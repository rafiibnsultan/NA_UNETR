#################################################################################################
# Copyright (c) Rafi Ibn Sultan.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################

"""NATUNETR: a UNETR-style 3D segmentation network with a neighborhood-attention encoder.

The encoder (`NAT`) replaces global self-attention with sliding-window neighborhood
attention, which keeps the cost linear in the number of voxels and preserves the
locality that thin, elongated structures such as coronary arteries depend on. Passing
per-stage `dilations` turns the encoder into its dilated variant (DiNAT), widening the
receptive field without extra parameters.

The MLP inside every attention layer is LoRA-adapted, so a pretrained encoder can be
frozen and fine-tuned through a small number of low-rank updates.

Layout mirrors MONAI's SwinUNETR: five encoder stages feed skip connections into five
decoder stages, so checkpoints and helper code transfer between the two.
"""

from __future__ import annotations

from collections.abc import Sequence

import loralib as lora
import natten
import torch
import torch.nn as nn
from natten import NeighborhoodAttention3D as NeighborhoodAttention

from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath

# natten renamed the relative-position-bias argument in 0.17.
is_natten_post_017 = hasattr(natten, "context")

LORA_RANK = 8

__all__ = ["NATUNETR"]


class Mlp_NAT(nn.Module):
    """Feed-forward block with LoRA-adapted projections and an optional depthwise conv.

    The depthwise convolution between the two projections re-introduces local spatial
    mixing that token-wise MLPs otherwise discard.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
        use_dwconv: bool = False,
        spatial_dims: int = 3,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = lora.Linear(in_features, hidden_features, r=LORA_RANK)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.use_dwconv = use_dwconv
        if use_dwconv:
            if spatial_dims != 3:
                raise ValueError("This block expects 3D")
            # Depthwise: one group per channel.
            self.dw = nn.Conv3d(
                hidden_features, hidden_features, kernel_size=3, stride=1,
                padding=1, groups=hidden_features,
            )
        self.fc2 = lora.Linear(hidden_features, out_features, r=LORA_RANK)

    def forward(self, x):
        """x: (B, H, W, D, C)"""
        x = self.fc1(x)  # -> (B, H, W, D, hidden)
        if self.use_dwconv:
            x = x.permute(0, 4, 1, 2, 3)  # (B, hidden, H, W, D) for Conv3d
            x = self.dw(x)
            x = x.permute(0, 2, 3, 4, 1)  # back to channels-last
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvTokenizer(nn.Module):
    """Convolutional patch embedding: halves resolution and lifts to `embed_dim`."""

    def __init__(self, in_chans: int = 3, embed_dim: int = 96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv3d(
                embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1,
                padding=1, groups=embed_dim // 2, bias=False,
            ),  # depthwise
            nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 4, 1)  # (B, H, W, D, C)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvDownsampler(nn.Module):
    """Strided convolution that halves each spatial dim and doubles the channels."""

    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv3d(
            dim, 2 * dim, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False
        )
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 4, 1, 2, 3))  # Permute for Conv3d
        x = x.permute(0, 2, 3, 4, 1)  # Permute back after Conv3d
        x = self.norm(x)
        return x


class NATLayer(nn.Module):
    """Pre-norm neighborhood-attention block with an optional LayerScale residual."""

    def __init__(
        self,
        dim,
        num_heads,
        kernel_size,
        dilation=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        extra_args = {"rel_pos_bias": True} if is_natten_post_017 else {"bias": True}
        self.attn = NeighborhoodAttention(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            **extra_args,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp_NAT(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            use_dwconv=True,
            spatial_dims=3,
        )

        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class NATBlock(nn.Module):
    """One encoder stage: `depth` attention layers followed by an optional downsampler.

    Returns `(downsampled, pre_downsample)` so the caller can route the second value
    into the matching decoder stage as a skip connection.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        kernel_size: int,
        dilations=None,
        downsample=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                NATLayer(
                    dim=dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dilation=None if dilations is None else dilations[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    layer_scale=layer_scale,
                )
                for i in range(depth)
            ]
        )

        self.downsample = None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is None:
            return x, x
        return self.downsample(x), x


class NAT(nn.Module):
    """Neighborhood-attention backbone producing one feature map per stage.

    Args:
        embed_dim: channel width of the first stage; doubles at every downsample.
        mlp_ratio: hidden-to-input ratio inside each `Mlp_NAT`.
        depths: number of attention layers per stage.
        num_heads: attention heads per stage.
        kernel_size: neighborhood extent per stage.
        dilations: per-stage, per-layer dilation factors; `None` disables dilation.
        out_indices: stages whose outputs are normalized and returned.
        use_v2: insert a residual convolution block after each stage.
    """

    def __init__(
        self,
        embed_dim,
        mlp_ratio,
        depths,
        num_heads,
        kernel_size,
        drop_path_rate=0.2,
        in_chans=3,
        dilations=None,
        out_indices=(0, 1, 2, 3, 4),
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        frozen_stages=-1,
        pretrained=None,
        layer_scale=0.1,
        spatial_dims: int = 3,
        use_v2=False,
        **kwargs,
    ):
        super().__init__()
        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.num_features = [int(embed_dim * 2**i) for i in range(self.num_levels)]
        self.mlp_ratio = mlp_ratio

        self.patch_embed = ConvTokenizer(in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decays linearly across all layers of all stages.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.use_v2 = use_v2
        self.levels = nn.ModuleList()
        if self.use_v2:
            self.layers1c = nn.ModuleList()
            self.layers2c = nn.ModuleList()
            self.layers3c = nn.ModuleList()
            self.layers4c = nn.ModuleList()

        for i in range(self.num_levels):
            level = NATBlock(
                dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_size[i],
                dilations=None if dilations is None else dilations[i],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
            )
            self.levels.append(level)

            # v2 keeps a residual conv block per stage; the last stage has no downsample
            # and therefore no matching block.
            if self.use_v2 and i < 4:
                layerc = UnetrBasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=(embed_dim * 2) * 2**i,
                    out_channels=(embed_dim * 2) * 2**i,
                    kernel_size=3,
                    stride=1,
                    norm_name="instance",
                    res_block=True,
                )
                [self.layers1c, self.layers2c, self.layers3c, self.layers4c][i].append(layerc)

        # add a norm layer for each output
        self.out_indices = out_indices
        for i_layer in self.out_indices:
            self.add_module(f"norm{i_layer}", norm_layer(self.num_features[i_layer]))

        self.frozen_stages = frozen_stages
        if pretrained is not None:
            self.init_weights(pretrained)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 2:
            for i in range(0, self.frozen_stages - 1):
                m = self.levels[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        return self

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if pretrained is not None:
            raise TypeError("pretrained must be a str or None")

    def forward_embeddings(self, x):
        return self.patch_embed(x)

    def forward_tokens(self, x):
        convs = (
            [self.layers1c, self.layers2c, self.layers3c, self.layers4c] if self.use_v2 else []
        )
        outs = []
        for idx, level in enumerate(self.levels):
            x, xo = level(x)
            if self.use_v2 and idx < len(convs):
                x = x.permute(0, 4, 1, 2, 3)
                x = convs[idx][0](x.contiguous())
                x = x.permute(0, 2, 3, 4, 1)
            if idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(xo)
                outs.append(x_out.permute(0, 4, 1, 2, 3).contiguous())
        return outs

    def forward(self, x):
        x = self.forward_embeddings(x)
        return self.forward_tokens(x)

    def forward_features(self, x):
        return self.forward(x)


class NATUNETR(nn.Module):
    """UNETR-style encoder-decoder with a neighborhood-attention (NAT/DiNAT) backbone.

    Example:
        >>> model = NATUNETR(
        ...     in_channels=1,
        ...     out_channels=2,
        ...     kernel_size=(7, 7, 7, 3, 3),
        ...     dilations=None,   # a per-stage schedule switches this to DiNAT
        ...     use_v2=True,
        ...     feature_size=60,
        ... )
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: Sequence[Sequence[int]] | None = None,
        depths: Sequence[int] = (3, 4, 6, 18, 5),
        kernel_size: Sequence[int] = (9, 6, 3, 3, 3),
        num_heads: Sequence[int] = (3, 6, 12, 24, 48),
        feature_size: int = 48,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        use_v2: bool = False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            dilations: per-stage, per-layer dilation factors. `None` gives plain NAT;
                a schedule matching `depths` gives the dilated (DiNAT) variant.
            depths: number of layers in each stage.
            kernel_size: neighborhood extent of each stage's attention.
            num_heads: number of attention heads.
            feature_size: dimension of network feature size, must be divisible by 12.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: accepted for API parity with SwinUNETR; currently unused.
            spatial_dims: number of spatial dims.
            use_v2: using v2, which adds a residual convolution block at the beginning of each stage.
        """
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")
        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")
        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")
        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")
        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")
        if dilations is not None and len(dilations) != len(depths):
            raise ValueError(
                f"dilations should provide one schedule per stage: "
                f"got {len(dilations)} for {len(depths)} stages."
            )

        self.normalize = normalize

        self.natViT = NAT(
            in_chans=in_channels,
            embed_dim=feature_size,
            dilations=dilations,
            depths=depths,
            kernel_size=kernel_size,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_v2=use_v2,
        )

        # Encoder: one residual conv block per skip connection, plus the bottleneck.
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        # Decoder: each stage upsamples and fuses the matching encoder skip.
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels
        )

    @torch.jit.unused
    def forward(self, x_in):
        hidden_states_out = self.natViT(x_in)

        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)

        return self.out(out)
