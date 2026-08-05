"""Model zoo for LAD coronary artery segmentation.

Every builder freezes the encoder and returns the parameter groups that stay
trainable, so the caller can hand them straight to an optimizer:

    model, decoder_params, lora_params = get_model("dinatunetr", num_classes, device)
    optimizer = torch.optim.AdamW(decoder_params + lora_params, lr=1e-4)

`lora_params` is empty for architectures that carry no LoRA layers.
"""

from __future__ import annotations

from functools import partial

import loralib as lora
import torch.nn as nn
from monai.networks.layers import Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.networks.nets import (
    NATUNETR,
    UNETR,
    BasicUNetPlusPlus,
    DynUNet,
    SwinUNETR,
    UNet,
)

PATCH_SIZE = (96, 96, 96)
LORA_RANK = 8

# Dilation schedule per NATUNETR stage, matched to depths (3, 4, 6, 18, 5).
# Alternatives explored during development:
#   settings 1: [[1, 16], [1, 8], [1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4], [1, 2], [1, 1]]
#   settings 2: [[1, 16], [1, 8], [1, 2, 1, 3, 1, 4], [1, 2], [1, 1]]
DINAT_DILATIONS = [
    [1, 6, 1],           # Stage 0: depth 3
    [1, 2, 1, 2],        # Stage 1: depth 4
    [1, 1, 1, 1, 1, 1],  # Stage 2: depth 6
    [1, 2] * 9,          # Stage 3: depth 18
    [1, 1, 1, 1, 1],     # Stage 4: depth 5
]


# ---------------------------------------------------------------------- utils
def freeze(*modules: nn.Module) -> None:
    """Disable gradients on every parameter of the given modules."""
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False


def collect_params(*modules: nn.Module) -> list[nn.Parameter]:
    """Flatten the parameters of the given modules into a single list."""
    params: list[nn.Parameter] = []
    for module in modules:
        params += list(module.parameters())
    return params


def enable_lora(model: nn.Module) -> list[nn.Parameter]:
    """Re-enable gradients on LoRA weights anywhere in the model and return them."""
    params = []
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            params.append(param)
    return params


def attrs(model: nn.Module, names: list[str]) -> list[nn.Module]:
    """Look up a list of submodules by attribute name."""
    return [getattr(model, name) for name in names]


def split_encoder_decoder(
    model: nn.Module, encoder_names: list[str], decoder_names: list[str]
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Freeze the named encoder modules, keep the named decoder modules trainable.

    Shared by the UNETR-style architectures (NATUNETR, SwinUNETR, UNETR), which all
    expose their encoder and decoder stages as flat attributes. LoRA weights are
    re-enabled after freezing, since they live inside the frozen encoder.
    """
    freeze(*attrs(model, encoder_names))
    lora_params = enable_lora(model)
    decoder_params = collect_params(*attrs(model, decoder_names))
    return decoder_params, lora_params


class LoRAMLPBlock(nn.Module):
    """Drop-in replacement for MONAI's MLPBlock with LoRA-adapted linear layers."""

    def __init__(self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.fc1 = lora.Linear(hidden_size, mlp_dim, r=LORA_RANK)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = lora.Linear(mlp_dim, hidden_size, r=LORA_RANK)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


# ------------------------------------------------------------------- builders
def build_natunetr(num_classes, device, dilations=None, feature_size=60):
    """NATUNETR with neighborhood attention; `dilations=None` gives the plain variant."""
    model = NATUNETR(
        in_channels=1,
        out_channels=num_classes,
        kernel_size=(7, 7, 7, 3, 3),
        dilations=dilations,
        use_v2=True,
        feature_size=feature_size,
    ).to(device)

    decoder_params, lora_params = split_encoder_decoder(
        model,
        encoder_names=["encoder1", "encoder2", "encoder3", "encoder4", "encoder10", "natViT"],
        decoder_names=["decoder1", "decoder2", "decoder3", "decoder4", "decoder5", "out"],
    )
    return model, decoder_params, lora_params


def build_swinunetr(num_classes, device, use_v2=False, feature_size=48):
    model = SwinUNETR(
        img_size=PATCH_SIZE,
        in_channels=1,
        out_channels=num_classes,
        feature_size=feature_size,
        use_checkpoint=True,
        use_v2=use_v2,
    ).to(device)

    decoder_params, lora_params = split_encoder_decoder(
        model,
        encoder_names=["encoder1", "encoder2", "encoder3", "encoder4", "encoder10", "swinViT"],
        decoder_names=["decoder1", "decoder2", "decoder3", "decoder4", "decoder5", "out"],
    )
    return model, decoder_params, lora_params


def build_unetr(num_classes, device, feature_size=48):
    """UNETR whose ViT MLP blocks are swapped for LoRA-adapted equivalents."""
    model = UNETR(
        img_size=PATCH_SIZE,
        in_channels=1,
        out_channels=num_classes,
        feature_size=feature_size,
        norm_name="batch",
    ).to(device)

    for block in model.vit.blocks:
        hidden_size = block.mlp.linear1.in_features
        mlp_dim = block.mlp.linear1.out_features
        dropout_rate = block.mlp.dropout1.p if hasattr(block.mlp, "dropout1") else 0.0
        block.mlp = LoRAMLPBlock(hidden_size, mlp_dim, dropout_rate)

    # UNETR has no decoder1; its shallowest decoder stage is decoder2.
    decoder_params, lora_params = split_encoder_decoder(
        model,
        encoder_names=["encoder1", "encoder2", "encoder3", "encoder4", "vit"],
        decoder_names=["decoder2", "decoder3", "decoder4", "decoder5", "out"],
    )
    return model, decoder_params, lora_params


def build_unet(num_classes, device):
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    # MONAI nests UNet as Sequential(down, SkipConnection(submodule), up) at each
    # level, so both halves have to be reached by recursing through the skips.
    def freeze_encoder(module):
        if isinstance(module, nn.Sequential) and len(module) == 3:
            encoder, skip = module[0], module[1]
            freeze(encoder)
            if isinstance(skip, nn.Module) and hasattr(skip, "submodule"):
                freeze_encoder(skip.submodule)

    def decoder_modules(module, found=None):
        found = [] if found is None else found
        if isinstance(module, nn.Sequential) and len(module) == 3:
            found.append(module[2])  # decoder (up path)
            if isinstance(module[1], SkipConnection):
                decoder_modules(module[1].submodule, found)
        elif hasattr(module, "children"):
            for child in module.children():
                decoder_modules(child, found)
        return found

    freeze_encoder(model.model)
    return model, collect_params(*decoder_modules(model.model)), []


def build_unetplusplus(num_classes, device):
    model = BasicUNetPlusPlus(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        features=(32, 32, 64, 128, 256, 32),
    ).to(device)

    # BasicUNetPlusPlus names its stages by prefix: conv_* encode, upcat_* and
    # final_conv_* decode.
    encoders = [getattr(model, n) for n in dir(model) if n.startswith("conv_")]
    decoders = [
        getattr(model, n) for n in dir(model)
        if n.startswith("upcat_") or n.startswith("final_conv_")
    ]

    freeze(*encoders)
    return model, collect_params(*decoders), []


def build_nnunet(num_classes, device):
    model = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        kernel_size=[[3, 3, 3]] * 5,
        strides=[[1, 1, 1]] + [[2, 2, 2]] * 4,
        upsample_kernel_size=[[2, 2, 2]] * 4,
    ).to(device)

    freeze(model.input_block, *model.downsamples, model.bottleneck)
    decoder_params = collect_params(*model.upsamples, model.output_block)
    return model, decoder_params, []


def build_nnformer(num_classes, device):
    from nnFormer.nnFormer_seg import nnFormer  # optional third-party baseline

    model = nnFormer(input_channels=1, num_classes=num_classes).to(device)

    freeze(model.model_down)
    lora_params = enable_lora(model)
    decoder_params = collect_params(model.decoder, model.final)
    return model, decoder_params, lora_params


def build_mednext(num_classes, device):
    from MedNeXt.nnunet_mednext.network_architecture.mednextv1.MedNextV1 import (  # optional
        MedNeXt,
    )

    model = MedNeXt(
        in_channels=1,
        n_channels=32,
        n_classes=num_classes,
        exp_r=4,                            # Expansion ratio in Expansion Layer
        kernel_size=5,                      # Kernel Size in Depthwise Conv. Layer
        enc_kernel_size=None,               # (Separate) Kernel Size in Encoder
        dec_kernel_size=None,               # (Separate) Kernel Size in Decoder
        deep_supervision=False,             # Enable Deep Supervision
        do_res=True,                        # Residual connection in MedNeXt block
        do_res_up_down=True,                # Residual conn. in Resampling blocks
        checkpoint_style=None,              # Enable Gradient Checkpointing
        block_counts=[2] * 9,               # Depth-first no. of blocks per layer
        norm_type="group",                  # Type of Norm: 'group' or 'layer'
        dim="3d",                           # Supports '3d', '2d' arguments
    ).to(device)

    stages = range(4)
    freeze(
        model.stem,
        *attrs(model, [f"enc_block_{i}" for i in stages]),
        *attrs(model, [f"down_{i}" for i in stages]),
        model.bottleneck,
    )

    decoder_names = []
    for i in reversed(stages):
        decoder_names += [f"up_{i}", f"dec_block_{i}"]
    decoder_names.append("out_0")
    if model.do_ds:  # deep supervision heads
        decoder_names += [f"out_{i}" for i in range(1, 5)]

    return model, collect_params(*attrs(model, decoder_names)), []


BUILDERS = {
    "natunetr": partial(build_natunetr, dilations=None),
    "dinatunetr": partial(build_natunetr, dilations=DINAT_DILATIONS),
    "unet": build_unet,
    "unet++": build_unetplusplus,
    "nnunet": build_nnunet,
    "unetr": build_unetr,
    "swinunetr": partial(build_swinunetr, use_v2=False),
    "swinunetrv2": partial(build_swinunetr, use_v2=True),
    "nnformer": build_nnformer,
    "mednext": build_mednext,
}

MODEL_TYPES = list(BUILDERS)


def get_model(model_type: str, num_classes: int, device) -> tuple:
    """Build `model_type`, freeze its encoder, and return the trainable groups.

    Args:
        model_type: One of `MODEL_TYPES`.
        num_classes: Number of output channels, background included.
        device: Torch device the model is moved to before parameters are collected.

    Returns:
        (model, decoder_params, lora_params)
    """
    try:
        builder = BUILDERS[model_type]
    except KeyError:
        raise ValueError(
            f"Unknown model_type {model_type!r}; expected one of {MODEL_TYPES}"
        ) from None

    return builder(num_classes, device)
