# Instructions

How this project is laid out, how to install it, and how to run training and inference.

---

## Layout

| Path | Contents |
|---|---|
| [main.py](main.py) | Training entry point — argparse CLI, k-fold loop, validation |
| [models.py](models.py) | Model zoo; builds any architecture and returns its trainable parameter groups |
| [MONAI/](MONAI/) | Fork of MONAI carrying the NATUNETR implementation |

### Where the model lives

The architecture is **not** in this directory — it ships inside the MONAI fork so it can
be imported like any other MONAI network:

```
MONAI/monai/networks/nets/nat_unetr.py
```

```python
from monai.networks.nets import NATUNETR

model = NATUNETR(
    in_channels=1,
    out_channels=2,
    kernel_size=(7, 7, 7, 3, 3),
    dilations=None,      # a per-stage schedule switches this to DiNAT
    use_v2=True,
    feature_size=60,
)
```

That file defines `NATUNETR` (encoder-decoder), `NAT` (the attention backbone),
`NATBlock` / `NATLayer` (one stage / one attention layer), and `Mlp_NAT` (the
LoRA-adapted feed-forward block).

The fork is otherwise stock MONAI `1.4.1rc1` (commit `e1e3d8eb`). Changes against
upstream:

- `monai/networks/nets/nat_unetr.py` — new, the NATUNETR implementation
- `monai/networks/nets/__init__.py` — one line exporting `NATUNETR`
- `monai/networks/nets/swin_unetr.py` — adds an `Mlp_LoRA` block so the SwinUNETR
  baseline can be fine-tuned the same way as NATUNETR

---

## Installation

Requires Python ≥ 3.9 and a CUDA-capable GPU (neighborhood attention has no CPU kernel).

```bash
# 1. PyTorch — match the CUDA build to your driver, see pytorch.org
pip install torch

# 2. Neighborhood attention. The wheel must match your torch + CUDA versions,
#    see https://github.com/SHI-Labs/NATTEN for the right index URL.
pip install natten

# 3. LoRA
pip install loralib

# 4. The MONAI fork carrying NATUNETR — install this, NOT monai from PyPI
pip install -e MONAI

# 5. Runtime dependencies of the transforms and metrics used here
pip install nibabel scikit-image scikit-learn scipy einops
```

> **`pip install monai` will not work.** Stock MONAI has no `NATUNETR`; the import in
> [models.py](models.py) resolves only against this fork. If both are installed, the
> PyPI package will shadow the fork.

`nat_unetr.py` supports NATTEN both before and after the 0.17 API change (it detects
which relative-position-bias argument to pass), so either generation of wheel works.

---

## Data

One folder per case, each holding the scan and its mask:

```
data/
├── case_0001/
│   ├── image.nii.gz
│   └── mask_LADA-AG.nii.gz
├── case_0002/
│   ├── image.nii.gz
│   └── mask_LADA-AG.nii.gz
└── ...
```

Cases missing either file are skipped. Masks are expected as `{0, 255}` and are remapped
to `{0, 1}` by the `RemapLabels` transform. If your filenames differ, pass
`--image-name` / `--label-name` rather than editing the source.

---

## Training

```bash
python main.py --data-dir ./data --pretrained ./weights_backup/dinatunetr_imagecas_combined.pth
```

Training is a **fine-tuning** stage: it loads an ImageCAS-pretrained checkpoint, freezes
the encoder, and updates only the decoder and LoRA parameters. To train without a
checkpoint:

```bash
python main.py --from-scratch
```

More examples:

```bash
# Full 5-fold cross-validation
python main.py --folds 5 --n-splits 5

# A baseline instead of NATUNETR
python main.py --model-type swinunetr --epochs 300

# Plain Dice-Focal, no Hausdorff term
python main.py --loss single

# Larger patches on a smaller GPU budget
python main.py --roi 128 128 64 --batch-size 1 --samples-per-volume 2
```

`python main.py --help` lists every option with its default. The ones that matter most:

| Flag | Default | Meaning |
|---|---|---|
| `--model-type` | `dinatunetr` | one of the ten architectures in [models.py](models.py) |
| `--pretrained` | derived | checkpoint to fine-tune from |
| `--from-scratch` | off | skip loading a checkpoint |
| `--epochs` | `200` | total epochs per fold |
| `--stage1-epochs` | `50` | epochs on Dice-Focal alone before the Hausdorff term joins |
| `--loss` | `combined` | `combined` (uncertainty-weighted) or `single` |
| `--folds` | `1` | how many folds to actually run |
| `--n-splits` | `5` | folds the data is divided into |
| `--roi` | `96 96 96` | patch size for cropping and sliding-window inference |
| `--lr` | `1e-4` | AdamW learning rate |
| `--seed` | `768` | seeds torch/numpy/random and the `KFold` split |

### The two-stage loss

With `--loss combined`, the first `--stage1-epochs` epochs train on Dice-Focal alone.
After that a Hausdorff distance term joins, and the two are balanced by learned
homoscedastic uncertainty — `log_sigma1` and `log_sigma2` are optimized alongside the
network, so the weighting is not hand-tuned. Warming up on Dice-Focal first matters: the
Hausdorff term is unstable while predictions are still noise.

### Available architectures

`natunetr`, `dinatunetr`, `unet`, `unet++`, `nnunet`, `unetr`, `swinunetr`,
`swinunetrv2`, `nnformer`, `mednext`

Every builder in [models.py](models.py) freezes the encoder and returns
`(model, decoder_params, lora_params)`, so the optimizer sees only what should be
trained. `lora_params` is empty for architectures with no LoRA layers.

`nnformer` and `mednext` need third-party packages that are **not** bundled here
([nnFormer](https://github.com/282857341/nnFormer),
[MedNeXt](https://github.com/MIC-DKFZ/MedNeXt)). They are imported lazily, so the other
eight models work without them installed.

---

## Evaluation

Validation runs every `--val-interval` epochs on the held-out fold and reports three
metrics, all computed on post-processed predictions (argmax → largest connected
component → small-object removal → hole filling):

- **Dice** — overlap
- **HD95** — 95th-percentile Hausdorff distance
- **ASD** — average symmetric surface distance

The checkpoint at the best Dice is written to
`<weights-dir>/<model-type>_fold<N>.pth`. At the end, per-fold metrics are printed, plus
mean ± standard deviation when more than one fold ran.

### Applying a trained checkpoint

There is no separate inference script here. To evaluate a saved checkpoint on a case,
reuse the same transforms training used — otherwise the intensity window and contrast
adjustment won't match what the model saw:

```python
import torch
from monai.inferers import sliding_window_inference

from main import build_post_transforms, build_val_transforms, parse_args
from models import get_model

device = torch.device("cuda")
roi, num_classes = (96, 96, 96), 2

args = parse_args([])  # defaults for the preprocessing knobs
val_tf = build_val_transforms(args, roi)
post_pred, _ = build_post_transforms(num_classes, args.min_object_size)

model, _, _ = get_model("dinatunetr", num_classes, device)
model.load_state_dict(torch.load("weights/dinatunetr_fold1.pth", map_location="cpu"))
model.to(device).eval()

case = val_tf({"image": "data/case_0001/image.nii.gz",
               "label": "data/case_0001/mask_LADA-AG.nii.gz"})

with torch.no_grad():
    logits = sliding_window_inference(
        case["image"].unsqueeze(0).to(device), roi, args.sw_batch_size, model
    )
prediction = post_pred(logits[0])  # one-hot, channel 1 is the artery
```

Note the validation transform loads an image **and** a label, because it crops to the
foreground of the pair. For label-free inference you would need a variant of
`build_val_transforms` that operates on the image key alone.

---

## Reproducibility

`--seed` (default `768`) seeds torch, numpy and Python's `random`, and drives the
`KFold` split so folds are identical across runs. By default cuDNN runs
deterministically and autotuning is off; `--no-deterministic` trades that for speed.

The checkpoint `main.py` fine-tunes from is produced by a separate ImageCAS pretraining
stage that is not included here, and the pretrained weights are not distributed with the
code. Use `--from-scratch` to train without one.
