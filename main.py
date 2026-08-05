"""Fine-tune NATUNETR (or a baseline) on LAD coronary artery CT.

Expects the dataset laid out one folder per case:

    data/<case_id>/image.nii.gz
    data/<case_id>/mask_LADA-AG.nii.gz

Training starts from an ImageCAS-pretrained checkpoint, freezes the encoder, and
updates only the decoder + LoRA parameters.

Examples
--------
    python main.py --data-dir ./data --pretrained ./weights_backup/dinatunetr_imagecas_combined.pth
    python main.py --model-type swinunetr --folds 5 --epochs 300
    python main.py --from-scratch --loss single --epochs 50
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss, HausdorffDTLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.transforms import (
    AdjustContrastd,
    AsDiscrete,
    Compose,
    CropForegroundd,
    FillHoles,
    KeepLargestConnectedComponent,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandAdjustContrastd,
    RandAffined,
    RandBiasFieldd,
    RandCropByPosNegLabeld,
    RandGaussianSharpend,
    RandShiftIntensityd,
    RemoveSmallObjects,
    SavitzkyGolaySmoothd,
    ScaleIntensityRanged,
    SpatialPadd,
    ToTensord,
)
from sklearn.model_selection import KFold

from models import MODEL_TYPES, get_model

CLASSES = ("background", "arteries")


# ------------------------------------------------------------------ arguments
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument("--data-dir", default="./data", help="root folder holding one subfolder per case")
    data.add_argument("--image-name", default="image.nii.gz", help="image filename inside each case folder")
    data.add_argument("--label-name", default="mask_LADA-AG.nii.gz", help="mask filename inside each case folder")
    data.add_argument("--cache-rate", type=float, default=1.0, help="fraction of the dataset held in RAM")

    model = parser.add_argument_group("model")
    model.add_argument("--model-type", default="dinatunetr", choices=MODEL_TYPES)
    model.add_argument("--pretrained", default=None,
                       help="checkpoint to fine-tune from "
                            "(default: <weights-backup-dir>/<model-type>_imagecas_combined.pth)")
    model.add_argument("--from-scratch", action="store_true",
                       help="skip loading a pretrained checkpoint")
    model.add_argument("--weights-dir", default="./weights", help="where best-metric checkpoints are written")
    model.add_argument("--weights-backup-dir", default="./weights_backup",
                       help="where the pretrained checkpoint is looked up")

    train = parser.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--stage1-epochs", type=int, default=50,
                       help="epochs trained on Dice-Focal alone before the Hausdorff term kicks in")
    train.add_argument("--loss", dest="loss_name", default="combined", choices=["combined", "single"])
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--n-splits", type=int, default=5, help="number of cross-validation folds")
    train.add_argument("--folds", type=int, default=1,
                       help="how many folds to actually run (set to --n-splits for the full CV)")

    val = parser.add_argument_group("validation")
    val.add_argument("--val-interval", type=int, default=5, help="run validation every N epochs")
    val.add_argument("--sw-batch-size", type=int, default=4, help="sliding-window batch size at inference")
    val.add_argument("--min-object-size", type=int, default=64,
                     help="drop predicted components smaller than this many voxels")

    tf = parser.add_argument_group("preprocessing")
    tf.add_argument("--roi", type=int, nargs=3, default=[96, 96, 96], metavar=("X", "Y", "Z"),
                    help="patch size used for cropping and sliding-window inference")
    tf.add_argument("--samples-per-volume", type=int, default=4,
                    help="random positive/negative patches drawn per volume")
    tf.add_argument("--hu-min", type=float, default=-200, help="lower HU bound for intensity scaling")
    tf.add_argument("--hu-max", type=float, default=400, help="upper HU bound for intensity scaling")
    tf.add_argument("--gamma-range", type=float, nargs=2, default=[1.6, 1.8], metavar=("LO", "HI"),
                    help="random contrast gamma range used during training")
    tf.add_argument("--val-gamma", type=float, default=1.8, help="fixed contrast gamma used at validation")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                         help="torch device, e.g. cuda, cuda:0, cpu")
    runtime.add_argument("--num-workers", type=int, default=0,
                         help=">0 can be unstable with CacheDataset + random transforms")
    runtime.add_argument("--seed", type=int, default=768)
    runtime.add_argument("--no-deterministic", dest="deterministic", action="store_false",
                         help="allow cuDNN autotuning (faster, not reproducible)")
    runtime.add_argument("--no-amp", dest="amp", action="store_false",
                         help="disable mixed-precision training")

    args = parser.parse_args(argv)

    if args.folds > args.n_splits:
        parser.error(f"--folds ({args.folds}) cannot exceed --n-splits ({args.n_splits})")
    if args.stage1_epochs > args.epochs:
        parser.error(f"--stage1-epochs ({args.stage1_epochs}) cannot exceed --epochs ({args.epochs})")

    return args


def set_seed(seed, deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


# -------------------------------------------------------------------- dataset
def build_datalist(data_dir, image_name, label_name):
    """Collect {image, label} pairs from one subfolder per case."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    datalist = []
    for subfolder in sorted(os.listdir(data_dir)):
        subfolder_path = os.path.join(data_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        image_file = os.path.join(subfolder_path, image_name)
        label_file = os.path.join(subfolder_path, label_name)

        # Only keep cases where both the scan and its mask are present
        if os.path.exists(image_file) and os.path.exists(label_file):
            datalist.append({"image": image_file, "label": label_file})

    if not datalist:
        raise RuntimeError(
            f"No cases found under {data_dir} containing both {image_name} and {label_name}"
        )
    return datalist


# ----------------------------------------------------------------- transforms
class RemapLabels(MapTransform):
    def __init__(self, keys, mapping=None):
        """
        RemapLabels maps specified values in the label tensor to new values.
        Args:
            keys (list[str]): Keys in the data dictionary to apply this transform.
            mapping (dict): Dictionary defining original-to-new value mapping.
                            Default is {0: 0, 255: 1}.
        """
        super().__init__(keys)
        self.mapping = mapping if mapping else {0: 0, 255: 1}

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.remap(d[key])
        return d

    def remap(self, tensor):
        # Create a new tensor with the same shape and default value 0
        remapped = torch.zeros_like(tensor)
        for orig, new in self.mapping.items():
            remapped[tensor == orig] = new
        return remapped


def build_train_transforms(args, roi):
    return Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
            RemapLabels(keys=["label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.hu_min, a_max=args.hu_max,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            RandAdjustContrastd(
                keys=["image"],
                prob=0.8,
                gamma=tuple(args.gamma_range),  # slight to strong enhancement
            ),
            SavitzkyGolaySmoothd(keys=["image"], window_length=5, order=2),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi,
                pos=1,
                neg=1,
                num_samples=args.samples_per_volume,
                image_key="image",
                image_threshold=0,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=1.0, spatial_size=roi,
                rotate_range=(0, 0, np.pi / 30),
                scale_range=(0.05, 0.05, 0.05)),
            RandGaussianSharpend(
                keys=["image"],
                sigma1_x=(0.5, 1.0),
                sigma2_x=(1.0, 2.0),
                alpha=(0.3, 0.7),
                prob=0.5,
            ),
            RandBiasFieldd(
                keys=["image"],
                prob=0.3,
                coeff_range=(0.0, 0.1),
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )


def build_val_transforms(args, roi):
    return Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
            RemapLabels(keys=["label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.hu_min, a_max=args.hu_max,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            AdjustContrastd(keys=["image"], gamma=args.val_gamma),
            SavitzkyGolaySmoothd(keys=["image"], window_length=5, order=2),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=roi, mode="constant"),
            ToTensord(keys=["image", "label"]),
        ]
    )


def build_post_transforms(num_classes, min_object_size):
    post_pred = Compose([
        AsDiscrete(argmax=True, to_onehot=num_classes),
        KeepLargestConnectedComponent(applied_labels=[1]),
        RemoveSmallObjects(min_size=min_object_size),
        FillHoles(applied_labels=[1]),
    ])
    post_label = Compose([AsDiscrete(to_onehot=num_classes)])  # n organs and background
    return post_pred, post_label


# --------------------------------------------------------------------- losses
def combined_loss(logit_map, labels, dice_fn, hausdorff_fn, log_sigma1, log_sigma2, normalize=True):
    """
    Compute the combined loss using homoscedastic uncertainty.
    This function calculates the per-voxel Dice loss and Hausdorff loss,
    applies normalization to the Hausdorff loss if specified, and combines
    them using heteroscedastic uncertainty.
    Args:
        logit_map (torch.Tensor): The predicted logit map from the model.
        labels (torch.Tensor): The ground truth labels.
        dice_fn (Callable): Dice-Focal loss term.
        hausdorff_fn (Callable): Hausdorff distance loss term.
        log_sigma1 (nn.Parameter): Log of the learned uncertainty on the Dice term.
        log_sigma2 (nn.Parameter): Log of the learned uncertainty on the Hausdorff term.
        normalize (bool, optional): Whether to apply logarithmic scaling to
                                    the Hausdorff loss. Default is True.
    Returns:
        tuple: A tuple containing:
            - combined_loss (torch.Tensor): The combined loss with uncertainty.
            - dice_loss (torch.Tensor): The calculated Dice loss.
            - hausdorff_loss (torch.Tensor): The calculated Hausdorff loss.
    """
    # Calculate per-voxel losses
    dice_loss = dice_fn(logit_map, labels)
    hausdorff_loss = hausdorff_fn(logit_map, labels)

    if normalize:
        hausdorff_loss = torch.log(1 + hausdorff_loss)  # Smooth logarithmic scaling

    # Add noise to the Hausdorff loss for robustness
    hausdorff_loss += torch.randn_like(hausdorff_loss) * 0.01
    # Clamp sigma^2 maps to prevent instability
    sigma1_sq = torch.clamp(torch.exp(log_sigma1), min=1e-3, max=10)
    sigma2_sq = torch.clamp(torch.exp(log_sigma2), min=1e-3, max=10)

    # Apply heteroscedastic uncertainty to voxel-wise losses
    loss = (1 / (2 * sigma1_sq)) * dice_loss + (1 / (2 * sigma2_sq)) * hausdorff_loss
    loss += 0.1 * (log_sigma1 + log_sigma2)  # Regularization term

    # Reduce losses to a single value by averaging
    return torch.mean(loss), torch.mean(dice_loss), torch.mean(hausdorff_loss)


def resolve_pretrained(args):
    """Return the checkpoint path to fine-tune from, or None when training from scratch."""
    if args.from_scratch:
        return None

    path = args.pretrained or os.path.join(
        args.weights_backup_dir, f"{args.model_type}_imagecas_combined.pth"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {path}\n"
            "Pass --pretrained <path> to point at it, or --from-scratch to train without it."
        )
    return path


# ------------------------------------------------------------------- training
def run_fold(args, fold, train_files, val_files, device, roi, num_classes, pretrained):
    train_ds = CacheDataset(train_files, transform=build_train_transforms(args, roi),
                            cache_rate=args.cache_rate, num_workers=args.num_workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)

    val_ds = CacheDataset(data=val_files, transform=build_val_transforms(args, roi),
                          cache_rate=args.cache_rate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    post_pred, post_label = build_post_transforms(num_classes, args.min_object_size)

    print("Model type:", args.model_type)
    model, decoder_params, lora_params = get_model(args.model_type, num_classes, device)

    if pretrained is not None:
        state_dict = torch.load(pretrained, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device)  # move after loading

    dice_fn = DiceFocalLoss(
        include_background=True, to_onehot_y=True, softmax=True,
        weight=torch.tensor([0.1, 0.9]), alpha=0.8, gamma=2.0,
        lambda_dice=1.0, lambda_focal=1.0,
    )
    hausdorff_fn = HausdorffDTLoss(include_background=True, to_onehot_y=True, softmax=True)

    log_sigma1 = nn.Parameter(torch.tensor(0.0, requires_grad=True))  # uncertainty on Dice Loss
    log_sigma2 = nn.Parameter(torch.tensor(0.0, requires_grad=True))  # uncertainty on Hausdorff Loss

    # Fine-tuning: the encoder stays frozen, only decoder + LoRA weights are trained
    optimizer = torch.optim.AdamW(
        decoder_params + lora_params + [log_sigma1, log_sigma2],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    hd95 = HausdorffDistanceMetric(include_background=False, percentile=95)  # 95% Hausdorff
    ASD = SurfaceDistanceMetric(symmetric=True, include_background=False)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_metric = -1
    best_hd95 = float("nan")
    best_asd = float("nan")
    best_metric_epoch = -1

    total_start = time.time()
    dice_loss = torch.tensor(0)
    hausdorff_loss = torch.tensor(0)

    for epoch in range(args.epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.epochs}")
        model.train()
        epoch_loss = 0
        epoch_dice_loss = 0
        epoch_hausdorff_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logit_map = model(inputs)
                if args.model_type == "unet++":
                    logit_map = logit_map[0]

                if args.loss_name == "combined":
                    if epoch < args.stage1_epochs:
                        loss = dice_fn(logit_map, labels) + 0.1 * (log_sigma1 + log_sigma2)
                    else:
                        loss, dice_loss, hausdorff_loss = combined_loss(
                            logit_map, labels, dice_fn, hausdorff_fn, log_sigma1, log_sigma2
                        )
                else:
                    loss = dice_fn(logit_map, labels)
            scaler.scale(loss).backward()
            epoch_loss += loss.item()
            if args.loss_name == "combined":
                if epoch < args.stage1_epochs:
                    epoch_dice_loss = epoch_loss
                    epoch_hausdorff_loss += hausdorff_loss.item()
                else:
                    epoch_dice_loss += dice_loss.item()
                    epoch_hausdorff_loss += hausdorff_loss.item()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

        epoch_loss /= step
        if args.loss_name == "combined":
            epoch_dice_loss /= step
            epoch_hausdorff_loss /= step
            print(
                f"Epoch {epoch + 1} - "
                f"Average Combined Loss: {epoch_loss:.4f}, "
                f"Average Dice Loss: {epoch_dice_loss:.4f}, "
                f"Average Hausdorff Loss: {epoch_hausdorff_loss:.4f}, "
                f"log_sigma1: {log_sigma1.item()}, log_sigma2: {log_sigma2.item()}"
            )
        else:
            print(f"Epoch {epoch + 1} - Average Loss: {epoch_loss:.4f}")

        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                dice_metric.reset()
                hd95.reset()
                ASD.reset()
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    val_outputs = sliding_window_inference(val_inputs, roi, args.sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    hd95(y_pred=val_outputs, y=val_labels)
                    ASD(y_pred=val_outputs, y=val_labels)

                metric_dice = dice_metric.aggregate().item()
                metric_hd95 = hd95.aggregate().item()
                metric_asd = ASD.aggregate().item()

                if metric_dice > best_metric:
                    best_metric = metric_dice
                    best_hd95 = metric_hd95
                    best_asd = metric_asd
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(),
                        os.path.join(args.weights_dir, f"{args.model_type}_fold{fold + 1}.pth"),
                    )
                print(
                    f"Fold {fold + 1}, current epoch: {epoch + 1} current mean dice: {metric_dice:.4f}"
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch} "
                )
        print(f"Fold {fold + 1}, time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")

    print(
        f"Fold {fold + 1}, train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}, total time: {(time.time() - total_start):.4f}"
    )
    return {"dice": best_metric, "hd95": best_hd95, "asd": best_asd}


def main(argv=None):
    args = parse_args(argv)

    set_seed(args.seed, deterministic=args.deterministic)
    os.makedirs(args.weights_dir, exist_ok=True)

    device = torch.device(args.device)
    roi = tuple(args.roi)
    num_classes = len(CLASSES)
    pretrained = resolve_pretrained(args)

    datalist = build_datalist(args.data_dir, args.image_name, args.label_name)
    print(f"Found {len(datalist)} cases in {args.data_dir}")
    if pretrained:
        print(f"Fine-tuning from {pretrained}")
    else:
        print("Training from scratch")

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(datalist)):
        if fold >= args.folds:
            break
        print(f"\n🌀 Fold {fold + 1}/{args.n_splits}")

        train_files = [datalist[i] for i in train_idx]
        val_files = [datalist[i] for i in val_idx]
        fold_results.append(
            run_fold(args, fold, train_files, val_files, device, roi, num_classes, pretrained)
        )

    print(f"\n📊 Results over {len(fold_results)} fold(s):")
    for i, r in enumerate(fold_results):
        print(f"Fold {i + 1}: Dice {r['dice']:.4f}, HD95 {r['hd95']:.4f}, ASD {r['asd']:.4f}")

    if len(fold_results) > 1:
        for name, key in (("Dice", "dice"), ("HD95", "hd95"), ("ASD", "asd")):
            values = np.array([r[key] for r in fold_results])
            print(f"{name}: {values.mean():.4f} ± {values.std():.4f}")

    return fold_results


if __name__ == "__main__":
    main()
