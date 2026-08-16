#!/usr/bin/env python3
"""Train a simple 2D U-Net with Soft F2 and select by whole-volume Hard F2."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from liver_unet import (
    LiverUNet2D,
    binary_counts,
    metrics_from_counts,
    soft_f2_loss,
)
from liver_unet_data import (
    DEFAULT_TRAIN_CASES,
    DEFAULT_VAL_CASES,
    LiverSliceDataset,
    load_cached_case,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = (
    PROJECT_ROOT / "datasets/SLIVER07/processed_for_contrastive/unet_slice_cache"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: LiverUNet2D,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_items = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = soft_f2_loss(logits, targets)
        loss.backward()
        optimizer.step()

        batch_items = int(images.shape[0])
        total_loss += float(loss.item()) * batch_items
        total_items += batch_items
        with torch.no_grad():
            counts = binary_counts(torch.sigmoid(logits) >= 0.5, targets >= 0.5)
            true_positive += counts[0]
            false_positive += counts[1]
            false_negative += counts[2]

    return {
        "loss": total_loss / max(total_items, 1),
        **metrics_from_counts(true_positive, false_positive, false_negative),
    }


def predict_cached_volume(
    model: LiverUNet2D,
    image: np.ndarray,
    device: torch.device,
    image_size: int,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probability = np.zeros(image.shape, dtype=np.float32)
    original_size = image.shape[-2:]

    with torch.no_grad():
        for start in range(0, image.shape[0], batch_size):
            array = np.asarray(image[start : start + batch_size], dtype=np.float32)
            batch = torch.from_numpy(array[:, None] / 127.5 - 1.0)
            if batch.shape[-2:] != (image_size, image_size):
                batch = F.interpolate(
                    batch,
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            logits = model(batch.to(device))
            batch_probability = torch.sigmoid(logits)
            if batch_probability.shape[-2:] != original_size:
                batch_probability = F.interpolate(
                    batch_probability,
                    size=original_size,
                    mode="bilinear",
                    align_corners=False,
                )
            probability[start : start + batch.shape[0]] = (
                batch_probability[:, 0].cpu().numpy()
            )
    return probability


def validate_whole_volumes(
    model: LiverUNet2D,
    cache_dir: Path,
    case_ids: list[str],
    device: torch.device,
    image_size: int,
    batch_size: int,
    threshold: float,
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    total_tp = total_fp = total_fn = 0
    case_rows: list[dict[str, float | int | str]] = []
    component_counts: list[int] = []
    largest_component_ratios: list[float] = []

    for case_id in case_ids:
        image, target_array = load_cached_case(cache_dir, case_id)
        probability = predict_cached_volume(
            model,
            image,
            device,
            image_size,
            batch_size,
        )
        prediction = probability >= threshold
        target = np.asarray(target_array).astype(bool)
        tp = int(np.count_nonzero(prediction & target))
        fp = int(np.count_nonzero(prediction & ~target))
        fn = int(np.count_nonzero(~prediction & target))
        case_metrics = metrics_from_counts(tp, fp, fn)
        labels, component_count = ndi.label(prediction)
        if component_count:
            sizes = np.bincount(labels.ravel())[1:]
            largest_ratio = float(sizes.max() / max(sizes.sum(), 1))
        else:
            largest_ratio = 0.0
        component_counts.append(int(component_count))
        largest_component_ratios.append(largest_ratio)
        case_rows.append(
            {
                "case_id": case_id,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                **case_metrics,
                "component_count": int(component_count),
                "largest_component_ratio": largest_ratio,
            }
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
        print(
            f"  {case_id}: F2={case_metrics['f2']:.4f} "
            f"recall={case_metrics['recall']:.4f} components={component_count}"
        )

    aggregate = metrics_from_counts(total_tp, total_fp, total_fn)
    aggregate.update(
        {
            "mean_component_count": float(np.mean(component_counts)),
            "max_component_count": float(max(component_counts, default=0)),
            "mean_largest_component_ratio": float(np.mean(largest_component_ratios)),
        }
    )
    return aggregate, case_rows


def save_checkpoint(
    path: Path,
    model: LiverUNet2D,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "train_cases": list(args.train_cases),
            "val_cases": list(args.val_cases),
        },
        path,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-cases", nargs="+", default=list(DEFAULT_TRAIN_CASES))
    parser.add_argument("--val-cases", nargs="+", default=list(DEFAULT_VAL_CASES))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--max-train-slices", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = LiverSliceDataset(
        args.cache_dir,
        args.train_cases,
        augment=True,
        image_size=args.image_size,
        max_slices=args.max_train_slices,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    model = LiverUNet2D(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config = {
        "device": str(device),
        "parameter_count": parameter_count,
        "train_slice_count": len(train_dataset),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"device={device} parameters={parameter_count:,} "
        f"train_slices={len(train_dataset)}"
    )

    best_f2 = -math.inf
    epoch_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        print(f"epoch {epoch:03d}: whole-volume validation")
        val_metrics, current_case_rows = validate_whole_volumes(
            model,
            args.cache_dir,
            list(args.val_cases),
            device,
            args.image_size,
            args.val_batch_size,
            args.threshold,
        )
        row: dict[str, object] = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        epoch_rows.append(row)
        for case_row in current_case_rows:
            case_rows.append({"epoch": epoch, **case_row})
        write_csv(args.output_dir / "train_log.csv", epoch_rows)
        write_csv(args.output_dir / "validation_cases.csv", case_rows)

        save_checkpoint(args.output_dir / "last.pt", model, optimizer, args, epoch, val_metrics)
        if args.checkpoint_every > 0 and (
            epoch % args.checkpoint_every == 0 or epoch == args.epochs
        ):
            save_checkpoint(
                args.output_dir / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                args,
                epoch,
                val_metrics,
            )
        if val_metrics["f2"] > best_f2:
            best_f2 = val_metrics["f2"]
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, args, epoch, val_metrics)

        print(
            f"epoch {epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"val_F2={val_metrics['f2']:.4f} val_recall={val_metrics['recall']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"mean_components={val_metrics['mean_component_count']:.2f}"
        )

    print(f"best whole-volume F2: {best_f2:.4f}")
    print(f"model: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
