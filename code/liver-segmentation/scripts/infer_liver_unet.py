#!/usr/bin/env python3
"""Run raw 2D U-Net liver segmentation and write aligned NIfTI files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from liver_unet import LiverUNet2D
from liver_unet_data import normalize_ct_to_uint8
from train_liver_unet import choose_device


def image_like(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    result = sitk.GetImageFromArray(array)
    result.CopyInformation(reference)
    return result


def predict_volume(
    model: LiverUNet2D,
    image: np.ndarray,
    device: torch.device,
    image_size: int,
    batch_size: int,
    window_min: float,
    window_max: float,
) -> np.ndarray:
    normalized = normalize_ct_to_uint8(image, window_min, window_max)
    probability = np.zeros(image.shape, dtype=np.float32)
    original_size = image.shape[-2:]
    model.eval()
    with torch.no_grad():
        for start in range(0, image.shape[0], batch_size):
            array = normalized[start : start + batch_size].astype(np.float32)
            batch = torch.from_numpy(array[:, None] / 127.5 - 1.0)
            if batch.shape[-2:] != (image_size, image_size):
                batch = F.interpolate(
                    batch,
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            prediction = torch.sigmoid(model(batch.to(device)))
            if prediction.shape[-2:] != original_size:
                prediction = F.interpolate(
                    prediction,
                    size=original_size,
                    mode="bilinear",
                    align_corners=False,
                )
            probability[start : start + batch.shape[0]] = prediction[:, 0].cpu().numpy()
            print(f"slices {start + 1}-{min(start + batch.shape[0], image.shape[0])}/{image.shape[0]}")
    return probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-min", type=float, default=-100.0)
    parser.add_argument("--window-max", type=float, default=250.0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args = checkpoint.get("args", {})
    image_size = int(model_args.get("image_size", 256))
    base_channels = int(model_args.get("base_channels", 16))
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(model_args.get("threshold", 0.5))
    )
    model = LiverUNet2D(base_channels=base_channels)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    image_itk = sitk.ReadImage(str(args.input))
    image = sitk.GetArrayFromImage(image_itk)
    probability = predict_volume(
        model,
        image,
        device,
        image_size,
        args.batch_size,
        args.window_min,
        args.window_max,
    )
    mask = probability >= threshold
    contour = mask & ~ndi.binary_erosion(
        mask,
        structure=np.ones((3, 3, 3), dtype=bool),
    )
    labels, component_count = ndi.label(mask)
    if component_count:
        sizes = np.bincount(labels.ravel())[1:]
        largest_component_ratio = float(sizes.max() / max(sizes.sum(), 1))
    else:
        largest_component_ratio = 0.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(
        image_like(probability.astype(np.float32), image_itk),
        str(args.output_dir / "liver_probability.nii.gz"),
    )
    sitk.WriteImage(
        image_like(mask.astype(np.uint8), image_itk),
        str(args.output_dir / "liver_mask_raw.nii.gz"),
    )
    sitk.WriteImage(
        image_like(contour.astype(np.uint8), image_itk),
        str(args.output_dir / "liver_contour_raw.nii.gz"),
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "input": str(args.input),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "threshold": threshold,
        "image_size": image_size,
        "component_count": int(component_count),
        "largest_component_ratio": largest_component_ratio,
        "postprocessing_applied": False,
    }
    (args.output_dir / "inference_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"raw mask: {args.output_dir / 'liver_mask_raw.nii.gz'}")
    print(f"components: {component_count}, largest ratio: {largest_component_ratio:.4f}")


if __name__ == "__main__":
    main()
