#!/usr/bin/env python3
"""Generate a conservative coarse ROI for liver-region learning.

This script is intentionally simple: it creates a high-recall body bounding box
or a rough liver-side hint box from a CT NIfTI image. The result is meant as a
first unsupervised baseline, not as a final liver segmentation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count == 0:
        return mask

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def make_body_mask(image: np.ndarray, hu_threshold: int) -> np.ndarray:
    mask = image > hu_threshold
    mask = ndi.binary_closing(mask, structure=np.ones((3, 5, 5), dtype=bool))
    mask = ndi.binary_fill_holes(mask)
    mask = largest_component(mask)
    return mask


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("No foreground found. Try a lower --body-threshold.")

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0) + 1
    return int(z_min), int(z_max), int(y_min), int(y_max), int(x_min), int(x_max)


def expand_bbox(
    bbox: tuple[int, int, int, int, int, int],
    shape: tuple[int, int, int],
    margin_voxels: int,
) -> tuple[int, int, int, int, int, int]:
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    z_size, y_size, x_size = shape

    return (
        max(0, z_min - margin_voxels),
        min(z_size, z_max + margin_voxels),
        max(0, y_min - margin_voxels),
        min(y_size, y_max + margin_voxels),
        max(0, x_min - margin_voxels),
        min(x_size, x_max + margin_voxels),
    )


def apply_liver_hint(
    bbox: tuple[int, int, int, int, int, int],
    shape: tuple[int, int, int],
    index_side: str,
    keep_z_fraction: float,
    keep_x_fraction: float,
) -> tuple[int, int, int, int, int, int]:
    z_min, z_max, y_min, y_max, x_min, x_max = bbox

    z_len = z_max - z_min
    x_len = x_max - x_min

    # Keep a broad craniocaudal range by default. Many datasets disagree on
    # whether the first slice is superior or inferior.
    z_start = z_min
    z_end = min(shape[0], z_min + int(round(z_len * keep_z_fraction)))

    if index_side == "left":
        x_start = x_min
        x_end = min(shape[2], x_min + int(round(x_len * keep_x_fraction)))
    else:
        x_start = max(0, x_max - int(round(x_len * keep_x_fraction)))
        x_end = x_max

    return z_start, z_end, y_min, y_max, x_start, x_end


def crop_array(
    array: np.ndarray,
    bbox: tuple[int, int, int, int, int, int],
) -> np.ndarray:
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    return array[z_min:z_max, y_min:y_max, x_min:x_max]


def mask_from_bbox(
    shape: tuple[int, int, int],
    bbox: tuple[int, int, int, int, int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    mask[z_min:z_max, y_min:y_max, x_min:x_max] = 1
    return mask


def image_from_array_like(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(reference.GetSpacing())
    image.SetDirection(reference.GetDirection())
    image.SetOrigin(reference.GetOrigin())
    return image


def write_cropped_image(
    array: np.ndarray,
    reference: sitk.Image,
    bbox: tuple[int, int, int, int, int, int],
    output_path: Path,
) -> None:
    z_min, _, y_min, _, x_min, _ = bbox
    cropped = crop_array(array, bbox)
    image = sitk.GetImageFromArray(cropped)

    old_origin = np.asarray(reference.GetOrigin(), dtype=float)
    spacing = np.asarray(reference.GetSpacing(), dtype=float)
    direction = np.asarray(reference.GetDirection(), dtype=float).reshape(3, 3)
    start_index_xyz = np.asarray([x_min, y_min, z_min], dtype=float)
    new_origin = old_origin + direction @ (start_index_xyz * spacing)

    image.SetSpacing(reference.GetSpacing())
    image.SetDirection(reference.GetDirection())
    image.SetOrigin(tuple(float(v) for v in new_origin))
    sitk.WriteImage(image, str(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a high-recall coarse ROI for liver-region learning."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input CT NIfTI.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--strategy",
        choices=["body_bbox", "liver_hint"],
        default="body_bbox",
        help="body_bbox is safest; liver_hint crops one x-side of the body bbox.",
    )
    parser.add_argument(
        "--body-threshold",
        type=int,
        default=-500,
        help="HU threshold for separating body from air.",
    )
    parser.add_argument(
        "--margin-voxels",
        type=int,
        default=12,
        help="Voxel margin added after ROI selection.",
    )
    parser.add_argument(
        "--index-side",
        choices=["left", "right"],
        default="left",
        help="Array x-side to keep for liver_hint. Try both on a new dataset.",
    )
    parser.add_argument(
        "--keep-z-fraction",
        type=float,
        default=1.0,
        help="Fraction of body bbox z range retained by liver_hint.",
    )
    parser.add_argument(
        "--keep-x-fraction",
        type=float,
        default=0.70,
        help="Fraction of body bbox x range retained by liver_hint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image = sitk.ReadImage(str(args.input))
    array = sitk.GetArrayFromImage(image).astype(np.float32)

    body_mask = make_body_mask(array, args.body_threshold)
    bbox = bbox_from_mask(body_mask)

    if args.strategy == "liver_hint":
        bbox = apply_liver_hint(
            bbox=bbox,
            shape=array.shape,
            index_side=args.index_side,
            keep_z_fraction=args.keep_z_fraction,
            keep_x_fraction=args.keep_x_fraction,
        )

    bbox = expand_bbox(bbox, array.shape, args.margin_voxels)
    roi_mask = mask_from_bbox(array.shape, bbox)

    write_cropped_image(array, image, bbox, args.output_dir / "roi_image.nii.gz")
    sitk.WriteImage(
        image_from_array_like(roi_mask, image),
        str(args.output_dir / "roi_mask.nii.gz"),
    )

    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    metadata = {
        "input": str(args.input),
        "strategy": args.strategy,
        "array_shape_zyx": list(array.shape),
        "bbox_zyx": {
            "z": [z_min, z_max],
            "y": [y_min, y_max],
            "x": [x_min, x_max],
        },
        "crop_shape_zyx": [z_max - z_min, y_max - y_min, x_max - x_min],
        "crop_volume_fraction": float(roi_mask.mean()),
        "body_threshold": args.body_threshold,
        "margin_voxels": args.margin_voxels,
        "index_side": args.index_side,
    }
    (args.output_dir / "roi_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
