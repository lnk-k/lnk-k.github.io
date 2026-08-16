#!/usr/bin/env python3
"""Create liver/non-liver 2D patch data for contrastive learning.

This script uses complete CT volumes and liver masks. It samples positive
patches centered inside liver and negative patches centered in body-but-not-liver
regions, then writes PNG patches plus a CSV manifest.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage as ndi


def window_to_uint8(image_slice: np.ndarray, window_min: float, window_max: float) -> np.ndarray:
    clipped = np.clip(image_slice.astype(np.float32), window_min, window_max)
    scaled = (clipped - window_min) / max(window_max - window_min, 1e-6)
    return np.round(scaled * 255).astype(np.uint8)


def body_mask_from_ct(image_slice: np.ndarray, threshold: float) -> np.ndarray:
    mask = image_slice > threshold
    mask = ndi.binary_fill_holes(mask)
    labels, count = ndi.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def valid_centers(mask: np.ndarray, patch_size: int) -> np.ndarray:
    margin = patch_size // 2
    if mask.shape[0] <= patch_size or mask.shape[1] <= patch_size:
        return np.empty((0, 2), dtype=np.int64)
    valid = mask.copy()
    valid[:margin, :] = False
    valid[-margin:, :] = False
    valid[:, :margin] = False
    valid[:, -margin:] = False
    return np.argwhere(valid)


def crop_patch(image_slice: np.ndarray, center_y: int, center_x: int, patch_size: int) -> np.ndarray:
    half = patch_size // 2
    return image_slice[center_y - half : center_y + half, center_x - half : center_x + half]


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_patch(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def sample_points(points: np.ndarray, count: int, rng: random.Random) -> list[tuple[int, int]]:
    if len(points) == 0:
        return []
    indices = [rng.randrange(len(points)) for _ in range(count)]
    return [(int(points[i, 0]), int(points[i, 1])) for i in indices]


def process_case(
    row: dict[str, str],
    output_dir: Path,
    patch_size: int,
    patches_per_case: int,
    window_min: float,
    window_max: float,
    body_threshold: float,
    rng: random.Random,
) -> list[dict[str, str | int]]:
    case_id = row["case_id"]
    image = sitk.GetArrayFromImage(sitk.ReadImage(row["image_path"]))
    liver_mask = sitk.GetArrayFromImage(sitk.ReadImage(row["liver_mask_path"])).astype(bool)

    liver_pixels_per_slice = liver_mask.reshape(liver_mask.shape[0], -1).sum(axis=1)
    liver_slices = np.where(liver_pixels_per_slice > 200)[0]
    if len(liver_slices) == 0:
        return []

    rows: list[dict[str, str | int]] = []
    class_count = patches_per_case // 2

    for label_name, label_value in (("liver", 1), ("non_liver", 0)):
        saved = 0
        attempts = 0
        while saved < class_count and attempts < class_count * 20:
            attempts += 1
            z = int(rng.choice(liver_slices.tolist()))
            gray = window_to_uint8(image[z], window_min, window_max)
            body = body_mask_from_ct(image[z], body_threshold)

            if label_value == 1:
                center_mask = liver_mask[z]
            else:
                dilated_liver = ndi.binary_dilation(liver_mask[z], iterations=8)
                center_mask = body & ~dilated_liver

            centers = valid_centers(center_mask, patch_size)
            sampled = sample_points(centers, 1, rng)
            if not sampled:
                continue

            y, x = sampled[0]
            patch = crop_patch(gray, y, x, patch_size)
            if patch.shape != (patch_size, patch_size):
                continue

            patch_name = f"{case_id}_z{z:03d}_y{y:03d}_x{x:03d}_{label_name}_{saved:03d}.png"
            patch_path = output_dir / "patches" / label_name / patch_name
            save_patch(patch, patch_path)
            rows.append(
                {
                    "patch_path": str(patch_path),
                    "case_id": case_id,
                    "slice_index": z,
                    "center_y": y,
                    "center_x": x,
                    "label": label_value,
                    "label_name": label_name,
                }
            )
            saved += 1

    return rows


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample liver and non-liver patches for contrastive learning."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--patches-per-case", type=int, default=80)
    parser.add_argument("--window-min", type=float, default=-100.0)
    parser.add_argument("--window-max", type=float, default=250.0)
    parser.add_argument("--body-threshold", type=float, default=-500.0)
    parser.add_argument("--seed", type=int, default=20260702)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    manifest_rows = load_manifest(args.manifest)

    all_rows: list[dict[str, str | int]] = []
    for row in manifest_rows:
        case_rows = process_case(
            row=row,
            output_dir=args.output_dir,
            patch_size=args.patch_size,
            patches_per_case=args.patches_per_case,
            window_min=args.window_min,
            window_max=args.window_max,
            body_threshold=args.body_threshold,
            rng=rng,
        )
        all_rows.extend(case_rows)
        print(f"{row['case_id']}: saved {len(case_rows)} patches")

    write_csv(args.output_dir / "contrastive_patches.csv", all_rows)
    print(f"done: saved {len(all_rows)} patches")
    print(f"output: {args.output_dir}")


if __name__ == "__main__":
    main()
