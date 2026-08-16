#!/usr/bin/env python3
"""Extract liver-positive 2D slices and liver crops for contrastive learning.

Input:
  imagesTr/case_xxx.nii.gz
  labelsTr/case_xxx.nii.gz

Output:
  clean_slices/     full axial slices without drawings
  outline_slices/   full axial slices with a red liver contour for checking
  masks/            binary liver masks as PNG
  liver_crops/      clean liver-region crops, usually best for contrastive learning
  metadata.csv
  contrastive_manifest.csv

The red contour images are for human inspection. Use clean_slices or liver_crops
for model training so the model does not learn the artificial red line.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage as ndi


NIFTI_SUFFIXES = (".nii.gz", ".nii")


def strip_nifti_suffix(path: Path) -> str:
    name = path.name
    for suffix in NIFTI_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def parse_label_values(text: str) -> list[int] | None:
    text = text.strip().lower()
    if text in {"positive", "any", "all", ">0"}:
        return None
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def find_nifti_files(folder: Path) -> list[Path]:
    files = sorted(folder.glob("*.nii")) + sorted(folder.glob("*.nii.gz"))
    return sorted(set(files))


def find_matching_image(images_dir: Path, case_id: str, channel: str) -> Path | None:
    candidates = [
        images_dir / f"{case_id}.nii.gz",
        images_dir / f"{case_id}.nii",
        images_dir / f"{case_id}_{channel}.nii.gz",
        images_dir / f"{case_id}_{channel}.nii",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_array(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def make_mask(label_slice: np.ndarray, label_values: list[int] | None) -> np.ndarray:
    if label_values is None:
        return label_slice > 0
    return np.isin(label_slice, label_values)


def window_to_uint8(image_slice: np.ndarray, window_min: float, window_max: float) -> np.ndarray:
    clipped = np.clip(image_slice.astype(np.float32), window_min, window_max)
    scaled = (clipped - window_min) / max(window_max - window_min, 1e-6)
    return np.round(scaled * 255).astype(np.uint8)


def contour_from_mask(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    return mask & ~eroded


def overlay_contour(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = np.stack([gray, gray, gray], axis=-1)
    contour = contour_from_mask(mask)
    rgb[contour] = np.array([255, 0, 0], dtype=np.uint8)
    return rgb


def bbox_from_mask_2d(mask: np.ndarray, padding: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Cannot build bbox from an empty mask.")
    y_min = max(0, int(ys.min()) - padding)
    y_max = min(mask.shape[0], int(ys.max()) + padding + 1)
    x_min = max(0, int(xs.min()) - padding)
    x_max = min(mask.shape[1], int(xs.max()) + padding + 1)
    return x_min, y_min, x_max, y_max


def save_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def process_case(
    image_path: Path,
    label_path: Path,
    output_dir: Path,
    label_values: list[int] | None,
    window_min: float,
    window_max: float,
    min_mask_pixels: int,
    crop_padding: int,
    max_slices_per_case: int,
) -> list[dict[str, str | int]]:
    case_id = strip_nifti_suffix(label_path)
    image = read_array(image_path)
    label = read_array(label_path)

    if image.shape != label.shape:
        raise ValueError(
            f"Shape mismatch for {case_id}: image {image.shape}, label {label.shape}"
        )

    rows: list[dict[str, str | int]] = []
    kept = 0

    for z in range(label.shape[0]):
        mask = make_mask(label[z], label_values)
        mask_pixels = int(mask.sum())
        if mask_pixels < min_mask_pixels:
            continue

        if max_slices_per_case > 0 and kept >= max_slices_per_case:
            break

        gray = window_to_uint8(image[z], window_min, window_max)
        outline = overlay_contour(gray, mask)
        mask_png = (mask.astype(np.uint8) * 255)
        x_min, y_min, x_max, y_max = bbox_from_mask_2d(mask, crop_padding)
        crop = gray[y_min:y_max, x_min:x_max]

        file_id = f"{case_id}_z{z:03d}"
        clean_path = output_dir / "clean_slices" / f"{file_id}.png"
        outline_path = output_dir / "outline_slices" / f"{file_id}.png"
        mask_path = output_dir / "masks" / f"{file_id}.png"
        crop_path = output_dir / "liver_crops" / f"{file_id}.png"

        save_png(gray, clean_path)
        save_png(outline, outline_path)
        save_png(mask_png, mask_path)
        save_png(crop, crop_path)

        rows.append(
            {
                "case_id": case_id,
                "slice_index": z,
                "mask_pixels": mask_pixels,
                "bbox_x_min": x_min,
                "bbox_y_min": y_min,
                "bbox_x_max": x_max,
                "bbox_y_max": y_max,
                "clean_slice": str(clean_path),
                "outline_slice": str(outline_path),
                "mask": str(mask_path),
                "liver_crop": str(crop_path),
            }
        )
        kept += 1

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


def write_contrastive_manifest(output_dir: Path, rows: list[dict[str, str | int]]) -> None:
    manifest_rows = [
        {
            "image_path": row["liver_crop"],
            "case_id": row["case_id"],
            "slice_index": row["slice_index"],
            "label": "liver_positive",
        }
        for row in rows
    ]
    write_csv(output_dir / "contrastive_manifest.csv", manifest_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep liver-positive slices, draw liver contours, and export crops."
    )
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--label-values",
        default="1,2",
        help="Label values treated as liver area. Use '>0' for any positive label.",
    )
    parser.add_argument("--channel", default="0000", help="nnU-Net image channel suffix.")
    parser.add_argument("--window-min", type=float, default=-100.0)
    parser.add_argument("--window-max", type=float, default=250.0)
    parser.add_argument("--min-mask-pixels", type=int, default=80)
    parser.add_argument("--crop-padding", type=int, default=24)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--max-slices-per-case", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_values = parse_label_values(args.label_values)
    label_files = find_nifti_files(args.labels_dir)

    if args.case_limit > 0:
        label_files = label_files[: args.case_limit]

    all_rows: list[dict[str, str | int]] = []
    missing_images: list[str] = []

    for label_path in label_files:
        case_id = strip_nifti_suffix(label_path)
        image_path = find_matching_image(args.images_dir, case_id, args.channel)
        if image_path is None:
            missing_images.append(case_id)
            continue

        rows = process_case(
            image_path=image_path,
            label_path=label_path,
            output_dir=args.output_dir,
            label_values=label_values,
            window_min=args.window_min,
            window_max=args.window_max,
            min_mask_pixels=args.min_mask_pixels,
            crop_padding=args.crop_padding,
            max_slices_per_case=args.max_slices_per_case,
        )
        all_rows.extend(rows)
        print(f"{case_id}: kept {len(rows)} liver-positive slices")

    write_csv(args.output_dir / "metadata.csv", all_rows)
    write_contrastive_manifest(args.output_dir, all_rows)

    print(f"done: kept {len(all_rows)} slices")
    print(f"output: {args.output_dir}")
    if missing_images:
        print(f"warning: {len(missing_images)} labels had no matching image")


if __name__ == "__main__":
    main()
