#!/usr/bin/env python3
"""Select complete CT volumes that contain liver labels.

This keeps the original 3D CT instead of slicing it into a 2D dataset. It also
writes a binary liver mask volume and a few red-contour preview PNGs for human
checking. The preview images are not meant for contrastive learning input.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage as ndi


IMAGE_SUFFIXES = (".nii.gz", ".nii", ".mhd")


def strip_nifti_suffix(path: Path) -> str:
    name = path.name
    for suffix in IMAGE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def numeric_key(path: Path) -> str | None:
    match = re.search(r"(\d+)$", strip_nifti_suffix(path))
    return match.group(1) if match else None


def parse_label_values(text: str) -> list[int] | None:
    text = text.strip().lower()
    if text in {"positive", "any", "all", ">0"}:
        return None
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def find_nifti_files(folder: Path) -> list[Path]:
    files = (
        sorted(folder.glob("*.nii"))
        + sorted(folder.glob("*.nii.gz"))
        + sorted(folder.glob("*.mhd"))
    )
    return sorted(set(files))


def find_matching_image(images_dir: Path, case_id: str, channel: str) -> Path | None:
    key = re.search(r"(\d+)$", case_id)
    numeric_case_id = key.group(1) if key else None
    candidates = [
        images_dir / f"{case_id}.nii.gz",
        images_dir / f"{case_id}.nii",
        images_dir / f"{case_id}.mhd",
        images_dir / f"{case_id}_{channel}.nii.gz",
        images_dir / f"{case_id}_{channel}.nii",
    ]
    if numeric_case_id is not None:
        candidates.extend(
            [
                images_dir / f"liver-orig{numeric_case_id}.mhd",
                images_dir / f"liver-orig{numeric_case_id}.nii.gz",
                images_dir / f"liver-orig{numeric_case_id}.nii",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if numeric_case_id is not None:
        for image_path in find_nifti_files(images_dir):
            if numeric_key(image_path) == numeric_case_id:
                return image_path
    return None


def mask_from_label(label: np.ndarray, label_values: list[int] | None) -> np.ndarray:
    if label_values is None:
        return label > 0
    return np.isin(label, label_values)


def copy_image_metadata(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(reference.GetSpacing())
    image.SetOrigin(reference.GetOrigin())
    image.SetDirection(reference.GetDirection())
    return image


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Cannot build bbox from an empty mask.")
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0) + 1
    return int(z_min), int(z_max), int(y_min), int(y_max), int(x_min), int(x_max)


def window_to_uint8(image_slice: np.ndarray, window_min: float, window_max: float) -> np.ndarray:
    clipped = np.clip(image_slice.astype(np.float32), window_min, window_max)
    scaled = (clipped - window_min) / max(window_max - window_min, 1e-6)
    return np.round(scaled * 255).astype(np.uint8)


def contour_from_mask(mask: np.ndarray) -> np.ndarray:
    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    return mask & ~eroded


def overlay_contour(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[contour_from_mask(mask)] = np.array([255, 0, 0], dtype=np.uint8)
    return rgb


def evenly_sample_slices(slice_indices: np.ndarray, max_count: int) -> list[int]:
    if len(slice_indices) <= max_count:
        return [int(v) for v in slice_indices]
    positions = np.linspace(0, len(slice_indices) - 1, max_count)
    return [int(slice_indices[int(round(pos))]) for pos in positions]


def save_preview_pngs(
    image: np.ndarray,
    mask: np.ndarray,
    case_id: str,
    preview_dir: Path,
    max_previews: int,
    window_min: float,
    window_max: float,
) -> list[Path]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    liver_slices = np.where(mask.reshape(mask.shape[0], -1).sum(axis=1) > 0)[0]
    selected_slices = evenly_sample_slices(liver_slices, max_previews)
    preview_paths = []

    for z in selected_slices:
        gray = window_to_uint8(image[z], window_min, window_max)
        overlay = overlay_contour(gray, mask[z])
        path = preview_dir / f"{case_id}_z{z:03d}.png"
        Image.fromarray(overlay).save(path)
        preview_paths.append(path)

    return preview_paths


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def process_case(
    image_path: Path,
    label_path: Path,
    case_id: str,
    output_dir: Path,
    label_values: list[int] | None,
    min_mask_voxels: int,
    max_previews: int,
    window_min: float,
    window_max: float,
) -> dict[str, str | int | float] | None:
    image_itk = sitk.ReadImage(str(image_path))
    label_itk = sitk.ReadImage(str(label_path))
    image = sitk.GetArrayFromImage(image_itk)
    label = sitk.GetArrayFromImage(label_itk)

    if image.shape != label.shape:
        raise ValueError(
            f"Shape mismatch for {case_id}: image {image.shape}, label {label.shape}"
        )

    mask = mask_from_label(label, label_values)
    mask_voxels = int(mask.sum())
    if mask_voxels < min_mask_voxels:
        return None

    images_out = output_dir / "images"
    labels_out = output_dir / "labels_original"
    masks_out = output_dir / "liver_masks"
    previews_out = output_dir / "outline_previews" / case_id
    for folder in (images_out, labels_out, masks_out):
        folder.mkdir(parents=True, exist_ok=True)

    image_out = images_out / f"{case_id}_0000.nii.gz"
    label_out = labels_out / f"{case_id}.nii.gz"
    mask_out = masks_out / f"{case_id}_liver_mask.nii.gz"

    sitk.WriteImage(image_itk, str(image_out))
    sitk.WriteImage(label_itk, str(label_out))
    sitk.WriteImage(copy_image_metadata(mask.astype(np.uint8), label_itk), str(mask_out))
    preview_paths = save_preview_pngs(
        image=image,
        mask=mask,
        case_id=case_id,
        preview_dir=previews_out,
        max_previews=max_previews,
        window_min=window_min,
        window_max=window_max,
    )

    z_min, z_max, y_min, y_max, x_min, x_max = bbox_from_mask(mask)
    return {
        "case_id": case_id,
        "image_path": str(image_out),
        "label_path": str(label_out),
        "liver_mask_path": str(mask_out),
        "preview_dir": str(previews_out),
        "preview_count": len(preview_paths),
        "shape_z": image.shape[0],
        "shape_y": image.shape[1],
        "shape_x": image.shape[2],
        "mask_voxels": mask_voxels,
        "mask_fraction": float(mask_voxels / mask.size),
        "bbox_z_min": z_min,
        "bbox_z_max": z_max,
        "bbox_y_min": y_min,
        "bbox_y_max": y_max,
        "bbox_x_min": x_min,
        "bbox_x_max": x_max,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep complete CT volumes with liver labels and save liver masks."
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
    parser.add_argument("--min-mask-voxels", type=int, default=1000)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--max-previews", type=int, default=5)
    parser.add_argument("--window-min", type=float, default=-100.0)
    parser.add_argument("--window-max", type=float, default=250.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_values = parse_label_values(args.label_values)
    label_files = find_nifti_files(args.labels_dir)
    if args.case_limit > 0:
        label_files = label_files[: args.case_limit]

    rows: list[dict[str, str | int | float]] = []
    missing_images = []

    for label_path in label_files:
        raw_case_id = strip_nifti_suffix(label_path)
        key = numeric_key(label_path)
        case_id = f"sliver07_{key}" if key is not None else raw_case_id
        image_path = find_matching_image(args.images_dir, raw_case_id, args.channel)
        if image_path is None:
            missing_images.append(case_id)
            continue

        row = process_case(
            image_path=image_path,
            label_path=label_path,
            case_id=case_id,
            output_dir=args.output_dir,
            label_values=label_values,
            min_mask_voxels=args.min_mask_voxels,
            max_previews=args.max_previews,
            window_min=args.window_min,
            window_max=args.window_max,
        )
        if row is None:
            print(f"{case_id}: skipped, no enough liver label voxels")
            continue
        rows.append(row)
        print(f"{case_id}: kept complete CT, mask voxels={row['mask_voxels']}")

    write_csv(args.output_dir / "metadata.csv", rows)
    write_csv(
        args.output_dir / "contrastive_manifest.csv",
        [
            {
                "image_path": row["image_path"],
                "liver_mask_path": row["liver_mask_path"],
                "case_id": row["case_id"],
                "label": "liver_ct",
            }
            for row in rows
        ],
    )

    print(f"done: kept {len(rows)} complete CT volumes")
    print(f"output: {args.output_dir}")
    if missing_images:
        print(f"warning: {len(missing_images)} labels had no matching image")


if __name__ == "__main__":
    main()
