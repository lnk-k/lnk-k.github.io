#!/usr/bin/env python3
"""Run liver patch model over a complete CT volume.

The trained patch classifier is applied with a 2D sliding window on axial
slices. The output is a probability heatmap volume plus a coarse liver mask.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image
from PIL import ImageDraw
from scipy import ndimage as ndi

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is not installed. Install it first, for example:\n"
        "  .venv/bin/python -m pip install torch\n"
    ) from exc

from train_liver_supcon import SmallLiverEncoder, choose_device


def image_from_array_like(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(reference.GetSpacing())
    image.SetOrigin(reference.GetOrigin())
    image.SetDirection(reference.GetDirection())
    return image


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


def make_grid_centers(
    shape: tuple[int, int],
    patch_size: int,
    stride: int,
    center_mask: np.ndarray,
) -> list[tuple[int, int]]:
    height, width = shape
    half = patch_size // 2
    ys = list(range(half, height - half + 1, stride))
    xs = list(range(half, width - half + 1, stride))
    if ys and ys[-1] != height - half:
        ys.append(height - half)
    if xs and xs[-1] != width - half:
        xs.append(width - half)

    centers = []
    for y in ys:
        for x in xs:
            if center_mask[y, x]:
                centers.append((y, x))
    return centers


def patches_to_tensor(
    image_slice: np.ndarray,
    centers: list[tuple[int, int]],
    patch_size: int,
) -> torch.Tensor:
    half = patch_size // 2
    patches = []
    for y, x in centers:
        patch = image_slice[y - half : y + half, x - half : x + half]
        patches.append(patch)
    array = np.stack(patches).astype(np.float32) / 255.0
    array = (array[:, None, :, :] - 0.5) / 0.5
    return torch.from_numpy(array)


def load_model(checkpoint_path: Path, device: torch.device) -> SmallLiverEncoder:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = checkpoint.get("args", {})
    model = SmallLiverEncoder(embedding_dim=int(model_args.get("embedding_dim", 128)))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def add_patch_scores(
    heat_sum: np.ndarray,
    heat_count: np.ndarray,
    centers: list[tuple[int, int]],
    scores: np.ndarray,
    patch_size: int,
) -> None:
    half = patch_size // 2
    for (y, x), score in zip(centers, scores, strict=True):
        heat_sum[y - half : y + half, x - half : x + half] += float(score)
        heat_count[y - half : y + half, x - half : x + half] += 1.0


def clean_mask(mask: np.ndarray, min_voxels: int, keep_largest_component: bool) -> np.ndarray:
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3, 3), dtype=bool))
    labels, count = ndi.label(mask)
    if count == 0:
        return mask.astype(np.uint8)

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    if keep_largest_component:
        return (labels == int(sizes.argmax())).astype(np.uint8)

    keep = sizes >= min_voxels
    keep[0] = False
    return keep[labels].astype(np.uint8)


def contour_from_mask(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    return mask & ~eroded


def save_contour_slices(
    image: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
    window_min: float,
    window_max: float,
    only_mask_slices: bool,
    contact_sheet_count: int,
    contact_sheet_columns: int,
) -> None:
    contour_dir = output_dir / "predicted_contour_slices"
    contour_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    saved_indices: list[int] = []
    for z in range(image.shape[0]):
        if only_mask_slices and not mask[z].any():
            continue
        gray = window_to_uint8(image[z], window_min, window_max)
        rgb = np.stack([gray, gray, gray], axis=-1)
        contour = contour_from_mask(mask[z].astype(bool))
        rgb[contour] = np.array([255, 0, 0], dtype=np.uint8)
        path = contour_dir / f"slice_z{z:03d}.png"
        Image.fromarray(rgb).save(path)
        saved_paths.append(path)
        saved_indices.append(z)

    if not saved_paths:
        return

    if len(saved_paths) > contact_sheet_count:
        positions = np.linspace(0, len(saved_paths) - 1, contact_sheet_count)
        selected_ids = [int(round(pos)) for pos in positions]
    else:
        selected_ids = list(range(len(saved_paths)))

    selected_paths = [saved_paths[i] for i in selected_ids]
    selected_indices = [saved_indices[i] for i in selected_ids]
    images = [Image.open(path).convert("RGB") for path in selected_paths]
    width, height = images[0].size
    rows = int(np.ceil(len(images) / contact_sheet_columns))
    sheet = Image.new(
        "RGB",
        (contact_sheet_columns * width, rows * height),
        color=(0, 0, 0),
    )
    draw = ImageDraw.Draw(sheet)
    for idx, preview in enumerate(images):
        x = (idx % contact_sheet_columns) * width
        y = (idx // contact_sheet_columns) * height
        sheet.paste(preview, (x, y))
        draw.rectangle((x, y, x + 86, y + 24), fill=(0, 0, 0))
        draw.text((x + 6, y + 5), f"z={selected_indices[idx]}", fill=(255, 255, 255))

    sheet.save(output_dir / "predicted_liver_contour_contact_sheet.png")


def save_previews(
    image: np.ndarray,
    heatmap: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
    max_previews: int,
    window_min: float,
    window_max: float,
) -> None:
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    heat_per_slice = heatmap.reshape(heatmap.shape[0], -1).max(axis=1)
    candidate_slices = np.where(heat_per_slice > 0)[0]
    if len(candidate_slices) == 0:
        candidate_slices = np.arange(image.shape[0])
    if len(candidate_slices) > max_previews:
        positions = np.linspace(0, len(candidate_slices) - 1, max_previews)
        selected = [int(candidate_slices[int(round(pos))]) for pos in positions]
    else:
        selected = [int(v) for v in candidate_slices]

    for z in selected:
        gray = window_to_uint8(image[z], window_min, window_max)
        rgb = np.stack([gray, gray, gray], axis=-1)

        heat = np.clip(heatmap[z], 0.0, 1.0)
        rgb[..., 0] = np.maximum(rgb[..., 0], np.round(heat * 255).astype(np.uint8))
        rgb[..., 1] = np.round(rgb[..., 1] * (1.0 - 0.35 * heat)).astype(np.uint8)

        contour = contour_from_mask(mask[z].astype(bool))
        rgb[contour] = np.array([0, 255, 0], dtype=np.uint8)
        Image.fromarray(rgb).save(preview_dir / f"slice_z{z:03d}.png")


def infer_volume(
    model: SmallLiverEncoder,
    image: np.ndarray,
    device: torch.device,
    patch_size: int,
    stride: int,
    batch_size: int,
    slice_step: int,
    window_min: float,
    window_max: float,
    body_threshold: float,
) -> np.ndarray:
    heatmap = np.zeros(image.shape, dtype=np.float32)

    for z in range(0, image.shape[0], slice_step):
        gray = window_to_uint8(image[z], window_min, window_max)
        body = body_mask_from_ct(image[z], body_threshold)
        centers = make_grid_centers(gray.shape, patch_size, stride, body)
        if not centers:
            continue

        heat_sum = np.zeros(gray.shape, dtype=np.float32)
        heat_count = np.zeros(gray.shape, dtype=np.float32)

        for start in range(0, len(centers), batch_size):
            batch_centers = centers[start : start + batch_size]
            batch = patches_to_tensor(gray, batch_centers, patch_size).to(device)
            with torch.no_grad():
                _, logits = model(batch)
                scores = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            add_patch_scores(heat_sum, heat_count, batch_centers, scores, patch_size)

        heatmap[z] = heat_sum / np.maximum(heat_count, 1.0)
        print(f"slice {z + 1}/{image.shape[0]}: {len(centers)} patches")

    if slice_step > 1:
        known_slices = np.arange(0, image.shape[0], slice_step)
        for z in range(image.shape[0]):
            nearest = int(known_slices[np.argmin(np.abs(known_slices - z))])
            if z not in known_slices:
                heatmap[z] = heatmap[nearest]

    return heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer liver probability heatmap for a CT.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="Input CT, NIfTI or MHD.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--slice-step", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--min-voxels", type=int, default=5000)
    parser.add_argument(
        "--keep-all-components",
        action="store_true",
        help="Keep all large predicted components. Default keeps only the largest component.",
    )
    parser.add_argument("--window-min", type=float, default=-100.0)
    parser.add_argument("--window-max", type=float, default=250.0)
    parser.add_argument("--body-threshold", type=float, default=-500.0)
    parser.add_argument("--max-previews", type=int, default=8)
    parser.add_argument(
        "--save-contour-slices",
        action="store_true",
        help="Save CT slices with predicted liver mask contour drawn in red.",
    )
    parser.add_argument(
        "--all-slices",
        action="store_true",
        help="When saving contour slices, save all slices instead of predicted-liver slices only.",
    )
    parser.add_argument("--contact-sheet-count", type=int, default=12)
    parser.add_argument("--contact-sheet-columns", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    model = load_model(args.checkpoint, device)

    image_itk = sitk.ReadImage(str(args.input))
    image = sitk.GetArrayFromImage(image_itk)
    heatmap = infer_volume(
        model=model,
        image=image,
        device=device,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
        slice_step=args.slice_step,
        window_min=args.window_min,
        window_max=args.window_max,
        body_threshold=args.body_threshold,
    )
    mask = clean_mask(
        heatmap >= args.threshold,
        args.min_voxels,
        keep_largest_component=not args.keep_all_components,
    )

    heatmap_path = args.output_dir / "liver_probability_heatmap.nii.gz"
    mask_path = args.output_dir / "liver_candidate_mask.nii.gz"
    sitk.WriteImage(image_from_array_like(heatmap.astype(np.float32), image_itk), str(heatmap_path))
    sitk.WriteImage(image_from_array_like(mask.astype(np.uint8), image_itk), str(mask_path))
    save_previews(
        image=image,
        heatmap=heatmap,
        mask=mask,
        output_dir=args.output_dir,
        max_previews=args.max_previews,
        window_min=args.window_min,
        window_max=args.window_max,
    )
    if args.save_contour_slices:
        save_contour_slices(
            image=image,
            mask=mask,
            output_dir=args.output_dir,
            window_min=args.window_min,
            window_max=args.window_max,
            only_mask_slices=not args.all_slices,
            contact_sheet_count=args.contact_sheet_count,
            contact_sheet_columns=args.contact_sheet_columns,
        )

    print(f"heatmap: {heatmap_path}")
    print(f"mask: {mask_path}")
    print(f"previews: {args.output_dir / 'previews'}")
    if args.save_contour_slices:
        print(f"contour slices: {args.output_dir / 'predicted_contour_slices'}")
        print(f"contact sheet: {args.output_dir / 'predicted_liver_contour_contact_sheet.png'}")


if __name__ == "__main__":
    main()
