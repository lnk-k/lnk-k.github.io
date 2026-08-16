#!/usr/bin/env python3
"""Create NIfTI volumes for predicted liver contours.

Outputs:
  liver_contour_mask.nii.gz
    Binary contour-only volume. Best used as an overlay with the original CT.

  ct_with_liver_contour.nii.gz
    CT volume with contour voxels set to a bright value. Useful when a viewer
    cannot overlay masks, but it is no longer a pure CT intensity image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


def image_from_array_like(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(reference.GetSpacing())
    image.SetOrigin(reference.GetOrigin())
    image.SetDirection(reference.GetDirection())
    return image


def contour_from_mask(mask: np.ndarray, thickness: int) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool))
    contour = mask & ~eroded
    if thickness > 1:
        contour = ndi.binary_dilation(contour, iterations=thickness - 1)
    return contour


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create liver contour NIfTI files.")
    parser.add_argument("--image", required=True, type=Path, help="Original CT volume.")
    parser.add_argument("--mask", required=True, type=Path, help="Predicted liver mask volume.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--thickness", type=int, default=1)
    parser.add_argument(
        "--contour-value",
        type=float,
        default=2000.0,
        help="Intensity written into CT contour voxels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_itk = sitk.ReadImage(str(args.image))
    mask_itk = sitk.ReadImage(str(args.mask))
    image = sitk.GetArrayFromImage(image_itk).astype(np.float32)
    mask = sitk.GetArrayFromImage(mask_itk).astype(bool)

    if image.shape != mask.shape:
        raise ValueError(f"Shape mismatch: image {image.shape}, mask {mask.shape}")

    contour = contour_from_mask(mask, args.thickness)
    contour_mask = contour.astype(np.uint8)
    ct_with_contour = image.copy()
    ct_with_contour[contour] = args.contour_value

    contour_path = args.output_dir / "liver_contour_mask.nii.gz"
    overlay_path = args.output_dir / "ct_with_liver_contour.nii.gz"

    sitk.WriteImage(image_from_array_like(contour_mask, mask_itk), str(contour_path))
    sitk.WriteImage(image_from_array_like(ct_with_contour, image_itk), str(overlay_path))

    print(f"contour mask: {contour_path}")
    print(f"ct with contour: {overlay_path}")
    print(f"contour voxels: {int(contour_mask.sum())}")


if __name__ == "__main__":
    main()
