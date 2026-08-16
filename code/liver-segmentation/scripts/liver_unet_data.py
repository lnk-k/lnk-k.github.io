#!/usr/bin/env python3
"""Data preparation and memory-mapped slice loading for liver U-Net."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


DEFAULT_TRAIN_CASES = (
    "sliver07_002",
    "sliver07_003",
    "sliver07_004",
    "sliver07_005",
    "sliver07_006",
    "sliver07_007",
    "sliver07_009",
    "sliver07_010",
    "sliver07_011",
    "sliver07_013",
    "sliver07_014",
    "sliver07_016",
    "sliver07_017",
    "sliver07_018",
    "sliver07_019",
    "sliver07_020",
)

DEFAULT_VAL_CASES = (
    "sliver07_001",
    "sliver07_008",
    "sliver07_012",
    "sliver07_015",
)


def normalize_ct_to_uint8(
    image: np.ndarray,
    window_min: float,
    window_max: float,
) -> np.ndarray:
    clipped = np.clip(image.astype(np.float32), window_min, window_max)
    scaled = (clipped - window_min) / max(window_max - window_min, 1e-6)
    return np.round(scaled * 255.0).astype(np.uint8)


def cache_paths(cache_dir: Path, case_id: str) -> tuple[Path, Path, Path]:
    return (
        cache_dir / f"{case_id}_image.npy",
        cache_dir / f"{case_id}_mask.npy",
        cache_dir / f"{case_id}_metadata.json",
    )


def prepare_case_cache(
    case_id: str,
    images_dir: Path,
    masks_dir: Path,
    cache_dir: Path,
    window_min: float,
    window_max: float,
    overwrite: bool = False,
) -> None:
    image_cache, mask_cache, metadata_path = cache_paths(cache_dir, case_id)
    if not overwrite and image_cache.exists() and mask_cache.exists() and metadata_path.exists():
        print(f"cached: {case_id}")
        return

    image_path = images_dir / f"{case_id}_0000.nii.gz"
    mask_path = masks_dir / f"{case_id}_liver_mask.nii.gz"
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    image_itk = sitk.ReadImage(str(image_path))
    mask_itk = sitk.ReadImage(str(mask_path))
    image = sitk.GetArrayFromImage(image_itk)
    mask = sitk.GetArrayFromImage(mask_itk)
    if image.shape != mask.shape:
        raise ValueError(f"Shape mismatch for {case_id}: {image.shape} vs {mask.shape}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(image_cache, normalize_ct_to_uint8(image, window_min, window_max))
    np.save(mask_cache, (mask > 0).astype(np.uint8))
    metadata = {
        "case_id": case_id,
        "shape_zyx": list(image.shape),
        "window_min": window_min,
        "window_max": window_max,
        "image_spacing_xyz": list(image_itk.GetSpacing()),
        "mask_spacing_xyz": list(mask_itk.GetSpacing()),
        "geometry_matches": {
            "size": image_itk.GetSize() == mask_itk.GetSize(),
            "spacing": image_itk.GetSpacing() == mask_itk.GetSpacing(),
            "origin": image_itk.GetOrigin() == mask_itk.GetOrigin(),
            "direction": image_itk.GetDirection() == mask_itk.GetDirection(),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"prepared: {case_id} shape={image.shape}")


def verify_case_cache(cache_dir: Path, case_ids: list[str] | tuple[str, ...]) -> None:
    missing: list[Path] = []
    for case_id in case_ids:
        for path in cache_paths(cache_dir, case_id)[:2]:
            if not path.exists():
                missing.append(path)
    if missing:
        lines = "\n".join(f"  {path}" for path in missing[:8])
        raise FileNotFoundError(
            "U-Net slice cache is incomplete. Run prepare_liver_unet_data.py first.\n"
            + lines
        )


class LiverSliceDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        case_ids: list[str] | tuple[str, ...],
        augment: bool,
        image_size: int,
        max_slices: int = 0,
        seed: int = 0,
    ) -> None:
        verify_case_cache(cache_dir, case_ids)
        self.cache_dir = cache_dir
        self.augment = augment
        self.image_size = image_size
        self.images: dict[str, np.ndarray] = {}
        self.masks: dict[str, np.ndarray] = {}
        self.records: list[tuple[str, int]] = []
        for case_id in case_ids:
            image_path, mask_path, _ = cache_paths(cache_dir, case_id)
            image = np.load(image_path, mmap_mode="r")
            mask = np.load(mask_path, mmap_mode="r")
            if image.shape != mask.shape:
                raise ValueError(f"Cached shape mismatch for {case_id}")
            self.images[case_id] = image
            self.masks[case_id] = mask
            self.records.extend((case_id, z) for z in range(image.shape[0]))

        if max_slices > 0 and len(self.records) > max_slices:
            rng = random.Random(seed)
            self.records = sorted(rng.sample(self.records, max_slices))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        case_id, z = self.records[index]
        image = np.asarray(self.images[case_id][z])
        mask = np.asarray(self.masks[case_id][z])

        image_array = np.ascontiguousarray(image, dtype=np.float32) / 127.5 - 1.0
        mask_array = np.ascontiguousarray(mask, dtype=np.float32)
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

        if image_tensor.shape[-2:] != (self.image_size, self.image_size):
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="nearest",
            ).squeeze(0)

        if self.augment:
            gain = random.uniform(0.9, 1.1)
            shift = random.uniform(-0.08, 0.08)
            image_tensor = torch.clamp(image_tensor * gain + shift, -1.0, 1.0)
            if random.random() < 0.5:
                image_tensor = torch.clamp(
                    image_tensor + torch.randn_like(image_tensor) * 0.02,
                    -1.0,
                    1.0,
                )

        return image_tensor, mask_tensor


def load_cached_case(cache_dir: Path, case_id: str) -> tuple[np.ndarray, np.ndarray]:
    verify_case_cache(cache_dir, [case_id])
    image_path, mask_path, _ = cache_paths(cache_dir, case_id)
    return (
        np.load(image_path, mmap_mode="r"),
        np.load(mask_path, mmap_mode="r"),
    )
