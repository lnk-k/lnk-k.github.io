#!/usr/bin/env python3
"""Prepare uint8 memory-mapped CT/mask arrays for U-Net training."""

from __future__ import annotations

import argparse
from pathlib import Path

from liver_unet_data import (
    DEFAULT_TRAIN_CASES,
    DEFAULT_VAL_CASES,
    prepare_case_cache,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "datasets/SLIVER07/processed_for_contrastive"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=DATA_ROOT / "images")
    parser.add_argument("--masks-dir", type=Path, default=DATA_ROOT / "liver_masks")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DATA_ROOT / "unet_slice_cache",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=list(DEFAULT_TRAIN_CASES + DEFAULT_VAL_CASES),
    )
    parser.add_argument("--window-min", type=float, default=-100.0)
    parser.add_argument("--window-max", type=float, default=250.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for case_id in args.case_ids:
        prepare_case_cache(
            case_id=case_id,
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            cache_dir=args.cache_dir,
            window_min=args.window_min,
            window_max=args.window_max,
            overwrite=args.overwrite,
        )
    print(f"cache: {args.cache_dir}")


if __name__ == "__main__":
    main()
