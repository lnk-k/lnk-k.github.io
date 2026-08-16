#!/usr/bin/env python3
"""Predict whether a patch looks like liver using a trained SupCon model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is not installed. Install it first, for example:\n"
        "  .venv/bin/python -m pip install torch\n"
    ) from exc

from train_liver_supcon import SmallLiverEncoder, choose_device


def load_patch(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    return (tensor - 0.5) / 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict liver probability for one patch.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--patch", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args = checkpoint.get("args", {})
    model = SmallLiverEncoder(embedding_dim=int(model_args.get("embedding_dim", 128)))
    model.load_state_dict(checkpoint["model_state"])

    device = choose_device(args.device)
    model.to(device)
    model.eval()

    patch = load_patch(args.patch).to(device)
    with torch.no_grad():
        _, logits = model(patch)
        probability = F.softmax(logits, dim=1)[0]

    print(f"non_liver_probability={float(probability[0]):.4f}")
    print(f"liver_probability={float(probability[1]):.4f}")
    print(f"prediction={'liver' if int(probability.argmax()) == 1 else 'non_liver'}")


if __name__ == "__main__":
    main()
