#!/usr/bin/env python3
"""Train a small supervised-contrastive model for liver patch recognition.

Input CSV columns expected:
  patch_path, case_id, label

The script trains with two losses:
  1. supervised contrastive loss: same-class patches are pulled together
  2. cross entropy loss: a small classifier predicts liver vs non-liver
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is not installed. Install it first, for example:\n"
        "  .venv/bin/python -m pip install torch\n"
    ) from exc


@dataclass(frozen=True)
class PatchRow:
    patch_path: Path
    case_id: str
    label: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_manifest(path: Path) -> list[PatchRow]:
    rows: list[PatchRow] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                PatchRow(
                    patch_path=Path(row["patch_path"]),
                    case_id=row["case_id"],
                    label=int(row["label"]),
                )
            )
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def split_by_case(
    rows: list[PatchRow],
    val_fraction: float,
    seed: int,
) -> tuple[list[PatchRow], list[PatchRow], list[str], list[str]]:
    cases = sorted({row.case_id for row in rows})
    rng = random.Random(seed)
    rng.shuffle(cases)

    val_count = max(1, int(round(len(cases) * val_fraction)))
    val_cases = set(cases[:val_count])
    train_rows = [row for row in rows if row.case_id not in val_cases]
    val_rows = [row for row in rows if row.case_id in val_cases]

    if not train_rows or not val_rows:
        raise ValueError("Train/val split produced an empty split.")
    return train_rows, val_rows, sorted(set(cases) - val_cases), sorted(val_cases)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def augment_patch(image: Image.Image, rng: random.Random) -> torch.Tensor:
    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    rotations = rng.randrange(4)
    if rotations:
        image = image.rotate(90 * rotations)

    if rng.random() < 0.8:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 1.35))
    if rng.random() < 0.8:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.80, 1.20))

    tensor = pil_to_tensor(image)
    if rng.random() < 0.5:
        noise = torch.randn_like(tensor) * rng.uniform(0.01, 0.04)
        tensor = torch.clamp(tensor + noise, 0.0, 1.0)
    return (tensor - 0.5) / 0.5


def plain_patch(image: Image.Image) -> torch.Tensor:
    return (pil_to_tensor(image) - 0.5) / 0.5


class PatchDataset(Dataset):
    def __init__(self, rows: list[PatchRow], train: bool, seed: int) -> None:
        self.rows = rows
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image = Image.open(row.patch_path).convert("L")
        label = torch.tensor(row.label, dtype=torch.long)

        if self.train:
            rng = random.Random(self.seed + index + random.randrange(1_000_000))
            view1 = augment_patch(image, rng)
            view2 = augment_patch(image, rng)
        else:
            view1 = plain_patch(image)
            view2 = view1.clone()

        return view1, view2, label


class SmallLiverEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.representation = nn.Linear(256, 256)
        self.projector = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )
        self.classifier = nn.Linear(256, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x).flatten(1)
        representation = F.relu(self.representation(features))
        embedding = F.normalize(self.projector(representation), dim=1)
        logits = self.classifier(representation)
        return embedding, logits


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).float().to(embeddings.device)
    self_mask = torch.eye(labels.shape[0], device=embeddings.device)
    positive_mask = positive_mask * (1.0 - self_mask)

    logits = torch.matmul(embeddings, embeddings.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (1.0 - self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positives_per_anchor = positive_mask.sum(dim=1)
    valid = positives_per_anchor > 0
    if not valid.any():
        return embeddings.new_tensor(0.0)

    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (
        positives_per_anchor + 1e-12
    )
    return -mean_log_prob_pos[valid].mean()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    temperature: float,
    ce_weight: float,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    total_supcon = 0.0
    total_ce = 0.0
    total_correct = 0
    total_items = 0

    for view1, view2, labels in loader:
        view1 = view1.to(device)
        view2 = view2.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        images = torch.cat([view1, view2], dim=0)
        repeated_labels = torch.cat([labels, labels], dim=0)
        embeddings, logits = model(images)

        supcon = supervised_contrastive_loss(embeddings, repeated_labels, temperature)
        ce = F.cross_entropy(logits[: labels.shape[0]], labels)
        loss = supcon + ce_weight * ce

        if train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            predictions = logits[: labels.shape[0]].argmax(dim=1)
            total_correct += int((predictions == labels).sum().item())
            total_items += int(labels.numel())
            total_loss += float(loss.item()) * labels.numel()
            total_supcon += float(supcon.item()) * labels.numel()
            total_ce += float(ce.item()) * labels.numel()

    return {
        "loss": total_loss / max(total_items, 1),
        "supcon_loss": total_supcon / max(total_items, 1),
        "ce_loss": total_ce / max(total_items, 1),
        "accuracy": total_correct / max(total_items, 1),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, float],
    train_cases: list[str],
    val_cases: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
            "train_cases": train_cases,
            "val_cases": val_cases,
            "class_names": ["non_liver", "liver"],
        },
        path,
    )


def write_log(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train liver supervised contrastive model.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260702)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(args.manifest)
    train_rows, val_rows, train_cases, val_cases = split_by_case(
        rows, args.val_fraction, args.seed
    )

    train_loader = DataLoader(
        PatchDataset(train_rows, train=True, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        PatchDataset(val_rows, train=False, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = SmallLiverEncoder(embedding_dim=args.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    config = {
        "device": str(device),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "train_cases": train_cases,
        "val_cases": val_cases,
        "args": vars(args),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    best_val_accuracy = -math.inf
    log_rows: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.temperature,
            args.ce_weight,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                None,
                device,
                args.temperature,
                args.ce_weight,
            )

        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_supcon_loss": train_metrics["supcon_loss"],
            "train_ce_loss": train_metrics["ce_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_supcon_loss": val_metrics["supcon_loss"],
            "val_ce_loss": val_metrics["ce_loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        log_rows.append(row)
        write_log(args.output_dir / "train_log.csv", log_rows)

        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.3f}"
        )

        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            args,
            epoch,
            val_metrics,
            train_cases,
            val_cases,
        )
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                args,
                epoch,
                val_metrics,
                train_cases,
                val_cases,
            )

    print(f"best val accuracy: {best_val_accuracy:.3f}")
    print(f"model output: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
