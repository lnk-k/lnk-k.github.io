#!/usr/bin/env python3
"""Small 2D U-Net and F2 utilities for liver segmentation."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class LiverUNet2D(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = [base_channels * factor for factor in (1, 2, 4, 8, 16)]
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.bottleneck = ConvBlock(c4, c5)
        self.pool = nn.MaxPool2d(2)

        self.up4 = UpBlock(c5, c4, c4)
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.output = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))
        x = self.up4(bottleneck, e4)
        x = self.up3(x, e3)
        x = self.up2(x, e2)
        x = self.up1(x, e1)
        return self.output(x)


def soft_f2_score(
    logits: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Differentiable batch-level F2, matching the aggregate validation metric."""
    probability = torch.sigmoid(logits)
    target = target.to(dtype=probability.dtype)
    true_positive = (probability * target).sum()
    false_positive = (probability * (1.0 - target)).sum()
    false_negative = ((1.0 - probability) * target).sum()
    return (5.0 * true_positive + epsilon) / (
        5.0 * true_positive + 4.0 * false_negative + false_positive + epsilon
    )


def soft_f2_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - soft_f2_score(logits, target)


def binary_counts(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[int, int, int]:
    prediction = prediction.bool()
    target = target.bool()
    true_positive = int((prediction & target).sum().item())
    false_positive = int((prediction & ~target).sum().item())
    false_negative = int((~prediction & target).sum().item())
    return true_positive, false_positive, false_negative


def metrics_from_counts(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> dict[str, float]:
    epsilon = 1e-12
    recall = true_positive / max(true_positive + false_negative, 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    dice = (2.0 * true_positive) / max(
        2.0 * true_positive + false_positive + false_negative,
        epsilon,
    )
    f2 = (5.0 * true_positive) / max(
        5.0 * true_positive + 4.0 * false_negative + false_positive,
        epsilon,
    )
    return {
        "recall": recall,
        "precision": precision,
        "dice": dice,
        "f2": f2,
    }
