"""Losses for noisy positive/sampled-negative interaction labels."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLabelSmoothingBCEWithLogitsLoss(nn.Module):
    """BCE with separate smoothing for observed positives and sampled negatives."""

    def __init__(
        self,
        *,
        negative_smoothing: float = 0.0,
        positive_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("negative_smoothing", negative_smoothing),
            ("positive_smoothing", positive_smoothing),
        ):
            if not 0.0 <= value < 0.5:
                raise ValueError(f"{name} must be in [0, 0.5)")
        self.negative_smoothing = negative_smoothing
        self.positive_smoothing = positive_smoothing
        if focal_gamma < 0.0:
            raise ValueError("focal_gamma must be non-negative")
        self.focal_gamma = focal_gamma

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        targets = torch.where(
            labels > 0.5,
            torch.full_like(labels, 1.0 - self.positive_smoothing),
            torch.full_like(labels, self.negative_smoothing),
        )
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if self.focal_gamma:
            probabilities = torch.sigmoid(logits)
            target_probability = torch.where(
                labels > 0.5,
                probabilities,
                1.0 - probabilities,
            )
            loss = loss * (1.0 - target_probability).pow(self.focal_gamma)
        if sample_weights is None:
            return loss.mean()
        if sample_weights.shape != loss.shape:
            raise ValueError("sample_weights must match the per-row loss shape")
        denominator = sample_weights.sum()
        if denominator <= 0:
            raise ValueError("sample_weights must have a positive sum")
        return (loss * sample_weights).sum() / denominator
