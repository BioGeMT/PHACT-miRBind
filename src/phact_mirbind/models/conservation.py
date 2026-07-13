"""Seq + target-conservation pairwise miRBind CNN."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from phact_mirbind.models.blocks import (
    make_conv_stack,
    run_conv_stack,
    validate_conv_config,
)


class PairwiseConservationCNN(nn.Module):
    """Pairwise miRBind CNN with broadcast target conservation channels."""

    def __init__(
        self,
        *,
        num_pair_classes: int,
        conservation_channel_count: int,
        target_length: int = 50,
        mirna_length: int = 28,
        embedding_dim: int = 8,
        dropout_rate: float = 0.2,
        filter_sizes: tuple[int, ...] = (128, 64, 32),
        kernel_sizes: tuple[int, ...] = (6, 3, 3),
    ) -> None:
        super().__init__()
        validate_conv_config(filter_sizes, kernel_sizes)
        if conservation_channel_count < 1:
            raise ValueError("conservation_channel_count must be positive")

        self.num_pair_classes = num_pair_classes
        self.conservation_channel_count = conservation_channel_count
        self.target_length = target_length
        self.mirna_length = mirna_length
        self.embedding_dim = embedding_dim

        self.pair_linear = nn.Linear(num_pair_classes, embedding_dim)
        (
            self.conv_layers,
            self.bn_layers,
            self.pool_layers,
            self.dropout_layers,
        ) = make_conv_stack(
            in_channels=embedding_dim + conservation_channel_count,
            filter_sizes=filter_sizes,
            kernel_sizes=kernel_sizes,
            dropout_rate=dropout_rate,
        )

        self.flat_features = self._compute_flat_features()
        self.fc1 = nn.Linear(self.flat_features, 30)
        self.bn_fc = nn.BatchNorm1d(30)
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(30, 1)

    def _compute_flat_features(self) -> int:
        pair_indices = torch.zeros(1, self.mirna_length, self.target_length).long()
        conservation = torch.zeros(
            1,
            self.conservation_channel_count,
            self.mirna_length,
            self.target_length,
        )
        x = self._build_conv_input(pair_indices, conservation)
        x = run_conv_stack(x, self.conv_layers, self.bn_layers, self.pool_layers)
        return x.numel()

    def forward(
        self,
        pair_indices: torch.Tensor,
        conservation: torch.Tensor,
    ) -> torch.Tensor:
        features = self.extract_features(pair_indices, conservation)
        return self.classify_features(features)

    def extract_features(
        self,
        pair_indices: torch.Tensor,
        conservation: torch.Tensor,
    ) -> torch.Tensor:
        x = self._build_conv_input(pair_indices, conservation)
        x = run_conv_stack(
            x,
            self.conv_layers,
            self.bn_layers,
            self.pool_layers,
            self.dropout_layers,
        )
        return x.contiguous().view(x.size(0), -1)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout_fc(F.leaky_relu(self.bn_fc(self.fc1(features)), 0.1))
        return self.fc2(x).squeeze(-1)

    def predict_proba(
        self,
        pair_indices: torch.Tensor,
        conservation: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(pair_indices, conservation))

    def _build_conv_input(
        self,
        pair_indices: torch.Tensor,
        conservation: torch.Tensor,
    ) -> torch.Tensor:
        if conservation.shape[1] != self.conservation_channel_count:
            raise ValueError(
                f"Expected {self.conservation_channel_count} conservation channels, "
                f"got {conservation.shape[1]}"
            )

        pair_onehot = F.one_hot(
            pair_indices.long(),
            num_classes=self.num_pair_classes,
        ).float()
        sequence_embedding = self.pair_linear(pair_onehot).permute(0, 3, 1, 2)
        return torch.cat([sequence_embedding, conservation.float()], dim=1)
