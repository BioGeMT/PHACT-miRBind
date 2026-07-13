"""Seq + PHACT pairwise miRBind CNN."""

from __future__ import annotations

from phact_mirbind.models.conservation import PairwiseConservationCNN


class PairwisePhactCNN(PairwiseConservationCNN):
    """Pairwise miRBind CNN with miRNA and target PHACT score channels."""

    def __init__(
        self,
        *,
        num_pair_classes: int,
        phact_channel_count: int = 8,
        target_length: int = 50,
        mirna_length: int = 28,
        embedding_dim: int = 8,
        dropout_rate: float = 0.2,
        filter_sizes: tuple[int, ...] = (128, 64, 32),
        kernel_sizes: tuple[int, ...] = (6, 3, 3),
    ) -> None:
        super().__init__(
            num_pair_classes=num_pair_classes,
            conservation_channel_count=phact_channel_count,
            target_length=target_length,
            mirna_length=mirna_length,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
            filter_sizes=filter_sizes,
            kernel_sizes=kernel_sizes,
        )
        self.phact_channel_count = phact_channel_count
