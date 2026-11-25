"""miRNA/target pair-grid encoding."""

from __future__ import annotations

import numpy as np

from phact_mirbind.data.columns import (
    BASES,
    PADDING_PAIR_INDEX,
    build_pair_to_index,
)


def encode_pair_indices(
    target_sequence: str,
    mirna_sequence: str,
    *,
    target_length: int,
    mirna_length: int,
    pair_to_index: dict[tuple[str, str], int] | None = None,
) -> np.ndarray:
    """Encode miRNA-base/target-base pairs as integer indices.

    The output shape is ``[mirna_length, target_length]``. Unknown bases and
    padded positions use the explicit padding pair index.
    """
    pair_to_index = pair_to_index or build_pair_to_index()
    indices = np.full((mirna_length, target_length), PADDING_PAIR_INDEX, dtype=np.int64)

    for mirna_pos, mirna_base in enumerate(mirna_sequence[:mirna_length]):
        mirna_base = mirna_base if mirna_base in BASES else "N"
        for target_pos, target_base in enumerate(target_sequence[:target_length]):
            target_base = target_base if target_base in BASES else "N"
            if target_base == "N" or mirna_base == "N":
                continue
            indices[mirna_pos, target_pos] = pair_to_index[(mirna_base, target_base)]

    return indices
