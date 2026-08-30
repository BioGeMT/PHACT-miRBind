"""miRNA/target pair-grid encoding."""

from __future__ import annotations

import numpy as np
import torch

from phact_mirbind.data.columns import (
    BASES,
    PAIR_BASES,
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


def decode_pair_base_ids(
    pair_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover base identities and masks from a batch of pair grids.

    Base IDs follow ``PAIR_BASES`` (A, T, C, G). Unknown and padded positions
    have a false mask; their placeholder base ID must not be consumed.
    """
    if pair_indices.ndim != 3:
        raise ValueError("pair_indices must have shape [batch, mirna, target]")

    valid = pair_indices < PADDING_PAIR_INDEX - 1
    mirna_mask = valid.any(dim=2)
    target_mask = valid.any(dim=1)

    first_target = valid.to(torch.int64).argmax(dim=2, keepdim=True)
    mirna_pairs = pair_indices.gather(2, first_target).squeeze(2)
    mirna_base_ids = torch.div(mirna_pairs, len(PAIR_BASES), rounding_mode="floor")

    transposed_pairs = pair_indices.transpose(1, 2)
    first_mirna = valid.transpose(1, 2).to(torch.int64).argmax(dim=2, keepdim=True)
    target_pairs = transposed_pairs.gather(2, first_mirna).squeeze(2)
    target_base_ids = target_pairs.remainder(len(PAIR_BASES))

    # Invalid grid cells encode padding/unknown rather than a fifth base. Keep
    # the accompanying masks authoritative, but return safe placeholder IDs so
    # callers can perform an embedding lookup before applying those masks.
    mirna_base_ids = mirna_base_ids.masked_fill(~mirna_mask, 0)
    target_base_ids = target_base_ids.masked_fill(~target_mask, 0)

    return (
        mirna_base_ids.long(),
        mirna_mask,
        target_base_ids.long(),
        target_mask,
    )
