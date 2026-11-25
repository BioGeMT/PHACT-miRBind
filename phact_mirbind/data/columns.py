"""Shared dataset constants and column helpers."""

from __future__ import annotations

BASES = ("A", "C", "G", "T")
PAIR_BASES = ("A", "T", "C", "G")
PADDING_PAIR_INDEX = 17


def build_pair_to_index() -> dict[tuple[str, str], int]:
    """Return miRNA-base/target-base pair indices matching miRBind."""
    return {
        (mirna_base, target_base): idx
        for idx, (mirna_base, target_base) in enumerate(
            (mirna_base, target_base)
            for mirna_base in PAIR_BASES
            for target_base in PAIR_BASES
        )
    }
