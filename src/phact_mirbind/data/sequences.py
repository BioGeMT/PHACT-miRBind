"""Sequence normalization helpers."""

from __future__ import annotations


def normalize_sequence(sequence: str, desired_length: int) -> str:
    normalized = sequence.upper().replace("U", "T")
    normalized = "".join(base if base in "ATCG" else "N" for base in normalized)
    return normalized[:desired_length].ljust(desired_length, "N")
