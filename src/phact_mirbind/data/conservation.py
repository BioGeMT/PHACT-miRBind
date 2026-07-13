"""phyloP/phastCons parsing for Manakov target windows."""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np

from phact_mirbind.data.columns import (
    GENE_PHASTCONS_COLUMN,
    GENE_PHYLOP_COLUMN,
    LABEL_COLUMN,
    TARGET_SEQUENCE_COLUMN,
    column_index,
    resolve_mirna_column,
)

CONSERVATION_FEATURES = ("phylop", "phastcons")
FEATURE_TO_COLUMN = {
    "phylop": GENE_PHYLOP_COLUMN,
    "phastcons": GENE_PHASTCONS_COLUMN,
}


def parse_conservation_features(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    else:
        parts = tuple(str(part).strip().lower() for part in value if str(part).strip())

    invalid = [feature for feature in parts if feature not in CONSERVATION_FEATURES]
    if invalid:
        choices = ", ".join(CONSERVATION_FEATURES)
        raise ValueError(f"Unknown conservation feature(s): {invalid}. Choices: {choices}")
    if not parts:
        raise ValueError("At least one conservation feature is required")
    return parts


@dataclass(frozen=True)
class ConservationSchema:
    gene_idx: int
    mirna_idx: int
    label_idx: int
    feature_indices: tuple[int, ...]
    feature_columns: tuple[str, ...]

    @classmethod
    def from_header(cls, header: list[str], features: tuple[str, ...]) -> "ConservationSchema":
        columns = column_index(header)
        missing_required = [
            column
            for column in (TARGET_SEQUENCE_COLUMN, LABEL_COLUMN)
            if column not in columns
        ]
        if missing_required:
            raise ValueError(f"Missing required columns: {', '.join(missing_required)}")

        mirna_column = resolve_mirna_column(columns)
        feature_columns = tuple(FEATURE_TO_COLUMN[feature] for feature in features)
        missing_features = [
            column for column in feature_columns if column not in columns
        ]
        if missing_features:
            raise ValueError(
                f"Missing conservation columns: {', '.join(missing_features)}"
            )

        return cls(
            gene_idx=columns[TARGET_SEQUENCE_COLUMN],
            mirna_idx=columns[mirna_column],
            label_idx=columns[LABEL_COLUMN],
            feature_indices=tuple(columns[column] for column in feature_columns),
            feature_columns=feature_columns,
        )


def conservation_matrix(
    row: list[str],
    feature_indices: tuple[int, ...],
    features: tuple[str, ...],
    target_length: int,
) -> np.ndarray:
    matrix = np.empty((target_length, len(features)), dtype=np.float32)
    for feature_idx, (column_idx, feature) in enumerate(zip(feature_indices, features)):
        matrix[:, feature_idx] = parse_conservation_track(
            row[column_idx],
            feature,
            target_length,
        )
    return matrix


def parse_conservation_track(value: str, feature: str, target_length: int) -> np.ndarray:
    default = 0.0 if feature == "phylop" else 0.5
    scores = np.full((target_length,), default, dtype=np.float32)
    if value in {"", "NA", "NaN", "nan", "None", "null"}:
        return scores

    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return scores

    limit = min(len(parsed), target_length)
    if limit:
        scores[:limit] = np.asarray(parsed[:limit], dtype=np.float32)

    if feature == "phylop":
        scores = np.clip(scores / 10.0, -1.0, 1.0)
    return scores.astype(np.float32)
