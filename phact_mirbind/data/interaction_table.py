"""One CSV/TSV row per interaction, with JSON arrays for position scores."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from phact_mirbind.data.columns import BASES, resolve_mirna_column


MISSING = {"", "NA", "NaN", "nan", "None", "null"}
CONSERVATION_COLUMNS = {"phylop": "gene_phyloP", "phastcons": "gene_phastCons"}


@dataclass
class Interaction:
    id: str
    gene: str
    mirna: str
    label: float | None
    mirna_phact: np.ndarray
    target_phact: np.ndarray
    conservation: dict[str, np.ndarray]
    metadata: dict[str, str]


def table_header(path: str | Path) -> tuple[list[str], str]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        line = handle.readline()
    delimiter = "\t" if "\t" in line else ","
    header = next(csv.reader([line], delimiter=delimiter), [])
    if not header or len(header) != len(set(header)):
        raise ValueError(f"{path}: empty or duplicate column names")
    return header, delimiter


def numeric_array(value: str, column: str) -> np.ndarray | None:
    if value.strip() in MISSING:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, list) or any(isinstance(x, (list, dict, bool)) for x in parsed):
        raise ValueError(f"{column} must contain an array of numbers or NaN")
    array = np.asarray(parsed, dtype=np.float64)
    if np.isinf(array).any():
        raise ValueError(f"{column} contains infinite scores")
    return array


def score_profile(
    row: dict[str, str], axis: str, sequence_length: int, model_length: int,
) -> np.ndarray:
    columns = [f"{axis}_phact_{base}" for base in BASES]
    present = [column in row for column in columns]
    if any(present) and not all(present):
        raise ValueError(f"{axis} requires all four columns: {', '.join(columns)}")
    result = np.full((model_length, 4), np.nan, dtype=np.float64)
    if not any(present):
        return result
    arrays = [numeric_array(row[column], column) for column in columns]
    if all(array is None for array in arrays):
        return result
    if any(array is None for array in arrays):
        raise ValueError(f"{axis}: provide all four arrays or leave all four empty")
    lengths = {len(array) for array in arrays}
    allowed = {sequence_length, model_length} if axis == "mirna" else {model_length}
    if len(lengths) != 1 or not lengths <= allowed:
        raise ValueError(f"{axis} score arrays must have the same length, one of {sorted(allowed)}; got {sorted(lengths)}")
    count = min(sequence_length, model_length)
    result[:count] = np.stack(arrays, axis=1)[:count]
    missing = np.isnan(result)
    if (missing.any(axis=1) != missing.all(axis=1)).any():
        raise ValueError(f"{axis}: each position must contain four scores or four NaNs")
    return result


def normalized_sequence(value: str, column: str) -> str:
    sequence = value.strip().upper().replace("U", "T")
    if not sequence or set(sequence) - set("ACGTN"):
        raise ValueError(f"{column} must be a nonempty A/C/G/T/U/N sequence")
    return sequence


def iter_interactions(
    path: str | Path, *, required_axes: Sequence[str] = ("mirna", "target"),
    require_labels: bool = True, conservation_features: Sequence[str] = (),
    mirna_length: int = 28, target_length: int = 50,
) -> Iterator[Interaction]:
    path = Path(path)
    header, delimiter = table_header(path)
    mirna_column = resolve_mirna_column({column: i for i, column in enumerate(header)})
    required = ["gene", mirna_column]
    if require_labels:
        required.append("label")
    for axis in required_axes:
        required.extend(f"{axis}_phact_{base}" for base in BASES)
    for feature in conservation_features:
        required.append(CONSERVATION_COLUMNS[feature])
    absent = [column for column in required if column not in header]
    if absent:
        raise ValueError(f"{path}: missing columns: {', '.join(absent)}")
    seen_ids: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for number, row in enumerate(reader, 1):
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("row width does not match the header")
                sample_id = row.get("id", str(number)).strip()
                if not sample_id or sample_id in seen_ids:
                    raise ValueError(f"empty or duplicate id: {sample_id!r}")
                seen_ids.add(sample_id)
                gene = normalized_sequence(row["gene"], "gene")
                mirna = normalized_sequence(row[mirna_column], mirna_column)
                if len(gene) != target_length:
                    raise ValueError(f"gene must have {target_length} nucleotides, got {len(gene)}")
                label = None if row.get("label", "").strip() in MISSING else float(row["label"])
                if label is not None and label not in (0., 1.):
                    raise ValueError("label must be 0 or 1")
                if require_labels and label is None:
                    raise ValueError("label must be 0 or 1")
                mirna_scores = score_profile(row, "mirna", len(mirna), mirna_length)
                target_scores = score_profile(row, "target", len(gene), target_length)
                conservation = {}
                for feature, column in CONSERVATION_COLUMNS.items():
                    scores = numeric_array(row.get(column, ""), column)
                    if scores is None:
                        scores = np.full(target_length, np.nan)
                    if len(scores) != target_length:
                        raise ValueError(f"{column} must have {target_length} positions")
                    if feature == "phastcons" and ((scores < 0) | (scores > 1)).any():
                        raise ValueError("gene_phastCons finite scores must be in [0,1]")
                    conservation[feature] = scores
                yield Interaction(sample_id, gene, mirna, label, mirna_scores,
                                  target_scores, conservation, row)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}, data row {number}: {exc}") from exc


def json_scores(values: np.ndarray) -> str:
    return json.dumps([float(x) if np.isfinite(x) else float("nan") for x in values], separators=(",", ":"))
