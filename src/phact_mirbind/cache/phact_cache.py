"""Compact .pt cache writer for pair grids plus miRNA/target PHACT scores."""

from __future__ import annotations

import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from phact_mirbind.cache.manifest import CACHE_VERSION, write_cache_log_header
from phact_mirbind.cache.pair_cache import PairSchema
from phact_mirbind.data.columns import BASES
from phact_mirbind.data.pair_encoding import encode_pair_indices
from phact_mirbind.data.sequences import normalize_sequence

DEFAULT_MIRNA_PHACT_FILE = Path(
    "reports/phact_score_ranges/"
    "phact_mirna_manakov_position_qntnorm_transformed_scores.tsv"
)
DEFAULT_TARGET_PHACT_FILE = Path(
    "reports/phact_score_ranges/"
    "phact_target_manakov_position_qntnorm_transformed_scores.tsv"
)
MISSING_SCORE_VALUE = 0.5
MISSING_REDUCED_VALUE = 0.0
MISSING_TOKENS = {"", "NA", "NaN", "nan", "None", "null"}
TARGET_SCORE_COLUMNS = tuple(f"score_{base}" for base in BASES)
PHACT_REDUCTIONS = ("nucleotide", "actual_margin", "alt_mean")


def write_phact_cache_from_tsv(
    input_file: str | Path,
    output_dir: str | Path,
    *,
    output_prefix: str,
    phact_split: str,
    phact_model: str | None = None,
    phact_models: str | list[str] | tuple[str, ...] | None = None,
    target_phact_models: str | list[str] | tuple[str, ...] | None = None,
    mirna_phact_file: str | Path = DEFAULT_MIRNA_PHACT_FILE,
    target_phact_file: str | Path = DEFAULT_TARGET_PHACT_FILE,
    row_id_column: str = "manakov_row_id",
    shard_size: int = 100_000,
    target_length: int = 50,
    mirna_length: int = 28,
    phact_dtype: str = "float16",
    phact_reduction: str = "nucleotide",
) -> Path:
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    mirna_phact_file = Path(mirna_phact_file)
    target_phact_file = Path(target_phact_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    if phact_dtype not in {"float16", "float32"}:
        raise ValueError("phact_dtype must be float16 or float32")
    validate_phact_reduction(phact_reduction)

    requested_mirna_models = parse_phact_models(
        phact_models if phact_models is not None else (phact_model or "CountNodes_2")
    )

    selected_mirna_models, mirna_score_columns = resolve_named_phact_columns(
        read_header(mirna_phact_file),
        requested_mirna_models,
        axis_name="miRNA",
        path=mirna_phact_file,
    )
    target_header = read_header(target_phact_file)
    if target_phact_models is None:
        requested_target_models = (
            requested_mirna_models
            if has_named_phact_columns(target_header)
            else ("target_score",)
        )
    else:
        requested_target_models = parse_phact_models(target_phact_models)
    selected_target_models, target_score_columns = resolve_target_phact_columns(
        target_header,
        requested_target_models,
        target_phact_file,
    )

    log_path = output_dir / f"{output_prefix}_cache_log.tsv"
    manifest_path = output_dir / f"{output_prefix}_manifest.json"
    shards: list[dict[str, object]] = []
    total_rows = 0
    total_positive = 0
    shard_idx = 0
    shard_started = time.perf_counter()
    started = time.perf_counter()

    pairs = np.empty((shard_size, mirna_length, target_length), dtype=np.uint8)
    mirna_channel_count = reduced_channel_count(mirna_score_columns, phact_reduction)
    target_channel_count = reduced_channel_count(target_score_columns, phact_reduction)
    mirna_missing_channel_count = len(mirna_score_columns) // len(BASES)
    target_missing_channel_count = len(target_score_columns) // len(BASES)
    mirna_phact = np.empty(
        (shard_size, mirna_length, mirna_channel_count),
        dtype=phact_dtype,
    )
    target_phact = np.empty(
        (shard_size, target_length, target_channel_count),
        dtype=phact_dtype,
    )
    mirna_phact_missing = np.empty(
        (shard_size, mirna_length, mirna_missing_channel_count),
        dtype=np.uint8,
    )
    target_phact_missing = np.empty(
        (shard_size, target_length, target_missing_channel_count),
        dtype=np.uint8,
    )
    labels = np.empty((shard_size,), dtype=np.float32)
    write_cache_log_header(log_path)

    mirna_stream = PhactPositionScoreStream(
        mirna_phact_file,
        phact_split=phact_split,
        position_column="mirna_position_1based",
        score_columns=mirna_score_columns,
        position_count=mirna_length,
        dtype=phact_dtype,
        phact_reduction=phact_reduction,
    )
    target_stream = PhactPositionScoreStream(
        target_phact_file,
        phact_split=phact_split,
        position_column="target_position_1based",
        score_columns=target_score_columns,
        position_count=target_length,
        dtype=phact_dtype,
        phact_reduction=phact_reduction,
    )

    try:
        with input_file.open("r", newline="", buffering=1024 * 1024) as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader)
            pair_schema = PairSchema.from_header(header)
            row_id_idx = header.index(row_id_column) if row_id_column in header else None
            shard_rows = 0
            shard_positive = 0
            previous_source_row_id = 0

            for input_row_number, row in enumerate(reader, start=1):
                source_row_id = (
                    int(row[row_id_idx]) if row_id_idx is not None else input_row_number
                )
                if source_row_id <= previous_source_row_id:
                    raise ValueError(
                        "Input rows must be sorted by ascending source row id for "
                        f"streaming PHACT joins; got {source_row_id} after "
                        f"{previous_source_row_id}"
                    )
                previous_source_row_id = source_row_id

                target_sequence = normalize_sequence(
                    row[pair_schema.gene_idx],
                    target_length,
                )
                mirna_sequence = normalize_sequence(
                    row[pair_schema.mirna_idx],
                    mirna_length,
                )
                pairs[shard_rows] = encode_pair_indices(
                    target_sequence,
                    mirna_sequence,
                    target_length=target_length,
                    mirna_length=mirna_length,
                ).astype(np.uint8)
                mirna_scores, mirna_missing = mirna_stream.read_scores(source_row_id)
                target_scores, target_missing = target_stream.read_scores(source_row_id)
                mirna_phact[shard_rows] = mirna_scores
                target_phact[shard_rows] = target_scores
                mirna_phact_missing[shard_rows] = mirna_missing
                target_phact_missing[shard_rows] = target_missing
                label = float(row[pair_schema.label_idx])
                labels[shard_rows] = label

                shard_rows += 1
                shard_positive += int(label)
                total_rows += 1
                total_positive += int(label)

                if shard_rows == shard_size:
                    shards.append(
                        _flush_shard(
                            output_dir,
                            output_prefix,
                            shard_idx,
                            pairs,
                            mirna_phact,
                            target_phact,
                            mirna_phact_missing,
                            target_phact_missing,
                            labels,
                            shard_rows,
                            shard_positive,
                            shard_started,
                            log_path,
                        )
                    )
                    shard_idx += 1
                    shard_rows = 0
                    shard_positive = 0
                    shard_started = time.perf_counter()

            if shard_rows:
                shards.append(
                    _flush_shard(
                        output_dir,
                        output_prefix,
                        shard_idx,
                        pairs,
                        mirna_phact,
                        target_phact,
                        mirna_phact_missing,
                        target_phact_missing,
                        labels,
                        shard_rows,
                        shard_positive,
                        shard_started,
                        log_path,
                    )
                )
    finally:
        mirna_stream.close()
        target_stream.close()

    manifest = {
        "cache_version": CACHE_VERSION,
        "cache_type": "phact",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": str(input_file),
        "phact_split": phact_split,
        "target_length": target_length,
        "mirna_length": mirna_length,
        "source_columns": {
            "target_sequence": pair_schema.gene_column,
            "mirna_sequence": pair_schema.mirna_column,
            "label": pair_schema.label_column,
            "row_id": row_id_column if row_id_idx is not None else None,
            "mirna_phact": list(mirna_score_columns),
            "target_phact": list(target_score_columns),
        },
        "pair_dtype": "uint8",
        "phact_dtype": phact_dtype,
        "phact_reduction": phact_reduction,
        "phact_model": selected_mirna_models[0] if len(selected_mirna_models) == 1 else None,
        "phact_models": list(selected_mirna_models),
        "target_phact_models": list(selected_target_models),
        "phact_axis_channel_counts": {
            "mirna": mirna_channel_count,
            "target": target_channel_count,
        },
        "phact_axis_missingness_channel_counts": {
            "mirna": mirna_missing_channel_count,
            "target": target_missing_channel_count,
        },
        "phact_axis_total_channel_counts": {
            "mirna": mirna_channel_count + mirna_missing_channel_count,
            "target": target_channel_count + target_missing_channel_count,
        },
        "phact_score_channel_order": phact_channel_order(
            selected_mirna_models,
            selected_target_models,
            phact_reduction,
        ),
        "phact_channel_order": [
            *phact_axis_channel_order(
                "mirna",
                selected_mirna_models,
                phact_reduction,
            ),
            *(f"mirna_{model}_missing" for model in selected_mirna_models),
            *phact_axis_channel_order(
                "target",
                selected_target_models,
                phact_reduction,
            ),
            *(f"target_{model}_missing" for model in selected_target_models),
        ],
        "phact_missingness_channel_order": phact_missingness_channel_order(
            selected_mirna_models,
            selected_target_models,
        ),
        "phact_missing": {
            "fill_value": missing_fill_value(phact_reduction),
            "partial_missing": "error",
            "mirna_positions_filled": mirna_stream.missing_score_positions,
            "target_positions_filled": target_stream.missing_score_positions,
            "mirna_score_groups_filled": mirna_stream.missing_score_positions,
            "target_score_groups_filled": target_stream.missing_score_positions,
            "mask_value": 1,
        },
        "label_dtype": "float32",
        "row_count": total_rows,
        "positive_rows": total_positive,
        "shard_size": shard_size,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "shards": shards,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


class PhactPositionScoreStream:
    """Read compact position-score rows for one split in ascending row-id order."""

    def __init__(
        self,
        path: Path,
        *,
        phact_split: str,
        position_column: str,
        score_columns: tuple[str, ...],
        position_count: int,
        dtype: str,
        phact_reduction: str,
    ) -> None:
        self.path = path
        self.phact_split = phact_split
        self.position_count = position_count
        self.dtype = dtype
        self.phact_reduction = phact_reduction
        self.handle = path.open("r", newline="", buffering=1024 * 1024)
        self.reader = csv.reader(self.handle, delimiter="\t")
        header = next(self.reader)
        columns = {column: idx for idx, column in enumerate(header)}
        required = (
            "split",
            "manakov_row_id",
            position_column,
            "actual_nt",
            *score_columns,
        )
        missing = [column for column in required if column not in columns]
        if missing:
            self.close()
            raise ValueError(f"{path} missing columns: {', '.join(missing)}")

        self.split_idx = columns["split"]
        self.row_id_idx = columns["manakov_row_id"]
        self.position_idx = columns[position_column]
        self.actual_nt_idx = columns["actual_nt"]
        self.score_indices = tuple(columns[column] for column in score_columns)
        if len(self.score_indices) % len(BASES):
            self.close()
            raise ValueError(
                f"{path} score column count must be a multiple of {len(BASES)}"
            )
        self.current_row: list[str] | None = None
        self.seen_split = False
        self.finished_split = False
        self.previous_row_id = 0
        self.missing_score_positions = 0
        self._advance()

    def close(self) -> None:
        self.handle.close()

    def read_scores(self, row_id: int) -> tuple[np.ndarray, np.ndarray]:
        if row_id <= self.previous_row_id:
            raise ValueError(
                f"Requested non-ascending PHACT row id {row_id} from {self.path}"
            )

        while self.current_row is not None and self.current_row_id < row_id:
            self._skip_current_source_row()

        if self.current_row is None:
            raise ValueError(
                f"Missing {self.phact_split} PHACT scores for row id {row_id} "
                f"in {self.path}"
            )
        if self.current_row_id > row_id:
            raise ValueError(
                f"Missing {self.phact_split} PHACT scores for row id {row_id} "
                f"in {self.path}; next available row id is {self.current_row_id}"
            )

        scores = np.empty(
            (self.position_count, reduced_channel_count(self.score_indices, self.phact_reduction)),
            dtype=self.dtype,
        )
        missingness = np.empty(
            (self.position_count, len(self.score_indices) // len(BASES)),
            dtype=np.uint8,
        )
        seen_positions = np.zeros(self.position_count, dtype=np.bool_)
        while self.current_row is not None and self.current_row_id == row_id:
            position = int(self.current_row[self.position_idx])
            if position < 1 or position > self.position_count:
                raise ValueError(
                    f"{self.path} has out-of-range position {position} for row id "
                    f"{row_id}; expected 1..{self.position_count}"
                )
            position_idx = position - 1
            if seen_positions[position_idx]:
                raise ValueError(
                    f"{self.path} has duplicate position {position} for row id {row_id}"
                )
            values, missing_groups = parse_score_values(
                [self.current_row[idx] for idx in self.score_indices],
                self.current_row[self.actual_nt_idx],
                self.phact_reduction,
                self.path,
                row_id,
                position,
            )
            scores[position_idx] = values
            missingness[position_idx] = missing_groups
            seen_positions[position_idx] = True
            self.missing_score_positions += sum(missing_groups)
            self._advance()

        if not bool(seen_positions.all()):
            missing = np.flatnonzero(~seen_positions) + 1
            preview = ",".join(str(int(pos)) for pos in missing[:10])
            raise ValueError(
                f"{self.path} is missing positions for row id {row_id}: {preview}"
            )

        self.previous_row_id = row_id
        return scores, missingness

    @property
    def current_row_id(self) -> int:
        assert self.current_row is not None
        return int(self.current_row[self.row_id_idx])

    def _skip_current_source_row(self) -> None:
        row_id = self.current_row_id
        while self.current_row is not None and self.current_row_id == row_id:
            self._advance()

    def _advance(self) -> None:
        if self.finished_split:
            self.current_row = None
            return

        for row in self.reader:
            row_split = row[self.split_idx]
            if row_split != self.phact_split:
                if self.seen_split:
                    self.finished_split = True
                    self.current_row = None
                    return
                continue

            self.seen_split = True
            self.current_row = row
            return

        self.finished_split = True
        self.current_row = None


def read_header(path: Path) -> list[str]:
    with path.open("r", newline="") as handle:
        return next(csv.reader(handle, delimiter="\t"))


def safe_phact_model_name(model: str) -> str:
    safe = model.replace(".", "p")
    return re.sub(r"[^0-9A-Za-z_]+", "_", safe)


def parse_phact_models(
    value: str | list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    if isinstance(value, str):
        models = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        models = tuple(str(part).strip() for part in value if str(part).strip())
    if not models:
        raise ValueError("At least one PHACT model is required")
    return models


def validate_phact_reduction(phact_reduction: str) -> None:
    if phact_reduction not in PHACT_REDUCTIONS:
        choices = ", ".join(PHACT_REDUCTIONS)
        raise ValueError(f"phact_reduction must be one of: {choices}")


def missing_fill_value(phact_reduction: str) -> float:
    validate_phact_reduction(phact_reduction)
    if phact_reduction in {"nucleotide", "alt_mean"}:
        return MISSING_SCORE_VALUE
    return MISSING_REDUCED_VALUE


def reduced_channel_count(
    score_columns: tuple[object, ...],
    phact_reduction: str,
) -> int:
    validate_phact_reduction(phact_reduction)
    if len(score_columns) % len(BASES):
        raise ValueError(f"score column count must be a multiple of {len(BASES)}")
    if phact_reduction == "nucleotide":
        return len(score_columns)
    return len(score_columns) // len(BASES)


def phact_channel_order(
    mirna_models: tuple[str, ...],
    target_models: tuple[str, ...],
    phact_reduction: str,
) -> list[str]:
    if phact_reduction == "nucleotide":
        return [
            *(
                f"mirna_{model}_{base}"
                for model in mirna_models
                for base in BASES
            ),
            *(
                f"target_{model}_{base}"
                for model in target_models
                for base in BASES
            ),
        ]
    return [
        *(f"mirna_{model}_{phact_reduction}" for model in mirna_models),
        *(f"target_{model}_{phact_reduction}" for model in target_models),
    ]


def phact_axis_channel_order(
    axis: str,
    models: tuple[str, ...],
    phact_reduction: str,
) -> list[str]:
    if phact_reduction == "nucleotide":
        return [
            f"{axis}_{model}_{base}"
            for model in models
            for base in BASES
        ]
    return [f"{axis}_{model}_{phact_reduction}" for model in models]


def phact_missingness_channel_order(
    mirna_models: tuple[str, ...],
    target_models: tuple[str, ...],
) -> list[str]:
    return [
        *(f"mirna_{model}_missing" for model in mirna_models),
        *(f"target_{model}_missing" for model in target_models),
    ]


def available_named_phact_models(header: list[str]) -> list[str]:
    models: list[str] = []
    seen: set[str] = set()
    for column in header:
        match = re.fullmatch(r"phact_(.+)_([ACGT])", column)
        if match and match.group(1) not in seen:
            model = match.group(1)
            models.append(model)
            seen.add(model)
    return models


def has_named_phact_columns(header: list[str]) -> bool:
    return any(column.startswith("phact_") for column in header)


def resolve_named_phact_columns(
    header: list[str],
    phact_models: tuple[str, ...],
    *,
    axis_name: str,
    path: Path,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    columns = set(header)
    selected_models: list[str] = []
    selected_columns: list[str] = []
    for requested_model in phact_models:
        model, score_columns = resolve_one_named_phact_model(
            columns,
            requested_model,
            available_named_phact_models(header),
            axis_name,
            path,
        )
        selected_models.append(model)
        selected_columns.extend(score_columns)
    return tuple(selected_models), tuple(selected_columns)


def resolve_one_named_phact_model(
    columns: set[str],
    requested_model: str,
    available_models: list[str],
    axis_name: str,
    path: Path,
) -> tuple[str, tuple[str, ...]]:
    raw_model = requested_model.removeprefix("phact_")
    candidates: list[str] = []
    for model in (raw_model, safe_phact_model_name(raw_model)):
        if model not in candidates:
            candidates.append(model)

    for model in candidates:
        score_columns = tuple(f"phact_{model}_{base}" for base in BASES)
        if all(column in columns for column in score_columns):
            return model, score_columns

    available = ", ".join(available_models)
    raise ValueError(
        f"PHACT model {requested_model!r} not found in {axis_name} PHACT file "
        f"{path}. Available named models: {available}"
    )


def resolve_target_phact_columns(
    header: list[str],
    target_phact_models: tuple[str, ...],
    path: Path,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    columns = set(header)
    if has_named_phact_columns(header):
        return resolve_named_phact_columns(
            header,
            target_phact_models,
            axis_name="target",
            path=path,
        )

    if len(target_phact_models) == 1 and all(
        column in columns for column in TARGET_SCORE_COLUMNS
    ):
        return (target_phact_models[0],), TARGET_SCORE_COLUMNS

    available = ", ".join(TARGET_SCORE_COLUMNS)
    raise ValueError(
        f"Target PHACT file {path} has only one unmodelled score group "
        f"({available}). Multiple target PHACT models require columns named "
        "phact_<model>_A, phact_<model>_C, phact_<model>_G, phact_<model>_T."
    )


def parse_score_values(
    values: list[str],
    actual_nt: str,
    phact_reduction: str,
    path: Path,
    row_id: int,
    position: int,
) -> tuple[list[float], list[int]]:
    parsed: list[float] = []
    missing_groups: list[int] = []
    actual_nt = actual_nt.strip().upper().replace("U", "T")
    for group_start in range(0, len(values), len(BASES)):
        group = values[group_start : group_start + len(BASES)]
        missing = [value in MISSING_TOKENS for value in group]
        if all(missing):
            if phact_reduction == "nucleotide":
                parsed.extend([missing_fill_value(phact_reduction)] * len(group))
            else:
                parsed.append(missing_fill_value(phact_reduction))
            missing_groups.append(1)
        elif any(missing):
            raise ValueError(
                f"{path} has partial missing PHACT scores for row id {row_id}, "
                f"position {position}: {group}"
            )
        else:
            scores = [float(value) for value in group]
            if phact_reduction == "nucleotide":
                parsed.extend(scores)
                missing_groups.append(0)
            elif actual_nt not in BASES:
                parsed.append(missing_fill_value(phact_reduction))
                missing_groups.append(1)
            elif phact_reduction == "alt_mean":
                actual_index = BASES.index(actual_nt)
                parsed.append(
                    sum(score for idx, score in enumerate(scores) if idx != actual_index)
                    / (len(BASES) - 1)
                )
                missing_groups.append(0)
            else:
                actual_index = BASES.index(actual_nt)
                best_alt = max(
                    score for idx, score in enumerate(scores) if idx != actual_index
                )
                parsed.append(scores[actual_index] - best_alt)
                missing_groups.append(0)
    return parsed, missing_groups


def _flush_shard(
    output_dir: Path,
    output_prefix: str,
    shard_idx: int,
    pairs: np.ndarray,
    mirna_phact: np.ndarray,
    target_phact: np.ndarray,
    mirna_phact_missing: np.ndarray,
    target_phact_missing: np.ndarray,
    labels: np.ndarray,
    row_count: int,
    positive_rows: int,
    shard_started: float,
    log_path: Path,
) -> dict[str, object]:
    shard_name = f"{output_prefix}_shard_{shard_idx:05d}.pt"
    shard_path = output_dir / shard_name
    torch.save(
        {
            "cache_version": CACHE_VERSION,
            "pair_indices": torch.from_numpy(pairs[:row_count].copy()),
            "mirna_phact": torch.from_numpy(mirna_phact[:row_count].copy()),
            "target_phact": torch.from_numpy(target_phact[:row_count].copy()),
            "mirna_phact_missing": torch.from_numpy(
                mirna_phact_missing[:row_count].copy()
            ),
            "target_phact_missing": torch.from_numpy(
                target_phact_missing[:row_count].copy()
            ),
            "labels": torch.from_numpy(labels[:row_count].copy()),
        },
        shard_path,
    )
    elapsed = time.perf_counter() - shard_started
    size_bytes = shard_path.stat().st_size
    with log_path.open("a") as handle:
        handle.write(
            "\t".join(
                [
                    str(shard_idx),
                    shard_name,
                    str(row_count),
                    str(positive_rows),
                    str(size_bytes),
                    f"{elapsed:.3f}",
                ]
            )
            + "\n"
        )
    print(
        f"cached shard {shard_idx}: rows={row_count:,} "
        f"size={size_bytes / (1024 ** 3):.2f} GiB elapsed={elapsed:.1f}s",
        flush=True,
    )
    return {
        "file": shard_name,
        "rows": row_count,
        "positive_rows": positive_rows,
        "size_bytes": size_bytes,
        "elapsed_sec": round(elapsed, 3),
    }
