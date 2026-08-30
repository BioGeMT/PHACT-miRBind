#!/usr/bin/env python
"""Build wide per-Manakov-row miRNA PHACT score rows."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path

NUCLEOTIDES = ("A", "C", "G", "T")
def main() -> None:
    args = parse_args()
    split_paths = {
        "train": args.train_file,
        "test": args.test_file,
        "leftout": args.leftout_file,
    }
    model_order, profile_scores = load_phact_profiles(args.phact_mirna_scores)
    score_columns = [
        f"phact_{safe_model_name(model)}_{base}"
        for model in model_order
        for base in NUCLEOTIDES
    ]
    write_row_scores(
        split_paths=split_paths,
        output_path=args.output_file,
        summary_path=args.summary_file,
        source_phact_file=args.phact_mirna_scores,
        model_order=model_order,
        score_columns=score_columns,
        profile_scores=profile_scores,
        mirna_length=args.mirna_length,
        float_format=args.float_format,
        force=args.force,
        max_rows_per_split=args.max_rows_per_split,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--leftout-file", type=Path, required=True)
    parser.add_argument("--phact-mirna-scores", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--summary-file", type=Path, default=None)
    parser.add_argument("--mirna-length", type=int, default=28)
    parser.add_argument(
        "--float-format",
        type=str,
        default=".8g",
        help="Python float format specifier used for scores.",
    )
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_model_name(model: str) -> str:
    safe = model.replace(".", "p")
    return re.sub(r"[^0-9A-Za-z_]+", "_", safe)


def normalize_sequence(sequence: str) -> str:
    return sequence.strip().upper().replace("U", "T")


def parse_mature_id(mature_id: str) -> tuple[str, str]:
    return tuple(mature_id.rsplit("_", 1))  # type: ignore[return-value]


def load_phact_profiles(
    path: Path,
) -> tuple[list[str], dict[str, dict[int, list[float | None]]]]:
    model_order: list[str] = []
    model_index: dict[str, int] = {}
    rows: list[dict[str, str]] = []

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            model = row["phact_model"]
            if model not in model_index:
                model_index[model] = len(model_order)
                model_order.append(model)
            rows.append(row)

    column_count = len(model_order) * len(NUCLEOTIDES)
    profiles: dict[str, dict[int, list[float | None]]] = {}
    for row in rows:
        mature_id = f"{row['pre_mirna']}_{row['arm']}"
        position = int(row["arm_position_1based"])
        if position < 1:
            continue
        profile = profiles.setdefault(mature_id, {})
        values = profile.setdefault(position, [None] * column_count)
        base_offset = model_index[row["phact_model"]] * len(NUCLEOTIDES)
        for base_index, base in enumerate(NUCLEOTIDES):
            values[base_offset + base_index] = float(row[f"score_{base}"])

    return model_order, profiles


def write_row_scores(
    *,
    split_paths: dict[str, Path],
    output_path: Path,
    summary_path: Path | None,
    source_phact_file: Path,
    model_order: list[str],
    score_columns: list[str],
    profile_scores: dict[str, dict[int, list[float | None]]],
    mirna_length: int,
    float_format: str,
    force: bool,
    max_rows_per_split: int | None,
) -> None:
    if mirna_length < 1:
        raise ValueError("--mirna-length must be at least 1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = summary_path or output_path.with_suffix(".summary.json")
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if output_path.exists() and not force:
        raise FileExistsError(f"Output exists: {output_path}")
    if temp_path.exists():
        temp_path.unlink()

    profile_cache: dict[tuple[str, str], list[list[str]]] = {}
    stats: Counter[str] = Counter()
    started = time.perf_counter()
    header = [
        "split",
        "manakov_row_id",
        "mirna_position_1based",
        "actual_nt",
        *score_columns,
    ]

    with temp_path.open("w", buffering=1024 * 1024) as out:
        out.write("\t".join(header) + "\n")
        for split, path in split_paths.items():
            split_started = time.perf_counter()
            split_rows = 0
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                required = {"noncodingRNA", "mirgenedb_mature_id"}
                missing_columns = required.difference(reader.fieldnames or [])
                if missing_columns:
                    raise ValueError(f"{path} missing columns: {sorted(missing_columns)}")

                for row_id, row in enumerate(reader, start=1):
                    if max_rows_per_split is not None and split_rows >= max_rows_per_split:
                        break

                    split_rows += 1
                    stats[f"{split}_rows"] += 1
                    profile_key = (row["mirgenedb_mature_id"], row["noncodingRNA"])
                    rows_for_mirna = profile_cache.get(profile_key)
                    if rows_for_mirna is None:
                        rows_for_mirna, row_stats = build_rows_for_mirna(
                            mature_ids_raw=row["mirgenedb_mature_id"],
                            sequence=row["noncodingRNA"],
                            profile_scores=profile_scores,
                            column_count=len(score_columns),
                            mirna_length=mirna_length,
                            float_format=float_format,
                        )
                        profile_cache[profile_key] = rows_for_mirna
                        stats.update(row_stats)

                    for suffix in rows_for_mirna:
                        out.write(f"{split}\t{row_id}\t" + "\t".join(suffix) + "\n")

                    if split_rows == 1 or split_rows % 250_000 == 0:
                        elapsed = time.perf_counter() - split_started
                        print(
                            f"{split}: rows={split_rows:,} "
                            f"output_rows={split_rows * mirna_length:,} "
                            f"elapsed={elapsed:.1f}s",
                            flush=True,
                        )

            print(
                f"{split}: complete rows={split_rows:,} "
                f"output_rows={split_rows * mirna_length:,}",
                flush=True,
            )

    temp_path.replace(output_path)
    stats["model_count"] = len(model_order)
    stats["score_column_count"] = len(score_columns)
    stats["mirna_length"] = mirna_length
    stats["profile_cache_entries"] = len(profile_cache)
    stats["total_input_rows"] = sum(stats[f"{split}_rows"] for split in split_paths)
    stats["total_output_rows"] = stats["total_input_rows"] * mirna_length
    stats["seconds"] = round(time.perf_counter() - started, 3)
    summary = {
        "output_file": str(output_path),
        "source_phact_file": str(source_phact_file),
        "models": model_order,
        "score_columns": score_columns,
        "stats": dict(stats),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"output={output_path}", flush=True)
    print(f"summary={summary_path}", flush=True)
    print(f"rows_written={stats['total_output_rows']:,}", flush=True)


def build_rows_for_mirna(
    *,
    mature_ids_raw: str,
    sequence: str,
    profile_scores: dict[str, dict[int, list[float | None]]],
    column_count: int,
    mirna_length: int,
    float_format: str,
) -> tuple[list[list[str]], Counter[str]]:
    stats: Counter[str] = Counter()
    normalized_sequence = normalize_sequence(sequence)
    mature_ids = [] if mature_ids_raw == "NA" else mature_ids_raw.split(";")
    available_profiles = [
        profile_scores[mature_id]
        for mature_id in mature_ids
        if mature_id in profile_scores
    ]

    stats["unique_mirna_keys"] += 1
    stats[f"mapped_id_count_{len(mature_ids)}"] += 1
    stats[f"available_profile_count_{len(available_profiles)}"] += 1
    if not mature_ids:
        stats["no_mirgenedb_match_keys"] += 1
    elif not available_profiles:
        stats["no_phact_profile_keys"] += 1
    elif len(available_profiles) == 1:
        stats["one_phact_profile_keys"] += 1
    else:
        stats["multiple_phact_profile_keys"] += 1

    rows: list[list[str]] = []
    for position in range(1, mirna_length + 1):
        actual_nt = (
            normalized_sequence[position - 1]
            if position <= len(normalized_sequence)
            else "N"
        )
        if actual_nt not in NUCLEOTIDES:
            actual_nt = "N"
        score_values = collapse_position_scores(
            available_profiles,
            position,
            column_count,
        )
        formatted_scores = [
            format(value, float_format) if value is not None else "NA"
            for value in score_values
        ]
        rows.append([str(position), actual_nt, *formatted_scores])
    return rows, stats


def collapse_position_scores(
    profiles: list[dict[int, list[float | None]]],
    position: int,
    column_count: int,
) -> list[float | None]:
    if not profiles:
        return [None] * column_count

    collapsed: list[float | None] = []
    for column_index in range(column_count):
        values = [
            profile[position][column_index]
            for profile in profiles
            if position in profile and profile[position][column_index] is not None
        ]
        collapsed.append(sum(values) / len(values) if values else None)
    return collapsed


if __name__ == "__main__":
    main()
