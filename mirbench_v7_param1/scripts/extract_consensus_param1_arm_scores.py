#!/usr/bin/env python3
"""Extract consensus-tree parameter-1 scores on mature-miRNA coordinates."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import zipfile
from collections import Counter
from pathlib import Path


BASES = ("A", "C", "G", "T")
PARAMETER_COLUMN = "PHACTn_gapAware_wtNTnorm_wl_param_1"
DEFAULT_MEMBER = "results_qntnorm_transformed.tsv.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--member", default=DEFAULT_MEMBER)
    parser.add_argument("--coordinate-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_archive_scores(
    archive: Path,
    member: str,
) -> tuple[dict[tuple[str, int, str], tuple[str, float]], Counter[str]]:
    scores: dict[tuple[str, int, str], tuple[str, float]] = {}
    stats: Counter[str] = Counter()
    ids: set[str] = set()
    positions: set[tuple[str, int]] = set()

    with zipfile.ZipFile(archive) as zipped:
        if member not in zipped.namelist():
            raise ValueError(f"Archive member not found: {member}")
        with zipped.open(member) as compressed:
            with gzip.GzipFile(fileobj=compressed) as uncompressed:
                text = io.TextIOWrapper(uncompressed, encoding="utf-8", newline="")
                reader = csv.DictReader(text, delimiter="\t")
                required = {"ID", "Position", "Nucleotide", PARAMETER_COLUMN}
                missing = required.difference(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"Archive member missing columns: {sorted(missing)}")

                for row_number, row in enumerate(reader, start=2):
                    precursor = row["ID"]
                    position = int(row["Position"])
                    nucleotide = row["Nucleotide"].upper().replace("U", "T")
                    if nucleotide not in BASES:
                        raise ValueError(
                            f"Unexpected nucleotide {nucleotide!r} at archive row {row_number}"
                        )
                    raw_value = row[PARAMETER_COLUMN]
                    value = float(raw_value)
                    if not math.isfinite(value):
                        raise ValueError(f"Non-finite score at archive row {row_number}")
                    key = (precursor, position, nucleotide)
                    if key in scores:
                        raise ValueError(f"Duplicate archive score key: {key}")
                    scores[key] = (raw_value, value)
                    ids.add(precursor)
                    positions.add((precursor, position))

    for precursor, position in positions:
        missing_bases = [
            base for base in BASES if (precursor, position, base) not in scores
        ]
        if missing_bases:
            raise ValueError(
                f"Incomplete nucleotide scores for {precursor} position {position}: "
                f"{missing_bases}"
            )

    stats["archive_score_rows"] = len(scores)
    stats["archive_precursors"] = len(ids)
    stats["archive_precursor_positions"] = len(positions)
    return scores, stats


def load_coordinate_template(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: dict[tuple[str, str, int, int, str], dict[str, str]] = {}
    required = {
        "pre_mirna",
        "family",
        "arm",
        "arm_position_1based",
        "flanked_position_1based",
        "actual_nt",
    }

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Coordinate template missing columns: {sorted(missing)}")
        for row in reader:
            actual_nt = row["actual_nt"].upper().replace("U", "T")
            if actual_nt not in BASES:
                raise ValueError(
                    f"Unexpected template nucleotide {actual_nt!r} for "
                    f"{row['pre_mirna']} {row['arm']}"
                )
            key = (
                row["pre_mirna"],
                row["arm"],
                int(row["arm_position_1based"]),
                int(row["flanked_position_1based"]),
                actual_nt,
            )
            existing = seen.get(key)
            if existing is not None:
                if existing["family"] != row["family"]:
                    raise ValueError(f"Inconsistent family for template key: {key}")
                continue
            normalized = {
                "pre_mirna": row["pre_mirna"],
                "family": row["family"],
                "arm": row["arm"],
                "arm_position_1based": str(int(row["arm_position_1based"])),
                "flanked_position_1based": str(int(row["flanked_position_1based"])),
                "actual_nt": actual_nt,
            }
            seen[key] = normalized
            rows.append(normalized)

    return rows


def write_output(
    output: Path,
    template_rows: list[dict[str, str]],
    scores: dict[tuple[str, int, str], tuple[str, float]],
    force: bool,
) -> Counter[str]:
    if output.exists() and not force:
        raise FileExistsError(f"Output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    if temp.exists():
        temp.unlink()

    fieldnames = [
        "pre_mirna",
        "family",
        "arm",
        "arm_position_1based",
        "flanked_position_1based",
        "actual_nt",
        "phact_model",
        "score_A",
        "score_C",
        "score_G",
        "score_T",
        "actual_nt_score",
    ]
    stats: Counter[str] = Counter()
    precursor_ids: set[str] = set()
    arms: set[tuple[str, str]] = set()
    min_score = math.inf
    max_score = -math.inf

    try:
        with temp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
            writer.writeheader()
            for template in template_rows:
                precursor = template["pre_mirna"]
                position = int(template["flanked_position_1based"])
                score_strings: dict[str, str] = {}
                for base in BASES:
                    key = (precursor, position, base)
                    if key not in scores:
                        raise ValueError(f"Missing parameter-1 score: {key}")
                    raw, value = scores[key]
                    score_strings[base] = raw
                    min_score = min(min_score, value)
                    max_score = max(max_score, value)

                actual_nt = template["actual_nt"]
                writer.writerow(
                    {
                        **template,
                        "phact_model": "param_1",
                        **{f"score_{base}": score_strings[base] for base in BASES},
                        "actual_nt_score": score_strings[actual_nt],
                    }
                )
                stats["output_rows"] += 1
                precursor_ids.add(precursor)
                arms.add((precursor, template["arm"]))
    except BaseException:
        if temp.exists():
            temp.unlink()
        raise

    os.replace(temp, output)
    stats["output_precursors"] = len(precursor_ids)
    stats["output_arms"] = len(arms)
    stats["minimum_score"] = min_score
    stats["maximum_score"] = max_score
    return stats


def main() -> None:
    args = parse_args()
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    scores, archive_stats = load_archive_scores(args.archive, args.member)
    template_rows = load_coordinate_template(args.coordinate_template)
    output_stats = write_output(args.output, template_rows, scores, args.force)

    summary = {
        "archive": str(args.archive),
        "archive_sha256": sha256(args.archive),
        "member": args.member,
        "parameter_column": PARAMETER_COLUMN,
        "coordinate_template": str(args.coordinate_template),
        "coordinate_template_sha256": sha256(args.coordinate_template),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "model_name": "param_1",
        "score_order": list(BASES),
        "stats": {**dict(archive_stats), **dict(output_stats)},
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
