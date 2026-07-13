#!/usr/bin/env python
"""Append exact MirGeneDB mature ID/family annotations to Manakov TSVs."""

from __future__ import annotations

import argparse
import csv
import re
import time
from collections import defaultdict
from pathlib import Path

MATURE_ID_COLUMN = "mirgenedb_mature_id"
FAMILY_COLUMN = "mirgenedb_family"


def normalize_sequence(sequence: str) -> str:
    return sequence.strip().upper().replace("U", "T")


def parse_mature_fasta(path: Path) -> dict[str, list[str]]:
    sequence_to_ids: dict[str, list[str]] = defaultdict(list)
    header: str | None = None
    chunks: list[str] = []

    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequence_to_ids[normalize_sequence("".join(chunks))].append(header)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)

    if header is not None:
        sequence_to_ids[normalize_sequence("".join(chunks))].append(header)

    return dict(sequence_to_ids)


def build_family_lookup(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {row["MirGeneDB_ID"]: row["Family"] for row in reader}


def precursor_id(mature_id: str) -> str:
    return mature_id.rsplit("_", 1)[0]


def family_for_mature_id(mature_id: str, family_lookup: dict[str, str]) -> str:
    precursor = precursor_id(mature_id)
    if precursor in family_lookup:
        return family_lookup[precursor]

    unversioned = re.sub(r"-v[0-9]+$", "", precursor)
    return family_lookup[unversioned]


def build_exact_annotation_lookup(
    sequence_to_ids: dict[str, list[str]],
    family_lookup: dict[str, str],
) -> dict[str, tuple[str, str]]:
    annotations = {}
    for sequence, mature_ids in sequence_to_ids.items():
        families = sorted(
            {family_for_mature_id(mature_id, family_lookup) for mature_id in mature_ids}
        )
        if len(families) != 1:
            joined_ids = ";".join(mature_ids)
            joined_families = ";".join(families)
            raise ValueError(
                f"Conflicting MirGeneDB families for {sequence}: "
                f"{joined_ids} -> {joined_families}"
            )
        annotations[sequence] = (";".join(mature_ids), families[0])
    return annotations


def append_annotations(input_path: Path, annotations: dict[str, tuple[str, str]]) -> None:
    temp_path = input_path.with_suffix(input_path.suffix + ".mirgenedb.tmp")
    if temp_path.exists():
        temp_path.unlink()

    started = time.perf_counter()
    rows = 0
    matched = 0
    multi = 0

    with input_path.open() as in_handle, temp_path.open("w", buffering=1024 * 1024) as out:
        header_line = in_handle.readline()
        if not header_line:
            raise ValueError(f"Input file is empty: {input_path}")

        header = header_line.rstrip("\n").rstrip("\r").split("\t")
        existing_annotation_indexes = {
            index
            for index, column in enumerate(header)
            if column in {MATURE_ID_COLUMN, FAMILY_COLUMN}
        }
        if existing_annotation_indexes:
            header = [
                column
                for index, column in enumerate(header)
                if index not in existing_annotation_indexes
            ]
        try:
            sequence_index = header.index("noncodingRNA")
        except ValueError as exc:
            raise ValueError(f"{input_path} is missing noncodingRNA column") from exc

        out.write(
            header_line.rstrip("\n").rstrip("\r")
            + f"\t{MATURE_ID_COLUMN}\t{FAMILY_COLUMN}\n"
        )

        for line_number, line in enumerate(in_handle, start=2):
            raw_line = line.rstrip("\n").rstrip("\r")
            fields = [
                field
                for index, field in enumerate(raw_line.split("\t"))
                if index not in existing_annotation_indexes
            ]
            if len(fields) <= sequence_index:
                raise ValueError(f"Malformed row {line_number} in {input_path}")

            rows += 1
            annotation = annotations.get(normalize_sequence(fields[sequence_index]))
            if annotation is None:
                mature_id = family = "NA"
            else:
                mature_id, family = annotation
                matched += 1
                if ";" in mature_id:
                    multi += 1
            out.write("\t".join(fields) + f"\t{mature_id}\t{family}\n")

            if rows % 500_000 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"{input_path.name}: rows={rows:,} matched={matched:,} "
                    f"multi={multi:,} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    temp_path.replace(input_path)
    elapsed = time.perf_counter() - started
    print(
        f"{input_path.name}: complete rows={rows:,} matched={matched:,} "
        f"multi={multi:,} unmatched={rows - matched:,} elapsed={elapsed:.1f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--family-map", type=Path, required=True)
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()

    annotations = build_exact_annotation_lookup(
        parse_mature_fasta(args.fasta),
        build_family_lookup(args.family_map),
    )
    for path in args.files:
        append_annotations(path, annotations)


if __name__ == "__main__":
    main()
