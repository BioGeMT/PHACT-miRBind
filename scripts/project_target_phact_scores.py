#!/usr/bin/env python
"""Project existing genomic target PHACT scores onto a new row table."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path


WINDOW_LENGTH = 50
NUCLEOTIDES = {"A", "C", "G", "T"}


def normalize_chrom(value: str) -> str:
    chrom = value.strip()
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def parse_int(value: str) -> int:
    return int(float(value))


def normalize_sequence(value: str) -> str:
    return value.strip().upper().replace("U", "T")


def collect_wanted_positions(
    input_path: Path,
) -> tuple[dict[tuple[str, str], set[int]], Counter[str]]:
    wanted: dict[tuple[str, str], set[int]] = {}
    stats: Counter[str] = Counter()
    with input_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            stats["input_rows"] += 1
            chrom = normalize_chrom(row["chr"])
            strand = row["strand"]
            start = parse_int(row["start"])
            end = parse_int(row["end"])
            if strand not in {"+", "-"} or end - start + 1 != WINDOW_LENGTH:
                stats["invalid_intervals"] += 1
                continue
            wanted.setdefault((chrom, strand), set()).update(range(start, end + 1))
    stats["chrom_strand_groups"] = len(wanted)
    stats["unique_wanted_positions"] = sum(len(values) for values in wanted.values())
    return wanted, stats


class ManakovMetadataStream:
    def __init__(self, path: Path) -> None:
        self.handle = path.open(newline="")
        self.reader = csv.DictReader(self.handle, delimiter="\t")
        self.row_id = 0
        self.row: dict[str, str] | None = None

    def get(self, row_id: int) -> dict[str, str]:
        if row_id < self.row_id:
            raise ValueError(
                f"Manakov row IDs are not ascending: {row_id} after {self.row_id}"
            )
        while self.row_id < row_id:
            try:
                self.row = next(self.reader)
            except StopIteration as exc:
                raise ValueError(f"Missing Manakov metadata row {row_id}") from exc
            self.row_id += 1
        assert self.row is not None
        return self.row

    def close(self) -> None:
        self.handle.close()


def load_matching_scores(
    target_score_path: Path,
    split_paths: dict[str, Path],
    wanted: dict[tuple[str, str], set[int]],
) -> tuple[dict[tuple[str, str], dict[int, tuple[str, str, str, str]]], Counter[str]]:
    streams = {
        split: ManakovMetadataStream(path) for split, path in split_paths.items()
    }
    scores: dict[tuple[str, str], dict[int, tuple[str, str, str, str]]] = {}
    stats: Counter[str] = Counter()
    active_source: tuple[str, int] | None = None
    active_key: tuple[str, str] | None = None
    active_wanted: set[int] | None = None
    started = time.perf_counter()
    try:
        with target_score_path.open(buffering=1024 * 1024) as handle:
            header = handle.readline().rstrip("\r\n").split("\t")
            expected = [
                "split",
                "manakov_row_id",
                "target_position_1based",
                "genomic_position",
                "actual_nt",
                "score_A",
                "score_C",
                "score_G",
                "score_T",
            ]
            if header != expected:
                raise ValueError(f"Unexpected target PHACT header: {header}")

            for line in handle:
                fields = line.rstrip("\r\n").split("\t")
                stats["score_rows_scanned"] += 1
                split = fields[0]
                row_id = int(fields[1])
                source = (split, row_id)
                if source != active_source:
                    if split not in streams:
                        raise ValueError(f"Unexpected PHACT split: {split}")
                    metadata = streams[split].get(row_id)
                    active_key = (
                        normalize_chrom(metadata["chr"]),
                        metadata["strand"],
                    )
                    active_wanted = wanted.get(active_key)
                    active_source = source
                    stats["manakov_rows_scanned"] += 1

                if active_wanted is None or active_key is None:
                    continue
                position = int(fields[3])
                if position not in active_wanted:
                    continue
                values = (fields[5], fields[6], fields[7], fields[8])
                group_scores = scores.setdefault(active_key, {})
                existing = group_scores.get(position)
                if existing is None:
                    group_scores[position] = values
                    stats["matched_unique_positions"] += 1
                elif existing != values:
                    stats["conflicting_duplicate_scores"] += 1

                if stats["score_rows_scanned"] % 20_000_000 == 0:
                    print(
                        f"score_rows={stats['score_rows_scanned']:,} "
                        f"matched_positions={stats['matched_unique_positions']:,} "
                        f"elapsed={time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
    finally:
        for stream in streams.values():
            stream.close()
    return scores, stats


def write_projected_scores(
    input_path: Path,
    output_path: Path,
    scores: dict[tuple[str, str], dict[int, tuple[str, str, str, str]]],
    *,
    split: str,
) -> Counter[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    header = [
        "split",
        "manakov_row_id",
        "target_position_1based",
        "genomic_position",
        "actual_nt",
        "score_A",
        "score_C",
        "score_G",
        "score_T",
    ]
    with input_path.open(newline="") as source, output_path.open(
        "w", buffering=1024 * 1024
    ) as destination:
        reader = csv.DictReader(source, delimiter="\t")
        destination.write("\t".join(header) + "\n")
        for row_id, row in enumerate(reader, start=1):
            chrom = normalize_chrom(row["chr"])
            strand = row["strand"]
            start = parse_int(row["start"])
            end = parse_int(row["end"])
            sequence = normalize_sequence(row["gene"])[:WINDOW_LENGTH].ljust(
                WINDOW_LENGTH, "N"
            )
            group_scores = scores.get((chrom, strand), {})
            lines: list[str] = []
            for offset in range(WINDOW_LENGTH):
                target_position = offset + 1
                genomic_position = start + offset if strand == "+" else end - offset
                actual_nt = sequence[offset]
                if actual_nt not in NUCLEOTIDES:
                    actual_nt = "N"
                values = group_scores.get(genomic_position)
                if values is None:
                    values = ("NA", "NA", "NA", "NA")
                    stats["missing_position_rows"] += 1
                else:
                    stats["matched_position_rows"] += 1
                lines.append(
                    f"{split}\t{row_id}\t{target_position}\t{genomic_position}\t"
                    f"{actual_nt}\t" + "\t".join(values) + "\n"
                )
            destination.write("".join(lines))
            stats["input_rows"] += 1
            stats["output_rows"] += WINDOW_LENGTH
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--manakov-train", type=Path, required=True)
    parser.add_argument("--manakov-test", type=Path, required=True)
    parser.add_argument("--manakov-leftout", type=Path, required=True)
    parser.add_argument("--target-phact-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--summary-file", type=Path)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    wanted, interval_stats = collect_wanted_positions(args.input_file)
    print(
        f"input_rows={interval_stats['input_rows']:,} "
        f"wanted_positions={interval_stats['unique_wanted_positions']:,}",
        flush=True,
    )
    scores, scan_stats = load_matching_scores(
        args.target_phact_file,
        {
            "train": args.manakov_train,
            "test": args.manakov_test,
            "leftout": args.manakov_leftout,
        },
        wanted,
    )
    output_stats = write_projected_scores(
        args.input_file,
        args.output_file,
        scores,
        split=args.split,
    )
    summary_path = args.summary_file or args.output_file.with_suffix(".summary.json")
    summary = {
        "input_file": str(args.input_file),
        "target_phact_file": str(args.target_phact_file),
        "output_file": str(args.output_file),
        "split": args.split,
        "interval_stats": dict(interval_stats),
        "scan_stats": dict(scan_stats),
        "output_stats": dict(output_stats),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"output={args.output_file}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
