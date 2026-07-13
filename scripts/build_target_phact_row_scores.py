#!/usr/bin/env python
"""Build compact per-Manakov-row target PHACT score rows."""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SPLITS = {
    "train": Path("/home/dtzim01/manakov_datasets/AGO2_eCLIP_Manakov2022_train.tsv"),
    "test": Path("/home/dtzim01/manakov_datasets/AGO2_eCLIP_Manakov2022_test.tsv"),
    "leftout": Path("/home/dtzim01/manakov_datasets/AGO2_eCLIP_Manakov2022_leftout.tsv"),
}
DEFAULT_SOURCE = Path(
    "/home/dtzim01/drive-download-19Ntprvu-qbI1k4ZQphZ4QnuFoXNIgK2E/"
    "extracted/results_0226/AGO2_eCLIP_Manakov2022_qntnorm_transformed.tsv"
)
DEFAULT_OUTPUT = Path(
    "/home/dtzim01/PHACT-miRBind/reports/phact_score_ranges/"
    "phact_target_manakov_position_qntnorm_transformed_scores.tsv"
)
WINDOW_LEN = 50
NUCLEOTIDES = ("A", "C", "G", "T")


def main() -> None:
    args = parse_args()
    split_paths = {
        "train": args.train_file,
        "test": args.test_file,
        "leftout": args.leftout_file,
    }
    write_target_rows(
        split_paths=split_paths,
        target_score_file=args.target_score_file,
        output_file=args.output_file,
        force=args.force,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_SPLITS["train"])
    parser.add_argument("--test-file", type=Path, default=DEFAULT_SPLITS["test"])
    parser.add_argument("--leftout-file", type=Path, default=DEFAULT_SPLITS["leftout"])
    parser.add_argument("--target-score-file", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize_chrom(value: object) -> str:
    chrom = str(value)
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def parse_int(value: object) -> int:
    return int(float(value))


def normalize_seq(value: object) -> str:
    return str(value).strip().upper().replace("U", "T")


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    intervals.sort()
    merged: list[list[int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        elif end > merged[-1][1]:
            merged[-1][1] = end
    return [(start, end) for start, end in merged]


def scan_intervals(
    split_paths: dict[str, Path],
) -> tuple[dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]], Counter]:
    raw_intervals: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    rows_by_split: Counter[str] = Counter()
    bad_rows = 0
    for split, path in split_paths.items():
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                rows_by_split[split] += 1
                try:
                    chrom = normalize_chrom(row["chr"])
                    strand = row["strand"]
                    start = parse_int(row["start"])
                    end = parse_int(row["end"])
                    if strand not in {"+", "-"} or end - start + 1 != WINDOW_LEN:
                        bad_rows += 1
                        continue
                    raw_intervals[(chrom, strand)].append((start, end))
                except Exception:
                    bad_rows += 1

    intervals: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    total_union_bp = 0
    for key, raw in raw_intervals.items():
        merged = merge_intervals(raw)
        starts = np.array([start for start, _ in merged], dtype=np.int64)
        ends = np.array([end for _, end in merged], dtype=np.int64)
        lengths = ends - starts + 1
        offsets = np.empty(len(merged), dtype=np.int64)
        if len(merged):
            offsets[0] = total_union_bp
            if len(merged) > 1:
                offsets[1:] = total_union_bp + np.cumsum(lengths[:-1])
        intervals[key] = (starts, ends, offsets)
        total_union_bp += int(lengths.sum())

    stats = Counter(rows_by_split)
    stats["bad_interval_rows"] = bad_rows
    stats["union_bp"] = total_union_bp
    return intervals, stats


def global_indices(
    intervals: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    chrom: str,
    strand: str,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    starts, ends, offsets = intervals[(chrom, strand)]
    idx = np.searchsorted(starts, positions, side="right") - 1
    valid = idx >= 0
    mask = np.zeros(len(positions), dtype=np.bool_)
    global_idx = np.full(len(positions), -1, dtype=np.int64)
    if valid.any():
        valid_pos = positions[valid]
        valid_idx = idx[valid]
        keep = valid_pos <= ends[valid_idx]
        if keep.any():
            valid_locations = np.flatnonzero(valid)[keep]
            keep_idx = valid_idx[keep]
            keep_pos = valid_pos[keep]
            global_idx[valid_locations] = offsets[keep_idx] + (
                keep_pos - starts[keep_idx]
            )
            mask[valid_locations] = True
    return mask, global_idx


def load_target_scores(
    intervals: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    target_score_file: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total_union_bp = max(
        int(offsets[-1] + (ends[-1] - starts[-1] + 1))
        for starts, ends, offsets in intervals.values()
        if len(starts)
    )
    score_a = np.full(total_union_bp, np.nan, dtype=np.float64)
    score_c = np.full(total_union_bp, np.nan, dtype=np.float64)
    score_g = np.full(total_union_bp, np.nan, dtype=np.float64)
    score_t = np.full(total_union_bp, np.nan, dtype=np.float64)
    found = np.zeros(total_union_bp, dtype=np.bool_)
    usecols = ["Chromosome", "Position", "Strand", "A", "T", "G", "C"]
    dtype = {
        "Chromosome": "string",
        "Position": "int64",
        "Strand": "string",
        "A": "float64",
        "T": "float64",
        "G": "float64",
        "C": "float64",
    }
    seen_rows = 0
    for chunk_number, chunk in enumerate(
        pd.read_csv(
            target_score_file,
            sep="\t",
            usecols=usecols,
            dtype=dtype,
            chunksize=1_000_000,
        ),
        start=1,
    ):
        seen_rows += len(chunk)
        for (chrom_raw, strand_raw), group in chunk.groupby(
            ["Chromosome", "Strand"], sort=False, observed=True
        ):
            chrom = normalize_chrom(chrom_raw)
            strand = str(strand_raw)
            if (chrom, strand) not in intervals:
                continue
            positions = group["Position"].to_numpy(dtype=np.int64, copy=False)
            mask, global_idx = global_indices(intervals, chrom, strand, positions)
            if not mask.any():
                continue
            kept = group.loc[mask, ["A", "C", "G", "T"]]
            gi = global_idx[mask]
            score_a[gi] = kept["A"].to_numpy(dtype=np.float64, copy=False)
            score_c[gi] = kept["C"].to_numpy(dtype=np.float64, copy=False)
            score_g[gi] = kept["G"].to_numpy(dtype=np.float64, copy=False)
            score_t[gi] = kept["T"].to_numpy(dtype=np.float64, copy=False)
            found[gi] = True
        if chunk_number == 1 or chunk_number % 20 == 0:
            print(
                f"scanned_target_rows={seen_rows:,} "
                f"unique_positions_found={int(found.sum()):,}/{total_union_bp:,}",
                flush=True,
            )
    return score_a, score_c, score_g, score_t, found


def one_global_index(
    intervals: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    chrom: str,
    strand: str,
    position: int,
) -> int:
    starts, ends, offsets = intervals[(chrom, strand)]
    idx = int(np.searchsorted(starts, position, side="right") - 1)
    if idx < 0 or position > int(ends[idx]):
        return -1
    return int(offsets[idx] + (position - int(starts[idx])))


def fmt(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return format(float(value), ".17g")


def write_target_rows(
    *,
    split_paths: dict[str, Path],
    target_score_file: Path,
    output_file: Path,
    force: bool,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = output_file.with_suffix(output_file.suffix + ".tmp")
    if output_file.exists() and not force:
        raise FileExistsError(f"Output exists: {output_file}")
    if temp_file.exists():
        temp_file.unlink()

    started = time.perf_counter()
    intervals, split_stats = scan_intervals(split_paths)
    print(
        f"interval scan complete: rows_by_split={dict(split_stats)} "
        f"keys={len(intervals)} union_bp={split_stats['union_bp']:,}",
        flush=True,
    )
    score_a, score_c, score_g, score_t, found = load_target_scores(
        intervals, target_score_file
    )

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
    rows_written = 0
    missing_score_rows = 0
    with temp_file.open("w", buffering=1024 * 1024) as out:
        out.write("\t".join(header) + "\n")
        for split, path in split_paths.items():
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row_index, row in enumerate(reader, start=1):
                    chrom = normalize_chrom(row["chr"])
                    start = parse_int(row["start"])
                    end = parse_int(row["end"])
                    strand = row["strand"]
                    gene = normalize_seq(row["gene"])[:WINDOW_LEN].ljust(WINDOW_LEN, "N")
                    lines: list[str] = []
                    for offset in range(WINDOW_LEN):
                        target_position = offset + 1
                        genomic_position = start + offset if strand == "+" else end - offset
                        actual_nt = gene[offset] if gene[offset] in NUCLEOTIDES else "N"
                        gi = one_global_index(
                            intervals, chrom, strand, genomic_position
                        )
                        if gi < 0 or not found[gi]:
                            a = c = g = t = "NA"
                            missing_score_rows += 1
                        else:
                            a = fmt(score_a[gi])
                            c = fmt(score_c[gi])
                            g = fmt(score_g[gi])
                            t = fmt(score_t[gi])
                        lines.append(
                            f"{split}\t{row_index}\t{target_position}\t"
                            f"{genomic_position}\t{actual_nt}\t{a}\t{c}\t{g}\t{t}\n"
                        )
                    out.write("".join(lines))
                    rows_written += WINDOW_LEN
                    if row_index == 1 or row_index % 100_000 == 0:
                        print(
                            f"{split}: manakov_rows={row_index:,} "
                            f"total_output_rows={rows_written:,}",
                            flush=True,
                        )

    temp_file.replace(output_file)
    print(f"output={output_file}", flush=True)
    print(f"rows_written={rows_written}", flush=True)
    print(f"missing_score_rows={missing_score_rows}", flush=True)
    print(f"elapsed={time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
