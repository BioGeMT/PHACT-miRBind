#!/usr/bin/env python3
"""Measure how well training-only group priors transfer across Manakov splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


KEYS = {
    "target_sequence": ["gene"],
    "target_cluster": ["gene_cluster_ID"],
    "target_locus": ["chr", "start", "end", "strand"],
    "mirna_sequence": ["noncodingRNA"],
    "mirna_family": ["noncodingRNA_fam"],
    "mirna_seed": ["mirna_seed"],
    "target_sequence_mirna_seed": ["gene", "mirna_seed"],
    "target_cluster_mirna_family": ["gene_cluster_ID", "noncodingRNA_fam"],
}
SOURCE_COLUMNS = sorted(
    {
        "label",
        "noncodingRNA",
        *(column for columns in KEYS.values() for column in columns if column != "mirna_seed"),
    }
)
SMOOTHING = (1.0, 5.0, 20.0, 100.0)


def main() -> None:
    args = parse_args()
    frames = {
        "train": read_rows(args.train),
        "val": read_rows(args.val),
        "test": read_rows(args.test),
        "leftout": read_rows(args.leftout),
    }
    global_rate = float(frames["train"]["label"].mean())
    report: dict[str, object] = {
        "rows": {name: len(frame) for name, frame in frames.items()},
        "train_positive_rate": global_rate,
        "priors": {},
    }

    for key_name, key_columns in KEYS.items():
        counts = (
            frames["train"]
            .groupby(key_columns, dropna=False, observed=True)["label"]
            .agg(["sum", "count"])
        )
        candidates = []
        for smoothing in SMOOTHING:
            metrics = {
                split: evaluate_prior(
                    frames[split],
                    counts,
                    key_columns,
                    global_rate,
                    smoothing,
                )
                for split in ("val", "test", "leftout")
            }
            candidates.append({"smoothing": smoothing, "splits": metrics})
        selected = max(candidates, key=lambda row: row["splits"]["val"]["auprc"])
        report["priors"][key_name] = {
            "key_columns": key_columns,
            "train_groups": len(counts),
            "selected_by_val": selected,
            "candidates": candidates,
        }
        print(
            key_name,
            f"alpha={selected['smoothing']:g}",
            *(f"{split}={selected['splits'][split]['auprc']:.6f}"
              f"/{selected['splits'][split]['coverage']:.3f}"
              for split in ("val", "test", "leftout")),
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"output={args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--leftout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=SOURCE_COLUMNS,
        dtype={column: "string" for column in SOURCE_COLUMNS if column != "label"},
    )
    frame["label"] = frame["label"].astype(np.int8)
    frame["mirna_seed"] = (
        frame["noncodingRNA"]
        .str.upper()
        .str.replace("U", "T", regex=False)
        .str.slice(1, 8)
    )
    return frame


def evaluate_prior(
    frame: pd.DataFrame,
    counts: pd.DataFrame,
    key_columns: list[str],
    global_rate: float,
    smoothing: float,
) -> dict[str, float | int]:
    joined = frame[key_columns + ["label"]].merge(
        counts,
        left_on=key_columns,
        right_index=True,
        how="left",
        sort=False,
    )
    known = joined["count"].notna().to_numpy()
    sums = joined["sum"].fillna(0.0).to_numpy(dtype=np.float64)
    group_counts = joined["count"].fillna(0.0).to_numpy(dtype=np.float64)
    prediction = (sums + smoothing * global_rate) / (group_counts + smoothing)
    labels = joined["label"].to_numpy(dtype=np.int8)
    return {
        "auprc": float(average_precision_score(labels, prediction)),
        "coverage": float(known.mean()),
        "known_rows": int(known.sum()),
    }


if __name__ == "__main__":
    main()
