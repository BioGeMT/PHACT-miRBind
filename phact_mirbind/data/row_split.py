"""Original miRBind-style random row splitting."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def split_original_rows(
    input_file: Path,
    output_dir: Path,
    *,
    output_prefix: str = "manakov_original_rows",
    val_fraction: float = 0.1,
    seed: int = 42,
    include_row_id: bool = False,
    row_id_column: str = "manakov_row_id",
) -> dict[str, object]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"{output_prefix}_train.tsv"
    val_path = output_dir / f"{output_prefix}_val.tsv"
    summary_path = output_dir / f"{output_prefix}_summary.json"

    row_count = count_data_rows(input_file)
    val_count = int(row_count * val_fraction)
    train_count = row_count - val_count

    generator = torch.Generator().manual_seed(seed)
    permuted_indices = torch.randperm(row_count, generator=generator).tolist()
    val_indices = set(permuted_indices[train_count:])

    observed = {"train": 0, "val": 0}
    with input_file.open("r", newline="") as input_handle:
        header = input_handle.readline()
        output_header = append_row_id(header, row_id_column) if include_row_id else header
        with train_path.open("w", newline="") as train_handle, val_path.open(
            "w", newline=""
        ) as val_handle:
            train_handle.write(output_header)
            val_handle.write(output_header)

            for row_idx, line in enumerate(input_handle):
                output_line = (
                    append_row_id(line, str(row_idx + 1)) if include_row_id else line
                )
                if row_idx in val_indices:
                    val_handle.write(output_line)
                    observed["val"] += 1
                else:
                    train_handle.write(output_line)
                    observed["train"] += 1

    summary: dict[str, object] = {
        "input_file": str(input_file),
        "train_file": str(train_path),
        "val_file": str(val_path),
        "summary_file": str(summary_path),
        "split": "torch_random_split_rows",
        "val_fraction": val_fraction,
        "seed": seed,
        "include_row_id": include_row_id,
        "row_id_column": row_id_column if include_row_id else None,
        "row_count": row_count,
        "rows": observed,
        "matches_original_mirbind_counts": observed == {
            "train": train_count,
            "val": val_count,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def count_data_rows(input_file: Path) -> int:
    with input_file.open("r", newline="") as handle:
        next(handle)
        return sum(1 for _ in handle)


def append_row_id(line: str, value: str) -> str:
    return line.rstrip("\r\n") + f"\t{value}\n"
