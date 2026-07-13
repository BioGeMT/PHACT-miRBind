from pathlib import Path

import torch

from phact_mirbind.data.row_split import split_original_rows


def test_split_original_rows_matches_torch_random_split_membership(tmp_path: Path):
    input_file = tmp_path / "input.tsv"
    input_file.write_text(
        "gene\tnoncodingRNA\tlabel\n"
        + "".join(f"gene{i}\tmir{i}\t{i % 2}\n" for i in range(20))
    )

    summary = split_original_rows(
        input_file,
        tmp_path / "split",
        output_prefix="rows",
        val_fraction=0.1,
        seed=42,
    )

    train_lines = (tmp_path / "split" / "rows_train.tsv").read_text().splitlines()[1:]
    val_lines = (tmp_path / "split" / "rows_val.tsv").read_text().splitlines()[1:]

    permuted = torch.randperm(20, generator=torch.Generator().manual_seed(42)).tolist()
    expected_val_indices = set(permuted[18:])
    expected_val_genes = {f"gene{idx}" for idx in expected_val_indices}

    assert summary["rows"] == {"train": 18, "val": 2}
    assert {line.split("\t")[0] for line in val_lines} == expected_val_genes
    assert len(train_lines) == 18


def test_split_original_rows_can_preserve_source_row_id(tmp_path: Path):
    input_file = tmp_path / "input.tsv"
    input_file.write_text(
        "gene\tnoncodingRNA\tlabel\n"
        + "".join(f"gene{i}\tmir{i}\t{i % 2}\n" for i in range(10))
    )

    split_original_rows(
        input_file,
        tmp_path / "split",
        output_prefix="rows",
        val_fraction=0.2,
        seed=42,
        include_row_id=True,
        row_id_column="manakov_row_id",
    )

    train_lines = (tmp_path / "split" / "rows_train.tsv").read_text().splitlines()
    val_lines = (tmp_path / "split" / "rows_val.tsv").read_text().splitlines()

    assert train_lines[0].split("\t")[-1] == "manakov_row_id"
    assert val_lines[0].split("\t")[-1] == "manakov_row_id"
    assert all(int(line.split("\t")[-1]) >= 1 for line in train_lines[1:])
    assert all(int(line.split("\t")[-1]) >= 1 for line in val_lines[1:])
