import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from phact_mirbind.cache.dataloaders import (
    CachedPairwiseIterableDataset,
    pair_collate,
)
from phact_mirbind.cache.pair_cache import write_pair_cache_from_tsv
from phact_mirbind.data.columns import (
    MANAKOV_COLUMNS,
    PADDING_PAIR_INDEX,
    build_pair_to_index,
)
from phact_mirbind.data.pair_encoding import encode_pair_indices


def test_pair_encoding_matches_mirbind_mirna_target_orientation():
    pair_to_index = build_pair_to_index()

    encoded = encode_pair_indices(
        target_sequence="ATN",
        mirna_sequence="CG",
        target_length=3,
        mirna_length=2,
        pair_to_index=pair_to_index,
    )

    assert encoded.shape == (2, 3)
    assert encoded[0, 0] == pair_to_index[("C", "A")]
    assert encoded[0, 1] == pair_to_index[("C", "T")]
    assert encoded[1, 0] == pair_to_index[("G", "A")]
    assert encoded[1, 1] == pair_to_index[("G", "T")]
    assert encoded[0, 2] == PADDING_PAIR_INDEX
    assert encoded[1, 2] == PADDING_PAIR_INDEX


def test_pair_cache_round_trip(tmp_path: Path):
    data_file = tmp_path / "small.tsv"
    data_file.write_text(
        "gene\tmirna\tlabel\n"
        "ATC\tUG\t1\n"
        "TTA\tCA\t0\n"
    )
    manifest_path = write_pair_cache_from_tsv(
        data_file,
        tmp_path / "cache",
        output_prefix="small",
        shard_size=1,
        target_length=3,
        mirna_length=2,
    )
    dataset = CachedPairwiseIterableDataset(manifest_path)
    loader = DataLoader(dataset, batch_size=2, collate_fn=pair_collate)

    pair_indices, labels = next(iter(loader))

    assert pair_indices.shape == (2, 2, 3)
    assert labels.tolist() == [1.0, 0.0]


def test_pair_cache_reads_full_manakov_schema_by_name(tmp_path: Path):
    data_file = tmp_path / "manakov.tsv"
    row = [
        "ATC",
        "UG",
        "hsa-miR-test",
        "mir-test",
        "three_prime_utr",
        "1",
        "1",
        "100",
        "102",
        "+",
        "1",
        "UTR3",
        "UTR3",
        "10",
        "12",
        "cluster1",
        "[10.0, 0.0, -5.0]",
        "[1.0, 0.5, 0.0]",
    ]
    data_file.write_text("\t".join(MANAKOV_COLUMNS) + "\n" + "\t".join(row) + "\n")

    manifest_path = write_pair_cache_from_tsv(
        data_file,
        tmp_path / "cache",
        output_prefix="manakov",
        target_length=3,
        mirna_length=2,
    )
    manifest = json.loads(manifest_path.read_text())
    dataset = CachedPairwiseIterableDataset(manifest_path)
    pair_indices, label = next(iter(dataset))

    assert pair_indices.shape == (2, 3)
    assert label.item() == 1.0
    assert manifest["source_columns"] == {
        "target_sequence": "gene",
        "mirna_sequence": "noncodingRNA",
        "label": "label",
    }


def test_pair_cache_global_shuffle_matches_torch_randperm(tmp_path: Path):
    data_file = tmp_path / "rows.tsv"
    rows = [f"AAA\tAA\t{idx}" for idx in range(4)]
    data_file.write_text("gene\tnoncodingRNA\tlabel\n" + "\n".join(rows) + "\n")
    manifest_path = write_pair_cache_from_tsv(
        data_file,
        tmp_path / "cache",
        output_prefix="rows",
        shard_size=2,
        target_length=3,
        mirna_length=2,
    )
    dataset = CachedPairwiseIterableDataset(
        manifest_path,
        shuffle_mode="global",
        shuffle_seed=42,
    )

    labels_epoch1 = [int(label.item()) for _, label in dataset]
    labels_epoch2 = [int(label.item()) for _, label in dataset]
    expected_epoch1 = torch.randperm(
        4,
        generator=torch.Generator().manual_seed(42),
    ).tolist()
    expected_epoch2 = torch.randperm(
        4,
        generator=torch.Generator().manual_seed(43),
    ).tolist()

    assert labels_epoch1 == expected_epoch1
    assert labels_epoch2 == expected_epoch2


def test_max_rows_is_global_across_dataloader_workers(tmp_path: Path):
    data_file = tmp_path / "rows.tsv"
    rows = [f"AAA\tAA\t{idx}" for idx in range(8)]
    data_file.write_text("gene\tnoncodingRNA\tlabel\n" + "\n".join(rows) + "\n")
    manifest_path = write_pair_cache_from_tsv(
        data_file,
        tmp_path / "cache",
        output_prefix="rows",
        shard_size=2,
        target_length=3,
        mirna_length=2,
    )

    for shuffle_mode in ("none", "shard", "global"):
        dataset = CachedPairwiseIterableDataset(
            manifest_path,
            max_rows=3,
            shuffle_mode=shuffle_mode,
            shuffle_seed=42,
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=2,
            collate_fn=pair_collate,
        )
        labels = [int(label.item()) for _, label in loader]

        assert len(labels) == 3
        assert len(set(labels)) == 3
