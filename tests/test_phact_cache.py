import json
from pathlib import Path

import pytest
import torch

from phact_mirbind.cache.dataloaders import (
    CachedPhactPairwiseIterableDataset,
    make_phact_collate,
)
from phact_mirbind.cache.phact_cache import write_phact_cache_from_tsv
from phact_mirbind.data.columns import MANAKOV_COLUMNS


def test_phact_cache_builds_compact_scores_and_fills_missing_with_neutral(
    tmp_path: Path,
):
    input_file = tmp_path / "input.tsv"
    header = [*MANAKOV_COLUMNS, "manakov_row_id"]
    input_file.write_text(
        "\t".join(header)
        + "\n"
        + "\n".join(
            [
                "\t".join(manakov_row("ATC", "UG", 1, 1)),
                "\t".join(manakov_row("GGA", "CA", 0, 3)),
            ]
        )
        + "\n"
    )
    mirna_phact_file = tmp_path / "mirna_phact.tsv"
    mirna_phact_file.write_text(
        "split\tmanakov_row_id\tmirna_position_1based\tactual_nt\t"
        "phact_CountNodes_2_A\tphact_CountNodes_2_C\t"
        "phact_CountNodes_2_G\tphact_CountNodes_2_T\n"
        "train\t1\t1\tT\t0.1\t0.2\t0.3\t0.4\n"
        "train\t1\t2\tG\tNA\tNA\tNA\tNA\n"
        "train\t2\t1\tA\t0.0\t0.0\t0.0\t0.0\n"
        "train\t2\t2\tA\t0.0\t0.0\t0.0\t0.0\n"
        "train\t3\t1\tC\t0.9\t0.8\t0.7\t0.6\n"
        "train\t3\t2\tA\t0.5\t0.5\t0.5\t0.5\n"
    )
    target_phact_file = tmp_path / "target_phact.tsv"
    target_phact_file.write_text(
        "split\tmanakov_row_id\ttarget_position_1based\tgenomic_position\t"
        "actual_nt\tscore_A\tscore_C\tscore_G\tscore_T\n"
        "train\t1\t1\t10\tA\t0.11\t0.12\t0.13\t0.14\n"
        "train\t1\t2\t11\tT\tNA\tNA\tNA\tNA\n"
        "train\t1\t3\t12\tC\t0.31\t0.32\t0.33\t0.34\n"
        "train\t2\t1\t20\tA\t0.0\t0.0\t0.0\t0.0\n"
        "train\t2\t2\t21\tA\t0.0\t0.0\t0.0\t0.0\n"
        "train\t2\t3\t22\tA\t0.0\t0.0\t0.0\t0.0\n"
        "train\t3\t1\t30\tG\t0.91\t0.92\t0.93\t0.94\n"
        "train\t3\t2\t31\tG\t0.81\t0.82\t0.83\t0.84\n"
        "train\t3\t3\t32\tA\t0.71\t0.72\t0.73\t0.74\n"
    )

    manifest_path = write_phact_cache_from_tsv(
        input_file,
        tmp_path / "cache",
        output_prefix="test",
        phact_split="train",
        phact_model="CountNodes_2",
        mirna_phact_file=mirna_phact_file,
        target_phact_file=target_phact_file,
        target_length=3,
        mirna_length=2,
    )
    dataset = CachedPhactPairwiseIterableDataset(manifest_path)
    items = list(iter(dataset))
    batch_pairs, channels, labels = make_phact_collate()(items)
    _, mirna_channels, _ = make_phact_collate("mirna")(items)
    _, target_channels, _ = make_phact_collate("target")(items)
    manifest = json.loads(manifest_path.read_text())

    assert batch_pairs.shape == (2, 2, 3)
    assert channels.shape == (2, 10, 2, 3)
    assert mirna_channels.shape == (2, 5, 2, 3)
    assert target_channels.shape == (2, 5, 2, 3)
    assert labels.tolist() == [1.0, 0.0]
    assert torch.allclose(
        channels[0, 0, 0],
        torch.tensor([0.1, 0.1, 0.1]),
        atol=1e-3,
    )
    assert torch.allclose(
        channels[0, 0, 1],
        torch.tensor([0.5, 0.5, 0.5]),
        atol=1e-3,
    )
    assert torch.allclose(channels[0, 4, 0], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(channels[0, 4, 1], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(channels[0, 6, :, 1], torch.tensor([0.5, 0.5]), atol=1e-3)
    assert torch.allclose(channels[0, 7, :, 2], torch.tensor([0.33, 0.33]), atol=1e-3)
    assert torch.allclose(channels[0, 9, :, 1], torch.tensor([1.0, 1.0]))
    assert torch.allclose(mirna_channels, channels[:, :5], atol=1e-3)
    assert torch.allclose(target_channels, channels[:, 5:], atol=1e-3)
    assert manifest["phact_models"] == ["CountNodes_2"]
    assert manifest["target_phact_models"] == ["target_score"]
    assert manifest["phact_axis_channel_counts"] == {"mirna": 4, "target": 4}
    assert manifest["phact_axis_missingness_channel_counts"] == {
        "mirna": 1,
        "target": 1,
    }
    assert manifest["phact_axis_total_channel_counts"] == {"mirna": 5, "target": 5}
    assert dataset.phact_channel_count == 10
    assert manifest["phact_missing"]["fill_value"] == 0.5
    assert manifest["phact_missing"]["mirna_positions_filled"] == 1
    assert manifest["phact_missing"]["target_positions_filled"] == 1

    shard_path = manifest_path.parent / manifest["shards"][0]["file"]
    legacy_shard = torch.load(shard_path, map_location="cpu")
    legacy_shard.pop("mirna_phact_missing")
    legacy_shard.pop("target_phact_missing")
    torch.save(legacy_shard, shard_path)
    manifest.pop("phact_axis_total_channel_counts")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    legacy_dataset = CachedPhactPairwiseIterableDataset(manifest_path)
    _, legacy_channels, _ = make_phact_collate()(list(legacy_dataset))

    assert legacy_channels.shape == (2, 8, 2, 3)
    assert legacy_dataset.phact_channel_count == 8


def test_phact_cache_supports_multiple_named_models_on_both_axes(tmp_path: Path):
    input_file = tmp_path / "input.tsv"
    header = [*MANAKOV_COLUMNS, "manakov_row_id"]
    input_file.write_text(
        "\t".join(header)
        + "\n"
        + "\t".join(manakov_row("ATC", "UG", 1, 1))
        + "\n"
    )
    mirna_phact_file = tmp_path / "mirna_phact.tsv"
    mirna_phact_file.write_text(
        "split\tmanakov_row_id\tmirna_position_1based\tactual_nt\t"
        "phact_CountNodes_3_A\tphact_CountNodes_3_C\t"
        "phact_CountNodes_3_G\tphact_CountNodes_3_T\t"
        "phact_CountNodes_4_A\tphact_CountNodes_4_C\t"
        "phact_CountNodes_4_G\tphact_CountNodes_4_T\n"
        "train\t1\t1\tT\t0.1\t0.2\t0.3\t0.4\t0.5\t0.6\t0.7\t0.8\n"
        "train\t1\t2\tG\tNA\tNA\tNA\tNA\t0.9\t0.8\t0.7\t0.6\n"
    )
    target_phact_file = tmp_path / "target_phact.tsv"
    target_phact_file.write_text(
        "split\tmanakov_row_id\ttarget_position_1based\tgenomic_position\tactual_nt\t"
        "phact_CountNodes_3_A\tphact_CountNodes_3_C\t"
        "phact_CountNodes_3_G\tphact_CountNodes_3_T\t"
        "phact_CountNodes_4_A\tphact_CountNodes_4_C\t"
        "phact_CountNodes_4_G\tphact_CountNodes_4_T\n"
        "train\t1\t1\t10\tA\t0.11\t0.12\t0.13\t0.14\t0.15\t0.16\t0.17\t0.18\n"
        "train\t1\t2\t11\tT\t0.21\t0.22\t0.23\t0.24\tNA\tNA\tNA\tNA\n"
        "train\t1\t3\t12\tC\t0.31\t0.32\t0.33\t0.34\t0.35\t0.36\t0.37\t0.38\n"
    )

    manifest_path = write_phact_cache_from_tsv(
        input_file,
        tmp_path / "cache",
        output_prefix="test",
        phact_split="train",
        phact_models="CountNodes_3,CountNodes_4",
        target_phact_models="CountNodes_3,CountNodes_4",
        mirna_phact_file=mirna_phact_file,
        target_phact_file=target_phact_file,
        target_length=3,
        mirna_length=2,
    )
    dataset = CachedPhactPairwiseIterableDataset(manifest_path)
    items = list(iter(dataset))
    _, channels, _ = make_phact_collate("both")(items)
    _, mirna_channels, _ = make_phact_collate("mirna")(items)
    _, target_channels, _ = make_phact_collate("target")(items)
    manifest = json.loads(manifest_path.read_text())

    assert channels.shape == (1, 20, 2, 3)
    assert mirna_channels.shape == (1, 10, 2, 3)
    assert target_channels.shape == (1, 10, 2, 3)
    assert manifest["phact_models"] == ["CountNodes_3", "CountNodes_4"]
    assert manifest["target_phact_models"] == ["CountNodes_3", "CountNodes_4"]
    assert manifest["phact_axis_channel_counts"] == {"mirna": 8, "target": 8}
    assert manifest["phact_axis_missingness_channel_counts"] == {
        "mirna": 2,
        "target": 2,
    }
    assert manifest["phact_axis_total_channel_counts"] == {
        "mirna": 10,
        "target": 10,
    }
    assert torch.allclose(channels[0, 0, 0], torch.tensor([0.1, 0.1, 0.1]), atol=1e-3)
    assert torch.allclose(channels[0, 0, 1], torch.tensor([0.5, 0.5, 0.5]), atol=1e-3)
    assert torch.allclose(channels[0, 4, 1], torch.tensor([0.9, 0.9, 0.9]), atol=1e-3)
    assert torch.allclose(channels[0, 9, 1], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(channels[0, 14, :, 1], torch.tensor([0.5, 0.5]), atol=1e-3)
    assert torch.allclose(channels[0, 19, :, 1], torch.tensor([1.0, 1.0]))
    assert manifest["phact_missing"]["mirna_score_groups_filled"] == 1
    assert manifest["phact_missing"]["target_score_groups_filled"] == 1


def test_phact_cache_actual_margin_reduction_uses_actual_minus_best_alt(
    tmp_path: Path,
):
    input_file = tmp_path / "input.tsv"
    header = [*MANAKOV_COLUMNS, "manakov_row_id"]
    input_file.write_text(
        "\t".join(header)
        + "\n"
        + "\t".join(manakov_row("ATN", "UG", 1, 1))
        + "\n"
    )
    mirna_phact_file = tmp_path / "mirna_phact.tsv"
    mirna_phact_file.write_text(
        "split\tmanakov_row_id\tmirna_position_1based\tactual_nt\t"
        "phact_CountNodes_3_A\tphact_CountNodes_3_C\t"
        "phact_CountNodes_3_G\tphact_CountNodes_3_T\n"
        "train\t1\t1\tT\t0.1\t0.2\t0.3\t0.9\n"
        "train\t1\t2\tG\tNA\tNA\tNA\tNA\n"
    )
    target_phact_file = tmp_path / "target_phact.tsv"
    target_phact_file.write_text(
        "split\tmanakov_row_id\ttarget_position_1based\tgenomic_position\t"
        "actual_nt\tscore_A\tscore_C\tscore_G\tscore_T\n"
        "train\t1\t1\t10\tA\t0.8\t0.4\t0.2\t0.1\n"
        "train\t1\t2\t11\tT\t0.9\t0.1\t0.2\t0.3\n"
        "train\t1\t3\t12\tN\t0.9\t0.1\t0.2\t0.3\n"
    )

    manifest_path = write_phact_cache_from_tsv(
        input_file,
        tmp_path / "cache",
        output_prefix="test",
        phact_split="train",
        phact_models="CountNodes_3",
        mirna_phact_file=mirna_phact_file,
        target_phact_file=target_phact_file,
        target_length=3,
        mirna_length=2,
        phact_reduction="actual_margin",
    )
    dataset = CachedPhactPairwiseIterableDataset(manifest_path)
    items = list(iter(dataset))
    _, channels, _ = make_phact_collate("both")(items)
    _, mirna_channels, _ = make_phact_collate("mirna")(items)
    _, target_channels, _ = make_phact_collate("target")(items)
    manifest = json.loads(manifest_path.read_text())

    assert channels.shape == (1, 4, 2, 3)
    assert mirna_channels.shape == (1, 2, 2, 3)
    assert target_channels.shape == (1, 2, 2, 3)
    assert manifest["phact_reduction"] == "actual_margin"
    assert manifest["phact_axis_channel_counts"] == {"mirna": 1, "target": 1}
    assert manifest["phact_missing"]["fill_value"] == 0.0
    assert torch.allclose(channels[0, 0, 0], torch.tensor([0.6, 0.6, 0.6]), atol=1e-3)
    assert torch.allclose(channels[0, 0, 1], torch.tensor([0.0, 0.0, 0.0]), atol=1e-3)
    assert torch.allclose(channels[0, 1, 1], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(channels[0, 2, :, 0], torch.tensor([0.4, 0.4]), atol=1e-3)
    assert torch.allclose(channels[0, 2, :, 1], torch.tensor([-0.6, -0.6]), atol=1e-3)
    assert torch.allclose(channels[0, 2, :, 2], torch.tensor([0.0, 0.0]), atol=1e-3)
    assert torch.allclose(channels[0, 3, :, 2], torch.tensor([1.0, 1.0]))


def test_phact_cache_alt_mean_reduction_uses_mean_of_non_actual_scores(
    tmp_path: Path,
):
    input_file = tmp_path / "input.tsv"
    header = [*MANAKOV_COLUMNS, "manakov_row_id"]
    input_file.write_text(
        "\t".join(header)
        + "\n"
        + "\t".join(manakov_row("ATN", "UG", 1, 1))
        + "\n"
    )
    mirna_phact_file = tmp_path / "mirna_phact.tsv"
    mirna_phact_file.write_text(
        "split\tmanakov_row_id\tmirna_position_1based\tactual_nt\t"
        "phact_CountNodes_3_A\tphact_CountNodes_3_C\t"
        "phact_CountNodes_3_G\tphact_CountNodes_3_T\n"
        "train\t1\t1\tT\t0.1\t0.2\t0.3\t0.9\n"
        "train\t1\t2\tG\tNA\tNA\tNA\tNA\n"
    )
    target_phact_file = tmp_path / "target_phact.tsv"
    target_phact_file.write_text(
        "split\tmanakov_row_id\ttarget_position_1based\tgenomic_position\t"
        "actual_nt\tscore_A\tscore_C\tscore_G\tscore_T\n"
        "train\t1\t1\t10\tA\t0.8\t0.4\t0.2\t0.1\n"
        "train\t1\t2\t11\tT\t0.9\t0.1\t0.2\t0.3\n"
        "train\t1\t3\t12\tN\t0.9\t0.1\t0.2\t0.3\n"
    )

    manifest_path = write_phact_cache_from_tsv(
        input_file,
        tmp_path / "cache",
        output_prefix="test",
        phact_split="train",
        phact_models="CountNodes_3",
        mirna_phact_file=mirna_phact_file,
        target_phact_file=target_phact_file,
        target_length=3,
        mirna_length=2,
        phact_reduction="alt_mean",
    )
    dataset = CachedPhactPairwiseIterableDataset(manifest_path)
    items = list(iter(dataset))
    _, channels, _ = make_phact_collate("both")(items)
    _, mirna_channels, _ = make_phact_collate("mirna")(items)
    _, target_channels, _ = make_phact_collate("target")(items)
    manifest = json.loads(manifest_path.read_text())

    assert channels.shape == (1, 4, 2, 3)
    assert mirna_channels.shape == (1, 2, 2, 3)
    assert target_channels.shape == (1, 2, 2, 3)
    assert manifest["phact_reduction"] == "alt_mean"
    assert manifest["phact_axis_channel_counts"] == {"mirna": 1, "target": 1}
    assert manifest["phact_missing"]["fill_value"] == 0.5
    assert manifest["phact_score_channel_order"] == [
        "mirna_CountNodes_3_alt_mean",
        "target_target_score_alt_mean",
    ]
    assert manifest["phact_channel_order"] == [
        "mirna_CountNodes_3_alt_mean",
        "mirna_CountNodes_3_missing",
        "target_target_score_alt_mean",
        "target_target_score_missing",
    ]
    assert torch.allclose(channels[0, 0, 0], torch.tensor([0.2, 0.2, 0.2]), atol=1e-3)
    assert torch.allclose(channels[0, 0, 1], torch.tensor([0.5, 0.5, 0.5]), atol=1e-3)
    assert torch.allclose(
        channels[0, 2, :, 0],
        torch.tensor([0.233333, 0.233333]),
        atol=1e-3,
    )
    assert torch.allclose(channels[0, 2, :, 1], torch.tensor([0.4, 0.4]), atol=1e-3)
    assert torch.allclose(channels[0, 2, :, 2], torch.tensor([0.5, 0.5]), atol=1e-3)
    assert torch.allclose(channels[0, 3, :, 2], torch.tensor([1.0, 1.0]))


def test_phact_cache_rejects_multiple_target_models_without_named_target_columns(
    tmp_path: Path,
):
    input_file = tmp_path / "input.tsv"
    header = [*MANAKOV_COLUMNS, "manakov_row_id"]
    input_file.write_text(
        "\t".join(header)
        + "\n"
        + "\t".join(manakov_row("ATC", "UG", 1, 1))
        + "\n"
    )
    mirna_phact_file = tmp_path / "mirna_phact.tsv"
    mirna_phact_file.write_text(
        "split\tmanakov_row_id\tmirna_position_1based\tactual_nt\t"
        "phact_CountNodes_3_A\tphact_CountNodes_3_C\t"
        "phact_CountNodes_3_G\tphact_CountNodes_3_T\t"
        "phact_CountNodes_4_A\tphact_CountNodes_4_C\t"
        "phact_CountNodes_4_G\tphact_CountNodes_4_T\n"
        "train\t1\t1\tT\t0.1\t0.2\t0.3\t0.4\t0.5\t0.6\t0.7\t0.8\n"
        "train\t1\t2\tG\t0.1\t0.2\t0.3\t0.4\t0.5\t0.6\t0.7\t0.8\n"
    )
    target_phact_file = tmp_path / "target_phact.tsv"
    target_phact_file.write_text(
        "split\tmanakov_row_id\ttarget_position_1based\tgenomic_position\t"
        "actual_nt\tscore_A\tscore_C\tscore_G\tscore_T\n"
        "train\t1\t1\t10\tA\t0.11\t0.12\t0.13\t0.14\n"
        "train\t1\t2\t11\tT\t0.21\t0.22\t0.23\t0.24\n"
        "train\t1\t3\t12\tC\t0.31\t0.32\t0.33\t0.34\n"
    )

    with pytest.raises(ValueError, match="only one unmodelled score group"):
        write_phact_cache_from_tsv(
            input_file,
            tmp_path / "cache",
            output_prefix="test",
            phact_split="train",
            phact_models="CountNodes_3,CountNodes_4",
            target_phact_models="CountNodes_3,CountNodes_4",
            mirna_phact_file=mirna_phact_file,
            target_phact_file=target_phact_file,
            target_length=3,
            mirna_length=2,
        )


def test_phact_cache_rejects_partial_missing_scores(tmp_path: Path):
    input_file = tmp_path / "input.tsv"
    header = [*MANAKOV_COLUMNS, "manakov_row_id"]
    input_file.write_text(
        "\t".join(header)
        + "\n"
        + "\t".join(manakov_row("ATC", "UG", 1, 1))
        + "\n"
    )
    mirna_phact_file = tmp_path / "mirna_phact.tsv"
    mirna_phact_file.write_text(
        "split\tmanakov_row_id\tmirna_position_1based\tactual_nt\t"
        "phact_CountNodes_2_A\tphact_CountNodes_2_C\t"
        "phact_CountNodes_2_G\tphact_CountNodes_2_T\n"
        "train\t1\t1\tT\t0.1\t0.2\t0.3\t0.4\n"
        "train\t1\t2\tG\t0.1\t0.2\t0.3\t0.4\n"
    )
    target_phact_file = tmp_path / "target_phact.tsv"
    target_phact_file.write_text(
        "split\tmanakov_row_id\ttarget_position_1based\tgenomic_position\t"
        "actual_nt\tscore_A\tscore_C\tscore_G\tscore_T\n"
        "train\t1\t1\t10\tA\t0.11\t0.12\t0.13\t0.14\n"
        "train\t1\t2\t11\tT\tNA\t0.22\t0.23\t0.24\n"
        "train\t1\t3\t12\tC\t0.31\t0.32\t0.33\t0.34\n"
    )

    with pytest.raises(ValueError, match="partial missing"):
        write_phact_cache_from_tsv(
            input_file,
            tmp_path / "cache",
            output_prefix="test",
            phact_split="train",
            phact_model="CountNodes_2",
            mirna_phact_file=mirna_phact_file,
            target_phact_file=target_phact_file,
            target_length=3,
            mirna_length=2,
        )


def manakov_row(gene: str, mirna: str, label: int, row_id: int) -> list[str]:
    return [
        gene,
        mirna,
        "mir-name",
        "mir-fam",
        "three_prime_utr",
        str(label),
        "1",
        "1",
        "3",
        "+",
        "1",
        "UTR3",
        "UTR3",
        "1",
        "3",
        "cluster1",
        "[]",
        "[]",
        str(row_id),
    ]
