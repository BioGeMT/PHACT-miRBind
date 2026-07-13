import json
from pathlib import Path

import torch

from phact_mirbind.cache.conservation_cache import (
    write_conservation_cache_from_tsv,
)
from phact_mirbind.cache.dataloaders import (
    CachedConservationPairwiseIterableDataset,
    make_conservation_collate,
)
from phact_mirbind.data.conservation import (
    parse_conservation_features,
)
from phact_mirbind.data.columns import MANAKOV_COLUMNS


def test_parse_conservation_features_supports_subsets():
    assert parse_conservation_features("phylop") == ("phylop",)
    assert parse_conservation_features("phastcons") == ("phastcons",)
    assert parse_conservation_features("phylop,phastcons") == (
        "phylop",
        "phastcons",
    )


def test_conservation_cache_build_and_feature_selection(tmp_path: Path):
    data_file = tmp_path / "conservation.tsv"
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
        "[10.0, -20.0, 5.0]",
        "[1.0, 0.5, 0.0]",
    ]
    data_file.write_text("\t".join(MANAKOV_COLUMNS) + "\n" + "\t".join(row) + "\n")

    manifest = write_conservation_cache_from_tsv(
        data_file,
        tmp_path / "cache",
        output_prefix="test",
        target_length=3,
        mirna_length=2,
    )
    dataset = CachedConservationPairwiseIterableDataset(
        manifest,
        conservation_features="phylop",
    )
    manifest_json = json.loads(manifest.read_text())
    pair_indices, compact, label = next(iter(dataset))
    collate = make_conservation_collate(mirna_length=2)
    batch_pairs, channels, labels = collate([(pair_indices, compact, label)])

    assert batch_pairs.shape == (1, 2, 3)
    assert channels.shape == (1, 1, 2, 3)
    assert labels.tolist() == [1.0]
    assert torch.allclose(channels[0, 0, 0], torch.tensor([1.0, -1.0, 0.5]))
    assert torch.allclose(channels[0, 0, 1], torch.tensor([1.0, -1.0, 0.5]))
    assert manifest_json["source_columns"] == {
        "target_sequence": "gene",
        "mirna_sequence": "noncodingRNA",
        "label": "label",
        "conservation": {
            "phylop": "gene_phyloP",
            "phastcons": "gene_phastCons",
        },
    }
