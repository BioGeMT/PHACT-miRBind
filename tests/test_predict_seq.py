import numpy as np

from phact_mirbind.cli.predict_seq import encode_batch_pair_indices
from phact_mirbind.data.pair_encoding import encode_pair_indices
from phact_mirbind.data.sequences import normalize_sequence


def test_batch_pair_encoding_matches_training_encoder() -> None:
    targets = ["ATCGUN", "tagc"]
    mirnas = ["CGTA", "uunn"]
    target_length = 6
    mirna_length = 5

    expected = np.stack(
        [
            encode_pair_indices(
                normalize_sequence(target, target_length),
                normalize_sequence(mirna, mirna_length),
                target_length=target_length,
                mirna_length=mirna_length,
            )
            for target, mirna in zip(targets, mirnas, strict=True)
        ]
    )

    actual = encode_batch_pair_indices(
        targets,
        mirnas,
        target_length=target_length,
        mirna_length=mirna_length,
    ).numpy()

    np.testing.assert_array_equal(actual, expected)
