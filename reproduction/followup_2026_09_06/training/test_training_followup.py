"""Focused checks for the follow-up's new input and ordering code."""

import numpy as np
import torch

from prepare_conservation import encoded_pairs
from run_condition import FollowupDataset, make_model, set_seed
from phact_mirbind.data.pair_encoding import encode_pair_indices


def test_vectorized_source_check_matches_canonical_pair_encoding():
    guides = ["ATCG", "GNAC", "T" * 30, ""]
    targets = ["GCTA", "ANTC", "C" * 55, "ATCG"]
    expected = np.stack([encode_pair_indices(target, guide, target_length=50, mirna_length=28)
                         for guide, target in zip(guides, targets)])
    np.testing.assert_array_equal(encoded_pairs(guides, targets), expected)


def test_real_and_control_initializations_are_identical():
    for axis in ("guide", "target"):
        set_seed(17)
        real, _ = make_model(f"{axis}_real")
        set_seed(17)
        control, _ = make_model(f"{axis}_shuffled")
        assert all(torch.equal(value, control.state_dict()[key])
                   for key, value in real.state_dict().items())


def test_epoch_permutations_match_release_and_evaluation_is_ordered():
    dataset = object.__new__(FollowupDataset)
    dataset.condition = "sequence"
    dataset.split = "train"
    dataset.seed = 43
    dataset.iteration_count = 0
    dataset.tensors = {"pair_indices": torch.arange(20), "labels": torch.arange(20)}
    for epoch in range(2):
        observed = torch.tensor([int(label) for _, label in dataset])
        expected = torch.randperm(20, generator=torch.Generator().manual_seed(43 + epoch))
        assert torch.equal(observed, expected)
    dataset.split = "test"
    assert [int(label) for _, label in dataset] == list(range(20))
