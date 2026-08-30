import torch

from phact_mirbind.models.conservation import PairwiseConservationCNN
from phact_mirbind.models.phact import PairwisePhactCNN
from phact_mirbind.models.seq_only import PairwiseSeqCNN


def test_pairwise_seq_cnn_forward_shape():
    model = PairwiseSeqCNN(
        num_pair_classes=18,
        target_length=50,
        mirna_length=28,
        embedding_dim=4,
        filter_sizes=(8, 4),
        kernel_sizes=(3, 3),
    )
    pair_indices = torch.zeros(2, 28, 50).long()

    logits = model(pair_indices)

    assert logits.shape == (2,)


def test_pairwise_seq_cnn_default_param_count_matches_original_mirbind():
    model = PairwiseSeqCNN(
        num_pair_classes=18,
        target_length=50,
        mirna_length=28,
        embedding_dim=8,
        filter_sizes=(128, 64, 32),
        kernel_sizes=(6, 3, 3),
    )

    assert sum(param.numel() for param in model.parameters()) == 147_249


def test_pairwise_conservation_cnn_forward_shape():
    model = PairwiseConservationCNN(
        num_pair_classes=18,
        conservation_channel_count=2,
        target_length=50,
        mirna_length=28,
        embedding_dim=4,
        filter_sizes=(8, 4),
        kernel_sizes=(3, 3),
    )
    pair_indices = torch.zeros(2, 28, 50).long()
    conservation = torch.zeros(2, 2, 28, 50)

    logits = model(pair_indices, conservation)

    assert logits.shape == (2,)


def test_pairwise_conservation_cnn_checks_channel_count():
    model = PairwiseConservationCNN(
        num_pair_classes=18,
        conservation_channel_count=2,
        target_length=50,
        mirna_length=28,
    )
    pair_indices = torch.zeros(2, 28, 50).long()
    conservation = torch.zeros(2, 1, 28, 50)

    try:
        model(pair_indices, conservation)
    except ValueError as exc:
        assert "Expected 2 conservation channels" in str(exc)
    else:
        raise AssertionError("Expected ValueError for wrong conservation channel count")


def test_pairwise_phact_cnn_forward_shape():
    model = PairwisePhactCNN(
        num_pair_classes=18,
        phact_channel_count=8,
        target_length=50,
        mirna_length=28,
        embedding_dim=4,
        filter_sizes=(8, 4),
        kernel_sizes=(3, 3),
    )
    pair_indices = torch.zeros(2, 28, 50).long()
    phact = torch.zeros(2, 8, 28, 50)

    logits = model(pair_indices, phact)

    assert logits.shape == (2,)
    assert model.encode(pair_indices, phact).shape == (2, 30)
