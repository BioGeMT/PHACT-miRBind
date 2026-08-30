import pytest
import torch
import torch.nn.functional as F

from phact_mirbind.training.losses import (
    AsymmetricLabelSmoothingBCEWithLogitsLoss,
)


def test_asymmetric_label_smoothing_uses_separate_class_targets() -> None:
    logits = torch.tensor([-1.0, 1.0])
    labels = torch.tensor([0.0, 1.0])
    loss = AsymmetricLabelSmoothingBCEWithLogitsLoss(
        negative_smoothing=0.05,
        positive_smoothing=0.1,
    )(logits, labels)

    expected = F.binary_cross_entropy_with_logits(
        logits,
        torch.tensor([0.05, 0.9]),
    )
    torch.testing.assert_close(loss, expected)


def test_asymmetric_label_smoothing_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="negative_smoothing"):
        AsymmetricLabelSmoothingBCEWithLogitsLoss(negative_smoothing=0.5)

    with pytest.raises(ValueError, match="focal_gamma"):
        AsymmetricLabelSmoothingBCEWithLogitsLoss(focal_gamma=-0.1)


def test_focal_gamma_downweights_easy_examples() -> None:
    logits = torch.tensor([4.0, -4.0])
    labels = torch.tensor([1.0, 0.0])
    plain = AsymmetricLabelSmoothingBCEWithLogitsLoss()(logits, labels)
    focal = AsymmetricLabelSmoothingBCEWithLogitsLoss(focal_gamma=1.0)(
        logits, labels
    )

    assert focal < plain


def test_sample_weights_scale_per_row_losses() -> None:
    logits = torch.tensor([0.0, 1.0])
    labels = torch.tensor([0.0, 1.0])
    weights = torch.tensor([0.25, 1.0])
    criterion = AsymmetricLabelSmoothingBCEWithLogitsLoss()

    actual = criterion(logits, labels, weights)
    row_losses = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    expected = (row_losses * weights).sum() / weights.sum()

    torch.testing.assert_close(actual, expected)
