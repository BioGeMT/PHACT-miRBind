import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from phact_mirbind.training.loop import (
    configure_new_input_channels_only,
    load_widened_state_dict,
    run_epoch,
    train_model,
)
from phact_mirbind.training.losses import AsymmetricLabelSmoothingBCEWithLogitsLoss


class TinyBinaryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


def loader() -> DataLoader:
    features = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    return DataLoader(TensorDataset(features, labels), batch_size=2)


def test_train_model_runs_test_and_leftout_only_after_training(tmp_path: Path):
    model = TinyBinaryModel()
    outputs = train_model(
        model=model,
        loaders={
            "train": loader(),
            "val": loader(),
            "test": loader(),
            "leftout": loader(),
        },
        optimizer=SGD(model.parameters(), lr=0.01),
        criterion=nn.BCEWithLogitsLoss(),
        device=torch.device("cpu"),
        output_dir=tmp_path,
        checkpoint_prefix="tiny",
        summary={"model_type": "tiny", "model_params": {}},
        num_epochs=2,
        patience=2,
        progress_every=0,
        show_progress=False,
    )

    history = json.loads(outputs["history"].read_text())
    final_evaluation = json.loads(outputs["final_evaluation"].read_text())

    assert len(history) == 2
    assert all("val" in record for record in history)
    assert all("test" not in record for record in history)
    assert all("leftout" not in record for record in history)
    assert set(final_evaluation) == {"best_epoch", "test", "leftout"}


def test_run_epoch_converts_bfloat16_probabilities_for_metrics(tmp_path: Path):
    model = TinyBinaryModel()

    class BFloat16Output(nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return model(features).to(torch.bfloat16)

    metrics = run_epoch(
        BFloat16Output(),
        loader(),
        nn.BCEWithLogitsLoss(),
        torch.device("cpu"),
        optimizer=None,
        epoch=1,
        phase="test",
        progress_every=0,
        progress_log_path=tmp_path / "progress.tsv",
        show_progress=False,
    )

    assert metrics["samples"] == 4


def test_run_epoch_accepts_per_row_sample_weights(tmp_path: Path):
    class WeightedDataset(TensorDataset):
        has_sample_weights = True

    features = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    weights = torch.tensor([0.0, 0.25, 1.0, 1.0])
    weighted_loader = DataLoader(
        WeightedDataset(features, labels, weights), batch_size=2
    )

    metrics = run_epoch(
        TinyBinaryModel(),
        weighted_loader,
        AsymmetricLabelSmoothingBCEWithLogitsLoss(),
        torch.device("cpu"),
        optimizer=None,
        epoch=1,
        phase="test",
        progress_every=0,
        progress_log_path=tmp_path / "progress.tsv",
        show_progress=False,
    )

    assert metrics["samples"] == 4


def test_load_widened_state_dict_preserves_narrow_model_function():
    torch.manual_seed(4)
    narrow = nn.Sequential(nn.Linear(4, 3), nn.LeakyReLU(), nn.Linear(3, 2))
    wide = nn.Sequential(nn.Linear(4, 5), nn.LeakyReLU(), nn.Linear(5, 2))
    features = torch.randn(8, 4)

    load_widened_state_dict(wide, narrow.state_dict())

    torch.testing.assert_close(wide(features), narrow(features))


def test_configure_new_input_channels_only_masks_existing_connections():
    class ConvModel(nn.Module):
        def __init__(self, input_channels: int) -> None:
            super().__init__()
            self.conv_layers = nn.ModuleList([nn.Conv2d(input_channels, 2, 1)])
            self.classifier = nn.Linear(2, 1)

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            hidden = self.conv_layers[0](features).mean(dim=(2, 3))
            return self.classifier(hidden).squeeze(-1)

    narrow = ConvModel(3)
    wide = ConvModel(5)
    load_widened_state_dict(wide, narrow.state_dict())

    trainable_weights = configure_new_input_channels_only(wide, narrow.state_dict())
    wide(torch.randn(4, 5, 2, 2)).sum().backward()
    gradient = wide.conv_layers[0].weight.grad

    assert trainable_weights == 4
    assert gradient is not None
    assert torch.count_nonzero(gradient[:, :3]) == 0
    assert torch.count_nonzero(gradient[:, 3:]) > 0
    assert not wide.classifier.weight.requires_grad
