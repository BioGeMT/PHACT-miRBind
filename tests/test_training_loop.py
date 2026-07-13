import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from phact_mirbind.training.loop import train_model


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
