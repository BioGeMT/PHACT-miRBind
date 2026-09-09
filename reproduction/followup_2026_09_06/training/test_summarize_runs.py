"""Exercise summary verification with small, real CPU artifacts."""

import csv
import json

import numpy as np
import pytest
import torch

import summarize_runs as summary


@pytest.fixture
def completed_runs(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setattr(summary, "ROOT", tmp_path)
    monkeypatch.setattr(summary, "PHACT_CACHE", cache)
    monkeypatch.setattr(summary, "EXPECTED_ROWS", {"test": 4, "leftout": 4})
    conditions = ["guide_real", "guide_shuffled"]
    (tmp_path / "protocol.json").write_text(json.dumps({"seeds": [17], "conditions": conditions}))
    labels = np.array([0, 1, 0, 1], dtype=np.float32)
    for split in ("test", "leftout"):
        directory = cache / split
        directory.mkdir(parents=True)
        torch.save({"labels": torch.from_numpy(labels)}, directory / "shard.pt")
        (directory / f"{split}_manifest.json").write_text(json.dumps({
            "shards": [{"file": "shard.pt", "rows": 4}], "row_count": 4,
        }))
    for condition in conditions:
        output = tmp_path / "runs" / "seed_17" / condition
        output.mkdir(parents=True)
        checkpoint = output / "model.pt"
        torch.manual_seed(17)
        torch.save({
            "model_state_dict": torch.nn.Linear(1, 1).state_dict(),
            "epoch": 2, "val_auprc": 0.8,
            "summary": {"condition": condition, "seed": 17, "initial_state_sha256": "matched"},
        }, checkpoint)
        result = dict(condition=condition, seed=17, smoke=False, best_epoch=2,
                      validation_ap=0.8, initial_state_sha256="matched",
                      checkpoint=str(checkpoint), checkpoint_sha256=summary.sha256(checkpoint))
        predictions = np.array([0.1, 0.9, 0.2, 0.8] if condition == "guide_real"
                               else [0.1, 0.4, 0.5, 0.8], dtype=np.float32)
        for split in ("test", "leftout"):
            path = output / f"{split}.npz"
            np.savez_compressed(path, ids=np.array([f"{split}_{i + 1}" for i in range(4)]),
                                labels=labels, predictions=predictions)
            result[split] = {"ap": 1.0 if condition == "guide_real" else 5 / 6,
                             "prediction_sha256": summary.sha256(path)}
        (output / "result.json").write_text(json.dumps(result))
    return tmp_path / "runs" / "seed_17" / "guide_real"


def replace_predictions(run, field, value):
    path = run / "test.npz"
    with np.load(path) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays[field][0] = value
    np.savez_compressed(path, **arrays)
    result_path = run / "result.json"
    result = json.loads(result_path.read_text())
    result["test"]["prediction_sha256"] = summary.sha256(path)
    result_path.write_text(json.dumps(result))


def test_valid_runs_produce_verified_metrics_and_paired_deltas(completed_runs):
    assert summary.summarize() == 2
    with (summary.ROOT / "seed_metrics.tsv").open() as handle:
        metrics = list(csv.DictReader(handle, delimiter="\t"))
    assert [(row["condition"], row["split"]) for row in metrics] == [
        ("guide_real", "test"), ("guide_shuffled", "test"),
        ("guide_real", "leftout"), ("guide_shuffled", "leftout"),
    ]
    assert [float(row["ap"]) for row in metrics] == pytest.approx([1, 5 / 6, 1, 5 / 6])
    with (summary.ROOT / "paired_seed_deltas.tsv").open() as handle:
        deltas = list(csv.DictReader(handle, delimiter="\t"))
    assert [float(row["ap_delta"]) for row in deltas] == pytest.approx([1 / 6, 1 / 6])


def test_corrupted_ids_fail_even_with_updated_prediction_hash(completed_runs):
    replace_predictions(completed_runs, "ids", "test_4")
    with pytest.raises(ValueError, match="Evaluation row mismatch"):
        summary.summarize()


def test_changed_checkpoint_fails_checksum_validation(completed_runs):
    checkpoint = completed_runs / "model.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Checkpoint checksum mismatch"):
        summary.summarize()


@pytest.mark.parametrize("invalid", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_probabilities_fail_even_with_updated_prediction_hash(completed_runs, invalid):
    replace_predictions(completed_runs, "predictions", invalid)
    with pytest.raises(ValueError, match="Invalid probabilities"):
        summary.summarize()
