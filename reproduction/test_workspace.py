from pathlib import Path
import workspace


def test_frozen_checkpoint_path_resolves_after_move(monkeypatch, tmp_path):
    monkeypatch.setattr(workspace, "WORKSPACE", tmp_path)
    checkpoint = tmp_path / "runs/followup-2026-09-06/training/runs/seed_17/sequence/model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"retained checkpoint")
    old = "/SCRATCH/dtzim01/phact/followup_2026-09-06/training/runs/seed_17/sequence/model.pt"
    assert workspace.historical_path(old).read_bytes() == b"retained checkpoint"


def test_unrelated_paths_are_not_rewritten():
    path = Path("/SCRATCH/dtzim01/phact/phact_experiments_other/file")
    assert workspace.historical_path(path) == path


def test_shared_dataset_uses_configured_copy(monkeypatch, tmp_path):
    monkeypatch.setattr(workspace, "DATASETS", tmp_path)
    assert workspace.historical_path("/home/dtzim01/manakov_datasets/test.tsv") == tmp_path / "test.tsv"


def test_preorganization_and_relative_training_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(workspace, "WORKSPACE", tmp_path)
    old = "/SCRATCH/dtzim01/phact_experiments/augmented_rows_v1/manakov_train_clean.tsv"
    expected = tmp_path / "runs/experiments/augmented_rows_v1/manakov_train_clean.tsv"
    assert workspace.historical_path(old) == expected
    relative = "data/presplit_phact_original_rows/manakov_original_rows_train.tsv"
    expected = tmp_path / "data/inputs/manakov_original_rows/manakov_original_rows_train.tsv"
    assert workspace.historical_path(relative) == expected
