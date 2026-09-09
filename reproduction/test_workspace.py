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
