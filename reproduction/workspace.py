"""Locate retained PHACT data without changing historical manifests."""
import os
from pathlib import Path

WORKSPACE = Path(os.environ.get("PHACT_WORKSPACE", Path.home() / "phact-workspace"))
DATASETS = Path(os.environ.get("PHACT_DATASETS", WORKSPACE / "data/inputs/manakov_datasets"))
REPO = Path(__file__).resolve().parents[1]

RELOCATIONS = {
    "/SCRATCH/dtzim01/phact/rambunctious_tablespoon_missed": "archive/agentomics-history",
    "/SCRATCH/dtzim01/phact/drive_import_20260813": "data/inputs/score_import_20260813",
    "/SCRATCH/dtzim01/phact/manakov_phact_ag_ready_leftout": "data/splits/manakov_agentomics_leftout",
    "/SCRATCH/dtzim01/phact/manakov_phact_ag_ready": "data/splits/manakov_agentomics_ready",
    "/SCRATCH/dtzim01/phact/phact_final": "analyses/final",
    "/SCRATCH/dtzim01/phact/runs/legacy_baselines": "runs/baselines",
    "/home/dtzim01/PHACT-miRBind/data/phact_cache_CountNodes_3": "data/caches/phact_cache_CountNodes_3",
    "/home/dtzim01/PHACT-miRBind/data/presplit_phact_original_rows": "data/inputs/manakov_original_rows",
    "/home/dtzim01/PHACT-miRBind/reports/phact_score_ranges": "data/scores/main_repo",
    "/home/dtzim01/PHACT-miRBind/outputs": "runs/baselines/main_repo_outputs",
    "/home/dtzim01/phact_mirbind_leaderboard_artifacts": "archive/legacy_leaderboard_artifacts",
    "/SCRATCH/dtzim01/phact/phact_param1_retraining": "runs/param1-training",
    "/SCRATCH/dtzim01/phact/phact_experiments": "runs/experiments",
    "/SCRATCH/dtzim01/phact/followup_2026-09-06": "runs/followup-2026-09-06",
    "/SCRATCH/dtzim01/phact/inputs/manakov_original_rows": "data/inputs/manakov_original_rows",
    "/SCRATCH/dtzim01/phact/derived/caches": "data/caches",
    "/SCRATCH/dtzim01/phact/derived/score_tables": "data/scores",
    "/SCRATCH/dtzim01/phact/phact_param1_leaderboard_artifacts/release": "releases/mirbench_v7",
}


def historical_path(value):
    """Translate known old roots; leave unrelated and relative paths unchanged."""
    path = Path(value)
    for old, new in RELOCATIONS.items():
        if path.is_relative_to(old):
            return WORKSPACE / new / path.relative_to(old)
    if path.is_relative_to("/home/dtzim01/manakov_datasets"):
        return DATASETS / path.relative_to("/home/dtzim01/manakov_datasets")
    return path
