#!/usr/bin/env python3
"""Build and validate the five PHACT-P1 miRBench v7 release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


RELEASE_NAME = "PHACT consensus parameter 1 / miRBench v7"
RELEASE_DATE = "2026-08-30"
CODE_TAG = "mirbench-v7-phact-param1-2026-08"

MODELS: list[dict[str, Any]] = [
    {
        "slug": "phact_p1_mirna_cnn",
        "name": "PHACT-P1 miRNA CNN",
        "family": "cnn",
        "run_dir": "outputs/param_1_mirna_cnn",
        "prediction_prefix": "model1",
        "expected": {"test": 0.8777361869042706, "leftout": 0.8702628283649391},
        "representation": "18-state miRNA-target pair grid plus four position-specific mature-miRNA PHACT-P1 channels",
        "architecture": "PairwisePhactCNN; 8-d pair embedding; 3 convolution blocks (128/64/32 filters; 6/3/3 kernels)",
    },
    {
        "slug": "phact_p1_mirna_target_cnn",
        "name": "PHACT-P1 miRNA + Target CNN",
        "family": "cnn",
        "run_dir": "outputs/param_1_mirna_target_cnn",
        "prediction_prefix": "model2",
        "expected": {"test": 0.8859630990269719, "leftout": 0.8649006093057304},
        "representation": "18-state pair grid plus four mature-miRNA and four target-position PHACT-P1 channels",
        "architecture": "PairwisePhactCNN; 8-d pair embedding; 3 convolution blocks (128/64/32 filters; 6/3/3 kernels)",
    },
    {
        "slug": "phact_p1_conservation_cnn",
        "name": "PHACT-P1 miRNA + Target + Conservation CNN",
        "family": "cnn",
        "run_dir": "outputs/param_1_conservation_conflict28_shift1_focal1_cnn",
        "prediction_prefix": "model3",
        "expected": {"test": 0.8870230962114197, "leftout": 0.8661646113254221},
        "representation": "18-state pair grid; four miRNA PHACT-P1, four target PHACT, and target phyloP/phastCons channels",
        "architecture": "PairwisePhactCNN; 8-d pair embedding; 3 convolution blocks (128/64/32 filters; 6/3/3 kernels)",
    },
    {
        "slug": "phact_p1_agentomics_selected_fusion",
        "name": "PHACT-P1 Multimodal Fusion",
        "family": "agentomics",
        "run_dir": "outputs/param_1_agentomics_model_5",
        "expected": {"test": 0.8899038583452401, "leftout": 0.8621774605396150},
        "representation": "miRBind2 and reverse-complement pair grids, miRNA/target PHACT-P1, conservation, metadata, and frozen RiNALMo embeddings for one selected mature candidate",
        "architecture": "Selected-candidate multimodal fusion network with sequence, positional, metadata, and frozen foundation-model branches",
    },
    {
        "slug": "phact_p1_agentomics_multicandidate_fusion",
        "name": "PHACT-P1 Multi-Candidate Fusion",
        "family": "agentomics",
        "run_dir": "outputs/param_1_agentomics_model_4",
        "expected": {"test": 0.8899782667360706, "leftout": 0.8608448391338781},
        "representation": "Model-5 multimodal features plus RiNALMo layers 3/6/9/12 and up to three mature-miRNA PHACT candidates",
        "architecture": "Layer-mix residual fusion followed by attention-based multi-candidate ambiguity fusion",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Root containing outputs/ and agentomics_eval/ from the final runs.",
    )
    parser.add_argument(
        "--cnn-predictions",
        type=Path,
        required=True,
        help="Directory containing model{1,2,3}_{test,leftout}.npz exports.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_file(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern!r} in {folder}, found {len(matches)}")
    return matches[0]


def read_labels(run_root: Path, split: str) -> pd.DataFrame:
    path = run_root / "agentomics_eval" / split / "labels.csv"
    labels = pd.read_csv(path)
    if labels.columns.tolist() != ["id", "label"]:
        raise ValueError(f"Unexpected label columns in {path}: {labels.columns.tolist()}")
    if labels["id"].duplicated().any() or set(labels["label"].unique()) - {0, 1}:
        raise ValueError(f"Invalid labels in {path}")
    return labels


def calculate_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.float64)
    if labels.ndim != 1 or predictions.ndim != 1 or len(labels) != len(predictions):
        raise ValueError("Labels and predictions must be equally sized one-dimensional arrays")
    if not np.isfinite(predictions).all() or ((predictions < 0) | (predictions > 1)).any():
        raise ValueError("Predictions must be finite probabilities in [0, 1]")
    binary = (predictions >= 0.5).astype(np.int8)
    clipped = np.clip(predictions, 1e-15, 1 - 1e-15)
    return {
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "auprc": float(average_precision_score(labels, predictions)),
        "auroc": float(roc_auc_score(labels, predictions)),
        "accuracy_at_0p5": float(accuracy_score(labels, binary)),
        "precision_at_0p5": float(precision_score(labels, binary, zero_division=0)),
        "recall_at_0p5": float(recall_score(labels, binary, zero_division=0)),
        "brier": float(brier_score_loss(labels, predictions)),
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
    }


def write_prediction_csv(path: Path, ids: pd.Series, predictions: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"id": ids.astype(str), "prediction": predictions})
    if frame["id"].duplicated().any():
        raise ValueError(f"Duplicate prediction IDs for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10g")
    checked = pd.read_csv(path)
    if checked.columns.tolist() != ["id", "prediction"] or len(checked) != len(frame):
        raise ValueError(f"Failed prediction CSV round-trip validation for {path}")
    return checked["prediction"].to_numpy(dtype=np.float64)


def cnn_predictions(model: dict[str, Any], split: str, cnn_root: Path, labels: pd.DataFrame) -> np.ndarray:
    path = cnn_root / f"{model['prediction_prefix']}_{split}.npz"
    with np.load(path) as data:
        if "labels" not in data or "predictions" not in data:
            raise ValueError(f"Missing labels/predictions in {path}")
        stored_labels = np.asarray(data["labels"], dtype=np.int8)
        predictions = np.asarray(data["predictions"], dtype=np.float64)
    expected_labels = labels["label"].to_numpy(dtype=np.int8)
    if not np.array_equal(stored_labels, expected_labels):
        raise ValueError(f"CNN label order differs from miRBench IDs for {model['slug']} {split}")
    return predictions


def agentomics_predictions(model: dict[str, Any], split: str, run_root: Path, labels: pd.DataFrame) -> np.ndarray:
    path = run_root / model["run_dir"] / "predictions" / f"{split}_predictions.csv"
    raw = pd.read_csv(path)
    required = {"id", "probability_1"}
    if not required.issubset(raw.columns) or raw["id"].duplicated().any():
        raise ValueError(f"Invalid Agentomics predictions in {path}")
    ordered = labels[["id"]].merge(
        raw[["id", "probability_1"]], on="id", how="left", validate="one_to_one"
    )
    if ordered["probability_1"].isna().any() or len(raw) != len(ordered):
        raise ValueError(f"Agentomics prediction IDs do not match labels for {model['slug']} {split}")
    return ordered["probability_1"].to_numpy(dtype=np.float64)


def add_text_to_tar(archive: tarfile.TarFile, arcname: str, content: str) -> None:
    payload = content.encode("utf-8")
    info = tarfile.TarInfo(arcname)
    info.size = len(payload)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


def bundle_agentomics(source: Path, destination: Path, slug: str) -> None:
    required = [source / "model.pt", source / "rinalmo_model"]
    if slug.endswith("multicandidate_fusion"):
        required.extend([source / "preprocessor.pkl", source / "architecture.json"])
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz", compresslevel=6) as archive:
        for name in ("model.pt", "preprocessor.pkl", "architecture.json", "final_evaluation.json"):
            path = source / name
            if path.is_file():
                archive.add(path, arcname=f"{slug}/{name}", recursive=False)
        archive.add(source / "rinalmo_model", arcname=f"{slug}/rinalmo_model")
        add_text_to_tar(
            archive,
            f"{slug}/README.txt",
            "Keep model.pt and rinalmo_model/ together. For the multi-candidate model, "
            "also keep preprocessor.pkl and architecture.json in this directory.\n",
        )


def build_metadata(model: dict[str, Any], run_root: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    run_dir = run_root / model["run_dir"]
    metadata: dict[str, Any] = {
        "model": model["name"],
        "slug": model["slug"],
        "representation": model["representation"],
        "architecture": model["architecture"],
        "evaluation": metrics,
    }
    if model["family"] == "cnn":
        summary = json.loads(unique_file(run_dir, "architecture_summary_*.json").read_text())
        metadata["model_type"] = summary["model_type"]
        metadata["model_params"] = summary["model_params"]
        metadata["training"] = {
            key: summary[key]
            for key in (
                "seed",
                "batch_size",
                "num_epochs",
                "patience",
                "learning_rate",
                "cache_shuffle_mode",
                "focal_gamma",
                "training_target_shift_max",
                "initial_checkpoint_mode",
                "phact_channel_mode",
                "phact_interaction_mode",
                "phact_missingness_included",
            )
        }
    else:
        source_metrics = json.loads((run_dir / "metrics.json").read_text())
        metadata["training"] = source_metrics["training_config"]
        if "architecture.json" in {p.name for p in run_dir.iterdir()}:
            metadata["architecture_config"] = json.loads((run_dir / "architecture.json").read_text())
    return metadata


def prepare_output(path: Path, force: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists():
        if not force:
            raise FileExistsError(f"Output exists: {path}; pass --force to replace it")
        if path == Path(path.anchor):
            raise ValueError("Refusing to replace a filesystem root")
        shutil.rmtree(path)
    (path / "model_weights").mkdir(parents=True)
    (path / "predictions").mkdir()
    (path / "metadata").mkdir()
    return path


def main() -> None:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    cnn_root = args.cnn_predictions.expanduser().resolve()
    output = prepare_output(args.output, args.force)
    labels_by_split = {split: read_labels(run_root, split) for split in ("test", "leftout")}
    manifest_models: list[dict[str, Any]] = []

    for model in MODELS:
        slug = model["slug"]
        run_dir = run_root / model["run_dir"]
        if model["family"] == "cnn":
            checkpoint = unique_file(run_dir, "pairwise_phact_model_*.pt")
            weight_path = output / "model_weights" / f"{slug}.pt"
            shutil.copy2(checkpoint, weight_path)
        else:
            weight_path = output / "model_weights" / f"{slug}.tar.gz"
            bundle_agentomics(run_dir, weight_path, slug)

        split_metrics: dict[str, Any] = {}
        prediction_paths: dict[str, str] = {}
        for split in ("test", "leftout"):
            labels = labels_by_split[split]
            if model["family"] == "cnn":
                predictions = cnn_predictions(model, split, cnn_root, labels)
            else:
                predictions = agentomics_predictions(model, split, run_root, labels)
            prediction_path = output / "predictions" / slug / f"{split}.csv"
            checked_predictions = write_prediction_csv(prediction_path, labels["id"], predictions)
            split_metrics[split] = calculate_metrics(
                labels["label"].to_numpy(dtype=np.int8), checked_predictions
            )
            difference = abs(split_metrics[split]["auprc"] - model["expected"][split])
            if difference > 2e-6:
                raise ValueError(
                    f"AUPRC mismatch for {slug} {split}: "
                    f"{split_metrics[split]['auprc']} vs {model['expected'][split]}"
                )
            prediction_paths[split] = prediction_path.relative_to(output).as_posix()

        metrics_path = output / "predictions" / slug / "metrics.json"
        metrics_path.write_text(json.dumps(split_metrics, indent=2, sort_keys=True) + "\n")
        metadata_path = output / "metadata" / f"{slug}.json"
        metadata_path.write_text(
            json.dumps(build_metadata(model, run_root, split_metrics), indent=2, sort_keys=True) + "\n"
        )
        manifest_models.append(
            {
                "slug": slug,
                "name": model["name"],
                "family": model["family"],
                "representation": model["representation"],
                "architecture": model["architecture"],
                "weights": weight_path.relative_to(output).as_posix(),
                "predictions": prediction_paths,
                "metrics": metrics_path.relative_to(output).as_posix(),
                "metadata": metadata_path.relative_to(output).as_posix(),
                "auprc": {split: split_metrics[split]["auprc"] for split in split_metrics},
            }
        )

    ordered = sorted(manifest_models, key=lambda item: item["auprc"]["test"], reverse=True)
    leaderboard_path = output / "leaderboard_rows.tsv"
    pd.DataFrame(
        [
            {
                "model_name": item["name"],
                "author": "Dimos",
                "test_auprc": item["auprc"]["test"],
                "leftout_auprc": item["auprc"]["leftout"],
                "slug": item["slug"],
            }
            for item in ordered
        ]
    ).to_csv(leaderboard_path, sep="\t", index=False, float_format="%.12g")

    readme_lines = [
        f"# {RELEASE_NAME}",
        "",
        "This directory contains the five validated PHACT-P1 submissions for miRBench v7.",
        "Prediction CSVs contain exactly `id,prediction`, where `prediction` is the positive-class probability.",
        "Agentomics archives include the frozen RiNALMo encoder required for offline inference.",
        "",
        "| Model | Test AUPRC | Left-out AUPRC |",
        "| --- | ---: | ---: |",
    ]
    for item in ordered:
        readme_lines.append(
            f"| {item['name']} | {item['auprc']['test']:.6f} | {item['auprc']['leftout']:.6f} |"
        )
    readme_lines.extend(
        [
            "",
            f"Code release: `{CODE_TAG}`.",
            "Verify every file with `sha256sum -c SHA256SUMS`.",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(readme_lines))

    release_files = []
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        release_files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "release": RELEASE_NAME,
        "release_date": RELEASE_DATE,
        "code_tag": CODE_TAG,
        "prediction_schema": ["id", "prediction"],
        "test_rows": int(len(labels_by_split["test"])),
        "leftout_rows": int(len(labels_by_split["leftout"])),
        "models": ordered,
        "files": release_files,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    checksum_paths = sorted(p for p in output.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    checksum_text = "".join(
        f"{sha256(path)}  {path.relative_to(output).as_posix()}\n" for path in checksum_paths
    )
    (output / "SHA256SUMS").write_text(checksum_text)
    print(json.dumps({"output": str(output), "models": len(ordered), "files": len(checksum_paths)}, indent=2))


if __name__ == "__main__":
    main()
