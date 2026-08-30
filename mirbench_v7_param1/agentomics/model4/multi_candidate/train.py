#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from baseline_model import (
    ArchitectureConfig,
    LayerMixConfig,
    PHACTGatedLayerMixFusionNet,
)
from candidate_representation import transform_candidate_representation
from inference import (
    build_foundation_cache_with_fallback,
    load_foundation_cache,
    make_batch,
    validate_representation,
)
from multi_candidate_model import (
    CandidateAmbiguityConfig,
    MultiCandidatePHACTFusionNet,
    architecture_payload,
    load_layer_mix_baseline,
)


PREFERRED_BATCH_SIZES = (768, 512, 384)
WEIGHT_DECAY = 1.0e-4
PATIENCE = 2
MAX_EPOCHS = 4
CANDIDATES = (
    {"name": "candidate_a", "seed": 20260802, "dropout": 0.10, "learning_rate": 3.0e-4},
    {"name": "candidate_b", "seed": 20260803, "dropout": 0.20, "learning_rate": 1.5e-4},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multi-candidate PHACT residual")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--initial-artifacts", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False


def atomic_json(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def labels_aligned(split_root: Path, ids: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(split_root / "labels.csv")
    label_column = "numeric_label" if "numeric_label" in frame else "label"
    if label_column not in frame:
        raise ValueError(f"No binary label column in {split_root / 'labels.csv'}")
    if frame["id"].duplicated().any():
        raise ValueError("Duplicate label ids")
    mapping = dict(zip(frame["id"].astype(str), frame[label_column]))
    missing = [sample_id for sample_id in ids.astype(str) if sample_id not in mapping]
    if missing:
        raise ValueError(f"Missing {len(missing)} labels")
    labels = np.asarray([mapping[sample_id] for sample_id in ids.astype(str)], dtype=np.float32)
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("Labels are not binary")
    return labels


def cache_name(split_root: Path) -> str:
    digest = hashlib.sha256(str(split_root.resolve()).encode()).hexdigest()[:12]
    return f"{split_root.name}_{digest}_candidate_layers.npz"


def ensure_cache(
    split_root: Path,
    cache_root: Path,
    rinalmo_dir: Path,
    device: torch.device,
) -> Tuple[Path, dict]:
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / cache_name(split_root)
    metadata_path = path.with_suffix(path.suffix + ".json")
    if path.is_file() and metadata_path.is_file():
        return path, json.loads(metadata_path.read_text())
    built, metadata = build_foundation_cache_with_fallback(
        split_root / "input", rinalmo_dir, path, device
    )
    atomic_json(metadata, metadata_path)
    return built, metadata


def iter_indices(
    n: int, batch_size: int, shuffle: bool, seed: int
) -> Iterable[np.ndarray]:
    order = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size].astype(np.int64, copy=False)


def metric_values(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1.0e-7, 1.0 - 1.0e-7)
    labels = np.asarray(labels, dtype=np.int64)
    hard = (probabilities >= 0.5).astype(np.int64)
    return {
        "auprc": float(average_precision_score(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, hard)),
        "precision": float(precision_score(labels, hard, zero_division=0)),
        "recall": float(recall_score(labels, hard, zero_division=0)),
        "f1": float(f1_score(labels, hard, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, hard)),
    }


def to_labels(labels: np.ndarray, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(labels[indices])).to(device=device)


@torch.inference_mode()
def evaluate(
    model: MultiCandidatePHACTFusionNet,
    arrays: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Tuple[dict, np.ndarray]:
    model.eval()
    probabilities = np.empty(len(labels), dtype=np.float32)
    pair_dtype = amp_dtype if device.type == "cuda" else torch.float32
    for indices in iter_indices(len(labels), batch_size, False, 0):
        kwargs = make_batch(arrays, cache, indices, device, pair_dtype)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(**kwargs).float()
        probabilities[indices] = torch.sigmoid(logits).cpu().numpy()
    return metric_values(labels, probabilities), probabilities


def train_epoch(
    model: MultiCandidatePHACTFusionNet,
    arrays: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
    labels: np.ndarray,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    seed: int,
    epoch: int,
) -> float:
    model.train(True)
    pair_dtype = amp_dtype if device.type == "cuda" else torch.float32
    total = 0.0
    seen = 0
    batches = math.ceil(len(labels) / batch_size)
    for batch_number, indices in enumerate(
        iter_indices(len(labels), batch_size, True, seed + epoch), start=1
    ):
        optimizer.zero_grad(set_to_none=True)
        kwargs = make_batch(arrays, cache, indices, device, pair_dtype)
        target = to_labels(labels, indices, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(**kwargs)
            loss = F.binary_cross_entropy_with_logits(logits.float(), target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.candidate_parameters(), 5.0)
        optimizer.step()
        total += float(loss.item()) * len(indices)
        seen += len(indices)
        if batch_number % 250 == 0 or batch_number == batches:
            print(
                json.dumps(
                    {
                        "phase": "train",
                        "epoch": epoch,
                        "batch": batch_number,
                        "batches": batches,
                        "mean_loss": total / seen,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return total / seen


def build_model(
    baseline: Mapping[str, object], candidate: Mapping[str, object], device: torch.device
) -> Tuple[MultiCandidatePHACTFusionNet, dict]:
    cfg = ArchitectureConfig(**dict(baseline["architecture_config"]))
    layer_cfg = LayerMixConfig(**dict(baseline["layer_mix_config"]))
    candidate_cfg = CandidateAmbiguityConfig(dropout=float(candidate["dropout"]))
    model = MultiCandidatePHACTFusionNet(cfg, layer_cfg, candidate_cfg)
    compatibility = load_layer_mix_baseline(model, baseline)
    model.configure_candidate_training()
    return model.to(device), compatibility


def clone_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def better(metrics: Mapping[str, float], epochs: int, best: Mapping[str, object] | None) -> bool:
    if best is None:
        return True
    return (
        float(metrics["auprc"]),
        -float(metrics["log_loss"]),
        -int(epochs),
    ) > (
        float(best["metrics"]["auprc"]),
        -float(best["metrics"]["log_loss"]),
        -int(best["optimization_epochs"]),
    )


def training_smoke(
    baseline: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
    labels: np.ndarray,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict:
    candidate = CANDIDATES[0]
    model, compatibility = build_model(baseline, candidate, device)
    cfg = ArchitectureConfig(**dict(baseline["architecture_config"]))
    layer_cfg = LayerMixConfig(**dict(baseline["layer_mix_config"]))
    reference = PHACTGatedLayerMixFusionNet(cfg, layer_cfg)
    reference.load_state_dict(baseline["state_dict"], strict=True)
    reference.to(device).eval()
    extra_indices = np.flatnonzero(
        arrays["candidate_profile_mask"][:, 1:].sum(axis=1) > 0
    )
    other_indices = np.flatnonzero(
        arrays["candidate_profile_mask"][:, 1:].sum(axis=1) == 0
    )
    indices = np.concatenate([extra_indices, other_indices])[: min(32, len(labels))].astype(
        np.int64, copy=False
    )
    pair_dtype = amp_dtype if device.type == "cuda" else torch.float32
    kwargs = make_batch(arrays, cache, indices, device, pair_dtype)
    baseline_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"candidate_phact", "candidate_profile_mask"}
    }
    model.eval()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
    ):
        initial = model(**kwargs).float()
        expected = reference(**baseline_kwargs).float()
    epoch0_parity = float(torch.max(torch.abs(initial - expected)).item())

    optimizer = torch.optim.AdamW(
        model.candidate_parameters(), lr=float(candidate["learning_rate"]), weight_decay=WEIGHT_DECAY
    )
    model.train(True)
    target = to_labels(labels, indices, device)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
        loss = F.binary_cross_entropy_with_logits(model(**kwargs).float(), target)
    loss.backward()
    adapter_gradient = model.candidate_residual_adapter.weight.grad
    first_adapter_gradient = bool(
        adapter_gradient is not None and adapter_gradient.abs().sum().item() > 0
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
        loss2 = F.binary_cross_entropy_with_logits(model(**kwargs).float(), target)
    loss2.backward()
    upstream_gradient = bool(
        model.candidate_query.weight.grad is not None
        and model.candidate_query.weight.grad.abs().sum().item() > 0
    )
    report = {
        "rows": len(indices),
        "epoch0_max_abs_logit_difference": epoch0_parity,
        "finite_loss": bool(torch.isfinite(loss).item() and torch.isfinite(loss2).item()),
        "zero_adapter_first_step_gradient": first_adapter_gradient,
        "upstream_gradient_after_adapter_update": upstream_gradient,
        "compatibility": compatibility,
        "candidate_phact_shape": list(arrays["candidate_phact"].shape),
        "samples_with_extra_profile": int(
            (arrays["candidate_profile_mask"][:, 1:].sum(axis=1) > 0).sum()
        ),
    }
    if epoch0_parity != 0.0 or not all(
        [report["finite_loss"], first_adapter_gradient, upstream_gradient]
    ):
        raise RuntimeError(f"Multi-candidate smoke failed: {report}")
    del model, reference, optimizer, kwargs, baseline_kwargs
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def main() -> None:
    args = parse_args()
    train_root = args.train_data.resolve()
    validation_root = args.validation_data.resolve()
    initial = args.initial_artifacts.resolve()
    artifacts = args.artifacts_dir.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    if any(artifacts.iterdir()):
        raise RuntimeError(f"Artifacts directory must be empty: {artifacts}")
    for required in (initial / "model.pt", initial / "preprocessor.pkl", initial / "rinalmo_model"):
        if not required.exists():
            raise FileNotFoundError(required)

    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
        if device.type == "cuda"
        else torch.float32
    )
    baseline = torch.load(initial / "model.pt", map_location="cpu", weights_only=False)
    with (initial / "preprocessor.pkl").open("rb") as handle:
        preprocessor = pickle.load(handle)

    cache_root = artifacts.parent / "multi_candidate_foundation_cache"
    train_cache_path, train_cache_metadata = ensure_cache(
        train_root, cache_root, initial / "rinalmo_model", device
    )
    validation_cache_path, validation_cache_metadata = ensure_cache(
        validation_root, cache_root, initial / "rinalmo_model", device
    )
    train_arrays = transform_candidate_representation(train_root, preprocessor)
    validation_arrays = transform_candidate_representation(validation_root, preprocessor)
    train_ids = validate_representation(train_arrays)
    validation_ids = validate_representation(validation_arrays)
    train_labels = labels_aligned(train_root, train_ids)
    validation_labels = labels_aligned(validation_root, validation_ids)
    train_cache = load_foundation_cache(train_cache_path, train_ids)
    validation_cache = load_foundation_cache(validation_cache_path, validation_ids)

    full_run = len(train_labels) > 1000
    candidate_specs: Sequence[Mapping[str, object]] = CANDIDATES if full_run else CANDIDATES[:1]
    max_epochs = MAX_EPOCHS if full_run else 1
    smoke = training_smoke(
        baseline, train_arrays, train_cache, train_labels, device, amp_dtype
    )

    selected_batch_size = min(PREFERRED_BATCH_SIZES[-1], len(train_labels))
    for size in PREFERRED_BATCH_SIZES:
        try:
            probe, _ = build_model(baseline, candidate_specs[0], device)
            probe_indices = np.arange(min(size, len(train_labels)), dtype=np.int64)
            kwargs = make_batch(
                train_arrays,
                train_cache,
                probe_indices,
                device,
                amp_dtype if device.type == "cuda" else torch.float32,
            )
            target = to_labels(train_labels, probe_indices, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                probe_loss = F.binary_cross_entropy_with_logits(probe(**kwargs).float(), target)
            probe_loss.backward()
            selected_batch_size = min(size, len(train_labels)) if not full_run else size
            del probe, kwargs, target
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

    history: list[dict] = []
    best: dict | None = None
    for candidate in candidate_specs:
        seed_everything(int(candidate["seed"]))
        model, compatibility = build_model(baseline, candidate, device)
        metrics, probabilities = evaluate(
            model,
            validation_arrays,
            validation_cache,
            validation_labels,
            selected_batch_size,
            device,
            amp_dtype,
        )
        history.append(
            {
                "candidate": candidate["name"],
                "seed": candidate["seed"],
                "epoch": 0,
                "optimization_epochs": 0,
                "train_loss": None,
                **metrics,
            }
        )
        if better(metrics, 0, best):
            best = {
                "candidate": candidate,
                "epoch": 0,
                "optimization_epochs": 0,
                "metrics": metrics,
                "probabilities": probabilities.copy(),
                "state_dict": clone_state(model),
                "compatibility": compatibility,
            }

        optimizer = torch.optim.AdamW(
            model.candidate_parameters(),
            lr=float(candidate["learning_rate"]),
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1
        )
        candidate_best = float(metrics["auprc"])
        no_improve = 0
        for epoch in range(1, max_epochs + 1):
            train_loss_value = train_epoch(
                model,
                train_arrays,
                train_cache,
                train_labels,
                optimizer,
                selected_batch_size,
                device,
                amp_dtype,
                int(candidate["seed"]),
                epoch,
            )
            metrics, probabilities = evaluate(
                model,
                validation_arrays,
                validation_cache,
                validation_labels,
                selected_batch_size,
                device,
                amp_dtype,
            )
            history.append(
                {
                    "candidate": candidate["name"],
                    "seed": candidate["seed"],
                    "epoch": epoch,
                    "optimization_epochs": epoch,
                    "train_loss": train_loss_value,
                    **metrics,
                }
            )
            print(json.dumps(history[-1], sort_keys=True), flush=True)
            if float(metrics["auprc"]) > candidate_best + 1.0e-12:
                candidate_best = float(metrics["auprc"])
                no_improve = 0
            else:
                no_improve += 1
            if better(metrics, epoch, best):
                best = {
                    "candidate": candidate,
                    "epoch": epoch,
                    "optimization_epochs": epoch,
                    "metrics": metrics,
                    "probabilities": probabilities.copy(),
                    "state_dict": clone_state(model),
                    "compatibility": compatibility,
                }
            scheduler.step(metrics["auprc"])
            atomic_json(
                {"history": history, "best": None if best is None else best["metrics"]},
                artifacts / "metrics_in_progress.json",
            )
            if no_improve >= PATIENCE:
                break
        del model, optimizer, scheduler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if best is None:
        raise RuntimeError("No eligible checkpoint")
    candidate_cfg = CandidateAmbiguityConfig(dropout=float(best["candidate"]["dropout"]))
    cfg = ArchitectureConfig(**dict(baseline["architecture_config"]))
    layer_cfg = LayerMixConfig(**dict(baseline["layer_mix_config"]))
    checkpoint = {
        "format_version": 1,
        "model_name": "MultiCandidatePHACTFusionNet",
        "architecture_config": cfg.__dict__,
        "layer_mix_config": layer_cfg.__dict__,
        "candidate_ambiguity_config": candidate_cfg.__dict__,
        "state_dict": best["state_dict"],
        "selected_candidate": best["candidate"]["name"],
        "best_candidate_epoch": best["epoch"],
        "optimization_epochs": best["optimization_epochs"],
        "seed": best["candidate"]["seed"],
        "score_representation": "consensus_tree_param_1_A_C_G_T_plus_finite_mask",
    }
    atomic_torch_save(checkpoint, artifacts / "model.pt")
    with (artifacts / "preprocessor.pkl").open("wb") as handle:
        pickle.dump(preprocessor, handle, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.copytree(initial / "rinalmo_model", artifacts / "rinalmo_model")
    atomic_json(architecture_payload(cfg, layer_cfg, candidate_cfg), artifacts / "architecture.json")
    summary = {
        "best": {
            "candidate": best["candidate"]["name"],
            "epoch": best["epoch"],
            **best["metrics"],
        },
        "history": history,
        "training_config": {
            "candidates": list(CANDIDATES),
            "maximum_epochs": max_epochs,
            "patience": PATIENCE,
            "batch_size": selected_batch_size,
            "weight_decay": WEIGHT_DECAY,
            "loss": "unweighted_BCEWithLogitsLoss",
            "device": str(device),
            "amp_dtype": str(amp_dtype),
        },
        "data": {
            "train_rows": len(train_labels),
            "validation_rows": len(validation_labels),
            "train_samples_with_extra_profile": int(
                (train_arrays["candidate_profile_mask"][:, 1:].sum(axis=1) > 0).sum()
            ),
            "validation_samples_with_extra_profile": int(
                (validation_arrays["candidate_profile_mask"][:, 1:].sum(axis=1) > 0).sum()
            ),
        },
        "smoke": smoke,
        "foundation_cache_metadata": {
            "train": train_cache_metadata,
            "validation": validation_cache_metadata,
        },
        "initial_layer_mix_model_sha256": file_sha256(initial / "model.pt"),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(summary, artifacts / "metrics.json")
    atomic_json(smoke, artifacts / "smoke_test.json")
    with (artifacts / "validation_metrics_by_epoch.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (artifacts / "validation_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label", "probability_1"])
        for sample_id, label, probability in zip(
            validation_ids, validation_labels, best["probabilities"]
        ):
            writer.writerow([sample_id, int(label), f"{float(probability):.9g}"])
    (artifacts / "metrics_in_progress.json").unlink(missing_ok=True)
    print(json.dumps({"status": "ok", **summary["best"]}, sort_keys=True))


if __name__ == "__main__":
    main()
