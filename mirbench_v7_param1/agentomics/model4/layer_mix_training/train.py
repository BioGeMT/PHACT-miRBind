from __future__ import annotations

import argparse
import copy
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
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from helpers.training_reporter import TrainingReporter
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

from data_representation import build_mirbind2_onehot, build_rc_pairwise_grid, load_labels, transform_representation
from foundation_layer_cache import build_cache
from layer_mix_model import (
    ArchitectureConfig,
    LayerMixConfig,
    PHACTGatedLayerMixFusionNet,
    architecture_payload,
    load_iteration30_classifier_weights,
)

ROOT = Path(__file__).resolve().parent
INITIAL_CHECKPOINT = ROOT / "baseline_model.pt"
RINALMO_SOURCE = ROOT / "rinalmo_model"
CACHE_ROOT = ROOT / "foundation_cache"
WEIGHT_DECAY = 1e-4
PATIENCE = 2
PREFERRED_BATCH_SIZES = (768, 512, 384)
CACHE_BATCH_SIZES = (1024, 768, 512, 256)

CANDIDATES = (
    {
        "name": "layers_a",
        "seed": 20260729,
        "warmup_lr": 3e-4,
        "new_lr": 2e-4,
        "adapter_lr": 1e-5,
        "metadata_lr": 3e-6,
    },
    {
        "name": "layers_b",
        "seed": 20260730,
        "warmup_lr": 1.5e-4,
        "new_lr": 1e-4,
        "adapter_lr": 5e-6,
        "metadata_lr": 1e-6,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the standalone PHACT-gated RiNALMo layer-mix classifier")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--initial-artifacts", required=True)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str))
    os.replace(temporary, path)


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def aligned_labels(split_dir: Path, ids: np.ndarray) -> np.ndarray:
    frame = load_labels(split_dir)
    mapping = dict(zip(frame["id"].astype(str), frame["label"].astype(np.float32)))
    missing = [str(sample_id) for sample_id in ids if str(sample_id) not in mapping]
    if missing:
        raise KeyError(f"Missing labels for {len(missing)} IDs")
    result = np.asarray([mapping[str(sample_id)] for sample_id in ids], dtype=np.float32)
    if not np.isin(result, [0.0, 1.0]).all():
        raise ValueError("Labels must be binary")
    return result


def split_cache_name(split_dir: Path) -> str:
    resolved = str(split_dir.resolve())
    token = hashlib.sha256(resolved.encode()).hexdigest()[:12]
    return f"{split_dir.name}_{token}_layers.npz"


def ensure_foundation_cache(split_dir: Path, device: torch.device) -> Tuple[Path, dict]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    output = CACHE_ROOT / split_cache_name(split_dir)
    metadata_path = output.with_suffix(output.suffix + ".json")
    if output.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        return output, metadata
    last_error: Exception | None = None
    for batch_size in CACHE_BATCH_SIZES:
        try:
            if output.exists():
                output.unlink()
            temporary = output.with_name(output.name + ".tmp.npz")
            if temporary.exists():
                temporary.unlink()
            metadata = build_cache(split_dir, RINALMO_SOURCE, output, batch_size, str(device))
            metadata["extraction_batch_size"] = batch_size
            atomic_json(metadata, metadata_path)
            return output, metadata
        except torch.cuda.OutOfMemoryError as exc:
            last_error = exc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError("All foundation-cache batch sizes exhausted") from last_error


def load_cache_for_training(path: Path, expected_ids: np.ndarray) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        needed = (
            "ids", "layer_indices", "mirna_pooled", "target_pooled",
            "mirna_row_to_unique", "target_row_to_unique", "mirna_valid", "target_valid",
        )
        missing = [key for key in needed if key not in data.files]
        if missing:
            raise ValueError(f"Foundation cache missing {missing}")
        cache = {key: data[key] for key in needed}
    if tuple(cache["layer_indices"].tolist()) != (3, 6, 9, 12):
        raise ValueError("Unexpected hidden-state order")
    if not np.array_equal(cache["ids"].astype(str), expected_ids.astype(str)):
        raise ValueError("Foundation cache ID order mismatch")
    if cache["mirna_pooled"].shape[1:] != (4, 480) or cache["target_pooled"].shape[1:] != (4, 480):
        raise ValueError("Foundation cache shape mismatch")
    if not np.all(cache["mirna_valid"] == 1) or not np.all(cache["target_valid"] == 1):
        raise ValueError("Invalid foundation pool")
    return cache


def to_device(array: np.ndarray, indices: np.ndarray, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array[indices]))
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device=device, non_blocking=True)


def make_layer_pairs(cache: Mapping[str, np.ndarray], indices: np.ndarray, device: torch.device) -> torch.Tensor:
    mirna_unique = cache["mirna_row_to_unique"][indices].astype(np.int64, copy=False)
    target_unique = cache["target_row_to_unique"][indices].astype(np.int64, copy=False)
    mirna = torch.from_numpy(np.ascontiguousarray(cache["mirna_pooled"][mirna_unique])).to(device, non_blocking=True)
    target = torch.from_numpy(np.ascontiguousarray(cache["target_pooled"][target_unique])).to(device, non_blocking=True)
    return torch.cat((mirna, target, torch.abs(mirna - target), mirna * target), dim=2)


def make_batch(
    arrays: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
    indices: np.ndarray,
    labels: np.ndarray | None,
    device: torch.device,
    pair_dtype: torch.dtype,
) -> Tuple[dict, torch.Tensor | None]:
    mirna_codes = to_device(arrays["mirna_codes"], indices, device, torch.long)
    target_codes = to_device(arrays["target_codes"], indices, device, torch.long)
    kwargs = {
        "pairwise_onehot": build_mirbind2_onehot(mirna_codes, target_codes, pair_dtype),
        "rc_pairwise_grid": build_rc_pairwise_grid(mirna_codes, target_codes, pair_dtype),
        "aux_numeric": to_device(arrays["aux_numeric"], indices, device, torch.float32),
        "target_tensor": to_device(arrays["target_tensor"], indices, device, torch.float32),
        "mirna_phact_tensor": to_device(arrays["mirna_phact_tensor"], indices, device, torch.float32),
        "feature_idx": to_device(arrays["feature_index"], indices, device, torch.long),
        "dominant_region_idx": to_device(arrays["dominant_region_index"], indices, device, torch.long),
        "layer_pair_vectors": make_layer_pairs(cache, indices, device),
    }
    y = None if labels is None else to_device(labels, indices, device, torch.float32)
    return kwargs, y


def iter_indices(n: int, batch_size: int, shuffle: bool, rng: np.random.Generator) -> Iterable[np.ndarray]:
    order = rng.permutation(n) if shuffle else np.arange(n, dtype=np.int64)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def metrics_from_predictions(y: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    y = np.asarray(y, dtype=np.int64)
    hard = (probability >= 0.5).astype(np.int64)
    return {
        "auprc": float(average_precision_score(y, probability)),
        "auroc": float(roc_auc_score(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probability)),
        "accuracy": float(accuracy_score(y, hard)),
        "precision": float(precision_score(y, hard, zero_division=0)),
        "recall": float(recall_score(y, hard, zero_division=0)),
        "f1": float(f1_score(y, hard, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, hard)),
    }


@torch.inference_mode()
def evaluate(
    model: PHACTGatedLayerMixFusionNet,
    arrays: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Tuple[dict, np.ndarray]:
    model.eval()
    probabilities = np.empty(len(labels), dtype=np.float32)
    total_loss = 0.0
    pair_dtype = amp_dtype if device.type == "cuda" else torch.float32
    for indices in iter_indices(len(labels), batch_size, False, np.random.default_rng(0)):
        kwargs, y = make_batch(arrays, cache, indices, labels, device, pair_dtype)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(**kwargs).float()
            loss = F.binary_cross_entropy_with_logits(logits, y, reduction="sum")
        probabilities[indices] = torch.sigmoid(logits).cpu().numpy()
        total_loss += float(loss.item())
    metrics = metrics_from_predictions(labels, probabilities)
    metrics["mean_bce"] = total_loss / len(labels)
    return metrics, probabilities


def train_one_epoch(
    model: PHACTGatedLayerMixFusionNet,
    arrays: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
    labels: np.ndarray,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    seed: int,
    local_epoch: int,
    reporter_epoch: int,
    reporter: TrainingReporter,
) -> float:
    model.train(True)
    pair_dtype = amp_dtype if device.type == "cuda" else torch.float32
    total = 0.0
    seen = 0
    batch_count = math.ceil(len(labels) / batch_size)
    for batch_number, indices in enumerate(
        iter_indices(len(labels), batch_size, True, np.random.default_rng(seed + local_epoch)), start=1
    ):
        optimizer.zero_grad(set_to_none=True)
        kwargs, y = make_batch(arrays, cache, indices, labels, device, pair_dtype)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(**kwargs)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 5.0)
        optimizer.step()
        amount = len(indices)
        total += float(loss.item()) * amount
        seen += amount
        if batch_number % 250 == 0 or batch_number == batch_count:
            reporter.report_batch(epoch=reporter_epoch, batch=batch_number, train_loss=total / seen)
    return total / seen


def clone_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def better_checkpoint(metrics: Mapping[str, float], epochs: int, incumbent: Mapping[str, object] | None) -> bool:
    if incumbent is None:
        return True
    current_key = (float(metrics["auprc"]), -float(metrics["log_loss"]), -int(epochs))
    incumbent_key = (
        float(incumbent["metrics"]["auprc"]),
        -float(incumbent["metrics"]["log_loss"]),
        -int(incumbent["optimization_epochs"]),
    )
    return current_key > incumbent_key


def build_model(initial_checkpoint: dict, device: torch.device) -> Tuple[PHACTGatedLayerMixFusionNet, dict]:
    cfg = ArchitectureConfig(**initial_checkpoint["architecture_config"])
    layer_cfg = LayerMixConfig()
    model = PHACTGatedLayerMixFusionNet(cfg, layer_cfg)
    compatibility = load_iteration30_classifier_weights(model, initial_checkpoint)
    return model.to(device), compatibility


def optimizer_for_stage(model: PHACTGatedLayerMixFusionNet, candidate: Mapping[str, object], stage: str):
    model.configure_for_stage(stage)
    groups = model.parameter_groups()
    if stage == "warmup":
        if groups["adapter_and_head"] or groups["phact_and_metadata"]:
            raise RuntimeError("Base tensors unexpectedly trainable during warm-up")
        params = [{"params": groups["new_branch"], "lr": float(candidate["warmup_lr"])}]
    else:
        params = [
            {"params": groups["new_branch"], "lr": float(candidate["new_lr"])},
            {"params": groups["adapter_and_head"], "lr": float(candidate["adapter_lr"])},
            {"params": groups["phact_and_metadata"], "lr": float(candidate["metadata_lr"])},
        ]
    return torch.optim.AdamW(params, weight_decay=WEIGHT_DECAY)


def training_smoke(
    initial_checkpoint: dict,
    arrays: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
    labels: np.ndarray,
    device: torch.device,
    amp_dtype: torch.dtype,
    batch_size: int,
) -> dict:
    seed_everything(20260729)
    model, compatibility = build_model(initial_checkpoint, device)
    model.configure_for_stage("warmup")
    optimizer = optimizer_for_stage(model, CANDIDATES[0], "warmup")
    indices = np.arange(min(32, len(labels)), dtype=np.int64)
    before = clone_state(model)
    model.train(True)
    kwargs, y = make_batch(arrays, cache, indices, labels, device, amp_dtype if device.type == "cuda" else torch.float32)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
        logits = model(**kwargs)
        loss = F.binary_cross_entropy_with_logits(logits.float(), y)
    loss.backward()
    adapter_gradient = model.layer_residual_adapter.weight.grad
    report = {
        "rows": len(indices),
        "finite_logits": bool(torch.isfinite(logits).all().item()),
        "finite_loss": bool(torch.isfinite(loss).item()),
        "nonzero_zero_adapter_gradient": bool(adapter_gradient is not None and adapter_gradient.abs().sum().item() > 0),
        "frozen_gradient_count": int(sum(p.grad is not None for p in model.parameters() if not p.requires_grad)),
        "iteration30_compatibility": compatibility,
        "device": str(device),
        "amp_dtype": str(amp_dtype),
        "batch_size": batch_size,
    }
    optimizer.step()
    updated_adapter = model.layer_residual_adapter.weight.detach().abs().sum().item() > 0
    optimizer.zero_grad(set_to_none=True)
    kwargs2, y2 = make_batch(arrays, cache, indices, labels, device, amp_dtype if device.type == "cuda" else torch.float32)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
        loss2 = F.binary_cross_entropy_with_logits(model(**kwargs2).float(), y2)
    loss2.backward()
    report["adapter_updated"] = bool(updated_adapter)
    report["upstream_projector_gradient_after_update"] = bool(
        model.layer_projector[0].weight.grad is not None and model.layer_projector[0].weight.grad.abs().sum().item() > 0
    )
    report["gate_gradient_after_update"] = bool(
        model.layer_gate.weight.grad is not None and model.layer_gate.weight.grad.abs().sum().item() > 0
    )
    report["frozen_unchanged"] = all(
        torch.equal(value, model.state_dict()[name].detach().cpu())
        for name, value in before.items()
        if name.startswith(("seq_encoder.", "rc_main_branch.", "rc_seed_branch."))
    )
    required = (
        report["finite_logits"], report["finite_loss"], report["nonzero_zero_adapter_gradient"],
        report["frozen_gradient_count"] == 0, report["adapter_updated"],
        report["upstream_projector_gradient_after_update"], report["gate_gradient_after_update"],
        report["frozen_unchanged"],
    )
    if not all(required):
        raise RuntimeError(f"Training smoke failed: {report}")
    del model, optimizer, kwargs, kwargs2
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def save_artifacts(
    artifacts: Path,
    initial_checkpoint: dict,
    selected: Mapping[str, object],
    preprocessor: dict,
    history: dict,
    smoke: dict,
    cache_metadata: dict,
    val_ids: np.ndarray,
    val_labels: np.ndarray,
) -> None:
    cfg = ArchitectureConfig(**initial_checkpoint["architecture_config"])
    layer_cfg = LayerMixConfig()
    checkpoint = {
        "format_version": 1,
        "architecture_config": cfg.__dict__,
        "layer_mix_config": layer_cfg.__dict__,
        "state_dict": selected["state_dict"],
        "selected_candidate": selected["candidate"],
        "best_candidate_epoch": int(selected["candidate_epoch"]),
        "optimization_epochs": int(selected["optimization_epochs"]),
        "seed": int(selected["seed"]),
        "representation_module_version": "consensus_param1_phact_gated_rinalmo_layer_mix",
        "rinalmo_subdir": "rinalmo_model",
    }
    atomic_torch_save(checkpoint, artifacts / "model.pt")
    with (artifacts / "preprocessor.pkl").open("wb") as handle:
        pickle.dump(preprocessor, handle, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.copytree(RINALMO_SOURCE, artifacts / "rinalmo_model")
    atomic_json(
        {
            "format_version": 1,
            "model_name": "PHACTGatedLayerMixFusionNet",
            "score_representation": "consensus_tree_param_1_A_C_G_T_plus_finite_mask",
            **architecture_payload(cfg, layer_cfg),
        },
        artifacts / "architecture.json",
    )
    atomic_json(
        {
            "format_version": 1,
            "mirna_phact_shape": [cfg.mirna_phact_input_channels, 28],
            "mirna_phact_score_channels": ["phact_param_1_A", "phact_param_1_C", "phact_param_1_G", "phact_param_1_T"],
            "mirna_phact_last_channel": "finite_position_mask",
            "foundation_hidden_state_indices": [3, 6, 9, 12],
            "preprocessing_fit": "initial_model5_training_split_only",
        },
        artifacts / "representation_contract.json",
    )
    layer_metadata = {
        "selected_hidden_state_indices": [3, 6, 9, 12],
        "selected_state_meaning": ["after_block_3", "after_block_6", "after_block_9", "after_block_12_final_layernorm"],
        "hidden_size": 480,
        "pair_vector_order": ["mirna", "target", "absolute_difference", "elementwise_product"],
        "sequence_orientation": "supplied_original_target",
        "pooling": "mean_biological_tokens_only",
        "pooler_used": False,
        "lm_or_task_head_used": False,
        "preprocessor_sha256": file_sha256(artifacts / "preprocessor.pkl"),
        "rinalmo_config_sha256": file_sha256(artifacts / "rinalmo_model" / "config.json"),
        "training_cache_metadata": cache_metadata,
    }
    atomic_json(layer_metadata, artifacts / "layer_feature_metadata.json")
    atomic_json(history, artifacts / "metrics.json")
    atomic_json(smoke, artifacts / "smoke_test.json")
    keys = [
        "candidate", "seed", "epoch", "optimization_epochs", "stage", "train_loss", "auprc", "auroc",
        "log_loss", "brier", "accuracy", "precision", "recall", "f1", "mcc", "mean_bce",
    ]
    with (artifacts / "validation_metrics_by_epoch.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history["epochs"])
    probabilities = np.asarray(selected["probabilities"], dtype=np.float32)
    with (artifacts / "validation_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label", "probability_1"])
        for sample_id, label, probability in zip(val_ids, val_labels, probabilities):
            writer.writerow([str(sample_id), int(label), f"{float(probability):.9g}"])


def main() -> None:
    global INITIAL_CHECKPOINT, RINALMO_SOURCE, CACHE_ROOT
    args = parse_args()
    train_dir = Path(args.train_data).resolve()
    validation_dir = Path(args.validation_data).resolve()
    artifacts = Path(args.artifacts_dir).resolve()
    initial_artifacts = Path(args.initial_artifacts).resolve()
    INITIAL_CHECKPOINT = initial_artifacts / "model.pt"
    RINALMO_SOURCE = initial_artifacts / "rinalmo_model"
    CACHE_ROOT = artifacts.parent / "layer_mix_foundation_cache"
    artifacts.mkdir(parents=True, exist_ok=True)
    if any(artifacts.iterdir()):
        raise RuntimeError(f"Artifacts directory must be empty: {artifacts}")
    for required in (INITIAL_CHECKPOINT,):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not (RINALMO_SOURCE / "model.safetensors").is_file():
        raise FileNotFoundError("Artifact-local RiNALMo source is absent")

    reporter = TrainingReporter()
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (
        torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16 if device.type == "cuda"
        else torch.float32
    )
    initial_checkpoint = torch.load(INITIAL_CHECKPOINT, map_location="cpu", weights_only=False)
    preprocessor = initial_checkpoint.get("preprocessor")
    if not isinstance(preprocessor, dict):
        raise ValueError("Initial Model 5 checkpoint does not contain its fitted preprocessor")

    # Build label-free foundation caches and transform with the frozen split-0 preprocessor.
    train_cache_path, train_cache_metadata = ensure_foundation_cache(train_dir, device)
    validation_cache_path, validation_cache_metadata = ensure_foundation_cache(validation_dir, device)
    train_arrays = transform_representation(train_dir, preprocessor, tokenizer=None)
    validation_arrays = transform_representation(validation_dir, preprocessor, tokenizer=None)
    train_labels = aligned_labels(train_dir, train_arrays["ids"])
    validation_labels = aligned_labels(validation_dir, validation_arrays["ids"])
    train_cache = load_cache_for_training(train_cache_path, train_arrays["ids"])
    validation_cache = load_cache_for_training(validation_cache_path, validation_arrays["ids"])

    full_run = len(train_labels) > 1000
    candidate_specs = CANDIDATES if full_run else CANDIDATES[:1]
    warmup_epochs = 1
    finetune_epochs = 4 if full_run else 1

    # Select the largest requested true minibatch that passes a forward/backward smoke.
    selected_batch_size = None
    smoke = None
    for batch_size in PREFERRED_BATCH_SIZES:
        if batch_size > len(train_labels) and len(train_labels) > 0:
            trial_size = len(train_labels)
        else:
            trial_size = batch_size
        try:
            smoke = training_smoke(
                initial_checkpoint, train_arrays, train_cache, train_labels,
                device, amp_dtype, trial_size,
            )
            selected_batch_size = trial_size if not full_run else batch_size
            break
        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if selected_batch_size is None or smoke is None:
        raise RuntimeError("All classifier batch-size fallbacks exhausted")

    global_best: dict | None = None
    history = {
        "configuration": {
            "candidates": [dict(item) for item in candidate_specs],
            "warmup_epochs": warmup_epochs,
            "maximum_finetune_epochs": finetune_epochs,
            "patience_after_warmup": PATIENCE,
            "batch_size": selected_batch_size,
            "weight_decay": WEIGHT_DECAY,
            "device": str(device),
            "amp_dtype": str(amp_dtype),
            "train_rows": len(train_labels),
            "validation_rows": len(validation_labels),
            "initial_model_sha256": file_sha256(INITIAL_CHECKPOINT),
        },
        "epochs": [],
    }
    reporter_epoch = 0

    for candidate in candidate_specs:
        seed = int(candidate["seed"])
        seed_everything(seed)
        model, compatibility = build_model(initial_checkpoint, device)
        # Eligible immutable epoch 0.
        epoch0_metrics, epoch0_probabilities = evaluate(
            model, validation_arrays, validation_cache, validation_labels,
            selected_batch_size, device, amp_dtype,
        )
        epoch0 = {
            "candidate": candidate["name"], "seed": seed, "epoch": 0,
            "optimization_epochs": 0, "stage": "initialization", "train_loss": None,
            **epoch0_metrics,
        }
        history["epochs"].append(epoch0)
        reporter.report_epoch(
            epoch=reporter_epoch, train_loss=None, val_loss=epoch0_metrics["log_loss"],
            val_metric_name="AUPRC", val_metric=epoch0_metrics["auprc"],
            early_stopping_patience_remaining=PATIENCE,
        )
        reporter_epoch += 1
        candidate_best_metric = float(epoch0_metrics["auprc"])
        candidate_best_logloss = float(epoch0_metrics["log_loss"])
        no_improve = 0
        optimization_epochs = 0
        if better_checkpoint(epoch0_metrics, 0, global_best):
            global_best = {
                "candidate": candidate["name"], "seed": seed, "candidate_epoch": 0,
                "optimization_epochs": 0, "metrics": epoch0_metrics,
                "state_dict": clone_state(model), "probabilities": epoch0_probabilities.copy(),
                "compatibility": compatibility,
            }

        # One new-branch-only warm-up epoch.
        warm_optimizer = optimizer_for_stage(model, candidate, "warmup")
        optimization_epochs += 1
        warm_loss = train_one_epoch(
            model, train_arrays, train_cache, train_labels, warm_optimizer,
            selected_batch_size, device, amp_dtype, seed, 1, reporter_epoch, reporter,
        )
        warm_metrics, warm_probabilities = evaluate(
            model, validation_arrays, validation_cache, validation_labels,
            selected_batch_size, device, amp_dtype,
        )
        warm_row = {
            "candidate": candidate["name"], "seed": seed, "epoch": 1,
            "optimization_epochs": optimization_epochs, "stage": "warmup", "train_loss": warm_loss,
            **warm_metrics,
        }
        history["epochs"].append(warm_row)
        warm_improved = (
            warm_metrics["auprc"] > candidate_best_metric + 1e-12
            or (
                abs(warm_metrics["auprc"] - candidate_best_metric) <= 1e-12
                and warm_metrics["log_loss"] < candidate_best_logloss - 1e-12
            )
        )
        if warm_improved:
            candidate_best_metric = float(warm_metrics["auprc"])
            candidate_best_logloss = float(warm_metrics["log_loss"])
        if better_checkpoint(warm_metrics, optimization_epochs, global_best):
            global_best = {
                "candidate": candidate["name"], "seed": seed, "candidate_epoch": 1,
                "optimization_epochs": optimization_epochs, "metrics": warm_metrics,
                "state_dict": clone_state(model), "probabilities": warm_probabilities.copy(),
                "compatibility": compatibility,
            }
        reporter.report_epoch(
            epoch=reporter_epoch, train_loss=warm_loss, val_loss=warm_metrics["log_loss"],
            val_metric_name="AUPRC", val_metric=warm_metrics["auprc"],
            early_stopping_patience_remaining=PATIENCE,
        )
        reporter_epoch += 1
        del warm_optimizer

        # Up to four fine-tuning epochs; patience starts after warm-up.
        fine_optimizer = optimizer_for_stage(model, candidate, "finetune")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            fine_optimizer, mode="max", factor=0.5, patience=1,
        )
        for fine_epoch in range(1, finetune_epochs + 1):
            optimization_epochs += 1
            local_epoch = 1 + fine_epoch
            train_loss_value = train_one_epoch(
                model, train_arrays, train_cache, train_labels, fine_optimizer,
                selected_batch_size, device, amp_dtype, seed, local_epoch,
                reporter_epoch, reporter,
            )
            validation_metrics, probabilities = evaluate(
                model, validation_arrays, validation_cache, validation_labels,
                selected_batch_size, device, amp_dtype,
            )
            row = {
                "candidate": candidate["name"], "seed": seed, "epoch": local_epoch,
                "optimization_epochs": optimization_epochs, "stage": "finetune",
                "train_loss": train_loss_value, **validation_metrics,
            }
            history["epochs"].append(row)
            improved = (
                validation_metrics["auprc"] > candidate_best_metric + 1e-12
                or (
                    abs(validation_metrics["auprc"] - candidate_best_metric) <= 1e-12
                    and validation_metrics["log_loss"] < candidate_best_logloss - 1e-12
                )
            )
            if improved:
                candidate_best_metric = float(validation_metrics["auprc"])
                candidate_best_logloss = float(validation_metrics["log_loss"])
                no_improve = 0
            else:
                no_improve += 1
            if better_checkpoint(validation_metrics, optimization_epochs, global_best):
                global_best = {
                    "candidate": candidate["name"], "seed": seed,
                    "candidate_epoch": local_epoch, "optimization_epochs": optimization_epochs,
                    "metrics": validation_metrics, "state_dict": clone_state(model),
                    "probabilities": probabilities.copy(), "compatibility": compatibility,
                }
            scheduler.step(validation_metrics["auprc"])
            reporter.report_epoch(
                epoch=reporter_epoch, train_loss=train_loss_value,
                val_loss=validation_metrics["log_loss"], val_metric_name="AUPRC",
                val_metric=validation_metrics["auprc"],
                early_stopping_patience_remaining=max(0, PATIENCE - no_improve),
            )
            reporter_epoch += 1
            history["best_so_far"] = {
                "candidate": global_best["candidate"],
                "candidate_epoch": global_best["candidate_epoch"],
                "optimization_epochs": global_best["optimization_epochs"],
                **global_best["metrics"],
            }
            atomic_json(history, artifacts / "metrics_in_progress.json")
            if no_improve >= PATIENCE:
                break
        del model, fine_optimizer, scheduler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if global_best is None:
        raise RuntimeError("No eligible checkpoint was produced")
    history["best"] = {
        "candidate": global_best["candidate"],
        "seed": global_best["seed"],
        "candidate_epoch": global_best["candidate_epoch"],
        "optimization_epochs": global_best["optimization_epochs"],
        **global_best["metrics"],
    }
    history["elapsed_seconds"] = time.time() - started
    cache_metadata = {"train": train_cache_metadata, "validation": validation_cache_metadata}
    save_artifacts(
        artifacts, initial_checkpoint, global_best, preprocessor, history, smoke,
        cache_metadata, validation_arrays["ids"], validation_labels,
    )
    (artifacts / "metrics_in_progress.json").unlink(missing_ok=True)

    # Production reconstruction and deterministic prediction smoke using only saved files.
    saved = torch.load(artifacts / "model.pt", map_location="cpu", weights_only=False)
    reload_model = PHACTGatedLayerMixFusionNet(
        ArchitectureConfig(**saved["architecture_config"]),
        LayerMixConfig(**saved["layer_mix_config"]),
    )
    reload_model.load_state_dict(saved["state_dict"], strict=True)
    reload_model.to(device).eval()
    smoke_indices = np.arange(min(64, len(validation_labels)), dtype=np.int64)
    with torch.inference_mode():
        kwargs, _ = make_batch(
            validation_arrays, validation_cache, smoke_indices, None, device,
            amp_dtype if device.type == "cuda" else torch.float32,
        )
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            p1 = torch.sigmoid(reload_model(**kwargs).float()).cpu().numpy()
            p2 = torch.sigmoid(reload_model(**kwargs).float()).cpu().numpy()
    reload_report = {
        "rows": len(smoke_indices),
        "finite": bool(np.isfinite(p1).all()),
        "in_unit_interval": bool(((p1 >= 0) & (p1 <= 1)).all()),
        "deterministic_max_abs_difference": float(np.max(np.abs(p1 - p2))) if len(p1) else 0.0,
        "strict_state_reload": True,
        "artifact_local_foundation_present": (artifacts / "rinalmo_model" / "model.safetensors").is_file(),
    }
    if not reload_report["finite"] or not reload_report["in_unit_interval"] or reload_report["deterministic_max_abs_difference"] != 0.0:
        raise RuntimeError(f"Production reload smoke failed: {reload_report}")
    atomic_json(reload_report, artifacts / "production_reload_smoke.json")
    print(json.dumps({
        "status": "ok",
        "best_candidate": global_best["candidate"],
        "best_epoch": global_best["candidate_epoch"],
        "best_validation_auprc": global_best["metrics"]["auprc"],
        "elapsed_seconds": history["elapsed_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
