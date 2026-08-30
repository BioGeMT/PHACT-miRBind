#!/usr/bin/env python3
"""Standalone inference for the iteration-33 PHACT-gated RiNALMo classifier.

The command consumes only the fixed input/ interface.  It rebuilds all
train-fitted baseline features with the saved preprocessor, extracts frozen
RiNALMo-micro block outputs 3/6/9/12 into a temporary deduplicated cache, and
emits one probability for every samples.tsv row in the original order.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import pickle
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from data_representation import (
    build_mirbind2_onehot,
    build_rc_pairwise_grid,
    transform_representation,
)
from foundation_layer_cache import build_cache
from layer_mix_model import ArchitectureConfig, LayerMixConfig, PHACTGatedLayerMixFusionNet


FOUNDATION_BATCH_SIZES: Tuple[int, ...] = (1024, 768, 512, 256)
PREDICTION_BATCH_SIZES: Tuple[int, ...] = (768, 512, 384)
SELECTED_LAYERS: Tuple[int, ...] = (3, 6, 9, 12)
HIDDEN_SIZE = 480


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict AGO2 miRNA-target interactions")
    parser.add_argument("--input", required=True, help="Path to the split input/ folder")
    parser.add_argument("--output", required=True, help="Destination prediction CSV")
    parser.add_argument("--artifacts-dir", required=True, help="Saved training artifacts")
    return parser.parse_args()


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_input_folder(path: Path) -> Tuple[Path, Path]:
    """Return the canonical input folder and split root expected by representation code."""
    path = Path(path).expanduser().resolve()
    if (path / "samples.tsv").is_file():
        input_dir = path
    elif (path / "input" / "samples.tsv").is_file():
        # Kept for defensive compatibility; the documented interface passes input/ itself.
        input_dir = path / "input"
    else:
        raise FileNotFoundError(f"No samples.tsv found in input folder: {path}")
    if input_dir.name != "input":
        raise ValueError("--input must be the fixed split input/ folder")
    required = (
        "samples.tsv",
        "sample_mirna_candidates.tsv",
        "phact_mirna_positions.tsv",
        "phact_target_positions.tsv",
        "mirgenedb_premirna_orthologues.tsv",
    )
    missing = [name for name in required if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Input folder is missing required files: {missing}")
    return input_dir, input_dir.parent


def validate_artifacts(artifacts_dir: Path) -> dict:
    artifacts_dir = artifacts_dir.expanduser().resolve()
    required = (
        "model.pt",
        "preprocessor.pkl",
        "architecture.json",
        "representation_contract.json",
        "layer_feature_metadata.json",
        "rinalmo_model/config.json",
        "rinalmo_model/model.safetensors",
        "rinalmo_model/tokenizer_config.json",
        "rinalmo_model/vocab.txt",
    )
    missing = [name for name in required if not (artifacts_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Artifacts directory is incomplete: {missing}")
    metadata = json.loads((artifacts_dir / "layer_feature_metadata.json").read_text())
    if tuple(metadata.get("selected_hidden_state_indices", ())) != SELECTED_LAYERS:
        raise ValueError("Artifact hidden-state order is not [3,6,9,12]")
    if int(metadata.get("hidden_size", -1)) != HIDDEN_SIZE:
        raise ValueError("Artifact RiNALMo hidden width is not 480")
    if metadata.get("pair_vector_order") != ["mirna", "target", "absolute_difference", "elementwise_product"]:
        raise ValueError("Artifact pair-vector order is incompatible")
    expected_preprocessor_hash = metadata.get("preprocessor_sha256")
    if expected_preprocessor_hash and file_sha256(artifacts_dir / "preprocessor.pkl") != expected_preprocessor_hash:
        raise ValueError("preprocessor.pkl checksum does not match layer metadata")
    expected_config_hash = metadata.get("rinalmo_config_sha256")
    if expected_config_hash and file_sha256(artifacts_dir / "rinalmo_model/config.json") != expected_config_hash:
        raise ValueError("RiNALMo config checksum does not match layer metadata")
    return metadata


def is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def choose_cuda_dtype() -> torch.dtype:
    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def remove_cache_files(cache_path: Path) -> None:
    candidates = (
        cache_path,
        cache_path.with_suffix(cache_path.suffix + ".json"),
        cache_path.with_name(cache_path.name + ".tmp.npz"),
    )
    for candidate in candidates:
        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            pass


def build_temporary_foundation_cache(
    input_dir: Path,
    rinalmo_dir: Path,
    cache_path: Path,
    preferred_device: torch.device,
) -> Tuple[dict, torch.device, int]:
    """Build one label-free cache, retrying CUDA batches then falling back to CPU."""
    devices = [preferred_device]
    if preferred_device.type == "cuda":
        devices.append(torch.device("cpu"))
    last_error: BaseException | None = None
    for device in devices:
        batch_sizes: Sequence[int] = FOUNDATION_BATCH_SIZES if device.type == "cuda" else (256,)
        for batch_size in batch_sizes:
            remove_cache_files(cache_path)
            try:
                log(f"foundation extraction: device={device}, batch_size={batch_size}")
                metadata = build_cache(input_dir, rinalmo_dir, cache_path, batch_size, str(device))
                metadata["extraction_batch_size"] = int(batch_size)
                metadata["extraction_device"] = str(device)
                return metadata, device, batch_size
            except BaseException as exc:
                last_error = exc
                if device.type == "cuda" and is_cuda_oom(exc):
                    log(f"foundation extraction CUDA OOM at batch_size={batch_size}; retrying")
                    clear_cuda()
                    continue
                raise
    raise RuntimeError("Foundation extraction failed on CUDA and CPU") from last_error


def load_foundation_cache(path: Path, expected_ids: np.ndarray) -> Dict[str, np.ndarray]:
    required = (
        "ids",
        "layer_indices",
        "mirna_pooled",
        "target_pooled",
        "mirna_row_to_unique",
        "target_row_to_unique",
        "mirna_valid",
        "target_valid",
    )
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(f"Temporary foundation cache is missing: {missing}")
        cache = {name: archive[name] for name in required}
    if tuple(int(x) for x in cache["layer_indices"].tolist()) != SELECTED_LAYERS:
        raise ValueError("Temporary foundation cache layer order mismatch")
    cache_ids = cache["ids"].astype(str)
    expected = expected_ids.astype(str)
    if not np.array_equal(cache_ids, expected):
        raise ValueError("Temporary foundation cache does not preserve samples.tsv ID order")
    if cache["mirna_pooled"].ndim != 3 or cache["mirna_pooled"].shape[1:] != (4, HIDDEN_SIZE):
        raise ValueError("Unexpected miRNA foundation cache shape")
    if cache["target_pooled"].ndim != 3 or cache["target_pooled"].shape[1:] != (4, HIDDEN_SIZE):
        raise ValueError("Unexpected target foundation cache shape")
    n_rows = len(expected)
    if len(cache["mirna_row_to_unique"]) != n_rows or len(cache["target_row_to_unique"]) != n_rows:
        raise ValueError("Foundation row-map length mismatch")
    if not np.all(cache["mirna_valid"] == 1) or not np.all(cache["target_valid"] == 1):
        raise ValueError("Foundation extraction produced an invalid sequence pool")
    return cache


def load_classifier(artifacts_dir: Path, device: torch.device) -> PHACTGatedLayerMixFusionNet:
    checkpoint = torch.load(artifacts_dir / "model.pt", map_location="cpu", weights_only=False)
    required = ("state_dict", "architecture_config", "layer_mix_config")
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise ValueError(f"model.pt is missing required entries: {missing}")
    model = PHACTGatedLayerMixFusionNet(
        ArchitectureConfig(**checkpoint["architecture_config"]),
        LayerMixConfig(**checkpoint["layer_mix_config"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.requires_grad_(False)
    model.eval()
    return model.to(device)


def to_device(
    array: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    selected = np.ascontiguousarray(array[indices])
    return torch.from_numpy(selected).to(device=device, dtype=dtype, non_blocking=(device.type == "cuda"))


def make_layer_pairs(
    cache: Mapping[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    mirna_indices = cache["mirna_row_to_unique"][indices].astype(np.int64, copy=False)
    target_indices = cache["target_row_to_unique"][indices].astype(np.int64, copy=False)
    mirna = torch.from_numpy(np.ascontiguousarray(cache["mirna_pooled"][mirna_indices])).to(
        device, non_blocking=(device.type == "cuda")
    )
    target = torch.from_numpy(np.ascontiguousarray(cache["target_pooled"][target_indices])).to(
        device, non_blocking=(device.type == "cuda")
    )
    return torch.cat((mirna, target, torch.abs(mirna - target), mirna * target), dim=2)


def make_batch(
    arrays: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    pair_dtype: torch.dtype,
) -> dict:
    mirna_codes = to_device(arrays["mirna_codes"], indices, device, torch.long)
    target_codes = to_device(arrays["target_codes"], indices, device, torch.long)
    return {
        "pairwise_onehot": build_mirbind2_onehot(mirna_codes, target_codes, pair_dtype),
        "rc_pairwise_grid": build_rc_pairwise_grid(mirna_codes, target_codes, pair_dtype),
        "aux_numeric": to_device(arrays["aux_numeric"], indices, device, torch.float32),
        "target_tensor": to_device(arrays["target_tensor"], indices, device, torch.float32),
        "mirna_phact_tensor": to_device(arrays["mirna_phact_tensor"], indices, device, torch.float32),
        "feature_idx": to_device(arrays["feature_index"], indices, device, torch.long),
        "dominant_region_idx": to_device(arrays["dominant_region_index"], indices, device, torch.long),
        "layer_pair_vectors": make_layer_pairs(cache, indices, device),
    }


def predict_once(
    model: PHACTGatedLayerMixFusionNet,
    arrays: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> np.ndarray:
    n_rows = len(arrays["ids"])
    probabilities = np.empty(n_rows, dtype=np.float32)
    pair_dtype = amp_dtype if device.type == "cuda" else torch.float32
    amp_context = (
        lambda: torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
        if device.type == "cuda"
        else nullcontext()
    )
    progress_step = max(100_000, batch_size)
    next_progress = progress_step
    model.eval()
    with torch.inference_mode():
        for start in range(0, n_rows, batch_size):
            stop = min(start + batch_size, n_rows)
            indices = np.arange(start, stop, dtype=np.int64)
            kwargs = make_batch(arrays, cache, indices, device, pair_dtype)
            with amp_context():
                logits = model(**kwargs)
            batch_probabilities = torch.sigmoid(logits.float()).cpu().numpy()
            if batch_probabilities.shape != (stop - start,):
                raise ValueError(f"Unexpected classifier output shape: {batch_probabilities.shape}")
            if not np.isfinite(batch_probabilities).all():
                raise FloatingPointError("Classifier produced a non-finite probability")
            probabilities[start:stop] = batch_probabilities
            del kwargs, logits, batch_probabilities
            if stop >= next_progress or stop == n_rows:
                log(f"prediction progress: {stop}/{n_rows}")
                next_progress += progress_step
    return probabilities


def predict_with_fallback(
    model: PHACTGatedLayerMixFusionNet,
    arrays: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
    preferred_device: torch.device,
) -> Tuple[np.ndarray, PHACTGatedLayerMixFusionNet, torch.device, int]:
    amp_dtype = choose_cuda_dtype() if preferred_device.type == "cuda" else torch.float32
    if preferred_device.type == "cuda":
        for batch_size in PREDICTION_BATCH_SIZES:
            try:
                log(f"classifier inference: device={preferred_device}, batch_size={batch_size}, amp={amp_dtype}")
                probabilities = predict_once(model, arrays, cache, batch_size, preferred_device, amp_dtype)
                return probabilities, model, preferred_device, batch_size
            except BaseException as exc:
                if not is_cuda_oom(exc):
                    raise
                log(f"classifier CUDA OOM at batch_size={batch_size}; retrying")
                clear_cuda()
        log("classifier CUDA batches exhausted; falling back to CPU")
        model = model.to("cpu").eval()
        clear_cuda()
        cpu_device = torch.device("cpu")
        probabilities = predict_once(model, arrays, cache, PREDICTION_BATCH_SIZES[-1], cpu_device, torch.float32)
        return probabilities, model, cpu_device, PREDICTION_BATCH_SIZES[-1]
    probabilities = predict_once(model, arrays, cache, PREDICTION_BATCH_SIZES[-1], preferred_device, torch.float32)
    return probabilities, model, preferred_device, PREDICTION_BATCH_SIZES[-1]


def atomic_write_predictions(output: Path, ids: np.ndarray, probabilities: np.ndarray) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if len(ids) != len(probabilities):
        raise ValueError("Probability count does not match input sample count")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("Probabilities must all be finite and within [0,1]")
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["id", "prediction", "probability_0", "probability_1"])
            for sample_id, probability in zip(ids.astype(str), probabilities):
                p1 = float(probability)
                p0 = 1.0 - p1
                writer.writerow([sample_id, int(p1 >= 0.5), format(p0, ".17g"), format(p1, ".17g")])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    started = time.time()
    input_dir, split_root = resolve_input_folder(Path(args.input))
    output = Path(args.output).expanduser().resolve()
    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    validate_artifacts(artifacts_dir)

    cache_path = output.parent / f".{output.name}.foundation.{os.getpid()}.npz"
    remove_cache_files(cache_path)
    preferred_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preferred_device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        log(f"using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        log("CUDA unavailable; using CPU")

    try:
        with (artifacts_dir / "preprocessor.pkl").open("rb") as handle:
            preprocessor = pickle.load(handle)
        log("reconstructing frozen train-fitted PHACT/sequence representation")
        arrays = transform_representation(split_root, preprocessor, tokenizer=None)
        ids = np.asarray(arrays["ids"], dtype=object)
        if len(ids) == 0:
            raise ValueError("samples.tsv contains no samples")
        if len(set(ids.astype(str).tolist())) != len(ids):
            raise ValueError("samples.tsv IDs must be unique")

        _, extraction_device, extraction_batch = build_temporary_foundation_cache(
            input_dir,
            artifacts_dir / "rinalmo_model",
            cache_path,
            preferred_device,
        )
        cache = load_foundation_cache(cache_path, ids)
        # The encoder is released by build_cache; clear its reserved CUDA memory before classifier loading.
        clear_cuda()
        classifier_device = preferred_device
        try:
            model = load_classifier(artifacts_dir, classifier_device)
        except BaseException as exc:
            if classifier_device.type != "cuda" or not is_cuda_oom(exc):
                raise
            log("classifier load exhausted CUDA memory; falling back to CPU")
            clear_cuda()
            classifier_device = torch.device("cpu")
            model = load_classifier(artifacts_dir, classifier_device)

        probabilities, model, prediction_device, prediction_batch = predict_with_fallback(
            model, arrays, cache, classifier_device
        )
        if len(probabilities) != len(ids):
            raise RuntimeError("Inference did not produce one probability per input sample")
        atomic_write_predictions(output, ids, probabilities)
        log(
            "inference complete: "
            f"rows={len(ids)}, extraction_device={extraction_device}, extraction_batch={extraction_batch}, "
            f"prediction_device={prediction_device}, prediction_batch={prediction_batch}, "
            f"seconds={time.time()-started:.1f}, output={output}"
        )
    finally:
        # The cache is an implementation detail and must never become a persistent artifact.
        remove_cache_files(cache_path)


if __name__ == "__main__":
    main()
