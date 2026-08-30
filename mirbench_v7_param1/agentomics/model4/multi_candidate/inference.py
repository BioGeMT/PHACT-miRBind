#!/usr/bin/env python3
"""Label-free inference for the standalone MultiCandidatePHACTFusionNet.

The script reconstructs the exact train-time representation from a supplied
``input/`` folder, extracts RiNALMo-micro block features with artifact-local
weights, performs one-model inference, and atomically writes one prediction per
``samples.tsv`` row.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import pickle
import random
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch

from baseline_model import ArchitectureConfig, LayerMixConfig
from candidate_representation import transform_candidate_representation
from data_representation import build_mirbind2_onehot, build_rc_pairwise_grid
from foundation_layer_cache import build_cache
from multi_candidate_model import (
    CandidateAmbiguityConfig,
    MultiCandidatePHACTFusionNet,
)

MODEL_BATCH_SIZES = (768, 512, 384, 256, 128, 64)
FOUNDATION_BATCH_SIZES = (1024, 768, 512, 256, 128, 64)
EXPECTED_LAYER_INDICES = [3, 6, 9, 12]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run label-free MultiCandidatePHACTFusionNet inference."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--artifacts-dir", type=Path, required=True
    )
    return parser.parse_args()


def seed_everything(seed: int = 20260802) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_input_folder(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "samples.tsv").is_file():
        input_dir = path
    elif (path / "input" / "samples.tsv").is_file():
        input_dir = path / "input"
    else:
        raise FileNotFoundError(
            f"--input must be an input folder containing samples.tsv: {path}"
        )
    required = (
        "samples.tsv",
        "sample_mirna_candidates.tsv",
        "phact_mirna_positions.tsv",
        "phact_target_positions.tsv",
        "mirgenedb_premirna_orthologues.tsv",
    )
    missing = [name for name in required if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")
    return input_dir


def verify_artifacts(artifacts_dir: Path) -> Dict[str, Path]:
    artifacts_dir = artifacts_dir.expanduser().resolve()
    paths = {
        "model": artifacts_dir / "model.pt",
        "preprocessor": artifacts_dir / "preprocessor.pkl",
        "rinalmo": artifacts_dir / "rinalmo_model",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing inference artifacts: {missing}")
    if not paths["rinalmo"].is_dir():
        raise NotADirectoryError(paths["rinalmo"])
    return paths


def choose_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def amp_dtype_for(device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError)
        and "out of memory" in str(exc).lower()
        and torch.cuda.is_available()
    )


def remove_cache_files(cache_path: Path) -> None:
    candidates = (
        cache_path,
        cache_path.with_suffix(cache_path.suffix + ".json"),
        cache_path.with_name(cache_path.name + ".tmp.npz"),
    )
    for path in candidates:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def build_foundation_cache_with_fallback(
    input_dir: Path,
    rinalmo_dir: Path,
    cache_path: Path,
    preferred_device: torch.device,
) -> Tuple[Path, dict]:
    """Build a label-free cache, reducing extraction batch size after CUDA OOM."""
    device_order = [preferred_device]
    if preferred_device.type == "cuda":
        device_order.append(torch.device("cpu"))
    last_oom: BaseException | None = None
    for device in device_order:
        batch_sizes = FOUNDATION_BATCH_SIZES if device.type == "cuda" else (128,)
        for batch_size in batch_sizes:
            remove_cache_files(cache_path)
            try:
                metadata = build_cache(
                    input_dir,
                    rinalmo_dir,
                    cache_path,
                    int(batch_size),
                    str(device),
                )
                metadata = dict(metadata)
                metadata["extraction_batch_size"] = int(batch_size)
                metadata["extraction_device"] = str(device)
                return cache_path, metadata
            except BaseException as exc:
                if device.type == "cuda" and is_cuda_oom(exc):
                    last_oom = exc
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                raise
    raise RuntimeError("RiNALMo cache extraction exhausted all fallbacks") from last_oom


def load_foundation_cache(
    cache_path: Path, expected_ids: np.ndarray
) -> Dict[str, np.ndarray]:
    needed = (
        "ids",
        "layer_indices",
        "mirna_pooled",
        "target_pooled",
        "mirna_row_to_unique",
        "target_row_to_unique",
        "mirna_valid",
        "target_valid",
    )
    with np.load(cache_path, allow_pickle=False) as archive:
        absent = [key for key in needed if key not in archive.files]
        if absent:
            raise KeyError(f"Foundation cache is missing arrays: {absent}")
        cache = {key: archive[key] for key in needed}
    cache["ids"] = cache["ids"].astype(str)
    if not np.array_equal(cache["ids"], expected_ids.astype(str)):
        raise ValueError("Foundation cache ID/order differs from samples.tsv")
    if cache["layer_indices"].astype(int).tolist() != EXPECTED_LAYER_INDICES:
        raise ValueError("Foundation cache hidden-state block mismatch")
    if not np.all(cache["mirna_valid"] == 1):
        raise ValueError("Invalid RiNALMo miRNA pooling mask")
    if not np.all(cache["target_valid"] == 1):
        raise ValueError("Invalid RiNALMo target pooling mask")
    if cache["mirna_pooled"].ndim != 3 or cache["mirna_pooled"].shape[1:] != (4, 480):
        raise ValueError(f"Unexpected miRNA foundation shape: {cache['mirna_pooled'].shape}")
    if cache["target_pooled"].ndim != 3 or cache["target_pooled"].shape[1:] != (4, 480):
        raise ValueError(f"Unexpected target foundation shape: {cache['target_pooled'].shape}")
    for name in ("mirna_pooled", "target_pooled"):
        if not np.isfinite(cache[name]).all():
            raise ValueError(f"Non-finite values in {name}")
    return cache


def load_preprocessor(path: Path) -> dict:
    with path.open("rb") as handle:
        preprocessor = pickle.load(handle)
    if not isinstance(preprocessor, dict):
        raise TypeError("preprocessor.pkl must contain a dictionary")
    return preprocessor


def instantiate_model(model_path: Path, device: torch.device) -> MultiCandidatePHACTFusionNet:
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if payload.get("model_name") != "MultiCandidatePHACTFusionNet":
        raise ValueError(f"Unexpected model payload: {payload.get('model_name')}")
    cfg = ArchitectureConfig(**dict(payload["architecture_config"]))
    layer_cfg = LayerMixConfig(**dict(payload["layer_mix_config"]))
    candidate_cfg = CandidateAmbiguityConfig(
        **dict(payload["candidate_ambiguity_config"])
    )
    model = MultiCandidatePHACTFusionNet(cfg, layer_cfg, candidate_cfg)
    result = model.load_state_dict(payload["state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Strict checkpoint load failed: {result}")
    model.freeze_inherited_parameters()
    model.to(device)
    model.eval()
    return model


def to_device(
    array: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    selected = np.ascontiguousarray(array[indices])
    return torch.from_numpy(selected).to(device=device, dtype=dtype, non_blocking=True)


def make_layer_pairs(
    cache: Mapping[str, np.ndarray], indices: np.ndarray, device: torch.device
) -> torch.Tensor:
    mirna_rows = cache["mirna_row_to_unique"][indices].astype(np.int64, copy=False)
    target_rows = cache["target_row_to_unique"][indices].astype(np.int64, copy=False)
    if mirna_rows.size and (
        mirna_rows.min() < 0 or mirna_rows.max() >= cache["mirna_pooled"].shape[0]
    ):
        raise IndexError("miRNA foundation row mapping is out of range")
    if target_rows.size and (
        target_rows.min() < 0 or target_rows.max() >= cache["target_pooled"].shape[0]
    ):
        raise IndexError("target foundation row mapping is out of range")
    mirna = torch.from_numpy(
        np.ascontiguousarray(cache["mirna_pooled"][mirna_rows])
    ).to(device=device, non_blocking=True)
    target = torch.from_numpy(
        np.ascontiguousarray(cache["target_pooled"][target_rows])
    ).to(device=device, non_blocking=True)
    return torch.cat(
        (mirna, target, torch.abs(mirna - target), mirna * target), dim=2
    )


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
        "pairwise_onehot": build_mirbind2_onehot(
            mirna_codes, target_codes, pair_dtype
        ),
        "rc_pairwise_grid": build_rc_pairwise_grid(
            mirna_codes, target_codes, pair_dtype
        ),
        "aux_numeric": to_device(
            arrays["aux_numeric"], indices, device, torch.float32
        ),
        "target_tensor": to_device(
            arrays["target_tensor"], indices, device, torch.float32
        ),
        "mirna_phact_tensor": to_device(
            arrays["mirna_phact_tensor"], indices, device, torch.float32
        ),
        "candidate_phact": to_device(
            arrays["candidate_phact"], indices, device, torch.float32
        ),
        "candidate_profile_mask": to_device(
            arrays["candidate_profile_mask"], indices, device, torch.float32
        ),
        "feature_idx": to_device(
            arrays["feature_index"], indices, device, torch.long
        ),
        "dominant_region_idx": to_device(
            arrays["dominant_region_index"], indices, device, torch.long
        ),
        "layer_pair_vectors": make_layer_pairs(cache, indices, device),
    }


def validate_representation(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    required = {
        "ids": None,
        "mirna_codes": (28,),
        "target_codes": (50,),
        "aux_numeric": (104,),
        "target_tensor": (12, 50),
        "mirna_phact_tensor": (5, 28),
        "candidate_phact": (3, 5, 28),
        "candidate_profile_mask": (3,),
        "feature_index": None,
        "dominant_region_index": None,
    }
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError(f"Representation is missing arrays: {missing}")
    ids = np.asarray(arrays["ids"]).astype(str)
    n = len(ids)
    if n == 0:
        raise ValueError("Input contains no samples")
    if len(set(ids.tolist())) != n:
        raise ValueError("samples.tsv IDs must be unique")
    for key, trailing_shape in required.items():
        value = np.asarray(arrays[key])
        if len(value) != n:
            raise ValueError(f"Row count mismatch for {key}: {len(value)} != {n}")
        if trailing_shape is not None and value.shape[1:] != trailing_shape:
            raise ValueError(f"Shape mismatch for {key}: {value.shape}")
        if key != "ids" and not np.isfinite(value).all():
            raise ValueError(f"Non-finite values in {key}")
    if not np.array_equal(
        arrays["candidate_phact"][:, 0], arrays["mirna_phact_tensor"]
    ):
        raise ValueError("Candidate slot 0 differs from inherited selected profile")
    return ids


def iter_indices(n: int, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, n, batch_size):
        yield np.arange(start, min(start + batch_size, n), dtype=np.int64)


def predict_at_batch_size(
    model: MultiCandidatePHACTFusionNet,
    arrays: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    n = len(arrays["ids"])
    probabilities = np.empty(n, dtype=np.float32)
    amp_dtype = amp_dtype_for(device)
    pair_dtype = amp_dtype if amp_dtype is not None else torch.float32
    offset = 0
    with torch.inference_mode():
        for indices in iter_indices(n, batch_size):
            kwargs = make_batch(arrays, cache, indices, device, pair_dtype)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_dtype is not None
                else nullcontext()
            )
            with amp_context:
                logits = model(**kwargs)
            batch_probability = torch.sigmoid(logits.float()).reshape(-1).cpu().numpy()
            if len(batch_probability) != len(indices):
                raise RuntimeError("Model did not return one logit per input sample")
            probabilities[offset : offset + len(indices)] = batch_probability
            offset += len(indices)
            del kwargs, logits, batch_probability
    if offset != n:
        raise RuntimeError(f"Predicted {offset} of {n} samples")
    if not np.isfinite(probabilities).all():
        raise ValueError("Model produced non-finite probabilities")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("Model probabilities fall outside [0, 1]")
    return probabilities


def predict_with_fallback(
    model_path: Path,
    arrays: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
    preferred_device: torch.device,
) -> Tuple[np.ndarray, str, int, str]:
    devices = [preferred_device]
    if preferred_device.type == "cuda":
        devices.append(torch.device("cpu"))
    last_oom: BaseException | None = None
    for device in devices:
        model = instantiate_model(model_path, device)
        batch_sizes: Sequence[int] = MODEL_BATCH_SIZES if device.type == "cuda" else (128,)
        for batch_size in batch_sizes:
            try:
                probabilities = predict_at_batch_size(
                    model, arrays, cache, device, int(batch_size)
                )
                dtype_name = (
                    str(amp_dtype_for(device)).replace("torch.", "")
                    if device.type == "cuda"
                    else "float32"
                )
                return probabilities, str(device), int(batch_size), dtype_name
            except BaseException as exc:
                if device.type == "cuda" and is_cuda_oom(exc):
                    last_oom = exc
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                raise
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    raise RuntimeError("Model inference exhausted all batch/device fallbacks") from last_oom


def atomic_write_predictions(
    output_path: Path, ids: np.ndarray, probability_1: np.ndarray
) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["id", "prediction", "probability_0", "probability_1"])
            for sample_id, p1 in zip(ids.tolist(), probability_1.tolist()):
                p1_float = float(p1)
                writer.writerow(
                    [
                        str(sample_id),
                        int(p1_float >= 0.5),
                        f"{1.0 - p1_float:.10g}",
                        f"{p1_float:.10g}",
                    ]
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    seed_everything()
    input_dir = resolve_input_folder(args.input)
    artifact_paths = verify_artifacts(args.artifacts_dir)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = output_path.parent / (
        f".{output_path.name}.{os.getpid()}.{uuid.uuid4().hex}.rinalmo_cache.npz"
    )
    try:
        preprocessor = load_preprocessor(artifact_paths["preprocessor"])
        arrays = transform_candidate_representation(
            input_dir, preprocessor, tokenizer=None, foundation_cache_path=None
        )
        ids = validate_representation(arrays)
        preferred_device = choose_device()
        build_foundation_cache_with_fallback(
            input_dir, artifact_paths["rinalmo"], cache_path, preferred_device
        )
        cache = load_foundation_cache(cache_path, ids)
        probabilities, device_name, batch_size, dtype_name = predict_with_fallback(
            artifact_paths["model"], arrays, cache, preferred_device
        )
        if len(probabilities) != len(ids):
            raise RuntimeError(
                f"Prediction count mismatch: {len(probabilities)} != {len(ids)}"
            )
        atomic_write_predictions(output_path, ids, probabilities)
        summary = {
            "rows": int(len(ids)),
            "device": device_name,
            "batch_size": int(batch_size),
            "amp_dtype": dtype_name,
            "output": str(output_path),
        }
        print(json.dumps(summary, sort_keys=True))
    finally:
        remove_cache_files(cache_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
