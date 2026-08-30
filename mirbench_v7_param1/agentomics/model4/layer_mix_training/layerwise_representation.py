#!/usr/bin/env python3
"""Leakage-safe layerwise extension of the iteration-30 representation.

The baseline transform and train-fitted preprocessor are reused without
refitting.  Hidden-state caches are label-free and aligned by samples.tsv row
order.  Large pair grids remain lazily materialized per minibatch.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from data_representation import (
    AUX_NUMERIC_FEATURE_NAMES,
    MIRNA_LENGTH,
    MIRNA_PHACT_CHANNELS,
    MIRNA_PHACT_SCORE_CHANNELS,
    RINALMO_CONCAT_DIM,
    RINALMO_HIDDEN_SIZE,
    TARGET_BRANCH_CHANNELS,
    TARGET_LENGTH,
    build_mirbind2_onehot,
    build_rc_pairwise_grid,
    make_online_rinalmo_pair_vector,
    transform_representation,
)

SELECTED_LAYER_INDICES: Tuple[int, ...] = (3, 6, 9, 12)
LAYER_PAIR_DIM = 4 * RINALMO_HIDDEN_SIZE


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_preprocessor(path: Path) -> dict:
    """Load, but never fit or mutate, the iteration-30 preprocessor."""
    with Path(path).open("rb") as handle:
        preprocessor = pickle.load(handle)
    required = {
        "aux_numeric_feature_names",
        "aux_numeric_scaler",
        "positional_scalers",
        "target_phact_scalers",
        "target_phact_channel_order",
        "mirna_phact_columns",
        "mirna_phact_scalers",
        "category_vocabs",
    }
    missing = sorted(required - set(preprocessor))
    if missing:
        raise ValueError(f"Frozen preprocessor lacks keys: {missing}")
    if list(preprocessor["aux_numeric_feature_names"]) != list(AUX_NUMERIC_FEATURE_NAMES):
        raise ValueError("104-feature order does not match iteration-30 source")
    if len(preprocessor["aux_numeric_feature_names"]) != 104:
        raise ValueError("Expected exactly 104 numeric features")
    if len(preprocessor["mirna_phact_columns"]) != MIRNA_PHACT_SCORE_CHANNELS:
        raise ValueError("Expected exactly 64 miRNA PHACT score columns")
    if set(preprocessor["category_vocabs"]) != {"feature", "dominant_region"}:
        raise ValueError("Unexpected category vocabulary set")
    return preprocessor


def load_foundation_cache(path: Path) -> Dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        cache = {name: data[name] for name in data.files}
    required = {
        "ids", "layer_indices", "mirna_pooled", "target_pooled",
        "mirna_row_to_unique", "target_row_to_unique", "mirna_valid", "target_valid",
    }
    missing = sorted(required - set(cache))
    if missing:
        raise ValueError(f"Foundation cache lacks arrays: {missing}")
    if tuple(cache["layer_indices"].tolist()) != SELECTED_LAYER_INDICES:
        raise ValueError("Unexpected foundation layer order")
    if cache["mirna_pooled"].shape[1:] != (4, RINALMO_HIDDEN_SIZE):
        raise ValueError("Unexpected miRNA pooled shape")
    if cache["target_pooled"].shape[1:] != (4, RINALMO_HIDDEN_SIZE):
        raise ValueError("Unexpected target pooled shape")
    if cache["mirna_pooled"].dtype != np.float32 or cache["target_pooled"].dtype != np.float32:
        raise TypeError("Foundation pools must be float32")
    return cache


def construct_pair_vectors(mirna_layers: np.ndarray, target_layers: np.ndarray) -> np.ndarray:
    """Construct [miRNA,target,absolute difference,product] for each layer."""
    if mirna_layers.shape != target_layers.shape or mirna_layers.ndim != 3:
        raise ValueError("Expected matching [rows,layers,hidden] arrays")
    result = np.concatenate(
        [mirna_layers, target_layers, np.abs(mirna_layers - target_layers), mirna_layers * target_layers],
        axis=2,
    ).astype(np.float32, copy=False)
    if result.shape[1:] != (4, LAYER_PAIR_DIM):
        raise ValueError(f"Unexpected layer-pair shape {result.shape}")
    if not np.isfinite(result).all():
        raise FloatingPointError("Layer pair vectors contain non-finite values")
    return result


def attach_foundation_layers(arrays: Dict[str, np.ndarray], cache_path: Path) -> Dict[str, np.ndarray]:
    cache = load_foundation_cache(cache_path)
    ids = arrays["ids"].astype(str)
    if not np.array_equal(ids, cache["ids"].astype(str)):
        raise ValueError("Foundation cache IDs/order do not exactly match representation rows")
    mi_inv = cache["mirna_row_to_unique"].astype(np.int64, copy=False)
    tg_inv = cache["target_row_to_unique"].astype(np.int64, copy=False)
    if len(mi_inv) != len(ids) or len(tg_inv) != len(ids):
        raise ValueError("Foundation inverse-map length mismatch")
    if not np.all(cache["mirna_valid"] == 1) or not np.all(cache["target_valid"] == 1):
        raise ValueError("Foundation cache contains invalid unique-sequence pools")
    mirna_layers = cache["mirna_pooled"][mi_inv]
    target_layers = cache["target_pooled"][tg_inv]
    layer_pair_vectors = construct_pair_vectors(mirna_layers, target_layers)
    arrays = dict(arrays)
    arrays["foundation_mirna_layers"] = mirna_layers.astype(np.float32, copy=False)
    arrays["foundation_target_layers"] = target_layers.astype(np.float32, copy=False)
    arrays["layer_pair_vectors"] = layer_pair_vectors
    # The original baseline adapter consumes the same selected final-layer pair.
    arrays["rinalmo_pair_vector"] = layer_pair_vectors[:, 3, :]
    return arrays


def transform_layerwise_representation(
    split_dir: Path,
    preprocessor_path: Path,
    foundation_cache_path: Path,
) -> Dict[str, np.ndarray]:
    preprocessor = load_frozen_preprocessor(preprocessor_path)
    baseline = transform_representation(Path(split_dir), preprocessor, tokenizer=None)
    return attach_foundation_layers(baseline, foundation_cache_path)


def materialize_pairwise_batch(
    arrays: Mapping[str, np.ndarray],
    indices: Sequence[int],
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialize the two large established pair tensors only for a minibatch."""
    idx = np.asarray(indices, dtype=np.int64)
    mi = torch.from_numpy(arrays["mirna_codes"][idx]).to(device)
    tg = torch.from_numpy(arrays["target_codes"][idx]).to(device)
    return build_mirbind2_onehot(mi, tg, dtype), build_rc_pairwise_grid(mi, tg, dtype)


def expected_shapes() -> dict:
    return {
        "mirbind2_pair_per_sample": [MIRNA_LENGTH, TARGET_LENGTH, 18],
        "reverse_complement_pair_grid_per_sample": [20, MIRNA_LENGTH, TARGET_LENGTH],
        "target_conservation_phact_per_sample": [TARGET_BRANCH_CHANNELS, TARGET_LENGTH],
        "mirna_phact_per_sample": [MIRNA_PHACT_CHANNELS, MIRNA_LENGTH],
        "aux_numeric_per_sample": 104,
        "foundation_sequence_layers_per_sample": [4, RINALMO_HIDDEN_SIZE],
        "foundation_layer_pair_vectors_per_sample": [4, LAYER_PAIR_DIM],
        "baseline_rinalmo_pair_vector_per_sample": RINALMO_CONCAT_DIM,
    }
