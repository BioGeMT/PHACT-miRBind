#!/usr/bin/env python3
"""Leakage-safe multi-candidate PHACT extension of the frozen fusion representation.

The inherited representation and preprocessor are intentionally unchanged.  This
module adds ordered candidate-level miRNA PHACT tensors using only files under an
input directory and the frozen train-fitted miRNA PHACT scalers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import data_representation as inherited
from layerwise_representation import attach_foundation_layers, materialize_pairwise_batch

MAX_CANDIDATE_PROFILES = 3
CANDIDATE_CHANNELS = inherited.MIRNA_PHACT_CHANNELS
CANDIDATE_LENGTH = inherited.MIRNA_LENGTH
_MISSING_MATURE = {"", "NA", "NAN", "NONE", "NULL"}


def resolve_split_root(path: Path) -> Path:
    """Resolve either a split root or its fixed ``input/`` folder.

    No label file is opened.  The returned root is used only because the
    inherited, immutable transformer addresses tables as ``root/input/*.tsv``.
    """
    path = Path(path).resolve()
    if (path / "input" / "samples.tsv").is_file():
        return path
    if (path / "samples.tsv").is_file() and path.name == "input":
        return path.parent
    raise FileNotFoundError(
        f"Expected a split root containing input/samples.tsv or the input folder itself: {path}"
    )


def _ordered_profile_candidates(
    split_root: Path, sample_ids: Sequence[str]
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Return profile-bearing candidates in stable numeric candidate order."""
    path = split_root / "input" / "sample_mirna_candidates.tsv"
    columns = ["row_idx", "slot", "candidate_index", "mirgenedb_mature_id"]
    if not path.is_file():
        return pd.DataFrame(columns=columns), {
            "candidate_file_present": False,
            "profile_bearing_rows": 0,
            "retained_rows": 0,
            "truncated_rows": 0,
        }

    ids = pd.Index(np.asarray(sample_ids, dtype=object), dtype="object")
    if ids.has_duplicates:
        raise ValueError("sample IDs must be unique")
    id_to_row = pd.Series(np.arange(len(ids), dtype=np.int64), index=ids)
    cand = pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "id",
            "candidate_index",
            "mirgenedb_mature_id",
            "has_phact_profile",
        ],
        dtype={"id": str, "mirgenedb_mature_id": str},
        keep_default_na=False,
        na_values=[],
    )
    cand["source_order"] = np.arange(len(cand), dtype=np.int64)
    cand["row_idx"] = cand["id"].map(id_to_row)
    has_profile = (
        pd.to_numeric(cand["has_phact_profile"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.int8)
        == 1
    )
    mature = cand["mirgenedb_mature_id"].astype(str).str.strip()
    valid_mature = ~mature.str.upper().isin(_MISSING_MATURE)
    valid = cand["row_idx"].notna().to_numpy() & has_profile & valid_mature.to_numpy()
    total = int(valid.sum())
    if not total:
        return pd.DataFrame(columns=columns), {
            "candidate_file_present": True,
            "profile_bearing_rows": 0,
            "retained_rows": 0,
            "truncated_rows": 0,
        }

    out = pd.DataFrame(
        {
            "row_idx": cand.loc[valid, "row_idx"].to_numpy(dtype=np.int64),
            "candidate_index": pd.to_numeric(
                cand.loc[valid, "candidate_index"], errors="coerce"
            )
            .fillna(10**9)
            .to_numpy(dtype=np.int64),
            "source_order": cand.loc[valid, "source_order"].to_numpy(dtype=np.int64),
            "mirgenedb_mature_id": mature.loc[valid].to_numpy(dtype=object),
        }
    )
    out.sort_values(
        ["row_idx", "candidate_index", "source_order"],
        kind="mergesort",
        inplace=True,
    )
    out["slot"] = out.groupby("row_idx", sort=False).cumcount().astype(np.int64)
    retained = out.loc[out["slot"] < MAX_CANDIDATE_PROFILES, columns].reset_index(drop=True)
    return retained, {
        "candidate_file_present": True,
        "profile_bearing_rows": total,
        "retained_rows": int(len(retained)),
        "truncated_rows": int(total - len(retained)),
    }


def build_candidate_phact(
    split_root: Path,
    sample_ids: Sequence[str],
    preprocessor: Dict[str, object],
    inherited_slot0: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Build ordered candidate tensors with frozen training-only preprocessing.

    Returns
    -------
    candidate_phact:
        Float32 ``[N, 3, 5, 28]``. Score channels 0:4 use the inherited
        train-fitted standardizers; channel 4 is the position mask.
    candidate_profile_mask:
        Float32 ``[N, 3]``.  It is one only when that slot has a joined,
        nonempty profile.  Every absent slot is represented by exact zeros.
    metadata:
        Label-free audit counts.  Candidate identities are never exposed as
        model features.
    """
    split_root = resolve_split_root(split_root)
    n = len(sample_ids)
    ordered, ordering_summary = _ordered_profile_candidates(split_root, sample_ids)

    frozen_cols = list(preprocessor["mirna_phact_columns"])
    if len(frozen_cols) != inherited.MIRNA_PHACT_SCORE_CHANNELS:
        raise ValueError(
            "Frozen miRNA-PHACT schema has "
            f"{len(frozen_cols)} score channels; expected {inherited.MIRNA_PHACT_SCORE_CHANNELS}"
        )
    observed_cols = inherited.discover_mirna_phact_columns(split_root)
    if observed_cols and observed_cols != frozen_cols:
        raise ValueError("miRNA-PHACT columns/order differ from the frozen training schema")

    raw_profiles, raw_masks, profile_summary = inherited.load_mirna_raw_profiles(
        split_root, frozen_cols
    )
    standardized = inherited.standardize_mirna_profiles(
        raw_profiles,
        raw_masks,
        frozen_cols,
        preprocessor["mirna_phact_scalers"],
    )

    tensor = np.zeros(
        (n, MAX_CANDIDATE_PROFILES, CANDIDATE_CHANNELS, CANDIDATE_LENGTH),
        dtype=np.float32,
    )
    profile_mask = np.zeros((n, MAX_CANDIDATE_PROFILES), dtype=np.float32)
    joined_rows = 0
    for row in ordered.itertuples(index=False):
        profile = standardized.get(str(row.mirgenedb_mature_id))
        if profile is None:
            continue
        ridx, slot = int(row.row_idx), int(row.slot)
        adjusted = inherited._adjust_mirna_tensor_channels(profile[None, ...])[0]
        if adjusted[inherited.MIRNA_PHACT_SCORE_CHANNELS].any():
            tensor[ridx, slot] = adjusted
            profile_mask[ridx, slot] = 1.0
            joined_rows += 1

    # Make the legacy selected profile an exact compatibility anchor.  The
    # ordering rule is the same; assignment from the already-built tensor also
    # guarantees bitwise slot-0 parity if NumPy internals ever change.
    if inherited_slot0 is not None:
        slot0 = np.asarray(inherited_slot0)
        expected = (n, CANDIDATE_CHANNELS, CANDIDATE_LENGTH)
        if slot0.shape != expected:
            raise ValueError(f"inherited_slot0 shape {slot0.shape}, expected {expected}")
        tensor[:, 0] = slot0.astype(np.float32, copy=False)
        profile_mask[:, 0] = (
            slot0[:, inherited.MIRNA_PHACT_SCORE_CHANNELS, :].sum(axis=1) > 0
        ).astype(np.float32)

    if not np.isfinite(tensor).all() or not np.isfinite(profile_mask).all():
        raise ValueError("non-finite candidate representation")
    absent = profile_mask == 0
    if np.any(tensor[absent] != 0):
        raise ValueError("absent candidate slots must be exact zero")
    pos_mask = tensor[:, :, inherited.MIRNA_PHACT_SCORE_CHANNELS, :]
    scores = tensor[:, :, : inherited.MIRNA_PHACT_SCORE_CHANNELS, :]
    if np.any(scores * (1.0 - pos_mask[:, :, None, :]) != 0):
        raise ValueError("positions outside a candidate profile must be exact zero")

    metadata = {
        **ordering_summary,
        "joined_retained_rows": int(joined_rows),
        "samples": int(n),
        "samples_with_profile": int((profile_mask.sum(axis=1) > 0).sum()),
        "samples_with_extra_profile": int((profile_mask[:, 1:].sum(axis=1) > 0).sum()),
        "slot_counts": [int(profile_mask[:, i].sum()) for i in range(MAX_CANDIDATE_PROFILES)],
        "profile_table": profile_summary,
    }
    return tensor, profile_mask, metadata


def transform_candidate_representation(
    input_or_split: Path,
    preprocessor: Dict[str, object],
    tokenizer=None,
    foundation_cache_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Transform an input folder without labels and append candidate profiles."""
    split_root = resolve_split_root(input_or_split)
    arrays = inherited.transform_representation(split_root, preprocessor, tokenizer=tokenizer)
    candidate, mask, _ = build_candidate_phact(
        split_root,
        arrays["ids"].tolist(),
        preprocessor,
        inherited_slot0=arrays["mirna_phact_tensor"],
    )
    arrays["candidate_phact"] = candidate
    arrays["candidate_profile_mask"] = mask
    if foundation_cache_path is not None:
        arrays = attach_foundation_layers(arrays, Path(foundation_cache_path))
    return arrays


def candidate_representation_shapes() -> Dict[str, object]:
    shapes = dict(inherited.representation_shapes())
    shapes.update(
        {
            "candidate_phact": [
                MAX_CANDIDATE_PROFILES,
                CANDIDATE_CHANNELS,
                CANDIDATE_LENGTH,
            ],
            "candidate_profile_mask": [MAX_CANDIDATE_PROFILES],
            "foundation_layer_pair_vectors": [4, 1920],
        }
    )
    return shapes


__all__ = [
    "MAX_CANDIDATE_PROFILES",
    "build_candidate_phact",
    "candidate_representation_shapes",
    "materialize_pairwise_batch",
    "resolve_split_root",
    "transform_candidate_representation",
]
