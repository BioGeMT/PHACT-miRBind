#!/usr/bin/env python3
"""Build deterministic, label-free RiNALMo hidden-layer caches.

The cache stores stable unique-sequence pools and row-to-unique maps.  It never
opens labels.csv.  multimolecule is imported before Transformers registration.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

# Registration order is required by the installed multimolecule package.
import multimolecule  # noqa: F401
from transformers import AutoModel, AutoTokenizer

from data_representation import sanitize_rna_for_rinalmo, tokenize_rinalmo_sequences

SELECTED_LAYER_INDICES: Tuple[int, ...] = (3, 6, 9, 12)
HIDDEN_SIZE = 480
EXPECTED_BLOCKS = 12


def stable_unique(values: Sequence[str]) -> Tuple[List[str], np.ndarray]:
    """Stable first-occurrence unique values and an int32 inverse map."""
    seen: Dict[str, int] = {}
    unique: List[str] = []
    inverse = np.empty(len(values), dtype=np.int32)
    for i, value in enumerate(values):
        idx = seen.get(value)
        if idx is None:
            idx = len(unique)
            seen[value] = idx
            unique.append(value)
        inverse[i] = idx
    return unique, inverse


def resolve_input_dir(path: Path) -> Path:
    path = Path(path)
    if path.name == "input" and (path / "samples.tsv").is_file():
        return path
    if (path / "input" / "samples.tsv").is_file():
        return path / "input"
    if (path / "samples.tsv").is_file():
        return path
    raise FileNotFoundError(f"Could not locate samples.tsv under {path}")


def load_samples_input_only(path: Path) -> pd.DataFrame:
    input_dir = resolve_input_dir(path)
    frame = pd.read_csv(
        input_dir / "samples.tsv",
        sep="\t",
        usecols=["id", "gene", "noncodingRNA"],
        dtype=str,
        keep_default_na=False,
        na_values=[],
    )
    if frame["id"].duplicated().any():
        raise ValueError("samples.tsv IDs must be unique")
    return frame


def load_encoder(model_dir: Path, device: torch.device):
    """Load only the pretrained encoder; remove any irrelevant pooler."""
    model_dir = str(Path(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=True
    )
    kwargs = dict(
        local_files_only=True,
        trust_remote_code=True,
        output_hidden_states=True,
        add_pooling_layer=False,
    )
    try:
        encoder = AutoModel.from_pretrained(model_dir, **kwargs)
    except TypeError:
        kwargs.pop("add_pooling_layer", None)
        encoder = AutoModel.from_pretrained(model_dir, **kwargs)
    if hasattr(encoder, "pooler"):
        encoder.pooler = None
    if getattr(encoder, "pooler", None) is not None:
        raise RuntimeError("RiNALMo pooler was not removed")
    blocks = getattr(getattr(encoder, "encoder", None), "layer", None)
    if blocks is None or len(blocks) != EXPECTED_BLOCKS:
        raise ValueError(f"Expected {EXPECTED_BLOCKS} RiNALMo blocks")
    if int(getattr(encoder.config, "hidden_size", -1)) != HIDDEN_SIZE:
        raise ValueError("Expected RiNALMo hidden width 480")
    encoder.requires_grad_(False)
    encoder.eval().to(device)
    return tokenizer, encoder


def _first_tensor(output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"Cannot obtain hidden tensor from {type(output)!r}")


def biological_mean_pool(hidden: torch.Tensor, biological_mask: torch.Tensor) -> torch.Tensor:
    mask = biological_mask.to(dtype=hidden.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / denom


def extract_selected_pools(
    sequences: Sequence[str],
    tokenizer,
    encoder,
    fixed_length: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool outputs after blocks 3, 6, 9, and normalized block 12.

    The installed multimolecule implementation does not expose intermediate
    hidden states even when output_hidden_states=True, so read-only hooks are
    used for blocks 3/6/9.  The returned last_hidden_state is used for state 12
    because final LayerNorm is applied after the final block.
    """
    n = len(sequences)
    pooled = np.empty((n, len(SELECTED_LAYER_INDICES), HIDDEN_SIZE), dtype=np.float32)
    token_counts = np.empty(n, dtype=np.int16)
    validity = np.empty(n, dtype=np.uint8)
    blocks = encoder.encoder.layer
    captured: Dict[int, torch.Tensor] = {}
    handles = []
    for state_idx in SELECTED_LAYER_INDICES[:-1]:
        def hook(_module, _inputs, output, state_idx=state_idx):
            captured[state_idx] = _first_tensor(output)
        handles.append(blocks[state_idx - 1].register_forward_hook(hook))
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    try:
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            tok = tokenize_rinalmo_sequences(sequences[start:stop], tokenizer, fixed_length)
            ids = torch.from_numpy(tok["input_ids"]).to(device)
            attention = torch.from_numpy(tok["attention_mask"]).to(device)
            biological = torch.from_numpy(tok["biological_mask"]).to(device)
            captured.clear()
            amp = torch.autocast(device_type="cuda", dtype=amp_dtype) if device.type == "cuda" else nullcontext()
            with torch.inference_mode(), amp:
                result = encoder(
                    input_ids=ids,
                    attention_mask=attention,
                    output_hidden_states=True,
                    return_dict=True,
                )
                final_hidden = result.last_hidden_state
                state_tensors = [captured[i] for i in SELECTED_LAYER_INDICES[:-1]] + [final_hidden]
                batch_pool = torch.stack(
                    [biological_mean_pool(x, biological) for x in state_tensors], dim=1
                ).float()
            arr = batch_pool.cpu().numpy()
            pooled[start:stop] = arr
            counts = biological.sum(dim=1).cpu().numpy().astype(np.int16)
            token_counts[start:stop] = counts
            validity[start:stop] = np.isfinite(arr).all(axis=(1, 2)).astype(np.uint8)
            if not np.all(validity[start:stop] == 1):
                raise FloatingPointError("Non-finite RiNALMo pooled state")
            expected = np.asarray(
                [min(len(sanitize_rna_for_rinalmo(x)), fixed_length - 2) for x in sequences[start:stop]],
                dtype=np.int16,
            )
            if not np.array_equal(counts, expected):
                raise AssertionError("Biological-token count mismatch")
    finally:
        for handle in handles:
            handle.remove()
    return pooled, token_counts, validity


def build_cache(
    input_path: Path,
    model_dir: Path,
    output: Path,
    batch_size: int = 128,
    device_name: str = "auto",
) -> dict:
    random.seed(20260729)
    np.random.seed(20260729)
    torch.manual_seed(20260729)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260729)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    samples = load_samples_input_only(input_path)
    ids = samples["id"].astype(str).tolist()
    mirna_clean = [sanitize_rna_for_rinalmo(x) for x in samples["noncodingRNA"].tolist()]
    target_clean = [sanitize_rna_for_rinalmo(x) for x in samples["gene"].tolist()]
    unique_mirna, mirna_inverse = stable_unique(mirna_clean)
    unique_target, target_inverse = stable_unique(target_clean)
    tokenizer, encoder = load_encoder(model_dir, device)
    mirna_pooled, mirna_counts, mirna_valid = extract_selected_pools(
        unique_mirna, tokenizer, encoder, 30, batch_size, device
    )
    target_pooled, target_counts, target_valid = extract_selected_pools(
        unique_target, tokenizer, encoder, 52, batch_size, device
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        ids=np.asarray(ids, dtype=str),
        layer_indices=np.asarray(SELECTED_LAYER_INDICES, dtype=np.int16),
        mirna_unique_sequences=np.asarray(unique_mirna, dtype=str),
        target_unique_sequences=np.asarray(unique_target, dtype=str),
        mirna_pooled=mirna_pooled,
        target_pooled=target_pooled,
        mirna_row_to_unique=mirna_inverse,
        target_row_to_unique=target_inverse,
        mirna_valid=mirna_valid,
        target_valid=target_valid,
        mirna_token_counts=mirna_counts,
        target_token_counts=target_counts,
    )
    os.replace(temporary, output)
    metadata = {
        "format_version": 1,
        "label_free": True,
        "row_count": len(ids),
        "unique_mirna_count": len(unique_mirna),
        "unique_target_count": len(unique_target),
        "selected_hidden_state_indices": list(SELECTED_LAYER_INDICES),
        "selected_state_meaning": ["after_block_3", "after_block_6", "after_block_9", "after_block_12_final_layernorm"],
        "hidden_size": HIDDEN_SIZE,
        "pair_vector_order": ["mirna", "target", "absolute_difference", "elementwise_product"],
        "sequence_cleaning": "uppercase; U_to_T; non_ACGT_to_N; empty_to_N; T_to_U_for_RiNALMo",
        "pooling": "mean over attention=1 and special-token=0 biological tokens",
        "pooler_used": False,
        "lm_or_task_head_used": False,
        "dtype": "float32",
        "device": str(device),
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path, help="input/ or split directory")
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="auto")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = build_cache(args.input, args.model_dir, args.output, args.batch_size, args.device)
    print(json.dumps(report, sort_keys=True))
