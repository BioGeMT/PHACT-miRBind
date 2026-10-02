#!/usr/bin/env python3
"""Leakage-safe iteration-13 fusion representation with online RiNALMo tokens.

All fitted statistics and vocabularies are learned from the supplied training
split only. Inference transformation never reads labels.
"""
from __future__ import annotations
import csv, gc, json, math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
SEED = 20260716

BASES = ["A", "C", "G", "T"]

BASE_TO_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}

N_CODE = 4

MIRNA_LENGTH = 28

TARGET_LENGTH = 50

MIRBIND2_PAIR_BASE_ORDER = ["A", "T", "C", "G"]

MIRBIND2_NUCLEOTIDE_PAIRS = [(a, b) for a in MIRBIND2_PAIR_BASE_ORDER for b in MIRBIND2_PAIR_BASE_ORDER]

MIRBIND2_PAIR_STATE_NAMES = [a + b for a, b in MIRBIND2_NUCLEOTIDE_PAIRS] + ["NN_PADDING_PAIR", "UNKNOWN_OR_N"]

MIRBIND2_NUM_PAIRS = 17

MIRBIND2_ONEHOT_CHANNELS = 18

MIRBIND2_CHECKPOINT_FILENAME = "pairwise_onehot_model_20260105_200141.pt"

CONSERVATION_CHANNELS = 4

TARGET_PHACT_CHANNELS_N = 8

TARGET_BRANCH_CHANNELS = CONSERVATION_CHANNELS + TARGET_PHACT_CHANNELS_N

MIRNA_PHACT_SCORE_CHANNELS = 4

MIRNA_PHACT_CHANNELS = MIRNA_PHACT_SCORE_CHANNELS + 1

TARGET_PHACT_SCORE_COLS = ["score_A", "score_C", "score_G", "score_T"]

TARGET_PHACT_CHANNEL_ORDER = [
    "standardized_score_A",
    "standardized_score_C",
    "standardized_score_G",
    "standardized_score_T",
    "standardized_actual_score",
    "standardized_nonactual_mean",
    "standardized_actual_minus_nonactual_mean",
    "target_phact_finite_mask",
]

BASE_SEQ_FEATURE_NAMES = [
    "gene_len",
    "mirna_len",
    "gene_gc_frac",
    "mirna_gc_frac",
    "seed_max_contiguous_rc_match_len_2_8",
    "seed6_rc_match_count_2_7",
    "seed7_rc_match_count_2_8",
]

EXACT_RC_COUNT_SEGMENTS = [
    ("mirna_1_7", 0, 7),
    ("mirna_1_8", 0, 8),
    ("mirna_2_9", 1, 9),
    ("mirna_2_10", 1, 10),
]

EXACT_RC_COUNT_FEATURE_NAMES = [f"exact_rc_count_{name}" for name, _, _ in EXACT_RC_COUNT_SEGMENTS]

ALIGNMENT_SEGMENTS = [
    ("seed_2_7", 1, 7),
    ("seed_2_8", 1, 8),
    ("seed_1_8", 0, 8),
    ("extended_2_13", 1, 13),
    ("central_9_16", 8, 16),
    ("supplementary_13_22", 12, 22),
]

ALIGNMENT_FEATURE_SUFFIXES = [
    "segment_valid",
    "max_wc_frac",
    "max_wobble_frac",
    "max_pair_frac",
    "max_score_frac",
    "max_contiguous_wc_frac",
    "max_contiguous_pair_frac",
    "top3_score_frac_mean",
    "windows_score_ge_0p75_frac",
    "windows_score_ge_0p85_frac",
    "best_start_norm",
    "best_center_distance_frac",
]

RICH_ALIGNMENT_FEATURE_NAMES = [f"{seg}_{suffix}" for seg, _, _ in ALIGNMENT_SEGMENTS for suffix in ALIGNMENT_FEATURE_SUFFIXES]

NEW_ALIGNMENT_FEATURE_NAMES = EXACT_RC_COUNT_FEATURE_NAMES + RICH_ALIGNMENT_FEATURE_NAMES

SEQ_FEATURE_NAMES = BASE_SEQ_FEATURE_NAMES + NEW_ALIGNMENT_FEATURE_NAMES

SUMMARY_STATS = ["mean", "std", "min", "max", "median", "q25", "q75", "finite_fraction", "empty_indicator"]

CONSERVATION_SOURCES = ["gene_phyloP", "gene_phastCons"]

PHACT_AUX_FEATURES = ["candidate_count", "has_selected_mirna_phact", "has_target_phact_any_position"]

AUX_NUMERIC_FEATURE_NAMES = (
    BASE_SEQ_FEATURE_NAMES
    + [f"{src}_{stat}" for src in CONSERVATION_SOURCES for stat in SUMMARY_STATS]
    + PHACT_AUX_FEATURES
    + NEW_ALIGNMENT_FEATURE_NAMES
)

RC_PAIRWISE_BASE_ORDER = ["A", "C", "G", "T"]

RC_PAIRWISE_CHANNELS = 20

RC_PAIRWISE_CHANNEL_NAMES = (
    [f"pair_{a}{b}" for a in RC_PAIRWISE_BASE_ORDER for b in RC_PAIRWISE_BASE_ORDER]
    + ["watson_crick_complement", "GT_or_TG_wobble", "valid_acgt_pair_mask", "mirna_seed_2_8_mask"]
)

CATEGORY_MIN_COUNT = 100

MISSING_TOKEN = "__MISSING__"

OTHER_TOKEN = "__OTHER__"

PH_TARGET_CHUNKSIZE = 2_000_000

TRAIN_METRIC_SUBSET_SIZE = 250_000

VALID_METRIC_EPS = 1e-7

WEIGHT_DECAY = 1e-4

GRAD_CLIP_NORM = 5.0

EARLY_STOPPING_PATIENCE = 3

FINE_TUNE_EPOCHS = 8

BATCH_SIZE_CANDIDATES = [768, 512]

RINALMO_MODEL_ID = "multimolecule/rinalmo-micro"

RINALMO_HIDDEN_SIZE = 480

RINALMO_CONCAT_DIM = 4 * RINALMO_HIDDEN_SIZE

RINALMO_BRANCH_HIDDEN_DIM = 256

RINALMO_FEATURE_DIM = 96

RINALMO_DROPOUT = 0.20

RINALMO_EMBED_BATCH_CANDIDATES = [768, 512, 256]

RINALMO_CACHE_CHUNK_SIZE = 50_000

CANDIDATES = [
    {
        "name": "fm_a",
        "warmup_lr": 5e-4,
        "body_lr": 1e-5,
        "classifier_lr": 5e-5,
        "new_lr": 5e-4,
    },
    {
        "name": "fm_b",
        "warmup_lr": 3e-4,
        "body_lr": 3e-6,
        "classifier_lr": 2e-5,
        "new_lr": 3e-4,
    },
]

@dataclass
class RepresentationConfig:
    target_length: int = TARGET_LENGTH
    mirna_length: int = MIRNA_LENGTH
    sequence_branch: str = "supplementary_miRBind2_pairwise_onehot_plus_reverse_complement_pairwise_cnn"
    target_orientation: str = "branch_A_gene_as_supplied_no_reverse_complement;branch_B_right_padded_target_then_reverse_complemented"
    sequence_cleaning: str = "uppercase_DNA_U_to_T_invalid_to_N"
    padding_truncation: str = "right_pad_with_N_right_truncate_prefix"
    pair_axis_order: str = "miRBind2_miRNA_position_by_target_position_channel_last;RC_channel_first_20_by_28_by_50"
    rc_pairwise_channel_names: Tuple[str, ...] = tuple(RC_PAIRWISE_CHANNEL_NAMES)
    mirbind2_pair_state_names: Tuple[str, ...] = tuple(MIRBIND2_PAIR_STATE_NAMES)
    mirbind2_num_pairs: int = MIRBIND2_NUM_PAIRS
    mirbind2_onehot_channels: int = MIRBIND2_ONEHOT_CHANNELS
    conservation_sources: Tuple[str, ...] = tuple(CONSERVATION_SOURCES)
    conservation_positional_channels: Tuple[str, ...] = (
        "standardized_gene_phyloP",
        "standardized_gene_phastCons",
        "gene_phyloP_finite_mask",
        "gene_phastCons_finite_mask",
    )
    target_phact_channels: Tuple[str, ...] = tuple(TARGET_PHACT_CHANNEL_ORDER)
    mirna_phact_tensor: str = "4_consensus_param_1_base_channels_plus_mirna_phact_finite_mask"
    mirna_candidate_selection: str = "lowest_candidate_index_with_has_phact_profile_1_else_missing"
    auxiliary_numeric_features: Tuple[str, ...] = tuple(AUX_NUMERIC_FEATURE_NAMES)
    categorical_features: Tuple[str, ...] = ("feature", "dominant_region")
    rinalmo_model_id: str = RINALMO_MODEL_ID
    rinalmo_hidden_size: int = RINALMO_HIDDEN_SIZE
    rinalmo_embedding_dim_per_sequence: int = RINALMO_HIDDEN_SIZE
    rinalmo_concat_dim: int = RINALMO_CONCAT_DIM
    rinalmo_vector_construction: str = "concat_mirna_target_absdiff_product"
    rinalmo_sequence_cleaning: str = "uppercase_DNA_clean_then_T_to_U_keep_N_full_unpadded_mirna_target_len_50"
    rinalmo_pooling: str = "mean_last_hidden_state_over_attention_mask_1_and_special_tokens_mask_0_fallback_drop_saved_special_token_ids"
    rinalmo_frozen: bool = True
    categorical_min_count: int = CATEGORY_MIN_COUNT
    missing_token: str = MISSING_TOKEN
    other_token: str = OTHER_TOKEN

def make_encoding_table() -> np.ndarray:
    table = np.full(256, N_CODE, dtype=np.uint8)
    for b, c in BASE_TO_CODE.items():
        table[ord(b)] = c
        table[ord(b.lower())] = c
    table[ord("U")] = BASE_TO_CODE["T"]
    table[ord("u")] = BASE_TO_CODE["T"]
    return table

ENCODING_TABLE = make_encoding_table()

COMPLEMENT_TRANS = str.maketrans("ACGTN", "TGCAN")

def sanitize_dna(seq: object) -> str:
    if not isinstance(seq, str):
        return ""
    seq = seq.upper().replace("U", "T")
    return "".join(ch if ch in "ACGT" else "N" for ch in seq)

def reverse_complement_str(seq: str) -> str:
    return sanitize_dna(seq).translate(COMPLEMENT_TRANS)[::-1]

def encode_sequences(seqs: Iterable[str], length: int) -> np.ndarray:
    seqs_list = list(seqs)
    out = np.full((len(seqs_list), length), N_CODE, dtype=np.uint8)
    table = ENCODING_TABLE
    for i, seq in enumerate(seqs_list):
        if not isinstance(seq, str) or not seq:
            continue
        raw = seq.encode("ascii", "ignore")[:length]
        if not raw:
            continue
        codes = table[np.frombuffer(raw, dtype=np.uint8)]
        out[i, : codes.shape[0]] = codes
    return out

def load_labels(split_dir: Path) -> pd.DataFrame:
    labels_path = split_dir / "labels.csv"
    labels = pd.read_csv(labels_path, dtype={"id": str})
    label_col = "numeric_label" if "numeric_label" in labels.columns else "label"
    if label_col not in labels.columns:
        raise ValueError(f"No label or numeric_label column in {labels_path}")
    labels = labels[["id", label_col]].rename(columns={label_col: "label"})
    labels["label"] = labels["label"].astype(np.float32)
    return labels

def read_samples(split_dir: Path) -> pd.DataFrame:
    samples_path = split_dir / "input" / "samples.tsv"
    usecols = ["id", "gene", "noncodingRNA", "feature", "dominant_region", "gene_phyloP", "gene_phastCons"]
    samples = pd.read_csv(samples_path, sep="\t", usecols=usecols, dtype=str, keep_default_na=False, na_values=[])
    missing = [c for c in usecols if c not in samples.columns]
    if missing:
        raise ValueError(f"Missing required samples.tsv columns: {missing}")
    return samples

def load_split_frame(split_dir_str: str, require_labels: bool = True) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    split_dir = Path(split_dir_str)
    samples = read_samples(split_dir)
    labels_arr: Optional[np.ndarray] = None
    if require_labels:
        labels = load_labels(split_dir)
        if samples["id"].duplicated().any():
            raise ValueError("samples.tsv contains duplicate ids")
        if labels["id"].duplicated().any():
            raise ValueError("labels.csv contains duplicate ids")
        merged = samples[["id"]].merge(labels, on="id", how="left", sort=False)
        if merged["label"].isna().any():
            raise ValueError("Some samples.tsv IDs do not have labels")
        if len(merged) != len(samples):
            raise ValueError("Label join changed row count")
        labels_arr = merged["label"].to_numpy(dtype=np.float32)
    return samples, labels_arr

def count_overlapping(text: str, pattern: str) -> int:
    if not pattern:
        return 0
    count = 0
    start = 0
    while True:
        idx = text.find(pattern, start)
        if idx < 0:
            return count
        count += 1
        start = idx + 1

def max_seed_common_substring(seed_rc: str, target: str) -> int:
    m = len(seed_rc)
    if m == 0 or not target:
        return 0
    max_len = min(7, m, len(target))
    for length in range(max_len, 0, -1):
        for start in range(m - length + 1):
            if seed_rc[start : start + length] in target:
                return length
    return 0

def longest_true_run(flags: Sequence[bool]) -> int:
    best = 0
    cur = 0
    for flag in flags:
        if flag:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best

def rich_alignment_features_for_segment(gene: str, mirna: str, start: int, end: int) -> List[float]:
    """Antiparallel deterministic target-window alignment features.

    start/end are zero-based miRNA slice coordinates. Target windows remain in
    original target orientation; target offset k pairs with miRNA base end-1-k.
    This is copied from iteration 6 and intentionally contains no fitted state.
    """
    length = end - start
    if length <= 0 or len(mirna) < end or len(gene) < length:
        return [0.0] * len(ALIGNMENT_FEATURE_SUFFIXES)
    segment = mirna[start:end]
    n_windows = len(gene) - length + 1
    if n_windows <= 0:
        return [0.0] * len(ALIGNMENT_FEATURE_SUFFIXES)
    best_wc = best_wobble = best_pair = best_score = 0.0
    best_contig_wc = best_contig_pair = 0
    best_start = 0
    score_fracs: List[float] = []
    ge075 = 0
    ge085 = 0
    for s in range(n_windows):
        window = gene[s : s + length]
        wc_flags: List[bool] = []
        wobble_flags: List[bool] = []
        pair_flags: List[bool] = []
        wc_count = 0
        wobble_count = 0
        for k, target_nt in enumerate(window):
            mirna_nt = segment[length - 1 - k]
            wc = (
                (mirna_nt == "A" and target_nt == "T")
                or (mirna_nt == "T" and target_nt == "A")
                or (mirna_nt == "C" and target_nt == "G")
                or (mirna_nt == "G" and target_nt == "C")
            )
            wobble = (mirna_nt == "G" and target_nt == "T") or (mirna_nt == "T" and target_nt == "G")
            wc_flags.append(wc)
            wobble_flags.append(wobble)
            pair_flags.append(wc or wobble)
            wc_count += int(wc)
            wobble_count += int(wobble)
        pair_count = wc_count + wobble_count
        score = float(wc_count) + 0.5 * float(wobble_count)
        score_frac = score / float(length)
        score_fracs.append(score_frac)
        if score_frac >= 0.75:
            ge075 += 1
        if score_frac >= 0.85:
            ge085 += 1
        if score > best_score:
            best_score = score
            best_start = s
        best_wc = max(best_wc, float(wc_count))
        best_wobble = max(best_wobble, float(wobble_count))
        best_pair = max(best_pair, float(pair_count))
        best_contig_wc = max(best_contig_wc, longest_true_run(wc_flags))
        best_contig_pair = max(best_contig_pair, longest_true_run(pair_flags))
    top_scores = sorted(score_fracs, reverse=True)[:3]
    best_center = best_start + (length - 1) / 2.0
    target_center = (len(gene) - 1) / 2.0 if gene else 0.0
    return [
        1.0,
        best_wc / float(length),
        best_wobble / float(length),
        best_pair / float(length),
        best_score / float(length),
        float(best_contig_wc) / float(length),
        float(best_contig_pair) / float(length),
        float(np.mean(top_scores)) if top_scores else 0.0,
        float(ge075) / float(n_windows),
        float(ge085) / float(n_windows),
        float(best_start) / 49.0,
        abs(best_center - target_center) / 25.0,
    ]

def compute_sequence_features(genes: Sequence[str], mirnas: Sequence[str]) -> np.ndarray:
    rows: List[List[float]] = []
    append = rows.append
    for gene_raw, mirna_raw in zip(genes, mirnas):
        gene = sanitize_dna(gene_raw)
        mirna = sanitize_dna(mirna_raw)
        gene_len = len(gene)
        mirna_len = len(mirna)
        gene_gc = (gene.count("G") + gene.count("C")) / gene_len if gene_len else 0.0
        mirna_gc = (mirna.count("G") + mirna.count("C")) / mirna_len if mirna_len else 0.0
        seed_2_8 = mirna[1:8] if mirna_len >= 2 else ""
        seed_2_7 = mirna[1:7] if mirna_len >= 2 else ""
        seed7_rc = reverse_complement_str(seed_2_8) if seed_2_8 else ""
        seed6_rc = reverse_complement_str(seed_2_7) if seed_2_7 else ""
        max_match = max_seed_common_substring(seed7_rc, gene)
        seed6_count = count_overlapping(gene, seed6_rc) if len(seed6_rc) == 6 and "N" not in seed6_rc else 0
        seed7_count = count_overlapping(gene, seed7_rc) if len(seed7_rc) == 7 and "N" not in seed7_rc else 0
        features: List[float] = [
            float(gene_len),
            float(mirna_len),
            float(gene_gc),
            float(mirna_gc),
            float(max_match),
            float(seed6_count),
            float(seed7_count),
        ]
        for _, start, end in EXACT_RC_COUNT_SEGMENTS:
            if mirna_len >= end:
                segment = mirna[start:end]
                segment_rc = reverse_complement_str(segment)
                cnt = count_overlapping(gene, segment_rc) if len(segment_rc) == (end - start) and "N" not in segment_rc else 0
            else:
                cnt = 0
            features.append(float(cnt))
        for _, start, end in ALIGNMENT_SEGMENTS:
            features.extend(rich_alignment_features_for_segment(gene, mirna, start, end))
        append(features)
    out = np.asarray(rows, dtype=np.float32)
    if out.shape[1] != len(SEQ_FEATURE_NAMES):
        raise RuntimeError(f"Sequence feature dimension mismatch: {out.shape[1]} vs {len(SEQ_FEATURE_NAMES)}")
    return out

def parse_numeric_vector(value: object) -> np.ndarray:
    if not isinstance(value, str):
        return np.empty(0, dtype=np.float32)
    text = value.strip()
    if not text or text.upper() == "NA":
        return np.empty(0, dtype=np.float32)
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return np.empty(0, dtype=np.float32)
    arr = np.fromstring(text, sep=",", dtype=np.float32)
    if arr.size == 0 and text.strip():
        vals: List[float] = []
        for part in text.split(","):
            try:
                vals.append(float(part))
            except Exception:
                vals.append(float("nan"))
        arr = np.asarray(vals, dtype=np.float32)
    return arr

def fit_conservation_positional_scalers(samples: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    scalers: Dict[str, Dict[str, float]] = {}
    for src in CONSERVATION_SOURCES:
        total = 0
        s = 0.0
        ss = 0.0
        for value in samples[src].tolist():
            arr = parse_numeric_vector(value)
            if arr.size:
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    total += int(finite.size)
                    vals = finite.astype(np.float64, copy=False)
                    s += float(vals.sum())
                    ss += float(np.square(vals).sum())
        mean = s / total if total else 0.0
        var = max(ss / total - mean * mean, 0.0) if total else 0.0
        std = math.sqrt(var) if var > 1e-12 else 1.0
        scalers[src] = {"mean": float(mean), "std": float(std), "finite_count": int(total)}
    return scalers

def summarize_vector(arr: np.ndarray) -> List[float]:
    if arr.size == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    vals = finite.astype(np.float64, copy=False)
    return [
        float(vals.mean()),
        float(vals.std(ddof=0)),
        float(vals.min()),
        float(vals.max()),
        float(np.median(vals)),
        float(np.quantile(vals, 0.25)),
        float(np.quantile(vals, 0.75)),
        float(finite.size / max(1, arr.size)),
        0.0,
    ]

def conservation_summaries_and_tensor(samples: pd.DataFrame, positional_scalers: Dict[str, Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(samples)
    summaries = np.zeros((n, len(CONSERVATION_SOURCES) * len(SUMMARY_STATS)), dtype=np.float32)
    tensor = np.zeros((n, CONSERVATION_CHANNELS, TARGET_LENGTH), dtype=np.float32)
    for i, (_, row) in enumerate(samples.iterrows()):
        offset = 0
        for src_idx, src in enumerate(CONSERVATION_SOURCES):
            arr = parse_numeric_vector(row[src])
            summaries[i, offset : offset + len(SUMMARY_STATS)] = np.asarray(summarize_vector(arr), dtype=np.float32)
            offset += len(SUMMARY_STATS)
            if arr.size:
                limit = min(TARGET_LENGTH, arr.size)
                vals = arr[:limit].astype(np.float32, copy=False)
                finite = np.isfinite(vals)
                scaler = positional_scalers[src]
                standardized = np.zeros(TARGET_LENGTH, dtype=np.float32)
                mask = np.zeros(TARGET_LENGTH, dtype=np.float32)
                if finite.any():
                    standardized[:limit][finite] = (vals[finite] - scaler["mean"]) / scaler["std"]
                    mask[:limit][finite] = 1.0
                tensor[i, src_idx, :] = standardized
                tensor[i, src_idx + 2, :] = mask
    return summaries, tensor

def fit_standard_scaler(x: np.ndarray) -> Dict[str, List[float]]:
    x64 = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(x64, axis=0)
    std = np.nanstd(x64, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    return {"mean": mean.astype(float).tolist(), "scale": std.astype(float).tolist()}

def apply_standard_scaler(x: np.ndarray, scaler: Dict[str, List[float]]) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    out = (np.asarray(x, dtype=np.float32) - mean) / scale
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32, copy=False)

def normalize_category(v: object) -> str:
    if not isinstance(v, str):
        return MISSING_TOKEN
    text = v.strip()
    if not text or text.upper() in {"NA", "NAN", "NONE", "NULL"}:
        return MISSING_TOKEN
    return text

def fit_category_vocab(values: Sequence[str], min_count: int = CATEGORY_MIN_COUNT) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for v in values:
        key = normalize_category(v)
        counts[key] = counts.get(key, 0) + 1
    vocab = {MISSING_TOKEN: 0, OTHER_TOKEN: 1}
    for key, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if key in vocab:
            continue
        if count >= min_count:
            vocab[key] = len(vocab)
    return vocab

def encode_categories(values: Sequence[str], vocab: Dict[str, int]) -> np.ndarray:
    other = vocab[OTHER_TOKEN]
    return np.asarray([vocab.get(normalize_category(v), other) for v in values], dtype=np.int64)

def default_scalar_stats() -> Dict[str, float]:
    return {"mean": 0.0, "std": 1.0, "count": 0}

def stats_from_sums(total: int, s: float, ss: float) -> Dict[str, float]:
    mean = s / total if total else 0.0
    var = max(ss / total - mean * mean, 0.0) if total else 0.0
    std = math.sqrt(var) if var > 1e-12 else 1.0
    return {"mean": float(mean), "std": float(std), "count": int(total)}

def read_candidate_info(split_dir: Path, sample_ids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, Dict[str, object]]:
    n = len(sample_ids)
    candidate_count = np.zeros(n, dtype=np.float32)
    has_selected = np.zeros(n, dtype=np.float32)
    selected_mature = np.empty(n, dtype=object)
    selected_mature[:] = None
    path = split_dir / "input" / "sample_mirna_candidates.tsv"
    if not path.exists():
        return candidate_count, has_selected, selected_mature, pd.DataFrame(columns=["row_idx", "mirgenedb_mature_id", "candidate_index"]), {"file_present": False}
    id_to_idx = pd.Series(np.arange(n, dtype=np.int64), index=pd.Index(sample_ids, dtype="object"))
    cand = pd.read_csv(
        path,
        sep="\t",
        usecols=["id", "candidate_index", "candidate_count", "mirgenedb_mature_id", "has_phact_profile"],
        dtype={"id": str, "mirgenedb_mature_id": str},
        keep_default_na=False,
        na_values=[],
    )
    row_idx = cand["id"].map(id_to_idx)
    valid_id = row_idx.notna().to_numpy()
    if valid_id.any():
        idx_arr = row_idx[valid_id].to_numpy(dtype=np.int64)
        counts = pd.to_numeric(cand.loc[valid_id, "candidate_count"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        np.maximum.at(candidate_count, idx_arr, counts)
    has_profile = pd.to_numeric(cand["has_phact_profile"], errors="coerce").fillna(0).to_numpy(dtype=np.int8) == 1
    mature = cand["mirgenedb_mature_id"].astype(str)
    valid_mature = (~mature.str.upper().isin(["NA", "", "NAN"])).to_numpy()
    valid_sel = valid_id & has_profile & valid_mature
    selected_df = pd.DataFrame(columns=["row_idx", "mirgenedb_mature_id", "candidate_index"])
    if valid_sel.any():
        tmp = pd.DataFrame(
            {
                "row_idx": row_idx[valid_sel].to_numpy(dtype=np.int64),
                "mirgenedb_mature_id": mature[valid_sel].to_numpy(dtype=object),
                "candidate_index": pd.to_numeric(cand.loc[valid_sel, "candidate_index"], errors="coerce").fillna(10**9).to_numpy(dtype=np.int64),
            }
        )
        tmp = tmp.sort_values(["row_idx", "candidate_index"], kind="mergesort").drop_duplicates("row_idx", keep="first")
        selected_df = tmp.reset_index(drop=True)
        ridx = selected_df["row_idx"].to_numpy(dtype=np.int64)
        mids = selected_df["mirgenedb_mature_id"].to_numpy(dtype=object)
        selected_mature[ridx] = mids
        has_selected[ridx] = 1.0
    summary = {
        "file_present": True,
        "rows": int(len(cand)),
        "sample_count": int(n),
        "selected_profile_samples": int(has_selected.sum()),
        "candidate_count_mean": float(candidate_count.mean()) if n else 0.0,
    }
    return candidate_count, has_selected, selected_mature, selected_df, summary

def discover_mirna_phact_columns(split_dir: Path) -> List[str]:
    path = split_dir / "input" / "phact_mirna_positions.tsv"
    if not path.exists():
        return []
    header = pd.read_csv(path, sep="\t", nrows=0)
    cols = [c for c in header.columns if c.startswith("phact_") and c.rsplit("_", 1)[-1] in BASES]
    return sorted(cols)

def load_mirna_raw_profiles(split_dir: Path, phact_cols: Sequence[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, object]]:
    profiles: Dict[str, np.ndarray] = {}
    masks: Dict[str, np.ndarray] = {}
    path = split_dir / "input" / "phact_mirna_positions.tsv"
    if not path.exists() or not phact_cols:
        return profiles, masks, {"file_present": path.exists(), "profiles": 0, "rows": 0}
    usecols = ["mirgenedb_mature_id", "mirna_position_1based"] + list(phact_cols)
    df = pd.read_csv(path, sep="\t", usecols=usecols, dtype={"mirgenedb_mature_id": str}, keep_default_na=False, na_values=[])
    for mid, grp in df.groupby("mirgenedb_mature_id", sort=False):
        mat = np.zeros((len(phact_cols), MIRNA_LENGTH), dtype=np.float32)
        mask = np.zeros(MIRNA_LENGTH, dtype=np.float32)
        pos = pd.to_numeric(grp["mirna_position_1based"], errors="coerce").to_numpy(dtype=np.float64) - 1.0
        valid_pos = np.isfinite(pos) & (pos >= 0) & (pos < MIRNA_LENGTH)
        if valid_pos.any():
            pidx = pos[valid_pos].astype(np.int64)
            vals = grp.loc[valid_pos, list(phact_cols)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            finite_row = np.isfinite(vals).any(axis=1)
            if finite_row.any():
                mat[:, pidx[finite_row]] = np.where(np.isfinite(vals[finite_row]), vals[finite_row], 0.0).T
                mask[pidx[finite_row]] = 1.0
        profiles[str(mid)] = mat
        masks[str(mid)] = mask
    return profiles, masks, {"file_present": True, "profiles": int(len(profiles)), "rows": int(len(df))}

def fit_mirna_phact_scalers(
    selected_df: pd.DataFrame,
    profiles: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    phact_cols: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    sums = np.zeros(len(phact_cols), dtype=np.float64)
    sums_sq = np.zeros(len(phact_cols), dtype=np.float64)
    counts = np.zeros(len(phact_cols), dtype=np.int64)
    if len(selected_df) > 0:
        freq = selected_df["mirgenedb_mature_id"].value_counts()
        for mid, sample_count in freq.items():
            mid_s = str(mid)
            if mid_s not in profiles:
                continue
            mat = profiles[mid_s].astype(np.float64, copy=False)
            mask = masks[mid_s].astype(bool)
            if not mask.any():
                continue
            vals = mat[:, mask]
            finite = np.isfinite(vals)
            c = finite.sum(axis=1).astype(np.int64) * int(sample_count)
            sums += np.where(finite, vals, 0.0).sum(axis=1) * int(sample_count)
            sums_sq += np.where(finite, vals * vals, 0.0).sum(axis=1) * int(sample_count)
            counts += c
    scalers: Dict[str, Dict[str, float]] = {}
    for i, c in enumerate(phact_cols):
        scalers[c] = stats_from_sums(int(counts[i]), float(sums[i]), float(sums_sq[i]))
    return scalers

def standardize_mirna_profiles(
    profiles: Dict[str, np.ndarray], masks: Dict[str, np.ndarray], phact_cols: Sequence[str], scalers: Dict[str, Dict[str, float]]
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if not phact_cols:
        return out
    means = np.asarray([scalers[c]["mean"] for c in phact_cols], dtype=np.float32)[:, None]
    stds = np.asarray([scalers[c]["std"] for c in phact_cols], dtype=np.float32)[:, None]
    for mid, raw in profiles.items():
        tens = np.zeros((len(phact_cols) + 1, MIRNA_LENGTH), dtype=np.float32)
        mask = masks.get(mid, np.zeros(MIRNA_LENGTH, dtype=np.float32)).astype(bool)
        if mask.any():
            vals = raw.copy().astype(np.float32)
            vals[:, mask] = (vals[:, mask] - means) / stds
            vals[:, ~mask] = 0.0
            vals[~np.isfinite(vals)] = 0.0
            tens[: len(phact_cols), :] = vals
            tens[len(phact_cols), mask] = 1.0
        out[mid] = tens
    return out

def build_mirna_phact_tensor(n: int, selected_df: pd.DataFrame, std_profiles: Dict[str, np.ndarray], n_score_cols: int) -> np.ndarray:
    tensor = np.zeros((n, n_score_cols + 1, MIRNA_LENGTH), dtype=np.float32)
    if len(selected_df) == 0 or not std_profiles:
        return tensor
    for row in selected_df.itertuples(index=False):
        ridx = int(row.row_idx)
        mid = str(row.mirgenedb_mature_id)
        prof = std_profiles.get(mid)
        if prof is not None and 0 <= ridx < n:
            tensor[ridx, :, :] = prof
    return tensor

def fit_and_transform_mirna_phact(
    train_split_dir: Path,
    train_sample_ids: Sequence[str],
    val_split_dir: Path,
    val_sample_ids: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    train_candidate_count, train_has_selected, _, train_selected_df, train_cand_summary = read_candidate_info(train_split_dir, train_sample_ids)
    val_candidate_count, val_has_selected, _, val_selected_df, val_cand_summary = read_candidate_info(val_split_dir, val_sample_ids)
    phact_cols = discover_mirna_phact_columns(train_split_dir)
    train_profiles, train_masks, train_profile_summary = load_mirna_raw_profiles(train_split_dir, phact_cols)
    mirna_scalers = fit_mirna_phact_scalers(train_selected_df, train_profiles, train_masks, phact_cols)
    train_std_profiles = standardize_mirna_profiles(train_profiles, train_masks, phact_cols, mirna_scalers)
    train_tensor = build_mirna_phact_tensor(len(train_sample_ids), train_selected_df, train_std_profiles, len(phact_cols))
    del train_profiles, train_masks, train_std_profiles
    gc.collect()

    val_profiles, val_masks, val_profile_summary = load_mirna_raw_profiles(val_split_dir, phact_cols)
    val_std_profiles = standardize_mirna_profiles(val_profiles, val_masks, phact_cols, mirna_scalers)
    val_tensor = build_mirna_phact_tensor(len(val_sample_ids), val_selected_df, val_std_profiles, len(phact_cols))
    del val_profiles, val_masks, val_std_profiles
    gc.collect()

    def adjust_cols(x: np.ndarray) -> np.ndarray:
        if x.shape[1] == MIRNA_PHACT_CHANNELS:
            return x
        y = np.zeros((x.shape[0], MIRNA_PHACT_CHANNELS, MIRNA_LENGTH), dtype=np.float32)
        if x.shape[1] > 0:
            score_n = min(x.shape[1] - 1, MIRNA_PHACT_SCORE_CHANNELS)
            if score_n > 0:
                y[:, :score_n, :] = x[:, :score_n, :]
            y[:, MIRNA_PHACT_SCORE_CHANNELS, :] = x[:, x.shape[1] - 1, :]
        return y

    train_tensor = adjust_cols(train_tensor)
    val_tensor = adjust_cols(val_tensor)
    metadata = {
        "mirna_phact_columns": list(phact_cols),
        "mirna_phact_scalers": mirna_scalers,
        "train_candidate_summary": train_cand_summary,
        "validation_candidate_summary": val_cand_summary,
        "train_profile_summary": train_profile_summary,
        "validation_profile_summary": val_profile_summary,
    }
    return train_tensor, val_tensor, np.column_stack([train_candidate_count, train_has_selected]), np.column_stack([val_candidate_count, val_has_selected]), metadata

def update_stats(values: np.ndarray, count: int, sums: float, sums_sq: float) -> Tuple[int, float, float]:
    vals = values[np.isfinite(values)]
    if vals.size:
        vals64 = vals.astype(np.float64, copy=False)
        count += int(vals64.size)
        sums += float(vals64.sum())
        sums_sq += float(np.square(vals64).sum())
    return count, sums, sums_sq

def fit_or_transform_target_phact(
    split_dir: Path,
    sample_ids: Sequence[str],
    fit: bool,
    scalers: Optional[Dict[str, Dict[str, float]]] = None,
    chunksize: int = PH_TARGET_CHUNKSIZE,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]], np.ndarray, Dict[str, object]]:
    n = len(sample_ids)
    tensor = np.zeros((n, TARGET_PHACT_CHANNELS_N, TARGET_LENGTH), dtype=np.float32)
    path = split_dir / "input" / "phact_target_positions.tsv"
    stats_names = ["score_A", "score_C", "score_G", "score_T", "actual_score", "nonactual_mean", "actual_minus_nonactual_mean"]
    counts = {name: 0 for name in stats_names}
    sums = {name: 0.0 for name in stats_names}
    sums_sq = {name: 0.0 for name in stats_names}
    rows_seen = 0
    rows_filled = 0
    if not path.exists():
        out_scalers = scalers if scalers is not None else {name: default_scalar_stats() for name in stats_names}
        return tensor, out_scalers, np.zeros(n, dtype=np.float32), {"file_present": False, "rows_seen": 0, "rows_filled": 0}
    id_to_idx = pd.Series(np.arange(n, dtype=np.int64), index=pd.Index(sample_ids, dtype="object"))
    base_code = {"A": 0, "C": 1, "G": 2, "T": 3}
    usecols = ["id", "target_position_1based", "actual_nt"] + TARGET_PHACT_SCORE_COLS
    for chunk in pd.read_csv(path, sep="\t", usecols=usecols, chunksize=chunksize):
        rows_seen += int(len(chunk))
        row_idx_ser = chunk["id"].map(id_to_idx)
        valid_id = row_idx_ser.notna().to_numpy()
        if not valid_id.any():
            del chunk
            continue
        row_idx = row_idx_ser.to_numpy(dtype=np.float64)
        pos = pd.to_numeric(chunk["target_position_1based"], errors="coerce").to_numpy(dtype=np.float64) - 1.0
        scores = chunk[TARGET_PHACT_SCORE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        actual_codes = chunk["actual_nt"].astype(str).str.upper().map(base_code).to_numpy(dtype=np.float64)
        all4 = np.isfinite(scores).all(axis=1)
        valid_actual = np.isfinite(actual_codes)
        valid_pos = np.isfinite(pos) & (pos >= 0) & (pos < TARGET_LENGTH)
        valid = valid_id & valid_pos & all4 & valid_actual
        if fit:
            valid_for_stats = valid_id & valid_pos
            for j, name in enumerate(TARGET_PHACT_SCORE_COLS):
                counts[name], sums[name], sums_sq[name] = update_stats(scores[valid_for_stats, j], counts[name], sums[name], sums_sq[name])
        if valid.any():
            vi = np.where(valid)[0]
            ridx = row_idx[vi].astype(np.int64)
            pidx = pos[vi].astype(np.int64)
            sc = scores[vi].astype(np.float32, copy=False)
            ac = actual_codes[vi].astype(np.int64)
            actual = sc[np.arange(sc.shape[0]), ac]
            nonactual = (sc.sum(axis=1) - actual) / 3.0
            diff = actual - nonactual
            if fit:
                counts["actual_score"], sums["actual_score"], sums_sq["actual_score"] = update_stats(actual, counts["actual_score"], sums["actual_score"], sums_sq["actual_score"])
                counts["nonactual_mean"], sums["nonactual_mean"], sums_sq["nonactual_mean"] = update_stats(nonactual, counts["nonactual_mean"], sums["nonactual_mean"], sums_sq["nonactual_mean"])
                counts["actual_minus_nonactual_mean"], sums["actual_minus_nonactual_mean"], sums_sq["actual_minus_nonactual_mean"] = update_stats(diff, counts["actual_minus_nonactual_mean"], sums["actual_minus_nonactual_mean"], sums_sq["actual_minus_nonactual_mean"])
            tensor[ridx, 0, pidx] = sc[:, 0]
            tensor[ridx, 1, pidx] = sc[:, 1]
            tensor[ridx, 2, pidx] = sc[:, 2]
            tensor[ridx, 3, pidx] = sc[:, 3]
            tensor[ridx, 4, pidx] = actual
            tensor[ridx, 5, pidx] = nonactual
            tensor[ridx, 6, pidx] = diff
            tensor[ridx, 7, pidx] = 1.0
            rows_filled += int(valid.sum())
        del chunk
    if fit:
        out_scalers = {name: stats_from_sums(counts[name], sums[name], sums_sq[name]) for name in stats_names}
    else:
        if scalers is None:
            raise ValueError("scalers must be supplied when fit=False")
        out_scalers = scalers
    mask = tensor[:, 7, :] > 0.5
    for ch, name in enumerate(stats_names):
        arr = tensor[:, ch, :]
        if mask.any():
            arr[mask] = (arr[mask] - out_scalers[name]["mean"]) / out_scalers[name]["std"]
        arr[~mask] = 0.0
    has_any = mask.any(axis=1).astype(np.float32)
    summary = {"file_present": True, "rows_seen": int(rows_seen), "rows_filled": int(rows_filled), "samples_with_any": int(has_any.sum())}
    return tensor, out_scalers, has_any, summary

def build_preprocessor(train_samples: pd.DataFrame, phact_metadata: Dict[str, object], target_phact_scalers: Dict[str, Dict[str, float]], raw_aux_train: np.ndarray, positional_scalers: Dict[str, Dict[str, float]], pretrained_summary: Dict[str, object]) -> Dict[str, object]:
    category_vocabs = {
        "feature": fit_category_vocab(train_samples["feature"].tolist()),
        "dominant_region": fit_category_vocab(train_samples["dominant_region"].tolist()),
    }
    return {
        "aux_numeric_scaler": fit_standard_scaler(raw_aux_train),
        "positional_scalers": positional_scalers,
        "category_vocabs": category_vocabs,
        "aux_numeric_feature_names": AUX_NUMERIC_FEATURE_NAMES,
        "sequence_encoding": {
            "model": "miRBind2_pairwise_onehot",
            "target_orientation": "as_supplied",
            "mirna_length": MIRNA_LENGTH,
            "target_length": TARGET_LENGTH,
            "pair_base_order": MIRBIND2_PAIR_BASE_ORDER,
            "pair_state_names": MIRBIND2_PAIR_STATE_NAMES,
            "num_pairs": MIRBIND2_NUM_PAIRS,
            "onehot_channels": MIRBIND2_ONEHOT_CHANNELS,
            "pad_or_trim": "right_pad_N_keep_left_prefix",
            "unknown_index": 17,
            "padding_pair_index": 16,
        },
        "rc_pairwise_encoding": {
            "model": "reverse_complement_pairwise_grid",
            "shape": [RC_PAIRWISE_CHANNELS, MIRNA_LENGTH, TARGET_LENGTH],
            "base_order": RC_PAIRWISE_BASE_ORDER,
            "channel_names": RC_PAIRWISE_CHANNEL_NAMES,
            "target_policy": "right_pad_or_truncate_to_50_then_reverse_complement",
            "mirna_policy": "right_pad_or_truncate_to_28_no_reverse_complement",
            "seed_positions_1based_inclusive": [2, 8],
            "valid_pair_channel": 18,
            "seed_mask_channel": 19,
        },
        "alignment_feature_rules": {
            "feature_order": NEW_ALIGNMENT_FEATURE_NAMES,
            "alignment_segments": [{"name": name, "start_0based": start, "end_exclusive": end} for name, start, end in ALIGNMENT_SEGMENTS],
            "alignment_feature_suffixes": ALIGNMENT_FEATURE_SUFFIXES,
            "exact_rc_count_segments": [{"name": name, "start_0based": start, "end_exclusive": end} for name, start, end in EXACT_RC_COUNT_SEGMENTS],
            "coordinate": "target_as_supplied; antiparallel segment windows; exact counts search reverse-complemented miRNA segment in target",
        },
        "target_phact_channel_order": TARGET_PHACT_CHANNEL_ORDER,
        "target_phact_scalers": target_phact_scalers,
        "mirna_phact_columns": phact_metadata.get("mirna_phact_columns", []),
        "mirna_phact_scalers": phact_metadata.get("mirna_phact_scalers", {}),
        "category_min_count": CATEGORY_MIN_COUNT,
        "target_length": TARGET_LENGTH,
        "mirna_length": MIRNA_LENGTH,
        "phact_preprocessing_summary": {k: v for k, v in phact_metadata.items() if k not in {"mirna_phact_scalers"}},
        "pretrained_summary": pretrained_summary,
    }

def transform_core_arrays(samples: pd.DataFrame, positional_scalers: Dict[str, Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mirna_codes = encode_sequences(samples["noncodingRNA"].tolist(), MIRNA_LENGTH)
    target_codes = encode_sequences(samples["gene"].tolist(), TARGET_LENGTH)
    seq_features = compute_sequence_features(samples["gene"].tolist(), samples["noncodingRNA"].tolist())
    cons_summary, cons_tensor = conservation_summaries_and_tensor(samples, positional_scalers)
    return mirna_codes, target_codes, seq_features, cons_summary, cons_tensor

PAIR_ORDER_LOOKUP_TENSOR: Optional[torch.Tensor] = None

def sanitize_rna_for_rinalmo(seq: object) -> str:
    """Clean a DNA/RNA string for frozen RiNALMo embedding."""
    cleaned_dna = sanitize_dna(seq)
    if not cleaned_dna:
        return "N"
    return cleaned_dna.replace("T", "U")

def build_mirbind2_onehot(mirna_codes: torch.Tensor, target_codes: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    global PAIR_ORDER_LOOKUP_TENSOR
    if PAIR_ORDER_LOOKUP_TENSOR is None or PAIR_ORDER_LOOKUP_TENSOR.device != mirna_codes.device:
        PAIR_ORDER_LOOKUP_TENSOR = torch.tensor([0, 2, 3, 1, -1], device=mirna_codes.device, dtype=torch.long)
    mi = mirna_codes.long() if mirna_codes.dtype != torch.long else mirna_codes
    tg = target_codes.long() if target_codes.dtype != torch.long else target_codes
    mi_order = PAIR_ORDER_LOOKUP_TENSOR[mi].unsqueeze(2)
    tg_order = PAIR_ORDER_LOOKUP_TENSOR[tg].unsqueeze(1)
    valid = (mi_order >= 0) & (tg_order >= 0)
    idx = torch.where(valid, mi_order * 4 + tg_order, torch.full_like(mi_order * 4 + tg_order, 17))
    x = torch.zeros((mi.shape[0], MIRNA_LENGTH, TARGET_LENGTH, MIRBIND2_ONEHOT_CHANNELS), device=mi.device, dtype=dtype)
    x.scatter_(3, idx.unsqueeze(3), 1.0)
    return x

def build_rc_pairwise_grid(mirna_codes: torch.Tensor, target_codes: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    mi = mirna_codes.long() if mirna_codes.dtype != torch.long else mirna_codes
    tg = target_codes.long() if target_codes.dtype != torch.long else target_codes
    complement_lookup = torch.tensor([3, 2, 1, 0, 4], device=tg.device, dtype=torch.long)
    tg_rc = complement_lookup[tg].flip(dims=[1])
    mi_e = mi.unsqueeze(2)
    tg_e = tg_rc.unsqueeze(1)
    valid = (mi_e < 4) & (tg_e < 4)
    pair_idx = mi_e * 4 + tg_e
    x = torch.zeros((mi.shape[0], RC_PAIRWISE_CHANNELS, MIRNA_LENGTH, TARGET_LENGTH), device=mi.device, dtype=dtype)
    pair_idx_clamped = torch.where(valid, pair_idx, torch.zeros_like(pair_idx))
    pair_onehot = torch.zeros((mi.shape[0], MIRNA_LENGTH, TARGET_LENGTH, 16), device=mi.device, dtype=dtype)
    pair_onehot.scatter_(3, pair_idx_clamped.unsqueeze(3), 1.0)
    pair_onehot = pair_onehot * valid.unsqueeze(3).to(dtype)
    x[:, :16, :, :] = pair_onehot.permute(0, 3, 1, 2).contiguous()
    wc = ((mi_e == 0) & (tg_e == 3)) | ((mi_e == 3) & (tg_e == 0)) | ((mi_e == 1) & (tg_e == 2)) | ((mi_e == 2) & (tg_e == 1))
    wobble = ((mi_e == 2) & (tg_e == 3)) | ((mi_e == 3) & (tg_e == 2))
    x[:, 16, :, :] = (wc & valid).to(dtype)
    x[:, 17, :, :] = (wobble & valid).to(dtype)
    x[:, 18, :, :] = valid.to(dtype)
    mirna_valid = (mi < 4).to(dtype)
    seed_mask = torch.zeros((mi.shape[0], MIRNA_LENGTH), device=mi.device, dtype=dtype)
    seed_mask[:, 1:8] = mirna_valid[:, 1:8]
    x[:, 19, :, :] = seed_mask.unsqueeze(2).expand(-1, -1, TARGET_LENGTH)
    return x

def _adjust_mirna_tensor_channels(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == MIRNA_PHACT_CHANNELS:
        return x.astype(np.float32, copy=False)
    y = np.zeros((x.shape[0], MIRNA_PHACT_CHANNELS, MIRNA_LENGTH), dtype=np.float32)
    if x.shape[1] > 0:
        score_n = min(x.shape[1] - 1, MIRNA_PHACT_SCORE_CHANNELS)
        if score_n > 0:
            y[:, :score_n, :] = x[:, :score_n, :]
        y[:, MIRNA_PHACT_SCORE_CHANNELS, :] = x[:, x.shape[1] - 1, :]
    return y


def transform_mirna_phact_with_metadata(split_dir: Path, sample_ids: Sequence[str], metadata: Dict[str, object]):
    candidate_count, has_selected, _, selected_df, candidate_summary = read_candidate_info(Path(split_dir), sample_ids)
    phact_cols = list(metadata['mirna_phact_columns'])
    profiles, masks, profile_summary = load_mirna_raw_profiles(Path(split_dir), phact_cols)
    std_profiles = standardize_mirna_profiles(profiles, masks, phact_cols, metadata['mirna_phact_scalers'])
    tensor = _adjust_mirna_tensor_channels(build_mirna_phact_tensor(len(sample_ids), selected_df, std_profiles, len(phact_cols)))
    aux = np.column_stack([candidate_count, has_selected]).astype(np.float32)
    return tensor, aux, {'candidate_summary': candidate_summary, 'profile_summary': profile_summary}


def tokenize_rinalmo_sequences(sequences: Sequence[object], tokenizer, fixed_length: int) -> Dict[str, np.ndarray]:
    """Tokenize cleaned RNA and return fixed token arrays plus biological masks."""
    cleaned = [sanitize_rna_for_rinalmo(x) for x in sequences]
    encoded = tokenizer(
        cleaned,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=int(fixed_length),
        return_attention_mask=True,
        return_special_tokens_mask=True,
        return_tensors='np',
    )
    input_ids = np.asarray(encoded['input_ids'], dtype=np.int64)
    attention_mask = np.asarray(encoded['attention_mask'], dtype=np.int64)
    special = np.asarray(encoded['special_tokens_mask'], dtype=np.int64)
    biological_mask = ((attention_mask == 1) & (special == 0)).astype(np.int64)
    if input_ids.shape != attention_mask.shape or input_ids.shape != biological_mask.shape:
        raise ValueError('RiNALMo token/mask shape mismatch')
    if np.any(biological_mask.sum(axis=1) < 1):
        raise ValueError('Every sequence must retain at least one biological token')
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'biological_mask': biological_mask}


def deduplicate_token_batch(input_ids: torch.Tensor, attention_mask: torch.Tensor, biological_mask: torch.Tensor):
    """Stable first-occurrence deduplication for the online miRNA encoder path."""
    rows = input_ids.detach().cpu().tolist()
    first = {}
    unique_idx, inverse = [], []
    for i, row in enumerate(rows):
        key = tuple(row)
        if key not in first:
            first[key] = len(unique_idx)
            unique_idx.append(i)
        inverse.append(first[key])
    index = torch.tensor(unique_idx, device=input_ids.device, dtype=torch.long)
    inv = torch.tensor(inverse, device=input_ids.device, dtype=torch.long)
    return input_ids.index_select(0, index), attention_mask.index_select(0, index), biological_mask.index_select(0, index), inv


def make_online_rinalmo_pair_vector(mirna_hidden: torch.Tensor, target_hidden: torch.Tensor,
                                    mirna_biological_mask: torch.Tensor,
                                    target_biological_mask: torch.Tensor) -> torch.Tensor:
    """Exact iteration-13 pair contract from online final hidden states."""
    def pool(hidden, mask):
        w = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
    m = pool(mirna_hidden, mirna_biological_mask)
    t = pool(target_hidden, target_biological_mask)
    return torch.cat([m, t, torch.abs(m - t), m * t], dim=1)


def _assemble_raw_aux(seq_features, cons_summary, mirna_aux, target_any):
    phact_aux = np.column_stack([mirna_aux, target_any.astype(np.float32)])
    base = seq_features[:, :len(BASE_SEQ_FEATURE_NAMES)]
    align = seq_features[:, len(BASE_SEQ_FEATURE_NAMES):]
    raw = np.concatenate([base, cons_summary, phact_aux.astype(np.float32), align], axis=1).astype(np.float32)
    if raw.shape[1] != len(AUX_NUMERIC_FEATURE_NAMES):
        raise ValueError(f'Aux feature shape mismatch: {raw.shape[1]} vs {len(AUX_NUMERIC_FEATURE_NAMES)}')
    return raw


def fit_representation(train_split_dir: Path, tokenizer=None) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Fit every preprocessing object on this training split and transform it."""
    split_dir = Path(train_split_dir)
    samples = read_samples(split_dir)
    ids = samples['id'].astype(str).tolist()
    target_phact, target_scalers, target_any, target_summary = fit_or_transform_target_phact(split_dir, ids, fit=True)
    mirna_tensor, _, mirna_aux, _, mirna_meta = fit_and_transform_mirna_phact(split_dir, ids, split_dir, ids)
    positional = fit_conservation_positional_scalers(samples)
    mirna_codes, target_codes, seq_features, cons_summary, cons_tensor = transform_core_arrays(samples, positional)
    target_tensor = np.concatenate([cons_tensor, target_phact], axis=1).astype(np.float32, copy=False)
    raw_aux = _assemble_raw_aux(seq_features, cons_summary, mirna_aux, target_any)
    mirna_meta = dict(mirna_meta)
    mirna_meta['target_phact_train_summary'] = target_summary
    preprocessor = build_preprocessor(samples, mirna_meta, target_scalers, raw_aux, positional, {})
    preprocessor['representation_config'] = asdict(RepresentationConfig())
    preprocessor['rinalmo_tokenization'] = {
        'sequence_orientation': 'original_mature_miRNA_and_gene_as_supplied',
        'cleaning': 'uppercase_DNA_clean_U_to_T_then_T_to_U_keep_N',
        'mirna_fixed_token_length': MIRNA_LENGTH + 2,
        'target_fixed_token_length': TARGET_LENGTH + 2,
        'pooling': 'mean_final_hidden_state_over_biological_tokens_only',
        'pair_vector': 'concat_mirna_target_absdiff_product',
        'deduplicate_mirna_within_batch': True,
    }
    arrays = {
        'ids': np.asarray(ids, dtype=object), 'mirna_codes': mirna_codes, 'target_codes': target_codes,
        'aux_numeric': apply_standard_scaler(raw_aux, preprocessor['aux_numeric_scaler']),
        'target_tensor': target_tensor, 'mirna_phact_tensor': mirna_tensor,
        'feature_index': encode_categories(samples['feature'].tolist(), preprocessor['category_vocabs']['feature']),
        'dominant_region_index': encode_categories(samples['dominant_region'].tolist(), preprocessor['category_vocabs']['dominant_region']),
    }
    if tokenizer is not None:
        mt = tokenize_rinalmo_sequences(samples['noncodingRNA'].tolist(), tokenizer, MIRNA_LENGTH + 2)
        tt = tokenize_rinalmo_sequences(samples['gene'].tolist(), tokenizer, TARGET_LENGTH + 2)
        arrays.update({f'mirna_{k}': v for k, v in mt.items()})
        arrays.update({f'target_{k}': v for k, v in tt.items()})
    return arrays, preprocessor


def transform_representation(split_dir: Path, preprocessor: Dict[str, object], tokenizer=None) -> Dict[str, np.ndarray]:
    """Transform inference inputs using frozen train-fitted preprocessing only."""
    split_dir = Path(split_dir)
    samples = read_samples(split_dir)
    ids = samples['id'].astype(str).tolist()
    target_phact, _, target_any, _ = fit_or_transform_target_phact(split_dir, ids, fit=False, scalers=preprocessor['target_phact_scalers'])
    mirna_tensor, mirna_aux, _ = transform_mirna_phact_with_metadata(split_dir, ids, preprocessor)
    mirna_codes, target_codes, seq_features, cons_summary, cons_tensor = transform_core_arrays(samples, preprocessor['positional_scalers'])
    raw_aux = _assemble_raw_aux(seq_features, cons_summary, mirna_aux, target_any)
    arrays = {
        'ids': np.asarray(ids, dtype=object), 'mirna_codes': mirna_codes, 'target_codes': target_codes,
        'aux_numeric': apply_standard_scaler(raw_aux, preprocessor['aux_numeric_scaler']),
        'target_tensor': np.concatenate([cons_tensor, target_phact], axis=1).astype(np.float32, copy=False),
        'mirna_phact_tensor': mirna_tensor,
        'feature_index': encode_categories(samples['feature'].tolist(), preprocessor['category_vocabs']['feature']),
        'dominant_region_index': encode_categories(samples['dominant_region'].tolist(), preprocessor['category_vocabs']['dominant_region']),
    }
    if tokenizer is not None:
        mt = tokenize_rinalmo_sequences(samples['noncodingRNA'].tolist(), tokenizer, MIRNA_LENGTH + 2)
        tt = tokenize_rinalmo_sequences(samples['gene'].tolist(), tokenizer, TARGET_LENGTH + 2)
        arrays.update({f'mirna_{k}': v for k, v in mt.items()})
        arrays.update({f'target_{k}': v for k, v in tt.items()})
    return arrays


def representation_shapes() -> Dict[str, object]:
    return {
        'mirbind2_pair': [MIRNA_LENGTH, TARGET_LENGTH, MIRBIND2_ONEHOT_CHANNELS],
        'reverse_complement_pair_grid': [RC_PAIRWISE_CHANNELS, MIRNA_LENGTH, TARGET_LENGTH],
        'target_conservation_phact': [TARGET_BRANCH_CHANNELS, TARGET_LENGTH],
        'mirna_phact': [MIRNA_PHACT_CHANNELS, MIRNA_LENGTH],
        'aux_numeric': len(AUX_NUMERIC_FEATURE_NAMES),
        'rinalmo_tokens': {'mirna': MIRNA_LENGTH + 2, 'target': TARGET_LENGTH + 2},
        'rinalmo_pair_vector': RINALMO_CONCAT_DIM,
    }
