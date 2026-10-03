"""Extract score summaries and immutable row references for conservation-aware folds."""

import csv
import gzip
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_mirna_p1_rows(root: Path, mapping_path: Path | None = None):
    root = Path(root)
    mapping_path = mapping_path or root / "mapping/mirna_positions.tsv"
    with mapping_path.open() as handle:
        positions = list(csv.DictReader(handle, delimiter="\t"))
    needed = {(row["score_id"], int(row["score_position_1based"])) for row in positions}
    values = {}
    with gzip.open(root / "scores/mirna_qntnorm_transformed.tsv.gz", "rt") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["ID"], int(row["Position"]))
            if key in needed:
                full_key = (*key, row["Nucleotide"])
                assert full_key not in values
                values[full_key] = row["PHACTn_gapAware_wtNTnorm_wl_param_1"]
    result = []
    for row in positions:
        key = (row["score_id"], int(row["score_position_1based"]))
        result.append(
            {
                "mirgenedb_mature_id": row["mirgenedb_mature_id"],
                "mirna_position_1based": row["mirna_position_1based"],
                "actual_nt": row["actual_nt"],
                **{"phact_param_1_" + base: values[(*key, base)] for base in "ACGT"},
            }
        )
    return result


def prepare_fold_features(data_root: Path, output_dir: Path) -> Path:
    root = Path(data_root).resolve()
    out = Path(output_dir).resolve()
    started = time.monotonic()

    def log(message):
        print(f"[{time.monotonic() - started:.0f}s] {message}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    if (out / "features.npz").exists():
        raise FileExistsError("Refusing to replace existing extracted features")
    frames = []
    sources = {}
    cols = [
        "gene",
        "noncodingRNA",
        "label",
        "chr",
        "start",
        "end",
        "strand",
        "dominant_region",
    ]
    split_counts = {}
    for split in ["train", "test"]:
        path = root / "manakov" / f"{split}.tsv"
        df = pd.read_csv(path, sep="\t", usecols=cols, dtype=str, keep_default_na=False)
        assert len(df) > 0
        split_counts[split] = len(df)
        frames.append(df)
        sources[str(path)] = {"rows": len(df), "sha256": sha(path)}
    df = pd.concat(frames, ignore_index=True)
    del frames
    df["noncodingRNA"] = df["noncodingRNA"].str.upper().str.replace("U", "T")
    df["gene"] = df["gene"].str.upper().str.replace("U", "T")
    mirna, sequences = pd.factorize(df["noncodingRNA"], sort=True)
    leftout_path = root / "manakov/leftout.tsv"
    left = pd.read_csv(leftout_path, sep="\t", usecols=["noncodingRNA"], dtype=str)
    left_sequences = set(left["noncodingRNA"].str.upper().str.replace("U", "T"))
    assert not set(sequences) & left_sequences
    sources[str(leftout_path)] = {
        "rows": len(left),
        "mirnas": len(left_sequences),
        "sha256": sha(leftout_path),
    }
    n = len(df)
    train_rows = split_counts["train"]
    starts = pd.to_numeric(df["start"]).to_numpy(dtype=np.int64)
    ends = pd.to_numeric(df["end"]).to_numpy(dtype=np.int64)
    assert (ends - starts == 49).all() and df["strand"].isin(["+", "-"]).all()
    window_strings = (
        "chr"
        + df["chr"].str.removeprefix("chr")
        + "|"
        + pd.Series(starts).astype(str)
        + "|"
        + pd.Series(ends).astype(str)
        + "|"
        + df["strand"]
        + "|"
        + df["gene"]
    )
    target, target_keys = pd.factorize(window_strings, sort=True)
    target_genes = np.array([s.rsplit("|", 1)[1] for s in target_keys])
    if any(len(seq) != 50 or not set(seq) <= set("ACGT") for seq in target_genes):
        raise ValueError(
            "Conservation summaries require 50-nt A/C/G/T target sequences"
        )
    seq_target, target_sequences = pd.factorize(df["gene"], sort=True)
    packed_pairs = (mirna.astype(np.uint64) << np.uint64(32)) | seq_target.astype(
        np.uint64
    )
    _, pair, pair_counts = np.unique(
        packed_pairs, return_inverse=True, return_counts=True
    )
    label = df["label"].to_numpy(dtype=np.int8)
    assert np.isin(label, [0, 1]).all()
    pair_min = np.full(len(pair_counts), 2, dtype=np.int8)
    pair_max = np.full(len(pair_counts), -1, dtype=np.int8)
    np.minimum.at(pair_min, pair, label)
    np.maximum.at(pair_max, pair, label)
    conflicts = pair_min != pair_max
    region, regions = pd.factorize(df["dominant_region"], sort=True)
    log(
        f"Indexed {n:,} rows, {len(sequences):,} miRNAs, {len(target_keys):,} genomic windows, {len(pair_counts):,} sequence pairs"
    )
    del (
        df,
        window_strings,
        starts,
        ends,
        packed_pairs,
        seq_target,
        left,
        target_sequences,
    )

    profile_rows = defaultdict(list)
    for row in load_mirna_p1_rows(root):
        profile_rows[row["mirgenedb_mature_id"]].append(row)
    profile_names = sorted(profile_rows)
    profile_index = {name: i for i, name in enumerate(profile_names)}
    profile_sequences = []
    profile_values = []
    profile_quartets = []
    for name in profile_names:
        rows = sorted(
            profile_rows[name], key=lambda row: int(row["mirna_position_1based"])
        )
        seq = "".join(row["actual_nt"] for row in rows)
        quartet = np.array(
            [[float(row["phact_param_1_" + b]) for b in "ACGT"] for row in rows]
        )
        assert np.isfinite(quartet).all()
        ref = quartet[np.arange(len(seq)), ["ACGT".index(b) for b in seq]]
        contrast = ref - (quartet.sum(axis=1) - ref) / 3
        profile_sequences.append(seq)
        profile_quartets.append(quartet)
        profile_values.append([float(contrast.mean()), float(contrast[1:8].mean())])
    profile_values = np.array(profile_values)
    target_features = np.full((len(target_keys), 4), np.nan)
    target_digests = np.zeros(len(target_keys), dtype="V16")
    target_seen = np.zeros(len(target_keys), dtype=bool)
    source_file = np.full(n, 255, dtype=np.uint8)
    source_row = np.zeros(n, dtype=np.int32)
    row_profile = np.full(n, -1, dtype=np.int16)
    verified_profiles = set()
    base_indices = np.zeros(256, dtype=np.int64)
    for j, b in enumerate(b"ACGT"):
        base_indices[b] = j
    array_files = ["train", "validation", "test"]
    for fi, split in enumerate(array_files):
        path = root / "datasets_p1" / f"{split}.csv"
        h = hashlib.sha256()
        with path.open("rb", buffering=8 * 1024 * 1024) as f:

            def lines():
                for line in f:
                    h.update(line)
                    yield line.decode()

            number = 0
            for number, row in enumerate(csv.DictReader(lines()), 1):
                _, origin, num = row["id"].split("_")
                assert origin in ["train", "test"]
                i = int(num) - 1 + (train_rows if origin == "test" else 0)
                assert 0 <= i < n and source_file[i] == 255
                m = mirna[i]
                t = target[i]
                assert row["noncodingRNA"].upper().replace("U", "T") == sequences[m]
                assert row["gene"].upper().replace("U", "T") == target_genes[t]
                assert int(row["label"]) == int(label[i])
                source_file[i] = fi
                source_row[i] = number
                profile = row["mirna_profile_id"]
                if profile:
                    pi = profile_index[profile]
                    assert profile_sequences[pi] == sequences[m]
                    row_profile[i] = pi
                    if pi not in verified_profiles:
                        observed = np.array(
                            [json.loads(row["mirna_phact_" + b]) for b in "ACGT"]
                        ).T
                        np.testing.assert_allclose(
                            observed, profile_quartets[pi], rtol=0, atol=1e-12
                        )
                        verified_profiles.add(pi)
                score_strings = [row["target_phact_" + b] for b in "ACGT"]
                digest = hashlib.blake2b(
                    "|".join(score_strings).encode(), digest_size=16
                ).digest()
                if target_seen[t]:
                    assert target_digests[t].tobytes() == digest, (
                        f"Inconsistent target scores at {target_keys[t]}"
                    )
                else:
                    target_digests[t] = digest
                    target_seen[t] = True
                    quartet = np.array([json.loads(s) for s in score_strings]).T
                    assert quartet.shape == (50, 4) and not np.isinf(quartet).any()
                    valid = np.isfinite(quartet).all(axis=1)
                    assert np.array_equal(valid, np.isfinite(quartet).any(axis=1))
                    target_features[t, 3] = valid.mean()
                    if valid.any():
                        bi = base_indices[
                            np.frombuffer(target_genes[t].encode(), dtype=np.uint8)
                        ]
                        ref = quartet[np.arange(50), bi]
                        contrast = (ref - (quartet.sum(axis=1) - ref) / 3)[valid]
                        target_features[t, :3] = [
                            contrast.mean(),
                            contrast.std(),
                            np.quantile(contrast, 0.9),
                        ]
                if number % 250000 == 0:
                    log(f"{split}: checked {number:,} score-array rows")
        sources[str(path)] = {"rows": number, "sha256": h.hexdigest()}
        log(f"Completed {split}: {number:,} rows")
    assert (source_file != 255).all() and target_seen.all()
    mirna_conservation = np.full((len(sequences), 2), np.nan)
    used_profiles = {}
    for m, seq in enumerate(sequences):
        profiles = np.unique(row_profile[mirna == m])
        profiles = profiles[profiles >= 0]
        used_profiles[seq] = [profile_names[p] for p in profiles]
        if len(profiles):
            mirna_conservation[m] = np.median(profile_values[profiles], axis=0)
    with (out / "features.npz.partial").open("wb") as f:
        np.savez_compressed(
            f,
            mirna=mirna.astype(np.int16),
            target=target.astype(np.int32),
            pair=pair.astype(np.int32),
            label=label,
            region=region.astype(np.int8),
            source_file=source_file,
            source_row=source_row,
            row_profile=row_profile,
            mirna_sequences=np.array(sequences, dtype=str),
            target_keys=np.array(target_keys, dtype=str),
            target_features=target_features,
            mirna_conservation=mirna_conservation,
            profile_names=np.array(profile_names),
            profile_values=profile_values,
            pair_counts=pair_counts,
            conflicting_pair=conflicts,
            regions=np.array(regions, dtype=str),
        )
    (out / "features.npz.partial").rename(out / "features.npz")
    report = {
        "rows": n,
        "original_split_counts": split_counts,
        "mirnas": len(sequences),
        "unique_target_windows": len(target_keys),
        "unique_sequence_pairs": len(pair_counts),
        "duplicate_pair_extra_rows": int(n - len(pair_counts)),
        "conflicting_label_pairs": int(conflicts.sum()),
        "rows_in_conflicting_pairs": int(pair_counts[conflicts].sum()),
        "conflicting_label_policy": "Retain original observations and labels; exact sequence pairs stay together in each run.",
        "original_leftout_mirna_overlap": 0,
        "sources": sources,
        "array_file_codes": dict(enumerate(array_files)),
        "mirna_summary": "Median across distinct assigned profile means for each exact mature sequence; profiles are not interaction-frequency weighted. No available profile gives NaN.",
        "mirna_profiles_by_sequence": used_profiles,
        "target_summary_columns": [
            "mean_ref_minus_mean_alt",
            "std_ref_minus_mean_alt",
            "q90_ref_minus_mean_alt",
            "coverage",
        ],
        "target_key": "chromosome, start, end, strand and actual 50-nt target sequence",
        "target_consistency": "Every repeated target key has identical serialized four-score arrays across all rows.",
        "source_script": str(Path(__file__).resolve()),
        "seconds": time.monotonic() - started,
    }
    (out / "feature_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    log("SUCCESS: fold features prepared; original datasets unchanged")
    return out / "features.npz"
