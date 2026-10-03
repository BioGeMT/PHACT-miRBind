"""One shared compact cache for the fixed folds; original score CSVs are unchanged."""

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from phact_mirbind.data.columns import PADDING_PAIR_INDEX, PAIR_BASES
from phact_mirbind.data.fold_features import sha


def bases(seq, length):
    seq = seq.upper().replace("U", "T")
    assert set(seq) <= set("ACGTN")
    lookup = {b: i for i, b in enumerate(PAIR_BASES)}
    result = np.full(length, 4, dtype=np.uint8)
    result[: min(length, len(seq))] = [lookup.get(b, 4) for b in seq[:length]]
    return result


def target_scores(row):
    values = np.array(
        [json.loads(row["target_phact_" + b]) for b in "ACGT"], dtype=np.float32
    ).T
    if values.shape != (50, 4) or np.isinf(values).any():
        raise ValueError(
            "Target scores must contain four 50-position arrays without infinities"
        )
    missing = np.isnan(values)
    if not np.array_equal(missing.any(axis=1), missing.all(axis=1)):
        raise ValueError(
            "A missing position requires NaN in all four target score arrays"
        )
    assert np.all(np.abs(values[np.isfinite(values)]) <= np.finfo(np.float16).max)
    return np.column_stack(
        [np.nan_to_num(values, nan=0.5), missing.all(axis=1)]
    ).astype(np.float16)


def prepare_training_cache(data_root: Path, folds_dir: Path, output_dir: Path) -> Path:
    """Write one reusable cache for all five runs, without expanding pair grids."""
    root = Path(data_root).resolve()
    folds = Path(folds_dir).resolve()
    out = Path(output_dir).resolve()
    started = time.monotonic()

    def log(message):
        print(f"[{time.monotonic() - started:.0f}s] {message}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Cache directory must be empty: {out}")
    with np.load(folds / "features.npz") as f:
        data = {
            k: f[k]
            for k in [
                "mirna",
                "target",
                "label",
                "target_keys",
                "mirna_sequences",
                "source_file",
                "source_row",
            ]
        }
    n = len(data["label"])
    nt = len(data["target_keys"])
    nm = len(data["mirna_sequences"])
    audit = json.loads((folds / "feature_audit.json").read_text())
    original_train_rows = next(
        v["rows"] for k, v in audit["sources"].items() if Path(k).name == "train.tsv"
    )
    left_count = next(
        v["rows"] for k, v in audit["sources"].items() if Path(k).name == "leftout.tsv"
    )
    source_audit = {Path(k).name: v for k, v in audit["sources"].items()}
    target_bases = np.lib.format.open_memmap(
        out / "target_bases.npy", mode="w+", dtype=np.uint8, shape=(nt + left_count, 50)
    )
    scores = np.lib.format.open_memmap(
        out / "target_scores.npy",
        mode="w+",
        dtype=np.float16,
        shape=(nt + left_count, 50, 5),
    )
    mirna_bases = np.full((nm + left_count, 28), 4, dtype=np.uint8)
    for m, seq in enumerate(data["mirna_sequences"]):
        mirna_bases[m] = bases(seq, 28)
    for t, key in enumerate(data["target_keys"]):
        target_bases[t] = bases(key.rsplit("|", 1)[1], 50)
    row_mirna = np.r_[
        data["mirna"].astype(np.int32), np.arange(nm, nm + left_count, dtype=np.int32)
    ]
    row_target = np.r_[data["target"], np.arange(nt, nt + left_count, dtype=np.int32)]
    labels = np.r_[data["label"], np.zeros(left_count, dtype=np.int8)]
    seen_rows = np.zeros(n, dtype=bool)
    seen_targets = np.zeros(nt, dtype=bool)
    sources = {}
    # Check each original source record and reuse its unique window's quartet.
    for fi, name in enumerate(["train", "validation", "test", "leftout"]):
        path = root / "datasets_p1" / f"{name}.csv"
        digest = hashlib.sha256()
        with path.open("rb", buffering=8 * 1024 * 1024) as f:

            def lines():
                for line in f:
                    digest.update(line)
                    yield line.decode()

            number = 0
            for number, row in enumerate(csv.DictReader(lines()), 1):
                if name == "leftout":
                    assert (
                        row["id"] == f"manakov_leftout_{number}"
                        and number <= left_count
                    )
                    i = n + number - 1
                    t = nt + number - 1
                    m = nm + number - 1
                    mirna_bases[m] = bases(row["noncodingRNA"], 28)
                    target_bases[t] = bases(row["gene"], 50)
                    scores[t] = target_scores(row)
                    labels[i] = int(row["label"])
                else:
                    _, origin, r = row["id"].split("_")
                    assert origin in ["train", "test"]
                    i = int(r) - 1 + (original_train_rows if origin == "test" else 0)
                    assert 0 <= i < n and not seen_rows[i]
                    assert (
                        data["source_file"][i] == fi and data["source_row"][i] == number
                    )
                    assert int(row["label"]) == data["label"][i]
                    m = int(row_mirna[i])
                    t = int(row_target[i])
                    seen_rows[i] = True
                    assert (
                        row["noncodingRNA"].upper().replace("U", "T")
                        == data["mirna_sequences"][m]
                    )
                    assert (
                        row["gene"].upper().replace("U", "T")
                        == data["target_keys"][t].rsplit("|", 1)[1]
                    )
                    if not seen_targets[t]:
                        scores[t] = target_scores(row)
                        seen_targets[t] = True
                if number % 250000 == 0:
                    log(f"{name}: {number:,} rows")
        sources[name] = {"rows": number, "sha256": digest.hexdigest()}
        if name != "leftout":
            assert digest.hexdigest() == source_audit[path.name]["sha256"]
        else:
            assert number == left_count
            raw_leftout = root / "manakov/leftout.tsv"
            assert sha(raw_leftout) == source_audit["leftout.tsv"]["sha256"]
            with raw_leftout.open() as raw:
                original = csv.DictReader(raw, delimiter="\t")
                with path.open() as prepared:
                    for raw_row, prepared_row in zip(
                        original, csv.DictReader(prepared), strict=True
                    ):
                        for col in ["gene", "noncodingRNA"]:
                            assert raw_row[col].upper().replace(
                                "U", "T"
                            ) == prepared_row[col].upper().replace("U", "T")
                        assert int(raw_row["label"]) == int(prepared_row["label"])
        log(f"Finished {name}: {number:,} rows")
    assert seen_rows.all() and seen_targets.all() and np.isin(labels, [0, 1]).all()
    scores.flush()
    target_bases.flush()
    for name, array in [
        ("mirna_bases", mirna_bases),
        ("row_mirna", row_mirna),
        ("row_target", row_target),
        ("labels", labels),
    ]:
        np.save(out / f"{name}.npy", array)
    with np.load(folds / "assignments.npz") as assignments:
        np.save(out / "roles.npy", assignments["roles"])
    metadata = {
        "pool_rows": n,
        "leftout_rows": left_count,
        "rows": len(labels),
        "unique_pool_target_windows": nt,
        "sources": sources,
        "assignments_sha256": sha(folds / "assignments.npz"),
        "base_order": "ATCG; 4 is unknown/padding",
        "target_channel_order": ["A", "C", "G", "T", "missing"],
        "missing_fill": 0.5,
        "score_storage": "float16, matching stable interaction_cache.py",
        "row_order": "Original train, original test, original leftout; same order as fold assignments for pool rows",
        "script_sha256": sha(Path(__file__)),
    }
    (out / "ready.json").write_text(json.dumps(metadata, indent=2) + "\n")
    log(
        "Cache ready; source IDs, labels, source pointers and pooled CSV hashes verified"
    )

    return out / "ready.json"


class FoldInputs:
    def __init__(
        self, cache_dir: Path, folds_dir: Path, variant: str, *, device="cuda"
    ):
        if variant not in {"seq_only", "target_phact"}:
            raise ValueError(f"Unknown model variant: {variant}")
        cache = Path(cache_dir)
        self.cache_dir = cache.resolve()
        self.device = torch.device(device)
        self.metadata = json.loads((cache / "ready.json").read_text())
        if (
            sha(Path(folds_dir) / "assignments.npz")
            != self.metadata["assignments_sha256"]
        ):
            raise ValueError("Cache and fold assignments do not match")
        self.variant = variant
        for name in [
            "mirna_bases",
            "target_bases",
            "row_mirna",
            "row_target",
            "labels",
        ]:
            values = np.load(cache / f"{name}.npy")
            tensor = torch.from_numpy(values).to(self.device)
            if name == "labels":
                tensor = tensor.float()
            setattr(self, name, tensor)
        self.target_scores = None
        if variant == "target_phact":
            self.target_scores = torch.from_numpy(
                np.load(cache / "target_scores.npy")
            ).to(self.device)
        self.roles = np.load(cache / "roles.npy")
        with np.load(Path(folds_dir) / "assignments.npz") as assignments:
            if not np.array_equal(self.roles, assignments["roles"]):
                raise ValueError("Cached row roles differ from fold assignments")

    def batch(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        m = self.mirna_bases[self.row_mirna[indices].long()].long()
        target_rows = self.row_target[indices].long()
        t = self.target_bases[target_rows].long()
        pairs = m[:, :, None] * 4 + t[:, None, :]
        pairs.masked_fill_(
            (m[:, :, None] >= 4) | (t[:, None, :] >= 4), PADDING_PAIR_INDEX
        )
        labels = self.labels[indices]
        if self.variant == "seq_only":
            return pairs, labels
        scores = (
            self.target_scores[target_rows]
            .permute(0, 2, 1)
            .unsqueeze(2)
            .expand(-1, -1, 28, -1)
        )
        return pairs, scores, labels


class FoldBatches:
    """Prebatched iterator compatible with the stable training/evaluation loop."""

    def __init__(self, inputs, indices, batch_size, *, shuffle=False, seed=42, name=""):
        self.inputs = inputs
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.name = name
        self.iteration = 0
        self.dataset = self
        self.manifest = {"row_count": len(indices)}
        self.has_sample_weights = False

    def __iter__(self):
        self.iteration += 1
        order = self.indices
        if self.shuffle:
            order = np.random.default_rng(
                self.seed + 10000 + self.iteration
            ).permutation(order)
        start = time.monotonic()
        for i in range(0, len(order), self.batch_size):
            yield self.inputs.batch(order[i : i + self.batch_size])
            if self.shuffle and (i // self.batch_size + 1) % 500 == 0:
                print(
                    f"{self.name} epoch {self.iteration}: {min(i + self.batch_size, len(order)):,}/{len(order):,}; "
                    f"{time.monotonic() - start:.1f}s",
                    flush=True,
                )
