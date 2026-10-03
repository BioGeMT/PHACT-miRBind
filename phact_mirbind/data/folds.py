"""Create fixed conservation-balanced outer miRNA folds and per-run row roles."""

import csv
import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from phact_mirbind.data.fold_features import sha

ROLE_NAMES = ["train", "validation", "seen_test", "unseen_test"]


def log(message):
    print(message, flush=True)


def quintiles(values):
    finite = np.isfinite(values)
    cuts = (
        np.quantile(values[finite], [0.2, 0.4, 0.6, 0.8])
        if finite.any()
        else np.array([])
    )
    codes = np.full(len(values), 5, dtype=np.int8)
    codes[finite] = np.searchsorted(cuts, values[finite], side="right")
    return codes, cuts.tolist()


def build_features(data):
    m = data["mirna"]
    t = data["target"]
    n = len(data["mirna_sequences"])
    row_counts = np.bincount(m, minlength=n)
    blocks = []
    weights = []
    names = []
    definitions = {}

    def add(name, values, weight):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        active = values.sum(axis=0) > 0
        values = values[:, active]
        blocks.append(values)
        if values.shape[1]:
            weights.extend([weight / values.shape[1]] * values.shape[1])
        names.extend(f"{name}_{i}" for i in np.flatnonzero(active))
        definitions[name] = {"total_weight": weight, "columns": int(values.shape[1])}

    def aggregate(codes, row_weights=None):
        x = np.zeros((n, 6))
        np.add.at(x, (m, codes), 1 if row_weights is None else row_weights)
        return x

    add("interaction_count", row_counts, 20)
    add("positive_count", np.bincount(m, weights=data["label"], minlength=n), 2)
    add("mirna_count", np.ones(n), 2)
    mirna_bins, mirna_cuts = quintiles(data["mirna_conservation"][:, 0])
    add("mirna_conservation_unique", np.eye(6)[mirna_bins], 3)
    row_bins = mirna_bins[m].copy()
    row_bins[data["row_profile"] < 0] = 5
    add("mirna_conservation_rows", aggregate(row_bins), 3)
    target_counts = np.bincount(t, minlength=len(data["target_features"]))
    inverse_counts = 1 / target_counts[t]
    target_cuts = {}
    for k, (name, weight) in enumerate([("mean", 6), ("std", 2), ("q90", 2)]):
        bins, cuts = quintiles(data["target_features"][:, k])
        target_cuts[name] = cuts
        add("target_" + name + "_rows", aggregate(bins[t]), weight)
        # Each genomic window contributes total weight one across the whole pool.
        # Fold diagnostics separately count actual distinct windows per fold.
        add(
            "target_" + name + "_window_weighted",
            aggregate(bins[t], inverse_counts),
            weight / 2,
        )
    add(
        "missing_target_rows",
        np.bincount(m, weights=data["target_features"][t, 3] == 0, minlength=n),
        2,
    )
    add(
        "missing_mirna_rows",
        np.bincount(m, weights=data["row_profile"] < 0, minlength=n),
        2,
    )
    return (
        np.concatenate(blocks, axis=1),
        np.array(weights),
        {
            "feature_names": names,
            "feature_blocks": definitions,
            "mirna_quintile_cutpoints": mirna_cuts,
            "target_quintile_cutpoints": target_cuts,
            "cutpoint_population": "Unique scored miRNAs for miRNA cuts; unique scored genomic windows for target cuts.",
            "missing_bin": 5,
            "bins": "0..4 are increasing conservation quintiles; 5 is unavailable.",
            "target_window_weighting": "Each interaction has inverse total occurrence count of its genomic window; actual distinct-window distributions are also reported.",
            "class_conditional_score_distributions": "Not optimized; positive/negative diagnostics are reported separately.",
        },
    )


def assign_groups(x, weights, proportions, seed, restarts, steps, inner=False):
    rng = np.random.default_rng(seed)
    total = x.sum(axis=0)
    active = total > 0
    z = x[:, active] / total[active]
    w = weights[active]
    proportions = np.array(proportions)
    k = len(proportions)
    n = len(x)
    allowed = np.ones((n, k), dtype=bool)
    if inner:
        # Prevent early stopping from being dominated by a single huge miRNA.
        allowed[:, 1] = x[:, 0] <= 0.4 * total[0] * proportions[1]
        assert allowed[:, 1].sum() > 10

    def cost(loads, props):
        relative = loads / np.asarray(props)[..., None] - 1
        base = np.sum(relative**2 * w, axis=-1)
        return base + 10000 * np.maximum(np.abs(relative[..., 0]) - 0.03, 0) ** 2

    best = None
    for attempt in range(restarts):
        loads = np.zeros((k, z.shape[1]))
        assignment = np.full(n, -1, dtype=np.int8)
        order = np.argsort(
            -x[:, 0] * rng.lognormal(0, 0.7 if inner else 0.15, n), kind="stable"
        )
        for i in order:
            delta = cost(loads + z[i], proportions) - cost(loads, proportions)
            delta[~allowed[i]] = np.inf
            f = int(np.argmin(delta + rng.uniform(0, 1e-10, k)))
            assignment[i] = f
            loads[f] += z[i]
        for step in range(steps):
            i = int(rng.integers(n))
            a = int(assignment[i])
            if rng.random() < 0.6:
                j = int(rng.integers(n))
                b = int(assignment[j])
                if a == b or not allowed[i, b] or not allowed[j, a]:
                    continue
                change = z[j] - z[i]
            else:
                j = None
                b = int(rng.integers(k))
                if a == b or not allowed[i, b]:
                    continue
                change = -z[i]
            proposal = np.array([loads[a] + change, loads[b] - change])
            props = proportions[[a, b]]
            improvement = cost(proposal, props).sum() - cost(loads[[a, b]], props).sum()
            if improvement < -1e-12:
                loads[[a, b]] = proposal
                assignment[i] = b
                if j is not None:
                    assignment[j] = a
        value = float(cost(loads, proportions).sum())
        if best is None or value < best[0]:
            best = (value, assignment.copy(), loads.copy())
    value, assignment, loads = best
    assert set(assignment) == set(range(k))
    counts = np.bincount(assignment, weights=x[:, 0], minlength=k)
    deviation = counts / (counts.sum() * proportions) - 1
    log(
        f"Balanced {n} groups into {k}: rows {counts.astype(int).tolist()}, deviations {np.round(deviation, 4).tolist()}"
    )
    return assignment, {
        "objective": value,
        "seed": seed,
        "restarts": restarts,
        "local_steps_per_restart": steps,
        "ordering_lognormal_sigma": 0.7 if inner else 0.15,
        "row_counts": counts.astype(int).tolist(),
        "relative_row_deviation": deviation.tolist(),
        "within_3_percent_row_target": bool((np.abs(deviation) <= 0.03).all()),
        "inner_max_single_group_fraction": 0.4 if inner else None,
    }


def distribution(values):
    finite = values[np.isfinite(values)]
    return {
        "n": len(values),
        "scored": len(finite),
        "missing_fraction": float(1 - len(finite) / max(len(values), 1)),
        "mean": float(finite.mean()) if len(finite) else None,
        "quantiles": np.quantile(finite, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]).tolist()
        if len(finite)
        else [],
    }


def summarize(data, index):
    m = data["mirna"][index]
    t = data["target"][index]
    y = data["label"][index]
    unique_m = np.unique(m)
    unique_t = np.unique(t)
    result = {
        "rows": len(index),
        "mirnas": len(unique_m),
        "target_windows": len(unique_t),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "positive_fraction": float(y.mean()),
        "mirna_missing_rows": int((data["row_profile"][index] < 0).sum()),
        "target_missing_rows": int((data["target_features"][t, 3] == 0).sum()),
        "mirna_conservation_unique": distribution(
            data["mirna_conservation"][unique_m, 0]
        ),
        "mirna_conservation_rows": distribution(data["mirna_conservation"][m, 0]),
        "mirna_seed_conservation_unique": distribution(
            data["mirna_conservation"][unique_m, 1]
        ),
        "target_conservation_rows": {},
        "target_conservation_unique": {},
        "target_by_label": {},
    }
    for k, name in enumerate(["mean", "std", "q90", "coverage"]):
        result["target_conservation_rows"][name] = distribution(
            data["target_features"][t, k]
        )
        result["target_conservation_unique"][name] = distribution(
            data["target_features"][unique_t, k]
        )
    for value in [0, 1]:
        result["target_by_label"][str(value)] = distribution(
            data["target_features"][t[y == value], 0]
        )
    return result


def build_conservation_folds(output_dir: Path, *, seed: int = 43) -> Path:
    out = Path(output_dir).resolve()
    started = time.monotonic()
    audit = json.loads((out / "feature_audit.json").read_text())
    original_train_rows = audit.get("original_split_counts", {}).get("train")
    if original_train_rows is None:
        original_train_rows = next(
            info["rows"]
            for path, info in audit["sources"].items()
            if Path(path).name == "train.tsv"
        )

    if any(
        (out / name).exists()
        for name in ["assignments.npz", "interaction_manifest.tsv.gz", "metadata.json"]
    ):
        raise FileExistsError(f"Fold artifacts already exist in {out}")
    with np.load(out / "features.npz") as f:
        data = {k: f[k] for k in f.files}
    n = len(data["label"])
    ng = len(data["mirna_sequences"])
    x, w, feature_definitions = build_features(data)
    outer, opt = assign_groups(x, w, [0.2] * 5, seed, 32, 8000)
    # Smaller validation cohorts need stronger availability balancing and a
    # broader search order to avoid repeatedly selecting one large miRNA.
    inner_weights = w.copy()
    for j, name in enumerate(feature_definitions["feature_names"]):
        if name.startswith("missing_mirna_rows"):
            inner_weights[j] = 40
        if name.startswith("missing_target_rows"):
            inner_weights[j] = 10
    roles = np.zeros((n, 5), dtype=np.uint8)
    pair_mirna = np.full(len(data["pair_counts"]), -1, dtype=np.int16)
    pair_min = np.full(len(data["pair_counts"]), 2, dtype=np.int8)
    pair_max = np.full(len(data["pair_counts"]), -1, dtype=np.int8)
    np.maximum.at(pair_mirna, data["pair"], data["mirna"])
    np.minimum.at(pair_min, data["pair"], data["label"])
    np.maximum.at(pair_max, data["pair"], data["label"])
    pair_label = np.where(pair_min == pair_max, pair_min, 2)
    guide_pairs = [np.flatnonzero(pair_mirna == m) for m in range(ng)]
    inner_assignment = np.zeros((ng, 5), dtype=np.uint8)
    diagnostics = {}
    inner_options = {}
    for fold in range(5):
        held = outer == fold
        remain = np.flatnonzero(~held)
        inner, options = assign_groups(
            x[remain],
            inner_weights,
            [0.9, 0.1],
            seed + 100 + fold,
            48,
            8000,
            inner=True,
        )
        val_m = remain[inner == 1]
        fit_m = remain[inner == 0]
        inner_assignment[val_m, fold] = 1
        inner_assignment[held, fold] = 3
        roles[:, fold] = inner_assignment[data["mirna"], fold]
        rng = np.random.default_rng(seed + 1000 + fold)
        seen_pairs = np.zeros(len(data["pair_counts"]), dtype=bool)
        for m in fit_m:
            pairs = guide_pairs[m]
            if len(pairs) < 20:
                continue
            for label_value in [0, 1, 2]:
                eligible = pairs[pair_label[pairs] == label_value]
                if len(eligible) < 2:
                    continue
                target = 0.05 * data["pair_counts"][eligible].sum()
                selected = 0
                number = 0
                for p in rng.permutation(eligible):
                    if number >= len(eligible) - 1:
                        break
                    size = int(data["pair_counts"][p])
                    if abs(selected + size - target) <= abs(selected - target):
                        seen_pairs[p] = True
                        selected += size
                        number += 1
                    if selected >= target:
                        break
        roles[seen_pairs[data["pair"]], fold] = 2
        assert (roles[held[data["mirna"]], fold] == 3).all()
        assert set(data["mirna"][roles[:, fold] == 2]) <= set(
            data["mirna"][roles[:, fold] == 0]
        )
        for code in [1, 3]:
            assert not set(data["mirna"][roles[:, fold] == code]) & set(
                data["mirna"][roles[:, fold] == 0]
            )
        pmin = np.full(len(pair_mirna), 4, dtype=np.uint8)
        pmax = np.zeros(len(pair_mirna), dtype=np.uint8)
        np.minimum.at(pmin, data["pair"], roles[:, fold])
        np.maximum.at(pmax, data["pair"], roles[:, fold])
        assert np.array_equal(pmin, pmax), "Exact pair crosses roles"
        counts = {
            name: int((roles[:, fold] == code).sum())
            for code, name in enumerate(ROLE_NAMES)
        }
        log(f"Fold {fold + 1} roles: {counts}")
        diagnostic = {
            name: summarize(data, np.flatnonzero(roles[:, fold] == code))
            for code, name in enumerate(ROLE_NAMES)
        }
        train_rows = np.flatnonzero(roles[:, fold] == 0)
        training_targets = set(data["target"][train_rows])
        training_sequences = set(data["target_keys"][list(training_targets)].tolist())
        training_target_seqs = {key.rsplit("|", 1)[1] for key in training_sequences}
        training_seeds = {
            data["mirna_sequences"][m][1:8]
            for m in np.unique(data["mirna"][train_rows])
        }
        for code, name in enumerate(ROLE_NAMES[1:], 1):
            idx = np.flatnonzero(roles[:, fold] == code)
            targets = data["target"][idx]
            exact = np.array([int(t) in training_targets for t in targets])
            shared_seq = np.array(
                [
                    data["target_keys"][t].rsplit("|", 1)[1] in training_target_seqs
                    for t in targets
                ]
            )
            shared_seed = np.array(
                [
                    data["mirna_sequences"][m][1:8] in training_seeds
                    for m in data["mirna"][idx]
                ]
            )
            diagnostic[name]["overlap_with_actual_training"] = {
                "rows_sharing_target_window": int(exact.sum()),
                "fraction_sharing_target_window": float(exact.mean()),
                "rows_sharing_exact_target_sequence": int(shared_seq.sum()),
                "fraction_sharing_exact_target_sequence": float(shared_seq.mean()),
                "rows_sharing_seed_2_8": int(shared_seed.sum()),
                "fraction_sharing_seed_2_8": float(shared_seed.mean()),
            }
        diagnostics[str(fold + 1)] = diagnostic
        inner_options[str(fold + 1)] = options
    assert np.all((roles == 3).sum(axis=1) == 1)
    np.savez_compressed(
        out / "assignments.npz",
        outer_fold_by_mirna=outer + 1,
        roles=roles,
        inner_assignment=inner_assignment,
    )
    log("Writing compact row manifest")
    with gzip.open(out / "interaction_manifest.tsv.gz", "wt") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "id",
                "label",
                "source_csv",
                "source_row_1based",
                "mirna_group",
                "target_window_id",
                "sequence_pair_id",
                "outer_fold",
                *[f"run_{i}_role" for i in range(1, 6)],
            ]
        )
        file_names = ["train.csv", "validation.csv", "test.csv"]
        for i in range(n):
            original = "train" if i < original_train_rows else "test"
            row_id = i + 1 if original == "train" else i - original_train_rows + 1
            writer.writerow(
                [
                    f"manakov_{original}_{row_id}",
                    int(data["label"][i]),
                    file_names[data["source_file"][i]],
                    int(data["source_row"][i]),
                    int(data["mirna"][i]) + 1,
                    int(data["target"][i]) + 1,
                    int(data["pair"][i]) + 1,
                    int(outer[data["mirna"][i]]) + 1,
                    *[ROLE_NAMES[r] for r in roles[i]],
                ]
            )
    with (out / "mirna_folds.tsv").open("w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "mirna_group",
                "sequence",
                "outer_fold",
                "interactions",
                "positive",
                "conservation_mean",
                "seed_conservation_mean",
                *[f"run_{i}_group_role" for i in range(1, 6)],
            ]
        )
        for m, seq in enumerate(data["mirna_sequences"]):
            writer.writerow(
                [
                    m + 1,
                    seq,
                    int(outer[m]) + 1,
                    int(x[m, 0]),
                    int(np.sum(data["label"][data["mirna"] == m])),
                    *[
                        float(v) if np.isfinite(v) else "NaN"
                        for v in data["mirna_conservation"][m]
                    ],
                    *[
                        "fit_and_seen_test" if code == 0 else ROLE_NAMES[code]
                        for code in inner_assignment[m]
                    ],
                ]
            )
    with gzip.open(out / "target_conservation.tsv.gz", "wt") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "target_window_id",
                "chromosome",
                "start",
                "end",
                "strand",
                "sequence",
                "mean_ref_minus_mean_alt",
                "std_ref_minus_mean_alt",
                "q90_ref_minus_mean_alt",
                "coverage",
            ]
        )
        for t, key in enumerate(data["target_keys"]):
            writer.writerow(
                [
                    t + 1,
                    *key.split("|"),
                    *[
                        float(v) if np.isfinite(v) else "NaN"
                        for v in data["target_features"][t]
                    ],
                ]
            )
    with (out / "fold_summary.tsv").open("w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "fold",
                "role",
                "rows",
                "mirnas",
                "positive",
                "negative",
                "mirna_missing_rows",
                "target_missing_rows",
                "mean_mirna_conservation_unique",
                "mean_target_conservation_rows",
                "mean_target_conservation_unique",
            ]
        )
        for fold, diagnostic in diagnostics.items():
            for name, d in diagnostic.items():
                writer.writerow(
                    [
                        fold,
                        name,
                        *[
                            d[k]
                            for k in [
                                "rows",
                                "mirnas",
                                "positive",
                                "negative",
                                "mirna_missing_rows",
                                "target_missing_rows",
                            ]
                        ],
                        d["mirna_conservation_unique"]["mean"],
                        d["target_conservation_rows"]["mean"]["mean"],
                        d["target_conservation_unique"]["mean"]["mean"],
                    ]
                )
    metadata = {
        "seed": seed,
        "outer_fold_count": 5,
        "outer_group": "Exact mature miRNA sequence",
        "scope": "Original Manakov train plus original test, with former validation rejoined to original train. Original leftout excluded and unchanged.",
        "outer_optimizer": opt,
        "inner_optimizers": inner_options,
        "balance_features": feature_definitions,
        "inner_feature_weight_overrides": {
            "missing_mirna_rows": 40,
            "missing_target_rows": 10,
        },
        "algorithm": "Seeded largest-first greedy assignment plus improving single-group moves and pairwise swaps; best balance objective across fixed restarts. No model outcomes used.",
        "row_balance": "Soft penalty beyond 3% deviation from requested row proportions; residual deviations reported.",
        "inner_validation": "Approximately 10% of the remaining outer training pool, whole miRNAs; no single validation miRNA may exceed 40% of the requested validation row count.",
        "seen_test": "Approximately 5% of remaining rows within eligible fitting miRNAs, whole exact sequence pairs stratified by label; miRNAs need >=20 distinct pairs and at least one pair per label stratum remains for training.",
        "role_codes": dict(enumerate(ROLE_NAMES)),
        "original_data_modified": False,
        "original_leftout_used_for_balance": False,
        "duplicate_policy": "Original labels retained; exact miRNA/target sequence pairs cannot cross roles. Conflicts counted in feature_audit.json.",
        "model_selection": "Use validation AUPRC for early stopping and tuning; seen_test/unseen_test are evaluation-only. Use identical manifests for every model/seed.",
        "pretraining": "Any supervised miRBind or DiscrimAlign pretraining must exclude that run's held-out rows/groups for independent evaluation.",
        "target_disjointness": "Targets are not held out as groups; exact window/sequence overlap is explicitly reported.",
        "diagnostics": diagnostics,
        "script": str(Path(__file__).resolve()),
        "elapsed_seconds": time.monotonic() - started,
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    log("SUCCESS: folds and audits written; no training started")
    return out / "assignments.npz"


def verify_conservation_folds(output_dir: Path, data_root: Path) -> dict:
    """Independently check saved assignments and every serialized row reference."""
    out = Path(output_dir).resolve()
    root = Path(data_root).resolve()
    with np.load(out / "features.npz") as f:
        data = {k: f[k] for k in f.files}
    with np.load(out / "assignments.npz") as f:
        outer = f["outer_fold_by_mirna"]
        roles = f["roles"]
    metadata = json.loads((out / "metadata.json").read_text())
    audit = json.loads((out / "feature_audit.json").read_text())
    original_train_rows = next(
        v["rows"] for k, v in audit["sources"].items() if Path(k).name == "train.tsv"
    )
    n = len(data["label"])
    m = data["mirna"]
    t = data["target"]
    pair = data["pair"]
    assert roles.shape == (n, 5) and np.isin(roles, [0, 1, 2, 3]).all()
    assert np.all((roles == 3).sum(axis=1) == 1)
    assert set(outer) == set(range(1, 6))
    counts = {}
    for f in range(5):
        assert np.array_equal(roles[:, f] == 3, outer[m] == f + 1)
        sets = {code: set(m[roles[:, f] == code]) for code in range(4)}
        assert sets[2] <= sets[0]
        assert not (sets[0] | sets[2]) & sets[1]
        assert not (sets[0] | sets[1] | sets[2]) & sets[3]
        low = np.full(len(data["pair_counts"]), 4, dtype=np.uint8)
        high = np.zeros(len(data["pair_counts"]), dtype=np.uint8)
        np.minimum.at(low, pair, roles[:, f])
        np.maximum.at(high, pair, roles[:, f])
        assert np.array_equal(low, high)
        counts[str(f + 1)] = {}
        for code, name in enumerate(ROLE_NAMES):
            mask = roles[:, f] == code
            record = metadata["diagnostics"][str(f + 1)][name]
            assert (
                int(mask.sum()) == record["rows"]
                and int(data["label"][mask].sum()) == record["positive"]
            )
            assert len(sets[code]) == record["mirnas"]
            assert np.unique(t[mask]).size == record["target_windows"]
            counts[str(f + 1)][name] = record["rows"]
    # Verify the serialized, human-readable manifest against compact arrays.
    offset = 0
    file_names = np.array(["train.csv", "validation.csv", "test.csv"])
    role_names = np.array(ROLE_NAMES)
    for frame in pd.read_csv(
        out / "interaction_manifest.tsv.gz", sep="\t", chunksize=250000
    ):
        stop = offset + len(frame)
        idx = np.arange(offset, stop)
        expected_ids = [
            f"manakov_train_{i + 1}"
            if i < original_train_rows
            else f"manakov_test_{i - original_train_rows + 1}"
            for i in idx
        ]
        assert frame["id"].tolist() == expected_ids
        assert np.array_equal(frame["label"], data["label"][idx])
        assert np.array_equal(frame["source_csv"], file_names[data["source_file"][idx]])
        assert np.array_equal(frame["source_row_1based"], data["source_row"][idx])
        for col, expected in [
            ("mirna_group", m[idx] + 1),
            ("target_window_id", t[idx] + 1),
            ("sequence_pair_id", pair[idx] + 1),
            ("outer_fold", outer[m[idx]]),
        ]:
            assert np.array_equal(frame[col], expected)
        for f in range(5):
            assert np.array_equal(frame[f"run_{f + 1}_role"], role_names[roles[idx, f]])
        offset = stop
    assert offset == n
    leftout = root / "manakov/leftout.tsv"
    assert sha(leftout) == next(
        v["sha256"]
        for k, v in audit["sources"].items()
        if Path(k).name == "leftout.tsv"
    )
    with leftout.open() as f:
        left_sequences = {
            r["noncodingRNA"].upper().replace("U", "T")
            for r in csv.DictReader(f, delimiter="\t")
        }
    assert not left_sequences & set(data["mirna_sequences"])

    result = {
        "status": "passed",
        "rows": n,
        "mirnas": len(outer),
        "original_leftout_unchanged": True,
        "rows_by_fold_and_role": counts,
        "assignments_sha256": sha(out / "assignments.npz"),
    }
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
