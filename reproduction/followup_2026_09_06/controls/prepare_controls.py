"""Create fixed, label-blind positional PHACT controls beside existing caches."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json

import numpy as np
import torch


CONTROL_SEED = 20260906


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_sequences(pairs):
    """Decode model-visible A,T,C,G IDs; retain unknown/padded positions as 4."""
    if pairs.ndim != 3 or not np.all((pairs < 16) | (pairs == 17)):
        raise ValueError("Expected pair grids containing only 0..15 and padding 17")
    rows = np.arange(len(pairs))
    valid = pairs < 16
    mirna_valid = valid.any(axis=2)
    target_valid = valid.any(axis=1)
    first_target = target_valid.argmax(axis=1)
    first_mirna = mirna_valid.argmax(axis=1)
    mirna = np.where(mirna_valid, pairs[rows, :, first_target] // 4, 4)
    target = np.where(target_valid, pairs[rows, first_mirna, :] % 4, 4)
    expected = np.where(
        mirna_valid[:, :, None] & target_valid[:, None, :],
        mirna[:, :, None] * 4 + target[:, None, :],
        17,
    )
    if not np.array_equal(pairs, expected):
        raise ValueError("Pair grid does not encode a consistent sequence pair")
    return {"mirna": mirna.astype(np.uint8), "target": target.astype(np.uint8)}


def position_permutation(sequence, missing, axis, seed=CONTROL_SEED):
    """One reproducible permutation per axis, represented sequence, and mask."""
    key = (
        f"phact-within-base-position-control-v2:{seed}:{axis}:".encode("ascii")
        + sequence.tobytes() + missing.tobytes()
    )
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(key).digest(), "little")))
    permutation = np.arange(len(sequence), dtype=np.uint8)
    for reference_base in range(4):
        active = np.flatnonzero((sequence == reference_base) & (missing == 0))
        permutation[active] = rng.permutation(active)
    return permutation


def reference_contrast(scores, sequences):
    scores = scores.astype(np.float32)
    # Pair-grid base IDs are A,T,C,G; PHACT channels are A,C,G,T.
    channels = np.array([0, 3, 1, 2, 0], dtype=np.uint8)[sequences]
    reference = np.take_along_axis(scores, channels[:, :, None], axis=2)[:, :, 0]
    return reference - (scores.sum(axis=2) - reference) / 3.0


def shuffle_profiles(scores, missing, sequences, axis, permutation_cache):
    if scores.dtype != np.float16 or scores.shape[2] != 4:
        raise ValueError("Expected float16 A,C,G,T quartets")
    if scores.shape[:2] != sequences.shape or missing.shape != sequences.shape + (1,):
        raise ValueError("Score, sequence, and mask shapes disagree")
    missing = missing[:, :, 0]
    if not np.isfinite(scores).all() or not np.all((missing == 0) | (missing == 1)):
        raise ValueError("Nonfinite scores or nonbinary missingness mask")
    if not np.all(scores[missing == 1] == 0.5):
        raise ValueError("Missing score values do not equal the declared fill value 0.5")

    keys = np.ascontiguousarray(np.concatenate((sequences, missing), axis=1))
    unique, inverse = np.unique(keys.view(f"V{keys.shape[1]}").ravel(), return_inverse=True)
    length = sequences.shape[1]
    unique_permutations = np.empty((len(unique), length), dtype=np.uint8)
    for index, opaque_key in enumerate(unique):
        key = opaque_key.tobytes()
        cache_key = (axis, key)
        permutation = permutation_cache.get(cache_key)
        if permutation is None:
            sequence_and_mask = np.frombuffer(key, dtype=np.uint8)
            permutation = position_permutation(
                sequence_and_mask[:length], sequence_and_mask[length:], axis
            )
            permutation_cache[cache_key] = permutation
        unique_permutations[index] = permutation
    permutations = unique_permutations[inverse]
    shuffled = scores[np.arange(len(scores))[:, None], permutations, :]

    active = (sequences < 4) & (missing == 0)
    if not np.array_equal(shuffled[~active], scores[~active]):
        raise AssertionError("Missing, ambiguous, or padded positions changed")
    if not np.all(np.sort(permutations, axis=1) == np.arange(length)):
        raise AssertionError("Position map is not a bijection")
    # A quartet is eight raw bytes: compare complete quartets, never sorted channels.
    before = np.ascontiguousarray(scores).view(np.uint64).reshape(len(scores), length)
    after = np.ascontiguousarray(shuffled).view(np.uint64).reshape(len(scores), length)
    if not np.array_equal(np.sort(before, axis=1), np.sort(after, axis=1)):
        raise AssertionError("A/C/G/T quartet multiset changed")
    if not np.array_equal(sequences[np.arange(len(scores))[:, None], permutations], sequences):
        raise AssertionError("A quartet moved to a different reference nucleotide")
    singleton_positions = 0
    for reference_base in range(4):
        base_positions = (sequences == reference_base) & active
        if not np.array_equal(np.sort(np.where(base_positions, before, 0), axis=1),
                              np.sort(np.where(base_positions, after, 0), axis=1)):
            raise AssertionError("Reference-base-conditional quartet multiset changed")
        singleton_positions += int((base_positions.sum(axis=1) == 1).sum())
    contrasts_before = reference_contrast(scores, sequences)
    contrasts_after = reference_contrast(shuffled, sequences)
    if not np.array_equal(np.sort(contrasts_before, axis=1), np.sort(contrasts_after, axis=1)):
        raise AssertionError("Reference-minus-alternative contrast multiset changed")
    if not np.isfinite(shuffled).all():
        raise AssertionError("Nonfinite output score")

    active_count = active.sum(axis=1)
    first_active = active.argmax(axis=1)
    reference = scores[np.arange(len(scores)), first_active, :]
    variable = (np.any(scores != reference[:, None, :], axis=2) & active).any(axis=1)
    changed = np.any(shuffled != scores, axis=(1, 2))
    if np.any(changed & ~variable):
        raise AssertionError("Positionally constant or wholly unavailable profile changed")
    contrast_range = (np.where(active, contrasts_before, -np.inf).max(axis=1)
                      - np.where(active, contrasts_before, np.inf).min(axis=1))
    contrast_flat = (active_count > 0) & (contrast_range <= 1e-6)
    return shuffled, {
        "rows": len(scores),
        "unique_keys_in_shard": len(unique),
        "changed_rows": int(changed.sum()),
        "eligible_variable_rows": int(variable.sum()),
        "eligible_constant_rows": int(((active_count > 0) & ~variable).sum()),
        "no_eligible_positions_rows": int((active_count == 0).sum()),
        "missing_positions": int((missing == 1).sum()),
        "scored_invalid_positions_preserved": int(((sequences == 4) & (missing == 0)).sum()),
        "eligible_positions": int(active_count.sum()),
        "moved_eligible_positions": int(((permutations != np.arange(length)) & active).sum()),
        "singleton_reference_base_positions_unchanged": singleton_positions,
        "reference_contrast_flat_rows": int(contrast_flat.sum()),
        "reference_contrast_flat_rows_with_quartet_changes": int((contrast_flat & changed).sum()),
    }


def prepare_split(cache_root, output_root, split, permutation_cache):
    base_manifest = cache_root / split / f"{split}_manifest.json"
    source = json.loads(base_manifest.read_text())
    destination = output_root / split
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "manifest.json").exists():
        raise FileExistsError(destination / "manifest.json")
    manifest = {
        "format_version": 1,
        "control": "fixed_label_blind_within_reference_base_position_quartet_shuffle",
        "control_seed": CONTROL_SEED,
        "source_cache": str(cache_root),
        "source_manifest": str(base_manifest),
        "source_manifest_sha256": sha256_file(base_manifest),
        "split": split,
        "row_count": source["row_count"],
        "channel_order": ["A", "C", "G", "T"],
        "key": "axis + model-visible sequence (A,T,C,G,unknown/padding) + missingness mask",
        "shuffled_positions": "within each reference-base group; valid encoded base and missingness mask == 0",
        "preserved": ["whole nucleotide quartets", "reference-base-conditional quartet multisets",
                      "reference-minus-alternative contrast multiset", "missing positions", "unknown/padded positions"],
        "shards": [],
    }
    for item in source["shards"]:
        base_path = base_manifest.parent / item["file"]
        sidecar_path = destination / item["file"]
        if sidecar_path.exists():
            raise FileExistsError(sidecar_path)
        source_hash = sha256_file(base_path)
        base = torch.load(base_path, map_location="cpu", weights_only=True)
        sequences = decode_sequences(base["pair_indices"].numpy())
        sidecar, audit = {}, {}
        for axis in ("mirna", "target"):
            field = f"{axis}_phact"
            shuffled, audit[axis] = shuffle_profiles(
                base[field].numpy(), base[f"{field}_missing"].numpy(),
                sequences[axis], axis, permutation_cache,
            )
            sidecar[field] = torch.from_numpy(shuffled)
        torch.save(sidecar, sidecar_path)
        reloaded = torch.load(sidecar_path, map_location="cpu", weights_only=True)
        if set(reloaded) != {"mirna_phact", "target_phact"}:
            raise AssertionError("Unexpected sidecar fields")
        for field in sidecar:
            if not torch.equal(sidecar[field], reloaded[field]):
                raise AssertionError("Saved sidecar differs from validated tensors")
        if source_hash != sha256_file(base_path):
            raise AssertionError("Source shard changed while creating the control")
        entry = {
            "base_shard": item["file"], "sidecar": sidecar_path.name,
            "rows": item["rows"], "base_sha256": source_hash,
            "sha256": sha256_file(sidecar_path), "audit": audit,
        }
        manifest["shards"].append(entry)
        print(json.dumps({"split": split, **entry}), flush=True)
    if sum(shard["rows"] for shard in manifest["shards"]) != source["row_count"]:
        raise AssertionError("Manifest row count mismatch")
    manifest["source_manifest_unchanged"] = sha256_file(base_manifest) == manifest["source_manifest_sha256"]
    if not manifest["source_manifest_unchanged"]:
        raise AssertionError("Source manifest changed")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    permutation_cache = {}
    manifests = [prepare_split(args.cache_root, args.output_root, split, permutation_cache)
                 for split in ("train", "val", "test", "leftout")]
    summary = {
        "control_seed": CONTROL_SEED,
        "unique_axis_sequence_mask_keys": len(permutation_cache),
        "checks_passed": ["quartet multiset", "finite scores", "frozen missing and invalid positions",
                          "unchanged constant profiles", "reference-base-conditional quartet multisets",
                          "reference-minus-alternative contrast multiset", "singleton base positions unchanged",
                          "sidecar round trip", "source file SHA256 unchanged"],
        "splits": {},
    }
    for manifest in manifests:
        totals = {}
        for axis in ("mirna", "target"):
            metrics = manifest["shards"][0]["audit"][axis]
            totals[axis] = {metric: sum(s["audit"][axis][metric] for s in manifest["shards"])
                            for metric in metrics if metric != "unique_keys_in_shard"}
        summary["splits"][manifest["split"]] = totals
    (args.output_root / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
