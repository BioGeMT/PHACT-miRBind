#!/usr/bin/env python3
"""Run one fixed PHACT follow-up condition using the released CNN and training loop."""

from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workspace import WORKSPACE, historical_path

import argparse
import hashlib
import json

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, IterableDataset

ROOT = WORKSPACE / "runs/followup-2026-09-06/training"
sys.path.insert(0, str(ROOT / "source" / "src"))

from phact_mirbind.cache.manifest import find_manifest
from phact_mirbind.cache.dataloaders import (
    make_conservation_collate, make_phact_collate, pair_collate,
)
from phact_mirbind.models.conservation import PairwiseConservationCNN
from phact_mirbind.models.phact import PairwisePhactCNN
from phact_mirbind.models.seq_only import PairwiseSeqCNN
from phact_mirbind.training.loop import set_seed, train_model

CONDITIONS = (
    "sequence", "guide_real", "guide_shuffled", "target_real",
    "target_shuffled", "conservation_both",
)
PHACT_CACHE = WORKSPACE / "runs/param1-training/cache/param_1_target_score"
EXPECTED_ROWS = {"train": 2245982, "val": 249553, "test": 324171, "leftout": 20054}


def sha256(path):
    with historical_path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class FollowupDataset(IterableDataset):
    """Keep compact tensors in CPU memory; retain the release's epoch permutations."""

    def __init__(self, condition, split, seed, max_rows=None):
        self.condition = condition
        self.split = split
        self.seed = seed
        self.max_rows = max_rows
        self.iteration_count = 0
        self.shuffle_mode = "global" if split == "train" else "none"
        cache_root = ROOT / "conservation_cache" if condition == "conservation_both" else PHACT_CACHE
        self.manifest_path = find_manifest(cache_root / split)
        self.manifest = json.loads(self.manifest_path.read_text())
        self.shards = [(self.manifest_path.parent / row["file"], row["rows"])
                       for row in self.manifest["shards"]]
        if sum(rows for _, rows in self.shards) != EXPECTED_ROWS[split]:
            raise ValueError(f"Unexpected row count: {split}")
        self.tensors = None

    def load(self):
        if self.tensors is not None:
            return
        columns = ["pair_indices", "labels"]
        if self.condition == "conservation_both":
            columns += ["conservation"]
            if self.manifest["conservation_features"] != ["phylop", "phastcons"]:
                raise ValueError("Unexpected conservation channel order")
        elif self.condition != "sequence":
            columns += ["mirna_phact", "target_phact"]
        collected = {column: [] for column in columns}
        row_indices = []
        control_dir = ROOT.parent / "controls" / self.split
        controls = {}
        if self.condition.endswith("shuffled"):
            manifest = json.loads((control_dir / "manifest.json").read_text())
            if manifest["control"] != "fixed_label_blind_within_reference_base_position_quartet_shuffle":
                raise ValueError("Expected the frozen within-reference-base positional control")
            if manifest["source_manifest_sha256"] != sha256(self.manifest_path):
                raise ValueError(f"Control source manifest changed: {self.split}")
            if manifest["split"] != self.split or manifest["row_count"] != EXPECTED_ROWS[self.split]:
                raise ValueError(f"Control split or row count mismatch: {self.split}")
            controls = {row["base_shard"]: row for row in manifest["shards"]}
        remaining = [self.max_rows // 2, self.max_rows // 2] if self.max_rows else None
        offset = 0
        for path, count in self.shards:
            shard = torch.load(path, map_location="cpu", weights_only=True)
            if controls:
                entry = controls[path.name]
                if entry["base_sha256"] != sha256(path):
                    raise ValueError(f"Control source shard changed: {path}")
                sidecar_path = control_dir / entry["sidecar"]
                if sha256(sidecar_path) != entry["sha256"]:
                    raise ValueError(f"Control checksum mismatch: {sidecar_path}")
                sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=True)
                axis = "mirna" if self.condition.startswith("guide") else "target"
                key = f"{axis}_phact"
                if sidecar[key].shape != shard[key].shape or entry["rows"] != count:
                    raise ValueError(f"Control shape mismatch: {path}")
                shard[key] = sidecar[key]
            if remaining is None:
                indices = torch.arange(count)
            else:
                parts = [torch.where(shard["labels"] == label)[0][:remaining[label]]
                         for label in (0, 1)]
                for label, part in enumerate(parts):
                    remaining[label] -= len(part)
                indices = torch.cat(parts).sort().values
            for column in columns:
                collected[column].append(shard[column][indices])
            row_indices.append(indices + offset)
            offset += count
            if remaining == [0, 0]:
                break
        if remaining is not None and any(remaining):
            raise ValueError(f"Insufficient rows of both classes for smoke: {self.split}")
        self.tensors = {column: torch.cat(parts) for column, parts in collected.items()}
        self.row_indices = torch.cat(row_indices).numpy()

    def __iter__(self):
        self.load()
        count = len(self.tensors["labels"])
        if self.split == "train":
            generator = torch.Generator().manual_seed(self.seed + self.iteration_count)
            order = torch.randperm(count, generator=generator).tolist()
        else:
            order = range(count)
        self.iteration_count += 1
        columns = ["pair_indices"]
        if self.condition == "conservation_both":
            columns += ["conservation"]
        elif self.condition != "sequence":
            columns += ["mirna_phact", "target_phact"]
        columns += ["labels"]
        for index in order:
            yield tuple(self.tensors[column][index] for column in columns)


def make_loader(condition, split, seed, max_rows=None):
    dataset = FollowupDataset(condition, split, seed, max_rows)
    if condition == "sequence":
        collate = pair_collate
    elif condition == "conservation_both":
        collate = make_conservation_collate(28)
    else:
        collate = make_phact_collate("mirna" if condition.startswith("guide") else "target")
    return DataLoader(dataset, batch_size=256, num_workers=0,
                      pin_memory=True, collate_fn=collate)


def make_model(condition):
    params = dict(num_pair_classes=18, target_length=50, mirna_length=28,
                  embedding_dim=8, dropout_rate=0.2,
                  filter_sizes=(128, 64, 32), kernel_sizes=(6, 3, 3))
    if condition == "sequence":
        model = PairwiseSeqCNN(**params)
    elif condition == "conservation_both":
        params["conservation_channel_count"] = 2
        model = PairwiseConservationCNN(**params)
    else:
        params["phact_channel_count"] = 4
        model = PairwisePhactCNN(**params)
    return model, params


def export_predictions(model, loader, split, output):
    model.eval()
    labels, predictions = [], []
    with torch.inference_mode():
        for *inputs, target in loader:
            logits = model(*(item.cuda(non_blocking=True) for item in inputs))
            labels.append(target.numpy())
            predictions.append(torch.sigmoid(logits).cpu().numpy())
    labels = np.concatenate(labels)
    predictions = np.concatenate(predictions)
    if not np.isfinite(predictions).all() or np.any((predictions < 0) | (predictions > 1)):
        raise ValueError(f"Invalid predictions: {split}")
    ids = np.array([f"{split}_{i + 1}" for i in loader.dataset.row_indices])
    np.savez_compressed(output / f"{split}.npz", ids=ids, labels=labels, predictions=predictions)
    return {"rows": len(labels), "positive_rows": int(labels.sum()),
            "ap": float(average_precision_score(labels, predictions)),
            "roc_auc": float(roc_auc_score(labels, predictions)),
            "prediction_sha256": sha256(output / f"{split}.npz")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a run: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    set_seed(args.seed)
    model, params = make_model(args.condition)
    initial_hash = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        initial_hash.update(name.encode())
        initial_hash.update(tensor.numpy().tobytes())
    model = model.cuda()
    loaders = {split: make_loader(args.condition, split, args.seed, 1024 if args.smoke else None)
               for split in ("train", "val")}
    for loader in loaders.values():
        loader.dataset.load()
        if set(loader.dataset.tensors["labels"].unique().tolist()) != {0.0, 1.0}:
            raise ValueError("Training and validation must each contain both classes")
    summary = dict(condition=args.condition, seed=args.seed, model_params=params,
                   initial_state_sha256=initial_hash.hexdigest(), batch_size=256,
                   learning_rate=0.001, num_epochs=2 if args.smoke else 50,
                   patience=7, missingness_included=False,
                   cache_shuffle_mode="global", smoke=args.smoke,
                   source_snapshot=str(ROOT / "source"), driver_sha256=sha256(__file__),
                   train_cache=str(loaders["train"].dataset.manifest_path),
                   val_cache=str(loaders["val"].dataset.manifest_path),
                   split_policy="original fixed 90/10 row split; split seed 42")
    artifacts = train_model(
        model=model, loaders=loaders, optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        criterion=torch.nn.BCEWithLogitsLoss(), device=torch.device("cuda"),
        output_dir=args.output, checkpoint_prefix="model", summary=summary,
        num_epochs=summary["num_epochs"], patience=7, progress_every=1000,
        final_eval_phases=(),
    )
    # Reload the selected checkpoint to validate the actual saved inference artifact.
    checkpoint = torch.load(artifacts["checkpoint"], map_location="cpu", weights_only=False)
    if not np.isfinite(checkpoint["val_auprc"]):
        raise ValueError("Selected checkpoint has non-finite validation AP")
    model.load_state_dict(checkpoint["model_state_dict"])
    results = dict(condition=args.condition, seed=args.seed, smoke=args.smoke,
                   best_epoch=checkpoint["epoch"], validation_ap=checkpoint["val_auprc"],
                   checkpoint=str(artifacts["checkpoint"]),
                   checkpoint_sha256=sha256(artifacts["checkpoint"]),
                   initial_state_sha256=initial_hash.hexdigest())
    for split in ("test", "leftout"):
        loader = make_loader(args.condition, split, args.seed, 1024 if args.smoke else None)
        results[split] = export_predictions(model, loader, split, args.output)
    temporary = args.output / "result.json.tmp"
    temporary.write_text(json.dumps(results, indent=2) + "\n")
    temporary.replace(args.output / "result.json")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
