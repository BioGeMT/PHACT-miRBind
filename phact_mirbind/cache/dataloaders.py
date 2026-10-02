"""Iterable datasets and collate functions for cached tensors."""

from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from phact_mirbind.cache.manifest import (
    CACHE_VERSION,
    find_manifest,
    normalize_shuffle_mode,
)
from phact_mirbind.data.columns import NUM_PAIR_CLASSES, PADDING_PAIR_INDEX
from phact_mirbind.data.conservation import (
    CONSERVATION_FEATURES,
    parse_conservation_features,
)
from phact_mirbind.data.pair_encoding import decode_pair_base_ids

PHACT_CHANNEL_MODES = ("mirna", "target", "both")
PHACT_INTERACTION_MODES = ("none", "outer")


class CachedPairwiseIterableDataset(IterableDataset):
    """Read compact pair-grid shards and emit ``pair_indices, label``."""

    def __init__(
        self,
        cache_path: str | Path,
        *,
        max_rows: int | None = None,
        shuffle: bool = False,
        shuffle_mode: str | None = None,
        shuffle_seed: int = 42,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.manifest_path = find_manifest(self.cache_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        self.max_rows = max_rows
        self.shuffle_mode = normalize_shuffle_mode(shuffle, shuffle_mode)
        self.shuffle_seed = shuffle_seed
        self._iteration_count = 0
        self.sample_weights: torch.Tensor | None = None
        self.has_sample_weights = False
        self.num_pair_classes = NUM_PAIR_CLASSES
        self.target_length = int(self.manifest["target_length"])
        self.mirna_length = int(self.manifest["mirna_length"])
        self.shards = [
            (
                self.manifest_path.parent / shard["file"],
                int(shard["rows"]),
            )
            for shard in self.manifest["shards"]
        ]
        self._refresh_shard_offsets()

        if int(self.manifest["cache_version"]) != CACHE_VERSION:
            raise ValueError(
                f"Unsupported cache version {self.manifest['cache_version']}; "
                f"expected {CACHE_VERSION}"
            )
        if max_rows is not None and max_rows < 1:
            raise ValueError("max_rows must be positive when provided")

    def set_sample_weights(self, path: str | Path) -> None:
        arrays = np.load(path)
        if "weights" not in arrays:
            raise ValueError(f"Sample-weight NPZ lacks weights: {path}")
        weights = torch.from_numpy(arrays["weights"].astype(np.float32, copy=False))
        expected_rows = int(self.manifest["row_count"])
        if len(weights) != expected_rows:
            raise ValueError(
                f"Sample weights have {len(weights)} rows; expected {expected_rows}"
            )
        if not torch.isfinite(weights).all() or torch.any(weights < 0):
            raise ValueError("Sample weights must be finite and non-negative")
        self.sample_weights = weights
        self.has_sample_weights = True

    def _refresh_shard_offsets(self) -> None:
        offset = 0
        self._shard_offsets: dict[Path, int] = {}
        for shard_path, row_count in self.shards:
            self._shard_offsets[shard_path] = offset
            offset += row_count

    def _sample_weight(self, shard_path: Path, row_idx: int) -> torch.Tensor:
        assert self.sample_weights is not None
        return self.sample_weights[self._shard_offsets[shard_path] + row_idx]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        iteration_count = self._iteration_count
        self._iteration_count += 1

        if self.shuffle_mode == "global":
            yield from self._iter_global_shuffle(iteration_count)
            return

        for shard_path, row_indices in self._iter_shard_rows(iteration_count):
            shard = torch.load(shard_path, map_location="cpu")
            for row_idx in row_indices:
                item = (shard["pair_indices"][row_idx], shard["labels"][row_idx])
                if self.has_sample_weights:
                    sample_weight = self._sample_weight(shard_path, row_idx)
                    yield (*item, sample_weight)
                else:
                    yield item

    def _iter_shard_rows(
        self,
        iteration_count: int,
    ) -> Iterator[tuple[Path, list[int]]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        seed = self.shuffle_seed + iteration_count
        shards = list(self.shards)
        if self.shuffle_mode == "shard":
            random.Random(seed).shuffle(shards)

        logical_offset = 0
        for shard_idx, (shard_path, row_count) in enumerate(shards):
            remaining = None
            if self.max_rows is not None:
                remaining = self.max_rows - logical_offset
                if remaining <= 0:
                    return

            rows_to_take = row_count if remaining is None else min(row_count, remaining)
            logical_offset += row_count
            if shard_idx % num_workers != worker_id:
                continue

            row_indices = list(range(row_count))
            if self.shuffle_mode == "shard":
                random.Random(seed + shard_idx + 1).shuffle(row_indices)
            yield shard_path, row_indices[:rows_to_take]

    def _iter_global_shuffle(
        self,
        iteration_count: int,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        generator = torch.Generator().manual_seed(self.shuffle_seed + iteration_count)

        cache = self._load_all_shards()
        total_rows = int(cache["labels"].shape[0])
        order = torch.randperm(total_rows, generator=generator)
        if self.max_rows is not None:
            order = order[: self.max_rows]
        if num_workers > 1:
            order = order[worker_id::num_workers]

        for global_idx in order.tolist():
            item = (cache["pair_indices"][global_idx], cache["labels"][global_idx])
            if self.has_sample_weights:
                assert self.sample_weights is not None
                sample_weight = self.sample_weights[global_idx]
                yield (*item, sample_weight)
            else:
                yield item

    def _load_all_shards(self) -> dict[str, torch.Tensor]:
        pairs: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for shard_path, _ in self.shards:
            shard = torch.load(shard_path, map_location="cpu")
            pairs.append(shard["pair_indices"])
            labels.append(shard["labels"])
        return {
            "pair_indices": torch.cat(pairs, dim=0),
            "labels": torch.cat(labels, dim=0),
        }


class CachedConservationPairwiseIterableDataset(CachedPairwiseIterableDataset):
    """Read pair-grid plus compact target-conservation shards."""

    def __init__(
        self,
        cache_path: str | Path,
        *,
        conservation_features: str | list[str] | tuple[str, ...] = CONSERVATION_FEATURES,
        max_rows: int | None = None,
        shuffle: bool = False,
        shuffle_mode: str | None = None,
        shuffle_seed: int = 42,
    ) -> None:
        super().__init__(
            cache_path,
            max_rows=max_rows,
            shuffle=shuffle,
            shuffle_mode=shuffle_mode,
            shuffle_seed=shuffle_seed,
        )
        self.cache_features = tuple(self.manifest["conservation_features"])
        self.selected_features = parse_conservation_features(conservation_features)
        missing_features = [
            feature for feature in self.selected_features if feature not in self.cache_features
        ]
        if missing_features:
            raise ValueError(
                "Requested conservation feature(s) not present in cache: "
                + ", ".join(missing_features)
            )
        self.feature_indices = tuple(
            self.cache_features.index(feature) for feature in self.selected_features
        )

    @property
    def conservation_channel_count(self) -> int:
        return len(self.selected_features)

    def __iter__(
        self,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        iteration_count = self._iteration_count
        self._iteration_count += 1

        if self.shuffle_mode == "global":
            yield from self._iter_global_shuffle(iteration_count)
            return

        for shard_path, row_indices in self._iter_shard_rows(iteration_count):
            shard = torch.load(shard_path, map_location="cpu")
            conservation = shard["conservation"][:, :, self.feature_indices]
            for row_idx in row_indices:
                yield (
                    shard["pair_indices"][row_idx],
                    conservation[row_idx],
                    shard["labels"][row_idx],
                )

    def _iter_global_shuffle(
        self,
        iteration_count: int,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        generator = torch.Generator().manual_seed(self.shuffle_seed + iteration_count)

        cache = self._load_all_shards()
        total_rows = int(cache["labels"].shape[0])
        order = torch.randperm(total_rows, generator=generator)
        if self.max_rows is not None:
            order = order[: self.max_rows]
        if num_workers > 1:
            order = order[worker_id::num_workers]

        for global_idx in order.tolist():
            yield (
                cache["pair_indices"][global_idx],
                cache["conservation"][global_idx],
                cache["labels"][global_idx],
            )

    def _load_all_shards(self) -> dict[str, torch.Tensor]:
        pairs: list[torch.Tensor] = []
        conservation: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for shard_path, _ in self.shards:
            shard = torch.load(shard_path, map_location="cpu")
            pairs.append(shard["pair_indices"])
            conservation.append(shard["conservation"][:, :, self.feature_indices])
            labels.append(shard["labels"])
        return {
            "pair_indices": torch.cat(pairs, dim=0),
            "conservation": torch.cat(conservation, dim=0),
            "labels": torch.cat(labels, dim=0),
        }


class CachedPhactPairwiseIterableDataset(CachedPairwiseIterableDataset):
    """Read pair-grid plus compact miRNA/target PHACT score shards."""

    def __init__(
        self,
        cache_path: str | Path,
        *,
        include_missingness: bool = True,
        max_rows: int | None = None,
        shuffle: bool = False,
        shuffle_mode: str | None = None,
        shuffle_seed: int = 42,
    ) -> None:
        self.include_missingness = include_missingness
        super().__init__(
            cache_path,
            max_rows=max_rows,
            shuffle=shuffle,
            shuffle_mode=shuffle_mode,
            shuffle_seed=shuffle_seed,
        )

    @property
    def phact_channel_count(self) -> int:
        return self.phact_channel_count_for_mode("both")

    @property
    def mirna_phact_channel_count(self) -> int:
        if not self.include_missingness:
            counts = self.manifest.get("phact_axis_channel_counts", {})
            return int(counts.get("mirna", 4))
        total_counts = self.manifest.get("phact_axis_total_channel_counts", {})
        if "mirna" in total_counts:
            return int(total_counts["mirna"])
        counts = self.manifest.get("phact_axis_channel_counts", {})
        return int(counts.get("mirna", 4))

    @property
    def target_phact_channel_count(self) -> int:
        if not self.include_missingness:
            counts = self.manifest.get("phact_axis_channel_counts", {})
            return int(counts.get("target", 4))
        total_counts = self.manifest.get("phact_axis_total_channel_counts", {})
        if "target" in total_counts:
            return int(total_counts["target"])
        counts = self.manifest.get("phact_axis_channel_counts", {})
        return int(counts.get("target", 4))

    def phact_channel_count_for_mode(self, channel_mode: str) -> int:
        validate_phact_channel_mode(channel_mode)
        if channel_mode == "mirna":
            return self.mirna_phact_channel_count
        if channel_mode == "target":
            return self.target_phact_channel_count
        return self.mirna_phact_channel_count + self.target_phact_channel_count

    def __iter__(
        self,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        iteration_count = self._iteration_count
        self._iteration_count += 1

        if self.shuffle_mode == "global":
            yield from self._iter_global_shuffle(iteration_count)
            return

        for shard_path, row_indices in self._iter_shard_rows(iteration_count):
            shard = torch.load(shard_path, map_location="cpu")
            mirna_phact = self._with_missingness(shard, "mirna")
            target_phact = self._with_missingness(shard, "target")

            for row_idx in row_indices:
                item = (
                    shard["pair_indices"][row_idx],
                    mirna_phact[row_idx],
                    target_phact[row_idx],
                    shard["labels"][row_idx],
                )
                if self.has_sample_weights:
                    sample_weight = self._sample_weight(shard_path, row_idx)
                    yield (*item, sample_weight)
                else:
                    yield item

    def _iter_global_shuffle(
        self,
        iteration_count: int,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        generator = torch.Generator().manual_seed(self.shuffle_seed + iteration_count)

        cache = self._load_all_shards()
        total_rows = int(cache["labels"].shape[0])
        order = torch.randperm(total_rows, generator=generator)
        if self.max_rows is not None:
            order = order[: self.max_rows]
        if num_workers > 1:
            order = order[worker_id::num_workers]

        for global_idx in order.tolist():
            item = (
                cache["pair_indices"][global_idx],
                cache["mirna_phact"][global_idx],
                cache["target_phact"][global_idx],
                cache["labels"][global_idx],
            )
            if self.has_sample_weights:
                assert self.sample_weights is not None
                sample_weight = self.sample_weights[global_idx]
                yield (*item, sample_weight)
            else:
                yield item

    def _load_all_shards(self) -> dict[str, torch.Tensor]:
        pairs: list[torch.Tensor] = []
        mirna_phact: list[torch.Tensor] = []
        target_phact: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for shard_path, _ in self.shards:
            shard = torch.load(shard_path, map_location="cpu")
            pairs.append(shard["pair_indices"])
            mirna_phact.append(self._with_missingness(shard, "mirna"))
            target_phact.append(self._with_missingness(shard, "target"))
            labels.append(shard["labels"])
        return {
            "pair_indices": torch.cat(pairs, dim=0),
            "mirna_phact": torch.cat(mirna_phact, dim=0),
            "target_phact": torch.cat(target_phact, dim=0),
            "labels": torch.cat(labels, dim=0),
        }

    def _with_missingness(
        self,
        shard: dict[str, torch.Tensor],
        axis: str,
    ) -> torch.Tensor:
        scores = shard[f"{axis}_phact"]
        if not self.include_missingness:
            return scores
        missingness = shard.get(f"{axis}_phact_missing")
        if missingness is None:
            return scores
        return torch.cat([scores, missingness.to(dtype=scores.dtype)], dim=-1)


class CompositeCachedPhactPairwiseIterableDataset(
    CachedPhactPairwiseIterableDataset
):
    """Stream compatible PHACT caches as one shuffled training dataset."""

    _COMPATIBILITY_KEYS = (
        "cache_type",
        "target_length",
        "mirna_length",
        "phact_dtype",
        "phact_reduction",
        "phact_models",
        "target_phact_models",
        "phact_axis_channel_counts",
        "phact_axis_missingness_channel_counts",
        "phact_axis_total_channel_counts",
        "phact_channel_order",
    )

    def __init__(
        self,
        cache_paths: Sequence[str | Path],
        *,
        max_rows: int | None = None,
        shuffle: bool = False,
        shuffle_mode: str | None = None,
        shuffle_seed: int = 42,
        include_missingness: bool = True,
    ) -> None:
        paths = tuple(Path(path) for path in cache_paths)
        if not paths:
            raise ValueError("cache_paths must contain at least one PHACT cache")

        super().__init__(
            paths[0],
            max_rows=max_rows,
            shuffle=shuffle,
            shuffle_mode=shuffle_mode,
            shuffle_seed=shuffle_seed,
            include_missingness=include_missingness,
        )
        self.cache_paths = paths
        manifests = [self.manifest]
        combined_shards = list(self.shards)

        for path in paths[1:]:
            dataset = CachedPhactPairwiseIterableDataset(
                path,
                include_missingness=include_missingness,
            )
            self._validate_compatible_manifest(path, dataset.manifest)
            manifests.append(dataset.manifest)
            combined_shards.extend(dataset.shards)

        self.shards = combined_shards
        self.manifest = deepcopy(self.manifest)
        self.manifest["row_count"] = sum(
            int(manifest["row_count"]) for manifest in manifests
        )
        self.manifest["positive_rows"] = sum(
            int(manifest["positive_rows"]) for manifest in manifests
        )
        self.manifest["composite_cache_paths"] = [str(path) for path in paths]
        self._refresh_shard_offsets()

    def _validate_compatible_manifest(
        self,
        path: Path,
        manifest: dict[str, object],
    ) -> None:
        mismatches = [
            key
            for key in self._COMPATIBILITY_KEYS
            if manifest.get(key) != self.manifest.get(key)
        ]
        if mismatches:
            raise ValueError(
                f"PHACT cache {path} is incompatible with {self.cache_path}; "
                f"mismatched manifest fields: {', '.join(mismatches)}"
            )


def validate_phact_channel_mode(channel_mode: str) -> None:
    if channel_mode not in PHACT_CHANNEL_MODES:
        choices = ", ".join(PHACT_CHANNEL_MODES)
        raise ValueError(f"channel_mode must be one of: {choices}")


def pair_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_indices = torch.stack([item[0] for item in batch]).long()
    labels = torch.stack([item[1] for item in batch]).float()
    return pair_indices, labels


def make_conservation_collate(mirna_length: int):
    def collate(
        batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair_indices = torch.stack([item[0] for item in batch]).long()
        compact = torch.stack([item[1] for item in batch]).float()
        labels = torch.stack([item[2] for item in batch]).float()
        channels = (
            compact.permute(0, 2, 1)
            .unsqueeze(2)
            .expand(compact.shape[0], compact.shape[2], mirna_length, compact.shape[1])
            .contiguous()
        )
        return pair_indices, channels, labels

    return collate


def make_phact_collate(
    channel_mode: str = "both",
    interaction_mode: str = "none",
    interaction_mirna_channels: int = 4,
    interaction_target_channels: int = 4,
    target_shift_max: int = 0,
    target_fill_values: Sequence[float] | None = None,
):
    validate_phact_channel_mode(channel_mode)
    if interaction_mode not in PHACT_INTERACTION_MODES:
        choices = ", ".join(PHACT_INTERACTION_MODES)
        raise ValueError(f"interaction_mode must be one of: {choices}")
    if interaction_mode != "none" and channel_mode != "both":
        raise ValueError("PHACT interaction features require channel_mode='both'")
    if interaction_mirna_channels < 1 or interaction_target_channels < 1:
        raise ValueError("interaction channel counts must be positive")
    if target_shift_max < 0:
        raise ValueError("target_shift_max must be non-negative")
    if target_shift_max and target_fill_values is None:
        raise ValueError("target_fill_values are required for target shifts")

    def collate(
        batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair_indices = torch.stack([item[0] for item in batch]).long()
        mirna_compact = torch.stack([item[1] for item in batch]).float()
        target_compact = torch.stack([item[2] for item in batch]).float()
        labels = torch.stack([item[3] for item in batch]).float()
        if target_shift_max:
            pair_indices, target_compact = shift_compact_target_axis(
                pair_indices,
                target_compact,
                int(torch.randint(-target_shift_max, target_shift_max + 1, ()).item()),
                target_fill_values,
            )
        batch_size = pair_indices.shape[0]
        mirna_length = mirna_compact.shape[1]
        target_length = target_compact.shape[1]

        mirna_channels = (
            mirna_compact.permute(0, 2, 1)
            .unsqueeze(3)
            .expand(batch_size, mirna_compact.shape[2], mirna_length, target_length)
        )
        target_channels = (
            target_compact.permute(0, 2, 1)
            .unsqueeze(2)
            .expand(batch_size, target_compact.shape[2], mirna_length, target_length)
        )
        if channel_mode == "mirna":
            channels = mirna_channels.contiguous()
        elif channel_mode == "target":
            channels = target_channels.contiguous()
        else:
            channels = torch.cat([mirna_channels, target_channels], dim=1).contiguous()
        if interaction_mode == "outer":
            if mirna_compact.shape[2] < interaction_mirna_channels:
                raise ValueError("miRNA cache has too few channels for interactions")
            if target_compact.shape[2] < interaction_target_channels:
                raise ValueError("target cache has too few channels for interactions")
            mirna_scores = (
                mirna_compact[:, :, :interaction_mirna_channels]
                .permute(0, 2, 1)
                .unsqueeze(2)
                .unsqueeze(4)
            )
            target_scores = (
                target_compact[:, :, :interaction_target_channels]
                .permute(0, 2, 1)
                .unsqueeze(1)
                .unsqueeze(3)
            )
            interaction_channels = (mirna_scores * target_scores).reshape(
                batch_size,
                interaction_mirna_channels * interaction_target_channels,
                mirna_length,
                target_length,
            )
            channels = torch.cat([channels, interaction_channels], dim=1).contiguous()
        if len(batch[0]) == 5:
            sample_weights = torch.stack([item[4] for item in batch]).float()
            return pair_indices, channels, labels, sample_weights
        return pair_indices, channels, labels

    return collate


def shift_compact_target_axis(
    pair_indices: torch.Tensor,
    target_compact: torch.Tensor,
    shift: int,
    fill_values: Sequence[float] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate the target axis and fill newly exposed positions neutrally."""
    if shift == 0:
        return pair_indices, target_compact
    if abs(shift) >= pair_indices.shape[-1]:
        raise ValueError("absolute target shift must be smaller than target length")
    if fill_values is None or len(fill_values) != target_compact.shape[-1]:
        raise ValueError("fill_values must match compact target channels")

    shifted_pairs = torch.full_like(pair_indices, PADDING_PAIR_INDEX)
    fills = torch.as_tensor(
        fill_values,
        dtype=target_compact.dtype,
        device=target_compact.device,
    ).view(1, 1, -1)
    shifted_target = fills.expand_as(target_compact).clone()
    if shift > 0:
        shifted_pairs[:, :, shift:] = pair_indices[:, :, :-shift]
        shifted_target[:, shift:] = target_compact[:, :-shift]
    else:
        shifted_pairs[:, :, :shift] = pair_indices[:, :, -shift:]
        shifted_target[:, :shift] = target_compact[:, -shift:]
    return shifted_pairs, shifted_target


def rinalmo_phact_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Collate compact PHACT features and recover both sequence token streams."""
    pair_indices = torch.stack([item[0] for item in batch]).long()
    mirna_base_ids, mirna_mask, target_base_ids, target_mask = (
        decode_pair_base_ids(pair_indices)
    )
    mirna_phact = torch.stack([item[1] for item in batch]).float()
    target_phact = torch.stack([item[2] for item in batch]).float()
    labels = torch.stack([item[3] for item in batch]).float()
    return (
        pair_indices,
        mirna_base_ids,
        mirna_mask,
        target_base_ids,
        target_mask,
        mirna_phact,
        target_phact,
        labels,
    )
