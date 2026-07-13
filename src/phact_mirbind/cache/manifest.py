"""Shared tensor-cache manifest helpers."""

from __future__ import annotations

from pathlib import Path

CACHE_VERSION = 1
SHUFFLE_MODES = {"none", "shard", "global"}


def find_manifest(cache_path: str | Path) -> Path:
    path = Path(cache_path)
    if path.is_file():
        return path
    manifests = sorted(path.glob("*_manifest.json"))
    if len(manifests) == 1:
        return manifests[0]
    if not manifests:
        raise FileNotFoundError(f"No *_manifest.json found in {path}")
    raise ValueError(f"Multiple manifests found in {path}; pass one explicitly")


def normalize_shuffle_mode(shuffle: bool, shuffle_mode: str | None) -> str:
    if shuffle_mode is None:
        return "shard" if shuffle else "none"

    normalized = shuffle_mode.strip().lower()
    if normalized not in SHUFFLE_MODES:
        choices = ", ".join(sorted(SHUFFLE_MODES))
        raise ValueError(f"shuffle_mode must be one of: {choices}")
    return normalized


def write_cache_log_header(log_path: Path) -> None:
    with log_path.open("w") as handle:
        handle.write("shard\tfile\trows\tpositive_rows\tsize_bytes\telapsed_sec\n")
