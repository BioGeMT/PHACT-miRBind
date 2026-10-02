"""Adapt one-profile interaction tables to the retained Agentomics split readers."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from phact_mirbind.cache.interaction_cache import source_key
from phact_mirbind.data.columns import BASES
from phact_mirbind.data.interaction_table import iter_interactions


def agentomics_input(path: str | Path, cache_root: str | Path) -> Path:
    """Accept an existing split directory or prepare one from a CSV/TSV table.

    A table represents one miRNA score profile per interaction. Synthetic profile
    IDs are storage keys, not MirGeneDB annotations. Existing directory input
    retains its original multi-candidate information.
    """
    path = Path(path).resolve()
    if path.is_dir():
        return path
    key = source_key(path, {"agentomics_array_schema": 1})
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / key
    if (destination / "arrays_manifest.json").is_file():
        required = ["labels.csv", "input/samples.tsv", "input/sample_mirna_candidates.tsv",
                    "input/phact_mirna_positions.tsv", "input/phact_target_positions.tsv"]
        if all((destination / file).is_file() for file in required):
            return destination
        raise FileNotFoundError(f"Incomplete Agentomics input cache: {destination}")
    if destination.exists():
        raise FileExistsError(f"Incomplete Agentomics input cache: {destination}")
    with tempfile.TemporaryDirectory(prefix=".building-", dir=cache_root) as temporary:
        output = Path(temporary) / "split"
        (output / "input").mkdir(parents=True)
        with ExitStack() as stack:
            def writer(filename: str, columns: list[str], delimiter: str = "\t"):
                handle = stack.enter_context((output / filename).open("w", newline=""))
                result = csv.writer(handle, delimiter=delimiter)
                result.writerow(columns)
                return result

            samples = writer("input/samples.tsv", ["id", "gene", "noncodingRNA", "feature",
                                                    "dominant_region", "gene_phyloP", "gene_phastCons"])
            labels = writer("labels.csv", ["id", "label"], ",")
            candidates = writer("input/sample_mirna_candidates.tsv", ["id", "candidate_index",
                                "candidate_count", "mirgenedb_mature_id", "has_phact_profile"])
            mirna = writer("input/phact_mirna_positions.tsv", ["mirgenedb_mature_id",
                           "mirna_position_1based"] + [f"phact_param_1_{base}" for base in BASES])
            targets = writer("input/phact_target_positions.tsv", ["id", "target_position_1based",
                             "actual_nt"] + [f"score_{base}" for base in BASES])
            known_profiles = set()
            rows = 0
            for item in iter_interactions(path, conservation_features=("phylop", "phastcons")):
                # Legacy readers use np.fromstring, which recognizes nan rather than JSON null.
                tracks = ["[" + ",".join(str(float(v)) for v in item.conservation[f]) + "]"
                          for f in ("phylop", "phastcons")]
                samples.writerow([item.id, item.gene, item.mirna, item.metadata.get("feature", "NA"),
                                  item.metadata.get("dominant_region", "NA"), *tracks])
                labels.writerow([item.id, int(item.label)])
                profile_id = "csv_profile_" + hashlib.sha256(
                    item.mirna.encode() + item.mirna_phact.tobytes(),
                ).hexdigest()
                available = np.isfinite(item.mirna_phact).all(axis=1)
                candidates.writerow([item.id, 0, 1, profile_id, int(available.any())])
                if profile_id not in known_profiles:
                    known_profiles.add(profile_id)
                    for position in np.flatnonzero(available):
                        mirna.writerow([profile_id, int(position) + 1, *item.mirna_phact[position]])
                for position in np.flatnonzero(np.isfinite(item.target_phact).all(axis=1)):
                    targets.writerow([item.id, int(position) + 1, item.gene[position],
                                      *item.target_phact[position]])
                rows += 1
                if rows % 100_000 == 0:
                    print(f"Prepared {rows:,} Agentomics rows from {path.name}", flush=True)
        if not rows:
            raise ValueError(f"{path}: no interactions")
        (output / "arrays_manifest.json").write_text(json.dumps({
            "source_file": str(path), "source_key": key, "row_count": rows,
            "profile_representation": "one supplied profile per interaction",
        }, indent=2) + "\n")
        output.rename(destination)
    return destination
