#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from pathlib import Path


REPO_DIR = Path("/home/dtzim01/PHACT-miRBind")
MANAKOV_DIR = Path("/home/dtzim01/manakov_datasets")
DEFAULT_OUTPUT_DIR = REPO_DIR / "agentomics_full"
DEFAULT_MIRNA_PHACT = (
    REPO_DIR
    / "reports/phact_score_ranges/phact_mirna_arm_position_qntnorm_transformed_all_models.tsv"
)
DEFAULT_TARGET_PHACT = (
    REPO_DIR
    / "reports/phact_score_ranges/phact_target_manakov_position_qntnorm_transformed_scores.tsv"
)
DEFAULT_ORTHOLOGUES = (
    REPO_DIR
    / "outputs/mirgenedb_orthologue_counts/"
    "mirgenedb_human_precursor_orthologue_presence_matrix.tsv"
)
DEFAULT_DESCRIPTION_TEMPLATE = REPO_DIR / "scripts/agentomics_dataset_description.md"
DATASET_NAME = "manakov_phact"
VERSION_SUFFIX_RE = re.compile(r"-v\d+$")
NUCLEOTIDES = ("A", "C", "G", "T")
SPLIT_SOURCES = {
    "train": MANAKOV_DIR / "AGO2_eCLIP_Manakov2022_train.tsv",
    "test": MANAKOV_DIR / "AGO2_eCLIP_Manakov2022_test.tsv",
    "leftout": MANAKOV_DIR / "AGO2_eCLIP_Manakov2022_leftout.tsv",
}
SAMPLE_COLUMNS = [
    "id",
    "gene",
    "noncodingRNA",
    "feature",
    "dominant_region",
    "gene_phyloP",
    "gene_phastCons",
    "mirgenedb_mature_id",
    "mirgenedb_premirna_id",
    "mirgenedb_family",
]
CANDIDATE_COLUMNS = [
    "id",
    "candidate_index",
    "candidate_count",
    "mirgenedb_mature_id",
    "has_phact_profile",
]


def normalize_sequence(sequence: str) -> str:
    return sequence.strip().upper().replace("U", "T")


def premirna_id_from_mature_id(mature_id: str) -> str:
    locus = mature_id.rsplit("_", 1)[0]
    return VERSION_SUFFIX_RE.sub("", locus)


def premirna_ids_from_cell(value: str) -> str:
    ids = [part.strip() for part in value.split(";") if part.strip() and part != "NA"]
    return ";".join(premirna_id_from_mature_id(mature_id) for mature_id in ids) or "NA"


def mature_ids_from_cell(value: str) -> list[str]:
    return [part.strip() for part in value.split(";") if part.strip() and part != "NA"]


def safe_model_name(model: str) -> str:
    safe = model.replace(".", "p")
    return re.sub(r"[^0-9A-Za-z_]+", "_", safe)


def split_dir(output_dir: Path, split: str) -> Path:
    if split == "train":
        return output_dir / "datasets" / DATASET_NAME / "train"
    return output_dir / "test_datasets" / DATASET_NAME / split


def sample_id(split: str, row_id: int) -> str:
    return f"{split}_{row_id}"


def write_metadata(dataset_dir: Path) -> None:
    metadata = {
        "name": DATASET_NAME,
        "task_type": "classification",
        "positive_class": "1",
        "negative_class": "0",
        "label_to_scalar": {
            "0": 0,
            "1": 1,
        },
        "source": "AGO2 eCLIP Manakov 2022 with PHACT and MirGeneDB annotations",
    }
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copyfile(DEFAULT_DESCRIPTION_TEMPLATE, dataset_dir / "dataset_description.md")


def load_mirna_phact_profiles(
    path: Path,
) -> tuple[list[str], list[str], dict[str, dict[int, tuple[str, list[str]]]]]:
    model_order: list[str] = []
    model_index: dict[str, int] = {}
    rows: list[dict[str, str]] = []

    with path.open(newline="", buffering=1024 * 1024) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            model = row["phact_model"]
            if model not in model_index:
                model_index[model] = len(model_order)
                model_order.append(model)
            rows.append(row)

    score_columns = [
        f"phact_{safe_model_name(model)}_{base}"
        for model in model_order
        for base in NUCLEOTIDES
    ]
    column_count = len(score_columns)
    profiles: dict[str, dict[int, tuple[str, list[str]]]] = {}
    for row in rows:
        mature_id = f"{row['pre_mirna']}_{row['arm']}"
        position = int(row["arm_position_1based"])
        profile = profiles.setdefault(mature_id, {})
        if position not in profile:
            profile[position] = (row["actual_nt"], ["NA"] * column_count)
        values = profile[position][1]
        base_offset = model_index[row["phact_model"]] * len(NUCLEOTIDES)
        for base_index, base in enumerate(NUCLEOTIDES):
            values[base_offset + base_index] = row[f"score_{base}"]

    return model_order, score_columns, profiles


def write_samples_and_labels(
    output_dir: Path,
    available_mirna_profiles: set[str],
    progress_every: int,
) -> tuple[dict[str, set[int]], dict[str, set[str]], dict[str, int]]:
    row_ids_by_split: dict[str, set[int]] = {}
    mirna_candidate_ids_by_split: dict[str, set[str]] = {}
    sample_counts: dict[str, int] = {}
    for split, source_path in SPLIT_SOURCES.items():
        destination = split_dir(output_dir, split)
        input_dir = destination / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        row_ids_by_split[split] = set()
        mirna_candidate_ids_by_split[split] = set()
        started = time.perf_counter()
        with (
            source_path.open(newline="", buffering=1024 * 1024) as source_handle,
            (destination / "labels.csv").open("w", newline="") as labels_handle,
            (input_dir / "samples.tsv").open("w", newline="") as samples_handle,
            (input_dir / "sample_mirna_candidates.tsv").open(
                "w",
                newline="",
            ) as candidates_handle,
        ):
            reader = csv.DictReader(source_handle, delimiter="\t")
            label_writer = csv.writer(labels_handle, lineterminator="\n")
            candidate_writer = csv.DictWriter(
                candidates_handle,
                fieldnames=CANDIDATE_COLUMNS,
                delimiter="\t",
                lineterminator="\n",
            )
            sample_writer = csv.DictWriter(
                samples_handle,
                fieldnames=SAMPLE_COLUMNS,
                delimiter="\t",
                lineterminator="\n",
            )
            label_writer.writerow(["id", "label"])
            sample_writer.writeheader()
            candidate_writer.writeheader()
            for row_id, row in enumerate(reader, start=1):
                if progress_every and row_id % progress_every == 0:
                    elapsed = time.perf_counter() - started
                    print(f"{split} samples rows={row_id:,} elapsed={elapsed:.1f}s", flush=True)
                current_id = sample_id(split, row_id)
                mirna_sequence = normalize_sequence(row["noncodingRNA"])
                mature_ids = mature_ids_from_cell(row["mirgenedb_mature_id"])
                row_ids_by_split[split].add(row_id)
                label_writer.writerow([current_id, row["label"]])
                sample_writer.writerow(
                    {
                        "id": current_id,
                        "gene": normalize_sequence(row["gene"]),
                        "noncodingRNA": mirna_sequence,
                        "feature": row["feature"],
                        "dominant_region": row["dominant_region"],
                        "gene_phyloP": row["gene_phyloP"],
                        "gene_phastCons": row["gene_phastCons"],
                        "mirgenedb_mature_id": row["mirgenedb_mature_id"],
                        "mirgenedb_premirna_id": premirna_ids_from_cell(
                            row["mirgenedb_mature_id"]
                        ),
                        "mirgenedb_family": row["mirgenedb_family"],
                    }
                )
                if not mature_ids:
                    candidate_writer.writerow(
                        {
                            "id": current_id,
                            "candidate_index": 0,
                            "candidate_count": 0,
                            "mirgenedb_mature_id": "NA",
                            "has_phact_profile": 0,
                        }
                    )
                else:
                    candidate_count = len(mature_ids)
                    for candidate_index, mature_id in enumerate(mature_ids, start=1):
                        mirna_candidate_ids_by_split[split].add(mature_id)
                        candidate_writer.writerow(
                            {
                                "id": current_id,
                                "candidate_index": candidate_index,
                                "candidate_count": candidate_count,
                                "mirgenedb_mature_id": mature_id,
                                "has_phact_profile": int(mature_id in available_mirna_profiles),
                            }
                        )
        sample_counts[split] = len(row_ids_by_split[split])
        print(
            f"{split} samples done rows={sample_counts[split]:,} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return row_ids_by_split, mirna_candidate_ids_by_split, sample_counts


def open_split_writers(
    output_dir: Path,
    file_name: str,
    header: list[str],
) -> tuple[dict[str, object], dict[str, csv.writer]]:
    handles: dict[str, object] = {}
    writers: dict[str, csv.writer] = {}
    for split in SPLIT_SOURCES:
        path = split_dir(output_dir, split) / "input" / file_name
        handle = path.open("w", newline="", buffering=1024 * 1024)
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        handles[split] = handle
        writers[split] = writer
    return handles, writers


def write_mirna_phact_positions(
    output_dir: Path,
    score_columns: list[str],
    profiles: dict[str, dict[int, tuple[str, list[str]]]],
    mirna_candidate_ids_by_split: dict[str, set[str]],
) -> dict[str, int]:
    started = time.perf_counter()
    counts = {split: 0 for split in SPLIT_SOURCES}
    output_header = [
        "mirgenedb_mature_id",
        "mirna_position_1based",
        "actual_nt",
        *score_columns,
    ]
    handles, writers = open_split_writers(
        output_dir,
        "phact_mirna_positions.tsv",
        output_header,
    )
    try:
        for split, mature_ids in mirna_candidate_ids_by_split.items():
            for mature_id in sorted(mature_ids):
                profile = profiles.get(mature_id)
                if profile is None:
                    continue
                for position, (actual_nt, values) in sorted(profile.items()):
                    writers[split].writerow([mature_id, position, actual_nt, *values])
                    counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    print(
        f"miRNA PHACT done written={sum(counts.values()):,} "
        f"elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return counts


def write_target_phact_positions(
    source_file: Path,
    output_dir: Path,
    row_ids_by_split: dict[str, set[int]],
    progress_every: int,
) -> dict[str, int]:
    started = time.perf_counter()
    counts = {split: 0 for split in SPLIT_SOURCES}
    source_rows = 0
    with source_file.open(newline="", buffering=1024 * 1024) as source_handle:
        reader = csv.reader(source_handle, delimiter="\t")
        header = next(reader)
        split_idx = header.index("split")
        row_id_idx = header.index("manakov_row_id")
        genomic_idx = header.index("genomic_position")
        output_header = ["id", *header[2:genomic_idx], *header[genomic_idx + 1 :]]
        handles, writers = open_split_writers(
            output_dir,
            "phact_target_positions.tsv",
            output_header,
        )
        try:
            for row in reader:
                source_rows += 1
                if progress_every and source_rows % progress_every == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"target PHACT source_rows={source_rows:,} written={sum(counts.values()):,} "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                split = row[split_idx]
                split_row_ids = row_ids_by_split.get(split)
                if split_row_ids is None:
                    continue
                row_id = int(row[row_id_idx])
                if row_id not in split_row_ids:
                    continue
                writers[split].writerow(
                    [
                        sample_id(split, row_id),
                        *row[2:genomic_idx],
                        *row[genomic_idx + 1 :],
                    ]
                )
                counts[split] += 1
        finally:
            for handle in handles.values():
                handle.close()
    print(
        f"target PHACT done source_rows={source_rows:,} written={sum(counts.values()):,} "
        f"elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return counts


def copy_orthologues_to_inputs(output_dir: Path, orthologues_file: Path) -> None:
    for split in SPLIT_SOURCES:
        shutil.copyfile(
            orthologues_file,
            split_dir(output_dir, split) / "input" / "mirgenedb_premirna_orthologues.tsv",
        )


def build_full(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    dataset_dir = output_dir / "datasets" / DATASET_NAME
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{output_dir} exists; pass --force to replace it")
        shutil.rmtree(output_dir)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(dataset_dir)
    model_order, score_columns, mirna_profiles = load_mirna_phact_profiles(
        args.mirna_phact_file
    )
    row_ids_by_split, mirna_candidate_ids_by_split, sample_counts = write_samples_and_labels(
        output_dir,
        set(mirna_profiles),
        args.progress_every,
    )
    copy_orthologues_to_inputs(output_dir, args.orthologues_file)
    mirna_counts = write_mirna_phact_positions(
        output_dir,
        score_columns,
        mirna_profiles,
        mirna_candidate_ids_by_split,
    )
    target_counts = write_target_phact_positions(
        args.target_phact_file,
        output_dir,
        row_ids_by_split,
        args.progress_every,
    )
    summary = {
        "output_dir": str(output_dir),
        "dataset_name": DATASET_NAME,
        "sample_rows": sample_counts,
        "mirna_phact_models": model_order,
        "mirna_candidate_ids": {
            split: len(ids) for split, ids in mirna_candidate_ids_by_split.items()
        },
        "mirna_phact_profiles": {
            split: sum(1 for mature_id in ids if mature_id in mirna_profiles)
            for split, ids in mirna_candidate_ids_by_split.items()
        },
        "mirna_phact_rows": mirna_counts,
        "target_phact_rows": target_counts,
        "mirna_phact_keyed_by_individual_mature_id": True,
        "mirna_multimap_profiles_averaged": False,
        "target_genomic_position_removed": True,
    }
    (output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mirna-phact-file", type=Path, default=DEFAULT_MIRNA_PHACT)
    parser.add_argument("--target-phact-file", type=Path, default=DEFAULT_TARGET_PHACT)
    parser.add_argument("--orthologues-file", type=Path, default=DEFAULT_ORTHOLOGUES)
    parser.add_argument("--progress-every", type=int, default=2_000_000)
    parser.add_argument("--force", action="store_true")
    build_full(parser.parse_args())


if __name__ == "__main__":
    main()
