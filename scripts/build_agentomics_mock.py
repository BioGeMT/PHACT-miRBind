#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path


REPO_DIR = Path("/home/dtzim01/PHACT-miRBind")
DEFAULT_OUTPUT_DIR = REPO_DIR / "agentomics_mock" / "manakov_phact"
DEFAULT_ORTHOLOGUES = (
    REPO_DIR
    / "outputs/mirgenedb_orthologue_counts/"
    "mirgenedb_human_precursor_orthologue_presence_matrix.tsv"
)
DEFAULT_PHACT_CACHE_DIR = (
    REPO_DIR
    / "data/phact_cache_CountNodes_2_0_MinNode_Mix_max05_Gauss_0_"
    "CountNodes_3_0_MinNode_Mix2_CountNodes_4_1_0p5_mean_2_5_0p1_"
    "median_3_CountNodes_1_target_target_auto"
)
DEFAULT_MIRNA_PHACT = (
    REPO_DIR
    / "reports/phact_score_ranges/phact_mirna_arm_position_qntnorm_transformed_all_models.tsv"
)
VERSION_SUFFIX_RE = re.compile(r"-v\d+$")
NUCLEOTIDES = ("A", "C", "G", "T")
SPLITS = ("train", "test", "leftout")
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


def format_split_id(split: str, row_id: int) -> str:
    return f"{split}_{row_id}"


def load_mirna_phact_profiles(
    path: Path,
) -> tuple[list[str], dict[str, dict[int, tuple[str, list[str]]]]]:
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
    profiles: dict[str, dict[int, tuple[str, list[str]]]] = {}
    for row in rows:
        mature_id = f"{row['pre_mirna']}_{row['arm']}"
        position = int(row["arm_position_1based"])
        profile = profiles.setdefault(mature_id, {})
        if position not in profile:
            profile[position] = (row["actual_nt"], ["NA"] * len(score_columns))
        values = profile[position][1]
        base_offset = model_index[row["phact_model"]] * len(NUCLEOTIDES)
        for base_index, base in enumerate(NUCLEOTIDES):
            values[base_offset + base_index] = row[f"score_{base}"]

    return score_columns, profiles


def sample_split_rows(
    split: str,
    source_file: Path,
    output_dir: Path,
    available_mirna_profiles: set[str],
    sample_count: int,
) -> tuple[dict[int, dict[str, str]], set[str]]:
    split_dir = output_dir / split
    input_dir = split_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    labels_path = split_dir / "labels.csv"
    samples_path = input_dir / "samples.tsv"
    sample_sequences: dict[int, dict[str, str]] = {}
    candidate_ids: set[str] = set()

    with (
        source_file.open(newline="") as source_handle,
        labels_path.open("w", newline="") as labels_handle,
        samples_path.open("w", newline="") as samples_handle,
        (input_dir / "sample_mirna_candidates.tsv").open("w", newline="") as candidates_handle,
    ):
        reader = csv.DictReader(source_handle, delimiter="\t")
        labels_writer = csv.writer(labels_handle, lineterminator="\n")
        candidate_writer = csv.DictWriter(
            candidates_handle,
            fieldnames=CANDIDATE_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        samples_writer = csv.DictWriter(
            samples_handle,
            fieldnames=SAMPLE_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        labels_writer.writerow(["id", "label"])
        samples_writer.writeheader()
        candidate_writer.writeheader()
        for row_id, row in enumerate(reader, start=1):
            if row_id > sample_count:
                break
            sample_id = format_split_id(split, row_id)
            target_sequence = normalize_sequence(row["gene"])
            mirna_sequence = normalize_sequence(row["noncodingRNA"])
            mature_ids = mature_ids_from_cell(row["mirgenedb_mature_id"])
            sample_sequences[row_id] = {
                "target": target_sequence,
                "mirna": mirna_sequence,
            }
            labels_writer.writerow([sample_id, row["label"]])
            samples_writer.writerow(
                {
                    "id": sample_id,
                    "gene": target_sequence,
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
                        "id": sample_id,
                        "candidate_index": 0,
                        "candidate_count": 0,
                        "mirgenedb_mature_id": "NA",
                        "has_phact_profile": 0,
                    }
                )
            else:
                candidate_count = len(mature_ids)
                for candidate_index, mature_id in enumerate(mature_ids, start=1):
                    candidate_ids.add(mature_id)
                    candidate_writer.writerow(
                        {
                            "id": sample_id,
                            "candidate_index": candidate_index,
                            "candidate_count": candidate_count,
                            "mirgenedb_mature_id": mature_id,
                            "has_phact_profile": int(mature_id in available_mirna_profiles),
                        }
                    )
    return sample_sequences, candidate_ids


def format_score(value: object) -> str:
    return f"{float(value):.8g}"


def write_mirna_phact_positions(
    split: str,
    output_dir: Path,
    score_columns: list[str],
    profiles: dict[str, dict[int, tuple[str, list[str]]]],
    candidate_ids: set[str],
) -> None:
    with (output_dir / split / "input" / "phact_mirna_positions.tsv").open(
        "w",
        newline="",
    ) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["mirgenedb_mature_id", "mirna_position_1based", "actual_nt", *score_columns]
        )
        for mature_id in sorted(candidate_ids):
            profile = profiles.get(mature_id)
            if profile is None:
                continue
            for position, (actual_nt, values) in sorted(profile.items()):
                writer.writerow([mature_id, position, actual_nt, *values])


def write_target_phact_positions(
    split: str,
    cache_dir: Path,
    output_dir: Path,
    sample_sequences: dict[int, dict[str, str]],
) -> None:
    import torch

    manifest_path = cache_dir / split / f"{split}_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    shard_path = cache_dir / split / manifest["shards"][0]["file"]
    shard = torch.load(shard_path, map_location="cpu")
    target_phact = shard["target_phact"]
    target_columns = manifest["source_columns"]["target_phact"]

    with (output_dir / split / "input" / "phact_target_positions.tsv").open(
        "w",
        newline="",
    ) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "target_position_1based", "actual_nt", *target_columns])
        for row_id, sequences in sample_sequences.items():
            sample_id = format_split_id(split, row_id)
            target_sequence = sequences["target"]
            for position_idx, actual_nt in enumerate(target_sequence):
                values = target_phact[row_id - 1, position_idx].tolist()
                writer.writerow(
                    [
                        sample_id,
                        position_idx + 1,
                        actual_nt,
                        *(format_score(value) for value in values),
                    ]
                )


def write_metadata(output_dir: Path, sample_count: int) -> None:
    metadata = {
        "name": "manakov_phact_mock",
        "task_type": "classification",
        "positive_class": "1",
        "negative_class": "0",
        "label_to_scalar": {
            "0": 0,
            "1": 1,
        },
        "sample_rows_per_split": sample_count,
        "splits": ["train", "test", "leftout"],
        "notes": "Small inspectable mock of the proposed Agentomics input files.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Manakov PHACT Agentomics Mock",
                "",
                "Small, direct-layout mock of the proposed Agentomics input.",
                "",
                "Layout:",
                "",
                "```text",
                "manakov_phact/",
                "  metadata.json",
                "  README.md",
                "  mirgenedb_premirna_orthologues.tsv",
                "  train/",
                "    labels.csv",
                "    input/",
                "      samples.tsv",
                "      sample_mirna_candidates.tsv",
                "      phact_mirna_positions.tsv",
                "      phact_target_positions.tsv",
                "  test/",
                "    labels.csv",
                "    input/",
                "      samples.tsv",
                "      sample_mirna_candidates.tsv",
                "      phact_mirna_positions.tsv",
                "      phact_target_positions.tsv",
                "  leftout/",
                "    labels.csv",
                "    input/",
                "      samples.tsv",
                "      sample_mirna_candidates.tsv",
                "      phact_mirna_positions.tsv",
                "      phact_target_positions.tsv",
                "```",
                "",
                "`labels.csv` contains only `id,label`.",
                "`samples.tsv` contains row-level sequence and annotation columns.",
                "`sample_mirna_candidates.tsv` contains one row per sample and exact",
                "MirGeneDB mature miRNA candidate; multimapped samples have multiple rows.",
                "`phact_mirna_positions.tsv` contains one row per individual",
                "`mirgenedb_mature_id` and mature miRNA position.",
                "`phact_target_positions.tsv` contains one row per target position.",
                "No miRNA PHACT profiles are averaged during dataset construction.",
                "The PHACT model column named `phact_mean_*` is one provided model,",
                "not a preprocessing average.",
                "`mirgenedb_premirna_orthologues.tsv` is precursor-level species presence",
                "from MirGeneDB Orthologues lists.",
                "",
                "Target PHACT position values in this mock are read from the cache.",
                "It is intentionally flatter than the exact Agentomics workspace wrapper.",
                "",
            ]
        )
    )


def build_mock(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    mirna_score_columns, mirna_profiles = load_mirna_phact_profiles(args.mirna_phact_file)
    for split in SPLITS:
        manifest_path = args.phact_cache_dir / split / f"{split}_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        sample_sequences, candidate_ids = sample_split_rows(
            split,
            Path(manifest["source_file"]),
            output_dir,
            set(mirna_profiles),
            args.sample_count,
        )
        write_mirna_phact_positions(
            split,
            output_dir,
            mirna_score_columns,
            mirna_profiles,
            candidate_ids,
        )
        write_target_phact_positions(
            split,
            args.phact_cache_dir,
            output_dir,
            sample_sequences,
        )
    shutil.copyfile(
        args.orthologues_file,
        output_dir / "train" / "input" / "mirgenedb_premirna_orthologues.tsv",
    )
    for split in ("test", "leftout"):
        shutil.copyfile(
            args.orthologues_file,
            output_dir / split / "input" / "mirgenedb_premirna_orthologues.tsv",
        )
    write_metadata(output_dir, args.sample_count)
    print(f"mock_dir={output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--phact-cache-dir", type=Path, default=DEFAULT_PHACT_CACHE_DIR)
    parser.add_argument("--mirna-phact-file", type=Path, default=DEFAULT_MIRNA_PHACT)
    parser.add_argument("--orthologues-file", type=Path, default=DEFAULT_ORTHOLOGUES)
    build_mock(parser.parse_args())


if __name__ == "__main__":
    main()
