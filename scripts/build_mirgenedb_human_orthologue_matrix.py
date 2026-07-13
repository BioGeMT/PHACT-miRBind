#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


VERSION_SUFFIX_RE = re.compile(r"-v\d+$")


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name: str | None = None
    seq_parts: list[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(seq_parts)))
                name = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
    if name is not None:
        records.append((name, "".join(seq_parts)))
    return records


def mature_locus(mature_id: str) -> str:
    return mature_id.rsplit("_", 1)[0]


def normalize_locus_version(locus: str) -> str:
    return VERSION_SUFFIX_RE.sub("", locus)


def family_lookup_key(mature_id: str) -> str:
    return normalize_locus_version(mature_locus(mature_id))


def orthologue_key(mature_id: str) -> tuple[str, str]:
    species, rest = mature_id.split("-", 1)
    locus, arm = rest.rsplit("_", 1)
    return locus, arm


def read_family_map(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {row["MirGeneDB_ID"]: row["Family"] for row in reader}


def read_species_order(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        species = [row["3_letter_code"].capitalize() for row in reader]
    return ["Hsa", *[code for code in species if code != "Hsa"]]


def write_matrix(args: argparse.Namespace) -> None:
    all_records = read_fasta(args.all_mature_fasta)
    hsa_records = read_fasta(args.hsa_mature_fasta)
    family_map = read_family_map(args.family_map)
    species_order = read_species_order(args.species_metadata)

    species_to_keys: dict[str, set[tuple[str, str]]] = {}
    for mature_id, _seq in all_records:
        species = mature_id.split("-", 1)[0]
        species_to_keys.setdefault(species, set()).add(orthologue_key(mature_id))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.output_dir / "mirgenedb_human_mature_orthologue_presence_matrix.tsv"
    summary_path = args.output_dir / "mirgenedb_human_mature_orthologue_presence_matrix.summary.json"

    header = [
        "mirgenedb_mature_id",
        "family",
        "mature_sequence",
        "orthologue_species_count_excluding_hsa",
        *species_order,
    ]

    with matrix_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for mature_id, sequence in hsa_records:
            key = orthologue_key(mature_id)
            presence = [
                1 if key in species_to_keys.get(species, set()) else 0
                for species in species_order
            ]
            family = family_map.get(family_lookup_key(mature_id), "NA")
            writer.writerow([
                mature_id,
                family,
                sequence.replace("U", "T"),
                sum(value for species, value in zip(species_order, presence) if species != "Hsa"),
                *presence,
            ])

    summary = {
        "human_mature_mirnas": len(hsa_records),
        "species_columns": len(species_order),
        "source_all_mature_fasta": str(args.all_mature_fasta),
        "source_hsa_mature_fasta": str(args.hsa_mature_fasta),
        "source_species_metadata": str(args.species_metadata),
        "source_family_map": str(args.family_map),
        "family_lookup_rule": (
            "Family map is precursor-level, so mature IDs are stripped of _3p/_5p "
            "and terminal version suffixes such as -v1 before lookup."
        ),
        "counting_rule": (
            "Rows are Hsa mature MirGeneDB IDs. Species column is 1 if all_mature.fas "
            "contains the same MirGeneDB precursor key and mature arm in that species."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"matrix={matrix_path}")
    print(f"summary={summary_path}")
    print(f"rows={len(hsa_records)}")
    print(f"species_columns={len(species_order)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all-mature-fasta",
        type=Path,
        default=Path("mirna_alignment/miRNA_mature_files/all_mature.fas"),
    )
    parser.add_argument(
        "--hsa-mature-fasta",
        type=Path,
        default=Path("mirna_alignment/miRNA_mature_files/hsa_mature.fas"),
    )
    parser.add_argument(
        "--species-metadata",
        type=Path,
        default=Path("mirna_alignment/miRNA_mature_files/mirgenedb_species_with_taxonomy.csv"),
    )
    parser.add_argument(
        "--family-map",
        type=Path,
        default=Path("mirna_alignment/miRNA_mature_files/mirgenedb_family_mappings.tsv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mirgenedb_orthologue_counts"),
    )
    write_matrix(parser.parse_args())


if __name__ == "__main__":
    main()
