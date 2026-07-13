#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup


DEFAULT_BASE_URL = "https://mirgenedb.org"
USER_AGENT = "PHACT-miRBind orthologue matrix builder"


@dataclass(frozen=True)
class HumanPrecursor:
    mirgenedb_id: str
    family: str
    show_url: str


def fetch_soup(url: str) -> BeautifulSoup:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return BeautifulSoup(response.read(), "html.parser")


def read_species_order(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        species = [row["3_letter_code"].capitalize() for row in reader]
    return ["Hsa", *[code for code in species if code != "Hsa"]]


def fetch_human_precursors(base_url: str) -> list[HumanPrecursor]:
    soup = fetch_soup(urljoin(base_url, "/browse/hsa"))
    table = soup.find("table", {"class": "browse"})
    if table is None:
        raise RuntimeError("Could not find MirGeneDB human browse table")

    precursors: list[HumanPrecursor] = []
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 3:
            continue
        link = cells[0].find("a", href=True)
        if link is None:
            continue
        mirgenedb_id = link.get_text(" ", strip=True)
        if not mirgenedb_id.startswith("Hsa-"):
            continue
        precursors.append(
            HumanPrecursor(
                mirgenedb_id=mirgenedb_id,
                family=cells[2].get_text(" ", strip=True),
                show_url=urljoin(base_url, link["href"]),
            )
        )
    return precursors


def fetch_orthologue_ids(show_url: str) -> list[str]:
    soup = fetch_soup(show_url)
    for row in soup.find_all("tr"):
        header = row.find("th")
        if header and header.get_text(" ", strip=True) == "Orthologues":
            return [
                link.get_text(" ", strip=True)
                for link in row.find_all("a")
                if link.get_text(" ", strip=True)
            ]
    return []


def species_code(mirgenedb_id: str) -> str:
    return mirgenedb_id.split("-", 1)[0]


def write_matrix(args: argparse.Namespace) -> None:
    species_order = read_species_order(args.species_metadata)
    precursors = fetch_human_precursors(args.base_url)
    if not precursors:
        raise RuntimeError("No human MirGeneDB precursor rows found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.output_dir / "mirgenedb_human_precursor_orthologue_presence_matrix.tsv"
    summary_path = (
        args.output_dir
        / "mirgenedb_human_precursor_orthologue_presence_matrix.summary.json"
    )

    rows: list[list[object]] = []
    unknown_species: set[str] = set()
    total_orthologue_ids = 0
    max_orthologue_ids = 0
    max_orthologue_species = 0

    for index, precursor in enumerate(precursors, start=1):
        orthologue_ids = fetch_orthologue_ids(precursor.show_url)
        orthologue_species = {species_code(orthologue_id) for orthologue_id in orthologue_ids}
        unknown_species.update(orthologue_species - set(species_order))
        total_orthologue_ids += len(orthologue_ids)
        max_orthologue_ids = max(max_orthologue_ids, len(orthologue_ids))
        max_orthologue_species = max(max_orthologue_species, len(orthologue_species))

        presence = [
            1 if species == "Hsa" or species in orthologue_species else 0
            for species in species_order
        ]
        rows.append(
            [
                precursor.mirgenedb_id,
                precursor.family,
                len(orthologue_species),
                *presence,
            ]
        )

        if args.progress_every and index % args.progress_every == 0:
            print(
                f"fetched {index}/{len(precursors)} precursor pages",
                flush=True,
            )
        if args.sleep_sec:
            time.sleep(args.sleep_sec)

    header = [
        "mirgenedb_premirna_id",
        "family",
        "orthologue_species_count_excluding_hsa",
        *species_order,
    ]
    with matrix_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)

    summary = {
        "source_browse_url": urljoin(args.base_url, "/browse/hsa"),
        "human_precursor_rows": len(rows),
        "species_columns": len(species_order),
        "counting_rule": (
            "Rows are human MirGeneDB precursor genes from /browse/hsa. "
            "For each row, species is 1 when that species appears in the "
            "source precursor page's Orthologues table row; mature 5p/3p arms "
            "are not checked. Hsa is set to 1 for every human precursor."
        ),
        "total_orthologue_precursor_ids_listed": total_orthologue_ids,
        "max_orthologue_precursor_ids_for_one_human_precursor": max_orthologue_ids,
        "max_orthologue_species_for_one_human_precursor": max_orthologue_species,
        "unknown_species_codes": sorted(unknown_species),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"matrix={matrix_path}")
    print(f"summary={summary_path}")
    print(f"rows={len(rows)}")
    print(f"species_columns={len(species_order)}")
    print(f"unknown_species_codes={','.join(sorted(unknown_species)) or 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a human pre-miRNA x species matrix from MirGeneDB online "
            "Orthologues lists."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--species-metadata",
        type=Path,
        default=Path("mirna_alignment/miRNA_mature_files/mirgenedb_species_with_taxonomy.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mirgenedb_orthologue_counts"),
    )
    parser.add_argument("--sleep-sec", type=float, default=0.02)
    parser.add_argument("--progress-every", type=int, default=50)
    write_matrix(parser.parse_args())


if __name__ == "__main__":
    main()
