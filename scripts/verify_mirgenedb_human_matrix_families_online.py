#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from urllib.request import urlopen

from bs4 import BeautifulSoup


VERSION_SUFFIX_RE = re.compile(r"-v\d+$")


def precursor_from_mature_id(mature_id: str) -> str:
    locus = mature_id.rsplit("_", 1)[0]
    return VERSION_SUFFIX_RE.sub("", locus)


def fetch_online_human_families(url: str) -> dict[str, str]:
    html = urlopen(url, timeout=30).read()
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"class": "browse"})
    if table is None:
        raise RuntimeError("Could not find MirGeneDB browse table")

    families: dict[str, str] = {}
    for row in table.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
        if len(cells) >= 3 and cells[0].startswith("Hsa-"):
            families[cells[0].split()[0]] = cells[2]
    return families


def verify(args: argparse.Namespace) -> None:
    online_families = fetch_online_human_families(args.url)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mismatches: list[tuple[str, str, str, str]] = []
    missing_online: list[tuple[str, str, str]] = []
    checked = 0

    with args.matrix.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            mature_id = row["mirgenedb_mature_id"]
            precursor_id = precursor_from_mature_id(mature_id)
            matrix_family = row["family"]
            online_family = online_families.get(precursor_id)
            checked += 1
            if online_family is None:
                missing_online.append((mature_id, precursor_id, matrix_family))
            elif matrix_family != online_family:
                mismatches.append((mature_id, precursor_id, matrix_family, online_family))

    report_path = args.output_dir / "mirgenedb_human_mature_family_online_verification.tsv"
    with report_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "status",
            "mirgenedb_mature_id",
            "online_precursor_id",
            "matrix_family",
            "online_family",
        ])
        for mature_id, precursor_id, matrix_family, online_family in mismatches:
            writer.writerow([
                "mismatch",
                mature_id,
                precursor_id,
                matrix_family,
                online_family,
            ])
        for mature_id, precursor_id, matrix_family in missing_online:
            writer.writerow([
                "missing_online_precursor",
                mature_id,
                precursor_id,
                matrix_family,
                "NA",
            ])

    summary = {
        "source_url": args.url,
        "online_human_precursor_rows": len(online_families),
        "matrix_mature_rows_checked": checked,
        "family_mismatches": len(mismatches),
        "missing_online_precursors": len(missing_online),
        "verification_tsv": str(report_path),
    }
    summary_path = args.output_dir / "mirgenedb_human_mature_family_online_verification.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    if mismatches or missing_online:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(
            "outputs/mirgenedb_orthologue_counts/"
            "mirgenedb_human_mature_orthologue_presence_matrix.tsv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mirgenedb_orthologue_counts"),
    )
    parser.add_argument("--url", default="https://mirgenedb.org/browse/hsa")
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
