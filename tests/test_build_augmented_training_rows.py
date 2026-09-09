import csv
import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_augmented_training_rows.py"
SPEC = importlib.util.spec_from_file_location("build_augmented_training_rows", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


HEADER = [
    "gene",
    "noncodingRNA",
    "noncodingRNA_name",
    "noncodingRNA_fam",
    "feature",
    "label",
    "chr",
    "start",
    "end",
    "strand",
    "Nunique",
    "dominant_region",
    "regions_present",
    "read_start_in_sel_tx_1based",
    "read_end_in_sel_tx_1based",
    "gene_cluster_ID",
    "gene_phyloP",
    "gene_phastCons",
]


def row(gene: str, mirna: str, family: str, label: str) -> dict[str, str]:
    values = {column: "" for column in HEADER}
    values.update(
        {
            "gene": gene,
            "noncodingRNA": mirna,
            "noncodingRNA_name": mirna,
            "noncodingRNA_fam": family,
            "label": label,
            "chr": "1",
            "start": "1",
            "end": "50",
            "strand": "+",
            "gene_cluster_ID": "1",
        }
    )
    return values


def write_tsv(path: Path, rows: list[dict[str, str]], *, row_ids: bool = False) -> None:
    fields = [*HEADER, "manakov_row_id"] if row_ids else HEADER
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index, values in enumerate(rows, start=1):
            output = dict(values)
            if row_ids:
                output["manakov_row_id"] = str(index)
            writer.writerow(output)


def test_filters_evaluation_pairs_leftout_families_and_duplicates(tmp_path: Path):
    train = tmp_path / "train.tsv"
    val = tmp_path / "val.tsv"
    test = tmp_path / "test.tsv"
    leftout = tmp_path / "leftout.tsv"
    gse = tmp_path / "gse.tsv"
    write_tsv(
        train,
        [row("TRAIN", "M1", "train-family", "1"), row("DROP", "M2", "f", "0")],
        row_ids=True,
    )
    write_tsv(val, [row("VAL", "M3", "f", "1")])
    write_tsv(test, [row("TEST", "M4", "f", "1")])
    write_tsv(leftout, [row("LEFT", "M5", "heldout-family", "1")])
    write_tsv(
        gse,
        [
            row("TRAIN", "M1", "train-family", "0"),
            row("VAL", "M3", "f", "1"),
            row("TEST", "M4", "f", "1"),
            row("LEFT", "M5", "heldout-family", "1"),
            row("NEW1", "M6", "heldout-family", "1"),
            row("NEW2", "M5", "different-name", "0"),
            row("KEEP", "M7", "new-family", "1"),
            row("KEEP", "M7", "new-family", "1"),
        ],
    )
    excluded = tmp_path / "excluded.txt"
    excluded.write_text("2\n")
    output_dir = tmp_path / "output"

    val_pairs, _, _, _ = MODULE.read_reference(val)
    test_pairs, _, _, _ = MODULE.read_reference(test)
    left_pairs, left_families, left_mirnas, _ = MODULE.read_reference(
        leftout, collect_families=True
    )
    train_pairs, train_stats = MODULE.write_clean_manakov_train(
        train,
        output_dir / "manakov_train_clean.tsv",
        MODULE.read_excluded_row_ids(excluded),
    )
    totals, sources = MODULE.write_filtered_gse_rows(
        [("gse", gse)],
        output_dir / "gse.tsv",
        manakov_train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        leftout_pairs=left_pairs,
        leftout_families=left_families,
        leftout_mirnas=left_mirnas,
    )

    with (output_dir / "gse.tsv").open(newline="") as handle:
        kept = list(csv.DictReader(handle, delimiter="\t"))

    assert train_stats["output_rows"] == 1
    assert train_stats["excluded_problematic_row_ids"] == 1
    assert [item["gene"] for item in kept] == ["KEEP"]
    assert totals["output_rows"] == 1
    assert sources["gse"]["reason_manakov_train_label_conflict"] == 1
    assert sources["gse"]["reason_manakov_val_pair"] == 1
    assert sources["gse"]["reason_manakov_test_pair"] == 1
    assert sources["gse"]["reason_manakov_leftout_pair"] == 1
    assert sources["gse"]["reason_leftout_family"] == 2
    assert sources["gse"]["reason_leftout_mirna"] == 2
    assert sources["gse"]["reason_prior_gse_pair"] == 1


def test_summary_records_evaluation_hashes(tmp_path: Path, monkeypatch):
    train = tmp_path / "train.tsv"
    val = tmp_path / "val.tsv"
    test = tmp_path / "test.tsv"
    leftout = tmp_path / "leftout.tsv"
    gse = tmp_path / "gse.tsv"
    write_tsv(train, [row("TRAIN", "M1", "f", "1")], row_ids=True)
    write_tsv(val, [row("VAL", "M2", "f", "0")])
    write_tsv(test, [row("TEST", "M3", "f", "1")])
    write_tsv(leftout, [row("LEFT", "M4", "heldout", "0")])
    write_tsv(gse, [row("KEEP", "M5", "new", "1")])
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT_PATH),
            "--manakov-train",
            str(train),
            "--manakov-val",
            str(val),
            "--manakov-test",
            str(test),
            "--manakov-leftout",
            str(leftout),
            "--gse",
            f"gse={gse}",
            "--output-dir",
            str(output_dir),
        ],
    )

    MODULE.main()

    summary = json.loads((output_dir / "augmentation_summary.json").read_text())
    assert summary["evaluation_sha256"]["test"] == MODULE.sha256(test)
    assert summary["evaluation_sha256"]["leftout"] == MODULE.sha256(leftout)
    assert summary["gse_total"]["output_rows"] == 1
