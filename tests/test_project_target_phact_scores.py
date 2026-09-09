import csv
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "project_target_phact_scores.py"
SPEC = importlib.util.spec_from_file_location("project_target_phact_scores", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["gene", "noncodingRNA", "label", "chr", "start", "end", "strand"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def data_row(*, gene: str, start: int, strand: str = "+") -> dict[str, str]:
    return {
        "gene": gene,
        "noncodingRNA": "ACGT",
        "label": "1",
        "chr": "1",
        "start": str(start),
        "end": str(start + 49),
        "strand": strand,
    }


def test_projects_scores_by_chromosome_strand_and_position(tmp_path: Path):
    input_path = tmp_path / "input.tsv"
    manakov_train = tmp_path / "manakov_train.tsv"
    empty = tmp_path / "empty.tsv"
    target_scores = tmp_path / "target.tsv"
    output = tmp_path / "output.tsv"
    sequence = "A" * 50
    write_rows(input_path, [data_row(gene=sequence, start=100)])
    write_rows(manakov_train, [data_row(gene=sequence, start=100)])
    write_rows(empty, [])
    target_scores.write_text(
        "split\tmanakov_row_id\ttarget_position_1based\tgenomic_position\t"
        "actual_nt\tscore_A\tscore_C\tscore_G\tscore_T\n"
        "train\t1\t1\t100\tA\t0.1\t0.2\t0.3\t0.4\n"
        "train\t1\t2\t101\tA\t0.5\t0.6\t0.7\t0.8\n"
    )

    wanted, interval_stats = MODULE.collect_wanted_positions(input_path)
    scores, scan_stats = MODULE.load_matching_scores(
        target_scores,
        {"train": manakov_train, "test": empty, "leftout": empty},
        wanted,
    )
    output_stats = MODULE.write_projected_scores(
        input_path,
        output,
        scores,
        split="train",
    )

    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    assert interval_stats["unique_wanted_positions"] == 50
    assert scan_stats["matched_unique_positions"] == 2
    assert output_stats["matched_position_rows"] == 2
    assert output_stats["missing_position_rows"] == 48
    assert rows[0]["score_A"] == "0.1"
    assert rows[1]["score_T"] == "0.8"
    assert rows[2]["score_A"] == "NA"


def test_reverse_strand_uses_end_to_start_coordinates(tmp_path: Path):
    input_path = tmp_path / "input.tsv"
    output = tmp_path / "output.tsv"
    write_rows(input_path, [data_row(gene="C" * 50, start=100, strand="-")])
    scores = {("chr1", "-"): {149: ("1", "2", "3", "4")}}

    MODULE.write_projected_scores(input_path, output, scores, split="train")

    with output.open(newline="") as handle:
        first = next(csv.DictReader(handle, delimiter="\t"))
    assert first["genomic_position"] == "149"
    assert first["actual_nt"] == "C"
    assert first["score_G"] == "3"
