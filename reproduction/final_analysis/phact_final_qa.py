#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workspace import WORKSPACE, historical_path

import csv
import math
from collections import Counter, defaultdict

from PIL import Image


root = WORKSPACE / "analyses/final"

with (root / "all_model_test_leftout_auprc.tsv").open(newline="") as handle:
    score_rows = list(csv.DictReader(handle, delimiter="\t"))
assert len(score_rows) == 50
assert len({(row["category"], row["model"]) for row in score_rows}) == len(score_rows)
assert len({row["run_id"] for row in score_rows}) == len(score_rows)
assert Counter(row["presentation_status"] for row in score_rows) == {
    "primary": 48,
    "diagnostic": 1,
    "duplicate": 1,
}
for row in score_rows:
    assert row["run_id"]
    assert row["legacy_category"]
    assert row["legacy_model_name"]
    assert row["model_explanation"].endswith(".")
    for field in ("test_auprc", "leftout_auprc"):
        value = float(row[field])
        assert 0 <= value <= 1, (row["model"], field, value)
    for raw_path in row["source_artifact"].split("; "):
        assert historical_path(raw_path).is_file(), raw_path

categories = Counter(row["category"] for row in score_rows)
score_keys = {row["run_id"] for row in score_rows}
assert next(
    row for row in score_rows if row["run_id"] == "agentomics_iteration_37"
)["presentation_status"] == "diagnostic"
assert next(
    row for row in score_rows if row["run_id"] == "agentomics_iteration_38"
)["presentation_status"] == "duplicate"

with (root / "all_models_plot_annotations.tsv").open(newline="") as handle:
    annotation_rows = list(csv.DictReader(handle, delimiter="\t"))
assert len(annotation_rows) == 9
assert all(row["run_id"] in score_keys for row in annotation_rows)
assert not {"agentomics_iteration_37", "agentomics_iteration_38"} & {
    row["run_id"] for row in annotation_rows
}
assert sum("category leader" in row["selection_reason"] for row in annotation_rows) == 8
assert sum("overall best test" in row["selection_reason"] for row in annotation_rows) == 1
assert sum("overall best left-out" in row["selection_reason"] for row in annotation_rows) == 1

model_image = Image.open(root / "all_models_test_vs_leftout_auprc.png")
assert model_image.width >= 3000 and model_image.height >= 1500, model_image.size

for superseded in (
    "mir17_let7_countnodes3_alt_minus_ref_full_premirna.tsv",
    "mir17_let7_countnodes3_alt_minus_ref_full_premirna.png",
    "mir17_let7_countnodes3_alt_minus_ref_full_premirna.pdf",
    "mir17_let7_countnodes3_alt_minus_ref_mature.tsv",
    "mir17_let7_countnodes3_alt_minus_ref_mature.png",
    "mir17_let7_countnodes3_alt_minus_ref_mature.pdf",
    "top5_manakov_mirnas.tsv",
    "top5_manakov_countnodes3_all_nucleotides_plot_data.tsv",
    "top5_manakov_countnodes3_all_nucleotides_premirna_profiles.png",
    "top5_manakov_countnodes3_all_nucleotides_premirna_profiles.pdf",
    "top5_manakov_countnodes3_coverage.json",
    "top5_manakov_countnodes3_plot_data.tsv",
    "top5_manakov_countnodes3_premirna_profiles.png",
    "top5_manakov_countnodes3_premirna_profiles.pdf",
    "top10_manakov_mirnas.tsv",
    "top10_manakov_countnodes3_all_nucleotides_plot_data.tsv",
    "top10_manakov_countnodes3_all_nucleotides_premirna_profiles.png",
    "top10_manakov_countnodes3_all_nucleotides_premirna_profiles.pdf",
    "top10_manakov_countnodes3_coverage.json",
    "top10_manakov_countnodes3_plot_data.tsv",
    "top10_manakov_countnodes3_premirna_profiles.png",
    "top10_manakov_countnodes3_premirna_profiles.pdf",
):
    assert not (root / superseded).exists(), superseded

with (
    root / "mir17_let7_countnodes3_ref_minus_alt_full_premirna.tsv"
).open(newline="") as handle:
    full_rows = list(csv.DictReader(handle, delimiter="\t"))
with (
    root / "mir17_let7_countnodes3_ref_minus_alt_mature.tsv"
).open(newline="") as handle:
    mature_rows = list(csv.DictReader(handle, delimiter="\t"))
with (
    root / "mir17_let7_countnodes3_old_vs_consensustree_ref_minus_alt.tsv"
).open(newline="") as handle:
    comparison_rows = list(csv.DictReader(handle, delimiter="\t"))

expected_lengths = {"Hsa-Mir-17-P1a": 60, "Hsa-Let-7-P1b": 67}
expected_mature_lengths = {"Hsa-Mir-17-P1a": 23, "Hsa-Let-7-P1b": 22}
assert len(full_rows) == sum(expected_lengths.values()) == 127
assert len(mature_rows) == sum(expected_mature_lengths.values()) == 45
assert len(comparison_rows) == sum(expected_lengths.values()) == 127

full_by_precursor: dict[str, list[dict[str, str]]] = defaultdict(list)
for row in full_rows:
    precursor_id = row["precursor_id"]
    full_by_precursor[precursor_id].append(row)
    reference_nt = row["reference_nt"]
    assert reference_nt in {"A", "C", "G", "T"}
    nucleotide_scores = {
        nucleotide: float(row[f"score_{nucleotide}"])
        for nucleotide in ("A", "C", "G", "T")
    }
    assert all(0 <= value <= 1.1 for value in nucleotide_scores.values())
    reference_score = float(row["reference_score"])
    assert abs(reference_score - nucleotide_scores[reference_nt]) < 1e-12
    alternative_nts = sorted({"A", "C", "G", "T"} - {reference_nt})
    assert row["alternative_nts"] == ",".join(alternative_nts)
    expected_mean = sum(nucleotide_scores[nt] for nt in alternative_nts) / 3
    observed_mean = float(row["mean_alternative_score"])
    observed_delta = float(row["reference_minus_mean_alternative"])
    assert abs(observed_mean - expected_mean) < 1e-12
    assert abs(observed_delta - (reference_score - expected_mean)) < 1e-12

for precursor_id, expected_length in expected_lengths.items():
    rows = full_by_precursor[precursor_id]
    positions = [int(row["precursor_position_1based"]) for row in rows]
    assert positions == list(range(1, expected_length + 1))

mature_by_precursor: dict[str, list[dict[str, str]]] = defaultdict(list)
for row in mature_rows:
    mature_by_precursor[row["precursor_id"]].append(row)
for precursor_id, expected_length in expected_mature_lengths.items():
    rows = mature_by_precursor[precursor_id]
    mature_positions = [int(row["mature_position_1based"]) for row in rows]
    assert mature_positions == list(range(1, expected_length + 1))
    assert all(row["mature_id"].endswith("_5p") for row in rows)

comparison_by_precursor: dict[str, list[dict[str, str]]] = defaultdict(list)
contrast_differences = []
for row in comparison_rows:
    precursor_id = row["precursor_id"]
    comparison_by_precursor[precursor_id].append(row)
    position = int(row["precursor_position_1based"])
    mature_start = int(row["mature_start_in_precursor_1based"])
    mature_end = int(row["mature_end_in_precursor_1based"])
    assert (row["is_mature"] == "True") == (mature_start <= position <= mature_end)
    reference_nt = row["reference_nt"]
    alternative_nts = sorted({"A", "C", "G", "T"} - {reference_nt})
    assert row["alternative_nts"] == ",".join(alternative_nts)
    source_contrasts = {}
    for source in ("old", "new"):
        scores = {
            nucleotide: float(row[f"{source}_score_{nucleotide}"])
            for nucleotide in ("A", "C", "G", "T")
        }
        assert all(math.isfinite(value) for value in scores.values())
        reference_score = float(row[f"{source}_reference_score"])
        mean_alternative = float(row[f"{source}_mean_alternative_score"])
        contrast = float(row[f"{source}_reference_minus_mean_alternative"])
        assert abs(reference_score - scores[reference_nt]) < 1e-12
        expected_mean = sum(scores[nucleotide] for nucleotide in alternative_nts) / 3
        assert abs(mean_alternative - expected_mean) < 1e-12
        assert abs(contrast - (reference_score - expected_mean)) < 1e-12
        source_contrasts[source] = contrast
    contrast_differences.append(
        abs(source_contrasts["new"] - source_contrasts["old"])
    )

for precursor_id, expected_length in expected_lengths.items():
    positions = [
        int(row["precursor_position_1based"])
        for row in comparison_by_precursor[precursor_id]
    ]
    assert positions == list(range(1, expected_length + 1))
assert max(contrast_differences) > 1e-6

full_image = Image.open(
    root / "mir17_let7_countnodes3_ref_minus_alt_full_premirna.png"
)
mature_image = Image.open(
    root / "mir17_let7_countnodes3_ref_minus_alt_mature.png"
)
assert full_image.width >= 2500 and full_image.height >= 900
assert mature_image.width >= 2500 and mature_image.height >= 900
comparison_images = []
for filename in (
    "mir17_countnodes3_old_vs_consensustree_ref_minus_alt_full_premirna.png",
    "let7_countnodes3_old_vs_consensustree_ref_minus_alt_full_premirna.png",
):
    image = Image.open(root / filename)
    assert image.width >= 1800 and image.height >= 900, image.size
    comparison_images.append(image)

best_test = max(score_rows, key=lambda row: float(row["test_auprc"]))
best_leftout = max(score_rows, key=lambda row: float(row["leftout_auprc"]))
print(f"categories={dict(sorted(categories.items()))}")
print(
    f"model_plot_annotations={len(annotation_rows)} "
    f"model_image={model_image.width}x{model_image.height}"
)
print(
    f"full_profile_rows={len(full_rows)} mature_profile_rows={len(mature_rows)} "
    f"full_image={full_image.width}x{full_image.height} "
    f"mature_image={mature_image.width}x{mature_image.height}"
)
print(
    f"old_new_rows={len(comparison_rows)} "
    f"max_contrast_change={max(contrast_differences):.9f} "
    f"comparison_images={[image.size for image in comparison_images]}"
)
print(
    f"best_test={best_test['model']} {float(best_test['test_auprc']):.9f}"
)
print(
    f"best_leftout={best_leftout['model']} "
    f"{float(best_leftout['leftout_auprc']):.9f}"
)
