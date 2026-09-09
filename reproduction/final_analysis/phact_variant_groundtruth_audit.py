#!/usr/bin/env python3
"""Audit old/new CountNodes_3 associations with pri-miRNA variants."""

from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workspace import WORKSPACE

import csv
import gzip
import json
import math
import random
from collections import Counter, defaultdict
from statistics import mean, median


SCRATCH = WORKSPACE
SEQUENCE_TABLE = WORKSPACE / "data/inputs/reference/hsa_premirnas_flank30.tsv"
EVALUATION_ROOT = (
    SCRATCH / "data/inputs/score_import_20260813/old/evaluation_set/evaluation_set"
)
OLD_SCORE_TABLE = (
    SCRATCH
    / "data/inputs/score_import_20260813/old/PHACT_scores_ALL_0226/results_0226/orthologs_qntnorm_transformed.tsv.gz"
)
NEW_SCORE_TABLE = (
    SCRATCH
    / "data/inputs/score_import_20260813/new/PHACT_scores_orthologs_with_consensusTree/results_qntnorm_transformed.tsv.gz"
)
SCORE_COLUMN = "PHACTn_gapAware_wtNTnorm_wl_param_CountNodes_3"
NUCLEOTIDES = {"A", "C", "G", "T"}
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def dna(value: str) -> str:
    return value.strip().upper().replace("U", "T")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def read_sequences() -> dict[str, str]:
    sequences = {}
    with SEQUENCE_TABLE.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sequences[row["mirgenedb_id"]] = dna(row["flank30_sequence"])
    return sequences


def read_scores(path: Path) -> dict[tuple[str, int, str], float]:
    scores = {}
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if SCORE_COLUMN not in (reader.fieldnames or []):
            raise ValueError(f"Missing {SCORE_COLUMN} in {path}")
        for row in reader:
            scores[(row["ID"], int(row["Position"]), row["Nucleotide"])] = float(
                row[SCORE_COLUMN]
            )
    return scores


def read_variants(path: Path, source: str) -> list[dict[str, object]]:
    variants = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            precursor_id = row["id"].removesuffix("_pri")
            position_field = "PositionVCF" if "PositionVCF" in row else "Position"
            reference_field = "ReferenceAlleleVCF" if "ReferenceAlleleVCF" in row else "REF"
            alternate_field = "AlternateAlleleVCF" if "AlternateAlleleVCF" in row else "ALT"
            variants.append(
                {
                    "source": source,
                    "chromosome": row["Chromosome"],
                    "precursor_id": precursor_id,
                    "strand": row["strand"],
                    "genomic_position_1based": int(row[position_field]),
                    "reference_genomic": dna(row[reference_field]),
                    "alternate_genomic": dna(row[alternate_field]),
                    "clinical_significance": row.get("ClinicalSignificance", ""),
                    "allele_frequency": (
                        float(row["allele_frequency"])
                        if row.get("allele_frequency") not in {None, "", ".", "NA"}
                        else None
                    ),
                }
            )
    return variants


def infer_genomic_start(
    precursor_id: str,
    variants: list[dict[str, object]],
    transcript_sequence: str,
) -> dict[str, object]:
    strands = {str(row["strand"]) for row in variants}
    chromosomes = {str(row["chromosome"]) for row in variants}
    if len(strands) != 1 or len(chromosomes) != 1:
        raise ValueError(f"Inconsistent locus metadata for {precursor_id}")
    strand = next(iter(strands))
    genomic_sequence = (
        transcript_sequence if strand == "+" else reverse_complement(transcript_sequence)
    )
    single_base_variants = [
        row
        for row in variants
        if len(str(row["reference_genomic"])) == 1
        and str(row["reference_genomic"]) in NUCLEOTIDES
    ]
    candidate_support: Counter[int] = Counter()
    for row in single_base_variants:
        position = int(row["genomic_position_1based"])
        reference = str(row["reference_genomic"])
        for sequence_index, nucleotide in enumerate(genomic_sequence):
            if nucleotide == reference:
                candidate_support[position - sequence_index] += 1
    if not candidate_support:
        return {
            "precursor_id": precursor_id,
            "start": None,
            "best_matches": 0,
            "second_matches": 0,
            "single_base_variants": 0,
            "contained_variants": 0,
            "reference_match_fraction": 0.0,
        }

    def candidate_metrics(start: int) -> tuple[int, int]:
        end = start + len(genomic_sequence) - 1
        contained = sum(
            start <= int(row["genomic_position_1based"]) <= end for row in variants
        )
        matches = sum(
            start <= int(row["genomic_position_1based"]) <= end
            and genomic_sequence[int(row["genomic_position_1based"]) - start]
            == str(row["reference_genomic"])
            for row in single_base_variants
        )
        return matches, contained

    ranked = sorted(
        (
            (*candidate_metrics(start), candidate_support[start], start)
            for start in candidate_support
        ),
        reverse=True,
    )
    best_matches, contained, _, start = ranked[0]
    second_matches = ranked[1][0] if len(ranked) > 1 else 0
    return {
        "precursor_id": precursor_id,
        "start": start,
        "best_matches": best_matches,
        "second_matches": second_matches,
        "single_base_variants": len(single_base_variants),
        "contained_variants": contained,
        "reference_match_fraction": best_matches / len(single_base_variants),
    }


def orient_allele(allele: str, strand: str) -> str:
    return allele if strand == "+" else allele.translate(COMPLEMENT)


def average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2
        for index in ordered[cursor:end]:
            ranks[index] = average_rank
        cursor = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else float("nan")


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(average_ranks(left), average_ranks(right))


def summarize_values(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "min": min(values),
        "median": median(values),
        "mean": mean(values),
        "max": max(values),
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def cluster_bootstrap_correlations(
    rows: list[dict[str, object]], iterations: int = 1000
) -> dict[str, object]:
    rows_by_precursor: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_precursor[str(row["precursor_id"])].append(row)
    precursor_ids = sorted(rows_by_precursor)
    random_generator = random.Random(20260816)
    estimates = {"old": [], "new": [], "new_minus_old": []}
    for _ in range(iterations):
        sampled_rows = []
        for _ in precursor_ids:
            sampled_rows.extend(
                rows_by_precursor[random_generator.choice(precursor_ids)]
            )
        rarity = [
            -math.log10(float(row["allele_frequency"])) for row in sampled_rows
        ]
        old_delta = [float(row["old_delta"]) for row in sampled_rows]
        new_delta = [float(row["new_delta"]) for row in sampled_rows]
        old_correlation = spearman(old_delta, rarity)
        new_correlation = spearman(new_delta, rarity)
        estimates["old"].append(old_correlation)
        estimates["new"].append(new_correlation)
        estimates["new_minus_old"].append(new_correlation - old_correlation)
    interval_summaries = {
        key: {
            "median": median(values),
            "confidence_interval_95pct": [
                percentile(values, 0.025),
                percentile(values, 0.975),
            ],
        }
        for key, values in estimates.items()
    }
    return {
        "iterations": iterations,
        "precursor_clusters": len(precursor_ids),
        **interval_summaries,
    }


def main() -> None:
    sequences = read_sequences()
    old_scores = read_scores(OLD_SCORE_TABLE)
    new_scores = read_scores(NEW_SCORE_TABLE)
    score_ids = {key[0] for key in old_scores} & {key[0] for key in new_scores}

    clinvar = read_variants(
        EVALUATION_ROOT / "clinvar_intersect_mirna_pri_final.tsv", "clinvar"
    )
    gnomad = read_variants(
        EVALUATION_ROOT / "gnomad_intersect_mirna_pri_final.tsv", "gnomad"
    )
    all_variants = clinvar + gnomad
    variants_by_id: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in all_variants:
        variants_by_id[str(row["precursor_id"])].append(row)

    mappings = {}
    for precursor_id, variants in variants_by_id.items():
        if precursor_id not in sequences or precursor_id not in score_ids:
            continue
        mappings[precursor_id] = infer_genomic_start(
            precursor_id, variants, sequences[precursor_id]
        )

    joined = []
    failure_reasons: Counter[str] = Counter()
    seen = set()
    for row in all_variants:
        precursor_id = str(row["precursor_id"])
        deduplication_key = (
            row["source"],
            precursor_id,
            row["chromosome"],
            row["genomic_position_1based"],
            row["reference_genomic"],
            row["alternate_genomic"],
            row["clinical_significance"],
            row["allele_frequency"],
        )
        if deduplication_key in seen:
            failure_reasons["duplicate_row"] += 1
            continue
        seen.add(deduplication_key)
        if precursor_id not in sequences:
            failure_reasons["missing_sequence"] += 1
            continue
        if precursor_id not in score_ids:
            failure_reasons["missing_old_or_new_scores"] += 1
            continue
        mapping = mappings.get(precursor_id)
        if not mapping or mapping["start"] is None:
            failure_reasons["no_coordinate_mapping"] += 1
            continue
        reference_genomic = str(row["reference_genomic"])
        alternate_genomic = str(row["alternate_genomic"])
        if (
            len(reference_genomic) != 1
            or len(alternate_genomic) != 1
            or reference_genomic not in NUCLEOTIDES
            or alternate_genomic not in NUCLEOTIDES
        ):
            failure_reasons["not_single_nucleotide_substitution"] += 1
            continue
        sequence = sequences[precursor_id]
        genomic_index = int(row["genomic_position_1based"]) - int(mapping["start"])
        if not 0 <= genomic_index < len(sequence):
            failure_reasons["outside_inferred_interval"] += 1
            continue
        strand = str(row["strand"])
        transcript_index = genomic_index if strand == "+" else len(sequence) - 1 - genomic_index
        score_position = transcript_index + 1
        reference = orient_allele(reference_genomic, strand)
        alternate = orient_allele(alternate_genomic, strand)
        if sequence[transcript_index] != reference:
            failure_reasons["reference_sequence_mismatch"] += 1
            continue
        old_reference = old_scores.get((precursor_id, score_position, reference))
        old_alternate = old_scores.get((precursor_id, score_position, alternate))
        new_reference = new_scores.get((precursor_id, score_position, reference))
        new_alternate = new_scores.get((precursor_id, score_position, alternate))
        if None in {old_reference, old_alternate, new_reference, new_alternate}:
            failure_reasons["missing_score_state"] += 1
            continue
        joined.append(
            {
                **row,
                "score_position_1based": score_position,
                "reference_transcript": reference,
                "alternate_transcript": alternate,
                "old_delta": old_reference - old_alternate,
                "new_delta": new_reference - new_alternate,
                "old_absolute_delta": abs(old_reference - old_alternate),
                "new_absolute_delta": abs(new_reference - new_alternate),
            }
        )

    mapping_rows = list(mappings.values())
    mapping_summary = {
        "evaluation_precursor_ids": len(variants_by_id),
        "ids_with_sequence_and_both_score_versions": len(mappings),
        "ids_with_100pct_reference_match": sum(
            row["reference_match_fraction"] == 1 for row in mapping_rows
        ),
        "ids_with_at_least_95pct_reference_match": sum(
            row["reference_match_fraction"] >= 0.95 for row in mapping_rows
        ),
        "minimum_reference_match_fraction": min(
            row["reference_match_fraction"] for row in mapping_rows
        ),
        "ambiguous_best_match_ids": sum(
            row["best_matches"] == row["second_matches"] for row in mapping_rows
        ),
    }

    joined_by_source = Counter(str(row["source"]) for row in joined)
    gnomad_joined = [
        row
        for row in joined
        if row["source"] == "gnomad" and row["allele_frequency"] is not None
    ]
    allele_frequency = [float(row["allele_frequency"]) for row in gnomad_joined]
    rarity = [-math.log10(value) for value in allele_frequency if value > 0]
    positive_af_rows = [
        row for row in gnomad_joined if float(row["allele_frequency"]) > 0
    ]
    old_delta = [float(row["old_delta"]) for row in positive_af_rows]
    new_delta = [float(row["new_delta"]) for row in positive_af_rows]
    old_absolute_delta = [float(row["old_absolute_delta"]) for row in positive_af_rows]
    new_absolute_delta = [float(row["new_absolute_delta"]) for row in positive_af_rows]

    thresholds = {}
    for threshold in (1e-5, 1e-4, 1e-3, 1e-2):
        rare = [
            row
            for row in positive_af_rows
            if float(row["allele_frequency"]) <= threshold
        ]
        common = [
            row
            for row in positive_af_rows
            if float(row["allele_frequency"]) > threshold
        ]
        thresholds[str(threshold)] = {
            "rare_n": len(rare),
            "common_n": len(common),
            "old_median_delta_rare": median(float(row["old_delta"]) for row in rare),
            "old_median_delta_common": median(float(row["old_delta"]) for row in common),
            "new_median_delta_rare": median(float(row["new_delta"]) for row in rare),
            "new_median_delta_common": median(float(row["new_delta"]) for row in common),
        }

    clinvar_joined = [row for row in joined if row["source"] == "clinvar"]
    clinvar_by_significance = Counter(
        str(row["clinical_significance"]) for row in clinvar_joined
    )
    clinvar_values = defaultdict(list)
    for row in clinvar_joined:
        clinvar_values[str(row["clinical_significance"])].append(
            {
                "precursor_id": row["precursor_id"],
                "chromosome": row["chromosome"],
                "position": row["genomic_position_1based"],
                "reference": row["reference_genomic"],
                "alternate": row["alternate_genomic"],
                "old_delta": row["old_delta"],
                "new_delta": row["new_delta"],
            }
        )

    result = {
        "input_counts": {"clinvar": len(clinvar), "gnomad": len(gnomad)},
        "mapping": mapping_summary,
        "join": {
            "joined_by_source": dict(joined_by_source),
            "failure_reasons": dict(failure_reasons),
        },
        "gnomad": {
            "allele_frequency": summarize_values(allele_frequency),
            "old_vs_new_delta_spearman": spearman(old_delta, new_delta),
            "old_delta_vs_rarity_spearman": spearman(old_delta, rarity),
            "new_delta_vs_rarity_spearman": spearman(new_delta, rarity),
            "old_absolute_delta_vs_rarity_spearman": spearman(
                old_absolute_delta, rarity
            ),
            "new_absolute_delta_vs_rarity_spearman": spearman(
                new_absolute_delta, rarity
            ),
            "old_delta_vs_log_rarity_pearson": pearson(old_delta, rarity),
            "new_delta_vs_log_rarity_pearson": pearson(new_delta, rarity),
            "precursor_cluster_bootstrap": cluster_bootstrap_correlations(
                positive_af_rows
            ),
            "threshold_summaries": thresholds,
        },
        "clinvar": {
            "significance_counts": dict(clinvar_by_significance),
            "variants": dict(clinvar_values),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
