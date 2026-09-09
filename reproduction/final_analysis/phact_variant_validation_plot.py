#!/usr/bin/env python3
"""Plot old and consensus-tree CountNodes_3 scores against variant evidence."""

from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workspace import WORKSPACE

import csv
import math
import runpy
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


AUDIT_SCRIPT = Path(__file__).with_name("phact_variant_groundtruth_audit.py")
OUTPUT_DIR = WORKSPACE / "analyses/final"
OUTPUT_STEM = "phact_old_vs_consensustree_gnomad_clinvar"
OLD_COLOR = "#4E79A7"
NEW_COLOR = "#E15759"


def collect_joined_variants(audit: dict[str, object]) -> list[dict[str, object]]:
    sequences = audit["read_sequences"]()
    old_scores = audit["read_scores"](audit["OLD_SCORE_TABLE"])
    new_scores = audit["read_scores"](audit["NEW_SCORE_TABLE"])
    score_ids = {key[0] for key in old_scores} & {key[0] for key in new_scores}

    evaluation_root = audit["EVALUATION_ROOT"]
    variants = audit["read_variants"](
        evaluation_root / "clinvar_intersect_mirna_pri_final.tsv", "clinvar"
    )
    variants.extend(
        audit["read_variants"](
            evaluation_root / "gnomad_intersect_mirna_pri_final.tsv", "gnomad"
        )
    )

    variants_by_id: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in variants:
        variants_by_id[str(row["precursor_id"])].append(row)
    mappings = {
        precursor_id: audit["infer_genomic_start"](
            precursor_id, precursor_variants, sequences[precursor_id]
        )
        for precursor_id, precursor_variants in variants_by_id.items()
        if precursor_id in sequences and precursor_id in score_ids
    }
    if len(mappings) != 330 or any(
        mapping["reference_match_fraction"] != 1 for mapping in mappings.values()
    ):
        raise ValueError("Expected 330 unambiguous, exact precursor mappings")

    joined = []
    seen = set()
    for row in variants:
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
            continue
        seen.add(deduplication_key)
        if precursor_id not in mappings:
            continue
        reference_genomic = str(row["reference_genomic"])
        alternate_genomic = str(row["alternate_genomic"])
        if (
            len(reference_genomic) != 1
            or len(alternate_genomic) != 1
            or reference_genomic not in audit["NUCLEOTIDES"]
            or alternate_genomic not in audit["NUCLEOTIDES"]
        ):
            continue

        sequence = sequences[precursor_id]
        mapping = mappings[precursor_id]
        genomic_index = int(row["genomic_position_1based"]) - int(mapping["start"])
        if not 0 <= genomic_index < len(sequence):
            continue
        strand = str(row["strand"])
        transcript_index = (
            genomic_index if strand == "+" else len(sequence) - 1 - genomic_index
        )
        reference = audit["orient_allele"](reference_genomic, strand)
        alternate = audit["orient_allele"](alternate_genomic, strand)
        if sequence[transcript_index] != reference:
            continue
        score_position = transcript_index + 1
        score_key_reference = (precursor_id, score_position, reference)
        score_key_alternate = (precursor_id, score_position, alternate)
        score_values = (
            old_scores.get(score_key_reference),
            old_scores.get(score_key_alternate),
            new_scores.get(score_key_reference),
            new_scores.get(score_key_alternate),
        )
        if any(value is None for value in score_values):
            continue
        old_reference, old_alternate, new_reference, new_alternate = score_values
        joined.append(
            {
                **row,
                "score_position_1based": score_position,
                "reference_transcript": reference,
                "alternate_transcript": alternate,
                "old_delta": old_reference - old_alternate,
                "new_delta": new_reference - new_alternate,
            }
        )

    source_counts = {
        source: sum(row["source"] == source for row in joined)
        for source in ("clinvar", "gnomad")
    }
    if source_counts != {"clinvar": 17, "gnomad": 7251}:
        raise ValueError(f"Unexpected joined variant counts: {source_counts}")
    return joined


def write_joined_table(rows: list[dict[str, object]]) -> None:
    fields = [
        "source",
        "chromosome",
        "genomic_position_1based",
        "precursor_id",
        "strand",
        "reference_genomic",
        "alternate_genomic",
        "reference_transcript",
        "alternate_transcript",
        "score_position_1based",
        "clinical_significance",
        "allele_frequency",
        "old_delta",
        "new_delta",
    ]
    with (OUTPUT_DIR / f"{OUTPUT_STEM}_plot_data.tsv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                str(item["source"]),
                str(item["chromosome"]),
                int(item["genomic_position_1based"]),
                str(item["precursor_id"]),
            ),
        ):
            writer.writerow({field: row[field] for field in fields})


def gnomad_bin_summaries(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    bins = [
        ("AF <= 1e-5", lambda value: value <= 1e-5),
        ("1e-5 < AF <= 1e-4", lambda value: 1e-5 < value <= 1e-4),
        ("1e-4 < AF <= 1e-3", lambda value: 1e-4 < value <= 1e-3),
        ("1e-3 < AF <= 1e-2", lambda value: 1e-3 < value <= 1e-2),
        ("AF > 1e-2", lambda value: value > 1e-2),
    ]
    summaries = []
    for label, contains in bins:
        bin_rows = [
            row for row in rows if contains(float(row["allele_frequency"]))
        ]
        summary: dict[str, object] = {"frequency_bin": label, "n": len(bin_rows)}
        for version in ("old", "new"):
            values = np.asarray(
                [float(row[f"{version}_delta"]) for row in bin_rows], dtype=float
            )
            lower, center, upper = np.quantile(values, [0.25, 0.5, 0.75])
            summary[f"{version}_q1"] = lower
            summary[f"{version}_median"] = center
            summary[f"{version}_q3"] = upper
        summaries.append(summary)
    return summaries


def write_bin_table(summaries: list[dict[str, object]]) -> None:
    fields = [
        "frequency_bin",
        "n",
        "old_q1",
        "old_median",
        "old_q3",
        "new_q1",
        "new_median",
        "new_q3",
    ]
    with (OUTPUT_DIR / f"{OUTPUT_STEM}_gnomad_frequency_bins.tsv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(summaries)


def plot_gnomad(
    axis: plt.Axes,
    rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
    audit: dict[str, object],
) -> None:
    positions = np.arange(len(summaries), dtype=float)
    for version, color, marker, label in (
        ("old", OLD_COLOR, "o", "Old PHACT scores"),
        ("new", NEW_COLOR, "s", "New consensus-tree scores"),
    ):
        center = np.asarray(
            [float(row[f"{version}_median"]) for row in summaries]
        )
        lower = np.asarray([float(row[f"{version}_q1"]) for row in summaries])
        upper = np.asarray([float(row[f"{version}_q3"]) for row in summaries])
        axis.errorbar(
            positions,
            center,
            yerr=np.vstack([center - lower, upper - center]),
            color=color,
            marker=marker,
            markersize=8,
            linewidth=2.3,
            elinewidth=1.6,
            capsize=4,
            capthick=1.4,
            label=label,
            zorder=3,
        )

    rarity = [-math.log10(float(row["allele_frequency"])) for row in rows]
    old_delta = [float(row["old_delta"]) for row in rows]
    new_delta = [float(row["new_delta"]) for row in rows]
    old_correlation = audit["spearman"](old_delta, rarity)
    new_correlation = audit["spearman"](new_delta, rarity)
    axis.text(
        0.98,
        0.96,
        "Spearman correlation with rarity\n"
        f"Old: $\\rho$ = {old_correlation:.3f}\n"
        f"Consensus tree: $\\rho$ = {new_correlation:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=11.5,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#CBD5E1",
            "alpha": 0.96,
        },
    )
    frequency_labels = [
        "$\\leq 10^{-5}$",
        "$10^{-5}$ to $10^{-4}$",
        "$10^{-4}$ to $10^{-3}$",
        "$10^{-3}$ to $10^{-2}$",
        "$>10^{-2}$",
    ]
    axis.set_xticks(
        positions,
        [
            f"{label}\nn={int(summary['n']):,}"
            for label, summary in zip(frequency_labels, summaries)
        ],
    )
    axis.set_title(
        "A   gnomAD population frequency",
        loc="left",
        fontsize=16,
        fontweight="bold",
        pad=30,
    )
    axis.text(
        0,
        1.01,
        "7,251 variants across 330 precursor miRNAs",
        transform=axis.transAxes,
        fontsize=11.5,
        color="#4B5563",
        va="bottom",
    )
    axis.set_xlabel("Allele-frequency bin (rare to common)", fontsize=13, labelpad=12)
    axis.set_ylabel("PHACT(REF) - PHACT(ALT)", fontsize=13)


def plot_clinvar(axis: plt.Axes, rows: list[dict[str, object]]) -> None:
    categories = [
        "Benign",
        "Benign/Likely benign",
        "Likely benign",
        "Pathogenic",
    ]
    for category_index, category in enumerate(categories):
        category_rows = sorted(
            [row for row in rows if row["clinical_significance"] == category],
            key=lambda row: (
                str(row["precursor_id"]),
                int(row["genomic_position_1based"]),
            ),
        )
        centers = (
            np.asarray([category_index], dtype=float)
            if len(category_rows) == 1
            else category_index + np.linspace(-0.22, 0.22, len(category_rows))
        )
        for center, row in zip(centers, category_rows):
            old_x = center - 0.035
            new_x = center + 0.035
            old_delta = float(row["old_delta"])
            new_delta = float(row["new_delta"])
            axis.plot(
                [old_x, new_x],
                [old_delta, new_delta],
                color="#9CA3AF",
                linewidth=1.0,
                alpha=0.8,
                zorder=1,
            )
            edgecolor = "#111827" if category == "Pathogenic" else "white"
            linewidth = 1.4 if category == "Pathogenic" else 0.7
            axis.scatter(
                old_x,
                old_delta,
                s=58,
                color=OLD_COLOR,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=3,
            )
            axis.scatter(
                new_x,
                new_delta,
                s=58,
                color=NEW_COLOR,
                marker="s",
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=3,
            )

    counts = [sum(row["clinical_significance"] == item for row in rows) for item in categories]
    axis.set_xticks(
        range(len(categories)),
        [
            f"Benign\nn={counts[0]}",
            f"Benign / likely\nbenign\nn={counts[1]}",
            f"Likely benign\nn={counts[2]}",
            f"Pathogenic\nn={counts[3]}",
        ],
    )
    pathogenic = next(row for row in rows if row["clinical_significance"] == "Pathogenic")
    axis.annotate(
        "Hsa-Mir-204-P2 G>A",
        xy=(3.035, float(pathogenic["new_delta"])),
        xytext=(2.52, 0.62),
        textcoords="data",
        fontsize=11,
        ha="center",
        arrowprops={"arrowstyle": "-", "color": "#6B7280", "linewidth": 1.0},
    )
    axis.text(
        0.02,
        0.96,
        "One pathogenic variant:\nclinical separation cannot be estimated",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#CBD5E1",
            "alpha": 1.0,
        },
    )
    axis.set_title(
        "B   ClinVar classifications",
        loc="left",
        fontsize=16,
        fontweight="bold",
        pad=30,
    )
    axis.text(
        0,
        1.01,
        "17 variants with matched old and new scores",
        transform=axis.transAxes,
        fontsize=11.5,
        color="#4B5563",
        va="bottom",
    )
    axis.set_ylabel("PHACT(REF) - PHACT(ALT)", fontsize=13)


def write_plot(
    joined: list[dict[str, object]],
    summaries: list[dict[str, object]],
    audit: dict[str, object],
) -> None:
    gnomad_rows = [
        row
        for row in joined
        if row["source"] == "gnomad"
        and row["allele_frequency"] is not None
        and float(row["allele_frequency"]) > 0
    ]
    clinvar_rows = [row for row in joined if row["source"] == "clinvar"]
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 7.6))
    plot_gnomad(axes[0], gnomad_rows, summaries, audit)
    plot_clinvar(axes[1], clinvar_rows)

    for axis in axes:
        axis.axhline(0, color="#6B7280", linewidth=1.0, zorder=0)
        axis.grid(axis="y", color="#DDE3EA", linewidth=0.8, alpha=0.9, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=11.5)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=OLD_COLOR,
            marker="o",
            linewidth=2.3,
            markersize=8,
            label="Old PHACT scores",
        ),
        Line2D(
            [0],
            [0],
            color=NEW_COLOR,
            marker="s",
            linewidth=2.3,
            markersize=8,
            label="New consensus-tree scores",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=2,
        frameon=False,
        fontsize=12,
        columnspacing=2.8,
    )
    figure.suptitle(
        "Old and consensus-tree PHACT scores versus variant evidence",
        fontsize=20,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.025,
        "Precursor-miRNA SNVs only. qntnorm-transformed CountNodes_3 contrast = PHACT(REF) - PHACT(ALT). "
        "gnomAD shows median and interquartile range per bin; population frequency is not a clinical label.",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#374151",
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.75, bottom=0.19, wspace=0.25)
    figure.savefig(OUTPUT_DIR / f"{OUTPUT_STEM}.png", dpi=260, bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / f"{OUTPUT_STEM}.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = runpy.run_path(str(AUDIT_SCRIPT), run_name="phact_variant_audit")
    joined = collect_joined_variants(audit)
    gnomad_rows = [
        row
        for row in joined
        if row["source"] == "gnomad"
        and row["allele_frequency"] is not None
        and float(row["allele_frequency"]) > 0
    ]
    summaries = gnomad_bin_summaries(gnomad_rows)
    write_joined_table(joined)
    write_bin_table(summaries)
    write_plot(joined, summaries, audit)
    print("gnomAD bins:")
    for row in summaries:
        print(
            f"  {row['frequency_bin']}: n={row['n']}, "
            f"old={row['old_median']:.4f}, new={row['new_median']:.4f}"
        )
    print(f"ClinVar variants: {sum(row['source'] == 'clinvar' for row in joined)}")


if __name__ == "__main__":
    main()
