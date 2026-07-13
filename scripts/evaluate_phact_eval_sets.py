#!/usr/bin/env python
"""Evaluate PHACT score variants on ClinVar and gnomAD eval sets."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


POSITIVE_CLINVAR = {
    "Pathogenic",
    "Likely pathogenic",
    "Pathogenic/Likely pathogenic",
}
NEGATIVE_CLINVAR = {
    "Benign",
    "Likely benign",
    "Benign/Likely benign",
}
NUCLEOTIDES = {"A", "C", "G", "T"}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    records.extend(evaluate_ortholog_scores(args))
    records.extend(evaluate_target_scores(args))

    output_path = args.output_dir / "phact_eval_model_comparison.tsv"
    pd.DataFrame(records).to_csv(output_path, sep="\t", index=False)
    print(f"Wrote {output_path}")

    best_path = args.output_dir / "phact_eval_model_best.tsv"
    best = (
        pd.DataFrame(records)
        .sort_values(["source", "eval_set", "metric", "value"], ascending=[True, True, True, False])
        .groupby(["source", "eval_set", "metric"], as_index=False)
        .head(5)
    )
    best.to_csv(best_path, sep="\t", index=False)
    print(f"Wrote {best_path}")


def evaluate_ortholog_scores(args: argparse.Namespace) -> list[dict[str, object]]:
    print("Evaluating ortholog PHACT on miRNA-pri eval sets", flush=True)
    clinvar = load_mirna_clinvar(args)
    gnomad = load_mirna_gnomad(args)
    records: list[dict[str, object]] = []

    for score_file in sorted(args.results_dir.glob("orthologs_*.tsv")):
        score_table = pd.read_csv(score_file, sep="\t")
        model_columns = score_table.columns[3:].tolist()
        score_table["Nucleotide"] = score_table["Nucleotide"].str.upper()

        clinvar_joined = clinvar.merge(
            score_table,
            left_on=["base_id", "phact_position", "alt"],
            right_on=["ID", "Position", "Nucleotide"],
            how="inner",
        )
        clinvar_ref_joined = clinvar.merge(
            score_table,
            left_on=["base_id", "phact_position", "ref"],
            right_on=["ID", "Position", "Nucleotide"],
            how="inner",
        )
        gnomad_joined = gnomad.merge(
            score_table,
            left_on=["base_id", "phact_position", "alt"],
            right_on=["ID", "Position", "Nucleotide"],
            how="inner",
        )
        gnomad_ref_joined = gnomad.merge(
            score_table,
            left_on=["base_id", "phact_position", "ref"],
            right_on=["ID", "Position", "Nucleotide"],
            how="inner",
        )
        print(
            f"  {score_file.name}: ClinVar mapped {len(clinvar_joined):,}/{len(clinvar):,}; "
            f"gnomAD mapped {len(gnomad_joined):,}/{len(gnomad):,}",
            flush=True,
        )

        for model_column in model_columns:
            model_name = clean_model_name(model_column)
            records.extend(
                clinvar_metrics(
                    clinvar_joined["label"].to_numpy(),
                    clinvar_joined[model_column].to_numpy(),
                    source="ortholog",
                    score_file=score_file.name,
                    model=model_name,
                    n_total=len(clinvar_joined),
                )
            )
            clinvar_delta = paired_delta(
                clinvar_joined,
                clinvar_ref_joined,
                model_column,
            )
            records.extend(
                clinvar_metrics(
                    clinvar_delta["label"].to_numpy(),
                    clinvar_delta["delta"].to_numpy(),
                    source="ortholog",
                    score_file=score_file.name,
                    model=f"{model_name}:ref_minus_alt",
                    n_total=len(clinvar_delta),
                    lower_is_worse=False,
                )
            )
            records.extend(
                gnomad_metrics(
                    gnomad_joined["allele_frequency"].to_numpy(),
                    gnomad_joined[model_column].to_numpy(),
                    source="ortholog",
                    score_file=score_file.name,
                    model=model_name,
                    n_total=len(gnomad_joined),
                )
            )
            gnomad_delta = paired_delta(
                gnomad_joined,
                gnomad_ref_joined,
                model_column,
            )
            records.extend(
                gnomad_metrics(
                    gnomad_delta["allele_frequency"].to_numpy(),
                    gnomad_delta["tolerance_delta"].to_numpy(),
                    source="ortholog",
                    score_file=score_file.name,
                    model=f"{model_name}:alt_minus_ref",
                    n_total=len(gnomad_delta),
                )
            )

    return records


def paired_delta(
    alt_joined: pd.DataFrame,
    ref_joined: pd.DataFrame,
    model_column: str,
) -> pd.DataFrame:
    key_columns = ["id", "genomic_position", "ref", "alt"]
    keep_columns = key_columns + ["label", "allele_frequency", model_column]
    left = alt_joined[[column for column in keep_columns if column in alt_joined]].rename(
        columns={model_column: "alt_score"}
    )
    right = ref_joined[[column for column in key_columns + [model_column] if column in ref_joined]].rename(
        columns={model_column: "ref_score"}
    )
    paired = left.merge(right, on=key_columns, how="inner")
    paired["delta"] = paired["ref_score"] - paired["alt_score"]
    paired["tolerance_delta"] = paired["alt_score"] - paired["ref_score"]
    return paired


def evaluate_target_scores(args: argparse.Namespace) -> list[dict[str, object]]:
    print("Evaluating AGO target PHACT on target eval sets", flush=True)
    con = duckdb.connect()
    records: list[dict[str, object]] = []

    for score_file in sorted(args.results_dir.glob("AGO2_eCLIP_Manakov2022_*.tsv")):
        score_name = score_file.stem.removeprefix("AGO2_eCLIP_Manakov2022_")
        print(f"  {score_file.name}: ClinVar join", flush=True)
        clinvar = con.execute(target_clinvar_query(score_file, args)).fetch_df()
        print(f"  {score_file.name}: ClinVar mapped {len(clinvar):,}", flush=True)
        records.extend(
            clinvar_metrics(
                clinvar["label"].to_numpy(),
                clinvar["alt_score"].to_numpy(),
                source="ago_target",
                score_file=score_file.name,
                model=score_name,
                n_total=len(clinvar),
            )
        )
        if "ref_score" in clinvar:
            records.extend(
                clinvar_metrics(
                    clinvar["label"].to_numpy(),
                    (clinvar["ref_score"] - clinvar["alt_score"]).to_numpy(),
                    source="ago_target",
                    score_file=score_file.name,
                    model=f"{score_name}:ref_minus_alt",
                    n_total=len(clinvar),
                    lower_is_worse=False,
                )
            )

        print(f"  {score_file.name}: gnomAD join/correlation", flush=True)
        gnomad = con.execute(target_gnomad_query(score_file, args)).fetch_df().iloc[0]
        records.append(
            {
                "source": "ago_target",
                "score_file": score_file.name,
                "model": score_name,
                "eval_set": "target_gnomad",
                "metric": "pearson_alt_score_ln_af",
                "value": float(gnomad["pearson_alt_score_ln_af"]),
                "n": int(gnomad["n"]),
                "n_positive": np.nan,
                "note": "higher is better; higher PHACT score should track higher allele frequency",
            }
        )
        records.append(
            {
                "source": "ago_target",
                "score_file": score_file.name,
                "model": f"{score_name}:alt_minus_ref",
                "eval_set": "target_gnomad",
                "metric": "pearson_tolerance_delta_ln_af",
                "value": float(gnomad["pearson_tolerance_delta_ln_af"]),
                "n": int(gnomad["n"]),
                "n_positive": np.nan,
                "note": "higher is better; alt_score-ref_score should track higher allele frequency",
            }
        )

    return records


def load_mirna_clinvar(args: argparse.Namespace) -> pd.DataFrame:
    path = args.eval_dir / "clinvar_intersect_mirna_pri_final.tsv"
    df = pd.read_csv(path, sep="\t")
    df = df.rename(
        columns={
            "PositionVCF": "genomic_position",
            "ReferenceAlleleVCF": "ref",
            "AlternateAlleleVCF": "alt",
        }
    )
    df["label"] = df["ClinicalSignificance"].map(clinvar_label)
    df = df[df["label"].notna()].copy()
    return add_mirna_phact_positions(df, args)


def load_mirna_gnomad(args: argparse.Namespace) -> pd.DataFrame:
    path = args.eval_dir / "gnomad_intersect_mirna_pri_final.tsv"
    df = pd.read_csv(path, sep="\t")
    df = df.rename(
        columns={
            "Position": "genomic_position",
            "REF": "ref",
            "ALT": "alt",
        }
    )
    df = df[df["allele_frequency"] > 0].copy()
    return add_mirna_phact_positions(df, args)


def add_mirna_phact_positions(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    coords = pd.read_csv(args.mirna_positions, sep="\t")
    coords["base_id"] = coords["ID"].str.replace(r"_pre$", "", regex=True)
    df["base_id"] = df["id"].str.replace(r"_pri$", "", regex=True)
    df["alt"] = df["alt"].str.upper()
    df["ref"] = df["ref"].str.upper()
    df = df[df["alt"].isin(NUCLEOTIDES) & df["ref"].isin(NUCLEOTIDES)].copy()
    merged = df.merge(
        coords[["base_id", "Strand", "Start_Flanking", "End_Flanking"]],
        on="base_id",
        how="inner",
    )
    plus = merged["Strand"] == "+"
    merged["phact_position"] = np.where(
        plus,
        merged["genomic_position"] - merged["Start_Flanking"] + 1,
        merged["End_Flanking"] - merged["genomic_position"] + 1,
    ).astype(int)
    return merged[merged["phact_position"] > 0].copy()


def clinvar_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    source: str,
    score_file: str,
    model: str,
    n_total: int,
    lower_is_worse: bool = True,
) -> list[dict[str, object]]:
    keep = np.isfinite(scores) & np.isfinite(labels)
    labels = labels[keep].astype(int)
    scores = scores[keep].astype(float)
    if lower_is_worse:
        predictor = -scores
        note = "pathogenic predicted by lower ALT score"
    else:
        predictor = scores
        note = "pathogenic predicted by larger score"

    rows: list[dict[str, object]] = []
    n_positive = int(labels.sum()) if len(labels) else 0
    for metric_name, metric_fn in [
        ("roc_auc_pathogenic", roc_auc_score),
        ("auprc_pathogenic", average_precision_score),
    ]:
        value = np.nan
        if len(np.unique(labels)) == 2:
            value = float(metric_fn(labels, predictor))
        rows.append(
            {
                "source": source,
                "score_file": score_file,
                "model": model,
                "eval_set": "clinvar",
                "metric": metric_name,
                "value": value,
                "n": int(len(labels)),
                "n_positive": n_positive,
                "note": note,
            }
        )
    return rows


def gnomad_metrics(
    allele_frequency: np.ndarray,
    scores: np.ndarray,
    *,
    source: str,
    score_file: str,
    model: str,
    n_total: int,
) -> list[dict[str, object]]:
    keep = np.isfinite(scores) & np.isfinite(allele_frequency) & (allele_frequency > 0)
    af = allele_frequency[keep].astype(float)
    score = scores[keep].astype(float)
    log_af = np.log(af)
    return [
        {
            "source": source,
            "score_file": score_file,
            "model": model,
            "eval_set": "gnomad",
            "metric": "pearson_alt_score_ln_af",
            "value": float(pd.Series(score).corr(pd.Series(log_af), method="pearson")),
            "n": int(len(score)),
            "n_positive": np.nan,
            "note": "higher is better; higher PHACT score should track higher allele frequency",
        },
        {
            "source": source,
            "score_file": score_file,
            "model": model,
            "eval_set": "gnomad",
            "metric": "spearman_alt_score_af",
            "value": float(pd.Series(score).corr(pd.Series(af), method="spearman")),
            "n": int(len(score)),
            "n_positive": np.nan,
            "note": "higher is better; higher PHACT score should track higher allele frequency",
        },
    ]


def target_clinvar_query(score_file: Path, args: argparse.Namespace) -> str:
    return f"""
WITH eval AS (
    SELECT
        Chromosome,
        strand AS Strand,
        PositionVCF::BIGINT AS Position,
        upper(ReferenceAlleleVCF) AS REF,
        upper(AlternateAlleleVCF) AS ALT,
        CASE
            WHEN ClinicalSignificance IN ('Pathogenic', 'Likely pathogenic', 'Pathogenic/Likely pathogenic') THEN 1
            WHEN ClinicalSignificance IN ('Benign', 'Likely benign', 'Benign/Likely benign') THEN 0
            ELSE NULL
        END AS label
    FROM read_csv('{args.eval_dir / "clinvar_intersect_ago2_targets_final.tsv"}',
        delim='\\t', header=true,
        columns={{
            'Chromosome': 'VARCHAR',
            'strand': 'VARCHAR',
            'ClinicalSignificance': 'VARCHAR',
            'PositionVCF': 'BIGINT',
            'ReferenceAlleleVCF': 'VARCHAR',
            'AlternateAlleleVCF': 'VARCHAR'
        }})
    WHERE label IS NOT NULL
      AND upper(ReferenceAlleleVCF) IN ('A', 'C', 'G', 'T')
      AND upper(AlternateAlleleVCF) IN ('A', 'C', 'G', 'T')
),
joined AS (
    SELECT
        e.Chromosome,
        e.Strand,
        e.Position,
        e.REF,
        e.ALT,
        e.label,
        max(CASE e.ALT WHEN 'A' THEN s.A WHEN 'C' THEN s.C WHEN 'G' THEN s.G WHEN 'T' THEN s.T END) AS alt_score,
        max(CASE e.REF WHEN 'A' THEN s.A WHEN 'C' THEN s.C WHEN 'G' THEN s.G WHEN 'T' THEN s.T END) AS ref_score
    FROM eval e
    JOIN read_csv('{score_file}',
        delim='\\t', header=true,
        columns={{
            'Chromosome': 'VARCHAR',
            'Position': 'BIGINT',
            'Strand': 'VARCHAR',
            'inputBlock': 'VARCHAR',
            'A': 'DOUBLE',
            'T': 'DOUBLE',
            'G': 'DOUBLE',
            'C': 'DOUBLE'
        }}) s
    ON e.Chromosome = s.Chromosome
       AND e.Position = s.Position
       AND e.Strand = s.Strand
    GROUP BY e.Chromosome, e.Strand, e.Position, e.REF, e.ALT, e.label
)
SELECT label, alt_score, ref_score
FROM joined
WHERE alt_score IS NOT NULL
"""


def target_gnomad_query(score_file: Path, args: argparse.Namespace) -> str:
    return f"""
WITH eval AS (
    SELECT
        Chromosome,
        strand AS Strand,
        Position::BIGINT AS Position,
        upper(REF) AS REF,
        upper(ALT) AS ALT,
        allele_frequency::DOUBLE AS allele_frequency
    FROM read_csv('{args.eval_dir / "gnomad_intersect_ago2_targets_final.tsv"}',
        delim='\\t', header=true,
        columns={{
            'Chromosome': 'VARCHAR',
            'strand': 'VARCHAR',
            'Position': 'BIGINT',
            'allele_frequency': 'DOUBLE',
            'allele_number': 'BIGINT',
            'REF': 'VARCHAR',
            'ALT': 'VARCHAR'
        }})
    WHERE allele_frequency > 0
      AND upper(REF) IN ('A', 'C', 'G', 'T')
      AND upper(ALT) IN ('A', 'C', 'G', 'T')
),
joined AS (
    SELECT
        e.Chromosome,
        e.Strand,
        e.Position,
        e.REF,
        e.ALT,
        e.allele_frequency,
        max(CASE e.ALT WHEN 'A' THEN s.A WHEN 'C' THEN s.C WHEN 'G' THEN s.G WHEN 'T' THEN s.T END) AS alt_score,
        max(CASE e.REF WHEN 'A' THEN s.A WHEN 'C' THEN s.C WHEN 'G' THEN s.G WHEN 'T' THEN s.T END) AS ref_score
    FROM eval e
    JOIN read_csv('{score_file}',
        delim='\\t', header=true,
        columns={{
            'Chromosome': 'VARCHAR',
            'Position': 'BIGINT',
            'Strand': 'VARCHAR',
            'inputBlock': 'VARCHAR',
            'A': 'DOUBLE',
            'T': 'DOUBLE',
            'G': 'DOUBLE',
            'C': 'DOUBLE'
        }}) s
    ON e.Chromosome = s.Chromosome
       AND e.Position = s.Position
       AND e.Strand = s.Strand
    GROUP BY e.Chromosome, e.Strand, e.Position, e.REF, e.ALT, e.allele_frequency
)
SELECT
    count(*) AS n,
    corr(alt_score, ln(allele_frequency)) AS pearson_alt_score_ln_af,
    corr(alt_score - ref_score, ln(allele_frequency)) AS pearson_tolerance_delta_ln_af
FROM joined
WHERE alt_score IS NOT NULL AND ref_score IS NOT NULL
"""


def clinvar_label(value: str) -> int | float:
    if value in POSITIVE_CLINVAR:
        return 1
    if value in NEGATIVE_CLINVAR:
        return 0
    return np.nan


def clean_model_name(column: str) -> str:
    for prefix in (
        "PHACTn_gapAware_BranchNorm_wl_param_",
        "PHACTn_gapAware_wtNTnorm_wl_param_",
    ):
        if column.startswith(prefix):
            return column.removeprefix(prefix)
    return column


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(
            "/home/dtzim01/drive-download-19Ntprvu-qbI1k4ZQphZ4QnuFoXNIgK2E/"
            "extracted/results_0226"
        ),
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path(
            "/home/dtzim01/drive-download-19Ntprvu-qbI1k4ZQphZ4QnuFoXNIgK2E/"
            "extracted/evaluation_set"
        ),
    )
    parser.add_argument(
        "--mirna-positions",
        type=Path,
        default=Path(
            "/home/dtzim01/PHACT-miRBind/PHACTn/workflow_WGA/scripts/"
            "hsa_pre_miRNA_positions.tsv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/dtzim01/PHACT-miRBind/reports/phact_eval_sets"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
