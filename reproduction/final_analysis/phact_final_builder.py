#!/usr/bin/env python3
"""Build the final PHACT model-score table and focused miRNA profile plots."""

from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workspace import WORKSPACE

import csv
import gzip
import json
import math
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SCRATCH = WORKSPACE
EXPERIMENTS = SCRATCH / "runs/experiments"
AGENTOMICS = SCRATCH / "archive/agentomics-history"
OUTPUT_DIR = SCRATCH / "analyses/final"
OLD_COUNTNODES3_TABLE = (
    SCRATCH
    / "data/inputs/score_import_20260813/old/PHACT_scores_ALL_0226/results_0226/orthologs_qntnorm_transformed.tsv.gz"
)
CONSENSUS_TREE_COUNTNODES3_TABLE = (
    SCRATCH
    / "data/inputs/score_import_20260813/new/PHACT_scores_orthologs_with_consensusTree/results_qntnorm_transformed.tsv.gz"
)
COUNTNODES3_COLUMN = "PHACTn_gapAware_wtNTnorm_wl_param_CountNodes_3"


@dataclass(frozen=True)
class ScoreRow:
    category_order: int
    category: str
    model: str
    features_or_variant: str
    test_auprc: float
    leftout_auprc: float
    source_artifact: str
    notes: str = ""
    run_id: str = ""
    legacy_category: str = ""
    legacy_model_name: str = ""
    model_explanation: str = ""
    presentation_status: str = "primary"


@dataclass(frozen=True)
class ModelPresentation:
    run_id: str
    display_name: str
    explanation: str
    status: str = "primary"


CATEGORY_DISPLAY_NAMES = {
    "Original sequence CNN": "Sequence-only baselines",
    "Conservation CNN": "Conservation-augmented CNNs",
    "Original PHACT CNN": "Original PHACT-CNN variants",
    "Experimental PHACT CNN": "PHACT-CNN extensions",
    "RiNALMo fusion": "PHACT–RiNALMo fusion",
    "Auxiliary feature model": "Feature baselines and representation probes",
    "Ensemble or stacker": "Prediction-level combinations",
    "Candidate-row ablation": "Data-curation sensitivity checks",
    "Agentomics model": "Agentomics architecture-search finalists",
}


MODEL_PRESENTATION = {
    "PairwiseSeqCNN": ModelPresentation(
        "seq_only",
        "Sequence CNN (PairwiseSeqCNN)",
        "The repository sequence baseline operating on the 28×50 nucleotide-pair grid without evolutionary channels.",
    ),
    "Official miRBench Manakov CNN": ModelPresentation(
        "official_mirbench_manakov_cnn",
        "Official miRBench CNN",
        "The official sequence-only Manakov CNN evaluated through the local data reader.",
    ),
    "PairwiseConservationCNN — phyloP": ModelPresentation(
        "conservation_phylop",
        "Sequence CNN + phyloP",
        "The pairwise sequence CNN augmented with target-position phyloP scores.",
    ),
    "PairwiseConservationCNN — phastCons": ModelPresentation(
        "conservation_phastcons",
        "Sequence CNN + phastCons",
        "The pairwise sequence CNN augmented with target-position phastCons scores.",
    ),
    "PairwiseConservationCNN — phyloP + phastCons": ModelPresentation(
        "conservation_both",
        "Sequence CNN + phyloP + phastCons",
        "The pairwise sequence CNN augmented with both target conservation tracks.",
    ),
    "PHACT CNN — CountNodes_3 miRNA": ModelPresentation(
        "phact_CountNodes_3_mirna",
        "PHACT-CNN — miRNA CountNodes_3",
        "The sequence CNN augmented with four-state CountNodes_3 scores along the miRNA axis.",
    ),
    "PHACT CNN — target": ModelPresentation(
        "phact_CountNodes_3_target",
        "PHACT-CNN — target PHACT",
        "The sequence CNN augmented with four-state PHACT scores along the target axis.",
    ),
    "PHACT CNN — CountNodes_3 both axes": ModelPresentation(
        "phact_CountNodes_3_both",
        "PHACT-CNN — miRNA + target PHACT",
        "The sequence CNN augmented with miRNA CountNodes_3 and target PHACT channels.",
    ),
    "PHACT CNN — actual-margin reduction": ModelPresentation(
        "phact_CountNodes_3_both_actual_margin",
        "PHACT-CNN — actual-nucleotide margin",
        "Both PHACT axes are reduced to the score margin for the observed nucleotide.",
    ),
    "PHACT CNN — alternative-mean reduction": ModelPresentation(
        "phact_CountNodes_3_both_alt_mean",
        "PHACT-CNN — alternative-nucleotide mean",
        "Both PHACT axes are reduced using the mean score across alternative nucleotides.",
    ),
    "PHACT CNN — all 16 miRNA models + target": ModelPresentation(
        "phact_all16_mirna_target",
        "PHACT-CNN — 16 miRNA tracks + target PHACT",
        "The sequence CNN receives all 16 miRNA PHACT model channels together with target PHACT.",
    ),
    "Clean-Manakov control": ModelPresentation(
        "phact_cnn_clean_manakov_control_lr5e5",
        "PHACT-CNN — conflict-filtered control",
        "A CountNodes_3 PHACT control trained after replacing 28 identified conflicting training rows.",
    ),
    "Balanced-50k GSE augmentation": ModelPresentation(
        "phact_cnn_augmented_balanced50k_lr5e5",
        "PHACT-CNN — balanced GSE-50k augmentation",
        "The CountNodes_3 model is augmented with a leakage-filtered, class-balanced 50,000-row GSE subset.",
    ),
    "PHACT + conservation clean Manakov": ModelPresentation(
        "phact_cnn_conservation_clean_manakov_lr1e4",
        "PHACT-CNN — target conservation",
        "The CountNodes_3 model adds target phyloP and phastCons tracks without external GSE augmentation.",
    ),
    "PHACT + conservation + conservative GSE": ModelPresentation(
        "phact_cnn_conservation_augmented_full_lr5e5",
        "PHACT-CNN — conservation + filtered GSE",
        "The conservation-augmented PHACT-CNN is trained with a leakage-filtered GSE augmentation set.",
    ),
    "PHACT + conservation + 25k GSE positives": ModelPresentation(
        "phact_cnn_conservation_gse_positive25k_lr5e5",
        "PHACT-CNN — conservation + GSE-positive-25k",
        "The conservation-augmented model adds 25,000 positive GSE training examples.",
    ),
    "PHACT + conservation + 25k positives + smoothing": ModelPresentation(
        "phact_cnn_conservation_gse_positive25k_negsmooth002_lr5e5",
        "PHACT-CNN — GSE-positive-25k + label smoothing",
        "The 25,000-positive GSE augmentation uses 0.02 smoothing on negative training labels.",
    ),
    "PHACT + conservation + 100k GSE positives": ModelPresentation(
        "phact_cnn_conservation_gse_positive100k_lr5e5",
        "PHACT-CNN — conservation + GSE-positive-100k",
        "The conservation-augmented model adds 100,000 positive GSE training examples.",
    ),
    "PHACT + conservation + target-shift training": ModelPresentation(
        "phact_cnn_conservation_conflict28_shift1_lr5e5",
        "PHACT-CNN — target-shift augmentation",
        "The conflict-filtered conservation model is trained with random target shifts of at most one position.",
    ),
    "PHACT + conservation + shift + focal loss": ModelPresentation(
        "phact_cnn_conservation_conflict28_shift1_focal1_lr5e5",
        "PHACT-CNN — target shift + focal loss",
        "The target-shift training variant replaces binary cross-entropy with focal loss at gamma 1.",
    ),
    "Full-Manakov final refit": ModelPresentation(
        "phact_cnn_full_manakov_conflict28_shift1_finalrefit",
        "PHACT-CNN — train+validation refit",
        "The target-shift model is refit on the combined training and validation rows after conflict replacement.",
    ),
    "Widened PHACT CNN": ModelPresentation(
        "phact_cnn_wide_clean_manakov_lr1e4",
        "PHACT-CNN — widened backbone",
        "The PHACT-CNN convolution widths are increased to 256, 128, and 64 filters.",
    ),
    "PHACT outer-product interactions": ModelPresentation(
        "phact_outer_clean_manakov_lr1e4",
        "PHACT-CNN — cross-axis PHACT interactions",
        "The model appends explicit miRNA-by-target PHACT interaction channels.",
    ),
    "Conservative GSE fine-tune": ModelPresentation(
        "phact_cnn_augmented_conservative_finetune",
        "PHACT-CNN — filtered-GSE fine-tuning",
        "The original CountNodes_3 checkpoint is fine-tuned on a leakage-filtered GSE dataset.",
    ),
    "RiNALMo cross-encoding residual": ModelPresentation(
        "rinalmo_cross_residual_augmented_conservative_frozen",
        "PHACT–RiNALMo — frozen residual fusion",
        "Frozen RiNALMo and miRBind/PHACT representations are combined through a trained residual head.",
    ),
    "Fully fine-tuned RiNALMo-PHACT": ModelPresentation(
        "fully_finetuned_rinalmo_phact",
        "PHACT–RiNALMo — fine-tuned fusion",
        "A RiNALMo-micro representation is integrated by PHACT-conditioned pooling while the miRBind branch remains frozen.",
    ),
    "Seed/region metadata LightGBM": ModelPresentation(
        "seed_metadata_no_leakage",
        "Seed-and-context feature baseline",
        "LightGBM uses seed-match counts, GC content, miRNA length, and target-region metadata.",
    ),
    "RNAduplex LightGBM": ModelPresentation(
        "rnaduplex",
        "RNAduplex feature baseline",
        "LightGBM uses ViennaRNA duplex energy and alignment-derived features.",
    ),
    "PHACT latent LightGBM": ModelPresentation(
        "phact_latent",
        "PHACT latent-space probe",
        "LightGBM tests the predictive content of frozen 30-dimensional PHACT-CNN latent vectors.",
    ),
    "Target-shift prediction ensemble": ModelPresentation(
        "phact_shift_ensemble",
        "Target-shift test-time ensemble",
        "Predictions from target shifts −1, 0, and +1 are blended using validation-selected weights.",
    ),
    "Shift-trained prediction ensemble": ModelPresentation(
        "phact_shift_trained_ensemble",
        "Shift-trained test-time ensemble",
        "A target-shift-trained PHACT-CNN is blended across shifted inference views.",
    ),
    "Focal/shift prediction ensemble": ModelPresentation(
        "phact_focal_shift_ensemble",
        "Focal-loss shift ensemble",
        "The focal-loss PHACT-CNN is blended across shifted target views at inference.",
    ),
    "Final-refit shift ensemble": ModelPresentation(
        "phact_finalrefit_shift_ensemble",
        "Refit-and-shift ensemble",
        "The train+validation refit is blended with its shifted-target predictions.",
    ),
    "Validation-selected heterogeneous ensemble": ModelPresentation(
        "validation_selected_ensemble",
        "Heterogeneous validation-selected blend",
        "Validation-selected weights combine shift PHACT, seed/context, miRNA-only PHACT, and RNAduplex predictions.",
    ),
    "Prediction stacker": ModelPresentation(
        "prediction_stacker",
        "14-family LightGBM stacker",
        "LightGBM combines ranked and logit-transformed predictions from 14 model families.",
    ),
    "Test-derived candidate baseline": ModelPresentation(
        "candidate_ablation_test_baseline",
        "Reviewed-test rows retained",
        "A one-epoch diagnostic retains all 22 reviewed test-derived candidate rows at full weight.",
    ),
    "Test-derived candidates downweighted": ModelPresentation(
        "candidate_ablation_test_downweight025",
        "Reviewed-test rows downweighted",
        "A one-epoch diagnostic assigns the 22 reviewed test-derived candidate rows a weight of 0.25.",
    ),
    "Test-derived candidates removed": ModelPresentation(
        "candidate_ablation_test_remove",
        "Reviewed-test rows removed",
        "A one-epoch diagnostic removes the 22 reviewed test-derived candidate rows from training.",
    ),
    "Validation-derived candidates downweighted": ModelPresentation(
        "candidate_ablation_validation_downweight025",
        "Reviewed-validation rows downweighted",
        "A one-epoch diagnostic assigns 53 reviewed validation-derived candidate rows a weight of 0.25.",
    ),
    "Validation-derived candidates removed": ModelPresentation(
        "candidate_ablation_validation_remove",
        "Reviewed-validation rows removed",
        "A one-epoch diagnostic removes 53 reviewed validation-derived candidate rows from training.",
    ),
    "Agentomics iteration 13": ModelPresentation(
        "agentomics_iteration_13",
        "Agentomics — frozen RiNALMo-micro fusion",
        "A multimodal sequence, conservation, PHACT, and metadata classifier with frozen final-layer RiNALMo-micro embeddings.",
    ),
    "Agentomics iteration 15": ModelPresentation(
        "agentomics_iteration_15",
        "Agentomics — frozen RiNALMo-mega fusion",
        "The multimodal fusion model uses frozen mean- and max-pooled RiNALMo-mega embeddings.",
    ),
    "Agentomics iteration 30": ModelPresentation(
        "agentomics_iteration_30",
        "Agentomics — partial RiNALMo fine-tuning",
        "The iteration-13 fusion model fine-tunes RiNALMo-micro blocks 10–11 together with its PHACT and fusion modules.",
    ),
    "Agentomics iteration 31": ModelPresentation(
        "agentomics_iteration_31",
        "Agentomics — RiNALMo fine-tuning + miRNA-group ranking",
        "The partially fine-tuned RiNALMo model adds paired within-miRNA sampling and a group-ranking objective.",
    ),
    "Agentomics iteration 32": ModelPresentation(
        "agentomics_iteration_32",
        "Agentomics — alignment-token transformer",
        "A Transformer encodes 77 shift-wise PHACT alignment tokens and adds a residual to the iteration-13 fusion model.",
    ),
    "Agentomics iteration 33": ModelPresentation(
        "agentomics_iteration_33",
        "Agentomics — PHACT-gated RiNALMo layer mixing",
        "A PHACT-conditioned gate mixes RiNALMo-micro blocks 3, 6, 9, and 12 before residual fusion.",
    ),
    "Agentomics iteration 34": ModelPresentation(
        "agentomics_iteration_34",
        "Agentomics — evolutionary FiLM conditioning",
        "A local 63-channel evolutionary tensor modulates the sequence-pair CNN through feature-wise affine conditioning.",
    ),
    "Agentomics iteration 36": ModelPresentation(
        "agentomics_iteration_36",
        "Agentomics — multi-candidate PHACT fusion",
        "An ordered multi-candidate miRNA-PHACT representation adds a trained residual to the fusion classifier.",
    ),
    "Agentomics iteration 37": ModelPresentation(
        "agentomics_iteration_37",
        "Agentomics — soft-alignment trial",
        "A differentiable local affine-gap alignment branch was trained, but checkpoint selection retained the unchanged inherited model.",
        "diagnostic",
    ),
    "Agentomics iteration 38": ModelPresentation(
        "agentomics_iteration_38",
        "Agentomics — orthologue-adapter trial",
        "The orthologue adapter remained exactly zero, making this artifact prediction-equivalent to iteration 36.",
        "duplicate",
    ),
}


def apply_presentation_metadata(rows: list[ScoreRow]) -> list[ScoreRow]:
    raw_names = {row.model for row in rows}
    missing = raw_names - MODEL_PRESENTATION.keys()
    extra = MODEL_PRESENTATION.keys() - raw_names
    if missing or extra:
        raise ValueError(
            f"Presentation metadata mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )

    presented = []
    for row in rows:
        metadata = MODEL_PRESENTATION[row.model]
        presented.append(
            replace(
                row,
                category=CATEGORY_DISPLAY_NAMES[row.category],
                model=metadata.display_name,
                run_id=metadata.run_id,
                legacy_category=row.category,
                legacy_model_name=row.model,
                model_explanation=metadata.explanation,
                presentation_status=metadata.status,
            )
        )
    return presented


def read_json(path: Path) -> dict | list:
    with path.open() as handle:
        return json.load(handle)


def only_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern} in {directory}, found {len(matches)}")
    return matches[0]


def final_eval_row(
    *,
    directory: Path,
    category_order: int,
    category: str,
    model: str,
    features: str,
    notes: str = "",
) -> ScoreRow:
    path = only_file(directory, "final_evaluation_*.json")
    metrics = read_json(path)
    return ScoreRow(
        category_order=category_order,
        category=category,
        model=model,
        features_or_variant=features,
        test_auprc=float(metrics["test"]["auprc"]),
        leftout_auprc=float(metrics["leftout"]["auprc"]),
        source_artifact=str(path),
        notes=notes,
    )


def prediction_metric_row(
    *,
    directory: Path,
    category_order: int,
    category: str,
    model: str,
    features: str,
    notes: str = "",
) -> ScoreRow:
    test_path = directory / "test_predictions.metrics.json"
    leftout_path = directory / "leftout_predictions.metrics.json"
    test = read_json(test_path)
    leftout = read_json(leftout_path)
    return ScoreRow(
        category_order=category_order,
        category=category,
        model=model,
        features_or_variant=features,
        test_auprc=float(test["auprc"]),
        leftout_auprc=float(leftout["auprc"]),
        source_artifact=f"{test_path}; {leftout_path}",
        notes=notes,
    )


def explicit_row(
    *,
    category_order: int,
    category: str,
    model: str,
    features: str,
    test: float,
    leftout: float,
    source: Path,
    notes: str = "",
) -> ScoreRow:
    return ScoreRow(
        category_order,
        category,
        model,
        features,
        float(test),
        float(leftout),
        str(source),
        notes,
    )


def collect_model_scores() -> list[ScoreRow]:
    rows: list[ScoreRow] = []

    original_runs = [
        (
            "seq_only",
            1,
            "Original sequence CNN",
            "PairwiseSeqCNN",
            "28x50 nucleotide-pair grid; no evolutionary channels",
        ),
        (
            "conservation_phylop",
            2,
            "Conservation CNN",
            "PairwiseConservationCNN — phyloP",
            "Sequence pair grid + target phyloP",
        ),
        (
            "conservation_phastcons",
            2,
            "Conservation CNN",
            "PairwiseConservationCNN — phastCons",
            "Sequence pair grid + target phastCons",
        ),
        (
            "conservation_both",
            2,
            "Conservation CNN",
            "PairwiseConservationCNN — phyloP + phastCons",
            "Sequence pair grid + both target conservation tracks",
        ),
        (
            "phact_CountNodes_3_mirna",
            3,
            "Original PHACT CNN",
            "PHACT CNN — CountNodes_3 miRNA",
            "Sequence grid + miRNA CountNodes_3 A/C/G/T scores",
        ),
        (
            "phact_CountNodes_3_target",
            3,
            "Original PHACT CNN",
            "PHACT CNN — target",
            "Sequence grid + target PHACT A/C/G/T scores",
        ),
        (
            "phact_CountNodes_3_both",
            3,
            "Original PHACT CNN",
            "PHACT CNN — CountNodes_3 both axes",
            "Sequence grid + miRNA CountNodes_3 + target PHACT",
        ),
        (
            "phact_CountNodes_3_both_actual_margin",
            3,
            "Original PHACT CNN",
            "PHACT CNN — actual-margin reduction",
            "Both PHACT axes reduced to actual-nucleotide margin",
        ),
        (
            "phact_CountNodes_3_both_alt_mean",
            3,
            "Original PHACT CNN",
            "PHACT CNN — alternative-mean reduction",
            "Both PHACT axes reduced to alternative-nucleotide mean",
        ),
        (
            "phact_CountNodes_2_0_MinNode_Mix_max05_Gauss_0_CountNodes_3_0_MinNode_Mix2_CountNodes_4_1_0p5_mean_2_5_0p1_median_3_CountNodes_1_target_target_auto_both",
            3,
            "Original PHACT CNN",
            "PHACT CNN — all 16 miRNA models + target",
            "All miRNA PHACT model channels plus target PHACT",
        ),
    ]
    for directory, order, category, model, features in original_runs:
        rows.append(
            final_eval_row(
                directory=WORKSPACE / "runs/baselines/main_repo_outputs" / directory,
                category_order=order,
                category=category,
                model=model,
                features=features,
                notes="Original repository run",
            )
        )

    official_source = EXPERIMENTS / "manakov_test_failure_analysis" / "summary.json"
    official_test = read_json(official_source)["models"]["official_cnn"]["auprc"]
    official_leftout_source = (
        EXPERIMENTS / "manakov_leftout_failure_analysis" / "summary.json"
    )
    official_leftout = read_json(official_leftout_source)["models"]["official_cnn"][
        "auprc"
    ]
    rows.append(
        explicit_row(
            category_order=1,
            category="Original sequence CNN",
            model="Official miRBench Manakov CNN",
            features="Official Keras sequence CNN evaluated through the local reader",
            test=official_test,
            leftout=official_leftout,
            source=official_source,
            notes=f"Left-out metric also read from {official_leftout_source}",
        )
    )

    experimental_runs = [
        (
            "phact_cnn_clean_manakov_control_lr5e5",
            "Clean-Manakov control",
            "CountNodes_3 PHACT; 28 conflict rows removed/replaced; LR 5e-5",
        ),
        (
            "phact_cnn_augmented_balanced50k_lr5e5",
            "Balanced-50k GSE augmentation",
            "CountNodes_3 PHACT + leakage-filtered balanced GSE subset",
        ),
        (
            "phact_cnn_conservation_clean_manakov_lr1e4",
            "PHACT + conservation clean Manakov",
            "CountNodes_3 PHACT + target phyloP/phastCons",
        ),
        (
            "phact_cnn_conservation_augmented_full_lr5e5",
            "PHACT + conservation + conservative GSE",
            "CountNodes_3 PHACT + conservation + leakage-filtered GSE",
        ),
        (
            "phact_cnn_conservation_gse_positive25k_lr5e5",
            "PHACT + conservation + 25k GSE positives",
            "Positive-only 25k GSE augmentation",
        ),
        (
            "phact_cnn_conservation_gse_positive25k_negsmooth002_lr5e5",
            "PHACT + conservation + 25k positives + smoothing",
            "Positive-only augmentation; negative smoothing 0.02",
        ),
        (
            "phact_cnn_conservation_gse_positive100k_lr5e5",
            "PHACT + conservation + 100k GSE positives",
            "Positive-only 100k GSE augmentation",
        ),
        (
            "phact_cnn_conservation_conflict28_shift1_lr5e5",
            "PHACT + conservation + target-shift training",
            "Conflict-28 replacement; random target shift up to one position",
        ),
        (
            "phact_cnn_conservation_conflict28_shift1_focal1_lr5e5",
            "PHACT + conservation + shift + focal loss",
            "Conflict-28 replacement; target shift 1; focal gamma 1",
        ),
        (
            "phact_cnn_full_manakov_conflict28_shift1_finalrefit",
            "Full-Manakov final refit",
            "Train+validation refit; conflict replacements; target shift 1",
        ),
        (
            "phact_cnn_wide_clean_manakov_lr1e4",
            "Widened PHACT CNN",
            "Function-preserving widening to 256,128,64 filters",
        ),
        (
            "phact_outer_clean_manakov_lr1e4",
            "PHACT outer-product interactions",
            "Appended miRNA-by-target PHACT interaction channels",
        ),
    ]
    for directory, model, features in experimental_runs:
        rows.append(
            final_eval_row(
                directory=EXPERIMENTS / directory,
                category_order=4,
                category="Experimental PHACT CNN",
                model=model,
                features=features,
            )
        )
    rows.append(
        prediction_metric_row(
            directory=EXPERIMENTS / "phact_cnn_augmented_conservative_finetune",
            category_order=4,
            category="Experimental PHACT CNN",
            model="Conservative GSE fine-tune",
            features="CountNodes_3 PHACT; leakage-safe GSE; initialized from original PHACT CNN",
            notes="Scores come from exported split predictions; no final_evaluation JSON",
        )
    )

    rows.append(
        final_eval_row(
            directory=EXPERIMENTS
            / "rinalmo_cross_residual_augmented_conservative_frozen",
            category_order=5,
            category="RiNALMo fusion",
            model="RiNALMo cross-encoding residual",
            features="Frozen RiNALMo + frozen miRBind/PHACT baselines; trained residual head",
        )
    )
    failure_test = read_json(
        EXPERIMENTS / "manakov_test_failure_analysis" / "summary.json"
    )
    failure_leftout = read_json(
        EXPERIMENTS / "manakov_leftout_failure_analysis" / "summary.json"
    )
    rows.append(
        explicit_row(
            category_order=5,
            category="RiNALMo fusion",
            model="Fully fine-tuned RiNALMo-PHACT",
            features="RiNALMo-micro + PHACT-conditioned pooling + frozen miRBind",
            test=failure_test["models"]["rinalmo"]["auprc"],
            leftout=failure_leftout["models"]["rinalmo"]["auprc"],
            source=EXPERIMENTS / "manakov_test_failure_analysis" / "summary.json",
            notes="Left-out metric is from the corresponding left-out failure summary",
        )
    )

    auxiliary = [
        (
            "seed_metadata_no_leakage",
            "Seed/region metadata LightGBM",
            "Seed-match counts, GC, miRNA length, and target-region metadata",
            EXPERIMENTS / "seed_metadata_baseline_no_leakage" / "metrics.json",
        ),
        (
            "rnaduplex",
            "RNAduplex LightGBM",
            "ViennaRNA duplex energy and alignment-derived features",
            EXPERIMENTS / "rnaduplex_features" / "lightgbm" / "metrics.json",
        ),
        (
            "phact_latent",
            "PHACT latent LightGBM",
            "LightGBM on frozen 30-dimensional PHACT-CNN latent vectors",
            EXPERIMENTS / "phact_latent_shift_model" / "lightgbm" / "metrics.json",
        ),
    ]
    for _, model, features, path in auxiliary:
        metrics = read_json(path)["splits"]
        rows.append(
            explicit_row(
                category_order=6,
                category="Auxiliary feature model",
                model=model,
                features=features,
                test=metrics["test"]["auprc"],
                leftout=metrics["leftout"]["auprc"],
                source=path,
            )
        )

    ensemble_metrics = [
        (
            "Target-shift prediction ensemble",
            "Validation-selected blend of target shifts -1/0/+1",
            EXPERIMENTS / "phact_shift_ensemble" / "metrics.json",
        ),
        (
            "Shift-trained prediction ensemble",
            "Target-shift-trained PHACT CNN blended across inference shifts",
            EXPERIMENTS / "phact_shift_trained_ensemble" / "metrics.json",
        ),
        (
            "Focal/shift prediction ensemble",
            "Focal-loss PHACT CNN blended across target shifts",
            EXPERIMENTS / "phact_focal_shift_ensemble" / "metrics.json",
        ),
        (
            "Final-refit shift ensemble",
            "Full-Manakov refit blended with its shifted prediction",
            EXPERIMENTS / "phact_finalrefit_shift_ensemble" / "metrics.json",
        ),
        (
            "Validation-selected heterogeneous ensemble",
            "Shift PHACT + seed metadata + miRNA-only PHACT + RNAduplex",
            EXPERIMENTS / "validation_selected_ensemble" / "metrics.json",
        ),
        (
            "Prediction stacker",
            "LightGBM stacker over ranked/logit outputs from 14 model families",
            EXPERIMENTS / "prediction_stacker" / "metrics.json",
        ),
    ]
    for model, features, path in ensemble_metrics:
        metrics = read_json(path)
        splits = metrics.get("splits", metrics.get("ensemble"))
        test = splits["test"]
        leftout = splits["leftout"]
        rows.append(
            explicit_row(
                category_order=7,
                category="Ensemble or stacker",
                model=model,
                features=features,
                test=test["auprc"] if isinstance(test, dict) else test,
                leftout=(
                    leftout["auprc"] if isinstance(leftout, dict) else leftout
                ),
                source=path,
                notes="Development ensemble; validation-selected where stated",
            )
        )

    ablation_dirs = [
        (
            "Test-derived candidate baseline",
            EXPERIMENTS
            / "manakov_test_failure_analysis"
            / "candidate_ablation_seed42_baseline",
            "22 reviewed rows retained at weight 1.0",
        ),
        (
            "Test-derived candidates downweighted",
            EXPERIMENTS
            / "manakov_test_failure_analysis"
            / "candidate_ablation_seed42_downweight025",
            "22 reviewed rows weighted 0.25",
        ),
        (
            "Test-derived candidates removed",
            EXPERIMENTS
            / "manakov_test_failure_analysis"
            / "candidate_ablation_seed42_remove",
            "22 reviewed rows weighted 0",
        ),
        (
            "Validation-derived candidates downweighted",
            EXPERIMENTS
            / "manakov_validation_failure_analysis"
            / "candidate_ablation_seed42_downweight025",
            "53 reviewed rows weighted 0.25",
        ),
        (
            "Validation-derived candidates removed",
            EXPERIMENTS
            / "manakov_validation_failure_analysis"
            / "candidate_ablation_seed42_remove",
            "53 reviewed rows weighted 0",
        ),
    ]
    for model, directory, features in ablation_dirs:
        rows.append(
            prediction_metric_row(
                directory=directory,
                category_order=8,
                category="Candidate-row ablation",
                model=model,
                features=features,
                notes="One-epoch controlled diagnostic; not a primary model claim",
            )
        )

    agentomics_path = AGENTOMICS / "top10_test_evaluation" / "summary.json"
    agentomics = read_json(agentomics_path)
    by_iteration: dict[int, dict[str, dict]] = defaultdict(dict)
    for record in agentomics:
        if record.get("status") == "success":
            by_iteration[int(record["iteration"])][str(record["split"])] = record
    for iteration in sorted(by_iteration):
        splits = by_iteration[iteration]
        if set(splits) != {"test", "leftout"}:
            continue
        notes = "Agentomics top-10 finalist"
        if iteration == 37:
            notes += "; best-iteration snapshot with soft-alignment branch selected at epoch 0"
        if iteration == 38:
            notes += "; artifact-equivalent to iteration 36"
        rows.append(
            explicit_row(
                category_order=9,
                category="Agentomics model",
                model=f"Agentomics iteration {iteration}",
                features="Automated multimodal PHACT/sequence model search finalist",
                test=splits["test"]["AUPRC"],
                leftout=splits["leftout"]["AUPRC"],
                source=agentomics_path,
                notes=notes,
            )
        )

    if not rows:
        raise RuntimeError("No model scores were collected")
    presented_rows = apply_presentation_metadata(rows)
    return sorted(
        presented_rows,
        key=lambda row: (row.category_order, row.model.casefold()),
    )


def write_score_tables(rows: list[ScoreRow]) -> None:
    output_rows: list[dict[str, object]] = []
    for row in rows:
        category_rows = [candidate for candidate in rows if candidate.category == row.category]
        test_ranked = sorted(
            category_rows,
            key=lambda candidate: (-candidate.test_auprc, candidate.model.casefold()),
        )
        leftout_ranked = sorted(
            category_rows,
            key=lambda candidate: (-candidate.leftout_auprc, candidate.model.casefold()),
        )
        record = asdict(row)
        record["test_rank_within_category"] = test_ranked.index(row) + 1
        record["leftout_rank_within_category"] = leftout_ranked.index(row) + 1
        output_rows.append(record)

    fieldnames = [
        "category_order",
        "category",
        "run_id",
        "model",
        "model_explanation",
        "features_or_variant",
        "test_auprc",
        "leftout_auprc",
        "test_rank_within_category",
        "leftout_rank_within_category",
        "presentation_status",
        "legacy_category",
        "legacy_model_name",
        "source_artifact",
        "notes",
    ]
    with (OUTPUT_DIR / "all_model_test_leftout_auprc.tsv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)

    with (OUTPUT_DIR / "all_model_test_leftout_auprc.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    markdown = [
        "# Test and left-out AUPRC by model category",
        "",
        "Metrics are copied from the recorded evaluation artifacts listed in the source column.",
        "Test and left-out have been inspected repeatedly and should be treated as development monitoring rather than untouched final estimates.",
        "",
    ]
    for category_order in sorted({row.category_order for row in rows}):
        category_rows = [row for row in rows if row.category_order == category_order]
        category = category_rows[0].category
        markdown.extend(
            [
                f"## {category}",
                "",
                "| Model | Explanation | Test AUPRC | Left-out AUPRC | Run ID |",
                "|---|---|---:|---:|---|",
            ]
        )
        for row in sorted(category_rows, key=lambda item: -item.test_auprc):
            model = row.model.replace("|", "\\|")
            explanation = row.model_explanation.replace("|", "\\|")
            markdown.append(
                f"| {model} | {explanation} | {row.test_auprc:.6f} | "
                f"{row.leftout_auprc:.6f} | `{row.run_id}` |"
            )
        markdown.append("")
    (OUTPUT_DIR / "all_model_test_leftout_auprc.md").write_text(
        "\n".join(markdown) + "\n"
    )


def write_model_performance_plot(rows: list[ScoreRow]) -> None:
    category_colors = {
        "Sequence-only baselines": "#4E79A7",
        "Conservation-augmented CNNs": "#59A14F",
        "Original PHACT-CNN variants": "#F28E2B",
        "PHACT-CNN extensions": "#E15759",
        "PHACT–RiNALMo fusion": "#B07AA1",
        "Feature baselines and representation probes": "#76B7B2",
        "Prediction-level combinations": "#EDC948",
        "Data-curation sensitivity checks": "#A0A0A0",
        "Agentomics architecture-search finalists": "#FF9DA7",
    }
    plot_rows = [row for row in rows if row.presentation_status != "duplicate"]
    primary_rows = [row for row in plot_rows if row.presentation_status == "primary"]
    ordered_categories = [
        category
        for _, category in sorted(
            {(row.category_order, row.category) for row in rows}
        )
    ]

    annotation_reasons: dict[str, list[str]] = defaultdict(list)
    for category in ordered_categories:
        if category == "Data-curation sensitivity checks":
            continue
        category_rows = [row for row in primary_rows if row.category == category]
        leader = max(
            category_rows,
            key=lambda row: (
                (row.test_auprc + row.leftout_auprc) / 2,
                row.leftout_auprc,
                row.test_auprc,
            ),
        )
        annotation_reasons[leader.run_id].append(
            "category leader by mean test/left-out AUPRC"
        )

    best_test = max(primary_rows, key=lambda row: row.test_auprc)
    best_leftout = max(primary_rows, key=lambda row: row.leftout_auprc)
    annotation_reasons[best_test.run_id].append(
        "overall best test AUPRC"
    )
    annotation_reasons[best_leftout.run_id].append(
        "overall best left-out AUPRC"
    )

    annotated_rows = [row for row in primary_rows if row.run_id in annotation_reasons]
    annotation_fields = [
        "category",
        "run_id",
        "model",
        "model_explanation",
        "test_auprc",
        "leftout_auprc",
        "selection_reason",
    ]
    with (OUTPUT_DIR / "all_models_plot_annotations.tsv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=annotation_fields, delimiter="\t")
        writer.writeheader()
        for row in annotated_rows:
            writer.writerow(
                {
                    "category": row.category,
                    "run_id": row.run_id,
                    "model": row.model,
                    "model_explanation": row.model_explanation,
                    "test_auprc": row.test_auprc,
                    "leftout_auprc": row.leftout_auprc,
                    "selection_reason": "; ".join(annotation_reasons[row.run_id]),
                }
            )

    figure, axes = plt.subplots(1, 2, figsize=(16, 12.5))
    for axis in axes:
        for category in ordered_categories:
            category_rows = [
                row
                for row in plot_rows
                if row.category == category and row.presentation_status == "primary"
            ]
            axis.scatter(
                [row.test_auprc for row in category_rows],
                [row.leftout_auprc for row in category_rows],
                s=52,
                color=category_colors[category],
                alpha=0.8,
                edgecolor="white",
                linewidth=0.65,
                zorder=2,
            )
            diagnostic_rows = [
                row
                for row in plot_rows
                if row.category == category and row.presentation_status == "diagnostic"
            ]
            axis.scatter(
                [row.test_auprc for row in diagnostic_rows],
                [row.leftout_auprc for row in diagnostic_rows],
                s=58,
                facecolors="none",
                edgecolors=category_colors[category],
                linewidth=1.5,
                zorder=2,
            )
        axis.scatter(
            [row.test_auprc for row in annotated_rows],
            [row.leftout_auprc for row in annotated_rows],
            s=105,
            facecolors=[category_colors[row.category] for row in annotated_rows],
            edgecolor="#1F2933",
            linewidth=1.3,
            zorder=3,
        )
        axis.set_xlabel("Test AUPRC", fontsize=14)
        axis.set_ylabel("Left-out AUPRC", fontsize=14)
        axis.tick_params(axis="both", labelsize=12)
        axis.grid(color="#DDE3EA", linewidth=0.7, alpha=0.8, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)

    test_values = [row.test_auprc for row in plot_rows]
    leftout_values = [row.leftout_auprc for row in plot_rows]
    axes[0].set_xlim(min(test_values) - 0.005, max(test_values) + 0.005)
    axes[0].set_ylim(min(leftout_values) - 0.005, max(leftout_values) + 0.005)
    axes[0].set_title(
        f"All {len(plot_rows)} non-duplicate artifacts",
        loc="left",
        fontweight="bold",
        fontsize=15,
    )

    axes[1].set_xlim(0.875, 0.892)
    axes[1].set_ylim(0.858, 0.874)
    axes[1].set_title(
        "Competitive region", loc="left", fontweight="bold", fontsize=15
    )

    short_labels = {
        "Sequence CNN (PairwiseSeqCNN)": "Sequence CNN",
        "Sequence CNN + phastCons": "phastCons CNN",
        "PHACT-CNN — 16 miRNA tracks + target PHACT": "16-track PHACT-CNN",
        "PHACT-CNN — miRNA CountNodes_3": "miRNA CountNodes_3",
        "PHACT-CNN — conservation + filtered GSE": "Conservation + filtered GSE",
        "PHACT–RiNALMo — frozen residual fusion": "Frozen RiNALMo residual",
        "PHACT latent-space probe": "PHACT latent-space probe",
        "14-family LightGBM stacker": "14-family stacker",
        "Agentomics — frozen RiNALMo-mega fusion": "Frozen RiNALMo-mega fusion",
    }
    highlight_summaries = {
        "Sequence CNN (PairwiseSeqCNN)": (
            "28×50 nucleotide-pair grid; no evolutionary scores."
        ),
        "Sequence CNN + phastCons": (
            "Sequence CNN plus target-position phastCons scores."
        ),
        "PHACT-CNN — 16 miRNA tracks + target PHACT": (
            "All 16 miRNA PHACT tracks plus target PHACT."
        ),
        "PHACT-CNN — miRNA CountNodes_3": (
            "Sequence CNN plus miRNA A/C/G/T CountNodes_3 scores."
        ),
        "PHACT-CNN — conservation + filtered GSE": (
            "PHACT/conservation CNN trained with leakage-filtered GSE rows."
        ),
        "PHACT–RiNALMo — frozen residual fusion": (
            "Frozen RiNALMo and miRBind/PHACT branches with a trained residual head."
        ),
        "PHACT latent-space probe": (
            "LightGBM fitted to frozen 30-dimensional PHACT-CNN embeddings."
        ),
        "14-family LightGBM stacker": (
            "LightGBM over rank/logit outputs from 14 model families."
        ),
        "Agentomics — frozen RiNALMo-mega fusion": (
            "Sequence/PHACT/context fusion with frozen RiNALMo-mega embeddings."
        ),
    }
    if set(highlight_summaries) != {row.model for row in annotated_rows}:
        raise ValueError("Highlighted-model explanations do not match annotations")
    label_offsets = {
        "Sequence CNN (PairwiseSeqCNN)": (10, -19),
        "Sequence CNN + phastCons": (-92, -10),
        "PHACT-CNN — 16 miRNA tracks + target PHACT": (10, -27),
        "PHACT-CNN — miRNA CountNodes_3": (10, 7),
        "PHACT-CNN — conservation + filtered GSE": (10, 13),
        "PHACT–RiNALMo — frozen residual fusion": (-100, 11),
        "PHACT latent-space probe": (10, -17),
        "14-family LightGBM stacker": (-92, 15),
        "Agentomics — frozen RiNALMo-mega fusion": (-120, -7),
    }
    for row in annotated_rows:
        axes[1].annotate(
            short_labels.get(row.model, row.model),
            xy=(row.test_auprc, row.leftout_auprc),
            xytext=label_offsets.get(row.model, (8, 8)),
            textcoords="offset points",
            fontsize=10.5,
            ha="left",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
            },
            arrowprops={"arrowstyle": "-", "color": "#6B7280", "linewidth": 0.7},
            zorder=4,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=8.5,
            markerfacecolor=category_colors[category],
            markeredgecolor="white",
            label=category,
        )
        for category in ordered_categories
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.445),
        ncol=3,
        frameon=False,
        fontsize=11,
        columnspacing=2.0,
        handletextpad=0.7,
    )
    figure.text(
        0.075,
        0.385,
        "Highlighted models",
        ha="left",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    first_row_y = 0.35
    row_spacing = 0.036
    for index, row in enumerate(annotated_rows):
        row_y = first_row_y - index * row_spacing
        figure.text(
            0.075,
            row_y,
            "●",
            color=category_colors[row.category],
            ha="left",
            va="center",
            fontsize=12,
        )
        figure.text(
            0.095,
            row_y,
            f"{short_labels[row.model]}: {highlight_summaries[row.model]}",
            color="#111827",
            ha="left",
            va="center",
            fontsize=11,
        )
    figure.text(
        0.5,
        0.025,
        "Black outline + plot label: highlighted model. Hollow Agentomics marker: diagnostic branch not selected.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#111827",
    )
    figure.suptitle(
        "Test versus left-out performance across PHACT model families",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )
    figure.tight_layout(rect=(0, 0.50, 1, 0.94), w_pad=3.0)
    figure.savefig(
        OUTPUT_DIR / "all_models_test_vs_leftout_auprc.png",
        dpi=220,
        bbox_inches="tight",
    )
    figure.savefig(
        OUTPUT_DIR / "all_models_test_vs_leftout_auprc.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


FOCUSED_MIRNAS = [
    {
        "slug": "mir17",
        "display_name": "miR-17-5p",
        "precursor_id": "Hsa-Mir-17-P1a",
        "mature_id": "Hsa-Mir-17-P1a_5p",
        "color": "#0072B2",
    },
    {
        "slug": "let7",
        "display_name": "let-7-5p",
        "precursor_id": "Hsa-Let-7-P1b",
        "mature_id": "Hsa-Let-7-P1b_5p",
        "color": "#D55E00",
    },
]


def read_fasta(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current_id: str | None = None
    current_parts: list[str] = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_parts)
                current_id = line[1:].split()[0]
                current_parts = []
            else:
                current_parts.append(line.upper().replace("U", "T"))
    if current_id is not None:
        sequences[current_id] = "".join(current_parts)
    return sequences


def precursor_sequences() -> dict[str, str]:
    path = WORKSPACE / "data/inputs/reference/hsa_premirnas_flank30.tsv"
    sequences: dict[str, str] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sequences[row["mirgenedb_id"]] = (
                row["pre_sequence"].strip().upper().replace("U", "T")
            )
    return sequences


def collect_countnodes3_scores(
    reader: csv.DictReader,
    selected_precursors: set[str],
    sequences: dict[str, str],
    source: str,
) -> dict[str, list[dict[str, object]]]:
    if COUNTNODES3_COLUMN not in (reader.fieldnames or []):
        raise ValueError(f"Missing {COUNTNODES3_COLUMN} in {source}")
    values: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row in reader:
        precursor_id = row["ID"]
        if precursor_id not in selected_precursors:
            continue
        pre_position = int(row["Position"]) - 30
        sequence = sequences.get(precursor_id)
        if sequence is None or not 1 <= pre_position <= len(sequence):
            continue
        nucleotide = row["Nucleotide"].upper().replace("U", "T")
        if nucleotide not in {"A", "C", "G", "T"}:
            continue
        values[precursor_id].append(
            {
                "precursor_position_1based": pre_position,
                "scored_nucleotide": nucleotide,
                "countnodes3_score": float(row[COUNTNODES3_COLUMN]),
            }
        )

    return {
        precursor_id: sorted(
            rows,
            key=lambda row: (
                int(row["precursor_position_1based"]),
                str(row["scored_nucleotide"]),
            ),
        )
        for precursor_id, rows in values.items()
    }


def load_countnodes3_all_scores(
    selected_precursors: set[str], sequences: dict[str, str]
) -> dict[str, list[dict[str, object]]]:
    archive_path = SCRATCH / "data/inputs/consensus_tree_scores.zip"
    member = "results_qntnorm_transformed.tsv.gz"
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member) as compressed:
            with gzip.open(compressed, mode="rt", newline="") as text:
                return collect_countnodes3_scores(
                    csv.DictReader(text, delimiter="\t"),
                    selected_precursors,
                    sequences,
                    f"{archive_path}:{member}",
                )


def load_countnodes3_gzip_scores(
    table_path: Path,
    selected_precursors: set[str],
    sequences: dict[str, str],
) -> dict[str, list[dict[str, object]]]:
    with gzip.open(table_path, mode="rt", newline="") as text:
        return collect_countnodes3_scores(
            csv.DictReader(text, delimiter="\t"),
            selected_precursors,
            sequences,
            str(table_path),
        )


def build_reference_minus_alt_rows(
    scores_by_precursor: dict[str, list[dict[str, object]]] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    precursor_by_id = precursor_sequences()
    mature_by_id = read_fasta(WORKSPACE / "data/inputs/reference/hsa_mature.fa")
    selected_precursors = {
        str(selection["precursor_id"]) for selection in FOCUSED_MIRNAS
    }
    if scores_by_precursor is None:
        scores_by_precursor = load_countnodes3_all_scores(
            selected_precursors, precursor_by_id
        )

    full_rows: list[dict[str, object]] = []
    mature_rows: list[dict[str, object]] = []
    for selection in FOCUSED_MIRNAS:
        precursor_id = str(selection["precursor_id"])
        mature_id = str(selection["mature_id"])
        precursor_sequence = precursor_by_id[precursor_id]
        mature_sequence = mature_by_id[mature_id]
        mature_start_zero_based = precursor_sequence.find(mature_sequence)
        if mature_start_zero_based < 0 or precursor_sequence.count(mature_sequence) != 1:
            raise ValueError(
                f"Expected one exact {mature_id} match in {precursor_id}"
            )
        mature_start = mature_start_zero_based + 1
        mature_end = mature_start_zero_based + len(mature_sequence)

        scores_by_position: dict[int, dict[str, float]] = defaultdict(dict)
        for point in scores_by_precursor.get(precursor_id, []):
            position = int(point["precursor_position_1based"])
            nucleotide = str(point["scored_nucleotide"])
            scores_by_position[position][nucleotide] = float(
                point["countnodes3_score"]
            )
        expected_positions = set(range(1, len(precursor_sequence) + 1))
        if set(scores_by_position) != expected_positions:
            raise ValueError(f"Incomplete CountNodes_3 positions for {precursor_id}")

        for position in range(1, len(precursor_sequence) + 1):
            nucleotide_scores = scores_by_position[position]
            if set(nucleotide_scores) != {"A", "C", "G", "T"}:
                raise ValueError(
                    f"Incomplete nucleotide scores for {precursor_id}:{position}"
                )
            reference_nt = precursor_sequence[position - 1]
            reference_score = nucleotide_scores[reference_nt]
            alternative_nts = sorted({"A", "C", "G", "T"} - {reference_nt})
            mean_alternative_score = sum(
                nucleotide_scores[nucleotide] for nucleotide in alternative_nts
            ) / len(alternative_nts)
            is_mature = mature_start <= position <= mature_end
            row = {
                "mirna": selection["display_name"],
                "precursor_id": precursor_id,
                "mature_id": mature_id,
                "precursor_length_nt": len(precursor_sequence),
                "mature_length_nt": len(mature_sequence),
                "mature_start_in_precursor_1based": mature_start,
                "mature_end_in_precursor_1based": mature_end,
                "precursor_position_1based": position,
                "mature_position_1based": (
                    position - mature_start + 1 if is_mature else ""
                ),
                "reference_nt": reference_nt,
                "score_A": nucleotide_scores["A"],
                "score_C": nucleotide_scores["C"],
                "score_G": nucleotide_scores["G"],
                "score_T": nucleotide_scores["T"],
                "reference_score": reference_score,
                "alternative_nts": ",".join(alternative_nts),
                "mean_alternative_score": mean_alternative_score,
                "reference_minus_mean_alternative": (
                    reference_score - mean_alternative_score
                ),
            }
            full_rows.append(row)
            if is_mature:
                mature_rows.append(row)
    return full_rows, mature_rows


def write_reference_minus_alt_table(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def delta_limits_from_values(values: list[float]) -> tuple[float, float]:
    value_range = max(max(values) - min(values), 0.1)
    padding = value_range * 0.08
    return min(min(values) - padding, -padding / 2), max(
        max(values) + padding, padding / 2
    )


def delta_limits(rows: list[dict[str, object]]) -> tuple[float, float]:
    return delta_limits_from_values(
        [float(row["reference_minus_mean_alternative"]) for row in rows]
    )


def style_delta_axis(axis: plt.Axes, y_limits: tuple[float, float]) -> None:
    axis.axhline(0, color="#4B5563", linewidth=0.9, zorder=1)
    axis.set_ylim(*y_limits)
    axis.grid(axis="y", color="#D8DEE9", linewidth=0.7, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def write_mir17_let7_outputs() -> None:
    full_rows, mature_rows = build_reference_minus_alt_rows()
    write_reference_minus_alt_table(
        OUTPUT_DIR / "mir17_let7_countnodes3_ref_minus_alt_full_premirna.tsv",
        full_rows,
    )
    write_reference_minus_alt_table(
        OUTPUT_DIR / "mir17_let7_countnodes3_ref_minus_alt_mature.tsv",
        mature_rows,
    )

    full_limits = delta_limits(full_rows)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for axis, selection in zip(axes, FOCUSED_MIRNAS):
        points = [
            row
            for row in full_rows
            if row["precursor_id"] == selection["precursor_id"]
        ]
        mature_start = int(points[0]["mature_start_in_precursor_1based"])
        mature_end = int(points[0]["mature_end_in_precursor_1based"])
        axis.axvspan(
            mature_start - 0.5,
            mature_end + 0.5,
            color="#E5E7EB",
            alpha=0.75,
            zorder=0,
            label="Mature 5p interval",
        )
        axis.plot(
            [row["precursor_position_1based"] for row in points],
            [row["reference_minus_mean_alternative"] for row in points],
            color=selection["color"],
            linewidth=2.0,
            marker="o",
            markersize=2.8,
            label="Reference − mean alternative",
            zorder=2,
        )
        axis.set_title(
            f"{selection['display_name']}\n{selection['precursor_id']}",
            loc="left",
            fontweight="bold",
        )
        axis.set_xlim(0.5, int(points[0]["precursor_length_nt"]) + 0.5)
        style_delta_axis(axis, full_limits)
        axis.legend(frameon=False, fontsize=8, loc="lower right")
    figure.suptitle(
        "CountNodes_3 reference-minus-alternative score across full pre-miRNAs",
        fontsize=15,
        fontweight="bold",
    )
    figure.supxlabel("Pre-miRNA position (1-based)", y=0.02)
    figure.supylabel("Reference score − mean alternative score", x=0.01)
    figure.tight_layout(rect=(0.025, 0.045, 1, 0.91), w_pad=2.5)
    figure.savefig(
        OUTPUT_DIR / "mir17_let7_countnodes3_ref_minus_alt_full_premirna.png",
        dpi=220,
        bbox_inches="tight",
    )
    figure.savefig(
        OUTPUT_DIR / "mir17_let7_countnodes3_ref_minus_alt_full_premirna.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)

    mature_limits = delta_limits(mature_rows)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for axis, selection in zip(axes, FOCUSED_MIRNAS):
        points = [
            row
            for row in mature_rows
            if row["precursor_id"] == selection["precursor_id"]
        ]
        axis.plot(
            [row["mature_position_1based"] for row in points],
            [row["reference_minus_mean_alternative"] for row in points],
            color=selection["color"],
            linewidth=2.2,
            marker="o",
            markersize=4.0,
            zorder=2,
        )
        axis.set_title(
            f"{selection['display_name']}\n{selection['mature_id']}",
            loc="left",
            fontweight="bold",
        )
        axis.set_xlim(0.5, int(points[0]["mature_length_nt"]) + 0.5)
        style_delta_axis(axis, mature_limits)
    figure.suptitle(
        "CountNodes_3 reference-minus-alternative score within mature miRNAs",
        fontsize=15,
        fontweight="bold",
    )
    figure.supxlabel("Mature miRNA position (1-based)", y=0.02)
    figure.supylabel("Reference score − mean alternative score", x=0.01)
    figure.tight_layout(rect=(0.025, 0.045, 1, 0.91), w_pad=2.5)
    figure.savefig(
        OUTPUT_DIR / "mir17_let7_countnodes3_ref_minus_alt_mature.png",
        dpi=220,
        bbox_inches="tight",
    )
    figure.savefig(
        OUTPUT_DIR / "mir17_let7_countnodes3_ref_minus_alt_mature.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def build_old_vs_new_countnodes3_rows() -> list[dict[str, object]]:
    sequences = precursor_sequences()
    selected_precursors = {
        str(selection["precursor_id"]) for selection in FOCUSED_MIRNAS
    }
    old_scores = load_countnodes3_gzip_scores(
        OLD_COUNTNODES3_TABLE, selected_precursors, sequences
    )
    new_scores = load_countnodes3_gzip_scores(
        CONSENSUS_TREE_COUNTNODES3_TABLE, selected_precursors, sequences
    )
    old_rows, _ = build_reference_minus_alt_rows(old_scores)
    new_rows, _ = build_reference_minus_alt_rows(new_scores)

    key = lambda row: (
        str(row["precursor_id"]),
        int(row["precursor_position_1based"]),
    )
    old_by_key = {key(row): row for row in old_rows}
    new_by_key = {key(row): row for row in new_rows}
    if old_by_key.keys() != new_by_key.keys():
        raise ValueError("Old and consensus-tree score positions do not match")

    common_fields = [
        "mirna",
        "precursor_id",
        "mature_id",
        "precursor_length_nt",
        "mature_length_nt",
        "mature_start_in_precursor_1based",
        "mature_end_in_precursor_1based",
        "precursor_position_1based",
        "mature_position_1based",
        "reference_nt",
        "alternative_nts",
    ]
    comparison_rows: list[dict[str, object]] = []
    for row_key in sorted(old_by_key):
        old_row = old_by_key[row_key]
        new_row = new_by_key[row_key]
        for field in common_fields:
            if old_row[field] != new_row[field]:
                raise ValueError(f"Old/new metadata mismatch at {row_key}: {field}")
        comparison_row = {field: old_row[field] for field in common_fields}
        comparison_row["is_mature"] = bool(old_row["mature_position_1based"])
        for source, source_row in (("old", old_row), ("new", new_row)):
            for nucleotide in ("A", "C", "G", "T"):
                comparison_row[f"{source}_score_{nucleotide}"] = source_row[
                    f"score_{nucleotide}"
                ]
            comparison_row[f"{source}_reference_score"] = source_row[
                "reference_score"
            ]
            comparison_row[f"{source}_mean_alternative_score"] = source_row[
                "mean_alternative_score"
            ]
            comparison_row[f"{source}_reference_minus_mean_alternative"] = (
                source_row["reference_minus_mean_alternative"]
            )
        comparison_rows.append(comparison_row)
    return comparison_rows


def write_old_vs_new_countnodes3_outputs() -> None:
    rows = build_old_vs_new_countnodes3_rows()
    table_path = (
        OUTPUT_DIR
        / "mir17_let7_countnodes3_old_vs_consensustree_ref_minus_alt.tsv"
    )
    write_reference_minus_alt_table(table_path, rows)

    comparison_limits = delta_limits_from_values(
        [
            float(row[field])
            for row in rows
            for field in (
                "old_reference_minus_mean_alternative",
                "new_reference_minus_mean_alternative",
            )
        ]
    )
    for selection in FOCUSED_MIRNAS:
        precursor_id = str(selection["precursor_id"])
        points = [row for row in rows if row["precursor_id"] == precursor_id]
        mature_start = int(points[0]["mature_start_in_precursor_1based"])
        mature_end = int(points[0]["mature_end_in_precursor_1based"])

        figure, axis = plt.subplots(figsize=(10.5, 6.3))
        axis.axvspan(
            mature_start - 0.5,
            mature_end + 0.5,
            color="#E5E7EB",
            alpha=0.8,
            zorder=0,
            label=(
                f"Mature: {selection['mature_id']} "
                f"(positions {mature_start}–{mature_end})"
            ),
        )
        axis.plot(
            [row["precursor_position_1based"] for row in points],
            [row["old_reference_minus_mean_alternative"] for row in points],
            color="#0072B2",
            linewidth=2.0,
            marker="o",
            markersize=3.0,
            label="Old PHACT scores",
            zorder=2,
        )
        axis.plot(
            [row["precursor_position_1based"] for row in points],
            [row["new_reference_minus_mean_alternative"] for row in points],
            color="#D55E00",
            linewidth=2.0,
            marker="s",
            markersize=2.8,
            label="Consensus-tree scores",
            zorder=3,
        )
        axis.set_title(
            f"{selection['display_name']} — {precursor_id}",
            loc="left",
            fontsize=14,
            fontweight="bold",
        )
        axis.set_xlabel("Pre-miRNA position (1-based)")
        axis.set_ylabel("Reference score − mean alternative score")
        axis.set_xlim(0.5, int(points[0]["precursor_length_nt"]) + 0.5)
        style_delta_axis(axis, comparison_limits)
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            frameon=False,
            fontsize=8.5,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
        )
        figure.tight_layout(rect=(0, 0.1, 1, 1))
        output_stem = (
            f"{selection['slug']}_countnodes3_old_vs_consensustree_"
            "ref_minus_alt_full_premirna"
        )
        figure.savefig(
            OUTPUT_DIR / f"{output_stem}.png",
            dpi=220,
            bbox_inches="tight",
        )
        figure.savefig(
            OUTPUT_DIR / f"{output_stem}.pdf",
            bbox_inches="tight",
        )
        plt.close(figure)


def write_readme(score_rows: list[ScoreRow]) -> None:
    categories = Counter(row.category for row in score_rows)
    lines = [
        "# PHACT final summaries",
        "",
        "This directory consolidates recorded model evaluation scores and the requested CountNodes_3 precursor profiles.",
        "",
        "## Files",
        "",
        "- `all_model_test_leftout_auprc.tsv`: machine-readable master score table with presentation names, stable run IDs, explanations, legacy labels, and source artifacts.",
        "- `all_model_test_leftout_auprc.csv`: CSV copy of the master table.",
        "- `all_model_test_leftout_auprc.md`: grouped human-readable table.",
        "- `all_models_test_vs_leftout_auprc.png` and `.pdf`: category-colored comparison of non-duplicate model artifacts with a competitive-region zoom.",
        "- `all_models_plot_annotations.tsv`: models labeled in the comparison figure and their selection reasons.",
        "- `mir17_let7_countnodes3_ref_minus_alt_full_premirna.tsv`: position-level miR-17 and let-7 scores across the full precursors.",
        "- `mir17_let7_countnodes3_ref_minus_alt_full_premirna.png` and `.pdf`: full-pre-miRNA reference-minus-alternative profiles.",
        "- `mir17_let7_countnodes3_ref_minus_alt_mature.tsv`: mature-region subset with mature-relative coordinates.",
        "- `mir17_let7_countnodes3_ref_minus_alt_mature.png` and `.pdf`: mature-only reference-minus-alternative profiles.",
        "- `mir17_let7_countnodes3_old_vs_consensustree_ref_minus_alt.tsv`: paired old and consensus-tree CountNodes_3 values for both focused precursors.",
        "- `mir17_countnodes3_old_vs_consensustree_ref_minus_alt_full_premirna.png` and `.pdf`: old-versus-new miR-17 precursor profile with its mature interval annotated.",
        "- `let7_countnodes3_old_vs_consensustree_ref_minus_alt_full_premirna.png` and `.pdf`: old-versus-new let-7 precursor profile with its mature interval annotated.",
        "",
        "## Definitions",
        "",
        "- Model scores are AUPRC values copied from recorded test and left-out artifacts.",
        "- `model` and `category` are concise presentation labels; `run_id`, `legacy_model_name`, and `source_artifact` preserve provenance.",
        "- Comparison-plot labels select the highest mean of test and left-out AUPRC in each non-ablation category, plus any separate overall test or left-out winner.",
        "- Agentomics iteration 37 is shown as a hollow diagnostic point because checkpoint selection retained its inherited model. Iteration 38 remains in the master table but is omitted from the plot because it is prediction-equivalent to iteration 36.",
        "- The focused profiles use MirGeneDB `Hsa-Mir-17-P1a_5p` and `Hsa-Let-7-P1b_5p`, whose mature sequences map uniquely within their corresponding precursors.",
        "- At each position, the plotted value is the qntnorm-transformed CountNodes_3 score of the reference nucleotide minus the mean score of the three non-reference nucleotides.",
        "- Old-versus-new figures compare `PHACT_scores_ALL_0226` with `PHACT_scores_orthologs_with_consensusTree` using matched precursor ID, position, nucleotide, and CountNodes_3 column name.",
        "- Positive values indicate that the reference nucleotide has a higher CountNodes_3 score than the alternative-nucleotide mean; this is a score contrast, not an experimentally measured mutation effect.",
        "",
        "## Evaluation caution",
        "",
        "Test and left-out results have been inspected repeatedly during development. They are monitoring results, not untouched final estimates.",
        "",
        "## Included model counts",
        "",
    ]
    for category in sorted(categories):
        lines.append(f"- {category}: {categories[category]}")
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n")


SUPERSEDED_OUTPUTS = [
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
]


def remove_superseded_outputs() -> None:
    for filename in SUPERSEDED_OUTPUTS:
        (OUTPUT_DIR / filename).unlink(missing_ok=True)


def validate_outputs(score_rows: list[ScoreRow]) -> None:
    expected_files = [
        "README.md",
        "all_model_test_leftout_auprc.tsv",
        "all_model_test_leftout_auprc.csv",
        "all_model_test_leftout_auprc.md",
        "all_models_test_vs_leftout_auprc.png",
        "all_models_test_vs_leftout_auprc.pdf",
        "all_models_plot_annotations.tsv",
        "mir17_let7_countnodes3_ref_minus_alt_full_premirna.tsv",
        "mir17_let7_countnodes3_ref_minus_alt_full_premirna.png",
        "mir17_let7_countnodes3_ref_minus_alt_full_premirna.pdf",
        "mir17_let7_countnodes3_ref_minus_alt_mature.tsv",
        "mir17_let7_countnodes3_ref_minus_alt_mature.png",
        "mir17_let7_countnodes3_ref_minus_alt_mature.pdf",
        "mir17_let7_countnodes3_old_vs_consensustree_ref_minus_alt.tsv",
        "mir17_countnodes3_old_vs_consensustree_ref_minus_alt_full_premirna.png",
        "mir17_countnodes3_old_vs_consensustree_ref_minus_alt_full_premirna.pdf",
        "let7_countnodes3_old_vs_consensustree_ref_minus_alt_full_premirna.png",
        "let7_countnodes3_old_vs_consensustree_ref_minus_alt_full_premirna.pdf",
    ]
    missing = [name for name in expected_files if not (OUTPUT_DIR / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing outputs: {missing}")
    remaining_superseded = [
        name for name in SUPERSEDED_OUTPUTS if (OUTPUT_DIR / name).exists()
    ]
    if remaining_superseded:
        raise RuntimeError(f"Superseded outputs remain: {remaining_superseded}")
    if len({row.run_id for row in score_rows}) != len(score_rows):
        raise ValueError("Every score row must have a unique run ID")
    if any(
        not row.run_id
        or not row.legacy_category
        or not row.legacy_model_name
        or not row.model_explanation
        for row in score_rows
    ):
        raise ValueError("Every score row must contain complete presentation metadata")
    if Counter(row.presentation_status for row in score_rows) != {
        "primary": 48,
        "diagnostic": 1,
        "duplicate": 1,
    }:
        raise ValueError("Unexpected presentation-status counts")
    if any(
        not math.isfinite(row.test_auprc)
        or not math.isfinite(row.leftout_auprc)
        or not 0 <= row.test_auprc <= 1
        or not 0 <= row.leftout_auprc <= 1
        for row in score_rows
    ):
        raise ValueError("All AUPRC values must be finite and in [0, 1]")
    expected_profile_rows = {
        "mir17_let7_countnodes3_ref_minus_alt_full_premirna.tsv": 127,
        "mir17_let7_countnodes3_ref_minus_alt_mature.tsv": 45,
        "mir17_let7_countnodes3_old_vs_consensustree_ref_minus_alt.tsv": 127,
    }
    for filename, expected_rows in expected_profile_rows.items():
        with (OUTPUT_DIR / filename).open() as handle:
            actual_rows = sum(1 for _ in handle) - 1
        if actual_rows != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} data rows in {filename}, found {actual_rows}"
            )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    score_rows = collect_model_scores()
    write_score_tables(score_rows)
    write_model_performance_plot(score_rows)
    write_mir17_let7_outputs()
    write_old_vs_new_countnodes3_outputs()
    remove_superseded_outputs()
    write_readme(score_rows)
    validate_outputs(score_rows)
    print(f"Wrote {len(score_rows)} model rows to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
