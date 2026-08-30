# PHACT consensus parameter-1 models for miRBench v7

This directory contains the exact source and reproducibility workflow for the
five PHACT-P1 models submitted to the miRBench v7 leaderboard. All five use the
new consensus-tree parameter-1 scores on the mature-miRNA axis; these retain
position-specific signal instead of the flat mature profiles seen in the older
score export.

The recorded leaderboard results are:

| Model | Representation and architecture | Test AUPRC | Left-out AUPRC |
| --- | --- | ---: | ---: |
| PHACT-P1 Multi-Candidate Fusion | Multimodal-fusion features, RiNALMo layers 3/6/9/12, and attention over up to three mature candidates | 0.889978 | 0.860845 |
| PHACT-P1 Multimodal Fusion | miRBind2 and reverse-complement pair grids, PHACT-P1, conservation, metadata, and frozen RiNALMo embeddings | 0.889904 | 0.862177 |
| PHACT-P1 miRNA + Target + Conservation CNN | Pair grid + 4 miRNA P1 + 4 target PHACT + target phyloP/phastCons channels | 0.887023 | 0.866165 |
| PHACT-P1 miRNA + Target CNN | Pair grid + 4 mature-miRNA P1 + 4 target PHACT channels | 0.885963 | 0.864901 |
| PHACT-P1 miRNA CNN | Pair grid + 4 mature-miRNA P1 channels | 0.877736 | 0.870263 |

The scores above are calculated on the standard 324,171-row test set and
20,054-row left-out set. AUPRC is calculated from the positive-class
probability, not a thresholded class prediction.

## Environment

The three CNNs use the root project environment:

```bash
git clone https://github.com/BioGeMT/PHACT-miRBind.git
cd PHACT-miRBind
git checkout mirbench-v7-phact-param1-2026-08
uv sync
```

The two Agentomics models use the environment recorded in
[`agentomics/model5/environment.yml`](agentomics/model5/environment.yml). They
require a CUDA GPU for practical full-data training. Their downloadable weight
archives include the frozen RiNALMo model required for offline inference.

## Data preparation

The workflow expects the miRBench v7 Manakov-format train, test, and left-out
TSVs, the established target-position PHACT score table, the coordinate
template for the mature miRNAs, and the consensus-tree output archive.

Extract the consensus parameter-1 scores and convert them to the two public
representations used by the models:

```bash
uv run python mirbench_v7_param1/scripts/extract_consensus_param1_arm_scores.py \
  --archive CONSENSUS_TREE_OUTPUT.tar.gz \
  --coordinate-template MIRNA_COORDINATES.tsv \
  --output work/consensus_param1_arm_scores.tsv \
  --summary work/consensus_param1_arm_scores.summary.json

uv run python mirbench_v7_param1/scripts/build_param1_profile_table.py \
  --input work/consensus_param1_arm_scores.tsv \
  --output work/phact_mirna_positions_param_1.tsv

uv run python scripts/build_mirna_phact_row_scores.py \
  --train-file AGO2_eCLIP_Manakov2022_train.tsv \
  --test-file AGO2_eCLIP_Manakov2022_test.tsv \
  --leftout-file AGO2_eCLIP_Manakov2022_leftout.tsv \
  --phact-mirna-scores work/consensus_param1_arm_scores.tsv \
  --output-file work/phact_mirna_param1_row_scores.tsv \
  --summary-file work/phact_mirna_param1_row_scores.summary.json
```

Build one PHACT cache per split. The following test-set command is the template;
repeat it with the corresponding input, prefix, and `--phact-split` for train
and left-out (the held-out validation cache is split from train using the same
split definition as the release run):

```bash
uv run build-phact-cache \
  --input-file AGO2_eCLIP_Manakov2022_test.tsv \
  --output-dir work/cache/param_1_target_score/test \
  --output-prefix test \
  --phact-split test \
  --phact-models param_1 \
  --target-phact-models target_score \
  --mirna-phact-file work/phact_mirna_param1_row_scores.tsv \
  --target-phact-file TARGET_PHACT_ROW_SCORES.tsv \
  --phact-reduction nucleotide
```

The conservation model uses the same cache with phyloP and phastCons appended
by [`../scripts/append_conservation_to_phact_cache.py`](../scripts/append_conservation_to_phact_cache.py).
Its conflict-cleaned training inputs are produced by
[`../scripts/build_conflict_cleaned_phact_caches.py`](../scripts/build_conflict_cleaned_phact_caches.py)
and [`../scripts/extract_conflict_replacement_rows.py`](../scripts/extract_conflict_replacement_rows.py).

## Train the three CNNs

The exact commands and order are captured in
[`run_cnn_models.sh`](run_cnn_models.sh):

```bash
uv run bash mirbench_v7_param1/run_cnn_models.sh work/cache work/outputs
```

Models 1 and 2 are trained directly. Model 3 widens Model 2 with the two
conservation channels, performs a short clean-data stage, and then fine-tunes
for two epochs with conflict replacements, one-position target shifting, and
focal loss (`gamma=1`).

## Train the two Agentomics models

Prepare a deterministic miRNA-grouped 80/20 train/validation split:

```bash
python mirbench_v7_param1/agentomics/model5/create_split.py \
  --source AGENTOMICS_TRAIN_ROOT \
  --output work/agentomics_splits \
  --param1-profile work/phact_mirna_positions_param_1.tsv
```

Each test input directory must contain `samples.tsv`,
`sample_mirna_candidates.tsv`, `phact_mirna_positions.tsv`,
`phact_target_positions.tsv`, and `mirgenedb_premirna_orthologues.tsv`; its
sibling `labels.csv` contains `id,label` and is used only for evaluation.

Train Model 5 first, then its layer-mix intermediate and Model 4. The script
performs that dependency order and evaluates both final models:

```bash
bash mirbench_v7_param1/run_agentomics_models.sh \
  work/agentomics_splits \
  work/agentomics_eval \
  work/outputs \
  MIRBIND2_INITIALIZATION_CHECKPOINT.pt
```

Model 5 chooses one mature candidate with an available PHACT profile. Model 4
extends it with layer-mixed RiNALMo features and explicit multi-candidate
ambiguity modelling. Neither leaderboard result is an ensemble.

## Inference and release validation

For the CNNs, export predictions from a checkpoint and cache with:

```bash
uv run python scripts/export_cached_phact_predictions.py \
  --checkpoint MODEL.pt \
  --cache CACHE_DIR \
  --output predictions.npz \
  --device cuda
```

For Agentomics inference, unpack a weight archive and run its corresponding
`model_inference/inference.py` (Model 5) or `multi_candidate/inference.py`
(Model 4), passing the unpacked directory as `--artifacts-dir`.

The release builder converts every result to the leaderboard schema
`id,prediction`, recalculates all metrics, checks the expected AUPRCs, packages
the five weight artifacts, and writes SHA-256 checksums:

```bash
uv run python mirbench_v7_param1/scripts/prepare_release_artifacts.py \
  --run-root work \
  --cnn-predictions work/cnn_prediction_npz \
  --output work/release

cd work/release
sha256sum -c SHA256SUMS
```

Only the five final model artifacts and their test/left-out predictions belong
in the leaderboard release. Training caches, validation predictions, temporary
RiNALMo embedding caches, and intermediate checkpoints are intentionally not
published.
