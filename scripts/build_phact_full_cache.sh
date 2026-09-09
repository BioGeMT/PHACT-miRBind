#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
: "${PHACT_WORKSPACE:?Set PHACT_WORKSPACE to the data workspace}"

SOURCE_DATA=${SOURCE_DATA:-${PHACT_WORKSPACE}/data/inputs/manakov_datasets}
REPO_DATA=${PHACT_WORKSPACE}/data/caches
PHACT_MODELS=${PHACT_MODELS:-${PHACT_MODEL:-CountNodes_2}}
PHACT_REDUCTION=${PHACT_REDUCTION:-nucleotide}
SAFE_MODELS=${PHACT_MODELS//[^a-zA-Z0-9_]/_}
SAFE_TARGET_MODELS=${TARGET_PHACT_MODELS:-target_auto}
SAFE_TARGET_MODELS=${SAFE_TARGET_MODELS//[^a-zA-Z0-9_]/_}
SAFE_REDUCTION=${PHACT_REDUCTION//[^a-zA-Z0-9_]/_}
SPLIT_DIR=${SPLIT_DIR:-"$PHACT_WORKSPACE/data/inputs/manakov_original_rows"}
if [[ "$PHACT_MODELS" != *,* && -z "${TARGET_PHACT_MODELS:-}" ]]; then
  DEFAULT_CACHE_DIR="$REPO_DATA/phact_cache_${SAFE_MODELS}"
else
  DEFAULT_CACHE_DIR="$REPO_DATA/phact_cache_${SAFE_MODELS}_target_${SAFE_TARGET_MODELS}"
fi
if [[ "$PHACT_REDUCTION" != "nucleotide" ]]; then
  DEFAULT_CACHE_DIR="${DEFAULT_CACHE_DIR}_${SAFE_REDUCTION}"
fi
CACHE_DIR=${CACHE_DIR:-"$DEFAULT_CACHE_DIR"}
MIRNA_PHACT_FILE=${MIRNA_PHACT_FILE:-${PHACT_WORKSPACE}/data/scores/main_repo/phact_mirna_manakov_position_qntnorm_transformed_scores.tsv}
TARGET_PHACT_FILE=${TARGET_PHACT_FILE:-${PHACT_WORKSPACE}/data/scores/main_repo/phact_target_manakov_position_qntnorm_transformed_scores.tsv}

TRAIN_SOURCE=${TRAIN_SOURCE:-"$SOURCE_DATA/AGO2_eCLIP_Manakov2022_train.tsv"}
TEST_SOURCE=${TEST_SOURCE:-"$SOURCE_DATA/AGO2_eCLIP_Manakov2022_test.tsv"}
LEFTOUT_SOURCE=${LEFTOUT_SOURCE:-"$SOURCE_DATA/AGO2_eCLIP_Manakov2022_leftout.tsv"}

TARGET_MODEL_ARGS=()
if [[ -n "${TARGET_PHACT_MODELS:-}" ]]; then
  TARGET_MODEL_ARGS=(--target-phact-models "$TARGET_PHACT_MODELS")
fi

if [[ ! -f "$SPLIT_DIR/manakov_original_rows_summary.json" ]]; then
  uv run split-original-rows \
    --input-file "$TRAIN_SOURCE" \
    --output-dir "$SPLIT_DIR" \
    --output-prefix manakov_original_rows \
    --val-fraction 0.1 \
    --seed 42 \
    --include-row-id \
    --row-id-column manakov_row_id
fi

if [[ ! -f "$CACHE_DIR/train/train_manifest.json" ]]; then
  uv run build-phact-cache \
    --input-file "$SPLIT_DIR/manakov_original_rows_train.tsv" \
    --output-dir "$CACHE_DIR/train" \
    --output-prefix train \
    --phact-split train \
    --phact-models "$PHACT_MODELS" \
    "${TARGET_MODEL_ARGS[@]}" \
    --phact-reduction "$PHACT_REDUCTION" \
    --mirna-phact-file "$MIRNA_PHACT_FILE" \
    --target-phact-file "$TARGET_PHACT_FILE"
fi

if [[ ! -f "$CACHE_DIR/val/val_manifest.json" ]]; then
  uv run build-phact-cache \
    --input-file "$SPLIT_DIR/manakov_original_rows_val.tsv" \
    --output-dir "$CACHE_DIR/val" \
    --output-prefix val \
    --phact-split train \
    --phact-models "$PHACT_MODELS" \
    "${TARGET_MODEL_ARGS[@]}" \
    --phact-reduction "$PHACT_REDUCTION" \
    --mirna-phact-file "$MIRNA_PHACT_FILE" \
    --target-phact-file "$TARGET_PHACT_FILE"
fi

if [[ ! -f "$CACHE_DIR/test/test_manifest.json" ]]; then
  uv run build-phact-cache \
    --input-file "$TEST_SOURCE" \
    --output-dir "$CACHE_DIR/test" \
    --output-prefix test \
    --phact-split test \
    --phact-models "$PHACT_MODELS" \
    "${TARGET_MODEL_ARGS[@]}" \
    --phact-reduction "$PHACT_REDUCTION" \
    --mirna-phact-file "$MIRNA_PHACT_FILE" \
    --target-phact-file "$TARGET_PHACT_FILE"
fi

if [[ ! -f "$CACHE_DIR/leftout/leftout_manifest.json" ]]; then
  uv run build-phact-cache \
    --input-file "$LEFTOUT_SOURCE" \
    --output-dir "$CACHE_DIR/leftout" \
    --output-prefix leftout \
    --phact-split leftout \
    --phact-models "$PHACT_MODELS" \
    "${TARGET_MODEL_ARGS[@]}" \
    --phact-reduction "$PHACT_REDUCTION" \
    --mirna-phact-file "$MIRNA_PHACT_FILE" \
    --target-phact-file "$TARGET_PHACT_FILE"
fi

echo "cache_dir=$CACHE_DIR"
