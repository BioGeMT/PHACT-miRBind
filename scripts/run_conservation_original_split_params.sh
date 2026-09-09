#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
: "${PHACT_WORKSPACE:?Set PHACT_WORKSPACE to the data workspace}"

SOURCE_DATA=${SOURCE_DATA:-${PHACT_WORKSPACE}/datasets/original/manakov_datasets}
REPO_DATA=${PHACT_WORKSPACE}/models/caches
SPLIT_DIR="$PHACT_WORKSPACE/datasets/splits/presplit_conservation_original_rows"
CACHE_DIR="$REPO_DATA/conservation_cache_original_rows"
FEATURES=${CONSERVATION_FEATURES:-phylop,phastcons}

TRAIN_SOURCE=${TRAIN_SOURCE:-"$SOURCE_DATA/AGO2_eCLIP_Manakov2022_train.tsv"}
TEST_SOURCE=${TEST_SOURCE:-"$SOURCE_DATA/AGO2_eCLIP_Manakov2022_test.tsv"}
LEFTOUT_SOURCE=${LEFTOUT_SOURCE:-"$SOURCE_DATA/AGO2_eCLIP_Manakov2022_leftout.tsv"}

if [[ ! -f "$CACHE_DIR/train/train_manifest.json" || ! -f "$CACHE_DIR/val/val_manifest.json" ]]; then
  if [[ ! -f "$SPLIT_DIR/manakov_original_rows_summary.json" ]]; then
    uv run split-original-rows \
      --input-file "$TRAIN_SOURCE" \
      --output-dir "$SPLIT_DIR" \
      --output-prefix manakov_original_rows \
      --val-fraction 0.1 \
      --seed 42
  fi

  if [[ ! -f "$CACHE_DIR/train/train_manifest.json" ]]; then
    uv run build-conservation-cache \
      --input-file "$SPLIT_DIR/manakov_original_rows_train.tsv" \
      --output-dir "$CACHE_DIR/train" \
      --output-prefix train \
      --conservation-features "$FEATURES"
  fi

  if [[ ! -f "$CACHE_DIR/val/val_manifest.json" ]]; then
    uv run build-conservation-cache \
      --input-file "$SPLIT_DIR/manakov_original_rows_val.tsv" \
      --output-dir "$CACHE_DIR/val" \
      --output-prefix val \
      --conservation-features "$FEATURES"
  fi
fi

if [[ ! -f "$CACHE_DIR/test/test_manifest.json" ]]; then
  uv run build-conservation-cache \
    --input-file "$TEST_SOURCE" \
    --output-dir "$CACHE_DIR/test" \
    --output-prefix test \
    --conservation-features "$FEATURES"
fi

if [[ ! -f "$CACHE_DIR/leftout/leftout_manifest.json" ]]; then
  uv run build-conservation-cache \
    --input-file "$LEFTOUT_SOURCE" \
    --output-dir "$CACHE_DIR/leftout" \
    --output-prefix leftout \
    --conservation-features "$FEATURES"
fi

uv run train-conservation-mirbind \
  --train-cache "$CACHE_DIR/train" \
  --val-cache "$CACHE_DIR/val" \
  --test-cache "$CACHE_DIR/test" \
  --leftout-cache "$CACHE_DIR/leftout" \
  --output-dir "${PHACT_WORKSPACE}/models/new/conservation" \
  --conservation-features "$FEATURES" \
  --batch-size 256 \
  --num-epochs 50 \
  --patience 7 \
  --learning-rate 0.001 \
  --dropout-rate 0.2 \
  --embedding-dim 8 \
  --filter-sizes "128,64,32" \
  --kernel-sizes "6,3,3" \
  --cache-shuffle-mode global \
  --progress-every 1000
