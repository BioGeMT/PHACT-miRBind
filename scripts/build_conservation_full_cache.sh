#!/usr/bin/env bash
set -euo pipefail

cd /home/dtzim01/PHACT-miRBind

SOURCE_DATA=${SOURCE_DATA:-/home/dtzim01/manakov_datasets}
REPO_DATA=/home/dtzim01/PHACT-miRBind/data
SPLIT_DIR=${SPLIT_DIR:-"$REPO_DATA/presplit_conservation_original_rows"}
CACHE_DIR=${CACHE_DIR:-"$REPO_DATA/conservation_cache_full"}
FEATURES=phylop,phastcons

TRAIN_SOURCE=${TRAIN_SOURCE:-"$SOURCE_DATA/AGO2_eCLIP_Manakov2022_train.tsv"}
TEST_SOURCE=${TEST_SOURCE:-"$SOURCE_DATA/AGO2_eCLIP_Manakov2022_test.tsv"}
LEFTOUT_SOURCE=${LEFTOUT_SOURCE:-"$SOURCE_DATA/AGO2_eCLIP_Manakov2022_leftout.tsv"}

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

echo "cache_dir=$CACHE_DIR"
