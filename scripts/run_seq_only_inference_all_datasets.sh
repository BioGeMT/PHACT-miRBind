#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/dtzim01/PHACT-miRBind}
DATA_DIR=${DATA_DIR:-/home/dtzim01/manakov_datasets}
CHECKPOINT=${CHECKPOINT:-outputs/seq_only/pairwise_seq_model_20260629_201939.pt}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/seq_only_inference}
BATCH_SIZE=${BATCH_SIZE:-4096}
DEVICE=${DEVICE:-auto}

cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR"

run_prediction() {
  local dataset_name=$1
  local input_file=$2
  local output_file="$OUTPUT_DIR/${dataset_name}_mirbind2_predictions.tsv"

  echo "[$(date -Is)] start dataset=$dataset_name input=$input_file output=$output_file"
  uv run predict-seq-mirbind \
    --input-file "$input_file" \
    --output-file "$output_file" \
    --checkpoint "$CHECKPOINT" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --force
  echo "[$(date -Is)] done dataset=$dataset_name output=$output_file"
}

run_prediction \
  AGO2_eCLIP_Manakov2022_train \
  "$DATA_DIR/AGO2_eCLIP_Manakov2022_train.tsv"

run_prediction \
  AGO2_eCLIP_Manakov2022_test \
  "$DATA_DIR/AGO2_eCLIP_Manakov2022_test.tsv"

run_prediction \
  AGO2_eCLIP_Manakov2022_leftout \
  "$DATA_DIR/AGO2_eCLIP_Manakov2022_leftout.tsv"

run_prediction \
  AGO2_CLASH_Hejret2023_train \
  "$DATA_DIR/AGO2_CLASH_Hejret2023_train.tsv"

run_prediction \
  AGO2_CLASH_Hejret2023_test \
  "$DATA_DIR/AGO2_CLASH_Hejret2023_test.tsv"

run_prediction \
  AGO2_eCLIP_Klimentova2022_test \
  "$DATA_DIR/AGO2_eCLIP_Klimentova2022_test.tsv"

echo "[$(date -Is)] all seq-only inference complete output_dir=$OUTPUT_DIR"
