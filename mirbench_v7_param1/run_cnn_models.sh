#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 CACHE_ROOT OUTPUT_ROOT" >&2
  exit 2
fi

CACHE_ROOT=$1
OUTPUT_ROOT=$2
BASE_CACHE="$CACHE_ROOT/param_1_target_score"
CONS_CACHE="$CACHE_ROOT/param_1_target_score_conservation"

checkpoint_in() {
  local output_dir=$1
  local checkpoints=("$output_dir"/pairwise_phact_model_*.pt)
  if [[ ${#checkpoints[@]} -ne 1 || ! -f "${checkpoints[0]}" ]]; then
    echo "Expected exactly one checkpoint in $output_dir" >&2
    exit 1
  fi
  printf '%s\n' "${checkpoints[0]}"
}

train-phact-mirbind \
  --train-cache "$BASE_CACHE/train" \
  --val-cache "$BASE_CACHE/val" \
  --test-cache "$BASE_CACHE/test" \
  --leftout-cache "$BASE_CACHE/leftout" \
  --output-dir "$OUTPUT_ROOT/param_1_mirna_cnn" \
  --exclude-missingness \
  --phact-channel-mode mirna \
  --batch-size 256 --num-epochs 50 --patience 7 --learning-rate 0.001 \
  --dropout-rate 0.2 --embedding-dim 8 \
  --filter-sizes 128,64,32 --kernel-sizes 6,3,3 \
  --cache-shuffle-mode global --seed 42 --progress-every 1000
train-phact-mirbind \
  --train-cache "$BASE_CACHE/train" \
  --val-cache "$BASE_CACHE/val" \
  --test-cache "$BASE_CACHE/test" \
  --leftout-cache "$BASE_CACHE/leftout" \
  --output-dir "$OUTPUT_ROOT/param_1_mirna_target_cnn" \
  --exclude-missingness \
  --phact-channel-mode both \
  --batch-size 256 --num-epochs 50 --patience 7 --learning-rate 0.001 \
  --dropout-rate 0.2 --embedding-dim 8 \
  --filter-sizes 128,64,32 --kernel-sizes 6,3,3 \
  --cache-shuffle-mode global --seed 42 --progress-every 1000

MODEL2_CHECKPOINT=$(checkpoint_in "$OUTPUT_ROOT/param_1_mirna_target_cnn")

train-phact-mirbind \
  --train-cache "$CONS_CACHE/manakov_clean" \
  --additional-train-cache "$CONS_CACHE/gse_conflict_replacements" \
  --val-cache "$CONS_CACHE/val" \
  --test-cache "$CONS_CACHE/test" \
  --leftout-cache "$CONS_CACHE/leftout" \
  --output-dir "$OUTPUT_ROOT/param_1_conservation_clean_cnn" \
  --initial-checkpoint "$MODEL2_CHECKPOINT" \
  --initial-checkpoint-mode widen \
  --exclude-missingness \
  --phact-channel-mode both \
  --batch-size 512 --num-epochs 3 --patience 1 --learning-rate 0.0001 \
  --dropout-rate 0.2 --embedding-dim 8 \
  --filter-sizes 128,64,32 --kernel-sizes 6,3,3 \
  --cache-shuffle-mode global --seed 42 --progress-every 500

CLEAN_CHECKPOINT=$(checkpoint_in "$OUTPUT_ROOT/param_1_conservation_clean_cnn")

train-phact-mirbind \
  --train-cache "$CONS_CACHE/manakov_clean" \
  --additional-train-cache "$CONS_CACHE/gse_conflict_replacements" \
  --val-cache "$CONS_CACHE/val" \
  --test-cache "$CONS_CACHE/test" \
  --leftout-cache "$CONS_CACHE/leftout" \
  --output-dir "$OUTPUT_ROOT/param_1_conservation_conflict28_shift1_focal1_cnn" \
  --initial-checkpoint "$CLEAN_CHECKPOINT" \
  --initial-checkpoint-mode strict \
  --exclude-missingness \
  --phact-channel-mode both \
  --batch-size 256 --num-epochs 2 --patience 2 --learning-rate 0.00005 \
  --training-target-shift-max 1 --focal-gamma 1.0 \
  --dropout-rate 0.2 --embedding-dim 8 \
  --filter-sizes 128,64,32 --kernel-sizes 6,3,3 \
  --cache-shuffle-mode global --seed 42 --progress-every 1000
