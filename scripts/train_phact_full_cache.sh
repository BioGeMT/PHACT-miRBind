#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
: "${PHACT_WORKSPACE:?Set PHACT_WORKSPACE to the data workspace}"

PHACT_MODELS=${PHACT_MODELS:-${PHACT_MODEL:-CountNodes_2}}
PHACT_REDUCTION=${PHACT_REDUCTION:-nucleotide}
PHACT_CHANNEL_MODE=${PHACT_CHANNEL_MODE:-both}
SAFE_MODELS=${PHACT_MODELS//[^a-zA-Z0-9_]/_}
SAFE_TARGET_MODELS=${TARGET_PHACT_MODELS:-target_auto}
SAFE_TARGET_MODELS=${SAFE_TARGET_MODELS//[^a-zA-Z0-9_]/_}
SAFE_REDUCTION=${PHACT_REDUCTION//[^a-zA-Z0-9_]/_}
if [[ "$PHACT_MODELS" != *,* && -z "${TARGET_PHACT_MODELS:-}" ]]; then
  DEFAULT_CACHE_DIR=${PHACT_WORKSPACE}/data/caches/phact_cache_${SAFE_MODELS}
  DEFAULT_OUTPUT_DIR=${PHACT_WORKSPACE}/runs/new/phact_${SAFE_MODELS}_${PHACT_CHANNEL_MODE}
else
  DEFAULT_CACHE_DIR=${PHACT_WORKSPACE}/data/caches/phact_cache_${SAFE_MODELS}_target_${SAFE_TARGET_MODELS}
  DEFAULT_OUTPUT_DIR=${PHACT_WORKSPACE}/runs/new/phact_${SAFE_MODELS}_target_${SAFE_TARGET_MODELS}_${PHACT_CHANNEL_MODE}
fi
if [[ "$PHACT_REDUCTION" != "nucleotide" ]]; then
  DEFAULT_CACHE_DIR="${DEFAULT_CACHE_DIR}_${SAFE_REDUCTION}"
  DEFAULT_OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}_${SAFE_REDUCTION}"
fi
CACHE_DIR=${CACHE_DIR:-"$DEFAULT_CACHE_DIR"}
OUTPUT_DIR=${OUTPUT_DIR:-"$DEFAULT_OUTPUT_DIR"}

uv run train-phact-mirbind \
  --train-cache "$CACHE_DIR/train" \
  --val-cache "$CACHE_DIR/val" \
  --test-cache "$CACHE_DIR/test" \
  --leftout-cache "$CACHE_DIR/leftout" \
  --output-dir "$OUTPUT_DIR" \
  --phact-channel-mode "$PHACT_CHANNEL_MODE" \
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
