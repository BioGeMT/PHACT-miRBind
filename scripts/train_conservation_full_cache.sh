#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
: "${PHACT_WORKSPACE:?Set PHACT_WORKSPACE to the data workspace}"

FEATURES=${CONSERVATION_FEATURES:-phylop,phastcons}
SAFE_FEATURES=${FEATURES//,/_}
CACHE_DIR=${CACHE_DIR:-${PHACT_WORKSPACE}/models/caches/conservation_cache_full}
OUTPUT_DIR=${OUTPUT_DIR:-${PHACT_WORKSPACE}/models/new/conservation_$SAFE_FEATURES}

uv run train-conservation-mirbind \
  --train-cache "$CACHE_DIR/train" \
  --val-cache "$CACHE_DIR/val" \
  --test-cache "$CACHE_DIR/test" \
  --leftout-cache "$CACHE_DIR/leftout" \
  --output-dir "$OUTPUT_DIR" \
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
