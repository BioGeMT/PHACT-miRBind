#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 SPLITS_ROOT EVAL_ROOT OUTPUT_ROOT MIRBIND2_CHECKPOINT" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SPLITS_ROOT=$1
EVAL_ROOT=$2
OUTPUT_ROOT=$3
MIRBIND2_CHECKPOINT=$4
MODEL5_CODE="$SCRIPT_DIR/agentomics/model5"
MODEL4_CODE="$SCRIPT_DIR/agentomics/model4"
MODEL5_OUT="$OUTPUT_ROOT/param_1_agentomics_model_5"
LAYER_OUT="$OUTPUT_ROOT/param_1_agentomics_layer_mix_intermediate"
MODEL4_OUT="$OUTPUT_ROOT/param_1_agentomics_model_4"

python "$MODEL5_CODE/model_training/train.py" \
  --train-data "$SPLITS_ROOT/train" \
  --validation-data "$SPLITS_ROOT/validation" \
  --artifacts-dir "$MODEL5_OUT" \
  --mirbind2-checkpoint "$MIRBIND2_CHECKPOINT"

mkdir -p "$MODEL5_OUT/predictions"
for split in test leftout; do
  python "$MODEL5_CODE/model_inference/inference.py" \
    --input "$EVAL_ROOT/$split/input" \
    --output "$MODEL5_OUT/predictions/${split}_predictions.csv" \
    --artifacts-dir "$MODEL5_OUT"
done
python "$SCRIPT_DIR/scripts/evaluate_agentomics_predictions.py" \
  --model-name param_1_agentomics_model_5 \
  --predictions-dir "$MODEL5_OUT/predictions" \
  --test-labels "$EVAL_ROOT/test/labels.csv" \
  --leftout-labels "$EVAL_ROOT/leftout/labels.csv" \
  --output "$MODEL5_OUT/final_evaluation.json"

python "$MODEL4_CODE/layer_mix_training/train.py" \
  --train-data "$SPLITS_ROOT/train" \
  --validation-data "$SPLITS_ROOT/validation" \
  --initial-artifacts "$MODEL5_OUT" \
  --artifacts-dir "$LAYER_OUT"

python "$MODEL4_CODE/multi_candidate/train.py" \
  --train-data "$SPLITS_ROOT/train" \
  --validation-data "$SPLITS_ROOT/validation" \
  --initial-artifacts "$LAYER_OUT" \
  --artifacts-dir "$MODEL4_OUT"

mkdir -p "$MODEL4_OUT/predictions"
for split in test leftout; do
  python "$MODEL4_CODE/multi_candidate/inference.py" \
    --input "$EVAL_ROOT/$split/input" \
    --output "$MODEL4_OUT/predictions/${split}_predictions.csv" \
    --artifacts-dir "$MODEL4_OUT"
done
python "$SCRIPT_DIR/scripts/evaluate_agentomics_predictions.py" \
  --model-name param_1_agentomics_model_4 \
  --predictions-dir "$MODEL4_OUT/predictions" \
  --test-labels "$EVAL_ROOT/test/labels.csv" \
  --leftout-labels "$EVAL_ROOT/leftout/labels.csv" \
  --output "$MODEL4_OUT/final_evaluation.json"
