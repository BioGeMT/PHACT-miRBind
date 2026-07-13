#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/dtzim01/PHACT-miRBind}
MANAKOV_DIR=${MANAKOV_DIR:-/home/dtzim01/manakov_datasets}

cd "$REPO_DIR"

scripts/append_mirgenedb_exact_annotation.py \
  --fasta mirna_alignment/miRNA_mature_files/hsa_mature.fas \
  --family-map mirna_alignment/miRNA_mature_files/mirgenedb_family_mappings.tsv \
  "$MANAKOV_DIR/AGO2_eCLIP_Manakov2022_train.tsv" \
  "$MANAKOV_DIR/AGO2_eCLIP_Manakov2022_test.tsv" \
  "$MANAKOV_DIR/AGO2_eCLIP_Manakov2022_leftout.tsv"

scripts/build_target_phact_row_scores.py --force
scripts/build_mirna_phact_row_scores.py --force
