#!/bin/bash

set -e

INPUT_TSV="$1"
BASE_NAME=$(basename "$INPUT_TSV" .tsv)
BED_FILE="${BASE_NAME}_genes.bed"
VALIDATION_RESULTS="${BASE_NAME}_validation.tsv"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_tsv>"
    echo "Example: $0 manakov_positives.tsv"
    exit 1
fi

if [ ! -f "$INPUT_TSV" ]; then
    echo "Error: Input file $INPUT_TSV not found"
    exit 1
fi

echo "=== Converting TSV to BED ==="
python get_gene_coordinates.py "$INPUT_TSV" "$BED_FILE"
echo "Created: $BED_FILE"

echo "=== Checking hg38 reference ==="
if [ ! -f "hg38.fa" ]; then
    echo "hg38.fa not found. Downloading..."
    wget http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
    echo "Extracting..."
    gunzip hg38.fa.gz
    echo "hg38.fa ready"
fi

echo "=== Local validation ==="
python validate_sequences.py "$BED_FILE" hg38.fa -o "$VALIDATION_RESULTS"
echo "Results saved to: $VALIDATION_RESULTS"

echo "=== API validation (100 sequences) ==="
python validate_sequences_api.py "$BED_FILE" -g hg38 -n 100

echo "=== Complete ==="