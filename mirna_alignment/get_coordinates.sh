#!/bin/bash

INPUT_TSV="$1"

if [ -z "$INPUT_TSV" ]; then
    echo "Usage: $0 <input.tsv>"
    exit 1
fi

# Create BED file (0-based)
tail -n +2 "$INPUT_TSV" | awk -F'\t' '{printf "%s\t%d\t%s\n", $7, int($8)-1, $9}' > coordinates.bed

# Create 1-based TSV
echo -e "chr\tstart\tend" > coordinates_1based.tsv
tail -n +2 "$INPUT_TSV" | awk -F'\t' '{printf "%s\t%d\t%s\n", $7, int($8), $9}' >> coordinates_1based.tsv

echo "Created: coordinates.bed (0-based) and coordinates_1based.tsv (1-based)"