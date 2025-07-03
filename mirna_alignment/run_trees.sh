#!/bin/bash

echo "Building phylogenetic trees for miRNA alignments..."

# Number of parallel jobs
JOBS=32

# Input directories (4+ sequences minimum)
input_dirs=(
    "output/filtered_alignments_precursor_default_4seq"
    "output/filtered_alignments_precursor_high_4seq"
    "output/filtered_alignments_primary_default_4seq"
    "output/filtered_alignments_primary_high_4seq"
)

# Output directories
output_dirs=(
    "output/trees_precursor_default_4seq"
    "output/trees_precursor_high_4seq"
    "output/trees_primary_default_4seq"
    "output/trees_primary_high_4seq"
)

# Create output directories
for dir in "${output_dirs[@]}"; do
    mkdir -p "$dir"
done

# Process each input directory
for i in "${!input_dirs[@]}"; do
    input_dir="${input_dirs[$i]}"
    output_dir="${output_dirs[$i]}"
    
    echo "Processing $input_dir with $JOBS parallel jobs..."
    
    # Use xargs for parallel processing
    find "$input_dir" -name "*.aln" -print0 | \
    xargs -0 -I {} -P "$JOBS" bash -c '
        aln="$1"
        output_dir="$2"
        name=$(basename "$aln" .aln)
        echo "Building tree for $name..."
        iqtree -s "$aln" -m TEST -bb 1000 -nt 1 \
               --seqtype DNA --safe --seed 42 \
               --prefix "$output_dir/$name" \
               --quiet
    ' _ {} "$output_dir"
    
    echo "Completed $input_dir"
done

echo "All trees completed!"