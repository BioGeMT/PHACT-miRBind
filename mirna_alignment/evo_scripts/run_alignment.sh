#!/bin/bash

# Check if correct number of arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <input_directory> <output_directory> <num_processes>"
    echo "Example: $0 evo_results/annotated_precursors evo_results/annotated_precursors_aligned 64"
    exit 1
fi

# Get arguments
INPUT_DIR="$1"
OUTPUT_PARENT_DIR="$2"
PARALLEL_PROCESSES="$3"

mkdir -p "$OUTPUT_PARENT_DIR"

# Check if the input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory '$INPUT_DIR' does not exist!"
    exit 1
fi

# Count the number of FASTA files
FASTA_COUNT=$(find "$INPUT_DIR" -name "*.fa" | wc -l | tr -d '[:space:]')
if [ "$FASTA_COUNT" -eq 0 ]; then
    echo "ERROR: No .fa files found in '$INPUT_DIR'"
    exit 1
fi

echo "Found $FASTA_COUNT FASTA files to process"
echo "Starting parallel MLocARNA processing with $PARALLEL_PROCESSES processes"

# Create a temporary file containing the list of FASTA files
TMP_FILE=$(mktemp)
find "$INPUT_DIR" -name "*.fa" > "$TMP_FILE"

# Process function to handle each file
process_file() {
    local FASTA_FILE="$1"
    local OUTPUT_DIR="$2"
    
    FILENAME=$(basename -- "$FASTA_FILE")
    BASENAME="${FILENAME%.*}"
    FILE_OUTPUT_DIR="$OUTPUT_DIR/$BASENAME"

    echo "=================================================="
    echo "Processing $FILENAME..."
    echo "Started at $(date)"
    
    # Create output directory for this file
    mkdir -p "$FILE_OUTPUT_DIR"

    # Run MLocARNA and suppress the nested "results" directory
    mlocarna "$FASTA_FILE" \
        --tgtdir="$FILE_OUTPUT_DIR" \
        --threads=1 \
        --struct-weight=300 \
        --consensus-structure=alifold \
        --plfold-span=150 \
        --write-structure \
        --stockholm \
        --alifold-consensus-dp \
        --free-endgaps \
        --indel=-150 \
        --indel-opening=-750 

    # Move alignment files from "results/" to the parent output directory
    mv "$FILE_OUTPUT_DIR"/results/* "$FILE_OUTPUT_DIR"/ 2>/dev/null

    # Clean up empty "results/" directory
    rmdir "$FILE_OUTPUT_DIR"/results 2>/dev/null

    echo "Completed $FILENAME at $(date)"
    echo "=================================================="
}

# Export the process function and output directory
export -f process_file
export OUTPUT_PARENT_DIR

# Process each file using a Mac-compatible parallel approach
cat "$TMP_FILE" | while read -r FASTA_FILE; do
    (
        process_file "$FASTA_FILE" "$OUTPUT_PARENT_DIR"
    ) &
    
    # Limit the number of parallel processes
    while [ $(jobs -p | wc -l | tr -d '[:space:]') -ge $PARALLEL_PROCESSES ]; do
        sleep 1
    done
done

# Wait for all background jobs to complete
wait

# Clean up temporary file
rm -f "$TMP_FILE"

echo "All files processed successfully!"
echo "Results are available in $OUTPUT_PARENT_DIR"