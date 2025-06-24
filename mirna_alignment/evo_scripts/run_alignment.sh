#!/bin/bash

INPUT_DIR="$1"
OUTPUT_PARENT_DIR="$2"
PARALLEL_PROCESSES="$3"
CUSTOM_PARAMS="$4"

mkdir -p "$OUTPUT_PARENT_DIR" 2>/dev/null

TMP_FILE=$(mktemp)
find "$INPUT_DIR" -name "*.fa" > "$TMP_FILE"

process_file() {
    local FASTA_FILE="$1"
    local OUTPUT_DIR="$2"
    local PARAMS="$3"
    
    FILENAME=$(basename -- "$FASTA_FILE")
    BASENAME="${FILENAME%.*}"
    FILE_OUTPUT_DIR="$OUTPUT_DIR/$BASENAME"
    
    mkdir -p "$FILE_OUTPUT_DIR" 2>/dev/null

    # Base mlocarna command with essential parameters
    MLOCARNA_CMD="mlocarna \"$FASTA_FILE\" --tgtdir=\"$FILE_OUTPUT_DIR\" --threads=1 --write-structure --stockholm  --alifold-consensus-dp --plfold-span=150 --width 1000"
    
    # Add custom parameters if provided
    if [ -n "$PARAMS" ]; then
        MLOCARNA_CMD="$MLOCARNA_CMD $PARAMS"
    fi
    
    eval $MLOCARNA_CMD >/dev/null 2>&1
}

export -f process_file
export OUTPUT_PARENT_DIR
export CUSTOM_PARAMS

# Process files in parallel
cat "$TMP_FILE" | while read -r FASTA_FILE; do
    (
        process_file "$FASTA_FILE" "$OUTPUT_PARENT_DIR" "$CUSTOM_PARAMS"
    ) &
    
    # Control number of parallel processes
    while [ $(jobs -p | wc -l | tr -d '[:space:]') -ge $PARALLEL_PROCESSES ]; do
        sleep 1
    done
done

wait
rm -f "$TMP_FILE"