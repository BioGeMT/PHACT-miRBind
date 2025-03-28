#!/usr/bin/env python3

import os
import sys
import shutil
import argparse
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment

def count_sequences(alignment_file):
    #Count the number of sequences in an alignment file.
    try:
        # Try different formats
        for format in ['fasta', 'phylip', 'clustal']:
            try:
                with open(alignment_file, 'r') as f:
                    alignment = AlignIO.read(f, format)
                    return len(alignment)
            except Exception:
                continue
                
        # If we couldn't parse with Biopython, try a simple grep for FASTA headers
        if os.path.getsize(alignment_file) > 0:
            with open(alignment_file, 'r') as f:
                content = f.read()
                if '>' in content:
                    # Count FASTA headers
                    return content.count('>')
        
        print(f"Warning: Could not determine sequence count for {alignment_file}")
        return 0
        
    except Exception as e:
        print(f"Error processing {alignment_file}: {e}")
        return 0

def filter_alignments(input_dir, output_dir, min_sequences=10):
  
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all alignment files
    alignment_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.aln'):
                alignment_files.append(os.path.join(root, file))
    
    total_files = len(alignment_files)
    print(f"Found {total_files} alignment files")
    
    # Process each alignment file
    filtered_count = 0
    for i, aln_file in enumerate(alignment_files, 1):
        base_name = os.path.basename(aln_file)
        
        # Count sequences
        seq_count = count_sequences(aln_file)
        
        # Status update
        print(f"Processing {i}/{total_files}: {base_name} - {seq_count} sequences", end="")
        
        # Filter alignments with at least min_sequences
        if seq_count >= min_sequences:
            # Copy the file to output directory
            output_file = os.path.join(output_dir, base_name)
            shutil.copy2(aln_file, output_file)
            filtered_count += 1
            print(" - COPIED")
        else:
            print(" - SKIPPED")
    
    print(f"\nFiltering complete!")
    print(f"Total alignment files: {total_files}")
    print(f"Filtered alignment files with {min_sequences}+ sequences: {filtered_count}")
    print(f"Filtered alignments saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter alignment files based on sequence count")
    parser.add_argument("input_dir", help="Directory containing alignment files")
    parser.add_argument("output_dir", help="Directory to store filtered alignment files")
    parser.add_argument("--min", type=int, default=10, help="Minimum number of sequences required (default: 10)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)
        
    filter_alignments(args.input_dir, args.output_dir, args.min)