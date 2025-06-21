#!/usr/bin/env python3

import os
import shutil
import argparse

def count_sequences(aln_file):
    try:
        sequence_names = set()
        with open(aln_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('CLUSTAL') or line.startswith('*') or line.startswith(':') or line.startswith('.'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    seq_name = parts[0]
                    sequence_names.add(seq_name)
        
        return len(sequence_names)
    except Exception:
        return 0

def process_alignments(input_dir, output_dir, min_sequences=10, clean_comments=True):
    os.makedirs(output_dir, exist_ok=True)
    
    subfolders = [f.path for f in os.scandir(input_dir) if f.is_dir()]
    
    copied = 0
    total_folders = len(subfolders)
    
    for subfolder in subfolders:
        subfolder_name = os.path.basename(subfolder)
        aln_files = [f for f in os.listdir(subfolder) if f.endswith('.aln')]
        
        if not aln_files:
            continue
        
        selected_file = "result.aln" if "result.aln" in aln_files else aln_files[0]
        source_path = os.path.join(subfolder, selected_file)
        
        if not os.path.exists(source_path):
            continue
        
        seq_count = count_sequences(source_path)
        
        if seq_count >= min_sequences:
            output_filename = f"{subfolder_name}.aln"
            dest_path = os.path.join(output_dir, output_filename)
            
            if clean_comments:
                with open(source_path, 'r') as src, open(dest_path, 'w') as dst:
                    for line in src:
                        if not line.startswith('#A'):
                            dst.write(line)
            else:
                shutil.copy2(source_path, dest_path)
            
            copied += 1
    
    return copied, total_folders

def main():
    parser = argparse.ArgumentParser(description="Process and filter .aln alignment files")
    parser.add_argument('--input', dest='input_dir', required=True, 
                       help="Input directory with alignment subfolders")
    parser.add_argument('--output', dest='output_dir', required=True, 
                       help="Output directory for filtered alignments")
    parser.add_argument('--min', type=int, default=10, 
                       help='Minimum number of sequences required (default: 10)')
    parser.add_argument('--keep-comments', action='store_true', 
                       help="Keep comment lines starting with #A")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        return
    
    process_alignments(
        args.input_dir, 
        args.output_dir,
        args.min,
        clean_comments=not args.keep_comments
    )

if __name__ == "__main__":
    main()