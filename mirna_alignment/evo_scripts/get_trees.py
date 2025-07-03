#!/usr/bin/env python3

import os
import shutil
import argparse

def process_trees(input_dir, output_dir):
    """
    Extract .treefile files from tree directories and copy them to output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    copied = 0
    total_files = 0
    
    # Get all .treefile files in input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.treefile'):
            total_files += 1
            src_path = os.path.join(input_dir, filename)
            
            # Extract miRNA name (remove _annotated.treefile suffix)
            if filename.endswith('_annotated.treefile'):
                base_name = filename[:-19]  # Remove '_annotated.treefile'
            else:
                base_name = filename[:-9]   # Remove '.treefile'
            
            # Create output filename
            output_filename = f"{base_name}.treefile"
            dest_path = os.path.join(output_dir, output_filename)
            
            # Copy file
            shutil.copy2(src_path, dest_path)
            copied += 1
    
    return copied, total_files

def main():
    parser = argparse.ArgumentParser(description="Extract .treefile files from tree directories")
    parser.add_argument('--input', dest='input_dir', required=True, 
                       help="Input directory with .treefile files")
    parser.add_argument('--output', dest='output_dir', required=True, 
                       help="Output directory for processed trees")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    copied, total = process_trees(args.input_dir, args.output_dir)
    print(f"Processed {copied}/{total} tree files")

if __name__ == "__main__":
    main()