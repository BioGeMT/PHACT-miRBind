#!/usr/bin/env python3
import os
import re
import subprocess
from Bio import SeqIO

def run_rnafold(sequence):
    #Predict secondary structure using RNAfold.
    try:
        cmd = f"echo '{sequence}' | RNAfold --noPS 2> /dev/null"
        result = subprocess.check_output(cmd, shell=True, text=True)
        structure = result.split('\n')[1].split()[0]
        if len(structure) != len(sequence):
            print(f"Warning: RNAfold structure length mismatch. Using dots.")
            structure = '.' * len(sequence)
        return structure
    except Exception as e:
        print(f"RNAfold failed: {str(e)}")
        return '.' * len(sequence)

def parse_mature_sequences(mature_file):
    mature_db = {}
    for record in SeqIO.parse(mature_file, "fasta"):
        header = record.description
        seq = str(record.seq)
        
        parts = header.split('_', 1)
        if len(parts) < 2:
            continue
            
        full_id, arm = parts[0], parts[1].split('_')[0]
        species, precursor_id = full_id.split('-', 1)
        
        arm = '5p' if '5p' in header else '3p' if '3p' in header else None
        if not arm:
            continue
            
        key = (species, precursor_id)
        if key not in mature_db:
            mature_db[key] = {'5p': None, '3p': None}
            
        if not mature_db[key][arm]:
            mature_db[key][arm] = seq
            
    return mature_db

def get_id_char(num):
    if num <= 9:
        return str(num)
    else:
        # Use letters for numbers > 9: a=10, b=11, etc.
        return chr(ord('a') + (num - 10))

def process_precursors(input_dir, output_dir, mature_db):
    os.makedirs(output_dir, exist_ok=True)
    
    #  delete "chunks" subfolder if it exists
    chunks_path = os.path.join(input_dir, "chunks")
    if os.path.exists(chunks_path) and os.path.isdir(chunks_path):
        print(f"Removing 'chunks' subfolder from input directory")
        try:
            import shutil
            shutil.rmtree(chunks_path)
        except Exception as e:
            print(f"Error removing 'chunks' subfolder: {str(e)}")
    
    processed_dirs = set()
    
    for dir_idx, precursor_dir in enumerate(os.listdir(input_dir), 1):
        dir_path = os.path.join(input_dir, precursor_dir)
        if not os.path.isdir(dir_path):
            continue
            
        print(f"Processing directory {dir_idx}: {precursor_dir}")
        
        output_filename = f"{precursor_dir}_precursor_annotated.fa"
        base, ext = os.path.splitext(output_filename)
        counter = 1
        while output_filename in processed_dirs:
            output_filename = f"{base}_{counter}{ext}"
            counter += 1
        processed_dirs.add(output_filename)
        
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w') as fout:
            for fasta_file in os.listdir(dir_path):
                if not fasta_file.endswith(('.fa', '.fasta')):
                    continue
                    
                input_path = os.path.join(dir_path, fasta_file)
                print(f"  Processing: {precursor_dir}/{fasta_file}")
                
                for rec in SeqIO.parse(input_path, 'fasta'):
                    full_sequence = str(rec.seq).upper()
                    header = rec.description
                    seq_len = len(full_sequence)
                    
                    parts = header.split('-', 1)
                    if len(parts) != 2:
                        continue
                    species, precursor_id = parts[0], parts[1].split()[0]
                    
                    mature = mature_db.get((species, precursor_id), 
                                         {'5p': None, '3p': None})
                    seq5p = mature['5p']
                    seq3p = mature['3p']
                    
                    # Get structure exactly matching sequence length
                    structure = run_rnafold(full_sequence)
                    
                    # Initialize annotation arrays with exact sequence length
                    anchors = ['.' for _ in range(seq_len)]
                    ids = ['.' for _ in range(seq_len)]
                    
                    # Process 5p region
                    if seq5p:
                        counter_5p = 1
                        start = full_sequence.find(seq5p)
                        if start != -1:
                            for j in range(start, start + len(seq5p)):
                                if j < seq_len:  # Safety check
                                    anchors[j] = 'A'
                                    ids[j] = get_id_char(counter_5p)
                                    counter_5p += 1
                    
                    # Process 3p region
                    if seq3p:
                        counter_3p = 1
                        start = full_sequence.find(seq3p)
                        if start != -1:
                            for j in range(start, start + len(seq3p)):
                                if j < seq_len:  # Safety check
                                    anchors[j] = 'B'
                                    ids[j] = get_id_char(counter_3p)
                                    counter_3p += 1
                    
                    # Convert to strings ensuring exact length
                    anchor_str = ''.join(anchors)
                    id_str = ''.join(ids)
                    
                    # Verify all lines have exact same length
                    assert len(full_sequence) == len(structure) == len(anchor_str) == len(id_str), \
                           f"Length mismatch for {header}: seq={len(full_sequence)}, struct={len(structure)}, anchor={len(anchor_str)}, id={len(id_str)}"
                    
                    # Write lines with exact length matching
                    lines = [
                        f">{header}",
                        full_sequence,
                        structure + " #S",
                        anchor_str + " #1",
                        id_str + " #2\n"
                    ]
                    fout.write('\n'.join(lines))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Annotate pre-miRNAs with RNAfold structure and mature regions')
    parser.add_argument('--input', required=True, help='Input directory with precursor FASTA directories')
    parser.add_argument('--output', required=True, help='Output directory for annotated precursors')
    parser.add_argument('--mature', required=True, help='FASTA file with mature 5p/3p sequences')
    args = parser.parse_args()

    try:
        subprocess.check_output(["RNAfold", "--version"])
    except FileNotFoundError:
        print("Error: Install ViennaRNA package first: 'conda install -c bioconda viennarna'")
        exit(1)

    print("Loading mature sequences...")
    mature_db = parse_mature_sequences(args.mature)
    
    print("\nProcessing precursors...")
    process_precursors(args.input, args.output, mature_db)
    
    print("\nAnnotation complete! Output directory:", args.output)