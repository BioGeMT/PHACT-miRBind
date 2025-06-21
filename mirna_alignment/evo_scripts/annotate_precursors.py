#!/usr/bin/env python3
import os
import re
import subprocess
from Bio import SeqIO

def run_rnafold(sequence):
    cmd = f"echo '{sequence}' | RNAfold --noPS 2> /dev/null"
    result = subprocess.check_output(cmd, shell=True, text=True)
    structure = result.split('\n')[1].split()[0]
    if len(structure) != len(sequence):
        structure = '.' * len(sequence)
    return structure

def parse_mature_sequences(mature_file):
    mature_db = {}
    for record in SeqIO.parse(mature_file, "fasta"):
        header = record.description
        seq = str(record.seq)
        
        if '_5p' in header:
            arm = '5p'
            base_id = header.replace('_5p*', '').replace('_5p', '').replace('>', '')
        elif '_3p' in header:
            arm = '3p'
            base_id = header.replace('_3p*', '').replace('_3p', '').replace('>', '')
        else:
            continue
            
        parts = base_id.split('-', 1)
        if len(parts) != 2:
            continue
            
        species, precursor_id = parts[0], parts[1]
        
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
        return chr(ord('a') + (num - 10))

def process_precursors(input_dir, output_dir, mature_db):
    os.makedirs(output_dir, exist_ok=True)
    
    processed_files = set()
    
    for file_idx, fasta_file in enumerate(os.listdir(input_dir), 1):
        if not fasta_file.endswith(('.fa', '.fasta')):
            continue
            
        input_path = os.path.join(input_dir, fasta_file)
        if not os.path.isfile(input_path):
            continue
        
        base_name = os.path.splitext(fasta_file)[0]
        output_filename = f"{base_name}_annotated.fa"
        
        counter = 1
        while output_filename in processed_files:
            output_filename = f"{base_name}_{counter}_annotated.fa"
            counter += 1
        processed_files.add(output_filename)
        
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w') as fout:
            for rec in SeqIO.parse(input_path, 'fasta'):
                original_sequence = str(rec.seq).upper()
                full_sequence = original_sequence
                header = rec.description
                seq_len = len(full_sequence)
                
                parts = header.split('-', 1)
                if len(parts) != 2:
                    continue
                species, precursor_id = parts[0], parts[1].split()[0]
                
                if precursor_id.endswith('_pri'):
                    precursor_id = precursor_id[:-4]
                
                lookup_key = (species, precursor_id)
                mature = mature_db.get(lookup_key, {'5p': None, '3p': None})
                seq5p = mature['5p']
                seq3p = mature['3p']
                
                structure = run_rnafold(full_sequence)
                
                anchors = ['.' for _ in range(seq_len)]
                ids = ['.' for _ in range(seq_len)]
                
                if seq5p:
                    counter_5p = 1
                    start = full_sequence.find(seq5p)
                    if start != -1:
                        for j in range(start, start + len(seq5p)):
                            if j < seq_len:
                                anchors[j] = 'A'
                                ids[j] = get_id_char(counter_5p)
                                counter_5p += 1
                
                if seq3p:
                    counter_3p = 1
                    start = full_sequence.find(seq3p)
                    if start != -1:
                        for j in range(start, start + len(seq3p)):
                            if j < seq_len:
                                anchors[j] = 'B'
                                ids[j] = get_id_char(counter_3p)
                                counter_3p += 1
                
                anchor_str = ''.join(anchors)
                id_str = ''.join(ids)
                
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
    parser.add_argument('--input', required=True, help='Input directory with precursor FASTA files')
    parser.add_argument('--output', required=True, help='Output directory for annotated precursors')
    parser.add_argument('--mature', required=True, help='FASTA file with mature 5p/3p sequences')
    args = parser.parse_args()

    mature_db = parse_mature_sequences(args.mature)
    process_precursors(args.input, args.output, mature_db)