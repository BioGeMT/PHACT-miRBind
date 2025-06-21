#!/usr/bin/env python3
import os
import argparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from tqdm import tqdm
import multiprocessing
import csv
import sys
from Bio import SeqIO
import re

def load_fasta_sequences(fasta_file):
    sequences = {}
    try:
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences[record.id] = str(record.seq)
    except Exception:
        pass
    return sequences

def get_page(url, retry_count=2):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    }
    
    for attempt in range(retry_count + 1):
        try:
            time.sleep(random.uniform(0.2, 1.0))
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.text
            if response.status_code == 429:
                time.sleep((attempt + 1) * random.uniform(5, 10))
            elif attempt < retry_count:
                time.sleep(random.uniform(1, 3))
        except Exception:
            if attempt < retry_count:
                time.sleep(random.uniform(1, 3))
    return None

def parse_orthologues(html):
    soup = BeautifulSoup(html, "html.parser")
    orthologues = []
    for tr in soup.find_all("tr"):
        th = tr.find("th")
        if th and "Orthologues" in th.get_text():
            td = tr.find("td")
            if td:
                for a in td.find_all("a"):
                    text = a.get_text().strip()
                    if text:
                        orthologues.append(text)
            break
    return orthologues

def construct_url(mir_id):
    parts = mir_id.split("-")
    if len(parts) < 2:
        return None
    species = parts[0].lower()
    rest_of_id = mir_id[len(parts[0]) + 1:]
    url = f"https://mirgenedb.org/show/{species}/{rest_of_id}"
    return url

def find_sequence_by_name(target_id, fasta_sequences, sequence_type="primary"):
    candidates = []
    candidates.append(target_id)
    
    if sequence_type == "primary":
        candidates.extend([
            f"{target_id}_pri",
            f"{target_id}-pri", 
            f"{target_id}.pri",
            target_id.replace("-", "_") + "_pri",
            target_id.lower() + "_pri",
            target_id.upper() + "_pri"
        ])
    elif sequence_type == "precursor":
        candidates.extend([
            f"{target_id}_pre",
            f"{target_id}-pre",
            f"{target_id}.pre", 
            target_id.replace("-", "_") + "_pre",
            target_id.lower() + "_pre",
            target_id.upper() + "_pre"
        ])
    
    candidates = list(dict.fromkeys(candidates))
    
    matches = []
    for candidate in candidates:
        if candidate in fasta_sequences:
            matches.append((candidate, fasta_sequences[candidate]))
    
    if len(matches) == 0:
        return None, None
    elif len(matches) == 1:
        return matches[0]
    else:
        for candidate in candidates:
            for seq_id, sequence in matches:
                if seq_id == candidate:
                    return seq_id, sequence
        return matches[0]

def worker_function(mir_id, output_folder, pri_sequences, pre_sequences):
    try:
        pri_folder = os.path.join(output_folder, "primary")
        pre_folder = os.path.join(output_folder, "precursor")
        os.makedirs(pri_folder, exist_ok=True)
        os.makedirs(pre_folder, exist_ok=True)
        
        pri_file = os.path.join(pri_folder, f"{mir_id}.fasta")
        pre_file = os.path.join(pre_folder, f"{mir_id}.fasta")
        if os.path.exists(pri_file) and os.path.exists(pre_file):
            return {"mir_id": mir_id, "status": "skipped", "message": "Files exist"}
        
        main_seq_id, main_sequence = find_sequence_by_name(mir_id, pri_sequences, "primary")
        if not main_seq_id:
            return {"mir_id": mir_id, "status": "error", "message": "Main sequence not found"}
        
        url = construct_url(mir_id)
        if not url:
            return {"mir_id": mir_id, "status": "error", "message": "Invalid URL"}
        
        html = get_page(url)
        
        if not html and "-v" not in mir_id:
            alt_url = construct_url(f"{mir_id}-v1")
            html = get_page(alt_url)
        
        if not html:
            return {"mir_id": mir_id, "status": "error", "message": "Failed to get page"}
        
        orthologues_list = parse_orthologues(html)
        
        confirmed_orthologues = {}
        for orth_id in orthologues_list:
            orth_seq_id, orth_sequence = find_sequence_by_name(orth_id, pri_sequences, "primary")
            if orth_seq_id:
                confirmed_orthologues[orth_seq_id] = orth_sequence
        
        with open(pri_file, "w") as f:
            f.write(f">{main_seq_id}\n{main_sequence}\n")
            for orth_seq_id, orth_sequence in confirmed_orthologues.items():
                f.write(f">{orth_seq_id}\n{orth_sequence}\n")
        
        all_seq_ids = [main_seq_id] + list(confirmed_orthologues.keys())
        precursor_count = 0
        
        with open(pre_file, "w") as f:
            for seq_id in all_seq_ids:
                base_name = seq_id
                if base_name.endswith('_pri'):
                    base_name = base_name[:-4]
                elif base_name.endswith('-pri'):
                    base_name = base_name[:-4]
                
                pre_seq_id, pre_sequence = find_sequence_by_name(base_name, pre_sequences, "precursor")
                if pre_seq_id:
                    if seq_id.endswith('_pri'):
                        pre_header = seq_id.replace('_pri', '_pre')
                    elif seq_id.endswith('-pri'):
                        pre_header = seq_id.replace('-pri', '_pre')
                    else:
                        pre_header = f"{seq_id}_pre"
                    
                    f.write(f">{pre_header}\n{pre_sequence}\n")
                    precursor_count += 1
        
        return {
            "mir_id": mir_id,
            "status": "success",
            "confirmed_orthologues": len(confirmed_orthologues),
            "sequences_with_pre": precursor_count
        }
    
    except Exception as e:
        return {"mir_id": mir_id, "status": "error", "message": str(e)}

def process_chunk(chunk_file, output_folder, results_file, pri_sequences, pre_sequences):
    try:
        with open(chunk_file, 'r') as f:
            mir_ids = [line.strip() for line in f if line.strip()]
        
        for mir_id in mir_ids:
            try:
                result = worker_function(mir_id, output_folder, pri_sequences, pre_sequences)
                with open(results_file, 'a', newline='') as f:
                    fieldnames = ['mir_id', 'status', 'message', 'confirmed_orthologues', 'sequences_with_pre']
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                    writer.writerow(result)
            except Exception:
                pass
        
        return []
    except Exception:
        return []

def process_task(args):
    chunk_file, output_folder, results_file, pri_sequences, pre_sequences = args
    return process_chunk(chunk_file, output_folder, results_file, pri_sequences, pre_sequences)

def main():
    parser = argparse.ArgumentParser(description="Process MirGeneDB sequences.")
    parser.add_argument("--input", required=True, help="TSV file with 'mirgenedb_id' column.")
    parser.add_argument("--pri_fasta", required=True, help="Primary sequences FASTA file.")
    parser.add_argument("--pre_fasta", required=True, help="Precursor sequences FASTA file.")
    parser.add_argument("--output_folder", required=True, help="Output folder.")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Number of workers.")
    parser.add_argument("--prep_only", action="store_true", help="Only prepare chunk files.")
    args = parser.parse_args()
    
    pri_sequences = load_fasta_sequences(args.pri_fasta)
    pre_sequences = load_fasta_sequences(args.pre_fasta)
    
    if not pri_sequences or not pre_sequences:
        print("Error: Could not load FASTA files")
        sys.exit(1)
    
    os.makedirs(args.output_folder, exist_ok=True)
    chunks_dir = os.path.join(args.output_folder, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    
    chunk_files = [os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir) if f.endswith('.txt')]
    
    if not chunk_files or args.prep_only:
        df = pd.read_csv(args.input, sep="\t", low_memory=False)
        if 'mirgenedb_id' not in df.columns:
            print("Error: 'mirgenedb_id' column not found")
            sys.exit(1)
        
        all_ids = df['mirgenedb_id'].dropna().astype(str).unique().tolist()
        pri_folder = os.path.join(args.output_folder, "primary")
        pre_folder = os.path.join(args.output_folder, "precursor")
        
        to_process = []
        for mir_id in all_ids:
            pri_file = os.path.join(pri_folder, f"{mir_id}.fasta")
            pre_file = os.path.join(pre_folder, f"{mir_id}.fasta")
            if not (os.path.exists(pri_file) and os.path.exists(pre_file)):
                to_process.append(mir_id)
        
        random.shuffle(to_process)  
        chunk_size = max(1, len(to_process) // (args.workers * 2))
        
        for f in chunk_files:
            os.remove(f)
        
        chunk_files = []
        for i in range(0, len(to_process), chunk_size):
            chunk = to_process[i:i+chunk_size]
            chunk_file = os.path.join(chunks_dir, f"chunk_{i//chunk_size}.txt")
            with open(chunk_file, 'w') as f:
                for mir_id in chunk:
                    f.write(f"{mir_id}\n")
            chunk_files.append(chunk_file)
        
        if args.prep_only:
            return
    
    results_file = os.path.join(args.output_folder, "processing_results.csv")
    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['mir_id', 'status', 'message', 'confirmed_orthologues', 'sequences_with_pre'])
    
    pool = multiprocessing.Pool(processes=args.workers)
    tasks = [(chunk_file, args.output_folder, results_file, pri_sequences, pre_sequences) for chunk_file in chunk_files]
    
    for _ in tqdm(pool.imap_unordered(process_task, tasks), total=len(tasks)):
        pass
    
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()