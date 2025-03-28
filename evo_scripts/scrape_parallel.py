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

def get_page(url, retry_count=2):
    #Download the page and return its HTML text (or None on error).
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    }
    
    for attempt in range(retry_count + 1):
        try:
            # Add a small random delay
            time.sleep(random.uniform(0.2, 1.0))
            
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.text
            
            print(f"Error: Got status code {response.status_code} for URL {url}")
            
            # If rate limited, wait longer before retry
            if response.status_code == 429:
                wait_time = (attempt + 1) * random.uniform(5, 10)
                print(f"Rate limited. Waiting {wait_time:.2f} seconds before retry.")
                time.sleep(wait_time)
            elif attempt < retry_count:
                time.sleep(random.uniform(1, 3))
        except requests.exceptions.Timeout:
            print(f"Timeout while getting URL {url}")
            if attempt < retry_count:
                time.sleep(random.uniform(1, 3))
        except Exception as e:
            print(f"Exception while getting URL {url}: {e}")
            if attempt < retry_count:
                time.sleep(random.uniform(1, 3))
    
    return None

def parse_precursor(html):
    #Parse the HTML to extract the precursor sequence.
    soup = BeautifulSoup(html, "html.parser")
    precursor = None
    for tr in soup.find_all("tr"):
        th = tr.find("th")
        if th and "Precursor" in th.get_text():
            td = tr.find("td")
            if td:
                pre_tag = td.find("pre")
                if pre_tag:
                    # Remove all whitespace and newlines
                    precursor = "".join(pre_tag.get_text(separator="").split())
                    break
    return precursor

def parse_orthologues(html):
    # Parse the HTML to extract the list of orthologue ids.
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
    # Construct the URL for a MirGeneDB id.
    parts = mir_id.split("-")
    if len(parts) < 2:
        print(f"Invalid mir_id: {mir_id}")
        return None
    species = parts[0].lower()  # e.g. "hsa"
    rest_of_id = mir_id[len(parts[0]) + 1:]
    url = f"https://mirgenedb.org/show/{species}/{rest_of_id}"
    return url

def worker_function(mir_id, output_folder):
    #P rocess a single MirGeneDB ID.
    try:
        # Check if output folder for this id already exists
        id_folder = os.path.join(output_folder, mir_id)
        if os.path.isdir(id_folder):
            return {"mir_id": mir_id, "status": "skipped", "message": f"Output folder already exists: {id_folder}"}
        
        # Construct URL and get page
        url = construct_url(mir_id)
        if not url:
            return {"mir_id": mir_id, "status": "error", "message": "Failed to construct URL"}
        
        html = get_page(url)
        
        # Try with -v1 suffix if needed
        if not html and "-v" not in mir_id:
            alt_mir_id = f"{mir_id}-v1"
            alt_url = construct_url(alt_mir_id)
            html = get_page(alt_url)
            if html:
                mir_id = alt_mir_id
                url = alt_url
        
        if not html:
            return {"mir_id": mir_id, "status": "error", "message": "Failed to get page content"}
        
        # Parse main precursor
        precursor = parse_precursor(html)
        if not precursor:
            return {"mir_id": mir_id, "status": "warning", "message": "No precursor found", "orthologues": 0}
        
        # Parse orthologues
        orth_list = parse_orthologues(html)
        
        # Process orthologues
        orth_data = {}
        for orth in orth_list:
            orth_url = construct_url(orth)
            if not orth_url:
                continue
            
            orth_html = get_page(orth_url)
            if not orth_html:
                continue
            
            orth_precursor = parse_precursor(orth_html)
            if orth_precursor:
                orth_data[orth] = orth_precursor
        
        # Write output
        os.makedirs(id_folder, exist_ok=True)
        
        # Write FASTA file
        fasta_path = os.path.join(id_folder, f"{mir_id}_precursors.fasta")
        with open(fasta_path, "w") as f:
            # Write main sequence
            f.write(f">{mir_id}\n{precursor}\n")
            # Write orthologues
            for orth, seq in orth_data.items():
                f.write(f">{orth}\n{seq}\n")
        
        # Write TSV file
        tsv_path = os.path.join(id_folder, f"{mir_id}_precursors.tsv")
        with open(tsv_path, "w") as f:
            f.write("mir_id\tprecursor\n")
            f.write(f"{mir_id}\t{precursor}\n")
            for orth, seq in orth_data.items():
                f.write(f"{orth}\t{seq}\n")
        
        return {
            "mir_id": mir_id,
            "status": "success",
            "url": url,
            "orthologues": len(orth_data)
        }
    
    except Exception as e:
        return {"mir_id": mir_id, "status": "error", "message": str(e)}

def process_chunk(chunk_file, output_folder, results_file):
    # Process a chunk of MirGeneDB IDs from a file.
    try:
        with open(chunk_file, 'r') as f:
            mir_ids = [line.strip() for line in f if line.strip()]
        
        results = []
        for mir_id in mir_ids:
            try:
                result = worker_function(mir_id, output_folder)
                results.append(result)
                
                # Write result to the results file
                try:
                    with open(results_file, 'a', newline='') as f:
                        # Ensure we have consistent fieldnames
                        fieldnames = ['mir_id', 'status', 'message', 'url', 'orthologues']
                        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                        writer.writerow(result)
                except Exception as e:
                    print(f"Error writing result to CSV for {mir_id}: {e}")
            except Exception as e:
                print(f"Error processing ID {mir_id}: {e}")
        
        return results
    except Exception as e:
        print(f"Error processing chunk file {chunk_file}: {e}")
        return []

def process_task(args):
    # Helper function to unpack arguments for process_chunk.
    chunk_file, output_folder, results_file = args
    return process_chunk(chunk_file, output_folder, results_file)

def main():
    parser = argparse.ArgumentParser(
        description="Scrape MirGeneDB pages to retrieve precursor sequences with multiprocessing."
    )
    parser.add_argument("--input", required=True,
                        help="TSV file with a column named 'mirgenedb_id' containing ids.")
    parser.add_argument("--output_folder", required=True,
                        help="Folder for output.")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                        help=f"Number of worker processes. Default: {multiprocessing.cpu_count()}")
    parser.add_argument("--prep_only", action="store_true",
                        help="Only prepare the chunk files, don't start processing.")
    args = parser.parse_args()
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Create a folder for chunk files
    chunks_dir = os.path.join(args.output_folder, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Check if we need to prepare chunk files
    chunk_files = [os.path.join(chunks_dir, f) for f in os.listdir(chunks_dir) if f.endswith('.txt')]
    
    if not chunk_files or args.prep_only:
        print("Preparing chunk files...")
        
        # Read input file
        df = pd.read_csv(args.input, sep="\t", low_memory=False)
        if 'mirgenedb_id' not in df.columns:
            print(f"Error: Column 'mirgenedb_id' not found. Available columns: {df.columns.tolist()}")
            sys.exit(1)
        
        # Get unique IDs
        all_ids = df['mirgenedb_id'].dropna().astype(str).unique().tolist()
        print(f"Found {len(all_ids)} unique MirGeneDB IDs")
        
        # Filter already processed IDs
        to_process = []
        for mir_id in all_ids:
            id_folder = os.path.join(args.output_folder, mir_id)
            if not os.path.isdir(id_folder):
                to_process.append(mir_id)
        
        print(f"After filtering already processed IDs: {len(to_process)} IDs to process")
        
        # Create chunk files
        random.shuffle(to_process)  # Shuffle IDs to distribute work evenly
        chunk_size = max(1, len(to_process) // (args.workers * 2))  # Multiple chunks per worker
        
        # Delete old chunk files
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
        
        print(f"Created {len(chunk_files)} chunk files in {chunks_dir}")
        
        if args.prep_only:
            print("Chunk files prepared. Use without --prep_only to start processing.")
            return
    else:
        print(f"Found {len(chunk_files)} existing chunk files")
    
    # Create a results file
    results_file = os.path.join(args.output_folder, "processing_results.csv")
    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['mir_id', 'status', 'url', 'orthologues', 'message'])
    
    # Process chunks in parallel
    print(f"Starting parallel processing with {args.workers} workers")
    
    # Create pool and tasks
    pool = multiprocessing.Pool(processes=args.workers)
    tasks = [(chunk_file, args.output_folder, results_file) for chunk_file in chunk_files]
    
    # Use imap to process chunks one at a time per worker with a separate function
    for _ in tqdm(pool.imap_unordered(process_task, tasks), total=len(tasks)):
        pass
    
    pool.close()
    pool.join()
    
    print("Processing complete! Check the results file for details.")

if __name__ == "__main__":
    main()