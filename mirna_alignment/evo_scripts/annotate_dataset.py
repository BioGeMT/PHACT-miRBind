#!/usr/bin/env python3
import argparse
import csv
import os
import pandas as pd
import re

def deduplicate_by_prefix(headers):
    deduped = []
    seen = set()
    for header in headers:
        prefix = header.split("_")[0] if "_" in header else header
        if prefix not in seen:
            seen.add(prefix)
            deduped.append(header)
    return deduped

def parse_fasta(fasta_file):
    mapping = {}
    with open(fasta_file, 'r') as f:
        header = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    sequence = "".join(seq_lines)
                    sequence = sequence.replace('U', 'T').replace('u', 't')
                    mapping[sequence] = header
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            sequence = "".join(seq_lines)
            sequence = sequence.replace('U', 'T').replace('u', 't')
            mapping[sequence] = header
    return mapping

def process_mapping(raw_mapping):
    if pd.isna(raw_mapping):
        return None, None
    
    first_mapping = re.split(r'\s*[;,]\s*', raw_mapping.strip())[0]
    cleaned = re.sub(r"_[35]p[*]*$", "", first_mapping)
    ref_id = re.sub(r"-v[0-9]+$", "", cleaned)
    
    return cleaned, ref_id

def process_combined(tsv_file, fasta_file, mirgenedb_file, output_file):
    fasta_mapping = parse_fasta(fasta_file)
    
    rows = []
    stats = {"total": 0, "kept": 0, "dropped": 0, "multiple": 0, "dropped_multiple": 0}
    
    with open(tsv_file, 'r', newline='') as infile:
        reader = csv.DictReader(infile, delimiter="\t")
        
        fieldnames = reader.fieldnames.copy()
        new_column = "mirgenedb_name"
        if new_column not in fieldnames:
            fieldnames.append(new_column)
        
        for row in reader:
            stats["total"] += 1
            tsv_seq = row["noncodingRNA"].strip()
            
            match_found = False
            
            if tsv_seq in fasta_mapping:
                row[new_column] = fasta_mapping[tsv_seq]
                rows.append(row)
                stats["kept"] += 1
                match_found = True
            elif not match_found:
                first_matches = []
                for fasta_seq, header in fasta_mapping.items():
                    if fasta_seq in tsv_seq:
                        unmatched = len(tsv_seq) - len(fasta_seq)
                        first_matches.append((header, unmatched))
                if first_matches:
                    min_unmatched = min(diff for _, diff in first_matches)
                    best_matches = [header for header, diff in first_matches if diff == min_unmatched]
                    best_matches = deduplicate_by_prefix(best_matches)
                    
                    if len(best_matches) > 1:
                        stats["multiple"] += 1
                        stats["dropped_multiple"] += 1
                    else:
                        row[new_column] = best_matches[0]
                        rows.append(row)
                        stats["kept"] += 1
                        match_found = True
                
                if not match_found:
                    second_matches = []
                    for fasta_seq, header in fasta_mapping.items():
                        if tsv_seq in fasta_seq:
                            unmatched = len(fasta_seq) - len(tsv_seq)
                            second_matches.append((header, unmatched))
                    if second_matches:
                        min_unmatched = min(diff for _, diff in second_matches)
                        best_matches = [header for header, diff in second_matches if diff == min_unmatched]
                        best_matches = deduplicate_by_prefix(best_matches)
                        
                        if len(best_matches) > 1:
                            stats["multiple"] += 1
                            stats["dropped_multiple"] += 1
                        else:
                            row[new_column] = best_matches[0]
                            rows.append(row)
                            stats["kept"] += 1
                            match_found = True
            
            if not match_found and not row.get(new_column, None):
                stats["dropped"] += 1
    
    df_annotations = pd.DataFrame(rows)
    
    unique_mature_ids = set()
    for row in rows:
        if row.get(new_column):
            unique_mature_ids.add(row[new_column])
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df_mirgene = pd.read_csv(mirgenedb_file, sep="\t")
    
    mirgenedb_id_col = None
    for col in df_mirgene.columns:
        if col.lower() in ['mirgenedb_id', 'id']:
            mirgenedb_id_col = col
            break
    
    family_col = None
    for col in df_mirgene.columns:
        if col.lower() == 'family':
            family_col = col
            break
    
    results = [process_mapping(val) for val in df_annotations['mirgenedb_name']]
    df_annotations['mirgenedb_id'] = [r[0] for r in results]
    df_annotations['mirgene_ref'] = [r[1] for r in results]
    
    existing_ids = set(df_mirgene[mirgenedb_id_col].unique())
    mask = df_annotations['mirgene_ref'].isin(existing_ids)
    df_annotations.loc[~mask, 'mirgenedb_id'] = None
    df_annotations.loc[~mask, 'mirgene_ref'] = None
    
    df_result = df_annotations.merge(
        df_mirgene, 
        left_on="mirgene_ref", 
        right_on=mirgenedb_id_col, 
        how="left"
    )
    
    df_result = df_result.rename(columns={family_col: "mirgenedb_fam"})
    df_result.drop(columns=["mirgene_ref", mirgenedb_id_col], inplace=True, errors='ignore')
    
    unique_mature_count = df_result['mirgenedb_name'].nunique()
    unique_precursor_count = df_result['mirgenedb_id'].nunique()
    unique_family_count = df_result['mirgenedb_fam'].nunique()
    
    total_records = len(df_result)
    mapped_records = df_result['mirgenedb_id'].notna().sum()
    
    df_result.to_csv(output_file, sep="\t", index=False)

def main():
    parser = argparse.ArgumentParser(description="Map miRNA sequences to IDs and enrich with family data.")
    parser.add_argument("--fasta", required=True, help="Input FASTA file.")
    parser.add_argument("--tsv", required=True, help="Input TSV file.")
    parser.add_argument("--mirgenedb", required=True, help="MirGeneDB TSV file.")
    parser.add_argument("--output", required=True, help="Output TSV file.")
    args = parser.parse_args()

    process_combined(args.tsv, args.fasta, args.mirgenedb, args.output)

if __name__ == "__main__":
    main()