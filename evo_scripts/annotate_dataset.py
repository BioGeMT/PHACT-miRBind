#!/usr/bin/env python3
import argparse
import csv
import os
import sys
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
                # Save the previous sequence, if any.
                if header is not None:
                    sequence = "".join(seq_lines)
                    # Convert RNA (U) to DNA (T)
                    sequence = sequence.replace('U', 'T').replace('u', 't')
                    mapping[sequence] = header
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)
        # Save the last sequence in the file.
        if header is not None:
            sequence = "".join(seq_lines)
            # Convert RNA (U) to DNA (T)
            sequence = sequence.replace('U', 'T').replace('u', 't')
            mapping[sequence] = header
    return mapping

def process_mapping(raw_mapping):
    if pd.isna(raw_mapping):
        return None, None
    
    # Just use the first mapping if there are multiple (semicolon/comma separated)
    first_mapping = re.split(r'\s*[;,]\s*', raw_mapping.strip())[0]
    
    # Remove only the *3p/*5p suffixes but keep version
    cleaned = re.sub(r"_[35]p[*]*$", "", first_mapping)
    
    # Create reference version (without -v1 etc) for matching
    ref_id = re.sub(r"-v[0-9]+$", "", cleaned)
    
    return cleaned, ref_id

def process_combined(tsv_file, fasta_file, mirgenedb_file, output_file):
    """
    Combined workflow that:
    1. Processes TSV file to add mirgenedb_name column by matching sequences
    2. Enriches with mirgenedb family data
    """
    print(f"Phase 1: Processing sequences from {tsv_file}")
    
    # Phase 1: Add mirgenedb_name column
    fasta_mapping = parse_fasta(fasta_file)
    if not fasta_mapping:
        sys.exit("Error: No sequences found in the FASTA file.")
    
    # Process TSV to add mirgenedb_name
    rows = []
    stats = {"total": 0, "kept": 0, "dropped": 0, "multiple": 0, "dropped_multiple": 0}
    
    with open(tsv_file, 'r', newline='') as infile:
        reader = csv.DictReader(infile, delimiter="\t")
        if "noncodingRNA" not in reader.fieldnames:
            sys.exit(f"Error: '{tsv_file}' does not contain the required 'noncodingRNA' column.")
        
        # Prepare header: original columns + new column.
        fieldnames = reader.fieldnames.copy()
        new_column = "mirgenedb_name"
        if new_column not in fieldnames:
            fieldnames.append(new_column)
        
        for row in reader:
            stats["total"] += 1
            tsv_seq = row["noncodingRNA"].strip()
            
            match_found = False
            
            # 1. Exact match.
            if tsv_seq in fasta_mapping:
                row[new_column] = fasta_mapping[tsv_seq]
                rows.append(row)
                stats["kept"] += 1
                match_found = True
            elif not match_found:
                # 2. First substring check: FASTA sequence is a substring of TSV sequence.
                first_matches = []
                for fasta_seq, header in fasta_mapping.items():
                    if fasta_seq in tsv_seq:
                        unmatched = len(tsv_seq) - len(fasta_seq)
                        first_matches.append((header, unmatched))
                if first_matches:
                    min_unmatched = min(diff for _, diff in first_matches)
                    best_matches = [header for header, diff in first_matches if diff == min_unmatched]
                    best_matches = deduplicate_by_prefix(best_matches)
                    
                    # Check if multiple matches after deduplication
                    if len(best_matches) > 1:
                        stats["multiple"] += 1
                        stats["dropped_multiple"] += 1
                    else:
                        row[new_column] = best_matches[0]
                        rows.append(row)
                        stats["kept"] += 1
                        match_found = True
                
                # 3. Second substring check: TSV sequence is a substring of FASTA sequence.
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
                        
                        # Check if multiple matches after deduplication
                        if len(best_matches) > 1:
                            stats["multiple"] += 1
                            stats["dropped_multiple"] += 1
                        else:
                            row[new_column] = best_matches[0]
                            rows.append(row)
                            stats["kept"] += 1
                            match_found = True
            
            # 4. No match found: drop the row.
            if not match_found and not row.get(new_column, None):
                stats["dropped"] += 1
    
    # Create intermediate DataFrame
    df_annotations = pd.DataFrame(rows)
    
    # Count unique mature miRNA IDs
    unique_mature_ids = set()
    for row in rows:
        if row.get(new_column):
            unique_mature_ids.add(row[new_column])
    
    # Phase 1 summary
    print(f"  Total rows processed: {stats['total']}")
    print(f"  Rows kept: {stats['kept']}")
    print(f"  Rows dropped: {stats['dropped']}")
    print(f"  Rows with multiple mappings (dropped): {stats['multiple']}")
    print(f"  Unique mature miRNA IDs found: {len(unique_mature_ids)}")
    
    # Phase 2: Merge with MirGeneDB data
    print(f"\nPhase 2: Merging with MirGeneDB data from {mirgenedb_file}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        df_mirgene = pd.read_csv(mirgenedb_file, sep="\t")
        print(f"MirGeneDB columns: {df_mirgene.columns.tolist()}")
    except Exception as e:
        sys.exit(f"Error reading MirGeneDB file: {e}")
    print(f"Loaded {len(df_mirgene)} records from MirGeneDB file")
    
    # Determine the correct ID column name in MirGeneDB file
    mirgenedb_id_col = None
    for col in df_mirgene.columns:
        if col.lower() in ['mirgenedb_id', 'id']:
            mirgenedb_id_col = col
            break
    if not mirgenedb_id_col:
        sys.exit(f"Error: Could not find MirGeneDB ID column. Available columns: {df_mirgene.columns.tolist()}")
    
    # Determine the Family column name
    family_col = None
    for col in df_mirgene.columns:
        if col.lower() == 'family':
            family_col = col
            break
    if not family_col:
        sys.exit(f"Error: Could not find Family column. Available columns: {df_mirgene.columns.tolist()}")
    
    print(f"Using ID column: {mirgenedb_id_col}, Family column: {family_col}")
    
    # Process the mirgenedb_name column into cleaned IDs
    results = [process_mapping(val) for val in df_annotations['mirgenedb_name']]
    df_annotations['mirgenedb_id'] = [r[0] for r in results]
    df_annotations['mirgene_ref'] = [r[1] for r in results]
    
    # Keep only those rows where mirgene_ref exists in MirGeneDB
    existing_ids = set(df_mirgene[mirgenedb_id_col].unique())
    mask = df_annotations['mirgene_ref'].isin(existing_ids)
    df_annotations.loc[~mask, 'mirgenedb_id'] = None
    df_annotations.loc[~mask, 'mirgene_ref'] = None
    
    # Merge with MirGeneDB
    df_result = df_annotations.merge(
        df_mirgene, 
        left_on="mirgene_ref", 
        right_on=mirgenedb_id_col, 
        how="left"
    )
    
    # Clean up and rename columns
    df_result = df_result.rename(columns={family_col: "mirgenedb_fam"})
    
    # Drop intermediate columns
    df_result.drop(columns=["mirgene_ref", mirgenedb_id_col], inplace=True, errors='ignore')
    
    # Count unique values after merging
    unique_mature_count = df_result['mirgenedb_name'].nunique()
    unique_precursor_count = df_result['mirgenedb_id'].nunique()
    unique_family_count = df_result['mirgenedb_fam'].nunique()
    
    # Statistics for console output
    total_records = len(df_result)
    mapped_records = df_result['mirgenedb_id'].notna().sum()
    
    print(f"\nFinal Results:")
    print(f"  Total records: {total_records}")
    print(f"  Successfully mapped to families: {mapped_records} ({mapped_records/total_records*100:.1f}%)")
    print(f"  No family mapping: {total_records - mapped_records} ({(total_records-mapped_records)/total_records*100:.1f}%)")
    print(f"  Unique mature miRNA IDs (mirgenedb_name): {unique_mature_count}")
    print(f"  Unique precursor miRNA IDs (mirgenedb_id): {unique_precursor_count}")
    print(f"  Unique miRNA families (mirgenedb_fam): {unique_family_count}")
    
    # Save the final output
    df_result.to_csv(output_file, sep="\t", index=False)
    print(f"\nOutput file saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Process a TSV file (with 'noncodingRNA' column), map sequences to miRNA IDs, "
            "and enrich with miRNA family data in a single workflow.\n"
            "1. Match sequences against FASTA to identify miRNAs\n"
            "2. Merge with MirGeneDB to add family information\n"
            "Rows with multiple mappings are dropped."
        )
    )
    parser.add_argument("--fasta", required=True, help="Input FASTA file containing miRNA sequences.")
    parser.add_argument("--tsv", required=True, help="TSV file with 'noncodingRNA' column to process.")
    parser.add_argument("--mirgenedb", required=True, help="MirGeneDB ID to Family TSV file.")
    parser.add_argument("--output", required=True, help="Output TSV file name.")
    args = parser.parse_args()

    process_combined(args.tsv, args.fasta, args.mirgenedb, args.output)

if __name__ == "__main__":
    main()