import pandas as pd
import requests
import time
import argparse
from urllib.parse import quote

def reverse_complement(seq):
    """
    Return reverse complement of DNA sequence
    """
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement.get(base, base) for base in seq[::-1])

def get_ucsc_sequence(chr_name, start, end, genome='hg38'):
    """
    Get sequence from UCSC REST API
    """
    # UCSC API endpoint
    url = f"https://api.genome.ucsc.edu/getData/sequence"
    params = {
        'genome': genome,
        'chrom': chr_name,
        'start': start,
        'end': end
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        return data['dna'].upper()
        
    except Exception as e:
        print(f"API error for {chr_name}:{start}-{end}: {e}")
        return None

def validate_with_ucsc_api(bed_file, genome='hg38', max_sequences=100):
    """
    Validate sequences using UCSC REST API
    Note: Limited to small batches due to API rate limits
    """
    # Read BED file - now expecting 5 columns with strand
    bed_df = pd.read_csv(bed_file, sep='\t', 
                        names=['chr', 'start', 'end', 'sequence', 'strand'])
    
    # Limit sequences for API testing
    if len(bed_df) > max_sequences:
        print(f"Testing first {max_sequences} sequences (API rate limit)")
        bed_df = bed_df.head(max_sequences)
    
    results = []
    
    for idx, row in bed_df.iterrows():
        # Handle chromosome naming
        chr_name = str(row['chr'])
        if not chr_name.startswith('chr'):
            chr_name = f"chr{chr_name}"
        
        # Handle mitochondrial chromosome naming difference
        if chr_name == 'chrMT':
            chr_name = 'chrM'
        
        # Get sequence from UCSC
        ref_seq_forward = get_ucsc_sequence(chr_name, row['start'], row['end'], genome)
        
        if ref_seq_forward:
            # Apply strand correction
            if row['strand'] == '-':
                ref_seq = reverse_complement(ref_seq_forward)
            else:
                ref_seq = ref_seq_forward
                
            bed_seq = row['sequence'].upper()
            match = ref_seq == bed_seq
            
            results.append({
                'chr': row['chr'],
                'start': row['start'],
                'end': row['end'],
                'strand': row['strand'],
                'bed_sequence': bed_seq,
                'ref_sequence': ref_seq,
                'match': match
            })
            
            if not match:
                print(f"MISMATCH at {chr_name}:{row['start']}-{row['end']} ({row['strand']} strand)")
                print(f"  BED: {bed_seq[:50]}...")
                print(f"  REF: {ref_seq[:50]}...")
                if row['strand'] == '-':
                    print(f"  FWD: {ref_seq_forward[:50]}...")
        
        # Rate limiting
        time.sleep(0.1)  # 100ms delay between requests
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} sequences...")
    
    # Summary
    if results:
        results_df = pd.DataFrame(results)
        matches = results_df['match'].sum()
        total = len(results_df)
        print(f"\n=== VALIDATION SUMMARY ===")
        print(f"Tested sequences: {total}")
        print(f"Matches: {matches} ({matches/total*100:.1f}%)")
        print(f"Mismatches: {total-matches} ({(total-matches)/total*100:.1f}%)")
        
        return results_df
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Validate sequences using UCSC REST API')
    parser.add_argument('bed_file', help='BED file with sequences')
    parser.add_argument('-g', '--genome', default='hg38', help='Genome build (default: hg38)')
    parser.add_argument('-n', '--max_sequences', type=int, default=100, 
                       help='Max sequences to test (default: 100)')
    
    args = parser.parse_args()
    
    validate_with_ucsc_api(args.bed_file, args.genome, args.max_sequences)

if __name__ == "__main__":
    main()