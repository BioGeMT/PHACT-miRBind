import pandas as pd
import argparse
from pyfaidx import Fasta

def reverse_complement(seq):
    """
    Return reverse complement of DNA sequence
    """
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement.get(base, base) for base in seq[::-1])

def validate_sequences(bed_file, fasta_file, output_file=None):
    """
    Validate sequences in BED file against reference genome
    """
    # Load reference genome
    print(f"Loading reference genome: {fasta_file}")
    genome = Fasta(fasta_file)
    
    # Read BED file - now expecting 5 columns with strand
    print(f"Reading BED file: {bed_file}")
    bed_df = pd.read_csv(bed_file, sep='\t', 
                        names=['chr', 'start', 'end', 'sequence', 'strand'])
    
    results = []
    mismatches = 0
    
    for idx, row in bed_df.iterrows():
        # Get reference sequence (pyfaidx uses 0-based coordinates like BED)
        chr_name = str(row['chr'])
        if not chr_name.startswith('chr'):
            chr_name = f"chr{chr_name}"
        
        # Handle mitochondrial chromosome naming difference
        if chr_name == 'chrMT':
            chr_name = 'chrM'
        
        try:
            # Get reference sequence (forward strand)
            ref_seq_forward = str(genome[chr_name][row['start']:row['end']]).upper()
            
            # Apply strand correction
            if row['strand'] == '-':
                ref_seq = reverse_complement(ref_seq_forward)
            else:
                ref_seq = ref_seq_forward
            
            bed_seq = row['sequence'].upper()
            
            match = ref_seq == bed_seq
            if not match:
                mismatches += 1
                
            results.append({
                'chr': row['chr'],
                'start': row['start'],
                'end': row['end'],
                'strand': row['strand'],
                'bed_sequence': bed_seq,
                'ref_sequence': ref_seq,
                'ref_forward': ref_seq_forward,
                'match': match,
                'length_bed': len(bed_seq),
                'length_ref': len(ref_seq)
            })
            
            # Progress update
            if (idx + 1) % 10000 == 0:
                print(f"Processed {idx + 1:,} sequences...")
                
        except Exception as e:
            print(f"Error at {chr_name}:{row['start']}-{row['end']}: {e}")
            results.append({
                'chr': row['chr'],
                'start': row['start'],
                'end': row['end'],
                'strand': row['strand'],
                'bed_sequence': row['sequence'],
                'ref_sequence': 'ERROR',
                'ref_forward': 'ERROR',
                'match': False,
                'length_bed': len(row['sequence']),
                'length_ref': 0
            })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Summary
    total = len(results_df)
    matches = results_df['match'].sum()
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Total sequences: {total:,}")
    print(f"Matches: {matches:,} ({matches/total*100:.1f}%)")
    print(f"Mismatches: {mismatches:,} ({mismatches/total*100:.1f}%)")
    
    # Show first few mismatches
    if mismatches > 0:
        print(f"\nFirst few mismatches:")
        mismatched = results_df[~results_df['match']].head()
        for _, row in mismatched.iterrows():
            print(f"  {row['chr']}:{row['start']}-{row['end']} ({row['strand']} strand)")
            print(f"    BED: {row['bed_sequence'][:50]}...")
            print(f"    REF: {row['ref_sequence'][:50]}...")
            if row['strand'] == '-':
                print(f"    FWD: {row['ref_forward'][:50]}...")
    
    # Save results if requested
    if output_file:
        results_df.to_csv(output_file, sep='\t', index=False)
        print(f"\nDetailed results saved to: {output_file}")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Validate BED sequences against reference genome')
    parser.add_argument('bed_file', help='BED file with sequences')
    parser.add_argument('fasta_file', help='Reference genome FASTA file')
    parser.add_argument('-o', '--output', help='Output file for detailed results')
    
    args = parser.parse_args()
    
    validate_sequences(args.bed_file, args.fasta_file, args.output)

if __name__ == "__main__":
    main()