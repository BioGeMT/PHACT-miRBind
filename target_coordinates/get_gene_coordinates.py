import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert genomic data to BED format')
    parser.add_argument('input', help='Input file path')
    parser.add_argument('output', help='Output BED file path')
    
    args = parser.parse_args()
    
    # Read input file (assuming it's tab-separated)
    print(f"Reading input file: {args.input}")
    df = pd.read_csv(args.input, sep='\t')
    
    # Extract the 5 columns we need and create BED format
    bed_df = pd.DataFrame({
        'chr': df['chr'],
        'start': df['start'].astype(int) - 1,  # Convert to 0-based (subtract 1)
        'end': df['end'].astype(int),          # Keep end as-is
        'sequence': df['gene'],                # Gene sequence
        'strand': df['strand']                 # Strand information
    })
    
    # Save as BED file (tab-separated, no header)
    print(f"Writing output file: {args.output}")
    bed_df.to_csv(args.output, sep='\t', header=False, index=False)
    
    print("Conversion complete!")
    print(f"Converted {len(bed_df)} records to BED format")
    print("\nFirst few lines of output:")
    print(bed_df.head().to_string(header=False, index=False))

if __name__ == "__main__":
    main()