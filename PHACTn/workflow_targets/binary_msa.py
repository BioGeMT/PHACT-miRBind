from Bio.SeqIO.FastaIO import SimpleFastaParser
import argparse
from pathlib import Path
import logging

#NUCLEOTIDE = '1'
#GAP = '0'

logging.basicConfig(
    format='[%(levelname)s] %(message)s',
    level=logging.INFO
)


def convert_to_binary_fasta(input_path, output_path):
    """Convert a FASTA file to binary format (gap = 0, ATGC = 1, others = -)."""
    sequence_count = 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(input_path) as input_msa, open(output_path, mode="w") as out_binary:
        for header, seq in SimpleFastaParser(input_msa):
            sequence_count += 1
            if not seq:
                logging.warning(f"Empty sequence found for: {header}")
                
            bin_seq = ''.join(
                ('0' if nt == '-' else '1' if nt.upper() in "ATGC" else '-')
                for nt in seq
            )
            
            out_binary.write(f'>{header}\n{bin_seq}\n')
            logging.debug(f"Processed sequence: {header}")

    logging.info(f"Finished processing {sequence_count} sequences.")


def main():
    parser = argparse.ArgumentParser(description="Convert MSA to binary FASTA")
    parser.add_argument("--msa_file", type=str, required=True, help="Path to input MSA file")
    parser.add_argument("--binary_out", type=str, required=True, help="Path to output binary FASTA file")
    args = parser.parse_args()

    logging.info(f"Reading MSA from: {args.msa_file}")
    logging.info(f"Writing binary FASTA to: {args.binary_out}")

    convert_to_binary_fasta(args.msa_file, args.binary_out)


if __name__ == "__main__":
    main()