from Bio import SeqIO
from Bio.Seq import Seq
import argparse
from ete3 import Tree
import pandas as pd
import logging
from typing import List, Set

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_msa(input_file: str, output_file: str) -> List[str]:
    """
    Process an MSA file by removing sequences that are entirely gaps, Ns, or a mix of both.

    Parameters:
        input_file (str): Path to the input FASTA MSA file.
        output_file (str): Path to the filtered output FASTA file.

    Returns:
        list: IDs of the sequences retained in the filtered alignment.
    """
    alignment_names = []
    original_count = 0
    with open(output_file, "w") as out_f:
        for record in SeqIO.parse(input_file, "fasta"):
            original_count += 1
            seq_str = str(record.seq).replace("*", "-")
            if all(base in "-Nn" for base in seq_str):
                logging.info(f"Removing gap/N-only sequence: {record.id}")
                continue
            record.seq = Seq(seq_str)
            record.description = ""
            SeqIO.write(record, out_f, "fasta")
            alignment_names.append(record.id)
    logging.info(f"Processed MSA: Retained {len(alignment_names)} out of {original_count} sequences.")
    return alignment_names


def prune_tree(input_tree: str, output_tree: str, taxa_to_keep: List[str]) -> None:
    """
    Prune a phylogenetic tree to retain only the taxa present in the filtered alignment.

    Parameters:
        input_tree (str): Path to the original Newick tree file.
        output_tree (str): Path to the pruned tree file.
        taxa_to_keep (list): List of sequence IDs to retain in the tree.

    Raises:
        ValueError: If no taxa in the tree match the alignment IDs.
    """
    tree = Tree(input_tree, format=1)
    tree_leaves = set(leaf.name for leaf in tree)
    common_taxa = tree_leaves.intersection(taxa_to_keep)
    if not common_taxa:
        raise ValueError("No matching taxa found between the tree and the MSA!")
    logging.info(f"Pruning tree: Retaining {len(common_taxa)} out of {len(tree_leaves)} leaves.")
    tree.prune(common_taxa, preserve_branch_length=True)
    tree.write(outfile=output_tree, format=1)


def main(input_msa: str, input_tree: str, output_msa: str, output_tree: str) -> None:
    """
    Main function to process an MSA file:
    - Filters MSA .
    - Prunes the corresponding phylogenetic tree.

    Parameters:
        input_msa (str): Path to the input MSA FASTA file.
        input_tree (str): Path to the input Newick tree.
        output_msa (str): Path to the output filtered FASTA file.
        output_tree (str): Path to the output pruned tree file.
    """
    logging.info(f"Starting processing for MSA '{input_msa}' and tree '{input_tree}'.")
    alignment_names = process_msa(input_msa, output_msa)
    prune_tree(input_tree, output_tree, alignment_names)
    logging.info(f"Processing complete. Filtered files saved to '{output_msa}' and '{output_tree}'.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter all-gap sequences from a multiple sequence alignment (MSA) and prune a corresponding phylogenetic tree."
        )
    parser.add_argument('--input_msa', type=str, required=True, help='Input FASTA file')
    parser.add_argument('--input_tree', type=str, required=True, help='Input tree file')
    parser.add_argument('--output_msa', type=str, required=True, help='Output FASTA file')
    parser.add_argument('--output_tree', type=str, required=True, help='Output tree file')
    args = parser.parse_args()

    main(args.input_msa, args.input_tree, args.output_msa, args.output_tree)


