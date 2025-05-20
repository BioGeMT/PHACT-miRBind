from Bio import SeqIO
from Bio.Seq import Seq
import argparse
from ete3 import Tree
import pandas as pd

def get_strand(query_id, positions_file):
    """
    Retrieve the strand information for a given miRNA ID from the positions file.

    Parameters:
        query_id (str): The miRNA ID (e.g., mir_8a).
        positions_file (str): Path to the TSV file containing miRNA strand info.

    Returns:
        str: '+' or '-' strand.

    Raises:
        ValueError: If the miRNA ID is not found in the positions file.
    """
    df = pd.read_csv(positions_file, sep="\t")
    strand_series = df.loc[df["ID"] == f"{query_id}_pre", "Strand"]
    if strand_series.empty:
        raise ValueError(f"Strand info not found for {query_id} in {positions_file}")
    return strand_series.values[0]


def process_msa(input_file, output_file, strand):
    """
    Process an MSA file by removing all-gap sequences and reverse complementing
    if the miRNA is on the minus strand.

    Parameters:
        input_file (str): Path to the input FASTA MSA file.
        output_file (str): Path to the filtered output FASTA file.
        strand (str): Strand of the query miRNA ('+' or '-').

    Returns:
        list: IDs of the sequences retained in the filtered alignment.
    """
    alignment_names = []
    with open(output_file, "w") as out_f:
        for record in SeqIO.parse(input_file, "fasta"):
            seq_str = str(record.seq).replace("*", "-")
            if not any(base != "-" for base in seq_str):
                continue
            record.seq = Seq(seq_str)
            if strand == "-":
                record.seq = record.seq.reverse_complement()
            record.description = ""
            SeqIO.write(record, out_f, "fasta")
            alignment_names.append(record.id)
    return alignment_names


def prune_tree(input_tree, output_tree, taxa_to_keep):
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
    tree.prune(common_taxa, preserve_branch_length=True)
    tree.write(outfile=output_tree, format=1)


def main(input_msa, input_tree, output_msa, output_tree, query_id, positions_file):
    """
    Main function to process an MSA file:
    - Retrieves strand information for the query.
    - Filters and reverse-complements MSA if needed.
    - Prunes the corresponding phylogenetic tree.

    Parameters:
        input_msa (str): Path to the input MSA FASTA file.
        input_tree (str): Path to the input Newick tree.
        output_msa (str): Path to the output filtered FASTA file.
        output_tree (str): Path to the output pruned tree file.
        query_id (str): The miRNA query ID.
        positions_file (str): Path to the strand info file (TSV).
    """
    strand = get_strand(query_id, positions_file)
    alignment_names = process_msa(input_msa, output_msa, strand)
    prune_tree(input_tree, output_tree, alignment_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove sequences with gaps from a FASTA file')
    parser.add_argument('--input_msa', type=str, required=True, help='Input FASTA file')
    parser.add_argument('--input_tree', type=str, required=True, help='Input tree file')
    parser.add_argument('--output_msa', type=str, required=True, help='Output FASTA file')
    parser.add_argument('--output_tree', type=str, required=True, help='Output tree file')
    parser.add_argument('--query_id', type=str, required=True, help='Current miRNA ID')
    parser.add_argument('--positions_file', type=str, required=False, help='Positions file')
    args = parser.parse_args()

    main(args.input_msa, args.input_tree, args.output_msa, args.output_tree, args.query_id, args.positions_file)


