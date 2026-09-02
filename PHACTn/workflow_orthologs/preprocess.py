from Bio import AlignIO, Phylo
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from collections import defaultdict
import argparse
import os
import sys


def convert_clustal_to_fasta(aln_file: str) -> MultipleSeqAlignment:
    """
    Reads a Clustal-formatted alignment file and returns it as a FASTA-style
    alignment with all Uracil (U) bases replaced by Thymine (T).
    """
    try:
        alignment = AlignIO.read(aln_file, "clustal")
    except ValueError as e:
        print(f"Error: Could not parse the input file '{aln_file}' as Clustal format.")
        print(f"BioPython error: {e}")
        sys.exit(1)

    converted_records = [
        SeqRecord(Seq(str(record.seq).replace('U', 'T').replace('u', 't')), id=record.id, description="")
        for record in alignment
    ]

    return MultipleSeqAlignment(converted_records)


def remove_gap_columns(alignment: MultipleSeqAlignment, query_id: str) -> MultipleSeqAlignment:
    """
    Removes alignment columns where the query sequence contains gaps ('-'),
    producing a gap-free alignment relative to the query sequence.
    """
    sequence_dict = {record.id: record.seq for record in alignment}

    if query_id not in sequence_dict:
        raise ValueError(f"No matching sequence found for query: {query_id}")
    print(f"Found sequence {query_id} in the MSA file")

    query_seq = sequence_dict[query_id]
    columns_to_keep = [i for i, nt in enumerate(query_seq) if nt != '-']

    return MultipleSeqAlignment(
        SeqRecord(Seq("".join(record.seq[i] for i in columns_to_keep)), id=record.id, description="")
        for record in alignment
    )


def rename_by_species(alignment: MultipleSeqAlignment) -> MultipleSeqAlignment:
    """
    Renames each record to its 3-letter species code (the first 3 characters
    of its original ID). If several records share the same species code
    (paralogs), the first keeps the plain code and the rest are suffixed as
    "<code>-2", "<code>-3", etc.
    """
    records_by_species = defaultdict(list)
    for record in alignment:
        species_code = record.id[:3]
        records_by_species[species_code].append(record)

    renamed_records = []
    for species_code, records in records_by_species.items():
        renamed_records.append(SeqRecord(records[0].seq, id=species_code, description=""))
        for i, record in enumerate(records[1:], start=2):
            renamed_records.append(SeqRecord(record.seq, id=f"{species_code}-{i}", description=""))

    return MultipleSeqAlignment(renamed_records)


def add_missing_tree_species(alignment: MultipleSeqAlignment, tree_file: str) -> MultipleSeqAlignment:
    """
    Adds a fully-gapped record for every species present in the tree but
    absent from the alignment, so that the alignment covers all tree tips.
    """
    tree = Phylo.read(tree_file, "newick")
    tree_species = {leaf.name for leaf in tree.get_terminals()}

    aligned_species = {record.id.split("-")[0] for record in alignment}
    missing_species = tree_species - aligned_species

    alignment_length = alignment.get_alignment_length()
    gap_records = [
        SeqRecord(Seq("-" * alignment_length), id=species_code, description="")
        for species_code in missing_species
    ]

    return MultipleSeqAlignment(list(alignment) + gap_records)


def save_alignment(alignment: MultipleSeqAlignment, output_file: str) -> None:
    if len(alignment) == 0:
        raise ValueError("Empty alignment cannot be saved")
    with open(output_file, "w") as handle:
        AlignIO.write(alignment, handle, "fasta")


def binarize_alignment(alignment: MultipleSeqAlignment) -> MultipleSeqAlignment:
    """
    Converts an alignment to binary format (gap = 0, ATGC = 1, others = -).
    """
    binary_records = []
    for record in alignment:
        seq = str(record.seq)
        bin_seq = ''.join(
            ('0' if nt == '-' else '1' if nt.upper() in "ATGC" else '-')
            for nt in seq
        )
        binary_records.append(SeqRecord(Seq(bin_seq), id=record.id, description=""))

    return MultipleSeqAlignment(binary_records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert a Clustal alignment to FASTA (U -> T) and remove columns gapped in the query sequence.'
    )
    parser.add_argument('-msa', '--msa_file', required=True, help='Path to the input Clustal MSA file (.aln)')
    parser.add_argument('-id', '--query_id', required=True, help='ID of the query sequence in the MSA file')
    parser.add_argument('-tree', '--tree_file', required=True, help='Path to the reference species tree (newick)')
    parser.add_argument('-out', '--out_file', required=True, help='Path for the output FASTA file')
    parser.add_argument('-bin_out', '--binary_out_file', required=True, help='Path for the output binary FASTA file')
    args = parser.parse_args()

    if not os.path.exists(args.msa_file):
        raise FileNotFoundError(f"Input MSA file not found: {args.msa_file}")

    if not os.path.exists(args.tree_file):
        raise FileNotFoundError(f"Input tree file not found: {args.tree_file}")

    converted_alignment = convert_clustal_to_fasta(args.msa_file)
    trimmed_alignment = remove_gap_columns(converted_alignment, args.query_id)
    renamed_alignment = rename_by_species(trimmed_alignment)
    complete_alignment = add_missing_tree_species(renamed_alignment, args.tree_file)
    save_alignment(complete_alignment, args.out_file)

    print(f"Successfully preprocessed '{args.msa_file}' to '{args.out_file}'.")

    binary_alignment = binarize_alignment(complete_alignment)
    save_alignment(binary_alignment, args.binary_out_file)

    print(f"Successfully binarized '{args.out_file}' to '{args.binary_out_file}'.")
