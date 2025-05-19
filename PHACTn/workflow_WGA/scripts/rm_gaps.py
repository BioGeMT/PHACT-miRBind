from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
import os
import argparse


def find_sequence_in_msa(msa_file, query):
    """
    Searches for a query (human) sequence in a multiple sequence alignment (MSA) file.

    Parameters:
        msa_file (str): Path to the input MSA file in FASTA format.
        query (str): The sequence identifier to search for in the MSA.

    Returns:
        tuple: A tuple containing:
            - Seq: The found sequence object
            - MultipleSeqAlignment: The complete alignment object

    Raises:
        ValueError: If the query sequence ID is not found in the MSA.

    Example:
        >>> sequence, alignment = find_sequence_in_msa("alignment.fasta", "seq123")
        Found sequence seq123 in the MSA file
        >>> print(sequence)
        ATGCGATCG...

    Notes:
        - The function performs exact matching of sequence IDs (case-sensitive)
        - The returned alignment object can be used with Bio.AlignIO for further analysis
    """

    # Parse the MSA file
    alignment = AlignIO.read(msa_file, "fasta")

    # Create a dictionary mapping sequence IDs to sequences
    sequence_dict = {record.id: record.seq for record in alignment}

    # Check if the query exists in the dictionary
    if query not in sequence_dict:
        raise ValueError(f"No matching sequence found for query: {query}")
    print(f"Found sequence {query} in the MSA file")

    return sequence_dict[query], alignment


def remove_gap_columns(alignment, query_seq):
    """
    Removes alignment columns where the query sequence that belongs to human
    contains gaps ('-'), producing a gap-free alignment relative to the query sequence.

    Parameters:
        alignment (MultipleSeqAlignment): A Bio.AlignIO MultipleSeqAlignment object
        query_seq (Seq): The reference sequence (with gaps) to use for column filtering

    Returns:
        MultipleSeqAlignment: A new alignment object with gap columns removed

    Example:
        >>> from Bio import AlignIO
        >>> alignment = AlignIO.read("input.fasta", "fasta")
        >>> query_seq = alignment[0].seq  # Use first sequence as reference
        >>> trimmed = remove_gap_columns(alignment, query_seq)
        >>> print(f"Trimmed from {len(query_seq)} to {len(trimmed[0].seq)} columns")

    Notes:
        - All sequences in the alignment will be trimmed identically based on the query's gaps
        - The query sequence itself should be one of the sequences in the alignment
        - Gap characters ('-') in other sequences are preserved if the query has a base
    """
    columns_to_keep = [i for i, nt in enumerate(query_seq) if nt != '-']

    # Create a new alignment with only the columns that do not have gaps
    trimmed_alignment = MultipleSeqAlignment(
        SeqRecord(Seq("".join(record.seq[i] for i in columns_to_keep)), id=record.id, description="")
        for record in alignment
    )
    
    return trimmed_alignment


def save_alignment(alignment, output_file):
    """
    Writes a multiple sequence alignment to a file in FASTA format.

    Parameters:
        alignment (MultipleSeqAlignment): A BioPython MultipleSeqAlignment object 
                                         containing the sequences to be saved
        output_file (str): Path to the output file where the alignment will be written

    Returns:
        None: Output is written directly to the specified file

    Raises:
        ValueError: If the alignment object is empty or invalid
        IOError: If the output file cannot be written

    Example:
        >>> from Bio import AlignIO
        >>> alignment = AlignIO.read("input.fasta", "fasta")
        >>> save_alignment(alignment, "output.fasta")
        # Creates output.fasta with alignment data

    Notes:
        - Always overwrites existing files with the same name
        - Uses FASTA format which is widely compatible with other tools
        - Preserves all sequence IDs and annotations from the alignment object
    """
    if len(alignment) == 0:
        raise ValueError("Empty alignment cannot be saved")
    with open(output_file, "w") as handle:
        AlignIO.write(alignment, handle, "fasta")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse aln output to remove gaps')
    parser.add_argument('-msa', '--msa_file', required=True, help='Path to the MSA file')
    parser.add_argument('-id', '--query_id', required=True, help='ID of the sequence in the MSA file')
    parser.add_argument('-out', '--out_file', required=True, help='out file')
    args = parser.parse_args()

    msa_file = args.msa_file
    q_name = args.query_id
    edited_outFile = args.out_file

    if not os.path.exists(msa_file):
        raise FileNotFoundError(f"Input MSA file not found: {msa_file}")

    query_seq, alignment = find_sequence_in_msa(msa_file, q_name)
    trimmed_alignment = remove_gap_columns(alignment, query_seq)

    # Save the new alignment to a file
    output_file = edited_outFile
    save_alignment(trimmed_alignment, output_file)