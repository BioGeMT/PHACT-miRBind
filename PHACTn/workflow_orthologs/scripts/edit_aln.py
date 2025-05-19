from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
import os
import argparse


def edit_aln(aln_file, out_file):
    """
    Converts a Clustal-formatted alignment file to FASTA format while replacing all Uracil (U) 
    bases with Thymine (T) to standardize RNA sequences to DNA representation.

    Parameters:
        aln_file (str): Path to input alignment file in Clustal format.
        out_file (str): Path where the converted FASTA alignment will be saved.

    Returns:
        None: Output is written directly to the specified file.


    - Creates a new FASTA file at the specified output path
    - Prints conversion status message to stdout

    Example:
        >>> edit_aln("input.aln", "output.fasta")
        Converted input.aln to output.fasta with U->T conversion.
    """

    alignment = AlignIO.read(aln_file, "clustal")

    # Convert all U's to T's
    converted_records = []
    for record in alignment:
        dna_seq = Seq(str(record.seq).replace('U', 'T').replace('u', 't'))
        converted_records.append(SeqRecord(dna_seq, id=record.id, description=""))

    # Create a new alignment with converted sequences
    converted_alignment = MultipleSeqAlignment(converted_records)

    # Write the new alignment to FASTA format
    AlignIO.write(converted_alignment, out_file, "fasta")

    print(f"Converted {aln_file} to {out_file} with U->T conversion.")



def find_sequence_in_msa(msa_file, query):
    """
    Searches for a query (human) sequence in a multiple sequence alignment (MSA) file.

    This function reads a FASTA-formatted MSA file and looks for a sequence matching
    the provided query ID. If found, it returns both the sequence and the full alignment
    object for further processing.

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

    This is particularly useful for:
    - Preparing alignments for phylogenetic analysis
    - Extracting ungapped regions around a reference sequence
    - Creating position-specific scoring matrices

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

    This function provides a convenient wrapper around Bio.AlignIO.write() to save
    alignment objects to disk with proper error handling and file closure.

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
    base_name = os.path.basename(msa_file)


    edit_aln(msa_file, edited_outFile)
    query_seq, alignment = find_sequence_in_msa(edited_outFile, q_name)
    trimmed_alignment = remove_gap_columns(alignment, query_seq)

    # Save the new alignment to a file
    out_path_dirname = os.path.dirname(edited_outFile)
    out_path_basename =  os.path.splitext(os.path.basename(edited_outFile))[0]
    output_file =  os.path.join(out_path_dirname, f"{out_path_basename}_nogap.fasta")
    save_alignment(trimmed_alignment, output_file)



