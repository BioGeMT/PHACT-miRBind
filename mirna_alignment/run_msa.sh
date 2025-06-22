#!/bin/bash

# Step 1: Annotate dataset with miRGeneDB IDs
echo "Step 1: Annotating dataset with miRGeneDB IDs..."
python evo_scripts/annotate_dataset.py --fasta miRNA_mature_files/hsa_mature.fas --tsv /home/dtzim01/positives.tsv --mirgenedb miRNA_mature_files/mirgenedb_family_mappings.tsv --output output/manakov_positives_annotated.tsv

# Step 2: Get orthologues using both primary (flanking) and precursor (no_flanking) sequences
echo "Step 2: Getting orthologues..."
python evo_scripts/get_orthologues.py --input output/manakov_positives_annotated.tsv --pri_fasta miRNA_mature_files/flanking.fas --pre_fasta miRNA_mature_files/no_flanking.fas --output_folder output/orthologues --workers 64

# Step 3a: Annotate primary sequences with structure
echo "Step 3a: Annotating primary sequences with structure..."
python evo_scripts/annotate_precursors.py --input output/orthologues/primary --output output/annotated_primary --mature miRNA_mature_files/all_mature.fas 

# Step 3b: Annotate precursor sequences with structure  
echo "Step 3b: Annotating precursor sequences with structure..."
python evo_scripts/annotate_precursors.py --input output/orthologues/precursor --output output/annotated_precursor --mature miRNA_mature_files/all_mature.fas

# Step 4: Make alignment script executable
echo "Step 4: Making alignment script executable..."
chmod +x evo_scripts/run_alignment.sh

# Step 5: Run alignments with 3 parameter scenarios

# Conservative parameters (stricter)
echo "Step 5a: Running CONSERVATIVE alignments..."
evo_scripts/run_alignment.sh output/annotated_primary output/alignments_primary_conservative 64 "--struct-weight=300 --indel=-200 --indel-open=-1000 --max-diff-am=30 --max-diff=80 --min-prob=0.01 --alifold-consensus-dp --plfold-span=150 --plfold-winsize=300"
evo_scripts/run_alignment.sh output/annotated_precursor output/alignments_precursor_conservative 64 "--struct-weight=300 --indel=-200 --indel-open=-1000 --max-diff-am=30 --max-diff=80 --min-prob=0.01 --alifold-consensus-dp --plfold-span=150 --plfold-winsize=300"

# Default parameters (balanced, uses LocARNA defaults)
echo "Step 5b: Running DEFAULT alignments..."
evo_scripts/run_alignment.sh output/annotated_primary output/alignments_primary_default 64 ""
evo_scripts/run_alignment.sh output/annotated_precursor output/alignments_precursor_default 64 ""

# Step 6: Process alignments and create filtered output directories
echo "Step 6: Processing and filtering alignments..."

python evo_scripts/get_alignments.py --input output/alignments_primary_conservative --output output/filtered_alignments_primary_conservative
python evo_scripts/get_alignments.py --input output/alignments_primary_default --output output/filtered_alignments_primary_default

python evo_scripts/get_alignments.py --input output/alignments_precursor_conservative --output output/filtered_alignments_precursor_conservative
python evo_scripts/get_alignments.py --input output/alignments_precursor_default --output output/filtered_alignments_precursor_default