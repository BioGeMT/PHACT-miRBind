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

# High parameters (most strict structural constraints)
echo "Step 5a: Running HIGH alignments..."
evo_scripts/run_alignment.sh output/annotated_primary output/alignments_primary_high 64 "--struct-weight=400 --indel=-300 --indel-open=-1000 --max-diff-am=30 --max-diff=80"
evo_scripts/run_alignment.sh output/annotated_precursor output/alignments_precursor_high 64 "--struct-weight=400 --indel=-300 --indel-open=-1000 --max-diff-am=30 --max-diff=80"

# Default parameters (balanced structural constraints)
echo "Step 5b: Running DEFAULT alignments..."
evo_scripts/run_alignment.sh output/annotated_primary output/alignments_primary_default 64 "--struct-weight=200 --indel=-150 --indel-open=-750"
evo_scripts/run_alignment.sh output/annotated_precursor output/alignments_precursor_default 64 "--struct-weight=200 --indel=-150 --indel-open=-750"

# Low parameters (minimal structural constraints)
echo "Step 5c: Running LOW alignments..."
evo_scripts/run_alignment.sh output/annotated_primary output/alignments_primary_low 64 "--struct-weight=0 --indel=-50 --indel-open=-350 --max-diff-am=50 --max-diff=120"
evo_scripts/run_alignment.sh output/annotated_precursor output/alignments_precursor_low 64 "--struct-weight=0 --indel=-50 --indel-open=-350 --max-diff-am=50 --max-diff=120"

# Step 6: Process alignments and create filtered output directories
echo "Step 6: Processing and filtering alignments..."

python evo_scripts/get_alignments.py --input output/alignments_primary_high --output output/filtered_alignments_primary_high
python evo_scripts/get_alignments.py --input output/alignments_primary_default --output output/filtered_alignments_primary_default
python evo_scripts/get_alignments.py --input output/alignments_primary_low --output output/filtered_alignments_primary_low

python evo_scripts/get_alignments.py --input output/alignments_precursor_high --output output/filtered_alignments_precursor_high
python evo_scripts/get_alignments.py --input output/alignments_precursor_default --output output/filtered_alignments_precursor_default
python evo_scripts/get_alignments.py --input output/alignments_precursor_low --output output/filtered_alignments_precursor_low