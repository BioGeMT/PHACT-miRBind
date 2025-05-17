python evo_scripts/annotate_dataset.py --fasta miRNA_mature_files/hsa_mature.fas --tsv manakov_positives.tsv --mirgenedb miRNA_mature_files/mirgenedb_family_mappings.tsv --output output/manakov_positives_annotated.tsv

python evo_scripts/scrape_parallel.py --input output/manakov_positives_annotated.tsv --output_folder output/precursors --workers 8

python evo_scripts/annotate_precursors.py --input output/precursors --output output/annotated_precursors --mature miRNA_mature_files/all_mature.fas 

evo_scripts/./run_alignment.sh output/annotated_precursors output/annotated_precursors_MSA 8
