python evo_scripts/find_alignments.py --input output/annotated_precursors_aligned/ --output output/all_alignments/

python evo_scripts/filter_alignments.py output/all_alignments/ output/all_alignments_filtered/ --min 10

find output/all_alignments_filtered -name "*.aln" | xargs -P 64 -I {} bash -c '
  aln_file={}
  basename=$(basename "$aln_file" .aln)
  mkdir -p "output/iqtree_out/$basename"
  iqtree -s "$aln_file" -m MFP -nt 1 -pre "output/iqtree_out/$basename/$basename"
'