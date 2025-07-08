from ete3 import PhyloTree, TreeStyle, SeqMotifFace, AttrFace
import os

# This line is KEY for headless servers without Xvfb
# It tells the Qt backend to render off-screen
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


# --- Input Files ---
tree_file = "Hsa-Let-7-P1b.treefile"
alignment_file = "Hsa-Let-7-P1b_annotated.fasta"
output_file = "tree_with_alignment.png"

# --- Define a custom layout function ---
def my_layout(node):
    if node.is_leaf():
        # Add a text face for the sequence name
        name_face = AttrFace("name", fsize=10, fgcolor="black")
        node.add_face(name_face, column=0, position="aligned")

        # Add the sequence alignment face in a compact format
        # We've changed seq_format to "compact" to save space
        seq_face = SeqMotifFace(node.sequence, seq_format="compact", gap_format="line")

        # Set the background colors for the alignment
        seq_face.bg_col = {"A": "green", "T": "red", "C": "blue", "G": "orange", "U": "red", "-": "grey"}

        node.add_face(seq_face, column=1, position="aligned")


# --- Load the tree ---
t = PhyloTree(tree_file)

# --- Link the alignment to the tree ---
t.link_to_alignment(alignment_file, alg_format="fasta")


# --- Tree Style ---
ts = TreeStyle()
ts.show_leaf_name = False
ts.layout_fn = my_layout   # Set the custom layout function

# --- Render the tree with the alignment ---
# We've increased the width 'w' to make the tree wider
t.render(output_file, tree_style=ts, w=400, units="mm", dpi=300)

print(f"Tree and alignment saved to {output_file}")