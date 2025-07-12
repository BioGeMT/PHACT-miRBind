#!/usr/bin/env python3

from ete4 import PhyloTree, NCBITaxa
from ete4.treeview import TreeStyle, TextFace, faces
import pandas as pd
import sys
import random
import colorsys

def load_species_mapping(csv_file):
    """Load species mapping from CSV file"""
    df = pd.read_csv(csv_file)
    return dict(zip(df['3_letter_code'].str.lower(), df['taxid']))

def parse_stk_annotations(stk_file):
    """Parse STK file to identify flanking regions and secondary structure"""
    mature_mask = None
    structure_mask = None
    
    with open(stk_file, 'r') as f:
        for line in f:
            if line.startswith('#=GC cA1'):
                # Extract the annotation string after the identifier
                parts = line.strip().split()
                if len(parts) >= 2:
                    mature_mask = ''.join(parts[2:])  # Join all parts after 'cA1'
            elif line.startswith('#=GC SS_cons'):
                # Extract secondary structure annotation
                parts = line.strip().split()
                if len(parts) >= 2:
                    structure_mask = ''.join(parts[2:])  # Join all parts after 'SS_cons'
    
    if not mature_mask:
        return None, None
        
    # Find first A and last B to identify true flanking regions
    first_A = mature_mask.find('A')
    last_B = mature_mask.rfind('B')
    
    if first_A == -1 or last_B == -1:
        return None, None
        
    # Create flanking mask: True for 5'/3' flanking, False for mature+loop
    flanking_mask = []
    for i, char in enumerate(mature_mask):
        if i < first_A or i > last_B:
            flanking_mask.append(True)  # 5'/3' flanking region
        else:
            flanking_mask.append(False)  # Mature sequences + loop region
    
    return flanking_mask, structure_mask

def parse_primary_structures(primary_file):
    """Parse primary.fa file to extract individual sequence structures"""
    structures = {}
    current_name = None
    
    with open(primary_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Extract sequence name (remove '>')
                current_name = line[1:]
            elif line.endswith(' #S') and current_name:
                # This is a structure line
                structure = line[:-3]  # Remove ' #S'
                structures[current_name] = structure
                
    return structures

def get_aligned_structure(seq_name, aligned_sequence, individual_structures):
    """Align individual structure to match the consensus alignment"""
    if seq_name not in individual_structures:
        return None
    
    individual_struct = individual_structures[seq_name]
    aligned_struct = []
    struct_pos = 0
    
    # Map individual structure to aligned sequence positions
    for seq_char in aligned_sequence:
        if seq_char == '-':
            # Gap in alignment - use dot for structure
            aligned_struct.append('.')
        else:
            # Non-gap character - use corresponding structure position
            if struct_pos < len(individual_struct):
                aligned_struct.append(individual_struct[struct_pos])
            else:
                aligned_struct.append('.')
            struct_pos += 1
    
    return ''.join(aligned_struct)

def get_region_boundaries(flanking_mask):
    """Find major boundary positions between flanking and mature regions"""
    if not flanking_mask:
        return []
    
    boundaries = []
    
    # Find the first transition from flanking (True) to mature (False)
    for i in range(len(flanking_mask) - 1):
        if flanking_mask[i] == True and flanking_mask[i + 1] == False:
            boundaries.append(i + 1)  # 5' flanking to mature transition
            break
    
    # Find the last transition from mature (False) to flanking (True)
    for i in range(len(flanking_mask) - 1, 0, -1):
        if flanking_mask[i - 1] == False and flanking_mask[i] == True:
            boundaries.append(i)  # mature to 3' flanking transition
            break
    
    return boundaries

def get_ncbi_lineage(ncbi, taxid):
    """Get taxonomic lineage for a taxid"""
    try:
        lineage = ncbi.get_lineage(taxid)
        lineage_ranks = ncbi.get_rank(lineage)
        lineage_names = ncbi.get_taxid_translator(lineage)
        
        taxonomy = {'taxid': taxid}
        for tid in lineage:
            rank = lineage_ranks.get(tid)
            name = lineage_names.get(tid)
            if rank and name:
                taxonomy[rank] = name
        return taxonomy
    except:
        return {'taxid': taxid}

def generate_colors(n, seed=42):
    """Generate n distinct colors"""
    random.seed(seed)
    colors = []
    golden_ratio = 0.618033988749
    
    for i in range(n):
        hue = (i * golden_ratio) % 1.0
        saturation = random.uniform(0.4, 0.7)
        lightness = random.uniform(0.7, 0.9)
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    return colors

def get_species_taxid(seq_name, code_to_taxid):
    """Extract taxid from sequence name"""
    code = seq_name[:3].lower()
    taxid = code_to_taxid.get(code)
    return int(taxid) if taxid else None

def calculate_consensus(sequences):
    """Calculate consensus sequence"""
    if not sequences:
        return ""
    consensus = []
    for pos in range(len(sequences[0])):
        counts = {}
        for seq in sequences:
            if pos < len(seq) and seq[pos] in 'ATCGU-':
                counts[seq[pos].upper()] = counts.get(seq[pos].upper(), 0) + 1
        consensus.append(max(counts, key=counts.get) if counts else '-')
    return ''.join(consensus)

def create_sequence_faces(sequence, consensus, node, column_start=1, flanking_mask=None, structure_mask=None):
    """Create colored nucleotide faces, showing individual structures vs consensus in flanking regions"""
    colors = {'A': 'red', 'T': 'blue', 'G': 'green', 'C': 'orange', 'U': 'blue', '-': 'gray'}
    
    # Get individual structure for this sequence (needs to be aligned to match consensus)
    individual_struct = get_aligned_structure(node.name, sequence, individual_structures)
    
    # Find boundaries between flanking and mature regions for blank spaces
    boundaries = get_region_boundaries(flanking_mask) if flanking_mask else []
    
    for i, (seq_nt, cons_nt) in enumerate(zip(sequence, consensus)):
        # Check if this position is in a flanking region
        is_flanking = flanking_mask and i < len(flanking_mask) and flanking_mask[i]
        
        # Check if this is a boundary position (transition between regions)
        is_boundary = i in boundaries
        
        if is_boundary:
            # Add blank space at region boundaries
            face = TextFace(' ', fsize=200, ftype='courier', bold=False, fgcolor='white')
            face.background.color = 'white'
        elif is_flanking and structure_mask and individual_struct and i < len(structure_mask) and i < len(individual_struct):
            # In flanking regions: show individual structure symbols
            individual_char = individual_struct[i]
            consensus_char = structure_mask[i]
            
            if individual_char != consensus_char:
                # Plain text for differences from consensus structure
                face = TextFace(individual_char, fsize=100, ftype='courier', bold=True, fgcolor='black')
            elif individual_char in '()':
                # Light red for matching parentheses (base pairs)
                face = TextFace(individual_char, fsize=100, ftype='courier', bold=True, fgcolor='black')
                face.background.color = 'lightcoral'
            else:
                # Light blue for matching dots
                face = TextFace(individual_char, fsize=100, ftype='courier', bold=True, fgcolor='black')
                face.background.color = 'lightblue'
        elif not is_flanking and seq_nt.upper() != cons_nt.upper() and seq_nt != '-':
            # Color highlighting for mature sequences and loop (non-flanking)
            face = TextFace(seq_nt, fsize=100, ftype='courier', bold=True, fgcolor='white')
            face.background.color = colors.get(seq_nt.upper(), 'purple')
        else:
            # No color highlighting for flanking regions or consensus matches
            face = TextFace(seq_nt, fsize=100, ftype='courier', bold=True, fgcolor='black')
        face.margin_right = 0
        face.margin_left = 0
        faces.add_face_to_node(face, node, column=column_start + i, position='aligned')

def save_lineage_data(taxonomic_data, highlight_level, csv_filename):
    """Save taxonomic lineage data to CSV"""
    import os
    base_name = os.path.splitext(os.path.basename(csv_filename))[0]
    output_file = f'{base_name}_lineage.csv'
    
    output_data = []
    for taxid, tax_data in taxonomic_data.items():
        row = {'taxid': taxid, 'chosen_level': highlight_level}
        row.update(tax_data)
        row['highlighted'] = bool(tax_data.get(highlight_level))
        output_data.append(row)
    
    df = pd.DataFrame(output_data)
    df.to_csv(output_file, index=False)
    print(f"✓ Lineage data saved: {output_file}")

# Global variables
consensus_seq = ""
code_to_taxid = {}
taxonomic_data = {}
group_colors = {}
highlight_level = ""
flanking_mask = None
structure_mask = None
individual_structures = {}

def layout(node):
    """Tree layout function"""
    node.img_style['hz_line_width'] = 20
    node.img_style['vt_line_width'] = 20
    
    if node.is_leaf:
        node.img_style['size'] = 0
        
        # Get display name
        code = node.name[:3].lower()
        display_name = node.name
        
        if hasattr(node, 'sci_name'):
            display_name = node.sci_name
        elif code in code_to_taxid:
            taxid = code_to_taxid[code]
            if taxid in taxonomic_data and 'species' in taxonomic_data[taxid]:
                display_name = taxonomic_data[taxid]['species']
        
        # Get background color
        bg_color = None
        if code in code_to_taxid:
            taxid = code_to_taxid[code]
            if taxid in taxonomic_data:
                group = taxonomic_data[taxid].get(highlight_level)
                if group:
                    bg_color = group_colors.get(group)
        
        # Create name face
        name_face = TextFace(display_name, fsize=200, ftype='courier', bold=True, fgcolor='black')
        if bg_color:
            name_face.background.color = bg_color
        
        # Special border for Homo sapiens
        if 'Homo sapiens' in display_name:
            name_face.border.width = 30
            name_face.border.color = 'red'
            name_face.border.type = 0
        
        name_face.margin_right = 150
        name_face.margin_left = 150
        faces.add_face_to_node(name_face, node, column=0, position='aligned')
        
        # Add sequence highlighting
        if hasattr(node, 'props') and 'sequence' in node.props:
            create_sequence_faces(node.props['sequence'], consensus_seq, node, column_start=1, flanking_mask=flanking_mask, structure_mask=structure_mask)
    
    # Bootstrap values
    if not node.is_leaf and hasattr(node, 'support') and node.support:
        support_face = TextFace(f"{node.support:.0f}", fsize=150, fgcolor='blue', bold=True)
        faces.add_face_to_node(support_face, node, column=0, position='branch-top')

# Main execution
if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Usage: python tree.py <tree_file> <fasta_file> <taxonomic_level> <species_csv> <stk_file> <primary_file>")
        print("Levels: kingdom, phylum, class, order, family")
        sys.exit(1)
    
    tree_file, fasta_file, highlight_level, species_csv, stk_file, primary_file = sys.argv[1:7]
    highlight_level = highlight_level.lower()
    
    # Extract miRNA name and pre/pri type from tree file path
    import os
    tree_filename = os.path.basename(tree_file)  # e.g., "Hsa-Let-7-P1b.treefile"
    mirna_name = tree_filename.replace('.treefile', '')  # e.g., "Hsa-Let-7-P1b"
    
    # Determine pre/pri type from directory name
    tree_dir = os.path.dirname(tree_file)
    if 'pre_' in tree_dir:
        sequence_type = "Pre-miRNA"
    elif 'pri_' in tree_dir:
        sequence_type = "Primary Transcript"
    else:
        sequence_type = "miRNA"
    
    # Validate level
    if highlight_level not in ['kingdom', 'phylum', 'class', 'order', 'family']:
        print(f"Invalid level: {highlight_level}")
        sys.exit(1)
    
    # Load data
    code_to_taxid = load_species_mapping(species_csv)
    ncbi = NCBITaxa()
    
    # Parse STK annotations to identify flanking regions and secondary structure
    flanking_mask, structure_mask = parse_stk_annotations(stk_file)
    
    # Parse individual structures from primary.fa
    individual_structures = parse_primary_structures(primary_file)
    
    # Load tree
    with open(tree_file) as f:
        tree_string = f.read().strip()
    t = PhyloTree(tree_string, sp_naming_function=lambda name: get_species_taxid(name, code_to_taxid))
    
    # Get taxonomic data
    print("Getting NCBI taxonomic data...")
    taxonomic_data = {}
    for node in t:
        if node.is_leaf:
            code = node.name[:3].lower()
            if code in code_to_taxid:
                taxid = code_to_taxid[code]
                if taxid not in taxonomic_data:
                    taxonomic_data[taxid] = get_ncbi_lineage(ncbi, taxid)
    
    # Get groups and colors
    groups = set()
    for tax_data in taxonomic_data.values():
        group = tax_data.get(highlight_level)
        if group:
            groups.add(group)
    
    groups = sorted(groups)
    if groups:
        colors = generate_colors(len(groups))
        group_colors = dict(zip(groups, colors))
    
    # Setup tree - make all branches same length
    for node in t.traverse():
        node.dist = 1.0  # Set all branch lengths to 1.0
    
    # Link alignment
    with open(fasta_file) as f:
        alignment = f.read()
    t.link_to_alignment(alignment, alg_format='fasta')
    
    # Calculate consensus
    sequences = [leaf.props['sequence'] for leaf in t if hasattr(leaf, 'props') and 'sequence' in leaf.props]
    consensus_seq = calculate_consensus(sequences)
    
    # Render tree
    ts = TreeStyle()
    ts.layout_fn = layout
    ts.show_leaf_name = False
    ts.show_branch_length = False
    ts.show_branch_support = False
    ts.mode = 'r'
    ts.scale = 400
    ts.branch_vertical_margin = 5
    
    # Create comprehensive title
   # title_text = f'{mirna_name} {sequence_type} - Phylogenetic Tree ({highlight_level} taxonomy highlighted)'
    #ts.title.add_face(TextFace(title_text, fsize=300, bold=True), column=0)
    
    output_file = f'phylo_tree_{highlight_level}.png'
    t.render(output_file, w=8000, h=5000, dpi=300, tree_style=ts)
    
    # Save lineage data
    save_lineage_data(taxonomic_data, highlight_level, species_csv)
    
    print(f"✓ Tree rendered: {output_file}")
    print(f"✓ {len(groups)} {highlight_level} groups highlighted")
    if groups:
        print("Color legend:")
        for group, color in group_colors.items():
            print(f"  {color} → {group}")