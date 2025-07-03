# miRNA Phylogenetic Analysis Pipeline

This pipeline analyzes miRNA evolutionary relationships through multiple sequence alignment and phylogenetic tree construction.

First clone the repo and cd into it. Download the dataset file (manakov_positives.tsv) from:

https://drive.google.com/drive/folders/14p99NI1y7rHrbHucbiIQMpqtYNTpuoMP?usp=sharing


## Dependencies

First set up the conda environment with the env.yaml file:

```
conda env create -f env.yaml
conda activate align_phact-mirbind
```

The `run_msa.sh` script runs the complete alignment pipeline with multiple parameter sets.

Required packages:
- Python 3.8+
- pandas
- BioPython
- ViennaRNA
- LocARNA
- BeautifulSoup4
- tqdm, requests

## Dataset Summary

- Original dataset: 1,431,689 rows of human miRNAs
- 555 unique mature miRNAs identified
- 377 unique precursor miRNAs
- 229 unique miRNA families
- 377 precursors had multiple sequences to align 

## Input Dataset

The pipeline starts with a TSV file containing human miRNA sequences:

| Column            | Example                                      |
|-------------------|----------------------------------------------|
| gene              | CTACCTGATCCGTTTACTCACTATGCCCCCTTGCCATC...    |
| noncodingRNA      | CTGTACAGCCTCCTAGCTTTCC                       |
| noncodingRNA_name | hsa-let-7a-2-3p                              |
| noncodingRNA_fam  | let-7                                        |
| ...               | ...                                          |

The `noncodingRNA` column contains the mature miRNA sequence.

## Pipeline Steps

### 1. Dataset Annotation
**Script**: `evo_scripts/annotate_dataset.py`

Maps miRNA sequences to miRGeneDB IDs and adds family information.

**New Columns Added**:
| Column          | Description                        | Example           |
|-----------------|------------------------------------|-------------------|
| mirgenedb_name  | Mature miRGeneDB entry name        | Hsa-Let-7-P1d_3p* |
| mirgenedb_id    | Precursor miRGeneDB entry name     | Hsa-Let-7-P1d     |
| mirgenedb_fam   | miRGeneDB family name              | Let-7             |

### 2. Ortholog Retrieval
**Script**: `evo_scripts/get_orthologues.py`

Retrieves precursor sequences and orthologs from miRGeneDB. Creates separate folders for primary (with flanking) and precursor (without flanking) sequences.

**Output Structure**:
```
output/orthologues/
├── primary/          # Primary sequences with flanking regions
└── precursor/        # Precursor sequences with _pre suffix
```

### 3. Structure Annotation
**Script**: `evo_scripts/annotate_precursors.py`

Adds secondary structure predictions and marks mature miRNA locations for both primary and precursor sequences.

```
>Hsa-Let-7-P1b
GGGUCUGUCCACCUGCCGCGCCCCCCGGGCUGAGGUAGGAGGUUGUAUAGUUGAGGAGGACACCCAAGGAGAUCACUAUACGGCCUCCUAGCUUUCCCCAGGCUGCGCCCUGCACGGGACGGGGCCC
((.(((((((.(.(((.((((..((.(((..(((.((((((((((((((((.((((.(....))).......)))))))))))))))))).)))..))).))..))))...))).)))))))).)). #S
..............................AAAAAAAAAAAAAAAAAAAAAA.......................BBBBBBBBBBBBBBBBBBBBBB.............................. #1
..............................123456789abcdefghijklm.......................123456789abcdefghijklm.............................. #2
```

### 4. Multiple Sequence Alignment
**Script**: `evo_scripts/run_alignment.sh`

Performs structure-aware alignments using MLocARNA with two parameter sets:

**High Parameters (Stricter)**:
- `--struct-weight=400` (emphasizes structure over sequence)
- `--indel=-300`, `--indel-open=-1000` (strict gap penalties)
- `--max-diff-am=30`, `--max-diff=80` (additional structural constraints)

**Default Parameters (Balanced)**:
- `--struct-weight=200` (balanced structure/sequence weighting)
- `--indel=-150`, `--indel-open=-750` (moderate gap penalties)

**Base MLocARNA Parameters**:
- `--consensus-structure=alifold`: Uses Vienna RNAalifold
- `--plfold-span=150`: Maximum span for base pairs
- `--write-structure`: Outputs structural information
- `--stockholm`: Stockholm format output
- `--alifold-consensus-dp`: Dynamic programming for consensus
- `--quiet`: Silent execution

### 5. Alignment Processing
**Script**: `evo_scripts/get_alignments.py`

Processes alignment results with comprehensive status reporting:
- Pre-processing analysis (SUCCESS/FAILED/EMPTY/MISSING)
- Detailed statistics (success rates, sequence distribution)
- Processing results with filtering statistics

### 6. Phylogenetic Tree Construction
**Script**: `run_trees.sh`

Builds maximum likelihood phylogenetic trees using IQ-TREE for all filtered alignments.

**IQ-TREE Parameters**:
- `-m TEST`: Model selection (tests all standard DNA models)
- `-bb 1000`: Ultrafast bootstrap with 1000 replicates
- `-nt 1`: Single thread per job (for parallel execution)
- `--seqtype DNA`: Explicitly specify nucleotide sequences
- `--safe`: Prevent numerical underflow issues
- `--seed 42`: Reproducible results
- `--quiet`: Reduce output verbosity

- 5 miRNAs have precursor sequences too conserved for bootstrap analysis:
  - Hsa-Mir-3187, Hsa-Mir-423, Hsa-Mir-584, Hsa-Mir-642, Hsa-Mir-935-v1
- These trees are built without bootstrap support but retain valid topology and branch lengths
- Primary sequences for these miRNAs successfully build trees with bootstrap support

**Tree Processing**: `evo_scripts/get_trees.py`

Extracts clean `.treefile` outputs from IQ-TREE results for downstream analysis.

## Running the Pipeline

### Complete Pipeline
```
./run_msa.sh    # Multiple sequence alignments
./run_trees.sh  # Phylogenetic tree construction
```

**MSA Pipeline** (`run_msa.sh`):
1. Annotates dataset with miRGeneDB IDs
2. Retrieves orthologous sequences (primary and precursor)
3. Annotates sequences with secondary structure for both primary and precursor sequences
4. Makes alignment script executable
5. Runs alignments with 4 parameter combinations:
   - 2 parameter sets (high/default) × 2 sequence types (primary/precursor)
6. Filters alignments (4+ sequences minimum)

**Tree Pipeline** (`run_trees.sh`):
1. Builds phylogenetic trees for all filtered alignments
2. Uses IQ-TREE with model selection and bootstrap support
3. Processes 4 alignment parameter combinations in parallel
4. Extracts clean `.treefile` outputs for downstream analysis

## Output Structure

```
output/
├── manakov_positives_annotated.tsv              # Annotated dataset
├── orthologues/
│   ├── primary/                                 # Primary sequences with flanking
│   └── precursor/                               # Precursor sequences
├── annotated_primary/                           # Annotated primary sequences
├── annotated_precursor/                         # Annotated precursor sequences
├── alignments_primary_high/                     # High parameter alignments (primary)
├── alignments_primary_default/                  # Default alignments (primary)
├── alignments_precursor_high/                   # High parameter alignments (precursor)
├── alignments_precursor_default/                # Default alignments (precursor)
├── filtered_alignments_primary_high_4seq/       # Filtered alignments (4+ sequences, primary high)
├── filtered_alignments_primary_default_4seq/    # Filtered alignments (4+ sequences, primary default)
├── filtered_alignments_precursor_high_4seq/     # Filtered alignments (4+ sequences, precursor high)
├── filtered_alignments_precursor_default_4seq/  # Filtered alignments (4+ sequences, precursor default)
├── trees_primary_high_4seq/                     # Raw IQ-TREE output (primary high)
├── trees_primary_default_4seq/                  # Raw IQ-TREE output (primary default)
├── trees_precursor_high_4seq/                   # Raw IQ-TREE output (precursor high)
├── trees_precursor_default_4seq/                # Raw IQ-TREE output (precursor default)
├── trees_primary_high_clean/                    # Clean .treefile only (primary high) - 331 trees
├── trees_primary_default_clean/                 # Clean .treefile only (primary default) - 331 trees
├── trees_precursor_high_clean/                  # Clean .treefile only (precursor high) - 331 trees
└── trees_precursor_default_clean/               # Clean .treefile only (precursor default) - 331 trees
```

## Directory Structure

- `miRNA_mature_files/`: miRNA sequence files
  - `hsa_mature.fas`: Human mature miRNA sequences
  - `flanking.fas`: Primary sequences with flanking regions
  - `no_flanking.fas`: Precursor sequences without flanking
  - `all_mature.fas`: All mature miRNA sequences
  - `mirgenedb_family_mappings.tsv`: miRGeneDB family mappings
- `evo_scripts/`: Pipeline scripts
  - `annotate_dataset.py`: Annotate dataset with miRGeneDB IDs
  - `get_orthologues.py`: Retrieve orthologous sequences
  - `annotate_precursors.py`: Add secondary structure predictions
  - `run_alignment.sh`: Run multiple sequence alignments
  - `get_alignments.py`: Process and filter alignment results
  - `get_trees.py`: Extract clean .treefile outputs from IQ-TREE results
- `run_msa.sh`: Complete MSA pipeline script
- `run_trees.sh`: Phylogenetic tree construction script  
- `env.yaml`: Conda environment specification
