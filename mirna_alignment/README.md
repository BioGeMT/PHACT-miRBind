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
- 358 of 377 precursors had multiple sequences to align (resulting in 358 alignments)
- 279 alignments with 10+ sequences used for phylogenetics
- 279 phylogenetic trees built

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

## Running the Pipeline

### Complete Pipeline
```
./run_msa.sh
```

This single script runs the complete pipeline:
1. Annotates dataset with miRGeneDB IDs
2. Retrieves orthologous sequences (primary and precursor)
3. Annotates sequences with secondary structure for both primary and precursor sequences
4. Makes alignment script executable
5. Runs alignments with 4 parameter combinations:
   - 2 parameter sets (high/default) × 2 sequence types (primary/precursor)
6. Filters alignments

## Output Structure

```
output/
├── manakov_positives_annotated.tsv           # Annotated dataset
├── orthologues/
│   ├── primary/                              # Primary sequences with flanking
│   └── precursor/                            # Precursor sequences
├── annotated_primary/                        # Annotated primary sequences
├── annotated_precursor/                      # Annotated precursor sequences
├── alignments_primary_high/                  # High parameter alignments (primary)
├── alignments_primary_default/               # Default alignments (primary)
├── alignments_precursor_high/                # High parameter alignments (precursor)
├── alignments_precursor_default/             # Default alignments (precursor)
├── filtered_alignments_primary_high/         # Filtered high parameter alignments (primary)
├── filtered_alignments_primary_default/      # Filtered default alignments (primary)
├── filtered_alignments_precursor_high/       # Filtered high parameter alignments (precursor)
└── filtered_alignments_precursor_default/    # Filtered default alignments (precursor)
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
- `run_msa.sh`: Complete pipeline script
- `env.yaml`: Conda environment specification

## Features

- **Silent Execution**: All scripts run without verbose output or error messages
- **Multiple Parameter Sets**: Tests high parameter (strict) and default alignment stringency levels
- **Comprehensive Monitoring**: Detailed alignment success/failure statistics
- **Parallel Processing**: Efficient multi-core processing (64 workers by default)
- **Clean Output**: Organized results with clear naming conventions
- **Filtered Results**: Post-processing creates filtered alignment directories

## Results

The pipeline produces:
- Annotated miRNA dataset with miRGeneDB mappings
- Orthologous precursor sequences from multiple species
- Structure-annotated sequences
- Multiple sequence alignments with 4 different parameter combinations
- Filtered alignments ready for downstream analysis
- Comprehensive alignment statistics and success rates

