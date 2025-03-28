# miRNA Phylogenetic Analysis Pipeline

This pipeline analyzes miRNA evolutionary relationships through multiple sequence alignment and phylogenetic tree construction.

Clone the repository and download the human mirna dataset (manakov_positives.tsv):

https://drive.google.com/drive/folders/14p99NI1y7rHrbHucbiIQMpqtYNTpuoMP

## Dependencies

First set up the conda environment with the env.yaml file:

```
conda env create -f env.yaml
conda activate PHACT-miRBind
```

Required packages:
- Python 3.8+
- pandas
- BioPython
- ViennaRNA
- LocARNA
- IQ-TREE
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

Maps miRNA sequences to miRGeneDB IDs and adds family information. While the input data uses miRBase-style annotations, we convert to miRGeneDB IDs because miRGeneDB provides better evolutionary annotations.

**New Columns Added**:
| Column          | Description                        | Example           |
|-----------------|------------------------------------|-------------------|
| mirgenedb_name  | Mature miRGeneDB entry name        | Hsa-Let-7-P1d_3p* |
| mirgenedb_id    | Precursor miRGeneDB entry name     | Hsa-Let-7-P1d     |
| mirgenedb_fam   | miRGeneDB family name              | Let-7             |

**Options**:
- `--fasta`: Input FASTA with mature miRNA sequences
- `--tsv`: Input TSV file
- `--mirgenedb`: miRGeneDB family mappings
- `--output`: Output file

### 2. Precursor Sequence Retrieval
**Script**: `evo_scripts/scrape_parallel.py`

Retrieves precursor sequences and orthologs from miRGeneDB.

```
>Hsa-Let-7-P1b
GGGUCUGUCCACCUGCCGCGCCCCCCGGGCUGAGGUAGGAGGUUGUAUAGUUGAGGAGGACACCCAAGGAGAUCACUAUACGGCCUCCUAGCUUUCCCCAGGCUGCGCCCUGCACGGGACGGGGCCC
>Bta-Let-7-P1b
CUGUCUGUCCACCUGCCGCGCCCCCCGGGCUGAGGUAGGAGGUUGUAUAGUUGAGGAGGACACCCAAGGAGAUCACUAUACGGCCUCCUAGCUUUCCCCAGGCUGCGCCCUGCACGGGACGGCCCGG

```

**Options**:
- `--input`: Annotated TSV file
- `--output_folder`: Output directory
- `--workers`: Number of cpus

### 3. Precursor Annotation
**Script**: `evo_scripts/annotate_precursors.py`

Adds secondary structure predictions and marks mature miRNA locations.

```
>Hsa-Let-7-P1b
GGGUCUGUCCACCUGCCGCGCCCCCCGGGCUGAGGUAGGAGGUUGUAUAGUUGAGGAGGACACCCAAGGAGAUCACUAUACGGCCUCCUAGCUUUCCCCAGGCUGCGCCCUGCACGGGACGGGGCCC
((.(((((((.(.(((.((((..((.(((..(((.((((((((((((((((.((((.(....))).......)))))))))))))))))).)))..))).))..))))...))).)))))))).)). #S
..............................AAAAAAAAAAAAAAAAAAAAAA.......................BBBBBBBBBBBBBBBBBBBBBB.............................. #1
..............................123456789abcdefghijklm.......................123456789abcdefghijklm.............................. #2

```

**Options**:
- `--input`: Input directory with precursor files
- `--output`: Output directory
- `--mature`: FASTA with mature sequences

### 4. Multiple Sequence Alignment
**Script**: `evo_scripts/run_alignment.sh`

Performs structure-aware alignments using MLocARNA on precursor sets.

```
CLUSTAL W --- LocARNA 2.0.1

Hsa-Let-7-P1b      ------GGGUCUGUCCACC-UGC-C----GCGCCC--------CC----CGGG-CUGAGGUAGGAGGUUGUAUAGUUGAGGA--GGA-------------------------------------------------------------CACC-C----AAGGAG---------A---------------------------------------------------UCA-CUAUACGGCCUCCUAGCUUUCC-C-CA-G----G---CUGC---------GCC-CUGCACGGGACGGGGCCC------
Bta-Let-7-P1b      ------CUGUCUGUCCACC-UGC-C----GCGCCC--------CC----CGGG-CUGAGGUAGGAGGUUGUAUAGUUGAGGA--GGA-------------------------------------------------------------CACC-C----AAGGAG---------A---------------------------------------------------UCA-CUAUACGGCCUCCUAGCUUUCC-C-CA-G----G---CUGC---------GCC-CUGCACGGGACGGCCCGG------
```

**Parameters**:
- Input directory: annotated precursors
- Output directory: alignment results
- Number of processes: for parallelization

**MLocARNA Parameters**:
- `--struct-weight=300`: Increases weight for structural similarity (default=200). We use higher value to prioritize a bit structural conservation over sequence similarity.
- `--consensus-structure=alifold`: Uses Vienna RNAalifold.
- `--plfold-span=150`: Sets maximum span length for base pairs to 150 nucleotides, because we have sequences around 150nuc.
- `--write-structure`: Outputs structural information for visualization.
- `--stockholm`: Outputs alignment in Stockholm format with structure annotation.
- `--alifold-consensus-dp`: Uses dynamic programming for more accurate consensus structures.
- `--free-endgaps`: Allows free gaps at sequence ends.
- `--indel=-150`: Lower gap penalty (default=-500).
- `--indel-opening=-750`: Gap opening penalty that favors fewer but longer gaps.
- `--threads=1`: Single-threaded per alignment (we do parallelization at script level).

### 5. Alignment Organization
**Script**: `evo_scripts/find_alignments.py`

Finds alignment results and organizes them in a folder for further analysis.

**Options**:
- `--input`: Directory with alignment results
- `--output`: Output directory

### 6. Alignment Filtering
**Script**: `evo_scripts/filter_alignments.py`

Filters alignments to keep only those with >10 sequences.

**Options**:
- Input directory: alignment files
- Output directory: filtered alignments
- `--min`: Minimum sequences required (default: 10)

### 7. Phylogenetic Tree Construction
**Script**: Uses IQ-TREE via `run_iqtree.sh`

Builds evolutionary trees using IQ-TREE, processing filtered alignments in parallel.

**IQ-TREE Parameters**:
- `-m MFP`: ModelFinder Plus for best model selection
- `-nt 1`: Use 1 CPU thread per tree (we do parallelization across trees)
- `-pre`: Set prefix for output files

## Running the Pipeline

The pipeline is divided into two main parts:

### 1. MSA Generation (run_msa.sh)
```
./run_msa.sh
```
This script performs steps 1-4:
- Annotates the dataset
- Retrieves precursor sequences
- Annotates precursors with structure
- Performs multiple sequence alignment

### 2. Phylogenetic Analysis (run_iqtree.sh)
```
./run_iqtree.sh
```
This script performs steps 5-7:
- Finds and organizes alignments
- Filters alignments (minimum 10 sequences)
- Constructs phylogenetic trees using IQ-TREE

## Directory Structure

- `miRNA_mature_files/`: miRNA sequence files
- `evo_scripts/`: Pipeline scripts
- `run_msa.sh`: Script for the MSA generation part of the pipeline
- `run_iqtree.sh`: Script for the phylogenetic analysis part

## Results

The pipeline produces:
- Annotated miRNA dataset
- Precursor sequences with orthologs
- Annotated precursor sequences
- Multiple sequence alignments
- Phylogenetic trees

Results available at: https://drive.google.com/drive/folders/14p99NI1y7rHrbHucbiIQMpqtYNTpuoMP?usp=sharing
