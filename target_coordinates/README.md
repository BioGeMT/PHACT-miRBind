# Target Coordinates to BED with Validation

Converts genomic TSV files to BED format with optional sequence validation.

## Setup

```bash
pip install pandas pyfaidx requests
chmod +x get_gene_coordinates.sh
```

## Usage

```bash
./get_gene_coordinates.sh manakov_positives.tsv
```


## Input format

TSV with columns: chr, start, end, gene, strand

```
chr	start	end	gene	strand
10	102141236	102141285	CTACCTGATCC...	+
```

## Output

- `*_genes.bed` - **Main output**: BED format file
- `*_validation.tsv` - Validation results

## BED output format

5-column BED format:

```
10	102141235	102141285	CTACCTGATCC...	+
```

Columns: chr, start, end, sequence, strand

Note: Start coordinate is reduced by 1 (TSV uses 1-based, BED uses 0-based)
