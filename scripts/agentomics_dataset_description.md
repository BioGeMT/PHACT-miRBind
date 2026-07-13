# Manakov PHACT miRNA-target binding dataset

## Task

Binary classification of AGO2 eCLIP miRNA-target interaction samples from
Manakov et al.

- `label = 1`: positive AGO2 eCLIP interaction sample
- `label = 0`: negative interaction sample

The prediction unit is `id`. Training/evaluation labels are in `labels.csv`;
all inference-time inputs are under `input/`.

The public dataset contains `datasets/manakov_phact/train/`. Agentomics should
create its own train/validation split from this training data. Hidden evaluation
splits use the same structure under `test_datasets/manakov_phact/test/` and
`test_datasets/manakov_phact/leftout/`.

`id` prefixes such as `train_`, `test_`, and `leftout_` are split-specific row
identifiers.

## Layout

Each split folder contains:

```text
labels.csv
input/
  samples.tsv
  sample_mirna_candidates.tsv
  phact_mirna_positions.tsv
  phact_target_positions.tsv
  mirgenedb_premirna_orthologues.tsv
```

## Files

### `labels.csv`

| column | meaning |
|---|---|
| `id` | Sample ID. Joins to `samples.tsv.id`. |
| `label` | Binary label: `1` positive, `0` negative. |

### `input/samples.tsv`

One row per miRNA-target sample.

| column | meaning |
|---|---|
| `id` | Sample ID. |
| `gene` | Target RNA sequence, stored with DNA alphabet after U-to-T normalization. |
| `noncodingRNA` | Mature miRNA sequence, stored with DNA alphabet after U-to-T normalization. |
| `feature` | Original Manakov target-region feature annotation. |
| `dominant_region` | Dominant transcript/genomic region for the target site. |
| `gene_phyloP` | Original per-position target phyloP vector as a stringified numeric list. |
| `gene_phastCons` | Original per-position target phastCons vector as a stringified numeric list. |
| `mirgenedb_mature_id` | Exact MirGeneDB mature miRNA match; semicolon-separated for multimaps; `NA` if unmatched. |
| `mirgenedb_premirna_id` | Pre-miRNA ID(s) derived from `mirgenedb_mature_id`; semicolon-separated for multimaps; `NA` if unmatched. |
| `mirgenedb_family` | MirGeneDB family for the matched miRNA(s), or `NA`. |

### `input/sample_mirna_candidates.tsv`

Expanded MirGeneDB candidate table. This avoids parsing semicolon-separated
candidate IDs from `samples.tsv`.

| column | meaning |
|---|---|
| `id` | Sample ID. Joins to `samples.tsv.id`. |
| `candidate_index` | Candidate number within the sample; `0` when unmatched. |
| `candidate_count` | Number of candidate mature IDs; `0` when unmatched. |
| `mirgenedb_mature_id` | One candidate mature miRNA ID, or `NA` when unmatched. |
| `has_phact_profile` | `1` if this candidate has miRNA PHACT rows, else `0`. |

Multimapped samples have multiple rows with the same `id`. No candidate is
selected or averaged during dataset construction.

### `input/phact_mirna_positions.tsv`

One row per MirGeneDB mature miRNA ID and mature-miRNA position.

| column | meaning |
|---|---|
| `mirgenedb_mature_id` | Mature miRNA ID. Joins to `sample_mirna_candidates.tsv.mirgenedb_mature_id`. |
| `mirna_position_1based` | 1-based position within the mature miRNA sequence. |
| `actual_nt` | Observed nucleotide at this miRNA position. |
| `phact_<model>_A/C/G/T` | Quantile-normalized PHACT score for each nucleotide state. |

miRNA PHACT source table:
`reports/phact_score_ranges/phact_mirna_arm_position_qntnorm_transformed_all_models.tsv`.

Included miRNA PHACT models:

```text
CountNodes_2, 0_MinNode_Mix, max05_Gauss, 0, CountNodes_3,
0_MinNode_Mix2, CountNodes_4, 1, 0p5, mean, 2, 5, 0p1,
median, 3, CountNodes_1
```

Each model has `_A`, `_C`, `_G`, and `_T` columns. `0.5` and `0.1` from the
source table are written as `0p5` and `0p1` in column names. `phact_mean_*` is
a PHACT model name, not an average computed during dataset construction.

### `input/phact_target_positions.tsv`

One row per sample and target-sequence position.

| column | meaning |
|---|---|
| `id` | Sample ID. Joins to `samples.tsv.id`. |
| `target_position_1based` | 1-based position within `samples.tsv.gene`. |
| `actual_nt` | Observed target nucleotide at this position. |
| `score_A` | Target PHACT score for nucleotide A. |
| `score_C` | Target PHACT score for nucleotide C. |
| `score_G` | Target PHACT score for nucleotide G. |
| `score_T` | Target PHACT score for nucleotide T. |

Target PHACT source table:
`reports/phact_score_ranges/phact_target_manakov_position_qntnorm_transformed_scores.tsv`.

Target rows are sample-specific because each sample has its own 50-nt target
window. If no finite target PHACT score was available for a position, all four
score columns are `NA`.

### `input/mirgenedb_premirna_orthologues.tsv`

MirGeneDB precursor-level orthologue presence matrix.

| column | meaning |
|---|---|
| `mirgenedb_premirna_id` | Human MirGeneDB pre-miRNA ID. |
| `family` | MirGeneDB family. |
| `orthologue_species_count_excluding_hsa` | Number of non-human species columns with value `1`. |
| species columns | Binary `1/0` presence indicators from MirGeneDB Orthologues lists. |

If `samples.tsv.mirgenedb_premirna_id` contains semicolon-separated IDs, split
the cell before joining to this table.

## Joins

```text
labels.csv.id -> samples.tsv.id

samples.tsv.id
  -> sample_mirna_candidates.tsv.id
  -> sample_mirna_candidates.tsv.mirgenedb_mature_id
  -> phact_mirna_positions.tsv.mirgenedb_mature_id

samples.tsv.id -> phact_target_positions.tsv.id

samples.tsv.mirgenedb_premirna_id
  -> split semicolon-separated IDs if needed
  -> mirgenedb_premirna_orthologues.tsv.mirgenedb_premirna_id
```

## PHACT Scores

PHACTn stands for PHylogeny-Aware Computation of Tolerance for Nucleotide
substitutions. The PHACT tables provide nucleotide-specific evolutionary
features derived from orthologous sequence alignments and phylogenetic context.

Conceptually, PHACT asks how tolerant an aligned sequence position appears to be
to different nucleotide states, given the pattern seen across orthologues and
the phylogenetic relationships among species. Positions that are strongly
preserved across evolution, or where alternative nucleotide states are less
compatible with the phylogenetic pattern, can be interpreted as more
evolutionarily constrained. This kind of constraint is often useful as a proxy
for functional importance, because substitutions at important nucleotides are
more likely to have been selected against over evolutionary time.

Each position has four scores, one for each possible nucleotide state: `A`,
`C`, `G`, and `T`. The `actual_nt` column records the nucleotide observed in the
miRNA or target sequence at that position. The dataset keeps all four
nucleotide-state scores rather than reducing them to a single summary value.

miRNA positions include 16 named PHACT models. Target positions include one
qntnorm-transformed A/C/G/T score set.

## Missing Values

- `samples.tsv`: `NA` in MirGeneDB annotation columns means the mature miRNA
  sequence had no exact MirGeneDB mature match.
- `sample_mirna_candidates.tsv`: unmatched samples use
  `mirgenedb_mature_id=NA` and `has_phact_profile=0`.
- `phact_mirna_positions.tsv`: only available miRNA PHACT profiles are written;
  candidates with `has_phact_profile=0` have no rows here.
- `phact_target_positions.tsv`: `NA` in score columns means no target PHACT
  score was available for that target position.

## Supplementary Reference Methods

The dataset includes `supplementary/` with compact reference method material.

- `reference_methods/DiscrimAlign/`: compact DiscrimAlign source/scoring
  snapshot from `https://github.com/BioGeMT/DiscrimAlign`, commit
  `2f69b6664e86039919793e651f89602063d5433d`, plus selected trained miRNA
  model pickles.
- `reference_methods/miRBind2_seq_only/`: miRBind2 sequence-only pairwise
  representation/model files plus pretrained checkpoint
  `pairwise_onehot_model_20260105_200141.pt`.
