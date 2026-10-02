# PHACT-miRBind

Implementation of PHACT nucleotide scoring, miRNA alignment, and PHACT-augmented
miRNA–target binding models.

## Layout

- [`PHACTn/`](PHACTn/README.md): original PHACTn scoring workflows and configurations.
- [`mirna_alignment/`](mirna_alignment/README.md): orthologue retrieval, precursor processing, alignments and trees.
- `phact_mirbind/`: PHACT CNN and RiNALMo–PHACT fusion implementations, cache readers/writers, training and prediction.
- `agentomics/`: single-candidate, layer-mix and multi-candidate PHACT fusion implementations with their training and inference code.

The PHACT CNN supports miRNA-only, target-only or combined score channels.
Shared sequence-only and conservation CNN classes remain because the PHACT
models reuse their architecture and pretrained branches. Model implementations
and input representations are preserved; weights and datasets are supplied
externally.

## PHACTn and alignment

Follow the environment and workflow instructions in the respective directory
READMEs. Their implementation files are retained unchanged.

## Binding-model setup

```bash
uv sync --locked
```

The root environment covers `phact_mirbind/`. The Agentomics environment is
recorded in `agentomics/model5/environment.yml`.

## Interaction-array input

All PHACT trainers accept a CSV or TSV with one row per interaction. Arrays are
written as `[0.2,NaN,0.8,...]` inside cells. CSV writers must quote array cells;
TSV writers do not need to quote commas. Missing whole tracks can be empty or
all-NaN arrays. The reader also accepts `null` entries.

| Columns | Contents |
| --- | --- |
| `id` | Optional unique row ID; defaults to the 1-based input row number |
| `gene` | Target sequence, exactly 50 nucleotides |
| `noncodingRNA` | Mature miRNA sequence (`mirna` is also accepted) |
| `label` | 0 or 1 |
| `mirna_phact_A`, `mirna_phact_C`, `mirna_phact_G`, `mirna_phact_T` | Four P1 score arrays, each matching the mature sequence length, or already padded to 28 positions |
| `target_phact_A`, `target_phact_C`, `target_phact_G`, `target_phact_T` | Four target score arrays, each 50 positions; targets do not have a P1 parameter |
| `gene_phyloP`, `gene_phastCons` | Native target conservation arrays, each 50 positions |
| `feature`, `dominant_region` | Optional metadata used by Agentomics; absent categories become `NA` |

Array positions follow the corresponding sequence from left to right. Scores
must already use the intended normalization/transformation; the loaders do not
transform PHACT scores again. Prepared P1 data uses transformed wtNT miRNA scores
and the latest mapped transformed target scores. U is converted to T. miRNAs
are padded or truncated to 28 positions together with their scores. At a missing
PHACT position, all four nucleotide entries must be NaN. Core CNN inputs use 0.5
for missing scores plus an explicit missingness mask, including padded positions.
Agentomics retains its own fitted preprocessing and missingness masks.

```bash
uv run train-phact-mirbind \
  --train-file /path/to/train.csv \
  --val-file /path/to/validation.csv \
  --phact-channel-mode both \
  --output-dir /path/to/training-output
```

Use `--phact-channel-mode mirna`, `target` or `both`. Only the selected score
axes are required. phyloP/phastCons columns may remain in the table for all
models. To include those target tracks in a PHACT CNN or RiNALMo fusion, add
`--conservation-features phylop,phastcons`; core compact inputs use
`clip(phyloP/10,-1,1)` and native phastCons, with separate missingness masks.
The default PHACT models use sequence and their selected PHACT channels.

`--test-file` and `--leftout-file` are optional final evaluation inputs. Validation
selects checkpoints. Tables are validated and converted to reusable tensor
shards automatically under `OUTPUT_DIR/input_cache`; `--input-cache-dir` can
share that directory across runs. Input content and representation settings
identify caches. Length errors, partial score quartets, duplicate IDs, infinite
scores and invalid labels are rejected.

The RiNALMo trainer accepts the same file arguments and still requires
`--mirbind-checkpoint`. Agentomics training entry points accept the same CSV/TSV
paths with their existing `--train-data` and `--validation-data` arguments.
Install this repository into their Python environment using
`pip install --no-deps -e /path/to/PHACT-miRBind`. They prepare their split
representation under an `agentomics_input_cache` beside the output artifacts.
Their pretrained checkpoint arguments remain required. A flat interaction row
supplies one profile per miRNA; the multi-candidate architecture uses that one
candidate. Existing split-folder input preserves multiple locus candidates.

## Existing cache input and prediction

Build a compact cache from a Manakov-format row TSV and two row-position score
TSVs. Supply both score-table paths explicitly; the miRNA table contains named
PHACT nucleotide-score columns, while targets may use `score_A/C/G/T` columns.

```bash
uv run build-phact-cache   --input-file /path/to/train.tsv   --output-dir /path/to/cache/train   --output-prefix train   --phact-split train   --phact-models param_1   --target-phact-models target_score   --mirna-phact-file /path/to/mirna_row_scores.tsv   --target-phact-file /path/to/target_row_scores.tsv

uv run train-phact-mirbind   --train-cache /path/to/cache/train   --val-cache /path/to/cache/val   --test-cache /path/to/cache/test   --leftout-cache /path/to/cache/leftout   --phact-channel-mode both   --output-dir /path/to/training-output

uv run predict-phact-mirbind   --checkpoint /path/to/model.pt   --cache /path/to/cache/test   --output /path/to/predictions.npz
```

Use `--phact-channel-mode mirna` or `target` to select a single score axis.
Missing-score masks and historical caches without masks are supported. Prediction
reads the checkpoint's channel configuration. Run each command with `--help`
for its complete input and model options.

The separate `train-rinalmo-phact-mirbind` command implements RiNALMo fusion;
`agentomics/` contains the standalone Agentomics training and inference entry
points. The shared sequence and conservation cache/training commands remain
available for their baseline components.
