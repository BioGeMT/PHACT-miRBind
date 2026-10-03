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

## Conservation-aware five-fold comparison

The `prepare-cv-folds`, `build-cv-cache` and `train-cv-mirbind` commands reproduce
the sequence-only versus target-PHACT comparison using the existing model classes
and training loop. Dataset files, caches, weights and results live outside the
repository.

`--data-root` expects the prepared Manakov layout:

```text
DATA/
  manakov/{train,test,leftout}.tsv
  datasets_p1/{train,validation,test,leftout}.csv
  mapping/mirna_positions.tsv
  scores/mirna_qntnorm_transformed.tsv.gz
```

The CSVs use the interaction-array format above and IDs such as
`manakov_train_1`. The previous validation CSV supplies original training rows;
its old role is discarded. Original train and test are pooled in their original
row order. Original leftout is excluded from fold preparation and balancing,
checked for exact mature-sequence overlap, and retained as an external final
evaluation set.

Each run holds out one of five whole-miRNA groups as unseen test. Approximately
10% of the remaining rows form validation using other whole miRNAs. Within the
fitting miRNAs, approximately 5% of eligible exact sequence pairs form seen test.
Those miRNAs also occur in actual training. All remaining rows form training.
Repeated exact miRNA–target sequence pairs stay together, including conflicting
labels. A seen-test miRNA needs at least 20 distinct pairs; each selected label
stratum retains a training pair.

The seeded optimizer balances interaction and positive counts, miRNA counts,
miRNA conservation, target conservation distributions and missing scores. For
each scored nucleotide, conservation is REF minus the mean of the other three
nucleotide scores. MiRNA summaries use mature and seed positions; target
summaries use the 50-nt mean, standard deviation, 90th percentile and score
coverage. Distinct miRNA profile means are combined by their median. Balance is
approximate because whole miRNAs cannot be split; diagnostics report residual
differences. No model performance guides the assignments. Target windows and
related miRNA families can cross roles; this is an exact mature-miRNA holdout.

```bash
uv run prepare-cv-folds --data-root /path/to/DATA --output-dir /path/to/folds
uv run build-cv-cache --data-root /path/to/DATA \
  --folds-dir /path/to/folds --output-dir /path/to/cv-cache
uv run train-cv-mirbind --variant seq_only --fold 1 \
  --cache-dir /path/to/cv-cache --folds-dir /path/to/folds \
  --output-dir /path/to/runs
uv run train-cv-mirbind --variant target_phact --fold 1 \
  --cache-dir /path/to/cv-cache --folds-dir /path/to/folds \
  --output-dir /path/to/runs
```

Repeat training for folds 1–5. Two tmux windows can run the variants concurrently,
with each window processing its folds sequentially. Both variants share a
content-checked compact cache. The cache retains base IDs and unique target
score arrays and constructs pair grids on the selected device. CUDA training
uses BF16; `--device cpu` is available for small checks. This cache is loaded into
device memory, rather than streamed; it must fit there.

Defaults match the comparison: split seed 43, training seed 42, Adam at 0.001,
batch size 512, at most 30 epochs, and patience 7 on validation average precision
(AUPRC). Common sequence-model weights start identically between variants, and
epoch row shuffles match. The target model adds A/C/G/T score channels and a
missingness channel, with 0.5 placeholders for missing quartets, broadcast over
the miRNA axis. phyloP, phastCons and miRNA PHACT are not model inputs in this
comparison; miRNA PHACT summaries are used for split balancing.

Seen test, unseen test and original leftout are evaluated only after restoring
the checkpoint selected by validation AUPRC. Each fold writes training logs,
its selected checkpoint, `evaluation.json`, prediction NPZs with pooled row
indices, and a final `complete.json` marker. Existing run directories are
rejected rather than overwritten or implicitly resumed. Fold preparation and
cache creation likewise reject existing outputs. `--reuse-features` skips score
extraction when rebuilding assignments from an unchanged, audited feature set.
The verification step checks every saved manifest ID, label, source pointer and
role, as well as whole-miRNA and exact-pair isolation.

For a five-fold comparison, match variants within each fold. Report mean AUPRC
and sample SD across the five evaluations. Relative gain is
`100 * (target_AP - sequence_AP) / sequence_AP` within each fold, then averaged.
This SD describes fold variation; it is not a confidence interval. Training sets
overlap, and the original leftout observations repeat across runs. The pooled
original test is now part of cross-validation, not a separate untouched test.

Run the focused input, split and training checks with:

```bash
uv run python -m unittest discover -s tests -v
```
