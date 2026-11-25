# PHACT-miRBind

Implementation of PHACT nucleotide scoring, miRNA alignment, and PHACT-augmented
miRNA–target binding models.

## Layout

- [`PHACTn/`](PHACTn/README.md): original PHACTn scoring workflows and configurations.
- [`mirna_alignment/`](mirna_alignment/README.md): orthologue retrieval, precursor processing, alignments and trees.
- `phact_mirbind/`: sequence-only and target-PHACT CNNs, five-fold preparation, cache and training.

The target-PHACT CNN extends the conservation CNN class, which adds score
channels to the sequence-only CNN architecture. Weights and datasets are supplied
externally.

## PHACTn and alignment

Follow the environment and workflow instructions in the respective directory
READMEs. Their implementation files are retained unchanged.

## Binding-model setup

```bash
uv sync --locked
```

The root environment covers `phact_mirbind/`.

## Interaction-array input

The five-fold commands read a CSV with one row per interaction. Arrays are
written as `[0.2,NaN,0.8,...]` inside quoted cells.

| Columns | Contents |
| --- | --- |
| `id` | Row ID such as `manakov_train_1` |
| `gene` | Target sequence, exactly 50 nucleotides |
| `noncodingRNA` | Mature miRNA sequence |
| `label` | 0 or 1 |
| `target_phact_A`, `target_phact_C`, `target_phact_G`, `target_phact_T` | Four target score arrays, each 50 positions; targets do not have a P1 parameter |

Array positions follow the target sequence from left to right. Scores must
already use the intended normalization/transformation; the loaders do not
transform PHACT scores again. Prepared data uses the latest mapped transformed
target scores. U is converted to T. miRNAs are padded or truncated to 28
positions. At a missing PHACT position, all four nucleotide entries must be NaN.
CNN inputs use 0.5 for missing scores plus an explicit missingness channel.

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

`predict-cv-mirbind` applies a saved checkpoint to a CSV in the interaction-array
format above; `label` is not needed. It reads the variant and architecture from
the checkpoint, encodes inputs as the cache does, uses BF16 on CUDA as in
training, and writes a TSV with `id` and `probability`.

```bash
uv run predict-cv-mirbind --checkpoint /path/to/best_model.pt \
  --input /path/to/interactions.csv --output /path/to/predictions.tsv
```

Run the focused input, split and training checks with:

```bash
uv run python -m unittest discover -s tests -v
```
