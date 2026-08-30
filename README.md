# PHACT-miRBind

This repo contains three runnable miRBind-style model families:

- `PairwiseSeqCNN`: the seq-only pairwise CNN baseline with the original
  28x50 miRNA/target pair grid and 147,249 default parameters.
- `PairwiseConservationCNN`: the same pair-grid CNN with target-position
  `phyloP`, `phastCons`, or both appended as extra channels before the first
  convolution.
- `PairwisePhactCNN`: the pair-grid CNN augmented with position-specific PHACT
  channels from the miRNA, target, or both axes.

The original PHACTn workflows are kept under `PHACTn/`. The PyTorch cache,
training, and model implementations are under `src/phact_mirbind/`.

The exact five-model PHACT consensus parameter-1 release for miRBench v7,
including data preparation, training order, inference, and artifact validation,
is documented in [`mirbench_v7_param1/README.md`](mirbench_v7_param1/README.md).

## Layout

```text
src/phact_mirbind/
  cli/          command-line entry points
  data/         TSV row parsing, sequence normalization, pair encoding
  cache/        .pt cache writers, manifests, iterable datasets
  models/       seq-only, conservation, and PHACT CNNs
  training/     shared train/eval loop, metrics, logging
```

## Setup

```bash
cd PHACT-miRBind
uv sync
```

## Cache

Seq-only training uses a neutral pair cache:

```bash
uv run build-pair-cache \
  --input-file data/presplit_original_rows/manakov_original_rows_train.tsv \
  --output-dir data/pair_cache_original_rows/train \
  --output-prefix train
```

Conservation training uses a pair + conservation cache:

```bash
uv run build-conservation-cache \
  --input-file data/presplit_conservation_original_rows/manakov_original_rows_train.tsv \
  --output-dir data/conservation_cache_original_rows/train \
  --output-prefix train \
  --conservation-features phylop,phastcons
```

`phyloP` is normalized as `clamp(score / 10, -1, 1)`. `phastCons` is used as
provided, with missing values filled as `0.5`.

PHACT cache construction keeps the neutral score fill used by earlier runs and
also stores one binary missingness channel per PHACT score group. Newly built
caches append those masks to the model input, allowing a model to distinguish a
real neutral score from an unavailable score. Caches produced before this
addition remain readable and simply omit the mask channels.

## Train

Run the seq-only baseline:

```bash
scripts/run_seq_original_split_params.sh
```

Run the conservation-channel model:

```bash
scripts/run_conservation_original_split_params.sh
```

Use only one conservation source by setting `CONSERVATION_FEATURES`:

```bash
CONSERVATION_FEATURES=phylop scripts/run_conservation_original_split_params.sh
CONSERVATION_FEATURES=phastcons scripts/run_conservation_original_split_params.sh
```

Build and train the PHACT-channel model:

```bash
scripts/build_phact_full_cache.sh
scripts/train_phact_full_cache.sh
```

`train-phact-mirbind --phact-channel-mode` selects `mirna`, `target`, or
`both`. `--phact-reduction` on cache construction selects the full nucleotide
scores, `actual_margin`, or `alt_mean` representation.

The scripts create/reuse:

- `data/presplit_original_rows` and `data/pair_cache_original_rows`
- `data/presplit_conservation_original_rows` and
  `data/conservation_cache_original_rows`

## Test

```bash
uv run pytest
```
