# PHACT-miRBind

This repo contains four runnable miRBind-style model families:

- `PairwiseSeqCNN`: the seq-only pairwise CNN baseline with the original
  28x50 miRNA/target pair grid and 147,249 default parameters.
- `PairwiseConservationCNN`: the same pair-grid CNN with target-position
  `phyloP`, `phastCons`, or both appended as extra channels before the first
  convolution.
- `PairwisePhactCNN`: the pair-grid CNN augmented with position-specific PHACT
  channels from the miRNA, target, or both axes.
- `PairwiseRinalmoPhactFusion`: one shared, fully fine-tuned RiNALMo-micro
  backbone for the miRNA and target, PHACT-conditioned top-layer mixing and
  positional pooling, and a frozen pretrained miRBind branch fused before
  binary classification.

The original PHACTn workflows are kept under `PHACTn/`. The PyTorch cache,
training, and model implementations are under `src/phact_mirbind/`.

The exact five-model PHACT consensus parameter-1 release for miRBench v7,
including data preparation, training order, inference, and artifact validation,
is documented in [`mirbench_v7_param1/README.md`](mirbench_v7_param1/README.md).

The five published models are three PHACT-P1 CNN variants and two Agentomics
fusion models. The fully fine-tuned RiNALMo model above is a separate
experimental model family. The fixed September positional-control study and
final report scripts are documented under [reproduction/](reproduction/README.md).

## Layout

```text
src/phact_mirbind/
  cli/          command-line entry points
  data/         TSV row parsing, sequence normalization, pair encoding
  cache/        .pt cache writers, manifests, iterable datasets
  models/       sequence, conservation, PHACT CNNs, and RiNALMo fusion
  training/     shared train/eval loop, metrics, logging
```

## Setup

```bash
cd PHACT-miRBind
uv sync --locked --extra analysis
export PHACT_WORKSPACE=/path/to/phact
```

On node 4, the workspace is `/SCRATCH/dtzim01/phact`. Keep the checkout on
main and store inputs, caches, checkpoints, and figures in that workspace.
Shell training workflows require `PHACT_WORKSPACE`; their new outputs go
under `runs/new/`. Python CLIs take explicit input/output paths.

Use unique output directories for experiments. The retained release and fixed
follow-up directories are historical evidence, not destinations for new runs.

## Cache

Seq-only training uses a neutral pair cache:

```bash
uv run build-pair-cache \
  --input-file "$PHACT_WORKSPACE/data/inputs/manakov_original_rows/manakov_original_rows_train.tsv" \
  --output-dir "$PHACT_WORKSPACE/data/caches/pair_cache_original_rows/train" \
  --output-prefix train
```

Conservation training uses a pair + conservation cache:

```bash
uv run build-conservation-cache \
  --input-file "$PHACT_WORKSPACE/data/inputs/manakov_original_rows/manakov_original_rows_train.tsv" \
  --output-dir "$PHACT_WORKSPACE/data/caches/conservation_cache_original_rows/train" \
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
PHACT_MODELS=CountNodes_3 scripts/build_phact_full_cache.sh
PHACT_MODELS=CountNodes_3 scripts/train_phact_full_cache.sh
```

`train-phact-mirbind --phact-channel-mode` selects `mirna`, `target`, or
`both`. `--phact-reduction` on cache construction selects the full nucleotide
scores, `actual_margin`, or `alt_mean` representation.

`--initial-checkpoint-mode widen` function-preservingly initializes a wider
filter stack from `--initial-checkpoint`; keep the embedding dimension and
input PHACT channel count unchanged, and only increase `--filter-sizes`.
`--additional-train-cache` may be repeated to mix compatible training-only
caches. `--training-target-shift-max N` applies neutral-padded target-axis
translation augmentation only to training batches. Noisy sampled negatives can
be studied without relabeling through `--negative-label-smoothing` or
`--focal-gamma`; both default to ordinary binary cross-entropy behavior.

### Fully fine-tune RiNALMo-micro

The RiNALMo fusion model consumes the compact per-position tensors from an
existing PHACT cache. It recovers both nucleotide streams from the cached pair
grid, so old caches do not need to be rebuilt. It fine-tunes all RiNALMo layers
with a lower, layer-wise-decayed learning rate while training the PHACT pooling
and fusion head at a higher learning rate. The supplied miRBind checkpoint is
kept frozen and in evaluation mode.

```bash
uv run train-rinalmo-phact-mirbind \
  --train-cache "$PHACT_WORKSPACE/runs/param1-training/cache/param_1_target_score/train" \
  --val-cache "$PHACT_WORKSPACE/runs/param1-training/cache/param_1_target_score/val" \
  --test-cache "$PHACT_WORKSPACE/runs/param1-training/cache/param_1_target_score/test" \
  --leftout-cache "$PHACT_WORKSPACE/runs/param1-training/cache/param_1_target_score/leftout" \
  --mirbind-checkpoint "$PHACT_WORKSPACE/runs/baselines/main_repo_outputs/seq_only/pairwise_seq_model_20260629_201939.pt" \
  --output-dir "$PHACT_WORKSPACE/runs/new/rinalmo_finetune" \
  --gradient-checkpointing \
  --progress-bar
```

`--additional-train-cache` may be repeated; compatible shard lists are mixed as
one training dataset while validation, test, and leftout remain unchanged.
`--encoding-mode cross` places both RNAs in one RiNALMo context so attention can
cross molecules. `--initial-checkpoint` starts a new run from a prior full
RiNALMo-PHACT checkpoint, which is useful for a conservative augmentation
fine-tune. `--phact-baseline-checkpoint` turns the new head into a zero-initialized
residual correction on top of an existing PHACT CNN; pair it with
`--freeze-rinalmo` for a lower-risk, faster head-only experiment.

Defaults are RiNALMo-micro, BF16 on CUDA, a `2e-6` top-backbone learning rate,
`0.85` layer-wise decay, PHACT-conditioned mixing of the top four layers, a
`1e-4` fusion-head learning rate, and gradient-norm clipping at `1.0`. Use
`--no-bfloat16` when the selected GPU does not support BF16. The pretrained
model is downloaded automatically by Hugging Face on the first run; the current
local cache uses about 128 MB.

`multimolecule` provides the maintained Hugging Face conversion used here and
is AGPL-3.0 licensed. The original RiNALMo implementation is Apache-2.0 and its
published pretrained weights are CC BY 4.0; check those terms before
redistributing a trained derivative or packaging this code into another
service.

Cache-building workflows use `data/inputs/` or `data/splits/` for source rows
and `data/caches/` for new caches inside the workspace. The original P1 run
keeps its own cache and prepared-data subdirectories together for provenance.

## Test

```bash
uv run pytest -q tests
```

## Data workspace and retained analyses

Keep large inputs and generated results in a separate workspace. Set
`PHACT_WORKSPACE` before using the shell workflows; new runs go under
`$PHACT_WORKSPACE/runs/new`. The retained follow-up and final-analysis scripts
are documented in [reproduction/README.md](reproduction/README.md).

The target-score builder requires explicit train/test/leftout, raw target-score,
and output paths. The legacy evaluation helper requires `--results-dir`,
`--eval-dir`, and `--output-dir`; neither assumes an old home data directory.
