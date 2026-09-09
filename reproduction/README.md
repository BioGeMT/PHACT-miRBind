# Retained analyses and the September follow-up

The [result-to-input map](RESULTS.md) identifies the assets for each result group
and records the limits of the reproduction checks.

Code is committed here. Inputs, controls, checkpoints, predictions, generated
figures, and the original source snapshots stay in a separate workspace.
Set its physical path before using these scripts:

```bash
export PHACT_WORKSPACE=/path/to/phact
# Optional if the benchmark TSVs are held elsewhere:
# export PHACT_DATASETS=/path/to/manakov_datasets
```

The default when unset is `~/phact-workspace`; no directory is created on import.
Install the project with `uv sync --locked`. Plotting and verification also need
`uv sync --locked --extra analysis`. The frozen study uses Python 3.13 on node 4.

`final_analysis/` contains the final report builder, QA, variant audit, and
variant figure script. Builders write to `$PHACT_WORKSPACE/analyses/final`;
the final report builder replaces its own superseded figures. Use a separate
workspace/output destination when exploring changes to final figures.

`followup_2026_09_06/` preserves the within-reference-base positional-control
method, fixed training protocol, serial runner, summary and analysis code,
and checks for the completed 18 runs. The original scripts and frozen package
remain unchanged with the run data, so their recorded hashes remain valid.
The current drivers resolve historical paths through `workspace.py`; they do
not rewrite manifests or checkpoints. The training driver uses the retained
frozen package under the study's `training/source/src` to preserve the study's
model implementation. The one-off distributed launch/handoff scripts remain
with the historical run records; they are not general-purpose launch tools.

Verify the retained study without training:

```bash
uv run python reproduction/followup_2026_09_06/analysis_2026-09-07/verify_results.py
uv run python reproduction/followup_2026_09_06/analysis_2026-09-07/verify_controls.py
uv run python reproduction/final_analysis/phact_final_qa.py
```

Verification writes reports beside the retained study. `run_suite.py` resumes
the fixed study using its existing frozen-input manifest; do not repurpose
its protocol or write experimental runs over completed study results.

Expected workspace:

```text
phact/
  data/inputs/{manakov_datasets,manakov_original_rows,reference,gse_conservation,score_import_20260813}
  data/splits/
  data/scores/
  data/caches/
  runs/baselines/
  runs/param1-training/
  runs/experiments/
  runs/followup-2026-09-06/{controls,training,analysis,analysis_2026-09-07}
  analyses/{final,supporting}/
  releases/{mirbench_v7,source}/
  archive/
```

The release directory remains the published package. Its two Agentomics
archives have equivalent members to the uploaded split archives, although
the compressed archive checksums differ. Keep the release checksum set with
its corresponding packaging. Workspace-specific retention and move records
belong in `archive/organization-records`, outside Git.
