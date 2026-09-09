# Retained studies and analyses

The maintained scoring, model, training, inference and five-model release code
stays in this repository. Historical study scripts live with the external data
workspace. The [result-to-input map](RESULTS.md) describes their inputs and outputs.

Workspace locations (relative to `PHACT_WORKSPACE`):

| Content | Location |
| --- | --- |
| Final report, variant figures and September study code, including tests | `reports/source/reproduction/` |
| Historical error-review and candidate-weight scripts and tests | `models/experiments/review-source/` |
| Historical preparation and launch recipes | `archive/legacy-recipes/` |
| September inputs, frozen package, 18 trained runs and analysis outputs | `models/positional-controls/` |
| Score comparisons and figures | `scores/comparisons/` |
| Model comparison tables and figures | `models/comparisons/` |
| Combined reports | `reports/` |
| Published five-model artifacts | `models/benchmark/artifacts/` |

From the maintained checkout, set both physical locations and verify the retained
assets using the checkout's environment:

```bash
export PHACT_REPO="$PWD"
export PHACT_WORKSPACE=/path/to/phact
uv sync --locked --extra analysis
uv run pytest -q "$PHACT_WORKSPACE/reports/source/reproduction" \
  "$PHACT_WORKSPACE/models/experiments/review-source/tests"
uv run python "$PHACT_WORKSPACE/reports/source/reproduction/followup_2026_09_06/analysis_2026-09-07/verify_results.py"
uv run python "$PHACT_WORKSPACE/reports/source/reproduction/followup_2026_09_06/analysis_2026-09-07/verify_controls.py"
uv run python "$PHACT_WORKSPACE/reports/source/reproduction/final_analysis/phact_final_qa.py"
```

The relocated scripts require `PHACT_REPO`. Their internal directory layout must
stay intact. The September driver uses the frozen package in the retained study,
its original input manifest and fixed protocol. It is not a general training
launcher. Verification writes reports beside the study; figure builders replace
their own outputs, so use a separate output workspace for exploratory changes.
Historical recipes are retained for provenance, not as supported commands.

The source before relocation is retained in Git history at `c98ac2b` and in the
workspace preservation set. The cleanup record under
`archive/organization-records/2026-09-09/git-scope-cleanup/` stores an exact source
archive and the file hashes before and after the checkout-path adjustment.
The older preservation set predates this layout change; restore its source and
follow its original instructions, or also retain the cleanup supplement.
No new online archive has been uploaded.

The workspace is organized by subject: `scores/`, `datasets/`, `models/`,
`reports/`, and `archive/`. Shared scripts under `reports/source/reproduction/`
assemble and check results across subjects. Historical manifests retain their
original paths; the shared path resolver maps them to their current locations.
The topic-layout record and package supplement preserve this mapping.
