# Results, inputs, and reproduction scope

All workspace paths below are relative to `PHACT_WORKSPACE`. Keep each retained
run together with its inputs, source snapshot, protocol, checkpoints, and outputs.
The table maps result groups; the workspace audit additionally records individual
file paths, sizes and hashes for the 50 model-comparison rows and 18 follow-up runs.

| Result group | Code in this repository | Required retained inputs | Outputs to preserve |
| --- | --- | --- | --- |
| Five-model P1 release | `mirbench_v7_param1/` | `runs/param1-training/{data,cache,agentomics_splits,agentomics_eval}`, original benchmark TSVs, target PHACT scores, consensus P1 scores, pretrained components | `releases/mirbench_v7/`, including all five weights, predictions, metrics and metadata; `releases/source/` records the original source |
| Fifty-model comparison | `reproduction/final_analysis/phact_final_builder.py` | Metric/prediction sources identified in `analyses/final/all_model_test_leftout_auprc.tsv`; baseline, experiment and Agentomics result directories | Final comparison TSV/CSV/Markdown and figures in `analyses/final/` |
| P1 profile and subgroup evidence | `reproduction/followup_2026_09_06/analysis/analyze_param1_evidence.py` | Old and P1 arm score tables, P1 evaluation caches, benchmark labels, baseline and release predictions | All files in `analyses/final/p1_evidence/`; the retained report used 1,000 cluster-bootstrap replicates |
| Variant and miRNA profile figures | `reproduction/final_analysis/phact_variant_groundtruth_audit.py`, `phact_variant_validation_plot.py`, `phact_final_builder.py` | `data/inputs/score_import_20260813/`, `data/inputs/consensus_tree_scores.zip`, `data/inputs/reference/` | Variant/profile plot data, figures and audit records in `analyses/final/` |
| Fixed positional-control study | `reproduction/followup_2026_09_06/{controls,training}/` | P1 cache, fixed control sidecars, conservation cache, frozen package, protocol and original driver hashes | Entire `runs/followup-2026-09-06/`, especially all 18 checkpoint/result/prediction sets and aggregate tables |
| Exposure, subgroup and pooled-AP analyses | `reproduction/followup_2026_09_06/analysis/` | Benchmark TSVs, old/P1 caches and retained predictions; exact consumed files are listed in the two input manifests | Joined predictions, tables, audits and input manifests in the follow-up's `analysis/` directory |
| Historical augmentation experiments | `scripts/build_augmented_training_rows.py` and related filtering/scoring/training scripts | `data/inputs/gse_conservation/`, original Manakov rows, exclusions, score tables, and each version's prepared rows/caches | Version-specific inputs and results under `runs/experiments/`; v1, v2 and positive-precedence v3 select different rows |

## What can be verified now

The retained study verifiers check checkpoint and frozen-input hashes, model
initialisation, prediction labels, reported metrics and control tensors. They
resolve historical paths without rewriting the recorded scientific inputs.
`phact_final_qa.py` checks the final report tables, their source artifacts, and
figure dimensions. Run the commands in [README.md](README.md).

A clean environment installed from the lockfile with the analysis extra has been
checked on Linux/Python 3.12. The test suite and a separate synthetic CLI smoke
check passed: pair/conservation/PHACT cache creation, one CPU epoch for each CNN
family, final evaluation and sequence checkpoint inference. The retained 18-run
study and evaluation controls also passed verification in that fresh environment.
These checks do not reproduce full GPU training or establish bitwise agreement
across CUDA versions. The original study environment used Python 3.13.

## What the repository alone does not provide

The repository does not contain the scientific data, pretrained weights, cached
foundation features, or the frozen follow-up package. The follow-up driver is
for the retained fixed study; it expects its existing workspace and frozen-input
manifest. A fresh clone without those assets cannot reconstruct the complete
study. The two released Agentomics models have their own documented environment.

Online release weights and predictions have been located in Drive. Metadata and
file sizes establish their listed coverage; the remote binary files have not all
been downloaded and independently hashed. Complete online coverage of the current
raw inputs, final analyses and September follow-up has not been established.
Keep the scratch copies. Uploading a new full-study archive is a separate action.

The workspace audit under `archive/organization-records/2026-09-09/follow-through/`
contains the exact input/result index, fresh-environment logs, online folder
inventory, and cleanup decisions. Its machine-specific details remain outside Git.
