import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "find_error_training_neighbors.py"
SPEC = importlib.util.spec_from_file_location("find_error_training_neighbors", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_unique_kmers_normalizes_rna() -> None:
    assert MODULE.unique_kmers("auuA", 3) == {"ATT", "TTA"}


def test_longest_shared_segment_extends_seed_both_directions() -> None:
    first = "AAACCCGGGTTTAAA"
    second = "GGACCCGGGTTTACC"

    shared = MODULE.longest_shared_segment(first, second, seed_length=4)

    assert shared == 11


def test_longest_shared_segment_returns_zero_without_seed() -> None:
    assert MODULE.longest_shared_segment("AAAAAAAA", "CCCCCCCC", 4) == 0


def test_controls_match_label_and_family_counts() -> None:
    errors = pd.DataFrame(
        {"test_row_id": [1, 2, 3], "label": [0, 1, 1], "analysis_family": ["A", "A", "B"]}
    )
    pool = pd.DataFrame(
        {
            "test_row_id": [4, 5, 6, 7],
            "label": [0, 1, 1, 1],
            "analysis_family": ["A", "A", "B", "B"],
            "stacker_error_type": ["correct"] * 4,
        }
    )

    controls = MODULE.family_label_matched_controls(errors, pool, random_seed=42)

    assert controls.groupby(["label", "analysis_family"]).size().to_dict() == {
        (0, "A"): 1,
        (1, "A"): 1,
        (1, "B"): 1,
    }


def test_write_outputs_handles_no_training_neighbors(tmp_path: Path) -> None:
    errors = pd.DataFrame(
        {
            "test_row_id": [1],
            "analysis_set": ["priority_error"],
            "control_match": [""],
            "label": [1],
            "stacker_error_type": ["false_negative"],
            "analysis_signature": ["all_core_wrong"],
            "pred_stacker": [0.1],
            "gene": ["A" * 50],
            "noncodingRNA": ["C" * 23],
            "noncodingRNA_name": ["held-out"],
            "analysis_family": ["HELD-OUT"],
        }
    )

    summary, neighbors = MODULE.write_outputs(
        tmp_path,
        errors,
        heaps={},
        kmer_length=12,
        neighbors_per_label=3,
    )

    assert len(summary) == 1
    assert neighbors.empty
    assert list(neighbors.columns) == list(MODULE.NEIGHBOR_COLUMNS)
