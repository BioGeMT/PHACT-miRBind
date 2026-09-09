import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "analyze_manakov_test_failures.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_manakov_test_failures", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_rank_violation_counts() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.int8)
    predictions = np.array([0.1, 0.8, 0.2, 0.9], dtype=np.float32)

    violations = MODULE.rank_violation_counts(labels, predictions)

    assert violations.tolist() == [0, 1, 1, 0]


def test_family_value_prefers_mirgenedb_and_falls_back() -> None:
    assert (
        MODULE.family_value(
            {"mirgenedb_family": "MIR-17", "noncodingRNA_fam": "mir-17"}
        )
        == "MIR-17"
    )
    assert (
        MODULE.family_value(
            {"mirgenedb_family": "NA", "noncodingRNA_fam": "mir-21"}
        )
        == "MIR-21"
    )


def test_error_signatures_identify_rescues() -> None:
    labels = np.array([1, 1, 1, 1, 1], dtype=np.int8)
    predictions = {
        "original_phact": np.array([0.1, 0.1, 0.9, 0.9, 0.1]),
        "seq_only": np.array([0.1, 0.1, 0.1, 0.9, 0.1]),
        "conservation_phact": np.array([0.1, 0.1, 0.9, 0.1, 0.9]),
        "rinalmo": np.array([0.1, 0.9, 0.1, 0.1, 0.1]),
        "rinalmo_cross": np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
    }
    core_models = list(predictions)

    signatures, correct_count = MODULE.assign_error_signatures(
        labels, predictions, core_models
    )

    assert signatures.tolist() == [
        "all_core_wrong",
        "rinalmo_rescue",
        "phact_rescue",
        "conservation_hurts",
        "conservation_rescue",
    ]
    assert correct_count.tolist() == [0, 1, 2, 2, 1]
