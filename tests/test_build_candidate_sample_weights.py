import csv
import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "build_candidate_sample_weights.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_candidate_sample_weights", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_weights_matches_ids_and_appends_unit_weights(tmp_path: Path) -> None:
    train = tmp_path / "train.tsv"
    with train.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["manakov_row_id"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(
            [{"manakov_row_id": "10"}, {"manakov_row_id": "20"}]
        )

    weights, indices = MODULE.build_weights(train, {"20"}, 0.25, 2)

    assert weights.tolist() == [1.0, 0.25, 1.0, 1.0]
    assert indices == [1]
