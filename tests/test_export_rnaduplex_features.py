import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "export_rnaduplex_features.py"
SPEC = importlib.util.spec_from_file_location("export_rnaduplex_features", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_duplex_line() -> None:
    features = MODULE.parse_duplex_line(
        ".(((..((.(((...(((((.(.&.).)))))..))).))))).   "
        "7,29  :   3,22  (-13.30)"
    )

    assert features == (-13.3, 14.0, 7.0, 29.0, 23.0, 3.0, 22.0, 20.0)


def test_parse_positive_energy_with_padding() -> None:
    features = MODULE.parse_duplex_line(".((.&.)).  42,45  :   1,4   ( 0.80)")

    assert features == (0.8, 2.0, 42.0, 45.0, 4.0, 1.0, 4.0, 4.0)
