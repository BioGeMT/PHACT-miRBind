"""Shared dataset constants and column helpers."""

from __future__ import annotations

BASES = ("A", "C", "G", "T")
PAIR_BASES = ("A", "T", "C", "G")
PADDING_PAIR_INDEX = 17
NUM_PAIR_CLASSES = 18
DEFAULT_TARGET_LENGTH = 50
DEFAULT_MIRNA_LENGTH = 28
TARGET_SEQUENCE_COLUMN = "gene"
LABEL_COLUMN = "label"
MIRNA_SEQUENCE_COLUMNS = ("mirna", "noncodingRNA")
GENE_PHYLOP_COLUMN = "gene_phyloP"
GENE_PHASTCONS_COLUMN = "gene_phastCons"

MANAKOV_COLUMNS = (
    "gene",
    "noncodingRNA",
    "noncodingRNA_name",
    "noncodingRNA_fam",
    "feature",
    "label",
    "chr",
    "start",
    "end",
    "strand",
    "Nunique",
    "dominant_region",
    "regions_present",
    "read_start_in_sel_tx_1based",
    "read_end_in_sel_tx_1based",
    "gene_cluster_ID",
    "gene_phyloP",
    "gene_phastCons",
)


def build_pair_to_index() -> dict[tuple[str, str], int]:
    """Return miRNA-base/target-base pair indices matching miRBind."""
    return {
        (mirna_base, target_base): idx
        for idx, (mirna_base, target_base) in enumerate(
            (mirna_base, target_base)
            for mirna_base in PAIR_BASES
            for target_base in PAIR_BASES
        )
    }


def column_index(header: list[str]) -> dict[str, int]:
    return {column: idx for idx, column in enumerate(header)}


def resolve_mirna_column(columns: dict[str, int]) -> str:
    """Return the miRNA sequence column name used by this Manakov file."""
    for column in MIRNA_SEQUENCE_COLUMNS:
        if column in columns:
            return column
    choices = " or ".join(MIRNA_SEQUENCE_COLUMNS)
    raise ValueError(f"Missing required column: {choices}")
