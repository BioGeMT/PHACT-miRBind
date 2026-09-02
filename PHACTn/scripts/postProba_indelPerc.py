import pandas as pd
from pathlib import Path
import argparse
import logging



def combine_probabilities(nt_probs_path, gap_probs_path, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    asr_df = pd.read_csv(nt_probs_path, sep="\t", comment="#")
    asr_df.rename(columns=lambda x: x.strip(), inplace=True)

    indel_df = pd.read_csv(gap_probs_path, sep="\t", comment="#")
    gap_perc_df = indel_df[["Node", "Site", "p_0"]]

    all_proba = pd.merge(asr_df, gap_perc_df, on=["Node", "Site"], how="inner")

    nan_cols = all_proba.columns[all_proba.isna().any()].tolist()
    if nan_cols:
        logging.warning(f"NaN values found in columns: {', '.join(nan_cols)}")

    all_proba.rename(columns={"p_0": "gap_proba"}, inplace=True)
    all_proba["no_gap_proba"] = 1 - all_proba["gap_proba"]

    nucleotides = ["A", "C", "G", "T"]
    for nt in nucleotides:
        all_proba.rename(columns={f"p_{nt}": f"{nt}_prior"}, inplace=True)
        # Calculate posterior as: prior * (1 - gap)
        all_proba[f"p_{nt}"] = all_proba[f"{nt}_prior"] * all_proba["no_gap_proba"]

    cols_to_keep = ["Node", "Site", "State", "p_A", "p_C", "p_G", "p_T"]
    all_proba[cols_to_keep].to_csv(output_path, sep="\t", index=False)


logging.basicConfig(
    format='[%(levelname)s] %(message)s',
    level=logging.INFO
)

def main():
    parser = argparse.ArgumentParser(description="Combine nucleotide and gap probabilities into posterior table")
    parser.add_argument("--nt_probs", type=str, required=True, help="Path to nucleotide probability table (e.g., from ASR)")
    parser.add_argument("--gap_probs", type=str, required=True, help="Path to indel/gap probability table")
    parser.add_argument("--binary_out", type=str, required=True, help="Output path for combined posterior probability table")
    args = parser.parse_args()

    logging.info(f"Reading nucleotide probabilities from: {args.nt_probs}")
    logging.info(f"Reading gap probabilities from: {args.gap_probs}")
    logging.info(f"Writing combined posteriors to: {args.binary_out}")

    combine_probabilities(args.nt_probs, args.gap_probs, args.binary_out)

    logging.info("Processing complete.")

if __name__ == "__main__":
    main()