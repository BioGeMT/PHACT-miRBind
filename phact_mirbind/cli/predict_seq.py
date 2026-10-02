"""Append seq-only miRBind predictions to TSV rows."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from phact_mirbind.data.columns import (
    DEFAULT_MIRNA_LENGTH,
    DEFAULT_TARGET_LENGTH,
    MIRNA_SEQUENCE_COLUMNS,
    NUM_PAIR_CLASSES,
    PADDING_PAIR_INDEX,
    TARGET_SEQUENCE_COLUMN,
    resolve_mirna_column,
)
from phact_mirbind.models.seq_only import PairwiseSeqCNN
from phact_mirbind.training.loop import resolve_device

PAIR_BASE_TO_CODE = {
    ord("A"): 0,
    ord("T"): 1,
    ord("U"): 1,
    ord("C"): 2,
    ord("G"): 3,
}
BASE_CODE_LOOKUP = np.full(256, -1, dtype=np.int16)
for _base, _code in PAIR_BASE_TO_CODE.items():
    BASE_CODE_LOOKUP[_base] = _code


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device)
    predict_tsv(
        input_file=args.input_file,
        output_file=args.output_file,
        model=model,
        device=device,
        batch_size=args.batch_size,
        prediction_column=args.prediction_column,
        progress_every_rows=args.progress_every_rows,
        force=args.force,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run seq-only miRBind inference and append probabilities to a TSV"
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--prediction-column", type=str, default="mirbind2_prediction")
    parser.add_argument("--progress-every-rows", type=int, default=100_000)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> PairwiseSeqCNN:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    model_params = dict(checkpoint.get("model_params") or {})
    model_params.setdefault("num_pair_classes", NUM_PAIR_CLASSES)
    model_params.setdefault("target_length", DEFAULT_TARGET_LENGTH)
    model_params.setdefault("mirna_length", DEFAULT_MIRNA_LENGTH)
    model_params["filter_sizes"] = tuple(model_params.get("filter_sizes", (128, 64, 32)))
    model_params["kernel_sizes"] = tuple(model_params.get("kernel_sizes", (6, 3, 3)))

    model = PairwiseSeqCNN(**model_params)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_tsv(
    *,
    input_file: Path,
    output_file: Path,
    model: PairwiseSeqCNN,
    device: torch.device,
    batch_size: int,
    prediction_column: str,
    progress_every_rows: int,
    force: bool,
) -> None:
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = output_file.with_suffix(output_file.suffix + ".tmp")
    if output_file.exists() and not force:
        raise FileExistsError(f"Output file already exists: {output_file}")
    if temp_file.exists():
        if not force:
            raise FileExistsError(f"Temporary output file already exists: {temp_file}")
        temp_file.unlink()

    started = time.perf_counter()
    rows_processed = 0
    next_progress = progress_every_rows if progress_every_rows > 0 else None
    target_length = int(model.target_length)
    mirna_length = int(model.mirna_length)

    with input_file.open() as in_handle, temp_file.open("w", buffering=1024 * 1024) as out:
        header_line = in_handle.readline()
        if not header_line:
            raise ValueError(f"Input file is empty: {input_file}")
        header = header_line.rstrip("\n").rstrip("\r").split("\t")
        columns = {column: idx for idx, column in enumerate(header)}
        if prediction_column in columns:
            raise ValueError(f"Input already has prediction column: {prediction_column}")
        if TARGET_SEQUENCE_COLUMN not in columns:
            raise ValueError(f"Missing required column: {TARGET_SEQUENCE_COLUMN}")
        mirna_column = resolve_mirna_column(columns)
        target_index = columns[TARGET_SEQUENCE_COLUMN]
        mirna_index = columns[mirna_column]
        required_width = max(target_index, mirna_index) + 1

        out.write(header_line.rstrip("\n").rstrip("\r") + f"\t{prediction_column}\n")

        raw_lines: list[str] = []
        target_sequences: list[str] = []
        mirna_sequences: list[str] = []
        for line_number, line in enumerate(in_handle, start=2):
            raw_line = line.rstrip("\n").rstrip("\r")
            fields = raw_line.split("\t")
            if len(fields) < required_width:
                raise ValueError(
                    f"Row {line_number} has {len(fields)} columns, "
                    f"expected at least {required_width}: {input_file}"
                )
            raw_lines.append(raw_line)
            target_sequences.append(fields[target_index])
            mirna_sequences.append(fields[mirna_index])

            if len(raw_lines) == batch_size:
                rows_processed += write_prediction_batch(
                    out,
                    raw_lines,
                    target_sequences,
                    mirna_sequences,
                    model,
                    device,
                    target_length=target_length,
                    mirna_length=mirna_length,
                )
                next_progress = maybe_print_progress(
                    input_file,
                    rows_processed,
                    started,
                    progress_every_rows,
                    next_progress,
                )
                raw_lines.clear()
                target_sequences.clear()
                mirna_sequences.clear()

        if raw_lines:
            rows_processed += write_prediction_batch(
                out,
                raw_lines,
                target_sequences,
                mirna_sequences,
                model,
                device,
                target_length=target_length,
                mirna_length=mirna_length,
            )

    temp_file.replace(output_file)
    elapsed = time.perf_counter() - started
    rate = rows_processed / elapsed if elapsed > 0 else 0.0
    print(
        f"done input={input_file} output={output_file} rows={rows_processed:,} "
        f"seconds={elapsed:.1f} rows_per_sec={rate:.1f}",
        flush=True,
    )


def write_prediction_batch(
    out,
    raw_lines: list[str],
    target_sequences: list[str],
    mirna_sequences: list[str],
    model: PairwiseSeqCNN,
    device: torch.device,
    *,
    target_length: int,
    mirna_length: int,
) -> int:
    pair_indices = encode_batch_pair_indices(
        target_sequences,
        mirna_sequences,
        target_length=target_length,
        mirna_length=mirna_length,
    ).to(device, non_blocking=True)
    with torch.inference_mode():
        probabilities = model.predict_proba(pair_indices).detach().cpu().numpy()
    out.write(
        "".join(
            f"{raw_line}\t{float(probability):.8f}\n"
            for raw_line, probability in zip(raw_lines, probabilities, strict=True)
        )
    )
    return len(raw_lines)


def encode_batch_pair_indices(
    target_sequences: list[str],
    mirna_sequences: list[str],
    *,
    target_length: int,
    mirna_length: int,
) -> torch.Tensor:
    target_codes = sequences_to_base_codes(target_sequences, target_length)
    mirna_codes = sequences_to_base_codes(mirna_sequences, mirna_length)
    pair_indices = np.full(
        (len(target_sequences), mirna_length, target_length),
        PADDING_PAIR_INDEX,
        dtype=np.int16,
    )
    valid_pairs = (mirna_codes[:, :, None] >= 0) & (target_codes[:, None, :] >= 0)
    pair_values = mirna_codes[:, :, None] * 4 + target_codes[:, None, :]
    pair_indices[valid_pairs] = pair_values[valid_pairs]
    return torch.from_numpy(pair_indices)


def sequences_to_base_codes(sequences: list[str], desired_length: int) -> np.ndarray:
    codes = np.full((len(sequences), desired_length), -1, dtype=np.int16)
    for row_index, sequence in enumerate(sequences):
        encoded = str(sequence).upper().encode("ascii", "ignore")[:desired_length]
        if encoded:
            bases = np.frombuffer(encoded, dtype=np.uint8)
            codes[row_index, : len(bases)] = BASE_CODE_LOOKUP[bases]
    return codes


def maybe_print_progress(
    input_file: Path,
    rows_processed: int,
    started: float,
    progress_every_rows: int,
    next_progress: int | None,
) -> int | None:
    if progress_every_rows <= 0 or next_progress is None or rows_processed < next_progress:
        return next_progress

    elapsed = time.perf_counter() - started
    rate = rows_processed / elapsed if elapsed > 0 else 0.0
    print(
        f"{input_file.name}: rows={rows_processed:,} elapsed={elapsed:.1f}s "
        f"rows_per_sec={rate:.1f}",
        flush=True,
    )
    while next_progress <= rows_processed:
        next_progress += progress_every_rows
    return next_progress


if __name__ == "__main__":
    main()
