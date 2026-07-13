"""Training log writers."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
import sys


def json_ready(value: object) -> object:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    return value


def append_tsv_log(log_path: Path, record: dict[str, object]) -> None:
    columns = [
        "epoch",
        "epoch_seconds",
        "train_loss",
        "train_auprc",
        "train_samples",
        "train_batches",
        "train_seconds",
        "val_loss",
        "val_auprc",
        "val_samples",
        "val_batches",
        "val_seconds",
        "test_loss",
        "test_auprc",
        "test_samples",
        "test_batches",
        "test_seconds",
        "leftout_loss",
        "leftout_auprc",
        "leftout_samples",
        "leftout_batches",
        "leftout_seconds",
    ]
    write_header = not log_path.exists()
    values = {
        "epoch": record["epoch"],
        "epoch_seconds": f"{float(record.get('epoch_seconds', float('nan'))):.3f}",
        "train_loss": _metric(record, "train", "loss"),
        "train_auprc": _metric(record, "train", "auprc"),
        "train_samples": _metric(record, "train", "samples", integer=True),
        "train_batches": _metric(record, "train", "batches", integer=True),
        "train_seconds": _metric(record, "train", "seconds"),
        "val_loss": _metric(record, "val", "loss"),
        "val_auprc": _metric(record, "val", "auprc"),
        "val_samples": _metric(record, "val", "samples", integer=True),
        "val_batches": _metric(record, "val", "batches", integer=True),
        "val_seconds": _metric(record, "val", "seconds"),
        "test_loss": _metric(record, "test", "loss"),
        "test_auprc": _metric(record, "test", "auprc"),
        "test_samples": _metric(record, "test", "samples", integer=True),
        "test_batches": _metric(record, "test", "batches", integer=True),
        "test_seconds": _metric(record, "test", "seconds"),
        "leftout_loss": _metric(record, "leftout", "loss"),
        "leftout_auprc": _metric(record, "leftout", "auprc"),
        "leftout_samples": _metric(record, "leftout", "samples", integer=True),
        "leftout_batches": _metric(record, "leftout", "batches", integer=True),
        "leftout_seconds": _metric(record, "leftout", "seconds"),
    }
    with log_path.open("a") as handle:
        if write_header:
            handle.write("\t".join(columns) + "\n")
        handle.write("\t".join(str(values[column]) for column in columns) + "\n")


def print_epoch(record: dict[str, object]) -> None:
    epoch = record["epoch"]
    parts = [f"Epoch {epoch} ({float(record.get('epoch_seconds', 0.0)):.1f}s)"]
    for split_name in ["train", "val"]:
        metrics = record.get(split_name)
        if not isinstance(metrics, dict):
            continue
        parts.append(
            f"{split_name}: loss={metrics['loss']:.4f}, "
            f"auprc={metrics['auprc']:.4f}, "
            f"samples={int(metrics['samples']):,}, "
            f"sec={float(metrics['seconds']):.1f}"
        )
    status = record.get("status")
    if status:
        parts.append(str(status))
    print(" | ".join(parts))


def print_final_evaluation(record: dict[str, object]) -> None:
    parts = [f"Final eval best_epoch={record['best_epoch']}"]
    for split_name in ["test", "leftout"]:
        metrics = record.get(split_name)
        if not isinstance(metrics, dict):
            continue
        parts.append(
            f"{split_name}: loss={metrics['loss']:.4f}, "
            f"auprc={metrics['auprc']:.4f}, "
            f"samples={int(metrics['samples']):,}, "
            f"sec={float(metrics['seconds']):.1f}"
        )
    print(" | ".join(parts))


def write_progress_header(progress_log_path: Path) -> None:
    with progress_log_path.open("w") as handle:
        handle.write(
            "utc_time\tepoch\tphase\tbatches\tsamples\telapsed_sec\tmean_loss\n"
        )


def maybe_log_progress(
    progress_log_path: Path,
    *,
    epoch: int,
    phase: str,
    progress_every: int,
    batch_count: int,
    sample_count: int,
    total_loss: float,
    started: float,
) -> None:
    if progress_every <= 0 or batch_count % progress_every != 0:
        return
    elapsed = time.perf_counter() - started
    mean_loss = total_loss / max(batch_count, 1)
    utc_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with progress_log_path.open("a") as handle:
        handle.write(
            f"{utc_time}\t{epoch}\t{phase}\t{batch_count}\t{sample_count}\t"
            f"{elapsed:.3f}\t{mean_loss:.8g}\n"
        )


def maybe_show_progress(
    *,
    epoch: int,
    phase: str,
    batch_count: int,
    total_batches: int | None,
    sample_count: int,
    total_samples: int | None,
    total_loss: float,
    started: float,
    enabled: bool,
) -> bool:
    if not enabled or not sys.stdout.isatty():
        return False

    elapsed = time.perf_counter() - started
    mean_loss = total_loss / max(batch_count, 1)
    if total_batches:
        fraction = min(batch_count / total_batches, 1.0)
        width = 24
        filled = int(width * fraction)
        bar = "#" * filled + "." * (width - filled)
        sample_text = (
            f"{sample_count:,}/{total_samples:,}"
            if total_samples is not None
            else f"{sample_count:,}"
        )
        text = (
            f"\rEpoch {epoch} {phase} [{bar}] "
            f"{batch_count:,}/{total_batches:,} batches "
            f"{sample_text} samples loss={mean_loss:.4f} {elapsed:.1f}s"
        )
    else:
        text = (
            f"\rEpoch {epoch} {phase}: batch={batch_count:,} "
            f"samples={sample_count:,} loss={mean_loss:.4f} {elapsed:.1f}s"
        )
    print(text, end="", flush=True)
    return True


def finish_progress(shown: bool) -> None:
    if shown:
        print()


def _metric(
    record: dict[str, object],
    split_name: str,
    metric_name: str,
    *,
    integer: bool = False,
) -> str:
    metrics = record.get(split_name)
    if not isinstance(metrics, dict):
        return "NA"
    value = metrics[metric_name]
    if integer:
        return str(int(value))
    return f"{float(value):.8g}"
