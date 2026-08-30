"""Generic training/evaluation loop for cached miRBind models."""

from __future__ import annotations

import json
import math
import platform
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from phact_mirbind.training.logging import (
    append_tsv_log,
    finish_progress,
    json_ready,
    maybe_log_progress,
    maybe_show_progress,
    print_epoch,
    print_final_evaluation,
    write_progress_header,
)
from phact_mirbind.training.metrics import average_precision


def load_widened_state_dict(
    model: nn.Module,
    source_state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Load a narrower compatible model while preserving its initial function.

    Existing output units are copied exactly. Connections from newly added
    input units into existing output units start at zero, while newly added
    output units keep their random initialization. This supports widening CNN
    filter stacks without discarding a trained checkpoint.
    """

    target_state_dict = model.state_dict()
    if target_state_dict.keys() != source_state_dict.keys():
        missing = sorted(source_state_dict.keys() - target_state_dict.keys())
        extra = sorted(target_state_dict.keys() - source_state_dict.keys())
        raise ValueError(
            f"Incompatible state dictionaries; missing={missing}, extra={extra}"
        )

    widened_state_dict: dict[str, torch.Tensor] = {}
    for name, target in target_state_dict.items():
        source = source_state_dict[name]
        if source.ndim != target.ndim or any(
            source_size > target_size
            for source_size, target_size in zip(source.shape, target.shape)
        ):
            raise ValueError(
                f"Cannot widen {name} from {tuple(source.shape)} "
                f"to {tuple(target.shape)}"
            )
        if source.shape == target.shape:
            widened_state_dict[name] = source
            continue

        widened = target.clone()
        if widened.ndim >= 2:
            widened[: source.shape[0]].zero_()
        source_slices = tuple(slice(0, size) for size in source.shape)
        widened[source_slices] = source
        widened_state_dict[name] = widened

    model.load_state_dict(widened_state_dict, strict=True)


def configure_new_input_channels_only(
    model: nn.Module,
    source_state_dict: Mapping[str, torch.Tensor],
) -> int:
    """Freeze a widened model except for newly appended first-conv inputs."""
    parameter_name = "conv_layers.0.weight"
    parameters = dict(model.named_parameters())
    if parameter_name not in parameters or parameter_name not in source_state_dict:
        raise ValueError(f"Missing required parameter: {parameter_name}")
    parameter = parameters[parameter_name]
    source = source_state_dict[parameter_name]
    if parameter.ndim != 4 or source.ndim != 4:
        raise ValueError("first convolution weights must be four-dimensional")
    if parameter.shape[1] <= source.shape[1]:
        raise ValueError("model has no newly appended first-convolution inputs")
    if (
        parameter.shape[0] != source.shape[0]
        or parameter.shape[2:] != source.shape[2:]
    ):
        raise ValueError("first convolution changed beyond its input channels")

    for candidate in model.parameters():
        candidate.requires_grad_(False)
    parameter.requires_grad_(True)
    mask = torch.zeros_like(parameter)
    mask[:, source.shape[1] :] = 1
    parameter.register_hook(lambda gradient: gradient * mask)
    model._keep_frozen_modules_in_eval = True
    return int(mask.sum().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: Optimizer | None,
    epoch: int,
    phase: str,
    progress_every: int,
    progress_log_path: Path,
    show_progress: bool,
    amp_dtype: torch.dtype | None = None,
    grad_clip_norm: float | None = None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    if optimizer is not None and getattr(model, "_keep_frozen_modules_in_eval", False):
        for module in model.modules():
            if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.Dropout)):
                module.eval()
    total_loss = 0.0
    batch_count = 0
    sample_count = 0
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    started = time.perf_counter()
    total_samples = expected_sample_count(loader)
    total_batches = expected_batch_count(loader, total_samples)
    progress_shown = False

    try:
        for batch in loader:
            sample_weights = None
            if getattr(loader.dataset, "has_sample_weights", False):
                *model_inputs, label, sample_weights = batch
                sample_weights = sample_weights.to(device, non_blocking=True)
            else:
                *model_inputs, label = batch
            model_inputs = [
                item.to(device, non_blocking=True) for item in model_inputs
            ]
            label = label.to(device, non_blocking=True)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, amp_dtype):
                    logits = model(*model_inputs)
                    loss = (
                        criterion(logits, label, sample_weights)
                        if sample_weights is not None
                        else criterion(logits, label)
                    )
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            else:
                with torch.no_grad():
                    with autocast_context(device, amp_dtype):
                        logits = model(*model_inputs)
                        loss = (
                            criterion(logits, label, sample_weights)
                            if sample_weights is not None
                            else criterion(logits, label)
                        )

            total_loss += loss.item()
            batch_count += 1
            sample_count += int(label.numel())
            labels.append(label.detach().cpu().numpy())
            probabilities.append(
                torch.sigmoid(logits).float().detach().cpu().numpy()
            )
            maybe_log_progress(
                progress_log_path,
                epoch=epoch,
                phase=phase,
                progress_every=progress_every,
                batch_count=batch_count,
                sample_count=sample_count,
                total_loss=total_loss,
                started=started,
            )
            progress_shown = maybe_show_progress(
                epoch=epoch,
                phase=phase,
                batch_count=batch_count,
                total_batches=total_batches,
                sample_count=sample_count,
                total_samples=total_samples,
                total_loss=total_loss,
                started=started,
                enabled=show_progress,
            ) or progress_shown
    finally:
        finish_progress(progress_shown)

    return {
        "loss": total_loss / max(batch_count, 1),
        "auprc": average_precision(labels, probabilities),
        "batches": batch_count,
        "samples": sample_count,
        "seconds": time.perf_counter() - started,
    }


def train_model(
    *,
    model: nn.Module,
    loaders: Mapping[str, DataLoader],
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    output_dir: Path,
    checkpoint_prefix: str,
    summary: dict[str, object],
    num_epochs: int,
    patience: int,
    progress_every: int,
    epoch_eval_phases: tuple[str, ...] = ("val",),
    final_eval_phases: tuple[str, ...] = ("test", "leftout"),
    show_progress: bool = False,
    extra_checkpoint: dict[str, object] | None = None,
    amp_dtype: torch.dtype | None = None,
    grad_clip_norm: float | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = output_dir / f"{checkpoint_prefix}_{timestamp}.pt"
    history_path = output_dir / f"training_history_{timestamp}.json"
    log_path = output_dir / f"training_log_{timestamp}.tsv"
    progress_log_path = output_dir / f"batch_progress_{timestamp}.tsv"
    final_eval_path = output_dir / f"final_evaluation_{timestamp}.json"
    summary_path = output_dir / f"architecture_summary_{timestamp}.json"

    summary.update(
        {
            "argv": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "device": str(device),
            "progress_every": progress_every,
            "epoch_eval_phases": list(epoch_eval_phases),
            "final_eval_phases": list(final_eval_phases),
            "progress_bar": show_progress,
            "batch_progress_log": str(progress_log_path),
            "training_log": str(log_path),
            "training_history": str(history_path),
            "final_evaluation": str(final_eval_path),
            "checkpoint": str(checkpoint_path),
            "total_params": sum(param.numel() for param in model.parameters()),
            "trainable_params": sum(
                param.numel() for param in model.parameters() if param.requires_grad
            ),
            "amp_dtype": str(amp_dtype) if amp_dtype is not None else None,
            "grad_clip_norm": grad_clip_norm,
        }
    )
    summary_path.write_text(json.dumps(json_ready(summary), indent=2) + "\n")

    print(f"Using device: {device}")
    print(f"Model parameters: {summary['total_params']:,}")
    print(f"Architecture summary: {summary_path}")
    write_progress_header(progress_log_path)

    history: list[dict[str, object]] = []
    best_val_auprc: float | None = None
    best_epoch: int | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        epoch_started = time.perf_counter()
        record: dict[str, object] = {"epoch": epoch}
        record["train"] = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            epoch=epoch,
            phase="train",
            progress_every=progress_every,
            progress_log_path=progress_log_path,
            show_progress=show_progress,
            amp_dtype=amp_dtype,
            grad_clip_norm=grad_clip_norm,
        )
        for phase in epoch_eval_phases:
            record[phase] = run_epoch(
                model,
                loaders[phase],
                criterion,
                device,
                optimizer=None,
                epoch=epoch,
                phase=phase,
                progress_every=progress_every,
                progress_log_path=progress_log_path,
                show_progress=show_progress,
                amp_dtype=amp_dtype,
            )
        record["epoch_seconds"] = time.perf_counter() - epoch_started

        val_metrics = record["val"]
        assert isinstance(val_metrics, dict)
        val_auprc = float(val_metrics["auprc"])
        has_improved = best_val_auprc is None or (
            not np.isnan(val_auprc)
            and (np.isnan(best_val_auprc) or val_auprc > best_val_auprc)
        )
        if has_improved:
            best_val_auprc = val_auprc
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            record["status"] = f"best_val={best_val_auprc:.4f} saved"
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auprc": val_auprc,
                "model_params": summary.get("model_params", {}),
                "summary": summary,
                "history": history,
            }
            if extra_checkpoint:
                checkpoint.update(extra_checkpoint)
            torch.save(checkpoint, checkpoint_path)
        else:
            patience_counter += 1
            record["status"] = (
                f"best_val={best_val_auprc:.4f} "
                f"patience={patience_counter}/{patience}"
            )

        history.append(record)
        history_path.write_text(json.dumps(json_ready(history), indent=2) + "\n")
        append_tsv_log(log_path, record)
        print_epoch(record)

        if patience_counter >= patience:
            print(f"Early stopping after epoch {epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    assert best_epoch is not None

    final_record: dict[str, object] = {"best_epoch": best_epoch}
    for phase in final_eval_phases:
        final_record[phase] = run_epoch(
            model,
            loaders[phase],
            criterion,
            device,
            optimizer=None,
            epoch=best_epoch,
            phase=phase,
            progress_every=progress_every,
            progress_log_path=progress_log_path,
            show_progress=show_progress,
            amp_dtype=amp_dtype,
        )
    final_eval_path.write_text(json.dumps(json_ready(final_record), indent=2) + "\n")
    print_final_evaluation(final_record)

    print(f"Training history: {history_path}")
    print(f"Training log: {log_path}")
    print(f"Final evaluation: {final_eval_path}")
    print(f"Best model: {checkpoint_path}")
    return {
        "checkpoint": checkpoint_path,
        "history": history_path,
        "log": log_path,
        "progress_log": progress_log_path,
        "final_evaluation": final_eval_path,
        "summary": summary_path,
    }


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    if amp_dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def expected_sample_count(loader: DataLoader) -> int | None:
    dataset = getattr(loader, "dataset", None)
    manifest = getattr(dataset, "manifest", None)
    if not isinstance(manifest, dict) or "row_count" not in manifest:
        return None

    row_count = int(manifest["row_count"])
    max_rows = getattr(dataset, "max_rows", None)
    if max_rows is not None:
        row_count = min(row_count, int(max_rows))
    return row_count


def expected_batch_count(loader: DataLoader, sample_count: int | None) -> int | None:
    if sample_count is None:
        return None
    batch_size = getattr(loader, "batch_size", None)
    if not isinstance(batch_size, int) or batch_size < 1:
        return None
    return math.ceil(sample_count / batch_size)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())
