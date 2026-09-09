#!/usr/bin/env python3
"""Evaluate a RiNALMo-PHACT checkpoint and export aligned predictions."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from multimolecule import RiNALMoModel, RnaTokenizer
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from phact_mirbind.cache.dataloaders import (
    CachedPhactPairwiseIterableDataset,
    rinalmo_phact_collate,
)
from phact_mirbind.cli.predict_seq import load_model as load_mirbind
from phact_mirbind.cli.train_rinalmo_phact import load_phact_baseline
from phact_mirbind.models.rinalmo_phact import PairwiseRinalmoPhactFusion


PAIR_BASE_TOKENS = ("A", "U", "C", "G")


def main() -> None:
    args = parse_args()
    device_name = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(device_name)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    summary = checkpoint["summary"]
    model = load_model(checkpoint, summary, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_val_auprc": float(checkpoint["val_auprc"]),
        "splits": {},
    }
    split_caches = []
    if args.val_cache is not None:
        split_caches.append(("val", args.val_cache))
    split_caches.extend(
        [("test", args.test_cache), ("leftout", args.leftout_cache)]
    )
    for split, cache_path in split_caches:
        dataset = CachedPhactPairwiseIterableDataset(cache_path, shuffle_mode="none")
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=rinalmo_phact_collate,
        )
        split_metrics, predictions = evaluate(
            model,
            loader,
            device,
            use_bfloat16=args.bfloat16,
        )
        prediction_path = args.output_dir / f"{split}_predictions.csv"
        write_predictions(prediction_path, split, predictions)
        split_metrics["prediction_file"] = str(prediction_path)
        metrics["splits"][split] = split_metrics
        print(
            f"{split}: auprc={split_metrics['auprc']:.8f} "
            f"loss={split_metrics['loss']:.8f} rows={split_metrics['rows']}",
            flush=True,
        )

    metrics_path = args.output_dir / "evaluation.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"metrics={metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--leftout-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--no-bfloat16",
        dest="bfloat16",
        action="store_false",
        help="Disable BF16 autocast on CUDA.",
    )
    parser.set_defaults(bfloat16=True)
    return parser.parse_args()


def load_model(
    checkpoint: dict[str, object],
    summary: dict[str, object],
    device: torch.device,
) -> PairwiseRinalmoPhactFusion:
    model_params = summary["model_params"]
    rinalmo_name = str(model_params["rinalmo_model"])
    tokenizer = RnaTokenizer.from_pretrained(rinalmo_name)
    rinalmo = RiNALMoModel.from_pretrained(rinalmo_name, add_pooling_layer=False)
    mirbind_path = Path(str(summary["mirbind_checkpoint"]))
    mirbind = load_mirbind(mirbind_path, torch.device("cpu"))
    raw_phact_baseline = summary.get("phact_baseline_checkpoint")
    phact_baseline_path = (
        Path(str(raw_phact_baseline)) if raw_phact_baseline is not None else None
    )
    phact_baseline, phact_baseline_mode = load_phact_baseline(phact_baseline_path)
    model = PairwiseRinalmoPhactFusion(
        rinalmo=rinalmo,
        mirbind=mirbind,
        mirna_phact_channels=int(model_params["mirna_phact_channels"]),
        target_phact_channels=int(model_params["target_phact_channels"]),
        base_token_ids=tuple(
            tokenizer.convert_tokens_to_ids(token) for token in PAIR_BASE_TOKENS
        ),
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.eos_token_id,
        layers_to_mix=int(model_params["layers_to_mix"]),
        attention_dim=int(model_params["attention_dim"]),
        pair_projection_dim=int(model_params["pair_projection_dim"]),
        classifier_dim=int(model_params["classifier_dim"]),
        dropout_rate=float(model_params["dropout_rate"]),
        encoding_mode=str(model_params.get("encoding_mode", "independent")),
        phact_baseline=phact_baseline,
        phact_baseline_channel_mode=str(
            summary.get("phact_baseline_channel_mode", phact_baseline_mode)
        ),
        phact_baseline_score_channels=tuple(
            summary.get("phact_baseline_score_channels", (4, 4))
        ),
        correction_scale=float(summary.get("correction_scale", 0.25)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def evaluate(
    model: PairwiseRinalmoPhactFusion,
    loader: DataLoader,
    device: torch.device,
    *,
    use_bfloat16: bool,
) -> tuple[dict[str, float | int], np.ndarray]:
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    weighted_loss = 0.0
    row_count = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            *model_inputs, label = batch
            model_inputs = [item.to(device, non_blocking=True) for item in model_inputs]
            label = label.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bfloat16 and device.type == "cuda",
            ):
                logits = model(*model_inputs)
                loss = F.binary_cross_entropy_with_logits(logits, label)
            probability = torch.sigmoid(logits).float().cpu().numpy()
            batch_labels = label.cpu().numpy()
            labels.append(batch_labels)
            predictions.append(probability)
            batch_rows = int(label.numel())
            weighted_loss += float(loss) * batch_rows
            row_count += batch_rows
    labels_array = np.concatenate(labels)
    predictions_array = np.concatenate(predictions)
    return (
        {
            "loss": weighted_loss / row_count,
            "auprc": float(average_precision_score(labels_array, predictions_array)),
            "rows": row_count,
            "seconds": time.perf_counter() - started,
        },
        predictions_array,
    )


def write_predictions(path: Path, split: str, predictions: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "prediction"])
        for row_id, prediction in enumerate(predictions, start=1):
            writer.writerow([f"{split}_{row_id}", f"{float(prediction):.8f}"])


if __name__ == "__main__":
    main()
