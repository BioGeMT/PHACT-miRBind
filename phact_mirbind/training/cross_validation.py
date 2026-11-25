"""Train an unchanged stable miRBind model on one saved fold, with compact GPU inputs."""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from phact_mirbind.cache.fold_cache import FoldBatches, FoldInputs
from phact_mirbind.data.fold_features import sha
from phact_mirbind.models.phact import PairwisePhactCNN
from phact_mirbind.models.seq_only import PairwiseSeqCNN
from phact_mirbind.training.loop import autocast_context, set_seed, train_model

PARAMS = dict(
    num_pair_classes=18,
    target_length=50,
    mirna_length=28,
    embedding_dim=8,
    dropout_rate=0.2,
    filter_sizes=(128, 64, 32),
    kernel_sizes=(6, 3, 3),
)


def make_model(variant, seed, device="cuda"):
    if variant not in {"seq_only", "target_phact"}:
        raise ValueError(f"Unknown model variant: {variant}")
    set_seed(seed)
    baseline = PairwiseSeqCNN(**PARAMS)
    if variant == "seq_only":
        model = baseline
    else:
        model = PairwisePhactCNN(**PARAMS, phact_channel_count=5)
        # All common parameters start identically; extra score connections retain
        # their random initialization. No pretrained weights are used.
        state = model.state_dict()
        for name, value in baseline.state_dict().items():
            if state[name].shape == value.shape:
                state[name].copy_(value)
            else:
                assert name == "conv_layers.0.weight"
                state[name][:, :8].copy_(value)
        model.load_state_dict(state)
    return model.to(device)


def load_model(checkpoint_path, device):
    """Rebuild a saved five-fold model from its checkpoint and return it with its variant."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    variant = checkpoint["summary"]["model_type"]
    if variant not in {"seq_only", "target_phact"}:
        raise ValueError(f"Unknown model variant: {variant}")
    params = {
        name: tuple(value) if isinstance(value, list) else value
        for name, value in checkpoint["model_params"].items()
    }
    model_class = PairwisePhactCNN if variant == "target_phact" else PairwiseSeqCNN
    model = model_class(**params)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device), variant


def bf16_dtype(device):
    """Return the autocast dtype of the training protocol: BF16 on CUDA, none on CPU."""
    if device.type != "cuda":
        return None
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This protocol requires BF16 support on CUDA devices")
    return torch.bfloat16


def predict_probabilities(model, input_batches, device, amp_dtype):
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for x in input_batches:
            with autocast_context(device, amp_dtype):
                logits = model(*x)
            # Same probability calculation as stable training.loop.run_epoch.
            p = torch.sigmoid(logits).float()
            assert torch.isfinite(p).all()
            probabilities.append(p.cpu().numpy())
    return np.concatenate(probabilities)


def predict(model, loader, device, amp_dtype):
    labels = []

    def inputs():
        for *x, y in loader:
            labels.append(y.cpu().numpy())
            yield x

    probabilities = predict_probabilities(model, inputs(), device, amp_dtype)
    return np.concatenate(labels), probabilities


def train_fold(
    inputs: FoldInputs,
    output_dir: Path,
    *,
    fold: int,
    seed: int = 42,
    batch_size: int = 512,
    num_epochs: int = 30,
    patience: int = 7,
) -> Path:
    """Select on validation AP, then evaluate the restored best checkpoint."""
    if fold not in range(1, 6) or batch_size < 2 or num_epochs < 1 or patience < 1:
        raise ValueError(
            "Require fold 1..5, batch size >=2, and positive epochs/patience"
        )
    variant = inputs.variant
    device = inputs.device
    amp_dtype = bf16_dtype(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    run_name = f"{variant}/fold_{fold}"
    indices = {
        name: np.flatnonzero(inputs.roles[:, fold - 1] == code)
        for code, name in enumerate(["train", "val", "seen_test", "unseen_test"])
    }
    if inputs.metadata["rows"] > inputs.metadata["pool_rows"]:
        indices["leftout"] = np.arange(
            inputs.metadata["pool_rows"], inputs.metadata["rows"]
        )
    loaders = {
        name: FoldBatches(
            inputs, idx, batch_size, shuffle=name == "train", seed=seed, name=run_name
        )
        for name, idx in indices.items()
    }
    model = make_model(variant, seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    set_seed(seed)
    output = Path(output_dir) / variant / f"fold_{fold}"
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "model_type": variant,
        "model_params": {
            **PARAMS,
            **({"phact_channel_count": 5} if variant == "target_phact" else {}),
        },
        "fold": fold,
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": 0.001,
        "num_epochs": num_epochs,
        "patience": patience,
        "selection_metric": "validation average precision",
        "initialization": "Random from scratch, common weights identical between variants",
        "rows": {name: len(idx) for name, idx in indices.items()},
        "cache": str(inputs.cache_dir),
        "target_channels": ["A", "C", "G", "T", "missing"]
        if variant == "target_phact"
        else [],
        "target_missing_fill": 0.5,
        "additional_conservation_tracks": [],
        "assignments_sha256": inputs.metadata["assignments_sha256"],
        "source_sha256": {
            str(p): sha(p)
            for p in [
                Path(__file__),
                *[
                    Path(__file__).parents[1] / "models" / f"{name}.py"
                    for name in ["seq_only", "conservation", "phact", "blocks"]
                ],
                Path(__file__).with_name("loop.py"),
            ]
        },
    }
    print(
        f"START {run_name}; rows={summary['rows']}; batch={batch_size}; seed={seed}",
        flush=True,
    )
    paths = train_model(
        model=model,
        loaders={"train": loaders["train"], "val": loaders["val"]},
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        output_dir=output,
        checkpoint_prefix="best_model",
        summary=summary,
        num_epochs=num_epochs,
        patience=patience,
        progress_every=100,
        show_progress=False,
        final_eval_phases=(),
        amp_dtype=amp_dtype,
    )
    # train_model restores the checkpoint selected only by validation AUPRC.
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    evaluation = {
        "variant": variant,
        "fold": fold,
        "best_epoch": checkpoint["epoch"],
        "best_validation_auprc": checkpoint["val_auprc"],
        "checkpoint": str(paths["checkpoint"]),
        "roles": {},
    }
    for name in [
        "seen_test",
        "unseen_test",
        *(["leftout"] if "leftout" in loaders else []),
    ]:
        y, p = predict(model, loaders[name], device, amp_dtype)
        metric = {
            "rows": len(y),
            "auprc": float(average_precision_score(y, p)),
            "auroc": float(roc_auc_score(y, p)),
        }
        evaluation["roles"][name] = metric
        np.savez_compressed(
            output / f"{name}_predictions.npz",
            row_index=indices[name],
            label=y,
            probability=p,
        )
        print(f"EVALUATION {run_name} {name}: {json.dumps(metric)}", flush=True)
    (output / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    (output / "complete.json").write_text(
        json.dumps(
            {"status": "complete", "evaluation": str(output / "evaluation.json")}
        )
        + "\n"
    )
    print(f"COMPLETE {run_name}", flush=True)

    return output / "evaluation.json"
