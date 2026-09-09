#!/usr/bin/env python3
"""Independently verify the completed fixed study; write only to this analysis directory."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workspace import WORKSPACE, historical_path

import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone

sys.dont_write_bytecode = True
ROOT = WORKSPACE / "runs/followup-2026-09-06/training"
OUTPUT = ROOT.parent / "analysis_2026-09-07"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from run_condition import EXPECTED_ROWS, PHACT_CACHE, make_model, set_seed
from phact_mirbind.cache.manifest import find_manifest
from summarize_runs import PAIRS


def digest(path):
    with historical_path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    check(abs(actual - expected) <= 1e-12, f'{message}: {actual} != {expected}')


def describe(values):
    return dict(n=len(values), mean=statistics.mean(values),
                sample_sd=statistics.stdev(values), minimum=min(values), maximum=max(values))


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def main():
    torch.set_num_threads(4)
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    seeds, conditions = protocol['seeds'], protocol['conditions']
    expected = {(seed, condition) for seed in seeds for condition in conditions}
    check(len(expected) == 18 and seeds == [17, 43, 101], 'Unexpected study design')
    paths = list((ROOT / 'runs').glob('seed_*/*/result.json'))
    check(len(paths) == 18, 'Expected exactly 18 completed result files')
    frozen = json.loads((ROOT / 'frozen_inputs.json').read_text())
    for path, expected_hash in frozen.items():
        check(digest(path) == expected_hash, f'Frozen input changed: {path}')

    canonical = {}
    for split in ('test', 'leftout'):
        labels_by_cache = []
        for cache in (PHACT_CACHE, ROOT / 'conservation_cache'):
            manifest_path = find_manifest(cache / split)
            manifest = json.loads(manifest_path.read_text())
            labels = np.concatenate([
                torch.load(manifest_path.parent / shard['file'], map_location='cpu', weights_only=True)['labels'].numpy()
                for shard in manifest['shards']])
            check(labels.shape == (EXPECTED_ROWS[split],), f'Canonical labels shape: {cache}/{split}')
            labels_by_cache.append(labels)
        check(np.array_equal(*labels_by_cache), f'PHACT/conservation labels differ: {split}')
        canonical[split] = labels_by_cache[0]

    results, metrics, hashes, runs = {}, [], {}, []
    for path in sorted(paths):
        result = json.loads(path.read_text())
        seed, condition = result['seed'], result['condition']
        identity = (seed, condition)
        check(identity in expected and identity not in results, f'Unexpected/duplicate identity: {path}')
        check(path.parent == ROOT / 'runs' / f'seed_{seed}' / condition, f'Wrong run path: {path}')
        check(result['smoke'] is False, f'Smoke result: {path}')
        results[identity] = result
        checkpoint_path = historical_path(result['checkpoint'])
        check(checkpoint_path.parent == path.parent, f'Checkpoint outside run: {path}')
        check(digest(checkpoint_path) == result['checkpoint_sha256'], f'Checkpoint hash: {path}')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        summary = checkpoint['summary']
        for field, value in [('seed', seed), ('condition', condition), ('smoke', False),
                             ('initial_state_sha256', result['initial_state_sha256']),
                             ('driver_sha256', digest(ROOT / 'run_condition.py')),
                             ('batch_size', 256), ('learning_rate', 0.001),
                             ('num_epochs', 50), ('patience', 7), ('amp_dtype', None)]:
            check(summary[field] == value, f'Checkpoint {field}: {path}')
        check(checkpoint['epoch'] == result['best_epoch'], f'Checkpoint epoch: {path}')
        close(checkpoint['val_auprc'], result['validation_ap'], f'Checkpoint validation AP: {path}')
        set_seed(seed)
        model, model_params = make_model(condition)
        initial_hash = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            initial_hash.update(name.encode())
            initial_hash.update(tensor.numpy().tobytes())
        check(initial_hash.hexdigest() == result['initial_state_sha256'], f'Recreated initialisation: {path}')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        check(all(torch.isfinite(value).all().item() for value in model.state_dict().values()), f'Nonfinite checkpoint: {path}')
        histories = list(path.parent.glob('training_history_*.json'))
        check(len(histories) == 1, f'History count: {path}')
        history = json.loads(histories[0].read_text())
        check([row['epoch'] for row in history] == list(range(1, len(history) + 1)), f'Epoch sequence: {path}')
        check(1 <= len(history) <= 50, f'Number of epochs: {path}')
        for row in history:
            for phase in ('train', 'val'):
                check(row[phase]['samples'] == EXPECTED_ROWS[phase], f'Full-data count {phase}: {path}')
        best = max(history, key=lambda row: row['val']['auprc'])
        check(best['epoch'] == result['best_epoch'], f'History best epoch: {path}')
        close(best['val']['auprc'], result['validation_ap'], f'History best validation AP: {path}')
        check(len(history) == 50 or len(history) - best['epoch'] == 7, f'Early-stop rule: {path}')
        runs.append(dict(seed=seed, condition=condition, best_epoch=best['epoch'],
                         epochs_completed=len(history), validation_ap=result['validation_ap'],
                         stop_reason='maximum_epochs' if len(history) == 50 else 'early_stopping',
                         initial_state_sha256=result['initial_state_sha256'],
                         torch_version=summary['torch_version'], parameter_count=sum(p.numel() for p in model.parameters())))
        for split in ('test', 'leftout'):
            prediction_path = path.parent / f'{split}.npz'
            check(digest(prediction_path) == result[split]['prediction_sha256'], f'Prediction hash: {prediction_path}')
            labels = canonical[split]
            with np.load(prediction_path, allow_pickle=False) as archive:
                check(set(archive.files) == {'ids', 'labels', 'predictions'}, f'NPZ columns: {prediction_path}')
                check(np.array_equal(archive['labels'], labels), f'Canonical labels: {prediction_path}')
                expected_ids = np.array([f'{split}_{index + 1}' for index in range(len(labels))])
                check(np.array_equal(archive['ids'], expected_ids), f'Canonical row IDs: {prediction_path}')
                predictions = archive['predictions']
                check(predictions.shape == labels.shape, f'Prediction shape: {prediction_path}')
                check(np.isfinite(predictions).all() and np.all((predictions >= 0) & (predictions <= 1)), f'Probabilities: {prediction_path}')
                ap = float(average_precision_score(labels, predictions))
                roc_auc = float(roc_auc_score(labels, predictions))
            check(result[split]['rows'] == len(labels), f'Rows: {prediction_path}')
            check(result[split]['positive_rows'] == int(labels.sum()), f'Positive rows: {prediction_path}')
            close(ap, result[split]['ap'], f'Recomputed AP: {prediction_path}')
            close(roc_auc, result[split]['roc_auc'], f'Recomputed AUROC: {prediction_path}')
            metrics.append(dict(seed=seed, condition=condition, split=split, ap=ap, roc_auc=roc_auc,
                                rows=len(labels), positive_rows=int(labels.sum()),
                                best_epoch=result['best_epoch'], validation_ap=result['validation_ap']))
        for artifact in path.parent.iterdir():
            if artifact.is_file() and (artifact.suffix in ('.json', '.pt', '.npz')):
                hashes[str(artifact.relative_to(ROOT))] = digest(artifact)
        print(f'Verified seed {seed}, {condition}', flush=True)

    check(set(results) == expected, 'Missing planned results')
    aggregates = []
    for condition in conditions:
        for split in ('test', 'leftout'):
            selected = [row for row in metrics if row['condition'] == condition and row['split'] == split]
            aggregates.append(dict(condition=condition, split=split,
                                   ap=describe([row['ap'] for row in selected]),
                                   roc_auc=describe([row['roc_auc'] for row in selected])))
    deltas, delta_summary = [], []
    for first, second in PAIRS:
        comparison = f'{first} minus {second}'
        for seed in seeds:
            if second.endswith('shuffled'):
                check(results[seed, first]['initial_state_sha256'] == results[seed, second]['initial_state_sha256'], f'Unmatched initialisation: {seed}/{comparison}')
            for split in ('test', 'leftout'):
                deltas.append(dict(seed=seed, comparison=comparison, split=split,
                                   ap_delta=results[seed, first][split]['ap'] - results[seed, second][split]['ap'],
                                   roc_auc_delta=results[seed, first][split]['roc_auc'] - results[seed, second][split]['roc_auc']))
        for split in ('test', 'leftout'):
            rows = [row for row in deltas if row['comparison'] == comparison and row['split'] == split]
            values = [row['ap_delta'] for row in rows]
            delta_summary.append(dict(comparison=comparison, split=split, ap_delta=describe(values),
                                      positive_seeds=sum(value > 0 for value in values),
                                      negative_seeds=sum(value < 0 for value in values),
                                      roc_auc_delta=describe([row['roc_auc_delta'] for row in rows])))

    original_metrics = read_rows(ROOT / 'seed_metrics.tsv')
    original_deltas = read_rows(ROOT / 'paired_seed_deltas.tsv')
    check(len(original_metrics) == len(metrics) == 36, 'Original metric row count')
    check(len(original_deltas) == len(deltas) == 30, 'Original delta row count')
    for row in original_metrics:
        match = [item for item in metrics if (item['seed'], item['condition'], item['split']) == (int(row['seed']), row['condition'], row['split'])]
        check(len(match) == 1, 'Original metric identity')
        close(match[0]['ap'], float(row['ap']), 'Original AP')
        close(match[0]['validation_ap'], float(row['validation_ap']), 'Original validation AP')
        check(match[0]['best_epoch'] == int(row['best_epoch']), 'Original best epoch')
    for row in original_deltas:
        match = [item for item in deltas if (item['seed'], item['comparison'], item['split']) == (int(row['seed']), row['comparison'], row['split'])]
        check(len(match) == 1, 'Original delta identity')
        close(match[0]['ap_delta'], float(row['ap_delta']), 'Original paired AP delta')
    published = (ROOT / 'TRAINING_RESULTS.md').read_text()
    for row in aggregates:
        values = row['ap']
        expected_row = (f"| {row['condition']} | {row['split']} | 3 | {values['mean']:.6f} | "
                        f"{values['sample_sd']:.6f} | {values['minimum']:.6f}–{values['maximum']:.6f} |")
        check(expected_row in published, f'Original summary table: {row}')

    output = dict(status='passed', completed_utc=datetime.now(timezone.utc).isoformat(),
                  protocol=protocol, runs=runs, metrics=metrics, aggregates=aggregates,
                  paired_deltas=deltas, paired_delta_summary=delta_summary,
                  checks=dict(completed_non_smoke_runs=18, checkpoint_hashes=18,
                              recreated_initialisations=18, full_training_histories=18,
                              validation_selected_checkpoints=18, prediction_hashes=36,
                              canonical_row_and_label_matches=36, recomputed_AP=36,
                              recomputed_AUROC=36, frozen_input_hashes=len(frozen),
                              original_metric_rows_matched=36, original_delta_rows_matched=30,
                              original_summary_table_rows_matched=12,
                              conservation_and_PHACT_cache_labels_match=True),
                  canonical_label_hashes={split: hashlib.sha256(labels.tobytes()).hexdigest() for split, labels in canonical.items()},
                  artifact_sha256=hashes,
                  interpretation='Sample standard deviations describe three training seeds on one fixed development split. No across-seed confidence interval or fresh bootstrap was computed. Metrics are separate fitted models, not ensembles. Evaluation collections have prior development exposure.')
    destination = OUTPUT / 'verified_analysis.json'
    destination.write_text(json.dumps(output, indent=2) + '\n')
    print(f'All checks passed; wrote {destination}', flush=True)


if __name__ == '__main__':
    main()
