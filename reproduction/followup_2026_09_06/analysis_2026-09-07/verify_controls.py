#!/usr/bin/env python3
"""Verify actual evaluation control tensors against their frozen audit counts."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workspace import WORKSPACE, historical_path

import hashlib
import json

sys.dont_write_bytecode = True
import torch

ROOT = WORKSPACE / "runs/followup-2026-09-06"


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    torch.set_num_threads(4)
    verified = {}
    for split in ('test', 'leftout'):
        directory = ROOT / 'controls' / split
        manifest = json.loads((directory / 'manifest.json').read_text())
        source_manifest = historical_path(manifest['source_manifest'])
        assert digest(source_manifest) == manifest['source_manifest_sha256']
        counts = dict(mirna=0, target=0)
        for shard in manifest['shards']:
            base_path = source_manifest.parent / shard['base_shard']
            control_path = directory / shard['sidecar']
            assert digest(base_path) == shard['base_sha256']
            assert digest(control_path) == shard['sha256']
            base = torch.load(base_path, map_location='cpu', weights_only=True)
            control = torch.load(control_path, map_location='cpu', weights_only=True)
            for axis in counts:
                original, shuffled = base[f'{axis}_phact'], control[f'{axis}_phact']
                assert original.shape == shuffled.shape and original.shape[0] == shard['rows']
                assert torch.isfinite(original).all() and torch.isfinite(shuffled).all()
                changed_rows = int((original != shuffled).flatten(start_dim=1).any(dim=1).sum())
                assert changed_rows == shard['audit'][axis]['changed_rows']
                counts[axis] += changed_rows
            print(f'Verified {split}/{shard["base_shard"]}', flush=True)
        verified[split] = {axis: dict(changed_rows=count, rows=manifest['row_count'],
                                      fraction_changed=count / manifest['row_count'])
                           for axis, count in counts.items()}
    report = dict(status='passed', source_and_sidecar_hashes_checked=10,
                  tensor_row_change_counts=verified,
                  scope='Evaluation control hashes, shape, finite scores, and actual changed-row counts; no new shuffling or training.')
    (ROOT / 'analysis_2026-09-07/verified_controls.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
