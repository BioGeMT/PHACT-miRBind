#!/usr/bin/env python3
"""Report exposure denominators and distinguish family labels from missing values."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workspace import WORKSPACE, DATASETS, historical_path

import json

import pandas as pd

ROOT = WORKSPACE / "runs/followup-2026-09-06/analysis"
CACHE = WORKSPACE / "data/caches/phact_cache_CountNodes_3"
DATA = DATASETS
COLUMNS = ['noncodingRNA', 'noncodingRNA_name', 'noncodingRNA_fam', 'mirgenedb_family', 'gene_cluster_ID']
MISSING = {'', 'NA', 'NAN', 'NONE'}


def annotations(frame):
    result = pd.DataFrame(index=frame.index)
    result['exact_sequence'] = frame.noncodingRNA.str.upper().str.replace('U', 'T', regex=False)
    seed = result.exact_sequence.str.slice(1, 8)
    result['seed_2_8'] = seed.where(seed.str.fullmatch('[ACGT]{7}'), 'NA')
    result['original_family'] = frame.noncodingRNA_fam.str.strip().str.upper()
    result['mirgenedb_family'] = frame.mirgenedb_family.str.strip().str.upper()
    result['family_with_fallback'] = result.mirgenedb_family.where(
        ~result.mirgenedb_family.isin(MISSING), result.original_family)
    result['target_cluster'] = frame.gene_cluster_ID.str.strip().str.upper()
    return result


def read_chunks(path):
    return pd.read_csv(path, sep='\t', keep_default_na=False, dtype=str, usecols=COLUMNS, chunksize=250000)


def main():
    manifest_path = CACHE / 'train/train_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source = historical_path(manifest['source_file'])
    seen = {}
    training_rows = 0
    for frame in read_chunks(source):
        values = annotations(frame)
        for key in values:
            seen.setdefault(key, set()).update(set(values[key]) - MISSING)
        training_rows += len(frame)
    assert training_rows == manifest['row_count']
    sources = [{'split': 'fitting', 'path': str(source), 'rows': training_rows,
                'manifest': str(manifest_path)}]
    rows, shared_mirgenedb, missing_match = [], [], []
    for split in ['val', 'test', 'leftout']:
        split_manifest_path = CACHE / split / f'{split}_manifest.json'
        split_manifest = json.loads(split_manifest_path.read_text())
        path = historical_path(split_manifest['source_file'])
        frame = pd.concat(read_chunks(path), ignore_index=True)
        assert len(frame) == split_manifest['row_count']
        sources.append({'split': split, 'path': str(path), 'rows': len(frame),
                        'manifest': str(split_manifest_path)})
        values = annotations(frame)
        for name in values:
            column = values[name]
            missing = column.isin(MISSING)
            unique = set(column) - MISSING
            shared = unique & seen[name]
            matched = column.isin(seen[name]) & ~missing
            rows.append({'split': split, 'definition': name, 'rows': len(frame),
                         'training_distinct': len(seen[name]), 'distinct_nonmissing': len(unique),
                         'shared_distinct': len(shared), 'shared_distinct_fraction': len(shared) / len(unique),
                         'missing_rows': int(missing.sum()), 'matched_rows': int(matched.sum()),
                         'matched_row_fraction_all_rows': matched.mean(),
                         'matched_row_fraction_annotated': matched.sum() / (~missing).sum(),
                         'exact_guides_in_matched_rows': values.loc[matched, 'exact_sequence'].nunique()})
        if split == 'leftout':
            for index in frame.index[values.mirgenedb_family.isin(seen['mirgenedb_family'])]:
                shared_mirgenedb.append({'row_id': f'leftout_{index + 1}', **frame.loc[index].to_dict()})
            naive = values.mirgenedb_family.isin(seen['mirgenedb_family'] | {'NA'})
            missing_match.append({'split': split, 'naive_seen_rows_including_NA': int(naive.sum()),
                                  'rows': len(frame), 'fraction': naive.mean(),
                                  'unmapped_NA_rows': int((values.mirgenedb_family == 'NA').sum())})
    pd.DataFrame(rows).to_csv(ROOT / 'exposure_definitions.tsv', sep='\t', index=False)
    pd.DataFrame(shared_mirgenedb).to_csv(ROOT / 'leftout_shared_mirgenedb_rows.tsv', sep='\t', index=False)
    (ROOT / 'exposure_audit.json').write_text(json.dumps({'sources': sources, 'missing_value_match': missing_match,
                                                       'all_row_counts_match_cache_manifests': True}, indent=2) + '\n')
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(missing_match))


if __name__ == '__main__':
    main()
