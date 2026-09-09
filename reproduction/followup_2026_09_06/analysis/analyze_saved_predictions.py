#!/usr/bin/env python3
"""Describe retained PHACT prediction differences without training any model."""
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workspace import WORKSPACE, DATASETS, historical_path

import hashlib
import json

import numpy as np
import pandas as pd
import scipy
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

import analyze_param1_evidence as previous

ROOT = WORKSPACE
OUT = ROOT / 'runs/followup-2026-09-06/analysis'
DATA = DATASETS
OLD_CACHE = WORKSPACE / "data/caches/phact_cache_CountNodes_3"
RELEASE = WORKSPACE / "archive/legacy_leaderboard_artifacts"
MODEL_PATHS = {
    'sequence_model': 'seq_only', 'target': 'phact_countnodes3_target',
    'phylop': 'conservation_phylop', 'phastcons': 'conservation_phastcons',
    'conservation_both': 'conservation_both',
}
PAIR_BASES = {base: index for index, base in enumerate('ATCG')}
SCORE_BASES = {base: index for index, base in enumerate('ACGT')}
MISSING_FAMILIES = {'', 'NA', 'nan', 'None'}
INPUT_FILES = set()


def save(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, sep='\t', index=False, na_rep='NA')


def sequence_indices(sequences, length, codes):
    result = np.full((len(sequences), length), -1, dtype=np.int8)
    for index, sequence in enumerate(sequences):
        values = [codes.get(base, -1) for base in sequence[:length]]
        result[index, :len(values)] = values
    return result


def normalize_sequences(values):
    return values.str.upper().str.replace('U', 'T', regex=False)


def read_training_exposure():
    manifest_path = OLD_CACHE / 'train/train_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    path = historical_path(manifest['source_file'])
    INPUT_FILES.update([path, manifest_path])
    seen = {key: set() for key in ['sequence', 'family', 'target_cluster']}
    row_count = 0
    for chunk in pd.read_csv(path, sep='\t', keep_default_na=False, dtype=str,
                             usecols=['noncodingRNA', 'mirgenedb_family', 'gene_cluster_ID'],
                             chunksize=250000):
        seen['sequence'].update(normalize_sequences(chunk.noncodingRNA))
        seen['family'].update(set(chunk.mirgenedb_family) - MISSING_FAMILIES)
        seen['target_cluster'].update(chunk.gene_cluster_ID)
        row_count += len(chunk)
    assert row_count == manifest['row_count']
    return seen, row_count


def profile_properties(scores, missing, sequences, length):
    bases = sequence_indices(sequences, length, SCORE_BASES)
    valid = (~missing) & (bases >= 0)
    if np.any((~missing) & (bases < 0)):
        raise AssertionError('A scored position has no reference nucleotide')
    reference = np.take_along_axis(scores, np.maximum(bases, 0)[..., None], axis=2)[..., 0]
    contrast = reference - (scores.sum(axis=2) - reference) / 3.0
    count = valid.sum(axis=1)
    minimum = np.where(valid, contrast, np.inf).min(axis=1)
    maximum = np.where(valid, contrast, -np.inf).max(axis=1)
    ranges = maximum - minimum
    ranges[count == 0] = np.nan
    means = np.divide(np.where(valid, contrast, 0).sum(axis=1), count,
                      out=np.full(len(scores), np.nan), where=count > 0)
    categories = np.where(count == 0, 'missing', np.where(ranges <= 1e-6, 'flat', 'variable'))
    valid_length = (bases >= 0).sum(axis=1)
    coverage = np.divide(count, valid_length, out=np.zeros(len(count)), where=valid_length > 0)
    return {'category': categories, 'range': ranges, 'mean_contrast': means,
            'positions': count, 'fraction': coverage}


def add_cached_properties(frame, split):
    cache = ROOT / 'runs/param1-training/cache/param_1_target_score' / split
    manifest_path = cache / f'{split}_manifest.json'
    INPUT_FILES.add(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest['row_count'] == len(frame)
    output = []
    offset = 0
    old_paths = sorted((OLD_CACHE / split).glob('*_shard_*.pt'))
    paths = sorted(cache.glob('*_shard_*.pt'))
    assert len(paths) == len(old_paths)
    for path, old_path in zip(paths, old_paths):
        INPUT_FILES.update([path, old_path])
        shard = torch.load(path, map_location='cpu', weights_only=True)
        old = torch.load(old_path, map_location='cpu', weights_only=True)
        for key in ['pair_indices', 'target_phact', 'target_phact_missing', 'labels']:
            assert torch.equal(shard[key], old[key]), (path, key)
        del old
        count = len(shard['labels'])
        rows = frame.iloc[offset:offset + count]
        assert np.array_equal(shard['labels'].numpy(), rows.label.to_numpy())
        # Validate every cached pair cell against source sequences, in small batches.
        guide_bases = sequence_indices(rows.sequence, 28, PAIR_BASES)
        target_bases = sequence_indices(rows.target_sequence, 50, PAIR_BASES)
        for start in range(0, count, 5000):
            guide = guide_bases[start:start + 5000, :, None]
            target = target_bases[start:start + 5000, None, :]
            expected = np.where((guide >= 0) & (target >= 0), guide * 4 + target, 17)
            assert np.array_equal(expected, shard['pair_indices'][start:start + 5000].numpy())
        properties = {}
        for axis, sequences, length in [('guide', rows.sequence, 28), ('target', rows.target_sequence, 50)]:
            key = 'mirna' if axis == 'guide' else 'target'
            values = profile_properties(shard[f'{key}_phact'].float().numpy(),
                                        shard[f'{key}_phact_missing'][..., 0].bool().numpy(),
                                        sequences, length)
            properties.update({f'{axis}_{name}': value for name, value in values.items()})
        output.append(pd.DataFrame(properties))
        offset += count
    assert offset == len(frame)
    properties = pd.concat(output, ignore_index=True)
    for column in properties:
        frame[column] = properties[column].to_numpy()
    frame['target_coverage'] = np.where(frame.target_fraction == 0, 'missing',
                                       np.where(frame.target_fraction == 1, 'complete', 'partial'))
    return {'cache_rows': offset, 'all_pair_cells_match_source': True,
            'p1_target_tensors_bitwise_equal_original': True,
            'target_positions_missing': int(50 * len(frame) - frame.target_positions.sum())}


def load_split(split, seen):
    path = DATA / f'AGO2_eCLIP_Manakov2022_{split}.tsv'
    INPUT_FILES.add(path)
    columns = ['gene', 'noncodingRNA', 'noncodingRNA_name', 'mirgenedb_family',
               'noncodingRNA_fam', 'mirgenedb_mature_id', 'dominant_region', 'gene_cluster_ID', 'label']
    frame = pd.read_csv(path, sep='\t', usecols=columns, keep_default_na=False,
                        dtype={name: str for name in columns if name != 'label'})
    assert len(frame) == {'test': 324171, 'leftout': 20054}[split]
    frame['row_id'] = [f'{split}_{index + 1}' for index in range(len(frame))]
    frame['sequence'] = normalize_sequences(frame.noncodingRNA)
    frame['target_sequence'] = normalize_sequences(frame.gene)
    frame['family'] = frame.mirgenedb_family.where(~frame.mirgenedb_family.isin(MISSING_FAMILIES), 'UNMAPPED')
    frame['family_exposure'] = np.where(frame.family == 'UNMAPPED', 'unmapped',
                                       np.where(frame.family.isin(seen['family']), 'seen', 'unseen'))
    frame['sequence_exposure'] = np.where(frame.sequence.isin(seen['sequence']), 'seen', 'unseen')
    frame['target_cluster_exposure'] = np.where(frame.gene_cluster_ID.isin(seen['target_cluster']), 'seen', 'unseen')
    for model, folder in MODEL_PATHS.items():
        prediction_path = RELEASE / 'predictions' / folder / f'{split}.csv'
        INPUT_FILES.add(prediction_path)
        frame[model] = previous.load_prediction_csv(prediction_path, split, len(frame))
    p1_path = ROOT / 'releases/mirbench_v7/predictions/phact_p1_mirna_cnn' / f'{split}.csv'
    INPUT_FILES.add(p1_path)
    frame['p1_guide'] = previous.load_prediction_csv(p1_path, split, len(frame))
    npz = previous.model_prediction_paths(ROOT, RELEASE, split)['sequence_only']
    INPUT_FILES.add(npz)
    sequence_npz = previous.load_sequence_predictions(npz, frame.label.to_numpy())
    maximum_difference = float(np.max(np.abs(frame.sequence_model.to_numpy() - sequence_npz)))
    assert maximum_difference < 1e-6, (split, maximum_difference)
    # Use the same retained sequence baseline as the completed P1 evidence analysis.
    frame['sequence_model'] = sequence_npz
    frame.attrs['sequence_csv_npz_maximum_difference'] = maximum_difference
    return frame


def metrics(frame):
    positive = int(frame.label.sum())
    result = {'rows': len(frame), 'positive': positive, 'negative': len(frame) - positive,
              'prevalence': positive / len(frame) if len(frame) else np.nan,
              'miRNAs': int(frame.sequence.nunique()),
              'ap_status': 'defined' if 0 < positive < len(frame) else 'one_class'}
    for model in MODELS:
        result[f'ap_{model}'] = (float(average_precision_score(frame.label, frame[model]))
                                  if result['ap_status'] == 'defined' else np.nan)
    for model in MODELS[1:]:
        result[f'delta_{model}_minus_sequence'] = result[f'ap_{model}'] - result['ap_sequence_model']
    result['delta_target_minus_conservation_both'] = result['ap_target'] - result['ap_conservation_both']
    return result


def paired_bootstrap(frame, split, subgroup, pairs, seed):
    if frame.label.nunique() != 2:
        return []
    models = sorted({name for pair in pairs for name in pair})
    distributions, groups = previous.cluster_bootstrap_average_precision(
        frame.label.to_numpy(), frame.sequence.to_numpy(),
        {model: frame[model].to_numpy() for model in models}, 1000, seed, 5)
    rows = []
    for first, second in pairs:
        values = distributions[first] - distributions[second]
        low, high = np.quantile(values, [0.025, 0.975])
        rows.append({'split': split, 'subgroup': subgroup, 'model_a': first, 'model_b': second,
                     'rows': len(frame), 'clusters': groups, 'replicates': 1000, 'seed': seed,
                     'ap_a': average_precision_score(frame.label, frame[first]),
                     'ap_b': average_precision_score(frame.label, frame[second]),
                     'delta': average_precision_score(frame.label, frame[first]) - average_precision_score(frame.label, frame[second]),
                     'ci_low': low, 'ci_high': high, 'fraction_delta_positive': (values > 0).mean()})
    return rows


def group_results(frame, split, grouping):
    rows = []
    for name, group in frame.groupby(grouping, sort=True):
        row = {'split': split, grouping: name, **metrics(group)}
        for column in ['noncodingRNA_name', 'family', 'mirgenedb_mature_id', 'family_exposure', 'guide_category']:
            if column != grouping:
                row[column] = '|'.join(sorted(group[column].unique()))
        for property_name in ['guide_range', 'guide_mean_contrast', 'target_range', 'target_mean_contrast', 'target_fraction']:
            row[f'mean_{property_name}'] = float(group[property_name].mean())
        row['target_seen_fraction'] = float((group.target_cluster_exposure == 'seen').mean())
        row['utr3_fraction'] = float((group.dominant_region == 'UTR3').mean())
        row['mean_prediction_delta_target_minus_sequence'] = float((group.target - group.sequence_model).mean())
        rows.append(row)
    return rows


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


MODELS = ['sequence_model', 'target', 'phylop', 'phastcons', 'conservation_both', 'p1_guide']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    print('Reading training exposure', flush=True)
    seen, training_rows = read_training_exposure()
    overall, strata, bootstrap, guides, families, macros, associations = [], [], [], [], [], [], []
    validation = {'training_rows': training_rows, 'training_sequences': len(seen['sequence']),
                  'training_families': len(seen['family']), 'training_target_clusters': len(seen['target_cluster'])}
    for split_index, split in enumerate(['test', 'leftout']):
        print(f'Loading {split}', flush=True)
        frame = load_split(split, seen)
        validation[split] = add_cached_properties(frame, split)
        validation[split]['sequence_csv_npz_maximum_difference'] = frame.attrs['sequence_csv_npz_maximum_difference']
        overall.append({'split': split, **metrics(frame)})
        expected = {'test': [0.877697, 0.886028, 0.885284, 0.884541, 0.885649, 0.877736],
                    'leftout': [0.870129, 0.862818, 0.863503, 0.867934, 0.866147, 0.870263]}
        for model, score in zip(MODELS, expected[split]):
            assert abs(overall[-1][f'ap_{model}'] - score) < 0.0000006, (split, model)
        print(f'{split}: identities, original target tensors and reference AP verified', flush=True)
        for axis in ['guide', 'target']:
            available = frame[f'{axis}_range'].dropna()
            thresholds = np.quantile(available, [1/3, 2/3])
            values = np.where(frame[f'{axis}_range'].isna(), 'missing',
                              np.where(frame[f'{axis}_range'] <= thresholds[0], 'lower_third',
                                       np.where(frame[f'{axis}_range'] <= thresholds[1], 'middle_third', 'upper_third')))
            frame[f'{axis}_range_group'] = values
            validation[split][f'{axis}_range_thresholds'] = thresholds.tolist()
        for category in ['target_coverage', 'target_category', 'guide_category', 'family_exposure',
                         'sequence_exposure', 'target_cluster_exposure', 'dominant_region',
                         'guide_range_group', 'target_range_group']:
            for group_name, group in frame.groupby(category, sort=True):
                strata.append({'split': split, 'stratum_type': category, 'stratum': group_name, **metrics(group)})
        guide_rows = group_results(frame, split, 'sequence')
        family_rows = group_results(frame, split, 'family')
        guides.extend(guide_rows)
        families.extend(family_rows)
        for grouping, rows in [('miRNA', guide_rows), ('family', family_rows)]:
            table = pd.DataFrame(rows)
            valid = table[table.ap_status == 'defined']
            if grouping == 'family':
                valid = valid[valid.family != 'UNMAPPED']
            for comparator in ['sequence', 'conservation_both']:
                column = f'delta_target_minus_{comparator}'
                macros.append({'split': split, 'grouping': grouping, 'comparator': comparator,
                               'groups_total': len(table), 'groups_two_class_mapped': len(valid),
                               'one_class_groups': int((table.ap_status == 'one_class').sum()),
                               'positive_deltas': int((valid[column] > 0).sum()),
                               'negative_deltas': int((valid[column] < 0).sum()),
                               'mean_delta': valid[column].mean(), 'median_delta': valid[column].median(),
                               'row_weighted_mean_delta': np.average(valid[column], weights=valid.rows)})
        table = pd.DataFrame(guide_rows)
        for property_name in ['rows', 'prevalence', 'mean_guide_range', 'mean_guide_mean_contrast',
                              'mean_target_range', 'mean_target_mean_contrast', 'mean_target_fraction',
                              'target_seen_fraction', 'utr3_fraction']:
            selected = table.loc[(table.ap_status == 'defined') & (table.rows >= 20),
                                 [property_name, 'delta_target_minus_sequence']].dropna()
            coefficient = (spearmanr(selected.iloc[:, 0], selected.iloc[:, 1]).statistic
                           if selected.iloc[:, 0].nunique() > 1 else np.nan)
            associations.append({'split': split, 'property': property_name, 'guides': len(selected),
                                 'minimum_rows_per_guide': 20, 'spearman_rho': coefficient})
        print(f'{split}: bootstrap 1000 exact-miRNA replicates', flush=True)
        pairs = [('target', model) for model in ['sequence_model', 'phylop', 'phastcons', 'conservation_both']]
        bootstrap.extend(paired_bootstrap(frame, split, 'all', pairs, 20260830 + split_index))
        if split == 'leftout':
            for category in ['target_coverage', 'family_exposure', 'target_cluster_exposure']:
                for group_name, group in frame.groupby(category, sort=True):
                    bootstrap.extend(paired_bootstrap(group, split, f'{category}:{group_name}',
                                                       [('target', 'sequence_model'), ('target', 'conservation_both')],
                                                       20260906))
        row_columns = ['row_id', 'sequence', 'family', 'family_exposure', 'target_cluster_exposure',
                       'dominant_region', 'label', 'guide_category', 'guide_range', 'guide_mean_contrast',
                       'target_coverage', 'target_category', 'target_range', 'target_mean_contrast'] + MODELS
        frame[row_columns].to_csv(OUT / f'{split}_joined_predictions.tsv.gz', sep='\t', index=False, na_rep='NA')
        print(f'{split}: done', flush=True)
    for name, rows in [('overall_metrics', overall), ('stratum_metrics', strata), ('paired_bootstrap', bootstrap),
                       ('per_mirna', guides), ('per_family', families), ('macro_deltas', macros),
                       ('guide_property_associations', associations)]:
        save(name + '.tsv', rows)
    save('top_guide_deltas.tsv', [row for split in ['test', 'leftout'] for row in
         sorted([row for row in guides if row['split'] == split and row['rows'] >= 100 and row['ap_status'] == 'defined'],
                key=lambda row: abs(row['delta_target_minus_sequence']), reverse=True)[:20]])
    validation.update({'all_checks_passed': True, 'training_performed': False, 'gpu_used': False,
                       'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
                       'torch': torch.__version__, 'bootstrap_replicates': 1000})
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print('Hashing input provenance', flush=True)
    save('input_manifest.tsv', [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': checksum(path)}
                                for path in sorted(INPUT_FILES)])
    print('Complete', flush=True)


if __name__ == '__main__':
    main()
