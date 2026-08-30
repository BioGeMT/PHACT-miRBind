#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260708
SRC: Path
IN: Path
OUT: Path
PARAM1_PROFILE: Path
SPLITS = ['train', 'validation', 'mini_train']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create the deterministic miRNA-grouped Agentomics train/validation split.'
    )
    parser.add_argument('--source', type=Path, required=True, help='Labeled Agentomics split root.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--param1-profile', type=Path, required=True)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def read_labels():
    labels = pd.read_csv(SRC / 'labels.csv')
    if 'label' in labels.columns:
        col = 'label'
    elif 'numeric_label' in labels.columns:
        col = 'numeric_label'
    else:
        raise ValueError(f'No label column found in {SRC / "labels.csv"}: {labels.columns.tolist()}')
    labels = labels[['id', col]].copy()
    labels[col] = labels[col].astype(int)
    labels = labels.rename(columns={col: 'label'})
    if labels['id'].duplicated().any():
        raise ValueError('Duplicate ids in labels.csv')
    if set(labels['label'].unique()) - {0, 1}:
        raise ValueError('Labels are not binary 0/1')
    return labels


def objective(n, p, target_n, target_p):
    if n <= 0:
        # allow objective but penalize empty validation heavily
        return 1e9
    target_neg = target_n - target_p
    neg = n - p
    return ((n - target_n) / target_n) ** 2 + ((p - target_p) / target_p) ** 2 + ((neg - target_neg) / target_neg) ** 2


def improve_mask(mask, counts, pos, target_n, target_p, rng, passes=8):
    n = int(counts[mask].sum())
    p = int(pos[mask].sum())
    cur = objective(n, p, target_n, target_p)
    g = len(counts)
    for _ in range(passes):
        improved = False
        for i in rng.permutation(g):
            if mask[i]:
                nn = n - int(counts[i]); pp = p - int(pos[i])
            else:
                nn = n + int(counts[i]); pp = p + int(pos[i])
            if nn <= 0 or nn >= int(counts.sum()):
                continue
            cand = objective(nn, pp, target_n, target_p)
            # deterministic small tolerance
            if cand + 1e-18 < cur:
                mask[i] = ~mask[i]
                n, p, cur = nn, pp, cand
                improved = True
        if not improved:
            break
    return mask, n, p, cur


def choose_validation_groups(merged):
    grouped = (merged.groupby('noncodingRNA', sort=False)['label']
               .agg(n='count', pos='sum')
               .reset_index())
    counts = grouped['n'].to_numpy(dtype=np.int64)
    pos = grouped['pos'].to_numpy(dtype=np.int64)
    total_n = int(counts.sum())
    total_p = int(pos.sum())
    target_n = int(round(0.20 * total_n))
    target_p = target_n * (total_p / total_n)
    rng = np.random.default_rng(SEED)

    starts = []
    g = len(counts)

    # Greedy starts, with largest groups first and several random orders.
    for order in [np.argsort(-counts), np.argsort(counts), rng.permutation(g)]:
        mask = np.zeros(g, dtype=bool)
        n = 0; p = 0; cur = objective(1, 0, target_n, target_p)
        for i in order:
            nn = n + int(counts[i]); pp = p + int(pos[i])
            cand = objective(nn, pp, target_n, target_p)
            if n == 0 or cand < cur:
                mask[i] = True; n = nn; p = pp; cur = cand
        starts.append(mask)

    # Random starts around 20% of groups; local search fixes sample imbalance.
    for _ in range(250):
        starts.append(rng.random(g) < 0.20)

    best = None
    for mask in starts:
        if not mask.any() or mask.all():
            continue
        mask = mask.copy()
        mask, n, p, obj = improve_mask(mask, counts, pos, target_n, target_p, rng)
        if p <= 0 or n - p <= 0:
            continue
        if best is None or obj < best[0] or (abs(obj - best[0]) < 1e-18 and n < best[1]):
            best = (obj, n, p, mask.copy())
    if best is None:
        raise RuntimeError('Failed to choose validation groups')
    _, val_n, val_p, val_mask = best
    val_groups = set(grouped.loc[val_mask, 'noncodingRNA'].astype(str))
    return val_groups, grouped, {'target_n': target_n, 'target_pos': target_p, 'val_n': int(val_n), 'val_pos': int(val_p)}


def write_labels(labels, mask, split_name):
    out = labels.loc[mask, ['id', 'label']]
    out.to_csv(OUT / split_name / 'labels.csv', index=False)
    return {'n': int(len(out)), 'pos': int(out['label'].sum()), 'neg': int((1 - out['label']).sum()),
            'pos_rate': float(out['label'].mean())}


def write_split_map(labels, val_ids, mini_ids):
    path = OUT / 'split_id_map.tsv'
    with path.open('w') as f:
        for sid in labels['id'].astype(str).to_numpy():
            if sid in mini_ids:
                code = 'M'       # mini_train, also written to train
            elif sid in val_ids:
                code = 'V'
            else:
                code = 'T'
            f.write(f'{sid}\t{code}\n')
    return path


def awk_filter_by_first_column(map_path, src_file, out_files):
    # map codes: T=train only, V=validation only, M=train + mini_train
    awk_script = r'''
NR==FNR {grp[$1]=$2; next}
FNR==1 {print $0 > train_out; print $0 > val_out; print $0 > mini_out; next}
{
  s=grp[$1]
  if (s=="T" || s=="M") print $0 > train_out
  else if (s=="V") print $0 > val_out
  if (s=="M") print $0 > mini_out
}
'''
    cmd = [
        'awk', '-F', '\t',
        '-v', f'train_out={out_files["train"]}',
        '-v', f'val_out={out_files["validation"]}',
        '-v', f'mini_out={out_files["mini_train"]}',
        awk_script, str(map_path), str(src_file)
    ]
    env = os.environ.copy()
    env['LC_ALL'] = 'C'
    subprocess.run(cmd, check=True, env=env)


def split_semicolon_ids(series):
    out = set()
    for value in series.astype(str):
        if not value or value == 'NA' or value == 'nan':
            continue
        for part in value.split(';'):
            part = part.strip()
            if part and part != 'NA' and part != 'nan':
                out.add(part)
    return out


def filter_reference_tables(samples_key, candidate_path, train_ids, val_ids, mini_ids):
    split_id_sets = {'train': train_ids, 'validation': val_ids, 'mini_train': mini_ids}

    # Candidate mature IDs needed by each split for phact_mirna_positions.tsv.
    cand = pd.read_csv(candidate_path, sep='\t', dtype=str, keep_default_na=False,
                       usecols=['id', 'mirgenedb_mature_id'])
    mature_sets = {}
    for split, ids in split_id_sets.items():
        vals = cand.loc[cand['id'].isin(ids), 'mirgenedb_mature_id']
        mature_sets[split] = {v for v in vals.astype(str) if v and v != 'NA' and v != 'nan'}

    mirna_phact = pd.read_csv(PARAM1_PROFILE, sep='\t', dtype=str, keep_default_na=False)
    for split in SPLITS:
        out_path = OUT / split / 'input' / 'phact_mirna_positions.tsv'
        mirna_phact.loc[mirna_phact['mirgenedb_mature_id'].isin(mature_sets[split])].to_csv(out_path, sep='\t', index=False)

    # Pre-miRNA IDs needed by each split for mirgenedb_premirna_orthologues.tsv.
    premirna_sets = {}
    for split, ids in split_id_sets.items():
        premirna_sets[split] = split_semicolon_ids(samples_key.loc[samples_key['id'].isin(ids), 'mirgenedb_premirna_id'])

    ortho = pd.read_csv(IN / 'mirgenedb_premirna_orthologues.tsv', sep='\t', dtype=str, keep_default_na=False)
    for split in SPLITS:
        out_path = OUT / split / 'input' / 'mirgenedb_premirna_orthologues.tsv'
        ortho.loc[ortho['mirgenedb_premirna_id'].isin(premirna_sets[split])].to_csv(out_path, sep='\t', index=False)

    return {
        split: {
            'mature_ids_in_phact_mirna_positions': int(len(mature_sets[split])),
            'premirna_ids_in_orthologues': int(len(premirna_sets[split])),
        } for split in SPLITS
    }


def main():
    global SRC, IN, OUT, PARAM1_PROFILE
    args = parse_args()
    SRC = args.source.expanduser().resolve()
    IN = SRC / 'input'
    OUT = args.output.expanduser().resolve()
    PARAM1_PROFILE = args.param1_profile.expanduser().resolve()
    for required in (SRC / 'labels.csv', IN, PARAM1_PROFILE):
        if not required.exists():
            raise FileNotFoundError(required)
    existing = [OUT / split for split in SPLITS if (OUT / split).exists()]
    if existing and not args.force:
        raise FileExistsError(
            f'Output split directories already exist: {existing}; use --force to replace them'
        )
    np.random.seed(SEED)
    # Recreate only our required split directories.
    for split in SPLITS:
        d = OUT / split
        if d.exists():
            shutil.rmtree(d)
        (d / 'input').mkdir(parents=True, exist_ok=True)

    labels = read_labels()
    print(f'Loaded {len(labels)} labels', flush=True)

    samples_key = pd.read_csv(IN / 'samples.tsv', sep='\t', dtype=str, keep_default_na=False,
                              usecols=['id', 'noncodingRNA', 'mirgenedb_premirna_id'])
    if samples_key['id'].duplicated().any():
        raise ValueError('Duplicate ids in samples.tsv')
    merged = labels.merge(samples_key[['id', 'noncodingRNA']], on='id', how='inner', validate='one_to_one')
    if len(merged) != len(labels):
        raise ValueError('labels.csv and samples.tsv ids do not match one-to-one')

    val_groups, grouped, group_info = choose_validation_groups(merged)
    merged['is_val'] = merged['noncodingRNA'].isin(val_groups)
    val_ids = set(merged.loc[merged['is_val'], 'id'].astype(str))
    train_ids = set(merged.loc[~merged['is_val'], 'id'].astype(str))

    rng = np.random.default_rng(SEED)
    train_label_df = labels[labels['id'].isin(train_ids)]
    mini_chosen = []
    for cls in [0, 1]:
        ids = train_label_df.loc[train_label_df['label'] == cls, 'id'].astype(str).to_numpy()
        take = min(100, len(ids))
        mini_chosen.extend(rng.choice(ids, size=take, replace=False).tolist())
    mini_ids = set(mini_chosen)
    if not mini_ids.issubset(train_ids):
        raise ValueError('mini_train ids are not a subset of train ids')

    val_mask = labels['id'].isin(val_ids)
    train_mask = labels['id'].isin(train_ids)
    mini_mask = labels['id'].isin(mini_ids)
    if set(labels.loc[val_mask, 'id']).intersection(set(labels.loc[train_mask, 'id'])):
        raise ValueError('train and validation labels overlap')

    label_summary = {
        'train': write_labels(labels, train_mask, 'train'),
        'validation': write_labels(labels, val_mask, 'validation'),
        'mini_train': write_labels(labels, mini_mask, 'mini_train'),
    }
    print('Label counts: ' + json.dumps(label_summary, sort_keys=True), flush=True)

    map_path = write_split_map(labels, val_ids, mini_ids)

    # Filter sample-keyed input files in one pass per source file.
    for fname in ['samples.tsv', 'sample_mirna_candidates.tsv', 'phact_target_positions.tsv']:
        print(f'Filtering {fname}', flush=True)
        awk_filter_by_first_column(
            map_path,
            IN / fname,
            {split: OUT / split / 'input' / fname for split in SPLITS}
        )

    print('Filtering reference tables', flush=True)
    ref_summary = filter_reference_tables(samples_key, IN / 'sample_mirna_candidates.tsv', train_ids, val_ids, mini_ids)

    # Confirm no miRNA sequence overlap between train and validation.
    train_mirnas = set(merged.loc[~merged['is_val'], 'noncodingRNA'])
    val_mirnas = set(merged.loc[merged['is_val'], 'noncodingRNA'])
    overlap = train_mirnas.intersection(val_mirnas)
    if overlap:
        raise ValueError(f'noncodingRNA overlap between train and validation: {len(overlap)}')

    summary = {
        'seed': SEED,
        'source': str(SRC),
        'split_strategy': 'Deterministic grouped holdout by exact noncodingRNA sequence. Validation groups selected by randomized greedy/local search to approximate 20% of samples and the global class balance; mini_train is a deterministic 100-per-class subset of train.',
        'label_summary': label_summary,
        'group_info': {
            **group_info,
            'total_noncodingRNA_groups': int(grouped.shape[0]),
            'validation_noncodingRNA_groups': int(len(val_groups)),
            'train_noncodingRNA_groups': int(grouped.shape[0] - len(val_groups)),
            'noncodingRNA_overlap_count': int(len(overlap)),
        },
        'reference_table_summary': ref_summary,
        'label_columns_written': ['id', 'label'],
        'input_files': ['samples.tsv', 'sample_mirna_candidates.tsv', 'phact_mirna_positions.tsv', 'phact_target_positions.tsv', 'mirgenedb_premirna_orthologues.tsv'],
        'paths': {split: str(OUT / split) for split in SPLITS},
    }
    with (OUT / 'split_summary.json').open('w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    # Remove temporary split map after filtering; split assignments are represented by labels.csv files.
    try:
        map_path.unlink()
    except FileNotFoundError:
        pass
    print('Done: ' + json.dumps(summary['paths'], sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
