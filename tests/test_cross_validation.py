"""Check row isolation, input encoding and checkpoint-selected evaluation."""

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from phact_mirbind.cache.fold_cache import (
    FoldBatches,
    FoldInputs,
    prepare_training_cache,
    target_scores,
)
from phact_mirbind.data.fold_features import prepare_fold_features
from phact_mirbind.data.folds import (
    assign_groups,
    build_conservation_folds,
    verify_conservation_folds,
)
from phact_mirbind.data.pair_encoding import encode_pair_indices
from phact_mirbind.training.cross_validation import make_model, train_fold


def write_table(path, rows, delimiter=","):
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def make_data(root):
    for directory in ["manakov", "datasets_p1", "mapping", "scores"]:
        (root / directory).mkdir()
    rng = np.random.default_rng(91)
    sequences = ["".join(rng.choice(list("ACGT"), 22)) for _ in range(82)]
    targets = ["".join(rng.choice(list("ACGT"), 50)) for _ in range(24)]
    raw, prepared, mapping, score_rows = [], [], [], []
    for m, sequence in enumerate(sequences[:80]):
        for pos, base in enumerate(sequence, 1):
            mapping.append(
                dict(
                    score_id=f"p{m}",
                    score_position_1based=pos,
                    mirgenedb_mature_id=f"p{m}",
                    mirna_position_1based=pos,
                    actual_nt=base,
                )
            )
            for b in "ACGT":
                score_rows.append(
                    dict(
                        ID=f"p{m}",
                        Position=pos,
                        Nucleotide=b,
                        PHACTn_gapAware_wtNTnorm_wl_param_1=0.2
                        + (0.2 + m / 200 if b == base else 0),
                    )
                )
        mirna_arrays = {
            b: [0.2 + (0.2 + m / 200 if b == base else 0) for base in sequence]
            for b in "ACGT"
        }
        for t, target in enumerate(targets):
            i = len(raw)
            row = dict(
                gene=target,
                noncodingRNA=sequence,
                label=t % 2,
                chr="chr1",
                start=t * 60,
                end=t * 60 + 49,
                strand="+",
                dominant_region="UTR3",
            )
            raw.append(row)
            scores = {
                b: [
                    float("nan")
                    if t == 0
                    else 0.1 + (0.3 + t / 100 if b == base else 0)
                    for base in target
                ]
                for b in "ACGT"
            }
            prepared.append(
                dict(
                    id=f"manakov_train_{i + 1}"
                    if i < 1200
                    else f"manakov_test_{i - 1200 + 1}",
                    gene=target,
                    noncodingRNA=sequence,
                    label=t % 2,
                    mirna_profile_id=f"p{m}",
                    **{
                        f"mirna_phact_{b}": json.dumps(v)
                        for b, v in mirna_arrays.items()
                    },
                    **{f"target_phact_{b}": json.dumps(v) for b, v in scores.items()},
                )
            )
    write_table(root / "manakov/train.tsv", raw[:1200], "\t")
    write_table(root / "manakov/test.tsv", raw[1200:], "\t")
    write_table(root / "datasets_p1/train.csv", prepared[:1000])
    write_table(root / "datasets_p1/validation.csv", prepared[1000:1200])
    write_table(root / "datasets_p1/test.csv", prepared[1200:])
    left_raw, left_prepared = [], []
    for i in range(2):
        r = dict(raw[i], noncodingRNA=sequences[80 + i])
        left_raw.append(r)
        left_prepared.append(
            dict(
                prepared[i],
                id=f"manakov_leftout_{i + 1}",
                noncodingRNA=r["noncodingRNA"],
            )
        )
    write_table(root / "manakov/leftout.tsv", left_raw, "\t")
    write_table(root / "datasets_p1/leftout.csv", left_prepared)
    write_table(root / "mapping/mirna_positions.tsv", mapping, "\t")
    with gzip.open(root / "scores/mirna_qntnorm_transformed.tsv.gz", "wt") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(score_rows)
    return raw, prepared


class CrossValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.raw, cls.prepared = make_data(cls.root)
        cls.folds = cls.root / "folds"
        prepare_fold_features(cls.root, cls.folds)

        # Small search for the synthetic fixture; production defaults are also
        # checked against all saved Manakov assignments outside this unit test.
        def small_search(x, weights, proportions, seed, restarts, steps, inner=False):
            return assign_groups(x, weights, proportions, seed, 2, 50, inner)

        with patch("phact_mirbind.data.folds.assign_groups", side_effect=small_search):
            build_conservation_folds(cls.folds)
        cls.verification = verify_conservation_folds(cls.folds, cls.root)
        cls.cache = cls.root / "cache"
        prepare_training_cache(cls.root, cls.folds, cls.cache)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_saved_fold_isolation_and_manifest(self):
        self.assertEqual(self.verification["status"], "passed")
        self.assertEqual(self.verification["rows"], 1920)
        with gzip.open(self.folds / "interaction_manifest.tsv.gz", "rt") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(rows[1199]["id"], "manakov_train_1200")
        self.assertEqual(rows[1200]["id"], "manakov_test_1")
        with np.load(self.folds / "assignments.npz") as data:
            self.assertTrue(np.all((data["roles"] == 3).sum(axis=1) == 1))

    def test_compact_batch_matches_stable_pair_encoder(self):
        inputs = FoldInputs(self.cache, self.folds, "target_phact", device="cpu")
        pairs, scores, labels = inputs.batch([0, 17, 1919, 1920])
        source = [self.raw[0], self.raw[17], self.raw[-1]]
        for k, row in enumerate(source[:3]):
            expected = encode_pair_indices(
                row["gene"], row["noncodingRNA"], target_length=50, mirna_length=28
            )
            np.testing.assert_array_equal(pairs[k], expected)
        self.assertEqual(tuple(scores.shape), (4, 5, 28, 50))
        self.assertTrue(torch.all(scores[0, :4] == 0.5))
        self.assertTrue(torch.all(scores[0, 4] == 1))
        self.assertTrue(torch.all(scores[1, 4] == 0))
        self.assertEqual(labels.tolist(), [0, 1, 1, 0])

    def test_partial_nan_quartet_rejected(self):
        row = {f"target_phact_{b}": json.dumps([0.2] * 50) for b in "ACGT"}
        row["target_phact_A"] = json.dumps([float("nan")] + [0.2] * 49)
        with self.assertRaises((AssertionError, ValueError)):
            target_scores(row)

    def test_common_weights_and_shuffle_are_matched(self):
        baseline = make_model("seq_only", 42, "cpu").state_dict()
        augmented = make_model("target_phact", 42, "cpu").state_dict()
        for name, value in baseline.items():
            actual = augmented[name]
            if name == "conv_layers.0.weight":
                actual = actual[:, :8]
            torch.testing.assert_close(actual, value, rtol=0, atol=0)
        inputs = FoldInputs(self.cache, self.folds, "seq_only", device="cpu")
        a = FoldBatches(inputs, np.arange(32), 16, shuffle=True)
        b = FoldBatches(inputs, np.arange(32), 16, shuffle=True)
        for _ in range(2):
            for x, y in zip(a, b, strict=True):
                torch.testing.assert_close(x[0], y[0], rtol=0, atol=0)

    def test_cache_assignment_tampering_rejected(self):
        old = np.load(self.cache / "roles.npy")
        changed = old.copy()
        changed[0, 0] = (changed[0, 0] + 1) % 4
        np.save(self.cache / "roles.npy", changed)
        try:
            with self.assertRaisesRegex(ValueError, "row roles"):
                FoldInputs(self.cache, self.folds, "seq_only", device="cpu")
        finally:
            np.save(self.cache / "roles.npy", old)

    def test_checkpoint_selection_and_final_predictions(self):
        for variant in ["seq_only", "target_phact"]:
            inputs = FoldInputs(self.cache, self.folds, variant, device="cpu")
            path = train_fold(
                inputs,
                self.root / "runs",
                fold=1,
                batch_size=128,
                num_epochs=1,
                patience=1,
            )
            evaluation = json.loads(path.read_text())
            self.assertEqual(evaluation["best_epoch"], 1)
            for role in ["seen_test", "unseen_test", "leftout"]:
                with np.load(path.parent / f"{role}_predictions.npz") as predictions:
                    self.assertTrue(np.isfinite(predictions["probability"]).all())
                    self.assertEqual(
                        len(predictions["label"]), evaluation["roles"][role]["rows"]
                    )
            self.assertTrue((path.parent / "complete.json").exists())
            with self.assertRaises(FileExistsError):
                train_fold(inputs, self.root / "runs", fold=1, num_epochs=1, patience=1)


if __name__ == "__main__":
    unittest.main()
