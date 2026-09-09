import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from prepare_controls import decode_sequences, position_permutation, prepare_split, reference_contrast, shuffle_profiles


def make_grid(mirna, target):
    return np.where((mirna[:, :, None] < 4) & (target[:, None, :] < 4),
                    mirna[:, :, None] * 4 + target[:, None, :], 17).astype(np.uint8)


class PositionalControlTests(unittest.TestCase):
    def test_decoder_recovers_unknown_and_padding_without_changing_orientation(self):
        mirna = np.array([[0, 3, 4, 2, 4]], dtype=np.uint8)
        target = np.array([[4, 1, 2, 3, 0, 4]], dtype=np.uint8)
        decoded = decode_sequences(make_grid(mirna, target))
        np.testing.assert_array_equal(decoded["mirna"], mirna)
        np.testing.assert_array_equal(decoded["target"], target)

    def test_decoder_rejects_inconsistent_grid(self):
        grid = make_grid(np.array([[0, 1]], dtype=np.uint8), np.array([[1, 2]], dtype=np.uint8))
        grid[0, 0, 1] = 14
        with self.assertRaisesRegex(ValueError, "consistent"):
            decode_sequences(grid)

    def test_fixed_permutation_keeps_quartets_missing_and_ambiguous_sites(self):
        sequence = np.array([[0, 1, 2, 3, 0, 1, 2, 3, 4, 4]], dtype=np.uint8)
        missing = np.zeros((1, 10, 1), dtype=np.uint8)
        missing[0, 2, 0] = 1
        missing[0, 9, 0] = 1
        scores = np.arange(40, dtype=np.float16).reshape(1, 10, 4) / 64
        scores[missing[:, :, 0] == 1] = 0.5
        original = scores.copy()
        shuffled, audit = shuffle_profiles(scores, missing, sequence, "mirna", {})
        permutation = position_permutation(sequence[0], missing[0, :, 0], "mirna")
        np.testing.assert_array_equal(shuffled[0], original[0, permutation])
        np.testing.assert_array_equal(scores, original)
        np.testing.assert_array_equal(shuffled[0, [2, 8, 9]], original[0, [2, 8, 9]])
        self.assertEqual(audit["scored_invalid_positions_preserved"], 1)
        self.assertEqual(audit["changed_rows"], 1)

    def test_same_key_uses_same_permutation_for_different_profiles_and_split_orders(self):
        sequences = np.tile(np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint8), (2, 1))
        missing = np.zeros((2, 8, 1), dtype=np.uint8)
        scores = np.arange(64, dtype=np.float16).reshape(2, 8, 4) / 128
        joint, _ = shuffle_profiles(scores, missing, sequences, "target", {})
        for index in (1, 0):
            separate, _ = shuffle_profiles(scores[index:index + 1], missing[index:index + 1],
                                           sequences[index:index + 1], "target", {})
            np.testing.assert_array_equal(separate[0], joint[index])
        np.testing.assert_array_equal(joint[1] - joint[0], np.full((8, 4), 0.25, dtype=np.float16))

    def test_unavailable_and_positionally_constant_profiles_remain_identical(self):
        sequences = np.tile(np.array([0, 1, 2, 3, 4], dtype=np.uint8), (2, 1))
        missing = np.array([[[1]] * 5, [[0]] * 4 + [[1]]], dtype=np.uint8)
        scores = np.full((2, 5, 4), 0.5, dtype=np.float16)
        scores[1, :4] = [0.1, 0.2, 0.3, 0.4]
        shuffled, audit = shuffle_profiles(scores, missing, sequences, "mirna", {})
        np.testing.assert_array_equal(shuffled, scores)
        self.assertEqual(audit["no_eligible_positions_rows"], 1)
        self.assertEqual(audit["eligible_constant_rows"], 1)

    def test_shuffle_preserves_base_conditional_quartets_and_variable_contrast_multiset(self):
        sequences = np.array([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]], dtype=np.uint8)
        missing = np.zeros((1, 12, 1), dtype=np.uint8)
        scores = np.arange(48, dtype=np.float16).reshape(1, 12, 4) / 64
        shuffled, _ = shuffle_profiles(scores, missing, sequences, "target", {})
        for base in range(4):
            positions = sequences[0] == base
            self.assertEqual(sorted(map(tuple, scores[0, positions])), sorted(map(tuple, shuffled[0, positions])))
        np.testing.assert_array_equal(np.sort(reference_contrast(scores, sequences), axis=1),
                                      np.sort(reference_contrast(shuffled, sequences), axis=1))

    def test_contrast_flat_profile_stays_flat_and_singleton_group_stays_fixed(self):
        sequences = np.array([[0, 1, 2, 3, 0, 1, 2]], dtype=np.uint8)
        missing = np.zeros((1, 7, 1), dtype=np.uint8)
        scores = np.empty((1, 7, 4), dtype=np.float16)
        base_to_channel = [0, 3, 1, 2]
        for position, base in enumerate(sequences[0]):
            reference_channel = base_to_channel[base]
            alternatives = [channel for channel in range(4) if channel != reference_channel]
            scores[0, position, reference_channel] = 0.75
            scores[0, position, alternatives] = [0.125, 0.25, 0.375] if position < 4 else [0.375, 0.25, 0.125]
        shuffled, audit = shuffle_profiles(scores, missing, sequences, "mirna", {})
        np.testing.assert_array_equal(reference_contrast(shuffled, sequences), np.full((1, 7), 0.5))
        np.testing.assert_array_equal(shuffled[0, 3], scores[0, 3])
        self.assertEqual(audit["singleton_reference_base_positions_unchanged"], 1)
        self.assertEqual(audit["reference_contrast_flat_rows"], 1)

    def test_real_entrypoint_sidecars_are_label_blind_and_leave_source_unchanged(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            base = root / "cache" / "train"
            base.mkdir(parents=True)
            sequences = np.tile(np.array([0, 1, 2, 3, 0, 1], dtype=np.uint8), (2, 1))
            scores = np.arange(48, dtype=np.float16).reshape(2, 6, 4) / 64
            shard = {"pair_indices": torch.from_numpy(make_grid(sequences, sequences)),
                     "mirna_phact": torch.from_numpy(scores), "target_phact": torch.from_numpy(scores.copy()),
                     "mirna_phact_missing": torch.zeros((2, 6, 1), dtype=torch.uint8),
                     "target_phact_missing": torch.zeros((2, 6, 1), dtype=torch.uint8),
                     "labels": torch.tensor([0.0, 1.0])}
            source_path = base / "train_shard_00000.pt"
            torch.save(shard, source_path)
            (base / "train_manifest.json").write_text(json.dumps({"row_count": 2, "shards": [
                {"file": source_path.name, "rows": 2}]}))
            original = source_path.read_bytes()
            manifest = prepare_split(root / "cache", root / "first", "train", {})
            self.assertEqual(source_path.read_bytes(), original)
            sidecar = torch.load(root / "first" / "train" / source_path.name, weights_only=True)
            self.assertEqual(set(sidecar), {"mirna_phact", "target_phact"})
            self.assertTrue(manifest["source_manifest_unchanged"])
            shard["labels"] = 1 - shard["labels"]
            torch.save(shard, source_path)
            prepare_split(root / "cache", root / "second", "train", {})
            second = torch.load(root / "second" / "train" / source_path.name, weights_only=True)
            for field in sidecar:
                self.assertTrue(torch.equal(sidecar[field], second[field]))


if __name__ == "__main__":
    unittest.main()
