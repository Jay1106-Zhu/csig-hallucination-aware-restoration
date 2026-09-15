import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_verifiers import fit_verifier, predict_checkpoint, macro_summary
from core import config


class VerifierContracts(unittest.TestCase):
    def test_checkpoint_matches_logistic_and_mlp_predictions(self):
        features = np.array([[0, 1], [1, 0], [3, 4], [4, 3]], dtype=np.float32)
        labels = np.array([0, 0, 1, 1])
        groups = np.array(['a', 'a', 'b', 'b'])
        for kind in ['logistic', 'mlp']:
            checkpoint = fit_verifier(features, labels, groups, kind, 505, config())
            first = predict_checkpoint(features, checkpoint)
            second = predict_checkpoint(features, checkpoint)
            np.testing.assert_array_equal(first, second)
            self.assertTrue(np.isfinite(first).all())
            self.assertTrue(((first >= 0) & (first <= 1)).all())
            np.testing.assert_allclose(checkpoint['mean'], [2, 2])

    def test_single_class_training_has_explicit_constant(self):
        checkpoint = fit_verifier(np.ones((3, 2)), np.zeros(3), np.array(['a', 'b', 'c']), 'logistic', 505, config())
        self.assertEqual(checkpoint['kind'], 'constant')
        np.testing.assert_array_equal(predict_checkpoint(np.zeros((4, 2)), checkpoint), 0)

    def test_macro_uses_images_and_counts_undefined(self):
        rows = [{'image_id': 'a', 'roc_auc': .8, 'brier': .1, 'positive_count': 1, 'region_count': 100},
                {'image_id': 'b', 'roc_auc': None, 'brier': .3, 'positive_count': 0, 'region_count': 2}]
        summary = macro_summary(rows)
        self.assertEqual(summary['roc_auc'], .8)
        self.assertAlmostEqual(summary['brier'], .2)
        self.assertEqual(summary['roc_auc_defined_images'], 1)
        self.assertEqual(summary['image_count'], 2)


if __name__ == '__main__':
    unittest.main()
