import unittest
import sys
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT / 'scripts'))
sys.path.insert(0, str(PHASE_ROOT / 'dependencies'))

import numpy as np
import torch

from phase0_core import (
    ConfidenceMLP,
    aggregate_heatmap,
    binary_metrics,
    fit_scaler,
    grouped_folds,
    image_balanced_weights,
    patch_coordinates,
    patch_quality,
    statistical_features,
    transform_features,
    validate_feature_index,
)
from report_utils import select_failures, spatial_statistics
from audit_utils import validate_fold_checkpoint


class Phase0Contracts(unittest.TestCase):
    def test_grid_matches_guide(self):
        coordinates = patch_coordinates(4096, 3072, 256, 128)
        self.assertEqual(len(coordinates), 713)
        self.assertEqual(coordinates[0], (0, 0))
        self.assertEqual(coordinates[-1], (3840, 2816))

    def test_nondivisible_edges_are_covered_without_duplicate(self):
        coordinates = patch_coordinates(300, 270, 256, 128)
        self.assertEqual(coordinates, [(0, 0), (44, 0), (0, 14), (44, 14)])
        with self.assertRaises(ValueError):
            patch_coordinates(200, 270, 256, 128)

    def test_gain_and_strict_good_label(self):
        target = np.full((8, 8, 3), 100, dtype=np.uint8)
        worse = target + 20
        better = target + 10
        result = patch_quality(worse, better, target)
        self.assertAlmostEqual(result['gain'], 6.020599913, places=6)
        self.assertEqual(result['label'], 'GOOD')
        self.assertEqual(patch_quality(worse, worse, target)['label'], 'BAD')

    def test_identical_perfect_predictions_have_zero_gain(self):
        target = np.zeros((8, 8, 3), dtype=np.uint8)
        result = patch_quality(target, target, target)
        self.assertEqual(result['gain'], 0)
        self.assertEqual(result['label'], 'BAD')

    def test_groups_never_overlap_and_oof_covers_once(self):
        groups = np.array(['one'] * 5 + ['two'] * 7 + ['three'] * 3)
        observed = []
        for train, heldout, image_id in grouped_folds(groups):
            self.assertFalse(set(groups[train]) & set(groups[heldout]))
            self.assertEqual(set(groups[heldout]), {image_id})
            observed.extend(heldout.tolist())
        self.assertEqual(sorted(observed), list(range(len(groups))))

    def test_scaler_is_fit_only_on_training(self):
        training = np.array([[1, 2], [3, 2]], dtype=np.float32)
        heldout = np.array([[100, 4]], dtype=np.float32)
        scaler = fit_scaler(training)
        np.testing.assert_allclose(scaler['mean'], [2, 2])
        np.testing.assert_allclose(transform_features(heldout, scaler), [[98, 2]])

    def test_image_weights_equalize_original_images(self):
        weights = image_balanced_weights(np.array(['one', 'one', 'two']))
        self.assertAlmostEqual(float(weights[:2].sum()), float(weights[2]))
        self.assertAlmostEqual(float(weights.mean()), 1.0)

    def test_stats_are_finite_and_do_not_accept_gt(self):
        image = np.full((256, 256, 3), 128, dtype=np.uint8)
        features = statistical_features(image, image)
        self.assertEqual(features.shape, (24,))
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_allclose(features[-4:], 0)
        with self.assertRaises(TypeError):
            statistical_features(image, image, image)

    def test_overlap_heatmap_averages_predictions(self):
        heatmap = aggregate_heatmap(3, 2, [(0, 0), (1, 0)], [0.2, 0.8], 2)
        np.testing.assert_allclose(heatmap, [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8]])

    def test_single_class_auc_is_unavailable_not_fabricated(self):
        metrics = binary_metrics(np.zeros(4), np.zeros(4))
        self.assertIsNone(metrics['roc_auc'])
        self.assertEqual(metrics['accuracy'], 1)
        self.assertEqual(metrics['good_count'], 0)

    def test_feature_index_rejects_reordering(self):
        validate_feature_index(['a', 'b'], ['a', 'b'], np.ones((2, 3)))
        with self.assertRaises(ValueError):
            validate_feature_index(['a', 'b'], ['b', 'a'], np.ones((2, 3)))

    def test_mlp_probability_and_reload(self):
        model = ConfidenceMLP(24, 64)
        features = torch.ones((3, 24))
        probability = model(features)
        self.assertEqual(tuple(probability.shape), (3,))
        self.assertTrue(torch.all((probability >= 0) & (probability <= 1)))
        reloaded = ConfidenceMLP(24, 64)
        reloaded.load_state_dict(model.state_dict())
        torch.testing.assert_close(probability, reloaded(features))

    def test_failure_selection_is_bad_and_nonoverlapping(self):
        import pandas as pd

        table = pd.DataFrame({'x': [0, 128, 256, 512], 'y': [0] * 4,
                              'gain': [-10, -9, -5, 1], 'label': ['BAD', 'BAD', 'BAD', 'GOOD']})
        selected = select_failures(table, count=2, patch_size=256)
        self.assertEqual(selected.x.tolist(), [0, 256])
        self.assertTrue((selected.label == 'BAD').all())

    def test_spatial_agreement_reports_independent_expectation(self):
        import pandas as pd

        table = pd.DataFrame({'x': [0, 128, 0, 128], 'y': [0, 0, 128, 128],
                              'label': ['BAD', 'BAD', 'GOOD', 'GOOD'], 'gain': [-1, -1, 1, 1]})
        statistics = spatial_statistics(table)
        self.assertEqual(statistics['adjacent_pairs'], 4)
        self.assertEqual(statistics['adjacent_same_label'], 0.5)
        self.assertEqual(statistics['independent_expected_agreement'], 0.5)

    def test_audit_rejects_heldout_leakage_and_wrong_scaler(self):
        features = np.array([[1, 2], [3, 4]], dtype=np.float32)
        payload = {'training_images': ['one'], 'heldout_image': 'two',
                   'mean': torch.tensor([2., 3.]), 'scale': torch.tensor([1., 1.])}
        validate_fold_checkpoint(payload, features, ['one'], 'two')
        with self.assertRaises(ValueError):
            validate_fold_checkpoint({**payload, 'training_images': ['one', 'two']}, features, ['one'], 'two')
        with self.assertRaises(ValueError):
            validate_fold_checkpoint({**payload, 'mean': torch.tensor([5., 3.])}, features, ['one'], 'two')


if __name__ == '__main__':
    unittest.main()
