import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluation import binary_metrics, fit_normalizer, fuse_regions, pixel_metrics, structure_features
from core import context_crop, partition_regions


class EvaluationContracts(unittest.TestCase):
    def test_native_metrics_identity_and_degradation(self):
        source = np.full((65, 225, 3), 120, dtype=np.uint8)
        self.assertAlmostEqual(pixel_metrics(source, source)['ssim'], 1)
        self.assertGreater(pixel_metrics(source, source)['psnr'], 100)
        self.assertAlmostEqual(pixel_metrics(source + 10, source)['psnr'], 28.1308036)

    def test_structure_identity_and_edges(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        changed = image.copy()
        changed[:, 32:] = 255
        self.assertTrue(np.isfinite(structure_features(image, image)).all())
        self.assertGreater(np.linalg.norm(structure_features(image, changed)), 0)

    def test_undefined_auc_not_silently_zero(self):
        metrics = binary_metrics(np.array([0, 0]), np.array([0.1, 0.2]))
        self.assertIsNone(metrics['roc_auc'])
        self.assertIsNone(metrics['pr_auc'])
        self.assertAlmostEqual(metrics['brier'], 0.025)

    def test_normalizer_train_only_scene_weighted(self):
        features = np.array([[0.], [0.], [10.]])
        mean, scale = fit_normalizer(features, np.array(['a', 'a', 'b']))
        self.assertAlmostEqual(mean[0], 5)
        self.assertAlmostEqual(scale[0], 5)

    def test_fusion_selects_true_pixels_only(self):
        source = np.zeros((3, 5, 3), dtype=np.uint8)
        restored = np.full_like(source, 255)
        regions = partition_regions(5, 3, 2)
        fused, mask = fuse_regions(source, restored, regions, [0, 5])
        self.assertEqual(mask.sum(), 5)
        np.testing.assert_array_equal(fused[mask], 255)
        np.testing.assert_array_equal(fused[~mask], 0)

    def test_context_matches_numpy_reflect_at_edges(self):
        image = np.arange(5 * 7 * 3).reshape(5, 7, 3)
        reference = np.pad(image, ((12, 12), (12, 12), (0, 0)), mode='reflect')
        np.testing.assert_array_equal(context_crop(image, 1, 2, 12), reference[8:20, 7:19])


if __name__ == '__main__':
    unittest.main()
