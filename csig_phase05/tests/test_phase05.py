import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT.parent / 'csig_phase0/dependencies'))

import numpy as np
from core import (
    context_crop, generate_degradation, grouped_splits, metric_gains,
    pairwise_global, partition_regions, risk_coverage, token_correspondence,
    validate_annotation, weighted_scene_samples,
)


class Phase05Contracts(unittest.TestCase):
    def test_partitions_cover_every_pixel_once(self):
        regions = partition_regions(481, 321, 256)
        coverage = np.zeros((321, 481), dtype=np.int32)
        for region in regions:
            coverage[region['y']:region['y'] + region['height'], region['x']:region['x'] + region['width']] += 1
        np.testing.assert_array_equal(coverage, 1)

    def test_context_uses_reflection_and_keeps_center(self):
        image = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
        crop = context_crop(image, 16, 12, 16)
        np.testing.assert_array_equal(crop, image[4:20, 8:24])
        self.assertEqual(context_crop(image, 0, 0, 64).shape, (64, 64, 3))

    def test_synthetic_degradation_is_reproducible_and_aligned(self):
        image = np.full((64, 64, 3), 128, dtype=np.uint8)
        first = generate_degradation(image, 'noise_jpeg', 505)
        second = generate_degradation(image, 'noise_jpeg', 505)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, image.shape)
        self.assertFalse(np.array_equal(first, image))

    def test_all_gain_directions_mean_improvement(self):
        gains = metric_gains({'psnr': 20, 'ssim': 0.5, 'lpips': 0.4, 'dists': 0.3},
                             {'psnr': 22, 'ssim': 0.7, 'lpips': 0.2, 'dists': 0.1})
        self.assertTrue(all(value > 0 for value in gains.values()))

    def test_group_split_never_leaks_related_images(self):
        groups = np.repeat(np.arange(35), 4)
        seen = []
        for training, validation in grouped_splits(groups, 5):
            self.assertFalse(set(groups[training]) & set(groups[validation]))
            seen.extend(validation)
        self.assertEqual(sorted(seen), list(range(len(groups))))

    def test_pairwise_features_are_ordered_and_symmetric_parts(self):
        first = np.array([1., 0., 0.])
        second = np.array([0., 1., 0.])
        result = pairwise_global(first, second)
        self.assertEqual(result.shape, (14,))
        self.assertEqual(result[-2], 0)
        self.assertAlmostEqual(result[-1], np.sqrt(2))

    def test_token_identity_and_displacement(self):
        tokens = np.eye(4, dtype=np.float32)
        identity = token_correspondence(tokens, tokens, 2)
        np.testing.assert_allclose(identity['disagreement'], 0)
        self.assertEqual(identity['summary'][4], 0)
        permuted = token_correspondence(tokens, tokens[[1, 0, 3, 2]], 2)
        self.assertGreater(permuted['summary'][4], 0)

    def test_pending_or_ai_annotations_not_human_groundtruth(self):
        pending = {'severity': '', 'reviewer_type': '', 'annotation_status': 'pending'}
        self.assertFalse(validate_annotation(pending))
        self.assertFalse(validate_annotation({'severity': 2, 'reviewer_type': 'assistant', 'annotation_status': 'reviewed'}))
        self.assertTrue(validate_annotation({'severity': 2, 'reviewer_type': 'human', 'annotation_status': 'reviewed'}))
        with self.assertRaises(ValueError):
            validate_annotation({'severity': 4, 'reviewer_type': 'human', 'annotation_status': 'reviewed'})

    def test_risk_coverage_uses_scores_not_gt_for_selection(self):
        result = risk_coverage(np.array([0.8, 0.2, 0.9, 0.1]), np.array([1., -1., -2., 2.]), [0.5, 1.])
        self.assertAlmostEqual(result[0]['damage_rate'], 0.5)
        self.assertAlmostEqual(result[0]['mean_gain'], -0.5)
        self.assertEqual(result[0]['selected_indices'], [2, 0])

    def test_scene_weights_equal_mass(self):
        weights = weighted_scene_samples(np.array(['a', 'a', 'b']))
        self.assertAlmostEqual(weights[0] + weights[1], weights[2])


if __name__ == '__main__':
    unittest.main()
