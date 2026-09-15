import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from annotations import choose_regions, validate_review
from extract_features import stability_summary, feature_variants


class PipelineContracts(unittest.TestCase):
    def test_annotation_sampling_fixed_without_scores(self):
        first = choose_regions(192, 505, 2)
        self.assertEqual(first, choose_regions(192, 505, 2))
        self.assertEqual(len(set(first)), 2)

    def test_review_requires_identity_type_and_valid_severity(self):
        base = {'reviewer_id': 'reviewer_1', 'reviewed_at': '2026-09-15', 'reviewer_type': 'human', 'annotation_status': 'reviewed', 'severity': '2', 'hallucination_types': 'structure_invention'}
        self.assertTrue(validate_review(base))
        with self.assertRaises(ValueError):
            validate_review({**base, 'reviewer_id': ''})
        with self.assertRaises(ValueError):
            validate_review({**base, 'hallucination_types': 'fake'})
        self.assertFalse(validate_review({**base, 'reviewer_type': 'assistant'}))

    def test_identical_conditions_have_zero_sensitivity(self):
        images = np.full((4, 32, 32, 3), 100, dtype=np.uint8)
        cls = np.ones((4, 8), dtype=np.float32)
        tokens = np.ones((4, 16, 8), dtype=np.float32)
        features = stability_summary(images, cls, tokens, [0, 0, 0])
        np.testing.assert_allclose(features, 0)
        images[0] = 0
        self.assertGreater(stability_summary(images, cls, tokens, [0, 0, 0])[0], 0)

    def test_feature_variants_have_only_explicit_feature_inputs(self):
        features = {'dino_lq': np.zeros((3, 4)), 'dino_restored': np.ones((3, 4)), 'pairwise': np.zeros((3, 18)), 'token256': np.zeros((3, 16)), 'token512': np.zeros((3, 16)), 'structure': np.zeros((3, 37)), 'stability': np.zeros((3, 12)), 'clip': np.zeros((3, 8)), 'stats': np.zeros((3, 24))}
        variants = feature_variants(features)
        self.assertEqual(variants['V7'].shape, (3, 67))
        self.assertEqual(variants['V8'].shape, (3, 40))
        self.assertEqual(variants['V3'].shape, (3, 32))
        np.testing.assert_array_equal(variants['C1_absdiff'], 1)


if __name__ == '__main__':
    unittest.main()
