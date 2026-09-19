import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from roi_experiments import composite_candidate, context_box, screening_status


class ROIExperimentContracts(unittest.TestCase):
    def test_context_padding_is_clipped_without_resizing(self):
        self.assertEqual(context_box((2, 3, 10, 12), (20, 30), 5), (0, 0, 15, 17))

    def test_protected_pixels_and_roi_exterior_remain_bit_identical(self):
        base = np.full((14, 16, 3), 100, dtype=np.uint8)
        patch = np.full((8, 8, 3), 220, dtype=np.uint8)
        allowed = np.ones((8, 8), dtype=bool)
        allowed[3:5, 3:5] = False
        output = composite_candidate(base, patch, (4, 2, 12, 10), allowed, 2)
        np.testing.assert_array_equal(output[:2], base[:2])
        np.testing.assert_array_equal(output[10:], base[10:])
        np.testing.assert_array_equal(output[:, :4], base[:, :4])
        np.testing.assert_array_equal(output[:, 12:], base[:, 12:])
        np.testing.assert_array_equal(output[5:7, 7:9], base[5:7, 7:9])
        self.assertGreater(int(output[4, 6, 0]), 100)
        self.assertLessEqual(int(output[4, 6, 0]), 220)

    def test_unprotected_interior_uses_full_candidate_weight(self):
        base = np.full((14, 16, 3), 100, dtype=np.uint8)
        patch = np.full((8, 8, 3), 220, dtype=np.uint8)
        allowed = np.ones((8, 8), dtype=bool)
        output = composite_candidate(base, patch, (4, 2, 12, 10), allowed, 2)
        self.assertEqual(int(output[4, 6, 0]), 220)

    def test_small_positive_main_gain_is_screened_not_auto_rejected_at_005(self):
        status = screening_status(0.002, 0.001, 0.03, 0.02)
        self.assertEqual(status, "VISUAL_REVIEW_REQUIRED")

    def test_negative_primary_metric_cannot_pass_on_secondary_gain(self):
        self.assertEqual(screening_status(-0.002, 0.01, 0.03, 0.02), "KEEP_H200_PRIMARY_REGRESSION")

    def test_secondary_tradeoff_needs_review_not_auto_acceptance(self):
        self.assertEqual(screening_status(0.002, -0.001, 0.03, 0.02), "REVIEW_SECONDARY_TRADEOFF")

    def test_nonfinite_scores_cannot_produce_candidate_acceptance(self):
        with self.assertRaises(ValueError):
            screening_status(float("nan"), 0.01, 0.03, 0.02)


if __name__ == "__main__":
    unittest.main()
