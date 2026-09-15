import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from selective_restoration import bootstrap_difference, budget_oracle_selection, pixel_oracle


class SelectiveContracts(unittest.TestCase):
    def test_bootstrap_pairs_images_and_is_reproducible(self):
        result = bootstrap_difference([1, 2, 3], [0, 1, 2], 505, 200)
        self.assertEqual(result['mean_difference'], 1)
        self.assertEqual(result['lower_95'], 1)
        self.assertEqual(result, bootstrap_difference([1, 2, 3], [0, 1, 2], 505, 200))

    def test_budget_oracle_maximizes_native_sse_not_patch_psnr(self):
        self.assertEqual(budget_oracle_selection(np.array([1, 100, -10, 5]), 2).tolist(), [1, 3])

    def test_pixel_oracle_never_worse_than_either_source(self):
        lq = np.array([[[0, 0, 0], [50, 50, 50]]], dtype=np.uint8)
        restored = np.array([[[100, 100, 100], [0, 0, 0]]], dtype=np.uint8)
        gt = np.zeros_like(lq)
        oracle, mask = pixel_oracle(lq, restored, gt)
        np.testing.assert_array_equal(oracle, gt)
        np.testing.assert_array_equal(mask, [[False, True]])


if __name__ == '__main__':
    unittest.main()
