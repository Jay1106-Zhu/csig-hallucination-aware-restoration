import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_diffir_control import blend_roi, clip_box, feather_alpha


class DiffIRControlContracts(unittest.TestCase):
    def test_clip_box_stays_inside_image(self):
        self.assertEqual(clip_box((-4, 3, 20, 40), width=16, height=32), (0, 3, 16, 32))

    def test_invalid_box_is_rejected(self):
        with self.assertRaises(ValueError):
            clip_box((8, 4, 8, 20), width=16, height=32)

    def test_feather_alpha_has_zero_edges_and_one_interior(self):
        alpha = feather_alpha((9, 9), feather=2)

        self.assertEqual(alpha.shape, (9, 9))
        self.assertEqual(alpha[0, 0], 0.0)
        self.assertEqual(alpha[4, 4], 1.0)

    def test_blend_roi_only_changes_the_selected_box(self):
        base = np.zeros((10, 12, 3), dtype=np.uint8)
        candidate = np.full((4, 5, 3), 255, dtype=np.uint8)

        blended = blend_roi(base, candidate, (3, 2, 8, 6), feather=0)

        np.testing.assert_array_equal(blended[:2], 0)
        np.testing.assert_array_equal(blended[2:6, 3:8], 255)
        np.testing.assert_array_equal(blended[6:], 0)


if __name__ == "__main__":
    unittest.main()
