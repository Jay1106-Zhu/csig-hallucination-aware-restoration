import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from roi_experiments import infer_native


@unittest.skipUnless(importlib.util.find_spec("torch"), "Run with project .conda Python for tensor tests")
class NativeInferenceContracts(unittest.TestCase):
    def test_reflect_pad_and_unpad_preserve_rectangular_image_and_rgb(self):
        import torch

        rgb = np.random.default_rng(231).integers(0, 256, size=(35, 41, 3), dtype=np.uint8)
        restored, clipped = infer_native(torch.nn.Identity(), rgb, "cpu", 32)
        np.testing.assert_array_equal(restored, rgb)
        self.assertEqual(clipped, 0.0)

    def test_infinite_model_output_is_rejected_before_casting(self):
        import torch

        class NonfiniteModel(torch.nn.Module):
            def forward(self, image):
                return image * float("nan")

        with self.assertRaises(ValueError):
            infer_native(NonfiniteModel(), np.ones((32, 32, 3), dtype=np.uint8), "cpu", 32)


if __name__ == "__main__":
    unittest.main()
