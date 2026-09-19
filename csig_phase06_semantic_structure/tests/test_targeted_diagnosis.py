import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_targeted_diagnosis import render_overlay
from targeted_diagnosis import TARGET_CASES, diagnose_image, target_case_paths


class TargetedNightDiagnosisContracts(unittest.TestCase):
    def test_target_case_inventory_is_the_low_score_night_cluster(self):
        self.assertEqual(TARGET_CASES, (37, 38, 39, 41, 42, 65))

        root = Path(__file__).resolve().parents[3]
        paths = target_case_paths(root / "csig_dataset" / "测试集")

        self.assertEqual(tuple(paths), tuple(root / "csig_dataset" / "测试集" / f"case{case_id}.jpg" for case_id in TARGET_CASES))
        self.assertTrue(all(path.is_file() for path in paths))

    def test_diagnosis_returns_native_resolution_proxies(self):
        image = np.zeros((64, 96, 3), dtype=np.uint8)
        image[20:28, 12:70] = (255, 220, 40)
        image[40:58, 18:82] = (80, 80, 80)

        diagnosis = diagnose_image(image, case_id=42)

        self.assertEqual(diagnosis.case_id, 42)
        self.assertEqual(diagnosis.image_shape, (64, 96, 3))
        self.assertGreater(diagnosis.highlight_core_area_ratio, 0.0)
        self.assertGreaterEqual(diagnosis.halo_candidate_area_ratio, 0.0)
        self.assertGreaterEqual(diagnosis.line_area_ratio, 0.0)
        self.assertLessEqual(diagnosis.highlight_core_area_ratio, 1.0)
        self.assertTrue(diagnosis.highlight_core.shape == (64, 96))
        self.assertTrue(diagnosis.halo_candidate.shape == (64, 96))
        self.assertTrue(diagnosis.structure_edge.shape == (64, 96))
        self.assertTrue(diagnosis.structure_lines.shape == (64, 96))

    def test_diagnosis_serialization_excludes_large_pixel_maps(self):
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        image[8:16, 8:34] = (255, 255, 255)

        payload = diagnose_image(image, case_id=37).to_dict()

        self.assertEqual(payload["case_id"], 37)
        self.assertEqual(payload["image_shape"], [32, 48, 3])
        self.assertIn("highlight_rois", payload)
        self.assertNotIn("highlight_core", payload)
        self.assertNotIn("structure_lines", payload)

    def test_diagnosis_overlay_is_renderable(self):
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        image[8:16, 8:34] = (255, 255, 255)
        diagnosis = diagnose_image(image, case_id=42)

        overlay = render_overlay(image, diagnosis)

        self.assertEqual(overlay.size, (48, 32))
        self.assertEqual(overlay.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
