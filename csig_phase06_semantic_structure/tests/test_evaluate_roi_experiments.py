import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_roi_experiments import paired_metrics


class PairedMetricsContracts(unittest.TestCase):
    def test_secondary_is_not_added_to_primary_and_no_auto_acceptance(self):
        baseline = {"clipiqa": 0.4, "topiq": 0.5}
        candidate = {"clipiqa": 0.401, "topiq": 0.49}
        roi_baseline = {"clipiqa": 0.3, "topiq": 0.4}
        roi_candidate = {"clipiqa": 0.33, "topiq": 0.42}
        result = paired_metrics(baseline, candidate, roi_baseline, roi_candidate)
        self.assertAlmostEqual(result["global_clipiqa_delta"], 0.001)
        self.assertAlmostEqual(result["global_topiq_delta"], -0.01)
        self.assertAlmostEqual(result["roi_clipiqa_delta"], 0.03)
        self.assertIsNone(result["proxy_score"])
        self.assertEqual(result["screening"], "REVIEW_SECONDARY_TRADEOFF")
        self.assertEqual(result["visual_status"], "PENDING")


if __name__ == "__main__":
    unittest.main()
