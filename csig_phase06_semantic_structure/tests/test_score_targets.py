import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from score_targets import ImageScore, summarize_scores
from scoring_protocol import apply_hypir_proxy


class TargetScoringContracts(unittest.TestCase):
    def test_subset_scores_are_not_reported_as_full_set_proxy(self):
        scores = [
            ImageScore(case_id=37, path=Path("case37.png"), clipiqa=0.40, topiq=0.50),
            ImageScore(case_id=42, path=Path("case42.png"), clipiqa=0.60, topiq=0.70),
        ]

        summary = summarize_scores(scores, family="hypir")

        self.assertEqual(summary.count, 2)
        self.assertEqual(summary.clipiqa_mean, 0.50)
        self.assertEqual(summary.topiq_mean, 0.60)
        self.assertIsNone(summary.proxy_score)
        self.assertEqual(summary.scope, "targeted_diagnostic_subset")

    def test_fixed_affine_is_reported_only_for_all_100_distinct_cases(self):
        scores = [ImageScore(case_id=index, path=Path(f"case{index}.png"), clipiqa=0.5, topiq=0.6) for index in range(1, 101)]
        summary = summarize_scores(scores, family="hypir")
        self.assertAlmostEqual(summary.proxy_score, 2.9577007988480113)
        self.assertEqual(summary.scope, "full_set")

    def test_duplicate_cases_cannot_inflate_coverage_or_mean(self):
        row = ImageScore(case_id=42, path=Path("case42.png"), clipiqa=0.5, topiq=0.6)
        with self.assertRaises(ValueError):
            summarize_scores([row, row], family="hypir")

    def test_cross_model_summary_keeps_proxy_unset(self):
        scores = [
            ImageScore(case_id=37, path=Path("case37.png"), clipiqa=0.40, topiq=0.50),
        ]

        summary = summarize_scores(scores, family="cross_model")

        self.assertIsNone(summary.proxy_score)

    def test_empty_summary_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_scores([], family="hypir")


if __name__ == "__main__":
    unittest.main()
