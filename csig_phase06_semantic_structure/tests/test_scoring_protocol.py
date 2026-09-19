import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scoring_protocol import (
    HYPIR_PROXY_INTERCEPT,
    HYPIR_PROXY_SLOPE,
    apply_hypir_proxy,
    candidate_case_ids,
    proxy_for_family,
)


class ScoringProtocolContracts(unittest.TestCase):
    def test_hypir_proxy_uses_the_fixed_affine(self):
        clipiqa_mean = 0.5

        score = apply_hypir_proxy(clipiqa_mean)

        self.assertEqual(
            score,
            HYPIR_PROXY_SLOPE * clipiqa_mean + HYPIR_PROXY_INTERCEPT,
        )

    def test_hypir_proxy_is_not_applied_to_cross_model_candidates(self):
        self.assertIsNone(proxy_for_family(0.5, "cross_model"))
        self.assertEqual(proxy_for_family(0.5, "hypir"), apply_hypir_proxy(0.5))

    def test_case_ids_must_match_before_comparison(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            for case_id in (37, 42):
                (first / f"case{case_id}.png").touch()
                (second / f"case{case_id}.png").touch()

            self.assertEqual(candidate_case_ids(first), (37, 42))
            self.assertEqual(candidate_case_ids(first, second), (37, 42))

            (second / "case65.png").touch()
            with self.assertRaises(ValueError):
                candidate_case_ids(first, second)


if __name__ == "__main__":
    unittest.main()
