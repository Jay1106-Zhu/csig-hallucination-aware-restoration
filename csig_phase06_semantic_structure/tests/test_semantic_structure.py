import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from semantic_structure import (
    CandidateScores,
    decide_candidate,
    extract_structure_priors,
    propose_rois,
    resize_label_map_nearest,
    structure_mask,
)


class SemanticStructureContracts(unittest.TestCase):
    def test_line_prior_does_not_drop_short_facade_segments_in_large_images(self):
        image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        image[400:403, 200:360] = 255
        mask = np.ones((1024, 1024), dtype=bool)
        priors = extract_structure_priors(image, mask)
        self.assertGreater(np.count_nonzero(priors["lines"]), 100)

    def test_resize_label_map_preserves_discrete_labels(self):
        source = np.array([[0, 1], [2, 3]], dtype=np.int32)

        resized = resize_label_map_nearest(source, (4, 6))

        self.assertEqual(resized.shape, (4, 6))
        self.assertEqual(set(np.unique(resized)), {0, 1, 2, 3})
        np.testing.assert_array_equal(resized[:2, :3], 0)
        np.testing.assert_array_equal(resized[2:, 3:], 3)

    def test_structure_mask_selects_only_requested_classes(self):
        labels = np.array([[1, 2, 3], [4, 2, 1]], dtype=np.int32)

        selected = structure_mask(labels, {1, 3})

        np.testing.assert_array_equal(
            selected,
            np.array([[True, False, True], [False, False, True]]),
        )

    def test_structure_priors_are_native_resolution_and_masked(self):
        image = np.zeros((32, 40, 3), dtype=np.uint8)
        image[8:24, 8:32] = 255
        mask = np.zeros((32, 40), dtype=bool)
        mask[4:28, 4:36] = True

        priors = extract_structure_priors(image, mask)

        self.assertEqual(set(priors), {"edge", "horizontal", "vertical", "lines"})
        for prior in priors.values():
            self.assertEqual(prior.shape, mask.shape)
            self.assertEqual(prior.dtype, np.uint8)
            self.assertTrue(np.all(prior[~mask] == 0))

    def test_roi_proposals_clip_padding_and_filter_small_components(self):
        labels = np.zeros((10, 12), dtype=np.int32)
        labels[0:3, 0:4] = 7
        labels[8:9, 10:11] = 7

        rois = propose_rois(
            labels,
            {7: "building"},
            target_labels={"building"},
            min_area=3,
            padding=4,
        )

        self.assertEqual(len(rois), 1)
        self.assertEqual(rois[0].label, "building")
        self.assertEqual((rois[0].x0, rois[0].y0), (0, 0))
        self.assertEqual((rois[0].x1, rois[0].y1), (8, 7))

    def test_candidate_veto_keeps_h200_without_clear_gain(self):
        scores = CandidateScores(
            h200_overall=0.80,
            expert_overall=0.83,
            h200_structure=0.80,
            expert_structure=0.90,
            expert_artifact_free=True,
        )

        decision = decide_candidate(scores, minimum_gain=0.05)

        self.assertEqual(decision.winner, "H200")
        self.assertIn("gain", decision.reason)

    def test_candidate_veto_accepts_only_clean_structural_gain(self):
        scores = CandidateScores(
            h200_overall=0.80,
            expert_overall=0.90,
            h200_structure=0.70,
            expert_structure=0.90,
            expert_artifact_free=True,
        )

        decision = decide_candidate(scores, minimum_gain=0.05)

        self.assertEqual(decision.winner, "EXPERT")

    def test_candidate_veto_rejects_artifact_even_with_score_gain(self):
        scores = CandidateScores(
            h200_overall=0.80,
            expert_overall=0.95,
            h200_structure=0.70,
            expert_structure=0.95,
            expert_artifact_free=False,
        )

        decision = decide_candidate(scores, minimum_gain=0.05)

        self.assertEqual(decision.winner, "H200")
        self.assertIn("artifact", decision.reason)


if __name__ == "__main__":
    unittest.main()
