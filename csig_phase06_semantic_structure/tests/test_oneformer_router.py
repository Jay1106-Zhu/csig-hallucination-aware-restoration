import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from oneformer_router import STRUCTURE_LABELS, select_target_mask, target_label_ids


class OneFormerRouterContracts(unittest.TestCase):
    def test_ade20k_comma_aliases_are_not_silently_dropped(self):
        labels = {6: "road, route", 11: "sidewalk, pavement", 43: "signboard, sign", 61: "bridge, span"}
        self.assertEqual(target_label_ids(labels), {6: "road", 11: "sidewalk", 43: "signboard", 61: "bridge"})

    def test_structure_branch_protects_signboard_from_building_expert(self):
        labels = {1: "building", 43: "signboard", 2: "sky"}
        self.assertEqual(target_label_ids(labels, STRUCTURE_LABELS), {1: "building"})

    def test_target_label_ids_normalizes_model_label_names(self):
        labels = {0: "Building", 1: "window", 2: "sky"}

        selected = target_label_ids(labels, {"building", "sky"})

        self.assertEqual(selected, {0: "building", 2: "sky"})

    def test_select_target_mask_uses_only_requested_semantic_ids(self):
        label_map = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int32)

        selected = select_target_mask(label_map, {0, 2})

        np.testing.assert_array_equal(
            selected,
            np.array([[True, False, True], [True, False, True]]),
        )

    def test_empty_target_selection_returns_empty_mask(self):
        label_map = np.zeros((3, 4), dtype=np.int32)

        selected = select_target_mask(label_map, set())

        self.assertEqual(selected.dtype, bool)
        self.assertFalse(selected.any())


if __name__ == "__main__":
    unittest.main()
