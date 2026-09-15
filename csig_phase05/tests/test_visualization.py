import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from visualize import region_heatmap, token_heatmap
from core import partition_regions


class VisualizationContracts(unittest.TestCase):
    def test_region_map_respects_valid_edges(self):
        regions = partition_regions(5, 3, 2)
        heatmap = region_heatmap(5, 3, regions, np.arange(len(regions)))
        self.assertEqual(heatmap.shape, (3, 5))
        self.assertEqual(heatmap[2, 4], 5)

    def test_token_map_uses_processor_central_crop_not_full_context(self):
        regions = partition_regions(256, 256, 256)
        heatmap = token_heatmap(256, 256, regions, np.ones((1, 16, 16)))
        self.assertTrue(np.isnan(heatmap[0, 0]))
        self.assertEqual(np.isfinite(heatmap).sum(), 224 * 224)
        self.assertEqual(heatmap[128, 128], 1)


if __name__ == '__main__':
    unittest.main()
