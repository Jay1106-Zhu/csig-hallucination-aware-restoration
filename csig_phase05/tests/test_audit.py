import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_phase05 import require, assert_index_order


class AuditContracts(unittest.TestCase):
    def test_audit_fails_closed(self):
        with self.assertRaises(AssertionError):
            require(False, 'mapping mismatch')

    def test_index_reorder_is_rejected(self):
        first = [{'image_id': 'a', 'region_index': '0'}, {'image_id': 'b', 'region_index': '0'}]
        assert_index_order(first, first)
        with self.assertRaises(AssertionError):
            assert_index_order(first, first[::-1])


if __name__ == '__main__':
    unittest.main()
