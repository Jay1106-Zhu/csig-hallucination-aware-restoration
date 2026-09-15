import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from core import ROOT, read_json
from evaluation import read_csv


class DeliveryContracts(unittest.TestCase):
    def test_report_answers_all_final_questions_without_claiming_human_success(self):
        path = ROOT / 'reports/phase05_report.md'
        if not path.exists():
            self.skipTest('Report not generated yet')
        text = path.read_text(encoding='utf-8')
        for number in range(1, 11):
            self.assertIn(f'| Q{number} |', text)
        self.assertIn('HOLD', text)
        self.assertIn('pending', text)
        self.assertIn('1024', text)
        self.assertIn('not an equal-pixel budget', text)

    def test_annotation_interface_is_offline_and_score_blind(self):
        path = ROOT / 'data/annotations/index.html'
        if not path.exists():
            self.skipTest('Annotation interface not generated yet')
        text = path.read_text(encoding='utf-8')
        self.assertNotIn('__ROWS__', text)
        self.assertNotIn('__TYPES__', text)
        self.assertNotIn('fetch(', text)
        self.assertNotIn('probability', text)
        self.assertNotIn('psnr_gain', text)
        rows = read_csv(ROOT / 'data/annotations/hallucination_labels.csv')
        self.assertEqual(len(rows), 110)
        for row in rows:
            self.assertEqual(row['severity'], '')
            self.assertEqual(row['annotation_status'], 'pending')

    def test_completed_audit_preserves_phase0(self):
        path = ROOT / 'analysis/artifact_audit.json'
        if not path.exists():
            self.skipTest('Full computational audit not generated yet')
        audit = read_json(path)
        self.assertEqual(audit['status'], 'PASS')
        self.assertTrue(audit['phase0_unchanged'])
        self.assertEqual(audit['human_annotations_fabricated'], 0)
        self.assertEqual(audit['test_data_used'], 0)


if __name__ == '__main__':
    unittest.main()
