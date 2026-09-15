import sys
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))
from build_phase05_release import classify_stage_path, public_text, git_allowed, source_name
from publish_phase05_release import asset_matches


class Phase05PublicationTests(unittest.TestCase):
    def test_public_git_and_archive_text_share_lf_line_endings(self):
        self.assertEqual(public_text('csig_phase05/analysis/example.json', '{}\r\n'), '{}\n')

    def test_empty_git_index_is_not_accepted_as_verified(self):
        from unittest.mock import patch
        from build_phase05_release import check_git
        with patch('build_phase05_release.subprocess.check_output', return_value=b''):
            with self.assertRaises(ValueError):
                check_git()

    def test_remote_digest_size_and_state_must_all_match(self):
        digest = 'a' * 64
        asset = {'digest': 'sha256:' + digest, 'size': 123, 'state': 'uploaded'}
        self.assertTrue(asset_matches(asset, digest, 123))
        self.assertFalse(asset_matches({**asset, 'size': 0}, digest, 123))
        self.assertFalse(asset_matches({**asset, 'state': 'starter'}, digest, 123))
        self.assertFalse(asset_matches({**asset, 'digest': 'sha256:' + 'b' * 64}, digest, 123))
    def test_all_experiment_sections_have_release_package(self):
        for name in ['data/annotations/index.html', 'features/scenes/case1.npz', 'models/verifiers/fold.npz',
                     'outputs/hypir/coeff_200/result/case1.png', 'outputs/selective/case1/predicted.png',
                     'analysis/artifact_audit.json', 'logs/audit.out.log', 'reports/phase05_report.md',
                     'README.md', 'scripts/extract_features.py', 'tests/test_phase05.py', 'checkpoints/index.json']:
            self.assertTrue(classify_stage_path(name).endswith('.zip'))
        with self.assertRaises(ValueError):
            classify_stage_path('dependencies/package.py')
        with self.assertRaises(ValueError):
            classify_stage_path('../secret')

    def test_sensitive_install_and_large_artifacts_not_in_git(self):
        for name in ['csig_phase05/features/features.npz', 'publication/phase05/dist/asset.zip', '.env', 'csig_phase05/dependencies/source.py']:
            self.assertFalse(git_allowed(name))
        for name in ['csig_phase05/scripts/core.py', 'csig_phase05/data/annotations/index.html',
                     'csig_phase05/data/annotations/images/review_001.png', 'publication/phase05/README.md']:
            self.assertTrue(git_allowed(name))

    def test_public_source_manifest_relocates_real_images_only(self):
        import json
        original = [{'lq_path': '../../csig_dataset/验证集/case1_lq.jpg', 'gt_path': '../../csig_dataset/验证集/case1_gt.jpg', 'candidate_path': 'outputs/hypir/a.png'},
                    {'lq_path': 'data/input/example.png', 'gt_path': 'data/sources/train/example.jpg'}]
        exported = json.loads(public_text('csig_phase05/data/manifests/pairs.json', json.dumps(original)))
        self.assertEqual(exported[0]['lq_path'], 'data/csig_dataset/验证集/case1_lq.jpg')
        self.assertEqual(exported[0]['gt_path'], 'data/csig_dataset/验证集/case1_gt.jpg')
        self.assertEqual(exported[1], original[1])
        self.assertEqual(original[0]['candidate_path'], exported[0]['candidate_path'])

    def test_only_authorized_named_external_sources_are_exported(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'case1_lq.jpg'
            self.assertEqual(source_name(path), 'csig_phase05/data/csig_dataset/验证集/case1_lq.jpg')
            with self.assertRaises(ValueError):
                source_name(Path(folder) / 'credentials.txt')


if __name__ == '__main__':
    unittest.main()
