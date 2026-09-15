import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / 'check_release.py'
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location('check_release', MODULE_PATH)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
from build_release import sanitize_text


class PublicReleaseTests(unittest.TestCase):
    def test_denies_runtime_and_large_git_artifacts(self):
        for name in ['csig_phase0/data/input/case1.jpg', 'csig_phase0/models/clip/pytorch_model.bin',
                     'csig_phase0/checkpoints/model.pt', 'csig_phase0/features/clip.npy',
                     '.env',
                     'csig_phase0/analysis/environment.json']:
            with self.subTest(name=name):
                self.assertFalse(release.allowed_path(name))

    def test_accepts_full_phase0_visualizations_and_final_checkpoint(self):
        for name in ['README.md', 'csig_phase0/scripts/phase0_core.py',
                     'csig_phase0/reports/patch_quality.csv',
                     'csig_phase0/reports/visualization/case1_gain_heatmap.png',
                     'csig_phase0/reports/visualization/failure_case1_01.png',
                     'csig_phase0/checkpoints/confidence_mlp.pt']:
            with self.subTest(name=name):
                self.assertTrue(release.allowed_path(name))

    def test_rejects_host_path_and_token_without_echoing_secret(self):
        token = 'ghp_' + 'a' * 36
        issues = release.content_issues('password=' + token)
        self.assertIn('credential-like value', issues)
        self.assertNotIn(token, ' '.join(issues))
        host_path = 'C:' + '\\Users\\' + 'example-user\\data'
        self.assertIn('host-specific absolute path', release.content_issues(host_path))

    def test_revision_hash_is_not_a_secret(self):
        self.assertEqual(release.content_issues('revision=' + 'a' * 40), [])

    def test_archive_paths_reject_escape(self):
        for name in ['../outside.txt', '/root/file', 'C:/user/file', 'data/../../outside.txt']:
            with self.subTest(name=name):
                self.assertFalse(release.safe_archive_path(name))
        self.assertTrue(release.safe_archive_path('csig_phase0/data/patches/case1/input.npy'))

    def test_freeze_build_machine_paths_become_installed_versions(self):
        source = 'packaging @ file:///' + 'C:' + '/Users/build/packaging'
        exported = sanitize_text(source)
        self.assertTrue(exported.startswith('packaging=='))
        self.assertEqual(release.content_issues(exported), [])


if __name__ == '__main__':
    unittest.main()
