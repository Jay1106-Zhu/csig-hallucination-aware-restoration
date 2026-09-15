import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_FILES = {
    'artifact_audit.json', 'auc_sensitivity.csv', 'baselines.csv', 'confidence_metrics_per_seed.csv',
    'confidence_summary.json', 'decision.json', 'failure_annotations.csv', 'failure_cases.csv',
    'feature_extraction.json', 'folds.json', 'full_image_quality.csv', 'immutable_after.json',
    'oof_predictions.csv', 'patch_storage.json', 'seed_macro_summary.csv', 'spatial_statistics.csv',
    'supplementary_regions.csv', 'training_history.json',
}


def safe_archive_path(name):
    path = PurePosixPath(name.replace('\\', '/'))
    return not path.is_absolute() and '..' not in path.parts and ':' not in name and bool(path.parts)


def allowed_path(name):
    if not safe_archive_path(name):
        return False
    path = PurePosixPath(name)
    if any(part.startswith('.env') or part == '__pycache__' for part in path.parts):
        return False
    if len(path.parts) == 1:
        return path.suffix == '.md' or name in {'.gitignore', '.gitattributes'}
    if path.parts[0] == 'publication':
        return (len(path.parts) == 2 and path.suffix in {'.py', '.md', '.json', '.txt'}) or (
            len(path.parts) == 3 and path.parts[1] == 'tests' and path.suffix == '.py')
    if path.parts[0] != 'csig_phase0':
        return False
    if len(path.parts) == 2:
        return path.suffix == '.md' or path.name in {'requirements.txt', 'requirements_full_freeze.txt'}
    section = path.parts[1]
    if section in {'scripts', 'tests'}:
        return len(path.parts) == 3 and path.suffix == '.py'
    if section == 'configs':
        return len(path.parts) == 3 and path.suffix == '.json'
    if section == 'reports':
        return (len(path.parts) == 3 and path.suffix in {'.md', '.csv'}) or (
            len(path.parts) == 4 and path.parts[2] == 'visualization' and path.suffix in {'.png', '.jpg'})
    if section == 'analysis':
        return len(path.parts) == 3 and path.name in ANALYSIS_FILES
    return name == 'csig_phase0/checkpoints/confidence_mlp.pt'


def content_issues(text):
    issues = []
    credential_patterns = [r'gh[pousr]_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{35,}',
                           r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
                           r'(?i)(?:api[_-]?key|access[_-]?token|password)\s*[=:]\s*[\"\x27]?[A-Za-z0-9+/=_-]{24,}']
    if any(re.search(pattern, text) for pattern in credential_patterns):
        issues.append('credential-like value')
    if re.search(r'(?i)[A-Z]:[\\/]+(?:Users|MyProjects)[\\/]', text) or re.search(r'(?m)^[A-Za-z0-9_.-]+ @ file:/+', text):
        issues.append('host-specific absolute path')
    return issues


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    arguments = parser.parse_args()
    entries = subprocess.check_output(['git', 'ls-files', '--cached', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    failures, files = [], []
    for name in sorted(filter(None, entries)):
        if not allowed_path(name):
            failures.append({'path': name, 'issue': 'outside Git publication allowlist'})
        payload = subprocess.check_output(['git', 'show', ':' + name], cwd=ROOT)
        if len(payload) >= 90 * 1024 * 1024:
            failures.append({'path': name, 'issue': 'large artifact belongs in Release'})
        if Path(name).suffix not in {'.png', '.jpg', '.pt'}:
            for issue in content_issues(payload.decode('utf-8-sig')):
                failures.append({'path': name, 'issue': issue})
        if name != 'publication/git_manifest.json':
            files.append({'path': name, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()})
    report = {'status': 'passed' if not failures else 'failed', 'files': files, 'file_count': len(files),
              'total_bytes': sum(entry['bytes'] for entry in files), 'failures': failures,
              'scope': 'Exact staged Git blobs; Release artifacts verified separately.'}
    if arguments.write:
        (ROOT / 'publication/git_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'files'}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
