import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

PUBLICATION = Path(__file__).resolve().parent
ROOT = PUBLICATION.parents[1]
PHASE = ROOT / 'csig_phase05'
sys.path.insert(0, str(PUBLICATION.parent))
from build_release import sanitize_text
from check_release import allowed_path, content_issues, hash_file, safe_archive_path

TAG = 'phase05-2026-09-15'
REPOSITORY = 'Jay1106-Zhu/csig-hallucination-aware-restoration'
TEXT_SUFFIXES = {'.md', '.json', '.csv', '.txt', '.log', '.py', '.html'}


def classify_stage_path(name):
    if not safe_archive_path(name) or any(part in {'dependencies', '__pycache__', '.cache', '.git'} for part in PurePosixPath(name).parts):
        raise ValueError('Unsafe or excluded experiment path')
    first = PurePosixPath(name).parts[0]
    if first == 'data':
        return 'phase05-data-patches.zip'
    if first == 'features':
        return 'phase05-features.zip'
    if first in {'models', 'checkpoints'}:
        return 'phase05-verifiers.zip'
    if name.startswith('outputs/hypir/'):
        return 'phase05-hypir-outputs.zip'
    if name.startswith('outputs/selective/'):
        return 'phase05-selective-outputs.zip'
    return 'phase05-analysis-reports.zip'


def source_name(path):
    if path.name not in {f'case{number}_{kind}.jpg' for number in range(1, 6) for kind in ['lq', 'gt']}:
        raise ValueError('Unexpected external image source')
    return 'csig_phase05/data/csig_dataset/验证集/' + path.name


def public_text(name, text):
    exported = sanitize_text(text.replace('\r\n', '\n'))
    if name == 'csig_phase05/data/manifests/pairs.json':
        entries = json.loads(exported)
        for entry in entries:
            for key in ['lq_path', 'gt_path']:
                if entry[key].startswith('../../csig_dataset/'):
                    entry[key] = 'data/csig_dataset/验证集/' + PurePosixPath(entry[key]).name
        exported = json.dumps(entries, ensure_ascii=False, indent=2) + '\n'
    if name == 'csig_phase05/README.md':
        exported = ('> 发布状态更新：用户已于2026-09-15授权本轮全量公开。代码和轻量图表在Git；完整数据/特征/权重/候选/融合/日志见 `../publication/phase05/README.md` 与独立Release `phase05-2026-09-15`。下方“未发布”是计算验收时的历史快照，不影响当前交付状态。人工标注依然pending，结论仍为HOLD。\n\n' + exported)
    return exported


def git_allowed(name):
    if not safe_archive_path(name):
        return False
    path = PurePosixPath(name)
    if any(part.startswith('.env') or part in {'dependencies', '__pycache__', '.cache', 'dist', 'upload_state', 'downloads'} for part in path.parts):
        return False
    if name.startswith('publication/phase05/'):
        return (len(path.parts) == 3 and path.suffix in {'.py', '.md', '.json', '.txt'}) or (len(path.parts) == 4 and path.parts[2] == 'tests' and path.suffix == '.py')
    if name.startswith('csig_phase05/'):
        if path.suffix in {'.npz', '.npy', '.pt', '.pth', '.bin', '.safetensors', '.zip', '.log'}:
            return False
        if any(part in {'outputs', 'sources', 'input', 'patches', 'models'} for part in path.parts[1:]):
            return False
        if path.suffix in {'.png', '.jpg'}:
            return name.startswith(('csig_phase05/reports/visualization/', 'csig_phase05/data/annotations/images/'))
        return path.suffix in {'.py', '.md', '.json', '.csv', '.txt', '.html'}
    return allowed_path(name)


def source_packages():
    groups = {}
    expected = []
    for path in sorted(PHASE.rglob('*')):
        if not path.is_file() or any(part in {'dependencies', '__pycache__', '.cache', '.git'} for part in path.relative_to(PHASE).parts):
            continue
        name = path.relative_to(ROOT).as_posix()
        package = classify_stage_path(path.relative_to(PHASE).as_posix())
        groups.setdefault(package, []).append((path, name))
        expected.append(name)
    pairs = json.loads((PHASE / 'data/manifests/pairs.json').read_text(encoding='utf-8'))
    for entry in pairs:
        if entry['domain'] == 'real_csig':
            for key in ['lq_path', 'gt_path']:
                source = (PHASE / entry[key]).resolve()
                if hash_file(source) != entry[key.replace('_path', '_sha256')]:
                    raise ValueError('Real source integrity mismatch')
                groups['phase05-data-patches.zip'].append((source, source_name(source)))
    frozen = []
    for model in ['clip', 'dino']:
        directory = ROOT / 'csig_phase0/models' / model
        for path in sorted(directory.rglob('*')):
            if path.is_file() and '.cache' not in path.relative_to(directory).parts:
                frozen.append((path, path.relative_to(ROOT).as_posix()))
    groups['phase05-shared-encoders.zip'] = frozen
    sys.path.insert(0, str(PHASE / 'dependencies'))
    lpips_root = Path(importlib.util.find_spec('lpips').origin).parent
    metric_sources = [(Path.home() / '.cache/torch/hub/checkpoints' / filename, 'csig_phase05/models/perceptual/torch_hub/' + filename)
                      for filename in ['alexnet-owt-7be5be79.pth', 'vgg16-397923af.pth']]
    metric_sources += [(lpips_root / 'weights/v0.1/alex.pth', 'csig_phase05/models/perceptual/lpips_alex_v01.pth'),
                       (PHASE / 'dependencies/DISTS_pytorch/weights.pt', 'csig_phase05/models/perceptual/dists_weights.pt')]
    groups['phase05-perceptual-models.zip'] = metric_sources
    return groups, expected


def build_package(item):
    package_name, sources = item
    destination = PUBLICATION / 'dist' / package_name
    records, seen = [], set()
    with zipfile.ZipFile(destination, 'w', allowZip64=True) as archive:
        for source, name in sources:
            if not source.is_file() or not safe_archive_path(name) or name in seen:
                raise ValueError('Missing, unsafe or duplicate member: ' + name)
            seen.add(name)
            source_digest = hash_file(source)
            exported_content = None
            if source.suffix.lower() in TEXT_SUFFIXES:
                original = source.read_text(encoding='utf-8-sig')
                exported_content = public_text(name, original).encode('utf-8')
                issues = content_issues(exported_content.decode('utf-8'))
                if issues:
                    raise ValueError(f'Public content rejected: {name}: {issues}')
            if exported_content is not None:
                archive.writestr(name, exported_content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=3)
                digest, size = hashlib.sha256(exported_content).hexdigest(), len(exported_content)
            else:
                compression = zipfile.ZIP_STORED if source.suffix.lower() in {'.npz', '.png', '.jpg', '.jpeg'} else zipfile.ZIP_DEFLATED
                archive.write(source, name, compress_type=compression, compresslevel=3 if compression == zipfile.ZIP_DEFLATED else None)
                digest, size = source_digest, source.stat().st_size
            records.append({'path': name, 'bytes': size, 'sha256': digest, 'source_sha256': source_digest, 'metadata_sanitized': source_digest != digest})
    if destination.stat().st_size >= 2 * 1024**3:
        raise ValueError('Package exceeds 2 GiB safety bound')
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC verification failed')
    result = {'name': package_name, 'bytes': destination.stat().st_size, 'sha256': hash_file(destination), 'file_count': len(records), 'files': records}
    print(f'Built {package_name}: {len(records)} files, {result["bytes"] / 1024**2:.2f} MiB', flush=True)
    return result


def stage_public_git():
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    existing = {}
    for entry in subprocess.check_output(['git', 'ls-files', '-s', '-z'], cwd=ROOT).decode('utf-8').split('\0'):
        if entry:
            header, name = entry.split('\t', 1)
            existing[name] = header.split()[1]
    updates = []
    for name in sorted(set(filter(None, names))):
        if not git_allowed(name):
            continue
        if not name.startswith(('csig_phase05/', 'publication/phase05/')) and '/' in name:
            continue
        source = ROOT / name
        if not source.is_file():
            continue
        content = source.read_bytes()
        if source.suffix.lower() in TEXT_SUFFIXES:
            content = public_text(name, content.decode('utf-8-sig')).encode('utf-8')
            issues = content_issues(content.decode('utf-8'))
            if issues:
                raise ValueError(f'Unsafe Git export: {name}: {issues}')
        if len(content) >= 90 * 1024**2:
            raise ValueError('Large file unexpectedly selected for Git')
        expected_oid = hashlib.sha1(b'blob ' + str(len(content)).encode('ascii') + b'\0' + content).hexdigest()
        if existing.get(name) == expected_oid:
            continue
        object_id = subprocess.check_output(['git', 'hash-object', '-w', '--stdin'], input=content, cwd=ROOT).decode().strip()
        updates.append(f'100644 {object_id}\t{name}\0')
    subprocess.run(['git', 'update-index', '-z', '--index-info'], input=''.join(updates).encode('utf-8'), cwd=ROOT, check=True)
    print('Staged sanitized Git exports without rewriting local experiment files:', len(updates), flush=True)


def check_git():
    names = subprocess.check_output(['git', 'ls-files', '--cached', '-s', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    records, failures = [], []
    entries = {}
    for item in filter(None, names):
        header, name = item.split('\t', 1) if '\t' in item else (item, '')
        if name:
            entries[name] = header.split()[1]
    if not entries:
        raise ValueError('Refusing to pass an empty or undecodable Git index')
    blob_input = ''.join(oid + '\n' for oid in entries.values()).encode('ascii')
    blob_process = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    blob_output, _ = blob_process.communicate(blob_input)
    cursor = 0
    payloads = {}
    ordered_names = list(entries)
    for name in ordered_names:
        header_end = blob_output.index(b'\n', cursor)
        header = blob_output[cursor:header_end].split()
        cursor = header_end + 1
        size = int(header[2])
        payloads[name] = blob_output[cursor:cursor + size]
        cursor += size + 1
    for name in sorted(ordered_names):
        if not git_allowed(name):
            failures.append({'path': name, 'reason': 'Git allowlist rejection'})
        payload = payloads[name]
        if len(payload) >= 90 * 1024**2:
            failures.append({'path': name, 'reason': 'large Git blob'})
        if Path(name).suffix.lower() not in {'.png', '.jpg', '.pt'}:
            for issue in content_issues(payload.decode('utf-8-sig')):
                failures.append({'path': name, 'reason': issue})
        if name != 'publication/phase05/git_manifest.json':
            records.append({'path': name, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()})
    report = {'status': 'PASS' if not failures else 'FAIL', 'file_count': len(records), 'total_bytes': sum(row['bytes'] for row in records), 'failures': failures, 'files': records}
    (PUBLICATION / 'git_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'files'}, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


def build_release():
    (PUBLICATION / 'dist').mkdir(parents=True, exist_ok=True)
    sources, expected = source_packages()
    with ThreadPoolExecutor(max_workers=2) as executor:
        packages = list(executor.map(build_package, sources.items()))
    actual = [record['path'] for package in packages for record in package['files']]
    if len(actual) != len(set(actual)) or set(expected) - set(actual):
        raise ValueError('Incomplete or duplicate export mapping')
    manifest = {'tag': TAG, 'repository': REPOSITORY, 'packages': packages, 'scope': 'All Phase05 experiment files plus the ten real LQ/GT images and shared frozen encoder/metric weights',
                'excluded': ['dependency installation trees', 'bytecode/download caches except explicitly exported frozen evaluation weights', 'Git credentials and private upload state'],
                'sanitization': 'Host-specific paths replaced with ${CSIG_WORKSPACE}/${USER_HOME}; real paired image paths relocated inside the release. Original local files unchanged.',
                'hash_policy': 'Use these exported hashes for downloads. Embedded experiment inventory/audit hashes remain historical source hashes; source_sha256 preserves the mapping.',
                'human_labels': '110 pending, zero reviewed; this release does not fabricate human ground truth', 'scientific_decision': 'HOLD'}
    (PUBLICATION / 'artifact_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (PUBLICATION / 'SHA256SUMS.txt').write_text(''.join(row['sha256'] + '  ' + row['name'] + '\n' for row in packages), encoding='utf-8')
    (PUBLICATION / 'completeness_verification.json').write_text(json.dumps({'status': 'PASS', 'stage_file_count': len(expected), 'exported_file_count': len(actual), 'package_count': len(packages),
               'missing_stage_files': [], 'duplicate_export_members': [], 'original_phase05_untouched': True, 'phase0_release_replaced': False,
               'total_compressed_bytes': sum(row['bytes'] for row in packages)}, indent=2) + '\n', encoding='utf-8')
    print('Complete Phase05 packages and manifest ready', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage-git', action='store_true')
    parser.add_argument('--check-git', action='store_true')
    arguments = parser.parse_args()
    if arguments.stage_git:
        stage_public_git()
    elif arguments.check_git:
        check_git()
    else:
        build_release()
