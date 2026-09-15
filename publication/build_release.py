import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from check_release import ROOT, content_issues, hash_file, safe_archive_path

PHASE = ROOT / 'csig_phase0'
OUTPUT = ROOT / 'publication/dist'
TEXT_SUFFIXES = {'.json', '.csv', '.md', '.txt', '.log', '.py'}


def sanitize_text(text):
    text = re.sub(r'(?m)^([A-Za-z0-9_.-]+) @ file:[^\r\n]+',
                  lambda match: match[1] + '==' + importlib.metadata.version(match[1]), text)
    workspace = str(ROOT.parent)
    home = str(Path.home())
    for source, replacement in [(workspace, '${CSIG_WORKSPACE}'), (home, '${USER_HOME}')]:
        for spelling in [source.replace('\\', '\\\\'), source.replace('\\', '/'), source]:
            text = text.replace(spelling, replacement)
    return text


def sanitize_object(value):
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, dict):
        return {sanitize_object(key): sanitize_object(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_object(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_object(item) for item in value)
    return value


def package_sources():
    groups = {
        'phase0-data-patches.zip': ['data'],
        'phase0-frozen-models.zip': ['models'],
        'phase0-predictor-features.zip': ['checkpoints', 'features'],
        'phase0-hypir-outputs.zip': ['outputs'],
        'phase0-analysis-visualization.zip': ['analysis', 'reports', 'logs'],
    }
    result = {}
    for package, directories in groups.items():
        files = []
        for directory in directories:
            for path in sorted((PHASE / directory).rglob('*')):
                if path.is_file() and not path.name.startswith('release_') and not any(part in {'.cache', '__pycache__'} for part in path.relative_to(PHASE).parts):
                    files.append((path, path.relative_to(ROOT).as_posix()))
        result[package] = files
    manifest = json.loads((PHASE / 'data/manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['validation']:
        for source in ['input_path', 'gt_path']:
            path = Path(entry[source])
            result['phase0-data-patches.zip'].append((path, 'csig_phase0/data/csig_dataset/验证集/' + path.name))
    for name in ['requirements.txt', 'requirements_full_freeze.txt', 'phase0_done.md', 'README.md']:
        result['phase0-analysis-visualization.zip'].append((PHASE / name, 'csig_phase0/' + name))
    return result


def build_package(item):
    package, sources = item
    entries = []
    destination = OUTPUT / package
    seen = set()
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as archive:
        for source, name in sources:
            if not safe_archive_path(name) or name in seen:
                raise ValueError('Invalid or duplicate archive member')
            seen.add(name)
            transformed = False
            if source.name == 'confidence_mlp.pt':
                import torch

                payload = torch.load(source, map_location='cpu', weights_only=True)
                sanitized = sanitize_object(payload)
                buffer = io.BytesIO()
                torch.save(sanitized, buffer)
                content = buffer.getvalue()
                transformed = True
            elif source.suffix in TEXT_SUFFIXES:
                original = source.read_text(encoding='utf-8-sig', errors='strict')
                exported = sanitize_text(original)
                issues = content_issues(exported)
                if issues:
                    raise ValueError(f'Unsafe text in {name}: {issues}')
                content = exported.encode('utf-8')
                transformed = content != source.read_bytes()
            else:
                content = None
            if content is None:
                archive.write(source, arcname=name)
                digest = hash_file(source)
                size = source.stat().st_size
            else:
                import hashlib

                archive.writestr(name, content)
                digest = hashlib.sha256(content).hexdigest()
                size = len(content)
            entries.append({'path': name, 'bytes': size, 'sha256': digest, 'metadata_sanitized': transformed})
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('ZIP integrity verification failed')
    result = {'name': package, 'bytes': destination.stat().st_size, 'sha256': hash_file(destination),
              'file_count': len(entries), 'files': entries}
    print(f"Built {package}: {len(entries)} files, {result['bytes'] / 1024**2:.1f} MiB", flush=True)
    return result


def reuse_package(item):
    package, sources = item
    destination = OUTPUT / package
    expected = {name for _, name in sources}
    entries = []
    with zipfile.ZipFile(destination) as archive:
        if set(archive.namelist()) != expected:
            raise ValueError('Existing archive member list is incomplete')
        for member in archive.infolist():
            digest = hashlib.sha256()
            with archive.open(member) as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            entries.append({'path': member.filename, 'bytes': member.file_size, 'sha256': digest.hexdigest(),
                            'metadata_sanitized': member.filename.endswith(('manifest.json', 'confidence_mlp.pt'))})
    print('Verified existing', package, flush=True)
    return {'name': package, 'bytes': destination.stat().st_size, 'sha256': hash_file(destination),
            'file_count': len(entries), 'files': entries}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--reuse', action='store_true')
    arguments = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    def execute(item):
        if arguments.reuse and item[0] != 'phase0-analysis-visualization.zip' and (OUTPUT / item[0]).exists():
            return reuse_package(item)
        return build_package(item)
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        packages = list(executor.map(execute, package_sources().items()))
    release = {'tag': 'phase0-2026-09-15', 'repository': 'Jay1106-Zhu/csig-hallucination-aware-restoration',
               'packages': packages, 'excluded': ['dependency installation directories', 'download caches', 'Git authentication'],
               'sanitization': 'Host-specific paths replaced with ${CSIG_WORKSPACE}/${USER_HOME}; numeric model tensors unchanged.',
               'note': 'Original local artifacts are not modified. Source audit hashes refer to the original experiment; use this manifest for exported files.'}
    (ROOT / 'publication/artifact_manifest.json').write_text(json.dumps(release, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    checksums = ''.join(f"{package['sha256']}  {package['name']}\n" for package in packages)
    (ROOT / 'publication/SHA256SUMS.txt').write_text(checksums, encoding='utf-8')
    (ROOT / 'publication/requirements_full_freeze.txt').write_text(
        sanitize_text((PHASE / 'requirements_full_freeze.txt').read_text(encoding='utf-8-sig')), encoding='utf-8')
    print('Release manifest complete', flush=True)


if __name__ == '__main__':
    main()
