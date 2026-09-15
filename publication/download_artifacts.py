import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from check_release import ROOT, hash_file, safe_archive_path


def restore_archive(archive_path, package, destination, overwrite=False):
    destination = destination.resolve()
    members = {entry['path']: entry for entry in package['files']}
    preserved = []
    with zipfile.ZipFile(archive_path) as archive:
        if len(archive.infolist()) != len(members) or set(archive.namelist()) != set(members):
            raise ValueError('Archive members differ from manifest')
        for entry in archive.infolist():
            if not safe_archive_path(entry.filename):
                raise ValueError('Unsafe archive path')
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Archive target escapes destination')
            expected = members[entry.filename]
            digest = hashlib.sha256()
            with archive.open(entry) as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected['sha256'] or entry.file_size != expected['bytes']:
                raise ValueError('Archive member checksum mismatch')
            if target.exists():
                if hash_file(target) == expected['sha256']:
                    continue
                if not overwrite:
                    preserved.append(entry.filename)
                    continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    return preserved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--destination', type=Path, default=ROOT)
    arguments = parser.parse_args()
    manifest = json.loads((ROOT / 'publication/artifact_manifest.json').read_text(encoding='utf-8'))
    download_root = ROOT / 'publication/downloads'
    download_root.mkdir(parents=True, exist_ok=True)
    for package in manifest['packages']:
        name = package['name']
        if '/' in name or '\\' in name or not safe_archive_path(name):
            raise ValueError('Unsafe asset name')
        destination = download_root / name
        if not destination.exists() or hash_file(destination) != package['sha256']:
            url = f"https://github.com/{manifest['repository']}/releases/download/{manifest['tag']}/{name}"
            partial = destination.with_suffix('.zip.part')
            print('Downloading', name, flush=True)
            with urllib.request.urlopen(url, timeout=120) as source, partial.open('wb') as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            if partial.stat().st_size != package['bytes'] or hash_file(partial) != package['sha256']:
                raise ValueError('Downloaded asset checksum mismatch')
            partial.replace(destination)
        print('Verified', name, flush=True)
        if arguments.extract:
            preserved = restore_archive(destination, package, arguments.destination, arguments.overwrite)
            for name in preserved:
                print('Preserved existing differing file:', name)
    print('All Phase 0 assets verified', flush=True)


if __name__ == '__main__':
    main()
