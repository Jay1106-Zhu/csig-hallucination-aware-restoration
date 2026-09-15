import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PUBLICATION = Path(__file__).resolve().parent
ROOT = PUBLICATION.parents[1]
sys.path.insert(0, str(PUBLICATION.parent))
from check_release import hash_file, safe_archive_path
from download_artifacts import restore_archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--destination', type=Path, default=ROOT)
    parser.add_argument('--package', action='append')
    arguments = parser.parse_args()
    manifest = json.loads((PUBLICATION / 'artifact_manifest.json').read_text(encoding='utf-8'))
    destination = PUBLICATION / 'downloads'
    destination.mkdir(parents=True, exist_ok=True)
    valid_names = {package['name'] for package in manifest['packages']}
    if arguments.package and set(arguments.package) - valid_names:
        raise ValueError('Unknown requested package')
    for package in manifest['packages']:
        name = package['name']
        if arguments.package and name not in arguments.package:
            continue
        if not safe_archive_path(name) or '/' in name or '\\' in name:
            raise ValueError('Unsafe package name')
        target = destination / name
        if not target.exists() or hash_file(target) != package['sha256']:
            partial = target.with_suffix('.zip.part')
            url = f'https://github.com/{manifest["repository"]}/releases/download/{manifest["tag"]}/{name}'
            command = ['curl.exe' if os.name == 'nt' else 'curl', '--fail', '--location', '--retry', '3', '--connect-timeout', '30',
                       '--continue-at', '-', '--output', str(partial), url]
            subprocess.run(command, check=True)
            if partial.stat().st_size != package['bytes'] or hash_file(partial) != package['sha256']:
                raise ValueError('Download failed full SHA256 validation')
            partial.replace(target)
        print('Verified', name, flush=True)
        if arguments.extract:
            preserved = restore_archive(target, package, arguments.destination, arguments.overwrite)
            for path in preserved:
                print('Preserved existing differing file:', path)
    print('Requested Phase05 packages verified; extraction never overwrites differing files without --overwrite')


if __name__ == '__main__':
    main()
