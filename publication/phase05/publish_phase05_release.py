import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

PUBLICATION = Path(__file__).resolve().parent
sys.path.insert(0, str(PUBLICATION.parent))
from publish_release import API, REPOSITORY, authenticated_session, response_json
from check_release import hash_file


def asset_matches(asset, digest, size):
    return asset.get('digest') == 'sha256:' + digest and asset.get('size') == size and asset.get('state') == 'uploaded'


def native_upload(authorization, upload_base, source):
    output_directory = PUBLICATION / 'upload_state'
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / (source.name + '.json')
    configuration = ('header = "Authorization: ' + authorization + '"\n' + 'header = "Content-Type: application/octet-stream"\n')
    url = upload_base + '?' + urllib.parse.urlencode({'name': source.name})
    print('Uploading', source.name, source.stat().st_size, 'bytes', flush=True)
    command = ['curl.exe' if os.name == 'nt' else 'curl', '--config', '-', '--silent', '--show-error', '--fail-with-body',
               '--connect-timeout', '30', '--max-time', '7200', '--request', 'POST', '--data-binary', '@' + str(source),
               '--output', str(output), '--write-out', 'status=%{http_code} speed=%{speed_upload}', url]
    result = subprocess.run(command, input=configuration, text=True, capture_output=True, timeout=7210)
    if result.returncode:
        raise RuntimeError(f'Native upload failed for {source.name}; code={result.returncode}. Resume this script after checking draft state.')
    print('Uploaded', source.name, result.stdout, flush=True)
    return json.loads(output.read_text(encoding='utf-8'))


def public_verification(manifest, sources):
    repository = response_json(requests.get(f'{API}/repos/{REPOSITORY}', timeout=60))
    if repository['private']:
        raise RuntimeError('Anonymous repository is not public')
    public_release = None
    for attempt in range(5):
        response = requests.get(f'{API}/repos/{REPOSITORY}/releases/tags/{manifest["tag"]}', params={'verification': int(time.time())}, timeout=60)
        if response.ok:
            public_release = response.json()
            break
        time.sleep(3 * (attempt + 1))
    if public_release is None or public_release['draft']:
        raise RuntimeError('Anonymous release not available')
    assets = {entry['name']: entry for entry in public_release['assets']}
    if set(assets) != {source.name for source, digest in sources}:
        raise RuntimeError('Public asset set mismatch')
    verified = []
    for source, expected in sources:
        asset = assets[source.name]
        if not asset_matches(asset, expected, source.stat().st_size):
            raise RuntimeError('Remote digest/state/size mismatch: ' + source.name)
        headers = {'Range': 'bytes=0-65535'} if source.suffix == '.zip' else {}
        with requests.get(asset['browser_download_url'], headers=headers, stream=True, timeout=(60, 180)) as download:
            download.raise_for_status()
            if source.suffix == '.zip':
                sample = next(download.iter_content(65536))
                with source.open('rb') as stream:
                    if sample != stream.read(len(sample)):
                        raise RuntimeError('Anonymous asset download prefix mismatch')
                prefix_bytes = len(sample)
            else:
                digest = hashlib.sha256()
                for chunk in download.iter_content(1024 * 1024):
                    digest.update(chunk)
                if digest.hexdigest() != expected:
                    raise RuntimeError('Anonymous manifest download hash mismatch')
                prefix_bytes = source.stat().st_size
        verified.append({'name': source.name, 'bytes': asset['size'], 'sha256': expected, 'asset_id': asset['id'],
                         'browser_download_url': asset['browser_download_url'], 'server_full_sha256_verified': True,
                         'anonymous_download_verified_bytes': prefix_bytes, 'verification_method': 'GitHub full-asset SHA256 and size, plus independent anonymous byte download'})
        print('Publicly verified', source.name, flush=True)
    old = json.loads((PUBLICATION.parent / 'remote_verification.json').read_text(encoding='utf-8'))
    old_release = response_json(requests.get(f'{API}/repos/{REPOSITORY}/releases/tags/{old["tag"]}', timeout=60))
    old_assets = {entry['name']: entry for entry in old_release['assets']}
    for entry in old['assets']:
        if entry['name'] not in old_assets or not asset_matches(old_assets[entry['name']], entry['sha256'], entry['bytes']):
            raise RuntimeError('Old Phase0 release integrity changed')
    report = {'repository': REPOSITORY, 'visibility': 'public', 'tag': manifest['tag'], 'release_id': public_release['id'], 'release_url': public_release['html_url'],
              'all_assets_verified': True, 'anonymous_release_access_verified': True, 'old_phase0_release_unchanged': True, 'assets': verified,
              'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (PUBLICATION / 'remote_verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('Phase05 release publicly verified; Phase0 assets unchanged', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--verify-only', action='store_true')
    arguments = parser.parse_args()
    manifest = json.loads((PUBLICATION / 'artifact_manifest.json').read_text(encoding='utf-8'))
    sources = [(PUBLICATION / 'dist' / entry['name'], entry['sha256']) for entry in manifest['packages']]
    sources.extend((PUBLICATION / name, hash_file(PUBLICATION / name)) for name in ['artifact_manifest.json', 'SHA256SUMS.txt'])
    if arguments.verify_only:
        public_verification(manifest, sources)
        return
    session = authenticated_session()
    repository = response_json(session.get(f'{API}/repos/{REPOSITORY}', timeout=60))
    if repository['private']:
        raise RuntimeError('Expected public repository')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PUBLICATION.parents[1]).decode().strip()
    branch = response_json(session.get(f'{API}/repos/{REPOSITORY}/branches/main', timeout=60))
    if branch['commit']['sha'] != commit:
        raise RuntimeError('Push and verify main before publishing release')
    releases = response_json(session.get(f'{API}/repos/{REPOSITORY}/releases', timeout=60))
    release = next((entry for entry in releases if entry['tag_name'] == manifest['tag']), None)
    notes = (PUBLICATION / 'RELEASE_NOTES.md').read_text(encoding='utf-8')
    if release is None:
        release = response_json(session.post(f'{API}/repos/{REPOSITORY}/releases', timeout=60, json={
            'tag_name': manifest['tag'], 'target_commitish': commit, 'name': 'Phase 0.5 — complete artifacts (HOLD; human labels pending)',
            'body': notes, 'draft': True, 'prerelease': False}))
    assets = response_json(session.get(f'{API}/repos/{REPOSITORY}/releases/{release["id"]}/assets', timeout=60))
    pending = []
    for source, expected in sources:
        if hash_file(source) != expected:
            raise RuntimeError('Local package changed after manifest: ' + source.name)
        asset = next((entry for entry in assets if entry['name'] == source.name), None)
        if asset and not asset_matches(asset, expected, source.stat().st_size):
            if not release['draft']:
                raise RuntimeError('Refusing to overwrite a published mismatched asset')
            deleted = session.delete(f'{API}/repos/{REPOSITORY}/releases/assets/{asset["id"]}', timeout=60)
            if deleted.status_code != 204:
                raise RuntimeError('Could not clean incomplete current draft upload')
            asset = None
        if asset is None:
            pending.append((source, expected))
        else:
            print('Already verified', source.name, flush=True)
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        futures = {executor.submit(native_upload, session.headers['Authorization'], release['upload_url'].split('{')[0], source): (source, expected)
                   for source, expected in sorted(pending, key=lambda item: item[0].stat().st_size)}
        for future in as_completed(futures):
            source, expected = futures[future]
            asset = future.result()
            if not asset_matches(asset, expected, source.stat().st_size):
                raise RuntimeError('Uploaded asset failed full digest verification: ' + source.name)
            print('Verified uploaded SHA256', source.name, flush=True)
    response_json(session.patch(f'{API}/repos/{REPOSITORY}/releases/{release["id"]}', json={'draft': False, 'body': notes}, timeout=60))
    public_verification(manifest, sources)


if __name__ == '__main__':
    main()
