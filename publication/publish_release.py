import hashlib
import json
import os
import subprocess
import time
import urllib.parse
from pathlib import Path

import requests

from check_release import ROOT, hash_file

OWNER = 'Jay1106-Zhu'
REPOSITORY = OWNER + '/csig-hallucination-aware-restoration'
API = 'https://api.github.com'


def authenticated_session():
    environment = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='never')
    result = subprocess.run(['git', 'credential', 'fill'],
                            input=f'protocol=https\nhost=github.com\nusername={OWNER}\n\n',
                            text=True, capture_output=True, env=environment, check=False)
    if result.returncode:
        raise RuntimeError('Git Credential Manager authentication unavailable')
    credential = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if not credential.get('password'):
        raise RuntimeError('GitHub credential missing')
    session = requests.Session()
    session.headers.update({'Authorization': 'Bearer ' + credential['password'], 'Accept': 'application/vnd.github+json',
                            'User-Agent': 'CSIG-Phase0-Publisher'})
    identity = session.get(API + '/user', timeout=30)
    identity.raise_for_status()
    if identity.json()['login'] != OWNER:
        raise RuntimeError('Refusing to publish under unexpected account')
    return session


def response_json(response):
    if not response.ok:
        raise RuntimeError(f'GitHub API request failed with status {response.status_code}')
    return response.json()


def upload_asset(session, upload_base, source):
    response_directory = ROOT / 'publication/upload_state'
    response_directory.mkdir(parents=True, exist_ok=True)
    response_path = response_directory / (source.name + '.json')
    url = upload_base + '?' + urllib.parse.urlencode({'name': source.name})
    configuration = ('header = "Authorization: ' + session.headers['Authorization'] + '"\n'
                     'header = "Content-Type: application/octet-stream"\n')
    command = ['curl.exe' if os.name == 'nt' else 'curl', '--config', '-', '--silent', '--show-error',
               '--fail-with-body', '--connect-timeout', '30', '--max-time', '1800', '--request', 'POST',
               '--data-binary', '@' + str(source), '--output', str(response_path),
               '--write-out', 'status=%{http_code} speed=%{speed_upload} bytes_per_second', url]
    completed = subprocess.run(command, input=configuration, capture_output=True, text=True, timeout=1810)
    if completed.returncode:
        raise RuntimeError(f'Native HTTPS upload failed ({completed.returncode}); inspect local upload state')
    print(completed.stdout, flush=True)
    return json.loads(response_path.read_text(encoding='utf-8'))


def main():
    manifest = json.loads((ROOT / 'publication/artifact_manifest.json').read_text(encoding='utf-8'))
    session = authenticated_session()
    repository = response_json(session.get(f'{API}/repos/{REPOSITORY}', timeout=30))
    if repository['private']:
        raise RuntimeError('Repository must already be public')
    release_body = (ROOT / 'publication/RELEASE_NOTES.md').read_text(encoding='utf-8')
    existing = response_json(session.get(f'{API}/repos/{REPOSITORY}/releases', timeout=30))
    release = next((entry for entry in existing if entry['tag_name'] == manifest['tag']), None)
    if release is None:
        release = response_json(session.post(f'{API}/repos/{REPOSITORY}/releases', timeout=30, json={
            'tag_name': manifest['tag'], 'target_commitish': 'main', 'name': 'Phase 0 — complete experiment artifacts',
            'body': release_body, 'draft': True, 'prerelease': False}))
    release_id = release['id']
    upload_base = release['upload_url'].split('{')[0]
    sources = [(ROOT / 'publication/dist' / entry['name'], entry['sha256']) for entry in manifest['packages']]
    sources.extend((ROOT / 'publication' / name, hash_file(ROOT / 'publication' / name))
                   for name in ['artifact_manifest.json', 'SHA256SUMS.txt'])
    verified = []
    for source, expected in sources:
        if hash_file(source) != expected:
            raise RuntimeError('Local asset changed after packaging')
        assets = response_json(session.get(f'{API}/repos/{REPOSITORY}/releases/{release_id}/assets', timeout=30))
        asset = next((entry for entry in assets if entry['name'] == source.name), None)
        if asset and (asset.get('digest') != 'sha256:' + expected or asset['size'] != source.stat().st_size):
            if release['draft']:
                deleted = session.delete(f'{API}/repos/{REPOSITORY}/releases/assets/{asset["id"]}', timeout=30)
                if deleted.status_code != 204:
                    raise RuntimeError('Could not replace incomplete draft upload')
                asset = None
            else:
                raise RuntimeError('Published asset does not match local manifest')
        if asset is None:
            print(f'Uploading {source.name} ({source.stat().st_size / 1024**2:.1f} MiB)', flush=True)
            asset = upload_asset(session, upload_base, source)
        if asset['state'] != 'uploaded' or asset['size'] != source.stat().st_size:
            raise RuntimeError('Remote upload is incomplete')
        if asset.get('digest') != 'sha256:' + expected:
            digest = hashlib.sha256()
            with session.get(f'{API}/repos/{REPOSITORY}/releases/assets/{asset["id"]}',
                             headers={'Accept': 'application/octet-stream'}, stream=True, timeout=(30, 1800)) as download:
                download.raise_for_status()
                for chunk in download.iter_content(8 * 1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise RuntimeError('Remote asset checksum verification failed')
        verified.append({'name': source.name, 'bytes': asset['size'], 'sha256': expected,
                         'asset_id': asset['id'], 'browser_download_url': asset['browser_download_url'],
                         'digest_verified': True})
        print('Verified', source.name, flush=True)
    release = response_json(session.patch(f'{API}/repos/{REPOSITORY}/releases/{release_id}',
                                           json={'draft': False, 'body': release_body}, timeout=30))
    public = requests.get(f'{API}/repos/{REPOSITORY}/releases/tags/{manifest["tag"]}', timeout=30)
    public_release = response_json(public)
    if public_release['draft'] or len(public_release['assets']) != len(sources):
        raise RuntimeError('Anonymous release verification failed')
    verification = {'repository': REPOSITORY, 'visibility': 'public', 'tag': manifest['tag'],
                    'release_url': release['html_url'], 'all_assets_verified': True, 'assets': verified,
                    'anonymous_release_access_verified': True, 'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (ROOT / 'publication/remote_verification.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
    print('Release published and publicly verified', flush=True)


if __name__ == '__main__':
    main()
