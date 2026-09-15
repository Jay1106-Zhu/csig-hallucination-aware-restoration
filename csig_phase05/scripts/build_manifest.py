import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

from core import ROOT, REPOSITORY, WORKSPACE, config, generate_degradation, load_rgb, read_json, relative, sha256, write_json


def main():
    settings = config()
    for directory in ['data/manifests', 'data/annotations', 'data/patches', 'data/sources/bsds_train', 'data/input',
                      'features/dino_global', 'features/dino_tokens', 'features/clip', 'features/structure', 'features/stability',
                      'outputs/hypir', 'checkpoints', 'analysis', 'logs', 'reports/visualization', 'reports/tables']:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    immutable = {}
    for path in sorted((REPOSITORY / 'csig_phase0').rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            immutable[relative(path)] = sha256(path)
    for path in [WORKSPACE / 'HYPIR/test.py', WORKSPACE / 'HYPIR/HYPIR/enhancer/sd2.py', WORKSPACE / 'HYPIR/weights/HYPIR_sd2.pth']:
        immutable[relative(path)] = sha256(path)
    write_json(ROOT / 'analysis/immutable_before.json', immutable)
    pairs = []
    categories = ['text', 'text', 'animal', 'vegetation', 'architecture']
    old = read_json(REPOSITORY / 'csig_phase0/data/manifest.json')['validation']
    for index, entry in enumerate(old):
        input_path, target_path = Path(entry['input_path']), Path(entry['gt_path'])
        input_image, target_image = load_rgb(input_path), load_rgb(target_path)
        if input_image.shape != target_image.shape or sha256(input_path) != entry['input_sha256'] or sha256(target_path) != entry['gt_sha256']:
            raise ValueError('Phase 0 pairing/hash mismatch')
        blurred = [cv2.GaussianBlur(cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32), (512, 512)), (0, 0), 2) for image in [input_image, target_image]]
        shift, response = cv2.phaseCorrelate(*blurred)
        aligned = bool(np.linalg.norm(shift) < 5 and response > 0.03)
        if not aligned:
            raise ValueError(f'Alignment check requires review before continuing: {entry["image_id"]}, {shift}, {response}')
        stability = {str(coefficient): relative(WORKSPACE / f'baseline/experiments/coeff_t_{coefficient}/output/result' / (entry['image_id'] + '_lq.png')) for coefficient in settings['hypir_coefficients']}
        stability['200'] = relative(entry['hypir_path'])
        pairs.append({'image_id': entry['image_id'], 'scene_id': entry['image_id'], 'lq_path': relative(input_path), 'gt_path': relative(target_path),
                      'scene_category': categories[index], 'degradation_tags': ['real_unknown'], 'source': 'CSIG validation',
                      'domain': 'real_csig', 'pairing_verified': True, 'alignment_verified': aligned,
                      'alignment_method': 'same-sized original pair + low-frequency phase correlation; not semantic ground truth',
                      'alignment_shift_512': list(shift), 'alignment_response': response, 'split': 'validation',
                      'width': entry['width'], 'height': entry['height'], 'lq_sha256': entry['input_sha256'], 'gt_sha256': entry['gt_sha256'],
                      'candidate_path': relative(entry['hypir_path']), 'stability_paths': stability})
    repository, revision = settings['source_repository'], settings['source_revision']
    api_url = f'https://api.github.com/repos/{repository}/contents/BSDS500/data/images/train?ref={revision}'
    response = requests.get(api_url, timeout=45)
    response.raise_for_status()
    available = sorted(entry['name'] for entry in response.json() if entry['name'].endswith('.jpg'))
    generator = np.random.default_rng(settings['seed'])
    selected = sorted(generator.choice(available, settings['external_scenes'], replace=False).tolist())
    write_json(ROOT / 'data/manifests/source_selection.json', {'repository': repository, 'revision': revision, 'upstream_split': 'train',
                                                            'available_count': len(available), 'selected': selected, 'seed': settings['seed']})
    def download(name):
        destination = ROOT / 'data/sources/bsds_train' / name
        if not destination.exists():
            url = f'https://raw.githubusercontent.com/{repository}/{revision}/BSDS500/data/images/train/{name}'
            for attempt in range(3):
                try:
                    result = requests.get(url, timeout=60)
                    result.raise_for_status()
                    destination.write_bytes(result.content)
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise
        return destination
    with ThreadPoolExecutor(max_workers=6) as executor:
        sources = list(executor.map(download, selected))
    signatures = []
    for index, source in enumerate(sources):
        target = load_rgb(source)
        signature = cv2.resize(cv2.cvtColor(target, cv2.COLOR_RGB2GRAY), (32, 32)).astype(np.float32).ravel()
        signature -= signature.mean()
        signature /= max(np.linalg.norm(signature), 1e-8)
        signatures.append(signature)
        recipe = settings['degradation_recipes'][index % len(settings['degradation_recipes'])]
        image_id = 'bsds_' + source.stem
        input_path = ROOT / 'data/input' / (image_id + '.png')
        Image.fromarray(generate_degradation(target, recipe, settings['seed'] + index)).save(input_path)
        paths = {str(coefficient): f'outputs/hypir/coeff_{coefficient}/result/{image_id}.png' for coefficient in settings['hypir_coefficients']}
        pairs.append({'image_id': image_id, 'scene_id': image_id, 'lq_path': relative(input_path), 'gt_path': relative(source),
                      'scene_category': 'unreviewed_natural_scene', 'degradation_tags': [recipe], 'source': repository + '@' + revision + '/train/' + source.name,
                      'domain': 'synthetic_bsds', 'pairing_verified': True, 'alignment_verified': True,
                      'alignment_method': 'deterministic same-coordinate synthetic degradation; no geometry transform', 'split': 'validation',
                      'width': target.shape[1], 'height': target.shape[0], 'lq_sha256': sha256(input_path), 'gt_sha256': sha256(source),
                      'candidate_path': paths['200'], 'stability_paths': paths, 'degradation_seed': settings['seed'] + index})
    if len({pair['gt_sha256'] for pair in pairs}) != len(pairs):
        raise ValueError('Duplicate source images cannot count as independent scenes')
    similarities = np.stack(signatures) @ np.stack(signatures).T
    np.fill_diagonal(similarities, -1)
    near_duplicates = [(selected[first], selected[second], float(similarities[first, second]))
                       for first in range(len(selected)) for second in range(first + 1, len(selected)) if similarities[first, second] > 0.985]
    if near_duplicates:
        write_json(ROOT / 'analysis/near_duplicates.json', near_duplicates)
        raise ValueError('Near-duplicate scenes require grouped identity review')
    write_json(ROOT / 'data/manifests/pairs.json', pairs)
    write_json(ROOT / 'analysis/duplicate_audit.json', {'unique_gt_hashes': len(pairs), 'near_duplicate_threshold': 0.985,
                                                      'max_synthetic_thumbnail_correlation': float(similarities.max()), 'near_duplicates': []})
    font = ImageFont.load_default(size=18)
    for page in range(5):
        canvas = Image.new('RGB', (1200, 600), 'white')
        for slot, source in enumerate(sources[page * 10:(page + 1) * 10]):
            image = Image.open(source).convert('RGB')
            image.thumbnail((240, 260))
            column, row = (slot % 5) * 240, (slot // 5) * 300
            canvas.paste(image, (column, row + 25))
            ImageDraw.Draw(canvas).text((column + 3, row + 2), 'bsds_' + source.stem, fill='black', font=font)
        canvas.save(ROOT / f'reports/visualization/source_contact_{page + 1}.jpg', quality=90)
    import torch
    packages = {}
    for package in ['torch', 'torchvision', 'numpy', 'opencv-python', 'pandas', 'scikit-image', 'scikit-learn', 'lpips', 'DISTS-pytorch', 'transformers', 'scipy']:
        packages[package] = importlib.metadata.version(package)
    environment = {'python': sys.version, 'platform': platform.platform(), 'torch_cuda': torch.version.cuda,
                   'gpu': torch.cuda.get_device_name(0), 'git_commit': subprocess.check_output(['git', '-C', str(REPOSITORY), 'rev-parse', 'HEAD'], text=True).strip(),
                   'packages': packages, 'command': ['python', 'csig_phase05/scripts/build_manifest.py']}
    write_json(ROOT / 'analysis/environment.json', environment)
    (ROOT / 'requirements.txt').write_text('\n'.join(f'{name}=={version}' for name, version in packages.items()) + '\n', encoding='utf-8')
    print('Manifest ready: 55 scenes = 5 real CSIG + 50 synthetic BSDS train-derived pairs', flush=True)


if __name__ == '__main__':
    main()
