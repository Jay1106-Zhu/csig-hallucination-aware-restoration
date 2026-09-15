import hashlib
import io
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
WORKSPACE = Path(os.environ.get('CSIG_WORKSPACE', REPOSITORY.parent)).resolve()
sys.path.insert(0, str(ROOT / 'dependencies'))
sys.path.insert(0, str(REPOSITORY / 'csig_phase0/dependencies'))


def config():
    return json.loads((ROOT / 'configs/phase05.json').read_text(encoding='utf-8'))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path):
    return Path(os.path.relpath(Path(path).resolve(), ROOT)).as_posix()


def resolve(path):
    return (ROOT / path).resolve()


def load_rgb(path):
    with Image.open(path) as image:
        if image.mode != 'RGB':
            raise ValueError('RGB images required')
        return np.asarray(image).copy()


def partition_regions(width, height, size=256):
    if min(width, height, size) <= 0:
        raise ValueError('Positive dimensions required')
    return [{'x': column, 'y': row, 'width': min(size, width - column), 'height': min(size, height - row)}
            for row in range(0, height, size) for column in range(0, width, size)]


def context_crop(image, center_x, center_y, size):
    left = int(round(center_x - size / 2))
    top = int(round(center_y - size / 2))
    def reflect_indices(start, length):
        if length == 1:
            return np.zeros(size, dtype=int)
        period = 2 * (length - 1)
        indices = np.arange(start, start + size) % period
        return np.minimum(indices, period - indices)
    return image[reflect_indices(top, image.shape[0])[:, None], reflect_indices(left, image.shape[1])[None, :]].copy()


def generate_degradation(image, recipe, seed):
    generator = np.random.default_rng(seed)
    height, width = image.shape[:2]
    values = image.astype(np.float32)
    if recipe in {'blur', 'mixed'}:
        values = cv2.GaussianBlur(values, (0, 0), 2.0 if recipe == 'blur' else 1.2)
    if recipe in {'low_resolution', 'mixed'}:
        small = cv2.resize(values, (max(1, width // 4), max(1, height // 4)), interpolation=cv2.INTER_AREA)
        values = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    if recipe in {'noise_jpeg', 'mixed'}:
        values += generator.normal(0, 15 if recipe == 'noise_jpeg' else 6, values.shape)
    if recipe == 'lowlight_glow':
        values = (values / 255.0) ** 1.5 * 150
        columns, rows = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
        glow = np.exp(-((columns - 0.55) ** 2 + (rows + 0.5) ** 2) / 0.18) * 55
        values += glow[:, :, None] * np.array([1., 0.8, 0.5])
        values += generator.normal(0, 4, values.shape)
    output = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    if recipe in {'noise_jpeg', 'mixed'}:
        buffer = io.BytesIO()
        Image.fromarray(output).save(buffer, format='JPEG', quality=55)
        output = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert('RGB')).copy()
    if recipe not in {'blur', 'low_resolution', 'noise_jpeg', 'lowlight_glow', 'mixed'}:
        raise ValueError('Unknown degradation recipe')
    return output


def metric_gains(input_metrics, candidate_metrics):
    return {name + '_gain': (candidate_metrics[name] - input_metrics[name]) * (-1 if name in {'lpips', 'dists'} else 1)
            for name in ['psnr', 'ssim', 'lpips', 'dists']}


def grouped_splits(groups, folds=5):
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

    values = np.asarray(groups)
    splitter = GroupKFold(n_splits=folds) if len(np.unique(values)) >= 30 else LeaveOneGroupOut()
    yield from splitter.split(np.zeros(len(values)), groups=values)


def weighted_scene_samples(groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse]
    return weights / weights.mean()


def pairwise_global(first, second):
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    return np.concatenate([first, second, np.abs(first - second), first * second,
                           [np.dot(first, second), np.linalg.norm(first - second)]]).astype(np.float32)


def token_correspondence(first, second, grid=16):
    first = first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-8)
    second = second / np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-8)
    similarities = first @ second.T
    diagonal = np.diag(similarities).clip(-1, 1)
    residual = np.linalg.norm(first - second, axis=1)
    nearest = similarities.argmax(axis=1)
    positions = np.stack(np.meshgrid(np.arange(grid), np.arange(grid), indexing='ij'), axis=-1).reshape(-1, 2)
    displacement = np.linalg.norm(positions - positions[nearest], axis=1)
    local_mask = np.max(np.abs(positions[:, None] - positions[None, :]), axis=-1) <= 1
    local_match = np.where(local_mask, similarities, -2).max(axis=1)
    disagreement = 1 - diagonal
    summary = [disagreement.mean(), disagreement.std(), np.quantile(disagreement, 0.5), np.quantile(disagreement, 0.9),
               displacement.mean(), displacement.std(), np.quantile(displacement, 0.9), (displacement > 1).mean(),
               residual.mean(), residual.std(), local_match.mean(), local_match.std(), similarities.max(axis=1).mean(),
               np.sort(disagreement)[-max(1, len(disagreement) // 10):].mean(), disagreement.max(), (nearest == np.arange(len(first))).mean()]
    return {'summary': np.asarray(summary, dtype=np.float32), 'disagreement': disagreement.reshape(grid, grid).astype(np.float32),
            'nearest': nearest.reshape(grid, grid).astype(np.int16), 'local_confidence': local_match.reshape(grid, grid).astype(np.float32),
            'displacement': displacement.reshape(grid, grid).astype(np.float32)}


def validate_annotation(row):
    if row.get('reviewer_type') != 'human' or row.get('annotation_status') != 'reviewed':
        return False
    severity = row.get('severity')
    if severity in {'', None}:
        return False
    if int(severity) not in {0, 1, 2, 3} or float(severity) != int(severity):
        raise ValueError('Severity must be an integer from 0 to 3')
    return True


def risk_coverage(scores, gains, coverages):
    ordering = np.argsort(-np.asarray(scores), kind='stable')
    gains = np.asarray(gains)
    records = []
    for coverage in coverages:
        count = max(1, int(math.ceil(len(scores) * coverage)))
        selection = ordering[:count]
        records.append({'coverage': coverage, 'actual_patch_coverage': count / len(scores), 'selected_indices': selection.tolist(),
                        'damage_rate': float((gains[selection] <= 0).mean()), 'mean_gain': float(gains[selection].mean())})
    return records
