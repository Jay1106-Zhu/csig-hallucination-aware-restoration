import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def patch_coordinates(width, height, patch_size=256, stride=128):
    if min(width, height) < patch_size or not 0 < stride <= patch_size:
        raise ValueError('Image too small or invalid stride')
    columns = list(range(0, width - patch_size + 1, stride))
    rows = list(range(0, height - patch_size + 1, stride))
    if columns[-1] != width - patch_size:
        columns.append(width - patch_size)
    if rows[-1] != height - patch_size:
        rows.append(height - patch_size)
    return [(column, row) for row in rows for column in columns]


def patch_quality(input_image, hypir_image, gt_image):
    if input_image.shape != hypir_image.shape or input_image.shape != gt_image.shape:
        raise ValueError('Image shapes differ; resizing is forbidden')
    input_mse = np.mean((input_image.astype(np.float64) - gt_image) ** 2)
    hypir_mse = np.mean((hypir_image.astype(np.float64) - gt_image) ** 2)
    input_psnr = math.inf if input_mse == 0 else 10 * math.log10(255 ** 2 / input_mse)
    hypir_psnr = math.inf if hypir_mse == 0 else 10 * math.log10(255 ** 2 / hypir_mse)
    gain = 0.0 if input_mse == hypir_mse else hypir_psnr - input_psnr
    return {'psnr_input': input_psnr, 'psnr_hypir': hypir_psnr,
            'gain': gain, 'label': 'GOOD' if gain > 0 else 'BAD'}


def grouped_folds(groups):
    groups = np.asarray(groups)
    if len(np.unique(groups)) < 2:
        raise ValueError('At least two independent images required')
    for image_id in np.unique(groups):
        yield np.flatnonzero(groups != image_id), np.flatnonzero(groups == image_id), str(image_id)


def fit_scaler(features):
    values = np.asarray(features, dtype=np.float64)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return {'mean': values.mean(axis=0).astype(np.float32), 'scale': scale.astype(np.float32)}


def transform_features(features, scaler):
    return ((np.asarray(features, dtype=np.float32) - scaler['mean']) / scaler['scale']).astype(np.float32)


def image_balanced_weights(groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return (weights / weights.mean()).astype(np.float32)


def _image_statistics(image):
    normalized = image.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float32) / 255.0))) ** 2
    vertical = np.fft.fftshift(np.fft.fftfreq(gray.shape[0]))[:, None]
    horizontal = np.fft.fftshift(np.fft.fftfreq(gray.shape[1]))[None, :]
    high_band = np.sqrt(vertical ** 2 + horizontal ** 2) >= 0.25
    high_energy = spectrum[high_band].sum() / max(float(spectrum.sum()), 1e-12)
    return np.array([*normalized.mean(axis=(0, 1)), *normalized.std(axis=(0, 1)),
                     float((cv2.Canny(gray, 100, 200) > 0).mean()),
                     float(cv2.Laplacian(gray, cv2.CV_32F).var()), high_energy,
                     float(gray.std()) / 255.0], dtype=np.float32)


def statistical_features(input_image, hypir_image):
    residual = (hypir_image.astype(np.float32) - input_image.astype(np.float32)) / 255.0
    return np.concatenate([_image_statistics(input_image), _image_statistics(hypir_image),
                           [np.abs(residual).mean(), residual.std(), (residual ** 2).mean(),
                            np.quantile(np.abs(residual), 0.95)]]).astype(np.float32)


def aggregate_heatmap(width, height, coordinates, values, patch_size=256):
    total = np.zeros((height, width), dtype=np.float32)
    count = np.zeros_like(total)
    for (column, row), value in zip(coordinates, values, strict=True):
        total[row:row + patch_size, column:column + patch_size] += value
        count[row:row + patch_size, column:column + patch_size] += 1
    if np.any(count == 0):
        raise ValueError('Uncovered pixels in heatmap')
    return total / count


def binary_metrics(labels, probabilities, threshold=0.5):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(probabilities) >= threshold
    both_classes = len(np.unique(labels)) == 2
    return {
        'accuracy': float(accuracy_score(labels, predictions)),
        'precision': float(precision_score(labels, predictions, zero_division=0)),
        'recall': float(recall_score(labels, predictions, zero_division=0)),
        'f1': float(f1_score(labels, predictions, zero_division=0)),
        'roc_auc': float(roc_auc_score(labels, probabilities)) if both_classes else None,
        'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)) if both_classes else None,
        'bad_recall': float(recall_score(1 - labels, ~predictions, zero_division=0)),
        'bad_f1': float(f1_score(1 - labels, ~predictions, zero_division=0)),
        'good_count': int(labels.sum()), 'bad_count': int((1 - labels).sum()),
    }


def validate_feature_index(expected_ids, stored_ids, features):
    if list(expected_ids) != list(stored_ids) or len(features) != len(expected_ids):
        raise ValueError('Feature rows do not match patch order')
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite features')


class ConfidenceMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def logits(self, features):
        return self.layers(features).squeeze(-1)

    def forward(self, features):
        return torch.sigmoid(self.logits(features))
