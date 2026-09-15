import csv
from pathlib import Path

from core import ROOT, partition_regions, weighted_scene_samples
import cv2
import numpy as np

METRICS = ['psnr', 'ssim', 'lpips', 'dists']


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError('Refusing to write an ambiguous empty CSV')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def pixel_metrics(image, reference):
    if image.shape != reference.shape:
        raise ValueError('Pixel metrics require identical native geometry')
    first = image.astype(np.float32)
    second = reference.astype(np.float32)
    mse = np.mean((first.astype(np.float64) - second) ** 2)
    psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-12))
    window = min(11, min(image.shape[:2]) // 2 * 2 - 1)
    if window < 3:
        raise ValueError('SSIM requires a region at least 3 pixels wide')
    first_mean = cv2.GaussianBlur(first, (window, window), 1.5)
    second_mean = cv2.GaussianBlur(second, (window, window), 1.5)
    first_var = cv2.GaussianBlur(first * first, (window, window), 1.5) - first_mean ** 2
    second_var = cv2.GaussianBlur(second * second, (window, window), 1.5) - second_mean ** 2
    covariance = cv2.GaussianBlur(first * second, (window, window), 1.5) - first_mean * second_mean
    constants = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    score = ((2 * first_mean * second_mean + constants[0]) * (2 * covariance + constants[1])) / (
        (first_mean ** 2 + second_mean ** 2 + constants[0]) * (first_var + second_var + constants[1]))
    border = window // 2
    return {'psnr': float(psnr), 'ssim': float(score[border:-border, border:-border].mean())}


class PerceptualMetrics:
    def __init__(self):
        import torch
        import lpips
        from DISTS_pytorch import DISTS

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lpips = lpips.LPIPS(net='alex', verbose=False).eval().to(self.device)
        self.dists = DISTS(load_weights=False)
        weights = torch.load(ROOT / 'dependencies/DISTS_pytorch/weights.pt', map_location='cpu', weights_only=True)
        with torch.no_grad():
            self.dists.alpha.copy_(weights['alpha'])
            self.dists.beta.copy_(weights['beta'])
        self.dists.eval().to(self.device)
        for model in [self.lpips, self.dists]:
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    def tensor(self, image, full=False):
        import torch

        height, width = image.shape[:2]
        ratio = min(1., 1024 / max(height, width)) if full else 1.
        ratio = max(ratio, 64 / min(height, width))
        if ratio != 1:
            image = cv2.resize(image, (round(width * ratio), round(height * ratio)), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC)
        return torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255

    def compare(self, first, second, full=False, only_lpips=False):
        import torch

        with torch.inference_mode():
            first_tensor, second_tensor = self.tensor(first, full), self.tensor(second, full)
            result = {'lpips': float(self.lpips(first_tensor * 2 - 1, second_tensor * 2 - 1).item())}
            if not only_lpips:
                result['dists'] = float(self.dists(first_tensor, second_tensor).item())
        return result


def structure_features(first, second):
    gray_images = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255 for image in [first, second]]
    magnitudes, angles, laplacians, edges, highpass = [], [], [], [], []
    for gray in gray_images:
        horizontal = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        vertical = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        magnitude, angle = cv2.cartToPolar(horizontal, vertical)
        magnitudes.append(magnitude)
        angles.append(angle)
        laplacians.append(cv2.Laplacian(gray, cv2.CV_32F))
        edges.append(cv2.Canny(np.rint(gray * 255).astype(np.uint8), 100, 200) > 0)
        highpass.append(gray - cv2.GaussianBlur(gray, (0, 0), 1.0))
    values = []
    for first_map, second_map in [(magnitudes[0], magnitudes[1]), (laplacians[0], laplacians[1]), (highpass[0], highpass[1])]:
        residual = np.abs(first_map - second_map)
        values.extend([first_map.mean(), second_map.mean(), first_map.std(), second_map.std(), residual.mean(), residual.std(), np.quantile(residual, .9)])
    orientation_difference = np.abs(np.angle(np.exp(1j * (angles[0] - angles[1]))))
    valid = np.maximum(magnitudes[0], magnitudes[1]) > .1
    values.extend([orientation_difference[valid].mean() if valid.any() else 0., valid.mean()])
    histograms = [np.histogram(angle, bins=8, range=(0, 2 * np.pi), weights=magnitude)[0] for angle, magnitude in zip(angles, magnitudes)]
    histograms = [histogram / max(histogram.sum(), 1e-8) for histogram in histograms]
    values.extend(np.abs(histograms[0] - histograms[1]))
    union = np.logical_or(*edges).sum()
    values.extend([edges[0].mean(), edges[1].mean(), np.logical_xor(*edges).mean(), np.logical_and(*edges).sum() / max(union, 1)])
    for source, target in [(edges[0], edges[1]), (edges[1], edges[0])]:
        distances = cv2.distanceTransform((~target).astype(np.uint8), cv2.DIST_L2, 3)
        bound = float(np.hypot(*source.shape))
        values.append(float(np.clip(distances[source], 0, bound).mean()) / bound if source.any() else 0.)
    return np.asarray(values, dtype=np.float32)


def fit_normalizer(features, groups):
    weights = weighted_scene_samples(groups)
    mean = np.average(features.astype(np.float64), axis=0, weights=weights)
    scale = np.sqrt(np.average((features - mean) ** 2, axis=0, weights=weights))
    scale[scale < 1e-6] = 1
    return mean.astype(np.float32), scale.astype(np.float32)


def binary_metrics(labels, probabilities, threshold=.5):
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = probabilities >= threshold
    both_classes = len(np.unique(labels)) == 2
    calibration = 0.
    bins = np.minimum((probabilities * 10).astype(int), 9)
    for index in range(10):
        selected = bins == index
        if selected.any():
            calibration += selected.mean() * abs(labels[selected].mean() - probabilities[selected].mean())
    return {'roc_auc': float(roc_auc_score(labels, probabilities)) if both_classes else None,
            'pr_auc': float(average_precision_score(labels, probabilities)) if both_classes else None,
            'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)) if both_classes else None,
            'precision': float(precision_score(labels, predictions, zero_division=0)),
            'recall': float(recall_score(labels, predictions, zero_division=0)),
            'f1': float(f1_score(labels, predictions, zero_division=0)),
            'accuracy': float((predictions == labels).mean()), 'brier': float(np.mean((probabilities - labels) ** 2)),
            'ece': float(calibration), 'positive_count': int(labels.sum()), 'region_count': len(labels)}


def fuse_regions(source, restored, regions, selected):
    mask = np.zeros(source.shape[:2], dtype=bool)
    for index in selected:
        region = regions[index]
        mask[region['y']:region['y'] + region['height'], region['x']:region['x'] + region['width']] = True
    fused = source.copy()
    fused[mask] = restored[mask]
    return fused, mask
