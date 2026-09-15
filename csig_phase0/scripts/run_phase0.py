import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dependencies'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np
import pandas as pd
import torch
from PIL import Image

from phase0_core import (
    ConfidenceMLP, binary_metrics, fit_scaler, grouped_folds, image_balanced_weights,
    patch_coordinates, patch_quality, sha256, statistical_features,
    transform_features, validate_feature_index,
)
from prepare_phase0 import write_json


def load_config():
    return json.loads((ROOT / 'configs/phase0.json').read_text(encoding='utf-8'))


def load_manifest():
    return json.loads((ROOT / 'data/manifest.json').read_text(encoding='utf-8'))['validation']


def extract_patches():
    config = load_config()
    size = config['patch_size']
    records, statistics, full_image_metrics = [], [], []
    inference = json.loads((ROOT / 'analysis/hypir_inference.json').read_text(encoding='utf-8'))
    if inference['status'] != 'completed':
        raise RuntimeError('Official HYPIR inference must complete first')
    for entry in load_manifest():
        started = time.perf_counter()
        images = {}
        for name in ['input', 'hypir', 'gt']:
            with Image.open(entry[name + '_path']) as image:
                if image.mode != 'RGB' or image.size != (entry['width'], entry['height']):
                    raise ValueError('Original-sized RGB images required')
                images[name] = np.asarray(image).copy()
        coordinates = patch_coordinates(entry['width'], entry['height'], size, config['stride'])
        destination = ROOT / 'data/patches' / entry['image_id']
        destination.mkdir(parents=True, exist_ok=True)
        arrays = {name: np.lib.format.open_memmap(destination / (name + '.npy'), mode='w+', dtype=np.uint8,
                                                shape=(len(coordinates), size, size, 3)) for name in images}
        full_image_metrics.append({'image_id': entry['image_id'], **patch_quality(images['input'], images['hypir'], images['gt'])})
        for index, (column, row) in enumerate(coordinates):
            patches = {name: image[row:row + size, column:column + size] for name, image in images.items()}
            for name, patch in patches.items():
                arrays[name][index] = patch
            patch_id = f"{entry['image_id']}_y{row:04d}_x{column:04d}"
            records.append({'image_id': entry['image_id'], 'patch_id': patch_id, 'x': column, 'y': row,
                            'patch_index': index, **patch_quality(patches['input'], patches['hypir'], patches['gt'])})
            statistics.append(statistical_features(patches['input'], patches['hypir']))
        for array in arrays.values():
            array.flush()
        print('Patches', entry['image_id'], len(coordinates), round(time.perf_counter() - started, 2), 'seconds', flush=True)
    table = pd.DataFrame(records)
    table.to_csv(ROOT / 'reports/patch_quality.csv', index=False)
    pd.DataFrame(full_image_metrics).to_csv(ROOT / 'analysis/full_image_quality.csv', index=False)
    np.save(ROOT / 'features/stats.npy', np.asarray(statistics, dtype=np.float32))
    table[['image_id', 'patch_id', 'x', 'y', 'patch_index']].to_csv(ROOT / 'features/index.csv', index=False)
    write_json(ROOT / 'analysis/patch_storage.json', {
        'format': 'lossless RGB uint8 NPY arrays, per-image per-source, mmap-readable',
        'path': 'data/patches/{image_id}/{input,hypir,gt}.npy', 'shape': ['patch_count', size, size, 3],
        'row_index': 'features/index.csv: patch_index', 'count': len(table),
        'stats_columns': [f'{source}_{stat}' for source in ['input', 'hypir'] for stat in
                          ['r_mean', 'g_mean', 'b_mean', 'r_std', 'g_std', 'b_std', 'edge_density', 'laplacian_variance',
                           'fft_high_energy', 'gray_std']] + ['residual_mae', 'residual_std', 'residual_mse', 'residual_abs_q95'],
    })


def extract_features():
    from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPVisionModelWithProjection

    config = load_config()
    torch.set_num_threads(4)
    torch.manual_seed(231)
    model_info = json.loads((ROOT / 'analysis/feature_models.json').read_text(encoding='utf-8'))
    timings = {}
    for name in ['clip', 'dino']:
        started = time.perf_counter()
        model_path = model_info[name]['path']
        if name == 'clip':
            processor = CLIPImageProcessor.from_pretrained(model_path, local_files_only=True)
            model = CLIPVisionModelWithProjection.from_pretrained(model_path, local_files_only=True)
        else:
            processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
            model = AutoModel.from_pretrained(model_path, local_files_only=True)
        model.eval().requires_grad_(False).to('cuda')
        torch.cuda.reset_peak_memory_stats()
        all_features = []
        with torch.inference_mode():
            for entry in load_manifest():
                paired = []
                for source in ['input', 'hypir']:
                    patches = np.load(ROOT / 'data/patches' / entry['image_id'] / (source + '.npy'), mmap_mode='r')
                    batches = []
                    for offset in range(0, len(patches), config['feature_batch_size']):
                        values = processor(images=[Image.fromarray(patch) for patch in patches[offset:offset + config['feature_batch_size']]],
                                           return_tensors='pt')['pixel_values'].to('cuda')
                        with torch.autocast('cuda', dtype=torch.float16):
                            output = model(pixel_values=values)
                        embedding = output.image_embeds if name == 'clip' else output.last_hidden_state[:, 0]
                        batches.append(torch.nn.functional.normalize(embedding.float(), dim=1).cpu().numpy())
                    paired.append(np.concatenate(batches))
                all_features.append(np.concatenate(paired, axis=1))
                print('Features', name, entry['image_id'], flush=True)
        features = np.concatenate(all_features)
        np.save(ROOT / 'features' / (name + '.npy'), features)
        timings[name] = {'seconds': time.perf_counter() - started, 'shape': list(features.shape),
                         'peak_gpu_bytes': torch.cuda.max_memory_allocated(), 'sources': ['input', 'hypir'],
                         'dtype': str(features.dtype), 'normalized_per_view': True, 'autocast': 'float16'}
        del model, output, values
        gc.collect()
        torch.cuda.empty_cache()
        write_json(ROOT / 'analysis/feature_extraction.json', timings)


def feature_variants():
    table = pd.read_csv(ROOT / 'reports/patch_quality.csv')
    index = pd.read_csv(ROOT / 'features/index.csv')
    arrays = {}
    for name in ['clip', 'dino', 'stats']:
        arrays[name] = np.load(ROOT / 'features' / (name + '.npy'))
        validate_feature_index(table.patch_id, index.patch_id, arrays[name])
    arrays['fusion'] = np.concatenate([arrays['clip'], arrays['dino'], arrays['stats']], axis=1)
    arrays['stats_lq'] = arrays['stats'][:, :10].copy()
    arrays['fusion_lq'] = np.concatenate([arrays['clip'][:, :arrays['clip'].shape[1] // 2],
                                          arrays['dino'][:, :arrays['dino'].shape[1] // 2], arrays['stats_lq']], axis=1)
    return table, arrays


def train_model(features, labels, groups, config, seed):
    torch.manual_seed(seed)
    scaler = fit_scaler(features)
    normalized = torch.from_numpy(transform_features(features, scaler))
    targets = torch.from_numpy(labels.astype(np.float32))
    group_weights = image_balanced_weights(groups)
    positive = float(np.sum(group_weights * labels))
    negative = float(np.sum(group_weights * (1 - labels)))
    class_weights = np.where(labels > 0, (positive + negative) / max(2 * positive, 1e-8),
                            (positive + negative) / max(2 * negative, 1e-8))
    weights = torch.from_numpy((group_weights * class_weights).astype(np.float32))
    model = ConfidenceMLP(features.shape[1], config['hidden_dim'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    losses = []
    model.train()
    for epoch in range(config['epochs']):
        permutation = torch.randperm(len(features))
        epoch_loss = 0.0
        for offset in range(0, len(features), config['batch_size']):
            selection = permutation[offset:offset + config['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            per_sample = torch.nn.functional.binary_cross_entropy_with_logits(model.logits(normalized[selection]), targets[selection], reduction='none')
            loss = (per_sample * weights[selection]).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(selection)
        losses.append(epoch_loss / len(features))
    model.eval()
    payload = {'state_dict': model.state_dict(), 'mean': torch.from_numpy(scaler['mean']), 'scale': torch.from_numpy(scaler['scale']),
               'seed': seed, 'input_dim': features.shape[1], 'hidden_dim': config['hidden_dim'], 'training_images': sorted(set(groups.tolist()))}
    return model, scaler, payload, losses


def bootstrap_image_auc(values, seed, draws):
    finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if len(finite) < 2:
        return [None, None]
    generator = np.random.default_rng(seed)
    means = generator.choice(finite, size=(draws, len(finite)), replace=True).mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def evaluate_predictors():
    config = load_config()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    table, variants = feature_variants()
    labels = (table.label == 'GOOD').to_numpy(dtype=np.int64)
    groups = table.image_id.to_numpy()
    fold_records, per_seed_records, histories, splits = [], [], {}, []
    oof_table = table[['image_id', 'patch_id', 'label', 'gain']].copy()
    summary = {}
    for name, features in variants.items():
        started = time.perf_counter()
        probabilities = np.full((len(config['seeds']), len(table)), np.nan, dtype=np.float32)
        for train, heldout, image_id in grouped_folds(groups):
            for seed_index, seed in enumerate(config['seeds']):
                model, scaler, payload, losses = train_model(features[train], labels[train], groups[train], config, seed)
                with torch.inference_mode():
                    prediction = model(torch.from_numpy(transform_features(features[heldout], scaler))).numpy()
                probabilities[seed_index, heldout] = prediction
                per_seed_records.append({'feature': name, 'heldout_image': image_id, 'seed': seed,
                                         **binary_metrics(labels[heldout], prediction, config['threshold'])})
                destination = ROOT / 'checkpoints/folds' / name
                destination.mkdir(parents=True, exist_ok=True)
                payload.update({'heldout_image': image_id, 'feature': name})
                torch.save(payload, destination / f'{image_id}_seed{seed}.pt')
                histories[f'{name}/{image_id}/{seed}'] = losses
                if name == 'fusion':
                    splits.append({'heldout_image': image_id, 'seed': seed, 'train_images': payload['training_images'],
                                   'train_patch_count': len(train), 'heldout_patch_count': len(heldout),
                                   'scaler_fit_on': payload['training_images'], 'early_stopping': False})
        if not np.isfinite(probabilities).all():
            raise ValueError('OOF predictions missing')
        ensemble = probabilities.mean(axis=0)
        oof_table['prob_' + name] = ensemble
        for seed_index, seed in enumerate(config['seeds']):
            oof_table[f'prob_{name}_seed{seed}'] = probabilities[seed_index]
        metrics_by_image = []
        for _, heldout, image_id in grouped_folds(groups):
            metrics = binary_metrics(labels[heldout], ensemble[heldout], config['threshold'])
            metrics_by_image.append(metrics)
            fold_records.append({'feature': name, 'heldout_image': image_id, **metrics})
        summary[name] = {'feature_dim': features.shape[1], 'independent_images': len(metrics_by_image),
                         'valid_auc_images': sum(entry['roc_auc'] is not None for entry in metrics_by_image),
                         'macro': {metric: float(np.mean([entry[metric] for entry in metrics_by_image if entry[metric] is not None]))
                                   if any(entry[metric] is not None for entry in metrics_by_image) else None
                                   for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'balanced_accuracy', 'bad_recall', 'bad_f1']},
                         'image_bootstrap_auc_95': bootstrap_image_auc([entry['roc_auc'] for entry in metrics_by_image], 231, config['bootstrap_draws']),
                         'elapsed_seconds': time.perf_counter() - started}
        print('Predictor', name, json.dumps(summary[name]['macro']), flush=True)
        write_json(ROOT / 'analysis/confidence_summary.json', summary)
    baseline_rows = []
    for train, heldout, image_id in grouped_folds(groups):
        prior = float(labels[train].mean())
        for name, constant in [('always_BAD', 0.0), ('training_prior', prior)]:
            baseline_rows.append({'baseline': name, 'heldout_image': image_id, 'training_prior': prior,
                                  **binary_metrics(labels[heldout], np.full(len(heldout), constant))})
    pd.DataFrame(baseline_rows).to_csv(ROOT / 'analysis/baselines.csv', index=False)
    pd.DataFrame(fold_records).to_csv(ROOT / 'reports/confidence_metrics.csv', index=False)
    pd.DataFrame(per_seed_records).to_csv(ROOT / 'analysis/confidence_metrics_per_seed.csv', index=False)
    oof_table.to_csv(ROOT / 'analysis/oof_predictions.csv', index=False)
    write_json(ROOT / 'analysis/folds.json', splits)
    write_json(ROOT / 'analysis/training_history.json', histories)
    final_models = []
    for seed in config['seeds']:
        _, _, payload, _ = train_model(variants['fusion'], labels, groups, config, seed)
        final_models.append(payload)
    torch.save({'format_version': 1, 'feature': 'fusion', 'feature_order': ['clip_input', 'clip_hypir', 'dino_input', 'dino_hypir', 'stats'],
                'models': final_models, 'threshold': config['threshold'], 'config': config,
                'feature_models': json.loads((ROOT / 'analysis/feature_models.json').read_text(encoding='utf-8')),
                'evaluation_warning': 'Fit on ALL five validation images; use saved fold checkpoints for OOF evaluation.'},
               ROOT / 'checkpoints/confidence_mlp.pt')
    payload = torch.load(ROOT / 'checkpoints/confidence_mlp.pt', map_location='cpu', weights_only=True)
    for saved in payload['models']:
        loaded = ConfidenceMLP(saved['input_dim'], saved['hidden_dim'])
        loaded.load_state_dict(saved['state_dict'])
        with torch.inference_mode():
            values = loaded((torch.from_numpy(variants['fusion'][:8]) - saved['mean']) / saved['scale'])
        if not torch.isfinite(values).all():
            raise ValueError('Reloaded checkpoint produced invalid values')


def audit_sources():
    original = json.loads((ROOT / 'analysis/immutable_before.json').read_text(encoding='utf-8'))
    modified = [path for path, digest in original.items() if sha256(path) != digest]
    write_json(ROOT / 'analysis/immutable_after.json', {'checked_files': len(original), 'modified_files': modified,
                                                       'all_unchanged': not modified})
    if modified:
        raise RuntimeError('Read-only sources were modified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['patches', 'features', 'train', 'audit', 'all'])
    arguments = parser.parse_args()
    steps = {'patches': extract_patches, 'features': extract_features, 'train': evaluate_predictors, 'audit': audit_sources}
    for stage in (steps if arguments.stage == 'all' else [arguments.stage]):
        started = time.perf_counter()
        steps[stage]()
        print('Completed stage', stage, round(time.perf_counter() - started, 2), 'seconds', flush=True)
