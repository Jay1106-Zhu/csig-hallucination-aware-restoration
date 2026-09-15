import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dependencies'))

import numpy as np
import pandas as pd
import torch
from PIL import Image

from audit_utils import validate_fold_checkpoint
from phase0_core import ConfidenceMLP, binary_metrics, grouped_folds, patch_quality, sha256
from prepare_phase0 import write_json
from run_phase0 import audit_sources, feature_variants, load_config, load_manifest


def main():
    torch.set_num_threads(4)
    config = load_config()
    table, variants = feature_variants()
    labels = (table.label == 'GOOD').to_numpy(dtype=np.int64)
    groups = table.image_id.to_numpy()
    if len(table) != 3565 or table.patch_id.duplicated().any() or set(groups) != {'case1', 'case2', 'case3', 'case4', 'case5'}:
        raise ValueError('Unexpected data identity or patch count')
    verified_patches = 0
    for entry in load_manifest():
        subset = table[table.image_id == entry['image_id']]
        original_images = {}
        for name in ['input', 'hypir', 'gt']:
            with Image.open(entry[name + '_path']) as image:
                original_images[name] = np.asarray(image).copy()
        arrays = {name: np.load(ROOT / 'data/patches' / entry['image_id'] / (name + '.npy'), mmap_mode='r') for name in original_images}
        for name, array in arrays.items():
            if array.shape != (713, 256, 256, 3) or array.dtype != np.uint8:
                raise ValueError('Invalid patch storage')
        for row in subset.itertuples(index=False):
            for name, array in arrays.items():
                if not np.array_equal(array[row.patch_index], original_images[name][row.y:row.y + 256, row.x:row.x + 256]):
                    raise ValueError('Stored patch differs from native source')
            quality = patch_quality(arrays['input'][row.patch_index], arrays['hypir'][row.patch_index], arrays['gt'][row.patch_index])
            if quality['label'] != row.label or not np.isclose(quality['gain'], row.gain, atol=1e-9):
                raise ValueError('Patch label recomputation failed')
            verified_patches += 1
    oof = pd.read_csv(ROOT / 'analysis/oof_predictions.csv')
    if oof.patch_id.tolist() != table.patch_id.tolist():
        raise ValueError('OOF ordering mismatch')
    fold_metrics = pd.read_csv(ROOT / 'reports/confidence_metrics.csv')
    reconstructed_folds = 0
    maximum_difference = 0.0
    for name, features in variants.items():
        for train, heldout, image_id in grouped_folds(groups):
            seed_probabilities = []
            for seed in config['seeds']:
                payload = torch.load(ROOT / 'checkpoints/folds' / name / f'{image_id}_seed{seed}.pt', map_location='cpu', weights_only=True)
                validate_fold_checkpoint(payload, features[train], groups[train], image_id)
                model = ConfidenceMLP(payload['input_dim'], payload['hidden_dim'])
                model.load_state_dict(payload['state_dict'])
                model.eval()
                with torch.inference_mode():
                    probability = model((torch.from_numpy(features[heldout]) - payload['mean']) / payload['scale']).numpy()
                recorded = oof[f'prob_{name}_seed{seed}'].to_numpy()[heldout]
                difference = float(np.max(np.abs(probability - recorded)))
                maximum_difference = max(maximum_difference, difference)
                if not np.allclose(probability, recorded, atol=2e-7, rtol=1e-5):
                    raise ValueError('Checkpoint OOF reconstruction mismatch')
                seed_probabilities.append(probability)
                reconstructed_folds += 1
            ensemble = np.mean(seed_probabilities, axis=0)
            if not np.allclose(ensemble, oof['prob_' + name].to_numpy()[heldout], atol=2e-7, rtol=1e-5):
                raise ValueError('Ensemble OOF mismatch')
            metrics = binary_metrics(labels[heldout], ensemble)
            stored = fold_metrics[(fold_metrics.feature == name) & (fold_metrics.heldout_image == image_id)].iloc[0]
            for metric, value in metrics.items():
                if value is None:
                    if not pd.isna(stored[metric]):
                        raise ValueError('Undefined metric should be NA')
                elif not np.isclose(value, stored[metric], atol=1e-9):
                    raise ValueError(f'Metric recomputation mismatch: {name}/{image_id}/{metric}')
    failures = pd.read_csv(ROOT / 'analysis/failure_cases.csv')
    annotations = pd.read_csv(ROOT / 'analysis/failure_annotations.csv')
    if len(failures) != 20 or len(annotations) != 20 or failures.patch_id.tolist() != annotations.patch_id.tolist():
        raise ValueError('Twenty inspected failure cases required')
    for kind in ['gain', 'confidence']:
        images = list((ROOT / 'reports/visualization').glob(f'case*_{kind}_heatmap.png'))
        if len(images) != 5:
            raise ValueError('Missing heatmap')
        for path in images:
            with Image.open(path) as image:
                if np.asarray(image).std() <= 1:
                    raise ValueError('Blank visualization')
    inference = json.loads((ROOT / 'analysis/hypir_inference.json').read_text(encoding='utf-8'))
    for path, digest in inference['outputs'].items():
        if sha256(path) != digest:
            raise ValueError('HYPIR output changed')
    models = json.loads((ROOT / 'analysis/feature_models.json').read_text(encoding='utf-8'))
    for model in models.values():
        for name, digest in model['files'].items():
            if sha256(Path(model['path']) / name) != digest:
                raise ValueError('Frozen model changed')
    audit_sources()
    artifact_hashes = {}
    for folder in ['configs', 'scripts', 'tests', 'reports', 'analysis', 'checkpoints', 'features']:
        for path in sorted((ROOT / folder).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.name in {'artifact_audit.json', 'artifact_hashes.json'}:
                continue
            artifact_hashes[str(path.relative_to(ROOT)).replace('\\', '/')] = sha256(path)
    write_json(ROOT / 'analysis/artifact_hashes.json', artifact_hashes)
    results = {'status': 'passed', 'independent_images': 5, 'verified_patch_triplets': verified_patches,
               'good_patches': int(labels.sum()), 'bad_patches': int((1 - labels).sum()),
               'reconstructed_fold_checkpoints': reconstructed_folds, 'max_oof_probability_error': maximum_difference,
               'train_only_scalers_verified': True, 'all_fold_metrics_recomputed': True,
               'failure_panels': 20, 'manual_failure_annotations': 20, 'gain_heatmaps': 5, 'oof_confidence_heatmaps': 5,
               'source_hashes_unchanged': True, 'frozen_feature_models_unchanged': True,
               'artifact_hash_count': len(artifact_hashes), 'test_images_used_for_training_or_evaluation': 0}
    write_json(ROOT / 'analysis/artifact_audit.json', results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
