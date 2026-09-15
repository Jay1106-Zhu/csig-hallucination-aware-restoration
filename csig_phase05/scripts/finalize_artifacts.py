import importlib.util
from pathlib import Path

import numpy as np

from core import ROOT, read_json, sha256, write_json
from evaluation import read_csv, write_csv


def main():
    features = dict(np.load(ROOT / 'features/features.npz'))
    groups = {'dino_global': ['dino_lq', 'dino_restored', 'pairwise'], 'clip': ['clip'], 'structure': ['structure', 'stats'], 'stability': ['stability']}
    for group, blocks in groups.items():
        directory = ROOT / 'features' / group
        directory.mkdir(parents=True, exist_ok=True)
        for block in blocks:
            path = directory / f'{block}.npy'
            np.save(path, features[block])
        write_json(directory / 'index.json', {'row_index': '../index.csv', 'blocks': {block: {'file': block + '.npy', 'shape': list(features[block].shape), 'sha256': sha256(directory / f'{block}.npy')} for block in blocks}})
    token_directory = ROOT / 'features/dino_tokens'
    token_directory.mkdir(parents=True, exist_ok=True)
    write_json(token_directory / 'index.json', {'row_index': '../index.csv', 'scenes': [{
        'image_id': entry['image_id'], 'container': f'../scenes/{entry["image_id"]}.npz',
        'keys': ['raw_tokens256', 'raw_tokens512', 'raw_cls256', 'raw_cls512', 'disagreement_256', 'disagreement_512', 'nearest_256', 'nearest_512', 'local_confidence_256', 'local_confidence_512', 'displacement_256', 'displacement_512'],
        'sha256': sha256(ROOT / 'features/scenes' / f'{entry["image_id"]}.npz')} for entry in read_json(ROOT / 'data/manifests/pairs.json')],
        'storage': 'one container per scene; no duplicated large token tensors', 'token_grid': [16, 16], 'token_dimension': 384,
        'views256': ['LQ', '50', '100', '150', '200'], 'views512': ['LQ', '200']})
    checkpoint_directory = ROOT / 'checkpoints'
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    write_json(checkpoint_directory / 'index.json', {'root': '..', 'checkpoints': read_json(ROOT / 'analysis/folds.json'), 'storage': 'models/verifiers is the single canonical weight directory; only frozen-feature logistic and tiny MLP verifier weights'})
    table_directory = ROOT / 'reports/tables'
    table_directory.mkdir(parents=True, exist_ok=True)
    write_json(table_directory / 'index.json', {'tables': [{'path': '../../analysis/' + path.name, 'sha256': sha256(path), 'bytes': path.stat().st_size} for path in sorted((ROOT / 'analysis').glob('*.csv'))]})
    schema = {'feature_row_order': 'features/index.csv', 'global_dimension': 384, 'clip_dimension_per_view': 512,
              'pairwise_order': ['LQ384', 'restored384', 'absolute_difference384', 'elementwise_product384', 'cosine1', 'L2_distance1'],
              'token_summary16': ['same_position_disagreement_mean', 'same_position_disagreement_std', 'same_position_disagreement_median', 'same_position_disagreement_q90',
                                  'nearest_token_displacement_mean', 'nearest_token_displacement_std', 'nearest_token_displacement_q90', 'nearest_token_displacement_gt1_fraction',
                                  'normalized_residual_mean', 'normalized_residual_std', 'local_match_cosine_mean', 'local_match_cosine_std', 'global_nearest_cosine_mean',
                                  'top10pct_disagreement_mean', 'max_disagreement', 'nearest_is_same_position_fraction'],
              'structure_order37': ['gradient7', 'laplacian7', 'highpass7', 'orientation_difference_mean', 'orientation_valid_fraction', 'abs_orientation_histogram_difference8',
                                     'LQ_edge_fraction', 'restored_edge_fraction', 'edge_xor_fraction', 'edge_iou', 'LQ_to_restored_boundary_distance', 'restored_to_LQ_boundary_distance'],
              'structure_seven_stats': ['LQ_mean', 'restored_mean', 'LQ_std', 'restored_std', 'absolute_residual_mean', 'absolute_residual_std', 'absolute_residual_q90'],
              'stability12': ['pixel_variance_mean', 'pixel_variance_std', 'pixel_variance_q90', 'edge_variance_mean', 'LPIPS_to_main_mean', 'LPIPS_to_main_std', 'LPIPS_to_main_max',
                              'normalized_CLS_variance_mean', 'normalized_CLS_variance_max', 'normalized_token_variance_sum_mean', 'normalized_token_variance_sum_std', 'normalized_token_variance_sum_q90'],
              'stats24': 'unchanged Phase0 statistical_features: per-view RGB means3/std3, edge fraction, laplacian variance, highband FFT fraction, gray std; residual abs mean/std/MSE/q95',
              'no_gt_no_image_id_no_position_features': True}
    write_json(ROOT / 'features/feature_schema.json', schema)
    metric_weights = []
    hub = Path.home() / '.cache/torch/hub/checkpoints'
    for filename in ['alexnet-owt-7be5be79.pth', 'vgg16-397923af.pth']:
        path = hub / filename
        if not path.exists():
            raise FileNotFoundError('Required cached metric backbone ' + filename)
        metric_weights.append({'source': 'torchvision pretrained checkpoint', 'name': filename, 'sha256': sha256(path)})
    package = Path(importlib.util.find_spec('lpips').origin).parent
    for name, path in [('LPIPS alex calibrated v0.1', package / 'weights/v0.1/alex.pth'), ('DISTS alpha beta 0.1', ROOT / 'dependencies/DISTS_pytorch/weights.pt')]:
        metric_weights.append({'source': name, 'name': path.name, 'sha256': sha256(path)})
    write_json(ROOT / 'analysis/perceptual_models.json', {'weights': metric_weights, 'frozen': True, 'DISTS_loading': 'load_weights=False then copy official alpha/beta from installed package; no third-party source modification',
               'LPIPS_input': 'RGB float[-1,1]', 'DISTS_input': 'RGB float[0,1]', 'region_geometry': 'native valid pixels; min side64 fallback, observed min65',
               'full_image_geometry': 'long side <=1024 for perceptual only, area resize; native PSNR/SSIM', 'SSIM': 'RGB Gaussian11 sigma1.5, valid center average',
               'psnr_zero_mse': 'MSE floored at1e-12 to keep finite JSON values; no identical LQ/candidate-vs-GT region observed'})
    training = read_json(ROOT / 'analysis/training.json')
    write_json(ROOT / 'analysis/decision.json', {'decision': 'HOLD', 'computational_status': 'complete and independently audited', 'human_review_status': '0 reviewed of 110 pending',
               'phase1_authorized': False, 'github_published': False, 'reasons': ['insufficient real scenes and positive regions', 'candidate-aware primary weaker than LQ-only',
               'no significant fixed-coverage damage reduction vs random', 'predicted actual fusion below LQ', 'independent human truth missing']})
    (ROOT / 'logs/verifier_run_summary.log').write_text('Structured summary reconstructed from training.json and folds.json; not the original stdout stream.\n' +
            f"runs={training['run_count']} checkpoints={training['checkpoint_count']} seconds={training['seconds']}\n" +
            '\n'.join(sorted({row['model_id'] for row in read_json(ROOT / 'analysis/folds.json')})) + '\n', encoding='utf-8')
    records = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or any(part in {'dependencies', '__pycache__', '.git'} for part in path.relative_to(ROOT).parts):
            continue
        if path.name in {'artifact_inventory.json', 'SHA256SUMS.txt'}:
            continue
        records.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha256(path)})
    write_json(ROOT / 'analysis/artifact_inventory.json', {'scope': 'all local Phase05 artifacts except installed dependencies, pycache and this inventory', 'files': records, 'file_count': len(records), 'total_bytes': sum(row['bytes'] for row in records)})
    (ROOT / 'analysis/SHA256SUMS.txt').write_text('\n'.join(row['sha256'] + '  ' + row['path'] for row in records) + '\n', encoding='utf-8')
    print(f'Inventoried {len(records)} files, {sum(row["bytes"] for row in records) / 1024**3:.3f} GiB')


if __name__ == '__main__':
    main()
