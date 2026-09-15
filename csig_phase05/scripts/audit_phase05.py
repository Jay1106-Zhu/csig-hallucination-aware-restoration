import json
from collections import defaultdict

import cv2
import numpy as np

from core import ROOT, config, generate_degradation, load_rgb, metric_gains, partition_regions, read_json, resolve, sha256, write_json
from evaluation import METRICS, PerceptualMetrics, binary_metrics, fit_normalizer, pixel_metrics, read_csv
from extract_features import feature_variants
from train_verifiers import predict_checkpoint


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def assert_index_order(first, second):
    require([(row['image_id'], str(row['region_index'])) for row in first] == [(row['image_id'], str(row['region_index'])) for row in second], 'Index order mismatch')


def main():
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    cfg = config()
    entries = read_json(ROOT / 'data/manifests/pairs.json')
    require(len(entries) == 55 and len({entry['gt_sha256'] for entry in entries}) == 55, 'Duplicate or missing source images')
    before = read_json(ROOT / 'analysis/immutable_before.json')
    after = {path: sha256(resolve(path)) for path in before}
    require(before == after, 'Phase0 or original restoration code/weights modified')
    write_json(ROOT / 'analysis/immutable_after.json', after)
    metrics = read_csv(ROOT / 'analysis/restoration_gain.csv')
    regions = read_csv(ROOT / 'analysis/region_index.csv')
    feature_index = read_csv(ROOT / 'features/index.csv')
    assert_index_order(metrics, regions)
    assert_index_order(regions, feature_index)
    combined = dict(np.load(ROOT / 'features/features.npz'))
    require(all(len(values) == len(regions) and np.isfinite(values).all() for values in combined.values()), 'Bad feature tensor')
    metadata = read_json(ROOT / 'analysis/feature_extraction.json')
    require(metadata['features_sha256'] == sha256(ROOT / 'features/features.npz'), 'Feature hash mismatch')
    require(metadata['index_sha256'] == sha256(ROOT / 'features/index.csv'), 'Feature index hash mismatch')
    by_image = defaultdict(list)
    for position, row in enumerate(regions):
        by_image[row['image_id']].append(position)
    perceptual = PerceptualMetrics()
    checked_regions, checked_candidates, perceptual_samples = 0, 0, 0
    maximum_psnr_error, maximum_perceptual_error = 0., 0.
    for scene_number, entry in enumerate(entries):
        require(entry['split'] != 'test', 'Test data included')
        require(entry['pairing_verified'] and entry['alignment_verified'], 'Unverified pairing')
        lq, gt, restored = [load_rgb(resolve(entry[key])) for key in ['lq_path', 'gt_path', 'candidate_path']]
        require(lq.shape == gt.shape == restored.shape, 'Native geometry mismatch')
        require(sha256(resolve(entry['lq_path'])) == entry['lq_sha256'], 'LQ hash mismatch')
        require(sha256(resolve(entry['gt_path'])) == entry['gt_sha256'], 'GT hash mismatch')
        if entry['domain'] == 'synthetic_bsds':
            require('/train/' in entry['source'], 'External source is not train split')
            require(np.array_equal(generate_degradation(gt, entry['degradation_tags'][0], entry['degradation_seed']), lq), 'Synthetic degradation not reproducible')
        for coefficient, path in entry['stability_paths'].items():
            require(sha256(resolve(path)) == entry['stability_sha256'][coefficient], 'Candidate hash mismatch')
            require(load_rgb(resolve(path)).shape == lq.shape, 'Condition geometry mismatch')
            checked_candidates += 1
        archive = np.load(ROOT / regions[by_image[entry['image_id']][0]]['archive_path'])
        archive_arrays = {name: archive[name] for name in ['lq', 'gt', 'restored']}
        expected = partition_regions(entry['width'], entry['height'], cfg['analysis_patch_size'])
        require(json.loads(str(archive['regions_json'])) == expected, 'Archive geometry mismatch')
        require(sum(region['width'] * region['height'] for region in expected) == entry['width'] * entry['height'], 'Uncovered or overlapping area')
        for local_index, row_index in enumerate(by_image[entry['image_id']]):
            region = expected[local_index]
            for key in ['x', 'y', 'width', 'height']:
                require(region[key] == int(regions[row_index][key]), 'Region coordinate mismatch')
            slices = np.s_[region['y']:region['y'] + region['height'], region['x']:region['x'] + region['width']]
            patches = [image[slices] for image in [lq, gt, restored]]
            for name, patch in zip(['lq', 'gt', 'restored'], patches):
                require(np.array_equal(archive_arrays[name][local_index, :region['height'], :region['width']], patch), 'Lossless patch mismatch')
            input_quality, restored_quality = pixel_metrics(patches[0], patches[1]), pixel_metrics(patches[2], patches[1])
            for metric in ['psnr', 'ssim']:
                error = abs(float(metrics[row_index][metric + '_gain']) - (restored_quality[metric] - input_quality[metric]))
                maximum_psnr_error = max(maximum_psnr_error, error)
                require(error < 1e-5, 'Pixel gain mismatch')
            require(int(metrics[row_index]['psnr_benefit']) == int(restored_quality['psnr'] > input_quality['psnr']), 'PSNR label sign mismatch')
            require(int(metrics[row_index]['four_metric_majority']) == int(sum(float(metrics[row_index][metric + '_gain']) > 0 for metric in METRICS) >= 3), 'Majority target mismatch')
            for metric in ['lpips', 'dists']:
                require(abs(float(metrics[row_index][metric + '_gain']) - (float(metrics[row_index][metric + '_lq']) - float(metrics[row_index][metric + '_restored']))) < 1e-7, 'Perceptual gain sign mismatch')
            if local_index == 0:
                for image, suffix in [(patches[0], 'lq'), (patches[2], 'restored')]:
                    measured = perceptual.compare(image, patches[1])
                    for metric in ['lpips', 'dists']:
                        error = abs(measured[metric] - float(metrics[row_index][metric + '_' + suffix]))
                        maximum_perceptual_error = max(maximum_perceptual_error, error)
                        require(error < 1e-5, 'Perceptual metric reproduction mismatch')
                perceptual_samples += 1
            checked_regions += 1
        feature_scene = ROOT / 'features/scenes' / (entry['image_id'] + '.npz')
        require(sha256(feature_scene) == metadata['scene_hashes'][entry['image_id']], 'Scene feature hash mismatch')
        scene_features = np.load(feature_scene)
        for key in combined:
            require(np.array_equal(scene_features[key], combined[key][by_image[entry['image_id']]]), 'Combined feature ordering mismatch')
        require(scene_features['raw_tokens256'].shape == (len(expected), 5, 256, 384), 'Missing spatial condition tokens')
        require(scene_features['raw_tokens512'].shape == (len(expected), 2, 256, 384), 'Missing multi-scale tokens')
        require(np.isfinite(scene_features['raw_tokens256']).all(), 'Nonfinite spatial tokens')
        print('Audit data', scene_number + 1, '/55', entry['image_id'], flush=True)
    variants = feature_variants(combined)
    groups = np.array([row['scene_id'] for row in feature_index])
    predictions = read_csv(ROOT / 'analysis/oof_predictions.csv')
    probability_lookup = {(row['model_id'], int(row['row_index'])): float(row['probability']) for row in predictions}
    require(len(probability_lookup) == len(predictions), 'Duplicated OOF prediction')
    maximum_probability_error = 0.
    for record in read_json(ROOT / 'analysis/folds.json'):
        require(sha256(ROOT / record['checkpoint']) == record['sha256'], 'Checkpoint hash mismatch')
        checkpoint = dict(np.load(ROOT / record['checkpoint']))
        train, validation = checkpoint['train_indices'], checkpoint['validation_indices']
        require(not set(groups[train]) & set(groups[validation]), 'Scene leakage')
        require(sorted(set(groups[train])) == record['train_scenes'] and sorted(set(groups[validation])) == record['validation_scenes'], 'Fold manifest mismatch')
        example = next(row for row in predictions if row['model_id'] == record['model_id'])
        features = variants[example['variant']]
        mean, scale = fit_normalizer(features[train], groups[train])
        require(np.array_equal(mean, checkpoint['mean']) and np.array_equal(scale, checkpoint['scale']), 'Train-only scaler mismatch')
        probabilities = predict_checkpoint(features[validation], checkpoint)
        expected = np.array([probability_lookup[(record['model_id'], int(index))] for index in validation])
        error = float(np.max(np.abs(probabilities - expected)))
        maximum_probability_error = max(maximum_probability_error, error)
        require(error < 1e-6, 'OOF reconstruction failed')
    by_model_image = defaultdict(list)
    for row in predictions:
        require(int(row['label']) == int(metrics[int(row['row_index'])][row['target']]), 'OOF label differs from independent quality target')
        by_model_image[(row['model_id'], row['image_id'])].append(row)
    per_image_verified = 0
    for row in read_csv(ROOT / 'analysis/verifier_per_image.csv'):
        if row['scope'] == 'baseline':
            continue
        values = by_model_image[(row['model_id'], row['image_id'])]
        measured = binary_metrics([int(value['label']) for value in values], [float(value['probability']) for value in values])
        for name, value in measured.items():
            require(row[name] == '' if value is None else abs(float(row[name]) - value) < 1e-7, 'Per-image evaluation mismatch')
        per_image_verified += 1
    require(np.all([row['hallucination_rate'] == '' for row in read_csv(ROOT / 'analysis/risk_coverage.csv')]), 'Fabricated human hallucination rates')
    primary = read_json(ROOT / 'analysis/training.json')['primary_model']
    primary_scores = defaultdict(list)
    for row in predictions:
        if row['model_id'] == primary:
            primary_scores[row['image_id']].append(row)
    selective_records = read_json(ROOT / 'analysis/selective_artifacts.json')
    for scene_number, entry in enumerate(entries):
        lq, restored, gt = [load_rgb(resolve(entry[key])) for key in ['lq_path', 'candidate_path', 'gt_path']]
        regions_for_image = partition_regions(entry['width'], entry['height'], cfg['analysis_patch_size'])
        for record in [row for row in selective_records if row['image_id'] == entry['image_id']]:
            require(sha256(ROOT / record['image']) == record['image_sha256'], 'Selective output hash mismatch')
            require(sha256(ROOT / record['mask']) == record['mask_sha256'], 'Selective mask hash mismatch')
            from PIL import Image
            mask = np.asarray(Image.open(ROOT / record['mask'])) > 0
            output = load_rgb(ROOT / record['image'])
            require(np.array_equal(output[mask], restored[mask]) and np.array_equal(output[~mask], lq[~mask]), 'Fusion pixels not selected from declared sources')
            if record['mode'] == 'predicted':
                scores = np.array([float(row['probability']) for row in sorted(primary_scores[entry['image_id']], key=lambda row: int(row['region_index']))])
                expected = np.argsort(-scores, kind='stable')[:len(record['selected_regions'])].tolist()
                require(expected == record['selected_regions'], 'Predicted selection not OOF score-only')
            if record['mode'] == 'oracle_pixel':
                output_sse = ((output.astype(np.float64) - gt) ** 2).sum(axis=-1)
                best_sse = np.minimum(((lq.astype(np.float64) - gt) ** 2).sum(axis=-1), ((restored.astype(np.float64) - gt) ** 2).sum(axis=-1))
                require(np.array_equal(output_sse, best_sse), 'Pixel oracle upper bound violated')
    require(all(row['severity'] == '' and row['annotation_status'] == 'pending' for row in read_csv(ROOT / 'data/annotations/hallucination_labels.csv')), 'Raw blind labels are not pending')
    from annotations import choose_regions
    annotation_rows = read_csv(ROOT / 'data/annotations/hallucination_labels.csv')
    require(len(annotation_rows) == 110, 'Incorrect blind review count')
    for scene_number, entry in enumerate(entries):
        selected = [row for row in annotation_rows if row['image_id'] == entry['image_id']]
        expected = choose_regions(len(by_image[entry['image_id']]), cfg['seed'] + scene_number, 2)
        require([int(row['region_index']) for row in selected] == expected, 'Blind subset depends on scores or wrong seed')
        for row in selected:
            require(sha256(ROOT / 'data/annotations' / row['image_path']) == row['image_sha256'], 'Blind review image changed')
    audit = {'status': 'PASS', 'scope': 'computational artifacts; independent human review remains pending', 'source_images': len(entries), 'real_images': 5, 'synthetic_images': 50,
             'native_regions_verified': checked_regions, 'candidates_verified': checked_candidates, 'perceptual_recomputed_regions': perceptual_samples,
             'max_pixel_gain_reproduction_error': maximum_psnr_error, 'max_perceptual_reproduction_error': maximum_perceptual_error,
             'verifier_checkpoints_verified': len(read_json(ROOT / 'analysis/folds.json')), 'max_oof_probability_error': maximum_probability_error,
             'per_image_metrics_recomputed': per_image_verified, 'blind_sample_hashes_verified': len(annotation_rows),
             'selective_images_verified': len(selective_records), 'immutable_files_verified': len(before), 'phase0_unchanged': True,
             'test_data_used': 0, 'restoration_models_trained': 0, 'human_annotations_fabricated': 0,
             'limitations': ['source_id is a scene proxy, not independently adjudicated scene identity', 'real alignment audit is low-frequency translation screening only',
                             'human-dependent results unavailable', 'oracle budget is fixed region count, not fixed pixel count', 'finite sample bootstrap is descriptive with few informative images']}
    write_json(ROOT / 'analysis/artifact_audit.json', audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
