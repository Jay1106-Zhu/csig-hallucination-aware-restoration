import math
from collections import defaultdict

import numpy as np
from PIL import Image

from core import ROOT, config, load_rgb, metric_gains, partition_regions, read_json, resolve, risk_coverage, sha256, write_json
from evaluation import METRICS, PerceptualMetrics, fuse_regions, pixel_metrics, read_csv, write_csv


def bootstrap_difference(first, second, seed, draws):
    differences = np.asarray(first) - np.asarray(second)
    generator = np.random.default_rng(seed)
    samples = generator.choice(differences, (draws, len(differences)), replace=True).mean(axis=1)
    return {'image_count': len(differences), 'mean_difference': float(differences.mean()), 'lower_95': float(np.quantile(samples, .025)),
            'upper_95': float(np.quantile(samples, .975)), 'draws': draws, 'unit': 'paired source image', 'seed': seed}


def budget_oracle_selection(sse_improvements, count):
    return np.argsort(-np.asarray(sse_improvements), kind='stable')[:count]


def pixel_oracle(lq, restored, gt):
    input_error = ((lq.astype(np.float32) - gt) ** 2).sum(axis=-1)
    restored_error = ((restored.astype(np.float32) - gt) ** 2).sum(axis=-1)
    mask = restored_error < input_error
    output = lq.copy()
    output[mask] = restored[mask]
    return output, mask


def risk_tables(cfg, metric_rows):
    by_image = defaultdict(list)
    areas_by_image = defaultdict(list)
    for row in read_csv(ROOT / 'analysis/region_index.csv'):
        areas_by_image[row['image_id']].append(int(row['width']) * int(row['height']))
    for row in metric_rows:
        by_image[row['image_id']].append(row)
    predictions = defaultdict(list)
    for row in read_csv(ROOT / 'analysis/oof_predictions.csv'):
        predictions[(row['model_id'], row['image_id'])].append(row)
    risk_rows = []
    for (model_id, image_id), rows in predictions.items():
        rows.sort(key=lambda row: int(row['region_index']))
        metrics = by_image[image_id]
        gains = np.array([float(row['psnr_gain']) for row in metrics])
        scores = np.array([float(row['probability']) for row in rows])
        for result in risk_coverage(scores, gains, cfg['coverage_grid']):
            selected = result.pop('selected_indices')
            result.pop('mean_gain')
            areas = np.asarray(areas_by_image[image_id])
            result['actual_pixel_coverage'] = float(areas[selected].sum() / areas.sum())
            result['pixel_weighted_damage_rate'] = float(np.average(gains[selected] <= 0, weights=areas[selected]))
            risk_rows.append({'model_id': model_id, 'image_id': image_id, 'domain': rows[0]['domain'], **result,
                              'selected_count': len(selected), 'delivered_damage_fraction': float((gains[selected] <= 0).sum() / len(gains)), 'hallucination_rate': None,
                              **{name + '_gain': float(np.mean([float(metrics[index][name + '_gain']) for index in selected])) for name in METRICS},
                              **{name + '_selected': float(np.mean([float(metrics[index][name + '_restored']) for index in selected])) for name in METRICS}})
    for image_number, (image_id, metrics) in enumerate(by_image.items()):
        gains = np.array([float(row['psnr_gain']) for row in metrics])
        for name, scores in [('random', np.random.default_rng(cfg['seed'] + image_number).random(len(metrics))), ('oracle_rank_psnr', gains)]:
            for result in risk_coverage(scores, gains, cfg['coverage_grid']):
                selected = result.pop('selected_indices')
                result.pop('mean_gain')
                areas = np.asarray(areas_by_image[image_id])
                result['actual_pixel_coverage'] = float(areas[selected].sum() / areas.sum())
                result['pixel_weighted_damage_rate'] = float(np.average(gains[selected] <= 0, weights=areas[selected]))
                risk_rows.append({'model_id': name, 'image_id': image_id, 'domain': metrics[0]['domain'], **result, 'selected_count': len(selected),
                                  'delivered_damage_fraction': float((gains[selected] <= 0).sum() / len(gains)), 'hallucination_rate': None,
                                  **{metric + '_gain': float(np.mean([float(metrics[index][metric + '_gain']) for index in selected])) for metric in METRICS},
                                  **{metric + '_selected': float(np.mean([float(metrics[index][metric + '_restored']) for index in selected])) for metric in METRICS}})
    write_csv(ROOT / 'analysis/risk_coverage.csv', risk_rows)
    summary_groups = defaultdict(list)
    for row in risk_rows:
        for domain in ['all', row['domain']]:
            summary_groups[(row['model_id'], domain, row['coverage'])].append(row)
    summaries = []
    for (model_id, domain, coverage), rows in summary_groups.items():
        summaries.append({'model_id': model_id, 'domain': domain, 'coverage': coverage, 'image_count': len(rows), 'hallucination_rate': None,
                          **{key: float(np.mean([row[key] for row in rows])) for key in ['damage_rate', 'actual_patch_coverage', 'actual_pixel_coverage', 'pixel_weighted_damage_rate', 'delivered_damage_fraction'] + [metric + '_gain' for metric in METRICS]}})
    write_csv(ROOT / 'analysis/risk_coverage_summary.csv', summaries)
    primary = read_json(ROOT / 'analysis/training.json')['primary_model']
    comparison = []
    for domain in ['all', 'real_csig', 'synthetic_bsds']:
        model_rows = {row['image_id']: row for row in risk_rows if row['model_id'] == primary and row['coverage'] == cfg['primary_coverage'] and (domain == 'all' or row['domain'] == domain)}
        baseline_rows = {row['image_id']: row for row in risk_rows if row['model_id'] == 'random' and row['coverage'] == cfg['primary_coverage'] and (domain == 'all' or row['domain'] == domain)}
        for metric in ['damage_rate', 'psnr_gain', 'ssim_gain', 'lpips_gain', 'dists_gain']:
            comparison.append({'domain': domain, 'metric': metric, 'comparison': 'V7 logistic minus fixed random', 'coverage': cfg['primary_coverage'],
                               **bootstrap_difference([row[metric] for row in model_rows.values()], [baseline_rows[image_id][metric] for image_id in model_rows], cfg['seed'], cfg['bootstrap_scenes'])})
    write_csv(ROOT / 'analysis/paired_bootstrap.csv', comparison)
    return by_image, predictions, primary


def actual_fusions(cfg, by_image, predictions, primary):
    import torch
    torch.set_num_threads(4)
    perceptual = PerceptualMetrics()
    full_baselines = {row['image_id']: row for row in read_csv(ROOT / 'analysis/image_metrics.csv')}
    rows, artifact_records = [], []
    for image_number, entry in enumerate(read_json(ROOT / 'data/manifests/pairs.json')):
        image_id = entry['image_id']
        lq, restored, gt = [load_rgb(resolve(entry[key])) for key in ['lq_path', 'candidate_path', 'gt_path']]
        regions = partition_regions(entry['width'], entry['height'], cfg['analysis_patch_size'])
        region_metrics = by_image[image_id]
        gains = np.array([float(row['psnr_gain']) for row in region_metrics])
        scores = np.array([float(row['probability']) for row in sorted(predictions[(primary, image_id)], key=lambda row: int(row['region_index']))])
        random_scores = np.random.default_rng(cfg['seed'] + image_number).random(len(regions))
        count = max(1, math.ceil(len(regions) * cfg['primary_coverage']))
        input_error = ((lq.astype(np.float32) - gt) ** 2).sum(axis=-1)
        restored_error = ((restored.astype(np.float32) - gt) ** 2).sum(axis=-1)
        sse_improvements = np.array([(input_error - restored_error)[region['y']:region['y'] + region['height'], region['x']:region['x'] + region['width']].sum(dtype=np.float64) for region in regions])
        selections = {'predicted': np.argsort(-scores, kind='stable')[:count], 'random': np.argsort(-random_scores, kind='stable')[:count],
                      'oracle_budget_sse': budget_oracle_selection(sse_improvements, count), 'oracle_positive': np.flatnonzero(sse_improvements > 0)}
        baseline = full_baselines[image_id]
        base_quality = {metric: float(baseline[metric + '_lq']) for metric in METRICS}
        candidate_quality = {metric: float(baseline[metric + '_restored']) for metric in METRICS}
        for mode, quality, coverage in [('LQ', base_quality, 0.), ('HYPIR', candidate_quality, 1.), ('always_HYPIR', candidate_quality, 1.)]:
            rows.append({'image_id': image_id, 'domain': entry['domain'], 'mode': mode, 'requested_coverage': coverage, 'pixel_coverage': coverage, 'region_coverage': coverage,
                         'selected_damage_rate': float((gains <= 0).mean()) if coverage else None, 'delivered_damage_fraction': float((gains <= 0).mean()) * coverage,
                         'hallucination_rate': None, **quality, **metric_gains(base_quality, quality)})
        destination = ROOT / 'outputs/selective' / image_id
        destination.mkdir(parents=True, exist_ok=True)
        for mode, selected in selections.items():
            fused, mask = fuse_regions(lq, restored, regions, selected)
            quality = {**pixel_metrics(fused, gt), **perceptual.compare(fused, gt, full=True)}
            path = destination / f'{mode}.png'
            mask_path = destination / f'{mode}_mask.png'
            Image.fromarray(fused).save(path)
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
            rows.append({'image_id': image_id, 'domain': entry['domain'], 'mode': mode, 'requested_coverage': None if mode == 'oracle_positive' else cfg['primary_coverage'],
                         'pixel_coverage': float(mask.mean()), 'region_coverage': len(selected) / len(regions), 'selected_damage_rate': float((gains[selected] <= 0).mean()) if len(selected) else None,
                         'delivered_damage_fraction': float((gains[selected] <= 0).sum() / len(gains)), 'hallucination_rate': None, **quality, **metric_gains(base_quality, quality)})
            artifact_records.append({'image_id': image_id, 'mode': mode, 'selected_regions': selected.tolist(), 'image': path.relative_to(ROOT).as_posix(), 'image_sha256': sha256(path), 'mask': mask_path.relative_to(ROOT).as_posix(), 'mask_sha256': sha256(mask_path)})
        fused, mask = pixel_oracle(lq, restored, gt)
        quality = {**pixel_metrics(fused, gt), **perceptual.compare(fused, gt, full=True)}
        path, mask_path = destination / 'oracle_pixel.png', destination / 'oracle_pixel_mask.png'
        Image.fromarray(fused).save(path)
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        rows.append({'image_id': image_id, 'domain': entry['domain'], 'mode': 'oracle_pixel', 'requested_coverage': None, 'pixel_coverage': float(mask.mean()), 'region_coverage': None,
                     'selected_damage_rate': None, 'delivered_damage_fraction': None, 'hallucination_rate': None, **quality, **metric_gains(base_quality, quality)})
        artifact_records.append({'image_id': image_id, 'mode': 'oracle_pixel', 'image': path.relative_to(ROOT).as_posix(), 'image_sha256': sha256(path), 'mask': mask_path.relative_to(ROOT).as_posix(), 'mask_sha256': sha256(mask_path)})
        print('Fusion', image_number + 1, '/55', image_id, flush=True)
    write_csv(ROOT / 'analysis/selective_fusion_metrics.csv', rows)
    write_csv(ROOT / 'analysis/oracle_upper_bound.csv', [row for row in rows if row['mode'].startswith('oracle')])
    write_json(ROOT / 'analysis/selective_artifacts.json', artifact_records)
    summary = []
    for domain in ['all', 'real_csig', 'synthetic_bsds']:
        for mode in dict.fromkeys(row['mode'] for row in rows):
            selected = [row for row in rows if row['mode'] == mode and (domain == 'all' or row['domain'] == domain)]
            summary.append({'domain': domain, 'mode': mode, 'image_count': len(selected), 'images_psnr_below_lq': sum(row['psnr_gain'] < -1e-8 for row in selected),
                            **{metric: float(np.mean([row[metric] for row in selected])) for metric in METRICS + [metric + '_gain' for metric in METRICS] + ['pixel_coverage']}})
    write_csv(ROOT / 'analysis/selective_fusion_summary.csv', summary)


if __name__ == '__main__':
    cfg = config()
    metric_rows = read_csv(ROOT / 'analysis/restoration_gain.csv')
    by_image, predictions, primary = risk_tables(cfg, metric_rows)
    actual_fusions(cfg, by_image, predictions, primary)
