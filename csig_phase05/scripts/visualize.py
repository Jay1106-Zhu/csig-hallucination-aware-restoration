from collections import defaultdict

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from core import ROOT, config, load_rgb, partition_regions, read_json, resolve
from evaluation import read_csv

DESTINATION = ROOT / 'reports/visualization'


def region_heatmap(width, height, regions, values):
    result = np.full((height, width), np.nan, dtype=np.float32)
    for region, value in zip(regions, values, strict=True):
        result[region['y']:region['y'] + region['height'], region['x']:region['x'] + region['width']] = value
    return result


def token_heatmap(width, height, regions, token_maps):
    result = np.full((height, width), np.nan, dtype=np.float32)
    for region, values in zip(regions, token_maps, strict=True):
        center_x, center_y = region['x'] + region['width'] / 2, region['y'] + region['height'] / 2
        origin_x, origin_y = round(center_x - 128) + 16, round(center_y - 128) + 16
        left, top = max(region['x'], origin_x, 0), max(region['y'], origin_y, 0)
        right, bottom = min(region['x'] + region['width'], origin_x + 224, width), min(region['y'] + region['height'], origin_y + 224, height)
        dense = cv2.resize(values.astype(np.float32), (224, 224), interpolation=cv2.INTER_NEAREST)
        result[top:bottom, left:right] = dense[top - origin_y:bottom - origin_y, left - origin_x:right - origin_x]
    return result


def preview(image, limit=768):
    ratio = min(1., limit / max(image.shape[:2]))
    return cv2.resize(image, (round(image.shape[1] * ratio), round(image.shape[0] * ratio)), interpolation=cv2.INTER_NEAREST if image.ndim == 2 else cv2.INTER_AREA)


def finish(figure, name):
    figure.tight_layout()
    figure.savefig(DESTINATION / name, dpi=140, bbox_inches='tight')
    plt.close(figure)


def plot_scenes(primary_predictions, metric_rows):
    annotations = read_csv(ROOT / 'data/annotations/hallucination_labels.csv')
    for entry in read_json(ROOT / 'data/manifests/pairs.json'):
        image_id = entry['image_id']
        regions = partition_regions(entry['width'], entry['height'], config()['analysis_patch_size'])
        quality = [row for row in metric_rows if row['image_id'] == image_id]
        probabilities = [float(row['probability']) for row in primary_predictions if row['image_id'] == image_id]
        scene = np.load(ROOT / 'features/scenes' / f'{image_id}.npz')
        gain = region_heatmap(entry['width'], entry['height'], regions, [float(row['psnr_gain']) for row in quality])
        confidence = region_heatmap(entry['width'], entry['height'], regions, probabilities)
        disagreement = token_heatmap(entry['width'], entry['height'], regions, scene['disagreement_256'])
        np.savez_compressed(ROOT / 'analysis' / f'heatmaps_{image_id}.npz', psnr_gain=gain, confidence=confidence, token_disagreement=disagreement)
        images = [load_rgb(resolve(entry[key])) for key in ['lq_path', 'candidate_path', 'gt_path']]
        fused = load_rgb(ROOT / 'outputs/selective' / image_id / 'predicted.png')
        figure, axes = plt.subplots(2, 4, figsize=(16, 8))
        for axis, image, title in zip(axes[0], images + [fused], ['LQ', 'Frozen HYPIR', 'GT (offline only)', 'Predicted fusion (50% regions)']):
            axis.imshow(preview(image))
            axis.set_title(title)
        for axis, values, title, colormap, bounds in zip(axes[1, :3], [gain, confidence, disagreement], ['PSNR gain [dB]', 'V7 benefit probability', 'DINO token disagreement (central 224)'], ['coolwarm', 'viridis', 'magma'], [(-8, 8), (0, 1), (0, 1)]):
            plotted = axis.imshow(preview(values), cmap=colormap, vmin=bounds[0], vmax=bounds[1])
            axis.set_title(title)
            figure.colorbar(plotted, ax=axis, shrink=.7)
        axes[1, 3].imshow(preview(images[0]))
        ratio = min(1., 768 / max(entry['width'], entry['height']))
        for row in annotations:
            if row['image_id'] == image_id:
                axes[1, 3].add_patch(Rectangle((int(row['x']) * ratio, int(row['y']) * ratio), int(row['width']) * ratio, int(row['height']) * ratio, fill=False, color='orange', linewidth=2))
        axes[1, 3].set_title('Blind subset overlay; human severity PENDING')
        for axis in axes.ravel():
            axis.axis('off')
        figure.suptitle(f'{image_id} | {entry["domain"]} | not a human hallucination detector', fontsize=14)
        finish(figure, f'scene_{image_id}.jpg')
        if entry['domain'] == 'real_csig':
            for name, values, colormap in [('psnr_gain', gain, 'coolwarm'), ('confidence', confidence, 'viridis'), ('token_disagreement', disagreement, 'magma')]:
                figure, axis = plt.subplots(figsize=(7, 5))
                plotted = axis.imshow(preview(values), cmap=colormap)
                figure.colorbar(plotted, ax=axis)
                axis.set_title(f'{image_id}: {name}')
                axis.axis('off')
                finish(figure, f'{name}_{image_id}.png')
        print('Visualization', image_id, flush=True)


def plot_curves(primary_predictions):
    from sklearn.metrics import precision_recall_curve, roc_curve

    by_image = defaultdict(list)
    for row in primary_predictions:
        by_image[row['image_id']].append(row)
    for name in ['roc', 'pr']:
        figure, axis = plt.subplots(figsize=(8, 6))
        informative = 0
        for image_id, rows in by_image.items():
            labels = np.array([int(row['label']) for row in rows])
            probabilities = np.array([float(row['probability']) for row in rows])
            if len(np.unique(labels)) < 2:
                continue
            informative += 1
            if name == 'roc':
                horizontal, vertical, _ = roc_curve(labels, probabilities)
            else:
                vertical, horizontal, _ = precision_recall_curve(labels, probabilities)
            axis.plot(horizontal, vertical, label=f'{image_id} (+{labels.sum()}/{len(labels)})')
        if name == 'roc':
            axis.plot([0, 1], [0, 1], '--', color='gray')
        axis.set(xlabel='False positive rate' if name == 'roc' else 'Recall', ylabel='True positive rate' if name == 'roc' else 'Precision',
                 title=f'V7 logistic PSNR benefit: {informative}/{len(by_image)} defined images')
        axis.legend(fontsize=8)
        finish(figure, name + '_curve.png')
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for domain, axis in zip(['real_csig', 'synthetic_bsds'], axes):
        bin_probabilities, bin_labels = defaultdict(list), defaultdict(list)
        for rows in by_image.values():
            if rows[0]['domain'] != domain:
                continue
            probabilities = np.array([float(row['probability']) for row in rows])
            labels = np.array([int(row['label']) for row in rows])
            bins = np.minimum((probabilities * 10).astype(int), 9)
            for bin_index in range(10):
                selected = bins == bin_index
                if selected.any():
                    bin_probabilities[bin_index].append(probabilities[selected].mean())
                    bin_labels[bin_index].append(labels[selected].mean())
        axis.plot([0, 1], [0, 1], '--', color='gray')
        axis.plot([np.mean(bin_probabilities[index]) for index in sorted(bin_probabilities)], [np.mean(bin_labels[index]) for index in sorted(bin_labels)], 'o-')
        axis.set(title=domain + ' (image-macro within occupied bins)', xlabel='Predicted benefit probability', ylabel='Observed positive fraction', xlim=(0, 1), ylim=(0, 1))
    finish(figure, 'calibration.png')
    risk = read_csv(ROOT / 'analysis/risk_coverage_summary.csv')
    primary = read_json(ROOT / 'analysis/training.json')['primary_model']
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for column, domain in enumerate(['all', 'real_csig', 'synthetic_bsds']):
        for name, label in [(primary, 'V7 fixed primary'), ('mixed_groupcv_psnr_benefit_C1_lq_logistic_505', 'LQ-only'), ('random', 'Fixed random'), ('oracle_rank_psnr', 'GT oracle ranking')]:
            rows = [row for row in risk if row['model_id'] == name and row['domain'] == domain]
            for axis, metric in zip(axes[:, column], ['damage_rate', 'psnr_gain']):
                axis.plot([float(row['coverage']) for row in rows], [float(row[metric]) for row in rows], marker='.', label=label)
                axis.set(xlabel='Requested region coverage', ylabel=metric, title=domain)
                axis.axvline(.5, color='gray', linestyle=':')
        axes[0, column].legend(fontsize=7)
    finish(figure, 'risk_coverage.png')


def plot_comparisons():
    summaries = read_csv(ROOT / 'analysis/verifier_summary.csv')
    variants = ['C1_lq', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'C5_pair_stability']
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    for axis, domain in zip(axes, ['all', 'real_csig', 'synthetic_bsds']):
        rows = {row['variant']: row for row in summaries if row['scope'] == 'mixed_groupcv' and row['target'] == 'psnr_benefit' and row['family'] == 'logistic' and row['domain'] == domain}
        axis.bar(np.arange(len(variants)), [float(rows[variant]['roc_auc']) for variant in variants], color=['#de8f05' if variant == 'V7' else '#4885a3' for variant in variants])
        axis.set_xticks(np.arange(len(variants)), variants, rotation=70)
        axis.axhline(.5, color='gray', linestyle='--')
        axis.set(ylim=(0, 1), title=f'{domain}: {rows["V7"]["roc_auc_defined_images"]} informative images', ylabel='Image-macro ROC-AUC')
    finish(figure, 'verifier_ablation.png')
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, domain in zip(axes, ['all', 'real_csig', 'synthetic_bsds']):
        for seed in config()['verifier_seeds']:
            selected = {row['variant']: row for row in summaries if row['scope'] == 'mixed_groupcv' and row['target'] == 'psnr_benefit' and row['family'] == 'mlp' and row['domain'] == domain and int(row['seed']) == seed}
            axis.plot(range(8), [float(selected['V' + str(number)]['roc_auc']) for number in range(1, 9)], 'o-', label=str(seed))
        axis.set_xticks(range(8), ['V' + str(number) for number in range(1, 9)])
        axis.set(title=domain, ylim=(0, 1), ylabel='Image-macro AUC')
        axis.legend(title='All fixed MLP seeds')
    finish(figure, 'mlp_seed_variation.png')
    fusion = read_csv(ROOT / 'analysis/selective_fusion_summary.csv')
    modes = ['HYPIR', 'predicted', 'random', 'oracle_budget_sse', 'oracle_positive', 'oracle_pixel']
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, domain in zip(axes, ['all', 'real_csig', 'synthetic_bsds']):
        rows = {row['mode']: row for row in fusion if row['domain'] == domain}
        axis.bar(range(len(modes)), [float(rows[mode]['psnr_gain']) for mode in modes])
        axis.set_xticks(range(len(modes)), modes, rotation=65)
        axis.axhline(0, color='black', linewidth=.8)
        axis.set(title=domain, ylabel='Actual full-image PSNR gain vs LQ [dB]')
    finish(figure, 'actual_selective_fusion.png')
    grouped = read_csv(ROOT / 'analysis/scene_group_metrics.csv')
    figure, axis = plt.subplots(figsize=(12, 5))
    for position, row in enumerate(grouped):
        if row['roc_auc']:
            axis.bar(position, float(row['roc_auc']))
        else:
            axis.text(position, .1, 'NA', ha='center')
    axis.set_xticks(range(len(grouped)), [row['domain'].replace('_csig', '').replace('_bsds', '') + '\n' + row['category'] + f' (n={row["image_count"]})' for row in grouped], rotation=70)
    axis.set(ylim=(0, 1), ylabel='V7 image-macro AUC', title='Scene categories are assistant visual metadata, NOT hallucination labels')
    finish(figure, 'scene_group_performance.png')


def pending_human_figures():
    for name, title in [('hallucination_severity_heatmap.png', 'Human hallucination severity heatmap'), ('psnr_vs_hallucination.png', 'PSNR gain vs human hallucination label'),
                        ('dino_vs_hallucination.png', 'DINO disagreement vs human severity'), ('stability_vs_hallucination.png', 'Controlled sensitivity vs human severity'),
                        ('hallucination_confusion.png', 'PSNR BAD vs human hallucination: confusion / agreement')]:
        figure, axis = plt.subplots(figsize=(9, 4))
        axis.set_facecolor('#e5e7eb')
        axis.text(.5, .57, 'NOT COMPUTABLE: independent human labels pending', ha='center', fontsize=13, transform=axis.transAxes)
        axis.text(.5, .4, '110 blind samples prepared; 0 reviewed. Missing is not severity 0.', ha='center', fontsize=10, transform=axis.transAxes)
        axis.set(title=title, xticks=[], yticks=[])
        finish(figure, name)
    figure, axes = plt.subplots(4, 3, figsize=(13, 13))
    entries = {entry['image_id']: entry for entry in read_json(ROOT / 'data/manifests/pairs.json')}
    for row_index, (category, image_id, center) in enumerate([('text', 'case1', (0.5, 0.45)), ('animal', 'case3', (0.5, 0.5)), ('vegetation', 'case4', (0.5, 0.5)), ('night', None, None)]):
        for column, key in enumerate(['lq_path', 'candidate_path', 'gt_path']):
            axis = axes[row_index, column]
            if image_id:
                entry = entries[image_id]
                image = load_rgb(resolve(entry[key]))
                center_x, center_y = round(entry['width'] * center[0]), round(entry['height'] * center[1])
                axis.imshow(image[center_y - 256:center_y + 256, center_x - 256:center_x + 256])
            else:
                axis.text(.5, .5, 'No real paired night scene\nSynthetic darkening is NOT night coverage', ha='center', transform=axis.transAxes)
            axis.set_title(category + ': ' + ['LQ', 'HYPIR', 'GT'][column])
            axis.axis('off')
    finish(figure, 'representative_scene_comparisons.jpg')


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    primary = read_json(ROOT / 'analysis/training.json')['primary_model']
    predictions = [row for row in read_csv(ROOT / 'analysis/oof_predictions.csv') if row['model_id'] == primary]
    plot_curves(predictions)
    plot_comparisons()
    pending_human_figures()
    plot_scenes(predictions, read_csv(ROOT / 'analysis/restoration_gain.csv'))


if __name__ == '__main__':
    main()
