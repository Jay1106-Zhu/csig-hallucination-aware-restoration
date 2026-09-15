import json

import numpy as np

from core import ROOT, config, load_rgb, metric_gains, partition_regions, read_json, relative, resolve, sha256
from evaluation import PerceptualMetrics, pixel_metrics, write_csv


def save_region_archive(entry, regions, arrays):
    target = ROOT / 'data/patches' / f"{entry['image_id']}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, regions_json=np.array(json.dumps(regions)), **arrays)
    return target


def main():
    import torch
    torch.set_num_threads(4)
    cfg = config()
    entries = read_json(ROOT / 'data/manifests/pairs.json')
    perceptual = PerceptualMetrics()
    index_rows, gain_rows, image_rows = [], [], []
    for scene_number, entry in enumerate(entries, 1):
        lq = load_rgb(resolve(entry['lq_path']))
        gt = load_rgb(resolve(entry['gt_path']))
        restored = load_rgb(ROOT / entry['candidate_path'])
        if lq.shape != gt.shape or lq.shape != restored.shape:
            raise RuntimeError(f"native geometry mismatch: {entry['image_id']}")
        regions = partition_regions(lq.shape[1], lq.shape[0], cfg['analysis_patch_size'])
        patch_arrays = {'lq': [], 'gt': [], 'restored': []}
        for region_index, region in enumerate(regions):
            column, row, width, height = region['x'], region['y'], region['width'], region['height']
            patches = {'lq': lq[row:row + height, column:column + width], 'gt': gt[row:row + height, column:column + width], 'restored': restored[row:row + height, column:column + width]}
            base = pixel_metrics(patches['lq'], patches['gt'])
            candidate = pixel_metrics(patches['restored'], patches['gt'])
            if perceptual:
                base.update(perceptual.compare(patches['lq'], patches['gt']))
                candidate.update(perceptual.compare(patches['restored'], patches['gt']))
            gains = metric_gains(base, candidate)
            center_x, center_y = column + width / 2, row + height / 2
            padding = {}
            for size in cfg['multiscale_contexts']:
                left, top = round(center_x - size / 2), round(center_y - size / 2)
                padding[str(size)] = [max(0, -left), max(0, -top), max(0, left + size - lq.shape[1]), max(0, top + size - lq.shape[0])]
            index_rows.append({'image_id': entry['image_id'], 'scene_id': entry['scene_id'], 'domain': entry['domain'], 'region_index': region_index, **region, 'center_x': center_x, 'center_y': center_y, 'context_padding': json.dumps(padding)})
            gain_rows.append({'image_id': entry['image_id'], 'scene_id': entry['scene_id'], 'domain': entry['domain'], 'region_index': region_index, 'psnr_lq': base['psnr'], 'psnr_restored': candidate['psnr'], 'ssim_lq': base['ssim'], 'ssim_restored': candidate['ssim'], 'lpips_lq': base.get('lpips'), 'lpips_restored': candidate.get('lpips'), 'dists_lq': base.get('dists'), 'dists_restored': candidate.get('dists'), **gains, 'four_metric_majority': int(sum(gains[name] > 0 for name in ['psnr_gain', 'ssim_gain', 'lpips_gain', 'dists_gain']) >= 3), 'psnr_benefit': int(gains['psnr_gain'] > 0)})
            for name, patch in patches.items():
                padded = np.zeros((cfg['analysis_patch_size'], cfg['analysis_patch_size'], 3), dtype=np.uint8)
                padded[:height, :width] = patch
                patch_arrays[name].append(padded)
        archive_arrays = {name: np.stack(patches) for name, patches in patch_arrays.items()}
        archive = save_region_archive(entry, regions, archive_arrays)
        for row in index_rows[-len(regions):]:
            row['archive_path'] = relative(archive)
        full_lq = pixel_metrics(lq, gt)
        full_restored = pixel_metrics(restored, gt)
        if perceptual:
            full_lq.update(perceptual.compare(lq, gt, full=True))
            full_restored.update(perceptual.compare(restored, gt, full=True))
        image_rows.append({'image_id': entry['image_id'], 'scene_id': entry['scene_id'], 'domain': entry['domain'], 'scene_category': entry.get('scene_category', 'unknown'), 'width': lq.shape[1], 'height': lq.shape[0], 'region_count': len(regions), 'psnr_lq': full_lq['psnr'], 'psnr_restored': full_restored['psnr'], 'ssim_lq': full_lq['ssim'], 'ssim_restored': full_restored['ssim'], 'lpips_lq': full_lq.get('lpips'), 'lpips_restored': full_restored.get('lpips'), 'dists_lq': full_lq.get('dists'), 'dists_restored': full_restored.get('dists'), **metric_gains(full_lq, full_restored), 'candidate_sha256': sha256(ROOT / entry['candidate_path'])})
        print(f"[{scene_number}/{len(entries)}] {entry['image_id']}: {len(regions)} regions", flush=True)
    write_csv(ROOT / 'analysis/region_metrics.csv', gain_rows)
    write_csv(ROOT / 'analysis/restoration_gain.csv', gain_rows)
    write_csv(ROOT / 'analysis/image_metrics.csv', image_rows)
    write_csv(ROOT / 'analysis/region_index.csv', index_rows)
    (ROOT / 'analysis/region_metrics_meta.json').write_text(json.dumps({'metric_direction': 'all gains positive means improvement', 'regions': len(gain_rows), 'images': len(entries), 'perceptual_full_longside_max': 1024, 'candidate': 'coeff_t=200', 'gt_used_only_for_label_and_evaluation': True}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
