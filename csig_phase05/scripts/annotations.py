import argparse
import datetime
import json

import numpy as np
from PIL import Image, ImageDraw

from core import ROOT, config, load_rgb, partition_regions, read_json, resolve, sha256, validate_annotation, write_json
from evaluation import read_csv, write_csv

TYPES = {'texture_invention', 'structure_invention', 'semantic_corruption', 'text_glyph_corruption', 'boundary_corruption', 'object_invention', 'repeated_pattern', 'over_sharpening_artifact', 'uncertain'}


def choose_regions(count, seed, amount):
    return np.random.default_rng(seed).choice(count, size=min(count, amount), replace=False).tolist()


def validate_review(row):
    if not validate_annotation(row):
        return False
    if not row.get('reviewer_id', '').strip() or not row.get('reviewed_at', '').strip():
        raise ValueError('Human review requires reviewer_id and reviewed_at')
    datetime.date.fromisoformat(row['reviewed_at'][:10])
    types = set(filter(None, row.get('hallucination_types', '').split(';')))
    if types - TYPES or (int(row['severity']) > 0 and not types):
        raise ValueError('Invalid or missing hallucination types')
    return True


def prepare():
    cfg = config()
    destination = ROOT / 'data/annotations'
    (destination / 'images').mkdir(parents=True, exist_ok=True)
    rows = []
    for scene_number, entry in enumerate(read_json(ROOT / 'data/manifests/pairs.json')):
        regions = partition_regions(entry['width'], entry['height'], cfg['analysis_patch_size'])
        arrays = [load_rgb(resolve(entry[key])) for key in ['lq_path', 'candidate_path', 'gt_path']]
        for region_index in choose_regions(len(regions), cfg['seed'] + scene_number, cfg['annotation_regions_per_scene']):
            region = regions[region_index]
            sample_id = f'review_{len(rows):03d}'
            montage = Image.new('RGB', (768, 284), (32, 32, 32))
            painter = ImageDraw.Draw(montage)
            for view_index, (name, image) in enumerate(zip(['LQ', 'RESTORED', 'GT'], arrays)):
                patch = image[region['y']:region['y'] + region['height'], region['x']:region['x'] + region['width']]
                painter.text((view_index * 256 + 8, 6), name, fill='white')
                montage.paste(Image.fromarray(patch), (view_index * 256, 28))
            image_path = destination / 'images' / f'{sample_id}.png'
            montage.save(image_path)
            rows.append({'sample_id': sample_id, 'image_id': entry['image_id'], 'region_index': region_index, **region,
                         'image_path': f'images/{sample_id}.png', 'image_sha256': sha256(image_path), 'severity': '', 'hallucination_types': '',
                         'reviewer_type': '', 'reviewer_id': '', 'reviewed_at': '', 'annotation_status': 'pending', 'note': ''})
    pending = destination / 'hallucination_labels.csv'
    if pending.exists() and read_csv(pending) != [{key: str(value) for key, value in row.items()} for row in rows]:
        raise ValueError('Existing annotation rows changed; preserve and review instead of overwriting')
    write_csv(pending, rows)
    template = (ROOT / 'scripts/annotation_template.html').read_text(encoding='utf-8')
    (destination / 'index.html').write_text(template.replace('__ROWS__', json.dumps(rows, ensure_ascii=False)).replace('__TYPES__', json.dumps(sorted(TYPES))), encoding='utf-8')
    write_json(ROOT / 'analysis/annotation_status.json', {'selected_regions': len(rows), 'reviewed_human_regions': 0, 'status': 'pending independent human review', 'seed': cfg['seed'], 'no_scores_shown': True})
    print(f'{len(rows)} blind regions prepared; no human labels fabricated', flush=True)


def import_reviews(path):
    expected = {row['sample_id']: row for row in read_csv(ROOT / 'data/annotations/hallucination_labels.csv')}
    seen, reviewed = set(), []
    for row in read_csv(path):
        sample = row['sample_id']
        if sample not in expected or sample in seen:
            raise ValueError('Unknown or repeated sample')
        seen.add(sample)
        for key in ['image_id', 'region_index', 'x', 'y', 'width', 'height', 'image_sha256']:
            if row[key] != expected[sample][key]:
                raise ValueError('Review mapping mismatch: ' + key)
        if validate_review(row):
            reviewed.append(row)
    if not reviewed:
        raise ValueError('No eligible human reviews; nothing imported')
    write_csv(ROOT / 'data/annotations/reviewed_labels.csv', reviewed)
    write_json(ROOT / 'analysis/annotation_status.json', {'selected_regions': len(expected), 'reviewed_human_regions': len(reviewed), 'status': 'human review imported; rerun human-dependent analyses', 'import_sha256': sha256(path)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--import-csv')
    arguments = parser.parse_args()
    import_reviews(arguments.import_csv) if arguments.import_csv else prepare()
