import sys
import time

import cv2
import numpy as np
from PIL import Image

from core import ROOT, REPOSITORY, config, context_crop, load_rgb, pairwise_global, partition_regions, read_json, resolve, sha256, token_correspondence, write_json
from evaluation import PerceptualMetrics, structure_features, write_csv


def stability_summary(images, cls, tokens, perceptual_differences):
    values = images.astype(np.float32) / 255
    variance = values.var(axis=0)
    edges = np.stack([cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 100, 200).astype(np.float32) / 255 for image in images])
    cls_variance = cls.var(axis=0)
    token_variance = tokens.var(axis=0).sum(axis=-1)
    return np.asarray([variance.mean(), variance.std(), np.quantile(variance, .9), edges.var(axis=0).mean(),
                       np.mean(perceptual_differences), np.std(perceptual_differences), np.max(perceptual_differences),
                       cls_variance.mean(), cls_variance.max(), token_variance.mean(), token_variance.std(), np.quantile(token_variance, .9)], dtype=np.float32)


def feature_variants(features):
    global_pair = np.concatenate([features['dino_lq'], features['dino_restored']], axis=1)
    return {'V1': global_pair, 'V2': features['pairwise'], 'V3': np.concatenate([features['token256'], features['token512']], axis=1),
            'V4': features['structure'], 'V5': features['stability'], 'V6': np.concatenate([features['pairwise'], features['structure']], axis=1),
            'V7': np.concatenate([features['pairwise'], features['structure'], features['stability']], axis=1),
            'V8': np.concatenate([features['clip'], global_pair, features['stats']], axis=1), 'C1_lq': features['dino_lq'], 'C1_restored': features['dino_restored'],
            'C1_absdiff': np.abs(features['dino_lq'] - features['dino_restored']), 'C3_token256': features['token256'], 'C3_token512': features['token512'],
            'C5_pair_stability': np.concatenate([features['pairwise'], features['stability']], axis=1)}


def encode(images, model, processor, batch_size, is_clip=False):
    import torch

    embeddings, tokens = [], []
    with torch.inference_mode():
        for offset in range(0, len(images), batch_size):
            values = processor(images=[Image.fromarray(image) for image in images[offset:offset + batch_size]], return_tensors='pt')['pixel_values'].to('cuda')
            with torch.autocast('cuda', dtype=torch.float16):
                output = model(pixel_values=values)
            embedding = output.image_embeds if is_clip else output.last_hidden_state[:, 0]
            embeddings.append(torch.nn.functional.normalize(embedding.float(), dim=-1).cpu().numpy())
            if not is_clip:
                tokens.append(torch.nn.functional.normalize(output.last_hidden_state[:, 1:].float(), dim=-1).cpu().numpy())
    return np.concatenate(embeddings), None if is_clip else np.concatenate(tokens)


def main():
    import torch
    from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPVisionModelWithProjection

    sys.path.insert(0, str(REPOSITORY / 'csig_phase0/scripts'))
    from phase0_core import statistical_features

    cfg = config()
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    dino_path, clip_path = REPOSITORY / 'csig_phase0/models/dino', REPOSITORY / 'csig_phase0/models/clip'
    dino = AutoModel.from_pretrained(dino_path, local_files_only=True).eval().requires_grad_(False).to('cuda')
    dino_processor = AutoImageProcessor.from_pretrained(dino_path, local_files_only=True, use_fast=False)
    clip = CLIPVisionModelWithProjection.from_pretrained(clip_path, local_files_only=True).eval().requires_grad_(False).to('cuda')
    clip_processor = CLIPImageProcessor.from_pretrained(clip_path, local_files_only=True)
    perceptual = PerceptualMetrics()
    all_features, index_rows, feature_hashes = [], [], {}
    started = time.perf_counter()
    for scene_number, entry in enumerate(read_json(ROOT / 'data/manifests/pairs.json'), 1):
        scene_file = ROOT / 'features/scenes' / (entry['image_id'] + '.npz')
        scene_file.parent.mkdir(parents=True, exist_ok=True)
        images = [load_rgb(resolve(entry['lq_path']))] + [load_rgb(resolve(entry['stability_paths'][str(coefficient)])) for coefficient in cfg['hypir_coefficients']]
        regions = partition_regions(entry['width'], entry['height'], cfg['analysis_patch_size'])
        contexts = [[context_crop(image, region['x'] + region['width'] / 2, region['y'] + region['height'] / 2, 256) for region in regions] for image in images]
        cls_views, token_views = zip(*[encode(view, dino, dino_processor, cfg['feature_batch_size']) for view in contexts])
        condition_cls = np.stack(cls_views[1:])
        condition_tokens = np.stack(token_views[1:])
        contexts512 = [[context_crop(image, region['x'] + region['width'] / 2, region['y'] + region['height'] / 2, 512) for region in regions] for image in [images[0], images[-1]]]
        cls512, token512 = zip(*[encode(view, dino, dino_processor, cfg['feature_batch_size']) for view in contexts512])
        clip_views = [encode(view, clip, clip_processor, cfg['feature_batch_size'], True)[0] for view in [contexts[0], contexts[-1]]]
        features = {'dino_lq': cls_views[0], 'dino_restored': cls_views[-1], 'clip': np.concatenate(clip_views, axis=1),
                    'pairwise': [], 'token256': [], 'token512': [], 'structure': [], 'stability': [], 'stats': []}
        maps = {f'{name}_{size}': [] for size in [256, 512] for name in ['disagreement', 'nearest', 'local_confidence', 'displacement']}
        for region_index, region in enumerate(regions):
            features['pairwise'].append(pairwise_global(cls_views[0][region_index], cls_views[-1][region_index]))
            for size, first_tokens, second_tokens in [(256, token_views[0], token_views[-1]), (512, token512[0], token512[-1])]:
                correspondence = token_correspondence(first_tokens[region_index], second_tokens[region_index])
                features[f'token{size}'].append(correspondence['summary'])
                for name in ['disagreement', 'nearest', 'local_confidence', 'displacement']:
                    maps[f'{name}_{size}'].append(correspondence[name])
            patches = [image[region['y']:region['y'] + region['height'], region['x']:region['x'] + region['width']] for image in images]
            features['structure'].append(structure_features(patches[0], patches[-1]))
            features['stats'].append(statistical_features(contexts[0][region_index], contexts[-1][region_index]))
            perceptual_differences = [perceptual.compare(patch, patches[-1], only_lpips=True)['lpips'] for patch in patches[1:-1]]
            features['stability'].append(stability_summary(np.stack(patches[1:]), condition_cls[:, region_index], condition_tokens[:, region_index], perceptual_differences))
            index_rows.append({'image_id': entry['image_id'], 'region_index': region_index, 'scene_id': entry['scene_id'], 'domain': entry['domain']})
        features = {key: np.asarray(value, dtype=np.float32) for key, value in features.items()}
        if not all(np.isfinite(value).all() for value in features.values()):
            raise ValueError('Non-finite features')
        np.savez_compressed(scene_file, **features, **{key: np.asarray(value) for key, value in maps.items()},
                            raw_tokens256=np.stack(token_views, axis=1).astype(np.float16), raw_cls256=np.stack(cls_views, axis=1),
                            raw_tokens512=np.stack(token512, axis=1).astype(np.float16), raw_cls512=np.stack(cls512, axis=1))
        feature_hashes[entry['image_id']] = sha256(scene_file)
        all_features.append(features)
        print(f"Features {scene_number}/55 {entry['image_id']}", flush=True)
    combined = {key: np.concatenate([features[key] for features in all_features]) for key in all_features[0]}
    np.savez_compressed(ROOT / 'features/features.npz', **combined)
    write_csv(ROOT / 'features/index.csv', index_rows)
    write_json(ROOT / 'analysis/feature_extraction.json', {'seconds': time.perf_counter() - started, 'feature_shapes': {key: list(value.shape) for key, value in combined.items()},
               'variants': {key: list(value.shape) for key, value in feature_variants(combined).items()}, 'index_sha256': sha256(ROOT / 'features/index.csv'), 'features_sha256': sha256(ROOT / 'features/features.npz'),
               'scene_hashes': feature_hashes, 'sources': ['LQ', 'restored coefficients 50/100/150/200'], 'gt_in_features': False, 'frozen': True, 'token_grid': [16, 16],
               'tokens_saved': 'L2-normalized float16 tokens, float32 computation', 'dino_views256': ['LQ', '50', '100', '150', '200'], 'dino_views512': ['LQ', '200'],
               'precision': 'float16 autocast encoders; float32 normalized features', 'models': {name: {key: value for key, value in metadata.items() if key != 'path'} for name, metadata in read_json(REPOSITORY / 'csig_phase0/analysis/feature_models.json').items()},
               'stability_definition': 'controlled coefficient sensitivity, not random uncertainty', 'stats': 'unchanged Phase0 statistical_features on 256 contexts; read-only import'})


if __name__ == '__main__':
    main()
