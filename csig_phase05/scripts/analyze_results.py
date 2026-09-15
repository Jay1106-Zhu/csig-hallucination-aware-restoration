from collections import defaultdict

from core import ROOT, config, read_json, write_json
import numpy as np
from scipy.stats import spearmanr

from annotations import validate_review
from evaluation import binary_metrics, read_csv, write_csv
from extract_features import feature_variants
from selective_restoration import bootstrap_difference

CATEGORIES = {
    'case1': 'text', 'case2': 'text', 'case3': 'animal', 'case4': 'vegetation', 'case5': 'architecture',
    '100098': 'animal', '106025': 'animal', '112082': 'animal', '138078': 'landscape', '140055': 'vegetation',
    '147021': 'people', '153077': 'people', '159029': 'animal', '159045': 'animal', '159091': 'animal',
    '164074': 'animal', '166081': 'architecture', '176039': 'landscape', '183055': 'animal', '187029': 'people',
    '188063': 'people', '198023': 'people', '198054': 'people', '20008': 'landscape', '216041': 'animal',
    '225017': 'vegetation', '23025': 'people', '23080': 'people', '231015': 'architecture', '236017': 'animal',
    '245051': 'people', '246016': 'people', '249061': 'landscape', '25098': 'vegetation', '253036': 'animal',
    '254054': 'architecture', '27059': 'landscape', '271008': 'people', '274007': 'architecture', '286092': 'people',
    '309004': 'animal', '311081': 'architecture', '314016': 'animal', '323016': 'people', '368078': 'architecture',
    '374020': 'architecture', '385028': 'architecture', '41004': 'animal', '45077': 'animal', '46076': 'vegetation',
    '59078': 'architecture', '61086': 'architecture', '76002': 'architecture', '80099': 'animal', '92059': 'landscape'}


def numeric(row):
    return {key: None if value == '' else float(value) for key, value in row.items() if key in ['roc_auc', 'pr_auc', 'brier', 'ece', 'positive_count', 'region_count']}


def main():
    cfg = config()
    feature_metadata = read_json(ROOT / 'analysis/feature_extraction.json')
    feature_metadata['variants'] = {name: list(values.shape) for name, values in feature_variants(dict(np.load(ROOT / 'features/features.npz'))).items()}
    feature_metadata['token_field_of_view_original_pixels'] = {'256_context': 224, '512_context': 448}
    feature_metadata['processor_geometry'] = 'resize short edge to 256 then center crop 224; token heatmaps cover central 7/8, not entire context'
    write_json(ROOT / 'analysis/feature_extraction.json', feature_metadata)
    primary = read_json(ROOT / 'analysis/training.json')['primary_model']
    categories = {entry['image_id']: {'category': CATEGORIES[entry['image_id'].removeprefix('bsds_')], 'provenance': 'assistant visual review of source contact sheets; not human hallucination labels'} for entry in read_json(ROOT / 'data/manifests/pairs.json')}
    write_json(ROOT / 'data/manifests/scene_categories.json', categories)
    per_image = read_csv(ROOT / 'analysis/verifier_per_image.csv')
    comparisons = []
    for first, second in [('V2', 'V1'), ('V3', 'V1'), ('V2', 'C1_lq'), ('V7', 'C1_lq'), ('V7', 'V6'), ('C5_pair_stability', 'V2'), ('V7', 'V8'), ('C3_token512', 'C3_token256')]:
        for domain in ['all', 'real_csig', 'synthetic_bsds']:
            selected = [row for row in per_image if row['scope'] == 'mixed_groupcv' and row['target'] == 'psnr_benefit' and row['family'] == 'logistic' and (domain == 'all' or row['domain'] == domain)]
            first_rows = {row['image_id']: row for row in selected if row['variant'] == first}
            second_rows = {row['image_id']: row for row in selected if row['variant'] == second}
            for metric in ['roc_auc', 'pr_auc', 'brier']:
                images = [image_id for image_id in first_rows if image_id in second_rows and first_rows[image_id][metric] != '' and second_rows[image_id][metric] != '']
                if images:
                    comparisons.append({'first': first, 'second': second, 'domain': domain, 'metric': metric,
                                       **bootstrap_difference([float(first_rows[image_id][metric]) for image_id in images], [float(second_rows[image_id][metric]) for image_id in images], cfg['seed'], cfg['bootstrap_scenes'])})
    write_csv(ROOT / 'analysis/ablation_comparisons.csv', comparisons)
    grouped = defaultdict(list)
    for row in per_image:
        if row['model_id'] == primary:
            grouped[(row['domain'], categories[row['image_id']]['category'])].append(row)
    category_rows = []
    for (domain, category), rows in grouped.items():
        metrics = [numeric(row) for row in rows]
        category_rows.append({'domain': domain, 'category': category, 'image_count': len(rows), 'roc_auc_defined_images': sum(row['roc_auc'] is not None for row in metrics),
                              **{key: float(np.mean([row[key] for row in metrics if row[key] is not None])) if any(row[key] is not None for row in metrics) else None for key in ['roc_auc', 'pr_auc', 'brier', 'ece']}})
    write_csv(ROOT / 'analysis/scene_group_metrics.csv', category_rows)
    metrics = read_csv(ROOT / 'analysis/restoration_gain.csv')
    index = {(row['image_id'], row['region_index']): position for position, row in enumerate(metrics)}
    review_path = ROOT / 'data/annotations/reviewed_labels.csv'
    reviews = [row for row in read_csv(review_path) if validate_review(row)] if review_path.exists() else []
    lookup = {(row['image_id'], row['region_index']): row for row in reviews}
    targets = []
    for row in metrics:
        review = lookup.get((row['image_id'], row['region_index']))
        severity = int(review['severity']) if review else None
        targets.append({'image_id': row['image_id'], 'region_index': row['region_index'], 'psnr_benefit': row['psnr_benefit'], 'four_metric_majority': row['four_metric_majority'],
                        'human_severity': severity, 'human_hallucination_present': int(severity > 0) if review else None, 'human_hallucination_clear': int(severity >= 2) if review else None,
                        'useful_psnr_and_human_low': int(int(row['psnr_benefit']) and severity <= 1) if review else None, 'annotation_status': 'reviewed' if review else 'pending'})
    write_csv(ROOT / 'analysis/trust_targets.csv', targets)
    human_summary = {'reviewed_human_regions': len(reviews), 'status': 'pending independent human annotations' if not reviews else 'descriptive human analysis',
                     'psnr_bad_vs_hallucination_confusion': None, 'psnr_vs_human_spearman': None, 'token_vs_human_spearman': None, 'stability_vs_human_spearman': None,
                     'per_type_detectability': None, 'hallucination_trained_verifier': 'not trained; independent reviewed scene coverage required'}
    if reviews:
        features = np.load(ROOT / 'features/features.npz')
        selected = np.array([index[(row['image_id'], row['region_index'])] for row in reviews])
        severity = np.array([int(row['severity']) for row in reviews])
        gain = np.array([float(metrics[position]['psnr_gain']) for position in selected])
        confusion = [[int(np.sum(((gain <= 0) == bool(bad)) & ((severity > 0) == bool(hallucination)))) for hallucination in [0, 1]] for bad in [0, 1]]
        human_summary['psnr_bad_vs_hallucination_confusion'] = confusion
        for key, values in [('psnr', gain), ('token', features['token256'][selected, 0]), ('stability', features['stability'][selected, 0])]:
            if len(np.unique(severity)) > 1 and len(np.unique(values)) > 1:
                human_summary[key + '_vs_human_spearman'] = float(spearmanr(values, severity).statistic)
        human_summary['caution'] = 'Descriptive correlations only; regions are clustered by scene, no patch-iid significance claim.'
    write_json(ROOT / 'analysis/human_analysis.json', human_summary)
    distributions = []
    for entry in read_json(ROOT / 'data/manifests/pairs.json'):
        selected = [row for row in metrics if row['image_id'] == entry['image_id']]
        distributions.append({'image_id': entry['image_id'], 'domain': entry['domain'], 'category': categories[entry['image_id']]['category'], 'region_count': len(selected),
                              'psnr_positive': sum(int(row['psnr_benefit']) for row in selected), 'majority_positive': sum(int(row['four_metric_majority']) for row in selected),
                              **{key: float(np.mean([float(row[key]) for row in selected])) for key in ['psnr_gain', 'ssim_gain', 'lpips_gain', 'dists_gain']}})
    write_csv(ROOT / 'analysis/label_distribution.csv', distributions)


if __name__ == '__main__':
    main()
