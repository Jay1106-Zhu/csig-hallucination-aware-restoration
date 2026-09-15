import time

from core import ROOT, config, grouped_splits, sha256, weighted_scene_samples, write_json
import numpy as np
from scipy.special import expit

from evaluation import binary_metrics, fit_normalizer, read_csv, write_csv
from extract_features import feature_variants


def fit_verifier(features, labels, groups, kind, seed, cfg):
    mean, scale = fit_normalizer(features, groups)
    normalized = ((features - mean) / scale).astype(np.float32)
    weights = weighted_scene_samples(groups).astype(np.float32)
    checkpoint = {'mean': mean, 'scale': scale, 'kind': kind, 'seed': seed}
    if len(np.unique(labels)) == 1:
        return {**checkpoint, 'kind': 'constant', 'probability': float(labels[0]), 'history': np.array([])}
    if kind == 'logistic':
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=cfg['logistic_C'], max_iter=2000, random_state=seed)
        model.fit(normalized, labels, sample_weight=weights)
        checkpoint.update(weight=model.coef_[0], bias=model.intercept_[0], history=np.array([]), iterations=int(model.n_iter_[0]))
        if model.n_iter_[0] >= model.max_iter:
            raise RuntimeError('Logistic regression failed to converge')
    elif kind == 'mlp':
        import torch

        torch.set_num_threads(2)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        model = torch.nn.Sequential(torch.nn.Linear(normalized.shape[1], cfg['mlp_hidden_dim']), torch.nn.ReLU(), torch.nn.Linear(cfg['mlp_hidden_dim'], 1))
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['mlp_learning_rate'])
        features_tensor, labels_tensor, weights_tensor = torch.from_numpy(normalized), torch.tensor(labels, dtype=torch.float32), torch.from_numpy(weights)
        history = []
        for epoch in range(cfg['mlp_epochs']):
            optimizer.zero_grad()
            logits = model(features_tensor).flatten()
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, labels_tensor, reduction='none') * weights_tensor).mean()
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
        checkpoint.update(weight1=model[0].weight.detach().numpy(), bias1=model[0].bias.detach().numpy(),
                          weight2=model[2].weight.detach().numpy(), bias2=model[2].bias.detach().numpy(), history=np.asarray(history))
    else:
        raise ValueError('Unknown model family')
    return checkpoint


def predict_checkpoint(features, checkpoint):
    normalized = ((features - checkpoint['mean']) / checkpoint['scale']).astype(np.float32)
    kind = str(checkpoint['kind'])
    if kind == 'constant':
        return np.full(len(features), float(checkpoint['probability']))
    if kind == 'logistic':
        return expit(normalized @ checkpoint['weight'] + checkpoint['bias'])
    hidden = np.maximum(normalized @ checkpoint['weight1'].T + checkpoint['bias1'], 0)
    return expit((hidden @ checkpoint['weight2'].T + checkpoint['bias2']).reshape(-1))


def macro_summary(rows):
    fields = ['roc_auc', 'pr_auc', 'balanced_accuracy', 'precision', 'recall', 'f1', 'accuracy', 'brier', 'ece']
    result = {'image_count': len(rows), 'region_count': sum(row['region_count'] for row in rows), 'positive_count': sum(row['positive_count'] for row in rows)}
    for field in fields:
        values = [row[field] for row in rows if row.get(field) is not None]
        result[field] = float(np.mean(values)) if values else None
        result[field + '_defined_images'] = len(values)
    return result


def evaluate_run(variant, features, labels, index, splits, target, scope, kind, seed, cfg, predictions, per_image, summaries, fold_records):
    model_id = f'{scope}_{target}_{variant}_{kind}_{seed}'
    probabilities = np.full(len(index), np.nan)
    fold_ids = np.full(len(index), -1)
    groups = np.array([row['scene_id'] for row in index])
    for fold_number, (train, validation) in enumerate(splits):
        if set(groups[train]) & set(groups[validation]):
            raise ValueError('Scene leakage')
        checkpoint = fit_verifier(features[train], labels[train], groups[train], kind, seed, cfg)
        checkpoint.update(train_indices=train, validation_indices=validation)
        output = ROOT / 'models/verifiers' / f'{model_id}_fold{fold_number}.npz'
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **checkpoint)
        probabilities[validation] = predict_checkpoint(features[validation], checkpoint)
        fold_ids[validation] = fold_number
        fold_records.append({'model_id': model_id, 'fold': fold_number, 'train_scenes': sorted(set(groups[train])), 'validation_scenes': sorted(set(groups[validation])),
                             'train_region_count': len(train), 'validation_region_count': len(validation), 'train_positive_count': int(labels[train].sum()),
                             'checkpoint': output.relative_to(ROOT).as_posix(), 'sha256': sha256(output), 'constant_fallback': str(checkpoint['kind']) == 'constant'})
    run_rows = []
    for image_id in dict.fromkeys(row['image_id'] for row in index):
        selected = np.array([row['image_id'] == image_id for row in index]) & np.isfinite(probabilities)
        if not selected.any():
            continue
        domain = index[np.flatnonzero(selected)[0]]['domain']
        row = {'model_id': model_id, 'scope': scope, 'target': target, 'variant': variant, 'family': kind, 'seed': seed, 'image_id': image_id, 'domain': domain,
               **binary_metrics(labels[selected], probabilities[selected], cfg['decision_threshold'])}
        per_image.append(row)
        run_rows.append(row)
    for domain in ['all', 'real_csig', 'synthetic_bsds']:
        selected = [row for row in run_rows if domain == 'all' or row['domain'] == domain]
        if selected:
            summaries.append({'model_id': model_id, 'scope': scope, 'target': target, 'variant': variant, 'family': kind, 'seed': seed, 'domain': domain, **macro_summary(selected)})
    for row_number in np.flatnonzero(np.isfinite(probabilities)):
        predictions.append({'model_id': model_id, 'scope': scope, 'target': target, 'variant': variant, 'family': kind, 'seed': seed, **index[row_number],
                            'row_index': row_number, 'fold': int(fold_ids[row_number]), 'label': int(labels[row_number]), 'probability': float(probabilities[row_number])})
    print(model_id, 'completed', flush=True)


def main():
    cfg = config()
    started = time.perf_counter()
    index = read_csv(ROOT / 'features/index.csv')
    metrics = read_csv(ROOT / 'analysis/restoration_gain.csv')
    if [(row['image_id'], row['region_index']) for row in index] != [(row['image_id'], row['region_index']) for row in metrics]:
        raise ValueError('Feature and label ordering mismatch')
    feature_blocks = dict(np.load(ROOT / 'features/features.npz'))
    variants = feature_variants(feature_blocks)
    groups = np.array([row['scene_id'] for row in index])
    splits = list(grouped_splits(groups, cfg['group_folds']))
    predictions, per_image, summaries, fold_records = [], [], [], []
    for target in ['psnr_benefit', 'four_metric_majority']:
        labels = np.array([int(row[target]) for row in metrics])
        for variant, features in variants.items():
            if target != 'psnr_benefit' and variant != 'V7':
                continue
            for kind in ['logistic', 'mlp']:
                for seed in ([cfg['seed']] if kind == 'logistic' else cfg['verifier_seeds']):
                    evaluate_run(variant, features, labels, index, splits, target, 'mixed_groupcv', kind, seed, cfg, predictions, per_image, summaries, fold_records)
    labels = np.array([int(row['psnr_benefit']) for row in metrics])
    synthetic = np.array([number for number, row in enumerate(index) if row['domain'] == 'synthetic_bsds'])
    real = np.array([number for number, row in enumerate(index) if row['domain'] == 'real_csig'])
    synthetic_splits = [(synthetic[train], synthetic[validation]) for train, validation in grouped_splits(groups[synthetic], cfg['group_folds'])]
    for scope, domain_splits in [('synthetic_only', synthetic_splits), ('synthetic_to_real', [(synthetic, real)])]:
        for variant in ['V1', 'V7', 'V8', 'C1_lq']:
            evaluate_run(variant, variants[variant], labels, index, domain_splits, 'psnr_benefit', scope, 'logistic', cfg['seed'], cfg, predictions, per_image, summaries, fold_records)
    for baseline, probability in [('always_bad', 0.), ('always_good', 1.)]:
        baseline_rows = []
        for image_id in dict.fromkeys(row['image_id'] for row in index):
            selected = np.array([row['image_id'] == image_id for row in index])
            row = {'model_id': baseline, 'scope': 'baseline', 'target': 'psnr_benefit', 'variant': baseline, 'family': 'constant', 'seed': cfg['seed'],
                   'image_id': image_id, 'domain': index[np.flatnonzero(selected)[0]]['domain'], **binary_metrics(labels[selected], np.full(selected.sum(), probability))}
            baseline_rows.append(row)
            per_image.append(row)
        for domain in ['all', 'real_csig', 'synthetic_bsds']:
            summaries.append({'model_id': baseline, 'scope': 'baseline', 'target': 'psnr_benefit', 'variant': baseline, 'family': 'constant', 'seed': cfg['seed'], 'domain': domain,
                              **macro_summary([row for row in baseline_rows if domain == 'all' or row['domain'] == domain])})
    write_csv(ROOT / 'analysis/oof_predictions.csv', predictions)
    write_csv(ROOT / 'analysis/verifier_per_image.csv', per_image)
    write_csv(ROOT / 'analysis/verifier_summary.csv', summaries)
    write_csv(ROOT / 'analysis/verifier_metrics.csv', summaries)
    write_json(ROOT / 'analysis/folds.json', fold_records)
    write_json(ROOT / 'analysis/training.json', {'seconds': time.perf_counter() - started, 'checkpoint_count': len(fold_records), 'run_count': len(set(row['model_id'] for row in predictions)),
               'primary_model': 'mixed_groupcv_psnr_benefit_V7_logistic_505', 'seed_selection': 'none; logistic deterministic, all MLP seeds reported',
               'features_sha256': sha256(ROOT / 'features/features.npz'), 'label_sha256': sha256(ROOT / 'analysis/restoration_gain.csv'),
               'scaling': 'training-only scene-balanced mean and variance', 'weighting': 'equal total loss weight per scene; no outcome-based class reweighting',
               'target_limit': 'quality proxies only; human hallucination and usefulness conjunction remain pending', 'test_data_used': False})


if __name__ == '__main__':
    main()
