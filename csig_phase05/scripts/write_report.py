from collections import defaultdict

from core import ROOT, config, read_json
from evaluation import read_csv


def number(value, digits=4):
    if value in ['', None]:
        return 'NA'
    return f'{float(value):.{digits}f}'


def mean(rows, field):
    values = [float(row[field]) for row in rows if row[field] not in ['', None]]
    return sum(values) / len(values) if values else None


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + ['| ' + ' | '.join(str(value) for value in row) + ' |' for row in rows])


def main():
    cfg = config()
    entries = read_json(ROOT / 'data/manifests/pairs.json')
    training = read_json(ROOT / 'analysis/training.json')
    audit = read_json(ROOT / 'analysis/artifact_audit.json')
    human = read_json(ROOT / 'analysis/human_analysis.json')
    if human['reviewed_human_regions']:
        raise RuntimeError('This report is the fixed zero-human-label snapshot; review the new annotation analyses before writing an updated scientific report')
    summaries = read_csv(ROOT / 'analysis/verifier_summary.csv')
    fusion = read_csv(ROOT / 'analysis/selective_fusion_summary.csv')
    bootstrap = read_csv(ROOT / 'analysis/paired_bootstrap.csv')
    distributions = read_csv(ROOT / 'analysis/label_distribution.csv')
    primary = training['primary_model']
    primary_rows = [row for row in summaries if row['scope'] == 'mixed_groupcv' and row['target'] == 'psnr_benefit' and row['family'] == 'logistic']
    image_metrics = read_csv(ROOT / 'analysis/image_metrics.csv')
    risk = read_csv(ROOT / 'analysis/risk_coverage_summary.csv')
    ablations = read_csv(ROOT / 'analysis/ablation_comparisons.csv')
    lines = ['''# CSIG Phase 0.5 report: Generate → Verify → Abstain

Date: 2026-09-15. Protocol fixed before reading candidate quality labels. This report separates measured results, proxy targets, assistant metadata, and unavailable independent human truth.

## 中文结论摘要

**HOLD。可计算实验已完成并审计通过，但本阶段尚未完成真实人工标签验收，不进入Phase1。**

- 数据是5张真实CSIG配对+50张合成BSDS train配对，不是55张真实配对。
- 1160个区域只有13个PSNR正收益；主验证器V7宏AUC0.4050，低于LQ-only的0.6647，并且仅5张图可定义AUC。
- 固定50%区域选择的真实融合PSNR仍比LQ低2.0892dB；相对random的损害率差没有显著下降。
- 110份盲标注材料已就绪，但0条由真实人审核。幻觉严重度/类型、人工相关性和人工标签模型仍为NA，助手观察未冒充人。
- 全部任务、来源、原始表、模型、可视化、完整性和下一步待办在本阶段目录；Phase0历史内容未修改，本轮未推送GitHub。

## Executive decision

**HOLD** — do not enter Phase 1 Selective Restoration.

The computational pipeline is complete and auditable, but the candidate-aware primary verifier is not better than the LQ-only comparator on this protocol, actual predicted fusion remains far below LQ, the real domain has only five scenes, and the required independent human hallucination labels are still pending. No human severity, hallucination type, or usefulness result is fabricated below.

## Evidence → finding → path

| Evidence | Finding | Reproducible path |
| --- | --- | --- |
| 55 paired scenes, 1160 non-overlapping native regions | 5 real CSIG scenes + 50 synthetic BSDS train scenes; synthetic is not real coverage | `data/manifests/pairs.json`, `reports/dataset_report.md` |
| 4 coefficient conditions, 220 candidate files | HYPIR is deterministic here; conditions are controlled sensitivity, not random uncertainty | `analysis/hypir_runs.json`, `data/manifests/pairs.json` |
| Primary V7 logistic grouped OOF | Image-macro PSNR-benefit AUC is weak and defined on only 5 images | `analysis/verifier_summary.csv`, `analysis/oof_predictions.csv` |
| 50% actual fusion and fixed random baseline | Predicted selection does not produce a practical restoration gain | `analysis/selective_fusion_summary.csv`, `analysis/paired_bootstrap.csv` |
| 110 fixed blind review rows | Human target is unavailable, not zero | `data/annotations/index.html`, `data/annotations/hallucination_labels.csv`, `analysis/human_analysis.json` |
| Full audit | Source hashes, mappings, fold isolation and fusion provenance pass | `analysis/artifact_audit.json` |

The exact command entry points are `scripts/build_manifest.py`, `scripts/run_hypir.py`, `scripts/prepare_evaluation.py`, `scripts/extract_features.py`, `scripts/train_verifiers.py`, `scripts/selective_restoration.py`, `scripts/visualize.py`, and `scripts/audit_phase05.py`.

## Scope and protocol

''']
    lines.append(f'- Dataset: {len(entries)} scenes ({sum(entry["domain"] == "real_csig" for entry in entries)} real CSIG, {sum(entry["domain"] == "synthetic_bsds" for entry in entries)} synthetic BSDS500 train). The external source is pinned in `data/manifests/source_selection.json`; five recipes are balanced at ten scenes each.')
    lines.append('- Native images are partitioned into non-overlapping 256-pixel regions. Edge regions keep their true geometry for all metrics. DINO contexts are reflected 256/512 crops; the frozen processor resizes to 256 and center-crops 224, so token maps cover only the central 224/448 source pixels and record that limitation.')
    lines.append('- Restoration metrics are PSNR, SSIM, LPIPS, and DISTS with positive gain meaning improvement. GT is used only for labels/evaluation, never features, scaling, model selection, or OOF predictions.')
    lines.append('- Metric resolution matters: PSNR/SSIM use native RGB pixels, SSIM uses Gaussian11/sigma1.5 valid centers; LPIPS/DISTS use native valid regions (minimum observed side65), but full-image perceptual evaluation uniformly caps the long side at1024. Therefore native region LPIPS and capped full-image LPIPS must not be interchanged; resizing may hide high-frequency artifacts.')
    lines.append('- Models are frozen DINOv2-small and CLIP ViT-B/32. V1–V8, C1, C3, and C5 use Logistic(C=0.1); Tiny MLP uses hidden width 32, 30 epochs, seeds 505/506/507. All folds are grouped by scene and scalers fit on training scenes only.')
    lines.append('- Human review: fixed seed 505, two non-overlapping regions per scene, blind LQ/restored/GT montage. Current reviewed count is 0; pending is never converted to severity 0.')
    lines.append('- C5 pairwise+stability completes the original Q5 ablation but was added after the first primary results were observed; it is explicitly supplementary, not a new preregistered winner. The fixed primary remains V7 logistic at50% coverage.')
    lines.append('\n## Data and quality labels\n')
    lines.append(table(['domain', 'regions', 'PSNR benefit', 'four-metric majority'], [[domain, sum(int(row['region_count']) for row in distributions if row['domain'] == domain), sum(int(row['psnr_positive']) for row in distributions if row['domain'] == domain), sum(int(row['majority_positive']) for row in distributions if row['domain'] == domain)] for domain in ['real_csig', 'synthetic_bsds']]))
    lines.append('\nThe raw distribution is deliberately sparse: only 13/1160 regions have positive PSNR gain (8/960 real, 5/200 synthetic), while 28/1160 are positive on the four-metric majority proxy (6/960 real, 22/200 synthetic). These are quality proxies, not hallucination labels. Full-image HYPIR is below LQ in PSNR for 54/55 scenes; the one exception does not establish reliability.')
    lines.append('\n## Verifier results\n')
    rows = []
    for variant in ['C1_lq', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'C5_pair_stability']:
        selected = next(row for row in primary_rows if row['variant'] == variant and row['domain'] == 'all' and row['family'] == 'logistic')
        rows.append([variant, number(selected['roc_auc']), selected['roc_auc_defined_images'], number(selected['pr_auc']), number(selected['brier']), number(selected['ece'])])
    lines.append(table(['variant', 'ROC-AUC macro', 'AUC defined images', 'PR-AUC macro', 'Brier', 'ECE'], rows))
    lines.append(f'\nThe fixed primary is `{primary}`. Its all-domain image-macro ROC-AUC is {number(next(row["roc_auc"] for row in primary_rows if row["variant"] == "V7" and row["domain"] == "all"))}; LQ-only C1 is {number(next(row["roc_auc"] for row in primary_rows if row["variant"] == "C1_lq" and row["domain"] == "all"))}. V7 is therefore not a stable candidate-aware improvement. Real and synthetic AUCs are reported separately in `analysis/verifier_summary.csv`; undefined single-class image counts are retained, not replaced by zero.')
    lines.append('\n### Stability and domain transfer')
    lines.append(table(['domain', 'primary AUC', 'defined images', 'PR-AUC', 'balanced accuracy', 'precision', 'recall', 'F1', 'Brier', 'ECE'], [[row['domain'], number(row['roc_auc']), row['roc_auc_defined_images'], number(row['pr_auc']), number(row['balanced_accuracy']), number(row['precision']), number(row['recall']), number(row['f1']), number(row['brier']), number(row['ece'])] for row in primary_rows if row['variant'] == 'V7']))
    lines.append('- Fixed probability threshold0.5 predicts no positive regions for the primary model; high accuracy and a small Brier score mostly reflect class imbalance. The always-BAD baseline has macro Brier0.023485 and accuracy0.976515, essentially the same. Full seed-by-image results and all three MLP seeds are retained; no best seed is selected.')
    lines.append(f'- C5 pairwise+controlled-stability has all-domain AUC {number(next(row["roc_auc"] for row in primary_rows if row["variant"] == "C5_pair_stability" and row["domain"] == "all"))}; this is not evidence of independent information over V2. Coefficients 50/100/150/200 are controlled coefficient sensitivity under deterministic HYPIR, not a posterior uncertainty estimate.')
    transfer = next(row for row in summaries if row['model_id'] == 'synthetic_to_real_psnr_benefit_V7_logistic_505' and row['domain'] == 'real_csig')
    lines.append(f'- Synthetic-trained → real V7 transfer has AUC {number(transfer["roc_auc"])} over only {transfer["roc_auc_defined_images"]} informative real images; it cannot support deployment.')
    lines.append('\n## Selective restoration\n')
    fusion_rows = {row['mode']: row for row in fusion if row['domain'] == 'all'}
    lines.append(table(['mode', 'full PSNR gain', 'SSIM gain', 'LPIPS gain', 'DISTS gain', 'pixel coverage'], [[mode, number(fusion_rows[mode]['psnr_gain']), number(fusion_rows[mode]['ssim_gain']), number(fusion_rows[mode]['lpips_gain']), number(fusion_rows[mode]['dists_gain']), number(fusion_rows[mode]['pixel_coverage'])] for mode in ['HYPIR', 'predicted', 'random', 'oracle_budget_sse', 'oracle_positive', 'oracle_pixel']]))
    all_bootstrap = {row['metric']: row for row in bootstrap if row['domain'] == 'all'}
    lines.append(f'\nAt the pre-specified 50% point, predicted minus fixed-random damage-rate difference is {number(all_bootstrap["damage_rate"]["mean_difference"])} with bootstrap 95% interval [{number(all_bootstrap["damage_rate"]["lower_95"])}, {number(all_bootstrap["damage_rate"]["upper_95"])}]. Selected-region mean PSNR-gain difference is {number(all_bootstrap["psnr_gain"]["mean_difference"])} dB with interval [{number(all_bootstrap["psnr_gain"]["lower_95"])}, {number(all_bootstrap["psnr_gain"]["upper_95"])}]. This is NOT the full-image fusion PSNR difference in the table. The interval crosses zero. The fixed random selection is seeded per scene and the comparison unit is the source image.')
    lines.append('- The region SSE oracle is only a diagnostic upper bound: even it remains below LQ at the fixed region budget. The pixel oracle reaches positive gain because it may select every true pixel independently; it is not an implementable verifier and uses GT.')
    lines.append('- Coverage is a region-count budget, not an equal-pixel budget. Small BSDS edge regions make the oracle budget pixel coverage only0.2790 versus predicted0.4858 and random0.5099 overall. Real images have equal regions, all exactly0.5. Both region and pixel coverage are explicit in raw tables; the synthetic oracle comparison is not a matched-pixel claim.')
    lines.append(table(['domain', 'predicted-random damage difference', '95% lower', '95% upper', 'images'], [[row['domain'], number(row['mean_difference']), number(row['lower_95']), number(row['upper_95']), row['image_count']] for row in bootstrap if row['metric'] == 'damage_rate']))
    lines.append('- These percentile image bootstraps are descriptive: there are only two informative real AUC images and three informative synthetic AUC images, the source images may share latent scene provenance, and the random baseline is one fixed permutation rather than a Monte Carlo average. No coverage or model was selected from these intervals.')
    lines.append('\n## Human target status\n')
    lines.append(f'- Reviewed human regions: {human["reviewed_human_regions"]}. Q1 PSNR BAD vs human hallucination, Q2 DINO disagreement vs human severity, hallucination type detectability, confusion/agreement, and human-usefulness conjunction are **NA pending review**. See `data/annotations/hallucination_protocol.md` and open `data/annotations/index.html` offline.')
    lines.append('- Assistant visual categories are provenance metadata only. No real night/flare scene is present; synthetic `lowlight_glow` is not night coverage. OCR was not used and no text glyph error rate is claimed.')
    lines.append('\n## 指令第十七节：六项关键统计\n')
    answers = [
        ('Q1', 'PSNR BAD and human hallucination一致吗？', 'NA: no independent human rows; proxy PSNR and hallucination are kept as separate targets.'),
        ('Q2', 'DINO disagreement与human severity正相关吗？', 'NA: human severity is pending; no assistant label is substituted.'),
        ('Q3', 'Global / pairwise / token谁最好？', 'On PSNR-benefit proxy, C1 LQ-only 0.6647, V2 0.5946, V3 0.4543, V7 0.4050; no stable candidate-aware winner.'),
        ('Q4', '只看 LQ 与 LQ+Restored差距？', 'LQ-only is stronger than the candidate-aware primary on the proxy. Candidate/restored inputs do not add reliable validation value here.'),
        ('Q5', 'Stability提供独立信息吗？', 'Not demonstrated: C5 0.5969 is close to V2 and coefficient variation is deterministic sensitivity.'),
        ('Q6', '是否真的能 selective abstain？', 'No practical evidence: predicted fusion at 50% has −2.0892 dB full-image PSNR gain vs LQ; predicted minus random damage is not significant.'),
        ('Q7', '恢复候选在本协议中是否值得自动应用？', 'No. Full HYPIR is below LQ on 54/55 scenes and actual candidate selection still loses to LQ.'),
        ('Q8', 'Oracle upper bound够高吗？', 'Pixel oracle is high but GT-dependent; fixed-region SSE oracle remains negative, so implementable region selection has limited margin.'),
        ('Q9', '是否进入 Phase 1？', 'No; first add genuinely independent real paired scenes and completed human review, then rerun the fixed protocol.'),
        ('Q10', '最终结论？', '**HOLD**, not GO and not STOP: the pipeline is healthy, but evidence is insufficient and the candidate-aware gate fails the practical bar.'),
    ]
    lines.append(table(['question', 'answer', 'evidence-based response'], answers[:6]))
    lines.append('\n## 最终必须回答的 Q1–Q10\n')
    final_answers = [
        ['Q1', '是否值得继续深挖 candidate-aware signal？', '仅值得先补数据与人工标签，不支持马上训练 Gate；当前固定主模型不优于 LQ-only。'],
        ['Q2', 'DINO pairwise/token 比 Phase 0 pooled feature 更可靠吗？', '未证实；V2 比同协议 V1 提高约 0.0512 AUC，但仅5个有效图像、区间跨0，token更差。旧 Phase0 重叠 patch AUC不可直接比较。'],
        ['Q3', 'PSNR gain 是合适的 hallucination proxy 吗？', '不能据本结果认定。PSNR 与感知质量方向明显不同，人工真值缺失，不能将 PSNR BAD 等同幻觉。'],
        ['Q4', '哪些 hallucination 类型最容易检测？', 'NA：没有已审核人工类型标签；不得用助手观察替代分类检验。'],
        ['Q5', '哪些类型最难？', 'NA：同样需要真实人工类型标签，当前不做无证据排名。'],
        ['Q6', 'Stability 提供额外信息吗？', '未稳定证实；C5−V2 的宏 AUC差很小，V7−V6也无可靠收益。这里只是确定性系数敏感性。'],
        ['Q7', 'Selective restoration 是否真实降低 damage？', '相对全量恢复，少用候选会减少暴露；但在固定50%区域coverage下，相对random的damage差区间含0，所有55张 predicted融合PSNR仍低于LQ。'],
        ['Q8', 'Oracle upper bound 足够高吗？', '区域positive oracle宏增益只有0.0189dB，固定半数区域oracle仍为负；逐像素GT oracle约1.8363dB，但不是可部署上界达到证据。'],
        ['Q9', '是否值得进入 Phase 1？', '否。真实域只有5图、PSNR AUC有效图只有5张、人工审核0，且主验证器未稳定通过。'],
        ['Q10', '是否仍应 HOLD？', '**HOLD**。计算流程完成不等于科学假设成立；不启动恢复器/LoRA/MoE/专家/Gate训练。'],
    ]
    lines.append(table(['编号', '问题', '回答'], final_answers))
    lines.append('\n## Reproducibility audit\n')
    lines.append(f'- Audit status: `{audit["status"]}`. Verified {audit["native_regions_verified"]} regions, {audit["candidates_verified"]} controlled candidates, {audit["verifier_checkpoints_verified"]} checkpoints, and {audit["selective_images_verified"]} selective image/mask artifacts. Maximum pixel gain reproduction error {audit["max_pixel_gain_reproduction_error"]}; perceptual reproduction error {audit["max_perceptual_reproduction_error"]}; OOF reconstruction error {audit["max_oof_probability_error"]}.')
    lines.append('- Phase 0 immutable snapshot remains unchanged; test data used: 0; restoration models trained: 0; human annotations fabricated: 0. Large local arrays/checkpoints remain under `csig_phase05`; this phase is not automatically pushed to GitHub.')
    lines.append('- All figures are under `reports/visualization/`. Human-dependent figures explicitly show pending status rather than invented severity plots. Raw metric, target, feature, fold, risk, fusion, oracle, and hash tables are under `analysis/`.')
    lines.append('\n## Next authorized work only\n')
    lines.append('1. Have at least one independent human reviewer complete the fixed 110-row blind set and ideally a second reviewer repeat it.\n2. Import only with `python scripts/annotations.py --import-csv <export.csv>`, then rerun `scripts/analyze_results.py`; do not change the sample set or tune thresholds.\n3. Add independently paired real scenes covering text, night/flare, people, and optical degradations before any Phase 1 decision.\n4. Do not train restoration models or publish new artifacts until separately authorized.')
    (ROOT / 'reports/phase05_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (ROOT / 'reports/done_checklist.md').write_text('''# Phase 0.5 done checklist\n\n- [x] 55-scene manifest with real/synthetic provenance and hashes\n- [x] Official frozen HYPIR candidate and four controlled coefficients\n- [x] PSNR/SSIM/LPIPS/DISTS native region and full-image tables\n- [x] Blind annotation protocol, pending CSV, offline interface, and importer\n- [x] DINO global/pairwise/token 256+512, structure, statistics, controlled sensitivity\n- [x] V1–V8 plus C1/C3/C5 grouped Logistic and Tiny MLP OOF checkpoints\n- [x] Per-image macro metrics, undefined-class counts, calibration, risk–coverage\n- [x] Actual LQ/HYPIR/predicted/random/oracle fusion outputs and upper bounds\n- [x] All computable plots and scene grouping metadata; missing-human panels explicitly marked NA\n- [x] Full mapping/hash/leakage/reconstruction audit PASS\n- [ ] Independent human hallucination labels (external input pending)\n- [ ] Human severity/type/performance plots and analyses (NA panels are not completed human results)\n- [ ] Phase 1 authorization (blocked by HOLD)\n''', encoding='utf-8')


if __name__ == '__main__':
    main()
