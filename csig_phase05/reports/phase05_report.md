# CSIG Phase 0.5 report: Generate → Verify → Abstain

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


- Dataset: 55 scenes (5 real CSIG, 50 synthetic BSDS500 train). The external source is pinned in `data/manifests/source_selection.json`; five recipes are balanced at ten scenes each.
- Native images are partitioned into non-overlapping 256-pixel regions. Edge regions keep their true geometry for all metrics. DINO contexts are reflected 256/512 crops; the frozen processor resizes to 256 and center-crops 224, so token maps cover only the central 224/448 source pixels and record that limitation.
- Restoration metrics are PSNR, SSIM, LPIPS, and DISTS with positive gain meaning improvement. GT is used only for labels/evaluation, never features, scaling, model selection, or OOF predictions.
- Metric resolution matters: PSNR/SSIM use native RGB pixels, SSIM uses Gaussian11/sigma1.5 valid centers; LPIPS/DISTS use native valid regions (minimum observed side65), but full-image perceptual evaluation uniformly caps the long side at1024. Therefore native region LPIPS and capped full-image LPIPS must not be interchanged; resizing may hide high-frequency artifacts.
- Models are frozen DINOv2-small and CLIP ViT-B/32. V1–V8, C1, C3, and C5 use Logistic(C=0.1); Tiny MLP uses hidden width 32, 30 epochs, seeds 505/506/507. All folds are grouped by scene and scalers fit on training scenes only.
- Human review: fixed seed 505, two non-overlapping regions per scene, blind LQ/restored/GT montage. Current reviewed count is 0; pending is never converted to severity 0.
- C5 pairwise+stability completes the original Q5 ablation but was added after the first primary results were observed; it is explicitly supplementary, not a new preregistered winner. The fixed primary remains V7 logistic at50% coverage.

## Data and quality labels

| domain | regions | PSNR benefit | four-metric majority |
| --- | --- | --- | --- |
| real_csig | 960 | 8 | 6 |
| synthetic_bsds | 200 | 5 | 22 |

The raw distribution is deliberately sparse: only 13/1160 regions have positive PSNR gain (8/960 real, 5/200 synthetic), while 28/1160 are positive on the four-metric majority proxy (6/960 real, 22/200 synthetic). These are quality proxies, not hallucination labels. Full-image HYPIR is below LQ in PSNR for 54/55 scenes; the one exception does not establish reliability.

## Verifier results

| variant | ROC-AUC macro | AUC defined images | PR-AUC macro | Brier | ECE |
| --- | --- | --- | --- | --- | --- |
| C1_lq | 0.6647 | 5 | 0.4076 | 0.0235 | 0.0261 |
| V1 | 0.5434 | 5 | 0.3429 | 0.0235 | 0.0251 |
| V2 | 0.5946 | 5 | 0.3646 | 0.0235 | 0.0242 |
| V3 | 0.4543 | 5 | 0.3425 | 0.0263 | 0.0464 |
| V4 | 0.3742 | 5 | 0.2851 | 0.0258 | 0.0438 |
| V5 | 0.4362 | 5 | 0.3466 | 0.0243 | 0.0473 |
| V6 | 0.4067 | 5 | 0.2770 | 0.0235 | 0.0245 |
| V7 | 0.4050 | 5 | 0.2769 | 0.0235 | 0.0245 |
| V8 | 0.4731 | 5 | 0.3199 | 0.0235 | 0.0247 |
| C5_pair_stability | 0.5969 | 5 | 0.3650 | 0.0235 | 0.0242 |

The fixed primary is `mixed_groupcv_psnr_benefit_V7_logistic_505`. Its all-domain image-macro ROC-AUC is 0.4050; LQ-only C1 is 0.6647. V7 is therefore not a stable candidate-aware improvement. Real and synthetic AUCs are reported separately in `analysis/verifier_summary.csv`; undefined single-class image counts are retained, not replaced by zero.

### Stability and domain transfer
| domain | primary AUC | defined images | PR-AUC | balanced accuracy | precision | recall | F1 | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 0.4050 | 5 | 0.2769 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0235 | 0.0245 |
| real_csig | 0.6791 | 2 | 0.0396 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0083 | 0.0084 |
| synthetic_bsds | 0.2222 | 3 | 0.4352 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0250 | 0.0261 |
- Fixed probability threshold0.5 predicts no positive regions for the primary model; high accuracy and a small Brier score mostly reflect class imbalance. The always-BAD baseline has macro Brier0.023485 and accuracy0.976515, essentially the same. Full seed-by-image results and all three MLP seeds are retained; no best seed is selected.
- C5 pairwise+controlled-stability has all-domain AUC 0.5969; this is not evidence of independent information over V2. Coefficients 50/100/150/200 are controlled coefficient sensitivity under deterministic HYPIR, not a posterior uncertainty estimate.
- Synthetic-trained → real V7 transfer has AUC 0.4605 over only 2 informative real images; it cannot support deployment.

## Selective restoration

| mode | full PSNR gain | SSIM gain | LPIPS gain | DISTS gain | pixel coverage |
| --- | --- | --- | --- | --- | --- |
| HYPIR | -3.2812 | -0.0952 | 0.1200 | 0.0457 | 1.0000 |
| predicted | -2.0892 | -0.0483 | 0.0555 | 0.0472 | 0.4858 |
| random | -2.0897 | -0.0529 | 0.0471 | 0.0482 | 0.5099 |
| oracle_budget_sse | -1.0188 | -0.0184 | 0.0266 | 0.0308 | 0.2790 |
| oracle_positive | 0.0189 | 0.0038 | 0.0059 | 0.0028 | 0.0209 |
| oracle_pixel | 1.8363 | 0.1014 | 0.1813 | 0.0607 | 0.3557 |

At the pre-specified 50% point, predicted minus fixed-random damage-rate difference is 0.0087 with bootstrap 95% interval [-0.0009, 0.0273]. Selected-region mean PSNR-gain difference is -0.1796 dB with interval [-0.4140, 0.0199]. This is NOT the full-image fusion PSNR difference in the table. The interval crosses zero. The fixed random selection is seeded per scene and the comparison unit is the source image.
- The region SSE oracle is only a diagnostic upper bound: even it remains below LQ at the fixed region budget. The pixel oracle reaches positive gain because it may select every true pixel independently; it is not an implementable verifier and uses GT.
- Coverage is a region-count budget, not an equal-pixel budget. Small BSDS edge regions make the oracle budget pixel coverage only0.2790 versus predicted0.4858 and random0.5099 overall. Real images have equal regions, all exactly0.5. Both region and pixel coverage are explicit in raw tables; the synthetic oracle comparison is not a matched-pixel claim.
| domain | predicted-random damage difference | 95% lower | 95% upper | images |
| --- | --- | --- | --- | --- |
| all | 0.0087 | -0.0009 | 0.0273 | 55 |
| real_csig | -0.0042 | -0.0083 | 0.0000 | 5 |
| synthetic_bsds | 0.0100 | 0.0000 | 0.0300 | 50 |
- These percentile image bootstraps are descriptive: there are only two informative real AUC images and three informative synthetic AUC images, the source images may share latent scene provenance, and the random baseline is one fixed permutation rather than a Monte Carlo average. No coverage or model was selected from these intervals.

## Human target status

- Reviewed human regions: 0. Q1 PSNR BAD vs human hallucination, Q2 DINO disagreement vs human severity, hallucination type detectability, confusion/agreement, and human-usefulness conjunction are **NA pending review**. See `data/annotations/hallucination_protocol.md` and open `data/annotations/index.html` offline.
- Assistant visual categories are provenance metadata only. No real night/flare scene is present; synthetic `lowlight_glow` is not night coverage. OCR was not used and no text glyph error rate is claimed.

## 指令第十七节：六项关键统计

| question | answer | evidence-based response |
| --- | --- | --- |
| Q1 | PSNR BAD and human hallucination一致吗？ | NA: no independent human rows; proxy PSNR and hallucination are kept as separate targets. |
| Q2 | DINO disagreement与human severity正相关吗？ | NA: human severity is pending; no assistant label is substituted. |
| Q3 | Global / pairwise / token谁最好？ | On PSNR-benefit proxy, C1 LQ-only 0.6647, V2 0.5946, V3 0.4543, V7 0.4050; no stable candidate-aware winner. |
| Q4 | 只看 LQ 与 LQ+Restored差距？ | LQ-only is stronger than the candidate-aware primary on the proxy. Candidate/restored inputs do not add reliable validation value here. |
| Q5 | Stability提供独立信息吗？ | Not demonstrated: C5 0.5969 is close to V2 and coefficient variation is deterministic sensitivity. |
| Q6 | 是否真的能 selective abstain？ | No practical evidence: predicted fusion at 50% has −2.0892 dB full-image PSNR gain vs LQ; predicted minus random damage is not significant. |

## 最终必须回答的 Q1–Q10

| 编号 | 问题 | 回答 |
| --- | --- | --- |
| Q1 | 是否值得继续深挖 candidate-aware signal？ | 仅值得先补数据与人工标签，不支持马上训练 Gate；当前固定主模型不优于 LQ-only。 |
| Q2 | DINO pairwise/token 比 Phase 0 pooled feature 更可靠吗？ | 未证实；V2 比同协议 V1 提高约 0.0512 AUC，但仅5个有效图像、区间跨0，token更差。旧 Phase0 重叠 patch AUC不可直接比较。 |
| Q3 | PSNR gain 是合适的 hallucination proxy 吗？ | 不能据本结果认定。PSNR 与感知质量方向明显不同，人工真值缺失，不能将 PSNR BAD 等同幻觉。 |
| Q4 | 哪些 hallucination 类型最容易检测？ | NA：没有已审核人工类型标签；不得用助手观察替代分类检验。 |
| Q5 | 哪些类型最难？ | NA：同样需要真实人工类型标签，当前不做无证据排名。 |
| Q6 | Stability 提供额外信息吗？ | 未稳定证实；C5−V2 的宏 AUC差很小，V7−V6也无可靠收益。这里只是确定性系数敏感性。 |
| Q7 | Selective restoration 是否真实降低 damage？ | 相对全量恢复，少用候选会减少暴露；但在固定50%区域coverage下，相对random的damage差区间含0，所有55张 predicted融合PSNR仍低于LQ。 |
| Q8 | Oracle upper bound 足够高吗？ | 区域positive oracle宏增益只有0.0189dB，固定半数区域oracle仍为负；逐像素GT oracle约1.8363dB，但不是可部署上界达到证据。 |
| Q9 | 是否值得进入 Phase 1？ | 否。真实域只有5图、PSNR AUC有效图只有5张、人工审核0，且主验证器未稳定通过。 |
| Q10 | 是否仍应 HOLD？ | **HOLD**。计算流程完成不等于科学假设成立；不启动恢复器/LoRA/MoE/专家/Gate训练。 |

## Reproducibility audit

- Audit status: `PASS`. Verified 1160 regions, 220 controlled candidates, 324 checkpoints, and 275 selective image/mask artifacts. Maximum pixel gain reproduction error 0.0; perceptual reproduction error 0.0; OOF reconstruction error 0.0.
- Phase 0 immutable snapshot remains unchanged; test data used: 0; restoration models trained: 0; human annotations fabricated: 0. Large local arrays/checkpoints remain under `csig_phase05`; this phase is not automatically pushed to GitHub.
- All figures are under `reports/visualization/`. Human-dependent figures explicitly show pending status rather than invented severity plots. Raw metric, target, feature, fold, risk, fusion, oracle, and hash tables are under `analysis/`.

## Next authorized work only

1. Have at least one independent human reviewer complete the fixed 110-row blind set and ideally a second reviewer repeat it.
2. Import only with `python scripts/annotations.py --import-csv <export.csv>`, then rerun `scripts/analyze_results.py`; do not change the sample set or tune thresholds.
3. Add independently paired real scenes covering text, night/flare, people, and optical degradations before any Phase 1 decision.
4. Do not train restoration models or publish new artifacts until separately authorized.
