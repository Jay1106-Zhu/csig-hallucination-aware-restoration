# Codex Prompt：CSIG Phase 0.5 Hallucination Verification 实验实施指南

你现在接手的仓库是：

`https://github.com/Jay1106-Zhu/csig-hallucination-aware-restoration`

目标不是直接训练新的恢复模型、LoRA、MoE 或专家网络，而是基于已经完成的 Phase 0 结果，继续完成 **Phase 0.5：Hallucination Verification / Reliability Target Redesign**。

---

## 一、背景与当前已知结论

请先阅读仓库中以下文件，理解现有实验，不要重复 Phase 0 已完成工作：

- `README.md`
- `CSIG_Hallucination_Aware_Restoration_Strategy.md`
- `CSIG_Phase0_Experiment_Guide.md`
- `csig_phase0/README.md`
- `csig_phase0/phase0_done.md`
- `csig_phase0/reports/phase0_report.md`
- `NEXT_TASKS.md`
- `task_plan.md`
- `findings.md`
- `progress.md`

Phase 0 已经确认：

1. HYPIR 在当前 5 张 validation 原图上整体表现为负收益，平均 PSNR gain 约为 -3.51 dB。
2. 当前 patch 中约 98% 按 `ΔPSNR > 0` 规则属于 BAD。
3. frozen feature 上：
   - CLIP macro ROC-AUC ≈ 0.6644
   - DINOv2 macro ROC-AUC ≈ 0.7548
   - Fusion macro ROC-AUC ≈ 0.7195
4. 但独立原图只有 5 张，AUC 实际只在 3 张图上定义。
5. 排除只有 1 个 GOOD patch 的图像后，Fusion AUC 明显下降，因此当前结论只能视为探索性信号。
6. 只看 LQ 的 `fusion_lq` 表现明显较弱，说明可靠性信号更可能存在于 **LQ 与 restored candidate 的关系** 中，而不是仅凭输入提前预测。
7. 当前 `GOOD/BAD = ΔPSNR > 0` 不是完整的 hallucination 定义，不能把“像素误差变大”直接等价为“语义幻觉”。

所以当前正式结论是：

> HOLD：不进入 Gate 部署、不训练专家、不训练 MoE。  
> 下一步先扩充数据、重设计 target，并验证 candidate-aware hallucination / reliability signal。

---

# 二、你这次的唯一目标

完成一个新的独立阶段目录：

`csig_phase05/`

本阶段只做：

> **Phase 0.5：Candidate-aware Hallucination Verification**

核心研究问题：

> 给定 LQ 图像 `x` 和 HYPIR 恢复结果 `y`，能否利用 frozen foundation feature、局部结构一致性和恢复稳定性，预测该区域的恢复结果是否可信？

重点验证三件事：

1. **DINO pairwise / token-level correspondence 是否能识别 HYPIR 不可靠区域；**
2. **PSNR gain 标签与人工 hallucination 标签是否存在明显偏差；**
3. **简单 verifier 是否可以支持 selective restoration，而不是只做分类。**

---

# 三、严格禁止事项

除非后续人工明确授权，否则本阶段禁止：

- 不训练 HYPIR；
- 不修改 HYPIR 权重；
- 不训练 LoRA；
- 不训练 Diffusion / DiT；
- 不训练 Mixture-of-Experts；
- 不训练 Text / Animal / Plant Expert；
- 不修改原有 `csig_phase0/` 结果；
- 不覆盖旧报告；
- 不使用 test GT；
- 不使用 test 集进行模型选择、阈值选择或任何形式的训练；
- 不把 patch 数量当作独立样本数量；
- 不以单一 accuracy 作为主指标；
- 不因为 DINO 在 Phase 0 表现最好就默认它一定有效，必须重新独立验证。

任何新结果必须写入 `csig_phase05/`。

---

# 四、阶段目录结构

请创建：

```text
csig_phase05/
├── README.md
├── configs/
├── data/
│   ├── manifests/
│   ├── annotations/
│   └── patches/
├── features/
│   ├── dino_global/
│   ├── dino_tokens/
│   ├── clip/
│   ├── structure/
│   └── stability/
├── outputs/
│   └── hypir/
├── checkpoints/
├── analysis/
├── scripts/
├── tests/
├── logs/
└── reports/
    ├── visualization/
    └── tables/
```

同时更新根目录：

- `task_plan.md`
- `progress.md`
- `findings.md`

但不要修改 Phase 0 历史结论。

---

# 五、Phase 0.5A：数据扩展与数据审计

## 目标

当前 5 张图远远不足。

本阶段第一优先级是将独立 paired validation 图像扩展到：

> 推荐 50–100 张独立场景。

如果当前机器或数据条件暂时无法达到 50 张，也必须：

1. 完成数据发现；
2. 列出候选来源；
3. 构建 manifest；
4. 明确当前实际可用数量；
5. 不得通过重复裁剪同一张原图伪装成“更多独立数据”。

---

## 场景覆盖建议

尽量覆盖：

- vegetation
- animal
- Chinese / English text
- architecture
- human
- small object
- night
- flare / glow
- low light
- motion blur
- defocus blur
- digital zoom
- severe low-resolution
- mixed degradation

特别注意：

不要让“语义类别”和“退化类型”完全绑定。

例如不要形成：

```text
animal = blur
vegetation = low-resolution
text = flare
```

而应该尽量交叉：

```text
animal × blur
animal × low-resolution
animal × noise

vegetation × blur
vegetation × low-resolution
vegetation × flare

text × blur
text × low-resolution
text × flare
```

---

## manifest 必须记录

每张独立原图至少包含：

```json
{
  "image_id": "...",
  "lq_path": "...",
  "gt_path": "...",
  "scene_category": "...",
  "degradation_tags": ["...", "..."],
  "source": "...",
  "pairing_verified": true,
  "alignment_verified": true,
  "split": "validation"
}
```

输出：

- `csig_phase05/data/manifests/pairs.json`
- `csig_phase05/reports/dataset_report.md`

---

# 六、Phase 0.5B：重新设计标签

不能再只使用：

```text
GOOD = ΔPSNR > 0
BAD = otherwise
```

本阶段必须并行构建三类标签。

---

## Label 1：Restoration Gain

每个 patch / region 计算：

- PSNR gain
- SSIM gain
- LPIPS gain
- DISTS gain

其中：

```text
gain_metric = metric(restored, GT) - metric(LQ, GT)
```

注意：

- 对“越低越好”的 LPIPS / DISTS 要统一方向；
- 建议标准化成“数值越大表示改善越好”。

保存：

`csig_phase05/analysis/restoration_gain.csv`

---

## Label 2：Hallucination Risk

请构建人工标注协议。

优先从代表性图像中抽样 patch，标注：

```text
0 = no hallucination
1 = mild
2 = clear
3 = severe
```

附加 hallucination type：

- texture_invention
- structure_invention
- semantic_corruption
- text_glyph_corruption
- boundary_corruption
- object_invention
- repeated_pattern
- over_sharpening_artifact
- uncertain

要求：

1. 不要让标注员看模型预测；
2. 最好同时展示 LQ / restored / GT；
3. GT 只用于离线标注和评估；
4. 标注规范必须写成文档。

输出：

- `csig_phase05/data/annotations/hallucination_protocol.md`
- `csig_phase05/data/annotations/hallucination_labels.csv`

---

## Label 3：Usefulness / Trust Target

不要立刻固定唯一公式。

先保留：

```text
restoration_gain
hallucination_risk
```

两组独立 target。

之后再分析：

```text
useful = gain_good AND hallucination_low
```

或者：

```text
utility = normalized_gain - lambda * hallucination_risk
```

lambda 暂时不要用测试数据调。

---

# 七、Phase 0.5C：DINO Pairwise Verification

这是本阶段核心实验。

## 目标

验证：

> HYPIR hallucination 是否表现为 LQ 与 restored image 之间局部 foundation feature correspondence 异常。

使用 frozen DINOv2。

不要微调 DINO。

---

## Experiment C1：Global Feature Baseline

分别提取：

```text
f_lq
f_restored
```

构造：

```text
f_lq
f_restored
|f_lq - f_restored|
f_lq * f_restored
cosine_similarity
L2_distance
```

比较以下输入：

1. LQ only
2. Restored only
3. LQ + Restored concat
4. Absolute difference
5. Pairwise engineered feature
6. 原 Phase 0 fusion baseline

模型只允许：

- Logistic Regression
- Linear Probe
- 2-layer MLP

不要使用大网络。

---

## Experiment C2：DINO Token-Level Correspondence

必须保留 spatial token grid。

不要只做 pooled embedding。

对于每个 patch / region：

提取：

```text
DINO tokens(LQ)
DINO tokens(Restored)
```

计算：

- token-wise cosine similarity
- token residual norm
- nearest-neighbor correspondence
- local matching confidence
- spatial displacement statistics
- mean / std / quantile of token disagreement
- top-k largest disagreement
- disagreement heatmap

重点保存：

```text
token_disagreement_map
```

---

## Experiment C3：Multi-scale

至少尝试：

- patch 128
- patch 256
- patch 512

如果显存或时间不允许，可只跑 256 + 512。

目标是判断：

> hallucination 是更容易从局部细节尺度看出来，还是需要更大的上下文。

---

# 八、Phase 0.5D：Structure Features

构建不依赖训练的结构型 verifier 特征：

- gradient difference
- Sobel edge difference
- Laplacian response
- local orientation histogram
- edge density change
- boundary displacement proxy
- high-frequency energy change

如果文字区域可检测：

- OCR confidence
- OCR string agreement
- character count change
- glyph topology proxy

这些特征不要代替 DINO，而是作为补充。

输出：

`csig_phase05/features/structure/`

---

# 九、Phase 0.5E：Restoration Stability

## 目标

验证：

> 如果一个区域主要由生成 prior 猜测，而不是由输入证据支持，那么改变 HYPIR 的生成条件后，该区域是否更不稳定。

---

## 如果 HYPIR 支持随机性

对同一输入生成 K 个结果：

```text
K = 4 或 5
```

改变 seed。

计算：

- pixel variance
- LPIPS variance
- DINO feature variance
- edge variance
- OCR variance
- token-level disagreement variance

---

## 如果 HYPIR 基本 deterministic

允许使用 controlled perturbation：

- 不同 model_t
- 不同 coeff_t
- 不同 prompt
- 不同 restoration strength
- 不同 crop context
- 不同 tile / overlap context

但必须：

1. 明确记录每组参数；
2. 不能混淆为“随机 uncertainty”；
3. 命名为 `restoration_stability` 或 `controlled sensitivity`。

---

# 十、Phase 0.5F：Verifier 模型

模型目标不是追求复杂度。

先做：

```text
Logistic Regression
Linear classifier
Tiny MLP
```

输入组分别比较：

### V1
DINO global

### V2
DINO pairwise

### V3
DINO token disagreement summary

### V4
Structure only

### V5
Stability only

### V6
DINO pairwise + structure

### V7
DINO pairwise + structure + stability

### V8
CLIP + DINO + stats（复现 Phase 0 baseline）

---

# 十一、严格的数据划分规则

必须以：

> 原图 / scene 为独立单位。

不能 random patch split。

推荐：

```text
Leave-One-Image-Out
```

当独立原图 >= 30 时，改为：

```text
GroupKFold / Grouped Train-Val Split
```

所有 patch 来自同一原图必须进入同一 fold。

如果存在同一场景不同 crop，也要放同一组。

---

# 十二、评价指标

不要只报告 ROC-AUC。

至少包括：

## 分类 / 排序

- ROC-AUC
- PR-AUC
- Balanced Accuracy
- Precision
- Recall
- F1

## 校准

- Brier score
- ECE（如果实现可靠）

## 图像级统计

必须先按原图统计，再做 macro average。

不要把 10,000 个 patch 当 10,000 个独立样本。

---

# 十三、最重要的新指标：Selective Restoration

本阶段必须新增：

> Risk–Coverage Curve

定义：

当 verifier 只接受置信度最高的 top-X% restored patch 时，统计：

- coverage
- mean restoration gain
- damage rate
- hallucination rate
- PSNR
- SSIM
- LPIPS
- DISTS

至少报告：

```text
Coverage = 10%
20%
30%
40%
50%
60%
70%
80%
90%
100%
```

定义：

```text
damage = restored region worse than fallback
```

fallback 第一版先用 LQ。

如果实现方便，可以额外比较一个 conservative baseline，但不要训练新模型。

---

# 十四、Selective Fusion 原型

只允许做简单原型：

```text
output = alpha * HYPIR + (1 - alpha) * LQ
```

其中 alpha 可先是：

```text
binary mask
```

或经过平滑后的 confidence。

不要训练复杂 fusion network。

必须比较：

1. LQ
2. HYPIR
3. Oracle mask（只做上界分析）
4. Predicted verifier mask
5. Random mask
6. Always HYPIR

重点看：

> verifier 是否真的减少 HYPIR 带来的 damage，而不仅仅是分类指标更高。

---

# 十五、Oracle Upper Bound

必须增加 oracle 分析：

若使用 GT，可以计算：

```text
oracle_accept = HYPIR better than fallback
```

得到：

```text
Oracle Selective Restoration
```

目的是回答：

> 如果 verifier 完美，HYPIR + fallback 这个框架理论上还有多少收益空间？

如果 oracle 本身也没有收益：

直接记录结论：

> 当前 candidate pool 不值得做 Gate。

不要强行继续。

---

# 十六、需要输出的核心图表

至少生成：

1. DINO token disagreement heatmap
2. hallucination annotation overlay
3. PSNR gain heatmap
4. hallucination severity heatmap
5. verifier confidence heatmap
6. Risk–Coverage curve
7. PR curve
8. ROC curve
9. calibration plot
10. `PSNR gain vs hallucination label`
11. `DINO disagreement vs hallucination label`
12. `stability score vs hallucination label`
13. 每类场景分组表现
14. vegetation / animal / text / night 代表性对照

所有可视化放：

`csig_phase05/reports/visualization/`

---

# 十七、关键统计分析

至少回答：

## Q1

PSNR BAD 和人工 hallucination 是否一致？

输出：

```text
confusion matrix
correlation
agreement
代表性 disagreement cases
```

---

## Q2

DINO disagreement 是否和 hallucination severity 正相关？

计算：

- Spearman
- Pearson（可选）
- 分组箱线图 / violin plot

---

## Q3

DINO global vs DINO pairwise vs token-level 谁最好？

---

## Q4

只看 LQ 和 LQ+Restored 差距多大？

---

## Q5

restoration stability 是否能提供额外独立信息？

做消融：

```text
DINO
DINO + stability
DINO + structure
DINO + structure + stability
```

---

## Q6

模型是否真的能 selective abstain？

最终必须回答：

> 在固定 coverage 下，damage rate 是否显著下降？

---

# 十八、成功 / 失败判定

不要预设“必须成功”。

---

## Phase 0.5 成功条件

至少满足：

1. 独立图像数量显著高于 Phase 0；
2. candidate-aware verifier 在 image-grouped validation 下稳定优于 LQ-only；
3. DINO pairwise / token disagreement 至少有一种表现稳定；
4. selective restoration 在某个有意义 coverage 上显著降低 damage；
5. 结果不是由 1–2 张图支撑；
6. hallucination label 与单纯 PSNR label 有明确区别。

---

## HOLD 条件

若：

- verifier AUC / PR-AUC 接近随机；
- image-to-image 方差极大；
- selective fusion 无实际收益；
- oracle selective upper bound 很低；

则：

> HOLD，不进入 Phase 1。

---

# 十九、只有满足 Phase 0.5 后，才允许进入 Phase 1

Phase 1 不直接做 MoE。

下一阶段正式定义为：

> **Selective Restoration**

候选：

```text
HYPIR
vs
LQ / conservative fallback
```

目标：

> Generate → Verify → Abstain

只有 Selective Restoration 被证明有效以后，再做：

> Candidate Benefit Analysis

然后才考虑：

- Detail Expert
- Structure Expert
- Optical Expert
- Motion Expert

最终才考虑 MoE / Router。

---

# 二十、代码质量要求

所有实验必须：

- 固定随机 seed；
- 配置化；
- 不硬编码绝对路径；
- 保存完整命令；
- 保存环境信息；
- 保存 git commit SHA；
- 保存输入和模型哈希；
- 每项实验可单独复现；
- 有最基本单元测试；
- 有数据泄漏测试；
- 有 grouped split 测试；
- 有 patch-to-image mapping 审计；
- 失败实验也记录；
- 不删除异常结果。

---

# 二十一、推荐脚本结构

```text
scripts/
├── build_manifest.py
├── run_hypir.py
├── build_patches.py
├── extract_dino_global.py
├── extract_dino_tokens.py
├── compute_pairwise_features.py
├── compute_structure_features.py
├── run_stability.py
├── build_hallucination_annotation_set.py
├── train_verifier.py
├── evaluate_verifier.py
├── evaluate_selective_restoration.py
├── oracle_analysis.py
└── make_report.py
```

---

# 二十二、最终交付物

必须提交：

## 文档

- `csig_phase05/README.md`
- `csig_phase05/reports/phase05_report.md`
- `csig_phase05/reports/dataset_report.md`
- `csig_phase05/data/annotations/hallucination_protocol.md`

## 原始表

- restoration gain
- hallucination labels
- features index
- fold split
- verifier metrics
- selective restoration metrics
- oracle upper-bound metrics

## 图

完整 visualization 目录。

## Checkpoints

只保存：

- Linear / Logistic / Tiny MLP verifier

不保存任何新 restoration model。

---

# 二十三、最终报告必须明确回答

最后的 `phase05_report.md` 不允许只列指标。

必须用明确结论回答：

1. HYPIR 的错误是否具有可预测的 candidate-aware signal？
2. DINO pairwise / token-level correspondence 是否比原 Phase 0 pooled feature 更可靠？
3. PSNR gain 是否是合适的 hallucination proxy？
4. 哪些 hallucination 类型最容易被 verifier 检测？
5. 哪些最难？
6. stability 是否提供额外信息？
7. selective restoration 是否真实降低 damage？
8. oracle upper bound 是否足够高？
9. 是否值得进入 Phase 1 Selective Restoration？
10. 是否仍应 HOLD？

结论只允许：

```text
GO
HOLD
STOP
```

并给出理由。

---

# 二十四、当前建议的科研主线

请始终围绕以下主线设计代码和实验，不要跑偏成普通 image restoration benchmark：

> **Generate → Verify → Abstain**

研究目标不是“让 HYPIR 更锐”，而是：

> 判断生成出来的细节是否有输入证据支持，并在证据不足时拒绝使用生成结果。

长期目标才是：

```text
LQ
↓
Generative Candidate
↓
Evidence-aware Verification
↓
Reliability Map
↓
Selective Restoration
↓
Capability Experts / MoE
```

但本次只做到：

```text
Candidate-aware Verification
+
Selective Restoration Prototype
```

---

# 二十五、执行顺序

严格按以下顺序：

```text
Step 1
阅读旧实验与现有代码

Step 2
建立 csig_phase05 目录和计划

Step 3
数据发现与 manifest

Step 4
重新运行 / 验证 HYPIR candidate

Step 5
建立 restoration gain labels

Step 6
建立 hallucination annotation subset

Step 7
DINO global baseline

Step 8
DINO pairwise experiment

Step 9
DINO token correspondence experiment

Step 10
structure feature experiment

Step 11
stability experiment

Step 12
grouped verifier training

Step 13
risk-coverage evaluation

Step 14
selective fusion prototype

Step 15
oracle upper-bound analysis

Step 16
消融 + 可视化

Step 17
完整审计

Step 18
phase05_report.md

Step 19
更新 progress.md / findings.md / task_plan.md
```

不要提前训练 Expert。

---

# 二十六、执行原则

如果发现：

- 数据配准错误；
- GT 与 LQ 不对应；
- patch mapping 错误；
- DINO feature extraction 不稳定；
- HYPIR 输出与 Phase 0 不一致；
- grouped split 数据泄漏；
- hallucination annotation 不可复现；

立即停止后续实验，先修复基础问题。

优先保证：

> 数据可信 > 实验可复现 > 统计正确 > 模型复杂度。

不要为了得到漂亮 AUC 调阈值、删图、筛 seed 或挑最好 fold。

所有负结果都必须保留。

---

请现在开始执行 Phase 0.5，并在每个阶段完成后，把关键结果追加到：

- `progress.md`
- `findings.md`

最终以：

`csig_phase05/reports/phase05_report.md`

作为本轮主验收文件。
