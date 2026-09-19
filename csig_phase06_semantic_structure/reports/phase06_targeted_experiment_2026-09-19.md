# Phase06 Targeted Night Expert Experiment

日期：2026-09-19

> **2026-09-19 第二轮勘误：本文件下文保留第一轮原始记录，不代表当前结论。** 最新依据见 [第二轮实测报告](round02_targeted_experts_2026-09-19.md)。
>
> 1. OneFormer 实际预测了 `signboard, sign`：旧解析遗漏逗号别名；case42 Original 约 0.98447%、H200 约 0.7039%。不能再称没有 signboard，但 window/细粒度 ROI 路由仍未解决。
> 2. 旧 Hough 零线段含尺度实现问题，已修正并加回归测试；非零线段不等于结构真实性恢复。
> 3. `minimum_gain=0.05` 不是 investigation MD 的评测阈值；本轮不据此判断实验有效性。DiffIR 仍只是不具备足够采用证据的 control。
> 4. 六张子集的 `2.728522` 仅是历史 affine 算术结果，不符合完整 100 张的 proxy 口径；新 subset/cross-model 记录均为 `proxy_score=null`。
> 5. 已取得官方 FFTformer/Restormer 预训练权重并完成 16 个局部候选，不再存在“未下载所以无法开始”的阻塞。3 个仅进入复核名单，默认 H200 不变。

## 1. Scope

- 目标样例：`case37/38/39/41/42/65`。
- 约束：不训练、不跑 100 张全量；只做低分夜景样例的定向验证。
- 全局底图：固定使用已生成的 HYPIR H200 输出。
- 评测主依据：investigation 中定义的 `CLIPIQA global mean`；`TOPIQ NR` 只作辅助 sanity check。
- `ProxyScore` 只用于 HYPIR family 内部；跨模型结果不套 affine。`Original` 不作为 GT。

## 2. H200 Baseline

基线产物：`outputs/baseline_scores.csv`、`outputs/baseline_scores.json`。

| Case | CLIPIQA | TOPIQ NR |
|---:|---:|---:|
| 37 | 0.455901 | 0.360578 |
| 38 | 0.355007 | 0.435467 |
| 39 | 0.421603 | 0.415016 |
| 41 | 0.468417 | 0.433318 |
| 42 | 0.449529 | 0.442368 |
| 65 | 0.434007 | 0.378981 |

六张诊断集均值为 `CLIPIQA=0.430744`、`TOPIQ=0.410954`；仅作为 HYPIR-only diagnostic proxy 时 `ProxyScore=2.728522`，不能外推为官方全量分数。

## 3. DiffIR Control

候选：本地已有 `Deblurring-DiffIRS2.pth`，仅作为 motion-deblur control，不作为文字或建筑结构 expert。

- 样例：`case42`。
- ROI：`(x0,y0,x1,y1)=(700,400,1212,912)`，512×512，同分辨率推理，羽化边界 32 px。
- 全图候选：`outputs/experts/diffir_control/result/case42.png`。
- ROI 前后：`case42_base_roi.png`、`case42_diffir_roi.png`。
- 对比报告：`outputs/experts/diffir_control/case42_comparison.json`。

| Scope | H200 CLIPIQA | Control CLIPIQA | Delta | H200 TOPIQ | Control TOPIQ | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Global case42 | 0.449529 | 0.449542 | +0.000013 | 0.442368 | 0.443154 | +0.000786 |
| 512×512 ROI | 0.287569 | 0.317317 | +0.029749 | 0.382843 | 0.406918 | +0.024075 |

结构统计没有形成同步证据：Hough line count 为 `20 -> 21`，但平均线长为 `130.65 -> 120.19`，line prior ratio 为 `0.008694 -> 0.008106`。按现有 `minimum_gain=0.05` candidate veto，结论是 `KEEP_H200`，原因是“insufficient overall gain”。

因此：局部 metric 上升只能作为 screening signal，不能证明 DiffIR control 适合贴回；当前不纳入正式 pipeline。

## 4. OneFormer Router POC

模型：`shi-labs/oneformer_ade20k_swin_tiny`。

分别对 H200 输出和原始 `case42.jpg` 运行，label map 都回映射到原生 `4096×3072`。

- H200 输入报告：`outputs/oneformer_case42/case42_router.json`。
- 原图输入报告：`outputs/oneformer_case42_original/case42_router.json`。
- 可视化：对应目录下的 `case42_semantic_overlay.png`、`case42_structure_mask.png`、`case42_structure_lines.png`。

两种输入都出现相同失败模式：`building` 覆盖约 `95.9%`，`sky` 约 `2.7%–2.9%`，没有有效 `window` 或 `signboard` 区域，structure line ratio 为 `0`。这不满足 investigation 中的 Router 成功标准，因此不能直接用该 mask 驱动 structure expert ROI。

## 5. Decision

- `case42` 的 DiffIR control：拒绝贴回，保留为 control artifact。
- OneFormer ADE20K tiny：Router POC 不通过，不能作为当前 semantic router。
- H200：继续作为六张样例的默认底图。
- 当前真正缺口：仓库中没有已经验证的 same-resolution building/window/signage expert checkpoint；现有 DiffIR 权重属于 motion-deblur，不应改名为文字专家。

## 6. Next Targeted Work

1. 不启动训练、不跑全量；先取得或指定真正的 structure/text expert checkpoint。
2. 优先在 `case38`（六张中 CLIPIQA 最低）和 `case42`（结构信号最强）各跑一块 ROI。
3. 每个候选固定记录 global CLIPIQA/TOPIQ、ROI CLIPIQA/TOPIQ、line/window preservation、字符真实性、halo/tile seam，并默认 `KEEP_H200`。
4. 只有全图主指标达到明确增益且没有视觉 veto，才考虑扩展到其余四张；在此之前不跑 100 张。
