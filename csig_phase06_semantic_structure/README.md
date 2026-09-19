# Phase06：低分夜景局部专家实验

当前状态（2026-09-19）：**第二轮 16 个候选已完成定向实测；13 个不采用、3 个仅保留复核资格，默认继续使用 H200。** 不训练、不跑 100 张全量、不自动部署或发布。

## 从这里开始

- 主报告：[第二轮定向专家实验](reports/round02_targeted_experts_2026-09-19.md)。包含完整 16 项指标、AI 视觉意见、第一轮勘误、限制和复现方式。
- 评测依据：[investigation 的评分 MD](../../investigation/csig_2026-09-18_experiment_summary_md/05_proxy_scoring_alignment.md)：CLIPIQA 为主，TOPIQ NR 为辅助；两者不加权；本轮不套 affine。
- 公开数据表：[comparisons.csv](public_results/comparisons.csv)、[metrics.json](public_results/metrics.json)。Original 不是 GT，ROI 分数不代替全图主指标。
- 视觉证据：[四组六面板](public_results/contacts)、[三组带上下文对照](public_results/review)、[AI 筛查记录](public_results/visual_review.json)。独立人工审核未完成。
- 验证证据：[产物审计 PASS](public_results/audit.json)、[39 项测试日志](public_results/logs/round02_tests.txt)。

## 本轮做了什么

固定 case38/case42 的立面、格栅和红色招牌四个 ROI；测试 FFTformer RealBlur-J、Restormer real denoising，分别输入 Original/H200。网络只处理原尺寸 ROI 加上下文，最终以 H200 为底图做局部羽化。全部候选 ROI 外像素保持不变。

| 优先复核候选 | 全图 ΔCLIPIQA | 实际融合 ROI ΔCLIPIQA | 当前限制 |
|---|---:|---:|---|
| case38 facade FFTformer←H200 | +0.001430 | +0.103553 | 观感改善，不证明纠正窗内生成细节 |
| case42 red_sign Restormer←H200 | +0.004964 | +0.063312 | 背景更平滑，但光晕仍在、字符正确性未知 |
| case42 red_sign FFTformer←H200（次选） | +0.001413 | +0.020008 | 同一招牌的另一候选，不是独立场景验证 |

三个候选的全图/ROI TOPIQ 也都上升，但只作为辅助证据，尚未加入正式 pipeline。两个模型不是现成的语义条件结构或中文发光文字专家。

OneFormer 的 `signboard, sign` 别名遗漏和 4K 短线过滤问题已修正；router 仍只提供粗 mask，ROI 由人工预先固定。本轮结构 ROI 内保护像素为 0，不能把它宣称为结构/招牌交界处的实景保护验证。

## 目录与复现

| 目录/文件 | 用途 |
|---|---|
| `configs/round02.json` | 推理前冻结的 case、ROI、专家和输入对照 |
| `scripts/roi_experiments.py` | 原尺寸推理、固定区域融合及候选 manifest |
| `scripts/evaluate_roi_experiments.py` | 同环境全图/原始 ROI/实际融合 ROI 评分及对照图 |
| `models/round02_assets.json` | 官方资源来源、revision、SHA-256 与许可路径 |
| `outputs/round02/router/` | 修正后的 Original 语义分割、边缘和线先验 |
| `outputs/round02/experiments/` | 16 份候选、基准、指标、视觉意见及审计 |
| `reports/`、`logs/`、`tests/` | 报告、实测日志和契约测试 |

在 `CSIG` 根目录运行：

```powershell
$Phase = "new_formal/csig_phase06_semantic_structure"
.\.conda\python.exe -m unittest discover -s "$Phase/tests" -v
.\.conda\python.exe -m compileall -q "$Phase/scripts" "$Phase/tests"
```

重现推理与评分的命令见主报告，须使用新输出目录，不覆盖 `outputs/round02/experiments`。本次 `.conda` 环境为 torch `2.11.0+cu128`、阶段内 pyiqa `0.1.16`；默认系统 Python 不是该环境。

下一步只安排三项候选的独立人工复核。未确认真实性前保留 H200，不扩全量；本轮没有榜单提交。[第一轮报告](reports/phase06_targeted_experiment_2026-09-19.md) 保留历史原文并在开头注明勘误，不应继续沿用其旧阈值和“无招牌”解释。

## GitHub 上传范围

Git 主分支上传可审阅的 Phase06 源码、配置、测试、报告、脱敏后的指标/审计记录、官方源码片段和精选对照图。完整实验目录另存于 GitHub Release `phase06-2026-09-19` 的 `phase06-complete-2026-09-19.zip`，包含原始输入、两个 `.pth` 权重、4K label map、16 份完整候选图和本地依赖。权重来源与 SHA-256 仍记录在 `models/round02_assets.json`。
