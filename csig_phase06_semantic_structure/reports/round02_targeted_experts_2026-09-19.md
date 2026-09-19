# Phase06 第二轮：低分夜景的定向预训练专家筛查

日期：2026-09-19。状态：**本轮 16 个候选的推理、评分、AI 辅助视觉筛查与产物审计完成；不部署，默认仍为 H200。**

## 1. 先看结论

- 遵守“不训练、不跑全量”：只用 `case38`、`case42`，每图两个预先固定 ROI；两个预训练模型、Original/H200 两种输入，共 16 个候选。
- 严格依照 [05_proxy_scoring_alignment.md](../../../investigation/csig_2026-09-18_experiment_summary_md/05_proxy_scoring_alignment.md)：全图 CLIPIQA 为主，TOPIQ NR 为辅助；ROI 指标只作局部诊断。不加权，不把子集或跨模型候选代入 affine。
- **3 个候选在全图和实际融合 ROI 上，两项指标均上升**；优先复核 `case38 facade → FFTformer←H200`、`case42 red_sign → Restormer←H200`。同一 case42 招牌上的 `FFTformer←H200` 留作次选对照，不是第三个独立场景。
- **13 个不采用**：11 个全图主指标下降；另 2 个 Original 输入的 case42 招牌虽提高 CLIPIQA，但局部明显偏糊、TOPIQ 大幅下降，保留 H200。
- 正向结果目前主要支持“噪声/局部观感改善”，**不证明窗格真实恢复、中文字符正确、H200 幻觉被纠正或炫光已消除**。独立人工审核、其他场景泛化和榜单验证均未完成。

## 2. 第一轮勘误

第一轮历史报告见 [phase06_targeted_experiment_2026-09-19.md](phase06_targeted_experiment_2026-09-19.md)。以下发现修正了上一轮的解释，不改写历史实测图片和分数。

| 原解释 | 本轮核查与纠正 | 影响 |
|---|---|---|
| OneFormer 没有识别 signboard | ADE20K 实际标签是 `signboard, sign`；旧精确匹配遗漏了别名。旧 case42 Original label map 中该类约占 0.98447%，H200 约占 0.7039%。 | 已按逗号拆分别名；不能再称“完全没有招牌”。预测覆盖率不等于分割准确率。 |
| 4K structure line ratio 为零说明没有可用结构 | Hough 的最短线段随整图尺寸放大，过滤掉了短窗框/格栅线段。现将投票阈值与最短线段分别限制在 64、96 像素以内。 | 已加大图短线回归测试；非零线段也不等于结构恢复正确。 |
| `minimum_gain=0.05` 是拒绝 DiffIR 的正式依据 | MD 没有给出这个阈值，它只是旧 helper 的未校准默认参数。 | 本轮不调用该 helper、不新增硬收益门槛；旧 DiffIR 结论只代表“尚未证明可以采用”。 |
| 六张的 affine 数值可作为 diagnostic proxy | 完整 100 张才符合 MD 的 full-set proxy 口径。 | 旧 `2.728522` 仅为历史算术结果，不再作为本轮判据；新 subset summary 的 `proxy_score=null`。 |

`oneformer_router.py` 同时将 signage 从结构专家的可编辑类别中移除，避免把招牌自动归给建筑处理分支。`score_targets.py` 仅在 HYPIR family 且 case1–100 各出现一次时返回 affine 分数，重复 case 会拒绝。

## 3. 冻结的实验协议

来源：[round02.json](../configs/round02.json)、[candidate_manifest.json](../public_results/candidate_manifest.json)。ROI 在候选生成前固定，没有看分数后修改 ROI、羽化宽度或参数。

### 3.1 样例与 ROI

| 原图 | 分支 | `(x0,y0,x1,y1)`，右/下边界不包含 | 原尺寸 ROI |
|---|---|---|---|
| case38 | facade / structure | `(1824,1664,2592,2432)` | 768×768 |
| case38 | red_sign / signage | `(1728,1296,2400,1664)` | 672×368 |
| case42 | lattice / structure | `(1792,416,2816,1056)` | 1024×640 |
| case42 | red_sign / signage | `(1296,2024,1864,2432)` | 568×408 |

选择依据是 MD 的夜景建筑/重复结构/发光招牌共性；case38 是此前六张诊断集中 CLIPIQA 最低者，case42 是 MD 指出的非 halo 结构信号最强者。它们是探索样例，不是新独立 holdout。四块 ROI 和 16 个变体不能算作 16 张独立图片。

- 全图底图始终是同图 H200。Original 只作为专家输入对照和观察参照，**不是 GT**。
- 原尺寸 ROI 外扩 64px 上下文；仅补齐网络倍数，FFTformer 为 32、Restormer 为 8；输出裁回原 ROI，32px 羽化。
- 不外部缩放、不 tile、不重新运行 HYPIR；FP32，seed=231，关闭 TF32。
- 结构分支读取 Original 的语义 mask，并排除 17×17 膨胀后的 signboard 区域；signage 分支允许编辑冻结的招牌 ROI。
- **本轮四个 ROI 内实际 `allowed_ratio=1`，保护像素数均为 0**。因此实景产物验证了 ROI 外完全不变，但不能宣称本轮已实景验证“结构/招牌交界处的保护能力”；非空保护区域由契约测试验证。
- OneFormer 仅作粗分割和区域约束，没有将语义、edge 或 line 条件输入两套恢复网络。**这不是训练好的 semantic-conditioned structure expert。**

### 3.2 官方预训练资源与环境

| 模型 | 本轮 checkpoint / 来源 | 角色与限制 |
|---|---|---|
| FFTformer | `fftformer_realblur_j.pth`；`kkkls/FFTformer` revision `01306b5d4072146a011c8c6e9d4927cf4607f1d5` | RealBlur-J 去模糊候选，same-resolution；不是现成中文文字、去炫光或语义条件结构专家。 |
| Restormer | `restormer_real_denoising.pth`；`swz30/Restormer` revision `68dc6ac472db26f16361150cb7a96a1bc87da93f` | 真实噪声预训练候选；不是专用去炫光或中文文字专家。 |

架构直接取对应官方源码及 YAML，用 `strict=True` 加载权重，不执行 YAML 中的训练流程。来源、完整 SHA-256、文件大小和对应 MIT license 路径均记录于 [round02_assets.json](../models/round02_assets.json)。下载资源已经就绪，不存在“缺少权重所以不能开始”的阻塞。

- 推理/评分：`.conda/python.exe`，PyTorch `2.11.0+cu128`，NVIDIA GeForce RTX 5080 Laptop GPU。
- 指标：阶段内冻结 `pyiqa 0.1.16`，复用既有 CLIPIQA/TOPIQ NR scorer；PIL RGB → float32 `[0,1]`，不做外部 resize，保留 pyiqa 各指标内部预处理。
- MD 原环境为 torch `2.8.0+cu128`；本机版本不同，因此本轮在同一本机环境同时重测 H200 与候选，不把不同环境的绝对分数混用。
- 两份指标权重的 SHA-256 记录在 `metrics.json`；case42 H200 同进程重测两项 delta 都为 0。这是一次确定性检查，不是统计显著性或跨机器一致性证明。

### 3.3 判定方式

以同图 H200 为对照，先比较**融合全图 CLIPIQA**，再看实际融合 ROI 的变化，TOPIQ 仅检查 trade-off。跨模型分数仍只是 screening。全图主指标下降者本轮不采用；主指标上升但 TOPIQ/视觉明显变差者不因分数更高而采用；同向提升只进入视觉与独立人工复核名单。

不对两个指标加权，不构造新的质量分数，不将 ROI 均值代替全图均值，也不通过两个场景推测完整 100 张的收益。

## 4. 全部实测结果

完整精度见 [metrics.json](../public_results/metrics.json) 与 [comparisons.csv](../public_results/comparisons.csv)。下表全部 delta 都是**候选减同图 H200**；ROI 使用最终贴回后的 `blended_roi.png`，不是只看未融合的 raw patch。

同环境 H200 基线：

| 原图 | 全图 CLIPIQA | 全图 TOPIQ NR |
|---|---:|---:|
| case38 | 0.355006844 | 0.435466766 |
| case42 | 0.449529380 | 0.442367613 |

两图基线均值为 CLIPIQA `0.402268112`、TOPIQ `0.438917190`，只作诊断记录，`proxy_score=null`。

| 原图/ROI | 专家 | 输入 | 全图 ΔCLIPIQA | 全图 ΔTOPIQ | 融合 ROI ΔCLIPIQA | 融合 ROI ΔTOPIQ | 本轮决策 |
|---|---|---|---:|---:|---:|---:|---|
| 38 facade | FFTformer | Original | -0.026853 | -0.000299 | -0.250173 | -0.149090 | 保留 H200 |
| 38 facade | FFTformer | H200 | +0.001430 | +0.005148 | +0.103553 | +0.092090 | 优先复核，不采用 |
| 38 red_sign | FFTformer | Original | -0.008504 | -0.000809 | -0.196107 | -0.144567 | 保留 H200 |
| 38 red_sign | FFTformer | H200 | -0.000808 | +0.001048 | -0.013425 | -0.008269 | 保留 H200 |
| 42 lattice | FFTformer | Original | -0.004255 | -0.002708 | -0.053727 | -0.131785 | 保留 H200 |
| 42 lattice | FFTformer | H200 | -0.000919 | +0.000756 | -0.021708 | +0.019904 | 保留 H200 |
| 42 red_sign | FFTformer | Original | +0.011605 | -0.002376 | +0.058638 | -0.170567 | 模糊/trade-off，不采用 |
| 42 red_sign | FFTformer | H200 | +0.001413 | +0.000653 | +0.020008 | +0.013452 | 次选复核，不采用 |
| 38 facade | Restormer | Original | -0.024865 | +0.000342 | -0.242284 | -0.139353 | 保留 H200 |
| 38 facade | Restormer | H200 | -0.005800 | +0.002184 | -0.016992 | +0.010652 | 保留 H200 |
| 38 red_sign | Restormer | Original | -0.011922 | -0.000652 | -0.238896 | -0.171950 | 保留 H200 |
| 38 red_sign | Restormer | H200 | -0.003493 | +0.000643 | -0.072361 | -0.027182 | 保留 H200 |
| 42 lattice | Restormer | Original | -0.000933 | -0.002922 | -0.030883 | -0.149032 | 保留 H200 |
| 42 lattice | Restormer | H200 | -0.001467 | -0.000486 | -0.033743 | -0.013851 | 保留 H200 |
| 42 red_sign | Restormer | Original | +0.007570 | -0.001140 | +0.009233 | -0.166403 | 模糊/trade-off，不采用 |
| 42 red_sign | Restormer | H200 | +0.004964 | +0.000784 | +0.063312 | +0.025131 | 优先复核，不采用 |

正向候选的绝对分数：

| 候选 | 全图 CLIPIQA | 全图 TOPIQ | 融合 ROI CLIPIQA | 融合 ROI TOPIQ |
|---|---:|---:|---:|---:|
| case38 facade FFTformer←H200 | 0.356436580 | 0.440614760 | 0.476704597 | 0.500678599 |
| case42 red_sign FFTformer←H200 | 0.450942099 | 0.443020225 | 0.194713220 | 0.418162346 |
| case42 red_sign Restormer←H200 | 0.454493731 | 0.443151563 | 0.238016888 | 0.429841191 |

没有把三个候选拼成一个“最终提交版本”。尤其两条 case42 招牌候选是同一区域的替代选项，不能累计它们的收益。

## 5. 视觉筛查及 router 复核

记录：[visual_review.json](../public_results/visual_review.json)。这是**非盲法 AI 辅助观察**，不是独立人工标注。原始指标文件中的 `visual_status=PENDING` 保留不变；本文件将 AI 意见独立存档，不用它填充人工审核。

- **case38 facade / FFTformer←H200**：立面细噪声减少、窗框较清楚，所查看 64px 上下文中未发现明确新增硬拼接缝；窗内 H200 原有疑似生成细节仍在。保留复核资格，不能称已解决结构幻觉。
- **case42 red_sign / Restormer←H200**：背景更平滑，LED 点阵和主要笔画大体保留，但红色光晕仍明显，并有过度平滑/塑料感风险。保留复核资格，不宣称字符正确或去炫光成功。
- **case42 red_sign / FFTformer←H200**：类似的轻微噪声/观感改善，光晕仍在，指标提升小于同 ROI Restormer；作为次选对照。
- **case42 red_sign / 两条 Original 输入**：尽管全图 CLIPIQA 比 H200 高，招牌仍明显偏糊、与周围 H200 清晰度不一致，ROI TOPIQ 分别下降约 0.171、0.166；本轮不采用。
- **case38 red_sign**：没有一致的文字真实性收益；Original→FFTformer 的笔画周围有暗轮廓/振铃风险。四路全图主指标都下降。
- **case42 lattice**：Original 输入两路仍糊；H200 输入两路没有可确证的格栅几何修复。四路全图主指标都下降。

四组六面板位于 [contacts](../public_results/contacts)，顺序为 Original/H200、FFTformer←Original/H200、Restormer←Original/H200；三组带原尺寸上下文的 H200/候选对照位于 [review](../public_results/review)。对照文件未做内容缩放，查看工具可能缩略显示。

修正后的 Original 输入 router：

| 原图 | building 覆盖率 | signboard 覆盖率 | 其他主要类别 | line prior ratio |
|---|---:|---:|---|---:|
| case38 | 91.8289% | 1.7923% | tree 4.5105%、road 0.2699% | 0.000166575 |
| case42 | 96.1602% | 0.98447% | sky 2.7241% | 0.003372669 |

这说明可以得到粗建筑/天空/部分招牌预测，但没有单独分出 window；building 连通域仍几乎整图。**不能称自动精细 ROI router 已完成**，本轮 ROI 是人工冻结的。预测比例只是诊断统计，没有 GT 分割精度验证。

## 6. 测试、审计与证据链

使用项目 `.conda` 环境实测：**39 项单元/契约测试全部通过，0 skipped；scripts/tests 的 compileall 通过**。旧诊断可视化代码有一个 Pillow `mode` 参数弃用警告，未影响测试，本轮不改无关代码。

[audit.json](../public_results/audit.json) 为 PASS，核验包括：

- 完整候选集合恰好为冻结配置的 16 项，无缺失/重复；指标与视觉记录逐 ID 对齐。
- 配置、推理脚本、评分脚本、10 份专家源码/配置/许可/权重资源、2 份评分权重、4 份输入文件的 SHA-256 一致。
- 8 份基准 ROI 与输入精确裁剪一致；16 份融合 ROI 与输出全图同坐标裁剪逐像素一致。
- 全图尺寸不变；全部候选 ROI 外变化通道数为 0；用 raw ROI、mask 和固定 feather 重建，16 份全图逐像素完全相同。
- mask 与 Original router 对齐；本轮 ROI 内保护像素数为 0，不夸大空集合上的保护检查。非空保护 mask 的行为已由测试覆盖。
- 全部全图/ROI delta 与对应基线相减一致、没有加权；subset/cross-model proxy 均为 null。

审计不重新推理或重新计算全部 IQA；它核查已经得到的图像、哈希、记录与算术一致性。公开日志：[round02_tests.txt](../public_results/logs/round02_tests.txt)、[round02_audit.txt](../public_results/logs/round02_audit.txt)。

| Evidence | Finding | Path / 后续动作 |
|---|---|---|
| 16 项成对分数、四组六面板 | 3 项同向提升、13 项不采用；不是整条模型路线普遍有效 | 只保留三项局部候选作独立复核，H200 默认不变 |
| case42 Original 招牌 TOPIQ 下跌与视觉模糊 | 跨模型 CLIPIQA 上升不等于文字恢复更正确 | 对这两项执行不采用，不套 affine |
| 原始 label map 与别名回归测试 | 原“无招牌”结论包含实现遗漏 | 用修正后的粗 mask，不声称自动细粒度路由完成 |
| 全图逐像素/哈希核验 | 16 次编辑均限制在固定 ROI，证据链可追溯 | 冻结当前产物，不覆盖历史输出 |

## 7. 复现与下一步

在 `CSIG` 根目录运行测试；不要使用没有 torch 的系统默认 Python：

```powershell
$Phase = "new_formal/csig_phase06_semantic_structure"
.\.conda\python.exe -m unittest discover -s "$Phase/tests" -v
.\.conda\python.exe -m compileall -q "$Phase/scripts" "$Phase/tests"
```

需要重现这 16 个候选时，用**新输出目录**，复用已经归档的 Original router、冻结权重与配置；以下命令不代表本轮已再次运行：

```powershell
$Phase = "new_formal/csig_phase06_semantic_structure"
$Replay = "$Phase/outputs/round02_replay"
.\.conda\python.exe "$Phase/scripts/roi_experiments.py" --config "$Phase/configs/round02.json" --output-dir "$Replay" --device cuda
.\.conda\python.exe "$Phase/scripts/evaluate_roi_experiments.py" --experiment-dir "$Replay" --device cuda
```

当前单一下一步是：**由队友独立检查三项 shortlist 的原尺寸对照，重点确认窗内细节、笔画/LED 点阵是否受损及过度平滑风险；未通过前仍保留 H200。**

通过后才值得在剩余 `case37/39/41/65` 中预先固定少量同类 ROI，分别验证立面轻度后处理和发光招牌后处理能否重复改善。对于 halo，下一轮应另设真正匹配去炫光任务的预训练候选，而不是把本轮去噪增益当作去炫光证据；目前未启动该实验，也没有训练、全量推理、榜单提交或公开发布。
