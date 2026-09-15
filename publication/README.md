# Phase 0 完整公开交付

仓库：`Jay1106-Zhu/csig-hallucination-aware-restoration`，可见性 public。

本次按用户要求上传 **Phase 0 指导规定的全部实际产物**，不是仅上传源码。Git 保存便于浏览的文档、代码、指标和全部图片；同仓库 Release `phase0-2026-09-15` 保存完整二进制、数据和模型归档。

## 按原始指导逐项对应

| 原始要求 | Git 可直接查看 | 完整下载包 / 包内路径 |
|---|---|---|
| 环境、requirements.txt | `csig_phase0/requirements.txt`；`publication/requirements_full_freeze.txt` | analysis 包中的 environment.json、nvidia-smi 日志；nvcc 未安装情况如实记录 |
| 数据检查与 dataset_report.md | `csig_phase0/reports/dataset_report.md` | data 包中的 data/manifest.json 与 validation LQ/GT 原图 |
| 官方 HYPIR 输出及参数 | 原始配置及最终报告 | hypir-outputs 包：`csig_phase0/outputs/hypir/`；analysis 包：推理命令、checkpoint 路径/哈希、模型 commit |
| input / HYPIR / GT patch | `csig_phase0/reports/patch_quality.csv` | data 包：`csig_phase0/data/patches/{case1..case5}/{input,hypir,gt}.npy`；共 3565 组三源 patch |
| CLIP、DINOv2、图像统计特征 | 配置、字段说明和模型性能表 | predictor-features 包：`csig_phase0/features/{clip,dino,stats}.npy` 与 index.csv |
| MLP checkpoint | 模型源码、训练配置 | predictor-features 包：`csig_phase0/checkpoints/confidence_mlp.pt` 以及 90 个折模型 |
| Accuracy/Precision/Recall/F1/AUC | confidence_metrics.csv、macro_summary、OOF/seed 表 | analysis 包提供相同指标和完整训练历史 |
| gain/confidence 热图 | `csig_phase0/reports/visualization/` 中全部 PNG | analysis 包同时提供原尺寸浮点 NPY 热图 |
| 20 个 Input/HYPIR/GT 失败案例 | failure_case*_*.png、逐图拼图和人工记录 | analysis 包包含完整相同图片，另含 5 个补充语义区域 |
| 最终报告、YES/NO、下一阶段建议 | phase0_report.md、phase0_done.md、NEXT_TASKS.md | analysis 包保留报告快照 |
| 冻结特征模型 | 精确来源/revision 记录在 analysis 包 | frozen-models 包包含完整 CLIP 与 DINOv2-small 权重和配置 |

HYPIR 恢复器原始大模型位于原工作区外部，不属于 new_formal 产物；本轮上传其来源、官方 commit、精确参数与 checkpoint SHA-256，而不额外复制 Stable Diffusion/HYPIR 原始大权重。CLIP/DINO 特征模型则完整包含于 Release。

## 五个完整归档

1. `phase0-data-patches.zip`：validation 原图/GT、推理输入副本、全部三源 patch、数据清单。
2. `phase0-frozen-models.zip`：冻结 CLIP、DINOv2-small 的实际模型文件。
3. `phase0-predictor-features.zip`：全部特征、最终置信度模型和 90 个折 checkpoint。
4. `phase0-hypir-outputs.zip`：五张官方 HYPIR PNG 及 prompt 文件。
5. `phase0-analysis-visualization.zip`：完整 analysis、reports、所有可视化及原始实验日志。

Release 另附 `artifact_manifest.json` 和 `SHA256SUMS.txt`。清单列出每个 ZIP 的大小和 SHA-256，以及每个包内文件的 SHA-256，不能把“上传了压缩包”代替完整性验证。

## 下载与恢复

在 GitHub 仓库的 Releases 页面打开 `phase0-2026-09-15`，下载五个 ZIP 并解压到仓库根目录。也可以从仓库根目录运行：

```powershell
python publication/download_artifacts.py --extract
```

脚本无需 GitHub 登录，先验证每个归档及成员的 SHA-256，再恢复目录。已存在且不同的文件默认保留并汇报；仅在确实希望替换本地同名文件时追加 `--overwrite`。完整包共包含约 3.3 GB 未压缩内容，请预留压缩包加解压文件的空间。

```powershell
python -m unittest discover -s publication/tests -v
python -m unittest discover -s csig_phase0/tests -v
```

测试需安装 `csig_phase0/requirements.txt` 中相关依赖。完整 GPU 复现还需要官方 HYPIR 的环境及外部权重，不应把公开交付误认为已经安装好的 Python 环境。

## 元数据与原始结果

- 数值指标、图像、patch、特征和模型张量保持实际实验结果，不重新训练、不重推理。
- 含本机绝对路径的文本/JSON 使用 `${CSIG_WORKSPACE}`、`${USER_HOME}` 占位根目录；pip freeze 中 file:// 构建路径替换为实际安装版本。
- 最终 `confidence_mlp.pt` 中的路径元数据同样去除本机根目录；模型权重张量不变，但序列化文件 SHA-256 因此不同于本地原始 checkpoint。
- 原始本地文件保持不变。历史 `artifact_audit.json` 是完整本地实验的验收；本次导出的哈希以 `publication/artifact_manifest.json` 为准。
- 读取下载的元数据时将占位根目录替换为自己的工作区；重新运行 `prepare_phase0.py data` 也会生成本机可用路径。外部 HYPIR 与 csig_dataset 采用阶段 README 所述工作区布局。
- 不上传凭据、Python 安装依赖目录、下载缓存和 Git 内部文件。这些不属于 Phase 0 要求交付的数据或结果。
- 未擅自为仓库添加开源许可证。公开可见性不改变第三方模型、代码和数据各自的使用条件。

## 实验结论

Fusion AUC=0.7195 达到原方案探索门槛，但只有三个可评价原图，其中 case4 仅一个 GOOD。排除单正例原图后 AUC=0.5926。结论保持 **数值门槛 YES，实践 HOLD：先补独立数据，不直接训练 Gate/专家**。
