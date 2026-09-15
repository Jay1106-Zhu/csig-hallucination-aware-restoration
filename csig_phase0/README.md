# Phase 0：恢复可信度预测

固定官方 HYPIR t200/t200，使用五张验证原图做 patch 标签、冻结特征和原图留一 MLP 验证。只训练置信度预测器，不训练恢复器。

## 阅读结果
- `phase0_done.md`：简要交付。
- `reports/phase0_report.md`：指标、结论、限制、失败案例观察。
- `reports/protocol.md`：观察本轮结果前固定的实验协议。
- `reports/confidence_macro_summary.csv`：CLIP/DINO/stats/fusion/LQ-only 原图宏平均指标。
- `reports/confidence_metrics.csv`：每种特征、每张原图的 OOF 指标。
- `analysis/oof_predictions.csv`：三个固定 seed 和集成的逐 patch OOF 概率。
- `analysis/artifact_audit.json`：产物、标签、折划分、scaler、OOF 重建与哈希审计。

## 环境

以下历史实验命令从包含 `new_formal/`、`HYPIR/`、`csig_dataset/` 的工作区父目录运行。使用现有 `.conda` 和已安装的官方 HYPIR 依赖；不需要 nvcc。CUDA 由 PyTorch wheel 提供。独立克隆仓库可先按 `publication/README.md` 下载全部 Release 产物。

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = "$PWD\new_formal\csig_phase0\dependencies"
$env:HF_HUB_DISABLE_XET = '1'
.\.conda\python.exe -m unittest discover -s new_formal\csig_phase0\tests -v
```

补充依赖已隔离安装；仅在缺失时执行：

```powershell
.\.conda\python.exe -m pip install --target new_formal\csig_phase0\dependencies --no-deps scikit-learn==1.7.2 joblib==1.5.2 threadpoolctl==3.6.0
```

`requirements.txt` 固定本实验直接依赖；`analysis/environment.json` 包含 CUDA/GPU/版本。`requirements_full_freeze.txt` 是实际完整环境快照。不要盲目在旧环境中重装所有依赖。

## 分阶段复现

下列流程会重写本阶段对应产物，因此正式新实验应复制配置和脚本到新阶段/新 run 目录，而不是覆盖已验收的本次结果。

```powershell
.\.conda\python.exe new_formal\csig_phase0\scripts\prepare_phase0.py data
.\.conda\python.exe new_formal\csig_phase0\scripts\prepare_phase0.py models
.\.conda\python.exe new_formal\csig_phase0\scripts\prepare_phase0.py hypir
.\.conda\python.exe new_formal\csig_phase0\scripts\run_phase0.py all
.\.conda\python.exe new_formal\csig_phase0\scripts\report_phase0.py all
.\.conda\python.exe new_formal\csig_phase0\scripts\audit_phase0.py
```

- 特征模型只在 `models` 阶段联网下载；推理/特征提取使用本地离线权重。
- 冻结模型身份、精确 revision 和权重哈希保存在 `analysis/feature_models.json`。已有清单时下载阶段应复用固定 revision。
- `run_phase0.py` 支持 `patches`、`features`、`train`、`audit` 单独恢复执行。
- 官方 HYPIR 完整参数和命令在 `analysis/hypir_inference.json`；`patch_size=512` 是模型推理 tile，分析 patch 则为 256。
- 人工失败案例观察记录为 `analysis/failure_annotations.csv`，不是自动评测标签；复现新数据时必须重新检查。

## Patch 与特征读取

patch 以无损 RGB uint8 三维图片集合保存，而不是散落上万个 PNG：

```python
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

root = Path('new_formal/csig_phase0')
index = pd.read_csv(root / 'features/index.csv')
row = index.iloc[0]
patches = np.load(root / 'data/patches' / row.image_id / 'input.npy', mmap_mode='r')
image = Image.fromarray(patches[int(row.patch_index)])
```

- 同目录 `hypir.npy`、`gt.npy` 使用相同 patch_index；每张图形状 `(713,256,256,3)`。
- `features/clip.npy`：input CLIP 512 维 + HYPIR CLIP 512 维。
- `features/dino.npy`：input DINO 384 维 + HYPIR DINO 384 维。
- `features/stats.npy`：24 维，字段见 `analysis/patch_storage.json`。
- 所有特征行顺序必须与 `features/index.csv` 和 `reports/patch_quality.csv` 完全一致。

## Checkpoint 用途

`checkpoints/confidence_mlp.pt` 包含三个固定 seed 的 fusion MLP 和各自 scaler，使用全部 validation 拟合。**不能拿它在这五张图上的预测作为验证指标。**

严格 OOF 复现使用 `checkpoints/folds/{feature}/{heldout_image}_seed{seed}.pt`。每个文件含训练图像 ID、输入维度、mean/scale、模型状态和 heldout_image。

输入维度 1816：1024 CLIP +768 DINO +24 stats。模型为 Linear(1816,64)→ReLU→Linear(64,1)→Sigmoid；最终概率取三 seed 均值。类别重加权训练意味着概率未必校准。

## 限制

只有五张原图，3565 个重叠 patch 不是 3565 个独立样本。GOOD/BAD 指 PSNR 相对收益，不是语义幻觉真值。热图使用 OOF，图像 bootstrap 只重采样原图。无测试集评价、无实际 gate 融合、无 Phase 1 自动执行。
