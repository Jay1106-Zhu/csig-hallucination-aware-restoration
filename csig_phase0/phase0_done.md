# Phase 0 交付

日期：2026-09-15。
状态：已完成（小样本探索性验证，限制详见报告）。

## 环境
- Python 3.11.16，NVIDIA GeForce RTX 5080 Laptop GPU，torch 2.11.0+cu128，CUDA 12.8。
- 环境版本：requirements.txt；数据检查：reports/dataset_report.md。

## HYPIR 结果
- 官方 t200/t200，原尺寸 RGB PNG ×5：outputs/hypir/result/。
- 官方推理耗时 347.41 秒；原数据/源码/权重哈希验证：True。

## Patch 与 Confidence
- 5 张原图，3565 个重叠 patch；原图等权 BAD=98.3450%。
- CLIP/DINO/Fusion 宏平均 ROC-AUC：0.6644 / 0.7548 / 0.7195。
- 注意：AUC 只有三个有效原图；case4 仅一个 GOOD。去掉单正例原图后 Fusion AUC=0.5926（事后敏感性分析）。
- 分组：5 折原图留一 ×3 固定 seed；指标与最终 checkpoint 分离；全量模型：checkpoints/confidence_mlp.pt。

## 可视化与报告
- reports/visualization/：20 个失败对照、5 张 gain 热图、5 张 OOF confidence 热图及逐图概览。
- 完整报告：reports/phase0_report.md；原始表：reports/patch_quality.csv、reports/confidence_metrics.csv。

## 验证
- 15 项单元测试通过；全部 3565 组 patch 与源数据、标签一致。
- 独立重建 90 个折 checkpoint，验证训练集 scaler 和 OOF；报告：analysis/artifact_audit.json。

## 下一阶段
- 是否值得继续研究 Gate：**YES**（按固定 fusion 原图宏平均 AUC>0.65）。
- 实际推进：**HOLD，先补数据**。当前分类质量和正例覆盖不足，不建议进入 Gate/专家训练。
- 只完成 Phase 0，没有启动 Phase 1、恢复模型训练、LoRA 训练、MoE 或实际 gate 融合。
- 5 张图不足以做强泛化结论；PSNR BAD 不等于语义幻觉标签，后续需独立数据及语义评估。
