# Phase 0 — 全部实验产物

此 Release 与 main 分支共同交付 CSIG Phase 0 指导要求的全部产物。

- `phase0-data-patches.zip`：5 对 validation LQ/GT、输入副本及 3565 组 input/HYPIR/GT patch。
- `phase0-frozen-models.zip`：实际使用的冻结 CLIP 与 DINOv2-small 权重/配置。
- `phase0-predictor-features.zip`：CLIP/DINO/stats 特征、最终 confidence_mlp.pt 和 90 个原图留一折模型。
- `phase0-hypir-outputs.zip`：5 张原尺寸官方 HYPIR 输出及 prompts。
- `phase0-analysis-visualization.zip`：完整指标、OOF、训练历史、元数据、环境日志、全部热图/失败对照/报告。
- `artifact_manifest.json`、`SHA256SUMS.txt`：归档与包内文件完整性校验。

Git 中可直接查看全部图像可视化、代码、报告和数值指标；大文件在此 Release 完整提供，没有删去 Phase 0 要求的 patch、GT 对照、特征或置信度 checkpoint。

元数据中的本机绝对路径已替换为占位根目录；最终 checkpoint 只清理路径元数据，不改变模型张量。完整原始 HYPIR/Stable Diffusion 大权重是外部依赖，已记录来源、commit、参数与 checkpoint 哈希。

结果：CLIP/DINO/Fusion AUC=0.6644/0.7548/0.7195。只有三个有效原图且单正例场景抬高均值，去掉该场景后 Fusion AUC=0.5926。**研究门槛 YES，实际推进 HOLD。**

没有使用 test GT，没有训练恢复模型、LoRA 或 MoE，也没有启动 Phase 1。
