# 后续任务清单（未执行）

此文件只整理后续工作，不授权自动开始新阶段。以 Phase 0 最终报告的 YES/NO 和限制为决策依据。

Phase 0 当前结论：数值门槛为 YES（Fusion AUC=0.7195），但实际推进 **HOLD**。仅 59 个 GOOD、三个有效 AUC 原图；排除单正例原图后 AUC=0.5926。下一步应先补数据，而不是训练 Gate。

| 优先级 | 工作项 | 前置条件 | 验收产物 | 状态 |
|---|---|---|---|---|
| P0 | 扩充独立成对验证原图，覆盖植物、动物、汉字、小目标、夜景/flare | 确认数据来源和使用权限；不使用 test GT | 新阶段 `data/manifest.json`、场景数量与图像级划分 | 未授权 |
| P0 | 检查配准、PSNR 标签稳定性和人工语义幻觉标注 | 新独立验证数据 | 数据质量报告、预先固定标签协议、人工标注规范 | 未授权 |
| P1 | 在扩充数据上复验 frozen CLIP/DINO/stats + MLP | 不再只依赖五张开发图 | 原图级 OOF/独立验证指标、置信区间、校准报告 | 未授权 |
| P1 | Phase 1 Expert Benefit Analysis | 独立数据支持可靠性预测且用户授权 | 固定候选池、逐图和区域收益、oracle 上界分析 | 未授权 |
| P2 | Phase 2 Shared Restoration Adapter | 数据与训练可行性明确，完成 Phase 1 | 配置、训练日志、checkpoint、固定验证协议 | 未授权 |
| P2 | Phase 3 能力型 Expert Training | 单适配器收益成立；非语义分类专家 | Detail/Structure/Optical/Motion 对照与消融 | 未授权 |
| P3 | Phase 4 Reliability Gate Training | 候选收益、训练标签、数据划分和校准成立 | 原图验证、fallback/融合策略、收益与损害分析 | 未授权 |
| P3 | 语义空间约束与结构/OCR 分支 | gate 基线成立并有语义标注 | 语义约束、多专家、全局/空间、拒绝增强消融 | 未授权 |

## 当前停止边界
- Phase 0 只判断预测能力；不等于实际 Gate 已改善图像。
- 不因 patch 数多而忽略独立原图数少的问题。
- 无论 Phase 0 YES/NO，都不直接训练 LoRA/MoE 或开始专家分类。
- 新阶段使用新的独立目录；不能覆盖本次已固定的 Phase 0 结果。
