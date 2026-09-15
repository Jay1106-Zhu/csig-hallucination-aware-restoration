> 发布状态更新：用户已于2026-09-15授权本轮全量公开。代码和轻量图表在Git；完整数据/特征/权重/候选/融合/日志见 `../publication/phase05/README.md` 与独立Release `phase05-2026-09-15`。下方“未发布”是计算验收时的历史快照，不影响当前交付状态。人工标注依然pending，结论仍为HOLD。

# Phase 0.5 Candidate-aware Verification

状态：**计算实验已完成并审计 PASS；科学结论 HOLD；真实人工审核待完成。** 依照根目录 `CSIG_Phase05_Codex_Prompt(1).md`，所有新产物保存在本目录，Phase 0 保持只读。

研究主线：Generate → Verify → Abstain。只训练 Logistic/Linear/Tiny MLP verifier，不训练恢复器。

## 阅读入口

- 主报告与全部 Q1–Q10：`reports/phase05_report.md`。
- 数据来源与限制：`reports/dataset_report.md`；固定协议：`reports/protocol.md`。
- 验收清单：`reports/done_checklist.md`；独立审计：`analysis/artifact_audit.json`。
- 测试33项通过；审计重建3520条逐图指标、324个checkpoint/OOF，关键误差0。全量本地清单：`analysis/artifact_inventory.json` / `analysis/SHA256SUMS.txt`（约2.464GiB，不含安装依赖）。
- 数据：5 real CSIG + 50 synthetic BSDS train 源图，1160 非重叠区域，220 个四档候选；不是55个真实配对场景。
- 特征：DINO global/pairwise/token 双尺度、CLIP、结构、统计、确定性条件敏感性；仅训练324个轻量分组 verifier checkpoint。
- 主 V7 logistic 的 PSNR-benefit 宏 AUC=0.4050，LQ-only=0.6647，AUC只在5/55图定义。固定50%预测融合PSNR宏增益 −2.0892 dB，相对random损害下降不显著。

## 人工审核（唯一无法自动补齐的真值）

双击 `data/annotations/index.html`，按 `data/annotations/hallucination_protocol.md` 对110个固定样本盲审。**不要以0代替未审核，也不要用模型/助手判断冒充人。** 填写审核者ID后导出CSV：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.conda/python.exe new_formal/csig_phase05/scripts/annotations.py --import-csv <导出的CSV>
.conda/python.exe new_formal/csig_phase05/scripts/analyze_results.py
```

导入器保留原始 pending 表。此命令只生成已审核子集的描述统计，不自动宣称完成幻觉验证器/类型性能/分组人工标签实验；这些仍需检查标注覆盖并另行推进。当前报告为2026-09-15、0条人工标注的固定快照。

## 复现现有结果

从工作区根 `${CSIG_WORKSPACE}` 运行，使用已配置的 `.conda/python.exe`；需要只读 Phase0 模型、现有 HYPIR 环境和本阶段隔离依赖。离线检查/复算无需重跑恢复：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.conda/python.exe -m unittest discover -s new_formal/csig_phase05/tests
.conda/python.exe new_formal/csig_phase05/scripts/audit_phase05.py
```

首次构建顺序：`build_manifest.py` → `run_hypir.py` → `prepare_evaluation.py` → `annotations.py` → `extract_features.py` → `train_verifiers.py` → `selective_restoration.py` → `analyze_results.py` → `visualize.py` → `audit_phase05.py` → `write_report.py`。**已有产物不要重跑 build_manifest：它建立原始快照和源清单，不是增量更新入口。**

## 产物组织与范围

| 目录 | 内容 |
| --- | --- |
| `data/` | 来源原图、合成LQ、配对manifest、无损patch、盲标注材料 |
| `outputs/hypir/` | 四档冻结恢复候选 |
| `features/` | 原始空间tokens、特征矩阵、哈希索引 |
| `models/verifiers/` | 324个Logistic/MLP分组checkpoint；无新恢复模型 |
| `outputs/selective/` | 275张真实融合图及mask |
| `analysis/` | 四指标、独立targets、OOF、宏指标、风险曲线、oracle与审计 |
| `reports/visualization/` | 原图面板、token/gain/confidence、曲线和明确pending的人工图 |
| `logs/` | 官方推理、指标、特征、融合、可视化、测试及审计日志 |

本轮**未提交或推送GitHub**。轻量源码/表格/报告在Git忽略规则中可见；原图、依赖、完整特征/模型/输出保留本地，不混入旧Phase0 Release。公开再分发必须另行核对来源许可及打包范围。结论HOLD，不自动启动Phase1。
