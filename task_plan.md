# new_formal 执行计划

日期：2026-09-15。仅执行 Phase 0；所有新产物归档于 new_formal。

## 阶段
1. [完成] 阅读方案，核验数据、环境与 HYPIR 资源。
2. [完成] 建立目录、协议和防泄漏测试。
3. [完成] 实现 patch、特征、分组 MLP 验证。
4. [完成] 运行实验并生成可视化。
5. [完成] 验收产物，汇总结论与后续任务。

## 约束
- 原始数据和 HYPIR 只读，不使用 test GT，不训练恢复模型。
- 按原图分组，重叠 patch 不得跨训练/测试折；GT 只用于标签与评价。
- 保留两份原始 md；后续阶段只列计划，不自动执行。

## 问题记录
- codegraph 未索引：改用文件检索。
- 非 Git 仓库：用哈希及清单追溯。
- nvcc 不在 PATH：实测 PyTorch CUDA。
- apply_patch 的批处理包装不能传递多行参数：直接调用其原生 apply_patch 入口。
- 一次环境检查超时：提高超时并拆分检查。
- 可视化命令超时：检查实际产物确认已写完，不盲目重跑。

## Next Step
用户已授权创建 public GitHub 仓库。当前下一步是整理公开范围、预检文件，然后创建并推送；不运行新实验。

## 验收结论
- 5 张官方推理、3565 组无损 patch、CLIP/DINO/stats 特征、90 个原图留一 checkpoint、全量 fusion checkpoint、20 个失败对照及全部热图已生成。
- 15 项契约测试通过；独立审计全部 patch 和 90 个折模型/scaler/OOF/逐折指标通过，35 个原始文件哈希保持不变。
- CLIP/DINO/Fusion 宏 AUC=0.6644/0.7548/0.7195；实际仅三个 AUC 有定义的原图，case4 只有一个 GOOD。
- 数值门槛 YES；实践建议 HOLD，先补独立数据。去掉单正例原图的事后敏感性 AUC=0.5926，不能宣称 Gate 已实用。
- 交付入口 csig_phase0/phase0_done.md；全报告 csig_phase0/reports/phase0_report.md。

## GitHub 公开发布（2026-09-15）
1. [完成] 核实账号 Jay1106-Zhu、目标仓库名可用和本地资源范围。
2. [进行中] 建立默认拒绝的忽略规则、公开说明、去除本机信息的元数据、发布检查。
3. [进行中] 已在 new_formal 初始化 Git，并创建 csig-hallucination-aware-restoration public 仓库；上传 Git 与五个完整 Release 归档。
4. [待完成] 核对远端 public 状态、提交和实际公开文件，记录交付。

用户补充明确要求：Phase 0 指导要求的产物全部上传。公开范围增加 validation 原图/GT、全部对照图、patch、特征、置信度模型、冻结模型、官方输出、完整元数据和日志。小文件放 Git，大文件按类别放同仓库 Release；仅排除依赖安装目录、缓存、凭据；本机路径以占位根路径表示，原本地文件不变。不擅自添加开源许可证。
