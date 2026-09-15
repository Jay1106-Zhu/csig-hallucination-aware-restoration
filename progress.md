# 执行日志

## 2026-09-15 GitHub 发布
- 范围变更：用户明确要求 Phase 0 指导要求的内容全部上传，取消此前对原图/GT 对照、特征和 checkpoint 的发布排除；改用 Git + Release 完整交付。
- 用户明确授权为 new_formal 创建新的 public 仓库。
- 确认 GitHub 账号 Jay1106-Zhu；Git Credential Manager 已有此账号，不读取或展示凭据。
- 拟定仓库名 csig-hallucination-aware-restoration，账号下无同名仓库。
- 本地约 3.3 GB，大部分为无损 patch、预训练权重及热图数组；发布时保留本地，只提交审查后的轻量内容。
- 新范围采用完整 Git + Release：五个归档包含所有 Phase 0 实际产物，另增加 validation 原始 LQ/GT 副本。
- 发布检查发现 pip freeze 含构建机器的 file:// 路径，已加入回归测试并转换为实际安装版本；不丢弃环境信息。
- 修复测试依赖父目录名 new_formal 的导入问题，独立仓库根目录也能运行 15 项原实验测试；未修改模型逻辑。

## 2026-09-15 Phase 0
- 已阅读两份新方案和旧计划上下文，建立独立记录。
- 尚未修改数据、HYPIR 或启动训练。
- 已固定 Phase 0 协议与参数；先写 12 项契约测试，观测缺少实现的预期失败后编写核心函数。
- 缺失 scikit-learn 已连同 joblib/threadpoolctl 安装在本阶段 dependencies 内，不替换旧环境包。
- 主实验选择官方 model_t=200/coeff_t=200，重新执行官方入口，不依据本轮 GT 挑选历史最优系数。
- 12 项核心契约测试已全部通过；环境与数据清单已归档，确认 5 对 validation、100 张 test 元信息。
- 官方 HYPIR 已后台运行，原始代码 Git 状态仅有预先存在的 models/weights 未跟踪目录，无代码改动。
- CLIP 和 DINOv2-small 预训练权重已下载到本阶段 models/，固定具体 revision 并记录文件 SHA-256。
- 5 张官方推理完成，347.41 秒；patch 3565×3 组已无损保存；CLIP/DINO 提取 68.77 秒；六组特征×5折×3seed 的 MLP 训练完成（55.37 秒）。
- 20 个失败案例、五组 gain/confidence 热图与逐图 ROC 已生成；一次可视化调用超时但产物实际完成，后续核验而非重复运行。
- 原始输入/GT、官方源码与恢复权重哈希校验全部一致；正在做独立产物审计与人工观察。
- 完成人工查看 20 个失败案例和 5 个补充语义区域，保存逐例标注及观察范围限制。
- 最终 15 项契约测试通过。独立审计校验全部 3565 组三源 patch 和标签，重建 90 个折 checkpoint，重算 OOF 与逐折指标；最大概率重建差 2.98e-08。
- 源数据/源码/恢复权重共 35 个文件保持原哈希；CLIP/DINO 权重未改变；test 参与训练/评价数量为 0。
- 已生成中文最终报告、phase0_done、总 README、阶段 README 和 NEXT_TASKS；不修改两份原始方案，不启动后续阶段。
- 最终结论：探索门槛 YES，实践 HOLD。主 AUC=0.719535，但只有三个有效原图且单正例图像抬高均值；不应直接进入 Gate 训练。
