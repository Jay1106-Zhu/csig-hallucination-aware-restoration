# Phase 0 预先固定实验协议

日期：2026-09-15。在观察本轮 patch 标签和模型指标之前固定此协议。

## 范围和基线
- 只使用 csig_dataset/验证集中的五对图像；test 仅统计目录和图片元信息，不提取特征、不读取 test GT。
- 主实验重新运行未修改的官方 HYPIR/test.py，固定官方 model_t=200、coeff_t=200，upscale=1、空 prompt、seed=231、推理 tile=512/stride=256。
- 不根据本次 GT 选择 HYPIR 强度；历史 coeff_t=50 不混入本实验。该结论只适用于此 HYPIR 配置。
- 原始数据、官方源码和恢复权重在运行前后计算 SHA-256 验证只读。

## 标签与特征
- RGB uint8 原尺寸严格对齐，patch=256/stride=128；末端不足步长时追加边界 patch，图片不足 patch 时拒绝处理。
- 标签为 patch PSNR gain>0；gain<=0 为 BAD。PSNR 不是语义幻觉标注，不能将 BAD 比例等同于幻觉发生率。
- 同时保存 input/HYPIR/GT patch。GT 只在标签和报告阶段使用。
- 冻结 CLIP ViT-B/32 和 DINOv2-small，使用各自处理器；分别对 input 和 HYPIR 提取 L2 归一化嵌入，并按 input、HYPIR 顺序拼接。
- 统计特征包含双方 RGB mean/std、Canny edge density、Laplacian variance、FFT 高频能量、灰度标准差，以及四个输出-输入残差统计，共 24 维。
- 主模型 fusion=CLIP+DINO+stats；比较 CLIP、DINO、stats 及 LQ-only 消融，不把位置、图像 ID、GT 或 gain 当特征。

## 防泄漏评价
- 外层 5 折 leave-one-image-out；每折四张训练、一张评价，所有重叠 patch 随原图一起划分。
- StandardScaler 仅在训练折拟合；无测试折早停、调参或阈值选择。
- MLP=Linear(64)-ReLU-Linear(1)-Sigmoid；40 epochs，AdamW lr=0.001、weight_decay=0.01、batch=256、固定阈值 0.5。
- 训练 loss 同时平衡原始图像及 GOOD/BAD；记录全部折与 seed。种子 231/232/233 固定，OOF 使用三个模型概率平均，不选择最优 seed。
- 报告各图像 Accuracy/Precision/Recall/F1/ROC-AUC，GOOD 为正类；主指标是有效图像 AUC 的等权宏平均。单类图像 AUC 为 NA。
- 提供常量/训练先验基线、BAD 检测指标和 balanced accuracy，避免高 BAD 比例导致 accuracy 虚高。
- bootstrap 仅重采样五张原图，不重采样 patch；区间是小样本描述，不能宣称总体置信度可靠。
- 所有 confidence 热图来自该原图未参与训练的 OOF 模型；全量最终 checkpoint 不用于报告 OOF 指标。
- 最终 checkpoint 是三种固定 seed 的 fusion MLP 与 scaler 的集合，训练集为全部 validation，仅供后续研究。

## 结论规则与空间分析
- 按指导，fusion 原图宏平均 AUC>0.65 给出继续研究 YES，否则 NO；无论结果如何都不自动启动 Phase 1。
- 补充种子稳定性、有效折数和图像 bootstrap 区间，YES 不代表可直接部署或已证明融合提升。
- 按图像统计 BAD 比例、gain，以及水平/垂直邻接 BAD 一致率和基于本图 BAD 比例的独立期望；重叠导致相关，空间统计仅描述性。
- 20 个失败案例按每图四个 BAD patch，优先严重且互不重叠；人工标注内容和可观察变化，不凭 CLIP 猜测语义类别。
- 完整产物落在本阶段目录；数据路径、模型 revision、命令、耗时及哈希均存清单。
