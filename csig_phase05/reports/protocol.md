# Phase 0.5 预先固定协议

日期：2026-09-15。先固定参数、数据抽样及主结果定义，再读取新增候选标签或训练预测器。

## 数据及独立性
- 5 个 CSIG 真实成对场景全部保留；额外从固定 BSDS500 官方镜像 train 中使用 seed=505 随机抽取 50 个源图像，不使用任何 test 图像。
- 每个外部源图像只生成一个 LQ，五种确定性退化配方平衡分配，不依据语义/模型结果选图。新增域明确为 synthetic，不冒充真实 paired 数据。
- 原始图像/scene 是分组单位，源文件 SHA-256 和低分辨率图像相似性检查防止副本伪装成新样本；语义类别允许 unknown，不使用模型标签假装人工确认。
- 不改写数据，不重配准旧数据。先审计尺寸、配对来源、低频平移与候选哈希；若存在明确错配则停止并记录。

## 标签和人工审核
- 四种 gain 均统一为正值改善：PSNR/SSIM=candidate-LQ，LPIPS/DISTS=LQ-candidate。
- 主预测标签固定为 psnr_gain>0，另外评价四指标多数改善的 usefulness proxy；这些不是人工 hallucination 真值。
- 每 scene 由 seed 固定选两个互不重叠区域，盲展示 LQ/candidate/GT，不显示 gain、特征、预测或按错误选样。
- 人工 severity=0/1/2/3 和类型独立保留。没有人的审核就保持 pending，不从 PSNR、DINO 或助手判断生成“人工真值”。不调 lambda。

## 特征与分组
- 全图分为不重叠有效区域，默认 256；末端区域以其真实大小计算 gain，特征提取用中心反射扩展的 256/512 上下文，明确记录有效范围和 padding。
- 冻结 DINOv2-small 与 CLIP，使用 Phase 0 同一模型/处理器；不更新权重。DINO 保留 16×16 tokens，保存位置对应、最近邻、局部匹配、位移和分歧统计。
- V1 global concat；V2 pairwise；V3 token summary（256+512）；V4 structure；V5 controlled sensitivity；V6 V2+V4；V7 V2+V4+V5（主模型）；V8 CLIP+DINO+stats 同配方但在新区域/新分组重新拟合。
- C1 补充 LQ-only、restored-only、absolute difference；C3 分别评价 256 与 512 的 token summary。
- 至少 30 个 scene 时用 5 折 GroupKFold；同 scene 的全部区域始终同折。标准化只训练折拟合。模型 Logistic(C=0.1) 以及 Tiny MLP(32隐藏层,30 epochs,三固定seed)，不根据测试折挑阈值/seed/模型。
- 训练 loss 先按 scene 平衡，类别平衡如使用必须只取训练标签；主模型固定为 V7 logistic，Tiny MLP 为补充，不挑最佳。

## 稳定性
- 官方 HYPIR 为单次确定性恢复，不假设 seed 是随机不确定性。固定 model_t=200，coeff_t=50/100/150/200，空 prompt、seed231、tile512/stride256。
- 对旧五图复用已存在的同参数历史输出并核验输入/源码/参数/哈希；只对新增图运行官方入口。
- 指标包括条件间像素方差、边缘方差、相对主候选 LPIPS、DINO CLS/token 方差；称 controlled sensitivity，不称 Bayesian uncertainty。

## 评价、融合及结论
- ROC-AUC、PR-AUC（average precision）、balanced accuracy、precision/recall/F1、Brier、10-bin ECE，先逐原图再按 scene 宏平均；只有一类的图像 AUC/AP 不定义，报告有效数。
- 真实 CSIG 和合成 BSDS 分域；另外 synthetic-only 与 synthetic-trained→CSIG 的域外检查，不用 synthetic 提升掩盖真实域失败。
- coverage=10%…100%；每张图按 OOF confidence 排序，固定 50% 为主点。报告实际区域/像素 coverage、selected gain、damage、四种质量指标；无人工标签的 hallucination rate 为 NA。
- 实际融合使用不重叠有效区域的 binary mask：LQ、HYPIR/Always HYPIR、predicted、固定seed random、region oracle；额外逐像素 SSE oracle 作为 PSNR 数学上界。
- 实际融合的全图 PSNR/SSIM 用原尺寸；LPIPS/DISTS 的 full-image 评价统一长边≤1024，区域指标使用原区域（过小尺寸按固定规则放大）。与未缩放像素指标分列。
- 主比较是 coverage50 的 predicted−random damage-rate，按 scene 成对 bootstrap（2000次），真实/合成分别报告；所有 coverage 展示但不挑一个好看的点宣称显著。
- 成功必须同时有独立数据、稳定 candidate-aware 增益、实际损害降低、非单图支撑和人工标签证据。人工审核未完成或真实域不足时结论最多 HOLD，不进入 Phase 1。

## 实施补记（不改主模型/阈值）

- 按原指令第十七节Q5补全 `C5_pair_stability`=V2+stability。它在首轮主结果后补跑，明确为补充消融，不改变固定 V7/coverage50 验收，也不据此挑模型。
- 冻结DINO原处理器的short edge256→center crop224在256/512上下文上分别对应224/448源像素；token映射保留中心真实范围，不把输出拉伸成整个上下文。
- 实际全图融合计算固定主点50%，同时10%–100%的OOF risk–coverage全部输出；二者分开，未把patch平均指标称作全图融合指标。
- SSIM为RGB逐通道11×11 Gaussian(sigma1.5)、valid中心均值；PSNR为RGB原始像素MSE，不裁图边。LPIPS输入[-1,1]，DISTS输入[0,1]；区域最短边实际>=65未放大，全图感知指标统一长边<=1024。
