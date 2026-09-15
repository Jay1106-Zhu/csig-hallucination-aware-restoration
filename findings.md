# Phase 0 发现

## Phase 0.5 预检
- 主结果HOLD：V7 ROC-AUC0.404961，LQ-only0.664665；只有case2/case5和3张合成原图有两类，名义55原图不等于55个有效AUC。
- 50%区域coverage下V7−random damage差all=+0.008712（95%CI−0.000947至+0.027273）；real=−0.004167（上界0）；synthetic=+0.01（下界0），未证明显著下降。
- 实际全图融合V7宏PSNR gain−2.089230dB（55/55低于LQ）；oracle_positive仅+0.018888dB；逐像素GT oracle+1.836257dB不能当作可部署门控性能。
- DINO固定处理器会从256中心裁224；512上下文对应原448范围。token可视化遵从真实视野，未拉伸到整个256区域。
- 全图感知指标统一缩至长边1024，区域用原像素；因此LQ/HYPIR的全图LPIPS改善不能直接等同于原生局部纹理准确，主报告分开说明。
- 原生非重叠区域实测：real 960 区域仅 8 个 PSNR 改善、6 个四指标多数改善；synthetic 200 区域仅 5 个 PSNR 改善、22 个四指标多数改善。不能把数据扩展数量误写为有效正样本充足。
- 本阶段的非重叠分区不同于 Phase 0 重叠 patch；旧 59 个 GOOD 不能直接与当前 13 个进行同协议比较。
- 新阶段要求 50–100 个独立场景，四种 gain（PSNR/SSIM/LPIPS/DISTS）、全局/成对/token 多尺度 DINO、结构、controlled stability、分组 verifier、risk–coverage、实际融合与 oracle。
- BSDS500 的 BIDS 官方镜像可用，固定 commit a04b7c6c3a9f0ace74bf205c72a43d32e1c72722。仅用 train 原图，不接触其 test 图像；退化由已知同坐标生成，可直接验证配对。
- BSDS 原始用途是分割，不是低质恢复；将其用作合成扩展必须显著标记 domain=synthetic，不能掩盖真实 paired 数量仍为 5。
- 当前缺已完成的独立人工 severity 标注；因此 Q1/Q2/类别幻觉可检测性在人工审核前不能得出肯定结论。
- 数据已扩充至 55 个源图像（5 real +50 synthetic），五对旧数据的低频配准检查通过，所有 GT hash 唯一，缩略图近重复筛查未触发。
- 已查看外部源图联系表1/2：存在动物、植物、人物、建筑、水岸/雪地等内容，退化配方按固定顺序平衡分配。语义类别是助手目视审阅，不是人工 hallucination 标注。
- 已查看联系表3/4：补充人物、建筑、自然景观、植物、海洋动物和市场小物体；有非中文标牌局部，但没有真实 night/flare 成对场景，不把 synthetic lowlight_glow 当作夜景覆盖达标。

## 公开发布范围
- 后续用户已明确授权上传全部 Phase 0 交付，包括图像、patch、特征和模型；以下轻量发布方案已被此新要求替代。
- 全量目录不适合直接公开：包含约 2 GB 数据/patch、661 MB 权重、运行日志和本机路径。
- 未核验原始竞赛图片的再分发许可，因此不发布原图/GT/恢复图及其对照裁剪；纯数值指标、热图和 ROC 图可作为实验摘要。
- 旧 audit 是完整本地实验的历史验收，公开仓库不包含全部原始产物；发布清单和实验审计分开记录，避免混淆。

- 指导规定 patch=256、stride=128，gain=PSNR(HYPIR,GT)-PSNR(Input,GT)，GOOD iff gain>0。
- 目标仅为可靠性预测，交付 CLIP/DINO/统计特征、MLP、按图像评估及失败案例。
- 旧记录称 validation 5 对、test 100 张 LQ-only，待核实；独立单位必须是原图。
- GPU RTX 5080 Laptop 16GB；已有 .conda 环境和历史 HYPIR 产物。
- 已实测 CUDA 可用：PyTorch 2.11.0+cu128，Python 3.11.16。缺 nvcc 不影响预编译 PyTorch 推理。
- 已确认 validation 为 5 对 RGB JPEG；历史 t50 是调弱变体，本轮固定官方 t200 验证其错误可预测性。
- 模型特征用 input/HYPIR 成对嵌入，另做 LQ-only 消融；特征不含 GT、位置或图像 ID。
- 全流程完成：3565 patches，但仅 59 个 GOOD（case1=0、case2=23、case3=0、case4=1、case5=35）。两张原图没有正例，AUC 仅在三张图上定义。
- Fusion 宏 AUC=0.719535，DINO=0.754799，CLIP=0.664419；但 Fusion 的 case4 AUC=0.973315 只有一个 GOOD，case2=0.620731、case5=0.564560。表面达标高度脆弱，不能宣称 Gate 已被可靠验证。
- Fusion 固定阈值 0.5 时 GOOD recall 很低，必须同时对照 always-BAD 基线，报告探索性门槛和实际推进建议的差别。
- 已查看全图概览：case1 汉字页、case2 书架、case3 海鸥与水面、case4 植被、case5 钟楼。最差 PSNR 的 case1 四例都是空白底色，主要呈现新增颗粒而非字形；需补充目的性语义区域观察，不能把最差 20 例当语义样本分布。
- 人工查看 case2 最差四例：书脊/书页的平滑浅色区域及分界，HYPIR 增强为深色锐边、颗粒，与 GT 分界及纹理不一致。不是字符识别样本。
- 人工查看 case3 最差四例：均为水面背景，HYPIR 生成明显分层硬边、条纹及碎裂纹理，GT 仍为模糊水面；这些不能代表海鸥主体质量。
- case4 最差四例出现大块阔叶及亮线网状叶脉，GT 为明显不同的细长叶片/枝条分布，是较明确的结构不一致案例。case5 最差四例主要是墙面/线框：颗粒、线条锐化、三角形边角，与 GT 局部形态不符。
- 已查看五个补充语义区域和 case4 全图/热图。海鸥主体轮廓基本保留但细密羽毛不同于 GT；植物小目标和叶形明显不同；中文局部增加颗粒和硬边，但没有 OCR 标注，不能编造错字率。
- case4 confidence 几乎全低，不能将其解释为精细语义门控；59 GOOD/3506 BAD 的极端不平衡下，当前 predictor 固定阈值不足以实用。
