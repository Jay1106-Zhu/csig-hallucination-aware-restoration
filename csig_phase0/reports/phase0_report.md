# CSIG Phase 0：HYPIR 可靠性可预测性验证

执行日期：2026-09-15。状态：实验已运行；只训练 Confidence MLP，没有训练恢复模型。

## 1. 结论与适用范围

**继续研究 Confidence Gate：YES**。预先固定的 fusion 原图宏平均 ROC-AUC=0.7195，阈值为严格 >0.65。
CLIP AUC=0.6644；DINO AUC=0.7548；Fusion AUC=0.7195。
独立验证单位只有 **5 张原图**（3565 个重叠 patch 不是独立图片）；有效 AUC 图像数=3。
Fusion 图像级描述性 bootstrap 95% 区间：[0.564559629161399, 0.9733146067415731]。小样本和同域相关性使该区间不能支持部署保证。
**实际推进建议：HOLD / 不进入 Gate 或专家训练，先补充独立数据。** YES 仅表示符合指导的探索性数值门槛，不等于本假设已获可靠证据。
全体只有 59 个 GOOD：case1/case3 为零，case4 仅一个；去掉只有一个 GOOD 的 case4 后，Fusion 在其余两个可评价原图的宏 AUC=0.5926。
这一事后敏感性分析不用于挑模型、改阈值或替换主指标，只说明主指标对极少正例图像的脆弱性。
本实验预测的是相对输入的 PSNR 收益，不能证明语义真实性、生成幻觉检测准确率或融合后性能提升。

## 2. 环境与官方推理

- Python：3.11.16 | packaged by Anaconda, Inc. | (main, Aug 27 2026, 14:36:16) [MSC v.1942 64 bit (AMD64)]；GPU：NVIDIA GeForce RTX 5080 Laptop GPU；PyTorch：2.11.0+cu128；CUDA runtime：12.8。
- nvcc 不在 PATH；没有编译自定义 CUDA 算子，PyTorch CUDA 推理与特征提取实测可用。
- scikit-learn 等缺失依赖安装在阶段 dependencies/；环境版本见 requirements.txt 与 analysis/environment.json。
- 官方入口 HYPIR/test.py；仓库 commit=b61d107c6cef38f01a93c7833558869731cfa8c1；本轮重新运行，耗时 347.41 秒。
- 固定 model_t=200、coeff_t=200、推理 tile=512/stride=256、upscale=1、空 prompt、seed=231；加载现有官方预训练 LoRA 权重不等于训练 LoRA。
- 五张原尺寸 RGB PNG 位于 outputs/hypir/result/；完整命令与输入/输出/权重见 analysis/hypir_inference.json、data/manifest.json。
- 原数据、官方源码和恢复权重 35 个文件前后校验：all_unchanged=True。

### 原图 PSNR（dB，等权按图像）

| image_id | psnr_input | psnr_hypir | gain |
|---|---|---|---|
| case1 | 32.0288 | 29.3353 | -2.6935 |
| case2 | 28.1894 | 24.2140 | -3.9754 |
| case3 | 35.6221 | 28.8654 | -6.7567 |
| case4 | 18.0562 | 16.3524 | -1.7038 |
| case5 | 26.2756 | 23.8758 | -2.3998 |

原图平均 Input=28.0344；HYPIR=24.5286；平均 gain=-3.5058 dB。
原图 PSNR 不等于 patch PSNR 的平均值。主模型采用官方 t200；此报告不外推到历史调弱 t50 或其它恢复器。

## 3. 数据与标签

- validation 为 5 对 RGB JPEG，4 张 4096×3072、1 张 3072×4096；test 100 张 LQ 只检查图片头部，不提取 patch、不参与标签或训练。
- patch_size=256、stride=128，每图 713 个，合计 3565；无缩放、无 GT 特征、无随机 patch 划分。
- GOOD：PSNR(HYPIR,GT)-PSNR(Input,GT)>0；其它为 BAD。GT 限于标签、离线评价和对照图。
- 所有 input/hypir/gt patch 以逐图无损 uint8 NPY 保存，位置索引在 features/index.csv，读取方法见 README。

## 4. HYPIR 错误是否有空间规律？

| image_id | good_count | bad_count | bad_fraction | mean_patch_gain | adjacent_same_label | independent_expected_agreement | agreement_excess |
|---|---|---|---|---|---|---|---|
| case1 | 0 | 713 | 1.0000 | -4.6743 | 1.0000 | 1.0000 | 0.0000 |
| case2 | 23 | 690 | 0.9677 | -3.7271 | 0.9614 | 0.9376 | 0.0238 |
| case3 | 0 | 713 | 1.0000 | -8.7291 | 1.0000 | 1.0000 | 0.0000 |
| case4 | 1 | 712 | 0.9986 | -1.6235 | 0.9971 | 0.9972 | -0.0001 |
| case5 | 35 | 678 | 0.9509 | -2.3175 | 0.9388 | 0.9066 | 0.0321 |

原图等权平均 BAD 比例=98.3450%。邻接一致率减去基于各图类比例的独立期望，平均超额=0.0112。
热图与邻接统计用于描述空间分布；patch 有 50% 重叠，邻接相关性并不等于独立的空间显著性检验。
不能把“大片区域都是 BAD”直接归因于某个语义类别，也不能以此证明 hallucination 的位置可泛化预测。

## 5. 是否可以预测？

- 外层 Leave-One-Image-Out：4 张训练/1 张评价，共 5 折；每折仅在训练图像拟合 scaler。
- MLP：Linear(64)→ReLU→Linear(1)→Sigmoid；40 epochs；AdamW lr=0.001、weight_decay=0.01；batch=256。
- 三个固定 seed=231/232/233，OOF 概率均值；无测试折调参、早停、阈值搜索或最佳 seed 选择。
- 训练 loss 按原图和标签平衡；指标为原图等权平均，GOOD 为正类，阈值固定 0.5。
- CLIP/DINO 使用冻结 input 与 HYPIR 的成对特征；fusion_lq 只使用输入，可检验输出感知相对输入特征的差别。

| Features | dim | accuracy | precision | recall | f1 | roc_auc | balanced_accuracy | bad_recall | bad_f1 | AUC_image_bootstrap_95 |
|---|---|---|---|---|---|---|---|---|---|---|
| clip | 1024 | 0.9495 | 0.0182 | 0.0087 | 0.0118 | 0.6644 | 0.5046 | 0.9657 | 0.9735 | 0.5055–0.9719 |
| dino | 768 | 0.9562 | 0.0308 | 0.0348 | 0.0327 | 0.7548 | 0.5234 | 0.9714 | 0.9773 | 0.5721–0.9565 |
| stats | 24 | 0.7930 | 0.0770 | 0.0927 | 0.0290 | 0.6588 | 0.5234 | 0.8047 | 0.8760 | 0.5394–0.8792 |
| fusion | 1816 | 0.9419 | 0.0140 | 0.0261 | 0.0182 | 0.7195 | 0.5121 | 0.9573 | 0.9698 | 0.5646–0.9733 |
| stats_lq | 10 | 0.8104 | 0.0026 | 0.2174 | 0.0050 | 0.5097 | 0.5766 | 0.8237 | 0.8892 | 0.3153–0.7851 |
| fusion_lq | 906 | 0.8014 | 0.0250 | 0.0087 | 0.0129 | 0.5599 | 0.5056 | 0.8176 | 0.8386 | 0.4466–0.6233 |

### Fusion 逐原图 OOF 指标

| heldout_image | accuracy | precision | recall | f1 | roc_auc | balanced_accuracy | bad_recall | good_count | bad_count |
|---|---|---|---|---|---|---|---|---|---|
| case1 | 0.9144 | 0.0000 | 0.0000 | 0.0000 | NA | NA | 0.9144 | 0 | 713 |
| case2 | 0.9158 | 0.0698 | 0.1304 | 0.0909 | 0.6207 | 0.5362 | 0.9420 | 23 | 690 |
| case3 | 0.9299 | 0.0000 | 0.0000 | 0.0000 | NA | NA | 0.9299 | 0 | 713 |
| case4 | 0.9986 | 0.0000 | 0.0000 | 0.0000 | 0.9733 | 0.5000 | 1.0000 | 1 | 712 |
| case5 | 0.9509 | 0.0000 | 0.0000 | 0.0000 | 0.5646 | 0.5000 | 1.0000 | 35 | 678 |

### 简单基线（同样按图像宏平均）

| baseline | accuracy | precision | recall | f1 | roc_auc | balanced_accuracy | bad_recall |
|---|---|---|---|---|---|---|---|
| always_BAD | 0.9835 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 1.0000 |
| training_prior | 0.9835 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 1.0000 |

Accuracy 必须结合类别不均衡解读。AUC 衡量排序而非校准；训练使用类别重加权，因此输出虽然记为 P(GOOD)，不应直接当作已校准概率。

### 类别稀缺与敏感性分析

- case1/case3 的 AUC 不定义，不能填 0.5 或把它们计入 AUC 均值；主 AUC 因此是三个有效原图的等权平均。
- case4 的高 AUC 只依靠一个 GOOD 的排序位置；不是稳定的场景级泛化证据。
- bootstrap 实际只对三个 AUC 有定义的原图重采样；不对 patch 重采样，不为缺失 AUC 造值。与五图总体相比覆盖有限，属于执行时的必要缺失值处理。
- 以下仅展示至少两个 GOOD 且至少两个 BAD 的图像：

| feature | sensitivity_auc | eligible_images |
|---|---|---|
| clip | 0.5107 | 2 |
| dino | 0.6540 | 2 |
| fusion | 0.5926 | 2 |
| fusion_lq | 0.6166 | 2 |
| stats | 0.5487 | 2 |
| stats_lq | 0.3720 | 2 |

- 主 Fusion 的 GOOD precision/recall/F1 很低，always-BAD 的 accuracy 更高；该概率图不能直接作为拒绝增强或融合权重部署。

### Seed 稳定性

| feature | seed | roc_auc | accuracy | f1 | bad_recall |
|---|---|---|---|---|---|
| clip | 231 | 0.6653 | 0.9422 | 0.0125 | 0.9584 |
| clip | 232 | 0.6868 | 0.9574 | 0.0121 | 0.9735 |
| clip | 233 | 0.6643 | 0.9501 | 0.0300 | 0.9656 |
| dino | 231 | 0.7397 | 0.9411 | 0.0340 | 0.9563 |
| dino | 232 | 0.7371 | 0.9613 | 0.0320 | 0.9765 |
| dino | 233 | 0.7662 | 0.9565 | 0.0320 | 0.9717 |
| fusion | 231 | 0.7138 | 0.9380 | 0.0171 | 0.9533 |
| fusion | 232 | 0.7239 | 0.9296 | 0.0145 | 0.9448 |
| fusion | 233 | 0.7373 | 0.9461 | 0.0207 | 0.9616 |
| fusion_lq | 231 | 0.6041 | 0.7992 | 0.0138 | 0.8154 |
| fusion_lq | 232 | 0.5532 | 0.8028 | 0.0121 | 0.8190 |
| fusion_lq | 233 | 0.5278 | 0.7994 | 0.0138 | 0.8157 |
| stats | 231 | 0.6423 | 0.8008 | 0.0291 | 0.8125 |
| stats | 232 | 0.6530 | 0.7935 | 0.0273 | 0.8051 |
| stats | 233 | 0.6424 | 0.7835 | 0.0586 | 0.7939 |
| stats_lq | 231 | 0.5231 | 0.8208 | 0.0050 | 0.8341 |
| stats_lq | 232 | 0.4909 | 0.7734 | 0.0042 | 0.7866 |
| stats_lq | 233 | 0.5096 | 0.8238 | 0.0047 | 0.8373 |

## 6. 失败案例与可视化

- gain/confidence：reports/visualization/case*_gain_heatmap.png、case*_confidence_heatmap.png。
- confidence 图全部来自该原图完全未参与训练的 OOF 模型；重叠处取平均，原尺寸浮点图同时保存为 NPY。
- gain PNG 统一色轴 [-10,10] dB（超界截色，NPY 保留真实值）；confidence 色轴 [0,1]。
- 20 个 Input/HYPIR/GT 对照：failure_case*_*.png；每图四个，按最差 gain 选择且在本图不重叠。
- 每图拼图：failures_case*.png；全图+热图：case*_overview.png；逐图 ROC：fusion_roc_by_image.png。
- 选择规则只用于展示，不构成语义类别分布的无偏估计。失败区域类别来自人工查看，不使用自动类别预测代替观察。

- 补充语义观察：supplementary_semantic_gallery.png，覆盖汉字、书脊、动物头部、植被和钟表局部；坐标事先从输入概览人工指定，不参与模型选择。
### 人工观察记录

| patch_id | category | observation | scope |
|---|---|---|---|
| case1_y2176_x1920 | 平滑页面背景 | HYPIR 引入密集细颗粒而输入和 GT 较平滑 | 最差PSNR样本不含汉字 |
| case1_y2048_x2560 | 平滑页面背景 | HYPIR 底色略偏且有均匀颗粒 | 不能归类为字形错误 |
| case1_y2304_x2432 | 平滑页面背景 | HYPIR 生成细密纹理与 GT 平滑区域不一致 | 相对PSNR损害 |
| case1_y1920_x2304 | 平滑页面背景 | HYPIR 新增背景颗粒而 GT 无相同细节 | 相对PSNR损害 |
| case2_y3200_x2688 | 书脊边界 | HYPIR 将模糊分界生成深色硬边并新增颗粒 | 不含可评价的完整汉字 |
| case2_y3456_x2688 | 书脊边界 | HYPIR 生成高对比深色直线与 GT 边界观感不同 | 锐化不等于真实恢复 |
| case2_y3200_x2432 | 书脊边界 | HYPIR 强化双边界及细纹理且局部色调改变 | 相对PSNR损害 |
| case2_y2432_x2432 | 书脊边界 | HYPIR 线条更细黑锐利且平面有颗粒 | 与GT结构强度不匹配 |
| case3_y1664_x2944 | 水面高频背景 | HYPIR 新增分层细纹和锐利浪脊但 GT 仍模糊 | 不是海鸥主体样本 |
| case3_y1408_x1152 | 水面高频背景 | HYPIR 生成人为分层条带和碎裂状纹理 | 不能证明生成纹理真实 |
| case3_y1024_x1152 | 水面高频背景 | HYPIR 新增细碎网纹及硬边与 GT 不一致 | 背景生成不可靠 |
| case3_y1920_x2944 | 水面高频背景 | HYPIR 生成深色锐边和刻纹状浪脊 | 背景生成不可靠 |
| case4_y2816_x1152 | 植物叶片 | HYPIR 形成大块阔叶及不同叶脉而 GT 叶片布局不同 | 可见明显结构不一致 |
| case4_y2688_x0000 | 植物叶片 | HYPIR 生成大幅网格叶脉与 GT 较小叶片结构不符 | 可见明显结构不一致 |
| case4_y0640_x1920 | 植物叶片 | HYPIR 增加亮线叶脉和叶片形态且不对应 GT 枝叶 | 可见明显结构不一致 |
| case4_y1408_x0768 | 植物枝叶 | HYPIR 生成宽大片叶和亮边与 GT 细长叶片不符 | 不能将清晰感等同于忠实度 |
| case5_y2560_x2304 | 建筑边框 | HYPIR 在竖线底部生成尖锐三角连接而 GT 不同 | 局部结构失真 |
| case5_y0000_x2944 | 建筑平面 | HYPIR 表面颗粒更重且线框更硬 | 相对PSNR损害 |
| case5_y0000_x0000 | 建筑平面与竖边 | HYPIR 新增密集竖向纹理和局部反差 | 相对PSNR损害 |
| case5_y2560_x1920 | 建筑小边角 | HYPIR 将模糊小凸起生成锐利不规则尖角 | 小结构真实性无保证 |

以上是选出的 20 个失败案例内容，不是全体 BAD 区域的语义标注；未出现的类别不能推断安全。

### 补充语义区域人工观察

观察文件：`reports/visualization/supplementary_semantic_gallery.png`。对应坐标、PSNR gain 和 OOF 分数存于 `analysis/supplementary_regions.csv`。

这五个区域依据输入全图概览按内容目的性选取，不用于统计类别比例，也不参与阈值、模型或训练参数选择。

| 区域 | 可观察变化 | 不能据此推断的内容 |
|---|---|---|
| case1 中文局部 | HYPIR 笔画边缘更锐，但笔画表面与底色产生颗粒，局部形态与 GT 不完全一致 | 当前 patch 仅含部分字形，未测 OCR 准确率，不能计作已证实的错字率 |
| case2 书脊局部 | HYPIR 将模糊边界变为深色锐边，字形只见局部，GT 的对比度与纹理不同 | 不能从不完整字形判断整本书标题是否恢复正确 |
| case3 海鸥头部 | HYPIR 保留大体头部、眼睛和喙的位置，新增细密羽毛和眼周结构；细节与 GT 不完全对应 | 不能将清晰的羽毛自动视为真实；也不能把主体保留等同于背景安全 |
| case4 植被/粉色小目标 | HYPIR 输出的阔叶、叶脉及粉色目标结构与 GT 细长叶片和目标形态明显不同 | 不做物种、花/果实类型断言；现象是形态不一致，不是分类结论 |
| case5 钟面与指针边缘 | HYPIR 增加暗背景颗粒，指针边缘更硬，局部轮廓变化 | 未测读时正确率，不能推断整体钟面或罗马数字恢复成功 |

所有五个补充区域的 PSNR 标签都是 BAD，OOF 分数也低，但这是少量目的性例子，不构成语义类别上的预测能力统计。

### 空间结论边界

case4 概览显示 confidence 几乎整体很低：这符合当前极端 BAD 占比，不意味着模型已学到精细的语义选择机制。case1/case3 的所有 patch 都是 BAD，二分类 AUC 在这些图像上不可定义。水面、植物、页面、建筑区域表现出不同的损害形式，但只有五个场景，不能分离语义类别、拍摄条件、原图身份和恢复强度的影响。


## 7. 下一阶段建议与停止条件

- YES 只表示达到探索性排序门槛；先扩大独立、成对、多场景 validation，再讨论 Confidence Gate。
- 保留全量训练的 checkpoints/confidence_mlp.pt 供复现；它用过全部五张图，不能用于本报告的验证指标，也不是部署模型。
- 若继续：先补充独立原图，固定完全未参与开发的验证集；增加局部感知/语义真实性标注，确认图像配准和 PSNR 定义适用性。
- 仅当扩大数据后仍稳定预测收益，再授权 Phase 1 Expert Benefit Analysis；不得直接训练 LoRA/MoE、设计语义专家分类。
- 暂未实施 fusion/gate 的实际图像融合，因此没有输出画质提高的证据；本轮到 Phase 0 停止。

## 8. 证据索引与复现

| 证据 | 发现 | 行动 |
|---|---|---|
| data/manifest.json、reports/dataset_report.md | 只有五个独立配对场景 | 所有划分按图像，结论保留小样本限制 |
| reports/patch_quality.csv、analysis/spatial_statistics.csv | patch 收益及空间分布 | 只定义相对输入 PSNR 标签，不声称语义幻觉 |
| analysis/folds.json、analysis/oof_predictions.csv、reports/confidence_metrics.csv | 原图留一预测能力 | 按预设 AUC 门槛决定研究价值 |
| analysis/immutable_after.json | 原始工程只读 | 保留来源与输出哈希 |
- 单元测试日志：logs/unit_tests.log；实际运行日志：logs/pipeline.out.log、logs/hypir_inference.log。
- 独立产物审计：analysis/artifact_audit.json；核验全部 3565 组三源 patch 与源像素/标签、90 个折 checkpoint 的训练图像/scaler/OOF，并重算全部逐折指标。
- 15 项契约测试覆盖 patch 边界、严格标签、原图划分、单类 AUC、训练 scaler、特征对齐、热图重叠、失败区域与 checkpoint 泄漏检查。
- 配置：configs/phase0.json；协议：reports/protocol.md；入口：scripts/prepare_phase0.py、scripts/run_phase0.py、scripts/report_phase0.py。
- 特征模型仓库与 revision：
  - openai/clip-vit-base-patch32 @ 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
  - facebook/dinov2-small @ ed25f3a31f01632728cabb09d1542f84ab7b0056
- 原始依据为 new_formal 根目录的两份方案；模型来源标识及哈希由官方下载 API 和本地文件校验得到。
