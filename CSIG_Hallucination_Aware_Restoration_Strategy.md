# CSIG 图像恢复项目规划与策略文档

## Semantic-Controlled Hallucination-Aware Restoration Framework

------------------------------------------------------------------------

## 1. 项目背景

当前基于 HYPIR
的生成式图像恢复模型能够提升视觉质量，但实验发现其存在明显的生成幻觉问题：

-   绿植区域出现错误叶型、花序和不存在纹理；
-   动物区域主体保持较好，但复杂背景容易产生错误结构；
-   中文文字存在笔画错误和结构丢失；
-   夜景存在 flare、glow、低光污染；
-   远景裁剪放大区域存在严重信息缺失。

核心问题：

> 当前生成式恢复模型并不知道什么时候应该相信自己的生成结果。

因此目标从"生成更多细节"转向：

> 生成可信细节，并控制错误幻觉。

------------------------------------------------------------------------

# 2. 核心思想

提出：

**Semantic-Controlled Hallucination-Aware Restoration Framework**

整体流程：

    Low Quality Image
            |
    Semantic Understanding + Degradation Analysis
            |
    Restoration Experts
            |
    Candidate Restoration
            |
    Reliability Gate
            |
    Final Output

------------------------------------------------------------------------

# 3. 总体架构

## 3.1 Semantic Encoder

负责理解：

-   text
-   animal
-   vegetation
-   person
-   object

同时提取：

-   edge
-   contour
-   structure

作用：

不是决定恢复模型，而是约束恢复行为。

------------------------------------------------------------------------

## 3.2 Degradation Encoder

分析：

-   low resolution
-   motion blur
-   defocus
-   flare
-   noise
-   low light

输出连续退化表示，而不是简单分类。

------------------------------------------------------------------------

## 3.3 Restoration Experts

不按照语义划分专家，而按照恢复能力划分：

### Detail Restoration Expert

解决：

-   远景裁剪
-   数字变焦
-   信息缺失

### Structure Restoration Expert

解决：

-   边缘
-   轮廓
-   中文字形

### Optical Restoration Expert

解决：

-   flare
-   glare
-   夜景污染

### Motion Restoration Expert

解决：

-   motion blur
-   defocus

------------------------------------------------------------------------

# 4. Semantic Constraint

## 植物

目标：

减少：

-   假叶片
-   假花序
-   错误纹理

方法：

-   vegetation mask
-   edge constraint
-   structure loss

------------------------------------------------------------------------

## 动物

目标：

保护：

-   主体轮廓
-   羽毛结构

方法：

-   instance mask
-   boundary constraint

------------------------------------------------------------------------

## 中文文字

目标：

恢复：

-   笔画
-   字形

方法：

-   OCR feature
-   glyph structure

指标：

-   OCR accuracy

------------------------------------------------------------------------

# 5. Reliability Gate（核心创新）

恢复模型输出候选结果：

    Candidate Restoration

但候选结果可能包含：

-   真实恢复细节
-   幻觉细节

因此增加：

Reliability Gate。

输出：

    alpha map

最终：

    Output =
    alpha * Candidate
    +
    (1-alpha) * Baseline

Gate 学习目标：

不是判断类别，而是判断：

> 该区域恢复结果是否真正改善。

------------------------------------------------------------------------

# 6. 实验规划

## Phase 0：验证 Gate 可行性

目标：

验证 HYPIR 错误是否可预测。

流程：

1.  获取 HYPIR 输出；
2.  Patch 切分；
3.  对比 GT；
4.  标记：

-   good restoration
-   bad restoration

训练：

Confidence Predictor。

------------------------------------------------------------------------

## Phase 1：Expert Benefit Analysis

建立 Candidate Pool：

  Candidate        作用
  ---------------- ----------
  HYPIR Baseline   基础恢复
  Detail Adapter   细节恢复
  Text Adapter     文字恢复
  Blur Adapter     运动恢复
  Flare Adapter    光学恢复

计算：

不同候选相对 GT 的收益。

------------------------------------------------------------------------

## Phase 2：Adapter Training

训练：

Shared Restoration Adapter。

目标：

降低：

-   错误纹理
-   结构错误

Loss：

-   L1 Loss
-   Structure Loss
-   Perceptual Loss

------------------------------------------------------------------------

## Phase 3：Expert Training

训练：

-   Detail Expert
-   Text Expert
-   Blur Expert
-   Optical Expert

验证专家互补性。

------------------------------------------------------------------------

## Phase 4：Reliability Gate Training

冻结专家。

训练：

Gate 预测：

-   哪个候选更可靠；
-   哪些区域应该采用增强。

Loss：

-   improvement regression
-   ranking loss
-   spatial consistency

------------------------------------------------------------------------

# 7. 关键消融实验

## 语义约束

比较：

    Baseline
    +
    Semantic Constraint

------------------------------------------------------------------------

## 多专家

比较：

    Single Restoration
    vs
    Multi Expert

------------------------------------------------------------------------

## 门控方式

比较：

    Classification Gate
    vs
    Gain-based Gate

------------------------------------------------------------------------

## 空间控制

比较：

    Global Gate
    vs
    Spatial Gate

------------------------------------------------------------------------

## 是否允许拒绝增强

比较：

    Always Enhance
    vs
    Confidence Controlled

------------------------------------------------------------------------

# 8. 数据策略

构建 CSIG-like 数据集。

覆盖：

## Semantic

-   animal
-   vegetation
-   text
-   urban

## Degradation

-   low resolution
-   blur
-   flare
-   noise
-   low light

采用：

真实退化 + 合成退化。

重点构造：

    animal + blur
    vegetation + low resolution
    text + flare
    night + zoom

------------------------------------------------------------------------

# 9. 评价指标

基础：

-   PSNR
-   SSIM
-   LPIPS

结构：

文字：

-   OCR Accuracy

动物：

-   segmentation consistency

植物：

-   edge consistency

幻觉：

-   错误结构数量
-   不存在纹理统计

------------------------------------------------------------------------

# 10. 开发优先级

第一阶段：

验证 Reliability Gate。

第二阶段：

训练 Shared Adapter。

第三阶段：

加入 Text + Structure Branch。

第四阶段：

加入 Multi Expert。

第五阶段：

加入 Semantic Spatial Constraint。

------------------------------------------------------------------------

# 总结

本项目目标不是设计一个更强的增强模型，而是：

> 构建一个能够理解图像内容、分析退化，并控制生成细节可信度的自适应恢复系统。

核心方向：

-   Semantic-aware Restoration
-   Degradation-aware Routing
-   Mixture of Experts
-   Reliability-aware Fusion
-   Hallucination Suppression

核心创新：

> 从"生成更多细节"转向"生成可信细节"。
