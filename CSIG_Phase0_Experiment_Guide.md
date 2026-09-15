# CSIG Phase0 实验指导 Prompt

## 任务目标

搭建 CSIG 图像恢复项目 Phase0 实验环境，验证 HYPIR
生成式恢复结果是否存在可预测的可靠性差异。

核心假设：

> HYPIR 并非所有区域都可靠，部分区域会产生
> hallucination（错误生成细节）。如果可以预测哪些区域可信，则可以进一步设计
> confidence gate / adaptive fusion。

Phase0 不训练新的恢复模型，只进行分析验证。

------------------------------------------------------------------------

## 一、目录结构

创建：

    csig_phase0/
    ├── data/
    ├── models/
    ├── checkpoints/
    ├── outputs/
    ├── analysis/
    ├── features/
    ├── scripts/
    └── reports/

------------------------------------------------------------------------

## 二、环境

检查：

    python --version
    nvidia-smi
    nvcc --version

安装：

    torch
    torchvision
    opencv-python
    numpy
    pandas
    scikit-image
    scikit-learn
    matplotlib
    tqdm
    pillow
    transformers
    open_clip_torch

保存 requirements.txt。

------------------------------------------------------------------------

## 三、数据检查

读取：

    data/csig_dataset/

生成：

    reports/dataset_report.md

记录：

-   图片数量
-   分辨率
-   GT情况
-   文件格式

不要修改原始数据。

------------------------------------------------------------------------

## 四、HYPIR Baseline

运行官方 HYPIR inference。

保存：

    outputs/hypir/

记录：

-   checkpoint
-   输入
-   输出
-   推理参数

不要修改模型。

------------------------------------------------------------------------

## 五、实验1：Patch级误差分析

使用 validation 数据。

Patch：

    patch_size=256
    stride=128

保存：

    input patch
    hypir patch
    gt patch

计算：

    gain =
    PSNR(HYPIR,GT)
    -
    PSNR(Input,GT)

标签：

    gain > 0 GOOD
    gain <=0 BAD

输出：

    reports/patch_quality.csv

字段：

    image_id
    patch_id
    x
    y
    psnr_input
    psnr_hypir
    gain
    label

------------------------------------------------------------------------

## 六、实验2：错误预测特征

提取：

### CLIP

保存：

    features/clip.npy

### DINOv2（显存允许）

保存：

    features/dino.npy

### 图像统计

包括：

-   RGB mean/std
-   edge density
-   Laplacian variance
-   FFT 高频能量

------------------------------------------------------------------------

## 七、实验3：Confidence Predictor

目标：

预测 HYPIR patch 是否可靠。

模型：

MLP

结构：

    Feature
     |
    Linear
     |
    ReLU
     |
    Linear
     |
    Sigmoid

输出：

-   Accuracy
-   Precision
-   Recall
-   F1
-   ROC-AUC

保存：

    checkpoints/confidence_mlp.pt

------------------------------------------------------------------------

## 八、可视化

生成：

    reports/visualization/

包含：

1.  HYPIR gain heatmap
2.  confidence heatmap
3.  20个失败案例：

Input / HYPIR / GT

重点观察：

-   植物
-   动物
-   汉字
-   小目标
-   高频纹理

------------------------------------------------------------------------

## 九、最终报告

生成：

    reports/phase0_report.md

回答：

### 1. HYPIR错误是否具有空间规律？

统计：

-   BAD比例
-   失败区域类别

### 2. 是否可以预测？

报告：

    CLIP AUC
    DINO AUC
    Fusion AUC

### 3. 是否值得继续设计 Confidence Gate？

判断：

YES / NO

参考：

ROC-AUC \> 0.65 认为存在进一步研究价值。

------------------------------------------------------------------------

## 十、禁止事项

不要：

1.  修改 HYPIR；
2.  训练 LoRA；
3.  训练 MoE；
4.  使用 test GT；
5.  把 patch 当独立图片统计；
6.  直接设计专家分类。

目标只有：

> 验证恢复可信度是否可预测。

------------------------------------------------------------------------

## 十一、交付

最终生成：

    phase0_done.md

包含：

1.  环境信息
2.  HYPIR结果
3.  Patch统计
4.  Confidence指标
5.  可视化路径
6.  下一阶段建议
