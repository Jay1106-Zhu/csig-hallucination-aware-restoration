# Phase 0.5 全量公开交付

本轮按用户“和上次一样，全部上传”的授权发布到同一个 **public** 仓库，新增独立 Release `phase05-2026-09-15`，不替换上次 Phase0 Release。

Git存源码、配置、报告、原始指标表、全部轻量图表和110份盲标注界面图片。Release存完整数据、patch、全量特征、324个验证器checkpoint、220个候选、275张融合及mask、全部日志/分析和冻结模型权重，不是仅上传示例或文档。

## 一次下载与恢复

```powershell
git clone https://github.com/Jay1106-Zhu/csig-hallucination-aware-restoration.git new_formal
cd new_formal
python publication/phase05/download_phase05.py --extract
```

需要系统`curl`，下载可断点续传，整包SHA-256和逐文件清单校验。默认不覆盖已有不同文件；只有明确需要覆盖自己的目标目录时才加`--overwrite`。可用`--package phase05-features.zip`选择单包。

Release页面：`https://github.com/Jay1106-Zhu/csig-hallucination-aware-restoration/releases/tag/phase05-2026-09-15`。

## 完整分包

| 文件 | 内容 |
| --- | --- |
| `phase05-data-patches.zip` | BSDS train原图、合成LQ、真实CSIG原图/GT副本、无损patch、所有manifest、110份盲标注 |
| `phase05-features.zip` | 全量DINO多尺度空间tokens、CLIP/global/pairwise/structure/stability、特征schema与索引 |
| `phase05-verifiers.zip` | 324个轻量验证器checkpoint与训练/验证scene索引，无新恢复模型 |
| `phase05-hypir-outputs.zip` | coeff50/100/150/200的所有冻结HYPIR输出和随附文件 |
| `phase05-selective-outputs.zip` | 全量predicted/random/oracle融合及mask |
| `phase05-analysis-reports.zip` | 所有分析、原始表、原始热图、89张可视化、日志、报告、实验代码/测试/配置 |
| `phase05-shared-encoders.zip` | 原Phase0固定CLIP/DINO编码器权重及处理器，解压至原相对目录供只读复用 |
| `phase05-perceptual-models.zip` | AlexNet/VGG16冻结评价骨干、LPIPS校准参数、DISTS参数，含完整SHA-256 |

另有`artifact_manifest.json`和`SHA256SUMS.txt`，每包、每个成员、原本地文件均可追溯；完整性覆盖见`completeness_verification.json`。

## 公开导出与原始实验的区别

- 不改写本地Phase0/Phase05科学实验内容，仅公开副本把本机工作区/主目录替换为`${CSIG_WORKSPACE}`/`${USER_HOME}`；真实原图manifest路径改成包内`data/csig_dataset/验证集/`。
- `source_sha256`保留原文件摘要，`sha256`校验实际公开成员；内嵌实验inventory及immutable审计仍是执行时原文件哈希，**不能把它当成脱敏副本的下载校验清单**。
- 实验文档内“未发布”是计算完成时的历史状态；当前发布状态以本目录`GITHUB_DELIVERY.md`/`remote_verification.json`为准。
- 依赖安装树、无关缓存、凭据和上传临时状态不公开。所有模型/数据不新增或改变原权利许可；使用者仍须遵循CSIG/BSDS及上游模型适用条款。
- HYPIR恢复权重并非本阶段新训练产物，不复制整个旧工程/恢复权重；官方来源与参数已记录，需重跑生成时另行按HYPIR上游配置。CLIP/DINO和新评价权重已完整附带。

## 复现实验边界

当前结果是 **HOLD**。5个真实+50个合成源图不等于55个真实配对场景；110条人工幻觉标注仍为pending，发布并不使人工审核完成。没有Phase1训练。

完整历史数字审计在`csig_phase05/analysis/artifact_audit.json`。公开副本先用本发布清单核验；本机绝对路径/只读旧阶段依赖哈希不能在新机器上逐字复刻。324个checkpoint仍含原生权重、scaler及分组索引，公开导出不变更任何数值张量。

要重新计算LPIPS/DISTS，可将`models/perceptual/torch_hub`两个骨干文件放入当前用户torch hub的`checkpoints`目录，安装固定`DISTS-pytorch==0.1`与`lpips==0.1.4`。主实验依赖版本在`csig_phase05/requirements.txt`；不直接将整棵本地依赖树上传。
