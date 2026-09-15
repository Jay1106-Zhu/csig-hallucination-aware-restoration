# GitHub 完整交付

日期：2026-09-15。

- 仓库：`Jay1106-Zhu/csig-hallucination-aware-restoration`，public。
- Release：`phase0-2026-09-15`，public、非草稿。
- Git 直接提供全部代码、配置、原始规划、指标、中文报告，以及 48 张可视化图片。
- Release 提供 5 个完整 ZIP（合计 1434.43 MiB）和 2 个校验清单；245 个包内文件全部列入 SHA-256 清单。
- 包含 Phase 0 指导所需的全部 3565 组 input/HYPIR/GT patch、CLIP/DINO/stats 特征、最终 confidence_mlp.pt、90 个折 checkpoint、官方 HYPIR 输出、20 个失败对照、浮点热图及完整日志。
- 额外提供 5 对 validation LQ/GT 原图和完整冻结 CLIP/DINOv2-small 模型。
- 环境依赖安装目录、下载缓存和凭据不上传；HYPIR 原始大权重记录来源/commit/参数/SHA-256，不复制仓库外部的大模型。

## 验证

- 15 项原实验契约测试、7 项发布/下载安全测试通过。
- 实际解压 predictor-features 包，验证全部 95 个成员和 91 个 checkpoint。
- 导出的最终模型 12 个权重张量及 6 个 scaler 张量与本地逐元素一致；仅去除路径元数据。
- 5 个 ZIP 均通过本地 CRC 和 SHA-256；远端资产大小及 SHA-256 全部匹配。
- Git 发布前扫描通过，未提交凭据、依赖安装目录、下载缓存或本机绝对路径元数据。
- 匿名访问仓库、Release 均返回成功；实际下载 SHA256SUMS.txt 与本地一致。

详见 `artifact_manifest.json`、`SHA256SUMS.txt`、`completeness_verification.json`、`remote_verification.json`。

## 获取

```powershell
git clone https://github.com/Jay1106-Zhu/csig-hallucination-aware-restoration.git new_formal
cd new_formal
python publication/download_artifacts.py --extract
```

需要至少容纳压缩包与解压内容的空间。脚本默认保留已存在且不同的文件，避免覆盖正在进行的本地实验。
