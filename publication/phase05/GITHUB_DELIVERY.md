# Phase 0.5 全量公开交付记录

状态：分包与本地覆盖检查通过，正在推送Git与上传独立Release；尚不提前声明远端验证完成。

- 目标仓库：`Jay1106-Zhu/csig-hallucination-aware-restoration`（已确认public）。
- 新Release：`phase05-2026-09-15`；旧`phase0-2026-09-15`保持不动。
- 上传范围：全部1868个Phase05文件，另外10张真实LQ/GT、6个冻结编码器文件、4个冻结评价模型文件；不公开安装依赖、凭据与上传临时状态。
- 8个ZIP分包及两个清单；所有包和成员均有SHA-256，详见`artifact_manifest.json`。
- 人工标注仍是110条pending，结论HOLD；公开交付并不等于科学假设通过。

远端发布后，`remote_verification.json`记录逐资产GitHub全文件SHA256/大小、匿名实际下载验证，以及旧Phase0资产未变的证据。
