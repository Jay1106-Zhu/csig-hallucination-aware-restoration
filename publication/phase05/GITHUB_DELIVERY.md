# Phase 0.5 全量公开交付记录

状态：已完成公开推送与独立Release上传，远端验证全部通过。

- 目标仓库：`Jay1106-Zhu/csig-hallucination-aware-restoration`（已确认public）。
- Git提交：`78cbf2ecc46e83631c220cff6cd1724bbec36258`。
- 新Release：[`phase05-2026-09-15`](https://github.com/Jay1106-Zhu/csig-hallucination-aware-restoration/releases/tag/phase05-2026-09-15)；旧`phase0-2026-09-15`保持不动。
- 上传范围：全部1868个Phase05文件，另外10张真实LQ/GT、6个冻结编码器文件、4个冻结评价模型文件；不公开安装依赖、凭据与上传临时状态。
- 8个ZIP分包及两个清单；所有包和成员均有SHA-256，详见`artifact_manifest.json`。
- 远端包大小：`analysis-reports` 27,320,561 B；`data-patches` 326,333,189 B；`features` 1,536,060,533 B；`hypir-outputs` 377,449,986 B；`selective-outputs` 349,347,465 B；`shared-encoders` 479,957,846 B；`perceptual-models` 742,624,303 B；`verifiers` 24,827,072 B。
- 远端验证：10/10 资产服务端全量 SHA-256 与大小一致，匿名下载验证通过，`visibility=public`，`old_phase0_release_unchanged=true`；完整证据见`remote_verification.json`。
- 人工标注仍是110条pending，结论HOLD；公开交付并不等于科学假设通过。

`remote_verification.json`记录逐资产GitHub全文件SHA256/大小、匿名实际下载验证，以及旧Phase0资产未变的证据。
