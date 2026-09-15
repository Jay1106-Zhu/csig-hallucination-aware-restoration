# Phase 0.5 数据发现与审计

## 实际数量
- 独立源图像/scene：55。CSIG 真实 paired 场景仍为 5，另有 50 个 BSDS train 源图像的合成退化对。
- 没有将重复裁剪增加为独立样本，没有使用 CSIG test 或 BSDS test，没有下载任何测试 GT。
- BSDS train 有 200 个候选，按固定 seed=505 随机无放回选择 50 个。原始列表与选择存 `data/manifests/source_selection.json`。
- 合成组分别采用 blur、4×降采样再上采样、noise+JPEG、synthetic low-light/glow、mixed，各 10 个；对 GT 施加同坐标退化，无几何变换。
- 这不是新增 50 张真实 paired validation；模型在合成域上的收益不能直接外推到真实拍摄退化。

## 可追溯来源
- Berkeley/BIDS BSDS500：官方镜像 `BIDS/BSDS500`，固定 commit `a04b7c6c3a9f0ace74bf205c72a43d32e1c72722`，路径 `BSDS500/data/images/train/`。论文：Arbelaez et al., Contour Detection and Hierarchical Image Segmentation, TPAMI 2011。
- 真实扩展候选：RealBlur（真实运动模糊）、RealSR（真实超分辨率）、SIDD（真实噪声）、LOL（低照度配对）。这些仅列为后续数据来源方向，本轮未下载、未使用、未声称已通过许可/配准核验。
- 人物、动物、植被、建筑/自然场景通过源图联系表的助手视觉观察整理，分类和来源存 `data/manifests/scene_categories.json`，不是独立人的幻觉标注；中文仅来自原 CSIG 文字图，真实 night/flare 覆盖不足。synthetic lowlight_glow 不是实际夜景样本。

## 审计方法与边界
- 所有 GT 源文件 SHA-256 唯一；合成图缩略图相关性阈值 0.985，超阈值必须先审核，当前未触发。
- CSIG 五对尺寸/原 hash 一致；低频 phase correlation 检查整体平移。manifest 保存响应和512坐标系位移，仍不能排除非平移微配准或语义标注误差。
- synthetic pairing/alignment 由同一 GT 和固定退化代码直接建立，不靠猜测文件名配对。
- 源图 identity 是 scene 分组代理；缩略图去重不能严格证明不同照片一定来自不同场景。联系表保留用于人工追溯，不夸大独立性。
- 完整 manifest：`data/manifests/pairs.json`；来源哈希和旧阶段快照：`analysis/immutable_before.json`；近重复检查：`analysis/duplicate_audit.json`。

## 后续数据缺口
新增真实、不同设备/拍摄条件、语义×退化交叉覆盖的配对数据，尤其中文/人物/夜景/光学污染；须另行授权并先审计，不能用当前合成组替代这些缺口。

## 实测限制

- 非重叠区域共1160个：真实960、合成200。PSNR改善分别仅8和5；质量多数改善分别6和22。AUC有效原图真实2张、合成3张，不能把55张名义样本数当作55张有效排序验证样本。
- BSDS照片只有4个末端大小不同的有效区域；requested coverage=10%/20%都会取1个区域（实际区域25%）。像素coverage另列，不与区域coverage混用。
- 512上下文在BSDS上大量需要反射padding，不是更大真实视野。DINO处理器的224中心裁剪只保留输入上下文中央7/8；热图外围留空。
- manifest 的 split=validation 指本研究的分组验证用途；外部原数据来源明确为BSDS train，未使用BSDS test或CSIG test。
