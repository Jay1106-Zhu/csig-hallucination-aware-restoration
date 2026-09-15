# 幻觉盲标注协议

本任务需要独立真实人的审核。PSNR、其他质量指标、DINO 特征和助手观察都不能代替人工幻觉真值。当前没有人工提交，标签保持 pending。

## 审核单位

每个源图像 seed=505 固定抽取两个不重叠有效区域，共 110 个。打开同一区域的 LQ / restored / GT，可放大比较。不要看指标、预测、特征或按模型错误筛选样本。GT 仅用于离线标注。

## 严重度

- 0：无可见虚构/语义错误；差异符合去噪、合理锐化或细节缺失。
- 1：局部轻微可疑纹理/边缘/细节变化，通常不改变识别。
- 2：有明确虚构、重复、位移或有意义的结构/语义不一致。
- 3：严重错误改变文字、对象、几何或场景理解，或占据该区域的大部分。

无法判断时 severity 留空，status=uncertain、type=uncertain，并说明原因；不得用 0 代替缺失。

## 类型

允许多选，以分号分隔：texture_invention、structure_invention、semantic_corruption、text_glyph_corruption、boundary_corruption、object_invention、repeated_pattern、over_sharpening_artifact、uncertain。severity=0 时允许类型为空。

## 来源与导入

填写 reviewer_id、reviewed_at、severity、hallucination_types 和 note。只有 reviewer_type=human 且 annotation_status=reviewed 且 severity 为 0–3 整数的行参与评价。pending/assistant/synthetic/uncertain 不当作 0。

双击 index.html 在本机离线标注、导出 CSV，再运行 `python scripts/annotations.py --import-csv <导出文件>`。导入脚本检查固定索引、重复条目、来源及类型，保存 reviewed_labels.csv；不覆盖原始 pending CSV。建议第二位标注员独立审核同一组，再报告一致性。两指标目标保持独立，不依据本轮结果调 lambda。
