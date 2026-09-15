import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dependencies'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from phase0_core import aggregate_heatmap
from prepare_phase0 import write_json
from report_utils import select_failures, spatial_statistics


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def generate_visualizations():
    table = pd.read_csv(ROOT / 'reports/patch_quality.csv')
    manifest = read_json('data/manifest.json')['validation']
    output = ROOT / 'reports/visualization'
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    failures = []
    spatial_rows = []
    cover = Image.new('RGB', (500 * len(manifest), 450), 'white')
    for entry_index, entry in enumerate(manifest):
        image_id = entry['image_id']
        subset = table[table.image_id == image_id]
        spatial_rows.append({'image_id': image_id, **spatial_statistics(subset)})
        with Image.open(entry['input_path']) as original:
            thumbnail = original.copy()
            thumbnail.thumbnail((500, 410))
        cover.paste(thumbnail, (500 * entry_index, 30))
        ImageDraw.Draw(cover).text((500 * entry_index + 5, 5), image_id, fill='black', font=font)
        selected = select_failures(subset, 4, 256)
        gallery = Image.new('RGB', (768, 304 * len(selected)), 'white')
        arrays = {name: np.load(ROOT / 'data/patches' / image_id / (name + '.npy'), mmap_mode='r') for name in ['input', 'hypir', 'gt']}
        for rank, (_, row) in enumerate(selected.iterrows(), 1):
            record = row.to_dict()
            record.update({'rank_in_image': rank, 'selection': 'lowest PSNR gain, nonoverlapping, four per image'})
            failures.append(record)
            panel = Image.new('RGB', (768, 304), 'white')
            annotation = ImageDraw.Draw(panel)
            annotation.text((4, 2), f"{row.patch_id} | gain={row.gain:.3f} dB", font=font, fill='black')
            for source_index, source in enumerate(['input', 'hypir', 'gt']):
                panel.paste(Image.fromarray(arrays[source][int(row.patch_index)]), (source_index * 256, 48))
                annotation.text((source_index * 256 + 4, 26), source.upper(), font=font, fill='black')
            panel.save(output / f'failure_{image_id}_{rank:02d}.png')
            gallery.paste(panel, (0, (rank - 1) * 304))
        gallery.save(output / f'failures_{image_id}.png')
    cover.save(output / 'input_overview.jpg', quality=92)
    pd.DataFrame(failures).to_csv(ROOT / 'analysis/failure_cases.csv', index=False)
    pd.DataFrame(spatial_rows).to_csv(ROOT / 'analysis/spatial_statistics.csv', index=False)
    if len(failures) != 20:
        raise ValueError(f'Expected 20 failures, found {len(failures)}')
    print('Saved 20 failure panels and five galleries', flush=True)


def confidence_visualizations():
    from sklearn.metrics import roc_curve

    table = pd.read_csv(ROOT / 'reports/patch_quality.csv')
    predictions = pd.read_csv(ROOT / 'analysis/oof_predictions.csv')
    if table.patch_id.tolist() != predictions.patch_id.tolist():
        raise ValueError('OOF row ordering mismatch')
    table['confidence'] = predictions.prob_fusion.to_numpy()
    output = ROOT / 'reports/visualization'
    for entry in read_json('data/manifest.json')['validation']:
        subset = table[table.image_id == entry['image_id']]
        coordinates = list(zip(subset.x, subset.y))
        gain = aggregate_heatmap(entry['width'], entry['height'], coordinates, subset.gain, 256)
        confidence = aggregate_heatmap(entry['width'], entry['height'], coordinates, subset.confidence, 256)
        for kind, values in [('gain', gain), ('confidence', confidence)]:
            np.save(output / f"{entry['image_id']}_{kind}.npy", values)
            figure, axis = plt.subplots(figsize=(9, 7))
            options = {'cmap': 'RdBu', 'vmin': -10, 'vmax': 10} if kind == 'gain' else {'cmap': 'viridis', 'vmin': 0, 'vmax': 1}
            heatmap = axis.imshow(values, **options)
            axis.set_title(f"{entry['image_id']} | {'PSNR gain (dB)' if kind == 'gain' else 'OOF P(GOOD), 3 seeds'}")
            axis.set_axis_off()
            figure.colorbar(heatmap, ax=axis, fraction=0.04, extend='both' if kind == 'gain' else 'neither')
            figure.tight_layout()
            figure.savefig(output / f"{entry['image_id']}_{kind}_heatmap.png", dpi=130)
            plt.close(figure)
        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        with Image.open(entry['input_path']) as image:
            axes[0].imshow(image)
        axes[0].set_title('Input')
        axes[1].imshow(gain, cmap='RdBu', vmin=-10, vmax=10)
        axes[1].set_title('Gain, blue=GOOD red=BAD [-10,10] dB')
        axes[2].imshow(confidence, cmap='viridis', vmin=0, vmax=1)
        axes[2].set_title('OOF confidence [0,1]')
        for axis in axes:
            axis.set_axis_off()
        figure.tight_layout()
        figure.savefig(output / f"{entry['image_id']}_overview.png", dpi=130)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 6))
    for image_id, subset in table.groupby('image_id', sort=True):
        labels = (subset.label == 'GOOD').to_numpy()
        if len(np.unique(labels)) == 2:
            false_positive, true_positive, _ = roc_curve(labels, subset.confidence)
            axis.plot(false_positive, true_positive, label=image_id)
    axis.plot([0, 1], [0, 1], '--', color='gray')
    axis.set(xlabel='False positive rate', ylabel='True positive rate', title='Fusion OOF ROC by original image')
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / 'fusion_roc_by_image.png', dpi=140)
    plt.close(figure)
    print('Saved OOF heatmaps and ROC curves', flush=True)


def supplementary_visualizations():
    table = pd.read_csv(ROOT / 'reports/patch_quality.csv')
    predictions = pd.read_csv(ROOT / 'analysis/oof_predictions.csv')
    table['confidence'] = predictions.prob_fusion.to_numpy()
    records = []
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
    gallery = Image.new('RGB', (768, 320 * 5), 'white')
    for region_index, region in enumerate(read_json('configs/semantic_regions.json')):
        selected = table[(table.image_id == region['image_id']) & (table.x == region['x']) & (table.y == region['y'])]
        if len(selected) != 1:
            raise ValueError('Supplementary region must identify one existing patch')
        row = selected.iloc[0]
        records.append({**region, 'patch_id': row.patch_id, 'gain': row.gain, 'label': row.label, 'oof_confidence': row.confidence,
                        'selection': 'purposeful semantic illustration, not used for model selection or metrics'})
        panel = Image.new('RGB', (768, 320), 'white')
        draw = ImageDraw.Draw(panel)
        draw.text((4, 0), f"{row.patch_id} | {region['topic']}", font=font, fill='black')
        draw.text((4, 20), f"gain={row.gain:.3f} dB | {row.label} | OOF confidence={row.confidence:.4f}", font=font, fill='black')
        for source_index, source in enumerate(['input', 'hypir', 'gt']):
            array = np.load(ROOT / 'data/patches' / row.image_id / (source + '.npy'), mmap_mode='r')
            draw.text((source_index * 256 + 4, 43), source.upper(), font=font, fill='black')
            panel.paste(Image.fromarray(array[int(row.patch_index)]), (source_index * 256, 64))
        panel.save(ROOT / 'reports/visualization' / f'supplementary_{row.image_id}.png')
        gallery.paste(panel, (0, region_index * 320))
    gallery.save(ROOT / 'reports/visualization/supplementary_semantic_gallery.png')
    pd.DataFrame(records).to_csv(ROOT / 'analysis/supplementary_regions.csv', index=False)


def markdown_table(frame, precision=4):
    columns = frame.columns.tolist()
    lines = ['| ' + ' | '.join(columns) + ' |', '|' + '|'.join('---' for _ in columns) + '|']
    for row in frame.itertuples(index=False, name=None):
        formatted = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                formatted.append(f'{value:.{precision}f}' if np.isfinite(value) else 'NA')
            else:
                formatted.append(str(value))
        lines.append('| ' + ' | '.join(formatted) + ' |')
    return '\n'.join(lines)


def write_reports():
    config = read_json('configs/phase0.json')
    summary = read_json('analysis/confidence_summary.json')
    environment = read_json('analysis/environment.json')
    inference = read_json('analysis/hypir_inference.json')
    audit = read_json('analysis/immutable_after.json')
    models = read_json('analysis/feature_models.json')
    spatial = pd.read_csv(ROOT / 'analysis/spatial_statistics.csv')
    full = pd.read_csv(ROOT / 'analysis/full_image_quality.csv')
    folds = pd.read_csv(ROOT / 'reports/confidence_metrics.csv')
    seeds = pd.read_csv(ROOT / 'analysis/confidence_metrics_per_seed.csv')
    baselines = pd.read_csv(ROOT / 'analysis/baselines.csv')
    annotations_path = ROOT / 'analysis/failure_annotations.csv'
    annotations = pd.read_csv(annotations_path).fillna('') if annotations_path.exists() else None
    fusion_auc = summary['fusion']['macro']['roc_auc']
    sensitivity = folds[(folds.good_count >= 2) & (folds.bad_count >= 2)].groupby('feature', as_index=False).agg(
        sensitivity_auc=('roc_auc', 'mean'), eligible_images=('heldout_image', 'count'))
    sensitivity.to_csv(ROOT / 'analysis/auc_sensitivity.csv', index=False)
    fusion_sensitivity = float(sensitivity.loc[sensitivity.feature == 'fusion', 'sensitivity_auc'].iloc[0])
    decision = 'YES' if fusion_auc is not None and fusion_auc > config['auc_research_threshold'] else 'NO'
    summary_rows = []
    for name, result in summary.items():
        lower, upper = result['image_bootstrap_auc_95']
        summary_rows.append({'Features': name, 'dim': result['feature_dim'], **result['macro'],
                             'AUC_image_bootstrap_95': f'{lower:.4f}–{upper:.4f}' if lower is not None else 'NA'})
    seed_summary = seeds.groupby(['feature', 'seed'], as_index=False)[['roc_auc', 'accuracy', 'f1', 'bad_recall']].mean()
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(ROOT / 'reports/confidence_macro_summary.csv', index=False)
    seed_summary.to_csv(ROOT / 'analysis/seed_macro_summary.csv', index=False)
    bad_fraction = spatial.bad_fraction.mean()
    report = [
        '# CSIG Phase 0：HYPIR 可靠性可预测性验证', '', '执行日期：2026-09-15。状态：实验已运行；只训练 Confidence MLP，没有训练恢复模型。', '',
        '## 1. 结论与适用范围', '',
        f"**继续研究 Confidence Gate：{decision}**。预先固定的 fusion 原图宏平均 ROC-AUC={fusion_auc:.4f}，阈值为严格 >0.65。",
        f"CLIP AUC={summary['clip']['macro']['roc_auc']:.4f}；DINO AUC={summary['dino']['macro']['roc_auc']:.4f}；Fusion AUC={fusion_auc:.4f}。",
        f"独立验证单位只有 **5 张原图**（{int(spatial.patch_count.sum())} 个重叠 patch 不是独立图片）；有效 AUC 图像数={summary['fusion']['valid_auc_images']}。",
        f"Fusion 图像级描述性 bootstrap 95% 区间：{summary['fusion']['image_bootstrap_auc_95']}。小样本和同域相关性使该区间不能支持部署保证。",
        '**实际推进建议：HOLD / 不进入 Gate 或专家训练，先补充独立数据。** YES 仅表示符合指导的探索性数值门槛，不等于本假设已获可靠证据。',
        f'全体只有 59 个 GOOD：case1/case3 为零，case4 仅一个；去掉只有一个 GOOD 的 case4 后，Fusion 在其余两个可评价原图的宏 AUC={fusion_sensitivity:.4f}。',
        '这一事后敏感性分析不用于挑模型、改阈值或替换主指标，只说明主指标对极少正例图像的脆弱性。',
        '本实验预测的是相对输入的 PSNR 收益，不能证明语义真实性、生成幻觉检测准确率或融合后性能提升。', '',
        '## 2. 环境与官方推理', '',
        f"- Python：{environment['python']}；GPU：{environment['gpu']}；PyTorch：{environment['packages']['torch']}；CUDA runtime：{environment['torch_cuda']}。",
        '- nvcc 不在 PATH；没有编译自定义 CUDA 算子，PyTorch CUDA 推理与特征提取实测可用。',
        '- scikit-learn 等缺失依赖安装在阶段 dependencies/；环境版本见 requirements.txt 与 analysis/environment.json。',
        f"- 官方入口 HYPIR/test.py；仓库 commit={inference['source_commit']}；本轮重新运行，耗时 {inference['elapsed_seconds']:.2f} 秒。",
        '- 固定 model_t=200、coeff_t=200、推理 tile=512/stride=256、upscale=1、空 prompt、seed=231；加载现有官方预训练 LoRA 权重不等于训练 LoRA。',
        '- 五张原尺寸 RGB PNG 位于 outputs/hypir/result/；完整命令与输入/输出/权重见 analysis/hypir_inference.json、data/manifest.json。',
        f"- 原数据、官方源码和恢复权重 {audit['checked_files']} 个文件前后校验：all_unchanged={audit['all_unchanged']}。", '',
        '### 原图 PSNR（dB，等权按图像）', '', markdown_table(full[['image_id', 'psnr_input', 'psnr_hypir', 'gain']]), '',
        f"原图平均 Input={full.psnr_input.mean():.4f}；HYPIR={full.psnr_hypir.mean():.4f}；平均 gain={full.gain.mean():.4f} dB。",
        '原图 PSNR 不等于 patch PSNR 的平均值。主模型采用官方 t200；此报告不外推到历史调弱 t50 或其它恢复器。', '',
        '## 3. 数据与标签', '',
        '- validation 为 5 对 RGB JPEG，4 张 4096×3072、1 张 3072×4096；test 100 张 LQ 只检查图片头部，不提取 patch、不参与标签或训练。',
        '- patch_size=256、stride=128，每图 713 个，合计 3565；无缩放、无 GT 特征、无随机 patch 划分。',
        '- GOOD：PSNR(HYPIR,GT)-PSNR(Input,GT)>0；其它为 BAD。GT 限于标签、离线评价和对照图。',
        '- 所有 input/hypir/gt patch 以逐图无损 uint8 NPY 保存，位置索引在 features/index.csv，读取方法见 README。', '',
        '## 4. HYPIR 错误是否有空间规律？', '',
        markdown_table(spatial[['image_id', 'good_count', 'bad_count', 'bad_fraction', 'mean_patch_gain', 'adjacent_same_label', 'independent_expected_agreement', 'agreement_excess']]), '',
        f'原图等权平均 BAD 比例={bad_fraction:.4%}。邻接一致率减去基于各图类比例的独立期望，平均超额={spatial.agreement_excess.mean():.4f}。',
        '热图与邻接统计用于描述空间分布；patch 有 50% 重叠，邻接相关性并不等于独立的空间显著性检验。',
        '不能把“大片区域都是 BAD”直接归因于某个语义类别，也不能以此证明 hallucination 的位置可泛化预测。', '',
        '## 5. 是否可以预测？', '',
        '- 外层 Leave-One-Image-Out：4 张训练/1 张评价，共 5 折；每折仅在训练图像拟合 scaler。',
        '- MLP：Linear(64)→ReLU→Linear(1)→Sigmoid；40 epochs；AdamW lr=0.001、weight_decay=0.01；batch=256。',
        '- 三个固定 seed=231/232/233，OOF 概率均值；无测试折调参、早停、阈值搜索或最佳 seed 选择。',
        '- 训练 loss 按原图和标签平衡；指标为原图等权平均，GOOD 为正类，阈值固定 0.5。',
        '- CLIP/DINO 使用冻结 input 与 HYPIR 的成对特征；fusion_lq 只使用输入，可检验输出感知相对输入特征的差别。', '',
        markdown_table(summary_frame), '',
        '### Fusion 逐原图 OOF 指标', '',
        markdown_table(folds[folds.feature == 'fusion'][['heldout_image', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'balanced_accuracy', 'bad_recall', 'good_count', 'bad_count']]), '',
        '### 简单基线（同样按图像宏平均）', '',
        markdown_table(baselines.groupby('baseline')[['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'balanced_accuracy', 'bad_recall']].mean().reset_index()), '',
        'Accuracy 必须结合类别不均衡解读。AUC 衡量排序而非校准；训练使用类别重加权，因此输出虽然记为 P(GOOD)，不应直接当作已校准概率。', '',
        '### 类别稀缺与敏感性分析', '',
        '- case1/case3 的 AUC 不定义，不能填 0.5 或把它们计入 AUC 均值；主 AUC 因此是三个有效原图的等权平均。',
        '- case4 的高 AUC 只依靠一个 GOOD 的排序位置；不是稳定的场景级泛化证据。',
        '- bootstrap 实际只对三个 AUC 有定义的原图重采样；不对 patch 重采样，不为缺失 AUC 造值。与五图总体相比覆盖有限，属于执行时的必要缺失值处理。',
        '- 以下仅展示至少两个 GOOD 且至少两个 BAD 的图像：', '', markdown_table(sensitivity), '',
        '- 主 Fusion 的 GOOD precision/recall/F1 很低，always-BAD 的 accuracy 更高；该概率图不能直接作为拒绝增强或融合权重部署。', '',
        '### Seed 稳定性', '', markdown_table(seed_summary), '',
        '## 6. 失败案例与可视化', '',
        '- gain/confidence：reports/visualization/case*_gain_heatmap.png、case*_confidence_heatmap.png。',
        '- confidence 图全部来自该原图完全未参与训练的 OOF 模型；重叠处取平均，原尺寸浮点图同时保存为 NPY。',
        '- gain PNG 统一色轴 [-10,10] dB（超界截色，NPY 保留真实值）；confidence 色轴 [0,1]。',
        '- 20 个 Input/HYPIR/GT 对照：failure_case*_*.png；每图四个，按最差 gain 选择且在本图不重叠。',
        '- 每图拼图：failures_case*.png；全图+热图：case*_overview.png；逐图 ROC：fusion_roc_by_image.png。',
        '- 选择规则只用于展示，不构成语义类别分布的无偏估计。失败区域类别来自人工查看，不使用自动类别预测代替观察。', '',
        '- 补充语义观察：supplementary_semantic_gallery.png，覆盖汉字、书脊、动物头部、植被和钟表局部；坐标事先从输入概览人工指定，不参与模型选择。',
    ]
    if annotations is not None:
        report.extend(['### 人工观察记录', '', markdown_table(annotations), '',
                       '以上是选出的 20 个失败案例内容，不是全体 BAD 区域的语义标注；未出现的类别不能推断安全。', ''])
        semantic_notes = ROOT / 'reports/semantic_observations.md'
        if semantic_notes.exists():
            report.extend([semantic_notes.read_text(encoding='utf-8'), ''])
    else:
        report.extend(['人工语义观察尚未完成；不得声称已完成植物/动物/汉字/小目标的类别评估。', ''])
    report.extend([
        '## 7. 下一阶段建议与停止条件', '',
        ('- YES 只表示达到探索性排序门槛；先扩大独立、成对、多场景 validation，再讨论 Confidence Gate。' if decision == 'YES'
         else '- NO：当前固定方案未达到原图宏平均 AUC 门槛；不应据此继续堆叠 Gate 或进入专家训练。'),
        '- 保留全量训练的 checkpoints/confidence_mlp.pt 供复现；它用过全部五张图，不能用于本报告的验证指标，也不是部署模型。',
        '- 若继续：先补充独立原图，固定完全未参与开发的验证集；增加局部感知/语义真实性标注，确认图像配准和 PSNR 定义适用性。',
        '- 仅当扩大数据后仍稳定预测收益，再授权 Phase 1 Expert Benefit Analysis；不得直接训练 LoRA/MoE、设计语义专家分类。',
        '- 暂未实施 fusion/gate 的实际图像融合，因此没有输出画质提高的证据；本轮到 Phase 0 停止。', '',
        '## 8. 证据索引与复现', '',
        '| 证据 | 发现 | 行动 |', '|---|---|---|',
        '| data/manifest.json、reports/dataset_report.md | 只有五个独立配对场景 | 所有划分按图像，结论保留小样本限制 |',
        '| reports/patch_quality.csv、analysis/spatial_statistics.csv | patch 收益及空间分布 | 只定义相对输入 PSNR 标签，不声称语义幻觉 |',
        '| analysis/folds.json、analysis/oof_predictions.csv、reports/confidence_metrics.csv | 原图留一预测能力 | 按预设 AUC 门槛决定研究价值 |',
        '| analysis/immutable_after.json | 原始工程只读 | 保留来源与输出哈希 |',
        '- 单元测试日志：logs/unit_tests.log；实际运行日志：logs/pipeline.out.log、logs/hypir_inference.log。',
        '- 独立产物审计：analysis/artifact_audit.json；核验全部 3565 组三源 patch 与源像素/标签、90 个折 checkpoint 的训练图像/scaler/OOF，并重算全部逐折指标。',
        '- 15 项契约测试覆盖 patch 边界、严格标签、原图划分、单类 AUC、训练 scaler、特征对齐、热图重叠、失败区域与 checkpoint 泄漏检查。',
        '- 配置：configs/phase0.json；协议：reports/protocol.md；入口：scripts/prepare_phase0.py、scripts/run_phase0.py、scripts/report_phase0.py。',
        '- 特征模型仓库与 revision：',
        f"  - {models['clip']['repository']} @ {models['clip']['revision']}",
        f"  - {models['dino']['repository']} @ {models['dino']['revision']}",
        '- 原始依据为 new_formal 根目录的两份方案；模型来源标识及哈希由官方下载 API 和本地文件校验得到。',
    ])
    (ROOT / 'reports/phase0_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    delivery = [
        '# Phase 0 交付', '', '日期：2026-09-15。', '',
        '## 环境', f"- Python 3.11.16，{environment['gpu']}，torch {environment['packages']['torch']}，CUDA {environment['torch_cuda']}。",
        '- 环境版本：requirements.txt；数据检查：reports/dataset_report.md。', '',
        '## HYPIR 结果', '- 官方 t200/t200，原尺寸 RGB PNG ×5：outputs/hypir/result/。',
        f"- 官方推理耗时 {inference['elapsed_seconds']:.2f} 秒；原数据/源码/权重哈希验证：{audit['all_unchanged']}。", '',
        '## Patch 与 Confidence', f'- 5 张原图，3565 个重叠 patch；原图等权 BAD={bad_fraction:.4%}。',
        f"- CLIP/DINO/Fusion 宏平均 ROC-AUC：{summary['clip']['macro']['roc_auc']:.4f} / {summary['dino']['macro']['roc_auc']:.4f} / {fusion_auc:.4f}。",
        f'- 注意：AUC 只有三个有效原图；case4 仅一个 GOOD。去掉单正例原图后 Fusion AUC={fusion_sensitivity:.4f}（事后敏感性分析）。',
        '- 分组：5 折原图留一 ×3 固定 seed；指标与最终 checkpoint 分离；全量模型：checkpoints/confidence_mlp.pt。', '',
        '## 可视化与报告', '- reports/visualization/：20 个失败对照、5 张 gain 热图、5 张 OOF confidence 热图及逐图概览。',
        '- 完整报告：reports/phase0_report.md；原始表：reports/patch_quality.csv、reports/confidence_metrics.csv。', '',
        '## 验证', '- 15 项单元测试通过；全部 3565 组 patch 与源数据、标签一致。',
        '- 独立重建 90 个折 checkpoint，验证训练集 scaler 和 OOF；报告：analysis/artifact_audit.json。', '',
        '## 下一阶段', f'- 是否值得继续研究 Gate：**{decision}**（按固定 fusion 原图宏平均 AUC>0.65）。',
        '- 实际推进：**HOLD，先补数据**。当前分类质量和正例覆盖不足，不建议进入 Gate/专家训练。',
        '- 只完成 Phase 0，没有启动 Phase 1、恢复模型训练、LoRA 训练、MoE 或实际 gate 融合。',
        '- 5 张图不足以做强泛化结论；PSNR BAD 不等于语义幻觉标签，后续需独立数据及语义评估。',
    ]
    if annotations is None:
        delivery.insert(3, '状态：计算完成，但人工失败案例观察尚待完成。')
    else:
        delivery.insert(3, '状态：已完成（小样本探索性验证，限制详见报告）。')
    (ROOT / 'phase0_done.md').write_text('\n'.join(delivery) + '\n', encoding='utf-8')
    write_json(ROOT / 'analysis/decision.json', {'decision': decision, 'criterion': 'fusion image-macro ROC-AUC > 0.65',
                                               'value': fusion_auc, 'semantic_annotations_completed': annotations is not None,
                                               'independent_images': 5, 'valid_auc_images': 3,
                                               'practical_recommendation': 'HOLD: collect independent paired data first',
                                               'sensitivity_auc_without_single_positive_image': fusion_sensitivity,
                                               'gate_fusion_implemented': False})
    print('Wrote report; decision', decision, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['failures', 'heatmaps', 'supplementary', 'report', 'all'])
    arguments = parser.parse_args()
    functions = {'failures': generate_visualizations, 'heatmaps': confidence_visualizations,
                 'supplementary': supplementary_visualizations, 'report': write_reports}
    for stage in (functions if arguments.stage == 'all' else [arguments.stage]):
        functions[stage]()
