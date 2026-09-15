import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / 'dependencies'))
from phase0_core import sha256


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding='utf-8')


def prepare_data():
    import torch

    for name in ['data/input', 'models', 'checkpoints', 'outputs/hypir', 'analysis', 'features', 'reports/visualization', 'logs']:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    original = PROJECT / 'csig_dataset'
    records = []
    immutable = {}
    for input_path in sorted((original / '验证集').glob('*_lq.*')):
        image_id = input_path.stem.removesuffix('_lq')
        gt_path = input_path.with_name(image_id + '_gt' + input_path.suffix)
        with Image.open(input_path) as input_image, Image.open(gt_path) as gt_image:
            if input_image.size != gt_image.size or input_image.mode != 'RGB' or gt_image.mode != 'RGB':
                raise ValueError('Paired RGB images must have matching native size')
            record = {'image_id': image_id, 'width': input_image.width, 'height': input_image.height,
                      'mode': input_image.mode, 'format': input_image.format,
                      'input_path': str(input_path), 'gt_path': str(gt_path),
                      'hypir_path': str(ROOT / 'outputs/hypir/result' / (input_path.stem + '.png'))}
        for path in (input_path, gt_path):
            immutable[str(path)] = sha256(path)
        record['input_sha256'] = immutable[str(input_path)]
        record['gt_sha256'] = immutable[str(gt_path)]
        destination = ROOT / 'data/input' / input_path.name
        if not destination.exists():
            shutil.copy2(input_path, destination)
        if sha256(destination) != record['input_sha256']:
            raise ValueError('Input copy checksum mismatch')
        records.append(record)
    test_records = []
    for path in sorted((original / '测试集').rglob('*')):
        if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
            continue
        if '_gt' in path.stem.lower():
            test_records.append({'name': path.name, 'excluded': 'test GT not opened'})
            continue
        with Image.open(path) as image:
            test_records.append({'name': path.name, 'width': image.width, 'height': image.height,
                                 'format': image.format, 'mode': image.mode})
    for path in sorted((PROJECT / 'HYPIR').rglob('*.py')):
        immutable[str(path)] = sha256(path)
    for path in sorted((PROJECT / 'HYPIR/weights').glob('*.pth')):
        immutable[str(path)] = sha256(path)
    write_json(ROOT / 'data/manifest.json', {'validation': records, 'test_inventory_only': test_records})
    write_json(ROOT / 'analysis/immutable_before.json', immutable)
    packages = ['torch', 'torchvision', 'opencv-python', 'numpy', 'pandas', 'scikit-image', 'scikit-learn',
                'matplotlib', 'tqdm', 'pillow', 'transformers', 'open_clip_torch', 'huggingface-hub',
                'scipy', 'joblib', 'threadpoolctl']
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    environment = {'timestamp': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
                   'executable': sys.executable, 'platform': platform.platform(), 'packages': versions,
                   'cuda_available': torch.cuda.is_available(), 'torch_cuda': torch.version.cuda,
                   'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
                   'nvcc': shutil.which('nvcc')}
    gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True, errors='replace')
    (ROOT / 'logs/nvidia-smi.txt').write_text(gpu.stdout + gpu.stderr, encoding='utf-8')
    write_json(ROOT / 'analysis/environment.json', environment)
    (ROOT / 'requirements.txt').write_text('\n'.join(f'{key}=={value}' for key, value in versions.items() if value) + '\n', encoding='utf-8')
    report = ['# 数据检查报告', '', f'验证集：{len(records)} 对；独立原图数：{len(records)}。',
              f'测试集：{len(test_records)} 张，只做文件及头部元信息盘点，不用于实验。', '',
              '| image_id | 宽×高 | 模式 | 格式 | GT |', '|---|---|---|---|---|']
    report.extend(f"| {row['image_id']} | {row['width']}×{row['height']} | {row['mode']} | {row['format']} | 存在且同尺寸 |" for row in records)
    report.extend(['', '没有下载/合成额外样本，没有改写原始数据，未做 resize、配准或色彩空间替换。',
                   '详尽清单及 SHA-256：`data/manifest.json`；源数据/源码/恢复权重校验：`analysis/immutable_before.json`。',
                   '这里只能验证文件配对与尺寸，不能据此证明像素级配准无误。'])
    (ROOT / 'reports/dataset_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('Prepared', len(records), 'validation pairs;', len(test_records), 'test metadata entries', flush=True)


def run_hypir():
    import time

    config = json.loads((ROOT / 'configs/phase0.json').read_text(encoding='utf-8'))['hypir']
    command = [sys.executable, 'test.py', '--base_model_type', 'sd2', '--base_model_path', 'models/stable-diffusion-2-1-base',
               '--model_t', str(config['model_t']), '--coeff_t', str(config['coeff_t']), '--lora_rank', '256',
               '--lora_modules', 'to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj',
               '--weight_path', 'weights/HYPIR_sd2.pth', '--patch_size', str(config['patch_size']), '--stride', str(config['stride']),
               '--lq_dir', str(ROOT / 'data/input'), '--scale_by', 'factor', '--upscale', '1', '--captioner', 'empty',
               '--output_dir', str(ROOT / 'outputs/hypir'), '--seed', str(config['seed']), '--device', 'cuda']
    metadata = {'command': command, 'cwd': str(PROJECT / 'HYPIR'), 'params': config,
                'source_commit': subprocess.check_output(['git', '-C', str(PROJECT / 'HYPIR'), 'rev-parse', 'HEAD'], text=True).strip(),
                'preexisting_source_status': subprocess.check_output(['git', '-C', str(PROJECT / 'HYPIR'), 'status', '--short'], text=True),
                'checkpoint': str(PROJECT / 'HYPIR/weights/HYPIR_sd2.pth'), 'status': 'running'}
    started = time.perf_counter()
    write_json(ROOT / 'analysis/hypir_inference.json', metadata)
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1', PYTHONUTF8='1')
    with (ROOT / 'logs/hypir_inference.log').open('w', encoding='utf-8') as log:
        completed = subprocess.run(command, cwd=PROJECT / 'HYPIR', stdout=log, stderr=subprocess.STDOUT, env=environment)
    metadata.update({'returncode': completed.returncode, 'elapsed_seconds': time.perf_counter() - started,
                     'status': 'completed' if completed.returncode == 0 else 'failed'})
    metadata['outputs'] = {str(path): sha256(path) for path in sorted((ROOT / 'outputs/hypir/result').glob('*.png'))}
    write_json(ROOT / 'analysis/hypir_inference.json', metadata)
    if completed.returncode:
        raise RuntimeError('Official HYPIR failed; see logs/hypir_inference.log')
    print('HYPIR completed', flush=True)


def download_models():
    from huggingface_hub import HfApi, snapshot_download

    config = json.loads((ROOT / 'configs/phase0.json').read_text(encoding='utf-8'))
    metadata_path = ROOT / 'analysis/feature_models.json'
    model_info = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
    for name in ['clip', 'dino']:
        repository = config[name + '_model']
        pinned_revision = model_info.get(name, {}).get('revision')
        details = HfApi().model_info(repository, revision=pinned_revision)
        revision = details.sha
        available = {entry.rfilename for entry in details.siblings}
        weights = 'model.safetensors' if 'model.safetensors' in available else 'pytorch_model.bin'
        destination = ROOT / 'models' / name
        snapshot_download(repository, revision=revision, local_dir=destination,
                          allow_patterns=['config.json', 'preprocessor_config.json', weights], max_workers=2)
        if not (destination / weights).is_file():
            raise FileNotFoundError('Pretrained weights were not downloaded')
        model_info[name] = {'repository': repository, 'revision': revision, 'path': str(destination),
                            'files': {path.name: sha256(path) for path in destination.glob('*') if path.is_file()}}
        write_json(ROOT / 'analysis/feature_models.json', model_info)
        print('Downloaded', name, revision, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['data', 'hypir', 'models'])
    arguments = parser.parse_args()
    {'data': prepare_data, 'hypir': run_hypir, 'models': download_models}[arguments.stage]()
