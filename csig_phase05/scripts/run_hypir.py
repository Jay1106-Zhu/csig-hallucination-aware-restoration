import os
import shutil
import subprocess
import sys
import time

from PIL import Image

from core import ROOT, WORKSPACE, config, read_json, relative, resolve, sha256, write_json


def main():
    settings = config()
    pairs = read_json(ROOT / 'data/manifests/pairs.json')
    original_state = read_json(ROOT / 'analysis/immutable_before.json')
    history = []
    for coefficient in settings['hypir_coefficients']:
        output = ROOT / f'outputs/hypir/coeff_{coefficient}'
        command = [sys.executable, 'test.py', '--base_model_type', 'sd2', '--base_model_path', 'models/stable-diffusion-2-1-base',
                   '--model_t', '200', '--coeff_t', str(coefficient), '--lora_rank', '256',
                   '--lora_modules', 'to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj',
                   '--weight_path', 'weights/HYPIR_sd2.pth', '--patch_size', '512', '--stride', '256',
                   '--lq_dir', str(ROOT / 'data/input'), '--scale_by', 'factor', '--upscale', '1', '--captioner', 'empty',
                   '--output_dir', str(output), '--seed', '231', '--device', 'cuda']
        expected = [output / 'result' / (pair['image_id'] + '.png') for pair in pairs if pair['domain'] == 'synthetic_bsds']
        started = time.perf_counter()
        if not all(path.exists() for path in expected):
            environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
                               PYTHONUTF8='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4')
            with (ROOT / f'logs/hypir_coeff_{coefficient}.log').open('w', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=WORKSPACE / 'HYPIR', env=environment, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Official inference failed for coeff={coefficient}')
        for pair in pairs:
            if pair['domain'] == 'real_csig':
                source = resolve(pair['stability_paths'][str(coefficient)])
                if coefficient != 200:
                    metadata_path = WORKSPACE / f'baseline/experiments/coeff_t_{coefficient}/experiment_metadata.md'
                    metadata = metadata_path.read_text(encoding='utf-8')
                    if '- model_t: 200' not in metadata or f'- coeff_t: {coefficient}' not in metadata or '- seed: 231' not in metadata:
                        raise ValueError('Historic controlled condition parameters differ')
                    historic_input = WORKSPACE / 'baseline/input' / (pair['image_id'] + '_lq.jpg')
                    if sha256(historic_input) != pair['lq_sha256']:
                        raise ValueError('Historical stability input mismatch')
                destination = output / 'result' / (pair['image_id'] + '.png')
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                if sha256(source) != sha256(destination):
                    raise ValueError('Copied candidate changed')
                pair['stability_paths'][str(coefficient)] = relative(destination)
            path = resolve(pair['stability_paths'][str(coefficient)])
            with Image.open(path) as image:
                if image.mode != 'RGB' or image.size != (pair['width'], pair['height']):
                    raise ValueError(f'Candidate size differs: {pair["image_id"]}')
            pair.setdefault('stability_sha256', {})[str(coefficient)] = sha256(path)
            if coefficient == 200:
                pair['candidate_path'] = relative(path)
                pair['candidate_sha256'] = sha256(path)
        history.append({'coefficient': coefficient, 'model_t': 200, 'seed': 231, 'kind': 'controlled sensitivity, not random uncertainty',
                        'command': ['python' if argument == sys.executable else relative(argument) if str(ROOT) in argument else argument for argument in command],
                        'cwd': relative(WORKSPACE / 'HYPIR'), 'elapsed_seconds': time.perf_counter() - started,
                        'new_images': 50, 'reused_images': 5, 'checkpoint_sha256': original_state[relative(WORKSPACE / 'HYPIR/weights/HYPIR_sd2.pth')]})
        write_json(ROOT / 'analysis/hypir_runs.json', history)
        print('Completed coeff', coefficient, round(history[-1]['elapsed_seconds'], 1), 'seconds', flush=True)
    write_json(ROOT / 'data/manifests/pairs.json', pairs)
    print('All 55 main candidates and four controlled conditions verified', flush=True)


if __name__ == '__main__':
    main()
