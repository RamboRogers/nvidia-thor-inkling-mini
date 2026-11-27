#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Launch or smoke-test the Jetson Thor v6 profile on loopback."""
import argparse
import base64
import grp
import json
import os
from pathlib import Path
import platform
import shlex
import struct
import subprocess
import time
import urllib.request
import zlib

from build import TAG, capture, run
import fetch_model

MODEL = 'inkling-mini'
CONTEXT = 262144
IMAGE_LIMIT = 16
ENV = {
    'HOME': '/trial-cache',
    'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'DO_NOT_TRACK': '1',
    'VLLM_NO_USAGE_STATS': '1',
    'MAX_JOBS': '4',
    'NVCC_THREADS': '1',
    'OMP_NUM_THREADS': '2',
    'TORCHINDUCTOR_COMPILE_THREADS': '1',
    'CMAKE_BUILD_PARALLEL_LEVEL': '2',
    'INKLING2_DRAFT_QUANT': 'int8',
    'INKLING2_DRAFT_QUANT_LAYERS': '1',
    'INKLING2_THOR_BF16X': '1',
    'INKLING2_THOR_BF16X_DENSE': '0',
    'INKLING2_THOR_BF16X_HEAD': '1',
    'INKLING2_THOR_BF16X_SRC': '/opt/thor-opt',
    'INKLING2_THOR_BF16X_PREBUILT': '/opt/thor-opt/ext',
    'INKLING2_THOR_FA4_PREFILL': 'global',
    'INKLING2_THOR_SPLITCAP': '1',
    'INKLING2_DRAFT_HOTVOCAB': '65536',
    'INKLING2_DRAFT_HOTVOCAB_FILE': '/opt/thor-opt/inkling_draftbytes/hotvocab/inkling2-mini-hot-r2.json',
    'INKLING2_DRAFT_W8TILE': '1',
    'INKLING2_THOR_VCAP': 'cost:1=0.30,2=0.22,3=0.12,4=0.0,min=1,max=8',
}
SPEC = {'method': 'dspark', 'num_speculative_tokens': 7, 'draft_sample_method': 'probabilistic'}


def gpu_groups():
    for name in ('video', 'render'):
        try:
            yield str(grp.getgrnam(name).gr_gid)
        except KeyError:
            continue


def launch_args(args):
    opts = [
        '--pull=never', '--name', args.name, '--init', '--restart=no', '--network', 'bridge',
        '--runtime', 'nvidia', '--user', f'{os.getuid()}:{os.getgid()}',
        '--security-opt=no-new-privileges', '--cap-drop=ALL',
        '--shm-size', '4g', '--cpus', '12', '--memory', '90g', '--memory-swap', '90g', '--pids-limit', '1024',
        '--publish', f'127.0.0.1:{args.port}:8000/tcp',
        '--mount', f'type=bind,src={args.model_root / fetch_model.REV},dst=/model,readonly',
        '--mount', f'type=bind,src={args.cache_dir},dst=/trial-cache', '--workdir', '/tmp',
    ]
    for gid in gpu_groups():
        opts += ['--group-add', gid]
    for key, value in ENV.items():
        opts += ['--env', f'{key}={value}']
    return opts + [
        args.image, '/model', '--served-model-name', MODEL, '--host', '0.0.0.0', '--port', '8000',
        '--dtype', 'bfloat16', '--max-model-len', str(CONTEXT), '--max-num-seqs', '4',
        '--max-num-batched-tokens', '2048', '--gpu-memory-utilization', '0.35',
        '--kv-cache-memory-bytes', str(28 * 2**30), '--kv-cache-dtype', 'bfloat16',
        '--reasoning-parser', 'inkling', '--enable-auto-tool-choice', '--tool-call-parser', 'inkling',
        '--limit-mm-per-prompt', json.dumps({'image': IMAGE_LIMIT}),
        '--safetensors-load-strategy', 'eager', '--speculative-config', json.dumps(SPEC),
    ]


def start(args):
    if platform.system() != 'Linux' or platform.machine() != 'aarch64' or os.getuid() == 0:
        raise ValueError('start as a non-root user with Docker access on aarch64 Jetson Thor')
    caps = capture('nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader', timeout=10).decode().split()
    if caps != ['11.0']:
        raise ValueError(f'expected one SM110 GPU, got {caps}')
    raw = capture('docker', 'image', 'inspect', '--format',
                  '{{.Id}} {{.Os}}/{{.Architecture}} {{index .Config.Labels "io.github.ramborogers.inkling.profile"}}',
                  args.image, timeout=15).decode().split()
    if len(raw) != 3 or raw[1:] != ['linux/arm64', 'thor-v6'] or not raw[0].startswith('sha256:'):
        raise ValueError('expected the ARM64 thor-v6 image produced by build.py')
    args.image = raw[0]
    fetch_model.run(args.model_root, verify=True)
    mem = next((int(row.split()[1]) for row in Path('/proc/meminfo').read_text().splitlines()
                if row.startswith('MemAvailable:')), 0)
    if mem < 64 * 2**20:
        raise ValueError(f'need at least 64 GiB MemAvailable to attempt cold start; found {mem / 2**20:.1f} GiB')
    args.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    cid = capture('docker', 'create', *launch_args(args), timeout=90).decode().strip()
    try:
        run('docker', 'start', cid, timeout=90)
    except subprocess.CalledProcessError:
        # No force: never stop a running container or touch an existing name.
        subprocess.run(('docker', 'rm', cid), check=False, timeout=30)
        raise
    print(f'Launched {args.name}. Follow: docker logs -f {shlex.quote(args.name)}', flush=True)
    print(f'Check: python3 serve.py smoke --port {args.port} --wait 600', flush=True)


def http_json(port, endpoint, payload=None, timeout=10):
    raw = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}{endpoint}', data=raw,
                                 headers={'Content-Type': 'application/json'})
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with client.open(req, timeout=timeout) as resp:
        return json.load(resp)


def smoke_png():
    """Generate the solid red PNG used by the vision smoke test."""
    width = height = 64
    scanlines = (b'\x00' + bytes((220, 20, 20)) * width) * height

    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))

    header = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header) + chunk(b'IDAT', zlib.compress(scanlines)) + chunk(b'IEND', b'')


def smoke(port, wait):
    until = time.monotonic() + wait
    while True:
        try:
            models = http_json(port, '/v1/models')['data']
        except (OSError, ValueError, KeyError):
            if time.monotonic() >= until:
                raise
            time.sleep(2)
        else:
            break
    if [(m['id'], m.get('max_model_len')) for m in models] != [(MODEL, CONTEXT)]:
        raise ValueError('unexpected model ID or context limit')
    result = http_json(port, '/v1/chat/completions', {
        'model': MODEL, 'messages': [{'role': 'user', 'content': 'Reply with exactly this text and nothing else: READY'}],
        'temperature': 0, 'max_tokens': 256, 'reasoning_effort': 'none',
    }, timeout=60)
    c = result['choices'][0]
    if (result.get('model') != MODEL or c.get('finish_reason') != 'stop'
            or (c['message'].get('content') or '').strip() != 'READY'):
        raise ValueError('chat contract failed (expected exact READY with stop finish)')
    url = 'data:image/png;base64,' + base64.b64encode(smoke_png()).decode('ascii')
    result = http_json(port, '/v1/chat/completions', {
        'model': MODEL, 'messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': url}},
            {'type': 'text', 'text': 'What single primary color is this image? Reply in one word.'},
        ]}], 'temperature': 0, 'max_tokens': 128, 'reasoning_effort': 'low',
    }, timeout=60)
    c = result['choices'][0]
    color = (c['message'].get('content') or '').strip().lower().removesuffix('.')
    if result.get('model') != MODEL or c.get('finish_reason') != 'stop' or color != 'red':
        raise ValueError('vision contract failed (expected red with stop finish)')
    print(f'SMOKE OK (text + vision): {MODEL}, max_model_len={CONTEXT}, response=READY, color=red')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'start', 'smoke'))
    parser.add_argument('--port', type=int, default=8010)
    parser.add_argument('--name', default='inkling-thor-v6')
    parser.add_argument('--image', default=TAG)
    parser.add_argument('--model-root', type=Path, default=fetch_model.DEFAULT)
    parser.add_argument('--cache-dir', type=Path, default=fetch_model.DEFAULT.parent / 'runtime')
    parser.add_argument('--wait', type=int, default=0, help='seconds to wait for the model list on smoke')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.wait < 0:
        parser.error('port must be 1–65535 and wait must be non-negative')
    args.model_root = args.model_root.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    if any(',' in str(p) or '\n' in str(p) for p in (args.model_root, args.cache_dir)):
        parser.error('model/cache paths must not contain commas or newlines')
    try:
        if args.action == 'plan':
            print(shlex.join(['docker', 'run', '-d', *launch_args(args)]))
        elif args.action == 'start':
            start(args)
        else:
            smoke(args.port, args.wait)
    except (OSError, ValueError, IndexError, KeyError, subprocess.SubprocessError) as exc:
        parser.exit(1, f'{args.action} failed: {exc}\n')


if __name__ == '__main__':
    main()
