#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rebuild the v6 Thor runtime from pinned public sources, without model weights."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent
BASE = 'vllm/vllm-openai@sha256:ecdf0ccc663d86332459f5a50a38c30861a5079bc3924aa08fdc657ef10dd648'
BASE_ID = 'sha256:f95362fc89f98d28a429d7047caa9090ccab881e1c4ffd0385f643627211c6de'
VLLM = '/usr/local/lib/python3.14/dist-packages/vllm'
DONOR = 'https://github.com/cqingwang/Inkling-2-Mini.git'
DONOR_REV = '54e2d9456738591ecd057a1a29586b0a69bfbac6'
TAG = 'inkling-mini:thor-v6'


def run(*args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def capture(*args, **kwargs):
    return run(*args, stdout=subprocess.PIPE, **kwargs).stdout


def check_bytes(raw, expected, label):
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('SHA-256 mismatch: ' + label)
    return raw


def install(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def read_tar(raw, names):
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as tar:
        members = tar.getmembers()
        if len(members) != len(names) or {m.name for m in members} != set(names):
            raise ValueError('unexpected input archive inventory')
        if any(not m.isfile() for m in members):
            raise ValueError('non-file entry in input archive')
        result = {}
        for member in members:
            reader = tar.extractfile(member)
            if reader is None:
                raise ValueError('missing input file: ' + member.name)
            with reader:
                result[member.name] = reader.read()
        return result


def stage_patch(raw, manifest, output):
    output.mkdir()
    base_files = {p: row['base_sha256'] for p, row in manifest.items() if row['base_sha256']}
    for p, data in read_tar(raw, base_files).items():
        install(output / p, check_bytes(data, base_files[p], p))
    patch = ROOT / 'patches' / 'vllm-thor.patch'
    run('git', '-c', 'apply.ignoreWhitespace=no', 'apply', '--whitespace=nowarn', str(patch), cwd=output)
    if {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()} != set(manifest):
        raise ValueError('unexpected patched file inventory')
    for p, meta in manifest.items():
        data = check_bytes((output / p).read_bytes(), meta['patched_sha256'], p)
        compile(data, p, 'exec')


def stage_donor(work, manifest, output, licenses):
    repo = work / 'donor.git'
    run('git', 'init', '--quiet', '--bare', str(repo))
    run('git', '-C', str(repo), 'fetch', '--quiet', '--depth=1', '--no-tags', DONOR, DONOR_REV, timeout=180)
    got = capture('git', '-C', str(repo), 'rev-parse', 'FETCH_HEAD').decode().strip()
    if got != DONOR_REV:
        raise ValueError('incorrect donor revision')
    for p, sha in manifest.items():
        raw = capture('git', '-C', str(repo), 'show', f'{DONOR_REV}:{p}')
        dest = output / p.removeprefix('overlay/') if p.startswith('overlay/') else licenses / p
        install(dest, check_bytes(raw, sha, p))


def offline_container(*extra):
    return ('docker', 'run', '--rm', '--pull=never', '--runtime=runc', '--network=none',
            '--read-only', '--cap-drop=ALL', '--security-opt=no-new-privileges',
            '--user', f'{os.getuid()}:{os.getgid()}', *extra)


def compile_extensions(src, target, manifest, prebuilt):
    target.mkdir()
    origin = prebuilt
    if origin is None:
        origin = src.parent.parent / 'compiled'
        origin.mkdir()
        run(*offline_container('--cpus=1', '--memory=8g', '--memory-swap=8g', '--pids-limit=128',
                               '--tmpfs', '/tmp:rw,nosuid,nodev,size=1g', '-w', '/tmp',
                               '-e', 'HOME=/tmp', '-e', 'CUDA_HOME=/usr/local/cuda', '-e', 'MAX_JOBS=1',
                               '--mount', f'type=bind,src={src},dst=/opt/thor-bf16x-src,readonly',
                               '--mount', f'type=bind,src={ROOT / "compile_extensions.py"},dst=/compile.py,readonly',
                               '--mount', f'type=bind,src={origin},dst=/out',
                               '--entrypoint=python3', BASE_ID, '-IB', '/compile.py'), timeout=900)
    for name, pinned_sha in manifest.items():
        raw = (origin / name).read_bytes()
        if raw[:4] != b'\x7fELF' or raw[4:6] != b'\x02\x01' or raw[18:20] != b'\xb7\x00':
            raise ValueError('expected a little-endian AArch64 ELF library: ' + name)
        if prebuilt is not None:
            check_bytes(raw, pinned_sha, name)
        install(target / name, raw)
        print(f'{name}: sha256:{hashlib.sha256(raw).hexdigest()}', flush=True)


def image_identity(ref):
    return capture('docker', 'image', 'inspect', '--format', '{{.Id}} {{.Os}}/{{.Architecture}}', ref).decode().strip()


def build(prebuilt, tag, local_base):
    if platform.system() != 'Linux' or platform.machine() != 'aarch64' or os.getuid() == 0:
        raise ValueError('build as a non-root user with Docker access on aarch64 Linux (Jetson Thor)')
    if not local_base:
        run('docker', 'pull', '--platform', 'linux/arm64', BASE, timeout=1800)
    if image_identity(BASE) != BASE_ID + ' linux/arm64':
        raise ValueError('pinned vLLM base image identity does not match')
    manifest = json.loads((ROOT / 'patches/manifest.json').read_text())
    with tempfile.TemporaryDirectory(prefix='inkling-thor-build-') as path:
        work = Path(path).resolve()
        ctx = work / 'context'
        ctx.mkdir()
        base_paths = [p for p, row in manifest['vllm'].items() if row['base_sha256']]
        raw = capture(*offline_container('--cpus=1', '--memory=256m', '--entrypoint=/bin/tar',
                                         BASE_ID, '-C', VLLM, '-cf', '-', *base_paths), timeout=120)
        stage_patch(raw, manifest['vllm'], ctx / 'vllm')
        stage_donor(work, manifest['donor'], ctx / 'thor-opt', ctx / 'notices' / 'cqingwang')
        compatibility = ctx / 'notices' / 'thor-compatibility'
        install(compatibility / 'LICENSE', (ROOT / 'LICENSE').read_bytes())
        shutil.copyfile(ROOT / 'patches/NOTICE', compatibility / 'NOTICE')
        compile_extensions(ctx / 'thor-opt', ctx / 'thor-opt' / 'ext', manifest['prebuilt'], prebuilt)
        (ctx / 'Dockerfile').write_text(
            f'FROM {BASE}\n'
            'COPY thor-opt/ /opt/thor-opt/\n'
            f'COPY vllm/ {VLLM}/\n'
            'COPY notices/ /usr/share/doc/inkling-thor/\n'
            'LABEL org.opencontainers.image.source="https://github.com/RamboRogers/nvidia-thor-inkling-mini"\n'
            'LABEL io.github.ramborogers.inkling.profile="thor-v6"\n'
        )
        for item in ctx.rglob('*'):
            item.chmod(0o755 if item.is_dir() or item.suffix == '.so' else 0o644)
            os.utime(item, (0, 0))
        run('docker', 'build', '--pull=false', '--network=none', '--platform=linux/arm64',
            '--build-arg', 'SOURCE_DATE_EPOCH=0', '-t', tag, str(ctx), timeout=300)
    print(f'{tag}: {image_identity(tag)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default=TAG)
    parser.add_argument('--local-base', action='store_true', help='use the already-pulled digest; skip registry pull')
    parser.add_argument('--prebuilt-extensions', type=Path, metavar='DIR',
                        help='use exact SHA-256-pinned extensions from the original Thor run (normally compile from source)')
    args = parser.parse_args()
    try:
        build(args.prebuilt_extensions, args.tag, args.local_base)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f'Build failed: {exc}\n')


if __name__ == '__main__':
    main()
