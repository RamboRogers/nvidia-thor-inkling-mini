#!/usr/bin/env python3
"""Stage the original, revision-pinned Inkling snapshot. No HF credentials or remote code."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

REPO = 'thinkingmachines/Inkling-2-Mini'
REV = 'c9d028302cd0f035fdf8e7e91f089a4ce94d2195'
DEFAULT = Path.home() / '.cache' / 'inkling-thor' / 'models'
# LFS entries are SHA-256 of file bytes; shorter hashes are Git blob IDs.
FILES = {
    'chat_template.jinja': (6868, 'cae99e08917fe3d9ff21551a7c9a835981c69032'),
    'config.json': (2722, 'e6c170c3fb666c50e3b509493b6b7a8f948b7d15'),
    'generation_config.json': (98, '1b5d49d129001d2be58c2f254bfae3221eab6d96'),
    'hf_quant_config.json': (67924, 'ec915fa4d2b07bfe26240f71e981d34778022baf'),
    'model.safetensors.index.json': (109300, '3b835a38066039133931b604a0734dfae89a644e'),
    'mtp.safetensors': (635801416, '7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2'),
    'processor_config.json': (461, '0e6df76a75a15f8af795dc08346dad6741f6421e'),
    'special_tokens_map.json': (529, 'e4fe9735b528be1c630eb832d655fc8c4c2c309f'),
    'tiktoken/tokenizer.model': (3386494, 'f170372dd655728909228220d77cf6b8104a59f38a91262a7d2c4027906ec0d8'),
    'tokenizer.json': (26178678, '7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6'),
    'tokenizer_config.json': (13235, 'abf8fb60451e6c33d2cfa3c4dfab9f8c1e7516aa'),
    'model-00001-of-00001.safetensors': (19526019168, '65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1'),
}


def check_file(path, n, expected):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != n:
        raise ValueError(f'bad type or size: {path.name}')
    h = hashlib.sha256() if len(expected) == 64 else hashlib.sha1(f'blob {n}\0'.encode())
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    if h.hexdigest() != expected:
        raise ValueError(f'checksum mismatch: {path.name}')


def inventory(tree, complete):
    if tree.is_symlink() or not tree.is_dir():
        raise ValueError('snapshot must be a real directory')
    allowed = set(FILES) | ({'.verified'} if complete else {s + '.part' for s in FILES})
    for p in tree.rglob('*'):
        rel = str(p.relative_to(tree))
        if p.is_symlink() or not ((p.is_file() and rel in allowed) or (p.is_dir() and rel == 'tiktoken')):
            raise ValueError(f'unexpected snapshot entry: {rel}')


def run(root, verify=False):
    if not root.is_dir() or root.resolve() != root.absolute():
        raise ValueError('stage root must already exist and have no symlink components')
    final, stage = root / REV, root / (REV + '.partial')
    # Lock the directory itself; read-only verification never creates a file.
    fd = os.open(root, os.O_RDONLY)
    try:
        fcntl.flock(fd, (fcntl.LOCK_SH if verify else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        if verify or final.exists() or final.is_symlink():
            inventory(final, True)
            for name, (n, h) in FILES.items():
                check_file(final / name, n, h)
            if json.loads((final / '.verified').read_text()) != {'repo': REPO, 'revision': REV, 'files': list(FILES)}:
                raise ValueError('receipt mismatch')
        else:
            stage.mkdir(exist_ok=True)
            inventory(stage, False)
            if shutil.disk_usage(root).free < sum(n for n, _ in FILES.values()) + 8 * 2**30:
                raise ValueError('insufficient disk space including reserve')
            for name, (n, h) in FILES.items():
                dest = stage / name
                dest.parent.mkdir(exist_ok=True)
                if not dest.exists():
                    part = dest.with_suffix(dest.suffix + '.part')
                    if not part.exists() or part.stat().st_size < n:
                        print(f'fetch {name} ({n} bytes)', flush=True)
                        subprocess.run(['curl', '-qfsSL', '--proto', '=https', '--proto-redir', '=https', '--retry', '3', '--connect-timeout', '15', '--max-time', '7200', '-C', '-', '-o', str(part), f'https://huggingface.co/{REPO}/resolve/{REV}/{name}'], check=True)
                    check_file(part, n, h)
                    part.rename(dest)
                else:
                    check_file(dest, n, h)
                print(f'verified {name}', flush=True)
            with (stage / '.verified').open('x') as f:
                json.dump({'repo': REPO, 'revision': REV, 'files': list(FILES)}, f)
            stage.rename(final)
        print(f'{final}: verified {len(FILES)} runtime files', flush=True)
    finally:
        os.close(fd)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DEFAULT)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    run(args.root, args.verify_only)
