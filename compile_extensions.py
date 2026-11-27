# SPDX-License-Identifier: Apache-2.0
"""Compile two SM110a extensions inside the pinned CUDA 13.0 build image."""
from pathlib import Path
import shutil

# This module runs only in the pinned image, not in the host Python environment.
from torch.utils.cpp_extension import _write_ninja_file_and_build_library  # type: ignore[import-not-found]


SRC = Path('/opt/thor-bf16x-src')
CUDA_WHEEL_HEADERS = '/usr/local/lib/python3.14/dist-packages/nvidia/cu13/include'
FLAGS = [
    '-gencode=arch=compute_110a,code=sm_110a', '-O3', '-std=c++20',
    '--expt-relaxed-constexpr', '-lineinfo', '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
    '-U__CUDA_NO_HALF2_OPERATORS__',
]


def build(name, sources, includes):
    output = Path('/out')
    directory = output / name
    directory.mkdir()
    # Compile/link without importing the new CUDA extension in this CPU-only container.
    _write_ninja_file_and_build_library(
        name, [str(SRC / p) for p in sources],
        extra_cflags=['-std=c++20'], extra_cuda_cflags=FLAGS,
        extra_sycl_cflags=[], extra_ldflags=[],
        extra_include_paths=[*(str(SRC / p) for p in includes), CUDA_WHEEL_HEADERS],
        build_directory=str(directory), with_cuda=True, with_sycl=False, verbose=True,
    )
    library = name + '.so'
    shutil.copyfile(directory / library, output / library)


if __name__ == '__main__':
    build('inkling_pico12_ext', [
        'inkling_pico12/csrc/pico12_ext.cu',
        'inkling_pico12/third_party/pico/kernels/gemv16.cu',
    ], ['inkling_pico12/third_party/pico'])
    build('inkling_wide_ext', ['inkling_wide/csrc/wide_ext.cu'], ['inkling_wide/csrc'])
