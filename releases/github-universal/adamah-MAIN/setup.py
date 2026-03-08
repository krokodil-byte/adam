"""
ADAMAH build system — compiles the native Vulkan library and GLSL shaders at install time.

    pip install .
"""

import os
import sys
import subprocess
import shutil
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop


def get_pkg_dir():
    """Get the adamah package directory."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'adamah')


def compile_shaders(pkg_dir):
    """Compile all .comp GLSL shaders to .spv SPIR-V."""
    src_dir = os.path.join(pkg_dir, 'shaders', 'src')
    if not os.path.isdir(src_dir):
        print("ADAMAH: No shader sources found, using precompiled .spv")
        return True

    glslang = shutil.which('glslangValidator')
    if not glslang:
        print("WARNING: glslangValidator not found. Using precompiled shaders.")
        print("  Install with: sudo apt install glslang-tools")
        return True  # Non-fatal, precompiled .spv should be there

    ok = True
    for dtype_dir in sorted(os.listdir(src_dir)):
        dtype_src = os.path.join(src_dir, dtype_dir)
        if not os.path.isdir(dtype_src):
            continue

        dtype_out = os.path.join(pkg_dir, 'shaders', dtype_dir)
        os.makedirs(dtype_out, exist_ok=True)

        for comp_file in sorted(os.listdir(dtype_src)):
            if not comp_file.endswith('.comp'):
                continue
            src_path = os.path.join(dtype_src, comp_file)
            spv_name = comp_file.replace('.comp', '.spv')
            dst_path = os.path.join(dtype_out, spv_name)

            print(f"  {dtype_dir}/{comp_file} -> {spv_name}")
            ret = subprocess.run(
                [glslang, '-V', src_path, '-o', dst_path],
                capture_output=True, text=True
            )
            if ret.returncode != 0:
                print(f"  ERROR: {ret.stderr.strip()}")
                ok = False

    # Copy f32 shaders to root for backward compat
    f32_dir = os.path.join(pkg_dir, 'shaders', 'f32')
    shaders_dir = os.path.join(pkg_dir, 'shaders')
    if os.path.isdir(f32_dir):
        for spv in os.listdir(f32_dir):
            if spv.endswith('.spv'):
                shutil.copy2(os.path.join(f32_dir, spv), os.path.join(shaders_dir, spv))

    return ok


def compile_native(pkg_dir):
    """Compile adamah.c into the platform-native shared library."""
    c_file = os.path.join(pkg_dir, 'adamah.c')
    out_name = 'adamah_opt.dll' if os.name == 'nt' else 'adamah.so'
    so_file = os.path.join(pkg_dir, out_name)

    if not os.path.exists(c_file):
        if os.path.exists(so_file):
            print(f"ADAMAH: Using existing {out_name}")
            return True
        print(f"ERROR: adamah.c not found and no precompiled {out_name}")
        return False

    shader_path = os.path.join(pkg_dir, 'shaders')

    # Detect compiler
    cc = os.environ.get('CC', 'gcc')

    # Write shader path to a temporary header to avoid quoting issues with spaces
    shader_hdr = os.path.join(pkg_dir, '_shader_path.h')
    with open(shader_hdr, 'w') as f:
        f.write(f'#define SHADER_PATH "{shader_path}"\n')
    if os.name == 'nt':
        if os.path.basename(cc).lower().startswith('cl'):
            vulkan_sdk = os.environ.get('VULKAN_SDK')
            if not vulkan_sdk:
                print("ERROR: VULKAN_SDK is required for MSVC builds")
                return False
            include_dir = os.path.join(vulkan_sdk, 'Include')
            vulkan_lib = os.path.join(vulkan_sdk, 'Lib', 'vulkan-1.lib')
            if not os.path.exists(vulkan_lib):
                print(f"ERROR: Vulkan import library not found: {vulkan_lib}")
                return False
            cmd = [
                cc, '/nologo', '/LD', '/O2',
                f'/I{include_dir}',
                f'/FI{shader_hdr}',
                c_file,
                '/link',
                f'/OUT:{so_file}',
                vulkan_lib,
            ]
        else:
            cflags = ['-shared', '-O3', '-std=c11', '-include', shader_hdr]
            vulkan_sdk = os.environ.get('VULKAN_SDK')
            if vulkan_sdk:
                include_dir = os.path.join(vulkan_sdk, 'Include')
                if os.path.isdir(include_dir):
                    cflags.extend(['-I', include_dir])
            lib_candidates = [
                os.path.join(os.path.dirname(os.path.dirname(pkg_dir)), 'libvulkan-1.a'),
            ]
            if vulkan_sdk:
                lib_candidates.append(os.path.join(vulkan_sdk, 'Lib', 'libvulkan-1.a'))
            lib_arg = next((p for p in lib_candidates if os.path.exists(p)), '-lvulkan-1')
            cmd = [cc] + cflags + [c_file, '-o', so_file, lib_arg]
    else:
        cflags = ['-shared', '-fPIC', '-O2']
        if os.environ.get('ADAMAH_CROSS_COMPILE') != '1':
            cflags.append('-march=native')
        cflags.extend(['-include', shader_hdr])
        libs = ['-lvulkan', '-ldl', '-lm']
        cmd = [cc] + cflags + [c_file, '-o', so_file] + libs
    print(f"ADAMAH: Compiling {c_file}")
    print(f"  {' '.join(cmd)}")

    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"ERROR compiling {out_name}:\n{ret.stderr}")
        return False

    print(f"ADAMAH: {so_file} compiled successfully")
    return True


class BuildWithNative(build_py):
    """Custom build that compiles shaders and C library."""

    def run(self):
        pkg_dir = get_pkg_dir()

        print("\n=== ADAMAH Build ===")
        print("Compiling GLSL shaders...")
        compile_shaders(pkg_dir)

        print("Compiling native Vulkan library...")
        if not compile_native(pkg_dir):
            print("\nERROR: Failed to compile native ADAMAH library")
            if os.name == 'nt':
                print("Requirements: MSVC or gcc/clang, plus Vulkan SDK")
            else:
                print("Requirements: gcc/clang, libvulkan-dev")
                print("  sudo apt install gcc libvulkan-dev")
            sys.exit(1)

        print("=== Build complete ===\n")
        super().run()


class DevelopWithNative(develop):
    """Custom develop that compiles shaders and C library."""

    def run(self):
        pkg_dir = get_pkg_dir()
        print("\n=== ADAMAH Build (develop) ===")
        compile_shaders(pkg_dir)
        if not compile_native(pkg_dir):
            sys.exit(1)
        print("=== Build complete ===\n")
        super().run()


setup(
    cmdclass={
        'build_py': BuildWithNative,
        'develop': DevelopWithNative,
    },
)
