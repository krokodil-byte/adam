#!/usr/bin/env python3
"""First-run bootstrap for ADAM + ADAMAH runtime."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
ADAMAH_ROOT = ROOT / "adamah-MAIN"
ADAMAH_PKG = ADAMAH_ROOT / "adamah"
SHADER_PROFILE_STAMP = ADAMAH_PKG / "shaders" / ".profile"
RUNTIME_REQUIREMENTS = (
    ("numpy", "numpy"),
    ("gguf", "gguf"),
    ("jinja2", "jinja2"),
)


def _run(cmd, cwd: Path | None = None) -> None:
    res = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
    )
    if res.returncode != 0:
        msg = res.stderr.strip() or res.stdout.strip() or f"command failed: {' '.join(cmd)}"
        raise RuntimeError(msg)


def _shader_profile() -> str:
    return (os.environ.get("ADAMAH_SHADER_PROFILE")
            or os.environ.get("ADAM_RUNTIME_PROFILE")
            or "").strip().lower()


def compiled_shader_profile() -> str:
    try:
        return SHADER_PROFILE_STAMP.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return ""


def _shader_compile_args() -> list[str]:
    profile = _shader_profile()
    args: list[str] = []
    if profile in ("broadcom_v3dv", "broadcom_v3dv_exact", "broadcom_v3dv_approx", "broadcom_v3dv_trace"):
        args.append("-DADAMAH_PROFILE_BROADCOM_V3DV_BALANCED=1")
    elif profile == "broadcom_v3dv_narrow":
        args.append("-DADAMAH_PROFILE_BROADCOM_V3DV_NARROW=1")
    return args


def _shader_target_args() -> list[str]:
    profile = _shader_profile()
    if profile.startswith("broadcom_v3dv"):
        return ["--target-env", "vulkan1.1"]
    return []


def ensure_python_deps(auto_install: bool = True) -> None:
    missing = [pkg for mod, pkg in RUNTIME_REQUIREMENTS if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    if not auto_install:
        raise RuntimeError(f"Missing Python dependencies: {', '.join(missing)}")
    _run([sys.executable, "-m", "pip", "install", *missing], cwd=ROOT)


def _compile_shaders(pkg_dir: Path) -> None:
    src_dir = pkg_dir / "shaders" / "src"
    if not src_dir.is_dir():
        return
    glslang = shutil.which("glslangValidator")
    if not glslang:
        if not _has_essential_shaders(pkg_dir):
            raise RuntimeError(
                "glslangValidator not found and essential precompiled shaders are missing. "
                "Install glslang-tools or restore adamah-MAIN/adamah/shaders from git."
            )
        return
    compile_args = _shader_compile_args()
    target_args = _shader_target_args()
    profile = _shader_profile() or "default"
    print(f"ADAMAH: Compiling shaders for profile '{profile}'")
    for dtype_dir in sorted(src_dir.iterdir()):
        if not dtype_dir.is_dir():
            continue
        out_dir = pkg_dir / "shaders" / dtype_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for comp_file in sorted(dtype_dir.glob("*.comp")):
            dst = out_dir / (comp_file.stem + ".spv")
            _run([glslang, "-V", *target_args, *compile_args, str(comp_file), "-o", str(dst)], cwd=ROOT)
    f32_dir = pkg_dir / "shaders" / "f32"
    root_dir = pkg_dir / "shaders"
    if f32_dir.is_dir():
        for spv in f32_dir.glob("*.spv"):
            shutil.copy2(spv, root_dir / spv.name)
    if not _has_essential_shaders(pkg_dir):
        raise RuntimeError(
            "Shader compilation finished without producing essential SPIR-V files. "
            "Expected adamah-MAIN/adamah/shaders/map_op1.spv and shaders/f32/map_op1.spv."
        )
    SHADER_PROFILE_STAMP.parent.mkdir(parents=True, exist_ok=True)
    SHADER_PROFILE_STAMP.write_text(profile, encoding="utf-8")


def _has_essential_shaders(pkg_dir: Path) -> bool:
    root_op1 = pkg_dir / "shaders" / "map_op1.spv"
    f32_op1 = pkg_dir / "shaders" / "f32" / "map_op1.spv"
    return root_op1.exists() and f32_op1.exists()


def _shader_outputs_stale(pkg_dir: Path) -> bool:
    src_dir = pkg_dir / "shaders" / "src"
    if not src_dir.is_dir():
        return False
    for dtype_dir in src_dir.iterdir():
        if not dtype_dir.is_dir():
            continue
        out_dir = pkg_dir / "shaders" / dtype_dir.name
        for comp_file in dtype_dir.glob("*.comp"):
            dst = out_dir / (comp_file.stem + ".spv")
            if not dst.exists():
                return True
            try:
                if comp_file.stat().st_mtime > dst.stat().st_mtime:
                    return True
            except OSError:
                return True
    return False


def _sync_root_shader_copies(pkg_dir: Path) -> None:
    f32_dir = pkg_dir / "shaders" / "f32"
    root_dir = pkg_dir / "shaders"
    if not f32_dir.is_dir():
        return
    root_dir.mkdir(parents=True, exist_ok=True)
    for spv in f32_dir.glob("*.spv"):
        dst = root_dir / spv.name
        if not dst.exists():
            shutil.copy2(spv, dst)


def _write_shader_header(pkg_dir: Path) -> Path:
    shader_hdr = pkg_dir / "_shader_path.h"
    shader_hdr.write_text(f'#define SHADER_PATH "{pkg_dir / "shaders"}"\n', encoding="utf-8")
    return shader_hdr


def _build_linux(pkg_dir: Path, force_rebuild: bool = False) -> Path:
    out_file = pkg_dir / "adamah.so"
    if out_file.exists() and not force_rebuild:
        return out_file
    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        raise RuntimeError("No C compiler found. Install gcc or clang.")
    shader_hdr = _write_shader_header(pkg_dir)
    cmd = [
        cc,
        "-shared",
        "-fPIC",
        "-O2",
        "-include",
        str(shader_hdr),
        str(pkg_dir / "adamah.c"),
        "-o",
        str(out_file),
        "-lvulkan",
        "-ldl",
        "-lm",
    ]
    if os.environ.get("ADAMAH_CROSS_COMPILE") != "1":
        cmd.insert(3, "-march=native")
    _run(cmd, cwd=ROOT)
    return out_file


def _build_windows_msvc(pkg_dir: Path, force_rebuild: bool = False) -> Path | None:
    out_file = pkg_dir / "adamah_opt.dll"
    if out_file.exists() and not force_rebuild:
        return out_file
    cl = shutil.which("cl")
    vulkan_sdk = os.environ.get("VULKAN_SDK")
    if not cl or not vulkan_sdk:
        return None
    include_dir = Path(vulkan_sdk) / "Include"
    lib_dir = Path(vulkan_sdk) / "Lib"
    vulkan_lib = lib_dir / "vulkan-1.lib"
    if not vulkan_lib.exists():
        return None
    shader_hdr = _write_shader_header(pkg_dir)
    cmd = [
        "cl",
        "/nologo",
        "/LD",
        "/O2",
        f"/I{include_dir}",
        f"/FI{shader_hdr}",
        str(pkg_dir / "adamah.c"),
        "/link",
        f"/OUT:{out_file}",
        str(vulkan_lib),
    ]
    _run(cmd, cwd=ROOT)
    return out_file


def _build_windows_gnu(pkg_dir: Path, force_rebuild: bool = False) -> Path:
    out_file = pkg_dir / "adamah_opt.dll"
    if out_file.exists() and not force_rebuild:
        return out_file
    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        raise RuntimeError("No MinGW/clang compiler found. Install gcc or clang.")
    shader_hdr = _write_shader_header(pkg_dir)
    cmd = [
        cc,
        "-shared",
        "-O3",
        "-std=c11",
        "-include",
        str(shader_hdr),
        str(pkg_dir / "adamah.c"),
        "-o",
        str(out_file),
    ]
    vulkan_sdk = os.environ.get("VULKAN_SDK")
    if vulkan_sdk:
        inc = Path(vulkan_sdk) / "Include"
        if inc.exists():
            cmd.extend(["-I", str(inc)])
    lib_candidates = [
        ROOT / "libvulkan-1.a",
    ]
    if vulkan_sdk:
        lib_candidates.append(Path(vulkan_sdk) / "Lib" / "libvulkan-1.a")
    lib_arg = next((str(p) for p in lib_candidates if p.exists()), None)
    if lib_arg is not None:
        cmd.append(lib_arg)
    else:
        cmd.append("-lvulkan-1")
    _run(cmd, cwd=ROOT)
    return out_file


def ensure_native_adamah(force_rebuild: bool = False, rebuild_shaders: bool = False) -> Path:
    pkg_dir = ADAMAH_PKG
    candidates = ["adamah_opt.dll", "adamah_new.dll", "adamah.dll"] if os.name == "nt" else ["adamah.so"]
    _sync_root_shader_copies(pkg_dir)
    if not _has_essential_shaders(pkg_dir):
        rebuild_shaders = True
    if _shader_outputs_stale(pkg_dir):
        rebuild_shaders = True
    if rebuild_shaders:
        _compile_shaders(pkg_dir)
        _sync_root_shader_copies(pkg_dir)
    if not force_rebuild:
        for name in candidates:
            path = pkg_dir / name
            if path.exists():
                if not _has_essential_shaders(pkg_dir):
                    raise RuntimeError(
                        "ADAMAH native library exists but essential shaders are missing. "
                        "Run `git restore adamah-MAIN/adamah/shaders` or "
                        "`python install_runtime.py --rebuild-shaders` after installing glslang-tools."
                    )
                return path
    _compile_shaders(pkg_dir)
    _sync_root_shader_copies(pkg_dir)
    if os.name == "nt":
        built = _build_windows_msvc(pkg_dir, force_rebuild=force_rebuild)
        if built is not None:
            return built
        return _build_windows_gnu(pkg_dir, force_rebuild=force_rebuild)
    return _build_linux(pkg_dir, force_rebuild=force_rebuild)


def ensure_runtime(auto_install: bool = True, build_native: bool = True,
                   force_rebuild: bool = False,
                   rebuild_shaders: bool = False) -> Path | None:
    ensure_python_deps(auto_install=auto_install)
    if build_native:
        return ensure_native_adamah(force_rebuild=force_rebuild, rebuild_shaders=rebuild_shaders)
    if rebuild_shaders:
        _compile_shaders(ADAMAH_PKG)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap ADAM + ADAMAH runtime")
    parser.add_argument("--no-python", action="store_true", help="skip Python dependency install")
    parser.add_argument("--no-native", action="store_true", help="skip native library build check")
    parser.add_argument("--rebuild-native", action="store_true", help="force native rebuild")
    parser.add_argument("--rebuild-shaders", action="store_true", help="force shader rebuild for the active shader profile")
    args = parser.parse_args(argv)

    path = ensure_runtime(
        auto_install=not args.no_python,
        build_native=not args.no_native,
        force_rebuild=args.rebuild_native,
        rebuild_shaders=args.rebuild_shaders or args.rebuild_native,
    )
    if path is not None:
        print(f"Runtime ready: {path}")
    else:
        print("Runtime ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
