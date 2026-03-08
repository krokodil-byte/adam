#!/usr/bin/env python3
"""Prepare a single GitHub-ready universal runtime bundle."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASES = ROOT / "releases"
BUNDLE_NAME = "github-universal"
BUNDLE_ROOT = RELEASES / BUNDLE_NAME
ADAMAH_VERSION = "5.2.0"
ADAMAH_RELEASE_NAME = f"adamah-v{ADAMAH_VERSION}"
ADAMAH_RELEASE_ROOT = RELEASES / ADAMAH_RELEASE_NAME

COMMON_TOP_LEVEL = [
    "adam",
    "adamah_chat.py",
    "runtime_bootstrap.py",
    "install_runtime.py",
    "requirements-runtime.txt",
    "LICENSE",
]

COMMON_ADAMAH_ROOT = [
    "README.md",
    "setup.py",
    "pyproject.toml",
    "LICENSE",
]

COMMON_ADAMAH_PACKAGE = [
    "__init__.py",
    "uucis.py",
    "adamah.c",
    "shaders",
]


WINDOWS_INSTALL = (
    "@echo off\r\n"
    "setlocal\r\n"
    "cd /d %~dp0\r\n"
    "python install_runtime.py\r\n"
    "if errorlevel 1 (\r\n"
    "  echo.\r\n"
    "  echo Install failed.\r\n"
    "  pause\r\n"
    "  exit /b %errorlevel%\r\n"
    ")\r\n"
    "echo.\r\n"
    "echo Runtime ready. Starting chat...\r\n"
    "python adamah_chat.py\r\n"
    "pause\r\n"
)

WINDOWS_RUN = (
    "@echo off\r\n"
    "setlocal\r\n"
    "cd /d %~dp0\r\n"
    "python adamah_chat.py %*\r\n"
)

LINUX_INSTALL = (
    "#!/usr/bin/env bash\n"
    "set -euo pipefail\n"
    "cd \"$(dirname \"$0\")\"\n"
    "if command -v apt-get >/dev/null 2>&1; then\n"
    "  echo \"[install] Installing system packages with apt-get\"\n"
    "  sudo apt-get update\n"
    "  sudo apt-get install -y python3 python3-pip python3-venv gcc libvulkan-dev glslang-tools\n"
    "fi\n"
    "if [ ! -x .venv/bin/python3 ]; then\n"
    "  echo \"[install] Creating local virtualenv in .venv\"\n"
    "  python3 -m venv .venv\n"
    "fi\n"
    ".venv/bin/python3 -m pip install --upgrade pip setuptools wheel\n"
    ".venv/bin/python3 install_runtime.py\n"
    "echo\n"
    "echo \"Runtime ready. Starting chat...\"\n"
    ".venv/bin/python3 adamah_chat.py\n"
)

LINUX_RUN = (
    "#!/usr/bin/env bash\n"
    "set -euo pipefail\n"
    "cd \"$(dirname \"$0\")\"\n"
    "if [ -x .venv/bin/python3 ]; then\n"
    "  exec .venv/bin/python3 adamah_chat.py \"$@\"\n"
    "fi\n"
    "exec python3 adamah_chat.py \"$@\"\n"
)


def _copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(
            src,
            dst,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.egg-info"),
        )
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _clear_generated_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in path.iterdir():
        if child.name == ".git":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _bundle_readme() -> str:
    return (
        "# ADAMAH Chat Universal Bundle\n\n"
        "Single GitHub release for Windows and Linux.\n\n"
        "## First run\n\n"
        "- Windows: double-click `install_windows.bat`\n"
        "- Linux x64/x86/ARM: run or double-click `install_linux.sh` in a terminal-enabled file manager\n\n"
        "Those installers will:\n\n"
        "- create a local `.venv` on Linux (avoids Debian/Raspberry Pi OS pip restrictions)\n"
        "- install Python runtime dependencies (`numpy`, `gguf`, `jinja2`)\n"
        "- build the ADAMAH native library locally for the current machine\n"
        "- start `adamah_chat.py`\n\n"
        "## Later runs\n\n"
        "- Windows: `run_chat_windows.bat`\n"
        "- Linux: `run_chat_linux.sh`\n\n"
        "## Models\n\n"
        "Place one or more `.gguf` model files inside the bundle root before starting the chat.\n\n"
        "## Linux system packages\n\n"
        "The Linux installer attempts `apt-get` automatically when available. If your distro is not Debian-based, "
        "install equivalents for:\n\n"
        "- `python3`\n"
        "- `python3-pip`\n"
        "- `python3-venv`\n"
        "- `gcc` or `clang`\n"
        "- `libvulkan-dev`\n"
        "- `glslang-tools` (optional but recommended)\n"
    )


def _adamah_release_readme() -> str:
    return (
        f"# ADAMAH {ADAMAH_VERSION}\n\n"
        "Standalone ADAMAH package release.\n\n"
        "## Install\n\n"
        "```bash\n"
        "python3 -m venv .venv\n"
        ". .venv/bin/activate\n"
        "pip install .\n"
        "```\n\n"
        "Using a local virtualenv avoids `externally-managed-environment` errors on Debian and Raspberry Pi OS.\n\n"
        "## What is included\n\n"
        "- `pyproject.toml`\n"
        "- `setup.py`\n"
        "- `README.md`\n"
        "- `LICENSE`\n"
        "- `adamah/` package sources\n"
        "- all shader sources and precompiled `.spv` files\n\n"
        "## Notes\n\n"
        "- Native compilation still happens locally during install/bootstrap.\n"
        "- No precompiled platform binaries are required in this release asset.\n"
    )


def build_adamah_release() -> None:
    RELEASES.mkdir(parents=True, exist_ok=True)
    _clear_generated_dir(ADAMAH_RELEASE_ROOT)
    for rel in COMMON_ADAMAH_ROOT:
        _copy_path(ROOT / "adamah-MAIN" / rel, ADAMAH_RELEASE_ROOT / rel)
    adamah_pkg_root = ADAMAH_RELEASE_ROOT / "adamah"
    adamah_pkg_root.mkdir(exist_ok=True)
    for rel in COMMON_ADAMAH_PACKAGE:
        _copy_path(ROOT / "adamah-MAIN" / "adamah" / rel, adamah_pkg_root / rel)
    _write_text(ADAMAH_RELEASE_ROOT / "README.md", _adamah_release_readme())
    manifest = {
        "bundle": ADAMAH_RELEASE_NAME,
        "version": ADAMAH_VERSION,
        "package": "adamah",
        "install": "pip install .",
        "files": {
            "root": COMMON_ADAMAH_ROOT,
            "package": COMMON_ADAMAH_PACKAGE,
        },
    }
    _write_text(ADAMAH_RELEASE_ROOT / "bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
    archive = shutil.make_archive(
        str(ADAMAH_RELEASE_ROOT),
        "zip",
        root_dir=str(RELEASES),
        base_dir=ADAMAH_RELEASE_NAME,
    )
    print(f"Prepared {ADAMAH_RELEASE_ROOT}")
    print(f"Created {archive}")


def build_bundle() -> None:
    RELEASES.mkdir(parents=True, exist_ok=True)
    _clear_generated_dir(BUNDLE_ROOT)

    for rel in COMMON_TOP_LEVEL:
        _copy_path(ROOT / rel, BUNDLE_ROOT / rel)

    adamah_root = BUNDLE_ROOT / "adamah-MAIN"
    adamah_root.mkdir(exist_ok=True)
    for rel in COMMON_ADAMAH_ROOT:
        _copy_path(ROOT / "adamah-MAIN" / rel, adamah_root / rel)

    adamah_pkg_root = adamah_root / "adamah"
    adamah_pkg_root.mkdir(exist_ok=True)
    for rel in COMMON_ADAMAH_PACKAGE:
        _copy_path(ROOT / "adamah-MAIN" / "adamah" / rel, adamah_pkg_root / rel)

    _write_text(BUNDLE_ROOT / "install_windows.bat", WINDOWS_INSTALL)
    _write_text(BUNDLE_ROOT / "run_chat_windows.bat", WINDOWS_RUN)
    _write_text(BUNDLE_ROOT / "install_linux.sh", LINUX_INSTALL)
    _write_text(BUNDLE_ROOT / "run_chat_linux.sh", LINUX_RUN)
    _write_text(BUNDLE_ROOT / "README.md", _bundle_readme())

    manifest = {
        "bundle": BUNDLE_NAME,
        "top_level": COMMON_TOP_LEVEL,
        "adamah_root": COMMON_ADAMAH_ROOT,
        "adamah_package_common": COMMON_ADAMAH_PACKAGE,
        "platforms": {
            "windows": ["install_windows.bat", "run_chat_windows.bat"],
            "linux": ["install_linux.sh", "run_chat_linux.sh"],
        },
    }
    _write_text(BUNDLE_ROOT / "bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")

    archive = shutil.make_archive(str(BUNDLE_ROOT), "zip", root_dir=str(RELEASES), base_dir=BUNDLE_NAME)
    print(f"Prepared {BUNDLE_ROOT}")
    print(f"Created {archive}")


def main() -> None:
    build_bundle()
    build_adamah_release()


if __name__ == "__main__":
    main()
