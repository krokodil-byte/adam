#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if command -v apt-get >/dev/null 2>&1; then
  echo "[install] Installing system packages with apt-get"
  sudo apt-get update
  sudo apt-get install -y python3 python3-pip python3-venv gcc libvulkan-dev glslang-tools
fi
if [ ! -x .venv/bin/python3 ]; then
  echo "[install] Creating local virtualenv in .venv"
  python3 -m venv .venv
fi
.venv/bin/python3 -m pip install --upgrade pip setuptools wheel
.venv/bin/python3 install_runtime.py
echo
echo "Runtime ready. Starting chat..."
.venv/bin/python3 adamah_chat.py
