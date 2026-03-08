#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if command -v apt-get >/dev/null 2>&1; then
  echo "[install] Installing system packages with apt-get"
  sudo apt-get update
  sudo apt-get install -y python3 python3-pip gcc libvulkan-dev glslang-tools
fi
python3 install_runtime.py
echo
echo "Runtime ready. Starting chat..."
python3 adamah_chat.py
