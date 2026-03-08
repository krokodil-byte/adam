#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -x .venv/bin/python3 ]; then
  exec .venv/bin/python3 adamah_chat.py "$@"
fi
exec python3 adamah_chat.py "$@"
