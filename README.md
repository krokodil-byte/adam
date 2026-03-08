# ADAM (Universal Chat Bundle)

Single GitHub release for Windows and Linux.

## First run

- Windows: double-click `install_windows.bat`
- Linux x64/x86/ARM: run or double-click `install_linux.sh` in a terminal-enabled file manager

Those installers will:

- install Python runtime dependencies (`numpy`, `gguf`)
- build the ADAMAH native library locally for the current machine
- start `adamah_chat.py`

## Later runs

- Windows: `run_chat_windows.bat`
- Linux: `run_chat_linux.sh`

## Models

Place one or more `.gguf` model files inside the bundle root before starting the chat.

## Linux system packages

The Linux installer attempts `apt-get` automatically when available. If your distro is not Debian-based, install equivalents for:

- `python3`
- `python3-pip`
- `gcc` or `clang`
- `libvulkan-dev`
- `glslang-tools` (optional but recommended)
