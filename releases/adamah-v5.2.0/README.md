# ADAMAH 5.2.0

Standalone ADAMAH package release.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install .
```

Using a local virtualenv avoids `externally-managed-environment` errors on Debian and Raspberry Pi OS.

## What is included

- `pyproject.toml`
- `setup.py`
- `README.md`
- `LICENSE`
- `adamah/` package sources
- all shader sources and precompiled `.spv` files

## Notes

- Native compilation still happens locally during install/bootstrap.
- No precompiled platform binaries are required in this release asset.
