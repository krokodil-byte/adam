# ADAM

ADAM is the orchestration/runtime layer around ADAMAH for local LLM inference
and diagnostics.

It includes:
- model loading and runtime profiles
- decode planning and execution paths
- profiling and diagnostics tooling
- chat/runtime integration utilities

## Repository Layout

- `adam/`: Python runtime and model engine
- `adamah-MAIN/`: ADAMAH Vulkan compute backend
- `tests/diagnostics/`: validation and performance diagnostics
- `tools/`: helper scripts

## Quick Start

```bash
python -m pip install -r requirements-runtime.txt
```

```bash
PYTHONUTF8=1 PYTHONPATH=. python -X utf8 tests/diagnostics/diag_inference.py gemma3-1b.gguf
```

## Authorship and AI Assistance

- Concept, architecture, and product direction: **Samuele Scuglia**.
- Implementation support used during development: **Claude**, **Gemini**,
  and **Codex**.

## License

- ADAM: [LICENSE](LICENSE) (CC BY-NC 4.0)
- ADAMAH backend: [adamah-MAIN/LICENSE](adamah-MAIN/LICENSE) (CC BY-NC 4.0)

