#!/usr/bin/env python3
"""Profile B12/B13 full-decode stage cost via ADAMAH_FULL_DECODE_STAGE_MASK.

Broadcom-focused diagnostic: loads the model once, then runs BOS@pos=0 forward
multiple times with different stage masks and prints timing deltas.
"""

import os
import sys
import time

from adam.paths import setup

setup()

from adamah_chat import load_model  # noqa: E402


DEFAULT_MASKS = [
    0x3F,  # all enabled (default)
    0x01,  # qkv + attn_norm
    0x02,  # rope + kv_write + attention
    0x04,  # attn_out proj + post_attn_norm + ffn_norm
    0x08,  # ffn gate+up(+act)
    0x10,  # ffn down + post_ffn_norm
    0x20,  # final output norm only
]


def _parse_masks(argv):
    if len(argv) <= 2:
        return DEFAULT_MASKS
    masks = []
    for s in argv[2:]:
        try:
            masks.append(int(s, 0))
        except Exception:
            pass
    return masks or DEFAULT_MASKS


def main(argv=None):
    argv = argv or sys.argv
    if len(argv) < 2:
        print("Usage: diag_full_decode_stage_probe.py <model.gguf> [mask ...]")
        return 2
    model = argv[1]
    masks = _parse_masks(argv)

    # Broadcom profiling defaults
    os.environ.setdefault("ADAM_BROADCOM_LM_HEAD_MODE", "cpu_top1")
    os.environ.setdefault("ADAM_RUNTIME_PROFILE", "broadcom_v3dv")

    startup = {
        "runtime_profile": "broadcom_v3dv",
        "stream_chunk_mb": 64,
        "gpu_fused_topk": True,
    }
    engine, tokenizer, _cfg, _GenConfig = load_model(model, startup=startup)

    engine._timing_enabled = True
    engine._reset_timing()
    engine.reset()
    _ = engine._forward(tokenizer.bos_id, 0, return_logits=False, sample_mode="cpu_top1")  # warm-up

    print("\nmask        total_ms   core_ms    lm_ms")
    print("----------  --------  --------  -------")
    for mask in masks:
        os.environ["ADAMAH_FULL_DECODE_STAGE_MASK"] = hex(mask)
        engine.reset()
        before = dict(engine.timing)
        t0 = time.perf_counter()
        _ = engine._forward(tokenizer.bos_id, 0, return_logits=False, sample_mode="cpu_top1")
        total_ms = (time.perf_counter() - t0) * 1000.0
        core_ms = (engine.timing.get("core_batch", 0.0) - before.get("core_batch", 0.0)) * 1000.0
        lm_ms = (engine.timing.get("lm_head_batch", 0.0) - before.get("lm_head_batch", 0.0)) * 1000.0
        print(f"{mask:#010x}  {total_ms:8.2f}  {core_ms:8.2f}  {lm_ms:7.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
