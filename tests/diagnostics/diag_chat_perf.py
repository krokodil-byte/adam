#!/usr/bin/env python3
"""Two-turn chat performance probe with prompt-prep and KV reuse attribution."""
import argparse
import os
import platform
import time

from adam.paths import setup; setup()

from adamah_chat import (
    _assistant_history_message,
    _chat_reuse_plan,
    _render_messages_tokens,
    load_model,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure two-turn chat overhead, KV reuse, and prompt prep cost."
    )
    parser.add_argument("model", help="Path to GGUF model file")
    parser.add_argument("--profile", default="auto")
    parser.add_argument(
        "--kv-cap",
        type=int,
        default=None,
        help="KV cache cap override. Default: runtime-profile auto (e.g. broadcom_v3dv -> 256).",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prompt1", default="hi")
    parser.add_argument(
        "--prompt2",
        default="don't think that's true ahahahah ants are smaller aren't they?",
    )
    parser.add_argument("--trace", action="store_true", help="Print per-stage timing breakdown")
    parser.add_argument("--rows-per-group", type=int, default=None,
                        help="Override gpu_fused_rows_per_group (default: use profile value)")
    parser.add_argument("--barrier-count", action="store_true",
                        help="Print Vulkan pipeline barrier count after Turn 2 (B10 diagnostic)")
    parser.add_argument("--split-proj", action="store_true",
                        help="Broadcom experiment: disable B12 monolithic decode and use split per-op projections.")
    return parser.parse_args(argv)


def _is_arm_like_host() -> bool:
    try:
        machine = os.uname().machine.lower()
    except AttributeError:
        machine = platform.machine().lower()
    return machine in ("aarch64", "arm64", "armv7l", "armv6l")


def _resolve_profile_name(profile_arg: str) -> str:
    name = str(profile_arg or "").strip().lower()
    if not name or name in ("auto", "default", "fast", "trace"):
        return "broadcom_v3dv" if _is_arm_like_host() else "desktop_discrete"
    return name


def _apply_fast_upload_defaults(runtime_profile: str) -> dict[str, str]:
    profile = str(runtime_profile or "").strip().lower()
    if not profile.startswith("broadcom_v3dv"):
        return {}
    defaults = {
        # Fast iterative loop on Pi: eager raw blocks + persistent upload caches.
        "ADAM_STREAM_LOAD": "0",
        "ADAM_Q4_MAP_CACHE": "1",
        "ADAM_Q8_MAP_CACHE": "1",
        "ADAM_B12_Q8_CACHE": "1",
        # Larger upload chunks reduce Python/native call overhead.
        "ADAM_F32_MAP_SCATTER_CHUNK": str(16 * 1024 * 1024),
        "ADAM_Q4_MAP_SCATTER_CHUNK": str(16 * 1024 * 1024),
        "ADAM_Q8_MAP_SCATTER_CHUNK": str(16 * 1024 * 1024),
        "ADAM_B12_Q8_SCATTER_CHUNK": str(16 * 1024 * 1024),
    }
    applied: dict[str, str] = {}
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


def _render_ids(messages, cfg, tokenizer):
    prompt, _add_bos, tokens, prep = _render_messages_tokens(messages, cfg, tokenizer)
    return prompt, tokens, float(prep.get("render_s", 0.0)), float(prep.get("encode_s", 0.0))


def main(argv=None):
    args = parse_args(argv)
    resolved_profile = _resolve_profile_name(args.profile)
    print(f"[diag_chat_perf] profile={args.profile} -> runtime_profile={resolved_profile}")
    applied_env = _apply_fast_upload_defaults(resolved_profile)
    if applied_env:
        pairs = " ".join(f"{k}={v}" for k, v in sorted(applied_env.items()))
        print(f"[diag_chat_perf] broadcom fast-upload defaults: {pairs}")
    startup: dict = {
        "runtime_profile": resolved_profile,
        "stream_chunk_mb": 64,
        "gpu_fused_topk": True,
    }
    if args.split_proj:
        os.environ["ADAM_BROADCOM_SPLIT_PROJ"] = "1"
    if args.kv_cap is not None:
        startup["kv_cap"] = int(args.kv_cap)
    if args.rows_per_group is not None:
        startup["gpu_fused_rows_per_group"] = args.rows_per_group
    engine, tokenizer, cfg, GenConfig = load_model(args.model, startup=startup)
    gen_cfg = GenConfig(
        max_tokens=int(args.max_tokens),
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        repeat_penalty=1.0,
        eos_token_ids=(tokenizer.eos_id,),
    )

    turn1_messages = [{"role": "user", "content": args.prompt1}]
    _, turn1_ids, turn1_render_s, turn1_encode_s = _render_ids(turn1_messages, cfg, tokenizer)
    turn1_tokens, turn1_stats = engine.generate(turn1_ids, gen_cfg, stream=False, trace_decode=args.trace)
    turn1_text = tokenizer.decode(turn1_tokens)

    turn2_messages = turn1_messages + [
        _assistant_history_message(turn1_text, turn1_tokens),
        {"role": "user", "content": args.prompt2},
    ]
    _, turn2_ids, turn2_render_s, turn2_encode_s = _render_ids(turn2_messages, cfg, tokenizer)
    reuse_plan = _chat_reuse_plan(list(turn1_ids) + list(turn1_tokens), turn2_ids, True)
    reuse_prefix = int(reuse_plan.get("reuse_prefix_tokens", 0) or 0)
    turn2_tokens, turn2_stats = engine.generate(
        turn2_ids,
        gen_cfg,
        stream=False,
        reuse_prefix=reuse_prefix,
        trace_decode=args.trace,
    )
    turn2_text = tokenizer.decode(turn2_tokens)

    print("")
    print("=== Turn 1 ===")
    print(f"prompt tokens  : {len(turn1_ids)}")
    print(f"render         : {turn1_render_s * 1000.0:.2f} ms")
    print(f"tokenize       : {turn1_encode_s * 1000.0:.2f} ms")
    print(f"prefill        : {turn1_stats.get('prefill_s', 0.0) * 1000.0:.2f} ms")
    print(f"decode         : {turn1_stats.get('decode_s', 0.0) * 1000.0:.2f} ms")
    print(f"decode tps     : {turn1_stats.get('decode_tps', 0.0):.2f}")
    print(f"reply preview  : {turn1_text[:100]!r}")

    print("")
    print("=== Turn 2 ===")
    print(f"prompt tokens  : {len(turn2_ids)}")
    print(f"render         : {turn2_render_s * 1000.0:.2f} ms")
    print(f"tokenize       : {turn2_encode_s * 1000.0:.2f} ms")
    print(f"kv reuse hit   : {'yes' if reuse_plan.get('reuse_hit') else 'no'}")
    print(f"kv reused      : {int(turn2_stats.get('n_prompt_reused', reuse_prefix) or 0)}")
    print(f"reuse miss     : {reuse_plan.get('reuse_miss_reason')}")
    print(f"reuse miss idx : {int(reuse_plan.get('reuse_miss_index', -1) or -1)}")
    print(f"prefilled      : {int(turn2_stats.get('n_prompt_prefilled', 0) or 0)}")
    print(f"prefill        : {turn2_stats.get('prefill_s', 0.0) * 1000.0:.2f} ms")
    print(f"decode         : {turn2_stats.get('decode_s', 0.0) * 1000.0:.2f} ms")
    print(f"decode tps     : {turn2_stats.get('decode_tps', 0.0):.2f}")
    print(f"reply preview  : {turn2_text[:100]!r}")

    ts = turn2_stats.get("trace_summary") or {}
    tm = turn2_stats.get("timing") or {}
    ns = turn2_stats.get("native_stats") or {}
    dispatch_total = int(ns.get("dispatch_count", ts.get("dispatch_count_total", 0) or 0))
    submit_total = int(ns.get("submit_count", ts.get("submit_count_total", 0) or 0))
    barrier_total = int(ns.get("barrier_count", ts.get("barrier_count_total", 0) or 0))
    core_ms = float(ts.get("core_batch_ms_avg", tm.get("core_batch", 0.0)))
    lm_ms = float(ts.get("lm_head_batch_ms_avg", tm.get("lm_head_batch", 0.0)))
    rerank_ms = float(ts.get("rerank_batch_ms_avg", tm.get("rerank_batch", 0.0)))
    print("")
    print("=== Runtime metrics (Turn 2) ===")
    print(f"decode_tps          : {turn2_stats.get('decode_tps', 0.0):.2f}")
    print(f"dispatch_count      : {dispatch_total}")
    print(f"submit_count        : {submit_total}")
    print(f"barrier_count       : {barrier_total}")
    print(f"core_batch_ms       : {core_ms:.3f}")
    print(f"lm_head_batch_ms    : {lm_ms:.3f}")
    print(f"rerank_batch_ms     : {rerank_ms:.3f}")

    if args.barrier_count:
        gpu = getattr(engine, 'gpu', None)
        bc = gpu.get_last_barrier_count() if (gpu and getattr(gpu, '_has_get_last_barrier_count', False)) else None
        print("")
        print("=== Barrier count (B10 diagnostic) ===")
        if bc is not None:
            print(f"  barriers/batch : {bc}")
            t2_tps = turn2_stats.get('decode_tps', 0.0)
            if t2_tps > 0:
                print(f"  implied us/bar : {(1000.0 / t2_tps - 0.5) / bc * 1000.0:.1f} μs  (rough: (step_ms - 0.5ms) / barriers)")
        else:
            print("  get_last_barrier_count not available in this DLL")

    if args.trace:
        print("")
        print("=== Trace (Turn 2 decode, avg per token) ===")
        keys = [
            "embed",
            "norm",
            "qkv",
            "qk_norm",
            "rope",
            "attn",
            "attn_out",
            "ffn",
            "lm_head",
            "lm_head_shortlist",
            "sample_resolve",
            "core_batch",
            "lm_head_batch",
            "rerank_batch",
        ]
        total_ms = ts.get("forward_ms_avg", 0.0) + ts.get("sample_ms_avg", 0.0)
        shown_forward = 0.0
        for k in keys:
            v = ts.get(f"{k}_ms_avg", tm.get(k, 0.0))
            if isinstance(v, float) and v > 0.0:
                print(f"  {k:<20}: {v:7.3f} ms")
                if k not in ("sample_resolve",):
                    shown_forward += v
        forward_avg = ts.get("forward_ms_avg", 0.0)
        residual = forward_avg - shown_forward
        if residual > 1e-3:
            print(f"  {'unattributed_fwd':<20}: {residual:7.3f} ms")
        print(f"  {'forward_avg':<20}: {forward_avg:7.3f} ms")
        print(f"  {'sample_avg':<20}: {ts.get('sample_ms_avg', 0.0):7.3f} ms")
        print(f"  {'step_avg':<20}: {ts.get('step_ms_avg', total_ms):7.3f} ms")
        tps = ts.get('decode_tps') or turn2_stats.get('decode_tps', 0.0)
        print(f"  {'decode_tps':<20}: {tps:7.2f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
