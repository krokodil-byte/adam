#!/usr/bin/env python3
"""Local ADAM vs Ollama head-to-head diagnostic (same prompt, same token budget)."""
import argparse
import json
import subprocess
import urllib.error
import urllib.request

from adam.paths import setup
setup()

from adamah_chat import load_model


def _http_json(url: str, payload: dict):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _run_ollama(model: str, prompt: str, max_tokens: int):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_k": 1,
            "top_p": 1.0,
            "num_predict": int(max_tokens),
        },
    }
    out = _http_json("http://127.0.0.1:11434/api/generate", payload)
    eval_count = int(out.get("eval_count", 0) or 0)
    eval_ns = int(out.get("eval_duration", 0) or 0)
    tps = (eval_count / (eval_ns / 1e9)) if eval_ns > 0 else 0.0
    return {
        "response": (out.get("response") or "").strip(),
        "decode_tps": tps,
        "eval_count": eval_count,
        "eval_duration_ns": eval_ns,
        "prompt_eval_count": int(out.get("prompt_eval_count", 0) or 0),
        "prompt_eval_duration_ns": int(out.get("prompt_eval_duration", 0) or 0),
    }


def _ollama_backend_hint() -> str:
    try:
        txt = subprocess.check_output(
            ["ollama", "ps"], stderr=subprocess.STDOUT, text=True, timeout=10
        ).strip()
        if txt:
            return txt.splitlines()[-1].strip()
    except Exception:
        pass
    return "unknown"


def parse_args():
    p = argparse.ArgumentParser(description="Head-to-head ADAM vs Ollama (local).")
    p.add_argument("model", help="GGUF model path for ADAM")
    p.add_argument("--profile", default="auto")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--prompt", default="Write exactly: 2+2=4.")
    p.add_argument("--kv-cap", type=int, default=None)
    p.add_argument("--ollama-model", default="gemma3_1b_local")
    return p.parse_args()


def main():
    args = parse_args()
    startup = {
        "runtime_profile": args.profile,
        "stream_chunk_mb": 64,
        "gpu_fused_topk": True,
    }
    if args.kv_cap is not None:
        startup["kv_cap"] = int(args.kv_cap)
    engine, tokenizer, _cfg, GenConfig = load_model(args.model, startup=startup)
    gen_cfg = GenConfig(
        max_tokens=int(args.max_tokens),
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        repeat_penalty=1.0,
        eos_token_ids=(tokenizer.eos_id,),
    )
    prompt_ids = tokenizer.encode(args.prompt, add_bos=True)
    out_tokens, stats = engine.generate(prompt_ids, gen_cfg, stream=False, trace_decode=True)
    reply = tokenizer.decode(out_tokens)
    ts = stats.get("trace_summary") or {}
    ns = stats.get("native_stats") or {}

    print("=== ADAM ===")
    print(f"prompt                : {args.prompt!r}")
    print(f"decode_tps            : {stats.get('decode_tps', 0.0):.2f}")
    print(f"sampling_mode         : {stats.get('sampling_mode')}")
    print(f"dispatch_count        : {int(ns.get('dispatch_count', ts.get('dispatch_count_total', 0) or 0))}")
    print(f"submit_count          : {int(ns.get('submit_count', ts.get('submit_count_total', 0) or 0))}")
    print(f"barrier_count         : {int(ns.get('barrier_count', ts.get('barrier_count_total', 0) or 0))}")
    print(f"core_batch_ms         : {float(ts.get('core_batch_ms_avg', 0.0)):.3f}")
    print(f"lm_head_batch_ms      : {float(ts.get('lm_head_batch_ms_avg', 0.0)):.3f}")
    print(f"rerank_batch_ms       : {float(ts.get('rerank_batch_ms_avg', 0.0)):.3f}")
    print(f"reply_preview         : {reply[:100]!r}")

    print("")
    print("=== OLLAMA ===")
    try:
        oll = _run_ollama(args.ollama_model, args.prompt, args.max_tokens)
        print(f"model                 : {args.ollama_model}")
        print(f"decode_tps            : {oll['decode_tps']:.2f}")
        print(f"eval_count            : {oll['eval_count']}")
        print(f"eval_duration_ns      : {oll['eval_duration_ns']}")
        print(f"backend_hint          : {_ollama_backend_hint()}")
        print(f"reply_preview         : {oll['response'][:100]!r}")
        adam_tps = float(stats.get("decode_tps", 0.0))
        oll_tps = float(oll["decode_tps"])
        ratio = (adam_tps / oll_tps) if oll_tps > 0 else 0.0
        print("")
        print("=== Head-to-head ===")
        print(f"adam_vs_ollama_ratio  : {ratio:.3f}x")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"ollama_error          : {e}")


if __name__ == "__main__":
    main()
