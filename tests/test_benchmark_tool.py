#!/usr/bin/env python3
"""Smoke tests for the comparative benchmark helpers."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [ROOT, os.path.join(ROOT, "adamah-MAIN")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from adam.tools import benchmark


def _llama_ok_record():
    return benchmark._finalize_run_record({
        "backend": "llama",
        "status": "ok",
        "n_prompt": 5,
        "n_gen": 12,
        "prefill_tps": 45.0,
        "decode_tps": 40.0,
        "prefill_s": 0.1,
        "decode_s": 0.3,
        "total_s": 0.4,
        "sampling_mode": "greedy",
        "runtime_profile": None,
        "trace_summary": {},
        "text": "",
    }, 32)


def main():
    args = benchmark.parse_args(["--model", "gemma-1b.gguf"])
    assert args.model == "gemma-1b.gguf"
    assert args.preset == benchmark.DEFAULT_PRESET
    assert args.runs == benchmark.DEFAULT_RUNS
    assert args.warmup == benchmark.DEFAULT_WARMUP
    assert args.max_tokens == benchmark.DEFAULT_MAX_TOKENS
    assert benchmark._prompt_list(args) == list(benchmark.DEFAULT_PROMPTS)
    regression_args = benchmark.parse_args([
        "--model", "gemma-1b.gguf",
        "--preset", "exact_greedy_regression",
    ])
    assert regression_args.warmup == 0
    decode_args = benchmark.parse_args([
        "--model", "gemma-1b.gguf",
        "--preset", benchmark.DECODE_REGRESSION_PRESET,
    ])
    assert decode_args.runs == 1
    assert decode_args.warmup == 0
    assert decode_args.diagnose_trace is True
    assert decode_args.profile == "desktop_discrete"
    assert decode_args.stream_load == "off"
    assert decode_args.stream_chunk_mb == 64
    assert decode_args.kv_cap == 1024
    assert decode_args.gpu_fused_rows_per_group == 512
    assert decode_args.gpu_fused_topk == "on"
    chat_args = benchmark.parse_args([
        "--model", "gemma-1b.gguf",
        "--preset", benchmark.CHAT_TURN_REGRESSION_PRESET,
    ])
    assert chat_args.runs == 1
    assert chat_args.warmup == 0
    assert chat_args.diagnose_trace is True
    assert chat_args.profile == "desktop_discrete"
    assert chat_args.turn_prompt1 == benchmark.DEFAULT_TURN_PROMPT1
    assert chat_args.turn_prompt2 == benchmark.DEFAULT_TURN_PROMPT2

    args = benchmark.parse_args([
        "--model", "gemma-1b.gguf",
        "--prompt", "one",
        "--prompt", "two",
        "--profile", "desktop_discrete",
        "--adam-only",
        "--sweep",
        "--json-out", "bench.json",
    ])
    assert benchmark._prompt_list(args) == ["one", "two"]
    assert args.json_out == "bench.json"
    assert args.profile == "desktop_discrete"
    assert args.adam_only is True
    assert args.sweep is True

    timings = benchmark.parse_llama_timings(
        "llama_print_timings: prompt eval time = 32.00 ms / 8 tokens\n"
        "llama_print_timings: eval time = 160.00 ms / 16 runs\n"
    )
    assert timings["n_prompt"] == 8
    assert timings["n_gen"] == 16
    assert abs(timings["prefill_tps"] - 250.0) < 1e-6
    assert abs(timings["decode_tps"] - 100.0) < 1e-6

    ok = benchmark._finalize_run_record({
        "backend": "adam",
        "status": "ok",
        "n_prompt": 5,
        "n_gen": 12,
        "prefill_tps": 50.0,
        "decode_tps": 60.0,
        "prefill_s": 0.1,
        "decode_s": 0.2,
        "total_s": 0.3,
        "sampling_mode": "gpu_fused_topk",
        "runtime_profile": "default",
        "stream_load": False,
        "stream_chunk_mb": 64,
        "kv_cap": 1024,
        "pool_hot_mb": 512,
        "pool_cold_mb": 1024,
        "gpu_fused_rows_per_group": 256,
        "gpu_fused_topk": True,
        "gpu_approx_rerank": False,
        "gpu_approx_partial_k": 8,
        "trace_decode": False,
        "trace_summary": {},
        "prompt_render_ms": 5.5,
        "prompt_tokenize_ms": 1.5,
        "prompt_total_tokens": 55,
        "prompt_reused_tokens": 26,
        "prompt_prefilled_tokens": 29,
        "prefill_ms": 500.0,
        "decode_ms": 300.0,
        "total_turn_ms": 807.0,
        "reuse_hit": True,
        "reuse_prefix_tokens": 26,
        "reuse_miss_reason": None,
        "reuse_miss_index": -1,
        "chat_turn_metrics": {
            "turn1_prompt_render_ms": 20.0,
            "turn2_prompt_render_ms": 5.5,
            "turn2_prompt_total_tokens": 55,
            "reuse_hit": True,
            "reuse_prefix_tokens": 26,
        },
        "text": "ok",
    }, 32)
    short = benchmark._finalize_run_record({
        "backend": "adam",
        "status": "ok",
        "n_prompt": 5,
        "n_gen": 2,
        "prefill_tps": 80.0,
        "decode_tps": 120.0,
        "prefill_s": 0.1,
        "decode_s": 0.1,
        "total_s": 0.2,
        "sampling_mode": "gpu_fused_topk",
        "runtime_profile": "default",
        "stream_load": False,
        "stream_chunk_mb": 64,
        "kv_cap": 1024,
        "pool_hot_mb": 512,
        "pool_cold_mb": 1024,
        "gpu_fused_rows_per_group": 256,
        "gpu_fused_topk": True,
        "gpu_approx_rerank": False,
        "gpu_approx_partial_k": 8,
        "trace_decode": False,
        "trace_summary": {},
        "chat_turn_metrics": {},
        "text": "short",
    }, 32)
    assert ok["status"] == "ok"
    assert short["status"] == "short_run"

    summary = benchmark.summarize_records("adam", [ok, short])
    assert summary["status"] == "ok"
    assert summary["valid_runs"] == 1
    assert summary["short_runs"] == 1
    assert summary["decode_tps"] == 60.0
    assert summary["sampling_mode"] == "gpu_fused_topk"
    assert summary["stream_chunk_mb"] == 64
    assert summary["kv_cap"] == 1024
    assert summary["prompt_render_ms"] == 5.5
    assert summary["prompt_reused_tokens"] == 26.0
    assert summary["reuse_hit"] is True
    backend_summary = benchmark.summarize_backend_runs("adam", ok, [ok])
    assert backend_summary["first_run_decode_tps"] == 60.0
    assert backend_summary["steady_decode_tps"] == 60.0
    assert backend_summary["kpi_decode_tps"] == 60.0
    assert backend_summary["meets_kpi_target"] is False

    llama_ok = _llama_ok_record()
    prompt_results = [{
        "prompt": "one",
        "backends": {
            "adam": {
                "cold_first_run": ok,
                "runs": [ok],
                "warmup": [],
                "diagnostic": {
                    "trace_summary": {
                        "step_ms_avg": 11.0,
                        "sample_ms_avg": 1.0,
                        "forward_ms_avg": 10.0,
                        "dispatch_count_total": 96,
                        "dispatch_count_per_token": 12.0,
                    }
                },
                "summary": benchmark.summarize_backend_runs("adam", ok, [ok]),
            },
            "llama": {
                "cold_first_run": llama_ok,
                "runs": [llama_ok],
                "warmup": [],
                "diagnostic": None,
                "summary": benchmark.summarize_backend_runs("llama", llama_ok, [llama_ok]),
            },
            "ollama": {
                "cold_first_run": None,
                "runs": [benchmark._skip_record("ollama", "missing model")],
                "warmup": [],
                "diagnostic": None,
                "summary": benchmark.summarize_backend_runs(
                    "ollama", None, [benchmark._skip_record("ollama", "missing model")]
                ),
            },
        },
    }]
    prompt_results[0]["comparison"] = benchmark._build_prompt_comparison(
        "one", prompt_results[0]["backends"]
    )
    overall = benchmark.build_overall_summary(prompt_results)
    assert overall["adam"]["status"] == "ok"
    assert overall["adam"]["beats_llama"] is True
    assert overall["adam"]["beats_ollama"] is None
    assert overall["adam"]["decode_tps"] == 60.0
    assert overall["adam"]["first_run_decode_tps"] == 60.0
    assert overall["adam"]["kpi_decode_tps"] == 60.0

    sweep_results = [{
        "label": "profile=desktop_discrete stream=off chunk=64MB kv=1024 rows=256 pool=auto sampler=gpu_fused_topk",
        "startup": {"kv_cap": 1024},
        "prompt_results": [],
        "overall": overall["adam"],
    }]
    report = benchmark.build_json_report(args, prompt_results, overall, sweep_results=sweep_results)
    assert report["config"]["model"] == "gemma-1b.gguf"
    assert report["config"]["preset"] == benchmark.DEFAULT_PRESET
    assert report["config"]["profile"] == "desktop_discrete"
    assert report["config"]["turn_prompt1"] == benchmark.DEFAULT_TURN_PROMPT1
    assert report["config"]["turn_prompt2"] == benchmark.DEFAULT_TURN_PROMPT2
    assert report["config"]["sweep"] is True
    assert report["config"]["adam_only"] is True
    assert report["config"]["kpi_target_tps"] == benchmark.DEFAULT_KPI_TARGET_TPS
    assert report["prompt_results"][0]["backends"]["adam"]["runs"][0]["backend"] == "adam"
    assert report["prompt_results"][0]["backends"]["adam"]["diagnostic"]["trace_summary"]["step_ms_avg"] == 11.0
    assert report["prompt_results"][0]["backends"]["adam"]["diagnostic"]["trace_summary"]["core_batch_ms_avg"] == 0.0
    assert report["prompt_results"][0]["backends"]["adam"]["diagnostic"]["trace_summary"]["dispatch_count_total"] == 96
    assert report["prompt_results"][0]["backends"]["adam"]["diagnostic"]["trace_summary"]["descriptor_set_update_count_total"] == 0
    assert report["prompt_results"][0]["backends"]["adam"]["runs"][0]["chat_turn_metrics"]["turn2_prompt_total_tokens"] == 55
    assert report["overall"]["adam"]["beats_llama"] is True
    assert report["sweep"][0]["startup"]["kv_cap"] == 1024

    startup = benchmark._build_adam_startup(args, {"kv_cap": 512, "gpu_fused_topk": False})
    assert startup["runtime_profile"] == "desktop_discrete"
    assert startup["gpu_approx_rerank"] is False
    assert startup["gpu_fused_topk"] is False
    assert startup["kv_cap"] == 512

    sweep_cases = benchmark._build_sweep_cases(args)
    assert sweep_cases
    first_case = sweep_cases[0]
    assert "stream_load" in first_case["startup"]
    assert "gpu_fused_topk" in first_case["startup"]
    cmd = benchmark._apply_case_args(
        ["python", "benchmark.py"],
        {
            "stream_load": False,
            "stream_chunk_mb": 64,
            "kv_cap": 1024,
            "pool_hot_mb": 512,
            "pool_cold_mb": 1024,
            "gpu_fused_rows_per_group": 256,
            "gpu_fused_topk": True,
        },
    )
    assert "--stream-load" in cmd
    assert "--gpu-fused-topk" in cmd

    print("PASS benchmark helpers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
