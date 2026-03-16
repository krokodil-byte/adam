#!/usr/bin/env python3
"""Smoke tests for the comparative benchmark helpers."""
import json
import os

from adam.paths import ROOT, setup; setup()

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
    assert args.gpu_telemetry == benchmark.DEFAULT_GPU_TELEMETRY_MODE
    assert args.decode_ablation is False
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
    assert decode_args.direct_kv_cache_write is None
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
    ablation_args = benchmark.parse_args([
        "--model", "gemma-1b.gguf",
        "--decode-ablation",
    ])
    assert ablation_args.decode_ablation is True
    assert ablation_args.diagnose_trace is True

    args = benchmark.parse_args([
        "--model", "gemma-1b.gguf",
        "--prompt", "one",
        "--prompt", "two",
        "--profile", "desktop_discrete",
        "--adam-only",
        "--sweep",
        "--decode-ablation",
        "--json-out", "bench.json",
        "--experiment-name", "attention-ablation",
    ])
    assert benchmark._prompt_list(args) == ["one", "two"]
    assert args.json_out == "bench.json"
    assert args.profile == "desktop_discrete"
    assert args.adam_only is True
    assert args.sweep is True
    assert args.decode_ablation is True
    assert args.experiment_name == "attention-ablation"
    assert args.experiment_log == benchmark.DEFAULT_EXPERIMENT_LOG

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
        "direct_kv_cache_write": True,
        "gpu_approx_rerank": False,
        "gpu_approx_partial_k": 8,
        "trace_decode": False,
        "trace_summary": {
            "descriptor_cache_breakdown": {
                "matvec_topk_t_xq8": {"hit_count": 0, "miss_count": 4},
            },
        },
        "gpu_telemetry": {
            "available": True,
            "sample_count": 4,
            "gpu_name": "RTX Test",
            "gpu_index": 0,
            "gpu_util_avg": 82.0,
            "gpu_util_max": 97.0,
            "mem_util_avg": 34.0,
            "mem_util_max": 40.0,
            "mem_used_mb_avg": 1536.0,
            "mem_used_mb_max": 1600.0,
            "mem_total_mb": 8192.0,
            "sm_clock_mhz_avg": 1725.0,
            "sm_clock_mhz_max": 1800.0,
            "mem_clock_mhz_avg": 7001.0,
            "mem_clock_mhz_max": 7001.0,
            "power_w_avg": 146.0,
            "power_w_max": 170.0,
        },
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
        "direct_kv_cache_write": False,
        "gpu_approx_rerank": False,
        "gpu_approx_partial_k": 8,
        "trace_decode": False,
        "trace_summary": {},
        "gpu_telemetry": {
            "available": True,
            "sample_count": 3,
            "gpu_name": "RTX Test",
            "gpu_index": 0,
            "gpu_util_avg": 75.0,
            "gpu_util_max": 88.0,
        },
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
    assert summary["direct_kv_cache_write"] is True
    assert summary["gpu_telemetry"]["available"] is True
    assert summary["gpu_telemetry"]["gpu_name"] == "RTX Test"
    assert summary["gpu_telemetry"]["gpu_util_avg"] == 82.0
    assert summary["prompt_render_ms"] == 5.5
    assert summary["prompt_reused_tokens"] == 26.0
    assert summary["reuse_hit"] is True
    assert summary["descriptor_cache_breakdown"]["matvec_topk_t_xq8"]["miss_count"] == 4
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
    decode_ablations = [{
        "id": "baseline",
        "label": "baseline current fast path",
        "feature": "current fast path",
        "mode": "baseline",
        "startup": {},
        "overall": overall["adam"],
        "trace_summary": {"dispatch_count_per_token": 12.0},
        "delta_decode_tps": 0.0,
        "takeaway": "Reference current path.",
    }]
    report = benchmark.build_json_report(
        args,
        prompt_results,
        overall,
        sweep_results=sweep_results,
        decode_ablations=decode_ablations,
    )
    assert report["config"]["model"] == "gemma-1b.gguf"
    assert report["config"]["preset"] == benchmark.DEFAULT_PRESET
    assert report["config"]["profile"] == "desktop_discrete"
    assert report["config"]["decode_ablation"] is True
    assert report["config"]["direct_kv_cache_write"] is None
    assert report["config"]["experimental_qk_norm_rope"] is None
    assert report["config"]["experimental_fused_qkv_qk_norm_rope"] is None
    assert report["config"]["gpu_telemetry"] == benchmark.DEFAULT_GPU_TELEMETRY_MODE
    assert report["config"]["turn_prompt1"] == benchmark.DEFAULT_TURN_PROMPT1
    assert report["config"]["turn_prompt2"] == benchmark.DEFAULT_TURN_PROMPT2
    assert report["config"]["sweep"] is True
    assert report["config"]["adam_only"] is True
    assert report["config"]["kpi_target_tps"] == benchmark.DEFAULT_KPI_TARGET_TPS
    assert report["config"]["experiment_name"] == "attention-ablation"
    assert report["config"]["experiment_log"] == benchmark.DEFAULT_EXPERIMENT_LOG
    assert report["prompt_results"][0]["backends"]["adam"]["runs"][0]["backend"] == "adam"
    assert report["prompt_results"][0]["backends"]["adam"]["diagnostic"]["trace_summary"]["step_ms_avg"] == 11.0
    assert report["prompt_results"][0]["backends"]["adam"]["diagnostic"]["trace_summary"]["core_batch_ms_avg"] == 0.0
    assert report["prompt_results"][0]["backends"]["adam"]["diagnostic"]["trace_summary"]["dispatch_count_total"] == 96
    assert report["prompt_results"][0]["backends"]["adam"]["diagnostic"]["trace_summary"]["descriptor_set_update_count_total"] == 0
    assert report["prompt_results"][0]["backends"]["adam"]["runs"][0]["chat_turn_metrics"]["turn2_prompt_total_tokens"] == 55
    assert report["overall"]["adam"]["beats_llama"] is True
    assert report["sweep"][0]["startup"]["kv_cap"] == 1024
    assert report["decode_ablations"][0]["label"] == "baseline current fast path"

    experiment_entry = benchmark.build_experiment_entry(args, prompt_results, overall)
    assert experiment_entry["experiment_name"] == "attention-ablation"
    assert experiment_entry["model"] == "gemma-1b.gguf"
    assert experiment_entry["adam"]["steady_decode_tps"] == 60.0
    assert experiment_entry["adam"]["trace_summary"]["step_ms_avg"] == 11.0
    assert experiment_entry["adam"]["trace_summary"]["dispatch_count_total"] == 96

    log_dir = os.path.join(ROOT, "reports")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "test_experiments.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)
    try:
        benchmark.append_experiment_log(log_path, experiment_entry)
        with open(log_path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
        assert len(lines) == 1
        saved = json.loads(lines[0])
        assert saved["experiment_name"] == "attention-ablation"
        assert saved["adam"]["steady_decode_tps"] == 60.0
    finally:
        if os.path.exists(log_path):
            os.remove(log_path)

    startup = benchmark._build_adam_startup(args, {"kv_cap": 512, "gpu_fused_topk": False})
    assert startup["runtime_profile"] == "desktop_discrete"
    assert startup["gpu_approx_rerank"] is False
    assert startup["gpu_fused_topk"] is False
    assert startup["kv_cap"] == 512

    feature_args = benchmark.parse_args([
        "--model", "gemma-1b.gguf",
        "--experimental-qk-norm-rope", "off",
        "--experimental-merged-qkv", "off",
        "--experimental-fused-qkv-qk-norm-rope", "off",
        "--experimental-merged-gateup", "off",
        "--experimental-attn-softmax-value", "on",
        "--experimental-rmsnorm-add", "on",
    ])
    feature_startup = benchmark._build_adam_startup(feature_args)
    assert feature_startup["experimental_qk_norm_rope"] is False
    assert feature_startup["experimental_merged_qkv"] is False
    assert feature_startup["experimental_fused_qkv_qk_norm_rope"] is False
    assert feature_startup["experimental_merged_gateup"] is False
    assert feature_startup["experimental_attn_softmax_value"] is True
    assert feature_startup["experimental_rmsnorm_add"] is True

    direct_kv_args = benchmark.parse_args([
        "--model", "gemma-1b.gguf",
        "--direct-kv-cache-write", "on",
    ])
    direct_kv_startup = benchmark._build_adam_startup(direct_kv_args)
    assert direct_kv_startup["direct_kv_cache_write"] is True

    sweep_cases = benchmark._build_sweep_cases(args)
    assert sweep_cases
    ablation_cases = benchmark._build_decode_ablation_cases(args)
    assert ablation_cases[0]["id"] == "baseline"
    assert any(case["id"] == "merged_qkv_off" for case in ablation_cases)
    assert not any(case["id"] == "fused_qkv_qk_norm_rope_off" for case in ablation_cases)
    broadcom_args = benchmark.parse_args([
        "--model", "gemma-1b.gguf",
        "--profile", "broadcom_v3dv",
    ])
    broadcom_ablations = benchmark._build_decode_ablation_cases(broadcom_args)
    assert any(case["id"] == "fused_qkv_qk_norm_rope_off" for case in broadcom_ablations)
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
            "direct_kv_cache_write": True,
            "experimental_qk_norm_rope": False,
            "experimental_merged_qkv": False,
            "experimental_fused_qkv_qk_norm_rope": False,
            "experimental_merged_gateup": False,
            "experimental_attn_softmax_value": True,
            "experimental_rmsnorm_add": True,
        },
    )
    assert "--stream-load" in cmd
    assert "--gpu-fused-topk" in cmd
    assert "--direct-kv-cache-write" in cmd
    assert "--experimental-qk-norm-rope" in cmd
    assert "--experimental-fused-qkv-qk-norm-rope" in cmd
    assert "--experimental-attn-softmax-value" in cmd

    line = benchmark._gpu_telemetry_line(ok["gpu_telemetry"])
    assert "gpu telemetry:" in line
    assert "82.0% avg" in line

    parsed = benchmark._parse_nvidia_smi_csv_numbers("14, 8, 1015, 210, 405, 15.56", 6)
    assert parsed == [14.0, 8.0, 1015.0, 210.0, 405.0, 15.56]

    print("PASS benchmark helpers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
