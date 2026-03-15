#!/usr/bin/env python3
"""Comparative benchmark for ADAM runtime path vs llama.cpp and Ollama."""
import argparse
import itertools
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional

# Ensure the project root is in sys.path so 'adam' and 'adamah' are importable
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ADAMAH_DIR = os.path.join(_ROOT, "adamah-MAIN")
for _p in (_ROOT, _ADAMAH_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BACKENDS = ("adam", "llama", "ollama")
DEFAULT_PRESET = "exact_greedy_32_prewarmed"
DECODE_REGRESSION_PRESET = "decode_regression"
CHAT_TURN_REGRESSION_PRESET = "chat_turn_regression"
DEFAULT_OLLAMA_MODEL = "gemma3:1b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EXPERIMENT_LOG = os.path.join(_ROOT, "reports", "throughput_experiments.jsonl")
DEFAULT_WARMUP = 1
DEFAULT_RUNS = 3
DEFAULT_MAX_TOKENS = 32
DEFAULT_TRACE_TOKENS = 8
DEFAULT_SEED = 123
DEFAULT_LLAMA_NGL = 999
DEFAULT_KPI_TARGET_TPS = 100.0
DEFAULT_GPU_TELEMETRY_MODE = "auto"
MIN_VALID_GENERATED_TOKENS = 8
SWEEP_STREAM_LOAD = (False, True)
SWEEP_STREAM_CHUNK_MB = (8, 16, 32, 64)
SWEEP_KV_CAP = (256, 512, 1024, 2048)
SWEEP_ROWS_PER_GROUP = (128, 256, 512)
SWEEP_POOL_MODES = ("auto", "fixed")
SWEEP_SAMPLERS = ("gpu_fused_topk", "gpu_argmax")
DEFAULT_PROMPTS = (
    "Continue in plain text only: The capital of France is",
    "Continue in plain text only: The Fibonacci sequence begins with 1, 1, 2, 3, 5,",
    "Continue in plain text only: In Python, a function is defined with",
)
DEFAULT_TURN_PROMPT1 = "hi"
DEFAULT_TURN_PROMPT2 = "what can you tell me about bees"
LLAMA_PROMPT_RE = re.compile(
    r"^.*prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*(?:tokens|runs).*$",
    re.IGNORECASE | re.MULTILINE,
)
LLAMA_EVAL_RE = re.compile(
    r"^.*eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*(?:tokens|runs).*$",
    re.IGNORECASE | re.MULTILINE,
)
PRESET_DEFAULTS = {
    "exact_greedy_32_prewarmed": {
        "runs": DEFAULT_RUNS,
        "warmup": DEFAULT_WARMUP,
        "max_tokens": DEFAULT_MAX_TOKENS,
    },
    "exact_greedy_regression": {
        "runs": DEFAULT_RUNS,
        "warmup": 0,
        "max_tokens": DEFAULT_MAX_TOKENS,
    },
    DECODE_REGRESSION_PRESET: {
        "runs": 1,
        "warmup": 0,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "diagnose_trace": True,
        "profile": "desktop_discrete",
        "stream_load": "off",
        "stream_chunk_mb": 64,
        "kv_cap": 1024,
        "gpu_fused_rows_per_group": 512,
        "gpu_fused_topk": "on",
        "fusion_scheduler_mode": None,
    },
    CHAT_TURN_REGRESSION_PRESET: {
        "runs": 1,
        "warmup": 0,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "diagnose_trace": True,
        "profile": "desktop_discrete",
        "stream_load": "off",
        "stream_chunk_mb": 64,
        "kv_cap": 1024,
        "gpu_fused_rows_per_group": 512,
        "gpu_fused_topk": "on",
        "fusion_scheduler_mode": None,
        "turn_prompt1": DEFAULT_TURN_PROMPT1,
        "turn_prompt2": DEFAULT_TURN_PROMPT2,
    },
}

DECODE_ABLATION_CASES: List[Dict[str, Any]] = [
    {
        "id": "qk_norm_rope_off",
        "label": "disable fused qk_norm_rope",
        "feature": "fused Q/K norm+RoPE prep",
        "mode": "disable",
        "startup": {"experimental_qk_norm_rope": False},
    },
    {
        "id": "merged_qkv_off",
        "label": "disable merged qkv",
        "feature": "merged QKV projection",
        "mode": "disable",
        "startup": {"experimental_merged_qkv": False},
    },
    {
        "id": "fused_qkv_qk_norm_rope_off",
        "label": "disable fused merged qkv -> qk norm+rope",
        "feature": "fused merged QKV -> Q/K norm+RoPE prep",
        "mode": "disable",
        "startup": {"experimental_fused_qkv_qk_norm_rope": False},
    },
    {
        "id": "merged_gateup_off",
        "label": "disable merged gate+up",
        "feature": "merged gate+up FFN projection",
        "mode": "disable",
        "startup": {"experimental_merged_gateup": False},
    },
    {
        "id": "attn_softmax_value_on",
        "label": "enable fused attn softmax+value",
        "feature": "fused attention softmax+value",
        "mode": "enable",
        "startup": {"experimental_attn_softmax_value": True},
    },
    {
        "id": "rmsnorm_add_on",
        "label": "enable fused rmsnorm+add",
        "feature": "fused rmsnorm+add residual path",
        "mode": "enable",
        "startup": {"experimental_rmsnorm_add": True},
    },
]

TRACE_SUMMARY_DEFAULTS: Dict[str, Any] = {
    "step_ms_avg": 0.0,
    "sample_ms_avg": 0.0,
    "forward_ms_avg": 0.0,
    "embed_ms_avg": 0.0,
    "norm_ms_avg": 0.0,
    "qkv_ms_avg": 0.0,
    "qk_norm_ms_avg": 0.0,
    "rope_ms_avg": 0.0,
    "attn_ms_avg": 0.0,
    "attn_out_ms_avg": 0.0,
    "ffn_ms_avg": 0.0,
    "lm_head_ms_avg": 0.0,
    "lm_head_shortlist_ms_avg": 0.0,
    "sample_resolve_ms_avg": 0.0,
    "core_batch_ms_avg": 0.0,
    "lm_head_batch_ms_avg": 0.0,
    "rerank_batch_ms_avg": 0.0,
    "dispatch_count_total": 0,
    "dispatch_count_per_token": 0.0,
    "submit_count_total": 0,
    "submit_count_per_token": 0.0,
    "barrier_count_total": 0,
    "barrier_count_per_token": 0.0,
    "fusion_flush_count_total": 0,
    "fusion_flush_count_per_token": 0.0,
    "descriptor_set_update_count_total": 0,
    "descriptor_set_update_count_per_token": 0.0,
    "descriptor_cache_hit_count_total": 0,
    "descriptor_cache_hit_count_per_token": 0.0,
    "descriptor_cache_miss_count_total": 0,
    "descriptor_cache_miss_count_per_token": 0.0,
    "alias_conflict_count_total": 0,
    "alias_conflict_count_per_token": 0.0,
    "scheduler_mode": 0,
    "scheduler_mode_name": "legacy",
    "descriptor_cache_breakdown": {},
}

CHAT_TURN_METRICS_DEFAULTS: Dict[str, Any] = {
    "turn1_prompt_render_ms": 0.0,
    "turn1_prompt_tokenize_ms": 0.0,
    "turn1_prompt_total_tokens": 0,
    "turn1_prompt_reused_tokens": 0,
    "turn1_prompt_prefilled_tokens": 0,
    "turn1_prefill_ms": 0.0,
    "turn1_decode_ms": 0.0,
    "turn1_decode_tps": 0.0,
    "turn1_total_turn_ms": 0.0,
    "turn2_prompt_render_ms": 0.0,
    "turn2_prompt_tokenize_ms": 0.0,
    "turn2_prompt_total_tokens": 0,
    "turn2_prompt_reused_tokens": 0,
    "turn2_prompt_prefilled_tokens": 0,
    "turn2_prefill_ms": 0.0,
    "turn2_decode_ms": 0.0,
    "turn2_decode_tps": 0.0,
    "turn2_total_turn_ms": 0.0,
    "reuse_hit": False,
    "reuse_prefix_tokens": 0,
    "reuse_miss_reason": None,
    "reuse_miss_index": -1,
}

GPU_TELEMETRY_DEFAULTS: Dict[str, Any] = {
    "available": False,
    "sample_count": 0,
    "gpu_name": None,
    "gpu_index": None,
    "gpu_util_avg": 0.0,
    "gpu_util_max": 0.0,
    "mem_util_avg": 0.0,
    "mem_util_max": 0.0,
    "mem_used_mb_avg": 0.0,
    "mem_used_mb_max": 0.0,
    "mem_total_mb": 0.0,
    "sm_clock_mhz_avg": 0.0,
    "sm_clock_mhz_max": 0.0,
    "mem_clock_mhz_avg": 0.0,
    "mem_clock_mhz_max": 0.0,
    "power_w_avg": 0.0,
    "power_w_max": 0.0,
    "error": None,
}


def _empty_trace_summary() -> Dict[str, Any]:
    return dict(TRACE_SUMMARY_DEFAULTS)


def _empty_chat_turn_metrics() -> Dict[str, Any]:
    return dict(CHAT_TURN_METRICS_DEFAULTS)


def _empty_gpu_telemetry() -> Dict[str, Any]:
    return dict(GPU_TELEMETRY_DEFAULTS)


def _default_experiment_name(args: argparse.Namespace) -> str:
    parts = [
        str(getattr(args, "preset", DEFAULT_PRESET) or DEFAULT_PRESET),
        str(getattr(args, "profile", "auto") or "auto"),
    ]
    scheduler = getattr(args, "fusion_scheduler_mode", None)
    if scheduler:
        parts.append(f"sched={scheduler}")
    direct_kv = getattr(args, "direct_kv_cache_write", None)
    if direct_kv:
        parts.append(f"direct_kv={direct_kv}")
    rows = getattr(args, "gpu_fused_rows_per_group", None)
    if rows is not None:
        parts.append(f"rows={rows}")
    return " | ".join(parts)


def _profile_prefers_dispatch_cut(args: Optional[argparse.Namespace] = None) -> bool:
    profile = str(getattr(args, "profile", "") or "").strip().lower()
    explicit = getattr(args, "experimental_fused_qkv_qk_norm_rope", None) if args is not None else None
    return explicit == "on" or profile.startswith("broadcom_v3dv")


def _normalize_trace_summary(summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = _empty_trace_summary()
    if summary:
        for key, value in summary.items():
            normalized[key] = value
    return normalized


def _normalize_chat_turn_metrics(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = _empty_chat_turn_metrics()
    if metrics:
        for key, value in metrics.items():
            normalized[key] = value
    return normalized


def _normalize_gpu_telemetry(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = _empty_gpu_telemetry()
    if metrics:
        for key, value in metrics.items():
            normalized[key] = value
    return normalized


def _parse_nvidia_smi_csv_numbers(line: str, expected_fields: int) -> Optional[List[float]]:
    parts = [part.strip() for part in (line or "").split(",")]
    if len(parts) < expected_fields:
        return None
    values: List[float] = []
    for part in parts[:expected_fields]:
        lower = part.lower()
        if lower in {"n/a", "[not supported]", "not supported"}:
            values.append(0.0)
            continue
        try:
            values.append(float(part))
        except ValueError:
            return None
    return values


def _summarize_gpu_telemetry_samples(samples: List[Dict[str, float]],
                                     gpu_name: Optional[str] = None,
                                     gpu_index: Optional[int] = None,
                                     mem_total_mb: Optional[float] = None,
                                     error: Optional[str] = None) -> Dict[str, Any]:
    if not samples:
        telemetry = _empty_gpu_telemetry()
        telemetry["gpu_name"] = gpu_name
        telemetry["gpu_index"] = gpu_index
        telemetry["mem_total_mb"] = _safe_float(mem_total_mb)
        telemetry["error"] = error
        return telemetry

    def avg(key: str) -> float:
        return sum(_safe_float(sample.get(key)) for sample in samples) / max(1, len(samples))

    def maxv(key: str) -> float:
        return max((_safe_float(sample.get(key)) for sample in samples), default=0.0)

    telemetry = {
        "available": True,
        "sample_count": len(samples),
        "gpu_name": gpu_name,
        "gpu_index": gpu_index,
        "gpu_util_avg": avg("gpu_util"),
        "gpu_util_max": maxv("gpu_util"),
        "mem_util_avg": avg("mem_util"),
        "mem_util_max": maxv("mem_util"),
        "mem_used_mb_avg": avg("mem_used_mb"),
        "mem_used_mb_max": maxv("mem_used_mb"),
        "mem_total_mb": _safe_float(mem_total_mb),
        "sm_clock_mhz_avg": avg("sm_clock_mhz"),
        "sm_clock_mhz_max": maxv("sm_clock_mhz"),
        "mem_clock_mhz_avg": avg("mem_clock_mhz"),
        "mem_clock_mhz_max": maxv("mem_clock_mhz"),
        "power_w_avg": avg("power_w"),
        "power_w_max": maxv("power_w"),
        "error": error,
    }
    return telemetry


def _summarize_gpu_telemetry(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    source = []
    for record in records:
        telemetry = _normalize_gpu_telemetry(record.get("gpu_telemetry"))
        if telemetry.get("available") and _safe_int(telemetry.get("sample_count")) > 0:
            source.append(telemetry)
    if not source:
        return _empty_gpu_telemetry()

    def med(key: str) -> float:
        return _median([_safe_float(item.get(key)) for item in source])

    summary = {
        "available": True,
        "sample_count": _median([float(_safe_int(item.get("sample_count"))) for item in source]),
        "gpu_name": next((item.get("gpu_name") for item in source if item.get("gpu_name")), None),
        "gpu_index": next((item.get("gpu_index") for item in source if item.get("gpu_index") is not None), None),
        "gpu_util_avg": med("gpu_util_avg"),
        "gpu_util_max": med("gpu_util_max"),
        "mem_util_avg": med("mem_util_avg"),
        "mem_util_max": med("mem_util_max"),
        "mem_used_mb_avg": med("mem_used_mb_avg"),
        "mem_used_mb_max": med("mem_used_mb_max"),
        "mem_total_mb": med("mem_total_mb"),
        "sm_clock_mhz_avg": med("sm_clock_mhz_avg"),
        "sm_clock_mhz_max": med("sm_clock_mhz_max"),
        "mem_clock_mhz_avg": med("mem_clock_mhz_avg"),
        "mem_clock_mhz_max": med("mem_clock_mhz_max"),
        "power_w_avg": med("power_w_avg"),
        "power_w_max": med("power_w_max"),
        "error": next((item.get("error") for item in source if item.get("error")), None),
    }
    return summary


class _NvidiaSmiMonitor:
    _STATIC_QUERY = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    _LOOP_QUERY_FIELDS = (
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "clocks.sm",
        "clocks.mem",
        "power.draw",
    )

    def __init__(self, gpu_index: int = 0, interval_ms: int = 100):
        self.gpu_index = int(gpu_index)
        self.interval_ms = max(20, int(interval_ms))
        self.proc: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.samples: List[Dict[str, float]] = []
        self.gpu_name: Optional[str] = None
        self.mem_total_mb: float = 0.0
        self.error: Optional[str] = None
        self._stop = threading.Event()

    @classmethod
    def probe_static(cls, gpu_index: int = 0) -> Dict[str, Any]:
        try:
            proc = subprocess.run(
                cls._STATIC_QUERY,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
        except Exception as exc:
            telemetry = _empty_gpu_telemetry()
            telemetry["error"] = str(exc)
            return telemetry
        if proc.returncode != 0:
            telemetry = _empty_gpu_telemetry()
            telemetry["error"] = (proc.stderr or proc.stdout or "nvidia-smi failed").strip()
            return telemetry
        lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
        if not lines:
            telemetry = _empty_gpu_telemetry()
            telemetry["error"] = "nvidia-smi returned no GPU rows"
            return telemetry
        row = None
        for line in lines:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            if _safe_int(parts[0], -1) == int(gpu_index):
                row = parts
                break
        if row is None:
            row = [part.strip() for part in lines[0].split(",")]
        telemetry = _empty_gpu_telemetry()
        telemetry["available"] = True
        telemetry["gpu_index"] = _safe_int(row[0], gpu_index)
        telemetry["gpu_name"] = row[1] if len(row) > 1 else None
        telemetry["mem_total_mb"] = _safe_float(row[2] if len(row) > 2 else 0.0)
        return telemetry

    def start(self) -> bool:
        static = self.probe_static(self.gpu_index)
        self.gpu_name = static.get("gpu_name")
        self.mem_total_mb = _safe_float(static.get("mem_total_mb"))
        if not static.get("available"):
            self.error = static.get("error")
            return False
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            f"--query-gpu={','.join(self._LOOP_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self.interval_ms}",
        ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception as exc:
            self.error = str(exc)
            self.proc = None
            return False

        def _reader() -> None:
            assert self.proc is not None
            try:
                while not self._stop.is_set():
                    line = self.proc.stdout.readline() if self.proc.stdout else ""
                    if not line:
                        if self.proc.poll() is not None:
                            break
                        continue
                    values = _parse_nvidia_smi_csv_numbers(line.strip(), expected_fields=6)
                    if values is None:
                        continue
                    self.samples.append({
                        "gpu_util": values[0],
                        "mem_util": values[1],
                        "mem_used_mb": values[2],
                        "sm_clock_mhz": values[3],
                        "mem_clock_mhz": values[4],
                        "power_w": values[5],
                    })
            except Exception as exc:
                self.error = self.error or str(exc)

        self.thread = threading.Thread(target=_reader, name="nvidia-smi-monitor", daemon=True)
        self.thread.start()
        return True

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.proc is not None and self.proc.stderr:
            try:
                stderr = self.proc.stderr.read().strip()
            except Exception:
                stderr = ""
            if stderr and not self.error and "interrupted" not in stderr.lower():
                self.error = stderr
        return _summarize_gpu_telemetry_samples(
            self.samples,
            gpu_name=self.gpu_name,
            gpu_index=self.gpu_index,
            mem_total_mb=self.mem_total_mb,
            error=self.error,
        )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ADAM runtime path against llama.cpp and Ollama."
    )
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    parser.add_argument("--llama-bin", default=os.environ.get("LLAMA_BIN"))
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--json-out", dest="json_out")
    parser.add_argument("--output", dest="json_out")
    parser.add_argument("--experiment-name")
    parser.add_argument("--experiment-log", dest="experiment_log")
    parser.add_argument("--diagnose-trace", action="store_true")
    parser.add_argument(
        "--decode-ablation",
        action="store_true",
        help="Run targeted ADAM decode ablations to show which subgraphs are worth deeper optimization.",
    )
    parser.add_argument("--profile", help="Runtime profile override for ADAM")
    parser.add_argument("--sweep", action="store_true", help="Run ADAM-only regression sweep")
    parser.add_argument("--adam-only", action="store_true", help="Skip llama.cpp and Ollama")
    parser.add_argument("--skip-adam", action="store_true")
    parser.add_argument("--skip-llama", action="store_true")
    parser.add_argument("--skip-ollama", action="store_true")
    parser.add_argument("--llama-ngl", type=int, default=DEFAULT_LLAMA_NGL)
    parser.add_argument(
        "--gpu-telemetry",
        choices=("auto", "on", "off"),
        default=DEFAULT_GPU_TELEMETRY_MODE,
        help="Sample GPU utilization during benchmark runs when supported.",
    )
    parser.add_argument("--stream-load", choices=("on", "off"), help=argparse.SUPPRESS)
    parser.add_argument("--stream-chunk-mb", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--kv-cap", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--pool-hot-mb", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--pool-cold-mb", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--gpu-fused-rows-per-group", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--gpu-fused-topk", choices=("on", "off"), help=argparse.SUPPRESS)
    parser.add_argument("--fusion-scheduler-mode", help=argparse.SUPPRESS)
    parser.add_argument("--direct-kv-cache-write", choices=("on", "off"), help=argparse.SUPPRESS)
    parser.add_argument("--experimental-qk-norm-rope", choices=("on", "off"), help=argparse.SUPPRESS)
    parser.add_argument("--experimental-merged-qkv", choices=("on", "off"), help=argparse.SUPPRESS)
    parser.add_argument("--experimental-fused-qkv-qk-norm-rope", choices=("on", "off"), help=argparse.SUPPRESS)
    parser.add_argument("--experimental-merged-gateup", choices=("on", "off"), help=argparse.SUPPRESS)
    parser.add_argument("--experimental-attn-softmax-value", choices=("on", "off"), help=argparse.SUPPRESS)
    parser.add_argument("--experimental-rmsnorm-add", choices=("on", "off"), help=argparse.SUPPRESS)
    parser.add_argument("--turn-prompt1", help=argparse.SUPPRESS)
    parser.add_argument("--turn-prompt2", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    preset = PRESET_DEFAULTS.get(args.preset, PRESET_DEFAULTS[DEFAULT_PRESET])
    if args.runs is None:
        args.runs = int(preset["runs"])
    if args.warmup is None:
        args.warmup = int(preset["warmup"])
    if args.max_tokens is None:
        args.max_tokens = int(preset["max_tokens"])
    if not args.diagnose_trace and preset.get("diagnose_trace"):
        args.diagnose_trace = True
    if args.profile is None and preset.get("profile"):
        args.profile = str(preset["profile"])
    if args.stream_load is None and preset.get("stream_load") is not None:
        args.stream_load = str(preset["stream_load"])
    if args.stream_chunk_mb is None and preset.get("stream_chunk_mb") is not None:
        args.stream_chunk_mb = int(preset["stream_chunk_mb"])
    if args.kv_cap is None and preset.get("kv_cap") is not None:
        args.kv_cap = int(preset["kv_cap"])
    if args.gpu_fused_rows_per_group is None and preset.get("gpu_fused_rows_per_group") is not None:
        args.gpu_fused_rows_per_group = int(preset["gpu_fused_rows_per_group"])
    if args.gpu_fused_topk is None and preset.get("gpu_fused_topk") is not None:
        args.gpu_fused_topk = str(preset["gpu_fused_topk"])
    if args.fusion_scheduler_mode is None and preset.get("fusion_scheduler_mode") is not None:
        args.fusion_scheduler_mode = str(preset["fusion_scheduler_mode"])
    if args.direct_kv_cache_write is None and preset.get("direct_kv_cache_write") is not None:
        args.direct_kv_cache_write = str(preset["direct_kv_cache_write"])
    if args.turn_prompt1 is None and preset.get("turn_prompt1") is not None:
        args.turn_prompt1 = str(preset["turn_prompt1"])
    if args.turn_prompt2 is None and preset.get("turn_prompt2") is not None:
        args.turn_prompt2 = str(preset["turn_prompt2"])
    if args.experiment_log is None:
        env_log = os.environ.get("ADAM_BENCH_EXPERIMENT_LOG", "").strip()
        if env_log:
            args.experiment_log = env_log
    if args.experiment_name is None:
        env_name = os.environ.get("ADAM_BENCH_EXPERIMENT_NAME", "").strip()
        if env_name:
            args.experiment_name = env_name
    if args.experiment_name and not args.experiment_log:
        args.experiment_log = DEFAULT_EXPERIMENT_LOG
    if args.decode_ablation and not args.diagnose_trace:
        args.diagnose_trace = True
    if args.turn_prompt1 is None:
        args.turn_prompt1 = DEFAULT_TURN_PROMPT1
    if args.turn_prompt2 is None:
        args.turn_prompt2 = DEFAULT_TURN_PROMPT2
    return args


def _prompt_list(args: argparse.Namespace) -> List[str]:
    if args.preset == CHAT_TURN_REGRESSION_PRESET:
        return [f"turn1={_preview(args.turn_prompt1, 32)} | turn2={_preview(args.turn_prompt2, 48)}"]
    if args.prompt:
        return list(args.prompt)
    if args.suite or not args.prompt:
        return list(DEFAULT_PROMPTS)
    return list(DEFAULT_PROMPTS[:1])


def _preview(text: str, limit: int = 72) -> str:
    text = (text or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _median(values: List[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _valid_token_floor(max_tokens: int) -> int:
    return min(max(1, int(max_tokens)), MIN_VALID_GENERATED_TOKENS)


def _build_generation_kwargs(tokenizer_eos_id: Optional[int], max_tokens: int) -> Dict[str, Any]:
    kwargs = {
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "repeat_penalty": 1.0,
        "seed": DEFAULT_SEED,
    }
    if tokenizer_eos_id is not None:
        kwargs["eos_token_ids"] = (int(tokenizer_eos_id),)
    return kwargs


def _error_record(backend: str, error: str) -> Dict[str, Any]:
    return {
        "backend": backend,
        "status": "error",
        "error": str(error),
        "n_prompt": 0,
        "n_gen": 0,
        "prefill_tps": 0.0,
        "decode_tps": 0.0,
        "prefill_s": 0.0,
        "decode_s": 0.0,
        "total_s": 0.0,
        "sampling_mode": None,
        "runtime_profile": None,
        "decode_path": None,
        "stream_load": None,
        "stream_chunk_mb": None,
        "kv_cap": None,
        "pool_hot_mb": None,
        "pool_cold_mb": None,
        "gpu_fused_rows_per_group": None,
        "gpu_fused_topk": None,
        "fusion_scheduler_mode": None,
        "direct_kv_cache_write": None,
        "experimental_qk_norm_rope": None,
        "experimental_merged_qkv": None,
        "experimental_fused_qkv_qk_norm_rope": None,
        "experimental_merged_gateup": None,
        "experimental_attn_softmax_value": None,
        "experimental_rmsnorm_add": None,
        "gpu_approx_rerank": None,
        "gpu_approx_partial_k": None,
        "trace_decode": None,
        "trace_summary": _empty_trace_summary(),
        "gpu_telemetry": _empty_gpu_telemetry(),
        "prompt_render_ms": None,
        "prompt_tokenize_ms": None,
        "prompt_total_tokens": None,
        "prompt_reused_tokens": None,
        "prompt_prefilled_tokens": None,
        "prefill_ms": None,
        "decode_ms": None,
        "total_turn_ms": None,
        "reuse_hit": None,
        "reuse_prefix_tokens": None,
        "reuse_miss_reason": None,
        "reuse_miss_index": None,
        "chat_turn_metrics": _empty_chat_turn_metrics(),
        "text": "",
    }


def _skip_record(backend: str, reason: str) -> Dict[str, Any]:
    record = _error_record(backend, reason)
    record["status"] = "skipped"
    return record


def _finalize_run_record(record: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
    status = record.get("status")
    record["trace_summary"] = _normalize_trace_summary(record.get("trace_summary"))
    record["chat_turn_metrics"] = _normalize_chat_turn_metrics(record.get("chat_turn_metrics"))
    record["gpu_telemetry"] = _normalize_gpu_telemetry(record.get("gpu_telemetry"))
    if status in {"error", "skipped"}:
        return record
    if _safe_int(record.get("n_gen"), 0) < _valid_token_floor(max_tokens):
        record["status"] = "short_run"
    else:
        record["status"] = "ok"
    return record


def parse_llama_timings(output: str) -> Dict[str, Any]:
    output = output or ""
    prompt_match = LLAMA_PROMPT_RE.search(output)
    eval_match = next(
        (match for match in LLAMA_EVAL_RE.finditer(output)
         if "prompt eval time" not in match.group(0).lower()),
        None,
    )
    if not eval_match:
        return {}

    prompt_ms = _safe_float(prompt_match.group(1)) if prompt_match else 0.0
    prompt_tokens = _safe_int(prompt_match.group(2)) if prompt_match else 0
    eval_ms = _safe_float(eval_match.group(1))
    eval_tokens = _safe_int(eval_match.group(2))

    return {
        "n_prompt": prompt_tokens,
        "n_gen": eval_tokens,
        "prefill_s": prompt_ms / 1000.0,
        "decode_s": eval_ms / 1000.0,
        "prefill_tps": (prompt_tokens * 1000.0 / prompt_ms) if prompt_ms > 0 else 0.0,
        "decode_tps": (eval_tokens * 1000.0 / eval_ms) if eval_ms > 0 else 0.0,
    }


def summarize_records(backend: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in records if r.get("status") == "ok"]
    short = [r for r in records if r.get("status") == "short_run"]
    errors = [r for r in records if r.get("status") == "error"]
    skipped = [r for r in records if r.get("status") == "skipped"]

    source = valid or short
    descriptor_cache_breakdown = {}
    for r in source:
        breakdown = r.get("descriptor_cache_breakdown")
        if not breakdown:
            breakdown = (r.get("trace_summary") or {}).get("descriptor_cache_breakdown")
        if breakdown:
            descriptor_cache_breakdown = breakdown
            break
    return {
        "backend": backend,
        "status": "ok" if valid else ("short_run" if short else ("error" if errors else "skipped")),
        "runs_total": len(records),
        "valid_runs": len(valid),
        "short_runs": len(short),
        "error_runs": len(errors),
        "skipped_runs": len(skipped),
        "decode_tps": _median([_safe_float(r.get("decode_tps")) for r in source]),
        "prefill_tps": _median([_safe_float(r.get("prefill_tps")) for r in source]),
        "total_s": _median([_safe_float(r.get("total_s")) for r in source]),
        "prefill_s": _median([_safe_float(r.get("prefill_s")) for r in source]),
        "decode_s": _median([_safe_float(r.get("decode_s")) for r in source]),
        "n_prompt": _median([float(_safe_int(r.get("n_prompt"))) for r in source]),
        "n_gen": _median([float(_safe_int(r.get("n_gen"))) for r in source]),
        "sampling_mode": next((r.get("sampling_mode") for r in source if r.get("sampling_mode")), None),
        "runtime_profile": next((r.get("runtime_profile") for r in source if r.get("runtime_profile")), None),
        "decode_path": next((r.get("decode_path") for r in source if r.get("decode_path")), None),
        "stream_load": next((r.get("stream_load") for r in source if r.get("stream_load") is not None), None),
        "stream_chunk_mb": next((r.get("stream_chunk_mb") for r in source if r.get("stream_chunk_mb") is not None), None),
        "kv_cap": next((r.get("kv_cap") for r in source if r.get("kv_cap") is not None), None),
        "pool_hot_mb": next((r.get("pool_hot_mb") for r in source if r.get("pool_hot_mb") is not None), None),
        "pool_cold_mb": next((r.get("pool_cold_mb") for r in source if r.get("pool_cold_mb") is not None), None),
        "gpu_fused_rows_per_group": next(
            (r.get("gpu_fused_rows_per_group") for r in source if r.get("gpu_fused_rows_per_group") is not None),
            None,
        ),
        "gpu_fused_topk": next((r.get("gpu_fused_topk") for r in source if r.get("gpu_fused_topk") is not None), None),
        "fusion_scheduler_mode": next(
            (r.get("fusion_scheduler_mode") for r in source if r.get("fusion_scheduler_mode")),
            None,
        ),
        "direct_kv_cache_write": next(
            (r.get("direct_kv_cache_write") for r in source if r.get("direct_kv_cache_write") is not None),
            None,
        ),
        "experimental_qk_norm_rope": next(
            (r.get("experimental_qk_norm_rope") for r in source if r.get("experimental_qk_norm_rope") is not None),
            None,
        ),
        "experimental_merged_qkv": next(
            (r.get("experimental_merged_qkv") for r in source if r.get("experimental_merged_qkv") is not None),
            None,
        ),
        "experimental_fused_qkv_qk_norm_rope": next(
            (
                r.get("experimental_fused_qkv_qk_norm_rope")
                for r in source if r.get("experimental_fused_qkv_qk_norm_rope") is not None
            ),
            None,
        ),
        "experimental_merged_gateup": next(
            (r.get("experimental_merged_gateup") for r in source if r.get("experimental_merged_gateup") is not None),
            None,
        ),
        "experimental_attn_softmax_value": next(
            (
                r.get("experimental_attn_softmax_value")
                for r in source if r.get("experimental_attn_softmax_value") is not None
            ),
            None,
        ),
        "experimental_rmsnorm_add": next(
            (r.get("experimental_rmsnorm_add") for r in source if r.get("experimental_rmsnorm_add") is not None),
            None,
        ),
        "gpu_approx_rerank": next(
            (r.get("gpu_approx_rerank") for r in source if r.get("gpu_approx_rerank") is not None),
            None,
        ),
        "gpu_approx_partial_k": next(
            (r.get("gpu_approx_partial_k") for r in source if r.get("gpu_approx_partial_k") is not None),
            None,
        ),
        "trace_decode": next((r.get("trace_decode") for r in source if r.get("trace_decode") is not None), None),
        "prompt_render_ms": _median([
            _safe_float(r.get("prompt_render_ms")) for r in source if r.get("prompt_render_ms") is not None
        ]),
        "prompt_tokenize_ms": _median([
            _safe_float(r.get("prompt_tokenize_ms")) for r in source if r.get("prompt_tokenize_ms") is not None
        ]),
        "prompt_total_tokens": _median([
            float(_safe_int(r.get("prompt_total_tokens"))) for r in source if r.get("prompt_total_tokens") is not None
        ]),
        "prompt_reused_tokens": _median([
            float(_safe_int(r.get("prompt_reused_tokens"))) for r in source if r.get("prompt_reused_tokens") is not None
        ]),
        "prompt_prefilled_tokens": _median([
            float(_safe_int(r.get("prompt_prefilled_tokens"))) for r in source if r.get("prompt_prefilled_tokens") is not None
        ]),
        "prefill_ms": _median([
            _safe_float(r.get("prefill_ms")) for r in source if r.get("prefill_ms") is not None
        ]),
        "decode_ms": _median([
            _safe_float(r.get("decode_ms")) for r in source if r.get("decode_ms") is not None
        ]),
        "total_turn_ms": _median([
            _safe_float(r.get("total_turn_ms")) for r in source if r.get("total_turn_ms") is not None
        ]),
        "reuse_hit": next((r.get("reuse_hit") for r in source if r.get("reuse_hit") is not None), None),
        "reuse_prefix_tokens": _median([
            float(_safe_int(r.get("reuse_prefix_tokens"))) for r in source if r.get("reuse_prefix_tokens") is not None
        ]),
        "reuse_miss_reason": next(
            (r.get("reuse_miss_reason") for r in source if r.get("reuse_miss_reason")), None
        ),
        "reuse_miss_index": next(
            (r.get("reuse_miss_index") for r in source if r.get("reuse_miss_index") is not None), None
        ),
        "descriptor_cache_breakdown": descriptor_cache_breakdown,
        "gpu_telemetry": _summarize_gpu_telemetry(source),
        "error": errors[0].get("error") if errors else (skipped[0].get("error") if skipped else None),
    }


def summarize_backend_runs(backend: str,
                           cold_first_run: Optional[Dict[str, Any]],
                           steady_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = summarize_records(backend, steady_runs)
    cold = cold_first_run or {}
    summary["cold_status"] = cold.get("status")
    summary["first_run_decode_tps"] = _safe_float(cold.get("decode_tps")) if cold else 0.0
    summary["first_run_prefill_tps"] = _safe_float(cold.get("prefill_tps")) if cold else 0.0
    summary["first_run_total_s"] = _safe_float(cold.get("total_s")) if cold else 0.0
    summary["steady_decode_tps"] = _safe_float(summary.get("decode_tps"))
    summary["steady_prefill_tps"] = _safe_float(summary.get("prefill_tps"))
    summary["kpi_decode_tps"] = _safe_float(summary.get("steady_decode_tps"))
    summary["kpi_target_tps"] = DEFAULT_KPI_TARGET_TPS
    summary["meets_kpi_target"] = (
        summary.get("status") == "ok" and summary["kpi_decode_tps"] >= DEFAULT_KPI_TARGET_TPS
    )
    return summary


def _ratio_or_none(lhs: Optional[float], rhs: Optional[float]) -> Optional[float]:
    if lhs is None or rhs is None or rhs <= 0.0:
        return None
    return lhs / rhs


def _status_ok(summary: Dict[str, Any]) -> bool:
    return summary.get("status") == "ok" and _safe_float(summary.get("decode_tps")) > 0.0


def _build_prompt_comparison(prompt: str, backend_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    del prompt
    summaries = {name: result["summary"] for name, result in backend_results.items()}
    ok_pairs = [
        (name, _safe_float(summary.get("kpi_decode_tps", summary.get("decode_tps"))))
        for name, summary in summaries.items()
        if _status_ok(summary)
    ]
    winner = max(ok_pairs, key=lambda item: item[1])[0] if ok_pairs else None
    adam_tps = _safe_float(summaries["adam"].get("kpi_decode_tps", summaries["adam"].get("decode_tps"))) if _status_ok(summaries["adam"]) else None
    llama_tps = _safe_float(summaries["llama"].get("kpi_decode_tps", summaries["llama"].get("decode_tps"))) if _status_ok(summaries["llama"]) else None
    ollama_tps = _safe_float(summaries["ollama"].get("kpi_decode_tps", summaries["ollama"].get("decode_tps"))) if _status_ok(summaries["ollama"]) else None
    return {
        "winner": winner,
        "adam_vs_llama": _ratio_or_none(adam_tps, llama_tps),
        "adam_vs_ollama": _ratio_or_none(adam_tps, ollama_tps),
        "beats_llama": (adam_tps > llama_tps) if adam_tps is not None and llama_tps is not None else None,
        "beats_ollama": (adam_tps > ollama_tps) if adam_tps is not None and ollama_tps is not None else None,
    }


def build_overall_summary(prompt_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    combined: Dict[str, List[Dict[str, Any]]] = {name: [] for name in BACKENDS}
    cold_combined: Dict[str, List[Dict[str, Any]]] = {name: [] for name in BACKENDS}
    winners = []
    for prompt_result in prompt_results:
        winners.append(prompt_result.get("comparison", {}).get("winner"))
        for backend in BACKENDS:
            combined[backend].extend(prompt_result["backends"][backend]["runs"])
            cold = prompt_result["backends"][backend].get("cold_first_run")
            if cold:
                cold_combined[backend].append(cold)

    overall = {
        backend: summarize_backend_runs(
            backend,
            {"decode_tps": _median([_safe_float(r.get("decode_tps")) for r in cold_combined[backend]]),
             "prefill_tps": _median([_safe_float(r.get("prefill_tps")) for r in cold_combined[backend]]),
             "total_s": _median([_safe_float(r.get("total_s")) for r in cold_combined[backend]]),
             "status": "ok" if cold_combined[backend] else None},
            combined[backend],
        )
        for backend in BACKENDS
    }
    adam_ok = _status_ok(overall["adam"])
    llama_ok = _status_ok(overall["llama"])
    ollama_ok = _status_ok(overall["ollama"])
    adam_tps = _safe_float(overall["adam"].get("decode_tps")) if adam_ok else None
    llama_tps = _safe_float(overall["llama"].get("decode_tps")) if llama_ok else None
    ollama_tps = _safe_float(overall["ollama"].get("decode_tps")) if ollama_ok else None
    overall["adam"]["beats_llama"] = (adam_tps > llama_tps) if adam_tps is not None and llama_tps is not None else None
    overall["adam"]["beats_ollama"] = (adam_tps > ollama_tps) if adam_tps is not None and ollama_tps is not None else None
    overall["adam"]["ratio_vs_llama"] = _ratio_or_none(adam_tps, llama_tps)
    overall["adam"]["ratio_vs_ollama"] = _ratio_or_none(adam_tps, ollama_tps)
    overall["winners"] = winners
    return overall


def _normalize_run_record_trace(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(record)
    normalized["trace_summary"] = _normalize_trace_summary(normalized.get("trace_summary"))
    normalized["chat_turn_metrics"] = _normalize_chat_turn_metrics(normalized.get("chat_turn_metrics"))
    normalized["gpu_telemetry"] = _normalize_gpu_telemetry(normalized.get("gpu_telemetry"))
    return normalized


def _normalize_backend_trace_payload(backend_payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(backend_payload)
    payload["runs"] = [
        _normalize_run_record_trace(run) for run in payload.get("runs", [])
    ]
    payload["warmup"] = [
        _normalize_run_record_trace(run) for run in payload.get("warmup", [])
    ]
    if payload.get("cold_first_run") is not None:
        payload["cold_first_run"] = _normalize_run_record_trace(payload["cold_first_run"])
    diagnostic = payload.get("diagnostic")
    if diagnostic is not None:
        diagnostic = dict(diagnostic)
        diagnostic["trace_summary"] = _normalize_trace_summary(diagnostic.get("trace_summary"))
        diagnostic["gpu_telemetry"] = _normalize_gpu_telemetry(diagnostic.get("gpu_telemetry"))
        payload["diagnostic"] = diagnostic
    return payload


def _normalize_prompt_results(prompt_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for prompt_result in prompt_results:
        prompt_copy = dict(prompt_result)
        backends = {}
        for backend_name, backend_payload in prompt_copy.get("backends", {}).items():
            backends[backend_name] = _normalize_backend_trace_payload(backend_payload)
        prompt_copy["backends"] = backends
        normalized.append(prompt_copy)
    return normalized


def build_json_report(args: argparse.Namespace,
                      prompt_results: List[Dict[str, Any]],
                      overall: Dict[str, Dict[str, Any]],
                      sweep_results: Optional[List[Dict[str, Any]]] = None,
                      decode_ablations: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    prompt_results = _normalize_prompt_results(prompt_results)
    return {
        "config": {
            "model": args.model,
            "preset": args.preset,
            "llama_bin": args.llama_bin,
            "ollama_model": args.ollama_model,
            "ollama_url": args.ollama_url,
            "runs": args.runs,
            "warmup": args.warmup,
            "max_tokens": args.max_tokens,
            "seed": DEFAULT_SEED,
            "deterministic": True,
            "diagnose_trace": bool(args.diagnose_trace),
            "profile": args.profile,
            "decode_ablation": bool(args.decode_ablation),
            "fusion_scheduler_mode": args.fusion_scheduler_mode,
            "direct_kv_cache_write": args.direct_kv_cache_write,
            "experimental_qk_norm_rope": args.experimental_qk_norm_rope,
            "experimental_merged_qkv": args.experimental_merged_qkv,
            "experimental_fused_qkv_qk_norm_rope": args.experimental_fused_qkv_qk_norm_rope,
            "experimental_merged_gateup": args.experimental_merged_gateup,
            "experimental_attn_softmax_value": args.experimental_attn_softmax_value,
            "experimental_rmsnorm_add": args.experimental_rmsnorm_add,
            "gpu_telemetry": args.gpu_telemetry,
            "turn_prompt1": args.turn_prompt1,
            "turn_prompt2": args.turn_prompt2,
            "sweep": bool(args.sweep),
            "adam_only": bool(args.adam_only),
            "kpi_target_tps": DEFAULT_KPI_TARGET_TPS,
            "experiment_name": args.experiment_name,
            "experiment_log": args.experiment_log,
        },
        "prompt_results": prompt_results,
        "overall": overall,
        "sweep": sweep_results or [],
        "decode_ablations": decode_ablations or [],
    }


def _median_trace_summary_for_backend(prompt_results: List[Dict[str, Any]],
                                      backend: str = "adam") -> Dict[str, Any]:
    summaries: List[Dict[str, Any]] = []
    for prompt_result in prompt_results:
        backend_payload = (prompt_result.get("backends") or {}).get(backend) or {}
        diagnostic = backend_payload.get("diagnostic") or {}
        trace_summary = diagnostic.get("trace_summary") or {}
        if trace_summary:
            summaries.append(_normalize_trace_summary(trace_summary))

    if not summaries:
        return _empty_trace_summary()

    merged = _empty_trace_summary()
    for key, default in TRACE_SUMMARY_DEFAULTS.items():
        if key == "descriptor_cache_breakdown":
            merged[key] = next(
                (summary.get(key) for summary in summaries if summary.get(key)),
                {},
            )
            continue
        if isinstance(default, float):
            merged[key] = _median([_safe_float(summary.get(key)) for summary in summaries])
        elif isinstance(default, int):
            merged[key] = int(round(_median([float(_safe_int(summary.get(key))) for summary in summaries])))
        else:
            merged[key] = next(
                (summary.get(key) for summary in summaries if summary.get(key) not in (None, "")),
                default,
            )
    return merged


def build_experiment_entry(args: argparse.Namespace,
                           prompt_results: List[Dict[str, Any]],
                           overall: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    adam_overall = dict((overall or {}).get("adam") or {})
    adam_trace = _median_trace_summary_for_backend(prompt_results, backend="adam")
    experiment_name = str(
        getattr(args, "experiment_name", None) or _default_experiment_name(args)
    )
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment_name": experiment_name,
        "model": args.model,
        "preset": args.preset,
        "profile": args.profile,
        "config": {
            "runs": args.runs,
            "warmup": args.warmup,
            "max_tokens": args.max_tokens,
            "diagnose_trace": bool(args.diagnose_trace),
            "stream_load": args.stream_load,
            "stream_chunk_mb": args.stream_chunk_mb,
            "kv_cap": args.kv_cap,
            "gpu_fused_rows_per_group": args.gpu_fused_rows_per_group,
            "gpu_fused_topk": args.gpu_fused_topk,
            "fusion_scheduler_mode": args.fusion_scheduler_mode,
            "direct_kv_cache_write": args.direct_kv_cache_write,
            "experimental_qk_norm_rope": args.experimental_qk_norm_rope,
            "experimental_merged_qkv": args.experimental_merged_qkv,
            "experimental_fused_qkv_qk_norm_rope": args.experimental_fused_qkv_qk_norm_rope,
            "experimental_merged_gateup": args.experimental_merged_gateup,
            "experimental_attn_softmax_value": args.experimental_attn_softmax_value,
            "experimental_rmsnorm_add": args.experimental_rmsnorm_add,
            "gpu_telemetry": args.gpu_telemetry,
            "adam_only": bool(args.adam_only),
        },
        "adam": {
            "status": adam_overall.get("status"),
            "steady_decode_tps": _safe_float(adam_overall.get("steady_decode_tps", adam_overall.get("decode_tps"))),
            "first_run_decode_tps": _safe_float(adam_overall.get("first_run_decode_tps")),
            "prefill_tps": _safe_float(adam_overall.get("prefill_tps")),
            "total_s": _safe_float(adam_overall.get("total_s")),
            "sampling_mode": adam_overall.get("sampling_mode"),
            "runtime_profile": adam_overall.get("runtime_profile"),
            "gpu_telemetry": _normalize_gpu_telemetry(adam_overall.get("gpu_telemetry")),
            "trace_summary": adam_trace,
        },
    }


def append_experiment_log(path: str, entry: Dict[str, Any]) -> None:
    if not path:
        raise ValueError("experiment log path is required")
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        json.dump(entry, handle, ensure_ascii=True)
        handle.write("\n")


def _collect_runtime_knobs(engine: Any, trace_decode: bool) -> Dict[str, Any]:
    runtime_plan = dict(getattr(engine, "_runtime_plan", {}) or {})
    pool_plan = dict(runtime_plan.get("pool_plan", {}) or {})
    return {
        "decode_path": runtime_plan.get("decode_path", getattr(engine, "_decode_path", None)),
        "stream_load": runtime_plan.get("stream_load", getattr(engine, "_stream_load", None)),
        "stream_chunk_mb": runtime_plan.get("stream_chunk_mb", getattr(engine, "_stream_chunk_mb", None)),
        "kv_cap": runtime_plan.get("kv_cap", getattr(engine, "_kv_cap", None)),
        "pool_hot_mb": pool_plan.get("hot_mb", getattr(engine, "_pool_hot_mb", None)),
        "pool_cold_mb": pool_plan.get("cold_mb", getattr(engine, "_pool_cold_mb", None)),
        "gpu_fused_rows_per_group": runtime_plan.get(
            "gpu_fused_rows_per_group", getattr(engine, "_gpu_fused_rows_per_group", None)
        ),
        "gpu_fused_topk": runtime_plan.get("gpu_fused_topk", getattr(engine, "_gpu_fused_topk", None)),
        "fusion_scheduler_mode": runtime_plan.get(
            "fusion_scheduler_mode", getattr(engine, "_fusion_scheduler_mode", None)
        ),
        "direct_kv_cache_write": runtime_plan.get(
            "direct_kv_cache_write", getattr(engine, "_direct_kv_cache_write", None)
        ),
        "experimental_qk_norm_rope": runtime_plan.get(
            "experimental_qk_norm_rope", getattr(engine, "_experimental_qk_norm_rope", None)
        ),
        "experimental_merged_qkv": runtime_plan.get(
            "experimental_merged_qkv", getattr(engine, "_experimental_merged_qkv", None)
        ),
        "experimental_fused_qkv_qk_norm_rope": runtime_plan.get(
            "experimental_fused_qkv_qk_norm_rope",
            getattr(engine, "_experimental_fused_qkv_qk_norm_rope", None),
        ),
        "experimental_merged_gateup": runtime_plan.get(
            "experimental_merged_gateup", getattr(engine, "_experimental_merged_gateup", None)
        ),
        "experimental_attn_softmax_value": runtime_plan.get(
            "experimental_attn_softmax_value", getattr(engine, "_experimental_attn_softmax_value", None)
        ),
        "experimental_rmsnorm_add": runtime_plan.get(
            "experimental_rmsnorm_add", getattr(engine, "_experimental_rmsnorm_add", None)
        ),
        "gpu_approx_rerank": runtime_plan.get("gpu_approx_rerank", getattr(engine, "_gpu_approx_rerank", None)),
        "gpu_approx_partial_k": runtime_plan.get(
            "gpu_approx_partial_k", getattr(engine, "_gpu_approx_partial_k", None)
        ),
        "trace_decode": bool(trace_decode),
    }


def _build_adam_startup(args: argparse.Namespace, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    startup: Dict[str, Any] = {
        "runtime_mode": "fast",
        "trace_decode": False,
        "gpu_approx_rerank": False,
        "gpu_fused_topk": True,
    }
    if args.profile:
        startup["runtime_profile"] = args.profile
    if getattr(args, "stream_load", None) is not None:
        startup["stream_load"] = (args.stream_load == "on")
    if getattr(args, "stream_chunk_mb", None) is not None:
        startup["stream_chunk_mb"] = int(args.stream_chunk_mb)
    if getattr(args, "kv_cap", None) is not None:
        startup["kv_cap"] = int(args.kv_cap)
    if getattr(args, "pool_hot_mb", None) is not None:
        startup["pool_hot_mb"] = int(args.pool_hot_mb)
    if getattr(args, "pool_cold_mb", None) is not None:
        startup["pool_cold_mb"] = int(args.pool_cold_mb)
    if getattr(args, "gpu_fused_rows_per_group", None) is not None:
        startup["gpu_fused_rows_per_group"] = int(args.gpu_fused_rows_per_group)
    if getattr(args, "gpu_fused_topk", None) is not None:
        startup["gpu_fused_topk"] = (args.gpu_fused_topk == "on")
    if getattr(args, "fusion_scheduler_mode", None):
        startup["fusion_scheduler_mode"] = str(args.fusion_scheduler_mode)
    if getattr(args, "direct_kv_cache_write", None) is not None:
        startup["direct_kv_cache_write"] = (args.direct_kv_cache_write == "on")
    if getattr(args, "experimental_qk_norm_rope", None) is not None:
        startup["experimental_qk_norm_rope"] = (args.experimental_qk_norm_rope == "on")
    if getattr(args, "experimental_merged_qkv", None) is not None:
        startup["experimental_merged_qkv"] = (args.experimental_merged_qkv == "on")
    if getattr(args, "experimental_fused_qkv_qk_norm_rope", None) is not None:
        startup["experimental_fused_qkv_qk_norm_rope"] = (
            args.experimental_fused_qkv_qk_norm_rope == "on"
        )
    if getattr(args, "experimental_merged_gateup", None) is not None:
        startup["experimental_merged_gateup"] = (args.experimental_merged_gateup == "on")
    if getattr(args, "experimental_attn_softmax_value", None) is not None:
        startup["experimental_attn_softmax_value"] = (args.experimental_attn_softmax_value == "on")
    if getattr(args, "experimental_rmsnorm_add", None) is not None:
        startup["experimental_rmsnorm_add"] = (args.experimental_rmsnorm_add == "on")
    if extra:
        startup.update({key: value for key, value in extra.items() if value is not None})
    return startup


def _should_monitor_gpu(args: argparse.Namespace, backend: str) -> bool:
    mode = str(getattr(args, "gpu_telemetry", DEFAULT_GPU_TELEMETRY_MODE) or DEFAULT_GPU_TELEMETRY_MODE).strip().lower()
    if mode == "off":
        return False
    if mode == "on":
        return True
    return backend == "adam"


def _run_with_gpu_telemetry(backend: str, args: argparse.Namespace, fn, *run_args, **run_kwargs) -> Dict[str, Any]:
    monitor: Optional[_NvidiaSmiMonitor] = None
    monitor_enabled = _should_monitor_gpu(args, backend)
    if monitor_enabled:
        monitor = _NvidiaSmiMonitor(gpu_index=_safe_int(os.environ.get("ADAM_BENCH_GPU_INDEX"), 0))
        if not monitor.start():
            monitor = None
    result = fn(*run_args, **run_kwargs)
    telemetry = monitor.stop() if monitor is not None else _empty_gpu_telemetry()
    if telemetry.get("available") or telemetry.get("error"):
        result = dict(result)
        result["gpu_telemetry"] = telemetry
    return result


class ADAMRunner:
    def __init__(self, model_path: str, startup: Optional[Dict[str, Any]] = None):
        self.model_path = model_path
        self.startup = dict(startup or {})
        self.engine = None
        self.tokenizer = None
        self.cfg = None
        self._gen_config_cls = None

    def load(self, verbose: bool = True) -> "ADAMRunner":
        from adamah_chat import load_model

        self.engine, self.tokenizer, self.cfg, self._gen_config_cls = load_model(
            self.model_path, startup=self.startup
        )
        if verbose:
            print("[INIT] ADAM runtime path ready")
        return self

    def _chat_generation_config(self, max_tokens: int):
        eos_ids = {getattr(self.tokenizer, "eos_id", None)}
        for tok_str, tok_id in getattr(self.tokenizer, "_specials", {}).items():
            if "end" in tok_str.lower() or "eot" in tok_str.lower():
                eos_ids.add(int(tok_id))
        eos_ids.discard(None)
        return self._gen_config_cls(
            max_tokens=int(max_tokens),
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            repeat_penalty=1.0,
            seed=DEFAULT_SEED,
            eos_token_ids=tuple(sorted(int(tok_id) for tok_id in eos_ids)),
        )

    def run(self, prompt: str, max_tokens: int, trace_decode: bool = False) -> Dict[str, Any]:
        try:
            tokens = self.tokenizer.encode(prompt)
            cfg = self._gen_config_cls(
                **_build_generation_kwargs(getattr(self.tokenizer, "eos_id", None), max_tokens)
            )
            t0 = time.perf_counter()
            out_tokens, stats = self.engine.generate(
                tokens, cfg, stream=False, trace_decode=trace_decode
            )
            total_s = time.perf_counter() - t0
            record = {
                "backend": "adam",
                "status": "ok",
                "error": None,
                "n_prompt": _safe_int(stats.get("n_prompt"), len(tokens)),
                "n_gen": _safe_int(stats.get("n_gen"), len(out_tokens)),
                "prefill_tps": _safe_float(stats.get("prefill_tps")),
                "decode_tps": _safe_float(stats.get("decode_tps")),
                "prefill_s": _safe_float(stats.get("prefill_s")),
                "decode_s": _safe_float(stats.get("decode_s")),
                "total_s": _safe_float(stats.get("total_s"), total_s),
        "sampling_mode": stats.get("sampling_mode"),
        "runtime_profile": getattr(self.engine, "_runtime_profile", None),
        "decode_path": getattr(self.engine, "_decode_path", None),
        "trace_summary": _normalize_trace_summary(stats.get("trace_summary")),
        "text": self.tokenizer.decode(out_tokens),
            }
            record.update(_collect_runtime_knobs(self.engine, trace_decode))
            return _finalize_run_record(record, max_tokens)
        except Exception as exc:
            return _error_record("adam", exc)

    def run_chat_turn(self, prompt1: str, prompt2: str, max_tokens: int, trace_decode: bool = False) -> Dict[str, Any]:
        from adamah_chat import _assistant_history_message, _chat_reuse_plan, _render_messages_tokens

        try:
            cfg = self._chat_generation_config(max_tokens)

            turn1_messages = [{"role": "user", "content": prompt1}]
            turn1_start = time.perf_counter()
            _, _, turn1_ids, turn1_prep = _render_messages_tokens(turn1_messages, self.cfg, self.tokenizer)
            turn1_out, turn1_stats = self.engine.generate(turn1_ids, cfg, stream=False, trace_decode=False)
            turn1_total_s = time.perf_counter() - turn1_start
            turn1_text = self.tokenizer.decode(turn1_out)

            turn2_messages = turn1_messages + [
                _assistant_history_message(turn1_text, turn1_out),
                {"role": "user", "content": prompt2},
            ]
            turn2_start = time.perf_counter()
            _, _, turn2_ids, turn2_prep = _render_messages_tokens(turn2_messages, self.cfg, self.tokenizer)
            session_tokens = list(turn1_ids) + list(turn1_out)
            reuse_plan = _chat_reuse_plan(session_tokens, turn2_ids, True)
            reuse_prefix = int(reuse_plan.get("reuse_prefix_tokens", 0) or 0)
            turn2_out, turn2_stats = self.engine.generate(
                turn2_ids,
                cfg,
                stream=False,
                trace_decode=trace_decode,
                reuse_prefix=reuse_prefix,
            )
            turn2_total_s = time.perf_counter() - turn2_start
            turn2_text = self.tokenizer.decode(turn2_out)

            chat_turn_metrics = {
                "turn1_prompt_render_ms": float(turn1_prep.get("render_s", 0.0)) * 1000.0,
                "turn1_prompt_tokenize_ms": float(turn1_prep.get("encode_s", 0.0)) * 1000.0,
                "turn1_prompt_total_tokens": len(turn1_ids),
                "turn1_prompt_reused_tokens": int(turn1_stats.get("n_prompt_reused", 0) or 0),
                "turn1_prompt_prefilled_tokens": int(turn1_stats.get("n_prompt_prefilled", len(turn1_ids)) or 0),
                "turn1_prefill_ms": float(turn1_stats.get("prefill_s", 0.0)) * 1000.0,
                "turn1_decode_ms": float(turn1_stats.get("decode_s", 0.0)) * 1000.0,
                "turn1_decode_tps": _safe_float(turn1_stats.get("decode_tps")),
                "turn1_total_turn_ms": turn1_total_s * 1000.0,
                "turn2_prompt_render_ms": float(turn2_prep.get("render_s", 0.0)) * 1000.0,
                "turn2_prompt_tokenize_ms": float(turn2_prep.get("encode_s", 0.0)) * 1000.0,
                "turn2_prompt_total_tokens": len(turn2_ids),
                "turn2_prompt_reused_tokens": int(turn2_stats.get("n_prompt_reused", reuse_prefix) or 0),
                "turn2_prompt_prefilled_tokens": int(turn2_stats.get("n_prompt_prefilled", len(turn2_ids)) or 0),
                "turn2_prefill_ms": float(turn2_stats.get("prefill_s", 0.0)) * 1000.0,
                "turn2_decode_ms": float(turn2_stats.get("decode_s", 0.0)) * 1000.0,
                "turn2_decode_tps": _safe_float(turn2_stats.get("decode_tps")),
                "turn2_total_turn_ms": turn2_total_s * 1000.0,
                "reuse_hit": bool(reuse_plan.get("reuse_hit")),
                "reuse_prefix_tokens": reuse_prefix,
                "reuse_miss_reason": reuse_plan.get("reuse_miss_reason"),
                "reuse_miss_index": int(reuse_plan.get("reuse_miss_index", -1) or -1),
            }

            record = {
                "backend": "adam",
                "status": "ok",
                "error": None,
                "n_prompt": int(turn2_stats.get("n_prompt_prefilled", len(turn2_ids)) or 0),
                "n_gen": _safe_int(turn2_stats.get("n_gen"), len(turn2_out)),
                "prefill_tps": _safe_float(turn2_stats.get("prefill_tps")),
                "decode_tps": _safe_float(turn2_stats.get("decode_tps")),
                "prefill_s": _safe_float(turn2_stats.get("prefill_s")),
                "decode_s": _safe_float(turn2_stats.get("decode_s")),
                "total_s": turn2_total_s,
                "sampling_mode": turn2_stats.get("sampling_mode"),
                "runtime_profile": getattr(self.engine, "_runtime_profile", None),
                "decode_path": getattr(self.engine, "_decode_path", None),
                "trace_summary": _normalize_trace_summary(turn2_stats.get("trace_summary")),
                "prompt_render_ms": chat_turn_metrics["turn2_prompt_render_ms"],
                "prompt_tokenize_ms": chat_turn_metrics["turn2_prompt_tokenize_ms"],
                "prompt_total_tokens": chat_turn_metrics["turn2_prompt_total_tokens"],
                "prompt_reused_tokens": chat_turn_metrics["turn2_prompt_reused_tokens"],
                "prompt_prefilled_tokens": chat_turn_metrics["turn2_prompt_prefilled_tokens"],
                "prefill_ms": chat_turn_metrics["turn2_prefill_ms"],
                "decode_ms": chat_turn_metrics["turn2_decode_ms"],
                "total_turn_ms": chat_turn_metrics["turn2_total_turn_ms"],
                "reuse_hit": chat_turn_metrics["reuse_hit"],
                "reuse_prefix_tokens": chat_turn_metrics["reuse_prefix_tokens"],
                "reuse_miss_reason": chat_turn_metrics["reuse_miss_reason"],
                "reuse_miss_index": chat_turn_metrics["reuse_miss_index"],
                "chat_turn_metrics": chat_turn_metrics,
                "text": turn2_text,
            }
            record.update(_collect_runtime_knobs(self.engine, trace_decode))
            return _finalize_run_record(record, max_tokens)
        except Exception as exc:
            return _error_record("adam", exc)


class LlamaRunner:
    def __init__(self, model_path: str, llama_bin: Optional[str], ngl: int = DEFAULT_LLAMA_NGL):
        self.model_path = model_path
        self.llama_bin = llama_bin
        self.ngl = ngl

    def is_available(self) -> bool:
        return bool(self.llama_bin and os.path.exists(self.llama_bin))

    def unavailable_reason(self) -> str:
        if self.llama_bin:
            return f"llama.cpp binary not found: {self.llama_bin}"
        return "llama.cpp binary not configured"

    def run(self, prompt: str, max_tokens: int, trace_decode: bool = False) -> Dict[str, Any]:
        del trace_decode
        if not self.is_available():
            return _skip_record("llama", self.unavailable_reason())

        cmd = [
            self.llama_bin,
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(max_tokens),
            "-ngl", str(self.ngl),
            "--temp", "0",
            "--top-k", "1",
            "--top-p", "1",
            "--repeat-penalty", "1",
            "--seed", str(DEFAULT_SEED),
        ]
        try:
            t0 = time.perf_counter()
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
            )
            total_s = time.perf_counter() - t0
        except Exception as exc:
            return _error_record("llama", exc)

        combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        if proc.returncode != 0:
            return _error_record("llama", combined or f"llama.cpp exit code {proc.returncode}")

        timings = parse_llama_timings(combined)
        if not timings:
            return _error_record("llama", "missing llama.cpp timing lines")

        record = {
            "backend": "llama",
            "status": "ok",
            "error": None,
            "n_prompt": timings["n_prompt"],
            "n_gen": timings["n_gen"],
            "prefill_tps": timings["prefill_tps"],
            "decode_tps": timings["decode_tps"],
            "prefill_s": timings["prefill_s"],
            "decode_s": timings["decode_s"],
            "total_s": total_s,
            "sampling_mode": "greedy",
            "runtime_profile": None,
            "trace_summary": _empty_trace_summary(),
            "text": proc.stdout or "",
        }
        return _finalize_run_record(record, max_tokens)


class OllamaRunner:
    def __init__(self, model: str, url: str = DEFAULT_OLLAMA_URL):
        self.model = model
        self.url = url.rstrip("/")

    def is_available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/api/tags", timeout=5) as resp:
                payload = json.loads(resp.read())
        except Exception:
            return False
        models = [entry.get("name", "") for entry in payload.get("models", [])]
        return any(self.model == name or self.model in name for name in models)

    def unavailable_reason(self) -> str:
        return f"Ollama model unavailable: {self.model}"

    def run(self, prompt: str, max_tokens: int, trace_decode: bool = False) -> Dict[str, Any]:
        del trace_decode
        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": int(max_tokens),
                "temperature": 0,
                "top_k": 1,
                "top_p": 1,
                "repeat_penalty": 1,
                "seed": DEFAULT_SEED,
            },
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=300) as resp:
                payload = json.loads(resp.read())
            total_s = time.perf_counter() - t0
        except Exception as exc:
            return _error_record("ollama", exc)

        record = {
            "backend": "ollama",
            "status": "ok",
            "error": None,
            "n_prompt": _safe_int(payload.get("prompt_eval_count")),
            "n_gen": _safe_int(payload.get("eval_count")),
            "prefill_tps": (
                _safe_int(payload.get("prompt_eval_count")) /
                (_safe_float(payload.get("prompt_eval_duration")) / 1e9)
                if _safe_float(payload.get("prompt_eval_duration")) > 0 else 0.0
            ),
            "decode_tps": (
                _safe_int(payload.get("eval_count")) /
                (_safe_float(payload.get("eval_duration")) / 1e9)
                if _safe_float(payload.get("eval_duration")) > 0 else 0.0
            ),
            "prefill_s": _safe_float(payload.get("prompt_eval_duration")) / 1e9,
            "decode_s": _safe_float(payload.get("eval_duration")) / 1e9,
            "total_s": total_s,
            "sampling_mode": "greedy",
            "runtime_profile": None,
            "trace_summary": _empty_trace_summary(),
            "text": payload.get("response", ""),
        }
        return _finalize_run_record(record, max_tokens)


def _run_backend_suite(backend: str,
                       runner: Optional[Any],
                       prompt: str,
                       args: argparse.Namespace) -> Dict[str, Any]:
    if args.preset == CHAT_TURN_REGRESSION_PRESET and backend != "adam":
        skipped = _skip_record(backend, "chat_turn_regression is only implemented for ADAM")
        return {
            "backend": backend,
            "cold_first_run": None,
            "warmup": [],
            "runs": [],
            "diagnostic": None,
            "summary": summarize_backend_runs(backend, None, [skipped]),
        }

    if runner is None:
        return {
            "backend": backend,
            "cold_first_run": None,
            "warmup": [],
            "runs": [],
            "diagnostic": None,
            "summary": summarize_backend_runs(
                backend,
                None,
                [_skip_record(backend, "backend skipped by CLI")],
            ),
        }

    if hasattr(runner, "is_available") and not runner.is_available():
        skipped = _skip_record(backend, runner.unavailable_reason())
        return {
            "backend": backend,
            "cold_first_run": None,
            "warmup": [],
            "runs": [],
            "diagnostic": None,
            "summary": summarize_backend_runs(backend, None, [skipped]),
        }

    run_fn = runner.run
    run_args = (prompt, args.max_tokens)
    if args.preset == CHAT_TURN_REGRESSION_PRESET and backend == "adam":
        run_fn = runner.run_chat_turn
        run_args = (args.turn_prompt1, args.turn_prompt2, args.max_tokens)

    cold_first_run = _run_with_gpu_telemetry(backend, args, run_fn, *run_args, trace_decode=False)
    if cold_first_run.get("status") == "error":
        return {
            "backend": backend,
            "cold_first_run": cold_first_run,
            "warmup": [],
            "runs": [],
            "diagnostic": None,
            "summary": summarize_backend_runs(backend, cold_first_run, [cold_first_run]),
        }

    warmup = []
    for _ in range(max(0, int(args.warmup))):
        result = _run_with_gpu_telemetry(backend, args, run_fn, *run_args, trace_decode=False)
        warmup.append(result)
        if result.get("status") == "error":
            return {
                "backend": backend,
                "cold_first_run": cold_first_run,
                "warmup": warmup,
                "runs": [],
                "diagnostic": None,
                "summary": summarize_backend_runs(backend, cold_first_run, [result]),
            }

    runs = [
        _run_with_gpu_telemetry(backend, args, run_fn, *run_args, trace_decode=False)
        for _ in range(max(1, int(args.runs)))
    ]
    diagnostic = None
    if backend == "adam" and args.diagnose_trace:
        diag_args = run_args[:-1] + (min(int(args.max_tokens), DEFAULT_TRACE_TOKENS),)
        diagnostic = _run_with_gpu_telemetry(backend, args, run_fn, *diag_args, trace_decode=True)

    return {
        "backend": backend,
        "cold_first_run": cold_first_run,
        "warmup": warmup,
        "runs": runs,
        "diagnostic": diagnostic,
        "summary": summarize_backend_runs(backend, cold_first_run, runs),
    }


def _format_optional_number(value: Optional[float], width: int = 10, digits: int = 1) -> str:
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value:>{width}.{digits}f}"


def _format_ratio(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}x"


def _format_bool(value: Optional[bool]) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "no"


def _trace_line(trace_summary: Dict[str, Any]) -> str:
    summary = _normalize_trace_summary(trace_summary)
    if not trace_summary:
        return "trace unavailable"
    line = (
        "trace avg/token: total {step:.2f}ms | sample {sample:.2f}ms | "
        "forward {forward:.2f}ms | lm_head {lm_head:.2f}ms | shortlist {shortlist:.2f}ms | "
        "resolve {resolve:.2f}ms | "
        "core_batch {core_batch:.2f}ms | lm_batch {lm_batch:.2f}ms | "
        "rerank {rerank:.2f}ms | attn {attn:.2f}ms | ffn {ffn:.2f}ms"
    ).format(
        step=_safe_float(summary.get("step_ms_avg")),
        sample=_safe_float(summary.get("sample_ms_avg")),
        forward=_safe_float(summary.get("forward_ms_avg")),
        lm_head=_safe_float(summary.get("lm_head_ms_avg")),
        shortlist=_safe_float(summary.get("lm_head_shortlist_ms_avg")),
        resolve=_safe_float(summary.get("sample_resolve_ms_avg")),
        core_batch=_safe_float(summary.get("core_batch_ms_avg")),
        lm_batch=_safe_float(summary.get("lm_head_batch_ms_avg")),
        rerank=_safe_float(summary.get("rerank_batch_ms_avg")),
        attn=_safe_float(summary.get("attn_ms_avg")),
        ffn=_safe_float(summary.get("ffn_ms_avg")),
    )
    if any(_safe_float(summary.get(key)) > 0.0 for key in (
        "dispatch_count_total",
        "submit_count_total",
        "barrier_count_total",
        "fusion_flush_count_total",
        "descriptor_set_update_count_total",
        "descriptor_cache_hit_count_total",
        "descriptor_cache_miss_count_total",
        "alias_conflict_count_total",
    )):
        line += (
            " | dispatch {dispatch:.1f}/tok | submit {submit:.2f}/tok | "
            "barrier {barrier:.1f}/tok | flush {flush:.1f}/tok | "
            "desc {desc:.1f}/tok | desc cache {desc_hit:.1f} hit {desc_miss:.1f} miss | "
            "alias {alias:.1f}/tok"
        ).format(
            dispatch=_safe_float(summary.get("dispatch_count_per_token")),
            submit=_safe_float(summary.get("submit_count_per_token")),
            barrier=_safe_float(summary.get("barrier_count_per_token")),
            flush=_safe_float(summary.get("fusion_flush_count_per_token")),
            desc=_safe_float(summary.get("descriptor_set_update_count_per_token")),
            desc_hit=_safe_float(summary.get("descriptor_cache_hit_count_per_token")),
            desc_miss=_safe_float(summary.get("descriptor_cache_miss_count_per_token")),
            alias=_safe_float(summary.get("alias_conflict_count_per_token")),
        )
    if summary.get("scheduler_mode_name"):
        line += f" | scheduler {summary.get('scheduler_mode_name')}"
    breakdown = summary.get("descriptor_cache_breakdown") or {}
    if isinstance(breakdown, dict) and breakdown:
        top_misses = sorted(
            (
                (str(name), _safe_float((stats or {}).get("miss_count")))
                for name, stats in breakdown.items()
                if _safe_float((stats or {}).get("miss_count")) > 0.0
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:2]
        if top_misses:
            line += " | desc miss top " + ", ".join(
                f"{name}={miss:.1f}" for name, miss in top_misses
            )
    return line


def _gpu_telemetry_line(gpu_telemetry: Dict[str, Any]) -> str:
    telemetry = _normalize_gpu_telemetry(gpu_telemetry)
    if not telemetry.get("available"):
        error = telemetry.get("error")
        return f"gpu telemetry unavailable{': ' + str(error) if error else ''}"
    return (
        "gpu telemetry: "
        f"util {telemetry.get('gpu_util_avg', 0.0):.1f}% avg {telemetry.get('gpu_util_max', 0.0):.1f}% max | "
        f"mem {telemetry.get('mem_util_avg', 0.0):.1f}% avg {telemetry.get('mem_util_max', 0.0):.1f}% max | "
        f"used {telemetry.get('mem_used_mb_avg', 0.0):.0f}/{telemetry.get('mem_total_mb', 0.0):.0f}MB avg | "
        f"sm {telemetry.get('sm_clock_mhz_avg', 0.0):.0f}MHz avg | "
        f"power {telemetry.get('power_w_avg', 0.0):.1f}W avg {telemetry.get('power_w_max', 0.0):.1f}W max"
    )


def _fixed_pool_pair(profile: Optional[str]) -> Dict[str, int]:
    name = (profile or "").strip().lower()
    if name.startswith("broadcom") or name == "unified_small_gpu":
        return {"pool_hot_mb": 64, "pool_cold_mb": 32}
    if name == "nvidia_discrete":
        return {"pool_hot_mb": 768, "pool_cold_mb": 1536}
    return {"pool_hot_mb": 512, "pool_cold_mb": 1024}


def _build_sweep_cases(args: argparse.Namespace) -> List[Dict[str, Any]]:
    profile_name = args.profile or "auto"
    cases: List[Dict[str, Any]] = []
    seen = set()
    for stream_load, chunk_mb, kv_cap, rows_per_group, pool_mode, sampler in itertools.product(
        SWEEP_STREAM_LOAD,
        SWEEP_STREAM_CHUNK_MB,
        SWEEP_KV_CAP,
        SWEEP_ROWS_PER_GROUP,
        SWEEP_POOL_MODES,
        SWEEP_SAMPLERS,
    ):
        startup = {
            "stream_load": bool(stream_load),
            "stream_chunk_mb": int(chunk_mb),
            "kv_cap": int(kv_cap),
            "gpu_fused_rows_per_group": int(rows_per_group),
            "gpu_fused_topk": sampler == "gpu_fused_topk",
            "gpu_approx_rerank": False,
            "trace_decode": False,
        }
        if pool_mode == "fixed":
            startup.update(_fixed_pool_pair(args.profile))
        key = tuple(sorted(startup.items()))
        if key in seen:
            continue
        seen.add(key)
        label = (
            f"profile={profile_name} stream={'on' if stream_load else 'off'} "
            f"chunk={chunk_mb}MB kv={kv_cap} rows={rows_per_group} "
            f"pool={pool_mode} sampler={sampler}"
        )
        cases.append({"label": label, "startup": startup})
    return cases


def _print_sweep_summary(sweep_results: List[Dict[str, Any]]) -> None:
    if not sweep_results:
        return
    ranked = [
        result for result in sweep_results
        if result.get("overall", {}).get("status") in {"ok", "short_run"}
    ]
    ranked.sort(key=lambda result: _safe_float(result.get("overall", {}).get("steady_decode_tps")), reverse=True)
    print("")
    print("=" * 88)
    print("ADAM SWEEP TOP CASES")
    print("=" * 88)
    print(f"{'rank':<5} {'cold':>10} {'steady':>10} {'kpi':>10} {'status':<10} case")
    print("-" * 88)
    for idx, result in enumerate(ranked[:10], start=1):
        overall = result.get("overall", {})
        print(
            f"{idx:<5} {_format_optional_number(overall.get('first_run_decode_tps'))} "
            f"{_format_optional_number(overall.get('steady_decode_tps'))} "
            f"{_format_optional_number(overall.get('kpi_decode_tps'))} "
            f"{overall.get('status', 'n/a'):<10} {result.get('label', '')}"
        )
    if not ranked:
        print("no successful sweep cases")


def _apply_case_args(base_cmd: List[str], startup: Dict[str, Any]) -> List[str]:
    cmd = list(base_cmd)
    if startup.get("stream_load") is not None:
        cmd.extend(["--stream-load", "on" if startup.get("stream_load") else "off"])
    if startup.get("stream_chunk_mb") is not None:
        cmd.extend(["--stream-chunk-mb", str(int(startup["stream_chunk_mb"]))])
    if startup.get("kv_cap") is not None:
        cmd.extend(["--kv-cap", str(int(startup["kv_cap"]))])
    if startup.get("pool_hot_mb") is not None:
        cmd.extend(["--pool-hot-mb", str(int(startup["pool_hot_mb"]))])
    if startup.get("pool_cold_mb") is not None:
        cmd.extend(["--pool-cold-mb", str(int(startup["pool_cold_mb"]))])
    if startup.get("gpu_fused_rows_per_group") is not None:
        cmd.extend(["--gpu-fused-rows-per-group", str(int(startup["gpu_fused_rows_per_group"]))])
    if startup.get("gpu_fused_topk") is not None:
        cmd.extend(["--gpu-fused-topk", "on" if startup.get("gpu_fused_topk") else "off"])
    if startup.get("fusion_scheduler_mode"):
        cmd.extend(["--fusion-scheduler-mode", str(startup["fusion_scheduler_mode"])])
    if startup.get("direct_kv_cache_write") is not None:
        cmd.extend(["--direct-kv-cache-write", "on" if startup.get("direct_kv_cache_write") else "off"])
    if startup.get("experimental_qk_norm_rope") is not None:
        cmd.extend(["--experimental-qk-norm-rope", "on" if startup.get("experimental_qk_norm_rope") else "off"])
    if startup.get("experimental_merged_qkv") is not None:
        cmd.extend(["--experimental-merged-qkv", "on" if startup.get("experimental_merged_qkv") else "off"])
    if startup.get("experimental_fused_qkv_qk_norm_rope") is not None:
        cmd.extend([
            "--experimental-fused-qkv-qk-norm-rope",
            "on" if startup.get("experimental_fused_qkv_qk_norm_rope") else "off",
        ])
    if startup.get("experimental_merged_gateup") is not None:
        cmd.extend(["--experimental-merged-gateup", "on" if startup.get("experimental_merged_gateup") else "off"])
    if startup.get("experimental_attn_softmax_value") is not None:
        cmd.extend([
            "--experimental-attn-softmax-value",
            "on" if startup.get("experimental_attn_softmax_value") else "off",
        ])
    if startup.get("experimental_rmsnorm_add") is not None:
        cmd.extend(["--experimental-rmsnorm-add", "on" if startup.get("experimental_rmsnorm_add") else "off"])
    return cmd


def _run_sweep_case_subprocess(args: argparse.Namespace,
                               prompts: List[str],
                               case: Dict[str, Any]) -> Dict[str, Any]:
    fd, tmp_path = tempfile.mkstemp(prefix="adam_sweep_", suffix=".json")
    os.close(fd)
    try:
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--model", args.model,
            "--preset", args.preset,
            "--runs", str(args.runs),
            "--warmup", str(args.warmup),
            "--max-tokens", str(args.max_tokens),
            "--adam-only",
            "--skip-llama",
            "--skip-ollama",
            "--json-out", tmp_path,
        ]
        if args.diagnose_trace:
            cmd.append("--diagnose-trace")
        if getattr(args, "gpu_telemetry", None):
            cmd.extend(["--gpu-telemetry", str(args.gpu_telemetry)])
        if args.profile:
            cmd.extend(["--profile", args.profile])
        if args.preset == CHAT_TURN_REGRESSION_PRESET:
            cmd.extend(["--turn-prompt1", args.turn_prompt1, "--turn-prompt2", args.turn_prompt2])
        else:
            for prompt in prompts:
                cmd.extend(["--prompt", prompt])
        cmd = _apply_case_args(cmd, case["startup"])
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
        )
        if proc.returncode != 0:
            return {
                "label": case["label"],
                "startup": case["startup"],
                "prompt_results": [],
                "overall": summarize_backend_runs("adam", None, [_error_record("adam", proc.stderr or proc.stdout or f"subprocess exit {proc.returncode}")]),
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        with open(tmp_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        prompt_results = payload.get("prompt_results", [])
        return {
            "label": case["label"],
            "startup": case["startup"],
            "prompt_results": prompt_results,
            "overall": payload.get("overall", {}).get("adam", {}),
            "trace_summary": _median_trace_summary_for_backend(prompt_results, backend="adam"),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _run_adam_sweep(args: argparse.Namespace, prompts: List[str]) -> List[Dict[str, Any]]:
    sweep_results: List[Dict[str, Any]] = []
    for case in _build_sweep_cases(args):
        print(f"[SWEEP] {case['label']}")
        sweep_results.append(_run_sweep_case_subprocess(args, prompts, case))
    return sweep_results


def _build_decode_ablation_cases(args: Optional[argparse.Namespace] = None) -> List[Dict[str, Any]]:
    cases = [{
        "id": "baseline",
        "label": "baseline current fast path",
        "feature": "current fast path",
        "mode": "baseline",
        "startup": {},
    }]
    for case in DECODE_ABLATION_CASES:
        # The fused merged QKV->Q/K prep path is only interesting on profiles
        # where we are actively exploring dispatch-cut decode scheduling.
        if case.get("id") == "fused_qkv_qk_norm_rope_off" and not _profile_prefers_dispatch_cut(args):
            continue
        cases.append(dict(case))
    return cases


def _make_decode_ablation_result(case: Dict[str, Any],
                                 overall_adam: Dict[str, Any],
                                 trace_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "id": case["id"],
        "label": case["label"],
        "feature": case["feature"],
        "mode": case["mode"],
        "startup": dict(case.get("startup", {})),
        "overall": dict(overall_adam or {}),
        "trace_summary": _normalize_trace_summary(trace_summary),
    }


def _interpret_decode_ablation(case: Dict[str, Any], delta_tps: float) -> str:
    feature = str(case.get("feature") or case.get("label") or "this path")
    if case.get("mode") == "baseline":
        return "Reference current path."
    if case.get("mode") == "disable":
        if delta_tps <= -0.5:
            return f"{feature} is helping now; keep it and optimize deeper in that subgraph."
        if delta_tps >= 0.5:
            return f"{feature} is hurting desktop decode as implemented; do not deepen it before fixing it."
        return f"{feature} is roughly neutral on desktop right now."
    if delta_tps >= 0.5:
        return f"{feature} looks promising; this is a good candidate for deeper implementation."
    if delta_tps <= -0.5:
        return f"{feature} still hurts desktop decode; not the next desktop target."
    return f"{feature} is roughly neutral on desktop right now."


def _run_decode_ablations(args: argparse.Namespace,
                          prompts: List[str],
                          prompt_results: List[Dict[str, Any]],
                          overall: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    cases = _build_decode_ablation_cases(args)
    results = [
        _make_decode_ablation_result(
            cases[0],
            dict((overall or {}).get("adam") or {}),
            _median_trace_summary_for_backend(prompt_results, backend="adam"),
        )
    ]
    for case in cases[1:]:
        print(f"[ABLATE] {case['label']}")
        payload = _run_sweep_case_subprocess(args, prompts, case)
        results.append(
            _make_decode_ablation_result(
                case,
                payload.get("overall", {}),
                payload.get("trace_summary"),
            )
        )

    baseline_tps = _safe_float(results[0].get("overall", {}).get("steady_decode_tps"))
    for result in results:
        current_tps = _safe_float(result.get("overall", {}).get("steady_decode_tps"))
        result["delta_decode_tps"] = current_tps - baseline_tps
        result["takeaway"] = _interpret_decode_ablation(result, result["delta_decode_tps"])
    return results


def _print_decode_ablation_summary(results: List[Dict[str, Any]]) -> None:
    if not results:
        return
    baseline = results[0]
    ranked = sorted(results[1:], key=lambda item: _safe_float(item.get("delta_decode_tps")), reverse=True)
    base_overall = baseline.get("overall", {})
    base_trace = baseline.get("trace_summary", {})
    print("")
    print("=" * 104)
    print("DECODE ABLATION SUMMARY")
    print("=" * 104)
    print(
        "baseline: "
        f"steady={_safe_float(base_overall.get('steady_decode_tps')):.1f} tok/s | "
        f"core={_safe_float(base_trace.get('core_batch_ms_avg')):.2f} ms | "
        f"lm={_safe_float(base_trace.get('lm_head_batch_ms_avg')):.2f} ms | "
        f"dispatch={_safe_float(base_trace.get('dispatch_count_per_token')):.1f}/tok | "
        f"desc={_safe_float(base_trace.get('descriptor_set_update_count_per_token')):.1f}/tok | "
        f"flush={_safe_float(base_trace.get('fusion_flush_count_per_token')):.1f}/tok"
    )
    print("-" * 104)
    print(
        f"{'rank':<5} {'steady':>10} {'delta':>8} {'core':>8} {'lm':>8} "
        f"{'dispatch':>10} {'desc':>10} {'flush':>8} {'status':<10} case"
    )
    print("-" * 104)
    for idx, result in enumerate(ranked, start=1):
        summary = result.get("overall", {})
        trace = result.get("trace_summary", {})
        print(
            f"{idx:<5} "
            f"{_format_optional_number(summary.get('steady_decode_tps'))} "
            f"{result.get('delta_decode_tps', 0.0):>8.1f} "
            f"{_safe_float(trace.get('core_batch_ms_avg')):>8.2f} "
            f"{_safe_float(trace.get('lm_head_batch_ms_avg')):>8.2f} "
            f"{_safe_float(trace.get('dispatch_count_per_token')):>10.1f} "
            f"{_safe_float(trace.get('descriptor_set_update_count_per_token')):>10.1f} "
            f"{_safe_float(trace.get('fusion_flush_count_per_token')):>8.1f} "
            f"{str(summary.get('status', 'n/a')):<10} {result.get('label', '')}"
        )
    print("")
    print("Where To Implement Next")
    for result in ranked:
        print(f"- {result.get('label')}: {result.get('takeaway')}")


def _print_prompt_summary(prompt_result: Dict[str, Any]) -> None:
    prompt = prompt_result["prompt"]
    print("")
    print("=" * 88)
    print(f"PROMPT: {_preview(prompt, 88)}")
    print("=" * 88)
    print(
        f"{'backend':<10} {'status':<10} {'decode_tps':>10} {'prefill_tps':>11} "
        f"{'total_s':>8} {'beats_llama':>11} {'beats_ollama':>12}"
    )
    print("-" * 88)
    for backend in BACKENDS:
        summary = prompt_result["backends"][backend]["summary"]
        beats_llama = prompt_result["comparison"]["beats_llama"] if backend == "adam" else None
        beats_ollama = prompt_result["comparison"]["beats_ollama"] if backend == "adam" else None
        print(
            f"{backend:<10} {summary['status']:<10} "
            f"{_format_optional_number(summary.get('decode_tps'))} "
            f"{_format_optional_number(summary.get('prefill_tps'), width=11)} "
            f"{_format_optional_number(summary.get('total_s'), width=8, digits=2)} "
            f"{_format_bool(beats_llama):>11} {_format_bool(beats_ollama):>12}"
        )
        if summary.get("error"):
            print(f"  note: {summary['error']}")
        elif backend == "adam" and summary.get("runtime_profile"):
            print(
                "  runtime: "
                f"profile={summary.get('runtime_profile')} "
                f"path={summary.get('decode_path') or 'n/a'} "
                f"stream={'on' if summary.get('stream_load') else 'off'} "
                f"chunk={summary.get('stream_chunk_mb')}MB "
                f"kv={summary.get('kv_cap')} "
                f"rows={summary.get('gpu_fused_rows_per_group')} "
                f"scheduler={summary.get('fusion_scheduler_mode') or 'n/a'} "
                f"direct_kv={'on' if summary.get('direct_kv_cache_write') else 'off'} "
                f"trace={'on' if summary.get('trace_decode') else 'off'}"
            )
            print(
                "  throughput: "
                f"cold={summary.get('first_run_decode_tps', 0.0):.1f} "
                f"steady={summary.get('steady_decode_tps', 0.0):.1f} "
                f"kpi={summary.get('kpi_decode_tps', 0.0):.1f} "
                f"target={summary.get('kpi_target_tps', DEFAULT_KPI_TARGET_TPS):.0f}"
            )
            gpu_telemetry = summary.get("gpu_telemetry") or {}
            if gpu_telemetry.get("available") or gpu_telemetry.get("error"):
                print(f"  {_gpu_telemetry_line(gpu_telemetry)}")
            if _safe_float(summary.get("prompt_total_tokens")) > 0.0:
                reuse_note = (
                    f"hit {int(summary.get('reuse_prefix_tokens') or 0)}"
                    if summary.get("reuse_hit")
                    else (
                        f"miss {summary.get('reuse_miss_reason')}@{summary.get('reuse_miss_index')}"
                        if summary.get("reuse_miss_reason") else "miss"
                    )
                )
                print(
                    "  chat: "
                    f"render={_safe_float(summary.get('prompt_render_ms')):.1f}ms "
                    f"tok={_safe_float(summary.get('prompt_tokenize_ms')):.1f}ms "
                    f"prompt={int(_safe_float(summary.get('prompt_total_tokens')))} "
                    f"prefilled={int(_safe_float(summary.get('prompt_prefilled_tokens')))} "
                    f"reused={int(_safe_float(summary.get('prompt_reused_tokens')))} "
                    f"turn_total={_safe_float(summary.get('total_turn_ms')):.1f}ms "
                    f"reuse={reuse_note}"
                )
    winner = prompt_result["comparison"].get("winner") or "n/a"
    print(
        f"winner={winner} | adam/llama={_format_ratio(prompt_result['comparison']['adam_vs_llama'])} "
        f"| adam/ollama={_format_ratio(prompt_result['comparison']['adam_vs_ollama'])}"
    )
    diagnostic = prompt_result["backends"]["adam"].get("diagnostic")
    if diagnostic:
        print(f"  [adam] {_trace_line(diagnostic.get('trace_summary') or {})}")


def _print_overall_summary(overall: Dict[str, Dict[str, Any]]) -> None:
    print("")
    print("=" * 88)
    print("OVERALL MEDIANS")
    print("=" * 88)
    print(
        f"{'backend':<10} {'status':<10} {'decode_tps':>10} {'prefill_tps':>11} "
        f"{'total_s':>8} {'beats_llama':>11} {'beats_ollama':>12}"
    )
    print("-" * 88)
    for backend in BACKENDS:
        summary = overall[backend]
        beats_llama = summary.get("beats_llama") if backend == "adam" else None
        beats_ollama = summary.get("beats_ollama") if backend == "adam" else None
        print(
            f"{backend:<10} {summary['status']:<10} "
            f"{_format_optional_number(summary.get('decode_tps'))} "
            f"{_format_optional_number(summary.get('prefill_tps'), width=11)} "
            f"{_format_optional_number(summary.get('total_s'), width=8, digits=2)} "
            f"{_format_bool(beats_llama):>11} {_format_bool(beats_ollama):>12}"
        )
        if summary.get("error"):
            print(f"  note: {summary['error']}")
        elif backend == "adam":
            gpu_telemetry = summary.get("gpu_telemetry") or {}
            if gpu_telemetry.get("available") or gpu_telemetry.get("error"):
                print(f"  {_gpu_telemetry_line(gpu_telemetry)}")
        elif backend == "adam":
            print(
                f"  kpi: cold={summary.get('first_run_decode_tps', 0.0):.1f} "
                f"steady={summary.get('steady_decode_tps', 0.0):.1f} "
                f"target={summary.get('kpi_target_tps', DEFAULT_KPI_TARGET_TPS):.0f} "
                f"met={'yes' if summary.get('meets_kpi_target') else 'no'}"
            )
            if _safe_float(summary.get("prompt_total_tokens")) > 0.0:
                reuse_note = (
                    f"hit {int(summary.get('reuse_prefix_tokens') or 0)}"
                    if summary.get("reuse_hit")
                    else (
                        f"miss {summary.get('reuse_miss_reason')}@{summary.get('reuse_miss_index')}"
                        if summary.get("reuse_miss_reason") else "miss"
                    )
                )
                print(
                    "  chat: "
                    f"render={_safe_float(summary.get('prompt_render_ms')):.1f}ms "
                    f"tok={_safe_float(summary.get('prompt_tokenize_ms')):.1f}ms "
                    f"prompt={int(_safe_float(summary.get('prompt_total_tokens')))} "
                    f"prefilled={int(_safe_float(summary.get('prompt_prefilled_tokens')))} "
                    f"reused={int(_safe_float(summary.get('prompt_reused_tokens')))} "
                    f"turn_total={_safe_float(summary.get('total_turn_ms')):.1f}ms "
                    f"reuse={reuse_note}"
                )
    print(
        f"adam/llama={_format_ratio(overall['adam'].get('ratio_vs_llama'))} | "
        f"adam/ollama={_format_ratio(overall['adam'].get('ratio_vs_ollama'))}"
    )
    winners = [winner for winner in overall.get("winners", []) if winner]
    if winners:
        counts = {name: winners.count(name) for name in sorted(set(winners))}
        winners_line = ", ".join(f"{name}={count}" for name, count in counts.items())
        print(f"prompt winners: {winners_line}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.adam_only:
        args.skip_llama = True
        args.skip_ollama = True
    prompts = _prompt_list(args)

    print("=" * 88)
    print("ADAM runtime benchmark vs llama.cpp and Ollama")
    print("=" * 88)
    print(
        f"model={args.model} | prompts={len(prompts)} | warmup={args.warmup} | "
        f"runs={args.runs} | max_tokens={args.max_tokens} | deterministic=yes | "
        f"preset={args.preset} | profile={args.profile or 'auto'} | "
        f"kpi_target={DEFAULT_KPI_TARGET_TPS:.0f}"
    )

    runners: Dict[str, Optional[Any]] = {
        "adam": None,
        "llama": None,
        "ollama": None,
    }

    if not args.skip_adam:
        print("[INIT] Loading ADAM runtime path...")
        try:
            runners["adam"] = ADAMRunner(args.model, startup=_build_adam_startup(args)).load(verbose=True)
        except Exception as exc:
            print(f"[INIT] ADAM unavailable: {exc}")
            runners["adam"] = None
    else:
        print("[INIT] ADAM skipped")

    if not args.skip_llama:
        runners["llama"] = LlamaRunner(args.model, args.llama_bin, ngl=args.llama_ngl)
        if runners["llama"].is_available():
            print(f"[INIT] llama.cpp ready ({args.llama_bin})")
        else:
            print(f"[INIT] llama.cpp skipped: {runners['llama'].unavailable_reason()}")
    else:
        print("[INIT] llama.cpp skipped")

    if not args.skip_ollama:
        runners["ollama"] = OllamaRunner(args.ollama_model, args.ollama_url)
        if runners["ollama"].is_available():
            print(f"[INIT] Ollama ready ({args.ollama_model})")
        else:
            print(f"[INIT] Ollama skipped: {runners['ollama'].unavailable_reason()}")
    else:
        print("[INIT] Ollama skipped")

    prompt_results = []
    for prompt in prompts:
        backend_results = {
            backend: _run_backend_suite(backend, runners[backend], prompt, args)
            for backend in BACKENDS
        }
        prompt_result = {
            "prompt": prompt,
            "backends": backend_results,
            "comparison": _build_prompt_comparison(prompt, backend_results),
        }
        prompt_results.append(prompt_result)
        _print_prompt_summary(prompt_result)

    overall = build_overall_summary(prompt_results)
    _print_overall_summary(overall)
    sweep_results = []
    decode_ablation_results = []
    if args.sweep and not args.skip_adam:
        sweep_results = _run_adam_sweep(args, prompts)
        _print_sweep_summary(sweep_results)
    if args.decode_ablation and not args.skip_adam:
        decode_ablation_results = _run_decode_ablations(args, prompts, prompt_results, overall)
        _print_decode_ablation_summary(decode_ablation_results)

    if args.experiment_log:
        entry = build_experiment_entry(args, prompt_results, overall)
        append_experiment_log(args.experiment_log, entry)
        print(f"saved experiment log: {args.experiment_log}")

    if args.json_out:
        payload = build_json_report(
            args,
            prompt_results,
            overall,
            sweep_results=sweep_results,
            decode_ablations=decode_ablation_results,
        )
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        print(f"saved json: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
