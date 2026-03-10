#!/usr/bin/env python3
"""
ADAMAH Chat — Universal LLM Inference TUI
Pure Vulkan, zero CUDA. Loads any GGUF model via ADAM.
"""
import os, sys, time, glob

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ──────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
ADAMAH_DIR = os.path.join(ROOT, "adamah-MAIN")

# NOTE: Do NOT add adamah-MAIN/adamah to sys.path — that dir contains adamah.so
# which Python would try to import as a C extension (PyInit_adamah).
# The correct import path is adamah-MAIN (parent of the adamah package).
for p in [ROOT, ADAMAH_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from runtime_bootstrap import ensure_runtime

# ── ANSI Colors ────────────────────────────────────────────
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
RESET = "\033[0m"
BOX_TL = "╔"; BOX_TR = "╗"; BOX_BL = "╚"; BOX_BR = "╝"
BOX_H = "═"; BOX_V = "║"
FAST_LM_ROWS_PER_GROUP = 256

def box(text, width=52):
    pad = width - len(text) - 2
    print(f"{CYAN}{BOX_TL}{BOX_H * width}{BOX_TR}")
    print(f"{BOX_V}  {BOLD}{text}{RESET}{CYAN}{' ' * pad}{BOX_V}")
    print(f"{BOX_BL}{BOX_H * width}{BOX_BR}{RESET}")

def scan_models(search_dirs=None):
    """Find all .gguf files in project tree (deduplicated by realpath)."""
    dirs = search_dirs or [ROOT]
    models = []
    seen = set()
    for d in dirs:
        for p in glob.glob(os.path.join(d, "**/*.gguf"), recursive=True):
            rp = os.path.realpath(p)
            if rp not in seen:
                seen.add(rp)
                models.append(rp)
    return sorted(models, key=lambda p: os.path.basename(p).lower())

def model_info_short(path):
    """Quick peek at GGUF metadata without loading tensors."""
    import struct
    info = {"path": path, "name": os.path.basename(path), "size_gb": os.path.getsize(path) / 1e9}
    try:
        with open(path, 'rb') as f:
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != 0x46554747:
                info["error"] = "not GGUF"
                return info
            _ = struct.unpack('<I', f.read(4))[0]  # version
            n_tensors = struct.unpack('<Q', f.read(8))[0]
            n_kv = struct.unpack('<Q', f.read(8))[0]
            info["n_tensors"] = n_tensors
            # Read a few key metadata fields
            for _ in range(min(n_kv, 50)):
                kl = struct.unpack('<Q', f.read(8))[0]
                key = f.read(kl).decode('utf-8', errors='replace')
                vtype = struct.unpack('<I', f.read(4))[0]
                if vtype == 8:  # string
                    sl = struct.unpack('<Q', f.read(8))[0]
                    val = f.read(sl).decode('utf-8', errors='replace')
                elif vtype == 4:  # uint32
                    val = struct.unpack('<I', f.read(4))[0]
                elif vtype == 5:  # int32
                    val = struct.unpack('<i', f.read(4))[0]
                elif vtype == 10:  # uint64
                    val = struct.unpack('<Q', f.read(8))[0]
                elif vtype == 6:  # float32
                    val = struct.unpack('<f', f.read(4))[0]
                elif vtype == 9:  # array - skip
                    et = struct.unpack('<I', f.read(4))[0]
                    n = struct.unpack('<Q', f.read(8))[0]
                    size_map = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
                    if et in size_map:
                        f.seek(n * size_map[et], 1)
                    elif et == 8:
                        for _ in range(n):
                            sl2 = struct.unpack('<Q', f.read(8))[0]
                            f.seek(sl2, 1)
                    val = f"[{n} items]"
                else:
                    break
                if key == "general.architecture": info["arch"] = val
                elif key == "general.name": info["model_name"] = val
                elif ".block_count" in key: info["layers"] = val
                elif ".embedding_length" in key: info["embd"] = val
                elif "general.file_type" in key: info["file_type"] = val
    except Exception as e:
        info["error"] = str(e)
    return info

def select_model(models):
    """Interactive model selection."""
    if not models:
        print(f"{RED}No .gguf models found!{RESET}")
        print(f"{DIM}Place .gguf files in: {ROOT}{RESET}")
        sys.exit(1)
    if len(models) == 1:
        info = model_info_short(models[0])
        arch = info.get("arch", "?")
        name = info.get("model_name", info["name"])
        print(f"{GREEN}Found:{RESET} {name} ({arch}, {info['size_gb']:.1f}GB)")
        return models[0]

    print(f"\n{BOLD}Models found:{RESET}")
    infos = []
    for i, m in enumerate(models):
        info = model_info_short(m)
        infos.append(info)
        arch = info.get("arch", "?")
        layers = info.get("layers", "?")
        embd = info.get("embd", "?")
        name = info.get("model_name", info["name"])
        print(f"  {CYAN}[{i+1}]{RESET} {name}")
        print(f"      {DIM}{arch} | {layers}L | {embd}E | {info['size_gb']:.1f}GB{RESET}")

    while True:
        try:
            choice = input(f"\n{YELLOW}Select model [1]: {RESET}").strip()
            idx = int(choice) - 1 if choice else 0
            if 0 <= idx < len(models):
                return models[idx]
        except (ValueError, EOFError):
            pass
        print(f"{RED}Invalid choice{RESET}")

def _env_flag(name, default=None):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _mb_str(value_bytes):
    return f"{int(value_bytes) / (1024 * 1024):.0f}MB"


def _build_session_system_prompt(reasoning_notes: str | None = None) -> str | None:
    parts = []
    if reasoning_notes:
        parts.append(
            "Use the following working notes privately to answer the user's latest message. "
            "Do not mention the notes themselves unless the user explicitly asks for them.\n\n"
            "Working notes for the next reply:\n"
            f"{reasoning_notes.strip()}"
        )
    if not parts:
        return None
    return "\n\n".join(parts)


def _reasoning_cycles_from_env() -> int:
    value = os.environ.get("ADAM_REASONING_CYCLES")
    if value is not None:
        return max(0, int(value))
    return 2 if _env_flag("ADAM_REASONING", False) else 0


def _reasoning_trace_from_env() -> bool:
    return bool(_env_flag("ADAM_REASONING_TRACE", False))


def _reasoning_enabled(reasoning_cycles: int) -> bool:
    return reasoning_cycles > 0


def _auto_compaction_enabled(configured: bool, reasoning_cycles: int) -> bool:
    return configured


def _is_compaction_seed_message(msg: dict) -> bool:
    return (
        msg.get("role") == "system"
        and isinstance(msg.get("content"), str)
        and msg["content"].startswith("Context from a previous conversation:")
    )


def _build_compaction_seed_message(summary: str) -> dict:
    return {
        "role": "system",
        "content": (
            "Context from a previous conversation:\n"
            f"{summary.strip()}\n\n"
            "You are continuing the same conversation. Use this summary as background "
            "context and answer the user's next messages naturally."
        ),
    }


def _history_to_transcript(history) -> str:
    transcript = []
    for msg in history:
        role = msg.get("role", "user").capitalize()
        transcript.append(f"{role}: {msg.get('content', '').strip()}")
    return "\n".join(transcript).strip()


def _reasoning_stage_name(cycle_idx: int, total_cycles: int) -> str:
    if cycle_idx == 0:
        return "task framing"
    if cycle_idx + 1 == total_cycles:
        return "final polish"
    if cycle_idx == 1:
        return "answer plan"
    return "refinement"


def _build_reasoning_request(user_prompt: str, context_text: str,
                             draft: str | None, cycle_idx: int,
                             total_cycles: int) -> str:
    stage = _reasoning_stage_name(cycle_idx, total_cycles)
    base = [f"User request:\n{user_prompt.strip()}"]
    if context_text:
        base.append(f"Recent conversation context:\n{context_text}")

    if cycle_idx == 0 or not draft:
        base.append(
            "Create compact working notes for the next reasoning pass. "
            "Capture the task, hard constraints, likely answer shape, and the main risks. "
            "Do not answer the user directly. Return notes only."
        )
        return "\n\n".join(base)

    focus = {
        "answer plan": (
            "Transform the notes into a stronger answer plan. Keep the strongest points, "
            "remove fluff, and make the structure easier for the next pass to use."
        ),
        "refinement": (
            "Improve these notes for the next pass. Fix weak parts, tighten logic, "
            "remove repetition, and preserve only what helps the final answer."
        ),
        "final polish": (
            "Polish these notes into the smallest useful answer blueprint. "
            "Keep them sharp, concrete, and ready for a direct final response."
        ),
    }[stage]
    base.append(f"Current working notes:\n{draft.strip()}")
    base.append(f"Improve the notes. Focus: {focus}\nReturn notes only.")
    return "\n\n".join(base)


def _build_reasoning_context_text(history, user_prompt: str) -> str:
    convo = list(history) + [{"role": "user", "content": user_prompt}]
    return _history_to_transcript(convo)


def _run_reasoning_stage(engine, tokenizer, cfg, GenConfig, instruction: str,
                         seed=None, temperature: float = 0.2,
                         max_tokens: int = 128) -> str:
    prompt, add_bos = prepare_chat_messages(
        [{"role": "user", "content": instruction}],
        cfg.arch,
        tokenizer,
        chat_template=getattr(cfg, "chat_template", None),
        system_prompt=None,
    )
    tokens = tokenizer.encode(prompt, add_bos=add_bos)
    scratch_cfg = GenConfig(
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=20,
        top_p=0.9,
        seed=seed,
        eos_token_ids=(tokenizer.eos_id,),
    )
    out_tokens, _ = engine.generate(tokens, scratch_cfg, stream=False)
    return tokenizer.decode(out_tokens).strip()


def _runtime_profile_overrides(profile_name, cfg, unified):
    name = (profile_name or "").strip().lower()
    if not name or name in ("default", "auto"):
        return {"name": "default"}
    if name == "unified_small_gpu":
        return {
            "name": "unified_small_gpu",
            "reserve_ratio": 0.12,
            "kv_cap_max": 256,
            "stream_load": True,
            "stream_chunk_mb": 8,
            "gpu_approx_rerank": cfg.n_vocab >= 131072,
            "gpu_fused_rows_per_group": 128,
            "pool_hot_mb_max": 64,
            "pool_cold_mb_max": 32,
        }
    if name == "broadcom_v3dv":
        return {
            "name": "broadcom_v3dv",
            "reserve_ratio": 0.15,
            "kv_cap_max": 256,
            "stream_load": True,
            "stream_chunk_mb": 8,
            "gpu_approx_rerank": cfg.n_vocab >= 131072,
            "gpu_fused_rows_per_group": 128,
            "trace_decode": False,
            "pool_hot_mb_max": 64,
            "pool_cold_mb_max": 32,
        }
    if name == "broadcom_v3dv_exact":
        return {
            "name": "broadcom_v3dv_exact",
            "reserve_ratio": 0.15,
            "kv_cap_max": 256,
            "stream_load": True,
            "stream_chunk_mb": 8,
            "gpu_approx_rerank": False,
            "gpu_fused_rows_per_group": 256,
            "trace_decode": False,
            "pool_hot_mb_max": 64,
            "pool_cold_mb_max": 32,
        }
    if name == "broadcom_v3dv_approx":
        return {
            "name": "broadcom_v3dv_approx",
            "reserve_ratio": 0.15,
            "kv_cap_max": 256,
            "stream_load": True,
            "stream_chunk_mb": 8,
            "gpu_approx_rerank": cfg.n_vocab >= 131072,
            "gpu_fused_rows_per_group": 128,
            "trace_decode": False,
            "pool_hot_mb_max": 64,
            "pool_cold_mb_max": 32,
        }
    if name == "broadcom_v3dv_trace":
        return {
            "name": "broadcom_v3dv_trace",
            "reserve_ratio": 0.15,
            "kv_cap_max": 256,
            "stream_load": True,
            "stream_chunk_mb": 8,
            "gpu_approx_rerank": cfg.n_vocab >= 131072,
            "gpu_fused_rows_per_group": 128,
            "trace_decode": True,
            "pool_hot_mb_max": 64,
            "pool_cold_mb_max": 32,
        }
    if name == "broadcom_v3dv_narrow":
        return {
            "name": "broadcom_v3dv_narrow",
            "reserve_ratio": 0.15,
            "kv_cap_max": 256,
            "stream_load": True,
            "stream_chunk_mb": 8,
            "gpu_approx_rerank": cfg.n_vocab >= 131072,
            "gpu_fused_rows_per_group": 128,
            "trace_decode": False,
            "pool_hot_mb_max": 64,
            "pool_cold_mb_max": 32,
        }
    return {"name": name}


def build_runtime_plan(adamah_mod, loader, cfg, engine_cls):
    reserve_ratio = min(max(float(os.environ.get("ADAM_RUNTIME_RESERVE_RATIO", "0.10")), 0.0), 0.5)
    machine = ""
    try:
        machine = os.uname().machine.lower()
    except AttributeError:
        machine = ""
    arm_like = machine in ("aarch64", "arm64", "armv7l", "armv6l")
    host = {}
    device = {}
    try:
        host = adamah_mod.host_memory_info()
    except Exception:
        host = {}
    try:
        device = adamah_mod.probe_device()
    except Exception:
        device = {}

    unified = bool(device.get("is_unified_memory")) or arm_like
    profile_name = os.environ.get("ADAM_RUNTIME_PROFILE", "default")
    profile = _runtime_profile_overrides(profile_name, cfg, unified)
    if "reserve_ratio" in profile and os.environ.get("ADAM_RUNTIME_RESERVE_RATIO") is None:
        reserve_ratio = float(profile["reserve_ratio"])
    rows_per_group = max(
        1,
        _env_int(
            "ADAM_GPU_FUSED_ROWS_PER_GROUP",
            int(profile.get("gpu_fused_rows_per_group", FAST_LM_ROWS_PER_GROUP)),
        ),
    )
    approx_rerank = bool(
        _env_flag("ADAM_GPU_APPROX_RERANK", profile.get("gpu_approx_rerank", False))
    )
    trace_decode = bool(
        _env_flag("ADAM_TRACE_DECODE", profile.get("trace_decode", False))
    )
    host_avail = int(host.get("available_bytes") or 0)
    device_free = int(device.get("free_budget_bytes") or device.get("budget_bytes") or device.get("heap_bytes") or 0)
    usable_device = int(device_free * (1.0 - reserve_ratio)) if device_free > 0 else 0
    if unified and host_avail > 0:
        usable_device = min(usable_device or host_avail, int(host_avail * (1.0 - reserve_ratio)))

    kv_cap_env = os.environ.get("ADAM_KV_CAP")
    kv_cap = int(kv_cap_env) if kv_cap_env else engine_cls.KV_CAP_DEFAULT
    if unified and kv_cap_env is None:
        kv_cap = min(kv_cap, 512)
    if kv_cap_env is None and profile.get("kv_cap_max") is not None:
        kv_cap = min(kv_cap, int(profile["kv_cap_max"]))
    while True:
        gpu_est = engine_cls.estimate_persistent_gpu_bytes(
            cfg,
            loader.tensor_shapes,
            loader.tensor_types,
            kv_cap=kv_cap,
            gpu_tied_lm_head=True,
            gpu_approx_rerank=approx_rerank,
            gpu_fused_rows_per_group=rows_per_group,
        )
        if kv_cap <= 256 or usable_device <= 0:
            break
        limit_ratio = 0.80 if unified else 0.92
        if gpu_est["total_bytes"] <= int(usable_device * limit_ratio):
            break
        kv_cap //= 2

    pool_plan = {}
    try:
        pool_plan = adamah_mod.recommend_pool_sizes(
            working_set_bytes=gpu_est["total_bytes"],
            reserve_ratio=reserve_ratio,
        )
    except Exception:
        pool_plan = {}
    if pool_plan:
        if profile.get("pool_hot_mb_max") is not None:
            pool_plan["hot_mb"] = min(int(pool_plan.get("hot_mb", 0)), int(profile["pool_hot_mb_max"]))
        if profile.get("pool_cold_mb_max") is not None:
            pool_plan["cold_mb"] = min(int(pool_plan.get("cold_mb", 0)), int(profile["pool_cold_mb_max"]))

    eager_host_bytes = loader.estimate_raw_bytes() + loader.estimate_f32_bytes()
    stream_default = unified
    if host_avail > 0 and eager_host_bytes + 512 * 1024 * 1024 > int(host_avail * (1.0 - reserve_ratio)):
        stream_default = True
    if profile.get("stream_load") is not None and os.environ.get("ADAM_STREAM_LOAD") is None:
        stream_default = bool(profile["stream_load"])
    stream_load = _env_flag("ADAM_STREAM_LOAD", stream_default)

    chunk_env = os.environ.get("ADAM_STREAM_CHUNK_MB")
    if chunk_env is not None:
        stream_chunk_mb = max(8, min(256, int(chunk_env)))
    else:
        if host_avail > 0:
            host_chunk = max(8, min(256, host_avail // (64 * 1024 * 1024)))
        else:
            host_chunk = 32 if unified else 64
        if pool_plan:
            pool_chunk = max(8, min(256, max(1, pool_plan.get("hot_mb", host_chunk) // 2)))
        else:
            pool_chunk = host_chunk
        if unified:
            stream_chunk_mb = max(8, min(64, min(host_chunk, pool_chunk)))
        else:
            stream_chunk_mb = max(16, min(256, min(host_chunk, max(pool_chunk, 64))))
        if profile.get("stream_chunk_mb") is not None:
            stream_chunk_mb = min(stream_chunk_mb, int(profile["stream_chunk_mb"]))

    return {
        "profile": profile.get("name", "default"),
        "reserve_ratio": reserve_ratio,
        "host": host,
        "device": device,
        "usable_device_bytes": usable_device,
        "gpu_est": gpu_est,
        "pool_plan": pool_plan,
        "stream_load": bool(stream_load),
        "stream_chunk_mb": int(stream_chunk_mb),
        "kv_cap": int(kv_cap),
        "gpu_approx_rerank": approx_rerank,
        "gpu_fused_rows_per_group": int(rows_per_group),
        "trace_decode": trace_decode,
    }


def print_runtime_plan(plan):
    device = plan.get("device", {})
    host = plan.get("host", {})
    gpu_est = plan.get("gpu_est", {})
    pool_plan = plan.get("pool_plan", {})
    reserve_pct = int(round(plan.get("reserve_ratio", 0.10) * 100))
    profile = plan.get("profile", "default")

    if device:
        heap = device.get("heap_bytes", 0)
        free_budget = device.get("free_budget_bytes", device.get("budget_bytes", 0))
        print(f"{GREEN}GPU memory:{RESET} {device.get('device_type_name', 'gpu')} "
              f"heap={_mb_str(heap)} free_budget={_mb_str(free_budget)} "
              f"usable={_mb_str(plan.get('usable_device_bytes', 0))} reserve={reserve_pct}%")
    if host:
        print(f"{GREEN}Host RAM:{RESET} total={_mb_str(host.get('total_bytes', 0))} "
              f"available={_mb_str(host.get('available_bytes', 0))}")
    if gpu_est:
        print(f"{GREEN}Model GPU est:{RESET} total={_mb_str(gpu_est.get('total_bytes', 0))} "
              f"workspace={_mb_str(gpu_est.get('workspace_bytes', 0))} "
              f"kv={_mb_str(gpu_est.get('kv_bytes', 0))}")
    if pool_plan:
        print(f"{GREEN}GPU pools:{RESET} hot={pool_plan.get('hot_mb', 0)}MB "
              f"cold={pool_plan.get('cold_mb', 0)}MB")
    print(f"{GREEN}Runtime:{RESET} profile={profile} kv_cap={plan.get('kv_cap', 0)} "
          f"stream={'on' if plan.get('stream_load') else 'off'} "
          f"chunk={plan.get('stream_chunk_mb', 0)}MB "
          f"sampler={'approx_rerank' if plan.get('gpu_approx_rerank') else 'exact_fused_topk'} "
          f"rows={plan.get('gpu_fused_rows_per_group', FAST_LM_ROWS_PER_GROUP)} "
          f"trace={'on' if plan.get('trace_decode') else 'off'}")


def init_gpu_backend(adamah_mod, runtime_plan=None):
    """Initialize ADAMAH with adaptive pools and conservative fallback retries."""
    attempts = []
    hot_env = os.environ.get("ADAMAH_CACHE_MB")
    cold_env = os.environ.get("ADAMAH_COLD_CACHE_MB")
    if hot_env or cold_env:
        hot = int(hot_env) if hot_env else None
        cold = int(cold_env) if cold_env else None
        attempts.append((hot, cold))
    elif runtime_plan and runtime_plan.get("pool_plan"):
        hot = int(runtime_plan["pool_plan"].get("hot_mb", 0))
        cold = int(runtime_plan["pool_plan"].get("cold_mb", 0))
        attempts.append((hot, cold))
        attempts.extend([
            (max(16, hot // 2), max(8, cold // 2)),
            (max(16, hot // 4), max(8, cold // 4)),
        ])
    attempts.extend([
        (None, None),
        (512, 256),
        (256, 128),
        (128, 64),
        (64, 32),
        (32, 16),
        (16, 8),
    ])

    seen = set()
    last_err = None
    for hot_mb, cold_mb in attempts:
        key = (hot_mb, cold_mb)
        if key in seen:
            continue
        seen.add(key)
        try:
            if hot_mb is None and cold_mb is None:
                gpu = adamah_mod.init()
                print(f"{GREEN}GPU pools:{RESET} auto")
            else:
                gpu = adamah_mod.init(cache_mb=hot_mb, cold_cache_mb=cold_mb)
                print(f"{GREEN}GPU pools:{RESET} hot={hot_mb}MB cold={cold_mb}MB")
            return gpu
        except RuntimeError as e:
            last_err = e
            continue
    raise RuntimeError(f"GPU init failed after fallback attempts: {last_err}")

def load_model(model_path):
    """Load GGUF model with ADAMAH GPU backend via ADAM."""
    ensure_runtime()
    from adam.loaders.gguf import GGUFLoader
    from adam.tokenizers.gguf_tok import GGUFTokenizer
    from adam.models.engine import ADAMEngine, ModelConfig, GenerationConfig
    import adamah as adamah_mod

    print(f"\n{CYAN}Loading {os.path.basename(model_path)}...{RESET}")
    loader = GGUFLoader(model_path, keep_tensors=False, keep_raw_blocks=False)
    loader.load(verbose=True)

    # Auto-detect config from GGUF metadata (works for any architecture)
    cfg = ModelConfig.from_gguf_metadata(loader.metadata, verbose=True)
    runtime_plan = build_runtime_plan(adamah_mod, loader, cfg, ADAMEngine)
    print_runtime_plan(runtime_plan)

    if runtime_plan["stream_load"]:
        print(f"{GREEN}GGUF mode:{RESET} streamed upload, chunk={runtime_plan['stream_chunk_mb']}MB")
    else:
        loader.keep_tensors = True
        loader.keep_raw_blocks = True
        loader.materialize(verbose=True)
        print(f"{GREEN}GGUF mode:{RESET} eager host load")

    # Tokenizer (architecture-agnostic, reads from GGUF vocab)
    tokenizer = GGUFTokenizer(
        vocab=loader.get_tokenizer_vocab(),
        scores=loader.get_tokenizer_scores(),
        bos_id=loader.get_bos_token_id(),
        eos_id=loader.get_eos_token_id(),
        token_types=loader.metadata.get('tokenizer.ggml.token_type', []),
        add_space_prefix=loader.metadata.get('tokenizer.ggml.add_space_prefix', True),
    )

    # Init GPU
    print(f"{CYAN}Initializing GPU...{RESET}")
    gpu = init_gpu_backend(adamah_mod, runtime_plan)

    # Build engine
    engine = ADAMEngine(
        gpu, cfg, loader.tensors,
        raw_blocks=loader.raw_blocks,
        tensor_types=loader.tensor_types,
        tensor_shapes=loader.tensor_shapes,
        tensor_loader=(loader if runtime_plan["stream_load"] else None),
        stream_chunk_mb=runtime_plan["stream_chunk_mb"],
        kv_cap=runtime_plan["kv_cap"],
        adamah_mod=adamah_mod,
        verbose=True,
        production_mode=True,
        gpu_fused_topk=True,
        gpu_fused_rows_per_group=runtime_plan["gpu_fused_rows_per_group"],
        gpu_approx_rerank=runtime_plan["gpu_approx_rerank"],
        gpu_tied_lm_head=True,
    )
    engine._runtime_profile = runtime_plan["profile"]
    engine._trace_decode = bool(runtime_plan.get("trace_decode", False))
    sampler_name = ("gpu_approx_rerank" if runtime_plan["gpu_approx_rerank"]
                    else "exact gpu_fused_topk")
    print(f"{GREEN}Chat fast path:{RESET} {sampler_name}, "
          f"rows_per_group={runtime_plan['gpu_fused_rows_per_group']}, "
          f"trace={'on' if runtime_plan.get('trace_decode') else 'off'}, production_mode=on")

    return engine, tokenizer, cfg, GenerationConfig

def print_help():
    print(f"""
{BOLD}Commands:{RESET}
  {CYAN}/help{RESET}    — Show this help
  {CYAN}/info{RESET}    — Show model info
  {CYAN}/reset{RESET}   — Clear chat history and KV cache
  {CYAN}/clear{RESET}   — Alias for /reset
  {CYAN}/compact{RESET} — Roll history into one carry-over summary
  {CYAN}/reason N{RESET} — Set reasoning refinement cycles (e.g. /reason 3, /reason off)
  {CYAN}/reasonshow on|off{RESET} — Show reasoning passes in dim text
  {CYAN}/trace on|off{RESET} — Show per-token decode trace summary
  {CYAN}/temp N{RESET}  — Set temperature (e.g. /temp 0.7)
  {CYAN}/topk N{RESET}  — Set top-k (e.g. /topk 40)
  {CYAN}/topp P{RESET}  — Set top-p (e.g. /topp 0.95)
  {CYAN}/repeat N{RESET} — Set repeat penalty (e.g. /repeat 1.1)
  {CYAN}/seed N{RESET}  — Set RNG seed
  {CYAN}/max N{RESET}   — Set max tokens (e.g. /max 128)
  {CYAN}/quit{RESET}    — Exit
""")

def _token_text(tokenizer, token_id: int) -> str:
    vocab = getattr(tokenizer, "_vocab", None)
    if vocab and 0 <= token_id < len(vocab):
        return vocab[token_id]
    return ""


def _render_gguf_chat_template(chat_template: str, tokenizer, messages) -> str | None:
    try:
        import jinja2
    except Exception:
        return None

    env = jinja2.Environment(
        undefined=jinja2.StrictUndefined,
        trim_blocks=False,
        lstrip_blocks=False,
    )
    env.globals["raise_exception"] = (
        lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
    )
    try:
        return env.from_string(chat_template).render(
            messages=messages,
            add_generation_prompt=True,
            bos_token=_token_text(tokenizer, tokenizer.bos_id),
            eos_token=_token_text(tokenizer, tokenizer.eos_id),
        )
    except Exception:
        return None


def apply_chat_template(text: str, arch: str, chat_template: str | None = None,
                        tokenizer=None, system_prompt: str | None = None) -> str:
    """Wrap a user message in the model's expected prompt format."""
    if chat_template and tokenizer is not None:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        rendered = _render_gguf_chat_template(chat_template, tokenizer, messages)
        if rendered is not None:
            return rendered

    a = arch.lower()
    if a in ('gemma', 'gemma2', 'gemma3'):
        return f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
    elif a in ('llama', 'llama2', 'llama3'):
        # Prefer a user-only fallback. Most modern llama-family instruct models
        # ship an explicit GGUF chat template, and forcing a system prompt here
        # degrades smaller variants such as TinyLlama.
        if system_prompt:
            return f"<|system|>\n{system_prompt}</s>\n<|user|>\n{text}</s>\n<|assistant|>\n"
        return f"<|user|>\n{text}</s>\n<|assistant|>\n"
    elif a in ('mistral', 'mixtral'):
        return f"[INST] {text} [/INST]"
    elif a in ('qwen2', 'qwen'):
        return (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
    elif a in ('phi3', 'phi'):
        return f"<|user|>\n{text}<|end|>\n<|assistant|>\n"
    else:
        # Fallback: generic instruct format that works for many base models
        return f"### Instruction:\n{text}\n\n### Response:\n"


def render_chat_messages(messages, arch: str, tokenizer,
                         chat_template: str | None = None,
                         system_prompt: str | None = None) -> str:
    """Render a full conversation history into the model prompt format."""
    convo = list(messages)
    if system_prompt:
        has_system = bool(convo and convo[0].get("role") == "system")
        if not has_system:
            convo = [{"role": "system", "content": system_prompt}] + convo

    if chat_template and tokenizer is not None:
        rendered = _render_gguf_chat_template(chat_template, tokenizer, convo)
        if rendered is not None:
            return rendered

    a = arch.lower()
    if a in ('gemma', 'gemma2', 'gemma3'):
        parts = []
        for msg in convo:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            parts.append(f"<start_of_turn>{role}\n{msg['content']}<end_of_turn>\n")
        parts.append("<start_of_turn>model\n")
        return "".join(parts)
    if a in ('llama', 'llama2', 'llama3'):
        parts = []
        for msg in convo:
            if msg["role"] == "system":
                parts.append(f"<|system|>\n{msg['content']}</s>\n")
            elif msg["role"] == "assistant":
                parts.append(f"<|assistant|>\n{msg['content']}</s>\n")
            else:
                parts.append(f"<|user|>\n{msg['content']}</s>\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)
    if a in ('mistral', 'mixtral'):
        parts = []
        pending_user = None
        for msg in convo:
            if msg["role"] == "system":
                pending_user = f"{msg['content']}\n\n"
            elif msg["role"] == "user":
                pending_user = (pending_user or "") + msg["content"]
            elif msg["role"] == "assistant" and pending_user is not None:
                parts.append(f"[INST] {pending_user} [/INST]{msg['content']}")
                pending_user = None
        if pending_user is not None:
            parts.append(f"[INST] {pending_user} [/INST]")
        return "".join(parts)
    if a in ('qwen2', 'qwen'):
        parts = []
        for msg in convo:
            role = msg["role"]
            if role == "assistant":
                role = "assistant"
            parts.append(f"<|im_start|>{role}\n{msg['content']}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)
    if a in ('phi3', 'phi'):
        parts = []
        for msg in convo:
            role = msg["role"]
            if role == "assistant":
                parts.append(f"<|assistant|>\n{msg['content']}<|end|>\n")
            elif role == "system":
                parts.append(f"<|system|>\n{msg['content']}<|end|>\n")
            else:
                parts.append(f"<|user|>\n{msg['content']}<|end|>\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    # Generic fallback: flatten prior turns, end with the next response marker.
    parts = []
    for msg in convo:
        role = msg["role"].capitalize()
        parts.append(f"### {role}:\n{msg['content']}\n\n")
    parts.append("### Assistant:\n")
    return "".join(parts)


def prepare_chat_prompt(text: str, arch: str, tokenizer, chat_template: str | None = None,
                        system_prompt: str | None = None):
    prompt = apply_chat_template(
        text, arch, chat_template=chat_template, tokenizer=tokenizer,
        system_prompt=system_prompt,
    )
    bos_token = _token_text(tokenizer, tokenizer.bos_id)
    add_bos = not (bos_token and prompt.startswith(bos_token))
    return prompt, add_bos


def prepare_chat_messages(messages, arch: str, tokenizer,
                          chat_template: str | None = None,
                          system_prompt: str | None = None):
    prompt = render_chat_messages(
        messages, arch, tokenizer, chat_template=chat_template,
        system_prompt=system_prompt,
    )
    bos_token = _token_text(tokenizer, tokenizer.bos_id)
    add_bos = not (bos_token and prompt.startswith(bos_token))
    return prompt, add_bos

def _trim_messages_to_fit(engine, tokenizer, cfg, messages, gen_cfg, system_prompt=None):
    kv_cap = getattr(engine, "_kv_cap", 0)
    if kv_cap <= 0:
        prompt, add_bos = prepare_chat_messages(
            messages, cfg.arch, tokenizer, chat_template=getattr(cfg, "chat_template", None),
            system_prompt=system_prompt,
        )
        tokens = tokenizer.encode(prompt, add_bos=add_bos)
        return list(messages), prompt, add_bos, tokens, 0

    trimmed = list(messages)
    dropped = 0
    while True:
        prompt, add_bos = prepare_chat_messages(
            trimmed, cfg.arch, tokenizer, chat_template=getattr(cfg, "chat_template", None),
            system_prompt=system_prompt,
        )
        tokens = tokenizer.encode(prompt, add_bos=add_bos)
        if len(tokens) + gen_cfg.max_tokens <= kv_cap:
            return trimmed, prompt, add_bos, tokens, dropped
        base = 1 if trimmed and trimmed[0].get("role") == "system" else 0
        if len(trimmed) <= base + 1:
            return trimmed, prompt, add_bos, tokens, dropped
        drop_n = 2 if len(trimmed) >= base + 2 and trimmed[base + 1].get("role") == "assistant" else 1
        trimmed = trimmed[:base] + trimmed[base + drop_n:]
        dropped += drop_n


def _summarize_history(engine, tokenizer, cfg, GenConfig, history, seed=None):
    if not history:
        return None

    summary_request = (
        "Summarize the conversation above into a compact carry-over context for the next chat window. "
        "Keep only durable facts: goals, preferences, constraints, decisions, and unresolved tasks. "
        "Omit filler and transient phrasing. Return plain text only."
    )
    summary_messages = list(history) + [{"role": "user", "content": summary_request}]
    summary_cfg = GenConfig(
        max_tokens=192,
        temperature=0.2,
        top_k=20,
        top_p=0.9,
        seed=seed,
        eos_token_ids=(tokenizer.eos_id,),
    )
    prompt, add_bos = prepare_chat_messages(
        summary_messages,
        cfg.arch,
        tokenizer,
        chat_template=getattr(cfg, "chat_template", None),
        system_prompt=(
            "You compress chat history into short reusable memory. "
            "Be faithful and concise."
        ),
    )
    tokens = tokenizer.encode(prompt, add_bos=add_bos)
    kv_cap = getattr(engine, "_kv_cap", 0)
    if kv_cap > 0 and len(tokens) + summary_cfg.max_tokens > kv_cap:
        transcript_text = _history_to_transcript(history)
        if not transcript_text:
            return None
        prompt, add_bos = prepare_chat_messages(
            [{
                "role": "user",
                "content": (
                    "Summarize the conversation below into a compact carry-over context for the next chat window. "
                    "Keep only durable facts: goals, preferences, constraints, decisions, and unresolved tasks. "
                    "Omit filler and transient phrasing. Return plain text only.\n\n"
                    f"{transcript_text}"
                ),
            }],
            cfg.arch,
            tokenizer,
            chat_template=getattr(cfg, "chat_template", None),
            system_prompt=(
                "You compress chat history into short reusable memory. "
                "Be faithful and concise."
            ),
        )
        tokens = tokenizer.encode(prompt, add_bos=add_bos)
    out_tokens, _ = engine.generate(tokens, summary_cfg, stream=False)
    text = tokenizer.decode(out_tokens).strip()
    return text or None


def _build_reasoning_draft(engine, tokenizer, cfg, GenConfig, history,
                           user_prompt, cycles, seed=None):
    if cycles <= 0:
        return None, []

    context_text = _build_reasoning_context_text(history, user_prompt)
    draft = None
    trace = []
    for i in range(int(cycles)):
        summary_req = (
            f"Conversation transcript:\n{context_text}\n\n"
            "Write a compact working summary for answering the latest user message. "
            "Keep only the task, constraints, useful prior context, and likely answer direction. "
            "Do not answer the user directly. Return notes only."
        )
        if draft:
            summary_req += f"\n\nPrevious refined notes:\n{draft}"
        summary = _run_reasoning_stage(
            engine, tokenizer, cfg, GenConfig, summary_req, seed=seed, max_tokens=128
        )
        if not summary:
            break
        trace.append((f"cycle {i + 1}: summary", summary))

        critique_req = (
            f"User request:\n{user_prompt.strip()}\n\n"
            f"Working summary:\n{summary}\n\n"
            "Critique these notes. Find weak assumptions, missing constraints, unclear steps, "
            "or likely failure points. Return concise critique notes only."
        )
        critique = _run_reasoning_stage(
            engine, tokenizer, cfg, GenConfig, critique_req, seed=seed, max_tokens=96
        )
        if not critique:
            break
        trace.append((f"cycle {i + 1}: critique", critique))

        refine_req = (
            f"User request:\n{user_prompt.strip()}\n\n"
            f"Working summary:\n{summary}\n\n"
            f"Critique:\n{critique}\n\n"
            "Produce improved working notes for the final answer. "
            "Keep them concise, concrete, and immediately usable. Return notes only."
        )
        draft = _run_reasoning_stage(
            engine, tokenizer, cfg, GenConfig, refine_req, seed=seed, max_tokens=128
        )
        if not draft:
            break
        trace.append((f"cycle {i + 1}: refine", draft))
    return draft or None, trace


def chat_loop(engine, tokenizer, cfg, GenConfig):
    """Main interactive chat loop."""
    # Collect EOS token IDs: the main EOS plus any type-3 "end-of-turn" specials.
    # Gemma3 ends turns with <end_of_turn> (token 106) before <eos> (token 1).
    # Stopping on both prevents generating past the intended response boundary.
    eos_ids = {tokenizer.eos_id}
    for tok_str, tok_id in tokenizer._specials.items():
        if 'end' in tok_str.lower() or 'eot' in tok_str.lower():
            eos_ids.add(tok_id)
    gen_cfg = GenConfig(max_tokens=128, temperature=0.7, top_k=40, top_p=0.95, seed=42,
                        eos_token_ids=tuple(eos_ids))
    history = []
    reasoning_cycles = _reasoning_cycles_from_env()
    reasoning_trace = _reasoning_trace_from_env()
    decode_trace = bool(getattr(engine, "_trace_decode", False))
    context_compaction = bool(_env_flag("ADAM_CONTEXT_COMPACTION", True))

    print(f"\n{GREEN}Ready!{RESET} Type {CYAN}/help{RESET} for commands.\n")
    print(f"{DIM}Template: {cfg.arch}{RESET}\n")

    while True:
        try:
            prompt = input(f"{BOLD}{GREEN}> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Bye!{RESET}")
            break

        if not prompt:
            continue

        # Commands
        if prompt.startswith('/'):
            cmd = prompt.split()
            if cmd[0] in ('/quit', '/exit', '/q'):
                print(f"{DIM}Bye!{RESET}")
                break
            elif cmd[0] == '/help':
                print_help()
            elif cmd[0] == '/info':
                print(f"{BOLD}Model:{RESET} {cfg.arch}")
                print(f"  Layers: {cfg.n_layer}, Embedding: {cfg.n_embd}")
                print(f"  Heads: {cfg.n_head}, KV Heads: {cfg.n_head_kv}")
                print(f"  FF: {cfg.n_ff}, Vocab: {cfg.n_vocab}")
                print(f"  Context: {cfg.n_ctx}, RoPE base: {cfg.rope_base_global}")
                print(f"  Conversation messages: {len(history)}")
                print(
                    f"  Compacted carry-over: "
                    f"{'yes' if history and _is_compaction_seed_message(history[0]) else 'no'}"
                )
                print(f"  Reasoning cycles: {reasoning_cycles}")
                print(f"  Reasoning trace: {'on' if reasoning_trace else 'off'}")
                print(f"  Decode trace: {'on' if decode_trace else 'off'}")
                print(
                    f"  Context compaction: "
                    f"{'on' if _auto_compaction_enabled(context_compaction, reasoning_cycles) else 'off'}"
                )
                print(f"{BOLD}Gen config:{RESET} temp={gen_cfg.temperature}, max={gen_cfg.max_tokens}, top_k={gen_cfg.top_k}, top_p={gen_cfg.top_p}, repeat={gen_cfg.repeat_penalty}, seed={gen_cfg.seed}")
                mode = ("gpu_approx_rerank" if getattr(engine, "_gpu_approx_rerank", False)
                        else "gpu_fused_topk")
                rows = getattr(engine, "_gpu_fused_rows_per_group", FAST_LM_ROWS_PER_GROUP)
                profile = getattr(engine, "_runtime_profile", "default")
                print(f"{BOLD}GPU path:{RESET} {mode} rows_per_group={rows} profile={profile}")
            elif cmd[0] in ('/reset', '/clear'):
                history.clear()
                engine.reset()
                print(f"{YELLOW}Chat history and KV cache cleared.{RESET}")
            elif cmd[0] == '/compact':
                if not history:
                    print(f"{YELLOW}Nothing to compact yet.{RESET}")
                else:
                    summary = _summarize_history(engine, tokenizer, cfg, GenConfig, history, seed=gen_cfg.seed)
                    if summary:
                        history = [_build_compaction_seed_message(summary)]
                        engine.reset()
                        print(f"{GREEN}Conversation compacted into carry-over summary.{RESET}")
                    else:
                        print(f"{RED}Compaction failed to produce a summary.{RESET}")
            elif cmd[0] == '/reason' and len(cmd) > 1:
                value = cmd[1].lower()
                if value == 'off':
                    reasoning_cycles = 0
                    print(f"{GREEN}Reasoning cycles → 0{RESET}")
                elif value == 'on':
                    reasoning_cycles = max(reasoning_cycles, 2)
                    print(f"{GREEN}Reasoning cycles → {reasoning_cycles}{RESET}")
                else:
                    try:
                        reasoning_cycles = max(0, int(value))
                        print(f"{GREEN}Reasoning cycles → {reasoning_cycles}{RESET}")
                    except ValueError:
                        print(f"{RED}Usage: /reason 3  or  /reason off{RESET}")
            elif cmd[0] == '/reasonshow' and len(cmd) > 1:
                value = cmd[1].lower()
                if value in ('on', '1', 'true'):
                    reasoning_trace = True
                    print(f"{GREEN}Reasoning trace → on{RESET}")
                elif value in ('off', '0', 'false'):
                    reasoning_trace = False
                    print(f"{GREEN}Reasoning trace → off{RESET}")
                else:
                    print(f"{RED}Usage: /reasonshow on  or  /reasonshow off{RESET}")
            elif cmd[0] == '/trace' and len(cmd) > 1:
                value = cmd[1].lower()
                if value in ('on', '1', 'true'):
                    decode_trace = True
                    print(f"{GREEN}Decode trace → on{RESET}")
                elif value in ('off', '0', 'false'):
                    decode_trace = False
                    print(f"{GREEN}Decode trace → off{RESET}")
                else:
                    print(f"{RED}Usage: /trace on  or  /trace off{RESET}")
            elif cmd[0] == '/temp' and len(cmd) > 1:
                try:
                    gen_cfg.temperature = float(cmd[1])
                    print(f"{GREEN}Temperature → {gen_cfg.temperature}{RESET}")
                except ValueError:
                    print(f"{RED}Usage: /temp 0.7{RESET}")
            elif cmd[0] == '/topk' and len(cmd) > 1:
                try:
                    gen_cfg.top_k = max(1, int(cmd[1]))
                    print(f"{GREEN}Top-k → {gen_cfg.top_k}{RESET}")
                except ValueError:
                    print(f"{RED}Usage: /topk 40{RESET}")
            elif cmd[0] == '/topp' and len(cmd) > 1:
                try:
                    gen_cfg.top_p = min(1.0, max(0.0, float(cmd[1])))
                    print(f"{GREEN}Top-p → {gen_cfg.top_p}{RESET}")
                except ValueError:
                    print(f"{RED}Usage: /topp 0.95{RESET}")
            elif cmd[0] == '/repeat' and len(cmd) > 1:
                try:
                    gen_cfg.repeat_penalty = max(1.0, float(cmd[1]))
                    print(f"{GREEN}Repeat penalty → {gen_cfg.repeat_penalty}{RESET}")
                except ValueError:
                    print(f"{RED}Usage: /repeat 1.1{RESET}")
            elif cmd[0] == '/seed' and len(cmd) > 1:
                try:
                    gen_cfg.seed = int(cmd[1])
                    print(f"{GREEN}Seed → {gen_cfg.seed}{RESET}")
                except ValueError:
                    print(f"{RED}Usage: /seed 42{RESET}")
            elif cmd[0] == '/max' and len(cmd) > 1:
                try:
                    gen_cfg.max_tokens = int(cmd[1])
                    print(f"{GREEN}Max tokens → {gen_cfg.max_tokens}{RESET}")
                except ValueError:
                    print(f"{RED}Usage: /max 128{RESET}")
            else:
                print(f"{RED}Unknown command. Type /help{RESET}")
            continue

        system_prompt = None
        pending_messages = history + [{"role": "user", "content": prompt}]
        pending_messages, templated, add_bos, tokens, dropped = _trim_messages_to_fit(
            engine, tokenizer, cfg, pending_messages, gen_cfg, system_prompt=system_prompt
        )
        if dropped and _auto_compaction_enabled(context_compaction, reasoning_cycles) and history:
            try:
                summary = _summarize_history(engine, tokenizer, cfg, GenConfig, history, seed=gen_cfg.seed)
            except Exception:
                summary = None
            if summary:
                history = [_build_compaction_seed_message(summary)]
                engine.reset()
                pending_messages = history + [{"role": "user", "content": prompt}]
                pending_messages, templated, add_bos, tokens, dropped = _trim_messages_to_fit(
                    engine, tokenizer, cfg, pending_messages, gen_cfg, system_prompt=system_prompt
                )
                print(f"{DIM}[context compacted into carry-over summary]{RESET}")
        if _reasoning_enabled(reasoning_cycles):
            draft, reasoning_steps = _build_reasoning_draft(
                engine, tokenizer, cfg, GenConfig, history,
                prompt, reasoning_cycles, seed=gen_cfg.seed,
            )
            if draft:
                if reasoning_trace:
                    for idx, (stage, notes) in enumerate(reasoning_steps, 1):
                        print(f"{DIM}[reason {idx}/{reasoning_cycles} | {stage}]{RESET}")
                        print(f"{DIM}{notes}{RESET}")
                system_prompt = _build_session_system_prompt(draft)
                pending_messages, templated, add_bos, tokens, dropped_after_reason = _trim_messages_to_fit(
                    engine, tokenizer, cfg, pending_messages, gen_cfg, system_prompt=system_prompt
                )
                if dropped_after_reason:
                    print(f"{DIM}[reasoning fit trimmed {dropped_after_reason} older message(s)]{RESET}")
                print(f"{DIM}[reasoning refined in {reasoning_cycles} cycle(s)]{RESET}")
        if dropped:
            print(f"{DIM}[context trimmed: dropped {dropped} older message(s) to fit kv_cap]{RESET}")
        t0 = time.perf_counter()

        try:
            out_tokens, stats = engine.generate(tokens, gen_cfg, stream=False, trace_decode=decode_trace)
            elapsed = time.perf_counter() - t0
            text = tokenizer.decode(out_tokens)
            history = pending_messages + [{"role": "assistant", "content": text}]

            print(f"\n{MAGENTA}{text}{RESET}")
            n_gen = stats.get('n_gen', len(out_tokens))
            total_tps = n_gen / elapsed if elapsed > 0 else 0
            decode_tps = stats.get('decode_tps', 0.0)
            prefill_s = stats.get('prefill_s', 0.0)
            decode_s = stats.get('decode_s', 0.0)
            mode = stats.get('sampling_mode', 'unknown')
            print(
                f"{DIM}[{n_gen} tokens | decode {decode_tps:.1f} tok/s | "
                f"total {total_tps:.1f} tok/s | prefill {prefill_s:.2f}s | "
                f"decode {decode_s:.2f}s | {mode}]{RESET}\n"
            )
            trace_summary = stats.get('trace_summary') or {}
            if decode_trace and trace_summary:
                print(
                    f"{DIM}[trace avg/token: total {trace_summary.get('step_ms_avg', 0.0):.2f}ms | "
                    f"sample {trace_summary.get('sample_ms_avg', 0.0):.2f}ms | "
                    f"forward {trace_summary.get('forward_ms_avg', 0.0):.2f}ms | "
                    f"lm_head {trace_summary.get('lm_head_ms_avg', 0.0):.2f}ms | "
                    f"attn {trace_summary.get('attn_ms_avg', 0.0):.2f}ms | "
                    f"ffn {trace_summary.get('ffn_ms_avg', 0.0):.2f}ms]{RESET}\n"
                )
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")
            import traceback
            traceback.print_exc()

def main():
    ensure_runtime()
    box("ADAMAH Chat — Universal LLM Inference")
    print(f"{DIM}Pure Vulkan • Zero CUDA • Any GGUF model{RESET}\n")

    models = scan_models()
    model_path = select_model(models)
    engine, tokenizer, cfg, GenConfig = load_model(model_path)
    chat_loop(engine, tokenizer, cfg, GenConfig)

if __name__ == '__main__':
    main()
