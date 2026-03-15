#!/usr/bin/env python3
"""
ADAM Inference Diagnostic
=========================
Four-level sanity check for the ADAM/Gemma3 inference pipeline.

  Check 1 — Scale fix: verify scale_h locs buffer has enough elements and
             the workspace float value is 1/sqrt(head_dim).
  Check 2 — Tokenizer: print token IDs for the chat template and flag
             any spurious ▁ (token 236743) between turns.
  Check 3 — Forward pass: run one token (BOS) through the full model,
             print top-10 predicted tokens. A healthy model should predict
             recognizable words, not <unused> tokens.
  Check 4 — Brief generation: 8 greedy tokens for "What is 2+2?".

Run from the ADAM project root:
    python tests/test_inference_debug.py [path/to/model.gguf]
"""
import os, sys, glob, time
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [ROOT, os.path.join(ROOT, "adamah-MAIN")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from adam.loaders.gguf import GGUFLoader
from adam.tokenizers.gguf_tok import GGUFTokenizer
from adam.models.engine import ADAMEngine, ModelConfig, GenerationConfig
import adamah as A

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"
BOLD = "\033[1m"
RST  = "\033[0m"


def _is_broadcom_like_profile() -> bool:
    profile = str(
        os.environ.get("ADAM_RUNTIME_PROFILE")
        or os.environ.get("ADAMAH_SHADER_PROFILE")
        or ""
    ).strip().lower()
    return profile.startswith("broadcom_v3dv")


def _pool_attempts_for_device() -> list[tuple[int, int]]:
    try:
        device = A.probe_device()
    except Exception:
        device = {}
    unified = bool(device.get("is_unified_memory"))
    if unified or _is_broadcom_like_profile():
        return [(64, 32), (48, 24), (32, 16)]
    return [(1024, 512), (512, 256), (256, 128)]


def _should_stream_loader() -> bool:
    try:
        device = A.probe_device()
    except Exception:
        device = {}
    return bool(device.get("is_unified_memory")) or _is_broadcom_like_profile()

# ── Model path ───────────────────────────────────────────────────────────────
def find_model(argv):
    if len(argv) > 1 and argv[1].endswith('.gguf'):
        return argv[1]
    hits = glob.glob(os.path.join(ROOT, "**/*.gguf"), recursive=True)
    if not hits:
        sys.exit("No .gguf file found. Pass path as argument.")
    hits.sort(key=lambda p: os.path.basename(p).lower())   # alphabetical; gemma < tinyllama
    print(f"  Auto-selected: {os.path.basename(hits[0])} "
          f"(pass a path to override, e.g. python tests/test_inference_debug.py model.gguf)")
    return hits[0]

# ── Load ─────────────────────────────────────────────────────────────────────
def load_everything(model_path):
    print(f"\n{BOLD}Loading {os.path.basename(model_path)}...{RST}")
    stream_loader = _should_stream_loader()
    loader = GGUFLoader(
        model_path,
        keep_tensors=not stream_loader,
        keep_raw_blocks=not stream_loader,
    )
    loader.load(verbose=False)

    cfg = ModelConfig.from_gguf_metadata(loader.metadata, verbose=False)

    add_space_prefix = loader.metadata.get('tokenizer.ggml.add_space_prefix', True)
    tokenizer = GGUFTokenizer(
        vocab=loader.get_tokenizer_vocab(),
        scores=loader.get_tokenizer_scores(),
        bos_id=loader.get_bos_token_id(),
        eos_id=loader.get_eos_token_id(),
        token_types=loader.metadata.get('tokenizer.ggml.token_type', []),
        add_space_prefix=add_space_prefix,
    )

    print(f"  arch={cfg.arch}  n_head={cfg.n_head}  head_dim={cfg.head_dim}  "
          f"n_head_kv={cfg.n_head_kv}  n_vocab={cfg.n_vocab}")
    print(f"  attn_softcap={cfg.attn_softcap}  final_softcap={cfg.final_softcap}")
    print(f"  add_space_prefix={add_space_prefix}  "
          f"bos={tokenizer.bos_id}  eos={tokenizer.eos_id}")

    # Use a model-aware pool recommendation first; the raw no-arg recommendation
    # can overshoot badly because it does not account for the model's persistent
    # GPU footprint and can leave native state poisoned after a failed init.
    gpu = None
    attempts = []
    try:
        est = ADAMEngine.estimate_persistent_gpu_bytes(
            cfg,
            loader.tensor_shapes,
            loader.tensor_types,
            kv_cap=ADAMEngine.KV_CAP_DEFAULT,
            gpu_tied_lm_head=True,
            gpu_approx_rerank=False,
            gpu_fused_rows_per_group=ADAMEngine.SAMPLE_FUSED_ROWS_PER_GROUP_NON_GREEDY,
        )
        plan = A.recommend_pool_sizes(working_set_bytes=est["total_bytes"])
        attempts.append((int(plan["hot_mb"]), int(plan["cold_mb"])))
    except Exception:
        pass
    attempts.extend(_pool_attempts_for_device())
    seen = set()
    for hot_mb, cold_mb in attempts:
        key = (int(hot_mb), int(cold_mb))
        if key in seen:
            continue
        seen.add(key)
        try:
            gpu = A.init(cache_mb=hot_mb, cold_cache_mb=cold_mb)
            print(f"  [GPU pool: hot={hot_mb}MB cold={cold_mb}MB]")
            break
        except (RuntimeError, OSError) as exc:
            print(f"  [GPU pool failed: hot={hot_mb}MB cold={cold_mb}MB] {exc}")
    if gpu is None:
        raise RuntimeError("GPU init failed at all pool sizes — is VRAM full?")
    engine_kwargs = {
        "raw_blocks": loader.raw_blocks,
        "tensor_types": loader.tensor_types,
        "adamah_mod": A,
        "verbose": False,
    }
    if stream_loader:
        engine_kwargs.update({
            "tensor_shapes": loader.tensor_shapes,
            "tensor_loader": loader,
            "runtime_profile": "broadcom_v3dv",
            "stream_chunk_mb": 8,
            "kv_cap": 256,
            "gpu_fused_rows_per_group": 256,
            "fusion_scheduler_mode": "level_batched",
            "direct_kv_cache_write": True,
        })
        print("  [Loader mode: streamed tensors, broadcom-safe engine config]")
    engine = ADAMEngine(gpu, cfg, loader.tensors, **engine_kwargs)
    return engine, tokenizer, cfg, gpu

# ── CHECK 1: Attention scale broadcast ───────────────────────────────────────
def check_scale(engine, cfg, gpu):
    print(f"\n{BOLD}=== Check 1: Attention scale broadcast ==={RST}")
    ws_id = engine.ws_map_id
    scale_h, scale_sz, pos_scale = engine._ws_slots['scale']
    kv_cap = engine._kv_cap
    expected_sz = cfg.n_head * kv_cap

    ok_sz = scale_sz == expected_sz
    print(f"  scale_h locs count : {scale_sz:6d}  need >= {expected_sz:6d}  "
          + (PASS if ok_sz else FAIL))

    # Read the actual scale float from the workspace (using map gather)
    val = gpu.gather(ws_id, np.array([pos_scale], dtype=np.uint32)).view(np.float32)[0]
    expected_val = 1.0 / np.sqrt(cfg.head_dim)
    ok_val = abs(val - expected_val) < 1e-4
    print(f"  scale workspace val: {val:.6f}  expected {expected_val:.6f}  "
          + (PASS if ok_val else FAIL))

    # Read first few locs from the scale_h device buffer — should all == pos_scale
    try:
        first_locs = gpu.download_dev(scale_h, min(4, scale_sz), dtype=np.uint32)
        all_ok = np.all(first_locs == pos_scale)
        print(f"  scale locs[0..{min(4,scale_sz)-1}]  : {first_locs.tolist()}  "
              f"(all == {pos_scale}?)  " + (PASS if all_ok else FAIL))
    except Exception as e:
        print(f"  scale locs download: {WARN} skipped ({e})")
        all_ok = True  # not critical if download not supported

    # --- Softcap slots ---
    cap_ok = True
    for cap_name, cap_val in [('attn_softcap', cfg.attn_softcap),
                               ('final_softcap', cfg.final_softcap)]:
        if cap_val > 0:
            if cap_name in engine._ws_slots:
                _, _, p = engine._ws_slots[cap_name]
                v = gpu.gather(ws_id, np.array([p], dtype=np.uint32)).view(np.float32)[0]
                ok = abs(v - cap_val) < 1e-3
                print(f"  {cap_name:16s}: slot OK, val={v:.1f} expected={cap_val:.1f}  "
                      + (PASS if ok else FAIL))
                cap_ok = cap_ok and ok
            else:
                print(f"  {cap_name:16s}: slot MISSING from ws_slots  {FAIL}")
                cap_ok = False
        else:
            print(f"  {cap_name:16s}: disabled (0.0)")

    return ok_sz and ok_val and cap_ok

# ── CHECK 2: Tokenizer ────────────────────────────────────────────────────────
def check_tokenizer(tokenizer, cfg):
    print(f"\n{BOLD}=== Check 2: Tokenizer (chat template) ==={RST}")
    from adamah_chat import prepare_chat_prompt

    template_text, add_bos = prepare_chat_prompt(
        "Hello", cfg.arch, tokenizer, chat_template=getattr(cfg, "chat_template", None)
    )
    ids = tokenizer.encode(template_text, add_bos=add_bos)

    print(f"  Template: {template_text!r}")
    print(f"  Token sequence ({len(ids)} tokens):")

    vocab = tokenizer._vocab if hasattr(tokenizer, '_vocab') and tokenizer._vocab else None
    problems = []
    prev_str = ""
    for i, tid in enumerate(ids):
        tok_str = vocab[tid] if vocab and 0 <= tid < len(vocab) else f"<id={tid}>"
        marker = ""
        if tok_str == '▁' and prev_str and '<end_of_turn>' in prev_str:
            marker = f"  ← {WARN} spurious ▁ (Bug 2: add_space_prefix not honored)"
            problems.append(i)
        print(f"    [{i:2d}] {tid:7d}  {tok_str!r}{marker}")
        prev_str = tok_str

    if problems:
        print(f"  {FAIL} Spurious ▁ at position(s) {problems}. "
              "Fix: pass add_space_prefix=False to GGUFTokenizer.")
    else:
        print(f"  {PASS} No spurious ▁ tokens between turns.")

    return len(problems) == 0

# ── CHECK 3: Forward pass (BOS → top-10 logits) ───────────────────────────────
def check_forward(engine, tokenizer, cfg, gpu):
    print(f"\n{BOLD}=== Check 3: Forward pass (BOS token, pos=0) ==={RST}")

    engine.reset()
    t0 = time.perf_counter()
    logits = engine._forward(tokenizer.bos_id, 0)
    dt = time.perf_counter() - t0
    print(f"  _forward took {dt:.2f}s")

    top10 = np.argsort(logits)[-10:][::-1]
    vocab = tokenizer._vocab if hasattr(tokenizer, '_vocab') and tokenizer._vocab else None
    print("  Top-10 predictions after BOS:")

    unused_count = 0
    for rank, tid in enumerate(top10):
        tok_str = vocab[tid] if vocab and 0 <= tid < len(vocab) else f"<id={tid}>"
        is_unused = '<unused' in tok_str
        if is_unused:
            unused_count += 1
        flag = f"  ← {FAIL} <unused>!" if is_unused else ""
        print(f"    [{rank+1:2d}] {tid:7d}  {tok_str!r:35s}  logit={logits[tid]:.2f}{flag}")

    lo, hi = logits.min(), logits.max()
    print(f"  Logit range: [{lo:.2f}, {hi:.2f}]")
    if hi > 500:
        print(f"  {FAIL} Logits extremely large — attention likely still unscaled!")
        ok_range = False
    elif hi > 50:
        print(f"  {WARN} Logits somewhat large (max={hi:.1f}) — check scale.")
        ok_range = True
    else:
        print(f"  {PASS} Logit range looks sane.")
        ok_range = True

    ok = unused_count == 0 and ok_range
    print("  " + (PASS + " Top-10 tokens look reasonable"
                  if ok else FAIL + f" {unused_count}/10 top tokens are <unused>"))
    return ok

# ── CHECK 4: Brief generation ─────────────────────────────────────────────────
def check_topk_sampling(engine, tokenizer, cfg):
    print(f"\n{BOLD}=== Check 4b: GPU top-k shortlist ==={RST}")
    from adamah_chat import prepare_chat_prompt

    prompt = "Say hello in one short sentence."
    template_text, add_bos = prepare_chat_prompt(
        prompt, cfg.arch, tokenizer, chat_template=getattr(cfg, "chat_template", None)
    )
    tokens = tokenizer.encode(template_text, add_bos=add_bos)

    sample_cfg = GenerationConfig(
        max_tokens=1,
        temperature=0.8,
        top_k=16,
        top_p=0.95,
        repeat_penalty=1.1,
        seed=123,
        eos_token_ids=(tokenizer.eos_id,),
    )
    print(f"  Prompt: {prompt!r}  ({len(tokens)} input tokens)")

    engine.reset()
    logits = None
    for i, t in enumerate(tokens):
        logits = engine._forward(t, i)

    np.random.seed(sample_cfg.seed)
    cpu_tok = engine._sample(logits, sample_cfg, list(tokens))
    np.random.seed(sample_cfg.seed)
    gpu_tok = engine._sample_gpu_topk(sample_cfg, list(tokens))

    vocab = tokenizer._vocab if hasattr(tokenizer, '_vocab') and tokenizer._vocab else None
    def ts(tid): return vocab[tid] if vocab and 0 <= tid < len(vocab) else f"<id={tid}>"

    can_gpu_topk = engine._can_gpu_topk_sample(sample_cfg)
    print(f"  GPU top-k available: {can_gpu_topk}")
    print(f"  CPU sample token: {cpu_tok}  {ts(cpu_tok)!r}")
    print(f"  GPU sample token: {gpu_tok}  {ts(gpu_tok)!r}")

    ok = can_gpu_topk and cpu_tok == gpu_tok
    print("  " + (PASS + " GPU shortlist matches CPU sampling"
                  if ok else FAIL + " GPU shortlist diverges from CPU sampling"))
    return ok

def check_generation(engine, tokenizer, cfg, n_tokens=8):
    print(f"\n{BOLD}=== Check 4: Greedy generation ({n_tokens} tokens) ==={RST}")
    from adamah_chat import prepare_chat_prompt

    prompt = "What is 2 + 2?"
    template_text, add_bos = prepare_chat_prompt(
        prompt, cfg.arch, tokenizer, chat_template=getattr(cfg, "chat_template", None)
    )
    tokens = tokenizer.encode(template_text, add_bos=add_bos)

    print(f"  Prompt: {prompt!r}  ({len(tokens)} input tokens)")

    eos_ids = {tokenizer.eos_id}
    for tok_str, tok_id in tokenizer._specials.items():
        if 'end' in tok_str.lower() or 'eot' in tok_str.lower():
            eos_ids.add(tok_id)
    gen_cfg = GenerationConfig(max_tokens=n_tokens, temperature=0.0, top_k=1,
                               repeat_penalty=1.0, seed=0,
                               eos_token_ids=tuple(eos_ids))
    # Prefill: run every token through the model.
    # Print top-5 logits and hidden state norm at pos 0, 1, 2, and final pos.
    # A healthy model should predict sensible tokens, not all <unused> at max logit.
    engine.reset()
    vocab = tokenizer._vocab if hasattr(tokenizer, '_vocab') and tokenizer._vocab else None
    hid_locs = np.arange(engine._ws_slots['hidden'][2],
                         engine._ws_slots['hidden'][2] + cfg.n_embd, dtype=np.uint32)
    # Norm diagnostics for Gemma3 internal weights
    output_norm_w = engine.tensors.get('output_norm.weight')
    if output_norm_w is not None:
        print(f"  output_norm.weight norm={np.linalg.norm(output_norm_w):.2f} "
              f"min={output_norm_w.min():.4f} max={output_norm_w.max():.4f}")
    # Check post_attention_norm and post_ffw_norm for first 2 layers
    for L in (0, 1):
        for wname in (f'blk.{L}.post_attention_norm.weight', f'blk.{L}.post_ffw_norm.weight'):
            w = engine.tensors.get(wname)
            if w is not None:
                print(f"  {wname}: norm={np.linalg.norm(w):.4f} max={abs(w).max():.4f}")

    # Bisect layers: find where the norm explosion starts.
    # Temporarily patch n_layer to run a subset of layers.
    true_n_layer = engine.cfg.n_layer
    print("  [Bisect] norm after N layers (BOS@pos=0, then BOS@pos=1):")
    for n_layers in (1, 2, 4, 8, 16, true_n_layer):
        engine.cfg.n_layer = n_layers
        engine.reset()
        engine._forward(tokenizer.bos_id, 0)   # pos=0
        logits_b = engine._forward(tokenizer.bos_id, 1)  # pos=1
        hid_b = engine.gpu.gather(engine.ws_map_id, hid_locs).view(np.float32)
        top1 = int(np.argmax(logits_b))
        tn = vocab[top1] if vocab and 0 <= top1 < len(vocab) else f"<id={top1}>"
        print(f"    n_layers={n_layers:2d}: |hid|={np.linalg.norm(hid_b):.2f}  "
              f"top1={tn!r}={logits_b[top1]:.2f}")
    engine.cfg.n_layer = true_n_layer

    print("  Per-token logit scan:")
    for i, t in enumerate(tokens):
        logits_pre = engine._forward(t, i)
        tok_str = vocab[t] if vocab and 0 <= t < len(vocab) else f"<id={t}>"
        # Gather hidden state for norm (outside batch is OK — engine already ended batch)
        hid = engine.gpu.gather(engine.ws_map_id, hid_locs).view(np.float32)
        hid_norm = float(np.linalg.norm(hid))
        top5 = np.argsort(logits_pre)[-5:][::-1]
        top5_str = ", ".join(
            f"{vocab[tid]!r}={logits_pre[tid]:.1f}" if vocab and 0 <= tid < len(vocab)
            else f"<id={tid}>={logits_pre[tid]:.1f}"
            for tid in top5
        )
        lo, hi = logits_pre.min(), logits_pre.max()
        if i < 3 or i == len(tokens) - 1:
            print(f"    pos={i:2d} in={tok_str!r:20s} |hid|={hid_norm:7.2f}  "
                  f"range=[{lo:.1f},{hi:.1f}]  top5: {top5_str}")
        elif i == 3:
            print(f"    ... ({len(tokens)-4} tokens skipped) ...")

    # Generation
    engine.reset()
    t0 = time.perf_counter()
    out_tokens, stats = engine.generate(tokens, gen_cfg)
    print(f"  Sampling mode: {stats.get('sampling_mode', 'unknown')}")
    dt = time.perf_counter() - t0

    decoded = tokenizer.decode(out_tokens)
    tps = len(out_tokens) / dt if dt > 0 else 0

    print(f"  Output ({len(out_tokens)} tok, {tps:.1f} tok/s): {decoded!r}")
    print(f"  Token IDs: {out_tokens}")

    ref_decoded = getattr(engine, '_last_cpu_gen_text', None)
    if ref_decoded is not None:
        print(f"  CPU reference: {ref_decoded!r}")

    unused_count = sum(
        1 for t in out_tokens
        if vocab and 0 <= t < len(vocab) and '<unused' in vocab[t]
    )
    same_as_cpu = (ref_decoded is None or decoded.rstrip() == ref_decoded.rstrip())
    used_gpu_greedy = stats.get('sampling_mode') in {
        'gpu_argmax', 'gpu_fused_topk', 'gpu_approx_rerank'
    }
    ok = unused_count == 0 and same_as_cpu and used_gpu_greedy
    if unused_count != 0:
        print("  " + FAIL + f" {unused_count}/{len(out_tokens)} tokens are <unused>")
    elif not same_as_cpu:
        print("  " + FAIL + " GPU greedy output diverges from CPU reference")
    elif not used_gpu_greedy:
        print("  " + FAIL + " Greedy decode fell back to CPU sampling")
    else:
        print("  " + PASS + " GPU generation matches CPU reference")
    return ok

# ── CHECK 5: CPU reference comparison ────────────────────────────────────────
def check_cpu_reference(engine, tokenizer, cfg):
    """CPU reference for n_layers=1 at pos=0.

    5a. Trivial identity: attn_out[h] == V[h//gs] for single token (softmax([x])=1.0).
        This is exact in floating point — any deviation means a bug in row_copy or
        V-cache matmul addressing.
    5b. V projection: GPU v slot vs CPU numpy V — only Q4 quantisation error expected (~3-5%).
    5c. Final logits: GPU vs CPU numpy for n_layers=1.
    """
    print(f"\n{BOLD}=== Check 5: CPU reference (n_layers=1, pos=0) ==={RST}")

    c = cfg
    tensors = engine.tensors   # float32 dequantised by GGUF loader
    gemma_norm = c.arch.startswith('gemma')
    eps = c.norm_eps
    gs = max(1, c.n_head // c.n_head_kv)  # GQA group size

    def norm_w(name):
        w = tensors[name].flatten().astype(np.float32)
        return w  # GGUF stores actual weights directly; no +1 offset needed

    def rms_norm_cpu(x, w):
        rms = np.sqrt(np.mean(x ** 2) + eps)
        return (x / rms) * w

    def gelu(x):
        return np.float32(0.5) * x * (np.float32(1.0) +
               np.tanh(np.float32(np.sqrt(2.0 / np.pi)) * (x + np.float32(0.044715) * x ** 3)))

    def silu(x):
        return x * (np.float32(1.0) / (np.float32(1.0) + np.exp(-x.clip(-80, 80))))

    # ---- Run GPU forward with n_layers=1, pos=0 ----
    true_n_layer = engine.cfg.n_layer
    engine.cfg.n_layer = 1
    engine.reset()
    gpu_logits = engine._forward(tokenizer.bos_id, 0)
    engine.cfg.n_layer = true_n_layer

    g = engine.gpu
    ws_id = engine.ws_map_id

    # Gather V from the actual KV cache, not the temporary workspace staging slot.
    kv_step = engine._kv_cap * c.head_dim_kv
    gpu_v = np.concatenate([
        g.gather(
            ws_id,
            np.arange(
                engine._kc_v_base[0] + gi * kv_step,
                engine._kc_v_base[0] + gi * kv_step + c.head_dim_kv,
                dtype=np.uint32,
            ),
        ).view(np.float32)
        for gi in range(c.n_head_kv)
    ], axis=0)

    # Gather attn_out slot (written by weighted-sum matmul, never overwritten)
    _, ao_sz, ao_pos = engine._ws_slots['attn_out']
    ao_locs = np.arange(ao_pos, ao_pos + ao_sz, dtype=np.uint32)
    gpu_ao = g.gather(ws_id, ao_locs).view(np.float32)

    # ---- 5a: Trivial identity: for pos=0, attn_out[h] must equal cached V[h//gs] exactly ----
    # softmax([score]) = [1.0]  →  attn_out[h] = 1.0 * V_cache[h//gs][0] = V[h//gs]
    max_diff = 0.0
    for h in range(c.n_head):
        grp = h // gs
        diff = float(np.max(np.abs(
            gpu_ao[h * c.head_dim_kv : (h + 1) * c.head_dim_kv]
            - gpu_v[grp * c.head_dim_kv : (grp + 1) * c.head_dim_kv]
        )))
        max_diff = max(max_diff, diff)
    ok5a = max_diff < 1e-2
    print(f"  5a. attn_out == V (trivial identity, pos=0):  max_diff={max_diff:.3e}  "
          + (PASS if ok5a else FAIL + " ← attention copy broken!"))
    if not ok5a:
        print(f"      |gpu_v|={np.linalg.norm(gpu_v):.3f}  |gpu_ao|={np.linalg.norm(gpu_ao):.3f}")
        for h in range(min(c.n_head, 4)):
            grp = h // gs
            ao_h = gpu_ao[h * c.head_dim_kv : (h + 1) * c.head_dim_kv]
            v_h  = gpu_v[grp * c.head_dim_kv : (grp + 1) * c.head_dim_kv]
            diff_h = float(np.max(np.abs(ao_h - v_h)))
            print(f"      head {h}: ao={ao_h[:4]}  v={v_h[:4]}  diff={diff_h:.3e}")

    # ---- CPU reference computation (1 layer, pos=0) ----
    emb_t = tensors['token_embd.weight'].astype(np.float32)  # [n_vocab, n_embd]
    hidden = (emb_t[tokenizer.bos_id].copy() if emb_t.shape[0] == c.n_vocab
              else emb_t[:, tokenizer.bos_id].copy())
    if c.emb_scale != 1.0:
        hidden = hidden * np.float32(c.emb_scale)

    p = 'blk.0'

    # Pre-attention norm
    normed = rms_norm_cpu(hidden, norm_w(f'{p}.attn_norm.weight'))

    # QKV projections: tensors are [out, in] — compute normed @ W.T
    W_q = tensors[f'{p}.attn_q.weight'].astype(np.float32)
    W_k = tensors[f'{p}.attn_k.weight'].astype(np.float32)
    W_v = tensors[f'{p}.attn_v.weight'].astype(np.float32)
    q_cpu = normed @ W_q.T   # [q_dim]
    k_cpu = normed @ W_k.T   # [k_dim]
    v_cpu = normed @ W_v.T   # [k_dim]

    # QK norm (Gemma3): shared weight across heads
    qnn = f'{p}.attn_q_norm.weight'
    if qnn in tensors:
        q_h2 = q_cpu.reshape(c.n_head, c.head_dim)
        k_h2 = k_cpu.reshape(c.n_head_kv, c.head_dim_kv)
        qn_w = norm_w(qnn)
        kn_w = norm_w(f'{p}.attn_k_norm.weight')
        for i in range(c.n_head):
            q_h2[i] = rms_norm_cpu(q_h2[i], qn_w)
        for i in range(c.n_head_kv):
            k_h2[i] = rms_norm_cpu(k_h2[i], kn_w)
        q_cpu = q_h2.reshape(-1)
        k_cpu = k_h2.reshape(-1)

    # RoPE: no-op at pos=0 (cos=1, sin=0), skip

    # ---- 5b: V projection CPU vs cached GPU V ----
    v_rmse = float(np.sqrt(np.mean((gpu_v - v_cpu) ** 2)))
    v_std  = float(max(np.std(v_cpu), 1e-8))
    ok5b = v_rmse / v_std < 0.15
    print(f"  5b. V projection CPU vs GPU:          RMSE={v_rmse:.4f}  "
          f"std={v_std:.4f}  rel={v_rmse/v_std:.3f}  "
          + (PASS if ok5b else FAIL))

    # Attention: trivial for single token — softmax([score])=[1.0] → attn_out[h]=V[h//gs]
    v_heads = v_cpu.reshape(c.n_head_kv, c.head_dim_kv)
    attn_out = np.stack([v_heads[h // gs] for h in range(c.n_head)]).reshape(-1)

    # Output projection
    W_o = tensors[f'{p}.attn_output.weight'].astype(np.float32)
    o_proj = attn_out @ W_o.T

    # Post-attention norm (Gemma3)
    pan = f'{p}.post_attention_norm.weight'
    if pan in tensors:
        o_proj = rms_norm_cpu(o_proj, norm_w(pan))
    hidden = hidden + o_proj

    # FFN
    normed2 = rms_norm_cpu(hidden, norm_w(f'{p}.ffn_norm.weight'))
    W_gate = tensors[f'{p}.ffn_gate.weight'].astype(np.float32)
    W_up   = tensors[f'{p}.ffn_up.weight'].astype(np.float32)
    W_down = tensors[f'{p}.ffn_down.weight'].astype(np.float32)
    gate_out = normed2 @ W_gate.T
    up_out   = normed2 @ W_up.T
    gate_out = gelu(gate_out) if c.ffn_act == 'gelu' else silu(gate_out)
    act_out  = gate_out * up_out
    ffn_out  = act_out @ W_down.T

    # Post-FFN norm (Gemma3)
    pfn = f'{p}.post_ffw_norm.weight'
    if pfn in tensors:
        ffn_out = rms_norm_cpu(ffn_out, norm_w(pfn))
    hidden = hidden + ffn_out

    # Final norm
    normed_final = rms_norm_cpu(hidden, norm_w('output_norm.weight'))

    # LM head
    if 'output.weight' in tensors:
        W_lm = tensors['output.weight'].astype(np.float32)
        cpu_logits = normed_final @ W_lm.T
    else:
        W_lm = tensors['token_embd.weight'].astype(np.float32)  # [n_vocab, n_embd]
        cpu_logits = (W_lm @ normed_final if W_lm.shape[0] == c.n_vocab
                      else W_lm.T @ normed_final)

    if c.final_softcap > 0:
        cap = np.float32(c.final_softcap)
        cpu_logits = np.tanh(cpu_logits / cap) * cap

    # ---- 5c: GPU logits vs CPU reference ----
    rmse = float(np.sqrt(np.mean((gpu_logits - cpu_logits) ** 2)))
    std  = float(max(np.std(cpu_logits), 1e-8))
    rel  = rmse / std

    vocab = tokenizer._vocab if hasattr(tokenizer, '_vocab') and tokenizer._vocab else None
    def tok_str(tid): return vocab[tid] if vocab and 0 <= tid < len(vocab) else f"<id={tid}>"

    gpu_top1 = int(np.argmax(gpu_logits))
    cpu_top1 = int(np.argmax(cpu_logits))
    ok5c = rel < 0.15
    print(f"  5c. Logits GPU vs CPU (n_layers=1):   RMSE={rmse:.4f}  "
          f"std={std:.4f}  rel={rel:.3f}  "
          + (PASS if ok5c else FAIL))
    print(f"      GPU top1: {gpu_top1:7d}  {tok_str(gpu_top1)!r:30s}  "
          f"logit={gpu_logits[gpu_top1]:.3f}")
    print(f"      CPU top1: {cpu_top1:7d}  {tok_str(cpu_top1)!r:30s}  "
          f"logit={cpu_logits[cpu_top1]:.3f}")

    # Print top-5 worst logit differences for further diagnosis
    if not ok5c:
        worst = np.argsort(np.abs(gpu_logits - cpu_logits))[-10:][::-1]
        print("      Worst 10 logit differences:")
        for tid in worst:
            print(f"        tok {tid:7d} {tok_str(tid)!r:25s}  "
                  f"gpu={gpu_logits[tid]:.3f}  cpu={cpu_logits[tid]:.3f}  "
                  f"diff={gpu_logits[tid]-cpu_logits[tid]:+.3f}")

    return ok5a and ok5b and ok5c


# ── CHECK 6: Multi-layer CPU reference (pos=0, no RoPE needed) ───────────────
def check_multilayer_cpu(engine, tokenizer, cfg):
    """Check 6: Run BOS@pos=0 through n_layers∈{1,2,4,8} on both GPU and CPU.

    At pos=0, attention is trivial (attn_out == V) so we don't need RoPE.
    Comparing GPU and CPU hidden norms across layers identifies whether:
     - the norm explosion is expected model behaviour → GPU and CPU agree
     - the norm explosion is a GPU computation bug → GPU diverges from CPU
    """
    print(f"\n{BOLD}=== Check 6: Multi-layer CPU vs GPU norms (BOS@pos=0) ==={RST}")

    c = cfg
    tensors = engine.tensors
    gemma_norm = c.arch.startswith('gemma')
    eps = c.norm_eps
    gs = max(1, c.n_head // c.n_head_kv)

    def norm_w(name):
        w = tensors[name].flatten().astype(np.float32)
        return w  # GGUF stores actual weights directly; no +1 offset needed

    def rms_norm_cpu(x, w):
        rms = np.sqrt(np.mean(x ** 2) + eps)
        return (x / rms) * w

    def gelu(x):
        return np.float32(0.5) * x * (np.float32(1.0) +
               np.tanh(np.float32(np.sqrt(2.0/np.pi)) * (x + np.float32(0.044715) * x**3)))

    def silu(x):
        return x * (np.float32(1.0) / (np.float32(1.0) + np.exp(-x.clip(-80, 80))))

    emb_t = tensors['token_embd.weight'].astype(np.float32)
    emb0 = (emb_t[tokenizer.bos_id].copy() if emb_t.shape[0] == c.n_vocab
            else emb_t[:, tokenizer.bos_id].copy())
    if c.emb_scale != 1.0:
        emb0 = emb0 * np.float32(c.emb_scale)

    hid_locs = np.arange(engine._ws_slots['hidden'][2],
                         engine._ws_slots['hidden'][2] + c.n_embd, dtype=np.uint32)

    true_n_layer = engine.cfg.n_layer
    all_ok = True
    cpu_hidden = emb0.copy()

    test_layers = [n for n in (1, 2, 4, 8, 16, true_n_layer) if n <= true_n_layer]
    if test_layers[-1] != true_n_layer:
        test_layers.append(true_n_layer)

    print(f"  {'n':>4}  {'|hid|_gpu':>12}  {'|hid|_cpu':>12}  {'ratio':>7}  "
          f"{'gpu_top1':>12}  {'cpu_top1':>12}  status")
    prev_n = 0
    vocab = tokenizer._vocab if hasattr(tokenizer, '_vocab') and tokenizer._vocab else None
    def ts(tid): return (vocab[tid][:10] if vocab and 0 <= tid < len(vocab)
                         else f"<{tid}>")
    for n in test_layers:
        # ---- CPU: advance from prev_n to n ----
        for L in range(prev_n, n):
            p = f'blk.{L}'
            normed = rms_norm_cpu(cpu_hidden, norm_w(f'{p}.attn_norm.weight'))
            W_q = tensors[f'{p}.attn_q.weight'].astype(np.float32)
            W_k = tensors[f'{p}.attn_k.weight'].astype(np.float32)
            W_v = tensors[f'{p}.attn_v.weight'].astype(np.float32)
            q = normed @ W_q.T
            k = normed @ W_k.T
            v = normed @ W_v.T
            qnn = f'{p}.attn_q_norm.weight'
            if qnn in tensors:
                q_h2 = q.reshape(c.n_head, c.head_dim)
                k_h2 = k.reshape(c.n_head_kv, c.head_dim_kv)
                qn_w = norm_w(qnn)
                kn_w = norm_w(f'{p}.attn_k_norm.weight')
                for i in range(c.n_head):
                    q_h2[i] = rms_norm_cpu(q_h2[i], qn_w)
                for i in range(c.n_head_kv):
                    k_h2[i] = rms_norm_cpu(k_h2[i], kn_w)
                q = q_h2.reshape(-1)
                k = k_h2.reshape(-1)
            # RoPE: no-op at pos=0
            v_heads = v.reshape(c.n_head_kv, c.head_dim_kv)
            attn_out = np.stack([v_heads[h // gs] for h in range(c.n_head)]).reshape(-1)
            W_o = tensors[f'{p}.attn_output.weight'].astype(np.float32)
            o_proj = attn_out @ W_o.T
            pan = f'{p}.post_attention_norm.weight'
            if pan in tensors:
                o_proj = rms_norm_cpu(o_proj, norm_w(pan))
            cpu_hidden = cpu_hidden + o_proj
            normed2 = rms_norm_cpu(cpu_hidden, norm_w(f'{p}.ffn_norm.weight'))
            W_gate = tensors[f'{p}.ffn_gate.weight'].astype(np.float32)
            W_up   = tensors[f'{p}.ffn_up.weight'].astype(np.float32)
            W_down = tensors[f'{p}.ffn_down.weight'].astype(np.float32)
            gate_out = normed2 @ W_gate.T
            up_out   = normed2 @ W_up.T
            gate_out = gelu(gate_out) if c.ffn_act == 'gelu' else silu(gate_out)
            act_out  = gate_out * up_out
            ffn_out  = act_out @ W_down.T
            pfn = f'{p}.post_ffw_norm.weight'
            if pfn in tensors:
                ffn_out = rms_norm_cpu(ffn_out, norm_w(pfn))
            cpu_hidden = cpu_hidden + ffn_out
        prev_n = n

        cpu_norm = float(np.linalg.norm(cpu_hidden))

        # CPU final norm + LM head to get logits
        cpu_normed = rms_norm_cpu(cpu_hidden, norm_w('output_norm.weight'))
        if 'output.weight' in tensors:
            W_lm = tensors['output.weight'].astype(np.float32)
            cpu_logits = cpu_normed @ W_lm.T
        else:
            W_lm = tensors['token_embd.weight'].astype(np.float32)
            cpu_logits = (W_lm @ cpu_normed if W_lm.shape[0] == c.n_vocab
                          else W_lm.T @ cpu_normed)
        if c.final_softcap > 0:
            cap = np.float32(c.final_softcap)
            cpu_logits = np.tanh(cpu_logits / cap) * cap
        cpu_top1 = int(np.argmax(cpu_logits))

        # ---- GPU: run _forward(bos, 0) with n layers ----
        engine.cfg.n_layer = n
        engine.reset()
        gpu_logits = engine._forward(tokenizer.bos_id, 0)
        gpu_hid = engine.gpu.gather(engine.ws_map_id, hid_locs).view(np.float32)
        gpu_norm = float(np.linalg.norm(gpu_hid))
        gpu_top1 = int(np.argmax(gpu_logits))

        ratio = gpu_norm / (cpu_norm + 1e-8)
        ok = 0.85 < ratio < 1.15   # within 15% is fine given Q4 error
        all_ok = all_ok and ok
        print(f"  {n:4d}  {gpu_norm:12.2f}  {cpu_norm:12.2f}  {ratio:7.3f}  "
              f"  {ts(gpu_top1):>12s}  {ts(cpu_top1):>12s}  "
              + (PASS if ok else FAIL))

    engine.cfg.n_layer = true_n_layer

    if all_ok:
        print(f"  {PASS} GPU and CPU hidden norms agree across all layer counts.")
        print(f"  → Norm growth is expected model behaviour; bug is elsewhere.")
    else:
        print(f"  {FAIL} GPU diverges from CPU — there is a per-layer computation bug.")

    return all_ok


# ── CHECK 7: CPU F32 autoregressive generation ────────────────────────────────
def check_cpu_generation(engine, tokenizer, cfg, n_tokens=8):
    """Check 7: Full autoregressive generation using exact F32 dequantised weights.

    Runs the complete 26-layer forward pass on CPU, including RoPE, KV cache,
    and causal attention.  If this produces coherent output (e.g. '4' for
    'What is 2+2?') while the GPU produces garbage, the root cause is the
    double Q4 quantisation (GGUF Q4_K → f32 → engine Q4 re-quantisation).
    """
    print(f"\n{BOLD}=== Check 7: CPU F32 autoregressive generation ==={RST}")
    from adamah_chat import prepare_chat_prompt
    import time as _time

    c = cfg; tensors = engine.tensors
    gemma_norm = c.arch.startswith('gemma'); eps = c.norm_eps
    gs = max(1, c.n_head // c.n_head_kv)

    def norm_w(name):
        w = tensors[name].flatten().astype(np.float32)
        return w  # GGUF stores actual weights directly; no +1 offset needed

    def rms_norm_cpu(x, w):
        rms = np.sqrt(np.mean(x ** 2) + eps); return (x / rms) * w

    def gelu(x):
        return np.float32(0.5)*x*(np.float32(1.0)+np.tanh(
            np.float32(np.sqrt(2.0/np.pi))*(x+np.float32(0.044715)*x**3)))

    def silu(x):
        return x*(np.float32(1.0)/(np.float32(1.0)+np.exp(-x.clip(-80,80))))

    def rope_half(x, pos, freq_base):
        """RoPE with half-half convention: pairs are (x[d], x[d+half]) for d<half."""
        n, d = x.shape; half = d // 2
        freqs = np.float32(1.0) / (freq_base ** (
            np.arange(0, half, dtype=np.float32) * np.float32(2.0) / d))
        theta = np.float32(pos) * freqs
        c_t = np.cos(theta).astype(np.float32); s_t = np.sin(theta).astype(np.float32)
        x0, x1 = x[:, :half], x[:, half:]
        return np.concatenate([x0*c_t - x1*s_t, x0*s_t + x1*c_t], axis=1)

    # KV cache: [n_layer, n_head_kv, kv_cap, head_dim_kv]
    kv_cap = engine._kv_cap
    k_cache = np.zeros((c.n_layer, c.n_head_kv, kv_cap, c.head_dim_kv), np.float32)
    v_cache = np.zeros((c.n_layer, c.n_head_kv, kv_cap, c.head_dim_kv), np.float32)

    # Pre-load and cache weight matrices (avoids repeated astype calls per token)
    print("  Pre-loading weight matrices...", end=' ', flush=True)
    t_pre = _time.perf_counter()
    W = {}
    for name in tensors:
        if tensors[name].ndim == 2:
            W[name] = tensors[name].astype(np.float32)
    t_pre = _time.perf_counter() - t_pre
    print(f"done ({t_pre:.1f}s)")

    def forward_cpu(tok_id, pos):
        seq_len = pos + 1
        emb_t = W.get('token_embd.weight', tensors['token_embd.weight'].astype(np.float32))
        hidden = (emb_t[tok_id].copy() if emb_t.shape[0] == c.n_vocab
                  else emb_t[:, tok_id].copy())
        if c.emb_scale != 1.0: hidden *= np.float32(c.emb_scale)

        for L in range(c.n_layer):
            p = f'blk.{L}'; gl = c.is_global(L)
            freq = c.rope_base_global if gl else c.rope_base_local

            normed = rms_norm_cpu(hidden, norm_w(f'{p}.attn_norm.weight'))

            q = (normed @ W[f'{p}.attn_q.weight'].T).reshape(c.n_head, c.head_dim)
            k = (normed @ W[f'{p}.attn_k.weight'].T).reshape(c.n_head_kv, c.head_dim_kv)
            v = (normed @ W[f'{p}.attn_v.weight'].T).reshape(c.n_head_kv, c.head_dim_kv)

            qnn = f'{p}.attn_q_norm.weight'
            if qnn in tensors:
                qnw = norm_w(qnn); knw = norm_w(f'{p}.attn_k_norm.weight')
                for i in range(c.n_head):   q[i] = rms_norm_cpu(q[i], qnw)
                for i in range(c.n_head_kv): k[i] = rms_norm_cpu(k[i], knw)

            q = rope_half(q, pos, freq)
            k = rope_half(k, pos, freq)

            k_cache[L, :, pos, :] = k
            v_cache[L, :, pos, :] = v

            scale = np.float32(1.0 / np.sqrt(c.head_dim))
            attn_out = np.zeros((c.n_head, c.head_dim_kv), np.float32)
            for h in range(c.n_head):
                kv_h = h // gs
                K_h = k_cache[L, kv_h, :seq_len, :]
                V_h = v_cache[L, kv_h, :seq_len, :]
                sc = q[h] @ K_h.T * scale
                if c.attn_softcap > 0:
                    cap = np.float32(c.attn_softcap)
                    sc = np.tanh(sc / cap) * cap
                sc -= sc.max(); w_ = np.exp(sc); w_ /= w_.sum()
                attn_out[h] = w_ @ V_h

            o_proj = attn_out.reshape(-1) @ W[f'{p}.attn_output.weight'].T
            pan = f'{p}.post_attention_norm.weight'
            if pan in tensors: o_proj = rms_norm_cpu(o_proj, norm_w(pan))
            hidden = hidden + o_proj

            normed2 = rms_norm_cpu(hidden, norm_w(f'{p}.ffn_norm.weight'))
            gate_out = normed2 @ W[f'{p}.ffn_gate.weight'].T
            up_out   = normed2 @ W[f'{p}.ffn_up.weight'].T
            gate_out = gelu(gate_out) if c.ffn_act == 'gelu' else silu(gate_out)
            ffn_out  = (gate_out * up_out) @ W[f'{p}.ffn_down.weight'].T
            pfn = f'{p}.post_ffw_norm.weight'
            if pfn in tensors: ffn_out = rms_norm_cpu(ffn_out, norm_w(pfn))
            hidden = hidden + ffn_out

        normed_final = rms_norm_cpu(hidden, norm_w('output_norm.weight'))
        if 'output.weight' in W:
            logits = normed_final @ W['output.weight'].T
        else:
            lm = W.get('token_embd.weight', tensors['token_embd.weight'].astype(np.float32))
            logits = (lm @ normed_final if lm.shape[0] == c.n_vocab else lm.T @ normed_final)
        if c.final_softcap > 0:
            cap = np.float32(c.final_softcap)
            logits = np.tanh(logits / cap) * cap
        return logits

    prompt = "What is 2 + 2?"
    template_text, add_bos = prepare_chat_prompt(
        prompt, c.arch, tokenizer, chat_template=getattr(c, "chat_template", None)
    )
    tokens = tokenizer.encode(template_text, add_bos=add_bos)
    print(f"  Prompt: {prompt!r}  ({len(tokens)} tokens)")

    vocab = tokenizer._vocab if hasattr(tokenizer, '_vocab') and tokenizer._vocab else None
    def ts(tid): return vocab[tid] if vocab and 0 <= tid < len(vocab) else f"<id={tid}>"

    eos_ids = {tokenizer.eos_id}
    for tok_str, tok_id in tokenizer._specials.items():
        if 'end' in tok_str.lower() or 'eot' in tok_str.lower():
            eos_ids.add(tok_id)

    t0 = _time.perf_counter()
    logits = None
    for i, t in enumerate(tokens):
        logits = forward_cpu(t, i)
    print(f"  Prefill done: {len(tokens)} tokens in {_time.perf_counter()-t0:.1f}s")
    top5 = np.argsort(logits)[-5:][::-1]
    print(f"  After template top5: {', '.join(f'{ts(t)!r}={logits[t]:.2f}' for t in top5)}")

    out_tokens = []; pos = len(tokens)
    for step in range(n_tokens):
        nt = int(np.argmax(logits))
        if nt in eos_ids: break
        out_tokens.append(nt)
        logits = forward_cpu(nt, pos); pos += 1
    dt = _time.perf_counter() - t0

    decoded = tokenizer.decode(out_tokens)
    tps = (len(tokens) + len(out_tokens)) / dt
    print(f"  CPU F32 output ({len(out_tokens)} tok, {tps:.1f} tok/s): {decoded!r}")
    print(f"  Token IDs: {out_tokens}")

    engine._last_cpu_gen_tokens = tuple(out_tokens)
    engine._last_cpu_gen_text = decoded

    ok = len(out_tokens) > 0
    print("  " + (PASS + " CPU F32 generated tokens"
                  if ok else FAIL + " No tokens generated (all EOS?)"))
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    model_path = find_model(sys.argv)
    engine, tokenizer, cfg, gpu = load_everything(model_path)

    results = {}
    results['1_scale']      = check_scale(engine, cfg, gpu)
    results['2_tokenizer']  = check_tokenizer(tokenizer, cfg)
    results['5_cpu_ref']    = check_cpu_reference(engine, tokenizer, cfg)
    results['6_multilayer'] = check_multilayer_cpu(engine, tokenizer, cfg)
    results['7_cpu_gen']    = check_cpu_generation(engine, tokenizer, cfg)
    results['4b_topk']      = check_topk_sampling(engine, tokenizer, cfg)
    results['4_generation'] = check_generation(engine, tokenizer, cfg)
    results['3_forward']    = check_forward(engine, tokenizer, cfg, gpu)

    print(f"\n{BOLD}=== Summary ==={RST}")
    all_ok = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {name:16s}  {status}")
        all_ok = all_ok and ok
    print()
    if all_ok:
        print(f"{PASS} All checks passed — inference pipeline looks healthy.")
    else:
        print(f"{FAIL} Some checks failed — see details above.")
    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())
