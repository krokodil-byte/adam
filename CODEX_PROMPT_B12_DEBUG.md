# Codex Task: B12 shader residual bug — sh_hidden overwritten by normed copy
# Project: ADAM — Vulkan LLM inference, Gemma3-1B, RTX 3070
# Date: 2026-03-22

---

## Root cause (confirmed by Claude analysis)

`map_full_decode_step.comp` has **two identical bugs** — one in the attn block, one in FFN:

```glsl
// ATTN, lines 444-447:
for (uint i = tid; i < n_embd; i += WG_SIZE) {
    sh_hidden[i] = sh_tmp[i];   // ← DESTROYS original hidden state
}
barrier();

// FFN, lines 472-476:
rmsnorm_hidden_to_tmp(0u, n_embd, wa.ffn_norm[L]);
for (uint i = tid; i < n_embd; i += WG_SIZE) {
    sh_hidden[i] = sh_tmp[i];   // ← same bug
}
barrier();
```

After `rmsnorm_hidden_to_tmp`, `sh_tmp[0:n_embd]` holds the **pre-norm output** of
`sh_hidden`. The copy loop overwrites `sh_hidden` with this normed version — the
original residual base is lost.

When `rmsnorm_tmp_add_hidden` later does `sh_hidden[i] += y`, it adds the residual to
the **normed version** (magnitude ≈ 1) instead of the true residual state. Each layer
accumulates nearly zero net change. After 26 layers, `sh_hidden` ≈ original embedding
with `||hid||_gpu ≈ 34` vs `||hid||_cpu ≈ 33427`.

---

## Fix

Save `sh_hidden` before each copy loop, restore it before each residual add.

### Attn block (around lines 441–469)

```glsl
// pre-attn norm: hidden -> tmp[0:n_embd]
rmsnorm_hidden_to_tmp(0u, n_embd, wa.attn_norm[L]);

// ── NEW: save original sh_hidden for residual ──
uint save_attn = attn_out_off + n_q;   // safe region after all attn tmp data
for (uint i = tid; i < n_embd; i += WG_SIZE) {
    sh_tmp[save_attn + i] = sh_hidden[i];
}
barrier();
// ───────────────────────────────────────────────

// copy normed -> sh_hidden so matvec helpers read normed input
for (uint i = tid; i < n_embd; i += WG_SIZE) {
    sh_hidden[i] = sh_tmp[i];
}
barrier();

matvec_from_hidden_to_tmp(q_off, n_embd, n_q, wa.wq[L], wa.qp_wq[L], wa.group_size_attn);
matvec_from_hidden_to_tmp(k_off, n_embd, n_k, wa.wk[L], wa.qp_wk[L], wa.group_size_attn);
matvec_from_hidden_to_tmp(v_off, n_embd, n_v, wa.wv[L], wa.qp_wv[L], wa.group_size_attn);

apply_rope_qk(...);
kv_write_layer(...);
attention_layer(...);
matvec_from_tmp_to_tmp(attn_out_off, 0u, n_q, n_embd, wa.wo[L], wa.qp_wo[L], ...);

// ── NEW: restore original sh_hidden before residual ──
for (uint i = tid; i < n_embd; i += WG_SIZE) {
    sh_hidden[i] = sh_tmp[save_attn + i];
}
barrier();
// ──────────────────────────────────────────────────────

rmsnorm_tmp_add_hidden(0u, n_embd, wa.post_attn_norm[L]);
```

### FFN block (around lines 471–489)

```glsl
rmsnorm_hidden_to_tmp(0u, n_embd, wa.ffn_norm[L]);

// ── NEW: save sh_hidden (= post-attn residual) ──
uint save_ffn = n_ff + n_embd;   // after gateup + down proj output areas
for (uint i = tid; i < n_embd; i += WG_SIZE) {
    sh_tmp[save_ffn + i] = sh_hidden[i];
}
barrier();
// ────────────────────────────────────────────────

for (uint i = tid; i < n_embd; i += WG_SIZE) {
    sh_hidden[i] = sh_tmp[i];
}
barrier();

gateup_geglu_from_hidden(0u, n_embd, n_ff, wa.wg[L], wa.wu[L], wa.qp_wg[L], wa.qp_wu[L], wa.group_size_ffn);
matvec_from_tmp_to_tmp(0u, down_out_off, n_ff, n_embd, wa.wd[L], wa.qp_wd[L], wa.group_size_ffn);

// ── NEW: restore before residual ──
for (uint i = tid; i < n_embd; i += WG_SIZE) {
    sh_hidden[i] = sh_tmp[save_ffn + i];
}
barrier();
// ──────────────────────────────────

rmsnorm_tmp_add_hidden(down_out_off, n_embd, wa.post_ffn_norm[L]);
```

### Update the early-exit guard (line 422)

```glsl
// Old:
if (n_embd == 0u || n_layer == 0u ||
    n_embd > MAX_EMBD ||
    (n_ff + n_embd) > MAX_TMP ||
    (attn_out_off + n_q) > MAX_TMP ||
    head_dim > MAX_HEAD_DIM || head_dim_kv > MAX_HEAD_DIM ||
    pc.kv_cap == 0u || pc.seq_len == 0u || pc.seq_len > pc.kv_cap) {

// New (add two more space checks):
if (n_embd == 0u || n_layer == 0u ||
    n_embd > MAX_EMBD ||
    (n_ff + 2u * n_embd) > MAX_TMP ||          // ← n_ff + n_embd save slot
    (attn_out_off + n_q + n_embd) > MAX_TMP || // ← save_attn + n_embd
    head_dim > MAX_HEAD_DIM || head_dim_kv > MAX_HEAD_DIM ||
    pc.kv_cap == 0u || pc.seq_len == 0u || pc.seq_len > pc.kv_cap) {
```

For Gemma3-1B (n_embd=1152, n_ff≈3456, n_q=2048, n_k=n_v=1024):
- `n_ff + 2*n_embd = 3456 + 2304 = 5760 ≤ 8192` ✓
- `attn_out_off + n_q + n_embd = 4096 + 2048 + 1152 = 7296 ≤ 8192` ✓

---

## Build + test

```bash
cd /c/Users/samus/Documents/ADAM/adamah-MAIN/adamah
# Recompile SPIR-V:
glslc shaders/src/f32/map_full_decode_step.comp -o shaders/f32/map_full_decode_step.spv
cp shaders/f32/map_full_decode_step.spv shaders/map_full_decode_step.spv

# Rebuild DLL:
PATH="/c/mingw64/bin:$PATH" gcc -shared -O2 -march=native \
  -include _shader_path.h \
  -I"/c/VulkanSDK/1.4.341.1/Include" \
  adamah.c -o adamah.dll \
  /c/Users/samus/Documents/ADAM/libvulkan-1.a -lm -Wl,--export-all-symbols
```

Verify DLL ~220-230KB.

Run diagnostics:
```bash
cd /c/Users/samus/Documents/ADAM
PYTHONUTF8=1 PYTHONPATH=/c/Users/samus/Documents/ADAM \
  "/c/Users/samus/AppData/Local/Programs/Python/Python312/python.exe" -X utf8 \
  tests/diagnostics/diag_inference.py gemma3-1b.gguf
```

**Target**: 8/8 PASS, generation "2 + 2 = 4".
Check 6 ratio should be 0.85–1.15 for all layer counts.

Then perf:
```bash
PYTHONUTF8=1 PYTHONPATH=/c/Users/samus/Documents/ADAM \
  "/c/Users/samus/AppData/Local/Programs/Python/Python312/python.exe" -X utf8 \
  tests/diagnostics/diag_chat_perf.py gemma3-1b.gguf
```

**Target**: Turn 2 decode_tps >> 60 tok/s.

---

## Post results in AGENT_COLLAB.md

- Check 6 ratio range
- Generation result
- decode_tps from diag_chat_perf.py Turn 2
