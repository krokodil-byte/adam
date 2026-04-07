# Codex Task: B16 — thread-per-row matvec (V3D barrier fix)
# Project: ADAM — Vulkan LLM inference, Gemma3-1B, Pi 5 V3D + RTX 3070
# Date: 2026-04-04

---

## Root cause

Current `map_full_decode_step.comp` matvec functions use a **reduce-per-row** pattern:
- All WG_SIZE threads collaborate on one row at a time
- Each row requires `reduce_sum()` → 7 barriers (log2 tree) + 1 barrier = **8 barriers per row**
- For 26 layers with N rows per matvec:
  - wg (gate): 6912 rows × 8 = 55296
  - wu (up):   6912 rows × 8 = 55296
  - wd (down): 1152 rows × 8 = 9216
  - wq:        1024 rows × 8 = 8192
  - wk+wv+wo:  ~1664 rows × 8 = 13312
  - **Total per layer: ~141k barriers × 26 = ~3.2M barriers/token**
- At ~160ns/barrier on V3D: **~512ms** — matches observed `core_batch ~530ms` exactly
- DRAM bandwidth floor: 697MB weights / 8GB/s = ~87ms — currently 6× away

Stage-probe confirms: matvec blocks = 511–544ms, attention/norm = 11–15ms.
The bottleneck is entirely barrier overhead, not DRAM bandwidth.

---

## Fix: thread-per-row matvec

Each thread computes **one full output row independently** using `sh_hidden` (already in
shared memory). No cross-thread reduction needed → **1 barrier per matvec call total**.

`sh_hidden` is MAX_EMBD=2048 floats = 8KB, fits in shared memory, all threads read it freely.

### 1. `matvec_from_hidden_to_tmp`

**Before:**
```glsl
void matvec_from_hidden_to_tmp(uint dst_off, uint K, uint N,
                               ADDR_T w_addr, ADDR_T qp_addr, uint group_size) {
    uint tid = gl_LocalInvocationID.x;
    for (uint row = 0u; row < N; ++row) {
        float acc = 0.0;
        uint base = row * K;
        for (uint i = tid; i < K; i += WG_SIZE) {
            float w = dequant_q8(w_addr, qp_addr, base + i, group_size);
            acc += sh_hidden[i] * w;
        }
        float sum = reduce_sum(acc);
        if (tid == 0u) {
            sh_tmp[dst_off + row] = sum;
        }
        barrier();
    }
}
```

**After:**
```glsl
void matvec_from_hidden_to_tmp(uint dst_off, uint K, uint N,
                               ADDR_T w_addr, ADDR_T qp_addr, uint group_size) {
    uint tid = gl_LocalInvocationID.x;
    for (uint row = tid; row < N; row += WG_SIZE) {
        float acc = 0.0;
        uint base = row * K;
        for (uint i = 0u; i < K; ++i) {
            acc += sh_hidden[i] * dequant_q8(w_addr, qp_addr, base + i, group_size);
        }
        sh_tmp[dst_off + row] = acc;
    }
    barrier();
}
```

### 2. `matvec_from_tmp_to_tmp`

**Before:**
```glsl
void matvec_from_tmp_to_tmp(uint src_off, uint dst_off, uint K, uint N,
                            ADDR_T w_addr, ADDR_T qp_addr, uint group_size) {
    uint tid = gl_LocalInvocationID.x;
    for (uint row = 0u; row < N; ++row) {
        float acc = 0.0;
        uint base = row * K;
        for (uint i = tid; i < K; i += WG_SIZE) {
            float w = dequant_q8(w_addr, qp_addr, base + i, group_size);
            acc += sh_tmp[src_off + i] * w;
        }
        float sum = reduce_sum(acc);
        if (tid == 0u) {
            sh_tmp[dst_off + row] = sum;
        }
        barrier();
    }
}
```

**After:**
```glsl
void matvec_from_tmp_to_tmp(uint src_off, uint dst_off, uint K, uint N,
                            ADDR_T w_addr, ADDR_T qp_addr, uint group_size) {
    uint tid = gl_LocalInvocationID.x;
    for (uint row = tid; row < N; row += WG_SIZE) {
        float acc = 0.0;
        uint base = row * K;
        for (uint i = 0u; i < K; ++i) {
            acc += sh_tmp[src_off + i] * dequant_q8(w_addr, qp_addr, base + i, group_size);
        }
        sh_tmp[dst_off + row] = acc;
    }
    barrier();
}
```

### 3. `gateup_geglu_from_hidden`

**Before:**
```glsl
void gateup_geglu_from_hidden(uint dst_off, uint K, uint N_ff, ...) {
    uint tid = gl_LocalInvocationID.x;
    for (uint row = 0u; row < N_ff; ++row) {
        float gate_part = 0.0;
        float up_part = 0.0;
        uint base = row * K;
        for (uint i = tid; i < K; i += WG_SIZE) {
            float wg = dequant_q8(wg_addr, qp_g_addr, base + i, group_size);
            float wu = dequant_q8(wu_addr, qp_u_addr, base + i, group_size);
            float x = sh_hidden[i];
            gate_part += x * wg;
            up_part += x * wu;
        }
        float gate = reduce_sum(gate_part);
        float up = reduce_sum(up_part);
        if (tid == 0u) {
            sh_tmp[dst_off + row] = gelu(gate) * up;
        }
        barrier();
    }
}
```

**After:**
```glsl
void gateup_geglu_from_hidden(uint dst_off, uint K, uint N_ff, ...) {
    uint tid = gl_LocalInvocationID.x;
    for (uint row = tid; row < N_ff; row += WG_SIZE) {
        float gate_acc = 0.0;
        float up_acc = 0.0;
        uint base = row * K;
        for (uint i = 0u; i < K; ++i) {
            float x = sh_hidden[i];
            gate_acc += x * dequant_q8(wg_addr, qp_g_addr, base + i, group_size);
            up_acc   += x * dequant_q8(wu_addr, qp_u_addr, base + i, group_size);
        }
        sh_tmp[dst_off + row] = gelu(gate_acc) * up_acc;
    }
    barrier();
}
```

---

## Safety notes

- `sh_hidden[i]` is read-only during all three matvec calls — no write conflict.
- `sh_tmp[src_off + i]` in `matvec_from_tmp_to_tmp`: `src_off` and `dst_off` must not overlap.
  This is already guaranteed by the caller in `main()` (attn_out_off, down_out_off are distinct).
- `reduce_sum()` and `reduce_max()` functions can be left in place — they are still used by
  `rmsnorm_*` and `attention_layer`. Do NOT remove them.
- `sh_reduce[WG_SIZE]` shared array stays as-is.
- No changes needed to `main()`, `attention_layer`, `rmsnorm_*`, `apply_rope_qk`, `kv_write_layer`.
- No changes to `adamah.c` or Python engine.

---

## Expected impact

| Metric | Before | After |
|--------|--------|-------|
| Shader barriers/token | ~3.2M | ~130 |
| core_batch Pi 5 | ~530ms | ~90–130ms |
| Pi 5 tok/s | 1.24 | **~8–11** |
| RTX 3070 tok/s | ~65 | ≥65 (no regression expected) |

The DRAM bandwidth floor on Pi 5 is ~87ms/token. Thread-per-row gives sequential
per-thread access (row-major) — V3D TMU can prefetch linearly. Should approach floor.

---

## Build + validate

```bash
# On Pi 5:
cd ~/ADAM/adamah-MAIN/adamah
glslc shaders/src/f32/map_full_decode_step.comp -o shaders/f32/map_full_decode_step.spv
cp shaders/f32/map_full_decode_step.spv shaders/map_full_decode_step.spv
gcc -shared -O2 -march=native \
  -include _shader_path.h \
  -I"$VULKAN_SDK/include" \
  adamah.c -o adamah.so \
  -lvulkan -lm
# Verify: ~220KB

PYTHONUTF8=1 PYTHONPATH=~/ADAM python3 -X utf8 \
  tests/diagnostics/diag_inference.py gemma3-1b.gguf
# Must be: 8/8 PASS, "2+2=4"

PYTHONUTF8=1 PYTHONPATH=~/ADAM python3 -X utf8 \
  tests/diagnostics/diag_chat_perf.py gemma3-1b.gguf --max-tokens 8
# Target: Turn2 decode_tps > 8.0

# On RTX 3070 (regression check):
PYTHONUTF8=1 PYTHONPATH=/c/Users/samus/Documents/ADAM \
  "/c/Users/samus/AppData/Local/Programs/Python/Python312/python.exe" -X utf8 \
  tests/diagnostics/diag_inference.py gemma3-1b.gguf
# Must be: 8/8 PASS
```

---

## Post results in AGENT_COLLAB.md

- diag_inference Pi: PASS count
- diag_chat_perf Pi Turn2 tok/s + core_batch ms
- diag_inference RTX3070: PASS count (regression check)
- If still barrier-bound: report new dominant barrier from stage-probe
