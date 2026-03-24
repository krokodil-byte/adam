# Codex Task: B9 (fused gateup+act shader) + B10 (barrier count diagnostic)
# Project: ADAM — Vulkan LLM inference, Gemma3-1B, RTX 3070
# Date: 2026-03-21
# Read AGENT_COLLAB.md for full context before starting.

---

## Background

Current state after B8:
- 339 ops in decode plan (26 layers × 13 ops + final norm)
- ~60 tok/s on RTX 3070 — dispatch fusion eliminated 78 dispatches but gain was flat
- Post-mortem: level_batched already suppressed barriers inside same fusion level;
  inter-level barriers dominate. Each inter-level barrier costs ~88μs (unvalidated estimate).
- Each layer still has a standalone DECODE_OP_OP2 (swiGLU: silu(gate)*up) AFTER
  the fused_gateup dispatch. This is an inter-level barrier we can eliminate.

---

## B9: New shader — map_fused_gateup_act_t_xq8

### Goal
Extend `map_fused_gateup_t_xq8.comp` to also compute `silu(gate) * up` inline,
writing a single activated output instead of two separate gate/up buffers.
This eliminates one dispatch+barrier per layer × 26 = 26 barriers ≈ 2.3ms.

### Shader spec: map_fused_gateup_act_t_xq8.comp

Base file: copy `shaders/src/f32/map_fused_gateup_t_xq8.comp` and modify as follows:

**Bindings (7 total)** — drop the LocsCGate/LocsCUp pair, add single LocsCOut:
```glsl
layout(set=0, binding=0) buffer ActBuffer  { float map_act[]; };
layout(set=0, binding=1) buffer WtBuffer   { uint  map_wt[];  };
layout(set=0, binding=2) buffer LocsA      { uint  locs_a[];  };
layout(set=0, binding=3) buffer LocsBGate  { uint  locs_bg[]; };
layout(set=0, binding=4) buffer LocsBUp    { uint  locs_bu[]; };
layout(set=0, binding=5) buffer LocsCOut   { uint  locs_co[]; };
layout(set=0, binding=6) buffer QParams    { float qparams[]; };
```

**Push constants (16 bytes)**:
```glsl
layout(push_constant) uniform PushConstants {
    uint K;        // input dim (n_embd = 1152 for Gemma3-1B)
    uint N_ff;     // output dim = gate rows = up rows (n_ff = 6912)
    uint group_size;
    uint rows_per_wg;
} pc;
```

**Dispatch shape**: X = ceil(N_ff / rows_per_wg) workgroups, Y = 1 (no projection selector).
The shader runs once and produces both gate and up dot products per output row.

**Changes to main()**:
- The shader needs to accumulate TWO dot products per output row simultaneously:
  `sum_gate[r]` using weights from `locs_bg` and `sum_up[r]` using weights from `locs_bu`.
- At the reduction/write stage, compute `silu(sum_gate[r]) * sum_up[r]` and write to `locs_co[0] + row`.

**silu function** (add near top of shader):
```glsl
float silu(float x) {
    return x / (1.0 + exp(-x));
}
```

**Accumulation loop changes**: Each thread accumulates two accumulators per row:
```glsl
float sums_gate[MAX_ROWS_PER_WG];
float sums_up[MAX_ROWS_PER_WG];
// ... initialize to 0 ...
// Inside tile loop, for each row r:
//   sums_gate[r] += a_tile[i] * load_q8(b_gate_base + tile + i);
//   sums_up[r]   += a_tile[i] * load_q8(b_up_base   + tile + i);
```

**Shared memory**: `partial` array must hold BOTH gate and up partials.
Use `partial[2 * MAX_ROWS_PER_WG * WG_SIZE]` and offset:
- gate partial at `[r * WG_SIZE + lid]`
- up   partial at `[(MAX_ROWS_PER_WG + r) * WG_SIZE + lid]`

**Write stage** (lid == 0):
```glsl
if (lid == 0u) {
    for (uint r = ...) {
        float gate_val = partial[r * WG_SIZE];
        float up_val   = partial[(MAX_ROWS_PER_WG + r) * WG_SIZE];
        map_act[locs_co[0] + row] = silu(gate_val) * up_val;
    }
}
```

### C implementation

In `adamah.c`:

1. New pipeline: `VkPipeline fused_gateup_act_t_xq8_pipe` in the pipeline struct.

2. `create_pipeline` call in both init blocks:
   - shader: `"map_fused_gateup_act_t_xq8.spv"`
   - n_bindings: 7
   - push_size: 16

3. New fusion op code macro: `#define FUSE_OP_FUSED_GATEUP_ACT_T_XQ8  24`

4. New decode opcode: `#define ADAMAH_DECODE_OP_FUSED_GATEUP_ACT_T_XQ8  30`

5. Public function signature (Claude needs this exactly):
```c
int map_fused_gateup_act_t_xq8_dev(
    AdamahCtx *ctx,
    uint32_t map_act,     // F32 workspace map id
    uint32_t map_wt,      // Q8 weight map id
    uint32_t locs_a,      // locs handle for input
    uint32_t locs_bg,     // locs handle for gate weight rows
    uint32_t locs_bu,     // locs handle for up weight rows
    uint32_t locs_co,     // locs handle for activated output
    uint32_t K,           // input dim
    uint32_t N_ff         // output dim (gate rows = up rows)
);
```

6. The exec_internal and fusion_queue functions follow the same pattern as the existing
   `map_fused_gateup_t_xq8_dev` family. Mirror them exactly, replacing:
   - pipeline: use `fused_gateup_act_t_xq8_pipe`
   - n_bindings: 7 (drop locs_cg/locs_cu, add single locs_co)
   - push bytes: K, N_ff, group_size, rows_per_wg (16 bytes, not 20)

7. Decode step case for `ADAMAH_DECODE_OP_FUSED_GATEUP_ACT_T_XQ8`:
   - Fields: map_id (F32 ws), map_id2 (Q8 wt), h0=locs_a, h1=locs_bg, h2=locs_bu, h3=locs_co
   - u0=K, u1=N_ff
   - Call: `map_fused_gateup_act_t_xq8_dev(ctx, op.map_id, op.map_id2, op.h0, op.h1, op.h2, op.h3, op.u0, op.u1)`

### __init__.py additions
```python
DECODE_OP_FUSED_GATEUP_ACT_T_XQ8 = 30

# In _setup_ctypes try/except:
lib.map_fused_gateup_act_t_xq8_dev.argtypes = [
    c_void_p,  # ctx
    c_uint32, c_uint32,  # map_act, map_wt
    c_uint32, c_uint32, c_uint32, c_uint32,  # locs_a, locs_bg, locs_bu, locs_co
    c_uint32, c_uint32,  # K, N_ff
]
lib.map_fused_gateup_act_t_xq8_dev.restype = c_int
self._has_map_fused_gateup_act_t_xq8_dev = True
```

### SPV placement
Compile to `shaders/f32/map_fused_gateup_act_t_xq8.spv` then copy flat:
```
cp shaders/f32/map_fused_gateup_act_t_xq8.spv shaders/map_fused_gateup_act_t_xq8.spv
```
(load_spv() in adamah.c resolves SHADER_PATH/name without subdirectory.)

### Validation
`python tests/diagnostics/diag_inference.py gemma3-1b.gguf` → 8 PASS, "2 + 2 = 4"
Then `diag_chat_perf.py` (no --trace), post Turn 2 decode_tps here.
Expected: 313 ops in decode plan, ~77 tok/s.

---

## B10: Barrier count diagnostic

### Goal
Count actual `vkCmdPipelineBarrier2` (or `vkCmdPipelineBarrier`) calls per batch.
The 88μs/barrier estimate is based on aggregate timing only; actual barrier count is unknown.
This tells us whether further dispatch fusion will help at all.

### adamah.c changes
1. Add a counter field to the ctx struct (or as a static/global per-ctx): `uint32_t last_barrier_count`
2. Reset to 0 at `batch_begin()` (or at start of `batch_end()` recording)
3. Increment atomically (or non-atomically, single-threaded is fine) wherever `vkCmdPipelineBarrier2`
   or `vkCmdPipelineBarrier` is called inside a batch
4. After `vkEndCommandBuffer` / `vkQueueSubmit` in `batch_end()`, store final count in ctx

5. New public function:
```c
uint32_t adamah_get_last_barrier_count(AdamahCtx *ctx);
```
Returns `ctx->last_barrier_count`.

### __init__.py additions
```python
# In _setup_ctypes try/except:
lib.adamah_get_last_barrier_count.argtypes = [c_void_p]
lib.adamah_get_last_barrier_count.restype  = c_uint32
self._has_get_last_barrier_count = True

# Wrapper method:
def get_last_barrier_count(self):
    if not getattr(self, '_has_get_last_barrier_count', False):
        return None
    return self._lib.adamah_get_last_barrier_count(self._ctx)
```

### Usage by Claude (after Codex delivers)
Claude will add a `--barrier-count` flag to `diag_chat_perf.py` that prints the barrier count
after one full forward pass (post-warmup). No engine.py changes needed.

---

## Coordination protocol

1. Codex completes B9 shader + C API + __init__.py + SPV
2. Codex posts exact C function signature for `map_fused_gateup_act_t_xq8_dev` in AGENT_COLLAB.md findings log
3. Claude wires engine.py `_build_decode_plan()` (B9 Python side)
4. Codex completes B10 barrier counter + posts `get_last_barrier_count()` in AGENT_COLLAB.md
5. Claude adds `--barrier-count` to diag_chat_perf.py

Both B9 and B10 can be developed in parallel (they touch different code paths).
