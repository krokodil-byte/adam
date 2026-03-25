# Codex Task: B13 — cooperative multi-WG monolithic shader
# Project: ADAM — Vulkan LLM inference, Gemma3-1B, RTX 3070 + Pi 5 V3D
# Date: 2026-03-25

---

## Context

B12 is a monolithic shader that runs all 26 transformer layers in one
`vkCmdDispatch`. It is numerically correct (8/8 diag_inference PASS,
"2+2=4" generation) but runs at 1.33 tok/s vs B6's 61 tok/s because
`vkCmdDispatch(1, 1, 1)` activates only 1 SM on an RTX 3070 (46 SMs).

B13 fixes this by dispatching N_WG workgroups that cooperate via a
global spin-barrier in a dedicated sync buffer. Matmuls (the bottleneck,
~99% of FLOPs) are split across all N_WG workgroups in parallel.
Reductions (RMSNorm, softmax) are handled sequentially by WG 0 while
others spin — reductions are ~0.04% of FLOPs so serializing them is fine.

Target: N_WG = 46 for RTX 3070 → ~40–45× speedup over current B12
→ estimated 50–300 tok/s (vs 1.33 tok/s). For V3D, N_WG = 4–8
(much fewer SMs but dispatch overhead per vkCmdDispatch is very large
on V3DV, so even N_WG=8 should give major gains over B6's 339 dispatches).

---

## Architecture overview

### Key design decisions

1. **Hidden state moves from shared memory to global act[] buffer.**
   `sh_hidden[MAX_EMBD]` is removed as the inter-operation state carrier.
   Instead, a dedicated slot in the ws map (`hidden_slot`) holds the
   authoritative hidden state between operations. Each WG operates on
   its assigned slice `[my_start .. my_end)` of the global vector.

2. **Spin-barrier for inter-WG synchronization.**
   A new binding 5 (`SyncBuf`) holds a flat array of uint32 counters.
   The shader pre-allocates N_BARRIERS = 512 slots (2KB), each used
   exactly once across the full shader execution (monotone barrier ID).
   Before each dispatch, C zeroes the buffer with `vkCmdFillBuffer`.

3. **Matmul parallelism: each WG handles a slice of output rows.**
   For a matvec with N output rows:
   - `rows_per_wg = (N + n_wg - 1) / n_wg`
   - WG i handles rows `[i * rows_per_wg .. min((i+1)*rows_per_wg, N))`
   - Each WG reads the full input vector (n_embd floats) from act[] and
     writes its output rows to act[].
   - All N_WG WGs work in parallel → N_WG× speedup on matmuls.

4. **Reductions handled by WG 0 only.**
   RMSNorm, attention softmax max/sum: only WG 0 executes the reduction,
   reading/writing act[]. Other WGs skip and spin on the arrival counter.
   Reduction cost is negligible (~1152 MACs for RMSNorm vs ~16M for
   all matmuls per layer).

5. **N_WG is a runtime parameter**, stored in `full_decode_state` and
   passed as a push constant. The shader uses `pc.n_wg` everywhere.
   Default: match Vulkan `maxComputeWorkGroupCount` or cap at 64.
   Python side: Claude will wire `n_wg` from GPU caps at register time.

---

## Changes required

### A. `adamah.c`

#### A1. Add `n_wg` and `sync_buf` to `FullDecodeState`

In the struct that holds `full_decode_state` (search for the struct
containing `map_ws_id`, `map_kvcache_id`, `n_layer`, `registered`):

```c
// add to FullDecodeState:
uint32_t n_wg;
VkBuffer sync_buf;
VkDeviceMemory sync_mem;
```

#### A2. Create sync buffer in `adamah_full_decode_register()`

Add `uint32_t n_wg` parameter to the function signature:

```c
int adamah_full_decode_register(uint32_t map_ws_id, uint32_t map_kvcache_id,
                                const FullDecodeWeightAddrs *wa,
                                uint32_t n_layer, uint32_t n_wg);
```

Inside the function, after existing init, allocate the sync buffer:

```c
// 512 barrier counters × 4 bytes = 2048 bytes
VkDeviceSize sync_size = 512 * sizeof(uint32_t);
// Destroy old buffer if re-registering
if (ctx.full_decode_state.sync_buf != VK_NULL_HANDLE) {
    vkDestroyBuffer(ctx.device, ctx.full_decode_state.sync_buf, NULL);
    vkFreeMemory(ctx.device, ctx.full_decode_state.sync_mem, NULL);
    ctx.full_decode_state.sync_buf = VK_NULL_HANDLE;
}
// Allocate: use VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
// Memory: DEVICE_LOCAL preferred; must be visible to compute shader
// Use create_buffer_ex() if available, otherwise vkCreateBuffer/vkAllocateMemory
// ... (follow existing buffer creation pattern in the file)
ctx.full_decode_state.n_wg = n_wg;
```

Also free sync_buf in `adamah_destroy()` / cleanup paths alongside
`full_decode_wa_buf`.

#### A3. Update `adamah_full_decode_step()`

Add binding 5 (sync_buf) to the descriptor set. Current code writes
5 `VkDescriptorBufferInfo` entries (bindings 0–4). Add binding 5:

```c
// binding 5: sync buffer
VkDescriptorBufferInfo buf_infos[6] = {
    {.buffer = m_ws->buf,   .range = VK_WHOLE_SIZE},           // 0: act
    {.buffer = m_kvcache->buf, .range = VK_WHOLE_SIZE},        // 1: kvcache
    {.buffer = ctx.full_decode_wa_buf, .range = sizeof(FullDecodeWeightAddrs)}, // 2: wa
    {.buffer = ctx.hot_pool->buf, .offset = emb_off,  .range = emb_res->size_bytes}, // 3: locs_emb
    {.buffer = ctx.hot_pool->buf, .offset = out_off,  .range = out_res->size_bytes}, // 4: locs_out
    {.buffer = ctx.full_decode_state.sync_buf, .range = VK_WHOLE_SIZE}, // 5: sync
};
VkWriteDescriptorSet writes[6];
for (int i = 0; i < 6; i++) { ... }
vkUpdateDescriptorSets(ctx.device, 6, writes, 0, NULL);
```

Zero the sync buffer before dispatch (so counters start at 0):

```c
vkCmdFillBuffer(ctx.cmd, ctx.full_decode_state.sync_buf,
                0, VK_WHOLE_SIZE, 0u);
// memory barrier: transfer write → compute shader read
VkMemoryBarrier mb = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
};
vkCmdPipelineBarrier(ctx.cmd,
    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    0, 1, &mb, 0, NULL, 0, NULL);
```

Update push constants to include `n_wg` (20 bytes total):

```c
uint32_t push[5] = {pos, seq_len, ctx.full_decode_state.n_layer, kv_cap,
                    ctx.full_decode_state.n_wg};
vkCmdPushConstants(..., 0, 20, push);
```

Change dispatch:

```c
vkCmdDispatch(ctx.cmd, ctx.full_decode_state.n_wg, 1, 1);
```

#### A4. Update `create_pipeline` call for `full_decode_step_pipe`

Currently `create_pipeline(&ctx.full_decode_step_pipe, "map_full_decode_step.spv", 5, ...)`.
Change `5` to `6` to add the sync buffer binding.

#### A5. Update push constant range in pipeline creation

If the pipeline specifies push constant size explicitly (e.g., 16 bytes), update to 20 bytes.
Search for `VkPushConstantRange` near `full_decode_step` pipeline creation and update `.size`.

#### A6. Python binding update (`adamah/__init__.py`)

The Python wrapper for `full_decode_register` needs to accept and pass `n_wg`:

```python
def full_decode_register(self, map_ws_id, map_kvcache_id, wa, n_layer, n_wg=1):
    ...
    ret = self._lib.adamah_full_decode_register(
        ctypes.c_uint32(map_ws_id),
        ctypes.c_uint32(map_kvcache_id),
        ctypes.byref(wa),
        ctypes.c_uint32(n_layer),
        ctypes.c_uint32(n_wg),
    )
```

Claude will update `engine.py` to pass `n_wg` from GPU caps.

---

### B. `shaders/src/f32/map_full_decode_step.comp`

This is the larger change. The shader needs to be restructured so that:
- hidden state lives in global `act[]` between operations
- each WG owns a slice of output rows for matmuls
- reductions (RMSNorm, softmax) are done by WG 0 only
- a global spin-barrier syncs WGs between phases

#### B1. New header / push constants / bindings

```glsl
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#define WG_SIZE 128
#define ADAMAH_FULL_DECODE_MAX_LAYERS 26
#define MAX_HEAD_DIM 512u
#define N_BARRIERS 512u   // pre-allocated barrier slots

layout(local_size_x = WG_SIZE) in;

// ... (buffer_reference types and FullDecodeWeightAddrs struct unchanged) ...

layout(set = 0, binding = 0) coherent buffer ActBuf   { float act[]; };
layout(set = 0, binding = 1) coherent buffer KVCacheBuf { float kv[]; };
layout(set = 0, binding = 2) readonly buffer WeightAddrsBuf { FullDecodeWeightAddrs wa; };
layout(set = 0, binding = 3) readonly buffer LocsEmbIn  { uint locs_emb_in[]; };
layout(set = 0, binding = 4) readonly buffer LocsHiddenOut { uint locs_hidden_out[]; };
layout(set = 0, binding = 5) coherent buffer SyncBuf   { uint s[]; };  // NEW

layout(push_constant) uniform PC {
    uint pos;
    uint seq_len;
    uint n_layer;
    uint kv_cap;
    uint n_wg;     // NEW — number of workgroups dispatched
} pc;

// Shared memory: only intra-WG reduction scratch + local norm/attention state.
// sh_hidden is no longer the inter-op authoritative state — act[] is.
shared float sh_reduce[WG_SIZE];
shared float sh_score_max;
shared float sh_score_sum;
// Small local cache for attention scores (per-head, fits in shmem)
shared float sh_scores[1024]; // kv_cap ≤ 1024 for Gemma3-1B
```

**Remove** `shared float sh_hidden[MAX_EMBD]` and `shared float sh_tmp[MAX_TMP]`.

#### B2. Workgroup identity variables

```glsl
uint wg_id  = gl_WorkGroupID.x;   // 0 .. n_wg-1
uint tid    = gl_LocalInvocationID.x; // 0 .. WG_SIZE-1
bool is_wg0 = (wg_id == 0u);
```

#### B3. Global spin-barrier helper

```glsl
// bid: monotone barrier ID (0..N_BARRIERS-1), unique per sync point in the shader.
// All N_WG workgroups call this at the same logical point.
// WG 0's tid=0 increments the counter then spins; others just spin.
void global_arrive_wait(uint bid) {
    memoryBarrierBuffer();
    barrier();  // ensure all threads in this WG have reached this point
    if (tid == 0u) {
        atomicAdd(s[bid], 1u);
        // spin until all n_wg WGs have arrived
        uint cnt = 0u;
        while (cnt < pc.n_wg) {
            memoryBarrierBuffer();
            cnt = s[bid];
        }
    }
    barrier();
    memoryBarrierBuffer();
}
```

The barrier IDs are assigned as constants throughout the shader. Use
a preprocessor counter or a local `uint next_bid = 0u` at the start of
main() that increments after each `global_arrive_wait(next_bid++)` call.
Since GLSL doesn't have `++` on function args cleanly, use explicit IDs
(count them as you write: bid 0 is the first barrier, bid 1 the second, etc.)
or maintain a shared atomic counter.

Simplest: declare `shared uint sh_bid;` initialized to 0 by WG 0 thread 0
at shader start, then `uint bid = atomicAdd(sh_bid, 1u);` before each
`global_arrive_wait(bid)`. This requires all WGs to call it in lockstep
— which they do since the code is data-parallel.

Actually even simpler: use a fixed sequence (the shader always runs the
same operations in the same order). Pre-number all barriers 0, 1, 2, …
as a literal in the code. There are at most ~10 per layer × 26 = 260.

#### B4. Slice assignment for matmuls

```glsl
// For a matmul producing N output elements:
// WG i handles rows [row_start .. row_start + n_rows)
void get_slice(uint N, out uint row_start, out uint n_rows) {
    uint rpw = (N + pc.n_wg - 1u) / pc.n_wg;
    row_start = wg_id * rpw;
    n_rows    = (row_start < N) ? min(rpw, N - row_start) : 0u;
}
```

#### B5. New matvec helper (reads from act[], writes to act[])

```glsl
// Matvec: out[output_base + row_start .. row_start+n_rows] = W[...] * act[input_base..]
// K = input dimension, n_rows = number of output rows this WG computes
void matvec_act_to_act(uint input_base, uint output_base,
                       uint K, uint row_start, uint n_rows,
                       uint64_t w_addr, uint64_t qp_addr, uint group_size) {
    for (uint r = 0u; r < n_rows; r++) {
        uint row = row_start + r;
        float acc = 0.0;
        uint base = row * K;
        for (uint i = tid; i < K; i += WG_SIZE) {
            acc += act[input_base + i] * dequant_q8(w_addr, qp_addr, base + i, group_size);
        }
        // intra-WG reduction
        sh_reduce[tid] = acc;
        barrier();
        for (uint stride = WG_SIZE >> 1; stride > 0u; stride >>= 1u) {
            if (tid < stride) sh_reduce[tid] += sh_reduce[tid + stride];
            barrier();
        }
        if (tid == 0u) act[output_base + row] = sh_reduce[0];
        barrier();
    }
}
```

#### B6. RMSNorm (WG 0 only, reads/writes act[])

```glsl
// WG 0 computes RMSNorm of act[src_base..src_base+dim] into act[dst_base..dst_base+dim]
// using norm weights at wt_addr. Other WGs skip (controlled externally by `is_wg0`).
void rmsnorm_act_wg0(uint src_base, uint dst_base, uint dim, uint64_t wt_addr) {
    F32Buf w = F32Buf(wt_addr);
    float ss = 0.0;
    for (uint i = tid; i < dim; i += WG_SIZE) {
        float x = act[src_base + i];
        ss += x * x;
    }
    ss = reduce_sum(ss);  // intra-WG reduction (WG 0 only)
    float inv_rms = inversesqrt(ss / float(dim) + 1e-6);
    for (uint i = tid; i < dim; i += WG_SIZE) {
        act[dst_base + i] = act[src_base + i] * inv_rms * w.f[i];
    }
    barrier();
}

// Residual add with post-norm: act[hid] += rmsnorm(act[src])
void rmsnorm_add_act_wg0(uint src_base, uint hid_base, uint dim, uint64_t wt_addr) {
    F32Buf w = F32Buf(wt_addr);
    float ss = 0.0;
    for (uint i = tid; i < dim; i += WG_SIZE) {
        float x = act[src_base + i];
        ss += x * x;
    }
    ss = reduce_sum(ss);
    float inv_rms = inversesqrt(ss / float(dim) + 1e-6);
    for (uint i = tid; i < dim; i += WG_SIZE) {
        act[hid_base + i] += act[src_base + i] * inv_rms * w.f[i];
    }
    barrier();
}
```

#### B7. Slot layout in act[]

The ws map (act buffer) already has dedicated slots for all intermediate
tensors (q, k, v, normed, attn_out, gate, up, ffn_out, hidden, etc.).
These slot base indices are derived from `locs_emb_in[]` and `locs_hidden_out[]`
in the current shader (it reads `locs_emb_in[0]` as the embedding slot,
`locs_hidden_out[0]` as the hidden-out slot).

B13 needs to know the base index of each slot in act[]. These are the
first element of each loc array (already available in the existing shader
via the LocsEmbIn/LocsHiddenOut bindings). Add the full ws slot layout
as push constants OR derive from locs arrays at shader start.

**Simplest approach**: read the base slots from the existing locs buffers
at shader start:

```glsl
// In main(), WG 0 reads and broadcasts slot bases:
shared uint sh_hidden_base;  // act[] offset for hidden state
shared uint sh_normed_base;  // act[] offset for normed state (pre-attn and pre-ffn)
// etc.

// At shader start:
if (is_wg0 && tid == 0u) {
    sh_hidden_base = locs_emb_in[0];   // embedding input = hidden start
    // Other slots are derived from the layout knowledge:
    // These are passed via additional push constants or derived from the locs arrays
}
```

**Alternative (recommended)**: Add the slot bases as additional fields in
`FullDecodeWeightAddrs` or as new push constants. Since the struct is
already uploaded as a UBO, add:

```c
// In FullDecodeWeightAddrs (C and GLSL):
uint hidden_slot;   // ws offset for hidden state
uint normed_slot;   // ws offset for normed output
uint q_slot;        // ws offset for Q projection output
uint k_slot;        // ws offset for K
uint v_slot;        // ws offset for V
uint attn_out_slot; // ws offset for attention output (before output proj)
uint ffn_normed_slot; // ws offset for FFN normed input
uint gate_slot;     // ws offset for gate proj output
uint up_slot;       // ws offset for up proj output
uint ffn_out_slot;  // ws offset for FFN down proj output (before residual)
```

Claude will add these fields to `FullDecodeWeightAddrs` in Python and fill
them from `engine._ws_slots` during `_register_full_decode()`.

#### B8. Attention (WG 0 only, single-threaded across heads)

Gemma3-1B: n_head=4, n_head_kv=1. Attention is cheap (kv_cap=1024 scores
per head). Keep it on WG 0 only — no parallelism needed here. Copy the
existing attention logic from B12, reading Q/K/V from act[] slots.

#### B9. GEGLU activation (all WGs participate)

```glsl
// Element-wise GEGLU: act[gate_slot + i] = gelu(act[gate_slot+i]) * act[up_slot+i]
// Each WG handles its slice of [0..N_ff)
void geglu_slice(uint gate_base, uint up_base, uint N_ff,
                 uint row_start, uint n_rows) {
    for (uint i = row_start + tid; i < row_start + n_rows; i += WG_SIZE) {
        if (i < N_ff) {
            act[gate_base + i] = gelu(act[gate_base + i]) * act[up_base + i];
        }
    }
    barrier();
}
```

#### B10. Layer loop structure in main()

```glsl
void main() {
    uint wg_id = gl_WorkGroupID.x;
    uint tid   = gl_LocalInvocationID.x;
    bool is_wg0 = (wg_id == 0u);
    uint n_embd = wa.N_embd;
    uint next_bid = 0u;  // monotone barrier counter

    // ── Early exit guard ──
    if (n_embd == 0u || pc.n_layer == 0u || pc.kv_cap == 0u) return;

    // ── Load embedding → hidden slot ──
    // Only WG 0 (any WG would do but one is enough for n_embd=1152)
    if (is_wg0) {
        for (uint i = tid; i < n_embd; i += WG_SIZE)
            act[wa.hidden_slot + i] = act[locs_emb_in[i]];
        barrier();
    }
    global_arrive_wait(next_bid++);  // bid 0: all WGs wait for embed load

    for (uint L = 0u; L < pc.n_layer; L++) {

        // ── Pre-attn RMSNorm (WG 0 only) ──
        if (is_wg0) rmsnorm_act_wg0(wa.hidden_slot, wa.normed_slot, n_embd, wa.attn_norm[L]);
        global_arrive_wait(next_bid++);  // norm done, normed_slot ready

        // ── QKV matmul (all WGs) ──
        // Q: N_q outputs
        {
            uint row_start, n_rows; get_slice(wa.N_q, row_start, n_rows);
            matvec_act_to_act(wa.normed_slot, wa.q_slot, n_embd,
                              row_start, n_rows, wa.wq[L], wa.qp_wq[L], wa.group_size_attn);
        }
        // K: N_k outputs
        {
            uint row_start, n_rows; get_slice(wa.N_k, row_start, n_rows);
            matvec_act_to_act(wa.normed_slot, wa.k_slot, n_embd,
                              row_start, n_rows, wa.wk[L], wa.qp_wk[L], wa.group_size_attn);
        }
        // V: N_v outputs
        {
            uint row_start, n_rows; get_slice(wa.N_v, row_start, n_rows);
            matvec_act_to_act(wa.normed_slot, wa.v_slot, n_embd,
                              row_start, n_rows, wa.wv[L], wa.qp_wv[L], wa.group_size_attn);
        }
        global_arrive_wait(next_bid++);  // QKV done

        // ── QK-norm + RoPE + KV-cache write + attention (WG 0 only) ──
        if (is_wg0) {
            // apply attn_q_norm to act[q_slot..q_slot+N_q]
            // apply attn_k_norm to act[k_slot..k_slot+N_k]
            // apply RoPE to Q and K in act[]
            // write K/V to kvcache at layer L, position pos
            // compute attention: for each head h, dot Q[h] with K[h,0:pos+1],
            //   softmax with softcap, weighted sum over V → act[attn_out_slot]
            // (port directly from existing B12 shader logic, reading from act[])
        }
        global_arrive_wait(next_bid++);  // attention done, attn_out_slot ready

        // ── Save hidden for residual ──
        // hidden_slot contains the residual; attn_out_slot has the attn output.
        // Output proj: act[proj_out_slot] = W_o * act[attn_out_slot]
        // Use a temp region within attn_out_slot area or a dedicated proj_out_slot.

        // ── Output proj (all WGs) ──
        {
            uint row_start, n_rows; get_slice(n_embd, row_start, n_rows);
            // reads from attn_out_slot (n_q inputs), writes to proj_out_slot
            matvec_act_to_act(wa.attn_out_slot, wa.proj_out_slot, wa.N_q,
                              row_start, n_rows, wa.wo[L], wa.qp_wo[L], wa.group_size_attn);
        }
        global_arrive_wait(next_bid++);  // output proj done

        // ── Post-attn residual + norm (WG 0 only) ──
        // act[hidden_slot] += rmsnorm(act[proj_out_slot], post_attn_norm[L])
        if (is_wg0) rmsnorm_add_act_wg0(wa.proj_out_slot, wa.hidden_slot,
                                         n_embd, wa.post_attn_norm[L]);
        global_arrive_wait(next_bid++);  // hidden updated

        // ── FFN RMSNorm (WG 0 only) ──
        if (is_wg0) rmsnorm_act_wg0(wa.hidden_slot, wa.ffn_normed_slot,
                                     n_embd, wa.ffn_norm[L]);
        global_arrive_wait(next_bid++);

        // ── Gate + Up matmul (all WGs) ──
        {
            uint row_start, n_rows; get_slice(wa.N_ff, row_start, n_rows);
            matvec_act_to_act(wa.ffn_normed_slot, wa.gate_slot, n_embd,
                              row_start, n_rows, wa.wg[L], wa.qp_wg[L], wa.group_size_ffn);
            matvec_act_to_act(wa.ffn_normed_slot, wa.up_slot, n_embd,
                              row_start, n_rows, wa.wu[L], wa.qp_wu[L], wa.group_size_ffn);
        }
        global_arrive_wait(next_bid++);

        // ── GEGLU activation (all WGs) ──
        {
            uint row_start, n_rows; get_slice(wa.N_ff, row_start, n_rows);
            geglu_slice(wa.gate_slot, wa.up_slot, wa.N_ff, row_start, n_rows);
        }
        global_arrive_wait(next_bid++);

        // ── Down proj (all WGs) ──
        {
            uint row_start, n_rows; get_slice(n_embd, row_start, n_rows);
            matvec_act_to_act(wa.gate_slot, wa.ffn_out_slot, wa.N_ff,
                              row_start, n_rows, wa.wd[L], wa.qp_wd[L], wa.group_size_ffn);
        }
        global_arrive_wait(next_bid++);

        // ── Post-FFN residual + norm (WG 0 only) ──
        if (is_wg0) rmsnorm_add_act_wg0(wa.ffn_out_slot, wa.hidden_slot,
                                          n_embd, wa.post_ffn_norm[L]);
        global_arrive_wait(next_bid++);  // layer L done

    } // end layer loop

    // ── Final norm → locs_hidden_out[] ──
    // WG 0 applies output_norm to act[hidden_slot] and writes to locs_hidden_out
    if (is_wg0) {
        F32Buf w = F32Buf(wa.output_norm);
        float ss = 0.0;
        for (uint i = tid; i < n_embd; i += WG_SIZE) ss += act[wa.hidden_slot + i] * act[wa.hidden_slot + i];
        ss = reduce_sum(ss);
        float inv_rms = inversesqrt(ss / float(n_embd) + 1e-6);
        for (uint i = tid; i < n_embd; i += WG_SIZE)
            act[locs_hidden_out[i]] = act[wa.hidden_slot + i] * inv_rms * w.f[i];
    }
    // No final barrier needed (other WGs are done / spinning at layer exit)
}
```

---

## Barrier count estimate

Per layer: ~10 barriers (norm, QKV, attn, outproj, post-attn, ffn-norm,
gate-up, geglu, down, post-ffn). × 26 layers = 260 barriers + 1 (embed load) = 261.
N_BARRIERS=512 is sufficient.

---

## Slot bases: new FullDecodeWeightAddrs fields

Add to the end of the C and GLSL struct (before the scalar model dims):

```c
// C (in adamah.h / adamah.c FullDecodeWeightAddrs):
uint32_t hidden_slot;
uint32_t normed_slot;
uint32_t q_slot;
uint32_t k_slot;
uint32_t v_slot;
uint32_t attn_out_slot;  // where attention output goes before output proj
uint32_t proj_out_slot;  // output proj result (before residual)
uint32_t ffn_normed_slot;
uint32_t gate_slot;
uint32_t up_slot;
uint32_t ffn_out_slot;
```

Claude will fill these from `engine._ws_slots` dict in `_register_full_decode()`.

---

## Build + test

```bash
# Recompile shader:
cd /c/Users/samus/Documents/ADAM/adamah-MAIN/adamah
glslc shaders/src/f32/map_full_decode_step.comp -o shaders/f32/map_full_decode_step.spv
cp shaders/f32/map_full_decode_step.spv shaders/map_full_decode_step.spv

# Rebuild DLL (must be ~220-230KB):
PATH="/c/mingw64/bin:$PATH" gcc -shared -O2 -march=native \
  -include _shader_path.h \
  -I"/c/VulkanSDK/1.4.341.1/Include" \
  adamah.c -o adamah.dll \
  /c/Users/samus/Documents/ADAM/libvulkan-1.a -lm -Wl,--export-all-symbols
```

Test correctness:
```bash
cd /c/Users/samus/Documents/ADAM
PYTHONUTF8=1 PYTHONPATH=/c/Users/samus/Documents/ADAM \
  "/c/Users/samus/AppData/Local/Programs/Python/Python312/python.exe" -X utf8 \
  tests/diagnostics/diag_inference.py gemma3-1b.gguf
```
Target: 8/8 PASS, generation "2 + 2 = 4".

Test throughput:
```bash
PYTHONUTF8=1 PYTHONPATH=/c/Users/samus/Documents/ADAM \
  "/c/Users/samus/AppData/Local/Programs/Python/Python312/python.exe" -X utf8 \
  tests/diagnostics/diag_chat_perf.py gemma3-1b.gguf
```
Target: Turn 2 decode_tps >> 61 tok/s (B6 baseline).

---

## Notes for Codex

- Do NOT modify `engine.py` or `adamah_chat.py` — Claude handles Python wiring
- Do NOT modify any other shaders or C functions outside the `full_decode_*` area
- The `FullDecodeWeightAddrs` struct must stay identical in C and GLSL (byte-exact)
- If `atomicAdd` on `coherent` buffer doesn't compile on the glslc version,
  try adding `#extension GL_KHR_memory_scope_semantics : require`
- `coherent` on a buffer binding is required for the spin-wait to see other
  WGs' atomic writes without deadlocking
- Spin-wait correctness: all N_WG WGs MUST reach every `global_arrive_wait` call.
  Any early-return or conditional skip must be guarded so all WGs still call the barrier.
- If next_bid reaches N_BARRIERS (512) the shader silently continues with
  a corrupted counter — add an assert or cap in debug mode.
- V3D note: V3D may not support spin-waits across workgroups within a single dispatch.
  If tests fail on Pi, fall back to N_WG=1 (identical to current B12 behavior).
  Claude will add a `_b13_n_wg` property gated on device caps.

---

## B13 addendum — barrier overhead diagnosis + A/B test (2026-03-25)

### Observed result
B13 validated at **28.6 tok/s** on RTX 3070 (vs 1.33 tok/s B12, 61 tok/s B6).
Expected was ~300 tok/s. The gap points to barrier overhead dominating.

### Root cause hypothesis
The current `global_arrive_wait` uses a **globally contended coherent uint counter**:
all N_WG workgroups spin on `s[bid]` via `memoryBarrierBuffer()` in a loop.
On NVIDIA, coherent buffer reads between workgroups cross the L2 cache coherency
domain — each spin iteration triggers a full L2 cache line invalidation.
Measured cost: likely ~50–150 μs per barrier (not ~3 μs as assumed).
With 260 barriers × 100 μs = 26 ms → ~38 tok/s ceiling, consistent with observation.

**Correction**: `GL_EXT_shader_atomic_float2` is NOT the right tool here.
The barrier mechanism needs uint atomics + memory semantics, not float atomics.

### Requested A/B test

**Step 1: measure barrier cost directly.**

Add a GPU timestamp (via push constant round-trip or a counter in the sync buffer)
to isolate barrier cost from compute cost. Specifically:
- Record time before first `global_arrive_wait` and after last one (per token)
- Report `barrier_ms` and `compute_ms` separately in a debug build

If `barrier_ms >> compute_ms`, confirmed barrier-dominated.

**Step 2: implement and compare two alternative sync schemes.**

#### Scheme A — "mailbox/epoch" per-WG coordination

Instead of all WGs spinning on a single global counter, WG 0 acts as coordinator:

```glsl
// SyncBuf layout for mailbox scheme:
// s[0..N_WG-1]: per-WG arrival flags (each WG writes its own slot)
// s[N_WG]:      coordinator broadcast slot (WG 0 writes, others read)

void global_arrive_wait_mailbox(uint epoch) {
    memoryBarrierBuffer(); barrier();
    // Each WG writes to its own dedicated slot — no contention
    if (tid == 0u) {
        s[wg_id] = epoch + 1u;          // signal arrival (unique slot per WG)
        memoryBarrierBuffer();
        if (wg_id == 0u) {
            // WG 0 polls per-WG slots (N_WG reads, but each is hot in L1)
            for (uint w = 1u; w < pc.n_wg; w++) {
                while (s[w] <= epoch) { memoryBarrierBuffer(); }
            }
            // broadcast "all arrived"
            s[N_WG_MAX] = epoch + 1u;
            memoryBarrierBuffer();
        } else {
            // non-WG0: spin on broadcast slot (1 cache line, written once by WG0)
            while (s[N_WG_MAX] <= epoch) { memoryBarrierBuffer(); }
        }
    }
    barrier(); memoryBarrierBuffer();
}
```

Key difference: each WG writes to its **own slot** (no write contention).
Non-WG0 WGs spin on a **single broadcast slot** written exactly once by WG 0.
WG 0 polls N_WG slots, but each is written once and then hot in L2 cache.
Expected: 1–2 cache line invalidations per barrier instead of N_WG × continuous polling.

`epoch` increments by 1 each call (local uint variable in main(), no extra buffer slot needed).
`N_WG_MAX` is a compile-time constant (e.g., 64). SyncBuf needs N_WG_MAX+1 uint slots.

#### Scheme B — "chunked dispatch" fallback

If in-shader sync remains expensive regardless of scheme, fall back to splitting
the 26 layers into K groups dispatched separately (K << 339):

```
// Instead of 1 dispatch for 26 layers:
// Dispatch 1: layers 0..8   (9 layers, no cross-dispatch sync needed)
// Dispatch 2: layers 9..17
// Dispatch 3: layers 18..25 + final norm
// Total: 3 dispatches vs 339 for B6 → ~100× less dispatch overhead
```

Each dispatch: N_WG workgroups, no inter-WG sync needed (layers within a dispatch
are processed sequentially by each WG independently, with only intra-WG barriers).
Between dispatches: standard Vulkan pipeline barrier (cheap at 3 barriers vs 339).

This eliminates global spin-barriers entirely at the cost of 3 dispatches instead of 1.
Estimated: 3 × 48μs (barrier overhead) + compute ≈ 2.1ms → ~470 tok/s.

### Requested implementation order

1. Add timestamp/cost measurement to current B13 (1 day)
2. Implement mailbox scheme A (in-shader, drop-in replacement for spin-wait)
3. If scheme A ≥ 200 tok/s → ship. If not → implement scheme B (chunked dispatch).
4. Report all three measurements in AGENT_COLLAB.md.

### Zero-copy embedding (B13 add-on, independent of barrier work)

While fixing barriers, also add zero-copy token embedding to eliminate the
`map_row_gather_xq8_dev` pre-dispatch currently enqueued before `full_decode_step`:

Add to `FullDecodeWeightAddrs`:
```c
uint64_t token_embd;      // BDA: row 0 of token_embd.weight in Q8 map
uint64_t qp_token_embd;   // BDA: qparams for token_embd
uint32_t group_size_embd;
float    emb_scale;       // sqrt(n_embd) for Gemma3, 1.0 otherwise
```

Add `token_id` to push constants (6th uint32, total 24 bytes).

Shader: at start of `main()`, replace `locs_emb_in` read with:
```glsl
uint emb_row = pc.token_id * wa.N_embd;
for (uint i = tid; i < wa.N_embd; i += WG_SIZE)
    act[wa.hidden_slot + i] =
        dequant_q8(wa.token_embd, wa.qp_token_embd, emb_row + i, wa.group_size_embd)
        * wa.emb_scale;
barrier();
```

`binding 3` (LocsEmbIn) can be removed from the descriptor set.
Claude will update `engine.py` to skip `_enqueue_gpu_embed_token` when `_b12` active
and pass `tok` as the 6th push constant.

---

## Post results in AGENT_COLLAB.md

- decode_tps Turn 2 from diag_chat_perf.py (scheme A, B, baseline)
- barrier_ms vs compute_ms breakdown
- diag_inference.py result (8/8 or fail) after each change
- N_WG value used (auto or manual)
