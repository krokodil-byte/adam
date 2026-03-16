# CPU/GPU Hybrid Pipeline for Broadcom V3DV

## Phase 1: ADAMAH C infrastructure
1. `find_device_local()` on integrated GPU: prefer DEVICE_LOCAL | HOST_VISIBLE
2. Map hot_pool when HOST_VISIBLE (set `b->ptr`)
3. Add `adamah_batch_submit()` (non-blocking) + `adamah_batch_wait()`
4. Add `adamah_ws_map_ptr(map_id, offset, size)` → CPU pointer to workspace
5. Add invalidate/flush helpers for coherency

## Phase 2: Python bindings
1. `batch_submit()`, `batch_wait()`
2. `ws_numpy_view(map_id, offset, n_elems)` → numpy array backed by GPU memory

## Phase 3: CPU lightweight ops (engine.py)
1. `_cpu_rmsnorm(data, weight, eps)` → numpy NEON-optimized
2. `_cpu_rope(q, k, pos, base, head_dim)` → numpy
3. `_cpu_residual_add(a, b)` → in-place a += b
4. `_cpu_silu_gate(gate, up)` / `_cpu_gelu_gate(gate, up)` → numpy
5. `_cpu_attn_softmax(scores, scale, softcap)` → numpy

## Phase 4: Broadcom forward pass restructure
Per layer: 1 GPU batch (heavy matmuls only) + CPU boundary work
