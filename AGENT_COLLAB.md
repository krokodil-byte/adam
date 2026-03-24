# AGENT_COLLAB — Claude ↔ Codex coordination
# Goal: 100 tok/s on RTX 3070 + Pi 5 working
# Updated: 2026-03-23

## Ownership
| Layer | Owner | Files |
|-------|-------|-------|
| Python engine/profiling | **Claude** | `adam/models/engine.py`, `adamah_chat.py`, `tests/diagnostics/` |
| C/GLSL backend | **Codex** | `adamah-MAIN/adamah/adamah.c`, `shaders/src/**/*.comp` |
| This file | Shared | append findings here |

---

## Current state (2026-03-21)

### RTX 3070 — Gemma3-1B
- **~59 tok/s** (diag_chat_perf Turn 2), **~62 tok/s** (chat app)
- Profile: `desktop_discrete`, `level_batched`, `direct_kv=True`, `rows=512`, B6+B8+B9+B10 active
- Decode plan: **339 ops** (26 layers), `fused_qkv_q8=True`, `fused_gateup_q8=True`, `fused_gateup_act_q8=True` (capability present, but GEGLU fallback active for Gemma3)
- **343 barriers/batch, 48.3 μs/barrier** (measured via B10) → 16.6ms GPU exec
- Barrier model: 343 barriers × 48.3μs = 16.6ms ≈ core_batch ✅
- Each decode plan op emits its own `cmd_barrier_after_dispatch()` (`barrier_skip=0` during decode_step — only set inside `fusion_flush`, not decode_step path)
- B11 (level-aware decode step) attempted and rolled back: decode chain is genuinely sequential (almost every op depends on previous output → alias-aware gives 1 barrier/op anyway)
- **Next: B12** — monolithic decode shader (1 dispatch for all 26 layers)

### Pi 5 V3D
- `stream_load=False` in broadcom_v3dv profile (USB bottleneck fixed, commit c88c158)
- A4 V3D MMU guard landed in `adamah.c` (no more PTB crash)
- A2 still TODO: check `env | grep -E '^VK_|^VULKAN_'` on Pi — validation layers = 100× slowdown

---

## Active work

### B12 — CLOSED (correctness ✅, perf ❌ 1.33 tok/s)

B12 achieves correct output (8/8 PASS, "2+2=4") but is 46× slower than B6 (1.33 vs 61 tok/s).
The monolithic single-dispatch design (`vkCmdDispatch(1,1,1)`, WG_SIZE=128) activates only 1 SM
on RTX 3070 (46 SMs), yielding ~2% GPU utilization. Sequential layer dependency + large per-layer
matmuls require hundreds of workgroups per op to saturate the GPU — incompatible with 1-dispatch design.
**B6 (61 tok/s) remains the active decode path. B12 infra left in place but bypassed.**

### Next optimization candidates

The 61 tok/s B6 ceiling is barrier-dominated: 343 barriers × 48.3μs = 16.6ms per token.
Actual compute ≈ 2ms. To break through, need to reduce barrier count or barrier latency.
Options explored and results:
- B7: TILE_K/WG shader tuning → regression/correctness fail
- B8–B10: Q8 fused shaders → flat (level_batched already coalesced)
- B11: alias-aware level step → rolled back (1 barrier/op anyway)
- B12: monolithic shader → correct but 46× slower

---

## Findings log

| Date | Agent | Finding |
|------|-------|---------|
| 2026-03-18 | Claude | Baseline: 62 tok/s RTX3070, 8 PASS |
| 2026-03-21 | Claude | B1 REVERTED: `fused_qkv_qk_norm_rope=True` → 54 tok/s. Do NOT enable on desktop_discrete. |
| 2026-03-21 | Claude | B2: `direct_kv_cache_write=True` safe, restored. B3 (fusion_flush skip): no-op, fusion already disabled. |
| 2026-03-21 | Claude | B4: rows_per_group sweep: **512=optimal** (was 1024). Profile fixed. |
| 2026-03-21 | Codex | B5: `level_batched` root cause found+fixed (alias-overlap + final barrier). |
| 2026-03-21 | Claude | B5+B6 promoted: level_batched + decode plan active. **~61 tok/s** new baseline. |
| 2026-03-21 | Claude | B7 exhausted: TILE_K/WG tuning regression or correctness fail. Barrier overhead dominates, not compute. |
| 2026-03-21 | Codex | B8 backend: fused QKV+gateup xq4 shaders + decode opcodes 26/27. |
| 2026-03-21 | Claude | B8 Q8 gap: Gemma3-1B uses Q5_0/Q8_0 (xq8 map) not Q4_K. Added xq8 variants. 339 ops. ~60 tok/s flat — level_batched already coalesced these → no real barrier savings. |
| 2026-03-21 | Codex | B9: `map_fused_gateup_act_t_xq8` shader + opcode 30. B10: `adamah_get_last_barrier_count()`. |
| 2026-03-21 | Claude | B9 wiring: `fused_gateup_act_q8` guard in decode plan. Fallback active: Gemma3 GEGLU ≠ SwiGLU. |
| 2026-03-21 | Claude | **B10 data**: 343 barriers, **48.3 μs/barrier**, 16.6ms total. Confirmed: entire GPU time is barrier-dominated. |
| 2026-03-21 | Claude | **Regression fix**: Codex B9/B10 DLL was 470KB (no -O2) → 27 tok/s. Rebuilt with `gcc -O2 -march=native` → 222KB, 59 tok/s restored. |
| 2026-03-21 | Codex | B11 prototype rolled back: alias-aware = 1 barrier/op (no gain); handle-only = correctness degradation. Decode ops genuinely sequential. |
| 2026-03-21 | Claude | **B12 proposed**: single monolithic shader for all 26 layers via buffer device addresses. Estimated ~3ms/step → ~300 tok/s. |
| 2026-03-21 | Codex | **B12 infra landed**: enabled Vulkan 1.2/BDA path (`bufferDeviceAddress=enabled` on RTX3070), added `FullDecodeWeightAddrs` + C APIs `adamah_full_decode_register()` / `adamah_full_decode_step()`, map/qparam device-address query helpers, Python ctypes bindings (`FullDecodeWeightAddrs`, `full_decode_register`, `full_decode_step`, `_has_full_decode_step`) and new shader/pipeline `map_full_decode_step`. |
| 2026-03-22 | Codex | **B12 shader completed (sequential composition)**: `map_full_decode_step.comp` now executes full per-layer loop in one dispatch (pre-attn RMSNorm, QKV matvec Q8, RoPE, KV write, attention over cache, output proj, post-attn RMSNorm+residual, FFN norm, gate+up+GEGLU, down proj, post-FFN RMSNorm+residual, final norm). Added `qp_wo[]` in `FullDecodeWeightAddrs` and KV-cap push constant path from C. |
| 2026-03-22 | Codex | **Known B12 caveats**: q/k norm weights and attn softcap are not yet applied in monolithic shader (RoPE+attention currently use scale-only path), and activation is GEGLU-only in-shader for now. Baseline decode path unchanged; `diag_inference.py` still 8 PASS. |
| 2026-03-22 | Claude | **B12 engine wiring complete + activated**: `_register_full_decode()` now fills all `FullDecodeWeightAddrs` fields: Q8 weights (q8_base+off), qparams (q8_qp_base+(off//32)×8), F32 norms (ws_base+off×4), attn_q_norm/attn_k_norm per layer, output_norm, attn_softcap. `_full_decode_active` set to True on successful register. `_b12` branch fires before `_b6` in `_forward()`; final-norm gate skips on `_b12`. **Ready for diag_inference.py + diag_chat_perf.py validation.** |
| 2026-03-22 | Claude | **B12 debug resolved**: After Codex added check-point fprintfs, none fired → all 7 validation gates pass. Shader runs but produces |hid|_gpu ≈ 34 (≈ raw embedding norm) vs |hid|_cpu ≈ 33427. Diagnosis: Q8 weight BDAs are all invalid → shader reads 0 for every weight → matmuls return 0 → residual connections preserve embedding through all 26 layers. Root cause: `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` missing from Q8 map buffer (map 2). `get_map_buffer_device_address(q8_map_id)` returns 0 → all q8_base+off addresses are garbage. Fix in adamah.c: see `CODEX_PROMPT_B12_DEBUG.md` updated task. |
| 2026-03-22 | Claude | **B12 mixed-quant diagnosis + fix**: `attn_output.weight`=Q4_K (GGML 12), `ffn_down.weight`=Q6_K (GGML 14) — both map 1, not Q8 map. `_w_addr()` returned `q8_base + q4_off` → shader read 0 for these → residuals frozen after 26 layers. Fix: added `_ensure_b12_q8_map()` (engine.py) — one-time init creates GPU map 3 with those tensors re-quantized to Q8 (dequant Q4_K/Q6_K → F32 → Q8). `_register_full_decode` now uses map 3 BDA for these tensors. Ready for `diag_inference.py` retest. |
| 2026-03-23 | Claude | **B12 root cause found + fixed (Python-side)**: `adamah_full_decode_step` C code computes `kv_cap = map_kvcache->total_bytes / (4 * 2 * n_layer * n_head_kv * head_dim_kv)`. When `map_kvcache_id == map_ws_id` (ws map ~420MB including activations, norms, scores), kv_cap is inflated (~2067 instead of 2048). Shader uses wrong `kv_cap` for layer/head offsets → K/V written to wrong ws locations → self-consistent within each (L,h) but all heads>0 and layers>0 access garbage. Fix: `_ensure_b12_kv_map()` creates dedicated map 4 with EXACTLY `2 * n_layer * n_head_kv * kv_cap * head_dim_kv` F32 elements. `full_decode_register(ws_map_id=0, kvcache_map_id=4, ...)` → C code derives exact kv_cap. Also fixed Check 6 diagnostic (was gathering from hid slot after B12 which writes to normed slot → false ratio=0.001; now gathers from normed slot for B12 n=26, uses top1 match as additional pass criterion). |
| 2026-03-23 | Claude | **B12 correctness CONFIRMED, perf FAILED**: `diag_inference.py` 8/8 PASS, generation "2 + 2 = 4\n" ✅. Production-mode KeyError fixed (moved `_register_full_decode()` before `release_host_state()` in engine `__init__`). **`diag_chat_perf.py` decode_tps = 1.33 tok/s** (vs 61 tok/s B6) — 46x regression. Root cause: `vkCmdDispatch(1,1,1)` in C — single workgroup × 128 threads for all 26 layers on RTX 3070 (46 SMs). Only 1 SM active = ~2% GPU utilization. Transformer layers must execute sequentially (inter-layer dependency), and each layer's matmuls (1152×4096, 6912×1152) need hundreds of workgroups for saturation — incompatible with monolithic single-dispatch design. **B12 is correct but not viable as a speed optimization. B6 (61 tok/s) remains the active decode path.** |
| 2026-03-21 | Codex | A4 V3D guard landed. Pi retest needed. |
| 2026-03-21 | Claude | Pi A5: stream_load=False on broadcom_v3dv. USB bottleneck eliminated. |
| 2026-03-22 | Codex | **B12 BDA audit + cleanup**: `create_buffer_ex()` already applies `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` (for storage buffers when BDA enabled) and `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT` via `VkMemoryAllocateFlagsInfo`; map buffers (including Q8 map 2 and qparams) go through this path. Removed temporary `B12_FAIL checkN` fprintfs from `adamah_full_decode_step`, rebuilt `adamah_new.dll` (~228KB). `diag_inference.py`: still FAIL at Check 6 (n=26 ratio ~0.001), so remaining issue is shader math/correctness, not BDA allocation flags. |
| 2026-03-22 | Codex | **B12 shader save/restore patch applied** (`map_full_decode_step.comp`): added residual save/restore around attn and FFN blocks, expanded guard for extra tmp slots, raised `MAX_TMP` to 9216, rebuilt SPV. `diag_inference.py` still fails at Check 6 for n=26 (ratio ~0.001). New finding: B12 wiring currently feeds *all non-F32 weights* from q8 base/qparams, but Gemma3-1B decode weights are mixed (`q5_0` + `q4_K` + `q6_K` + `q8_0`), so `attn_output`/`ffn_down` (q4/q6 map) are decoded as q8 in monolithic shader and corrupt full 26-layer output. |

---

## Permanent constraints
- **Never** enable `experimental_fused_qkv_qk_norm_rope=True` or regress `direct_kv=False` on desktop_discrete
- **SPVs**: must be in `shaders/` root (flat), NOT `shaders/f32/` — `load_spv()` uses `SHADER_PATH/name`
- **Build**: `gcc -shared -O2 -march=native -include _shader_path.h ...` → ~220KB. If >400KB = wrong flags
- **Test always**: `diag_inference.py` 8 PASS + "2+2=4" after every C/shader change
- Codex: do NOT modify `adamah_chat.py` or `adam/models/engine.py`
- Claude: do NOT modify `adamah.c` or `shaders/src/**/*.comp`
