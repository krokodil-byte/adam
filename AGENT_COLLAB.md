# AGENT_COLLAB — Claude ↔ Codex coordination
# Goal: 100+ tok/s on RTX 3070 + Pi 5 V3D
# Updated: 2026-03-25

## Ownership
| Layer | Owner | Files |
|-------|-------|-------|
| Python engine / profiling | **Claude** | `adam/models/engine.py`, `adamah_chat.py`, `tests/diagnostics/` |
| C / GLSL backend | **Codex** | `adamah-MAIN/adamah/adamah.c`, `shaders/src/**/*.comp` |
| This file | Shared | append findings here |

---

## Current state (2026-03-25)

### RTX 3070 — Gemma3-1B
- **~61 tok/s** baseline — B6 decode plan (339 ops), `level_batched`, `direct_kv=True`, `rows=512`
- GPU time: 343 barriers × 48.3 μs = **16.6ms/token** (barrier-dominated; compute ≈ 2ms)
- B12 monolithic shader: correct (8/8 PASS, "2+2=4") but 1.33 tok/s (`vkCmdDispatch(1,1,1)` = 1 SM)
- **Active decode path: B6**

### Pi 5 V3D
- `stream_load=False` (USB bottleneck fixed)
- V3D MMU guard landed in `adamah.c` (no PTB crash)
- Perf unknown — needs retest after B13

---

## Active work

### [CODEX] B13 — cooperative multi-WG decode shader ← CURRENT PRIORITY

Prompt: `CODEX_PROMPT_B13.md`

**Core idea**: change `vkCmdDispatch(1, 1, 1)` → `vkCmdDispatch(N_WG, 1, 1)`.
Matmuls (99% of FLOPs) are split across N_WG workgroups in parallel.
Reductions (RMSNorm, softmax — 0.04% of FLOPs) stay on WG 0.
Inter-WG sync via coherent spin-barrier in a new small SyncBuf (binding 5).
Hidden state moves from `sh_hidden[]` (shared mem) to `act[]` (global mem).

**Expected**: N_WG=46 on RTX3070 → ~40× matmul speedup → ~50–300 tok/s.
For V3D (N_WG=4–8): eliminates 339 dispatches → major gain even with fewer SMs.

**C changes**: `adamah_full_decode_register()` gains `n_wg` param; allocates
sync_buf (512×uint32 = 2KB); `adamah_full_decode_step()` zeros sync_buf via
`vkCmdFillBuffer`, dispatches N_WG. `create_pipeline` bumps binding count 5→6.
Push constants grow 16→20 bytes (+n_wg).

**Shader changes**: new `global_arrive_wait(bid)` spin-barrier; per-WG slice
helpers; all matmuls use `matvec_act_to_act()`; reductions gated on `is_wg0`.
`FullDecodeWeightAddrs` gains 11 slot-base fields (hidden_slot, normed_slot,
q_slot, k_slot, v_slot, attn_out_slot, proj_out_slot, ffn_normed_slot,
gate_slot, up_slot, ffn_out_slot) — Claude fills from `engine._ws_slots`.

**Python (Claude)**: after Codex lands, Claude updates `_register_full_decode()`
to fill new slot fields, passes `n_wg` from GPU caps, wires Python binding.

---

## Findings log

| Date | Agent | Finding |
|------|-------|---------|
| 2026-03-18 | Claude | Baseline: 62 tok/s RTX3070, 8/8 PASS |
| 2026-03-21 | Claude | B1 REVERTED: fused_qkv_qk_norm_rope → 54 tok/s. Never enable on desktop_discrete. |
| 2026-03-21 | Claude | B4: rows_per_group 512=optimal (was 1024). Profile fixed. |
| 2026-03-21 | Codex  | B5: level_batched fix (alias-overlap + final barrier). |
| 2026-03-21 | Claude | B5+B6 promoted: **~61 tok/s** new baseline. |
| 2026-03-21 | Claude | B7 exhausted: shader TILE_K/WG tuning → regression. Barrier overhead dominates. |
| 2026-03-21 | Codex  | B8–B10: fused QKV+gateup Q8 shaders, barrier count API. |
| 2026-03-21 | Claude | B8–B10 wiring: 339 ops, flat perf (level_batched already coalesced). |
| 2026-03-21 | Claude | **B10 data**: 343 barriers × 48.3μs = 16.6ms. Entire GPU time is barrier-dominated. |
| 2026-03-21 | Codex  | B11 rolled back: alias-aware = 1 barrier/op anyway. No gain. |
| 2026-03-21 | Codex  | B12 infra: BDA path enabled, `adamah_full_decode_register/step`, Python bindings, shader scaffold. |
| 2026-03-22 | Codex  | B12 shader: full 26-layer loop, save/restore residual fix, MAX_TMP=9216. |
| 2026-03-22 | Claude | B12 wiring: Q8 remap map 3 (Q4_K/Q6_K → Q8 for attn_out/ffn_down), dedicated KV map 4. |
| 2026-03-23 | Claude | B12 correctness confirmed: 8/8 PASS, "2+2=4". Production-mode KeyError fixed. |
| 2026-03-23 | Claude | **B12 perf: 1.33 tok/s** — root cause: `vkCmdDispatch(1,1,1)` = 1 SM active. B6 stays active. |
| 2026-03-25 | Claude | **B13 designed**: cooperative N_WG dispatch, spin-barrier, matmul slicing. Prompt written. |
| 2026-03-21 | Codex  | A4: V3D MMU guard landed. |
| 2026-03-21 | Claude | A5: stream_load=False on broadcom_v3dv, USB bottleneck eliminated. |

---

## Permanent constraints
- **Never** enable `fused_qkv_qk_norm_rope=True` or `direct_kv=False` on desktop_discrete
- **SPVs**: must be in `shaders/` root (flat) — `load_spv()` uses `SHADER_PATH/name`
- **Build**: `gcc -shared -O2 -march=native -include _shader_path.h ...` → ~220KB. >400KB = wrong flags
- **Test always**: `diag_inference.py` 8/8 PASS + "2+2=4" after every C/shader change
- Codex: do NOT modify `adamah_chat.py` or `adam/models/engine.py`
- Claude: do NOT modify `adamah.c` or `shaders/src/**/*.comp`
