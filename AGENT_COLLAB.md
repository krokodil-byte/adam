# AGENT_COLLAB â€” Claude â†” Codex coordination
# Goal: 13+ tok/s on Pi 5 V3D (GPU-primary), 100+ tok/s on RTX 3070
# Updated: 2026-04-05

## Ownership
| Layer | Owner | Files |
|-------|-------|-------|
| Python engine / profiling | **Claude** | `adam/models/engine.py`, `adamah_chat.py`, `tests/diagnostics/` |
| C / GLSL backend | **Codex** | `adamah-MAIN/adamah/adamah.c`, `shaders/src/**/*.comp` |
| This file | Shared | append findings here |

---

## Current state

### RTX 3070 â€” Gemma3-1B
- **~65 tok/s** â€” B14 monolithic shader (N_WG=256/512, head-parallel attention)
- Monolithic path active: `_full_decode_active=True`, `diag_inference` 8/8 PASS

### Pi 5 V3D â€” Gemma3-1B
- **1.24 tok/s** â€” B12 monolithic, N_WG=1 (forced), `core_batch ~530ms/token`
- Root cause identified: **barrier overhead in matvec**, not DRAM bandwidth
  - Each `matvec_from_hidden_to_tmp` does `reduce_sum()` per output row â†’ 7 barriers Ã— N rows
  - For 26 layers: ~3.2M shader barriers â†’ ~519ms at ~160ns/barrier on V3D
  - Theoretical DRAM floor: ~87ms (697MB / 8GB/s) â†’ currently 6Ã— away
- Fix direction: **thread-per-row matvec** (each thread owns one full output row, zero reduce â†’ 1 barrier per matvec call)

---

## Active work

### [CODEX] B16 â€” thread-per-row matvec (V3D barrier fix)
Prompt: `CODEX_PROMPT_B16.md`

Change `matvec_from_hidden_to_tmp`, `matvec_from_tmp_to_tmp`, `gateup_geglu_from_hidden`
from reduce-per-row (7+1 barriers Ã— N rows) to thread-per-row (1 barrier total).
Expected: 3.2M barriers â†’ ~130, core_batch 530ms â†’ ~90ms, **~11 tok/s on Pi**.

---

## Permanent constraints
- **Never** enable `fused_qkv_qk_norm_rope=True` or `direct_kv=False` on desktop_discrete
- **Never** enable multi-WG on V3D (N_WG=1 forced on broadcom_v3dv)
- **SPVs**: `shaders/f32/` for Broadcom, `shaders/` root for desktop_discrete
- **Build**: `gcc -shared -O2 -march=native -include _shader_path.h ...` â†’ ~220KB. >400KB = wrong flags
- **Test always**: `diag_inference.py` 8/8 PASS + "2+2=4" after every C/shader change
- Codex: do NOT modify `adamah_chat.py` or `adam/models/engine.py` (except approved wiring)
- Claude: do NOT modify `adamah.c` or `shaders/src/**/*.comp`

---

## Key findings (condensed)

| Date | Finding |
|------|---------|
| 2026-03-21 | B6 baseline: 61 tok/s RTX3070 (level_batched, rows=512, direct_kv=True) |
| 2026-03-21 | B10 data: 343 barriers Ã— 48.3Âµs = 16.6ms â€” barrier-dominated on RTX3070 |
| 2026-03-23 | B12 monolithic correct (8/8 PASS) but 1.33 tok/s â€” dispatch(1,1,1) = 1 SM |
| 2026-03-25 | B13 cooperative N_WG: 36.88 tok/s RTX3070; b3=89% spin (attention WG0-only) |
| 2026-03-26 | B14 head-parallel attention: **65 tok/s** RTX3070, b3 spin Ã·3.3x |
| 2026-03-29 | Pi V3D: uint64/BDA crash (NIR `pack_64_2x32_split`) â€” fixed with ADAMAH_V3D_MONOLITHIC |
| 2026-04-01 | Pi N_WG=1 wins: 1.24 tok/s vs N_WG=4 â†’ 0.76 tok/s (spin overhead dominates) |
| 2026-04-02 | Pi stage-probe: matvec blocks = 511â€“544ms, attention/norm = 11â€“15ms |
| 2026-04-03 | Subgroup ops rejected by V3D (`Unknown intrinsic ... reduction_op=fadd`) |
| 2026-04-04 | Root cause: reduce_sum() per matvec row = 3.2M barriers/token = 519ms. Fix: thread-per-row |

| 2026-04-05 | Broadcom B12 steady confirmed: Turn2 decode 1.24 tok/s, core_batch 530.8ms, sample 261ms. |
| 2026-04-05 | Stage mask probe: 0x08=543.6ms, 0x10=265.6ms, 0x3f=534.1ms -> FFN path dominates. |
| 2026-04-05 | WG_SIZE sweep (64/128/256) on monolithic produced identical ~543.5ms for 0x3f. |
| 2026-04-05 | Split-proj (no monolithic) regresses to ~0.03 tok/s due dispatch/barrier overhead. |
| 2026-04-05 | Single-proj microbench (split path): gate q8 311ms, up q8 313ms, down q4 395ms. |
| 2026-04-05 | B18 attempt: B12 direct-q5 gate/up path implemented (raw Q5 map 5, packed selector=3), q4-direct scaffold added in shader/C. |
| 2026-04-05 | Safety: experimental B18 path now `ADAM_B12_Q5_RAW=0` by default (opt-in only) to prevent production regressions. |
| 2026-04-05 | Pi diag_inference currently fails Check5/Check6 even with B12 disabled (`ADAM_BROADCOM_SPLIT_PROJ=1 ADAM_BROADCOM_REQUIRE_B12=0`) -> regression is upstream of new B12 path. |
| 2026-04-05 | Pi perf A/B (`diag_chat_perf --max-tokens 8`): q5_raw ON vs OFF both Turn2 decode_tps ~1.24 (no measurable gain yet). |
| 2026-04-05 | B19: q4 direct down opt-in (`ADAM_B12_Q4_DIRECT_DOWN=1`) implemented; B12 Q8 remap shrinks 238MB -> 134MB (39 tensors), but Turn2 decode_tps remains ~1.24. |
| 2026-04-05 | B19+b18 combined (`ADAM_B12_Q4_DIRECT_DOWN=1 ADAM_B12_Q5_RAW=1`): map5 raw q5=285MB + map3=134MB, Turn2 decode_tps ~1.23 (slight regression). |
| 2026-04-06 | B20 infra landed in C+shader: `ADAMAH_BROADCOM_DECODE_BACKEND=auto|v3dv_levels|legacy_monolithic`, per-layer staged dispatch path (`0x07` then `0x18`, final `0x20`) in `adamah_full_decode_step`, push constants extended with `layer_base`, optional stage timing log via `ADAMAH_BCM_STAGE_TIMING=1`. |
| 2026-04-06 | B20 follow-up: FFN stage split in `v3dv_levels` (`0x08` gate-only + `0x10` down-only). Gate stage now supports multi-WG without spin via scratch handoff (`ADAMAH_BCM_WG_FFN`), down+post-ffn remains WG=1 for correctness. |
| 2026-04-07 | Telemetry fix: `engine.generate()` now resets/reads native stats even without `trace_decode`, so `diag_chat_perf` reports real `dispatch_count/submit_count/barrier_count` instead of zeros. |
| 2026-04-07 | Broadcom measurement (Pi, `--max-tokens 8`): forcing `ADAMAH_BROADCOM_DECODE_BACKEND=v3dv_levels` gives Turn2 `decode_tps ~0.43`, `dispatch_count=656`, `barrier_count=664` (high dispatch-recording overhead on V3DV). |
| 2026-04-07 | Safety rollback: in backend `auto`, Broadcom now defaults back to `legacy_monolithic` (v3dv_levels remains opt-in). Verified Turn2 restored to `decode_tps ~1.24`, `dispatch_count=32`, `barrier_count=48`. |
