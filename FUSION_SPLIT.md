# Portable Performance Split - ADAM / ADAMAH

## Goal
Reproduce the op-centric batching advantage shown in
`adamah-MAIN/benchmarks/benchmark_simple_batches.py` inside real transformer decode
and interactive chat.

Constraint: portable Vulkan only. No NVIDIA-only fast paths, no vendor-specific
scheduler work, no overlap in owned files.

## Current Baseline
- Decode benchmark: `python -X utf8 -m adam.tools.benchmark --model gemma3-1b.gguf --preset decode_regression --profile desktop_discrete --adam-only --skip-llama --skip-ollama`
- Desktop result on current tree:
  - about `53.7 tok/s`
  - about `18.6 ms/token`
  - `core_batch ~12.6 ms/token`
  - `lm_batch ~4.1 ms/token`
  - about `603 dispatches`, `603 barriers`, `603 descriptor updates`, `53 fusion flushes` per generated token
- Chat finding on current tree:
  - prompt render/tokenize is small on desktop
  - the remaining high-value chat bug is failed KV-prefix reuse on some follow-up turns when history is rebuilt from decoded text instead of canonical token history

## File Ownership
- Codex only:
  - `adamah_chat.py`
  - `adam/models/engine.py`
  - `adam/tools/benchmark.py`
  - `adam/tokenizers/gguf_tok.py`
  - `tests/*`
  - this document
- Claude only:
  - `adamah-MAIN/adamah/adamah.c`
  - `adamah-MAIN/adamah/__init__.py`
  - `adamah-MAIN/adamah/shaders/src/*` if needed

No shared edits across those boundaries in the same phase.

## Phase Order
1. Codex lands chat-turn benchmark coverage and token-stable reuse diagnostics.
2. Claude lands native scheduler modes and additive stats, defaulting to `legacy`.
3. Claude lands `alias_safe` overlap-aware hazard tracking and reduces scheduler churn.
4. Codex re-enables merged Python fast paths against `fusion_scheduler_mode != legacy`.
5. Claude lands `level_batched` only after both target profiles are green.

## Codex Scope
### Phase 1
- Add `chat_turn_regression` to `adam.tools.benchmark`.
- Emit chat-turn fields:
  - `prompt_render_ms`
  - `prompt_tokenize_ms`
  - `prompt_total_tokens`
  - `prompt_reused_tokens`
  - `prompt_prefilled_tokens`
  - `prefill_ms`
  - `decode_ms`
  - `decode_tps`
  - `total_turn_ms`
  - `reuse_hit`
  - `reuse_prefix_tokens`
  - `reuse_miss_reason`
  - `reuse_miss_index`
- Keep `decode_regression` as the decode-core KPI.
- Keep `tests/test_chat_turn_perf.py` as the direct two-turn probe.
- Refactor chat session state so assistant turns retain canonical generated token spans.
- Stop relying on `decode(out_tokens) -> render template -> re-encode` for KV reuse.
- Preserve external chat behavior.

### Later Codex Work
- Re-enable merged `QKV` and merged `gate+up` only when the backend scheduler mode is no longer `legacy`.
- Remove avoidable host overhead in the chat path:
  - repeated template compilation
  - repeated render/tokenize work where it can be cached safely
  - repeated suffix counting when only the newest turn changed

## Claude Scope
### Native Scheduler Work
- Add rollout knob `fusion_scheduler_mode` with:
  - `legacy`
  - `alias_safe`
  - `level_batched`
- Default remains `legacy`.
- Extend native stats with additive fields:
  - `dispatch_count`
  - `submit_count`
  - `barrier_count`
  - `fusion_flush_count`
  - `descriptor_set_update_count`
  - `descriptor_cache_hit_count`
  - `descriptor_cache_miss_count`
  - `alias_conflict_count`
  - `scheduler_mode`
- Implement overlap-aware hazard tracking for:
  - q/k norm temp outputs vs canonical q/k
  - `attn_out` row views vs contiguous `attn_out`
  - KV `row_copy`
  - shortlist and sampler scratch views that alias through different handle shapes
- Reintroduce dependency-level batching only through `alias_safe`.
- Keep `row_copy` immediate until overlap tracking proves it safe.

## Required Commands
- Hard correctness gate:
  - `python -X utf8 tests/test_inference_debug.py gemma3-1b.gguf`
- Desktop decode attribution:
  - `python -X utf8 -m adam.tools.benchmark --model gemma3-1b.gguf --preset decode_regression --profile desktop_discrete --adam-only --skip-llama --skip-ollama`
- Broadcom decode attribution:
  - `python -X utf8 -m adam.tools.benchmark --model gemma3-1b.gguf --preset decode_regression --profile broadcom_v3dv --adam-only --skip-llama --skip-ollama`
- Desktop chat attribution:
  - `python -X utf8 -m adam.tools.benchmark --model gemma3-1b.gguf --preset chat_turn_regression --profile desktop_discrete --adam-only --skip-llama --skip-ollama`
  - `python -X utf8 tests/test_chat_turn_perf.py gemma3-1b.gguf --profile desktop_discrete --max-tokens 16`
- Broadcom chat attribution:
  - `python -X utf8 -m adam.tools.benchmark --model gemma3-1b.gguf --preset chat_turn_regression --profile broadcom_v3dv --adam-only --skip-llama --skip-ollama`
  - `python -X utf8 tests/test_chat_turn_perf.py gemma3-1b.gguf --profile broadcom_v3dv --max-tokens 16`

## Done Criteria
- Decode correctness stays green.
- Normal append-only follow-up turns reuse the full prior canonical prefix unless compaction or reasoning invalidates the cache.
- Desktop decode improves materially over the current baseline.
- Broadcom shows the same directional improvement from the same portable work.
- No agent edits the other agent's owned files.
