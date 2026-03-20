#!/usr/bin/env python3
"""Prompt rendering smoke tests for metadata-driven chat templates."""
import os

from adam.paths import setup; setup()

from adamah_chat import (
    _assistant_history_message,
    _auto_compaction_enabled,
    _build_compaction_seed_message,
    _decode_path_overrides,
    _chat_reuse_plan,
    _chat_reuse_prefix_len,
    _desired_shader_profile,
    _build_reasoning_request,
    _build_session_system_prompt,
    _clamp_default_max_tokens,
    _clamp_requested_max_tokens,
    _gen_preset_defaults,
    _max_tokens_hard_cap,
    _max_tokens_soft_cap,
    _reasoning_enabled,
    _reasoning_stage_name,
    _render_messages_tokens,
    _resolve_runtime_profile_name,
    _runtime_preset_defaults,
    _runtime_profile_overrides,
    prepare_chat_prompt,
    prepare_chat_messages,
)
from adam.models.engine import ADAMEngine, GenerationConfig, ModelConfig


class DummyTokenizer:
    def __init__(self):
        self.bos_id = 1
        self.eos_id = 2
        self._vocab = ["<unk>", "<s>", "</s>"]
        self._specials = {}

    def encode(self, text, add_bos=True):
        ids = [100 + ord(ch) for ch in text]
        return ([self.bos_id] + ids) if add_bos else ids


class DummyGPU:
    _has_map_matvec_topk_t_xq4_dev = True
    _has_map_matvec_topk_t_xq8_dev = True
    _has_map_matvec_topk_t_xq4_ex_dev = True
    _has_map_matvec_topk_t_xq8_ex_dev = True


def main():
    tok = DummyTokenizer()

    tiny_template = (
        "{% for message in messages %}\n"
        "{% if message['role'] == 'user' %}\n"
        "{{ '<|user|>\\n' + message['content'] + eos_token }}\n"
        "{% endif %}\n"
        "{% if loop.last and add_generation_prompt %}\n"
        "{{ '<|assistant|>' }}\n"
        "{% endif %}\n"
        "{% endfor %}"
    )
    prompt, add_bos = prepare_chat_prompt(
        "hi", "llama", tok, chat_template=tiny_template
    )
    assert prompt == "\n\n<|user|>\nhi</s>\n\n\n<|assistant|>\n\n"
    assert add_bos is True

    gemma_template = "{{ bos_token }}<start_of_turn>user\n{{ messages[0]['content'] }}<end_of_turn>\n<start_of_turn>model\n"
    prompt, add_bos = prepare_chat_prompt(
        "hi", "gemma3", tok, chat_template=gemma_template
    )
    assert prompt == "<s><start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model"
    assert add_bos is False

    convo, add_bos = prepare_chat_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "how are you?"},
        ],
        "llama",
        tok,
        chat_template=tiny_template,
    )
    assert "<|user|>\nhi</s>" in convo
    assert "<|user|>\nhow are you?</s>" in convo
    assert convo.rstrip().endswith("<|assistant|>")
    assert add_bos is True

    assert _chat_reuse_prefix_len([1, 2, 3], [1, 2, 3, 4, 5], True) == 3
    assert _chat_reuse_prefix_len([1, 2, 3], [1, 2, 9, 4, 5], True) == 0
    assert _chat_reuse_prefix_len([1, 2, 3], [1, 2, 3], True) == 0
    assert _chat_reuse_prefix_len([1, 2, 3], [1, 2, 3, 4], False) == 0
    reuse_hit = _chat_reuse_plan([1, 2, 3], [1, 2, 3, 4], True)
    assert reuse_hit["reuse_hit"] is True
    assert reuse_hit["reuse_prefix_tokens"] == 3
    reuse_miss = _chat_reuse_plan([1, 2, 3], [1, 2, 9, 4], True)
    assert reuse_miss["reuse_hit"] is False
    assert reuse_miss["reuse_miss_reason"] == "prefix_mismatch"
    assert reuse_miss["reuse_miss_index"] == 2
    prompt_not_extended = _chat_reuse_plan([1, 2, 3], [1, 2, 3], True)
    assert prompt_not_extended["reuse_miss_reason"] == "prompt_not_extended"

    cfg_stub = type("Cfg", (), {"arch": "gemma3", "chat_template": None})()
    canon_prompt, add_bos, canon_tokens, prep = _render_messages_tokens(
        [
            {"role": "user", "content": "hi"},
            _assistant_history_message("WRONG", [9001, 9002]),
            {"role": "user", "content": "next"},
        ],
        cfg_stub,
        tok,
    )
    assert "<start_of_turn>user" in canon_prompt
    assert add_bos is True
    assert prep["render_s"] >= 0.0
    joined = ",".join(str(tok_id) for tok_id in canon_tokens)
    assert "9001,9002" in joined

    assert _build_session_system_prompt() is None
    assert _build_session_system_prompt("remember this") == (
        "Use the following working notes privately to answer the user's latest message. "
        "Do not mention the notes themselves unless the user explicitly asks for them.\n\n"
        "Working notes for the next reply:\nremember this"
    )
    assert _runtime_preset_defaults("fast")["runtime_mode"] == "fast"
    assert _runtime_preset_defaults("trace")["trace_decode"] is True
    assert _runtime_preset_defaults("desktop_long")["kv_cap"] == 16384
    assert _runtime_preset_defaults("broadcom_fast")["kv_cap"] == 256
    assert _runtime_preset_defaults("broadcom_trace")["runtime_profile"] == "broadcom_v3dv"
    assert _desired_shader_profile({"runtime_mode": "fast", "runtime_profile": "default"}) in ("desktop_discrete", "broadcom_v3dv")
    assert _max_tokens_soft_cap(256) == 128
    assert _max_tokens_hard_cap(256) == 224
    assert _clamp_default_max_tokens(256, 256) == 128
    assert _clamp_requested_max_tokens(512, 256) == 224
    factual = _gen_preset_defaults("factual")
    assert factual["top_k"] == 32
    assert factual["max_tokens"] == 256
    seed = _build_compaction_seed_message("goal: ship the app")
    assert seed["role"] == "system"
    assert "Context from a previous conversation:" in seed["content"]
    assert "goal: ship the app" in seed["content"]
    assert _reasoning_stage_name(0, 4) == "task framing"
    assert _reasoning_stage_name(1, 4) == "answer plan"
    assert _reasoning_stage_name(3, 4) == "final polish"
    req0 = _build_reasoning_request("write a reply", "User: hi", None, 0, 3)
    assert "User request:\nwrite a reply" in req0
    assert "Recent conversation context:\nUser: hi" in req0
    assert "Return notes only." in req0
    req1 = _build_reasoning_request("write a reply", "User: hi", "draft v1", 1, 3)
    assert "Current working notes:\ndraft v1" in req1
    assert "Focus:" in req1
    assert _reasoning_enabled(0) is False
    assert _reasoning_enabled(3) is True
    assert _auto_compaction_enabled(True, 0) is True
    assert _auto_compaction_enabled(True, 2) is True
    engine = ADAMEngine.__new__(ADAMEngine)
    cfg = GenerationConfig(repeat_penalty=1.1)
    assert engine._repeat_history([1, 2, 3], [], cfg) is None
    assert engine._repeat_history([1, 2, 3], [7, 8], cfg) == [7, 8]
    cfg.repeat_on_prompt = True
    assert engine._repeat_history([1, 2, 3], [7, 8], cfg) == [1, 2, 3, 7, 8]
    engine._runtime_profile = "broadcom_v3dv"
    assert engine._can_gpu_merge_approx_shortlist() is True
    engine._runtime_profile = "default"
    assert engine._can_gpu_merge_approx_shortlist() is False
    trace_summary = engine._summarize_decode_trace([
        {
            "step_ms": 10.0,
            "sample_ms": 2.0,
            "forward_ms": 8.0,
            "timing_ms": {"attn": 3.0, "ffn": 2.0, "lm_head": 1.0},
        },
        {
            "step_ms": 14.0,
            "sample_ms": 4.0,
            "forward_ms": 10.0,
            "timing_ms": {"attn": 5.0, "ffn": 3.0, "lm_head": 2.0},
        },
    ])
    assert trace_summary["step_ms_avg"] == 12.0
    assert trace_summary["sample_ms_avg"] == 3.0
    assert trace_summary["attn_ms_avg"] == 4.0
    assert _resolve_runtime_profile_name("default", unified=False) == "desktop_discrete"
    assert _resolve_runtime_profile_name("default", unified=True) == "broadcom_v3dv"
    assert _resolve_runtime_profile_name("trace", unified=True) == "broadcom_v3dv"
    assert _resolve_runtime_profile_name(
        "default",
        unified=False,
        device={"device_name": "NVIDIA GeForce RTX 3070"},
    ) == "desktop_discrete"
    small_prof = _runtime_profile_overrides("default", ModelConfig(), unified=True)
    assert small_prof["gpu_approx_rerank"] is False
    assert small_prof["gpu_fused_rows_per_group"] == 256
    assert small_prof["gpu_approx_partial_k"] == 8
    discrete_prof = _runtime_profile_overrides("default", ModelConfig(), unified=False)
    assert discrete_prof["stream_load"] is False
    assert discrete_prof["gpu_fused_rows_per_group"] == 512
    discrete_decode = _decode_path_overrides("desktop_discrete")
    assert discrete_decode["direct_kv_cache_write"] is True
    assert discrete_decode["experimental_fused_qkv_qk_norm_rope"] is True
    gemma_prof = _runtime_profile_overrides(
        "default",
        ModelConfig(n_vocab=262144),
        unified=True,
    )
    assert gemma_prof["gpu_approx_rerank"] is True
    assert gemma_prof["gpu_fused_rows_per_group"] == 256
    assert gemma_prof["gpu_approx_partial_k"] == 8
    prof = _runtime_profile_overrides(
        "broadcom_v3dv",
        ModelConfig(n_vocab=262144),
        unified=True,
    )
    assert prof["gpu_approx_rerank"] is True
    assert prof["gpu_approx_partial_k"] == 8
    engine = ADAMEngine.__new__(ADAMEngine)
    engine.gpu = DummyGPU()
    engine.cfg = ModelConfig(n_vocab=32768)
    engine._gpu_fused_topk = True
    engine._gpu_approx_rerank = True
    engine._gpu_fused_rows_per_group = 256
    engine._runtime_profile = "desktop_discrete"
    engine._lm_wt_name = "output.weight"
    engine._q4_tensors = {"output.weight"}
    engine._q8_tensors = set()
    engine._trans = {"output.weight": False}
    greedy_cfg = GenerationConfig(temperature=0.0, top_k=1, repeat_penalty=1.0)
    chat_cfg = GenerationConfig(temperature=0.7, top_k=40, repeat_penalty=1.08)
    assert engine._gpu_fused_topk_weight_kind() == "q4"
    assert engine._can_gpu_fused_topk_sample(greedy_cfg) is True
    assert engine._effective_gpu_fused_rows_per_group(greedy_cfg) == 256
    assert engine._effective_gpu_fused_rows_per_group(chat_cfg) == 256
    engine._can_gpu_argmax_sample = lambda cfg: True
    engine._can_gpu_approx_rerank_sample = lambda cfg: True
    engine._can_gpu_topk_sample = lambda cfg: True
    assert engine._select_sampling_mode(greedy_cfg) == "gpu_fused_topk"
    engine._gpu_fused_topk = False
    assert engine._select_sampling_mode(greedy_cfg) == "gpu_argmax"

    print("PASS adaptive chat template rendering")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
