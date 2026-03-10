#!/usr/bin/env python3
"""Prompt rendering smoke tests for metadata-driven chat templates."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [ROOT, os.path.join(ROOT, "adamah-MAIN")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from adamah_chat import (
    _auto_compaction_enabled,
    _build_compaction_seed_message,
    _build_reasoning_request,
    _build_session_system_prompt,
    _reasoning_enabled,
    _reasoning_stage_name,
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

    assert _build_session_system_prompt() is None
    assert _build_session_system_prompt("remember this") == (
        "Use the following working notes privately to answer the user's latest message. "
        "Do not mention the notes themselves unless the user explicitly asks for them.\n\n"
        "Working notes for the next reply:\nremember this"
    )
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
    prof = _runtime_profile_overrides("broadcom_v3dv_trace", ModelConfig(), unified=True)
    assert prof["trace_decode"] is True
    assert prof["gpu_approx_rerank"] is False

    print("PASS adaptive chat template rendering")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
