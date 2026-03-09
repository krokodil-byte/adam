#!/usr/bin/env python3
"""Prompt rendering smoke tests for metadata-driven chat templates."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [ROOT, os.path.join(ROOT, "adamah-MAIN")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from adamah_chat import prepare_chat_prompt, prepare_chat_messages


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

    print("PASS adaptive chat template rendering")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
