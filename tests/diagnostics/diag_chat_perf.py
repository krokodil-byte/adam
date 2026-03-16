#!/usr/bin/env python3
"""Two-turn chat performance probe with prompt-prep and KV reuse attribution."""
import argparse
import os
import time

from adam.paths import setup; setup()

from adamah_chat import (
    _assistant_history_message,
    _chat_reuse_plan,
    _render_messages_tokens,
    load_model,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure two-turn chat overhead, KV reuse, and prompt prep cost."
    )
    parser.add_argument("model", help="Path to GGUF model file")
    parser.add_argument("--profile", default="desktop_discrete")
    parser.add_argument("--kv-cap", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prompt1", default="hi")
    parser.add_argument(
        "--prompt2",
        default="don't think that's true ahahahah ants are smaller aren't they?",
    )
    return parser.parse_args(argv)


def _render_ids(messages, cfg, tokenizer):
    prompt, _add_bos, tokens, prep = _render_messages_tokens(messages, cfg, tokenizer)
    return prompt, tokens, float(prep.get("render_s", 0.0)), float(prep.get("encode_s", 0.0))


def main(argv=None):
    args = parse_args(argv)
    engine, tokenizer, cfg, GenConfig = load_model(
        args.model,
        startup={
            "runtime_profile": args.profile,
            "stream_load": False,
            "stream_chunk_mb": 64,
            "kv_cap": int(args.kv_cap),
            "gpu_fused_topk": True,
            "gpu_fused_rows_per_group": 512,
        },
    )
    gen_cfg = GenConfig(
        max_tokens=int(args.max_tokens),
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        repeat_penalty=1.0,
        eos_token_ids=(tokenizer.eos_id,),
    )

    turn1_messages = [{"role": "user", "content": args.prompt1}]
    _, turn1_ids, turn1_render_s, turn1_encode_s = _render_ids(turn1_messages, cfg, tokenizer)
    turn1_tokens, turn1_stats = engine.generate(turn1_ids, gen_cfg, stream=False)
    turn1_text = tokenizer.decode(turn1_tokens)

    turn2_messages = turn1_messages + [
        _assistant_history_message(turn1_text, turn1_tokens),
        {"role": "user", "content": args.prompt2},
    ]
    _, turn2_ids, turn2_render_s, turn2_encode_s = _render_ids(turn2_messages, cfg, tokenizer)
    reuse_plan = _chat_reuse_plan(list(turn1_ids) + list(turn1_tokens), turn2_ids, True)
    reuse_prefix = int(reuse_plan.get("reuse_prefix_tokens", 0) or 0)
    turn2_tokens, turn2_stats = engine.generate(
        turn2_ids,
        gen_cfg,
        stream=False,
        reuse_prefix=reuse_prefix,
    )
    turn2_text = tokenizer.decode(turn2_tokens)

    print("")
    print("=== Turn 1 ===")
    print(f"prompt tokens  : {len(turn1_ids)}")
    print(f"render         : {turn1_render_s * 1000.0:.2f} ms")
    print(f"tokenize       : {turn1_encode_s * 1000.0:.2f} ms")
    print(f"prefill        : {turn1_stats.get('prefill_s', 0.0) * 1000.0:.2f} ms")
    print(f"decode         : {turn1_stats.get('decode_s', 0.0) * 1000.0:.2f} ms")
    print(f"decode tps     : {turn1_stats.get('decode_tps', 0.0):.2f}")
    print(f"reply preview  : {turn1_text[:100]!r}")

    print("")
    print("=== Turn 2 ===")
    print(f"prompt tokens  : {len(turn2_ids)}")
    print(f"render         : {turn2_render_s * 1000.0:.2f} ms")
    print(f"tokenize       : {turn2_encode_s * 1000.0:.2f} ms")
    print(f"kv reuse hit   : {'yes' if reuse_plan.get('reuse_hit') else 'no'}")
    print(f"kv reused      : {int(turn2_stats.get('n_prompt_reused', reuse_prefix) or 0)}")
    print(f"reuse miss     : {reuse_plan.get('reuse_miss_reason')}")
    print(f"reuse miss idx : {int(reuse_plan.get('reuse_miss_index', -1) or -1)}")
    print(f"prefilled      : {int(turn2_stats.get('n_prompt_prefilled', 0) or 0)}")
    print(f"prefill        : {turn2_stats.get('prefill_s', 0.0) * 1000.0:.2f} ms")
    print(f"decode         : {turn2_stats.get('decode_s', 0.0) * 1000.0:.2f} ms")
    print(f"decode tps     : {turn2_stats.get('decode_tps', 0.0):.2f}")
    print(f"reply preview  : {turn2_text[:100]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
