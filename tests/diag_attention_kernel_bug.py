#!/usr/bin/env python3
"""
Reproduce the pure-GPU batched attention bug on Gemma3.

This script disables the engine's CPU attention fallback, runs two prompt
tokens through a 1-layer model, and compares:
  - GPU Q/K/V and KV cache rows
  - CPU attention recomputed from those GPU tensors
  - GPU softmax / attn_out stored in the workspace

Expected on a healthy backend:
  GPU softmax ~= CPU softmax-from-GPU-QK
  GPU attn_out ~= CPU attn_out-from-GPU-QKV
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [ROOT, os.path.join(ROOT, "adamah-MAIN")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from adam.loaders.gguf import GGUFLoader
from adam.tokenizers.gguf_tok import GGUFTokenizer
from adam.models.engine import ADAMEngine, ModelConfig
from adamah_chat import prepare_chat_prompt
import adamah as A


def gather_slot(engine, name):
    _, sz, pos = engine._ws_slots[name]
    locs = np.arange(pos, pos + sz, dtype=np.uint32)
    return engine.gpu.gather(engine.ws_map_id, locs).view(np.float32)


def gather_scores_rows(engine, n_cols):
    if hasattr(engine, "_scores_row_bases"):
        row_bases = np.asarray(engine._scores_row_bases, dtype=np.uint32)
        locs = np.concatenate([
            np.arange(int(base), int(base) + n_cols, dtype=np.uint32)
            for base in row_bases
        ])
        return engine.gpu.gather(engine.ws_map_id, locs).view(np.float32).reshape(len(row_bases), n_cols)
    return gather_slot(engine, 'scores')[:engine.cfg.n_head * n_cols].reshape(engine.cfg.n_head, n_cols)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "gemma3-1b.gguf")
    loader = GGUFLoader(path)
    loader.load(verbose=False)
    cfg = ModelConfig.from_gguf_metadata(loader.metadata, verbose=False)
    tok = GGUFTokenizer(
        vocab=loader.get_tokenizer_vocab(),
        scores=loader.get_tokenizer_scores(),
        bos_id=loader.get_bos_token_id(),
        eos_id=loader.get_eos_token_id(),
        token_types=loader.metadata.get('tokenizer.ggml.token_type', []),
        add_space_prefix=loader.metadata.get('tokenizer.ggml.add_space_prefix', True),
    )

    gpu = A.init()
    engine = ADAMEngine(
        gpu, cfg, loader.tensors,
        raw_blocks=loader.raw_blocks,
        tensor_types=loader.tensor_types,
        adamah_mod=A,
        verbose=False,
        cpu_attention_fallback=False,
    )
    engine.cfg.n_layer = 1

    text, add_bos = prepare_chat_prompt(
        "What is 2 + 2?", cfg.arch, tok, chat_template=getattr(cfg, "chat_template", None)
    )
    tokens = tok.encode(text, add_bos=add_bos)
    engine.reset()
    engine._forward(tokens[0], 0)
    engine._forward(tokens[1], 1)

    c = engine.cfg
    q = gather_slot(engine, 'q').reshape(c.n_head, c.head_dim)
    attn_out_gpu = gather_slot(engine, 'attn_out').reshape(c.n_head, c.head_dim_kv)
    softmax_gpu = gather_scores_rows(engine, 2)

    k_cache = engine.gpu.gather(
        engine.ws_map_id,
        np.arange(engine._kc_k_base[0], engine._kc_k_base[0] + 2 * c.head_dim_kv, dtype=np.uint32),
    ).view(np.float32).reshape(2, c.head_dim_kv)
    v_cache = engine.gpu.gather(
        engine.ws_map_id,
        np.arange(engine._kc_v_base[0], engine._kc_v_base[0] + 2 * c.head_dim_kv, dtype=np.uint32),
    ).view(np.float32).reshape(2, c.head_dim_kv)

    scale = np.float32(1.0 / np.sqrt(c.head_dim))
    scores = np.stack([q[h] @ k_cache.T * scale for h in range(c.n_head)])
    if c.attn_softcap > 0:
        cap = np.float32(c.attn_softcap)
        scores = np.tanh(scores / cap) * cap
    scores -= scores.max(axis=1, keepdims=True)
    softmax_cpu = np.exp(scores).astype(np.float32)
    softmax_cpu /= softmax_cpu.sum(axis=1, keepdims=True)
    attn_out_cpu = np.stack([softmax_cpu[h] @ v_cache for h in range(c.n_head)])

    sm_rmse = float(np.sqrt(np.mean((softmax_gpu - softmax_cpu) ** 2)))
    ao_rmse = float(np.sqrt(np.mean((attn_out_gpu - attn_out_cpu) ** 2)))
    print("softmax_gpu")
    print(softmax_gpu)
    print("softmax_cpu_from_gpu_qk")
    print(softmax_cpu)
    print(f"softmax_rmse={sm_rmse:.6f}")
    print()
    print(f"attn_out_rmse={ao_rmse:.6f}")

    ok = sm_rmse < 1e-3 and ao_rmse < 1e-2
    if ok:
        print("PASS pure-GPU attention matches CPU recomputation")
        return 0
    print("FAIL pure-GPU attention diverges from CPU recomputation")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
