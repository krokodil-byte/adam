"""
ADAM — Inference Engine v10
============================
Native Q4 GPU pipeline on ADAMAH 5.2.
Loads any GGUF transformer model and runs inference via Vulkan GPU (ADAMAH).

Key design choices:
  - batch_begin()/batch_end() wraps the entire layer loop → single fence wait per step
  - KV cache is per-head-contiguous: [n_head_kv, kv_cap, head_dim_kv] per layer
    so matmul_t can use stride=head_dim_kv (correct for any GQA ratio)
  - n_ops=n_head for attention matmuls → handles GQA (each query head picks its KV group)
  - output.weight cached as float32 at init → no per-token conversion
"""
import numpy as np
import os
import time, sys
from typing import Optional, List, Dict
from dataclasses import dataclass

@dataclass
class GenerationConfig:
    max_tokens: int = 128; temperature: float = 0.8; top_k: int = 40
    top_p: float = 0.95; repeat_penalty: float = 1.1; seed: Optional[int] = None
    repeat_on_prompt: bool = False
    eos_token_ids: tuple = (1,)  # Token IDs that stop generation

@dataclass
class ModelConfig:
    arch: str = "llama"
    n_vocab: int = 32000; n_ctx: int = 4096; n_embd: int = 4096
    n_head: int = 32; n_head_kv: int = 32; n_layer: int = 32
    n_ff: int = 11008; head_dim: int = 128; head_dim_kv: int = 128
    norm_eps: float = 1e-6
    rope_base_global: float = 10000.0; rope_base_local: float = 10000.0
    sliding_window: int = 0
    chat_template: Optional[str] = None
    ffn_act: str = 'silu'    # 'silu' | 'gelu'
    emb_scale: float = 1.0   # embedding scale (Gemma: sqrt(n_embd), others: 1.0)
    attn_softcap: float = 0.0  # attention logit soft-cap (Gemma3: 50.0, 0=disabled)
    final_softcap: float = 0.0 # final logit soft-cap (Gemma3: 30.0, 0=disabled)

    # GGML feed_forward.activation_type codes → act name
    _GGML_ACT = {8: 'gelu', 9: 'gelu', 10: 'silu', 6: 'relu'}
    # Architectures that use GELU gate (GeGLU) — others use SiLU (SwiGLU)
    _GELU_ARCHS = {'gemma', 'gemma2', 'gemma3', 'gpt2', 'gptj', 'gptneox',
                   'falcon', 'starcoder', 'starcoder2', 'refact'}

    def is_global(self, L):
        if self.arch == "gemma3": return (L + 1) % 6 == 0
        return True

    @classmethod
    def from_gguf_metadata(cls, metadata, verbose=True):
        arch = metadata.get('general.architecture', 'llama')
        c = cls(arch=arch)
        g = lambda key, default: metadata.get(f'{arch}.{key}', metadata.get(key, default))
        c.n_ctx = g('context_length', 4096)
        c.n_layer = g('block_count', 32)
        c.n_embd = g('embedding_length', 4096)
        c.n_ff = g('feed_forward_length', c.n_embd * 4)
        c.n_head = g('attention.head_count', 32)
        c.n_head_kv = g('attention.head_count_kv', c.n_head)
        c.norm_eps = g('attention.layer_norm_rms_epsilon', 1e-6)
        c.n_vocab = g('vocab_size', metadata.get('tokenizer.ggml.tokens', [None]))
        if isinstance(c.n_vocab, list): c.n_vocab = len(c.n_vocab)
        c.chat_template = metadata.get('tokenizer.chat_template')
        c.head_dim = c.n_embd // c.n_head if c.n_head > 0 else 128
        c.head_dim_kv = c.n_embd // c.n_head_kv if c.n_head_kv > 0 else c.head_dim
        if arch == 'gemma3':
            c.sliding_window = g('attention.sliding_window', 512)
            c.rope_base_global = g('rope.freq_base', 1e6)
            # Gemma3-1B uses the same freq_base for all layers.
            # Fall back to rope_base_global if no separate local key is present.
            c.rope_base_local = g('rope.local_freq_base', c.rope_base_global)
        else:
            c.sliding_window = g('attention.sliding_window', 0)
            c.rope_base_global = g('rope.freq_base', 10000.0)
            c.rope_base_local = c.rope_base_global

        # FFN activation: prefer GGUF metadata key, fall back to arch heuristic
        ggml_act = g('feed_forward.activation_type', None)
        if ggml_act is not None:
            c.ffn_act = cls._GGML_ACT.get(ggml_act, 'silu')
        else:
            c.ffn_act = 'gelu' if arch in cls._GELU_ARCHS else 'silu'

        # Embedding scale: Gemma family multiplies embeddings by sqrt(n_embd)
        import math
        c.emb_scale = math.sqrt(c.n_embd) if arch.startswith('gemma') else 1.0

        # Logit soft-capping (Gemma 3): tanh(x / cap) * cap
        c.attn_softcap = float(g('attention.logit_softcapping', 0.0))
        c.final_softcap = float(g('logit_softcapping', 0.0))
        # Hardcoded Gemma3 fallbacks — all Gemma3 sizes use these caps;
        # some GGUF converters omit the metadata keys.
        if arch == 'gemma3':
            if c.attn_softcap == 0.0: c.attn_softcap = 50.0
            if c.final_softcap == 0.0: c.final_softcap = 30.0

        if verbose:
            name = metadata.get('general.name', arch)
            caps = (f" attn_cap={c.attn_softcap} final_cap={c.final_softcap}"
                    if c.attn_softcap or c.final_softcap else "")
            print(f"[CFG] {name} ({arch}): L={c.n_layer} E={c.n_embd} H={c.n_head} "
                  f"KV={c.n_head_kv} FF={c.n_ff} V={c.n_vocab} "
                  f"act={c.ffn_act} emb_scale={c.emb_scale:.2f}{caps}")
        return c


class ADAMEngine:
    """ADAM inference engine — runs any GGUF transformer on ADAMAH Vulkan GPU."""

    KV_CAP_DEFAULT = 1024
    SAMPLE_TOPK_MAX = 64
    SAMPLE_APPROX_SHORTLIST_MAX = 256
    SAMPLE_REPEAT_MAX = 64
    SAMPLE_FUSED_ROWS_PER_GROUP = 256
    SAMPLE_FUSED_ROWS_PER_GROUP_NON_GREEDY = 512

    @classmethod
    def estimate_persistent_gpu_bytes(cls, cfg: ModelConfig,
                                      tensor_shapes: Dict[str, tuple],
                                      tensor_types: Dict[str, int],
                                      force_f32: bool = False,
                                      kv_cap: Optional[int] = None,
                                      gpu_tied_lm_head: bool = True,
                                      gpu_approx_rerank: bool = False,
                                      gpu_approx_partial_k: Optional[int] = None,
                                      gpu_fused_rows_per_group: Optional[int] = None) -> Dict[str, int]:
        kv_cap = int(kv_cap if kv_cap is not None else cls.KV_CAP_DEFAULT)
        rows_per_group = int(gpu_fused_rows_per_group or cls.SAMPLE_FUSED_ROWS_PER_GROUP)
        approx_partial_k = max(
            1,
            min(
                cls.SAMPLE_TOPK_MAX,
                int(gpu_approx_partial_k if gpu_approx_partial_k is not None else cls.SAMPLE_TOPK_MAX),
            ),
        )
        tied_lm_head = (
            gpu_tied_lm_head and
            'output.weight' not in tensor_shapes and
            'token_embd.weight' in tensor_shapes
        )

        q4_elems = 0
        q8_elems = 0
        f32_weight_elems = 0
        norm_elems = 0
        lm_q4_approx_elems = 0
        wt_matrix_names = []
        norm_names = []

        for name in sorted(tensor_shapes):
            sh = tuple(tensor_shapes[name])
            if name == 'token_embd.weight' and not tied_lm_head:
                continue
            if len(sh) == 2 and (name.startswith('blk.') or name == 'output.weight' or
                                 (tied_lm_head and name == 'token_embd.weight')):
                wt_matrix_names.append(name)
            elif len(sh) == 1 or name == 'output_norm.weight':
                norm_names.append(name)
            else:
                (wt_matrix_names if len(sh) == 2 else norm_names).append(name)

        for name in wt_matrix_names:
            sh = tuple(tensor_shapes[name])
            elems = 1
            for s in sh:
                elems *= int(s)
            is_lm_head_candidate = (
                name == 'output.weight' or (tied_lm_head and name == 'token_embd.weight')
            )
            trans = (
                name == 'token_embd.weight' and len(sh) == 2 and
                sh[0] == cfg.n_embd and sh[1] != cfg.n_embd
            )
            if force_f32:
                f32_weight_elems += elems
            elif tensor_types.get(name) in cls._GGML_Q8_MAP:
                q8_elems += elems
                if gpu_approx_rerank and is_lm_head_candidate and not trans:
                    lm_q4_approx_elems += elems
            else:
                q4_elems += elems

        for name in norm_names:
            elems = 1
            for s in tensor_shapes[name]:
                elems *= int(s)
            norm_elems += elems

        sc_sz = cfg.n_head * kv_cap
        pos = sc_sz + 1
        if cfg.emb_scale != 1.0:
            pos += 1
        if cfg.attn_softcap > 0:
            pos += 1
        if cfg.final_softcap > 0:
            pos += 1

        sample_fused_groups = (cfg.n_vocab + rows_per_group - 1) // rows_per_group
        sample_fused_partial_cap = sample_fused_groups * (
            approx_partial_k if gpu_approx_rerank else cls.SAMPLE_TOPK_MAX
        )
        slots = {
            'hidden': cfg.n_embd,
            'normed': cfg.n_embd,
            'q': cfg.n_head * cfg.head_dim,
            'k': cfg.n_head_kv * cfg.head_dim_kv,
            'v': cfg.n_head_kv * cfg.head_dim_kv,
            'attn_out': cfg.n_head * cfg.head_dim_kv,
            'o_proj': cfg.n_embd,
            'gate': cfg.n_ff,
            'up': cfg.n_ff,
            'act': cfg.n_ff,
            'ffn_out': cfg.n_embd,
            'logits': cfg.n_vocab,
            'sample_token': 1,
            'sample_rand': 1,
            'sample_topk_idx': cls.SAMPLE_TOPK_MAX,
            'sample_topk_val': cls.SAMPLE_TOPK_MAX,
            'sample_topk_prob': cls.SAMPLE_TOPK_MAX,
            'sample_short_sel': cls.SAMPLE_APPROX_SHORTLIST_MAX,
            'sample_exact_idx': cls.SAMPLE_APPROX_SHORTLIST_MAX,
            'sample_exact_val': cls.SAMPLE_APPROX_SHORTLIST_MAX,
            'sample_fused_idx': sample_fused_partial_cap,
            'sample_fused_val': sample_fused_partial_cap,
            'normed2': cfg.n_embd,
        }
        pos += sum(slots.values())
        pos += norm_elems
        if cfg.head_dim_kv > 0 and pos % cfg.head_dim_kv != 0:
            pos += cfg.head_dim_kv - (pos % cfg.head_dim_kv)
        kv_elems = 2 * cfg.n_layer * cfg.n_head_kv * kv_cap * cfg.head_dim_kv
        pos += kv_elems

        workspace_bytes = pos * 4
        q4_total_elems = q4_elems + lm_q4_approx_elems
        q4_bytes = 0 if force_f32 else ((q4_total_elems + 1) // 2) + (((q4_total_elems + cls.Q4_GROUP_SIZE - 1) // cls.Q4_GROUP_SIZE) * 8)
        q8_bytes = q8_elems + (((q8_elems + cls.Q8_GROUP_SIZE - 1) // cls.Q8_GROUP_SIZE) * 8)
        f32_weight_bytes = f32_weight_elems * 4
        total_bytes = workspace_bytes + q4_bytes + q8_bytes + f32_weight_bytes
        return {
            'workspace_bytes': workspace_bytes,
            'q4_bytes': q4_bytes,
            'q8_bytes': q8_bytes,
            'f32_weight_bytes': f32_weight_bytes,
            'kv_bytes': kv_elems * 4,
            'total_bytes': total_bytes,
        }

    def __init__(self, gpu, cfg: ModelConfig, tensors: Dict[str, np.ndarray],
                 adamah_mod=None, verbose=True, **kw):
        self.gpu = gpu; self.cfg = cfg; self.tensors = dict(tensors)
        self.A = adamah_mod; self.verbose = verbose
        self.raw_blocks = dict(kw.get('raw_blocks', {}))
        self.tensor_types = kw.get('tensor_types', {})
        self.tensor_shapes = dict(kw.get(
            'tensor_shapes', {name: tuple(arr.shape) for name, arr in self.tensors.items()}))
        self._tensor_loader = kw.get('tensor_loader')
        self._stream_chunk_mb = max(1, int(kw.get('stream_chunk_mb', 32)))
        self._kv_cap = int(kw.get('kv_cap', self.KV_CAP_DEFAULT))
        self._force_f32 = bool(kw.get('force_f32', False))
        self._cpu_attention_fallback = bool(kw.get('cpu_attention_fallback', False))
        self._production_mode = bool(kw.get('production_mode', False))
        self._gpu_tied_lm_head = bool(kw.get('gpu_tied_lm_head', True))
        self._gpu_approx_rerank = bool(kw.get('gpu_approx_rerank', False))
        self._gpu_fused_topk = bool(kw.get('gpu_fused_topk', True))
        self._direct_kv_cache_write = bool(kw.get('direct_kv_cache_write', False))
        self._gpu_approx_partial_k = max(
            1, min(self.SAMPLE_TOPK_MAX,
                   int(kw.get('gpu_approx_partial_k', self.SAMPLE_TOPK_MAX))))
        self._gpu_fused_rows_per_group = max(
            1, int(kw.get('gpu_fused_rows_per_group', self.SAMPLE_FUSED_ROWS_PER_GROUP)))
        if self._production_mode and self._cpu_attention_fallback:
            raise ValueError("production_mode requires GPU attention; cpu_attention_fallback is disabled")
        self._host_state_pruned = False
        self._timing_enabled = False
        self._trace_decode = False
        self._detect_dims()
        self._blk_names = tuple(f'blk.{L}' for L in range(self.cfg.n_layer))
        self._build_layout()
        self._alloc_workspace()
        self._upload_all()
        self._alloc_kv_cache()
        self._upload_locs()
        self._cache_lm_weight()
        # B12 Q8 re-quant (in _register_full_decode) needs self.tensors — run before release.
        self._full_decode_active = False
        self._full_decode_registered = False
        self._register_full_decode()
        if self._production_mode:
            self.release_host_state()
        self.timing = {}; self._reset_timing()
        # Detect runtime profile for hardware-specific optimizations
        self._runtime_profile = str(
            os.environ.get('ADAMAH_SHADER_PROFILE', os.environ.get('ADAM_RUNTIME_PROFILE', ''))
        ).strip().lower()
        self._is_broadcom = self._runtime_profile.startswith('broadcom_v3dv')
        self._is_integrated = getattr(self.gpu, '_is_integrated_gpu', False)
        # Auto-tune for Broadcom / integrated GPU (Raspberry Pi)
        if self._is_broadcom or self._is_integrated:
            # Default to level_batched fusion scheduler (enables barrier coalescing)
            default_scheduler = 'level_batched'
            # Larger rows_per_group reduces dispatch count for LM head sampling
            if self._gpu_fused_rows_per_group == self.SAMPLE_FUSED_ROWS_PER_GROUP:
                self._gpu_fused_rows_per_group = 1024
            if self.verbose:
                print(f"[GPU] Broadcom/integrated GPU detected — "
                      f"auto-tuning: scheduler={default_scheduler}, "
                      f"rows_per_group={self._gpu_fused_rows_per_group}")
        else:
            default_scheduler = 'legacy'
        self._fusion_scheduler_mode = str(
            kw.get('fusion_scheduler_mode', os.environ.get('ADAM_FUSION_SCHEDULER_MODE', default_scheduler))
        ).strip().lower() or default_scheduler
        if hasattr(self.gpu, 'fusion_set_scheduler_mode'):
            self.gpu.fusion_set_scheduler_mode(self._fusion_scheduler_mode)
        if hasattr(self.gpu, 'fusion_get_scheduler_mode'):
            self._fusion_scheduler_mode = str(self.gpu.fusion_get_scheduler_mode()).strip().lower()
        # The backend now flushes queued fusion ops into the active batch before
        # any immediate op and before batch_end(), so decode can safely keep
        # fusion enabled and let unary/binary/broadcast ops batch together.
        self.gpu.fusion_enable(True)
        self._decode_plan_active = False
        self._decode_plan_id = 0
        self._build_decode_plan()
        if self.verbose and self._cpu_attention_fallback:
            print("[GPU] Attention: CPU fallback enabled (workaround for batched attention kernel bug)")

    def _shape_of(self, name):
        if name in self.tensor_shapes:
            return tuple(self.tensor_shapes[name])
        if name in self.tensors:
            return tuple(self.tensors[name].shape)
        raise KeyError(name)

    def _size_of(self, name):
        ne = 1
        for s in self._shape_of(name):
            ne *= int(s)
        return ne

    def _register_loc_span(self, handle, map_id, start, count):
        if handle and hasattr(self.gpu, 'register_loc_span'):
            self.gpu.register_loc_span(int(handle), int(map_id), int(start), int(count))

    def _register_row_base_span(self, handle, map_id, start, count, stride, row_size):
        if handle and hasattr(self.gpu, 'register_row_base_span'):
            self.gpu.register_row_base_span(
                int(handle), int(map_id), int(start), int(count), int(stride), int(row_size)
            )

    def _fusion_alias_fast_enabled(self):
        return self._fusion_scheduler_mode != 'legacy'

    def _weight_kind(self, wt_name):
        if wt_name in self._q8_tensors:
            return 'q8'
        if wt_name in self._q4_tensors:
            return 'q4'
        return 'f32'

    def _all_tensor_names(self):
        if self.tensor_shapes:
            return list(self.tensor_shapes.keys())
        return list(self.tensors.keys())

    def _load_tensor_f32(self, name, keep=False):
        if name in self.tensors:
            return self.tensors[name]
        if self._tensor_loader is None:
            raise KeyError(name)
        arr = self._tensor_loader.load_tensor_f32(name)
        if keep:
            self.tensors[name] = arr
        return arr

    def _load_raw_block(self, name, keep=False):
        if name in self.raw_blocks:
            return self.raw_blocks[name]
        if self._tensor_loader is None:
            raise KeyError(name)
        raw = self._tensor_loader.load_tensor_raw(name)
        if keep:
            self.raw_blocks[name] = raw
        return raw

    def _iter_tensor_chunks(self, name, include_raw=False, include_f32=True):
        if self._tensor_loader is not None and name not in self.tensors and name not in self.raw_blocks:
            yield from self._tensor_loader.iter_tensor_chunks(
                name,
                max_chunk_mb=self._stream_chunk_mb,
                include_raw=include_raw,
                include_f32=include_f32,
            )
            return

        raw = self.raw_blocks.get(name) if include_raw else None
        arr = self.tensors.get(name) if include_f32 else None
        if arr is None and include_f32:
            arr = self._load_tensor_f32(name)
        if raw is None and include_raw:
            raw = self._load_raw_block(name)
        flat = (np.ascontiguousarray(arr.reshape(-1), dtype=np.float32)
                if arr is not None else None)
        yield 0, self._size_of(name), raw, flat

    # ============================================================
    # Auto-detect from tensor shapes
    # ============================================================
    def _detect_dims(self):
        c = self.cfg
        for n in self._all_tensor_names():
            sh = self._shape_of(n)
            if 'attn_q.weight' in n and len(sh) == 2:
                q_out = sh[1] if sh[0] == c.n_embd else sh[0]
                c.head_dim = q_out // c.n_head
                break
        for n in self._all_tensor_names():
            sh = self._shape_of(n)
            if 'attn_k.weight' in n and len(sh) == 2:
                k_out = sh[1] if sh[0] == c.n_embd else sh[0]
                c.head_dim_kv = k_out // c.n_head_kv
                break
        if 'token_embd.weight' in self.tensor_shapes:
            sh = self._shape_of('token_embd.weight')
            c.n_vocab = max(sh); c.n_embd = min(sh)
        for n in self._all_tensor_names():
            sh = self._shape_of(n)
            if 'ffn_gate.weight' in n and len(sh) == 2:
                c.n_ff = max(sh) if min(sh) == c.n_embd else min(sh)
                break
        if self.verbose:
            q_dim = c.n_head * c.head_dim; k_dim = c.n_head_kv * c.head_dim_kv
            print(f"[CFG] E={c.n_embd} H={c.n_head}(d={c.head_dim}) "
                  f"KV={c.n_head_kv}(d={c.head_dim_kv}) FF={c.n_ff} "
                  f"L={c.n_layer} V={c.n_vocab} q={q_dim} k={k_dim}")

    # ============================================================
    # Native Q4 / Q8 weight layout
    # map 1 = Q4 (Q4_K and everything else)
    # map 2 = Q8 (Q8_0 type=8 and Q5_0 type=6 — stored losslessly as int8)
    # ============================================================
    Q4_GROUP_SIZE = 32
    Q8_GROUP_SIZE = 32
    # GGML types routed to the Q8 map (stored as int8, full precision)
    _GGML_Q8_MAP = frozenset({6, 8})   # Q5_0=6, Q8_0=8

    def _build_layout(self):
        c = self.cfg
        self._off = {}; self._trans = {}; self.tensor_map_id = {}
        self._cpu_only = set(); self._q4_tensors = set(); self._norm_tensors = set()
        self._q8_tensors = set()
        self._lm_q4_approx_name = None
        self._lm_q4_approx_off = None
        wt_matrix_names = []; norm_names = []
        tied_lm_head = (
            self._gpu_tied_lm_head and
            'output.weight' not in self.tensor_shapes and
            'token_embd.weight' in self.tensor_shapes
        )

        def _qkv_sort_key(n):
            if '.attn_q.' in n: return n.replace('.attn_q.', '.attn_z0.')
            if '.attn_k.' in n: return n.replace('.attn_k.', '.attn_z1.')
            if '.attn_v.' in n: return n.replace('.attn_v.', '.attn_z2.')
            return n

        for name in sorted(self._all_tensor_names(), key=_qkv_sort_key):
            sh = self._shape_of(name)
            # token_embd always stays CPU-backed for the per-token embedding lookup.
            # If the model is weight-tied (no output.weight tensor), also upload it
            # to a GPU weight map so the LM head can stay on device.
            if name == 'token_embd.weight':
                self._cpu_only.add(name)
                if not tied_lm_head:
                    self._trans[name] = (len(sh) == 2 and sh[0] == c.n_embd and
                                         sh[1] != c.n_embd)
                    if self.verbose:
                        print(f"[GPU] CPU-only: '{name}' ({self._size_of(name):,} elems)")
                    continue
                if self.verbose:
                    print(f"[GPU] CPU lookup + GPU LM head: '{name}' ({self._size_of(name):,} elems)")
            if len(sh) == 2 and (name.startswith('blk.') or name == 'output.weight' or
                                 (tied_lm_head and name == 'token_embd.weight')):
                wt_matrix_names.append(name)
            elif len(sh) == 1 or name == 'output_norm.weight':
                norm_names.append(name)
            else:
                (wt_matrix_names if len(sh) == 2 else norm_names).append(name)

        q4_pos = 0; q8_pos = 0
        self._f32_wt_tensors = set()
        for name in wt_matrix_names:
            sh = self._shape_of(name)
            is_lm_head_candidate = (name == 'output.weight' or
                                    (tied_lm_head and name == 'token_embd.weight'))
            self._trans[name] = (
                name == 'token_embd.weight' and len(sh) == 2 and
                sh[0] == c.n_embd and sh[1] != c.n_embd
            )
            if self._force_f32:
                self._off[name] = q4_pos; self.tensor_map_id[name] = 1
                self._f32_wt_tensors.add(name)
                q4_pos += self._size_of(name)
            elif self.tensor_types.get(name) in self._GGML_Q8_MAP:
                # Q5_0 / Q8_0 → Q8 map (map 2), full int8 precision
                self._off[name] = q8_pos; self.tensor_map_id[name] = 2
                self._q8_tensors.add(name)
                q8_pos += self._size_of(name)
                if (self._gpu_approx_rerank and is_lm_head_candidate and
                        self._lm_q4_approx_off is None and not self._trans[name]):
                    self._lm_q4_approx_name = name
                    self._lm_q4_approx_off = q4_pos
                    q4_pos += self._size_of(name)
            else:
                # Q4_K and all other types → Q4 map (map 1)
                self._off[name] = q4_pos; self.tensor_map_id[name] = 1
                self._q4_tensors.add(name)
                q4_pos += self._size_of(name)

        self._q4_total_elems = 0 if self._force_f32 else q4_pos
        self._q8_total_elems = q8_pos
        self._f32_wt_total_elems = q4_pos if self._force_f32 else 0
        self._norm_names_ordered = norm_names
        self._norm_sizes = {n: self._size_of(n) for n in norm_names}

        if self.verbose:
            if self._force_f32:
                print(f"[GPU] F32 weights: {len(wt_matrix_names)} tensors "
                      f"{q4_pos * 4 / 1e6:.0f} MB")
            else:
                n4g = (q4_pos + self.Q4_GROUP_SIZE - 1) // self.Q4_GROUP_SIZE
                n8g = (q8_pos + self.Q8_GROUP_SIZE - 1) // self.Q8_GROUP_SIZE
                print(f"[GPU] Q4 map: {len(self._q4_tensors)} tensors {q4_pos*0.5/1e6:.0f} MB + "
                      f"{n4g*8/1e6:.0f} MB qparams")
                if self._q8_tensors:
                    print(f"[GPU] Q8 map: {len(self._q8_tensors)} tensors {q8_pos/1e6:.0f} MB + "
                          f"{n8g*8/1e6:.0f} MB qparams")
            print(f"[GPU] F32 norms: {len(norm_names)} tensors")

    def _upload_all(self):
        c = self.cfg; g = self.gpu
        g.set_dtype(self.A.DTYPE_F32)
        g.map_create(0, 1, self._ws_total_elems)
        self.ws_map_id = 0
        if self.verbose:
            print(f"[GPU] Workspace map 0: {self._ws_total_elems:,} F32 elems "
                  f"({self._ws_total_elems * 4 / 1e6:.0f} MB)")

        self.wt_map_id = 1
        if self._force_f32:
            g.map_init(1, 4, 1, self._f32_wt_total_elems)
            if self.verbose:
                print(f"[GPU] F32 weight map 1: {self._f32_wt_total_elems:,} elems "
                      f"({self._f32_wt_total_elems * 4 / 1e6:.0f} MB)")
            t0 = time.perf_counter()
            for name in sorted(self._f32_wt_tensors):
                off = self._off[name]
                for elem_off, take, _, data in self._iter_tensor_chunks(name, include_f32=True):
                    locs = np.arange(off + elem_off, off + elem_off + take, dtype=np.uint32)
                    g.scatter(self.wt_map_id, locs, data)
            if self.verbose:
                print(f"[GPU] F32 weight upload: {len(self._f32_wt_tensors)} tensors in "
                      f"{time.perf_counter()-t0:.1f}s")
        else:
            g.map_create_typed(1, self.A.DTYPE_Q4, 1, self._q4_total_elems, self.Q4_GROUP_SIZE)
            if self.verbose:
                print(f"[GPU] Q4 weight map 1: {self._q4_total_elems:,} elems "
                      f"({self._q4_total_elems * 0.5 / 1e6:.0f} MB)")
            t0 = time.perf_counter()
            n_groups = (self._q4_total_elems + self.Q4_GROUP_SIZE - 1) // self.Q4_GROUP_SIZE
            all_scales = np.ones(n_groups, dtype=np.float32)
            all_zeros = np.zeros(n_groups, dtype=np.float32)
            _GGML_Q4K = 12; n_direct = 0
            for name in sorted(self._q4_tensors):
                off = self._off[name]
                if self.tensor_types.get(name) == _GGML_Q4K and (
                        name in self.raw_blocks or self._tensor_loader is not None):
                    # Q4_K: extract original sub-block scales/zeros directly from raw bytes.
                    # Q4_K decode: val = sc*nibble - mn  →  ADAMAH: scale=sc, zero=-mn
                    # Scatter with these exact qparams recovers original nibbles bit-exactly.
                    for elem_off, take, raw, _ in self._iter_tensor_chunks(
                            name, include_raw=True, include_f32=False):
                        n_blk = take // 256
                        blks = np.frombuffer(raw, dtype=np.uint8)[:n_blk * 144].reshape(n_blk, 144)
                        d = np.ascontiguousarray(blks[:, 0:2]).view(np.float16).reshape(n_blk).astype(np.float32)
                        dm = np.ascontiguousarray(blks[:, 2:4]).view(np.float16).reshape(n_blk).astype(np.float32)
                        rs = blks[:, 4:16].astype(np.int32)
                        s2 = np.empty((n_blk, 8), np.float32)
                        m2 = np.empty((n_blk, 8), np.float32)
                        s2[:, 0:4] = rs[:, 0:4] & 0x3F;  m2[:, 0:4] = rs[:, 4:8] & 0x3F
                        s2[:, 4:8] = (rs[:, 8:12] & 0x0F) | ((rs[:, 0:4] >> 6) << 4)
                        m2[:, 4:8] = (rs[:, 8:12] >> 4)   | ((rs[:, 4:8] >> 6) << 4)
                        sc = d[:, None] * s2; mn = dm[:, None] * m2
                        gi = (off + elem_off) // self.Q4_GROUP_SIZE
                        all_scales[gi:gi + n_blk * 8] = np.where(sc.reshape(-1) > 0, sc.reshape(-1), 1.0)
                        all_zeros[gi:gi + n_blk * 8] = (-mn).reshape(-1)
                    n_direct += 1
                else:
                    for elem_off, take, _, data in self._iter_tensor_chunks(name, include_f32=True):
                        for gi_e in range(0, take, self.Q4_GROUP_SIZE):
                            chunk = data[gi_e:gi_e + self.Q4_GROUP_SIZE]
                            vmin, vmax = float(chunk.min()), float(chunk.max())
                            scale = (vmax - vmin) / 15.0 if vmax > vmin else 1.0
                            group_idx = (off + elem_off + gi_e) // self.Q4_GROUP_SIZE
                            all_scales[group_idx] = scale
                            all_zeros[group_idx] = vmin
            if self.verbose and n_direct:
                print(f"[GPU] Q4_K direct qparams: {n_direct}/{len(self._q4_tensors)} tensors")
            if self._lm_q4_approx_off is not None and self._lm_q4_approx_name is not None:
                off = self._lm_q4_approx_off
                for elem_off, take, _, data in self._iter_tensor_chunks(
                        self._lm_q4_approx_name, include_f32=True):
                    for gi_e in range(0, take, self.Q4_GROUP_SIZE):
                        chunk = data[gi_e:gi_e + self.Q4_GROUP_SIZE]
                        vmin, vmax = float(chunk.min()), float(chunk.max())
                        scale = (vmax - vmin) / 15.0 if vmax > vmin else 1.0
                        group_idx = (off + elem_off + gi_e) // self.Q4_GROUP_SIZE
                        all_scales[group_idx] = scale
                        all_zeros[group_idx] = vmin
            g.set_qparams(self.wt_map_id, all_scales, all_zeros)
            for name in sorted(self._q4_tensors):
                off = self._off[name]
                for elem_off, take, _, data in self._iter_tensor_chunks(name, include_f32=True):
                    locs = np.arange(off + elem_off, off + elem_off + take, dtype=np.uint32)
                    g.scatter(self.wt_map_id, locs, data)
            if self._lm_q4_approx_off is not None and self._lm_q4_approx_name is not None:
                off = self._lm_q4_approx_off
                for elem_off, take, _, data in self._iter_tensor_chunks(
                        self._lm_q4_approx_name, include_f32=True):
                    locs = np.arange(off + elem_off, off + elem_off + take, dtype=np.uint32)
                    g.scatter(self.wt_map_id, locs, data)
            if self.verbose:
                print(f"[GPU] Q4 upload: {len(self._q4_tensors)} tensors in "
                      f"{time.perf_counter()-t0:.1f}s")

            # ---- Q8 weight map (map 2): Q8_0 and Q5_0 stored as int8 ----
            self.q8_map_id = 2
            if self._q8_tensors:
                g.map_create_typed(2, self.A.DTYPE_Q8, 1, self._q8_total_elems, self.Q8_GROUP_SIZE)
                if self.verbose:
                    print(f"[GPU] Q8 weight map 2: {self._q8_total_elems:,} elems "
                          f"({self._q8_total_elems / 1e6:.0f} MB)")
                t0 = time.perf_counter()
                n8g = (self._q8_total_elems + self.Q8_GROUP_SIZE - 1) // self.Q8_GROUP_SIZE
                q8_scales = np.ones(n8g, dtype=np.float32)
                _GGML_Q8_0 = 8; _GGML_Q5_0 = 6
                # Q8 scatter (word_size=1) DMA-copies raw bytes — must pre-quantize to uint8.
                # Encoding: uint8_val = int8_val + 128 (so 128 → 0, 0 → -128*scale, 255 → 127*scale).
                # xq8 shader reads unsigned: val = float(u8) * scale + zero, with zero = -128 * scale.
                for name in sorted(self._q8_tensors):
                    off = self._off[name]
                    t_type = self.tensor_types.get(name)
                    for elem_off, take, raw, data_f32 in self._iter_tensor_chunks(
                            name,
                            include_raw=(t_type in (_GGML_Q8_0, _GGML_Q5_0)),
                            include_f32=(t_type != _GGML_Q8_0)):
                        n_blk = take // 32
                        gi = (off + elem_off) // self.Q8_GROUP_SIZE
                        if raw is not None and t_type == _GGML_Q8_0:
                            blks = np.frombuffer(raw, dtype=np.uint8)[:n_blk * 34].reshape(n_blk, 34)
                            d = np.ascontiguousarray(blks[:, 0:2]).view(np.float16).reshape(n_blk).astype(np.float32)
                            q8_scales[gi:gi + n_blk] = np.where(d > 0, d, 1.0)
                            int8_raw = blks[:, 2:34].reshape(-1)
                            uint8_vals = (int8_raw.view(np.int8).astype(np.int16) + 128).astype(np.uint8)
                        elif raw is not None and t_type == _GGML_Q5_0:
                            blks = np.frombuffer(raw, dtype=np.uint8)[:n_blk * 22].reshape(n_blk, 22)
                            d = np.ascontiguousarray(blks[:, 0:2]).view(np.float16).reshape(n_blk).astype(np.float32)
                            d_abs = np.abs(d)
                            d_safe = np.where(d_abs > 0, d_abs, 1.0)
                            q8_scales[gi:gi + n_blk] = d_safe
                            int8_per_group = np.round(
                                data_f32.reshape(n_blk, 32) / d_safe[:, None]
                            ).astype(np.int16).clip(-16, 15)
                            uint8_vals = (int8_per_group + 128).astype(np.uint8).reshape(-1)
                        else:
                            uint8_vals = np.empty(take, dtype=np.uint8)
                            for bi in range(0, take, self.Q8_GROUP_SIZE):
                                chunk = data_f32[bi:bi + self.Q8_GROUP_SIZE]
                                vmax = float(np.abs(chunk).max())
                                sc = vmax / 127.0 if vmax > 0 else 1.0
                                q8_scales[(off + elem_off + bi) // self.Q8_GROUP_SIZE] = sc
                                q8_int8 = np.round(chunk / sc).astype(np.int16).clip(-127, 127)
                                uint8_vals[bi:bi + len(chunk)] = (q8_int8 + 128).astype(np.uint8)
                        locs = np.arange(off + elem_off, off + elem_off + take, dtype=np.uint32)
                        g.scatter(self.q8_map_id, locs, uint8_vals)
                # xq8 shader: val = float(u8) * scale + zero; with zero = -128*scale → u8=128 ↦ 0
                q8_zeros = (-128.0 * q8_scales).astype(np.float32)
                g.set_qparams(self.q8_map_id, q8_scales, q8_zeros)
                if self.verbose:
                    print(f"[GPU] Q8 upload: {len(self._q8_tensors)} tensors in "
                          f"{time.perf_counter()-t0:.1f}s")

        # GGUF stores actual RMSNorm weights directly — no +1.0 offset needed.
        # (Older thinking assumed HuggingFace delta convention; GGUF converters
        # pre-compute 1+delta and store the actual multiplicative weight.)
        t0 = time.perf_counter()
        for name in self._norm_names_ordered:
            off = self._off[name]
            for elem_off, take, _, data in self._iter_tensor_chunks(name, include_f32=True):
                locs = np.arange(off + elem_off, off + elem_off + take, dtype=np.uint32)
                g.scatter(self.ws_map_id, locs, data)
        if self.verbose:
            print(f"[GPU] F32 norm upload: {len(self._norm_names_ordered)} tensors in "
                  f"{time.perf_counter()-t0:.1f}s")

    def _alloc_workspace(self):
        """Allocate workspace in map 0 (F32).

        Layout:
          [0]      scores    n_head * kv_cap   (element 0 for fixed softmax locs)
          [sc_sz]  scale     1                 1/sqrt(head_dim)
          [...]    activations (hidden, normed, q, k, v, attn_out, ...)
          [...]    norm weights (F32)
          [aligned] KV cache  per-head-contiguous
                    Layer L: K_heads[n_head_kv, kv_cap, head_dim_kv]
                              V_heads[n_head_kv, kv_cap, head_dim_kv]
        """
        c = self.cfg; g = self.gpu
        self._ws_slots = {}
        kv_cap = self._kv_cap
        k_dim = c.n_head_kv * c.head_dim_kv
        q_dim = c.n_head * c.head_dim
        gs = c.n_head // c.n_head_kv  # GQA group size
        self._sample_fused_groups = (
            c.n_vocab + self._gpu_fused_rows_per_group - 1) // self._gpu_fused_rows_per_group
        partial_k = self._gpu_approx_partial_k if self._gpu_approx_rerank else self.SAMPLE_TOPK_MAX
        self._sample_fused_partial_cap = self._sample_fused_groups * partial_k

        # --- Scores at element 0 ---
        pos = 0
        sc_sz = c.n_head * kv_cap
        sc_locs = np.arange(0, sc_sz, dtype=np.uint32)
        sc_h, _ = g.upload_dev(sc_locs)
        self._ws_slots['scores'] = (sc_h, sc_sz, 0)
        self._register_loc_span(sc_h, 0, 0, sc_sz)
        self._scores_row_bases = np.arange(0, sc_sz, kv_cap, dtype=np.uint32)
        self._scores_row_h, _ = g.upload_dev(self._scores_row_bases)
        self._register_row_base_span(self._scores_row_h, 0, 0, c.n_head, kv_cap, kv_cap)
        scores_active = np.empty(sc_sz, dtype=np.uint32)
        idx = 0
        for t in range(kv_cap):
            scores_active[idx:idx + c.n_head] = self._scores_row_bases + np.uint32(t)
            idx += c.n_head
        self._scores_active_h, _ = g.upload_dev(scores_active)
        pos += sc_sz

        # --- Scale scalar ---
        # The broadcast shader reads locs_scalar[loc_idx] for loc_idx in [0, n).
        # n = n_head * seq_len can be up to n_head * kv_cap, so the locs buffer
        # must have sc_sz elements (exec_broadcast_internal checks n*4 <= buf_bytes).
        # All entries point to the same scalar position in the workspace map.
        scale_locs = np.full(sc_sz, pos, dtype=np.uint32)
        scale_h, _ = g.upload_dev(scale_locs)
        self._ws_slots['scale'] = (scale_h, sc_sz, pos)
        self._register_loc_span(scale_h, 0, pos, 1)
        self._pos_scale = pos
        pos += 1

        # --- Embedding scale scalar ---
        if c.emb_scale != 1.0:
            emb_scale_locs = np.full(c.n_embd, pos, dtype=np.uint32)
            emb_scale_h, _ = g.upload_dev(emb_scale_locs)
            self._ws_slots['emb_scale'] = (emb_scale_h, c.n_embd, pos)
            self._register_loc_span(emb_scale_h, 0, pos, 1)
            self._pos_emb_scale = pos
            pos += 1

        # --- Softcap scalars (Gemma 3 and similar) ---
        if c.attn_softcap > 0:
            attn_cap_locs = np.full(sc_sz, pos, dtype=np.uint32)
            attn_cap_h, _ = g.upload_dev(attn_cap_locs)
            self._ws_slots['attn_softcap'] = (attn_cap_h, sc_sz, pos)
            self._register_loc_span(attn_cap_h, 0, pos, 1)
            pos += 1
        if c.final_softcap > 0:
            # final_softcap is applied to logits of shape [n_vocab], so the locs
            # buffer must have n_vocab elements (same broadcast-size rule as scale).
            final_cap_locs = np.full(c.n_vocab, pos, dtype=np.uint32)
            final_cap_h, _ = g.upload_dev(final_cap_locs)
            self._ws_slots['final_softcap'] = (final_cap_h, c.n_vocab, pos)
            self._register_loc_span(final_cap_h, 0, pos, 1)
            pos += 1

        # --- Activation slots ---
        slots = {
            'hidden':   c.n_embd,
            'normed':   c.n_embd,
            'q':        q_dim,
            'k':        k_dim,   # n_head_kv rows × head_dim_kv
            'v':        k_dim,
            'attn_out': c.n_head * c.head_dim_kv,
            'o_proj':   c.n_embd,
            'gate':     c.n_ff,
            'up':       c.n_ff,
            'act':      c.n_ff,
            'ffn_out':  c.n_embd,
            'logits':   c.n_vocab,
            'sample_token': 1,
            'sample_rand': 1,
            'sample_topk_idx': self.SAMPLE_TOPK_MAX,
            'sample_topk_val': self.SAMPLE_TOPK_MAX,
            'sample_topk_prob': self.SAMPLE_TOPK_MAX,
            'sample_short_sel': self.SAMPLE_APPROX_SHORTLIST_MAX,
            'sample_exact_idx': self.SAMPLE_APPROX_SHORTLIST_MAX,
            'sample_exact_val': self.SAMPLE_APPROX_SHORTLIST_MAX,
            'sample_fused_idx': self._sample_fused_partial_cap,
            'sample_fused_val': self._sample_fused_partial_cap,
            'normed2':  c.n_embd,
        }
        for name, sz in slots.items():
            locs = np.arange(pos, pos + sz, dtype=np.uint32)
            handle, _ = g.upload_dev(locs)
            self._ws_slots[name] = (handle, sz, pos)
            self._register_loc_span(handle, 0, pos, sz)
            pos += sz

        # --- Norm weights after activation slots ---
        for name in self._norm_names_ordered:
            sz = self._norm_sizes[name]
            self._off[name] = pos
            self.tensor_map_id[name] = 0
            self._trans[name] = False
            pos += sz

        # --- QK-norm locs for Q (workspace q slot) ---
        pos_q = self._ws_slots['q'][2]
        q_rms_locs = np.array([pos_q + i * c.head_dim
                                for i in range(c.n_head)], dtype=np.uint32)
        self._ws_rms_q_h, _ = g.upload_dev(q_rms_locs)
        self._register_row_base_span(self._ws_rms_q_h, 0, pos_q, c.n_head, c.head_dim, c.head_dim)
        # _ws_rms_q_h doubles as locs_a for scores matmul (same values).
        # QK RMSNorm must write to a genuinely separate workspace slice before
        # RoPE; using the q slot itself is logically in-place and corrupts rows.
        q_tmp_pos = self._ws_slots['normed'][2]
        q_tmp_locs = np.array([q_tmp_pos + i * c.head_dim
                               for i in range(c.n_head)], dtype=np.uint32)
        self._ws_rms_q_dst_h, _ = g.upload_dev(q_tmp_locs)
        self._register_row_base_span(self._ws_rms_q_dst_h, 0, q_tmp_pos, c.n_head, c.head_dim, c.head_dim)

        # --- QK-norm locs for K (workspace k slot) ---
        pos_k = self._ws_slots['k'][2]
        k_rms_locs = np.array([pos_k + i * c.head_dim_kv
                                for i in range(c.n_head_kv)], dtype=np.uint32)
        self._ws_rms_k_h, _ = g.upload_dev(k_rms_locs)
        self._register_row_base_span(self._ws_rms_k_h, 0, pos_k, c.n_head_kv, c.head_dim_kv, c.head_dim_kv)
        # Same fix for K: write RMSNorm output into the free normed2 slice, then
        # RoPE back into the actual k slot.
        k_tmp_pos = self._ws_slots['normed2'][2]
        k_tmp_locs = np.array([k_tmp_pos + i * c.head_dim_kv
                               for i in range(c.n_head_kv)], dtype=np.uint32)
        self._ws_rms_k_dst_h, _ = g.upload_dev(k_tmp_locs)
        self._register_row_base_span(self._ws_rms_k_dst_h, 0, k_tmp_pos, c.n_head_kv, c.head_dim_kv, c.head_dim_kv)

        # --- RoPE src handles (separate resource from q_h/k_h to avoid in-place issue) ---
        q_locs = np.arange(pos_q, pos_q + q_dim, dtype=np.uint32)
        self._q_rope_src_h, _ = g.upload_dev(q_locs.copy())
        self._register_loc_span(self._q_rope_src_h, 0, pos_q, q_dim)
        k_locs = np.arange(pos_k, pos_k + k_dim, dtype=np.uint32)
        self._k_rope_src_h, _ = g.upload_dev(k_locs.copy())
        self._register_loc_span(self._k_rope_src_h, 0, pos_k, k_dim)
        # Merged Q+K rope handles (valid when head_dim == head_dim_kv and Q,K are contiguous)
        if c.head_dim == c.head_dim_kv:
            qk_locs = np.arange(pos_q, pos_q + q_dim + k_dim, dtype=np.uint32)
            self._qk_ws_out_h, _ = g.upload_dev(qk_locs.copy())
            self._qk_rope_src_h, _ = g.upload_dev(qk_locs.copy())
            self._qk_rope_out_h, _ = g.upload_dev(qk_locs.copy())
            self._register_loc_span(self._qk_ws_out_h, 0, pos_q, q_dim + k_dim)
            self._register_loc_span(self._qk_rope_src_h, 0, pos_q, q_dim + k_dim)
            self._register_loc_span(self._qk_rope_out_h, 0, pos_q, q_dim + k_dim)
        else:
            self._qk_ws_out_h = None
            self._qk_rope_src_h = None

        # --- Merged QKV workspace output handle (q/k/v slots are contiguous) ---
        qkv_total = q_dim + k_dim + k_dim
        self._qkv_ws_out_h, _ = g.upload_dev(np.arange(pos_q, pos_q + qkv_total, dtype=np.uint32))
        self._register_loc_span(self._qkv_ws_out_h, 0, pos_q, qkv_total)

        # --- Merged gate+up workspace output handle (gate/up slots are contiguous) ---
        pos_gate = self._ws_slots['gate'][2]
        gateup_total = self._ws_slots['gate'][1] + self._ws_slots['up'][1]
        self._gateup_ws_out_h, _ = g.upload_dev(np.arange(pos_gate, pos_gate + gateup_total, dtype=np.uint32))
        self._register_loc_span(self._gateup_ws_out_h, 0, pos_gate, gateup_total)

        # --- row_copy source-base handles (scalar = element offset of slot start) ---
        self._k_src_base_h, _ = g.upload_dev(np.array([pos_k], dtype=np.uint32))
        self._register_loc_span(self._k_src_base_h, 0, pos_k, 1)
        pos_v = self._ws_slots['v'][2]
        self._v_src_base_h, _ = g.upload_dev(np.array([pos_v], dtype=np.uint32))
        self._register_loc_span(self._v_src_base_h, 0, pos_v, 1)
        # Direct KV projection reuses the same normalized hidden row for each KV
        # head; keep a small handle ready instead of rebuilding it every token.
        kv_src_locs = np.full(max(1, c.n_head_kv), self._ws_slots['normed'][2], dtype=np.uint32)
        self._kv_proj_src_h, _ = g.upload_dev(kv_src_locs)
        self._register_loc_span(self._kv_proj_src_h, 0, self._ws_slots['normed'][2], c.n_embd)

        # --- KV cache: per-head-contiguous layout ---
        # Each KV head g of layer L occupies kv_cap * head_dim_kv elements:
        #   K: kc_k_base[L] + g * kv_cap * head_dim_kv
        #   V: kc_v_base[L] + g * kv_cap * head_dim_kv
        # Align start to head_dim_kv so dst_loc = element_off / head_dim_kv is integer.
        kv_step = kv_cap * c.head_dim_kv        # elements per KV head
        self._kv_step = kv_step
        kv_per_layer = 2 * c.n_head_kv * kv_step
        if pos % c.head_dim_kv != 0:
            pos += c.head_dim_kv - (pos % c.head_dim_kv)
        kv_start = pos

        self._kc_k_base = []   # element offset of K cache start for layer L
        self._kc_v_base = []
        # row_base[g] = (kc_k_base + g*kv_step) / head_dim_kv (integer)
        self._kc_k_row_base = []  # numpy arrays, shape (n_head_kv,), per layer
        self._kc_v_row_base = []

        for L in range(c.n_layer):
            kb = kv_start + L * kv_per_layer
            vb = kb + c.n_head_kv * kv_step
            self._kc_k_base.append(kb)
            self._kc_v_base.append(vb)
            kb_row = kb // c.head_dim_kv
            vb_row = vb // c.head_dim_kv
            self._kc_k_row_base.append(
                np.array([kb_row + gi * kv_cap for gi in range(c.n_head_kv)],
                         dtype=np.uint32))
            self._kc_v_row_base.append(
                np.array([vb_row + gi * kv_cap for gi in range(c.n_head_kv)],
                         dtype=np.uint32))

        pos = kv_start + c.n_layer * kv_per_layer

        # --- Attention handles: n_ops=n_head matmuls ---
        # locs_b for scores matmul [L]: K cache base for each query head's KV group
        self._kc_k_locs_h = []
        self._kc_v_locs_h = []
        for L in range(c.n_layer):
            kb = self._kc_k_base[L]; vb = self._kc_v_base[L]
            k_locs = np.array([kb + (h // gs) * kv_step for h in range(c.n_head)],
                               dtype=np.uint32)
            v_locs = np.array([vb + (h // gs) * kv_step for h in range(c.n_head)],
                               dtype=np.uint32)
            hk, _ = g.upload_dev(k_locs); hv, _ = g.upload_dev(v_locs)
            self._kc_k_locs_h.append(hk); self._kc_v_locs_h.append(hv)
            # The attention read handles contain repeated GQA groups, but the
            # alias model only needs the covered KV-head regions.
            self._register_row_base_span(hk, 0, kb, c.n_head_kv, kv_step, kv_step)
            self._register_row_base_span(hv, 0, vb, c.n_head_kv, kv_step, kv_step)

        # locs_c for weighted sum output (fixed)
        pos_ao = self._ws_slots['attn_out'][2]
        ao_locs = np.array([pos_ao + h * c.head_dim_kv for h in range(c.n_head)],
                            dtype=np.uint32)
        self._ao_locs_h, _ = g.upload_dev(ao_locs)
        self._register_row_base_span(self._ao_locs_h, 0, pos_ao, c.n_head, c.head_dim_kv, c.head_dim_kv)

        # --- row_copy copy_spec handles (fixed base rows; `pos` is added at dispatch) ---
        self._kc_kv_copy_h = []
        for L in range(c.n_layer):
            kv_spec = np.empty(4 * c.n_head_kv, dtype=np.uint32)
            for gi in range(c.n_head_kv):
                base = gi * 4
                kv_spec[base] = gi
                kv_spec[base + 1] = self._kc_k_row_base[L][gi]
                kv_spec[base + 2] = gi + c.n_head_kv
                kv_spec[base + 3] = self._kc_v_row_base[L][gi]
            hk, _ = g.upload_dev(kv_spec)
            self._kc_kv_copy_h.append(hk)

        # Direct-V writes target one token slot per KV head. The handle contents
        # move every token, but the per-head stride stays fixed.
        self._kv_head_offsets = (
            np.arange(c.n_head_kv, dtype=np.uint32) * np.uint32(kv_step)
        )
        self._kc_v_write_arr = []
        self._kc_v_write_h = []
        self._kc_v_write_pos = [-1] * c.n_layer
        for L in range(c.n_layer):
            v_write = (self._kc_v_base[L] + self._kv_head_offsets).astype(np.uint32)
            hv, _ = g.upload_dev(v_write)
            self._kc_v_write_arr.append(v_write)
            self._kc_v_write_h.append(hv)
            self._register_row_base_span(
                hv, 0, self._kc_v_base[L], c.n_head_kv, kv_step, c.head_dim_kv
            )

        # --- Pre-allocated gather/scatter locs ---
        self._hid_locs = np.arange(self._ws_slots['hidden'][2],
                                   self._ws_slots['hidden'][2] + c.n_embd,
                                   dtype=np.uint32)
        self._normed_locs = np.arange(self._ws_slots['normed'][2],
                                      self._ws_slots['normed'][2] + c.n_embd,
                                      dtype=np.uint32)
        self._lo_locs = np.arange(self._ws_slots['logits'][2],
                                  self._ws_slots['logits'][2] + c.n_vocab,
                                  dtype=np.uint32)
        self._sample_locs = np.array([self._ws_slots['sample_token'][2]], dtype=np.uint32)
        self._sample_rand_locs = np.array([self._ws_slots['sample_rand'][2]], dtype=np.uint32)
        self._sample_topk_idx_locs = np.arange(
            self._ws_slots['sample_topk_idx'][2],
            self._ws_slots['sample_topk_idx'][2] + self.SAMPLE_TOPK_MAX,
            dtype=np.uint32,
        )
        self._sample_topk_val_locs = np.arange(
            self._ws_slots['sample_topk_val'][2],
            self._ws_slots['sample_topk_val'][2] + self.SAMPLE_TOPK_MAX,
            dtype=np.uint32,
        )
        self._sample_topk_prob_locs = np.arange(
            self._ws_slots['sample_topk_prob'][2],
            self._ws_slots['sample_topk_prob'][2] + self.SAMPLE_TOPK_MAX,
            dtype=np.uint32,
        )
        self._sample_short_sel_locs = np.arange(
            self._ws_slots['sample_short_sel'][2],
            self._ws_slots['sample_short_sel'][2] + self.SAMPLE_APPROX_SHORTLIST_MAX,
            dtype=np.uint32,
        )
        self._sample_exact_idx_locs = np.arange(
            self._ws_slots['sample_exact_idx'][2],
            self._ws_slots['sample_exact_idx'][2] + self.SAMPLE_APPROX_SHORTLIST_MAX,
            dtype=np.uint32,
        )
        self._sample_exact_val_locs = np.arange(
            self._ws_slots['sample_exact_val'][2],
            self._ws_slots['sample_exact_val'][2] + self.SAMPLE_APPROX_SHORTLIST_MAX,
            dtype=np.uint32,
        )
        self._sample_fused_idx_locs = np.arange(
            self._ws_slots['sample_fused_idx'][2],
            self._ws_slots['sample_fused_idx'][2] + self._sample_fused_partial_cap,
            dtype=np.uint32,
        )
        self._sample_fused_val_locs = np.arange(
            self._ws_slots['sample_fused_val'][2],
            self._ws_slots['sample_fused_val'][2] + self._sample_fused_partial_cap,
            dtype=np.uint32,
        )
        self._sample_repeat_ids_arr = np.zeros(self.SAMPLE_REPEAT_MAX, dtype=np.uint32)
        self._sample_short_sel_arr = np.zeros(self.SAMPLE_APPROX_SHORTLIST_MAX, dtype=np.float32)
        self._sample_rand_arr = np.zeros(1, dtype=np.float32)
        self._sample_repeat_ids_h, _ = g.upload_dev(self._sample_repeat_ids_arr)
        self._sample_fused_base_h, _ = g.upload_dev(
            np.array([self._ws_slots['sample_fused_idx'][2]], dtype=np.uint32))
        self._register_loc_span(self._sample_fused_base_h, 0, self._ws_slots['sample_fused_idx'][2], 1)
        self._sample_exact_base_h, _ = g.upload_dev(
            np.array([self._ws_slots['sample_exact_idx'][2]], dtype=np.uint32))
        self._register_loc_span(self._sample_exact_base_h, 0, self._ws_slots['sample_exact_idx'][2], 1)
        self._sample_short_sel_chunk_h = []
        self._sample_exact_idx_chunk_h = []
        self._sample_exact_val_chunk_h = []
        for start in range(0, self.SAMPLE_APPROX_SHORTLIST_MAX, self.SAMPLE_TOPK_MAX):
            stop = min(start + self.SAMPLE_TOPK_MAX, self.SAMPLE_APPROX_SHORTLIST_MAX)
            sh, _ = g.upload_dev(self._sample_short_sel_locs[start:stop].copy())
            ih, _ = g.upload_dev(self._sample_exact_idx_locs[start:stop].copy())
            vh, _ = g.upload_dev(self._sample_exact_val_locs[start:stop].copy())
            self._sample_short_sel_chunk_h.append(sh)
            self._sample_exact_idx_chunk_h.append(ih)
            self._sample_exact_val_chunk_h.append(vh)
        self._sample_fused_topk_n = 0
        self._sample_fused_partial_n = 0
        self._sample_rerank_n = 0
        self._sample_token_ready = False  # True when sampling is fused into forward batch

        self._ws_total_elems = pos
        if self.verbose:
            kv_mb = c.n_layer * kv_per_layer * 4 / 1e6
            print(f"[GPU] Workspace: {len(slots)} slots + "
                  f"{len(self._norm_names_ordered)} norms + "
                  f"KV cache (cap={kv_cap}) = {pos:,} elems "
                  f"({pos*4/1e6:.0f} MB, KV={kv_mb:.0f} MB)")

    def _alloc_kv_cache(self):
        """Initialize scores buffer and attention scale."""
        c = self.cfg; g = self.gpu; ws_id = self.ws_map_id
        sc_sz = c.n_head * self._kv_cap
        g.scatter(ws_id, np.arange(sc_sz, dtype=np.uint32),
                  np.full(sc_sz, -1e30, dtype=np.float32))
        g.scatter(ws_id, np.array([self._pos_scale], dtype=np.uint32),
                  np.array([1.0 / np.sqrt(c.head_dim)], dtype=np.float32))
        if c.emb_scale != 1.0:
            g.scatter(ws_id, np.array([self._pos_emb_scale], dtype=np.uint32),
                      np.array([c.emb_scale], dtype=np.float32))
        if c.attn_softcap > 0:
            _, _, p = self._ws_slots['attn_softcap']
            g.scatter(ws_id, np.array([p], dtype=np.uint32),
                      np.array([c.attn_softcap], dtype=np.float32))
        if c.final_softcap > 0:
            _, _, p = self._ws_slots['final_softcap']
            g.scatter(ws_id, np.array([p], dtype=np.uint32),
                      np.array([c.final_softcap], dtype=np.float32))
        if self.verbose:
            print(f"[GPU] KV cache: cap={self._kv_cap}, {c.n_layer} layers, "
                  f"n_head_kv={c.n_head_kv}, head_dim_kv={c.head_dim_kv}")

    def _upload_locs(self):
        c = self.cfg
        self._wh = {}
        for name in self._q4_tensors | self._q8_tensors | self._f32_wt_tensors:
            locs = np.array([self._off[name]], dtype=np.uint32)
            self._wh[name], _ = self.gpu.upload_dev(locs)

        # Merge helpers: detect per-layer groups of contiguous same-map weight tensors.
        # Q4 and Q8 maps each store all their tensors with the same on-GPU encoding
        # (Q4_K nibbles or uint8+scale respectively), so any tensors contiguous in
        # one map can be dispatched as a single wider matmul.
        def _detect_merge(name_list):
            """Return (wt_h, map_id) if all names are contiguous in the same map, else None."""
            if not name_list:
                return None
            if all(n in self._q4_tensors for n in name_list):
                map_id = 1
            elif all(n in self._q8_tensors for n in name_list):
                map_id = self.q8_map_id
            else:
                return None
            for i in range(1, len(name_list)):
                expected = self._off[name_list[i - 1]] + self._size_of(name_list[i - 1])
                if self._off[name_list[i]] != expected:
                    return None
            return self._wh[name_list[0]], map_id

        self._qkv_merged_layers = set()
        self._wh_qkv_wt = {}; self._qkv_map_id = {}
        self._qk_merged_layers = set()
        self._wh_qk_wt = {}; self._qk_map_id = {}
        self._gateup_merged_layers = set()
        self._wh_gateup_wt = {}; self._gateup_map_id = {}
        self._wh_v_heads = {}
        self._v_head_map_id = {}
        self._v_head_kind = {}
        for L in range(c.n_layer):
            p = f'blk.{L}'
            qn = f'{p}.attn_q.weight'; kn = f'{p}.attn_k.weight'; vn = f'{p}.attn_v.weight'
            r = _detect_merge([qn, kn, vn])
            if r:
                self._qkv_merged_layers.add(L)
                self._wh_qkv_wt[L], self._qkv_map_id[L] = r
            r = _detect_merge([qn, kn])
            if r:
                self._qk_merged_layers.add(L)
                self._wh_qk_wt[L], self._qk_map_id[L] = r
            if vn in self._wh and not self._trans.get(vn, False):
                head_span = c.head_dim_kv * c.n_embd
                v_head_locs = (
                    np.uint32(self._off[vn]) +
                    np.arange(c.n_head_kv, dtype=np.uint32) * np.uint32(head_span)
                )
                self._wh_v_heads[L], _ = self.gpu.upload_dev(v_head_locs)
                self._v_head_map_id[L] = self.tensor_map_id[vn]
                self._v_head_kind[L] = self._weight_kind(vn)
            gn = f'{p}.ffn_gate.weight'; un = f'{p}.ffn_up.weight'
            r = _detect_merge([gn, un])
            if r:
                self._gateup_merged_layers.add(L)
                self._wh_gateup_wt[L], self._gateup_map_id[L] = r
        if self.verbose:
            print(f"[GPU] QKV merge: {len(self._qkv_merged_layers)}/{c.n_layer} layers  "
                  f"gate+up merge: {len(self._gateup_merged_layers)}/{c.n_layer} layers")

        self._lm_q4_approx_h = 0
        if self._lm_q4_approx_off is not None:
            self._lm_q4_approx_h, _ = self.gpu.upload_dev(
                np.array([self._lm_q4_approx_off], dtype=np.uint32))

        q_head_dim = c.head_dim; kv_head_dim = c.head_dim_kv
        for name in self._norm_names_ordered:
            off = self._off[name]; sz = self._norm_sizes[name]
            if 'attn_q_norm' in name:
                n_rows = c.n_head; h_dim = q_head_dim
                locs = (np.array([off] * n_rows) if sz == h_dim
                        else np.array([off + i*h_dim for i in range(n_rows)]))
            elif 'attn_k_norm' in name:
                n_rows = c.n_head_kv; h_dim = kv_head_dim
                locs = (np.array([off] * n_rows) if sz == h_dim
                        else np.array([off + i*h_dim for i in range(n_rows)]))
            else:
                locs = np.array([off])
            self._wh[name], _ = self.gpu.upload_dev(locs.astype(np.uint32))

        self._gpu_token_embd_name = None
        self._gpu_token_embd_mode = None
        self._emb_gather_locs_arr = None
        self._emb_gather_locs_h = 0
        self._emb_gather_base = None
        self._emb_row_base = 0
        self._emb_dst_base = self._ws_slots['hidden'][2]
        self._pending_embed_tok = 0
        if 'token_embd.weight' in self.tensor_map_id:
            name = 'token_embd.weight'
            sh = self._shape_of(name)
            off = np.uint32(self._off[name])
            if (not self._trans[name]) and (name in self._q8_tensors):
                self._gpu_token_embd_name = name
                self._gpu_token_embd_mode = 'row_xq8'
                self._emb_row_base = int(off)
            elif self._trans[name]:
                base = off + np.arange(c.n_embd, dtype=np.uint32) * np.uint32(sh[1])
                self._gpu_token_embd_name = name
                self._gpu_token_embd_mode = 'gather'
                self._emb_gather_base = base
                self._emb_gather_locs_arr = np.empty(c.n_embd, dtype=np.uint32)
                self._emb_gather_locs_h, _ = self.gpu.upload_dev(self._emb_gather_locs_arr)
            else:
                base = off + np.arange(c.n_embd, dtype=np.uint32)
                self._gpu_token_embd_name = name
                self._gpu_token_embd_mode = 'gather'
                self._emb_gather_base = base
                self._emb_gather_locs_arr = np.empty(c.n_embd, dtype=np.uint32)
                self._emb_gather_locs_h, _ = self.gpu.upload_dev(self._emb_gather_locs_arr)

        if self.verbose:
            print(f"[GPU] Uploaded {len(self._wh)} loc handles")

    def _cache_lm_weight(self):
        """LM head now runs on GPU via Q4 map — no CPU cache needed."""
        self._lm_name = None; self._lm_weight = None
        # Determine which tensor is the LM head weight.
        # output.weight is in the Q4 map; if absent, fall back to token_embd.weight
        # (weight-tied models).  token_embd.weight is still CPU-only, so in that
        # case we keep the old CPU path.
        if 'output.weight' in self.tensor_map_id:
            self._lm_wt_name = 'output.weight'
        elif 'token_embd.weight' in self.tensor_map_id:
            self._lm_wt_name = 'token_embd.weight'
        elif 'token_embd.weight' in self._cpu_only:
            self._lm_wt_name = None   # weight-tied, use CPU fallback
            lm = self._load_tensor_f32('token_embd.weight', keep=True).astype(np.float32)
            self._lm_weight = lm
            if self.verbose:
                print(f"[GPU] LM head: weight-tied, CPU fallback ({lm.nbytes/1e6:.0f} MB)")
        else:
            self._lm_wt_name = None
        self._lm_q4_approx_ready = (
            self._lm_wt_name is not None and
            self._lm_wt_name == self._lm_q4_approx_name and
            self._lm_q4_approx_off is not None and
            self._lm_q4_approx_h != 0
        )
        if self.verbose and self._lm_wt_name:
            print(f"[GPU] LM head: GPU matmul via '{self._lm_wt_name}'")
        if self.verbose and self._lm_q4_approx_ready:
            print(f"[GPU] LM head: Q4 shortlist copy ready for '{self._lm_wt_name}'")

    def _host_runtime_keep_names(self):
        keep = set()
        if self._gpu_token_embd_name is None and 'token_embd.weight' in self.tensors:
            keep.add('token_embd.weight')
        if self._lm_weight is not None and 'token_embd.weight' in self.tensors:
            keep.add('token_embd.weight')
        return keep

    def release_host_state(self):
        if self._host_state_pruned:
            return
        keep_names = self._host_runtime_keep_names()
        tensor_bytes = sum(arr.nbytes for arr in self.tensors.values())
        raw_bytes = sum(len(b) for b in self.raw_blocks.values())
        if keep_names:
            self.tensors = {name: arr for name, arr in self.tensors.items() if name in keep_names}
        else:
            self.tensors = {}
        self.raw_blocks = {}
        self._host_state_pruned = True
        if self.verbose:
            kept = ', '.join(sorted(keep_names)) if keep_names else 'none'
            print(f"[GPU] Production mode: released host tensors/raw "
                  f"({tensor_bytes/1e6:.0f} MB tensors, {raw_bytes/1e6:.0f} MB raw; kept: {kept})")

    def _can_gpu_argmax_sample(self, cfg) -> bool:
        """Fast path: deterministic greedy sampling entirely on GPU."""
        deterministic = (cfg.temperature <= 0.0) or (cfg.top_k == 1)
        return deterministic and cfg.repeat_penalty == 1.0 and self._lm_wt_name is not None

    def _effective_sample_top_k(self, cfg) -> int:
        """Collapse deterministic decode to top-1 so LM head work matches the sampler."""
        if cfg.temperature <= 0.0:
            return 1
        top_k = int(cfg.top_k)
        if top_k <= 0:
            return 0
        return min(top_k, self.SAMPLE_TOPK_MAX, self.cfg.n_vocab)

    def _can_gpu_topk_sample(self, cfg) -> bool:
        """Fast path: GPU shortlist + GPU top-p normalization over the best k candidates."""
        if self._lm_wt_name is None:
            return False
        top_k = self._effective_sample_top_k(cfg)
        if top_k <= 0 or top_k > self.SAMPLE_TOPK_MAX:
            return False
        return True

    def _gpu_fused_topk_weight_kind(self) -> Optional[str]:
        if not self._lm_wt_name or self._trans[self._lm_wt_name]:
            return None
        if (self._lm_wt_name in self._q8_tensors and
                getattr(self.gpu, '_has_map_matvec_topk_t_xq8_dev', False)):
            return 'q8'
        if (self._lm_wt_name in self._q4_tensors and
                getattr(self.gpu, '_has_map_matvec_topk_t_xq4_dev', False)):
            return 'q4'
        return None

    def _effective_gpu_fused_rows_per_group(self, cfg) -> int:
        del cfg
        base = max(1, int(self._gpu_fused_rows_per_group))
        is_broadcom = bool(
            getattr(
                self,
                '_is_broadcom',
                str(getattr(self, '_runtime_profile', '')).startswith('broadcom_v3dv'),
            )
        )
        is_integrated = bool(
            getattr(
                self,
                '_is_integrated',
                getattr(getattr(self, 'gpu', None), '_is_integrated_gpu', False),
            )
        )
        # On Broadcom, use even larger groups for greedy (top_k=1) to minimize
        # dispatch count - the LM head dominates decode time on low-end GPUs.
        if (is_broadcom or is_integrated) and base < 2048:
            return max(base, 1024)
        return base

    def _can_gpu_fused_topk_sample(self, cfg) -> bool:
        if not self._gpu_fused_topk:
            return False
        if not self._can_gpu_topk_sample(cfg):
            return False
        return self._gpu_fused_topk_weight_kind() is not None

    def _can_gpu_approx_rerank_sample(self, cfg) -> bool:
        if not self._gpu_approx_rerank:
            return False
        if self._lm_wt_name is None:
            return False
        top_k = self._effective_sample_top_k(cfg)
        if top_k <= 0 or top_k > self.SAMPLE_APPROX_SHORTLIST_MAX:
            return False
        if not self._lm_q4_approx_ready:
            return False
        if self._lm_wt_name not in self._q8_tensors:
            return False
        if self._trans[self._lm_wt_name]:
            return False
        return (getattr(self.gpu, '_has_map_matvec_topk_t_xq4_ex_dev', False) and
                getattr(self.gpu, '_has_map_matvec_rerank_t_xq8_dev', False))

    def _prepare_repeat_ids(self, prev):
        recent = sorted({t for t in prev[-self.SAMPLE_REPEAT_MAX:] if 0 <= t < self.cfg.n_vocab})
        if not recent:
            return 0
        self._sample_repeat_ids_arr.fill(0)
        self._sample_repeat_ids_arr[:len(recent)] = np.asarray(recent, dtype=np.uint32)
        self.gpu.upload_dev(self._sample_repeat_ids_arr, handle=self._sample_repeat_ids_h)
        return len(recent)

    def _can_gpu_categorical_sample(self) -> bool:
        return bool(getattr(self.gpu, '_has_map_sample_categorical_dev', False))

    def _can_gpu_resolve_idx(self) -> bool:
        return bool(getattr(self.gpu, '_has_map_resolve_idx_dev', False))

    def _gpu_write_sample_rand(self):
        self._sample_rand_arr[0] = np.float32(np.random.random())
        self.gpu.scatter(self.ws_map_id, self._sample_rand_locs, self._sample_rand_arr)

    def _gpu_read_sample_token(self) -> int:
        tok = self.gpu.gather(self.ws_map_id, self._sample_locs).view(np.float32)
        return int(tok[0]) if tok.size else 0

    def _sample_gpu_argmax(self) -> int:
        """Sample next token by reducing the GPU logits buffer to one scalar index."""
        if self._sample_token_ready:
            self._sample_token_ready = False
            return self._gpu_read_sample_token()
        g = self.gpu; ws_id = self.ws_map_id
        lo_h, _ = self._wsh('logits')
        tok_h, _ = self._wsh('sample_token')
        g.map_argmax_dev(ws_id, lo_h, tok_h, self.cfg.n_vocab)
        tok = g.gather(ws_id, self._sample_locs).view(np.float32)
        return int(tok[0])

    def _sample_gpu_topk(self, cfg, prev) -> int:
        """Sample from a GPU-computed top-k shortlist with GPU top-p probabilities."""
        if self._sample_token_ready:
            self._sample_token_ready = False
            return self._gpu_read_sample_token()
        g = self.gpu; ws_id = self.ws_map_id
        lo_h, _ = self._wsh('logits')
        idx_h, _ = self._wsh('sample_topk_idx')
        val_h, _ = self._wsh('sample_topk_val')
        prob_h, _ = self._wsh('sample_topk_prob')
        top_k = self._effective_sample_top_k(cfg)

        if cfg.repeat_penalty != 1.0 and prev:
            if getattr(g, '_has_map_repeat_penalty_dev', False):
                n_rep = self._prepare_repeat_ids(prev)
                if n_rep:
                    g.map_repeat_penalty_dev(ws_id, lo_h, self._sample_repeat_ids_h, n_rep,
                                             float(cfg.repeat_penalty))
            else:
                recent = sorted({t for t in prev[-self.SAMPLE_REPEAT_MAX:] if 0 <= t < self.cfg.n_vocab})
                if recent:
                    locs = self._wsp('logits') + np.asarray(recent, dtype=np.uint32)
                    vals = g.gather(ws_id, locs).view(np.float32)
                    adj = vals.copy()
                    pos = adj > 0
                    adj[pos] /= np.float32(cfg.repeat_penalty)
                    adj[~pos] *= np.float32(cfg.repeat_penalty)
                    g.scatter(ws_id, locs, adj.astype(np.float32))

        can_dev_sample = self._can_gpu_categorical_sample()
        if can_dev_sample:
            rand_h, _ = self._wsh('sample_rand')
            tok_h, _ = self._wsh('sample_token')
            g.batch_begin()
            try:
                g.map_topk_dev(ws_id, lo_h, idx_h, val_h, self.cfg.n_vocab, top_k)
                if cfg.temperature > 0.0:
                    g.map_topp_dev(ws_id, val_h, prob_h, top_k, float(cfg.temperature), float(cfg.top_p))
                    self._gpu_write_sample_rand()
                    g.map_sample_categorical_dev(ws_id, idx_h, prob_h, rand_h, tok_h, top_k)
            finally:
                g.batch_end()
            if cfg.temperature > 0.0:
                return self._gpu_read_sample_token()

        idx = g.gather(ws_id, self._sample_topk_idx_locs[:top_k]).view(np.float32).astype(np.int32)
        vals = g.gather(ws_id, self._sample_topk_val_locs[:top_k]).view(np.float32)
        valid = np.isfinite(vals)
        if not np.any(valid):
            return 0

        idx = idx[valid]
        vals = vals[valid]
        if cfg.temperature <= 0.0:
            return int(idx[0])

        if not can_dev_sample:
            g.map_topp_dev(ws_id, val_h, prob_h, len(idx), float(cfg.temperature), float(cfg.top_p))
        p = g.gather(ws_id, self._sample_topk_prob_locs[:len(idx)]).view(np.float32)
        keep = p > 0.0
        if not np.any(keep):
            return int(idx[0])
        idx = idx[keep]
        p = p[keep]
        p /= p.sum()
        return int(idx[np.random.choice(len(p), p=p)])

    def _sample_gpu_approx_rerank(self, cfg) -> int:
        """Sample from an exact rerank over a GPU-computed approximate shortlist."""
        if self._sample_token_ready:
            self._sample_token_ready = False
            self._sample_rerank_n = 0
            return self._gpu_read_sample_token()
        n = self._sample_rerank_n
        if n <= 0:
            return 0
        self._sample_rerank_n = 0
        g = self.gpu; ws_id = self.ws_map_id
        exact_idx_h, _ = self._wsh('sample_exact_idx')
        exact_val_h, _ = self._wsh('sample_exact_val')
        topk_idx_h, _ = self._wsh('sample_topk_idx')
        topk_val_h, _ = self._wsh('sample_topk_val')
        prob_h, _ = self._wsh('sample_topk_prob')
        tok_h, _ = self._wsh('sample_token')
        resolved_h, _ = self._wsh('sample_short_sel')
        top_k = min(self._effective_sample_top_k(cfg), n, self.SAMPLE_TOPK_MAX)
        if top_k <= 0:
            top_k = min(n, self.SAMPLE_TOPK_MAX)
        can_dev_sample = self._can_gpu_categorical_sample()
        can_resolve = self._can_gpu_resolve_idx()
        use_dev_path = can_resolve and (cfg.temperature <= 0.0 or can_dev_sample)

        if use_dev_path:
            rand_h, _ = self._wsh('sample_rand')
            g.batch_begin()
            try:
                g.map_topk_dev(ws_id, exact_val_h, topk_idx_h, topk_val_h, n, top_k)
                if self.cfg.final_softcap > 0:
                    fcap_h, _ = self._wsh('final_softcap')
                    g.map_broadcast_dev(ws_id, self.A.OP_DIV, topk_val_h, fcap_h, topk_val_h, top_k)
                    g.map_op1_dev(ws_id, self.A.OP_TANH, topk_val_h, topk_val_h, top_k)
                    g.map_broadcast_dev(ws_id, self.A.OP_MUL, topk_val_h, fcap_h, topk_val_h, top_k)
                if cfg.temperature <= 0.0:
                    g.map_resolve_idx_dev(ws_id, self._sample_exact_base_h, topk_idx_h, tok_h, 1)
                else:
                    g.map_resolve_idx_dev(ws_id, self._sample_exact_base_h, topk_idx_h, resolved_h, top_k)
                    if can_dev_sample:
                        g.map_topp_dev(ws_id, topk_val_h, prob_h, top_k,
                                       float(cfg.temperature), float(cfg.top_p))
                        self._gpu_write_sample_rand()
                        g.map_sample_categorical_dev(ws_id, resolved_h, prob_h, rand_h, tok_h, top_k)
            finally:
                g.batch_end()
            if cfg.temperature <= 0.0:
                return self._gpu_read_sample_token()
            if can_dev_sample:
                return self._gpu_read_sample_token()

        idx = g.gather(ws_id, self._sample_exact_idx_locs[:n]).view(np.float32).astype(np.int32)
        vals = g.gather(ws_id, self._sample_exact_val_locs[:n]).view(np.float32)
        valid = np.isfinite(vals)
        if not np.any(valid):
            return 0
        idx = idx[valid]
        vals = vals[valid]
        if idx.size == 0:
            return 0
        if self.cfg.final_softcap > 0:
            cap = np.float32(self.cfg.final_softcap)
            vals = np.tanh(vals / cap) * cap
        if cfg.top_k > 0 and idx.size > cfg.top_k:
            keep = np.argsort(vals)[-cfg.top_k:][::-1]
            idx = idx[keep]
            vals = vals[keep]
        if cfg.temperature <= 0.0:
            return int(idx[np.argmax(vals)])

        logits = vals.astype(np.float32) / np.float32(cfg.temperature)
        p = np.exp(logits - np.max(logits))
        p /= p.sum()
        if cfg.top_p < 1.0:
            si = np.argsort(p)[::-1]
            cs = np.cumsum(p[si])
            cutoff = np.searchsorted(cs, cfg.top_p) + 1
            mask = np.zeros_like(p, dtype=bool)
            mask[si[:cutoff]] = True
            idx = idx[mask]
            p = p[mask]
            p /= p.sum()
        return int(idx[np.random.choice(len(p), p=p)])

    def _consume_gpu_fused_topk(self, cfg) -> int:
        top_k = self._sample_fused_topk_n
        if top_k <= 0:
            return 0
        t0 = time.perf_counter()
        g = self.gpu; ws_id = self.ws_map_id
        self._sample_fused_topk_n = 0

        # Fast path: sampling was already fused into the forward batch — just gather.
        if self._sample_token_ready:
            self._sample_token_ready = False
            tok = self._gpu_read_sample_token()
            self.timing['sample_resolve'] += time.perf_counter() - t0
            return tok

        pos_h, _ = self._wsh('sample_topk_idx')
        prob_h, _ = self._wsh('sample_topk_prob')
        tok_h, _ = self._wsh('sample_token')
        resolved_h, _ = self._wsh('sample_short_sel')
        can_resolve = self._can_gpu_resolve_idx()
        can_dev_sample = self._can_gpu_categorical_sample()

        if top_k == 1:
            if can_resolve:
                g.batch_begin()
                try:
                    g.map_resolve_idx_dev(ws_id, self._sample_fused_base_h, pos_h, tok_h, 1)
                finally:
                    g.batch_end()
                tok = self._gpu_read_sample_token()
            else:
                pos = g.gather(ws_id, self._sample_topk_idx_locs[:1]).view(np.float32).astype(np.int32)
                pos = np.clip(pos, 0, max(self._sample_fused_partial_n - 1, 0))
                tok_locs = self._sample_fused_idx_locs[pos]
                idx = g.gather(ws_id, tok_locs).view(np.float32).astype(np.int32)
                tok = int(idx[0]) if idx.size else 0
            self.timing['sample_resolve'] += time.perf_counter() - t0
            return tok

        if can_resolve:
            rand_h, _ = self._wsh('sample_rand')
            g.batch_begin()
            try:
                if cfg.temperature <= 0.0:
                    g.map_resolve_idx_dev(ws_id, self._sample_fused_base_h, pos_h, tok_h, 1)
                else:
                    g.map_resolve_idx_dev(ws_id, self._sample_fused_base_h, pos_h, resolved_h, top_k)
                    if can_dev_sample:
                        self._gpu_write_sample_rand()
                        g.map_sample_categorical_dev(ws_id, resolved_h, prob_h, rand_h, tok_h, top_k)
            finally:
                g.batch_end()
            if cfg.temperature <= 0.0:
                tok = self._gpu_read_sample_token()
                self.timing['sample_resolve'] += time.perf_counter() - t0
                return tok
            if can_dev_sample:
                tok = self._gpu_read_sample_token()
                self.timing['sample_resolve'] += time.perf_counter() - t0
                return tok

        pos = g.gather(ws_id, self._sample_topk_idx_locs[:top_k]).view(np.float32).astype(np.int32)
        pos = np.clip(pos, 0, max(self._sample_fused_partial_n - 1, 0))
        tok_locs = self._sample_fused_idx_locs[pos]
        idx = g.gather(ws_id, tok_locs).view(np.float32).astype(np.int32)
        if cfg.temperature <= 0.0:
            tok = int(idx[0]) if idx.size else 0
            self.timing['sample_resolve'] += time.perf_counter() - t0
            return tok

        p = g.gather(ws_id, self._sample_topk_prob_locs[:top_k]).view(np.float32)
        keep = p > 0.0
        if not np.any(keep):
            tok = int(idx[0]) if idx.size else 0
            self.timing['sample_resolve'] += time.perf_counter() - t0
            return tok
        idx = idx[keep]
        p = p[keep]
        p /= p.sum()
        tok = int(idx[np.random.choice(len(p), p=p)])
        self.timing['sample_resolve'] += time.perf_counter() - t0
        return tok

    def _can_gpu_merge_approx_shortlist(self) -> bool:
        profile = str(getattr(self, "_runtime_profile", ""))
        return profile.startswith("broadcom_v3dv")

    # ============================================================
    # Workspace handle helpers
    # ============================================================
    def _wsh(self, slot_name):
        h, sz, _ = self._ws_slots[slot_name]; return h, sz

    def _wsp(self, slot_name):
        return self._ws_slots[slot_name][2]

    # ============================================================
    # Cross-map auto-dispatch helpers
    # ============================================================
    def _gpu_matmul_t(self, act_h, wt_name, out_h, M, K, N):
        g = self.gpu; wt_mid = self.tensor_map_id[wt_name]; ws_id = self.ws_map_id
        if wt_mid == ws_id:
            g.map_matmul_t_dev(ws_id, act_h, self._wh[wt_name], out_h, M, K, N)
        elif wt_name in self._q8_tensors:
            g.map_matmul_t_xq8_dev(ws_id, wt_mid, act_h, self._wh[wt_name], out_h, M, K, N)
        elif wt_name in self._q4_tensors:
            g.map_matmul_t_xq4_dev(ws_id, wt_mid, act_h, self._wh[wt_name], out_h, M, K, N)
        else:
            g.map_matmul_t_x_dev(ws_id, wt_mid, act_h, self._wh[wt_name], out_h, M, K, N)

    def _gpu_matmul(self, act_h, wt_name, out_h, M, K, N):
        g = self.gpu; wt_mid = self.tensor_map_id[wt_name]; ws_id = self.ws_map_id
        if wt_mid == ws_id:
            g.map_matmul_dev(ws_id, act_h, self._wh[wt_name], out_h, M, K, N)
        elif wt_name in self._q8_tensors:
            g.map_matmul_xq8_dev(ws_id, wt_mid, act_h, self._wh[wt_name], out_h, M, K, N)
        elif wt_name in self._q4_tensors:
            g.map_matmul_xq4_dev(ws_id, wt_mid, act_h, self._wh[wt_name], out_h, M, K, N)
        else:
            g.map_matmul_x_dev(ws_id, wt_mid, act_h, self._wh[wt_name], out_h, M, K, N)

    def _gpu_rmsnorm(self, src_h, wt_name, dst_h, n_rows, dim, eps):
        g = self.gpu; wt_mid = self.tensor_map_id[wt_name]; ws_id = self.ws_map_id
        if wt_mid == ws_id:
            g.map_rmsnorm_dev(ws_id, src_h, self._wh[wt_name], dst_h, n_rows, dim, eps)
        else:
            g.map_rmsnorm_x_dev(ws_id, wt_mid, src_h, self._wh[wt_name], dst_h, n_rows, dim, eps)

    def _gpu_proj(self, in_h, wt_name, out_h, M, K, N):
        if self._trans[wt_name]:
            self._gpu_matmul(in_h, wt_name, out_h, M, K, N)
        else:
            self._gpu_matmul_t(in_h, wt_name, out_h, M, K, N)

    def _gpu_proj_merged(self, in_h, wt_handle, wt_map_id, weight_kind, transposed, out_h, M, K, N):
        g = self.gpu
        ws_id = self.ws_map_id
        if transposed:
            if wt_map_id == ws_id:
                g.map_matmul_dev(ws_id, in_h, wt_handle, out_h, M, K, N)
            elif weight_kind == 'q8':
                g.map_matmul_xq8_dev(ws_id, wt_map_id, in_h, wt_handle, out_h, M, K, N)
            elif weight_kind == 'q4':
                g.map_matmul_xq4_dev(ws_id, wt_map_id, in_h, wt_handle, out_h, M, K, N)
            else:
                g.map_matmul_x_dev(ws_id, wt_map_id, in_h, wt_handle, out_h, M, K, N)
        else:
            if wt_map_id == ws_id:
                g.map_matmul_t_dev(ws_id, in_h, wt_handle, out_h, M, K, N)
            elif weight_kind == 'q8':
                g.map_matmul_t_xq8_dev(ws_id, wt_map_id, in_h, wt_handle, out_h, M, K, N)
            elif weight_kind == 'q4':
                g.map_matmul_t_xq4_dev(ws_id, wt_map_id, in_h, wt_handle, out_h, M, K, N)
            else:
                g.map_matmul_t_x_dev(ws_id, wt_map_id, in_h, wt_handle, out_h, M, K, N)

    def _gpu_proj_heads_direct(self, src_h, wt_handle, wt_map_id, weight_kind, dst_h, K, N, n_ops):
        g = self.gpu
        ws_id = self.ws_map_id
        if wt_map_id == ws_id:
            g.map_matmul_t_dev(ws_id, src_h, wt_handle, dst_h, 1, K, N, n_ops)
        elif weight_kind == 'q8':
            g.map_matmul_t_xq8_dev(ws_id, wt_map_id, src_h, wt_handle, dst_h, 1, K, N, n_ops)
        elif weight_kind == 'q4':
            g.map_matmul_t_xq4_dev(ws_id, wt_map_id, src_h, wt_handle, dst_h, 1, K, N, n_ops)
        else:
            g.map_matmul_t_x_dev(ws_id, wt_map_id, src_h, wt_handle, dst_h, 1, K, N, n_ops)

    def _prepare_direct_v_write(self, layer_idx: int, pos: int):
        if self._kc_v_write_pos[layer_idx] == pos:
            return
        base = np.uint32(self._kc_v_base[layer_idx] + pos * self.cfg.head_dim_kv)
        np.add(base, self._kv_head_offsets, out=self._kc_v_write_arr[layer_idx])
        self.gpu.upload_dev(self._kc_v_write_arr[layer_idx], handle=self._kc_v_write_h[layer_idx])
        self._register_row_base_span(
            self._kc_v_write_h[layer_idx],
            0,
            int(base),
            self.cfg.n_head_kv,
            self._kv_step,
            self.cfg.head_dim_kv,
        )
        self._kc_v_write_pos[layer_idx] = pos

    def _prepare_gpu_embed_token(self, tok: int):
        name = self._gpu_token_embd_name
        if name is None:
            raise RuntimeError("GPU token embedding requested but no GPU embedding weight is available")
        c = self.cfg
        g = self.gpu
        if self._gpu_token_embd_mode == 'row_xq8':
            self._pending_embed_tok = int(tok)
            return

        if self._trans[name]:
            np.add(self._emb_gather_base, np.uint32(tok), out=self._emb_gather_locs_arr)
        else:
            np.add(self._emb_gather_base, np.uint32(tok * c.n_embd), out=self._emb_gather_locs_arr)
        g.upload_dev(self._emb_gather_locs_arr, handle=self._emb_gather_locs_h)

    def _enqueue_gpu_embed_token(self, dst_h: int):
        name = self._gpu_token_embd_name
        if name is None:
            raise RuntimeError("GPU token embedding requested but no GPU embedding weight is available")
        c = self.cfg
        g = self.gpu
        mid = self.tensor_map_id[name]
        if self._gpu_token_embd_mode == 'row_xq8':
            g.map_row_gather_xq8_dev(
                self.ws_map_id,
                mid,
                self._emb_row_base,
                self._emb_dst_base,
                self._pending_embed_tok,
                c.n_embd,
                float(c.emb_scale),
            )
            return

        src_h, _, _, _ = g.map_gather_dev(mid, self._emb_gather_locs_h, c.n_embd)
        g.map_scatter_dev(self.ws_map_id, dst_h, c.n_embd, src_h)
        if c.emb_scale != 1.0:
            emb_scale_h, _ = self._wsh('emb_scale')
            g.map_broadcast_dev(self.ws_map_id, self.A.OP_MUL, dst_h, emb_scale_h, dst_h, c.n_embd)

    def _gather_contiguous(self, start, size):
        locs = np.arange(start, start + size, dtype=np.uint32)
        return self.gpu.gather(self.ws_map_id, locs).view(np.float32)

    def _cpu_attention_step(self, layer_idx, seq_len):
        c = self.cfg; g = self.gpu; ws_id = self.ws_map_id
        q_pos = self._ws_slots['q'][2]
        ao_pos = self._ws_slots['attn_out'][2]
        q = self._gather_contiguous(q_pos, c.n_head * c.head_dim).reshape(c.n_head, c.head_dim)
        attn_out = np.empty((c.n_head, c.head_dim_kv), dtype=np.float32)
        gs = max(1, c.n_head // c.n_head_kv)
        scale = np.float32(1.0 / np.sqrt(c.head_dim))

        for gi in range(c.n_head_kv):
            k_base = self._kc_k_base[layer_idx] + gi * self._kv_cap * c.head_dim_kv
            v_base = self._kc_v_base[layer_idx] + gi * self._kv_cap * c.head_dim_kv
            k = self._gather_contiguous(k_base, seq_len * c.head_dim_kv).reshape(seq_len, c.head_dim_kv)
            v = self._gather_contiguous(v_base, seq_len * c.head_dim_kv).reshape(seq_len, c.head_dim_kv)
            q_slice = q[gi * gs:(gi + 1) * gs]
            scores = (q_slice @ k.T) * scale
            if c.attn_softcap > 0:
                cap = np.float32(c.attn_softcap)
                scores = np.tanh(scores / cap) * cap
            scores -= scores.max(axis=1, keepdims=True)
            probs = np.exp(scores).astype(np.float32)
            probs /= probs.sum(axis=1, keepdims=True)
            attn_out[gi * gs:(gi + 1) * gs] = probs @ v

        g.scatter(ws_id,
                  np.arange(ao_pos, ao_pos + attn_out.size, dtype=np.uint32),
                  attn_out.reshape(-1))

    # ============================================================
    # B6: Decode plan — pre-compiled forward pass (zero Python overhead)
    # ============================================================
    def _build_decode_plan(self):
        """Pre-compile the full forward pass into a flat DecodePlanOp array.

        Requirements: fused QK-norm+RoPE path (Gemma3), no cpu_attention_fallback.
        Falls back to Python loop if not supported.
        """
        A = self.A
        if not getattr(self.gpu, '_has_decode_plan', False):
            self._decode_plan_active = False
            return
        if self._cpu_attention_fallback:
            self._decode_plan_active = False
            return

        c = self.cfg
        ws_id = self.ws_map_id

        # Require fused QK-norm+RoPE (Gemma3 only): both norm weights in same map
        qnn0 = 'blk.0.attn_q_norm.weight'
        knn0 = 'blk.0.attn_k_norm.weight'
        if (qnn0 not in self._wh or self._qk_rope_out_h is None
                or self.tensor_map_id.get(qnn0) != self.tensor_map_id.get(knn0)):
            self._decode_plan_active = False
            return

        RTP = A.DECODE_RUNTIME_POS
        RTS = A.DECODE_RUNTIME_SEQ_LEN

        def _mk(**kw):
            return A.DecodePlanOp(**kw)

        def rmsnorm_op(src_h, wt_name, dst_h, n_rows, dim):
            wt_mid = self.tensor_map_id[wt_name]
            wt_h = self._wh[wt_name]
            if wt_mid == ws_id:
                return _mk(kind=A.DECODE_OP_RMSNORM, map_id=ws_id,
                            h0=src_h, h1=wt_h, h2=dst_h,
                            u0=n_rows, u1=dim, f0=c.norm_eps)
            else:
                return _mk(kind=A.DECODE_OP_RMSNORM_X, map_id=ws_id, map_id2=wt_mid,
                            h0=src_h, h1=wt_h, h2=dst_h,
                            u0=n_rows, u1=dim, f0=c.norm_eps)

        def rmsnorm_add_op(norm_in_h, wt_name, res_h, dst_h, n_rows, dim):
            wt_h = self._wh[wt_name]
            return _mk(kind=A.DECODE_OP_RMSNORM_ADD, map_id=ws_id,
                        h0=norm_in_h, h1=wt_h, h2=res_h, h3=dst_h,
                        u0=n_rows, u1=dim, f0=c.norm_eps)

        def proj_op(in_h, wt_name, out_h, M, K, N):
            wt_mid = self.tensor_map_id[wt_name]
            wt_h = self._wh[wt_name]
            kind = self._weight_kind(wt_name)
            trans = self._trans[wt_name]
            if trans:
                if wt_mid == ws_id:
                    return _mk(kind=A.DECODE_OP_MATMUL_X, map_id=ws_id,
                                h0=in_h, h1=wt_h, h2=out_h, u0=M, u1=K, u2=N, u3=1)
                elif kind == 'q8':
                    return _mk(kind=A.DECODE_OP_MATMUL_XQ8, map_id=ws_id, map_id2=wt_mid,
                                h0=in_h, h1=wt_h, h2=out_h, u0=M, u1=K, u2=N, u3=1)
                elif kind == 'q4':
                    return _mk(kind=A.DECODE_OP_MATMUL_XQ4, map_id=ws_id, map_id2=wt_mid,
                                h0=in_h, h1=wt_h, h2=out_h, u0=M, u1=K, u2=N, u3=1)
                else:
                    return _mk(kind=A.DECODE_OP_MATMUL_X, map_id=ws_id, map_id2=wt_mid,
                                h0=in_h, h1=wt_h, h2=out_h, u0=M, u1=K, u2=N, u3=1)
            else:
                if wt_mid == ws_id:
                    return _mk(kind=A.DECODE_OP_MATMUL_T, map_id=ws_id,
                                h0=in_h, h1=wt_h, h2=out_h, u0=M, u1=K, u2=N, u3=1)
                elif kind == 'q8':
                    return _mk(kind=A.DECODE_OP_MATMUL_T_XQ8, map_id=ws_id, map_id2=wt_mid,
                                h0=in_h, h1=wt_h, h2=out_h, u0=M, u1=K, u2=N, u3=1)
                elif kind == 'q4':
                    return _mk(kind=A.DECODE_OP_MATMUL_T_XQ4, map_id=ws_id, map_id2=wt_mid,
                                h0=in_h, h1=wt_h, h2=out_h, u0=M, u1=K, u2=N, u3=1)
                else:
                    return _mk(kind=A.DECODE_OP_MATMUL_T_X, map_id=ws_id, map_id2=wt_mid,
                                h0=in_h, h1=wt_h, h2=out_h, u0=M, u1=K, u2=N, u3=1)

        hid_h    = self._ws_slots['hidden'][0]
        normed_h = self._ws_slots['normed'][0]
        q_h      = self._ws_slots['q'][0]
        k_h      = self._ws_slots['k'][0]
        v_h      = self._ws_slots['v'][0]
        ao_h     = self._ws_slots['attn_out'][0]
        op_h     = self._ws_slots['o_proj'][0]
        n2_h     = self._ws_slots['normed2'][0]
        gate_h   = self._ws_slots['gate'][0]
        up_h     = self._ws_slots['up'][0]
        act_h    = self._ws_slots['act'][0]
        ffn_h    = self._ws_slots['ffn_out'][0]

        q_dim  = c.n_head * c.head_dim
        k_dim  = c.n_head_kv * c.head_dim_kv
        fused_gate_op = A.OP_GEGLU if c.ffn_act == 'gelu' else A.OP_SWIGLU
        attn_scale = float(1.0 / np.sqrt(c.head_dim))
        softcap    = float(c.attn_softcap)

        _has_fused_qkv_q4 = (hasattr(A, 'DECODE_OP_FUSED_QKV_T_XQ4') and
                              getattr(self.gpu, '_has_map_fused_qkv_t_xq4_dev', False))
        _has_fused_gateup_q4 = (hasattr(A, 'DECODE_OP_FUSED_GATEUP_T_XQ4') and
                                 getattr(self.gpu, '_has_map_fused_gateup_t_xq4_dev', False))
        _has_fused_qkv_q8 = (hasattr(A, 'DECODE_OP_FUSED_QKV_T_XQ8') and
                              getattr(self.gpu, '_has_map_fused_qkv_t_xq8_dev', False))
        _has_fused_gateup_q8 = (hasattr(A, 'DECODE_OP_FUSED_GATEUP_T_XQ8') and
                                 getattr(self.gpu, '_has_map_fused_gateup_t_xq8_dev', False))
        _has_fused_qkv = _has_fused_qkv_q4 or _has_fused_qkv_q8
        _has_fused_gateup = _has_fused_gateup_q4 or _has_fused_gateup_q8
        _has_fused_gateup_act_q8 = (hasattr(A, 'DECODE_OP_FUSED_GATEUP_ACT_T_XQ8') and
                                     getattr(self.gpu, '_has_map_fused_gateup_act_t_xq8_dev', False))

        def fused_qkv_op_or_fallback(in_h, qname, kname, vname, out_qh, out_kh, out_vh):
            q_mid = self.tensor_map_id[qname]
            if (not self._trans[qname] and not self._trans[kname]
                    and not self._trans[vname]
                    and self.tensor_map_id[kname] == q_mid
                    and self.tensor_map_id[vname] == q_mid):
                kind_q = self._weight_kind(qname)
                kind_k = self._weight_kind(kname)
                kind_v = self._weight_kind(vname)
                if (_has_fused_qkv_q4
                        and kind_q == 'q4' and kind_k == 'q4' and kind_v == 'q4'):
                    return [_mk(
                        kind=A.DECODE_OP_FUSED_QKV_T_XQ4,
                        map_id=ws_id, map_id2=q_mid,
                        h0=in_h,
                        h1=self._wh[qname], h2=self._wh[kname], h3=self._wh[vname],
                        h4=out_qh, h5=out_kh,
                        u0=out_vh, u1=c.n_embd, u2=q_dim, u3=k_dim, u4=k_dim,
                    )]
                if (_has_fused_qkv_q8
                        and kind_q == 'q8' and kind_k == 'q8' and kind_v == 'q8'):
                    return [_mk(
                        kind=A.DECODE_OP_FUSED_QKV_T_XQ8,
                        map_id=ws_id, map_id2=q_mid,
                        h0=in_h,
                        h1=self._wh[qname], h2=self._wh[kname], h3=self._wh[vname],
                        h4=out_qh, h5=out_kh,
                        u0=out_vh, u1=c.n_embd, u2=q_dim, u3=k_dim, u4=k_dim,
                    )]
            return [
                proj_op(in_h, qname, out_qh, 1, c.n_embd, q_dim),
                proj_op(in_h, kname, out_kh, 1, c.n_embd, k_dim),
                proj_op(in_h, vname, out_vh, 1, c.n_embd, k_dim),
            ]

        def fused_gateup_op_or_fallback(in_h, gate_name, up_name, out_gate_h, out_up_h):
            gate_mid = self.tensor_map_id[gate_name]
            if (not self._trans[gate_name] and not self._trans[up_name]
                    and self.tensor_map_id[up_name] == gate_mid):
                kind_g = self._weight_kind(gate_name)
                kind_u = self._weight_kind(up_name)
                if _has_fused_gateup_q4 and kind_g == 'q4' and kind_u == 'q4':
                    return [_mk(
                        kind=A.DECODE_OP_FUSED_GATEUP_T_XQ4,
                        map_id=ws_id, map_id2=gate_mid,
                        h0=in_h,
                        h1=self._wh[gate_name], h2=self._wh[up_name],
                        h3=out_gate_h, h4=out_up_h,
                        u0=c.n_embd, u1=c.n_ff, u2=c.n_ff,
                    )]
                if _has_fused_gateup_q8 and kind_g == 'q8' and kind_u == 'q8':
                    return [_mk(
                        kind=A.DECODE_OP_FUSED_GATEUP_T_XQ8,
                        map_id=ws_id, map_id2=gate_mid,
                        h0=in_h,
                        h1=self._wh[gate_name], h2=self._wh[up_name],
                        h3=out_gate_h, h4=out_up_h,
                        u0=c.n_embd, u1=c.n_ff, u2=c.n_ff,
                    )]
            return [
                proj_op(in_h, gate_name, out_gate_h, 1, c.n_embd, c.n_ff),
                proj_op(in_h, up_name, out_up_h, 1, c.n_embd, c.n_ff),
            ]

        def fused_gateup_act_op_or_fallback(in_h, gate_name, up_name, out_act_h):
            """Fused gate+up projection + swiGLU activation → single output (act_h).
            Only activates for Q8 swiGLU (silu); falls back to 3 separate ops otherwise."""
            gate_mid = self.tensor_map_id[gate_name]
            if (_has_fused_gateup_act_q8
                    and fused_gate_op == A.OP_SWIGLU
                    and not self._trans[gate_name] and not self._trans[up_name]
                    and self.tensor_map_id[up_name] == gate_mid):
                kind_g = self._weight_kind(gate_name)
                kind_u = self._weight_kind(up_name)
                if kind_g == 'q8' and kind_u == 'q8':
                    return [_mk(
                        kind=A.DECODE_OP_FUSED_GATEUP_ACT_T_XQ8,
                        map_id=ws_id, map_id2=gate_mid,
                        h0=in_h,
                        h1=self._wh[gate_name], h2=self._wh[up_name],
                        h3=out_act_h,
                        u0=c.n_embd, u1=c.n_ff,
                    )]
            return [
                *fused_gateup_op_or_fallback(in_h, gate_name, up_name, gate_h, up_h),
                _mk(kind=A.DECODE_OP_OP2, map_id=ws_id,
                    op_code=fused_gate_op, h0=gate_h, h1=up_h, h2=out_act_h, u0=c.n_ff),
            ]

        all_ops = []
        for L in range(c.n_layer):
            p  = self._blk_names[L]
            gl = c.is_global(L)
            freq = float(c.rope_base_global if gl else c.rope_base_local)
            qnn  = f'{p}.attn_q_norm.weight'
            knn  = f'{p}.attn_k_norm.weight'
            pan  = f'{p}.post_attention_norm.weight'
            pfn  = f'{p}.post_ffw_norm.weight'
            norm_mid = self.tensor_map_id[qnn]

            # 1. Pre-attention RMSNorm: hidden → normed
            all_ops.append(rmsnorm_op(hid_h, f'{p}.attn_norm.weight', normed_h, 1, c.n_embd))

            # 2-4. QKV projections (fused single dispatch if all Q4 same-map, else 3 ops)
            all_ops.extend(fused_qkv_op_or_fallback(
                normed_h,
                f'{p}.attn_q.weight', f'{p}.attn_k.weight', f'{p}.attn_v.weight',
                q_h, k_h, v_h))

            # 5. Fused QK-norm + RoPE (cross-map, writes Q+K back into workspace)
            all_ops.append(_mk(
                kind=A.DECODE_OP_QK_NORM_ROPE_X,
                map_id=ws_id, map_id2=norm_mid,
                h0=self._ws_rms_q_h, h1=self._wh[qnn],
                h2=self._ws_rms_k_h, h3=self._wh[knn],
                h4=self._qk_rope_out_h,
                u0=c.n_head, u1=c.head_dim,
                u2=c.n_head_kv, u3=c.head_dim_kv,
                u4=RTP,
                f0=freq,
            ))

            # 6. KV copy to cache (K and V together; workspace layout has K then V contiguous)
            all_ops.append(_mk(
                kind=A.DECODE_OP_ROW_COPY_OFFSET,
                map_id=ws_id,
                h0=self._kc_kv_copy_h[L],
                h1=self._k_src_base_h,
                u0=RTP,
                u1=2 * c.n_head_kv,
                u2=c.head_dim_kv,
            ))

            # 7. Attention scores: Q @ K^T  [n_head × 1 × seq_len]
            all_ops.append(_mk(
                kind=A.DECODE_OP_MATMUL_T,
                map_id=ws_id,
                h0=self._ws_rms_q_h,
                h1=self._kc_k_locs_h[L],
                h2=self._scores_row_h,
                u0=1, u1=c.head_dim, u2=RTS, u3=c.n_head,
            ))

            # 8. Fused softmax + weighted-V sum → attn_out
            all_ops.append(_mk(
                kind=A.DECODE_OP_ATTN_SOFTMAX_VALUE,
                map_id=ws_id,
                h0=self._scores_row_h,
                h1=self._kc_v_locs_h[L],
                h2=self._ao_locs_h,
                u0=c.n_head, u1=RTS, u2=c.head_dim_kv,
                f0=attn_scale, f1=softcap,
            ))

            # 9. Output projection
            all_ops.append(proj_op(ao_h, f'{p}.attn_output.weight', op_h,
                                   1, c.n_head * c.head_dim_kv, c.n_embd))

            # 10. Post-attention norm + residual
            if pan in self._wh:
                all_ops.append(rmsnorm_add_op(op_h, pan, hid_h, hid_h, 1, c.n_embd))
            else:
                all_ops.append(_mk(kind=A.DECODE_OP_OP2, map_id=ws_id,
                                   op_code=A.OP_ADD, h0=hid_h, h1=op_h, h2=hid_h, u0=c.n_embd))

            # 11. FFN pre-norm: hidden → normed2
            all_ops.append(rmsnorm_op(hid_h, f'{p}.ffn_norm.weight', n2_h, 1, c.n_embd))

            # 12-14. Gate+up proj + activation → act_h (fused single dispatch if Q8 swiGLU, else 3 ops)
            all_ops.extend(fused_gateup_act_op_or_fallback(
                n2_h, f'{p}.ffn_gate.weight', f'{p}.ffn_up.weight', act_h))

            # 15. FFN down projection
            all_ops.append(proj_op(act_h, f'{p}.ffn_down.weight', ffn_h, 1, c.n_ff, c.n_embd))

            # 16. Post-FFN norm + residual
            if pfn in self._wh:
                all_ops.append(rmsnorm_add_op(ffn_h, pfn, hid_h, hid_h, 1, c.n_embd))
            else:
                all_ops.append(_mk(kind=A.DECODE_OP_OP2, map_id=ws_id,
                                   op_code=A.OP_ADD, h0=hid_h, h1=ffn_h, h2=hid_h, u0=c.n_embd))

        # Final output norm (included in plan so decode_step covers it)
        all_ops.append(rmsnorm_op(hid_h, 'output_norm.weight', normed_h, 1, c.n_embd))

        self._decode_plan_id = 0
        self._decode_plan_n_layer = c.n_layer
        self.gpu.register_decode_plan(self._decode_plan_id, all_ops)
        self._decode_plan_active = True
        if self.verbose:
            print(f"[GPU] B9 decode plan: {len(all_ops)} ops ({c.n_layer} layers)"
                  f" fused_qkv_q8={_has_fused_qkv_q8}"
                  f" fused_gateup_q8={_has_fused_gateup_q8}"
                  f" fused_gateup_act_q8={_has_fused_gateup_act_q8}")

    # ============================================================
    # B12: Monolithic decode shader registration
    # ============================================================

    _B12_PROJ_SUFFIXES = [
        'attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight',
        'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight',
    ]
    _B12_Q8_MAP_ID = 3   # new GPU map for B12-only re-quantized Q4/Q6 tensors
    _B12_KV_MAP_ID = 4   # dedicated KV cache for B12 (exact size so C code derives kv_cap correctly)

    def _ensure_b12_kv_map(self):
        """One-time: create a dedicated F32 KV-cache map (map 4) for B12 with EXACT size.

        The C code computes kv_cap = map_kvcache_total_bytes / (4 * 2 * n_layer * n_head_kv
        * head_dim_kv). If we pass ws_map_id for kvcache (which includes activations + norms
        + KV cache), kv_cap is inflated and the shader addresses wrong KV locations.
        This map has exactly 2 * n_layer * n_head_kv * kv_cap * head_dim_kv F32 elements,
        so the C code derives the correct kv_cap."""
        if getattr(self, '_b12_kv_map_created', False):
            return
        c = self.cfg
        n_kv_elems = 2 * c.n_layer * c.n_head_kv * self._kv_cap * c.head_dim_kv
        self.gpu.map_create(self._B12_KV_MAP_ID, 1, n_kv_elems)
        self._b12_kv_map_created = True
        if self.verbose:
            print(f"[GPU] B12 KV map {self._B12_KV_MAP_ID}: {n_kv_elems:,} F32 elems "
                  f"({n_kv_elems * 4 / 1e6:.0f} MB, kv_cap={self._kv_cap})")

    def _ensure_b12_q8_map(self):
        """One-time: create Q8 re-quant map (map 3) for projection tensors not already in Q8 map.

        Uses ALL model layers (len(_blk_names)) so the map is valid for any n_layer sub-test.
        Idempotent — safe to call from reset() multiple times."""
        if getattr(self, '_b12_q8_map_created', False):
            return

        self._b12_q8_off = {}   # tensor-name → element offset inside B12 Q8 map

        # Collect all unique non-Q8 projection tensors across all model layers
        non_q8 = []
        seen = set()
        for L in range(len(self._blk_names)):
            for suffix in self._B12_PROJ_SUFFIXES:
                name = f'{self._blk_names[L]}.{suffix}'
                if (name in self._off and name not in seen and
                        self.tensor_map_id.get(name) not in (self.ws_map_id, self.q8_map_id)):
                    non_q8.append(name)
                    seen.add(name)

        if not non_q8:
            self._b12_q8_map_created = True
            return

        # Total element count for the new map
        total_elems = sum(self._size_of(n) for n in non_q8)
        n_groups = (total_elems + self.Q8_GROUP_SIZE - 1) // self.Q8_GROUP_SIZE

        g = self.gpu
        g.map_create_typed(self._B12_Q8_MAP_ID, self.A.DTYPE_Q8, 1, total_elems, self.Q8_GROUP_SIZE)

        b12_scales = np.ones (n_groups, dtype=np.float32)
        b12_zeros  = np.zeros(n_groups, dtype=np.float32)

        pos = 0
        for name in non_q8:
            self._b12_q8_off[name] = pos
            for elem_off, take, _, data_f32 in self._iter_tensor_chunks(name, include_f32=True):
                uint8_vals = np.empty(take, dtype=np.uint8)
                for bi in range(0, take, self.Q8_GROUP_SIZE):
                    chunk = data_f32[bi:bi + self.Q8_GROUP_SIZE]
                    vmax  = float(np.abs(chunk).max())
                    sc    = vmax / 127.0 if vmax > 0 else 1.0
                    gi    = (pos + elem_off + bi) // self.Q8_GROUP_SIZE
                    b12_scales[gi] = sc
                    b12_zeros [gi] = -128.0 * sc
                    q8 = np.round(chunk / sc).astype(np.int16).clip(-127, 127)
                    uint8_vals[bi:bi + len(chunk)] = (q8 + 128).astype(np.uint8)
                locs = np.arange(pos + elem_off, pos + elem_off + take, dtype=np.uint32)
                g.scatter(self._B12_Q8_MAP_ID, locs, uint8_vals)
            pos += self._size_of(name)

        g.set_qparams(self._B12_Q8_MAP_ID, b12_scales, b12_zeros)
        self._b12_q8_map_created = True
        if self.verbose:
            print(f"[GPU] B12 Q8 remap: {len(non_q8)} tensors, "
                  f"{total_elems/1e6:.0f}MB → map {self._B12_Q8_MAP_ID}")

    def _register_full_decode(self):
        """Build FullDecodeWeightAddrs and register the monolithic decode shader (B12)."""
        A = self.A
        if not (getattr(self.gpu, '_has_full_decode_step', False) and
                getattr(self.gpu, '_has_map_device_address', False)):
            return

        c = self.cfg
        q_dim = c.n_head * c.head_dim
        k_dim = c.n_head_kv * c.head_dim_kv

        # Ensure B12 Q8 re-quant map exists for any Q4/Q6 projection tensors
        self._ensure_b12_q8_map()

        ws_base    = self.gpu.get_map_buffer_device_address(self.ws_map_id)
        q8_base    = self.gpu.get_map_buffer_device_address(self.q8_map_id)
        q8_qp_base = self.gpu.get_map_qparam_device_address(self.q8_map_id)
        b12_base    = self.gpu.get_map_buffer_device_address(self._B12_Q8_MAP_ID) \
                      if getattr(self, '_b12_q8_off', None) else 0
        b12_qp_base = self.gpu.get_map_qparam_device_address(self._B12_Q8_MAP_ID) \
                      if getattr(self, '_b12_q8_off', None) else 0

        wa = A.FullDecodeWeightAddrs()

        def _w_addr(name):
            """Byte-level VkDeviceAddress for a weight tensor."""
            mid = self.tensor_map_id.get(name)
            off = self._off.get(name, 0)
            if mid == self.ws_map_id:
                return ws_base + off * 4   # F32 norm: 4 bytes/elem
            elif mid == self.q8_map_id:
                return q8_base + off       # native Q8: 1 byte/elem
            else:
                # Q4/Q6 tensor remapped to B12 Q8 map
                b12_off = self._b12_q8_off.get(name, None)
                return (b12_base + b12_off) if b12_off is not None else 0

        def _qp_addr(name):
            """Byte-level VkDeviceAddress for a tensor's Q8 qparam block."""
            mid = self.tensor_map_id.get(name)
            if mid == self.q8_map_id:
                off = self._off.get(name, 0)
                return q8_qp_base + (off // self.Q8_GROUP_SIZE) * 8
            else:
                # Q4/Q6 tensor: use B12 Q8 map qparams
                b12_off = self._b12_q8_off.get(name, None)
                return (b12_qp_base + (b12_off // self.Q8_GROUP_SIZE) * 8) if b12_off is not None else 0

        # Per-layer weight BDAs (data + qparams)
        weight_fields = [
            ('wq',  'qp_wq',  'attn_q.weight'),
            ('wk',  'qp_wk',  'attn_k.weight'),
            ('wv',  'qp_wv',  'attn_v.weight'),
            ('wo',  'qp_wo',  'attn_output.weight'),
            ('wg',  'qp_wg',  'ffn_gate.weight'),
            ('wu',  'qp_wu',  'ffn_up.weight'),
            ('wd',  'qp_wd',  'ffn_down.weight'),
        ]
        for w_field, qp_field, suffix in weight_fields:
            w_arr  = getattr(wa, w_field)
            qp_arr = getattr(wa, qp_field)
            for L in range(min(c.n_layer, 26)):
                name = f'{self._blk_names[L]}.{suffix}'
                if name in self._off:
                    w_arr[L]  = _w_addr(name)
                    qp_arr[L] = _qp_addr(name)

        # Per-layer norm BDAs (F32 in ws map — no qparams needed)
        norm_fields = [
            ('attn_norm',      'attn_norm.weight'),
            ('attn_q_norm',    'attn_q_norm.weight'),
            ('attn_k_norm',    'attn_k_norm.weight'),
            ('ffn_norm',       'ffn_norm.weight'),
            ('post_attn_norm', 'post_attention_norm.weight'),
            ('post_ffn_norm',  'post_ffw_norm.weight'),
        ]
        for field, suffix in norm_fields:
            arr = getattr(wa, field)
            for L in range(min(c.n_layer, 26)):
                name = f'{self._blk_names[L]}.{suffix}'
                if name in self._off:
                    arr[L] = _w_addr(name)

        wa.output_norm    = _w_addr('output_norm.weight')
        wa.attn_softcap   = float(c.attn_softcap)

        # Model dimensions
        wa.N_embd      = c.n_embd
        wa.N_q         = q_dim
        wa.N_k         = k_dim
        wa.N_v         = k_dim
        wa.N_ff        = c.n_ff
        wa.n_head      = c.n_head
        wa.n_head_kv   = c.n_head_kv
        wa.head_dim    = c.head_dim
        wa.head_dim_kv = c.head_dim_kv
        wa.group_size_attn = self.Q8_GROUP_SIZE
        wa.group_size_ffn  = self.Q8_GROUP_SIZE

        # Ensure dedicated B12 KV map exists so C code derives correct kv_cap
        self._ensure_b12_kv_map()

        try:
            self.gpu.full_decode_register(self.ws_map_id, self._B12_KV_MAP_ID, wa, c.n_layer)
            self._full_decode_registered = True
            self._full_decode_active = True
            self._full_decode_n_layer = c.n_layer
            if self.verbose:
                print(f"[GPU] B12 full_decode_register: ws={ws_base:#x} q8={q8_base:#x}"
                      f" q8_qp={q8_qp_base:#x} softcap={c.attn_softcap}"
                      f" → _full_decode_active=True")
        except Exception as e:
            self._full_decode_registered = False
            if self.verbose:
                print(f"[GPU] B12 full_decode_register failed: {e}")

    # ============================================================
    # Forward pass
    # ============================================================
    def _forward(self, tok, pos, return_logits=True, sample_mode=None, sample_cfg=None, sample_prev=None):
        c = self.cfg; g = self.gpu; ws_id = self.ws_map_id
        q_dim = c.n_head * c.head_dim; k_dim = c.n_head_kv * c.head_dim_kv
        seq_len = pos + 1
        trace_split = bool(getattr(self, "_trace_decode", False))
        te = self._timing_enabled

        # ---- Embedding lookup / prep ----
        if te: t0 = time.perf_counter()
        hid_h, _ = self._wsh('hidden')
        use_gpu_embed = False

        if self._gpu_token_embd_name is not None:
            self._prepare_gpu_embed_token(tok)
            use_gpu_embed = True
        elif 'token_embd.weight' in self._cpu_only:
            emb_t = self._load_tensor_f32('token_embd.weight', keep=True)
            emb_data = (emb_t[tok] if emb_t.shape[0] == c.n_vocab
                        else emb_t[:, tok]).astype(np.float32)
            if c.emb_scale != 1.0:
                emb_data = emb_data * np.float32(c.emb_scale)
            g.scatter(ws_id, self._hid_locs, emb_data)
        else:
            off = self._off['token_embd.weight']
            mid = self.tensor_map_id['token_embd.weight']
            emb_locs = np.arange(off + tok * c.n_embd,
                                 off + (tok + 1) * c.n_embd, dtype=np.uint32)
            emb_data = g.gather(mid, emb_locs).view(np.float32)
            if c.emb_scale != 1.0:
                emb_data = emb_data * np.float32(c.emb_scale)
            g.scatter(ws_id, self._hid_locs, emb_data)
        if te: self.timing['embed'] += time.perf_counter() - t0

        # ---- Pre-extract fixed workspace handles (same every token) ----
        hid_h    = self._ws_slots['hidden'][0]
        normed_h = self._ws_slots['normed'][0]
        q_h      = self._ws_slots['q'][0]
        k_h      = self._ws_slots['k'][0]
        v_h      = self._ws_slots['v'][0]
        ao_h     = self._ws_slots['attn_out'][0]
        op_h     = self._ws_slots['o_proj'][0]
        n2_h     = self._ws_slots['normed2'][0]
        gate_h   = self._ws_slots['gate'][0]
        up_h     = self._ws_slots['up'][0]
        act_h    = self._ws_slots['act'][0]
        ffn_h    = self._ws_slots['ffn_out'][0]

        # ---- Start batch: all GPU layer ops accumulate in one command buffer ----
        g.batch_begin()
        if use_gpu_embed:
            self._enqueue_gpu_embed_token(hid_h)

        # B12: monolithic shader — primary path (B13 will make it fast via N_WG dispatch)
        _b12 = (self._full_decode_active and not te
                and c.n_layer == getattr(self, '_full_decode_n_layer', -1))
        if _b12:
            self.gpu.full_decode_step(hid_h, normed_h, pos, seq_len)

        # B6: decode plan fallback when B12 not available
        _b6 = (not _b12 and self._decode_plan_active and not te
               and c.n_layer == self._decode_plan_n_layer)
        if _b6:
            self.gpu.decode_step(self._decode_plan_id, pos, seq_len)

        if not _b6 and not _b12:
         for L in range(c.n_layer):
            p = self._blk_names[L]; gl = c.is_global(L)
            alias_fast = self._fusion_alias_fast_enabled()

            # ---- Pre-attention RMSNorm ----
            if te: t0 = time.perf_counter()
            self._gpu_rmsnorm(hid_h, f'{p}.attn_norm.weight', normed_h,
                              1, c.n_embd, c.norm_eps)
            if te: self.timing['norm'] += time.perf_counter() - t0

            # ---- QKV projections: Q → q slot, K,V → k,v slots ----
            if te: t0 = time.perf_counter()
            q_name = f'{p}.attn_q.weight'
            k_name = f'{p}.attn_k.weight'
            v_name = f'{p}.attn_v.weight'
            direct_kv = self._direct_kv_cache_write and L in self._wh_v_heads
            direct_v_queued = False
            if alias_fast and L in self._qkv_merged_layers and not direct_kv:
                self._gpu_proj_merged(
                    normed_h,
                    self._wh_qkv_wt[L],
                    self._qkv_map_id[L],
                    self._weight_kind(q_name),
                    self._trans[q_name],
                    self._qkv_ws_out_h,
                    1,
                    c.n_embd,
                    q_dim + k_dim + k_dim,
                )
            elif alias_fast and direct_kv and self._qk_ws_out_h is not None and L in self._qk_merged_layers:
                self._gpu_proj_merged(
                    normed_h,
                    self._wh_qk_wt[L],
                    self._qk_map_id[L],
                    self._weight_kind(q_name),
                    self._trans[q_name],
                    self._qk_ws_out_h,
                    1,
                    c.n_embd,
                    q_dim + k_dim,
                )
                self._prepare_direct_v_write(L, pos)
                if self._fusion_scheduler_mode == 'alias_safe':
                    self._gpu_proj_heads_direct(
                        self._kv_proj_src_h,
                        self._wh_v_heads[L],
                        self._v_head_map_id[L],
                        self._v_head_kind[L],
                        self._kc_v_write_h[L],
                        c.n_embd,
                        c.head_dim_kv,
                        c.n_head_kv,
                    )
                    direct_v_queued = True
            else:
                self._gpu_proj(normed_h, q_name, q_h, 1, c.n_embd, q_dim)
                self._gpu_proj(normed_h, k_name, k_h, 1, c.n_embd, k_dim)
                if direct_kv:
                    self._prepare_direct_v_write(L, pos)
                    # In alias_safe we can keep this queued and let the
                    # upcoming K row_copy flush materialize it. Level-batched
                    # still cannot prove the normed_h alias, so execute it
                    # after the row_copy boundary below.
                    if self._fusion_scheduler_mode == 'alias_safe':
                        self._gpu_proj_heads_direct(
                            self._kv_proj_src_h,
                            self._wh_v_heads[L],
                            self._v_head_map_id[L],
                            self._v_head_kind[L],
                            self._kc_v_write_h[L],
                            c.n_embd,
                            c.head_dim_kv,
                            c.n_head_kv,
                        )
                        direct_v_queued = True
                else:
                    self._gpu_proj(normed_h, v_name, v_h, 1, c.n_embd, k_dim)
            if te: self.timing['qkv'] += time.perf_counter() - t0

            # ---- QK Norm (Gemma 3 only): out-of-place (in-place silently fails in batch) ----
            if te: t0 = time.perf_counter()
            qnn = f'{p}.attn_q_norm.weight'
            if qnn in self._wh:
                knn = f'{p}.attn_k_norm.weight'
                freq = c.rope_base_global if gl else c.rope_base_local
                fused_qk_norm_rope = (
                    self._qk_rope_out_h is not None
                    and self.tensor_map_id[qnn] == self.tensor_map_id[knn]
                )
                if fused_qk_norm_rope:
                    g.map_qk_norm_rope_x_dev(
                        ws_id,
                        self.tensor_map_id[qnn],
                        self._ws_rms_q_h,
                        self._wh[qnn],
                        self._ws_rms_k_h,
                        self._wh[knn],
                        self._qk_rope_out_h,
                        c.n_head,
                        c.head_dim,
                        c.n_head_kv,
                        c.head_dim_kv,
                        pos,
                        freq,
                    )
                else:
                    # Keep the Q/K temp views immediate and ordered until the
                    # alias model proves the unfused subgraph safe end-to-end.
                    g.fusion_flush()
                    with g.fusion_disabled():
                        self._gpu_rmsnorm(self._ws_rms_q_h, qnn, self._ws_rms_q_dst_h,
                                          c.n_head, c.head_dim, c.norm_eps)
                        self._gpu_rmsnorm(self._ws_rms_k_h, knn, self._ws_rms_k_dst_h,
                                          c.n_head_kv, c.head_dim_kv, c.norm_eps)
                        g.map_rope_dev(ws_id, self._ws_rms_q_dst_h, q_h,
                                       1, c.n_head, c.head_dim, pos, freq)
                        g.map_rope_dev(ws_id, self._ws_rms_k_dst_h, k_h,
                                       1, c.n_head_kv, c.head_dim_kv, pos, freq)
            else:
                freq = c.rope_base_global if gl else c.rope_base_local
                g.map_rope_dev(ws_id, self._q_rope_src_h, q_h, 1, c.n_head, c.head_dim, pos, freq)
                g.map_rope_dev(ws_id, self._k_rope_src_h, k_h, 1, c.n_head_kv, c.head_dim_kv, pos, freq)
            if te: self.timing['qk_norm'] += time.perf_counter() - t0

            # ---- RoPE: Q and K in workspace (out-of-place: same data positions,
            #      different src resource to avoid in-place Vulkan driver issue) ----
            if te: t0 = time.perf_counter()
            if te: self.timing['rope'] += time.perf_counter() - t0

            # ---- Copy K,V from workspace to KV cache (per-head-contiguous) ----
            g.map_row_copy_offset_dev(
                ws_id,
                self._kc_kv_copy_h[L],
                self._k_src_base_h,
                pos,
                c.n_head_kv if direct_kv else 2 * c.n_head_kv,
                c.head_dim_kv,
            )
            if direct_kv and not direct_v_queued:
                with g.fusion_disabled():
                    self._gpu_proj_heads_direct(
                        self._kv_proj_src_h,
                        self._wh_v_heads[L],
                        self._v_head_map_id[L],
                        self._v_head_kind[L],
                        self._kc_v_write_h[L],
                        c.n_embd,
                        c.head_dim_kv,
                        c.n_head_kv,
                    )

            # ---- Attention ----
            if te: t0 = time.perf_counter()
            if self._cpu_attention_fallback:
                g.batch_end()
                self._cpu_attention_step(L, seq_len)
                g.batch_begin()
            else:
                # scores[h, t] = Q_h @ K_cache_{h//gs}[t]^T
                # locs_a[h] = pos_q + h*head_dim  (_ws_rms_q_h reused)
                # locs_b[h] = kc_k_base[L] + (h//gs)*kv_step  (K cache head h's group)
                # locs_c[h] = fixed score row base for head h
                g.map_matmul_t_dev(ws_id,
                                   self._ws_rms_q_h,
                                   self._kc_k_locs_h[L],
                                   self._scores_row_h,
                                   1, c.head_dim, seq_len, c.n_head)

                # Scale, optional softcap, and softmax in one pass over fixed-stride rows.


                # row INDICES [0..n_head-1] — NOT element positions [h*seq_len].
                g.map_attn_softmax_value_dev(
                    ws_id,
                    self._scores_row_h,
                    self._kc_v_locs_h[L],
                    self._ao_locs_h,
                    c.n_head,
                    seq_len,
                    c.head_dim_kv,
                    float(1.0 / np.sqrt(c.head_dim)),
                    float(c.attn_softcap),
                )
                # `attn_out` is produced through row-base handles but consumed as a
                # contiguous slot by `o_proj`; keep that ordered for now.
                # Broadcom (level_batched) and desktop_discrete (direct_kv path with
                # map_attn_softmax_value_dev) both write attn_out as a single fused op —
                # the fusion system can order o_proj correctly without an explicit flush.
                if not (self._is_broadcom or self._is_integrated
                        or self._runtime_profile == 'desktop_discrete'):
                    g.fusion_flush()
            if te: self.timing['attn'] += time.perf_counter() - t0

            # ---- Output projection ----
            if te: t0 = time.perf_counter()
            o_in = c.n_head * c.head_dim_kv
            self._gpu_proj(ao_h, f'{p}.attn_output.weight', op_h, 1, o_in, c.n_embd)

            pan = f'{p}.post_attention_norm.weight'
            if pan in self._wh:
                g.map_rmsnorm_add_dev(ws_id, op_h, self._wh[pan], hid_h, hid_h,
                                      1, c.n_embd, c.norm_eps)
            else:
                g.map_op2_dev(ws_id, self.A.OP_ADD, hid_h, op_h, hid_h, c.n_embd)
            if te: self.timing['attn_out'] += time.perf_counter() - t0

            # ---- FFN ----
            if te: t0 = time.perf_counter()
            self._gpu_rmsnorm(hid_h, f'{p}.ffn_norm.weight', n2_h, 1, c.n_embd, c.norm_eps)

            gate_name = f'{p}.ffn_gate.weight'
            up_name = f'{p}.ffn_up.weight'
            if alias_fast and L in self._gateup_merged_layers:
                self._gpu_proj_merged(
                    n2_h,
                    self._wh_gateup_wt[L],
                    self._gateup_map_id[L],
                    self._weight_kind(gate_name),
                    self._trans[gate_name],
                    self._gateup_ws_out_h,
                    1,
                    c.n_embd,
                    2 * c.n_ff,
                )
            else:
                self._gpu_proj(n2_h, gate_name, gate_h, 1, c.n_embd, c.n_ff)
                self._gpu_proj(n2_h, up_name,   up_h,   1, c.n_embd, c.n_ff)
            fused_gate_op = self.A.OP_GEGLU if c.ffn_act == 'gelu' else self.A.OP_SWIGLU
            g.map_op2_dev(ws_id, fused_gate_op, gate_h, up_h, act_h, c.n_ff)
            self._gpu_proj(act_h, f'{p}.ffn_down.weight', ffn_h, 1, c.n_ff, c.n_embd)

            pfn = f'{p}.post_ffw_norm.weight'
            if pfn in self._wh:
                g.map_rmsnorm_add_dev(ws_id, ffn_h, self._wh[pfn], hid_h, hid_h,
                                      1, c.n_embd, c.norm_eps)
            else:
                g.map_op2_dev(ws_id, self.A.OP_ADD, hid_h, ffn_h, hid_h, c.n_embd)
            if te: self.timing['ffn'] += time.perf_counter() - t0

        # ---- Final norm (still in batch; B6/B12 path includes this in decode plan/shader) ----
        normed_final_h, _ = self._wsh('normed')
        if not _b6 and not _b12:
            if te: t0 = time.perf_counter()
            self._gpu_rmsnorm(hid_h, 'output_norm.weight', normed_final_h, 1, c.n_embd, c.norm_eps)
            if te: self.timing['norm'] += time.perf_counter() - t0

        if trace_split:
            t_batch = time.perf_counter()
            g.batch_end()
            self.timing['core_batch'] += time.perf_counter() - t_batch
            g.batch_begin()

        # ---- LM head: stays inside batch so we only pay one fence wait ----
        if te: t0 = time.perf_counter()
        lo_h, _ = self._wsh('logits')
        approx_partial_k = 0
        approx_penalty_n = 0
        approx_partial_val_h = None
        if self._lm_wt_name:
            if sample_mode == 'gpu_approx_rerank':
                approx_partial_k = min(self._gpu_approx_partial_k, self.cfg.n_vocab)
                if sample_cfg.repeat_penalty != 1.0 and sample_prev:
                    approx_penalty_n = self._prepare_repeat_ids(sample_prev)
                partial_idx_h, _ = self._wsh('sample_fused_idx')
                partial_val_h, _ = self._wsh('sample_fused_val')
                approx_partial_val_h = partial_val_h
                rows_per_group = self._effective_gpu_fused_rows_per_group(sample_cfg)
                sample_fused_groups = (c.n_vocab + rows_per_group - 1) // rows_per_group
                self._sample_fused_partial_n = sample_fused_groups * approx_partial_k
                self._sample_rerank_n = 0
                g.map_matvec_topk_t_xq4_ex_dev(
                    ws_id,
                    self.wt_map_id,
                    normed_final_h,
                    self._lm_q4_approx_h,
                    self._sample_repeat_ids_h,
                    partial_idx_h,
                    partial_val_h,
                    c.n_embd,
                    c.n_vocab,
                    approx_partial_k,
                    approx_penalty_n,
                    float(sample_cfg.repeat_penalty),
                    rows_per_group,
                )
            elif sample_mode == 'gpu_fused_topk':
                top_k = self._effective_sample_top_k(sample_cfg)
                n_penalty = 0
                if sample_cfg.repeat_penalty != 1.0 and sample_prev:
                    n_penalty = self._prepare_repeat_ids(sample_prev)
                partial_idx_h, _ = self._wsh('sample_fused_idx')
                partial_val_h, _ = self._wsh('sample_fused_val')
                rows_per_group = self._effective_gpu_fused_rows_per_group(sample_cfg)
                sample_fused_groups = (c.n_vocab + rows_per_group - 1) // rows_per_group
                self._sample_fused_partial_n = sample_fused_groups * top_k
                wt_mid = self.tensor_map_id[self._lm_wt_name]
                if te: shortlist_t0 = time.perf_counter()
                wt_kind = self._gpu_fused_topk_weight_kind()
                if wt_kind == 'q8' and getattr(g, '_has_map_matvec_topk_t_xq8_ex_dev', False):
                    g.map_matvec_topk_t_xq8_ex_dev(
                        ws_id,
                        wt_mid,
                        normed_final_h,
                        self._wh[self._lm_wt_name],
                        self._sample_repeat_ids_h,
                        partial_idx_h,
                        partial_val_h,
                        c.n_embd,
                        c.n_vocab,
                        top_k,
                        n_penalty,
                        float(sample_cfg.repeat_penalty),
                        rows_per_group,
                    )
                elif wt_kind == 'q8':
                    g.map_matvec_topk_t_xq8_dev(
                        ws_id,
                        wt_mid,
                        normed_final_h,
                        self._wh[self._lm_wt_name],
                        self._sample_repeat_ids_h,
                        partial_idx_h,
                        partial_val_h,
                        c.n_embd,
                        c.n_vocab,
                        top_k,
                        n_penalty,
                        float(sample_cfg.repeat_penalty),
                    )
                elif wt_kind == 'q4' and getattr(g, '_has_map_matvec_topk_t_xq4_ex_dev', False):
                    g.map_matvec_topk_t_xq4_ex_dev(
                        ws_id,
                        wt_mid,
                        normed_final_h,
                        self._wh[self._lm_wt_name],
                        self._sample_repeat_ids_h,
                        partial_idx_h,
                        partial_val_h,
                        c.n_embd,
                        c.n_vocab,
                        top_k,
                        n_penalty,
                        float(sample_cfg.repeat_penalty),
                        rows_per_group,
                    )
                elif wt_kind == 'q4':
                    g.map_matvec_topk_t_xq4_dev(
                        ws_id,
                        wt_mid,
                        normed_final_h,
                        self._wh[self._lm_wt_name],
                        self._sample_repeat_ids_h,
                        partial_idx_h,
                        partial_val_h,
                        c.n_embd,
                        c.n_vocab,
                        top_k,
                        n_penalty,
                        float(sample_cfg.repeat_penalty),
                    )
                else:
                    raise RuntimeError(f"gpu_fused_topk unavailable for LM head '{self._lm_wt_name}'")
                idx_h, _ = self._wsh('sample_topk_idx')
                if top_k == 1:
                    g.map_argmax_dev(ws_id, partial_val_h, idx_h, self._sample_fused_partial_n)
                else:
                    val_h, _ = self._wsh('sample_topk_val')
                    prob_h, _ = self._wsh('sample_topk_prob')
                    g.map_topk_dev(ws_id, partial_val_h, idx_h, val_h, self._sample_fused_partial_n, top_k)
                    if c.final_softcap > 0:
                        fcap_h, _ = self._wsh('final_softcap')
                        g.map_broadcast_dev(ws_id, self.A.OP_DIV, val_h, fcap_h, val_h, top_k)
                        g.map_op1_dev(ws_id, self.A.OP_TANH, val_h, val_h, top_k)
                        g.map_broadcast_dev(ws_id, self.A.OP_MUL, val_h, fcap_h, val_h, top_k)
                    if sample_cfg.temperature > 0.0:
                        g.map_topp_dev(ws_id, val_h, prob_h, top_k, float(sample_cfg.temperature), float(sample_cfg.top_p))
                self._sample_fused_topk_n = top_k
                # ---- Fuse resolve + categorical sample into forward batch ----
                # This eliminates a separate batch_begin/batch_end (fence wait)
                # in _consume_gpu_fused_topk, saving ~0.5-1ms per token.
                can_resolve = self._can_gpu_resolve_idx()
                can_dev_sample = self._can_gpu_categorical_sample()
                if can_resolve:
                    tok_h, _ = self._wsh('sample_token')
                    if top_k == 1 or (sample_cfg and sample_cfg.temperature <= 0.0):
                        g.map_resolve_idx_dev(ws_id, self._sample_fused_base_h, idx_h, tok_h, 1)
                        self._sample_token_ready = True
                    elif can_dev_sample and sample_cfg and sample_cfg.temperature > 0.0:
                        resolved_h, _ = self._wsh('sample_short_sel')
                        g.map_resolve_idx_dev(ws_id, self._sample_fused_base_h, idx_h, resolved_h, top_k)
                        rand_h, _ = self._wsh('sample_rand')
                        self._gpu_write_sample_rand()
                        g.map_sample_categorical_dev(ws_id, resolved_h, prob_h, rand_h, tok_h, top_k)
                        self._sample_token_ready = True
                if te: self.timing['lm_head_shortlist'] += time.perf_counter() - shortlist_t0
            else:
                # GPU path: output.weight is in the Q4/Q8 map → one GPU matmul
                self._gpu_proj(normed_final_h, self._lm_wt_name, lo_h, 1, c.n_embd, c.n_vocab)
                # Final logit soft-cap: tanh(x / cap) * cap (Gemma 3: cap=30.0)
                if c.final_softcap > 0:
                    fcap_h, _ = self._wsh('final_softcap')
                    g.map_broadcast_dev(ws_id, self.A.OP_DIV, lo_h, fcap_h, lo_h, c.n_vocab)
                    g.map_op1_dev(ws_id, self.A.OP_TANH, lo_h, lo_h, c.n_vocab)
                    g.map_broadcast_dev(ws_id, self.A.OP_MUL, lo_h, fcap_h, lo_h, c.n_vocab)
                # ---- Fuse argmax/topk sampling into forward batch ----
                if sample_mode == 'gpu_argmax':
                    tok_h, _ = self._wsh('sample_token')
                    g.map_argmax_dev(ws_id, lo_h, tok_h, c.n_vocab)
                    self._sample_token_ready = True
                elif sample_mode == 'gpu_topk' and sample_cfg is not None:
                    # Apply repeat penalty on GPU if supported
                    if sample_cfg.repeat_penalty != 1.0 and sample_prev:
                        if getattr(g, '_has_map_repeat_penalty_dev', False):
                            n_rep = self._prepare_repeat_ids(sample_prev)
                            if n_rep:
                                g.map_repeat_penalty_dev(ws_id, lo_h, self._sample_repeat_ids_h,
                                                         n_rep, float(sample_cfg.repeat_penalty))
                    # topk → topp → categorical sample — all inside the batch
                    _top_k = self._effective_sample_top_k(sample_cfg)
                    _idx_h, _ = self._wsh('sample_topk_idx')
                    _val_h, _ = self._wsh('sample_topk_val')
                    _prob_h, _ = self._wsh('sample_topk_prob')
                    g.map_topk_dev(ws_id, lo_h, _idx_h, _val_h, c.n_vocab, _top_k)
                    if sample_cfg.temperature > 0.0:
                        g.map_topp_dev(ws_id, _val_h, _prob_h, _top_k,
                                       float(sample_cfg.temperature), float(sample_cfg.top_p))
                        if self._can_gpu_categorical_sample():
                            _rand_h, _ = self._wsh('sample_rand')
                            tok_h, _ = self._wsh('sample_token')
                            self._gpu_write_sample_rand()
                            g.map_sample_categorical_dev(ws_id, _idx_h, _prob_h, _rand_h, tok_h, _top_k)
                            self._sample_token_ready = True
        else:
            # CPU fallback: weight-tied (token_embd.weight) — gather hidden,
            # multiply on CPU, scatter result.  One gather + one scatter per token.
            pass  # handled after batch_end below

        # ---- Fold Broadcom rerank into main batch (saves one fence wait) ----
        _rerank_folded = False
        if self._lm_wt_name and sample_mode == 'gpu_approx_rerank':
            partial_n = self._sample_fused_partial_n
            if partial_n > 0 and self._can_gpu_merge_approx_shortlist() and approx_partial_val_h is not None:
                shortlist_n = min(self.SAMPLE_TOPK_MAX, self.SAMPLE_APPROX_SHORTLIST_MAX, partial_n)
                if shortlist_n > 0:
                    short_sel_h, _ = self._wsh('sample_short_sel')
                    val_h, _ = self._wsh('sample_exact_val')
                    g.map_topk_dev(
                        ws_id,
                        approx_partial_val_h,
                        short_sel_h,
                        val_h,
                        partial_n,
                        shortlist_n,
                    )
                    wt_mid = self.tensor_map_id[self._lm_wt_name]
                    g.map_matvec_rerank_t_xq8_dev(
                        ws_id,
                        wt_mid,
                        normed_final_h,
                        self._wh[self._lm_wt_name],
                        self._sample_fused_base_h,
                        self._sample_short_sel_chunk_h[0],
                        self._sample_repeat_ids_h,
                        self._sample_exact_idx_chunk_h[0],
                        self._sample_exact_val_chunk_h[0],
                        c.n_embd,
                        shortlist_n,
                        approx_penalty_n,
                        float(sample_cfg.repeat_penalty),
                    )
                    self._sample_rerank_n = shortlist_n
                    _rerank_folded = True
                    # Also fuse final sampling into the batch
                    _can_resolve = self._can_gpu_resolve_idx()
                    _can_dev = self._can_gpu_categorical_sample()
                    if _can_resolve and shortlist_n > 0:
                        _exact_val_h, _ = self._wsh('sample_exact_val')
                        _topk_idx_h, _ = self._wsh('sample_topk_idx')
                        _topk_val_h, _ = self._wsh('sample_topk_val')
                        _top_k = min(self._effective_sample_top_k(sample_cfg), shortlist_n, self.SAMPLE_TOPK_MAX)
                        if _top_k <= 0: _top_k = min(shortlist_n, self.SAMPLE_TOPK_MAX)
                        g.map_topk_dev(ws_id, _exact_val_h, _topk_idx_h, _topk_val_h, shortlist_n, _top_k)
                        if c.final_softcap > 0:
                            fcap_h, _ = self._wsh('final_softcap')
                            g.map_broadcast_dev(ws_id, self.A.OP_DIV, _topk_val_h, fcap_h, _topk_val_h, _top_k)
                            g.map_op1_dev(ws_id, self.A.OP_TANH, _topk_val_h, _topk_val_h, _top_k)
                            g.map_broadcast_dev(ws_id, self.A.OP_MUL, _topk_val_h, fcap_h, _topk_val_h, _top_k)
                        tok_h, _ = self._wsh('sample_token')
                        if sample_cfg.temperature <= 0.0:
                            _ebase_h = self._sample_exact_base_h
                            g.map_resolve_idx_dev(ws_id, _ebase_h, _topk_idx_h, tok_h, 1)
                            self._sample_token_ready = True
                        elif _can_dev:
                            _ebase_h = self._sample_exact_base_h
                            _resolved_h, _ = self._wsh('sample_short_sel')
                            _prob_h, _ = self._wsh('sample_topk_prob')
                            g.map_resolve_idx_dev(ws_id, _ebase_h, _topk_idx_h, _resolved_h, _top_k)
                            g.map_topp_dev(ws_id, _topk_val_h, _prob_h, _top_k,
                                           float(sample_cfg.temperature), float(sample_cfg.top_p))
                            _rand_h, _ = self._wsh('sample_rand')
                            self._gpu_write_sample_rand()
                            g.map_sample_categorical_dev(ws_id, _resolved_h, _prob_h, _rand_h, tok_h, _top_k)
                            self._sample_token_ready = True

        if trace_split:
            if te: self.timing['lm_head'] += time.perf_counter() - t0
            t_batch = time.perf_counter()
            g.batch_end()
            self.timing['lm_head_batch'] += time.perf_counter() - t_batch
        else:
            # ---- End batch: one fence wait for ALL layer + LM-head GPU ops ----
            g.batch_end()
            if te: self.timing['lm_head'] += time.perf_counter() - t0

        if self._lm_wt_name and sample_mode == 'gpu_approx_rerank' and not _rerank_folded:
            shortlist_n = 0
            partial_n = self._sample_fused_partial_n
            if partial_n > 0:
                partial_vals = g.gather(
                    ws_id, self._sample_fused_val_locs[:partial_n]).view(np.float32)
                valid_pos = np.flatnonzero(np.isfinite(partial_vals))
                if valid_pos.size:
                    shortlist_n = min(self.SAMPLE_APPROX_SHORTLIST_MAX, int(valid_pos.size))
                    if shortlist_n < valid_pos.size:
                        keep_rel = np.argpartition(
                            partial_vals[valid_pos], -shortlist_n)[-shortlist_n:]
                        keep_pos = valid_pos[keep_rel]
                    else:
                        keep_pos = valid_pos
                    keep_pos = keep_pos[np.argsort(partial_vals[keep_pos])[::-1]]
                    self._sample_short_sel_arr[:shortlist_n] = keep_pos.astype(np.float32, copy=False)
                    short_sel_h, _ = self._wsh('sample_short_sel')
                    idx_h, _ = self._wsh('sample_exact_idx')
                    val_h, _ = self._wsh('sample_exact_val')
                    g.batch_begin()
                    try:
                        g.scatter(
                            ws_id,
                            self._sample_short_sel_locs[:shortlist_n],
                            self._sample_short_sel_arr[:shortlist_n],
                        )
                        wt_mid = self.tensor_map_id[self._lm_wt_name]
                        for chunk_i, start in enumerate(range(0, shortlist_n, self.SAMPLE_TOPK_MAX)):
                            chunk_n = min(self.SAMPLE_TOPK_MAX, shortlist_n - start)
                            g.map_matvec_rerank_t_xq8_dev(
                                ws_id,
                                wt_mid,
                                normed_final_h,
                                self._wh[self._lm_wt_name],
                                self._sample_fused_base_h,
                                self._sample_short_sel_chunk_h[chunk_i],
                                self._sample_repeat_ids_h,
                                self._sample_exact_idx_chunk_h[chunk_i],
                                self._sample_exact_val_chunk_h[chunk_i],
                                c.n_embd,
                                chunk_n,
                                approx_penalty_n,
                                float(sample_cfg.repeat_penalty),
                            )
                    finally:
                        if trace_split:
                            t_batch = time.perf_counter()
                            g.batch_end()
                            self.timing['rerank_batch'] += time.perf_counter() - t_batch
                        else:
                            g.batch_end()
                    self._sample_rerank_n = shortlist_n

        if not self._lm_wt_name:
            # CPU fallback (weight-tied models): gather from 'normed' slot
            # (output_norm now writes there out-of-place)
            hid_data = g.gather(ws_id, self._normed_locs).view(np.float32)
            wt = self._lm_weight
            logits_cpu = (wt @ hid_data) if wt.shape[0] == c.n_vocab else (wt.T @ hid_data)
            if c.final_softcap > 0:
                cap = np.float32(c.final_softcap)
                logits_cpu = np.tanh(logits_cpu / cap) * cap
            g.scatter(ws_id, self._lo_locs, logits_cpu)

        if not return_logits:
            if te and not trace_split:
                self.timing['lm_head'] += time.perf_counter() - t0
            return None

        logits = g.gather(ws_id, self._lo_locs).view(np.float32)
        if te and not trace_split:
            self.timing['lm_head'] += time.perf_counter() - t0
        return logits

    # ============================================================
    # Sampling
    # ============================================================
    def _sample(self, logits, cfg, prev):
        logits = logits.copy()
        if cfg.repeat_penalty != 1.0 and prev:
            for t in set(prev[-64:]):
                if t < len(logits):
                    if logits[t] > 0: logits[t] /= cfg.repeat_penalty
                    else: logits[t] *= cfg.repeat_penalty
        if cfg.temperature <= 0: return int(np.argmax(logits))
        logits /= cfg.temperature
        if cfg.top_k > 0:
            tk = min(cfg.top_k, len(logits))
            ti = np.argpartition(logits, -tk)[-tk:]
            m = np.full_like(logits, -np.inf); m[ti] = logits[ti]; logits = m
        p = np.exp(logits - np.max(logits)); p /= p.sum()
        if cfg.top_p < 1.0:
            si = np.argsort(p)[::-1]; cs = np.cumsum(p[si])
            co = np.searchsorted(cs, cfg.top_p) + 1
            m = np.zeros_like(p); m[si[:co]] = p[si[:co]]; p = m / m.sum()
        return int(np.random.choice(len(p), p=p))

    def _repeat_history(self, prompt_tokens, generated_tokens, cfg):
        if cfg.repeat_penalty == 1.0:
            return None
        if cfg.repeat_on_prompt:
            hist = list(prompt_tokens) + list(generated_tokens)
        else:
            hist = list(generated_tokens)
        return hist if hist else None

    def _summarize_native_decode_stats(self, native_stats, n_tokens):
        if not native_stats:
            return {}
        denom = float(max(0, int(n_tokens)))
        summary = {}
        for key in (
            'dispatch_count',
            'submit_count',
            'barrier_count',
            'fusion_flush_count',
            'descriptor_set_update_count',
            'descriptor_cache_hit_count',
            'descriptor_cache_miss_count',
            'alias_conflict_count',
        ):
            total = int(native_stats.get(key, 0))
            summary[f'{key}_total'] = total
            summary[f'{key}_per_token'] = (total / denom) if denom > 0.0 else 0.0
        if native_stats.get('scheduler_mode') is not None:
            summary['scheduler_mode'] = int(native_stats.get('scheduler_mode', 0))
        if native_stats.get('scheduler_mode_name') is not None:
            summary['scheduler_mode_name'] = str(native_stats.get('scheduler_mode_name'))
        return summary

    def _summarize_decode_trace(self, trace_steps, native_stats=None, n_tokens=None):
        if not trace_steps:
            return self._summarize_native_decode_stats(native_stats or {}, n_tokens or 0)
        denom = float(len(trace_steps))
        stage_keys = (
            'embed', 'norm', 'qkv', 'qk_norm', 'rope', 'attn', 'attn_out',
            'ffn', 'lm_head', 'lm_head_shortlist', 'sample_resolve',
            'core_batch', 'lm_head_batch', 'rerank_batch'
        )
        summary = {
            'step_ms_avg': sum(t['step_ms'] for t in trace_steps) / denom,
            'sample_ms_avg': sum(t['sample_ms'] for t in trace_steps) / denom,
            'forward_ms_avg': sum(t['forward_ms'] for t in trace_steps) / denom,
        }
        for key in stage_keys:
            summary[f'{key}_ms_avg'] = sum(t['timing_ms'].get(key, 0.0) for t in trace_steps) / denom
        summary.update(self._summarize_native_decode_stats(native_stats or {}, n_tokens or len(trace_steps)))
        return summary

    def _select_sampling_mode(self, config) -> str:
        effective_top_k = self._effective_sample_top_k(config)
        gpu_fused_topk = self._can_gpu_fused_topk_sample(config)
        gpu_approx_rerank = (effective_top_k > 1) and self._can_gpu_approx_rerank_sample(config)
        gpu_argmax = self._can_gpu_argmax_sample(config)
        gpu_topk = self._can_gpu_topk_sample(config)
        if effective_top_k == 1 and gpu_fused_topk:
            return 'gpu_fused_topk'
        if gpu_argmax:
            return 'gpu_argmax'
        if gpu_approx_rerank:
            return 'gpu_approx_rerank'
        if gpu_fused_topk:
            return 'gpu_fused_topk'
        if gpu_topk:
            return 'gpu_topk'
        return 'cpu'

    # ============================================================
    # Generate
    # ============================================================
    def generate(self, token_ids, config=None, stream=True, **kw):
        if config is None: config = GenerationConfig()
        if config.seed is not None: np.random.seed(config.seed)
        progress_interval_s = float(kw.get('progress_interval_s', 0.25))
        trace_decode = bool(kw.get('trace_decode', False))
        reuse_prefix = max(0, int(kw.get('reuse_prefix', 0) or 0))
        g = self.gpu
        gpu_argmax = self._can_gpu_argmax_sample(config)
        gpu_topk = self._can_gpu_topk_sample(config)
        sampling_mode = self._select_sampling_mode(config)

        total_needed = len(token_ids) + config.max_tokens
        if total_needed > self._kv_cap:
            raise RuntimeError(
                f"KV cache too small: need {total_needed} tokens but "
                f"kv_cap={self._kv_cap}. Reinitialize with kv_cap>={total_needed}.")

        if reuse_prefix >= len(token_ids):
            reuse_prefix = 0
        self.reset()
        self._timing_enabled = trace_decode
        self._trace_decode = trace_decode
        prompt_prefill_n = max(0, len(token_ids) - reuse_prefix)
        all_tok = list(token_ids[:reuse_prefix]); out_tok = []

        tp = time.perf_counter()
        logits = None
        last_prompt_idx = len(token_ids) - 1
        for i in range(reuse_prefix, len(token_ids)):
            t = token_ids[i]
            need_logits = (i == last_prompt_idx) and (sampling_mode == 'cpu')
            sample_mode = sampling_mode if (
                i == last_prompt_idx and
                sampling_mode in ('gpu_fused_topk', 'gpu_approx_rerank')) else None
            repeat_prev = self._repeat_history(token_ids, out_tok, config) if sample_mode else None
            logits = self._forward(t, i, return_logits=need_logits,
                                   sample_mode=sample_mode, sample_cfg=config,
                                   sample_prev=repeat_prev)
        t_pre = time.perf_counter() - tp

        if trace_decode and hasattr(g, 'reset_native_stats'):
            g.reset_native_stats()

        td = time.perf_counter(); ttimes = []
        last_progress = td
        trace_steps = []
        for step in range(config.max_tokens):
            t0 = time.perf_counter()
            t_sample0 = time.perf_counter()
            if sampling_mode == 'gpu_approx_rerank':
                nt = self._sample_gpu_approx_rerank(config)
            elif sampling_mode == 'gpu_fused_topk':
                nt = self._consume_gpu_fused_topk(config)
            elif gpu_argmax:
                nt = self._sample_gpu_argmax()
            elif gpu_topk:
                if self._sample_token_ready:
                    self._sample_token_ready = False
                    nt = self._gpu_read_sample_token()
                else:
                    repeat_prev = self._repeat_history(token_ids, out_tok, config)
                    nt = self._sample_gpu_topk(config, repeat_prev)
            else:
                repeat_prev = self._repeat_history(token_ids, out_tok, config)
                nt = self._sample(logits, config, repeat_prev)
            sample_dt = time.perf_counter() - t_sample0
            if nt in config.eos_token_ids: break
            all_tok.append(nt); out_tok.append(nt)
            sample_mode = sampling_mode if sampling_mode in ('gpu_fused_topk', 'gpu_approx_rerank', 'gpu_topk', 'gpu_argmax') else None
            repeat_prev = self._repeat_history(token_ids, out_tok, config) if sample_mode in ('gpu_fused_topk', 'gpu_approx_rerank', 'gpu_topk') else None
            timing_before = dict(self.timing) if trace_decode else None
            t_forward0 = time.perf_counter()
            logits = self._forward(nt, len(token_ids) + step, return_logits=(sampling_mode == 'cpu'),
                                   sample_mode=sample_mode, sample_cfg=config,
                                   sample_prev=repeat_prev)
            forward_dt = time.perf_counter() - t_forward0
            dt = time.perf_counter() - t0; ttimes.append(dt)
            if trace_decode:
                timing_after = dict(self.timing)
                delta = {
                    key: max(0.0, (timing_after.get(key, 0.0) - timing_before.get(key, 0.0))) * 1000.0
                    for key in timing_after
                }
                trace_steps.append({
                    'step': step,
                    'token_id': int(nt),
                    'step_ms': dt * 1000.0,
                    'sample_ms': sample_dt * 1000.0,
                    'forward_ms': forward_dt * 1000.0,
                    'timing_ms': delta,
                })
            now = time.perf_counter()
            if stream and (now - last_progress >= progress_interval_s or step + 1 == config.max_tokens):
                sys.stdout.write(f"\r  [{step+1}/{config.max_tokens}] {1/dt:.1f} tok/s")
                sys.stdout.flush()
                last_progress = now
        t_dec = time.perf_counter() - td
        if stream: print()

        np_ = len(token_ids); ng = len(out_tok)
        native_stats = (
            g.get_native_stats()
            if trace_decode and hasattr(g, 'get_native_stats')
            else {}
        )
        active_rows = (
            self._effective_gpu_fused_rows_per_group(config)
            if sampling_mode in ('gpu_fused_topk', 'gpu_approx_rerank')
            else None
        )
        return out_tok, {
            'n_prompt': prompt_prefill_n,
            'n_prompt_total': np_,
            'n_prompt_reused': reuse_prefix,
            'n_prompt_prefilled': prompt_prefill_n,
            'n_gen': ng,
            'prefill_s': t_pre, 'decode_s': t_dec,
            'total_s': time.perf_counter() - tp,
            'prefill_tps': prompt_prefill_n/t_pre if t_pre > 0 else 0,
            'decode_tps': ng/t_dec if t_dec > 0 else 0,
            'avg_ms': np.mean(ttimes)*1000 if ttimes else 0,
            'sampling_mode': sampling_mode,
            'gpu_fused_rows_per_group_active': active_rows,
            'timing': dict(self.timing),
            'native_stats': native_stats if trace_decode else {},
            'trace_summary': (
                self._summarize_decode_trace(trace_steps, native_stats=native_stats, n_tokens=ng)
                if trace_decode else {}
            ),
            'trace_steps': trace_steps if trace_decode else [],
        }

    def reset(self):
        """Reset scores buffer for a new generation."""
        c = self.cfg; g = self.gpu; ws_id = self.ws_map_id
        sc_sz = c.n_head * self._kv_cap
        g.scatter(ws_id, np.arange(sc_sz, dtype=np.uint32),
                  np.full(sc_sz, -1e30, dtype=np.float32))
        self._sample_fused_topk_n = 0
        self._sample_fused_partial_n = 0
        self._sample_rerank_n = 0
        self._sample_token_ready = False
        self._reset_timing()

    def _reset_timing(self):
        self.timing = {k: 0.0 for k in [
            'embed', 'norm', 'qkv', 'qk_norm', 'rope', 'attn', 'attn_out',
            'ffn', 'lm_head', 'lm_head_shortlist', 'sample_resolve',
            'core_batch', 'lm_head_batch', 'rerank_batch']}
