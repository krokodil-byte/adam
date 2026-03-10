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

    @classmethod
    def estimate_persistent_gpu_bytes(cls, cfg: ModelConfig,
                                      tensor_shapes: Dict[str, tuple],
                                      tensor_types: Dict[str, int],
                                      force_f32: bool = False,
                                      kv_cap: Optional[int] = None,
                                      gpu_tied_lm_head: bool = True,
                                      gpu_approx_rerank: bool = False,
                                      gpu_fused_rows_per_group: Optional[int] = None) -> Dict[str, int]:
        kv_cap = int(kv_cap if kv_cap is not None else cls.KV_CAP_DEFAULT)
        rows_per_group = int(gpu_fused_rows_per_group or cls.SAMPLE_FUSED_ROWS_PER_GROUP)
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
        sample_fused_partial_cap = sample_fused_groups * cls.SAMPLE_TOPK_MAX
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
        self._gpu_fused_rows_per_group = max(
            1, int(kw.get('gpu_fused_rows_per_group', self.SAMPLE_FUSED_ROWS_PER_GROUP)))
        if self._production_mode and self._cpu_attention_fallback:
            raise ValueError("production_mode requires GPU attention; cpu_attention_fallback is disabled")
        self._host_state_pruned = False
        self._detect_dims()
        self._build_layout()
        self._alloc_workspace()
        self._upload_all()
        self._alloc_kv_cache()
        self._upload_locs()
        self._cache_lm_weight()
        if self._production_mode:
            self.release_host_state()
        self.timing = {}; self._reset_timing()
        # The backend now flushes queued fusion ops into the active batch before
        # any immediate op and before batch_end(), so decode can safely keep
        # fusion enabled and let unary/binary/broadcast ops batch together.
        self.gpu.fusion_enable(True)
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

        for name in sorted(self._all_tensor_names()):
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
        self._sample_fused_partial_cap = self._sample_fused_groups * self.SAMPLE_TOPK_MAX

        # --- Scores at element 0 ---
        pos = 0
        sc_sz = c.n_head * kv_cap
        sc_locs = np.arange(0, sc_sz, dtype=np.uint32)
        sc_h, _ = g.upload_dev(sc_locs)
        self._ws_slots['scores'] = (sc_h, sc_sz, 0)
        self._scores_row_bases = np.arange(0, sc_sz, kv_cap, dtype=np.uint32)
        self._scores_row_h, _ = g.upload_dev(self._scores_row_bases)
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
        self._pos_scale = pos
        pos += 1

        # --- Embedding scale scalar ---
        if c.emb_scale != 1.0:
            emb_scale_locs = np.full(c.n_embd, pos, dtype=np.uint32)
            emb_scale_h, _ = g.upload_dev(emb_scale_locs)
            self._ws_slots['emb_scale'] = (emb_scale_h, c.n_embd, pos)
            self._pos_emb_scale = pos
            pos += 1

        # --- Softcap scalars (Gemma 3 and similar) ---
        if c.attn_softcap > 0:
            attn_cap_locs = np.full(sc_sz, pos, dtype=np.uint32)
            attn_cap_h, _ = g.upload_dev(attn_cap_locs)
            self._ws_slots['attn_softcap'] = (attn_cap_h, sc_sz, pos)
            pos += 1
        if c.final_softcap > 0:
            # final_softcap is applied to logits of shape [n_vocab], so the locs
            # buffer must have n_vocab elements (same broadcast-size rule as scale).
            final_cap_locs = np.full(c.n_vocab, pos, dtype=np.uint32)
            final_cap_h, _ = g.upload_dev(final_cap_locs)
            self._ws_slots['final_softcap'] = (final_cap_h, c.n_vocab, pos)
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
        # _ws_rms_q_h doubles as locs_a for scores matmul (same values)
        # Separate dst handle (same locs, different resource) for out-of-place RMSNorm
        self._ws_rms_q_dst_h, _ = g.upload_dev(q_rms_locs.copy())

        # --- QK-norm locs for K (workspace k slot) ---
        pos_k = self._ws_slots['k'][2]
        k_rms_locs = np.array([pos_k + i * c.head_dim_kv
                                for i in range(c.n_head_kv)], dtype=np.uint32)
        self._ws_rms_k_h, _ = g.upload_dev(k_rms_locs)
        # Separate dst handle for out-of-place RMSNorm
        self._ws_rms_k_dst_h, _ = g.upload_dev(k_rms_locs.copy())

        # --- RoPE src handles (separate resource from q_h/k_h to avoid in-place issue) ---
        q_locs = np.arange(pos_q, pos_q + q_dim, dtype=np.uint32)
        self._q_rope_src_h, _ = g.upload_dev(q_locs.copy())
        k_locs = np.arange(pos_k, pos_k + k_dim, dtype=np.uint32)
        self._k_rope_src_h, _ = g.upload_dev(k_locs.copy())

        # --- row_copy source-base handles (scalar = element offset of slot start) ---
        self._k_src_base_h, _ = g.upload_dev(np.array([pos_k], dtype=np.uint32))
        pos_v = self._ws_slots['v'][2]
        self._v_src_base_h, _ = g.upload_dev(np.array([pos_v], dtype=np.uint32))

        # --- KV cache: per-head-contiguous layout ---
        # Each KV head g of layer L occupies kv_cap * head_dim_kv elements:
        #   K: kc_k_base[L] + g * kv_cap * head_dim_kv
        #   V: kc_v_base[L] + g * kv_cap * head_dim_kv
        # Align start to head_dim_kv so dst_loc = element_off / head_dim_kv is integer.
        kv_step = kv_cap * c.head_dim_kv        # elements per KV head
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

        # locs_c for weighted sum output (fixed)
        pos_ao = self._ws_slots['attn_out'][2]
        ao_locs = np.array([pos_ao + h * c.head_dim_kv for h in range(c.n_head)],
                            dtype=np.uint32)
        self._ao_locs_h, _ = g.upload_dev(ao_locs)

        # --- row_copy copy_spec handles (fixed base rows; `pos` is added at dispatch) ---
        self._kc_k_copy_h = []; self._kc_v_copy_h = []
        for L in range(c.n_layer):
            k_spec = np.empty(2 * c.n_head_kv, dtype=np.uint32)
            v_spec = np.empty(2 * c.n_head_kv, dtype=np.uint32)
            for gi in range(c.n_head_kv):
                k_spec[gi*2] = gi; k_spec[gi*2+1] = self._kc_k_row_base[L][gi]
                v_spec[gi*2] = gi; v_spec[gi*2+1] = self._kc_v_row_base[L][gi]
            hk, _ = g.upload_dev(k_spec); hv, _ = g.upload_dev(v_spec)
            self._kc_k_copy_h.append(hk); self._kc_v_copy_h.append(hv)

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
        self._sample_repeat_ids_h, _ = g.upload_dev(self._sample_repeat_ids_arr)
        self._sample_fused_base_h, _ = g.upload_dev(
            np.array([self._ws_slots['sample_fused_idx'][2]], dtype=np.uint32))
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

    def _can_gpu_topk_sample(self, cfg) -> bool:
        """Fast path: GPU shortlist + GPU top-p normalization over the best k candidates."""
        if self._lm_wt_name is None:
            return False
        if cfg.top_k <= 1 or cfg.top_k > self.SAMPLE_TOPK_MAX:
            return False
        return True

    def _can_gpu_fused_topk_sample(self, cfg) -> bool:
        if not self._gpu_fused_topk:
            return False
        if not self._can_gpu_topk_sample(cfg):
            return False
        if not getattr(self.gpu, '_has_map_matvec_topk_t_xq8_dev', False):
            return False
        if not self._lm_wt_name:
            return False
        return (self._lm_wt_name in self._q8_tensors) and (not self._trans[self._lm_wt_name])

    def _can_gpu_approx_rerank_sample(self, cfg) -> bool:
        if not self._gpu_approx_rerank:
            return False
        if self._lm_wt_name is None:
            return False
        if cfg.top_k <= 1 or cfg.top_k > self.SAMPLE_APPROX_SHORTLIST_MAX:
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

    def _sample_gpu_argmax(self) -> int:
        """Sample next token by reducing the GPU logits buffer to one scalar index."""
        g = self.gpu; ws_id = self.ws_map_id
        lo_h, _ = self._wsh('logits')
        tok_h, _ = self._wsh('sample_token')
        g.map_argmax_dev(ws_id, lo_h, tok_h, self.cfg.n_vocab)
        tok = g.gather(ws_id, self._sample_locs).view(np.float32)
        return int(tok[0])

    def _sample_gpu_topk(self, cfg, prev) -> int:
        """Sample from a GPU-computed top-k shortlist with GPU top-p probabilities."""
        g = self.gpu; ws_id = self.ws_map_id
        lo_h, _ = self._wsh('logits')
        idx_h, _ = self._wsh('sample_topk_idx')
        val_h, _ = self._wsh('sample_topk_val')
        prob_h, _ = self._wsh('sample_topk_prob')
        top_k = min(int(cfg.top_k), self.SAMPLE_TOPK_MAX, self.cfg.n_vocab)

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

        g.map_topk_dev(ws_id, lo_h, idx_h, val_h, self.cfg.n_vocab, top_k)

        idx = g.gather(ws_id, self._sample_topk_idx_locs[:top_k]).view(np.float32).astype(np.int32)
        vals = g.gather(ws_id, self._sample_topk_val_locs[:top_k]).view(np.float32)
        valid = np.isfinite(vals)
        if not np.any(valid):
            return 0

        idx = idx[valid]
        vals = vals[valid]
        if cfg.temperature <= 0.0:
            return int(idx[0])

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
        n = self._sample_rerank_n
        if n <= 0:
            return 0
        idx = self.gpu.gather(
            self.ws_map_id, self._sample_exact_idx_locs[:n]).view(np.float32).astype(np.int32)
        vals = self.gpu.gather(
            self.ws_map_id, self._sample_exact_val_locs[:n]).view(np.float32)
        self._sample_rerank_n = 0

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
        pos = self.gpu.gather(self.ws_map_id, self._sample_topk_idx_locs[:top_k]).view(np.float32).astype(np.int32)
        pos = np.clip(pos, 0, max(self._sample_fused_partial_n - 1, 0))
        tok_locs = self._sample_fused_idx_locs[pos]
        idx = self.gpu.gather(self.ws_map_id, tok_locs).view(np.float32).astype(np.int32)
        if cfg.temperature <= 0.0:
            self._sample_fused_topk_n = 0
            return int(idx[0]) if idx.size else 0

        p = self.gpu.gather(self.ws_map_id, self._sample_topk_prob_locs[:top_k]).view(np.float32)
        keep = p > 0.0
        self._sample_fused_topk_n = 0
        if not np.any(keep):
            return int(idx[0]) if idx.size else 0
        idx = idx[keep]
        p = p[keep]
        p /= p.sum()
        return int(idx[np.random.choice(len(p), p=p)])

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
    # Forward pass
    # ============================================================
    def _forward(self, tok, pos, return_logits=True, sample_mode=None, sample_cfg=None, sample_prev=None):
        c = self.cfg; g = self.gpu; ws_id = self.ws_map_id
        q_dim = c.n_head * c.head_dim; k_dim = c.n_head_kv * c.head_dim_kv
        seq_len = pos + 1

        # ---- Embedding lookup / prep ----
        t0 = time.perf_counter()
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
        self.timing['embed'] += time.perf_counter() - t0

        # ---- Start batch: all GPU layer ops accumulate in one command buffer ----
        g.batch_begin()
        if use_gpu_embed:
            self._enqueue_gpu_embed_token(hid_h)

        for L in range(c.n_layer):
            p = f'blk.{L}'; gl = c.is_global(L)

            # ---- Pre-attention RMSNorm ----
            t0 = time.perf_counter()
            normed_h, _ = self._wsh('normed')
            self._gpu_rmsnorm(hid_h, f'{p}.attn_norm.weight', normed_h,
                              1, c.n_embd, c.norm_eps)
            self.timing['norm'] += time.perf_counter() - t0

            # ---- QKV projections: Q → q slot, K,V → k,v slots ----
            t0 = time.perf_counter()
            q_h, _ = self._wsh('q')
            k_h, _ = self._wsh('k')
            v_h, _ = self._wsh('v')
            self._gpu_proj(normed_h, f'{p}.attn_q.weight', q_h, 1, c.n_embd, q_dim)
            self._gpu_proj(normed_h, f'{p}.attn_k.weight', k_h, 1, c.n_embd, k_dim)
            self._gpu_proj(normed_h, f'{p}.attn_v.weight', v_h, 1, c.n_embd, k_dim)
            self.timing['qkv'] += time.perf_counter() - t0

            # ---- QK Norm (Gemma 3 only): out-of-place (in-place silently fails in batch) ----
            t0 = time.perf_counter()
            qnn = f'{p}.attn_q_norm.weight'
            if qnn in self._wh:
                self._gpu_rmsnorm(self._ws_rms_q_h, qnn, self._ws_rms_q_dst_h,
                                  c.n_head, c.head_dim, c.norm_eps)
                knn = f'{p}.attn_k_norm.weight'
                self._gpu_rmsnorm(self._ws_rms_k_h, knn, self._ws_rms_k_dst_h,
                                  c.n_head_kv, c.head_dim_kv, c.norm_eps)
            self.timing['qk_norm'] += time.perf_counter() - t0

            # ---- RoPE: Q and K in workspace (out-of-place: same data positions,
            #      different src resource to avoid in-place Vulkan driver issue) ----
            t0 = time.perf_counter()
            freq = c.rope_base_global if gl else c.rope_base_local
            g.map_rope_dev(ws_id, self._q_rope_src_h, q_h, 1, c.n_head, c.head_dim, pos, freq)
            g.map_rope_dev(ws_id, self._k_rope_src_h, k_h, 1, c.n_head_kv, c.head_dim_kv, pos, freq)
            self.timing['rope'] += time.perf_counter() - t0

            # ---- Copy K,V from workspace to KV cache (per-head-contiguous) ----
            g.map_row_copy_offset_dev(ws_id, self._kc_k_copy_h[L], self._k_src_base_h,
                                      pos, c.n_head_kv, c.head_dim_kv)
            g.map_row_copy_offset_dev(ws_id, self._kc_v_copy_h[L], self._v_src_base_h,
                                      pos, c.n_head_kv, c.head_dim_kv)

            # ---- Attention ----
            t0 = time.perf_counter()
            ao_h, _ = self._wsh('attn_out')
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
                g.map_attn_softmax_abs_dev(
                    ws_id,
                    self._scores_row_h,
                    self._scores_row_h,
                    seq_len,
                    c.n_head,
                    float(1.0 / np.sqrt(c.head_dim)),
                    float(c.attn_softcap),
                )

                # attn_out[h, d] = sum_t attn[h,t] * V_cache_{h//gs}[t, d]
                # locs_a[h] = fixed score row base for head h
                # locs_b[h] = kc_v_base[L] + (h//gs)*kv_step
                # locs_c[h] = pos_ao + h*head_dim_kv  (fixed)
                g.map_matmul_dev(ws_id,
                                 self._scores_row_h,
                                 self._kc_v_locs_h[L],
                                 self._ao_locs_h,
                                 1, seq_len, c.head_dim_kv, c.n_head)
            self.timing['attn'] += time.perf_counter() - t0

            # ---- Output projection ----
            t0 = time.perf_counter()
            op_h, _ = self._wsh('o_proj')
            o_in = c.n_head * c.head_dim_kv
            self._gpu_proj(ao_h, f'{p}.attn_output.weight', op_h, 1, o_in, c.n_embd)

            pan = f'{p}.post_attention_norm.weight'
            if pan in self._wh:
                # Out-of-place: normed_h slot is free after QKV projections+RoPE.
                normed_h, _ = self._wsh('normed')
                self._gpu_rmsnorm(op_h, pan, normed_h, 1, c.n_embd, c.norm_eps)
                g.map_op2_dev(ws_id, self.A.OP_ADD, hid_h, normed_h, hid_h, c.n_embd)
            else:
                g.map_op2_dev(ws_id, self.A.OP_ADD, hid_h, op_h, hid_h, c.n_embd)
            self.timing['attn_out'] += time.perf_counter() - t0

            # ---- FFN ----
            t0 = time.perf_counter()
            n2_h, _ = self._wsh('normed2')
            self._gpu_rmsnorm(hid_h, f'{p}.ffn_norm.weight', n2_h, 1, c.n_embd, c.norm_eps)

            gate_h, _ = self._wsh('gate')
            up_h,   _ = self._wsh('up')
            act_h,  _ = self._wsh('act')
            ffn_h,  _ = self._wsh('ffn_out')

            self._gpu_proj(n2_h, f'{p}.ffn_gate.weight', gate_h, 1, c.n_embd, c.n_ff)
            self._gpu_proj(n2_h, f'{p}.ffn_up.weight',   up_h,   1, c.n_embd, c.n_ff)
            ffn_act_op = self.A.OP_GELU if c.ffn_act == 'gelu' else self.A.OP_SWISH
            g.map_op1_dev(ws_id, ffn_act_op, gate_h, gate_h, c.n_ff)
            g.map_op2_dev(ws_id, self.A.OP_MUL,  gate_h, up_h, act_h, c.n_ff)
            self._gpu_proj(act_h, f'{p}.ffn_down.weight', ffn_h, 1, c.n_ff, c.n_embd)

            pfn = f'{p}.post_ffw_norm.weight'
            if pfn in self._wh:
                # Out-of-place: n2_h ('normed2') slot is free after FFW projections.
                self._gpu_rmsnorm(ffn_h, pfn, n2_h, 1, c.n_embd, c.norm_eps)
                g.map_op2_dev(ws_id, self.A.OP_ADD, hid_h, n2_h, hid_h, c.n_embd)
            else:
                g.map_op2_dev(ws_id, self.A.OP_ADD, hid_h, ffn_h, hid_h, c.n_embd)
            self.timing['ffn'] += time.perf_counter() - t0

        # ---- Final norm (still in batch) ----
        # Out-of-place into 'normed' slot (free after all layers).
        t0 = time.perf_counter()
        normed_final_h, _ = self._wsh('normed')
        self._gpu_rmsnorm(hid_h, 'output_norm.weight', normed_final_h, 1, c.n_embd, c.norm_eps)
        self.timing['norm'] += time.perf_counter() - t0

        # ---- LM head: stays inside batch so we only pay one fence wait ----
        t0 = time.perf_counter()
        lo_h, _ = self._wsh('logits')
        approx_partial_k = 0
        approx_penalty_n = 0
        approx_partial_val_h = None
        if self._lm_wt_name:
            if sample_mode == 'gpu_approx_rerank':
                approx_partial_k = min(self.SAMPLE_TOPK_MAX, self.cfg.n_vocab)
                if sample_cfg.repeat_penalty != 1.0 and sample_prev:
                    approx_penalty_n = self._prepare_repeat_ids(sample_prev)
                partial_idx_h, _ = self._wsh('sample_fused_idx')
                partial_val_h, _ = self._wsh('sample_fused_val')
                approx_partial_val_h = partial_val_h
                self._sample_fused_partial_n = self._sample_fused_groups * approx_partial_k
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
                    self._gpu_fused_rows_per_group,
                )
            elif sample_mode == 'gpu_fused_topk':
                top_k = min(int(sample_cfg.top_k), self.SAMPLE_TOPK_MAX, self.cfg.n_vocab)
                n_penalty = 0
                if sample_cfg.repeat_penalty != 1.0 and sample_prev:
                    n_penalty = self._prepare_repeat_ids(sample_prev)
                partial_idx_h, _ = self._wsh('sample_fused_idx')
                partial_val_h, _ = self._wsh('sample_fused_val')
                self._sample_fused_partial_n = self._sample_fused_groups * top_k
                wt_mid = self.tensor_map_id[self._lm_wt_name]
                if getattr(g, '_has_map_matvec_topk_t_xq8_ex_dev', False):
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
                        self._gpu_fused_rows_per_group,
                    )
                else:
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
                idx_h, _ = self._wsh('sample_topk_idx')
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
            else:
                # GPU path: output.weight is in the Q4/Q8 map → one GPU matmul
                self._gpu_proj(normed_final_h, self._lm_wt_name, lo_h, 1, c.n_embd, c.n_vocab)
                # Final logit soft-cap: tanh(x / cap) * cap (Gemma 3: cap=30.0)
                if c.final_softcap > 0:
                    fcap_h, _ = self._wsh('final_softcap')
                    g.map_broadcast_dev(ws_id, self.A.OP_DIV, lo_h, fcap_h, lo_h, c.n_vocab)
                    g.map_op1_dev(ws_id, self.A.OP_TANH, lo_h, lo_h, c.n_vocab)
                    g.map_broadcast_dev(ws_id, self.A.OP_MUL, lo_h, fcap_h, lo_h, c.n_vocab)
        else:
            # CPU fallback: weight-tied (token_embd.weight) — gather hidden,
            # multiply on CPU, scatter result.  One gather + one scatter per token.
            pass  # handled after batch_end below

        # ---- End batch: one fence wait for ALL layer + LM-head GPU ops ----
        g.batch_end()

        if self._lm_wt_name and sample_mode == 'gpu_approx_rerank':
            shortlist_n = 0
            partial_n = self._sample_fused_partial_n
            if partial_n > 0:
                if self._can_gpu_merge_approx_shortlist() and approx_partial_val_h is not None:
                    shortlist_n = min(self.SAMPLE_TOPK_MAX, self.SAMPLE_APPROX_SHORTLIST_MAX, partial_n)
                    if shortlist_n > 0:
                        short_sel_h, _ = self._wsh('sample_short_sel')
                        val_h, _ = self._wsh('sample_exact_val')
                        g.batch_begin()
                        try:
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
                        finally:
                            g.batch_end()
                        self._sample_rerank_n = shortlist_n
                else:
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
                        # Keep the rerank stage inside one GPU batch after shortlist selection
                        # to avoid per-chunk submit/fence overhead on slower drivers.
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
            self.timing['lm_head'] += time.perf_counter() - t0
            return None

        logits = g.gather(ws_id, self._lo_locs).view(np.float32)
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

    def _summarize_decode_trace(self, trace_steps):
        if not trace_steps:
            return {}
        denom = float(len(trace_steps))
        stage_keys = ('embed', 'norm', 'qkv', 'qk_norm', 'rope', 'attn', 'attn_out', 'ffn', 'lm_head')
        summary = {
            'step_ms_avg': sum(t['step_ms'] for t in trace_steps) / denom,
            'sample_ms_avg': sum(t['sample_ms'] for t in trace_steps) / denom,
            'forward_ms_avg': sum(t['forward_ms'] for t in trace_steps) / denom,
        }
        for key in stage_keys:
            summary[f'{key}_ms_avg'] = sum(t['timing_ms'].get(key, 0.0) for t in trace_steps) / denom
        return summary

    # ============================================================
    # Generate
    # ============================================================
    def generate(self, token_ids, config=None, stream=True, **kw):
        if config is None: config = GenerationConfig()
        if config.seed is not None: np.random.seed(config.seed)
        progress_interval_s = float(kw.get('progress_interval_s', 0.25))
        trace_decode = bool(kw.get('trace_decode', False))
        gpu_argmax = self._can_gpu_argmax_sample(config)
        gpu_approx_rerank = (not gpu_argmax) and self._can_gpu_approx_rerank_sample(config)
        gpu_fused_topk = ((not gpu_argmax) and (not gpu_approx_rerank) and
                          self._can_gpu_fused_topk_sample(config))
        gpu_topk = ((not gpu_argmax) and (not gpu_approx_rerank) and
                    (not gpu_fused_topk) and self._can_gpu_topk_sample(config))
        sampling_mode = (
            'gpu_argmax' if gpu_argmax else
            ('gpu_approx_rerank' if gpu_approx_rerank else
             ('gpu_fused_topk' if gpu_fused_topk else
              ('gpu_topk' if gpu_topk else 'cpu')))
        )

        total_needed = len(token_ids) + config.max_tokens
        if total_needed > self._kv_cap:
            raise RuntimeError(
                f"KV cache too small: need {total_needed} tokens but "
                f"kv_cap={self._kv_cap}. Reinitialize with kv_cap>={total_needed}.")

        self.reset()
        all_tok = list(token_ids); out_tok = []

        tp = time.perf_counter()
        logits = None
        last_prompt_idx = len(token_ids) - 1
        for i, t in enumerate(token_ids):
            need_logits = (i == last_prompt_idx) and (sampling_mode == 'cpu')
            sample_mode = sampling_mode if (
                i == last_prompt_idx and
                sampling_mode in ('gpu_fused_topk', 'gpu_approx_rerank')) else None
            repeat_prev = self._repeat_history(token_ids, out_tok, config) if sample_mode else None
            logits = self._forward(t, i, return_logits=need_logits,
                                   sample_mode=sample_mode, sample_cfg=config,
                                   sample_prev=repeat_prev)
        t_pre = time.perf_counter() - tp

        td = time.perf_counter(); ttimes = []
        last_progress = td
        trace_steps = []
        for step in range(config.max_tokens):
            t0 = time.perf_counter()
            repeat_prev = self._repeat_history(token_ids, out_tok, config)
            t_sample0 = time.perf_counter()
            if sampling_mode == 'gpu_approx_rerank':
                nt = self._sample_gpu_approx_rerank(config)
            elif sampling_mode == 'gpu_fused_topk':
                nt = self._consume_gpu_fused_topk(config)
            elif gpu_argmax:
                nt = self._sample_gpu_argmax()
            elif gpu_topk:
                nt = self._sample_gpu_topk(config, repeat_prev)
            else:
                nt = self._sample(logits, config, repeat_prev)
            sample_dt = time.perf_counter() - t_sample0
            if nt in config.eos_token_ids: break
            all_tok.append(nt); out_tok.append(nt)
            sample_mode = sampling_mode if sampling_mode in ('gpu_fused_topk', 'gpu_approx_rerank') else None
            repeat_prev = self._repeat_history(token_ids, out_tok, config) if sample_mode else None
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
        return out_tok, {
            'n_prompt': np_, 'n_gen': ng,
            'prefill_s': t_pre, 'decode_s': t_dec,
            'total_s': time.perf_counter() - tp,
            'prefill_tps': np_/t_pre if t_pre > 0 else 0,
            'decode_tps': ng/t_dec if t_dec > 0 else 0,
            'avg_ms': np.mean(ttimes)*1000 if ttimes else 0,
            'sampling_mode': sampling_mode,
            'timing': dict(self.timing),
            'trace_summary': self._summarize_decode_trace(trace_steps) if trace_decode else {},
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
        self._reset_timing()

    def _reset_timing(self):
        self.timing = {k: 0.0 for k in [
            'embed', 'norm', 'qkv', 'qk_norm', 'rope', 'attn', 'attn_out',
            'ffn', 'lm_head']}
