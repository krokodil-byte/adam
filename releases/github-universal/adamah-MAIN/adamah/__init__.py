"""
ADAMAH - High-Performance Cross-Platform GPU Computing

Pure Vulkan compute library. Runs on any GPU: NVIDIA, AMD, Intel, ARM.
Full compute in float32, bfloat16, int8, int6, or int4 — dtype is transparent.

Usage:
    import adamah

    gpu = adamah.init()
    gpu.set_dtype(adamah.DTYPE_Q4)      # everything runs in q4 from here

    gpu.map_create(0, 768, 50000)        # allocates q4 packed in VRAM
    gpu.scatter(0, locs, data)           # data stored as q4
    gpu.map_op1_dev(0, adamah.OP_GELU, src, dst, n)  # q4 compute
    result = gpu.gather(0, locs)         # q4 packed out

    # Map-to-map ops — no CPU round-trip, all on GPU in native dtype:
    gpu.map_op2_dev(0, adamah.OP_ADD, a, b, dst, n)
    gpu.map_matmul_dev(0, a, b, c, M, K, N, 1)
    gpu.map_softmax_dev(0, src, dst, rows, cols)

CC BY-NC 4.0 - Samuele Scuglia - 2026
"""

import ctypes
import numpy as np
import os
from contextlib import contextmanager
from typing import Optional, Tuple

__version__ = "5.2.0"

# Import UUCIS wrapper for benchmark compatibility
try:
    from .uucis import UUCISView
except ImportError:
    UUCISView = None

# Operation codes - Unary operations (map_op1)
OP_NEG = 0
OP_ABS = 1
OP_SQRT = 2
OP_EXP = 3
OP_LOG = 4
OP_TANH = 5
OP_RELU = 6
OP_GELU = 7
OP_SIN = 8
OP_COS = 9
OP_TAN = 10
OP_ASIN = 11
OP_ACOS = 12
OP_ATAN = 13
OP_SINH = 14
OP_COSH = 15
OP_SIGMOID = 16
OP_SWISH = 17
OP_MISH = 18
OP_SELU = 19
OP_ELU = 20
OP_LEAKY_RELU = 21
OP_CEIL = 22
OP_FLOOR = 23
OP_ROUND = 24
OP_SIGN = 25
OP_RECIPROCAL = 26
OP_SQUARE = 27
OP_CUBE = 28
OP_SOFTPLUS = 29
OP_HARDSIGMOID = 30
OP_HARDSWISH = 31
OP_EXPM1 = 32
OP_LOG1P = 33

# Binary operations (map_op2)
OP_ADD = 0
OP_SUB = 1
OP_MUL = 2
OP_DIV = 3
OP_POW = 4
OP_MIN = 5
OP_MAX = 6
OP_MOD = 7
OP_EQ = 8
OP_NE = 9
OP_LT = 10
OP_LE = 11
OP_GT = 12
OP_GE = 13
OP_AND = 14
OP_OR = 15
OP_XOR = 16
OP_ATAN2 = 17
OP_STEP = 18
OP_SMOOTHSTEP = 19

# ============================================
# Helpers
# ============================================

def _unpack_ticket_handle(v: int) -> Tuple[int, int]:
    handle = v & 0xFFFFFFFF
    ticket = (v >> 32) & 0xFFFFFFFFFFFFFFFF
    return handle, ticket

# ============================================
# ArrayHandle - Wrapper for GPU arrays
# ============================================

class ArrayHandle:
    """GPU array handle (for lazy execution API)."""

    def __init__(self, gpu, handle_id: int, shape, dtype='float32'):
        self.gpu = gpu
        self.id = handle_id
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.dtype = np.dtype(dtype)
        self._freed = False

    def numpy(self) -> np.ndarray:
        """Download to numpy (triggers sync)."""
        if self._freed:
            raise RuntimeError("Cannot download freed array")
        self.gpu.sync()
        # TODO: implement array download
        raise NotImplementedError("ArrayHandle.numpy() not yet implemented")

    def free(self):
        """Explicitly free GPU memory."""
        if not self._freed:
            # TODO: implement array free
            self._freed = True

    def __del__(self):
        if not self._freed:
            try:
                self.free()
            except:
                pass

# ============================================
# Main Adamah Class
# ============================================

class Adamah:
    """ADAMAH GPU compute library."""

    _MAP_NUMPY_DTYPES = {
        0: np.float32,  # DTYPE_F32
        1: np.uint16,   # DTYPE_BF16 (raw bf16 words)
        2: np.uint8,    # DTYPE_Q8
        3: np.uint8,    # DTYPE_Q4 packed bytes
        4: np.uint8,    # DTYPE_Q6 packed bytes
    }

    def __init__(self, lib_path: Optional[str] = None, cache_mb: Optional[int] = None, cold_cache_mb: Optional[int] = None):
        """Initialize ADAMAH library."""
        if lib_path is None:
            env_lib = os.environ.get("ADAMAH_LIB_PATH")
            if env_lib:
                lib_path = env_lib
            else:
                # Auto-detect library path
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if os.name == 'nt':
                    candidates = ['adamah_opt.dll', 'adamah_new.dll', 'adamah.dll']
                else:
                    candidates = ['adamah.so']
                for lib_name in candidates:
                    cand = os.path.join(current_dir, lib_name)
                    if os.path.exists(cand):
                        lib_path = cand
                        break
                if lib_path is None:
                    lib_name = candidates[0]
                    raise FileNotFoundError(
                        f"{lib_name} not found in {current_dir}\n"
                        f"Please compile the ADAMAH native library first"
                    )

        self._lib = ctypes.CDLL(lib_path)
        self._setup_ctypes()

        # Initialize Vulkan / cache pools
        if cache_mb is not None or cold_cache_mb is not None:
            init_ex = getattr(self._lib, "adamah_init_ex", None)
            if init_ex is None:
                ret = self._lib.adamah_init()
            else:
                hot_mb = int(cache_mb if cache_mb is not None else cold_cache_mb)
                cold_mb = int(cold_cache_mb if cold_cache_mb is not None else hot_mb)
                ret = init_ex(ctypes.c_uint64(hot_mb * 1024 * 1024), ctypes.c_uint64(cold_mb * 1024 * 1024))
        else:
            ret = self._lib.adamah_init()
        if ret != 0:
            raise RuntimeError(f"adamah_init failed with code {ret}")

        # Metrics
        self._metrics = {
            'gather_calls': 0,
            'scatter_calls': 0,
            'op_calls': 0,
            'total_bytes_cpu_to_gpu': 0,
            'total_bytes_gpu_to_cpu': 0,
        }

        # Map metadata (for UUCIS compatibility)
        self._maps = {}

    def _setup_ctypes(self):
        """Configure ctypes function signatures."""
        # Init/shutdown
        self._lib.adamah_init.argtypes = []
        self._lib.adamah_init.restype = ctypes.c_int
        init_ex = getattr(self._lib, "adamah_init_ex", None)
        if init_ex is not None:
            init_ex.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
            init_ex.restype = ctypes.c_int

        self._lib.adamah_shutdown.argtypes = []
        self._lib.adamah_shutdown.restype = None

        self._lib.adamah_sync.argtypes = []
        self._lib.adamah_sync.restype = None

        self._lib.adamah_fusion_enable.argtypes = [ctypes.c_int]
        self._lib.adamah_fusion_enable.restype = None

        self._lib.adamah_fusion_is_enabled.argtypes = []
        self._lib.adamah_fusion_is_enabled.restype = ctypes.c_int

        self._lib.adamah_fusion_flush.argtypes = []
        self._lib.adamah_fusion_flush.restype = ctypes.c_int

        # Map operations
        self._lib.map_init.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self._lib.map_init.restype = ctypes.c_int

        self._lib.map_destroy.argtypes = [ctypes.c_uint32]
        self._lib.map_destroy.restype = ctypes.c_int

        self._lib.map_size.argtypes = [ctypes.c_uint32]
        self._lib.map_size.restype = ctypes.c_uint64

        # Scatter/gather
        self._lib.map_scatter.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
            ctypes.c_uint32
        ]
        self._lib.map_scatter.restype = ctypes.c_uint64

        self._lib.map_gather.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
            ctypes.c_uint32
        ]
        self._lib.map_gather.restype = ctypes.c_uint64

        # Operations
        self._lib.map_op1.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32
        ]
        self._lib.map_op1.restype = ctypes.c_int

        self._lib.map_op2.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32
        ]
        self._lib.map_op2.restype = ctypes.c_int

        self._lib.map_matmul.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32
        ]
        self._lib.map_matmul.restype = ctypes.c_int

        self._lib.map_softmax.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32, ctypes.c_uint32
        ]
        self._lib.map_softmax.restype = ctypes.c_int

        self._lib.map_layernorm.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float
        ]
        self._lib.map_layernorm.restype = ctypes.c_int

        self._lib.map_broadcast.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32
        ]
        self._lib.map_broadcast.restype = ctypes.c_int

        self._lib.map_reduce.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32
        ]
        self._lib.map_reduce.restype = ctypes.c_int

        # Device-only async helpers
        self._lib.map_upload_dev.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
        self._lib.map_upload_dev.restype = ctypes.c_uint64
        self._lib.map_download_dev.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
        self._lib.map_download_dev.restype = ctypes.c_int
        self._lib.map_gather_dev.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self._lib.map_gather_dev.restype = ctypes.c_uint64
        self._lib.map_scatter_dev.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self._lib.map_scatter_dev.restype = ctypes.c_uint64

        # Device-locs ops
        self._lib.map_op1_dev.argtypes = [ctypes.c_uint32, ctypes.c_uint32,
                                          ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self._lib.map_op1_dev.restype = ctypes.c_int
        self._lib.map_op2_dev.argtypes = [ctypes.c_uint32, ctypes.c_uint32,
                                          ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self._lib.map_op2_dev.restype = ctypes.c_int
        self._lib.map_matmul_dev.argtypes = [ctypes.c_uint32,
                                             ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                             ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self._lib.map_matmul_dev.restype = ctypes.c_int
        self._lib.map_softmax_dev.argtypes = [ctypes.c_uint32,
                                              ctypes.c_uint32, ctypes.c_uint32,
                                              ctypes.c_uint32, ctypes.c_uint32]
        self._lib.map_softmax_dev.restype = ctypes.c_int
        self._lib.map_softmax_abs_dev.argtypes = [ctypes.c_uint32,
                                                  ctypes.c_uint32, ctypes.c_uint32,
                                                  ctypes.c_uint32, ctypes.c_uint32]
        self._lib.map_softmax_abs_dev.restype = ctypes.c_int
        self._lib.map_attn_softmax_abs_dev.argtypes = [ctypes.c_uint32,
                                                       ctypes.c_uint32, ctypes.c_uint32,
                                                       ctypes.c_uint32, ctypes.c_uint32,
                                                       ctypes.c_float, ctypes.c_float]
        self._lib.map_attn_softmax_abs_dev.restype = ctypes.c_int
        self._lib.map_layernorm_dev.argtypes = [ctypes.c_uint32,
                                                ctypes.c_uint32, ctypes.c_uint32,
                                                ctypes.c_uint32, ctypes.c_uint32,
                                                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float]
        self._lib.map_layernorm_dev.restype = ctypes.c_int
        self._lib.map_broadcast_dev.argtypes = [ctypes.c_uint32, ctypes.c_uint32,
                                                ctypes.c_uint32, ctypes.c_uint32,
                                                ctypes.c_uint32, ctypes.c_uint32]
        self._lib.map_broadcast_dev.restype = ctypes.c_int
        self._lib.map_reduce_dev.argtypes = [ctypes.c_uint32, ctypes.c_uint32,
                                             ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self._lib.map_reduce_dev.restype = ctypes.c_int
        try:
            self._lib.map_repeat_penalty_dev.argtypes = [ctypes.c_uint32,
                                                         ctypes.c_uint32, ctypes.c_uint32,
                                                         ctypes.c_uint32, ctypes.c_float]
            self._lib.map_repeat_penalty_dev.restype = ctypes.c_int
            self._has_map_repeat_penalty_dev = True
        except AttributeError:
            self._has_map_repeat_penalty_dev = False
        try:
            self._lib.map_matvec_topk_t_xq4_dev.argtypes = [
                ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_float,
            ]
            self._lib.map_matvec_topk_t_xq4_dev.restype = ctypes.c_int
            self._has_map_matvec_topk_t_xq4_dev = True
        except AttributeError:
            self._has_map_matvec_topk_t_xq4_dev = False
        try:
            self._lib.map_matvec_topk_t_xq4_ex_dev.argtypes = [
                ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_float, ctypes.c_uint32,
            ]
            self._lib.map_matvec_topk_t_xq4_ex_dev.restype = ctypes.c_int
            self._has_map_matvec_topk_t_xq4_ex_dev = True
        except AttributeError:
            self._has_map_matvec_topk_t_xq4_ex_dev = False
        try:
            self._lib.map_matvec_topk_t_xq8_dev.argtypes = [
                ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_float,
            ]
            self._lib.map_matvec_topk_t_xq8_dev.restype = ctypes.c_int
            self._has_map_matvec_topk_t_xq8_dev = True
        except AttributeError:
            self._has_map_matvec_topk_t_xq8_dev = False
        try:
            self._lib.map_matvec_topk_t_xq8_ex_dev.argtypes = [
                ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_float, ctypes.c_uint32,
            ]
            self._lib.map_matvec_topk_t_xq8_ex_dev.restype = ctypes.c_int
            self._has_map_matvec_topk_t_xq8_ex_dev = True
        except AttributeError:
            self._has_map_matvec_topk_t_xq8_ex_dev = False
        try:
            self._lib.map_matvec_rerank_t_xq8_dev.argtypes = [
                ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_float,
            ]
            self._lib.map_matvec_rerank_t_xq8_dev.restype = ctypes.c_int
            self._has_map_matvec_rerank_t_xq8_dev = True
        except AttributeError:
            self._has_map_matvec_rerank_t_xq8_dev = False
        self._lib.map_argmax_dev.argtypes = [ctypes.c_uint32,
                                             ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self._lib.map_argmax_dev.restype = ctypes.c_int
        self._lib.map_topk_dev.argtypes = [ctypes.c_uint32,
                                           ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                           ctypes.c_uint32, ctypes.c_uint32]
        self._lib.map_topk_dev.restype = ctypes.c_int
        self._lib.map_topp_dev.argtypes = [ctypes.c_uint32,
                                           ctypes.c_uint32, ctypes.c_uint32,
                                           ctypes.c_uint32, ctypes.c_float, ctypes.c_float]
        self._lib.map_topp_dev.restype = ctypes.c_int

        # Transformer-specific ops
        self._lib.map_rmsnorm_dev.argtypes = [ctypes.c_uint32,
                                              ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                              ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float]
        self._lib.map_rmsnorm_dev.restype = ctypes.c_int
        self._lib.map_rope_dev.argtypes = [ctypes.c_uint32,
                                           ctypes.c_uint32, ctypes.c_uint32,
                                           ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                           ctypes.c_uint32, ctypes.c_float]
        self._lib.map_rope_dev.restype = ctypes.c_int
        self._lib.map_matmul_t_dev.argtypes = [ctypes.c_uint32,
                                               ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                               ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self._lib.map_matmul_t_dev.restype = ctypes.c_int
        self._lib.map_row_copy_dev.argtypes = [ctypes.c_uint32,
                                               ctypes.c_uint32, ctypes.c_uint32,
                                               ctypes.c_uint32, ctypes.c_uint32]
        self._lib.map_row_copy_dev.restype = ctypes.c_int
        self._lib.map_row_copy_offset_dev.argtypes = [ctypes.c_uint32,
                                                      ctypes.c_uint32, ctypes.c_uint32,
                                                      ctypes.c_uint32, ctypes.c_uint32,
                                                      ctypes.c_uint32]
        self._lib.map_row_copy_offset_dev.restype = ctypes.c_int
        self._lib.map_fma_dev.argtypes = [ctypes.c_uint32,
                                          ctypes.c_uint32, ctypes.c_uint32,
                                          ctypes.c_uint32, ctypes.c_uint32,
                                          ctypes.c_uint32]
        self._lib.map_fma_dev.restype = ctypes.c_int

        # Cross-map ops (F32 activations × F32/Q4 weights on separate maps)
        _xmap_args = [ctypes.c_uint32, ctypes.c_uint32,
                      ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                      ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        for _fn in ('map_matmul_t_x_dev', 'map_matmul_x_dev',
                    'map_matmul_t_xq4_dev', 'map_matmul_xq4_dev',
                    'map_matmul_t_xq8_dev', 'map_matmul_xq8_dev'):
            getattr(self._lib, _fn).argtypes = _xmap_args
            getattr(self._lib, _fn).restype = ctypes.c_int
        self._lib.map_row_gather_xq8_dev.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_float,
        ]
        self._lib.map_row_gather_xq8_dev.restype = ctypes.c_int
        self._lib.map_rmsnorm_x_dev.argtypes = [ctypes.c_uint32, ctypes.c_uint32,
                                                 ctypes.c_uint32, ctypes.c_uint32,
                                                 ctypes.c_uint32, ctypes.c_uint32,
                                                 ctypes.c_uint32, ctypes.c_float]
        self._lib.map_rmsnorm_x_dev.restype = ctypes.c_int

        # Sync for async tickets
        self._lib.adamah_synchronize.argtypes = [ctypes.c_uint64]
        self._lib.adamah_synchronize.restype = None
        self._lib.adamah_synchronize_all.argtypes = []
        self._lib.adamah_synchronize_all.restype = None

        # Batching
        self._lib.batch_begin.argtypes = []
        self._lib.batch_begin.restype = None

        self._lib.batch_end.argtypes = []
        self._lib.batch_end.restype = None

        # Dtype system
        self._try_setup_dtype()

    def _try_setup_dtype(self):
        """Setup dtype-related ctypes if available."""
        try:
            fn = self._lib.adamah_set_dtype
            fn.argtypes = [ctypes.c_uint32]
            fn.restype = ctypes.c_int

            fn2 = self._lib.adamah_get_dtype
            fn2.argtypes = []
            fn2.restype = ctypes.c_uint32

            fn3 = self._lib.map_init_dtype
            fn3.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
            fn3.restype = ctypes.c_int

            fn4 = self._lib.map_set_qparams
            fn4.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_float),
                            ctypes.POINTER(ctypes.c_float), ctypes.c_uint32]
            fn4.restype = ctypes.c_int

            fn5 = self._lib.map_get_dtype
            fn5.argtypes = [ctypes.c_uint32]
            fn5.restype = ctypes.c_uint32

            self._has_dtype = True
        except (AttributeError, OSError):
            self._has_dtype = False

    # ============================================
    # Dtype API
    # ============================================

    # Dtype constants
    DTYPE_F32 = 0
    DTYPE_BF16 = 1
    DTYPE_Q8 = 2
    DTYPE_Q4 = 3
    DTYPE_Q6 = 4

    def set_dtype(self, dtype: int, group_size: int = 128):
        """Set compute dtype. All subsequent map_create and ops use this dtype.
        
        Args:
            dtype: DTYPE_F32, DTYPE_BF16, DTYPE_Q8, DTYPE_Q4, or DTYPE_Q6
            group_size: Quantization group size (for q4/q6/q8, default 128)
        
        Everything is transparent after this call:
            gpu.set_dtype(adamah.DTYPE_Q4)
            gpu.map_create(0, 768, 50000)    # allocates q4 packed
            gpu.scatter(0, locs, data)        # data goes in q4
            gpu.map_op1_dev(0, OP_GELU, ...)  # computes in q4
            result = gpu.gather(0, locs)      # returns packed q4
        """
        if not self._has_dtype:
            raise RuntimeError("Dtype support not available - recompile adamah.so")
        ret = self._lib.adamah_set_dtype(ctypes.c_uint32(dtype))
        if ret != 0:
            names = {0:'f32', 1:'bf16', 2:'q8', 3:'q4', 4:'q6'}
            raise RuntimeError(f"Failed to set dtype {names.get(dtype, dtype)} (code {ret})")
        self._active_dtype = dtype
        self._group_size = group_size

    def get_dtype(self) -> int:
        """Get currently active dtype."""
        if not self._has_dtype:
            return 0
        return int(self._lib.adamah_get_dtype())

    def map_create_typed(self, map_id: int, dtype: int, pack_size: int, n_packs: int, group_size: int = 128):
        """Create a memory map with explicit dtype (advanced — prefer map_create after set_dtype).
        
        VRAM per 1M elements:
            f32: 4.00 MB | bf16: 2.00 MB | q8: 1.00 MB | q6: 0.80 MB | q4: 0.50 MB
        """
        if not self._has_dtype:
            raise RuntimeError("Dtype support not available - recompile adamah.so")
        ret = self._lib.map_init_dtype(
            ctypes.c_uint32(map_id), ctypes.c_uint32(dtype),
            ctypes.c_uint32(pack_size), ctypes.c_uint32(n_packs),
            ctypes.c_uint32(group_size)
        )
        if ret != 0:
            names = {0:'f32', 1:'bf16', 2:'q8', 3:'q4', 4:'q6'}
            raise RuntimeError(f"map_init_dtype({names.get(dtype, dtype)}) failed: {ret}")
        ws = {0: 4, 1: 2, 2: 1, 3: 1, 4: 1}[dtype]
        self._maps[map_id] = (ws, pack_size, n_packs, dtype)

    def set_qparams(self, map_id: int, scales: np.ndarray, zero_points: np.ndarray):
        """Set quantization parameters for a quantized map.
        
        For q4/q6/q8: sets per-group scale and zero_point.
        These live in separate GPU buffers bound to the shader automatically.
        """
        if not self._has_dtype:
            raise RuntimeError("Dtype support not available - recompile adamah.so")
        scales = np.ascontiguousarray(scales, dtype=np.float32)
        zero_points = np.ascontiguousarray(zero_points, dtype=np.float32)
        ret = self._lib.map_set_qparams(
            ctypes.c_uint32(map_id),
            scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            zero_points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint32(len(scales))
        )
        if ret != 0:
            raise RuntimeError(f"map_set_qparams failed with code {ret}")

    # ============================================
    # Map Operations — dtype-aware
    # ============================================

    def map_create(self, map_id: int, pack_size: int = 0, n_packs: int = 0,
                   word_size: int = 0, **kw):
        """Create memory map using the active dtype.
        
        After set_dtype(), just pass logical dimensions:
            gpu.map_create(0, pack_size=768, n_packs=50000)
        
        VRAM is allocated in the native format — q4 maps use ~0.5 bytes/element.
        
        For backward compat, word_size still works:
            gpu.map_create(0, word_size=4, pack_size=768, n_packs=50000)
        """
        dtype = getattr(self, '_active_dtype', 0)
        group_size = getattr(self, '_group_size', 128)
        
        # Backward compat: if word_size given and no dtype set, use legacy path
        if word_size > 0 and dtype == 0 and not hasattr(self, '_active_dtype'):
            ret = self._lib.map_init(
                ctypes.c_uint32(map_id), ctypes.c_uint32(word_size),
                ctypes.c_uint32(pack_size), ctypes.c_uint32(n_packs)
            )
            if ret != 0:
                raise RuntimeError(f"map_init failed with code {ret}")
            self._maps[map_id] = (word_size, pack_size, n_packs, 0)
            return

        # dtype-aware path
        if self._has_dtype:
            ret = self._lib.map_init_dtype(
                ctypes.c_uint32(map_id), ctypes.c_uint32(dtype),
                ctypes.c_uint32(pack_size), ctypes.c_uint32(n_packs),
                ctypes.c_uint32(group_size)
            )
            if ret != 0:
                names = {0:'f32', 1:'bf16', 2:'q8', 3:'q4', 4:'q6'}
                raise RuntimeError(f"map_create({names.get(dtype, '?')}) failed: {ret}")
        else:
            ws = word_size if word_size > 0 else 4
            ret = self._lib.map_init(
                ctypes.c_uint32(map_id), ctypes.c_uint32(ws),
                ctypes.c_uint32(pack_size), ctypes.c_uint32(n_packs)
            )
            if ret != 0:
                raise RuntimeError(f"map_init failed: {ret}")

        ws = {0: 4, 1: 2, 2: 1, 3: 1, 4: 1}.get(dtype, word_size if word_size > 0 else 4)
        self._maps[map_id] = (ws, pack_size, n_packs, dtype)

    # Alias for compatibility
    def map_init(self, map_id: int, word_size: int, pack_size: int, n_packs: int):
        """Alias for map_create (compatibility)."""
        return self.map_create(
            map_id,
            pack_size=pack_size,
            n_packs=n_packs,
            word_size=word_size,
        )

    def map_destroy(self, map_id: int):
        """Destroy memory map."""
        ret = self._lib.map_destroy(ctypes.c_uint32(map_id))
        if ret != 0:
            raise RuntimeError(f"map_destroy failed with code {ret}")

    def map_size(self, map_id: int) -> int:
        """Get map size in packs."""
        return int(self._lib.map_size(ctypes.c_uint32(map_id)))

    def scatter(self, map_id: int, locs: np.ndarray, data: np.ndarray) -> int:
        """Write data to map. Data is copied as-is — must match active dtype packing."""
        self._metrics['scatter_calls'] += 1
        self._metrics['total_bytes_cpu_to_gpu'] += data.nbytes
        locs = np.ascontiguousarray(locs, dtype=np.uint32)
        data = np.ascontiguousarray(data)
        ticket = self._lib.map_scatter(
            ctypes.c_uint32(map_id),
            locs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            data.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_uint32(len(locs))
        )
        return int(ticket)

    map_scatter = scatter

    def gather(self, map_id: int, locs: np.ndarray, n_packs: Optional[int] = None, n_locs: Optional[int] = None) -> np.ndarray:
        """Read data from map. Returns raw bytes matching active dtype packing."""
        self._metrics['gather_calls'] += 1
        locs = np.ascontiguousarray(locs, dtype=np.uint32)
        n = n_locs if n_locs is not None else (n_packs if n_packs is not None else len(locs))
        out = np.empty(n, dtype=np.uint32)
        self._lib.map_gather(
            ctypes.c_uint32(map_id),
            locs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_uint32(n)
        )
        self._metrics['total_bytes_gpu_to_cpu'] += out.nbytes
        return out

    # Alias for compatibility
    map_gather = gather

    # ============================================
    # Device-only async helpers
    # ============================================

    def upload_dev(self, data, handle: int = 0):
        """Upload CPU data into a device-local buffer. Returns (handle, ticket)."""
        arr = np.ascontiguousarray(data)
        ptr = arr.ctypes.data_as(ctypes.c_void_p)
        packed = self._lib.map_upload_dev(ctypes.c_uint32(handle), ptr, ctypes.c_uint32(arr.nbytes))
        return _unpack_ticket_handle(int(packed))

    def download_dev(self, handle: int, n_elems: int, dtype=np.float32):
        """Download device-local buffer into a numpy array."""
        out = np.empty(n_elems, dtype=dtype)
        ptr = out.ctypes.data_as(ctypes.c_void_p)
        ret = self._lib.map_download_dev(ctypes.c_uint32(handle), ptr, ctypes.c_uint32(out.nbytes))
        if ret != 0:
            raise RuntimeError(f"map_download_dev failed with code {ret}")
        return out

    def map_gather_dev(self, map_id: int, locs_handle: int, n_locs: int):
        """Device-only gather. Returns (dst_handle, ticket, n_elems, dtype)."""
        if map_id not in self._maps:
            raise ValueError(f"Map {map_id} not initialized")
        _, pack_size, _, _ = self._maps[map_id]
        packed = self._lib.map_gather_dev(ctypes.c_uint32(map_id), ctypes.c_uint32(locs_handle), ctypes.c_uint32(n_locs))
        dst_handle, ticket = _unpack_ticket_handle(int(packed))
        return dst_handle, ticket, n_locs * pack_size, np.float32

    def map_scatter_dev(self, map_id: int, locs_handle: int, n_locs: int, src_handle: int):
        """Device-only scatter. Returns ticket."""
        if map_id not in self._maps:
            raise ValueError(f"Map {map_id} not initialized")
        packed = self._lib.map_scatter_dev(ctypes.c_uint32(map_id), ctypes.c_uint32(locs_handle),
                                           ctypes.c_uint32(n_locs), ctypes.c_uint32(src_handle))
        return int(packed)

    def synchronize(self, ticket: int):
        """Wait for a specific async ticket."""
        self._lib.adamah_synchronize(ctypes.c_uint64(ticket))

    def synchronize_all(self):
        """Wait for all async submissions."""
        self._lib.adamah_synchronize_all()

    # ============================================
    # GPU Operations
    # ============================================

    def map_op1(self, map_id: int, op: int, locs_in: np.ndarray, locs_out: np.ndarray, n: Optional[int] = None):
        """Unary operation on map."""
        self._metrics['op_calls'] += 1

        locs_in = np.ascontiguousarray(locs_in, dtype=np.uint32)
        locs_out = np.ascontiguousarray(locs_out, dtype=np.uint32)

        # Auto-compute n if not provided (UUCIS compatibility)
        if n is None:
            n = len(locs_in)

        ret = self._lib.map_op1(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(op),
            locs_in.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(n)
        )

        if ret != 0:
            raise RuntimeError(f"map_op1 failed with code {ret}")

    def map_op1_dev(self, map_id: int, op: int, locs_in_handle: int, locs_out_handle: int, n: int):
        """Unary op using device-local locs buffers."""
        self._metrics['op_calls'] += 1
        ret = self._lib.map_op1_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(op),
            ctypes.c_uint32(locs_in_handle),
            ctypes.c_uint32(locs_out_handle),
            ctypes.c_uint32(n)
        )
        if ret != 0:
            raise RuntimeError(f"map_op1_dev failed with code {ret}")

    def map_op2(self, map_id: int, op: int, locs_a: np.ndarray, locs_b: np.ndarray,
                locs_out: np.ndarray, n: Optional[int] = None):
        """Binary operation on map."""
        self._metrics['op_calls'] += 1

        locs_a = np.ascontiguousarray(locs_a, dtype=np.uint32)
        locs_b = np.ascontiguousarray(locs_b, dtype=np.uint32)
        locs_out = np.ascontiguousarray(locs_out, dtype=np.uint32)

        # Auto-compute n if not provided (UUCIS compatibility)
        if n is None:
            n = len(locs_a)

        ret = self._lib.map_op2(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(op),
            locs_a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(n)
        )

        if ret != 0:
            raise RuntimeError(f"map_op2 failed with code {ret}")

    def map_op2_dev(self, map_id: int, op: int, locs_a_handle: int, locs_b_handle: int,
                    locs_out_handle: int, n: int):
        """Binary op using device-local locs buffers."""
        self._metrics['op_calls'] += 1
        ret = self._lib.map_op2_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(op),
            ctypes.c_uint32(locs_a_handle),
            ctypes.c_uint32(locs_b_handle),
            ctypes.c_uint32(locs_out_handle),
            ctypes.c_uint32(n)
        )
        if ret != 0:
            raise RuntimeError(f"map_op2_dev failed with code {ret}")

    def map_matmul(self, map_id: int, locs_a: np.ndarray, locs_b: np.ndarray,
                   locs_c: np.ndarray, M: int, K: int, N: int, n_ops: int = 1):
        """Matrix multiplication: C = A @ B."""
        self._metrics['op_calls'] += 1

        locs_a = np.ascontiguousarray(locs_a, dtype=np.uint32)
        locs_b = np.ascontiguousarray(locs_b, dtype=np.uint32)
        locs_c = np.ascontiguousarray(locs_c, dtype=np.uint32)

        ret = self._lib.map_matmul(
            ctypes.c_uint32(map_id),
            locs_a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(M),
            ctypes.c_uint32(K),
            ctypes.c_uint32(N),
            ctypes.c_uint32(n_ops)
        )

        if ret != 0:
            raise RuntimeError(f"map_matmul failed with code {ret}")

    def map_matmul_dev(self, map_id: int, locs_a_handle: int, locs_b_handle: int,
                       locs_c_handle: int, M: int, K: int, N: int, n_ops: int = 1):
        """Matmul using device-local locs buffers."""
        self._metrics['op_calls'] += 1
        ret = self._lib.map_matmul_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(locs_a_handle),
            ctypes.c_uint32(locs_b_handle),
            ctypes.c_uint32(locs_c_handle),
            ctypes.c_uint32(M),
            ctypes.c_uint32(K),
            ctypes.c_uint32(N),
            ctypes.c_uint32(n_ops)
        )
        if ret != 0:
            raise RuntimeError(f"map_matmul_dev failed with code {ret}")

    def map_softmax(self, map_id: int, locs_in: np.ndarray, locs_out: np.ndarray,
                    row_size: int, n_rows: Optional[int] = None):
        """Softmax operation."""
        self._metrics['op_calls'] += 1

        locs_in = np.ascontiguousarray(locs_in, dtype=np.uint32)
        locs_out = np.ascontiguousarray(locs_out, dtype=np.uint32)

        # Auto-compute n_rows if not provided (UUCIS compatibility)
        if n_rows is None:
            n_rows = len(locs_in)

        ret = self._lib.map_softmax(
            ctypes.c_uint32(map_id),
            locs_in.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(n_rows),
            ctypes.c_uint32(row_size)
        )

        if ret != 0:
            raise RuntimeError(f"map_softmax failed with code {ret}")

    def map_softmax_dev(self, map_id: int, locs_in_handle: int, locs_out_handle: int,
                        row_size: int, n_rows: int):
        """Softmax using device-local locs buffers."""
        self._metrics['op_calls'] += 1
        ret = self._lib.map_softmax_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(locs_in_handle),
            ctypes.c_uint32(locs_out_handle),
            ctypes.c_uint32(n_rows),
            ctypes.c_uint32(row_size)
        )
        if ret != 0:
            raise RuntimeError(f"map_softmax_dev failed with code {ret}")

    def map_softmax_abs_dev(self, map_id: int, locs_in_handle: int, locs_out_handle: int,
                            row_size: int, n_rows: int):
        """Softmax using absolute row-base locs buffers."""
        self._metrics['op_calls'] += 1
        ret = self._lib.map_softmax_abs_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(locs_in_handle),
            ctypes.c_uint32(locs_out_handle),
            ctypes.c_uint32(n_rows),
            ctypes.c_uint32(row_size)
        )
        if ret != 0:
            raise RuntimeError(f"map_softmax_abs_dev failed with code {ret}")

    def map_attn_softmax_abs_dev(self, map_id: int, locs_in_handle: int, locs_out_handle: int,
                                 row_size: int, n_rows: int, scale: float, cap: float):
        """Attention-specific absolute-row softmax with built-in scale and optional softcap."""
        self._metrics['op_calls'] += 1
        ret = self._lib.map_attn_softmax_abs_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(locs_in_handle),
            ctypes.c_uint32(locs_out_handle),
            ctypes.c_uint32(n_rows),
            ctypes.c_uint32(row_size),
            ctypes.c_float(scale),
            ctypes.c_float(cap),
        )
        if ret != 0:
            raise RuntimeError(f"map_attn_softmax_abs_dev failed with code {ret}")

    def map_layernorm(self, map_id: int, locs_src: np.ndarray, locs_dst: np.ndarray,
                      locs_gamma: np.ndarray, locs_beta: np.ndarray,
                      dim: int, eps: float = 1e-5, n_rows: Optional[int] = None):
        """Layer normalization."""
        self._metrics['op_calls'] += 1

        locs_src = np.ascontiguousarray(locs_src, dtype=np.uint32)
        locs_dst = np.ascontiguousarray(locs_dst, dtype=np.uint32)
        locs_gamma = np.ascontiguousarray(locs_gamma, dtype=np.uint32)
        locs_beta = np.ascontiguousarray(locs_beta, dtype=np.uint32)

        # Auto-compute n_rows if not provided (UUCIS compatibility)
        if n_rows is None:
            n_rows = len(locs_src)

        ret = self._lib.map_layernorm(
            ctypes.c_uint32(map_id),
            locs_src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_beta.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(n_rows),
            ctypes.c_uint32(dim),
            ctypes.c_float(eps)
        )

        if ret != 0:
            raise RuntimeError(f"map_layernorm failed with code {ret}")

    def map_layernorm_dev(self, map_id: int, locs_src_handle: int, locs_dst_handle: int,
                          locs_gamma_handle: int, locs_beta_handle: int,
                          dim: int, eps: float = 1e-5, n_rows: int = 0):
        """LayerNorm using device-local locs buffers."""
        self._metrics['op_calls'] += 1
        ret = self._lib.map_layernorm_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(locs_src_handle),
            ctypes.c_uint32(locs_dst_handle),
            ctypes.c_uint32(locs_gamma_handle),
            ctypes.c_uint32(locs_beta_handle),
            ctypes.c_uint32(n_rows),
            ctypes.c_uint32(dim),
            ctypes.c_float(eps)
        )
        if ret != 0:
            raise RuntimeError(f"map_layernorm_dev failed with code {ret}")

    # ============================================
    # Transformer-specific ops
    # ============================================

    def map_rmsnorm_dev(self, map_id: int, locs_src_handle: int, locs_wt_handle: int,
                        locs_dst_handle: int, n_rows: int, dim: int, eps: float = 1e-5):
        """RMSNorm: x / sqrt(mean(x²) + eps) * weight.
        
        Used by Gemma/LLaMA instead of LayerNorm. No mean subtraction, no beta.
        """
        self._metrics['op_calls'] += 1
        ret = self._lib.map_rmsnorm_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(locs_src_handle),
            ctypes.c_uint32(locs_wt_handle),
            ctypes.c_uint32(locs_dst_handle),
            ctypes.c_uint32(n_rows),
            ctypes.c_uint32(dim),
            ctypes.c_float(eps)
        )
        if ret != 0:
            raise RuntimeError(f"map_rmsnorm_dev failed with code {ret}")

    def map_rope_dev(self, map_id: int, locs_src_handle: int, locs_dst_handle: int,
                     n_tokens: int, n_heads: int, head_dim: int,
                     pos_offset: int = 0, freq_base: float = 10000.0):
        """Rotary Positional Encoding (RoPE).
        
        Applies cos/sin rotation to pairs of elements in each head.
        Standard for Gemma/LLaMA/GPT-NeoX.
        """
        self._metrics['op_calls'] += 1
        ret = self._lib.map_rope_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(locs_src_handle),
            ctypes.c_uint32(locs_dst_handle),
            ctypes.c_uint32(n_tokens),
            ctypes.c_uint32(n_heads),
            ctypes.c_uint32(head_dim),
            ctypes.c_uint32(pos_offset),
            ctypes.c_float(freq_base)
        )
        if ret != 0:
            raise RuntimeError(f"map_rope_dev failed with code {ret}")

    def map_matmul_t_dev(self, map_id: int, locs_a_handle: int, locs_b_handle: int,
                         locs_c_handle: int, M: int, K: int, N: int, n_ops: int = 1):
        """Matrix multiply with B transposed: C = A @ B^T.
        
        For attention scores: Q @ K^T where Q is [M, K] and K is [N, K].
        """
        self._metrics['op_calls'] += 1
        ret = self._lib.map_matmul_t_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(locs_a_handle),
            ctypes.c_uint32(locs_b_handle),
            ctypes.c_uint32(locs_c_handle),
            ctypes.c_uint32(M),
            ctypes.c_uint32(K),
            ctypes.c_uint32(N),
            ctypes.c_uint32(n_ops)
        )
        if ret != 0:
            raise RuntimeError(f"map_matmul_t_dev failed with code {ret}")

    # ============================================
    # Cross-map (dual-buffer) operations
    # ============================================

    def map_matmul_t_x_dev(self, map_act: int, map_wt: int,
                           locs_a_handle: int, locs_b_handle: int,
                           locs_c_handle: int, M: int, K: int, N: int, n_ops: int = 1):
        """Cross-map matmul with B transposed: C = A @ B^T.

        A (activations) from map_act, B (weights) from map_wt, C written to map_act.
        """
        self._metrics['op_calls'] += 1
        ret = self._lib.map_matmul_t_x_dev(
            ctypes.c_uint32(map_act), ctypes.c_uint32(map_wt),
            ctypes.c_uint32(locs_a_handle), ctypes.c_uint32(locs_b_handle),
            ctypes.c_uint32(locs_c_handle),
            ctypes.c_uint32(M), ctypes.c_uint32(K), ctypes.c_uint32(N),
            ctypes.c_uint32(n_ops)
        )
        if ret != 0:
            raise RuntimeError(f"map_matmul_t_x_dev failed with code {ret}")

    def map_matmul_x_dev(self, map_act: int, map_wt: int,
                         locs_a_handle: int, locs_b_handle: int,
                         locs_c_handle: int, M: int, K: int, N: int, n_ops: int = 1):
        """Cross-map matmul: C = A @ B.

        A (activations) from map_act, B (weights) from map_wt, C written to map_act.
        """
        self._metrics['op_calls'] += 1
        ret = self._lib.map_matmul_x_dev(
            ctypes.c_uint32(map_act), ctypes.c_uint32(map_wt),
            ctypes.c_uint32(locs_a_handle), ctypes.c_uint32(locs_b_handle),
            ctypes.c_uint32(locs_c_handle),
            ctypes.c_uint32(M), ctypes.c_uint32(K), ctypes.c_uint32(N),
            ctypes.c_uint32(n_ops)
        )
        if ret != 0:
            raise RuntimeError(f"map_matmul_x_dev failed with code {ret}")

    def map_rmsnorm_x_dev(self, map_act: int, map_wt: int,
                          locs_src_handle: int, locs_wt_handle: int,
                          locs_dst_handle: int, n_rows: int, dim: int, eps: float = 1e-5):
        """Cross-map RMSNorm: activations from map_act, norm weights from map_wt.

        dst (in map_act) = (src / sqrt(mean(src²) + eps)) * weight
        """
        self._metrics['op_calls'] += 1
        ret = self._lib.map_rmsnorm_x_dev(
            ctypes.c_uint32(map_act), ctypes.c_uint32(map_wt),
            ctypes.c_uint32(locs_src_handle), ctypes.c_uint32(locs_wt_handle),
            ctypes.c_uint32(locs_dst_handle),
            ctypes.c_uint32(n_rows), ctypes.c_uint32(dim),
            ctypes.c_float(eps)
        )
        if ret != 0:
            raise RuntimeError(f"map_rmsnorm_x_dev failed with code {ret}")

    def map_row_gather_xq8_dev(self, map_act: int, map_wt: int,
                               src_base: int, dst_base: int,
                               row_idx: int, row_size: int, scale: float = 1.0):
        """Copy one contiguous Q8 row from map_wt into contiguous F32 storage in map_act."""
        self._metrics['op_calls'] += 1
        ret = self._lib.map_row_gather_xq8_dev(
            ctypes.c_uint32(map_act), ctypes.c_uint32(map_wt),
            ctypes.c_uint32(src_base), ctypes.c_uint32(dst_base),
            ctypes.c_uint32(row_idx), ctypes.c_uint32(row_size),
            ctypes.c_float(scale),
        )
        if ret != 0:
            raise RuntimeError(f"map_row_gather_xq8_dev failed with code {ret}")

    def map_matmul_t_xq4_dev(self, map_act: int, map_wt: int,
                              locs_a_handle: int, locs_b_handle: int,
                              locs_c_handle: int, M: int, K: int, N: int, n_ops: int = 1):
        """Cross-map F32×Q4 matmul with B transposed: C = A @ B^T.

        A (F32, from map_act) × B (Q4, from map_wt) → C (F32, to map_act).
        Q4 weights are dequantized inline using the map's qparams.
        """
        self._metrics['op_calls'] += 1
        ret = self._lib.map_matmul_t_xq4_dev(
            ctypes.c_uint32(map_act), ctypes.c_uint32(map_wt),
            ctypes.c_uint32(locs_a_handle), ctypes.c_uint32(locs_b_handle),
            ctypes.c_uint32(locs_c_handle),
            ctypes.c_uint32(M), ctypes.c_uint32(K), ctypes.c_uint32(N),
            ctypes.c_uint32(n_ops)
        )
        if ret != 0:
            raise RuntimeError(f"map_matmul_t_xq4_dev failed with code {ret}")

    def map_matmul_xq4_dev(self, map_act: int, map_wt: int,
                            locs_a_handle: int, locs_b_handle: int,
                            locs_c_handle: int, M: int, K: int, N: int, n_ops: int = 1):
        """Cross-map F32×Q4 matmul: C = A @ B (non-transposed).

        A (F32, from map_act) × B (Q4, from map_wt) → C (F32, to map_act).
        Q4 weights are dequantized inline using the map's qparams.
        """
        self._metrics['op_calls'] += 1
        ret = self._lib.map_matmul_xq4_dev(
            ctypes.c_uint32(map_act), ctypes.c_uint32(map_wt),
            ctypes.c_uint32(locs_a_handle), ctypes.c_uint32(locs_b_handle),
            ctypes.c_uint32(locs_c_handle),
            ctypes.c_uint32(M), ctypes.c_uint32(K), ctypes.c_uint32(N),
            ctypes.c_uint32(n_ops)
        )
        if ret != 0:
            raise RuntimeError(f"map_matmul_xq4_dev failed with code {ret}")

    def map_matmul_t_xq8_dev(self, map_act: int, map_wt: int,
                              locs_a_handle: int, locs_b_handle: int,
                              locs_c_handle: int, M: int, K: int, N: int, n_ops: int = 1):
        """Cross-map F32xQ8 matmul with B transposed: C = A @ B^T."""
        self._metrics['op_calls'] += 1
        ret = self._lib.map_matmul_t_xq8_dev(
            ctypes.c_uint32(map_act), ctypes.c_uint32(map_wt),
            ctypes.c_uint32(locs_a_handle), ctypes.c_uint32(locs_b_handle),
            ctypes.c_uint32(locs_c_handle),
            ctypes.c_uint32(M), ctypes.c_uint32(K), ctypes.c_uint32(N),
            ctypes.c_uint32(n_ops)
        )
        if ret != 0:
            raise RuntimeError(f"map_matmul_t_xq8_dev failed with code {ret}")

    def map_matmul_xq8_dev(self, map_act: int, map_wt: int,
                            locs_a_handle: int, locs_b_handle: int,
                            locs_c_handle: int, M: int, K: int, N: int, n_ops: int = 1):
        """Cross-map F32xQ8 matmul: C = A @ B (non-transposed)."""
        self._metrics['op_calls'] += 1
        ret = self._lib.map_matmul_xq8_dev(
            ctypes.c_uint32(map_act), ctypes.c_uint32(map_wt),
            ctypes.c_uint32(locs_a_handle), ctypes.c_uint32(locs_b_handle),
            ctypes.c_uint32(locs_c_handle),
            ctypes.c_uint32(M), ctypes.c_uint32(K), ctypes.c_uint32(N),
            ctypes.c_uint32(n_ops)
        )
        if ret != 0:
            raise RuntimeError(f"map_matmul_xq8_dev failed with code {ret}")

    def map_row_copy_dev(self, map_id: int, copy_spec_handle: int, src_base_handle: int,
                         n_copies: int, row_size: int):
        """Copy rows by index — embedding table lookup.
        
        copy_spec is a buffer of [src_row_idx, dst_row_base] pairs.
        Copies row from src_base + row_idx * row_size to dst_row_base.
        """
        self._metrics['op_calls'] += 1
        ret = self._lib.map_row_copy_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(copy_spec_handle),
            ctypes.c_uint32(src_base_handle),
            ctypes.c_uint32(n_copies),
            ctypes.c_uint32(row_size)
        )
        if ret != 0:
            raise RuntimeError(f"map_row_copy_dev failed with code {ret}")

    def map_row_copy_offset_dev(self, map_id: int, copy_spec_handle: int,
                                src_base_handle: int, dst_row_offset: int,
                                n_copies: int, row_size: int):
        """Copy rows by index with a runtime row offset added to every destination row."""
        self._metrics['op_calls'] += 1
        ret = self._lib.map_row_copy_offset_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(copy_spec_handle),
            ctypes.c_uint32(src_base_handle),
            ctypes.c_uint32(dst_row_offset),
            ctypes.c_uint32(n_copies),
            ctypes.c_uint32(row_size)
        )
        if ret != 0:
            raise RuntimeError(f"map_row_copy_offset_dev failed with code {ret}")

    def map_fma_dev(self, map_id: int, locs_a_handle: int, locs_b_handle: int,
                    locs_c_handle: int, locs_dst_handle: int, n_locs: int):
        """Fused multiply-add: dst = a * b + c element-wise.
        
        For residual connections, scale+bias, weighted sums.
        """
        self._metrics['op_calls'] += 1
        ret = self._lib.map_fma_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(locs_a_handle),
            ctypes.c_uint32(locs_b_handle),
            ctypes.c_uint32(locs_c_handle),
            ctypes.c_uint32(locs_dst_handle),
            ctypes.c_uint32(n_locs)
        )
        if ret != 0:
            raise RuntimeError(f"map_fma_dev failed with code {ret}")

    def map_broadcast(self, map_id: int, op: int, locs_src: np.ndarray, locs_scalar: np.ndarray,
                      locs_dst: np.ndarray, n: Optional[int] = None):
        """Broadcast scalar operation (element-wise op with scalar)."""
        self._metrics['op_calls'] += 1
        # Broadcast kernels use a compact op enum: 0=mul, 1=div, 2=add, 3=sub.
        # Translate the public OP_* constants into that shader-specific encoding.
        op = {
            OP_MUL: 0,
            OP_DIV: 1,
            OP_ADD: 2,
            OP_SUB: 3,
        }.get(op, op)

        locs_src = np.ascontiguousarray(locs_src, dtype=np.uint32)
        locs_scalar = np.ascontiguousarray(locs_scalar, dtype=np.uint32)
        locs_dst = np.ascontiguousarray(locs_dst, dtype=np.uint32)

        # Auto-compute n if not provided
        if n is None:
            n = len(locs_src)

        ret = self._lib.map_broadcast(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(op),
            locs_src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_scalar.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(n)
        )

        if ret != 0:
            raise RuntimeError(f"map_broadcast failed with code {ret}")

    def map_broadcast_dev(self, map_id: int, op: int, locs_src_handle: int, locs_scalar_handle: int,
                          locs_dst_handle: int, n: int):
        """Broadcast using device-local locs buffers."""
        self._metrics['op_calls'] += 1
        op = {
            OP_MUL: 0,
            OP_DIV: 1,
            OP_ADD: 2,
            OP_SUB: 3,
        }.get(op, op)
        ret = self._lib.map_broadcast_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(op),
            ctypes.c_uint32(locs_src_handle),
            ctypes.c_uint32(locs_scalar_handle),
            ctypes.c_uint32(locs_dst_handle),
            ctypes.c_uint32(n)
        )
        if ret != 0:
            raise RuntimeError(f"map_broadcast_dev failed with code {ret}")

    def map_reduce(self, map_id: int, op: int, locs_src: np.ndarray, locs_dst: np.ndarray,
                   n: Optional[int] = None):
        """Reduce operation (sum/max/min along pack dimension)."""
        self._metrics['op_calls'] += 1

        locs_src = np.ascontiguousarray(locs_src, dtype=np.uint32)
        locs_dst = np.ascontiguousarray(locs_dst, dtype=np.uint32)

        # Auto-compute n if not provided
        if n is None:
            n = len(locs_src)

        ret = self._lib.map_reduce(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(op),
            locs_src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(n)
        )

        if ret != 0:
            raise RuntimeError(f"map_reduce failed with code {ret}")

    def map_reduce_dev(self, map_id: int, op: int, locs_src_handle: int, locs_dst_handle: int, n: int):
        """Reduce using device-local locs buffers."""
        self._metrics['op_calls'] += 1
        ret = self._lib.map_reduce_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(op),
            ctypes.c_uint32(locs_src_handle),
            ctypes.c_uint32(locs_dst_handle),
            ctypes.c_uint32(n)
        )
        if ret != 0:
            raise RuntimeError(f"map_reduce_dev failed with code {ret}")

    def map_argmax_dev(self, map_id: int, locs_src_handle: int, locs_dst_handle: int, n: int):
        """Argmax over `n` scalar locations, writing the winning source index into dst[0]."""
        self._metrics['op_calls'] += 1
        ret = self._lib.map_argmax_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(locs_src_handle),
            ctypes.c_uint32(locs_dst_handle),
            ctypes.c_uint32(n)
        )
        if ret != 0:
            raise RuntimeError(f"map_argmax_dev failed with code {ret}")

    def map_repeat_penalty_dev(self, map_id: int, locs_src_handle: int,
                               token_ids_handle: int, n_ids: int, penalty: float):
        """Apply repeat penalty in-place to a shortlist of token ids within a logits loc buffer."""
        if not getattr(self, '_has_map_repeat_penalty_dev', False):
            raise NotImplementedError("map_repeat_penalty_dev is not available in the loaded backend")
        self._metrics['op_calls'] += 1
        ret = self._lib.map_repeat_penalty_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(locs_src_handle),
            ctypes.c_uint32(token_ids_handle),
            ctypes.c_uint32(n_ids),
            ctypes.c_float(penalty),
        )
        if ret != 0:
            raise RuntimeError(f"map_repeat_penalty_dev failed with code {ret}")

    def map_matvec_topk_t_xq8_dev(self, map_act: int, map_wt: int,
                                  locs_a_handle: int, locs_b_handle: int,
                                  penalty_ids_handle: int,
                                  locs_idx_dst_handle: int, locs_val_dst_handle: int,
                                  K: int, N: int, k: int,
                                  n_penalty: int = 0, repeat_penalty: float = 1.0):
        """Compute partial top-k directly from a F32 activation vector times a Q8 weight matrix."""
        if not getattr(self, '_has_map_matvec_topk_t_xq8_dev', False):
            raise NotImplementedError("map_matvec_topk_t_xq8_dev is not available in the loaded backend")
        self._metrics['op_calls'] += 1
        ret = self._lib.map_matvec_topk_t_xq8_dev(
            ctypes.c_uint32(map_act),
            ctypes.c_uint32(map_wt),
            ctypes.c_uint32(locs_a_handle),
            ctypes.c_uint32(locs_b_handle),
            ctypes.c_uint32(penalty_ids_handle),
            ctypes.c_uint32(locs_idx_dst_handle),
            ctypes.c_uint32(locs_val_dst_handle),
            ctypes.c_uint32(K),
            ctypes.c_uint32(N),
            ctypes.c_uint32(k),
            ctypes.c_uint32(n_penalty),
            ctypes.c_float(repeat_penalty),
        )
        if ret != 0:
            raise RuntimeError(f"map_matvec_topk_t_xq8_dev failed with code {ret}")

    def map_matvec_topk_t_xq4_dev(self, map_act: int, map_wt: int,
                                  locs_a_handle: int, locs_b_handle: int,
                                  penalty_ids_handle: int,
                                  locs_idx_dst_handle: int, locs_val_dst_handle: int,
                                  K: int, N: int, k: int,
                                  n_penalty: int = 0, repeat_penalty: float = 1.0):
        """Compute partial top-k directly from a F32 activation vector times a Q4 weight matrix."""
        if not getattr(self, '_has_map_matvec_topk_t_xq4_dev', False):
            raise NotImplementedError("map_matvec_topk_t_xq4_dev is not available in the loaded backend")
        self._metrics['op_calls'] += 1
        ret = self._lib.map_matvec_topk_t_xq4_dev(
            ctypes.c_uint32(map_act),
            ctypes.c_uint32(map_wt),
            ctypes.c_uint32(locs_a_handle),
            ctypes.c_uint32(locs_b_handle),
            ctypes.c_uint32(penalty_ids_handle),
            ctypes.c_uint32(locs_idx_dst_handle),
            ctypes.c_uint32(locs_val_dst_handle),
            ctypes.c_uint32(K),
            ctypes.c_uint32(N),
            ctypes.c_uint32(k),
            ctypes.c_uint32(n_penalty),
            ctypes.c_float(repeat_penalty),
        )
        if ret != 0:
            raise RuntimeError(f"map_matvec_topk_t_xq4_dev failed with code {ret}")

    def map_matvec_topk_t_xq4_ex_dev(self, map_act: int, map_wt: int,
                                     locs_a_handle: int, locs_b_handle: int,
                                     penalty_ids_handle: int,
                                     locs_idx_dst_handle: int, locs_val_dst_handle: int,
                                     K: int, N: int, k: int,
                                     n_penalty: int = 0, repeat_penalty: float = 1.0,
                                     rows_per_group: int = 256):
        """Compute partial top-k directly from a F32 activation vector times a Q4 weight matrix."""
        if not getattr(self, '_has_map_matvec_topk_t_xq4_ex_dev', False):
            raise NotImplementedError("map_matvec_topk_t_xq4_ex_dev is not available in the loaded backend")
        self._metrics['op_calls'] += 1
        ret = self._lib.map_matvec_topk_t_xq4_ex_dev(
            ctypes.c_uint32(map_act),
            ctypes.c_uint32(map_wt),
            ctypes.c_uint32(locs_a_handle),
            ctypes.c_uint32(locs_b_handle),
            ctypes.c_uint32(penalty_ids_handle),
            ctypes.c_uint32(locs_idx_dst_handle),
            ctypes.c_uint32(locs_val_dst_handle),
            ctypes.c_uint32(K),
            ctypes.c_uint32(N),
            ctypes.c_uint32(k),
            ctypes.c_uint32(n_penalty),
            ctypes.c_float(repeat_penalty),
            ctypes.c_uint32(rows_per_group),
        )
        if ret != 0:
            raise RuntimeError(f"map_matvec_topk_t_xq4_ex_dev failed with code {ret}")

    def map_matvec_topk_t_xq8_ex_dev(self, map_act: int, map_wt: int,
                                     locs_a_handle: int, locs_b_handle: int,
                                     penalty_ids_handle: int,
                                     locs_idx_dst_handle: int, locs_val_dst_handle: int,
                                     K: int, N: int, k: int,
                                     n_penalty: int = 0, repeat_penalty: float = 1.0,
                                     rows_per_group: int = 256):
        """Compute partial top-k directly from a F32 activation vector times a Q8 weight matrix."""
        if not getattr(self, '_has_map_matvec_topk_t_xq8_ex_dev', False):
            raise NotImplementedError("map_matvec_topk_t_xq8_ex_dev is not available in the loaded backend")
        self._metrics['op_calls'] += 1
        ret = self._lib.map_matvec_topk_t_xq8_ex_dev(
            ctypes.c_uint32(map_act),
            ctypes.c_uint32(map_wt),
            ctypes.c_uint32(locs_a_handle),
            ctypes.c_uint32(locs_b_handle),
            ctypes.c_uint32(penalty_ids_handle),
            ctypes.c_uint32(locs_idx_dst_handle),
            ctypes.c_uint32(locs_val_dst_handle),
            ctypes.c_uint32(K),
            ctypes.c_uint32(N),
            ctypes.c_uint32(k),
            ctypes.c_uint32(n_penalty),
            ctypes.c_float(repeat_penalty),
            ctypes.c_uint32(rows_per_group),
        )
        if ret != 0:
            raise RuntimeError(f"map_matvec_topk_t_xq8_ex_dev failed with code {ret}")

    def map_matvec_rerank_t_xq8_dev(self, map_act: int, map_wt: int,
                                    locs_a_handle: int, locs_b_handle: int,
                                    partial_idx_base_handle: int,
                                    sel_locs_handle: int, penalty_ids_handle: int,
                                    locs_idx_dst_handle: int, locs_val_dst_handle: int,
                                    K: int, n_ids: int,
                                    n_penalty: int = 0, repeat_penalty: float = 1.0):
        """Compute exact logits for shortlisted token ids selected from a partial-topk buffer."""
        if not getattr(self, '_has_map_matvec_rerank_t_xq8_dev', False):
            raise NotImplementedError("map_matvec_rerank_t_xq8_dev is not available in the loaded backend")
        self._metrics['op_calls'] += 1
        ret = self._lib.map_matvec_rerank_t_xq8_dev(
            ctypes.c_uint32(map_act),
            ctypes.c_uint32(map_wt),
            ctypes.c_uint32(locs_a_handle),
            ctypes.c_uint32(locs_b_handle),
            ctypes.c_uint32(partial_idx_base_handle),
            ctypes.c_uint32(sel_locs_handle),
            ctypes.c_uint32(penalty_ids_handle),
            ctypes.c_uint32(locs_idx_dst_handle),
            ctypes.c_uint32(locs_val_dst_handle),
            ctypes.c_uint32(K),
            ctypes.c_uint32(n_ids),
            ctypes.c_uint32(n_penalty),
            ctypes.c_float(repeat_penalty),
        )
        if ret != 0:
            raise RuntimeError(f"map_matvec_rerank_t_xq8_dev failed with code {ret}")

    def map_topk_dev(self, map_id: int, locs_src_handle: int, locs_idx_dst_handle: int,
                     locs_val_dst_handle: int, n: int, k: int):
        """Top-k over `n` scalar locations, writing indices and values into device dst buffers."""
        self._metrics['op_calls'] += 1
        ret = self._lib.map_topk_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(locs_src_handle),
            ctypes.c_uint32(locs_idx_dst_handle),
            ctypes.c_uint32(locs_val_dst_handle),
            ctypes.c_uint32(n),
            ctypes.c_uint32(k),
        )
        if ret != 0:
            raise RuntimeError(f"map_topk_dev failed with code {ret}")

    def map_topp_dev(self, map_id: int, locs_src_handle: int, locs_dst_handle: int,
                     n: int, temperature: float, top_p: float):
        """Normalize a sorted shortlist and zero out entries beyond cumulative top-p."""
        self._metrics['op_calls'] += 1
        ret = self._lib.map_topp_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(locs_src_handle),
            ctypes.c_uint32(locs_dst_handle),
            ctypes.c_uint32(n),
            ctypes.c_float(temperature),
            ctypes.c_float(top_p),
        )
        if ret != 0:
            raise RuntimeError(f"map_topp_dev failed with code {ret}")

    # ============================================
    # Batching
    # ============================================

    def batch_begin(self):
        """Begin batching mode (for UUCIS compatibility)."""
        self._lib.batch_begin()

    def batch_end(self):
        """End batching mode (for UUCIS compatibility)."""
        self._lib.batch_end()

    @contextmanager
    def batch(self):
        """Context manager for manual batching."""
        self.batch_begin()
        try:
            yield
        finally:
            self.batch_end()

    def sync(self):
        """Synchronize GPU (wait for all operations)."""
        self._lib.adamah_sync()

    def fusion_enable(self, enable: bool):
        """Enable or disable the fusion op-queue.

        When enabled, supported ops queue in C and flush into the current
        batch command buffer. Immediate ops auto-flush any pending fusion
        work before they execute, so mixed decode paths can safely keep
        fusion enabled during batch_begin()/batch_end().
        """
        self._lib.adamah_fusion_enable(ctypes.c_int(1 if enable else 0))

    def fusion_is_enabled(self) -> bool:
        """Return whether fusion queueing is currently enabled."""
        return bool(self._lib.adamah_fusion_is_enabled())

    @contextmanager
    def fusion_disabled(self):
        """Temporarily bypass fusion so an op executes immediately and in-order."""
        was_enabled = self.fusion_is_enabled()
        if was_enabled:
            self.fusion_enable(False)
        try:
            yield
        finally:
            if was_enabled:
                self.fusion_enable(True)

    def fusion_flush(self):
        """Flush any pending fusion-queued ops immediately."""
        ret = self._lib.adamah_fusion_flush()
        if ret != 0:
            raise RuntimeError(f"adamah_fusion_flush failed with code {ret}")

    # ============================================
    # High-Level Operation Wrappers
    # ============================================
    
    # --- Unary Activations ---
    def sigmoid(self, map_id: int, locs_in_h: int, locs_out_h: int, n: int):
        """Sigmoid activation: 1 / (1 + exp(-x))"""
        return self.map_op1_dev(map_id, OP_SIGMOID, locs_in_h, locs_out_h, n)
    
    def swish(self, map_id: int, locs_in_h: int, locs_out_h: int, n: int):
        """Swish activation: x * sigmoid(x)"""
        return self.map_op1_dev(map_id, OP_SWISH, locs_in_h, locs_out_h, n)
    
    def mish(self, map_id: int, locs_in_h: int, locs_out_h: int, n: int):
        """Mish activation: x * tanh(softplus(x))"""
        return self.map_op1_dev(map_id, OP_MISH, locs_in_h, locs_out_h, n)
    
    def selu(self, map_id: int, locs_in_h: int, locs_out_h: int, n: int):
        """SELU activation (Scaled ELU)"""
        return self.map_op1_dev(map_id, OP_SELU, locs_in_h, locs_out_h, n)
    
    def elu(self, map_id: int, locs_in_h: int, locs_out_h: int, n: int):
        """ELU activation"""
        return self.map_op1_dev(map_id, OP_ELU, locs_in_h, locs_out_h, n)
    
    def leaky_relu(self, map_id: int, locs_in_h: int, locs_out_h: int, n: int):
        """Leaky ReLU: max(x, 0.01*x)"""
        return self.map_op1_dev(map_id, OP_LEAKY_RELU, locs_in_h, locs_out_h, n)
    
    def hardsigmoid(self, map_id: int, locs_in_h: int, locs_out_h: int, n: int):
        """Hard Sigmoid: clamp(0.2*x + 0.5, 0, 1)"""
        return self.map_op1_dev(map_id, OP_HARDSIGMOID, locs_in_h, locs_out_h, n)
    
    def hardswish(self, map_id: int, locs_in_h: int, locs_out_h: int, n: int):
        """Hard Swish: x * hardsigmoid(x)"""
        return self.map_op1_dev(map_id, OP_HARDSWISH, locs_in_h, locs_out_h, n)
    
    # --- Unary Math ---
    def reciprocal(self, map_id: int, locs_in_h: int, locs_out_h: int, n: int):
        """Reciprocal: 1/x"""
        return self.map_op1_dev(map_id, OP_RECIPROCAL, locs_in_h, locs_out_h, n)
    
    def square(self, map_id: int, locs_in_h: int, locs_out_h: int, n: int):
        """Square: x*x"""
        return self.map_op1_dev(map_id, OP_SQUARE, locs_in_h, locs_out_h, n)
    
    def cube(self, map_id: int, locs_in_h: int, locs_out_h: int, n: int):
        """Cube: x*x*x"""
        return self.map_op1_dev(map_id, OP_CUBE, locs_in_h, locs_out_h, n)
    
    def softplus(self, map_id: int, locs_in_h: int, locs_out_h: int, n: int):
        """Softplus: log(1 + exp(x))"""
        return self.map_op1_dev(map_id, OP_SOFTPLUS, locs_in_h, locs_out_h, n)
    
    def sign(self, map_id: int, locs_in_h: int, locs_out_h: int, n: int):
        """Sign: -1, 0, or +1"""
        return self.map_op1_dev(map_id, OP_SIGN, locs_in_h, locs_out_h, n)
    
    def ceil(self, map_id: int, locs_in_h: int, locs_out_h: int, n: int):
        """Ceiling function"""
        return self.map_op1_dev(map_id, OP_CEIL, locs_in_h, locs_out_h, n)
    
    def floor(self, map_id: int, locs_in_h: int, locs_out_h: int, n: int):
        """Floor function"""
        return self.map_op1_dev(map_id, OP_FLOOR, locs_in_h, locs_out_h, n)
    
    def round(self, map_id: int, locs_in_h: int, locs_out_h: int, n: int):
        """Round function"""
        return self.map_op1_dev(map_id, OP_ROUND, locs_in_h, locs_out_h, n)
    
    # --- Unary Trigonometric ---
    def tan(self, map_id: int, locs_in_h: int, locs_out_h: int, n: int):
        """Tangent"""
        return self.map_op1_dev(map_id, OP_TAN, locs_in_h, locs_out_h, n)
    
    def asin(self, map_id: int, locs_in_h: int, locs_out_h: int, n: int):
        """Arc sine"""
        return self.map_op1_dev(map_id, OP_ASIN, locs_in_h, locs_out_h, n)
    
    def acos(self, map_id: int, locs_in_h: int, locs_out_h: int, n: int):
        """Arc cosine"""
        return self.map_op1_dev(map_id, OP_ACOS, locs_in_h, locs_out_h, n)
    
    def atan(self, map_id: int, locs_in_h: int, locs_out_h: int, n: int):
        """Arc tangent"""
        return self.map_op1_dev(map_id, OP_ATAN, locs_in_h, locs_out_h, n)
    
    def sinh(self, map_id: int, locs_in_h: int, locs_out_h: int, n: int):
        """Hyperbolic sine"""
        return self.map_op1_dev(map_id, OP_SINH, locs_in_h, locs_out_h, n)
    
    def cosh(self, map_id: int, locs_in_h: int, locs_out_h: int, n: int):
        """Hyperbolic cosine"""
        return self.map_op1_dev(map_id, OP_COSH, locs_in_h, locs_out_h, n)
    
    # --- Binary Comparisons ---
    def equal(self, map_id: int, locs_a_h: int, locs_b_h: int, locs_out_h: int, n: int):
        """Element-wise equality (returns 1.0 or 0.0)"""
        return self.map_op2_dev(map_id, OP_EQ, locs_a_h, locs_b_h, locs_out_h, n)
    
    def not_equal(self, map_id: int, locs_a_h: int, locs_b_h: int, locs_out_h: int, n: int):
        """Element-wise inequality"""
        return self.map_op2_dev(map_id, OP_NE, locs_a_h, locs_b_h, locs_out_h, n)
    
    def less_than(self, map_id: int, locs_a_h: int, locs_b_h: int, locs_out_h: int, n: int):
        """Element-wise less than"""
        return self.map_op2_dev(map_id, OP_LT, locs_a_h, locs_b_h, locs_out_h, n)
    
    def less_equal(self, map_id: int, locs_a_h: int, locs_b_h: int, locs_out_h: int, n: int):
        """Element-wise less than or equal"""
        return self.map_op2_dev(map_id, OP_LE, locs_a_h, locs_b_h, locs_out_h, n)
    
    def greater_than(self, map_id: int, locs_a_h: int, locs_b_h: int, locs_out_h: int, n: int):
        """Element-wise greater than"""
        return self.map_op2_dev(map_id, OP_GT, locs_a_h, locs_b_h, locs_out_h, n)
    
    def greater_equal(self, map_id: int, locs_a_h: int, locs_b_h: int, locs_out_h: int, n: int):
        """Element-wise greater than or equal"""
        return self.map_op2_dev(map_id, OP_GE, locs_a_h, locs_b_h, locs_out_h, n)
    
    # --- Binary Logical ---
    def logical_and(self, map_id: int, locs_a_h: int, locs_b_h: int, locs_out_h: int, n: int):
        """Logical AND (treats non-zero as true)"""
        return self.map_op2_dev(map_id, OP_AND, locs_a_h, locs_b_h, locs_out_h, n)
    
    def logical_or(self, map_id: int, locs_a_h: int, locs_b_h: int, locs_out_h: int, n: int):
        """Logical OR"""
        return self.map_op2_dev(map_id, OP_OR, locs_a_h, locs_b_h, locs_out_h, n)
    
    def logical_xor(self, map_id: int, locs_a_h: int, locs_b_h: int, locs_out_h: int, n: int):
        """Logical XOR"""
        return self.map_op2_dev(map_id, OP_XOR, locs_a_h, locs_b_h, locs_out_h, n)
    
    # --- Binary Math ---
    def minimum(self, map_id: int, locs_a_h: int, locs_b_h: int, locs_out_h: int, n: int):
        """Element-wise minimum"""
        return self.map_op2_dev(map_id, OP_MIN, locs_a_h, locs_b_h, locs_out_h, n)
    
    def maximum(self, map_id: int, locs_a_h: int, locs_b_h: int, locs_out_h: int, n: int):
        """Element-wise maximum"""
        return self.map_op2_dev(map_id, OP_MAX, locs_a_h, locs_b_h, locs_out_h, n)
    
    def power(self, map_id: int, locs_a_h: int, locs_b_h: int, locs_out_h: int, n: int):
        """Element-wise power: a^b"""
        return self.map_op2_dev(map_id, OP_POW, locs_a_h, locs_b_h, locs_out_h, n)
    
    def modulo(self, map_id: int, locs_a_h: int, locs_b_h: int, locs_out_h: int, n: int):
        """Element-wise modulo"""
        return self.map_op2_dev(map_id, OP_MOD, locs_a_h, locs_b_h, locs_out_h, n)
    
    def atan2(self, map_id: int, locs_a_h: int, locs_b_h: int, locs_out_h: int, n: int):
        """Two-argument arctangent"""
        return self.map_op2_dev(map_id, OP_ATAN2, locs_a_h, locs_b_h, locs_out_h, n)

    # ============================================
    # Array API (TODO: future enhancement)
    # ============================================

    def array(self, data, target='vram'):
        """Upload numpy array to GPU (returns ArrayHandle)."""
        # TODO: implement handle-based array API
        raise NotImplementedError("array() API not yet implemented - use map_create/scatter for now")

    def add(self, a: ArrayHandle, b: ArrayHandle, target='vram'):
        """Element-wise addition (lazy)."""
        raise NotImplementedError("add() API not yet implemented - use map_op2 for now")

    def mul(self, a: ArrayHandle, b: ArrayHandle, target='vram'):
        """Element-wise multiplication (lazy)."""
        raise NotImplementedError("mul() API not yet implemented - use map_op2 for now")

    # ============================================
    # Metrics
    # ============================================

    def reset_metrics(self):
        """Reset transfer metrics."""
        for key in self._metrics:
            self._metrics[key] = 0

    def print_metrics(self):
        """Print transfer metrics."""
        print("=== ADAMAH Metrics ===")
        print(f"gather() calls: {self._metrics['gather_calls']}")
        print(f"scatter() calls: {self._metrics['scatter_calls']}")
        print(f"Operation calls: {self._metrics['op_calls']}")
        print(f"CPU→GPU: {self._metrics['total_bytes_cpu_to_gpu'] / 1e6:.2f} MB")
        print(f"GPU→CPU: {self._metrics['total_bytes_gpu_to_cpu'] / 1e6:.2f} MB")

    # ============================================
    # Cleanup
    # ============================================

    def shutdown(self):
        """Shutdown ADAMAH and cleanup resources."""
        self._lib.adamah_shutdown()

    def __del__(self):
        try:
            self.shutdown()
        except:
            pass

    @property
    def uucis(self):
        """UUCIS wrapper API (for benchmark compatibility)."""
        if not hasattr(self, '_uucis'):
            if UUCISView is None:
                raise ImportError("uucis.py not found - UUCIS API not available")
            self._uucis = UUCISView(self)
        return self._uucis

    def __repr__(self):
        dtype_names = {0: 'f32', 1: 'bf16', 2: 'q8', 3: 'q4', 4: 'q6'}
        dt = dtype_names.get(getattr(self, '_active_dtype', 0), '?')
        return f"Adamah(v{__version__}, dtype={dt})"


# ============================================
# Top-level API — import adamah; gpu = adamah.init()
# ============================================

# Data types (accessible as adamah.DTYPE_BF16 etc.)
DTYPE_F32 = 0
DTYPE_BF16 = 1
DTYPE_Q8 = 2
DTYPE_Q4 = 3
DTYPE_Q6 = 4

# Reduce ops
REDUCE_SUM = 0
REDUCE_MAX = 1
REDUCE_MIN = 2

# Broadcast ops
BROADCAST_MUL = 0
BROADCAST_DIV = 1
BROADCAST_ADD = 2
BROADCAST_SUB = 3

def init(cache_mb: Optional[int] = None, cold_cache_mb: Optional[int] = None) -> Adamah:
    """Initialize ADAMAH and return a GPU context.
    
    Args:
        cache_mb: Hot cache size in MB (default: 512)
        cold_cache_mb: Cold cache size in MB (default: 512)
    
    Returns:
        Adamah instance ready to use
    
    Example:
        gpu = adamah.init()
        gpu.set_dtype(adamah.DTYPE_BF16)
    """
    return Adamah(cache_mb=cache_mb, cold_cache_mb=cold_cache_mb)


# Everything importable from `import adamah`
__all__ = [
    # Core
    'Adamah', 'init', '__version__',
    
    # Data types
    'DTYPE_F32', 'DTYPE_BF16', 'DTYPE_Q8', 'DTYPE_Q4', 'DTYPE_Q6',
    
    # Unary ops
    'OP_NEG', 'OP_ABS', 'OP_SQRT', 'OP_EXP', 'OP_LOG', 'OP_TANH',
    'OP_RELU', 'OP_GELU', 'OP_SIN', 'OP_COS', 'OP_TAN',
    'OP_ASIN', 'OP_ACOS', 'OP_ATAN', 'OP_SINH', 'OP_COSH',
    'OP_SIGMOID', 'OP_SWISH', 'OP_MISH', 'OP_SELU', 'OP_ELU',
    'OP_LEAKY_RELU', 'OP_CEIL', 'OP_FLOOR', 'OP_ROUND', 'OP_SIGN',
    'OP_RECIPROCAL', 'OP_SQUARE', 'OP_CUBE', 'OP_SOFTPLUS',
    'OP_HARDSIGMOID', 'OP_HARDSWISH', 'OP_EXPM1', 'OP_LOG1P',
    
    # Binary ops
    'OP_ADD', 'OP_SUB', 'OP_MUL', 'OP_DIV', 'OP_POW',
    'OP_MIN', 'OP_MAX', 'OP_MOD',
    'OP_EQ', 'OP_NE', 'OP_LT', 'OP_LE', 'OP_GT', 'OP_GE',
    'OP_AND', 'OP_OR', 'OP_XOR', 'OP_ATAN2', 'OP_STEP', 'OP_SMOOTHSTEP',
    
    # Reduce / Broadcast
    'REDUCE_SUM', 'REDUCE_MAX', 'REDUCE_MIN',
    'BROADCAST_MUL', 'BROADCAST_DIV', 'BROADCAST_ADD', 'BROADCAST_SUB',
    
    # Handles
    'ArrayHandle',
]
