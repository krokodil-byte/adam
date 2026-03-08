import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

_UINT32_MAX = int(np.iinfo(np.uint32).max)

_UNARY_OPS = {
    # Basic math
    "NEG": 0,
    "ABS": 1,
    "SQRT": 2,
    "EXP": 3,
    "LOG": 4,
    "TANH": 5,
    "RELU": 6,
    "GELU": 7,
    # Trig
    "SIN": 8,
    "COS": 9,
    "TAN": 10,
    "ASIN": 11,
    "ACOS": 12,
    "ATAN": 13,
    "SINH": 14,
    "COSH": 15,
    # Activations
    "SIGMOID": 16,
    "SWISH": 17,
    "MISH": 18,
    "SELU": 19,
    "ELU": 20,
    "LEAKY_RELU": 21,
    # Rounding
    "CEIL": 22,
    "FLOOR": 23,
    "ROUND": 24,
    "SIGN": 25,
    # Extra math
    "RECIP": 26,
    "RECIPROCAL": 26,  # alias
    "SQR": 27,
    "SQUARE": 27,  # alias
    "CUBE": 28,
    "SOFTPLUS": 29,
    "HARDSIGMOID": 30,
    "HARDSWISH": 31,
    "EXPM1": 32,
    "LOG1P": 33,
}

_BINARY_OPS = {
    "ADD": 0,
    "SUB": 1,
    "MUL": 2,
    "DIV": 3,
    "POW": 4,
    "MIN": 5,
    "MAX": 6,
    "MOD": 7,
    "EQ": 8,
    "NE": 9,
    "LT": 10,
    "LE": 11,
    "GT": 12,
    "GE": 13,
    "AND": 14,
    "OR": 15,
    "XOR": 16,
    "ATAN2": 17,
    "STEP": 18,
    "SMOOTHSTEP": 19,
}

_REDUCE_OPS = {
    "SUM": 0,
    "MAX": 1,
    "MIN": 2,
}

_BROADCAST_OPS = {
    "ADD": 0,
    "SUB": 1,
    "MUL": 2,
    "DIV": 3,
}


def _dtype_for_wordlength(wordlength: int):
    if wordlength == 4:
        return np.float32
    if wordlength == 8:
        return np.float64
    return np.uint8


def _prod(values: Sequence[int]) -> int:
    total = 1
    for v in values:
        total *= int(v)
    return total


def _compute_strides(shape: Sequence[int]) -> Tuple[int, ...]:
    if not shape:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * int(shape[i + 1])
    return tuple(strides)


@dataclass
class MapMeta:
    dim: int
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    n_cells: int
    wordlength: int
    pack_size: int
    dtype: Any


class LocationEncoder:
    def __init__(self):
        self._meta: Dict[int, MapMeta] = {}

    def set_map(
        self,
        map_id: int,
        dim: int,
        n_cells: int,
        wordlength: int,
        shape: Optional[Sequence[int]] = None,
        pack_size: int = 1,
        dtype: Optional[Any] = None,
    ) -> MapMeta:
        dim = int(dim)
        if dim < 1:
            raise ValueError("dim must be >= 1")
        n_cells = int(n_cells)
        if n_cells < 0:
            raise ValueError("n_cells must be >= 0")
        if n_cells > _UINT32_MAX:
            raise ValueError(f"n_cells exceeds uint32 max ({_UINT32_MAX})")

        if shape is None:
            if dim == 1:
                shape = (n_cells,)
            else:
                shape = tuple([1] * (dim - 1) + [n_cells])
        else:
            if len(shape) != dim:
                raise ValueError("shape length must match dim")
            if _prod(shape) != n_cells:
                raise ValueError("product(shape) must equal n_cells")
            shape = tuple(int(v) for v in shape)

        strides = _compute_strides(shape)
        if dtype is None:
            dtype = _dtype_for_wordlength(wordlength)

        meta = MapMeta(
            dim=dim,
            shape=shape,
            strides=strides,
            n_cells=n_cells,
            wordlength=int(wordlength),
            pack_size=int(pack_size),
            dtype=dtype,
        )
        self._meta[int(map_id)] = meta
        return meta

    def remove(self, map_id: int) -> None:
        self._meta.pop(int(map_id), None)

    def get(self, map_id: int) -> Optional[MapMeta]:
        return self._meta.get(int(map_id))

    def encode(self, map_id: int, location_list) -> np.ndarray:
        meta = self.get(map_id)
        if meta is None:
            raise ValueError(f"Map {map_id} has no metadata; init with UUCIS first")

        if location_list is None:
            raise ValueError("location_list is required")

        # Coordinates: list of tuples/lists, or 2D ndarray
        if isinstance(location_list, np.ndarray):
            if location_list.ndim == 2:
                return self._coords_to_flat(meta, location_list)
            if location_list.ndim == 1:
                return self._flat_to_uint32(meta, location_list)
            raise ValueError("location_list ndarray must be 1D or 2D")

        if isinstance(location_list, tuple) and meta.dim > 1:
            return self._coords_to_flat(meta, [location_list])

        if isinstance(location_list, list):
            if not location_list:
                return np.ascontiguousarray(location_list, dtype=np.uint32)
            first = location_list[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                return self._coords_to_flat(meta, location_list)
            return self._flat_to_uint32(meta, location_list)

        return self._flat_to_uint32(meta, [location_list])

    def _coords_to_flat(self, meta: MapMeta, coords) -> np.ndarray:
        coords_arr = np.asarray(coords, dtype=np.int64)
        if coords_arr.size == 0:
            return np.ascontiguousarray(coords_arr.reshape(0), dtype=np.uint32)
        if coords_arr.ndim == 1:
            if meta.dim != 1:
                raise ValueError("coordinate list must have dim entries")
            coords_arr = coords_arr.reshape(-1, 1)
        if coords_arr.shape[1] != meta.dim:
            raise ValueError("coordinate list must have shape (n, dim)")

        if np.any(coords_arr < 0):
            raise ValueError("coordinates must be non-negative")

        for axis, size in enumerate(meta.shape):
            if np.any(coords_arr[:, axis] >= size):
                raise ValueError("coordinate out of bounds for shape")

        flat = (coords_arr * np.asarray(meta.strides, dtype=np.int64)).sum(axis=1)
        if np.any(flat < 0) or np.any(flat >= meta.n_cells):
            raise ValueError("flat index out of bounds")

        return np.ascontiguousarray(flat, dtype=np.uint32)

    def _flat_to_uint32(self, meta: MapMeta, locs) -> np.ndarray:
        arr = np.asarray(locs, dtype=np.int64)
        if arr.size == 0:
            return np.ascontiguousarray(arr.reshape(0), dtype=np.uint32)
        if np.any(arr < 0):
            raise ValueError("location indices must be non-negative")
        if np.any(arr >= meta.n_cells):
            raise ValueError("location index out of bounds")
        if np.any(arr > _UINT32_MAX):
            raise ValueError("location index exceeds uint32 max")
        return np.ascontiguousarray(arr, dtype=np.uint32)


class CachedVar:
    def __init__(self, handle: int, n_elems: int, dtype, ticket: Optional[int] = None):
        self.handle = int(handle)
        self.n_elems = int(n_elems)
        self.dtype = dtype
        self.ticket = ticket

    def __repr__(self):
        return f"CachedVar(handle={self.handle}, n_elems={self.n_elems}, dtype={self.dtype}, ticket={self.ticket})"


class UUCISView:
    def __init__(self, gpu):
        self._gpu = gpu
        self._loc = LocationEncoder()
        self._auto_batch_cached = True
        self._auto_batch_ops = False  # Disabled: fusion system in C handles batching
        self._batch_active = False
        self._batch_ops = 0
        self._batch_limit = 4096
        self._strict_cached_ops = True

    def set_cached_batching(self, enabled: bool) -> None:
        self._auto_batch_cached = bool(enabled)

    def set_auto_batching(self, enabled: bool, *, limit: Optional[int] = None) -> None:
        self._auto_batch_ops = bool(enabled)
        if limit is not None:
            self._batch_limit = max(1, int(limit))

    def set_strict_cached_ops(self, enabled: bool) -> None:
        self._strict_cached_ops = bool(enabled)

    # ------------------------------------------------------------------
    # Cached/device helpers
    # ------------------------------------------------------------------
    def cached(self, handle: int, n_elems: int, dtype, ticket: Optional[int] = None) -> CachedVar:
        return CachedVar(handle, n_elems, dtype, ticket=ticket)

    def to_cached(self, data) -> CachedVar:
        arr = np.ascontiguousarray(data)
        # Ensure uploads aren't recorded inside an active batch (staging buffers would explode).
        if self._batch_active:
            self._batch_flush()
        handle, ticket = self._gpu.upload_dev(arr)
        return CachedVar(handle, arr.size, arr.dtype, ticket=ticket)

    def cached_wait(self, cached_var: CachedVar) -> None:
        if cached_var is None:
            return
        if cached_var.ticket:
            self._batch_flush()
            self._gpu.synchronize(cached_var.ticket)
            cached_var.ticket = None

    def cached_download(self, cached_var: CachedVar) -> np.ndarray:
        self.cached_wait(cached_var)
        return self._gpu.download_dev(cached_var.handle, cached_var.n_elems, cached_var.dtype)

    def cache_locs(self, map_id: int, location_list) -> CachedVar:
        meta = self._ensure_meta(map_id)
        return self._ensure_locs_cached(map_id, location_list, meta)

    # ------------------------------------------------------------------
    # Initialization & Memory
    # ------------------------------------------------------------------
    def array_init(self, map_id: int, n_cells: int, wordlength: int):
        n_cells = self._validate_n_cells(n_cells)
        ret = self._gpu.map_init(map_id, word_size=int(wordlength), pack_size=1, n_packs=int(n_cells))
        if ret is None:
            ret = 0
        if ret == 0:
            self._loc.set_map(map_id, dim=1, n_cells=n_cells, wordlength=wordlength, shape=(int(n_cells),), pack_size=1)
        return ret

    def var_init(self, map_id: int, wordlength: int):
        return self.array_init(map_id, 1, wordlength)

    def cvar_init(self, map_id: int, wordlength: int) -> CachedVar:
        ret = self.var_init(map_id, wordlength)
        if ret != 0:
            raise RuntimeError(f"var_init failed with code {ret}")
        return self.cache_locs(map_id, [0])

    def carray_init(self, map_id: int, n_cells: int, wordlength: int) -> CachedVar:
        ret = self.array_init(map_id, n_cells, wordlength)
        if ret != 0:
            raise RuntimeError(f"array_init failed with code {ret}")
        return self.cache_locs(map_id, np.arange(int(n_cells), dtype=np.uint32))

    def map_init(
        self,
        map_id: int,
        dim: int,
        n_cells: int,
        wordlength: int,
        shape: Optional[Sequence[int]] = None,
        *,
        pack_size: Optional[int] = None,
    ):
        n_cells = self._validate_n_cells(n_cells)
        pack_size = self._validate_pack_size(pack_size)
        shape = self._normalize_shape(dim, n_cells, shape)
        ret = self._gpu.map_init(map_id, word_size=int(wordlength), pack_size=int(pack_size), n_packs=int(n_cells))
        if ret is None:
            ret = 0
        if ret == 0:
            self._loc.set_map(map_id, dim=dim, n_cells=n_cells, wordlength=wordlength, shape=shape, pack_size=pack_size)
        return ret

    def cmap_init(
        self,
        map_id: int,
        dim: int,
        n_cells: int,
        wordlength: int,
        shape: Optional[Sequence[int]] = None,
        *,
        pack_size: Optional[int] = None,
    ) -> CachedVar:
        ret = self.map_init(map_id, dim, n_cells, wordlength, shape=shape, pack_size=pack_size)
        if ret != 0:
            raise RuntimeError(f"map_init failed with code {ret}")
        return self.cache_locs(map_id, np.arange(int(n_cells), dtype=np.uint32))

    # ------------------------------------------------------------------
    # Scatter / Gather
    # ------------------------------------------------------------------
    def scatter(self, map_id: int, location_list, data):
        # Auto-sync: flush fusion queue before writing new data
        # This prevents race conditions with pending operations
        self._gpu.synchronize_all()
        
        meta = self._ensure_meta(map_id)
        if self._strict_cached_ops and not isinstance(location_list, CachedVar):
            raise RuntimeError("scatter requires cached locs (use cache_locs)")
        use_cached_locs = isinstance(location_list, CachedVar)
        use_cached_data = isinstance(data, CachedVar)

        if use_cached_locs or use_cached_data:
            self._batch_enter()
            if use_cached_data and not self._auto_batch_cached:
                self.cached_wait(data)
            locs_cached = self._ensure_locs_cached(map_id, location_list, meta)
            n_locs = locs_cached.n_elems
            if n_locs == 0:
                return 0

            if use_cached_data:
                if data.n_elems != n_locs * meta.pack_size:
                    raise ValueError("data length must match len(locs) * pack_size")
                src_cached = data
            else:
                arr = np.ascontiguousarray(data, dtype=meta.dtype)
                if arr.size != n_locs * meta.pack_size:
                    raise ValueError("data length must match len(locs) * pack_size")
                src_cached = self.to_cached(arr)
                if not self._auto_batch_cached:
                    self.cached_wait(src_cached)

            return self._gpu.map_scatter_dev(map_id, locs_cached.handle, n_locs, src_cached.handle)

        locs = self._loc.encode(map_id, location_list)

        arr = np.ascontiguousarray(data, dtype=meta.dtype)
        if arr.size != len(locs) * meta.pack_size:
            raise ValueError("data length must match len(locs) * pack_size")

        self._batch_enter()
        return self._gpu.map_scatter(map_id, locs, arr)

    def gather(self, map_id: int, location_list, target=None):
        # Auto-sync: flush fusion queue before gathering results
        self._gpu.synchronize_all()
        
        meta = self._ensure_meta(map_id)
        if self._strict_cached_ops and not isinstance(location_list, CachedVar):
            raise RuntimeError("gather requires cached locs (use cache_locs)")
        use_cached_locs = isinstance(location_list, CachedVar)
        use_cached_target = isinstance(target, CachedVar)

        if use_cached_locs or use_cached_target:
            locs_cached = self._ensure_locs_cached(map_id, location_list, meta)
            n_locs = locs_cached.n_elems
            if n_locs == 0:
                if use_cached_target:
                    target.handle = 0
                    target.n_elems = 0
                    target.dtype = meta.dtype
                    target.ticket = None
                    return target
                out = np.empty(0, dtype=meta.dtype)
                if isinstance(target, np.ndarray):
                    if target.size != out.size:
                        raise ValueError("target size must match gathered size")
                    target[...] = out.reshape(target.shape)
                    return target
                return out

            self._batch_enter()
            dst_handle, ticket, n_elems, dtype = self._gpu.map_gather_dev(map_id, locs_cached.handle, n_locs)
            if use_cached_target:
                target.handle = dst_handle
                target.n_elems = n_elems
                target.dtype = dtype
                target.ticket = ticket
                if not self._auto_batch_cached and ticket:
                    self._gpu.synchronize(ticket)
                    target.ticket = None
                return target

            self._batch_flush()
            out = self._gpu.download_dev(dst_handle, n_elems, dtype)
            if target is None:
                return out
            if isinstance(target, np.ndarray):
                if target.size != out.size:
                    raise ValueError("target size must match gathered size")
                target[...] = out.reshape(target.shape)
                return target
            return out

        locs = self._loc.encode(map_id, location_list)

        self._batch_flush()
        out = self._gpu.map_gather(map_id, locs, n_locs=len(locs))
        if target is None:
            return out

        if isinstance(target, np.ndarray):
            if target.size != out.size:
                raise ValueError("target size must match gathered size")
            target[...] = out.reshape(target.shape)
            return target

        return out

    # ------------------------------------------------------------------
    # Decode-State Primitives
    # ------------------------------------------------------------------
    def _ensure_single_cached_loc(self, map_id: int, locs, meta: MapMeta, what: str) -> CachedVar:
        if self._strict_cached_ops and not isinstance(locs, CachedVar):
            raise RuntimeError(f"{what} requires cached locs (use cache_locs)")
        cached = self._ensure_locs_cached(map_id, locs, meta)
        if cached.n_elems != 1:
            raise ValueError(f"{what} requires exactly one location")
        return cached

    def argmax(self, map_id: int, *, locs_src, locs_dst):
        """Argmax over scalar map cells; writes the winning source index into locs_dst[0]."""
        meta = self._ensure_meta(map_id)
        if meta.pack_size != 1:
            raise ValueError("argmax requires pack_size=1")
        if self._strict_cached_ops and not isinstance(locs_src, CachedVar):
            raise RuntimeError("argmax requires cached locs_src (use cache_locs)")
        src = self._ensure_locs_cached(map_id, locs_src, meta)
        dst = self._ensure_single_cached_loc(map_id, locs_dst, meta, "argmax")
        self._batch_enter()
        return self._gpu.map_argmax_dev(map_id, src.handle, dst.handle, src.n_elems)

    def topk(self, map_id: int, *, locs_src, locs_idx_dst, locs_val_dst, k: int):
        """Top-k over scalar map cells; writes source indices and values into cached dst locs."""
        meta = self._ensure_meta(map_id)
        if meta.pack_size != 1:
            raise ValueError("topk requires pack_size=1")
        if int(k) < 1:
            raise ValueError("k must be >= 1")
        if self._strict_cached_ops and not isinstance(locs_src, CachedVar):
            raise RuntimeError("topk requires cached locs_src (use cache_locs)")
        src = self._ensure_locs_cached(map_id, locs_src, meta)
        idx_dst = self._ensure_locs_cached(map_id, locs_idx_dst, meta)
        val_dst = self._ensure_locs_cached(map_id, locs_val_dst, meta)
        if idx_dst.n_elems < int(k) or val_dst.n_elems < int(k):
            raise ValueError("topk destination buffers must contain at least k locations")
        self._batch_enter()
        return self._gpu.map_topk_dev(map_id, src.handle, idx_dst.handle, val_dst.handle, src.n_elems, int(k))

    def topp(self, map_id: int, *, locs_src, locs_dst, n: int, temperature: float, top_p: float):
        """Top-p normalize a sorted shortlist; dst receives normalized kept probabilities."""
        meta = self._ensure_meta(map_id)
        if meta.pack_size != 1:
            raise ValueError("topp requires pack_size=1")
        if int(n) < 1:
            raise ValueError("n must be >= 1")
        src = self._ensure_locs_cached(map_id, locs_src, meta)
        dst = self._ensure_locs_cached(map_id, locs_dst, meta)
        if src.n_elems < int(n) or dst.n_elems < int(n):
            raise ValueError("topp source/destination buffers must contain at least n locations")
        self._batch_enter()
        return self._gpu.map_topp_dev(map_id, src.handle, dst.handle, int(n), float(temperature), float(top_p))

    def scalar_copy(self, map_id: int, *, locs_src, locs_dst):
        """Copy one scalar map cell to another scalar map cell."""
        meta = self._ensure_meta(map_id)
        src = self._ensure_single_cached_loc(map_id, locs_src, meta, "scalar_copy")
        dst = self._ensure_single_cached_loc(map_id, locs_dst, meta, "scalar_copy")
        with self._gpu.fusion_disabled():
            self._batch_enter()
            return self._gpu.map_op1_dev(map_id, 255, src.handle, dst.handle, 1)

    def scalar_move(self, map_id: int, *, locs_src, locs_dst):
        """Alias of scalar_copy for decode-state wiring."""
        return self.scalar_copy(map_id, locs_src=locs_src, locs_dst=locs_dst)

    def scalar_add(self, map_id: int, *, locs_src, locs_scalar, locs_dst):
        """dst = src + scalar for single-cell state updates."""
        meta = self._ensure_meta(map_id)
        src = self._ensure_single_cached_loc(map_id, locs_src, meta, "scalar_add")
        scalar = self._ensure_single_cached_loc(map_id, locs_scalar, meta, "scalar_add")
        dst = self._ensure_single_cached_loc(map_id, locs_dst, meta, "scalar_add")
        with self._gpu.fusion_disabled():
            self._batch_enter()
            return self._gpu.map_broadcast_dev(
                map_id, _BROADCAST_OPS["ADD"], src.handle, scalar.handle, dst.handle, 1
            )

    def scalar_increment(self, map_id: int, *, locs_src, locs_one, locs_dst):
        """Alias for scalar_add with a preloaded +1 cell."""
        return self.scalar_add(map_id, locs_src=locs_src, locs_scalar=locs_one, locs_dst=locs_dst)

    def scalar_eq(self, map_id: int, *, locs_a, locs_b, locs_dst):
        """dst = 1.0 iff a == b else 0.0."""
        meta = self._ensure_meta(map_id)
        a = self._ensure_single_cached_loc(map_id, locs_a, meta, "scalar_eq")
        b = self._ensure_single_cached_loc(map_id, locs_b, meta, "scalar_eq")
        dst = self._ensure_single_cached_loc(map_id, locs_dst, meta, "scalar_eq")
        with self._gpu.fusion_disabled():
            self._batch_enter()
            return self._gpu.map_op2_dev(map_id, _BINARY_OPS["EQ"], a.handle, b.handle, dst.handle, 1)

    def scalar_ge(self, map_id: int, *, locs_a, locs_b, locs_dst):
        """dst = 1.0 iff a >= b else 0.0."""
        meta = self._ensure_meta(map_id)
        a = self._ensure_single_cached_loc(map_id, locs_a, meta, "scalar_ge")
        b = self._ensure_single_cached_loc(map_id, locs_b, meta, "scalar_ge")
        dst = self._ensure_single_cached_loc(map_id, locs_dst, meta, "scalar_ge")
        with self._gpu.fusion_disabled():
            self._batch_enter()
            return self._gpu.map_op2_dev(map_id, _BINARY_OPS["GE"], a.handle, b.handle, dst.handle, 1)

    def scalar_or(self, map_id: int, *, locs_a, locs_b, locs_dst):
        """dst = logical_or(a, b) for single-cell state flags."""
        meta = self._ensure_meta(map_id)
        a = self._ensure_single_cached_loc(map_id, locs_a, meta, "scalar_or")
        b = self._ensure_single_cached_loc(map_id, locs_b, meta, "scalar_or")
        dst = self._ensure_single_cached_loc(map_id, locs_dst, meta, "scalar_or")
        with self._gpu.fusion_disabled():
            self._batch_enter()
            return self._gpu.map_op2_dev(map_id, _BINARY_OPS["OR"], a.handle, b.handle, dst.handle, 1)

    # ------------------------------------------------------------------
    # Persistence & Distribution
    # ------------------------------------------------------------------
    def map_save(self, map_id: int, path_or_stream):
        if hasattr(path_or_stream, "write"):
            raise NotImplementedError("stream save not supported; pass a file path")
        self._batch_flush()
        return self._gpu.map_save(map_id, path_or_stream)

    def map_load(self, map_id: int, path_or_stream):
        if hasattr(path_or_stream, "read"):
            raise NotImplementedError("stream load not supported; pass a file path")
        self._batch_flush()
        ret = self._gpu.map_load(map_id, path_or_stream)
        if ret == 0 and map_id in self._gpu._maps:
            word_size, pack_size, n_packs, dtype = self._gpu._maps[map_id]
            self._loc.set_map(
                map_id,
                dim=1,
                n_cells=n_packs,
                wordlength=word_size,
                shape=(int(n_packs),),
                pack_size=pack_size,
                dtype=dtype,
            )
        return ret

    def map_broadcast(self, source_id: int, target_map_id: int, mode, *, locs_src=None, locs_scalar=None, locs_dst=None):
        if source_id != target_map_id:
            raise NotImplementedError("broadcast across different maps is not supported")

        op_code = self._parse_broadcast_op(mode)
        meta = self._ensure_meta(source_id)
        if self._strict_cached_ops:
            if not isinstance(locs_src, CachedVar):
                raise RuntimeError("broadcast requires cached locs_src (use cache_locs)")
            if locs_scalar is not None and not isinstance(locs_scalar, CachedVar):
                raise RuntimeError("broadcast requires cached locs_scalar (use cache_locs)")
            if locs_dst is not None and not isinstance(locs_dst, CachedVar):
                raise RuntimeError("broadcast requires cached locs_dst (use cache_locs)")
        if self._has_cached_locs(locs_src, locs_scalar, locs_dst):
            src_locs = self._ensure_locs_cached(source_id, locs_src, meta)
            if locs_scalar is None:
                scalar_locs = src_locs
            else:
                scalar_locs = self._ensure_locs_cached(source_id, locs_scalar, meta)
            dst_locs = self._ensure_locs_cached(source_id, locs_dst, meta, default=src_locs)
            self._batch_enter()
            return self._gpu.map_broadcast_dev(
                source_id, op_code, src_locs.handle, scalar_locs.handle, dst_locs.handle, src_locs.n_elems
            )

        src_locs = self._ensure_locs(source_id, locs_src, meta)
        if locs_scalar is None:
            scalar_locs = src_locs
        else:
            scalar_locs = self._loc.encode(source_id, locs_scalar)
        dst_locs = self._ensure_locs(source_id, locs_dst, meta, default=src_locs)

        self._batch_enter()
        return self._gpu.map_broadcast(source_id, op_code, src_locs, scalar_locs, dst_locs)

    # ------------------------------------------------------------------
    # Scalar Operators (Op)
    # ------------------------------------------------------------------
    def op1(self, op_type, value, target=None):
        if self._is_map_id(value):
            return self.mop1(op_type, int(value), target)

        arr = np.asarray(value)
        res = self._apply_unary_cpu(op_type, arr)
        return self._write_scalar_result(res, target)

    def op2(self, op_type, value1, value2, target=None):
        if self._is_map_id(value1) or self._is_map_id(value2):
            if self._is_map_id(value1) and self._is_map_id(value2):
                return self.mop2(op_type, int(value1), int(value2), target)
            raise NotImplementedError("mixed scalar/map op2 is not supported")

        a = np.asarray(value1)
        b = np.asarray(value2)
        res = self._apply_binary_cpu(op_type, a, b)
        return self._write_scalar_result(res, target)

    def opN(self, op_type, values, target=None):
        values = list(values)
        if len(values) == 1:
            return self.op1(op_type, values[0], target)
        if len(values) == 2:
            kind, _ = self._parse_op_type(op_type)
            if kind == "REDUCE":
                arrs = [np.asarray(v) for v in values]
                res = self._apply_reduce_cpu(op_type, arrs)
                return self._write_scalar_result(res, target)
            return self.op2(op_type, values[0], values[1], target)

        if any(self._is_map_id(v) for v in values):
            raise NotImplementedError("opN with map ids is not supported")

        arrs = [np.asarray(v) for v in values]
        res = self._apply_reduce_cpu(op_type, arrs)
        return self._write_scalar_result(res, target)

    # ------------------------------------------------------------------
    # Map/Array Operators (Mop)
    # ------------------------------------------------------------------
    def mop1(self, op_type, map_id: int, target, *, locs_src=None, locs_dst=None, extra=None):
        meta = self._ensure_meta(map_id)
        self._validate_target_map(map_id, target)
        kind, name = self._parse_op_type(op_type)

        if kind == "UNARY" or (kind is None and name in _UNARY_OPS):
            op_code = _UNARY_OPS[name] if isinstance(name, str) else int(name)
            if self._strict_cached_ops and not isinstance(locs_src, CachedVar):
                raise RuntimeError("mop1 requires cached locs_src (use cache_locs)")
            if self._strict_cached_ops and locs_dst is not None and not isinstance(locs_dst, CachedVar):
                raise RuntimeError("mop1 requires cached locs_dst (use cache_locs)")
            if self._has_cached_locs(locs_src, locs_dst):
                src_locs = self._ensure_locs_cached(map_id, locs_src, meta)
                dst_locs = self._ensure_locs_cached(map_id, locs_dst, meta, default=src_locs)
                self._batch_enter()
                return self._gpu.map_op1_dev(map_id, op_code, src_locs.handle, dst_locs.handle, src_locs.n_elems)
            src_locs = self._ensure_locs(map_id, locs_src, meta)
            dst_locs = self._ensure_locs(map_id, locs_dst, meta, default=src_locs)
            self._batch_enter()
            return self._gpu.map_op1(map_id, op_code, src_locs, dst_locs)

        if kind == "REDUCE" or (kind is None and name in _REDUCE_OPS):
            op_code = _REDUCE_OPS[name] if isinstance(name, str) else int(name)
            if self._strict_cached_ops and not isinstance(locs_src, CachedVar):
                raise RuntimeError("mop1 reduce requires cached locs_src (use cache_locs)")
            if self._strict_cached_ops and locs_dst is not None and not isinstance(locs_dst, CachedVar):
                raise RuntimeError("mop1 reduce requires cached locs_dst (use cache_locs)")
            if self._has_cached_locs(locs_src, locs_dst):
                src_locs = self._ensure_locs_cached(map_id, locs_src, meta)
                dst_locs = self._ensure_locs_cached(map_id, locs_dst, meta, default=src_locs)
                self._batch_enter()
                return self._gpu.map_reduce_dev(map_id, op_code, src_locs.handle, dst_locs.handle, src_locs.n_elems)
            src_locs = self._ensure_locs(map_id, locs_src, meta)
            dst_locs = self._ensure_locs(map_id, locs_dst, meta, default=src_locs)
            self._batch_enter()
            return self._gpu.map_reduce(map_id, op_code, src_locs, dst_locs)

        if name == "SOFTMAX":
            if not extra or "row_size" not in extra:
                raise ValueError("extra['row_size'] is required for SOFTMAX")
            row_size = int(extra["row_size"])
            if self._strict_cached_ops and not isinstance(locs_src, CachedVar):
                raise RuntimeError("softmax requires cached locs_src (use cache_locs)")
            if self._strict_cached_ops and locs_dst is not None and not isinstance(locs_dst, CachedVar):
                raise RuntimeError("softmax requires cached locs_dst (use cache_locs)")
            if self._has_cached_locs(locs_src, locs_dst):
                src_locs = self._ensure_locs_cached(map_id, locs_src, meta)
                dst_locs = self._ensure_locs_cached(map_id, locs_dst, meta, default=src_locs)
                self._batch_enter()
                return self._gpu.map_softmax_dev(map_id, src_locs.handle, dst_locs.handle, row_size, src_locs.n_elems)
            src_locs = self._ensure_locs(map_id, locs_src, meta)
            dst_locs = self._ensure_locs(map_id, locs_dst, meta, default=src_locs)
            self._batch_enter()
            return self._gpu.map_softmax(map_id, src_locs, dst_locs, row_size)

        if name == "LAYERNORM":
            if not extra or "dim" not in extra:
                raise ValueError("extra['dim'] is required for LAYERNORM")
            dim = int(extra["dim"])
            eps = float(extra.get("eps", 1e-5)) if extra else 1e-5
            if not extra or "locs_gamma" not in extra or "locs_beta" not in extra:
                raise ValueError("extra['locs_gamma'] and extra['locs_beta'] are required for LAYERNORM")
            if self._strict_cached_ops and not isinstance(locs_src, CachedVar):
                raise RuntimeError("layernorm requires cached locs_src (use cache_locs)")
            if self._strict_cached_ops and locs_dst is not None and not isinstance(locs_dst, CachedVar):
                raise RuntimeError("layernorm requires cached locs_dst (use cache_locs)")
            if self._strict_cached_ops and not isinstance(extra.get("locs_gamma"), CachedVar):
                raise RuntimeError("layernorm requires cached locs_gamma (use cache_locs)")
            if self._strict_cached_ops and not isinstance(extra.get("locs_beta"), CachedVar):
                raise RuntimeError("layernorm requires cached locs_beta (use cache_locs)")
            if self._has_cached_locs(locs_src, locs_dst, extra.get("locs_gamma"), extra.get("locs_beta")):
                src_locs = self._ensure_locs_cached(map_id, locs_src, meta)
                dst_locs = self._ensure_locs_cached(map_id, locs_dst, meta, default=src_locs)
                locs_gamma = self._ensure_locs_cached(map_id, extra["locs_gamma"], meta)
                locs_beta = self._ensure_locs_cached(map_id, extra["locs_beta"], meta)
                self._batch_enter()
                return self._gpu.map_layernorm_dev(
                    map_id,
                    src_locs.handle,
                    dst_locs.handle,
                    locs_gamma.handle,
                    locs_beta.handle,
                    dim,
                    eps,
                    src_locs.n_elems,
                )
            src_locs = self._ensure_locs(map_id, locs_src, meta)
            dst_locs = self._ensure_locs(map_id, locs_dst, meta, default=src_locs)
            locs_gamma = self._loc.encode(map_id, extra["locs_gamma"])
            locs_beta = self._loc.encode(map_id, extra["locs_beta"])
            self._batch_enter()
            return self._gpu.map_layernorm(map_id, src_locs, dst_locs, locs_gamma, locs_beta, dim, eps)

        raise NotImplementedError(f"Unsupported mop1 op_type: {op_type}")

    def mop2(self, op_type, map_id_a: int, map_id_b: int, target, *, locs_a=None, locs_b=None, locs_dst=None, extra=None):
        if map_id_a != map_id_b:
            raise NotImplementedError("mop2 requires both operands in the same map")
        meta = self._ensure_meta(map_id_a)
        self._validate_target_map(map_id_a, target)

        kind, name = self._parse_op_type(op_type)

        if kind == "BINARY" or (kind is None and name in _BINARY_OPS):
            op_code = _BINARY_OPS[name] if isinstance(name, str) else int(name)
            if self._strict_cached_ops and not isinstance(locs_a, CachedVar):
                raise RuntimeError("mop2 requires cached locs_a (use cache_locs)")
            if self._strict_cached_ops and locs_b is not None and not isinstance(locs_b, CachedVar):
                raise RuntimeError("mop2 requires cached locs_b (use cache_locs)")
            if self._strict_cached_ops and locs_dst is not None and not isinstance(locs_dst, CachedVar):
                raise RuntimeError("mop2 requires cached locs_dst (use cache_locs)")
            if self._has_cached_locs(locs_a, locs_b, locs_dst):
                src_a = self._ensure_locs_cached(map_id_a, locs_a, meta)
                src_b = self._ensure_locs_cached(map_id_a, locs_b, meta, default=src_a)
                dst = self._ensure_locs_cached(map_id_a, locs_dst, meta, default=src_a)
                self._batch_enter()
                return self._gpu.map_op2_dev(
                    map_id_a, op_code, src_a.handle, src_b.handle, dst.handle, src_a.n_elems
                )
            src_a = self._ensure_locs(map_id_a, locs_a, meta)
            src_b = self._ensure_locs(map_id_a, locs_b, meta, default=src_a)
            dst = self._ensure_locs(map_id_a, locs_dst, meta, default=src_a)
            self._batch_enter()
            return self._gpu.map_op2(map_id_a, op_code, src_a, src_b, dst)

        if kind == "BROADCAST" or (kind is None and name in _BROADCAST_OPS):
            op_code = _BROADCAST_OPS[name] if isinstance(name, str) else int(name)
            if self._strict_cached_ops and not isinstance(locs_a, CachedVar):
                raise RuntimeError("broadcast requires cached locs_a (use cache_locs)")
            if self._strict_cached_ops and locs_b is not None and not isinstance(locs_b, CachedVar):
                raise RuntimeError("broadcast requires cached locs_b (use cache_locs)")
            if self._strict_cached_ops and locs_dst is not None and not isinstance(locs_dst, CachedVar):
                raise RuntimeError("broadcast requires cached locs_dst (use cache_locs)")
            if self._has_cached_locs(locs_a, locs_b, locs_dst):
                src_a = self._ensure_locs_cached(map_id_a, locs_a, meta)
                src_scalar = self._ensure_locs_cached(map_id_a, locs_b, meta, default=src_a)
                dst = self._ensure_locs_cached(map_id_a, locs_dst, meta, default=src_a)
                self._batch_enter()
                return self._gpu.map_broadcast_dev(
                    map_id_a, op_code, src_a.handle, src_scalar.handle, dst.handle, src_a.n_elems
                )
            src_a = self._ensure_locs(map_id_a, locs_a, meta)
            src_scalar = self._ensure_locs(map_id_a, locs_b, meta, default=src_a)
            dst = self._ensure_locs(map_id_a, locs_dst, meta, default=src_a)
            self._batch_enter()
            return self._gpu.map_broadcast(map_id_a, op_code, src_a, src_scalar, dst)

        if name == "MATMUL":
            if not extra:
                raise ValueError("extra is required for MATMUL")
            locs_a = extra.get("locs_a")
            locs_b = extra.get("locs_b")
            locs_c = extra.get("locs_c") or extra.get("locs_dst")
            M = int(extra["M"])
            K = int(extra["K"])
            N = int(extra["N"])
            if self._strict_cached_ops and not isinstance(locs_a, CachedVar):
                raise RuntimeError("matmul requires cached locs_a (use cache_locs)")
            if self._strict_cached_ops and not isinstance(locs_b, CachedVar):
                raise RuntimeError("matmul requires cached locs_b (use cache_locs)")
            if self._strict_cached_ops and not isinstance(locs_c, CachedVar):
                raise RuntimeError("matmul requires cached locs_c (use cache_locs)")
            if self._has_cached_locs(locs_a, locs_b, locs_c):
                locs_a = self._ensure_locs_cached(map_id_a, locs_a, meta)
                locs_b = self._ensure_locs_cached(map_id_a, locs_b, meta)
                locs_c = self._ensure_locs_cached(map_id_a, locs_c, meta)
                self._batch_enter()
                return self._gpu.map_matmul_dev(
                    map_id_a,
                    locs_a.handle,
                    locs_b.handle,
                    locs_c.handle,
                    M,
                    K,
                    N,
                    locs_a.n_elems,
                )
            locs_a = self._loc.encode(map_id_a, locs_a)
            locs_b = self._loc.encode(map_id_a, locs_b)
            locs_c = self._loc.encode(map_id_a, locs_c)
            self._batch_enter()
            return self._gpu.map_matmul(map_id_a, locs_a, locs_b, locs_c, M, K, N)

        raise NotImplementedError(f"Unsupported mop2 op_type: {op_type}")

    def mopN(self, op_type, ids, target=None, **kwargs):
        ids = list(ids)
        if len(ids) == 1:
            return self.mop1(op_type, ids[0], target, **kwargs)
        if len(ids) == 2:
            return self.mop2(op_type, ids[0], ids[1], target, **kwargs)
        raise NotImplementedError("mopN supports only 1 or 2 ids")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_meta(self, map_id: int) -> MapMeta:
        if map_id not in self._gpu._maps:
            self._loc.remove(map_id)
            raise ValueError(f"Map {map_id} not initialized")

        meta = self._loc.get(map_id)
        if meta is None:
            word_size, pack_size, n_packs, dtype = self._gpu._maps[map_id]
            meta = self._loc.set_map(
                map_id,
                dim=1,
                n_cells=n_packs,
                wordlength=word_size,
                shape=(int(n_packs),),
                pack_size=pack_size,
                dtype=dtype,
            )
        return meta

    def _ensure_locs(self, map_id: int, locs, meta: MapMeta, default=None) -> np.ndarray:
        if locs is None:
            if default is not None:
                return default
            return np.arange(meta.n_cells, dtype=np.uint32)
        return self._loc.encode(map_id, locs)

    def _ensure_locs_cached(self, map_id: int, locs, meta: MapMeta, default=None) -> CachedVar:
        if isinstance(locs, CachedVar):
            if not self._auto_batch_cached:
                self.cached_wait(locs)
            return locs
        if self._batch_active:
            self._batch_flush()
        if locs is None:
            if default is not None:
                if isinstance(default, CachedVar):
                    if not self._auto_batch_cached:
                        self.cached_wait(default)
                    return default
                locs_arr = np.ascontiguousarray(default, dtype=np.uint32)
            else:
                locs_arr = np.arange(meta.n_cells, dtype=np.uint32)
        else:
            locs_arr = self._loc.encode(map_id, locs)
        if locs_arr.size == 0:
            return CachedVar(0, 0, locs_arr.dtype)
        handle, ticket = self._gpu.upload_dev(locs_arr)
        cached = CachedVar(handle, locs_arr.size, locs_arr.dtype, ticket=ticket)
        if not self._auto_batch_cached:
            self.cached_wait(cached)
        return cached

    def _has_cached_locs(self, *locs) -> bool:
        for loc in locs:
            if isinstance(loc, CachedVar):
                return True
        return False

    def _batch_enter(self, ops: int = 1) -> None:
        if not self._auto_batch_ops:
            return
        if not self._batch_active:
            self._gpu.batch_begin()
            self._batch_active = True
            self._batch_ops = 0
        if self._batch_ops + ops > self._batch_limit:
            self._gpu.batch_end()
            self._batch_active = False
            self._batch_ops = 0
            self._gpu.batch_begin()
            self._batch_active = True
        self._batch_ops += ops

    def _batch_flush(self) -> None:
        if not self._batch_active:
            return
        self._gpu.batch_end()
        self._batch_active = False
        self._batch_ops = 0

    def _validate_n_cells(self, n_cells: int) -> int:
        n_cells = int(n_cells)
        if n_cells < 0:
            raise ValueError("n_cells must be >= 0")
        if n_cells > _UINT32_MAX:
            raise ValueError(f"n_cells exceeds uint32 max ({_UINT32_MAX})")
        return n_cells

    def _validate_pack_size(self, pack_size: Optional[int]) -> int:
        if pack_size is None:
            return 1
        pack_size = int(pack_size)
        if pack_size < 1:
            raise ValueError("pack_size must be >= 1")
        return pack_size

    def _normalize_shape(self, dim: int, n_cells: int, shape: Optional[Sequence[int]]) -> Tuple[int, ...]:
        dim = int(dim)
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if shape is None:
            if dim == 1:
                return (int(n_cells),)
            return tuple([1] * (dim - 1) + [int(n_cells)])
        if len(shape) != dim:
            raise ValueError("shape length must match dim")
        if _prod(shape) != int(n_cells):
            raise ValueError("product(shape) must equal n_cells")
        return tuple(int(v) for v in shape)

    def _is_map_id(self, value) -> bool:
        if isinstance(value, (int, np.integer)):
            return int(value) in self._gpu._maps
        return False

    def _validate_target_map(self, map_id: int, target) -> None:
        if self._is_map_id(target) and int(target) != int(map_id):
            raise NotImplementedError("target map must match source map")

    def _parse_op_type(self, op_type):
        if isinstance(op_type, str):
            t = op_type.strip().upper()
            if ":" in t:
                kind, name = t.split(":", 1)
                return kind, name
            return None, t
        return None, op_type

    def _parse_broadcast_op(self, mode):
        if isinstance(mode, str):
            key = mode.strip().upper()
            if key not in _BROADCAST_OPS:
                raise ValueError(f"Unknown broadcast mode: {mode}")
            return _BROADCAST_OPS[key]
        return int(mode)

    def _apply_unary_cpu(self, op_type, arr):
        kind, name = self._parse_op_type(op_type)
        if kind in ("UNARY", None) and isinstance(name, str):
            if name in _UNARY_OPS:
                return self._apply_unary_name(name, arr)
        return arr

    def _apply_binary_cpu(self, op_type, a, b):
        kind, name = self._parse_op_type(op_type)
        if kind in ("BINARY", None) and isinstance(name, str):
            if name in _BINARY_OPS:
                return self._apply_binary_name(name, a, b)
        raise NotImplementedError(f"Unsupported binary op_type for scalar op: {op_type}")

    def _apply_reduce_cpu(self, op_type, arrs):
        kind, name = self._parse_op_type(op_type)
        if kind in ("REDUCE", None) and isinstance(name, str):
            if name == "SUM":
                return np.sum(arrs, axis=0)
            if name == "MAX":
                return np.maximum.reduce(arrs)
            if name == "MIN":
                return np.minimum.reduce(arrs)
        raise NotImplementedError(f"Unsupported op_type for opN: {op_type}")

    def _apply_unary_name(self, name: str, arr):
        if name == "NEG":
            return -arr
        if name == "ABS":
            return np.abs(arr)
        if name == "SQRT":
            return np.sqrt(arr)
        if name == "EXP":
            return np.exp(arr)
        if name == "LOG":
            return np.log(arr)
        if name == "TANH":
            return np.tanh(arr)
        if name == "RELU":
            return np.maximum(arr, 0)
        if name == "GELU":
            return 0.5 * arr * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (arr + 0.044715 * np.power(arr, 3))))
        if name == "SIN":
            return np.sin(arr)
        if name == "COS":
            return np.cos(arr)
        if name == "RECIP":
            return 1.0 / arr
        if name == "SQR":
            return np.square(arr)
        return arr

    def _apply_binary_name(self, name: str, a, b):
        if name == "ADD":
            return a + b
        if name == "SUB":
            return a - b
        if name == "MUL":
            return a * b
        if name == "DIV":
            return a / b
        if name == "POW":
            return np.power(a, b)
        if name == "MIN":
            return np.minimum(a, b)
        if name == "MAX":
            return np.maximum(a, b)
        raise NotImplementedError(f"Unsupported binary op: {name}")

    def _write_scalar_result(self, res, target):
        if isinstance(target, CachedVar):
            return self.to_cached(res)
        if isinstance(target, np.ndarray):
            if target.size != np.size(res):
                raise ValueError("target size must match result size")
            target[...] = np.asarray(res).reshape(target.shape)
            return target
        if self._is_map_id(target):
            arr = np.asarray(res)
            locs = np.arange(arr.size, dtype=np.uint32)
            return self.scatter(int(target), locs, arr)
        return res

    # ============================================
    # Synchronization
    # ============================================
    
    def sync(self):
        """Synchronize: flush all pending operations and wait for completion.
        
        This is called automatically before gather(), but you can call it
        explicitly if you need to ensure all GPU operations have completed.
        """
        self._gpu.synchronize_all()
    
    def flush(self):
        """Alias for sync() - flush pending operations."""
        self.sync()
