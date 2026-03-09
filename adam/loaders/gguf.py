"""
ADAM — GGUF Loader v7
=====================
Reads GGUF model files in TWO modes:
  1. raw_blocks: Dict[name → bytes] — raw quantized blocks
  2. tensors: Dict[name → np.ndarray(f32)] — dequantized via gguf reference library

Dequantization is delegated to the gguf package (llama.cpp reference implementation)
to guarantee bit-exact correctness for all quantization types.
"""
import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from gguf import GGMLQuantizationType as _QType
import gguf as _gguf_lib

GGUF_MAGIC = 0x46554747
GGML_TYPE_F32=0; GGML_TYPE_F16=1; GGML_TYPE_Q4_0=2; GGML_TYPE_Q5_0=6
GGML_TYPE_Q8_0=8; GGML_TYPE_Q4_K=12; GGML_TYPE_Q6_K=14; GGML_TYPE_BF16=30

# Bytes per block and elements per block for each quantization type
QUANT_INFO = {
    GGML_TYPE_F32:  (4, 1),      # 4 bytes per element
    GGML_TYPE_F16:  (2, 1),      # 2 bytes per element
    GGML_TYPE_BF16: (2, 1),      # 2 bytes per element
    GGML_TYPE_Q8_0: (34, 32),    # 34 bytes per 32 elements
    GGML_TYPE_Q4_0: (18, 32),    # 18 bytes per 32 elements
    GGML_TYPE_Q5_0: (22, 32),    # 22 bytes per 32 elements
    GGML_TYPE_Q4_K: (144, 256),  # 144 bytes per 256 elements
    GGML_TYPE_Q6_K: (210, 256),  # 210 bytes per 256 elements
}

GGUF_TYPE_UINT8=0; GGUF_TYPE_INT8=1; GGUF_TYPE_UINT16=2; GGUF_TYPE_INT16=3
GGUF_TYPE_UINT32=4; GGUF_TYPE_INT32=5; GGUF_TYPE_FLOAT32=6; GGUF_TYPE_BOOL=7
GGUF_TYPE_STRING=8; GGUF_TYPE_ARRAY=9; GGUF_TYPE_UINT64=10; GGUF_TYPE_INT64=11
GGUF_TYPE_FLOAT64=12

@dataclass
class TensorInfo:
    name: str; shape: Tuple[int,...]; dtype: int; offset: int

class GGUFLoader:
    def __init__(self, path: str, keep_tensors: bool = True, keep_raw_blocks: bool = True):
        self.path = Path(path)
        self.keep_tensors = bool(keep_tensors)
        self.keep_raw_blocks = bool(keep_raw_blocks)
        self.metadata: Dict = {}
        self.tensor_infos: List[TensorInfo] = []
        self.tensor_info_by_name: Dict[str, TensorInfo] = {}
        self.tensors: Dict[str, np.ndarray] = {}      # f32 dequantized
        self.raw_blocks: Dict[str, bytes] = {}          # raw GGUF bytes
        self.tensor_types: Dict[str, int] = {}          # ggml type per tensor
        self.tensor_shapes: Dict[str, Tuple] = {}       # shape per tensor
        self._data_offset = 0

    def load(self, verbose=True):
        sz = self.path.stat().st_size
        if verbose: print(f"[GGUF] Loading {self.path.name} ({sz/1e9:.2f} GB)")
        with open(self.path, 'rb') as f:
            self._parse_header(f, verbose)
            self._parse_metadata(f, verbose)
            self._parse_tensor_infos(f, verbose)
            if self.keep_tensors or self.keep_raw_blocks:
                self._load_tensors(f, verbose)
            elif verbose:
                print(f"[GGUF] Indexed {len(self.tensor_infos)} tensors")
                print("[GGUF] Raw blocks: 0 MB (streamed on demand)")
                print("[GGUF] F32 dequant: 0 MB (streamed on demand)")
        return self

    def materialize(self, verbose=True):
        if self.keep_tensors is False and self.keep_raw_blocks is False:
            raise RuntimeError("materialize() requires keep_tensors or keep_raw_blocks enabled")
        if self.tensors or self.raw_blocks:
            return self
        with open(self.path, 'rb') as f:
            self._load_tensors(f, verbose)
        return self

    def _rs(self, f):
        l = struct.unpack('<Q', f.read(8))[0]
        return f.read(l).decode('utf-8')

    def _rv(self, f, t):
        R = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<B',10:'<Q',11:'<q',12:'<d'}
        if t in R:
            v = struct.unpack(R[t], f.read(struct.calcsize(R[t])))[0]
            return bool(v) if t==7 else v
        if t==8: return self._rs(f)
        if t==9:
            et=struct.unpack('<I',f.read(4))[0]; n=struct.unpack('<Q',f.read(8))[0]
            return [self._rv(f,et) for _ in range(n)]
        raise ValueError(f"Unknown type {t}")

    def _parse_header(self, f, v):
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != GGUF_MAGIC: raise ValueError(f"Bad magic {magic:#x}")
        ver = struct.unpack('<I', f.read(4))[0]
        self.n_tensors = struct.unpack('<Q', f.read(8))[0]
        self.n_kv = struct.unpack('<Q', f.read(8))[0]
        if v: print(f"[GGUF] v{ver}, {self.n_tensors} tensors, {self.n_kv} KV")

    def _parse_metadata(self, f, v):
        for _ in range(self.n_kv):
            k=self._rs(f); t=struct.unpack('<I',f.read(4))[0]
            self.metadata[k]=self._rv(f,t)
        if v:
            for k2,v2 in self.metadata.items():
                if isinstance(v2,list): print(f"  {k2} = [{type(v2[0]).__name__ if v2 else '?'} x {len(v2)}]")
                else: print(f"  {k2} = {v2}")

    def _parse_tensor_infos(self, f, v):
        for _ in range(self.n_tensors):
            nm=self._rs(f); nd=struct.unpack('<I',f.read(4))[0]
            sh=tuple(struct.unpack('<Q',f.read(8))[0] for _ in range(nd))
            dt=struct.unpack('<I',f.read(4))[0]; off=struct.unpack('<Q',f.read(8))[0]
            ti = TensorInfo(nm,sh,dt,off)
            self.tensor_infos.append(ti)
            self.tensor_info_by_name[nm] = ti
            self.tensor_types[nm] = dt
            self.tensor_shapes[nm] = sh
        align=self.metadata.get('general.alignment',32)
        pos=f.tell(); self._data_offset=(pos+align-1)//align*align
        if v:
            tc={}; nm={0:'f32',1:'f16',2:'q4_0',6:'q5_0',8:'q8_0',12:'q4_K',14:'q6_K',30:'bf16'}
            for ti in self.tensor_infos:
                n=nm.get(ti.dtype,f'type{ti.dtype}'); tc[n]=tc.get(n,0)+1
            print(f"[GGUF] Types: {tc}")

    def _load_tensors(self, f, v):
        total = len(self.tensor_infos)
        for i, ti in enumerate(self.tensor_infos):
            if v and (i%50==0 or i==total-1):
                print(f"[GGUF] Loading {i+1}/{total}: {ti.name} {ti.shape}")
            self._load_one(f, ti)
        if v:
            raw_bytes = sum(len(b) for b in self.raw_blocks.values())
            f32_bytes = sum(t.nbytes for t in self.tensors.values())
            print(f"[GGUF] Done: {total} tensors")
            print(f"[GGUF] Raw blocks: {raw_bytes/1e6:.0f} MB"
                  f"{' (kept for GPU zero-copy)' if self.keep_raw_blocks else ' (not retained)'}")
            print(f"[GGUF] F32 dequant: {f32_bytes/1e6:.0f} MB"
                  f"{' (kept)' if self.keep_tensors else ' (not retained)'}")

    def _load_one(self, f, ti):
        ne = 1
        for s in ti.shape: ne *= s
        f.seek(self._data_offset + ti.offset)

        if ti.dtype in QUANT_INFO:
            bpb, epb = QUANT_INFO[ti.dtype]
            n_blocks = ne // epb
            raw_size = n_blocks * bpb
            raw = f.read(raw_size)
            if self.keep_raw_blocks:
                self.raw_blocks[ti.name] = raw
            if self.keep_tensors:
                # Dequantize using the gguf reference library (llama.cpp).
                # ti.shape is in GGUF ne[] order (fastest dim first): (ne0, ne1, ...).
                # Reversing gives (ne1, ne0, ...) = (out, in) which is what
                # quant_shape_to_byte_shape expects and dequantize returns.
                qtype = _QType(ti.dtype)
                byte_shape = _gguf_lib.quant_shape_to_byte_shape(ti.shape[::-1], qtype)
                raw_np = np.frombuffer(raw, np.uint8).reshape(byte_shape)
                self.tensors[ti.name] = _gguf_lib.dequantize(raw_np, qtype)
        else:
            raise NotImplementedError(f"Unsupported type {ti.dtype} for {ti.name}")

    def _tensor_numel(self, ti: TensorInfo) -> int:
        ne = 1
        for s in ti.shape:
            ne *= s
        return ne

    def _tensor_raw_size(self, ti: TensorInfo) -> int:
        if ti.dtype not in QUANT_INFO:
            raise NotImplementedError(f"Unsupported type {ti.dtype} for {ti.name}")
        bpb, epb = QUANT_INFO[ti.dtype]
        ne = self._tensor_numel(ti)
        return (ne // epb) * bpb

    def get_tensor_info(self, name: str) -> TensorInfo:
        return self.tensor_info_by_name[name]

    def estimate_raw_bytes(self) -> int:
        return sum(self._tensor_raw_size(ti) for ti in self.tensor_infos)

    def estimate_f32_bytes(self) -> int:
        return sum(self._tensor_numel(ti) * 4 for ti in self.tensor_infos)

    def load_tensor_raw(self, name: str) -> bytes:
        if name in self.raw_blocks:
            return self.raw_blocks[name]
        ti = self.get_tensor_info(name)
        raw_size = self._tensor_raw_size(ti)
        with open(self.path, 'rb') as f:
            f.seek(self._data_offset + ti.offset)
            raw = f.read(raw_size)
        if self.keep_raw_blocks:
            self.raw_blocks[name] = raw
        return raw

    def load_tensor_f32(self, name: str) -> np.ndarray:
        if name in self.tensors:
            return self.tensors[name]
        ti = self.get_tensor_info(name)
        raw = self.load_tensor_raw(name)
        arr = self._dequant(ti.dtype, raw, self._tensor_numel(ti))
        if self.keep_tensors:
            self.tensors[name] = arr
        return arr

    def iter_tensor_chunks(self, name: str, max_chunk_mb: int = 32,
                           include_raw: bool = False, include_f32: bool = True):
        ti = self.get_tensor_info(name)
        ne = self._tensor_numel(ti)
        if ne <= 0:
            return

        if name in self.tensors or name in self.raw_blocks:
            raw = self.raw_blocks.get(name) if include_raw else None
            arr = self.tensors.get(name) if include_f32 else None
            if arr is None and include_f32:
                arr = self.load_tensor_f32(name)
            if raw is None and include_raw:
                raw = self.load_tensor_raw(name)
            yield 0, ne, raw, (arr.reshape(-1).astype(np.float32, copy=False) if arr is not None else None)
            return

        max_chunk_bytes = max(1, int(max_chunk_mb)) * 1024 * 1024
        bpb, epb = QUANT_INFO[ti.dtype]

        if include_f32:
            chunk_elems = max(epb, (max_chunk_bytes // 4 // epb) * epb)
        else:
            chunk_elems = max(epb, (max_chunk_bytes // bpb) * epb)
        chunk_elems = min(chunk_elems, ne)

        with open(self.path, 'rb') as f:
            f.seek(self._data_offset + ti.offset)
            elem_off = 0
            while elem_off < ne:
                take = min(chunk_elems, ne - elem_off)
                raw_size = (take // epb) * bpb
                raw = f.read(raw_size)
                arr = self._dequant(ti.dtype, raw, take) if include_f32 else None
                yield elem_off, take, (raw if include_raw else None), arr
                elem_off += take

    def release_tensors(self, keep_names=()):
        keep = set(keep_names)
        self.tensors = {name: arr for name, arr in self.tensors.items() if name in keep}

    def release_raw_blocks(self, keep_names=()):
        keep = set(keep_names)
        self.raw_blocks = {name: blk for name, blk in self.raw_blocks.items() if name in keep}

    def _dequant(self, dtype, data, ne):
        if dtype == GGML_TYPE_F32:
            return np.frombuffer(data, np.float32).copy()
        elif dtype == GGML_TYPE_F16:
            return np.frombuffer(data, np.float16).astype(np.float32)
        elif dtype == GGML_TYPE_BF16:
            raw = np.frombuffer(data, np.uint16)
            return (raw.astype(np.uint32) << 16).view(np.float32)
        elif dtype == GGML_TYPE_Q8_0:
            return self._dq8(data, ne)
        elif dtype == GGML_TYPE_Q4_0:
            return self._dq4_0(data, ne)
        elif dtype == GGML_TYPE_Q5_0:
            return self._dq5_0(data, ne)
        elif dtype == GGML_TYPE_Q4_K:
            return self._dq4k(data, ne)
        elif dtype == GGML_TYPE_Q6_K:
            return self._dq6k(data, ne)

    def _dq8(self, data, ne):
        nb=ne//32; r=np.empty(ne,np.float32); o=0
        for i in range(nb):
            s=np.frombuffer(data[o:o+2],np.float16)[0].astype(np.float32);o+=2
            q=np.frombuffer(data[o:o+32],np.int8).astype(np.float32);o+=32
            r[i*32:(i+1)*32]=q*s
        return r

    def _dq4_0(self, data, ne):
        # llama.cpp layout: y[0..15] = lower nibbles * s - 8*s,
        #                   y[16..31] = upper nibbles * s - 8*s
        nb=ne//32; r=np.empty(ne,np.float32); o=0
        for i in range(nb):
            s=np.frombuffer(data[o:o+2],np.float16)[0].astype(np.float32);o+=2
            nib=np.frombuffer(data[o:o+16],np.uint8);o+=16
            r[i*32:i*32+16]   = ((nib & 0x0F).astype(np.float32) - 8) * s
            r[i*32+16:i*32+32]= (((nib >> 4) & 0x0F).astype(np.float32) - 8) * s
        return r

    def _dq5_0(self, data, ne):
        nb=ne//32; r=np.empty(ne,np.float32); o=0
        for i in range(nb):
            s=np.frombuffer(data[o:o+2],np.float16)[0].astype(np.float32);o+=2
            qh=int.from_bytes(data[o:o+4],'little');o+=4
            ql=np.frombuffer(data[o:o+16],np.uint8);o+=16
            for j in range(32):
                lo=int(ql[j%16])&0x0F if j<16 else (int(ql[j-16])>>4)&0x0F
                hi=(qh>>j)&1; r[i*32+j]=s*((lo|(hi<<4))-16)
        return r

    def _dq4k(self, data, ne):
        # llama.cpp layout (dequantize_row_q4_K):
        # 4 groups of 64 elements, each group reads 32 bytes of quants.
        # First 32 elems of group: lower nibbles of the 32 bytes, scale[is].
        # Next  32 elems of group: upper nibbles of the 32 bytes, scale[is+1].
        nb=ne//256; r=np.empty(ne,np.float32)
        for i in range(nb):
            bo=i*144
            d =np.frombuffer(data[bo:bo+2],   np.float16)[0].astype(np.float32)
            dm=np.frombuffer(data[bo+2:bo+4], np.float16)[0].astype(np.float32)
            rs=np.frombuffer(data[bo+4:bo+16],np.uint8)
            qs=np.frombuffer(data[bo+16:bo+144],np.uint8)
            sc=np.zeros(8,np.float32); mn=np.zeros(8,np.float32)
            for j in range(8):
                if j<4: s2=int(rs[j])&0x3F; m2=int(rs[j+4])&0x3F
                else: s2=(int(rs[j+4])&0x0F)|((int(rs[j-4])>>6)<<4); m2=(int(rs[j+4])>>4)|((int(rs[j])>>6)<<4)
                sc[j]=d*s2; mn[j]=dm*m2
            is_=0
            for g in range(4):
                q=qs[g*32:g*32+32]
                lo=(q&0x0F).astype(np.float32); hi=((q>>4)&0x0F).astype(np.float32)
                base=i*256+g*64
                r[base:base+32]    = lo*sc[is_]   - mn[is_]
                r[base+32:base+64] = hi*sc[is_+1] - mn[is_+1]
                is_+=2
        return r

    def _dq6k(self, data, ne):
        # llama.cpp layout (dequantize_row_q6_K):
        # 2 halves of 128 elements. Within each half, l=0..31 fills
        # y[l], y[l+32], y[l+64], y[l+96] using two ql regions and qh.
        nb=ne//256; r=np.empty(ne,np.float32)
        for i in range(nb):
            bo=i*210
            ql=np.frombuffer(data[bo:bo+128],np.uint8)
            qh=np.frombuffer(data[bo+128:bo+192],np.uint8)
            sc=np.frombuffer(data[bo+192:bo+208],np.int8)
            d=np.frombuffer(data[bo+208:bo+210],np.float16)[0].astype(np.float32)
            for half in range(2):
                base=i*256+half*128
                for l in range(32):
                    is_=l//16
                    ql_a=int(ql[half*64+l]);    ql_b=int(ql[half*64+l+32])
                    qh_v=int(qh[half*32+l])
                    q1=(ql_a&0x0F)|(((qh_v>>0)&3)<<4)
                    q2=(ql_b&0x0F)|(((qh_v>>2)&3)<<4)
                    q3=(ql_a>>4)  |(((qh_v>>4)&3)<<4)
                    q4=(ql_b>>4)  |(((qh_v>>6)&3)<<4)
                    so=half*8+is_
                    r[base+l   ]=d*int(sc[so  ])*(q1-32)
                    r[base+l+32]=d*int(sc[so+2])*(q2-32)
                    r[base+l+64]=d*int(sc[so+4])*(q3-32)
                    r[base+l+96]=d*int(sc[so+6])*(q4-32)
        return r

    def get_tokenizer_vocab(self): return self.metadata.get('tokenizer.ggml.tokens')
    def get_tokenizer_scores(self): return self.metadata.get('tokenizer.ggml.scores')
    def get_bos_token_id(self): return self.metadata.get('tokenizer.ggml.bos_token_id', 2)
    def get_eos_token_id(self): return self.metadata.get('tokenizer.ggml.eos_token_id', 1)
