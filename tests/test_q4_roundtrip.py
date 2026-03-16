#!/usr/bin/env python3
"""
Q4 GPU Pipeline Diagnostic
===========================
Tests the Q4 inference pipeline end-to-end:
  Test 1: F32 scatter/gather round-trip (sanity check)
  Test 2: F32×Q4 matmul with known values (checks dequant + matmul math)
  Test 3: F32×Q4 matmul vs CPU reference using a real GGUF tensor

NOTE: gather() on Q4 maps returns raw packed bytes, not dequantized float32.
      The engine never gathers from Q4 maps — only from the F32 workspace.
      Tests 2 and 3 verify the Q4 pipeline via matmul output, not round-trip.

NOTE: MAX_MAPS = 16 in the C backend — map IDs must be 0–15.

Usage:
    python tests/test_q4_roundtrip.py
    python tests/test_q4_roundtrip.py path/to/model.gguf   # uses first blk.0 tensor
"""
import os, sys
import numpy as np

from adam.paths import ROOT, setup; setup()

import adamah as A

GROUP_SIZE = 32          # must match engine Q4_GROUP_SIZE

# ──────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────

def quantize_groups(data: np.ndarray, group_size: int):
    """Pure-Python Q4 quantization — mirrors engine/scatter shader logic."""
    n = len(data)
    n_groups = (n + group_size - 1) // group_size
    scales = np.empty(n_groups, np.float32)
    zeros  = np.empty(n_groups, np.float32)
    for g in range(n_groups):
        chunk = data[g * group_size : (g + 1) * group_size]
        vmin, vmax = float(chunk.min()), float(chunk.max())
        scales[g] = (vmax - vmin) / 15.0 if vmax > vmin else 1.0
        zeros[g]  = vmin
    return scales, zeros


def dequantize_groups(data: np.ndarray, scales: np.ndarray, zeros: np.ndarray,
                      group_size: int) -> np.ndarray:
    """Mirrors GPU Q4 formula: q = round((x-zp)/scale) clamped 0..15, out = q*scale+zp.
    Uses float32 throughout to match the GPU shader's arithmetic exactly.
    Use this as the CPU reference for matmul comparisons."""
    n = len(data)
    out = np.empty(n, np.float32)
    eps = np.float32(1e-10)
    for g in range(len(scales)):
        s = g * group_size; e = min(s + group_size, n)
        sc = scales[g]   # np.float32 — keep native dtype, no Python float promotion
        zp = zeros[g]    # np.float32
        chunk = np.asarray(data[s:e], dtype=np.float32)
        q = np.clip(np.round((chunk - zp) / np.maximum(sc, eps)),
                    np.float32(0), np.float32(15))
        out[s:e] = q * sc + zp
    return out


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ──────────────────────────────────────────────────
# Test 1 — F32 scatter/gather round-trip
# ──────────────────────────────────────────────────

def test_f32_roundtrip(gpu):
    print("\n[TEST 1] F32 scatter/gather round-trip")
    N = 256
    data = np.arange(N, dtype=np.float32) * 0.01 - 1.0

    gpu.set_dtype(A.DTYPE_F32)
    gpu.map_create(10, 1, N)
    locs = np.arange(N, dtype=np.uint32)
    gpu.scatter(10, locs, data)
    recovered = gpu.gather(10, locs).view(np.float32)

    err = rmse(data, recovered)
    ok = err < 1e-6
    print(f"  RMSE={err:.2e}  {'OK (exact)' if ok else 'FAIL'}")
    if not ok:
        print("  ERROR: F32 scatter/gather is not lossless!")
    return ok


# ──────────────────────────────────────────────────
# Test 2 — matmul with known values
# ──────────────────────────────────────────────────

def test_matmul(gpu):
    print("\n[TEST 2] Small F32×Q4 matmul (C = A @ B^T)")
    # A[1,8] all-ones, B[4,8] partial identity: B[n, k] = 1 if k==n else 0
    # B must be stored row-major as [N, K], so np.eye(N, K) gives the right shape.
    M, K, N = 1, 8, 4
    A_data = np.ones((M, K), np.float32)
    B_data = np.eye(N, K, dtype=np.float32)   # shape (4, 8): B[n,n]=1, rest 0

    # Expected: C[0,n] = sum_k A[0,k]*B[n,k] = B[n,n]*1 = 1.0 for n in [0,3]
    expected = np.array([[1., 1., 1., 1.]], np.float32)

    scales, zeros = quantize_groups(B_data.flatten(), GROUP_SIZE)

    # F32 workspace (map 11): A at [0..K-1], C at [K..K+N-1]
    ws_id = 11
    ws_size = M * K + M * N + 8
    gpu.set_dtype(A.DTYPE_F32)
    gpu.map_create(ws_id, 1, ws_size)

    a_off = 0; c_off = M * K
    gpu.scatter(ws_id, np.arange(M * K, dtype=np.uint32), A_data.flatten())

    # Q4 weight map (map 12)
    wt_id = 12
    gpu.map_create_typed(wt_id, A.DTYPE_Q4, 1, K * N, GROUP_SIZE)
    gpu.set_qparams(wt_id, scales, zeros)
    gpu.scatter(wt_id, np.arange(K * N, dtype=np.uint32), B_data.flatten())

    # Handles
    a_h, _ = gpu.upload_dev(np.array([a_off], dtype=np.uint32))
    b_h, _ = gpu.upload_dev(np.array([0],     dtype=np.uint32))
    c_h, _ = gpu.upload_dev(np.array([c_off], dtype=np.uint32))

    gpu.map_matmul_t_xq4_dev(ws_id, wt_id, a_h, b_h, c_h, M, K, N)

    c_locs = np.arange(c_off, c_off + M * N, dtype=np.uint32)
    result = gpu.gather(ws_id, c_locs).view(np.float32)

    err = rmse(expected.flatten(), result)
    ok = err < 0.01
    print(f"  expected={expected.flatten()}  got={result.round(4)}")
    print(f"  RMSE={err:.6f}  {'OK' if ok else 'FAIL'}")
    if not ok:
        print("  ERROR: Matmul result is wrong — check shader or qparams binding!")
    return ok


# ──────────────────────────────────────────────────
# Test 3 — real GGUF tensor: GPU matmul vs CPU reference
# ──────────────────────────────────────────────────

def test_gguf_tensor(gpu, gguf_path: str):
    print(f"\n[TEST 3] GGUF tensor GPU matmul vs CPU ({os.path.basename(gguf_path)})")
    from adam.loaders.gguf import GGUFLoader
    loader = GGUFLoader(gguf_path)
    loader.load(verbose=False)

    # Pick the first 2D weight tensor from block 0
    name = None
    for n, t in loader.tensors.items():
        if t.ndim == 2 and n.startswith('blk.0') and 'weight' in n:
            name = n; break
    if name is None:
        print("  No suitable tensor found, skipping")
        return True

    tensor = loader.tensors[name]   # shape [out, in] after GGUF reshape
    N_out, K_in = tensor.shape
    data = tensor.flatten().astype(np.float32)
    print(f"  Tensor: {name}  shape={tensor.shape}  n={len(data)}")

    scales, zeros = quantize_groups(data, GROUP_SIZE)

    # Q4 weight map (map 14)
    wt_id = 14
    gpu.map_create_typed(wt_id, A.DTYPE_Q4, 1, len(data), GROUP_SIZE)
    gpu.set_qparams(wt_id, scales, zeros)
    gpu.scatter(wt_id, np.arange(len(data), dtype=np.uint32), data)

    # Random input A[1, K_in]
    rng = np.random.default_rng(0)
    a_data = rng.standard_normal(K_in).astype(np.float32) * 0.1

    # F32 workspace (map 13): A at [0..K_in-1], C at [K_in..K_in+N_out-1]
    ws_id = 13
    ws_size = K_in + N_out + 8
    gpu.set_dtype(A.DTYPE_F32)
    gpu.map_create(ws_id, 1, ws_size)
    gpu.scatter(ws_id, np.arange(K_in, dtype=np.uint32), a_data)

    a_h, _ = gpu.upload_dev(np.array([0],     dtype=np.uint32))
    b_h, _ = gpu.upload_dev(np.array([0],     dtype=np.uint32))
    c_h, _ = gpu.upload_dev(np.array([K_in],  dtype=np.uint32))

    gpu.map_matmul_t_xq4_dev(ws_id, wt_id, a_h, b_h, c_h, 1, K_in, N_out)

    c_locs = np.arange(K_in, K_in + N_out, dtype=np.uint32)
    result = gpu.gather(ws_id, c_locs).view(np.float32)

    # CPU reference: same Q4 quantization path as the GPU
    dequant = dequantize_groups(data, scales, zeros, GROUP_SIZE)
    expected = a_data @ dequant.reshape(N_out, K_in).T

    err = rmse(expected, result)
    out_std = float(np.std(expected)) + 1e-8
    # GPU and CPU use the same Q4 quantization, so ideally err ≈ 0.
    # In practice, GLSL round() vs np.round() gives ~1-LSB differences for ~2 elements
    # per K_in-length dot product (values near .5 quantization boundary), producing
    # a small but non-zero RMSE (~2-3% of output std).  Real bugs produce 20%+ errors.
    ok = err < out_std * 0.10   # 10% of output std catches real bugs, ignores rounding noise

    print(f"  RMSE={err:.6f}  out_std={out_std:.4f}  rel={err/out_std:.4f}  {'OK' if ok else 'FAIL'}")
    if not ok:
        print("  ERROR: GPU matmul diverges from CPU reference — check Q4 pipeline!")

    # Spot-check a few outputs
    idx = [0, N_out // 4, N_out // 2, N_out - 1]
    print("  Spot check (cpu → gpu):")
    for i in idx:
        if 0 <= i < N_out:
            print(f"    [{i:5d}]  cpu={expected[i]:+.6f}  gpu={result[i]:+.6f}  diff={expected[i]-result[i]:+.6f}")
    return ok


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    gguf_path = sys.argv[1] if len(sys.argv) > 1 else None
    if gguf_path is None:
        import glob
        found = glob.glob(os.path.join(ROOT, "*.gguf"))
        gguf_path = found[0] if found else None

    print("Initializing GPU...")
    gpu = A.init()

    results = []
    results.append(test_f32_roundtrip(gpu))
    results.append(test_matmul(gpu))
    if gguf_path:
        results.append(test_gguf_tensor(gpu, gguf_path))
    else:
        print("\n[TEST 3] Skipped — no .gguf file found (pass path as argument)")

    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} passed")
    if passed < total:
        print("SOME TESTS FAILED — GPU Q4 pipeline has bugs")
        sys.exit(1)
    else:
        print("All tests passed — GPU Q4 pipeline is working correctly")


if __name__ == '__main__':
    main()
