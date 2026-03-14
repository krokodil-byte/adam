#!/usr/bin/env python3
"""
GPU decode primitives smoke test.

Checks the first building blocks for a stateful GPU decode loop:
  - argmax over logits already resident in a workspace map
  - scalar copy / increment / comparisons / OR for decode state
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAMAH_DIR = os.path.join(ROOT, "adamah-MAIN")
for p in [ROOT, ADAMAH_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import adamah as A


def test_argmax(u):
    u.array_init(8, 12, 4)
    logits = np.array([-3.0, 5.5, 2.0, 5.5, 4.0, -1.0], dtype=np.float32)
    logits_locs = u.cache_locs(8, np.arange(logits.size, dtype=np.uint32))
    u.scatter(8, logits_locs, logits)

    src = logits_locs
    dst = u.cache_locs(8, [10])
    u.argmax(8, locs_src=src, locs_dst=dst)

    winner = int(u.gather(8, dst).view(np.float32)[0])
    print(f"[argmax] winner={winner}")
    return winner == 1


def test_topk(u):
    u.array_init(10, 20, 4)
    logits = np.array([-3.0, 5.5, 2.0, 5.5, 4.0, -1.0], dtype=np.float32)
    logits_locs = u.cache_locs(10, np.arange(logits.size, dtype=np.uint32))
    idx_dst = u.cache_locs(10, [8, 9, 10])
    val_dst = u.cache_locs(10, [12, 13, 14])
    u.scatter(10, logits_locs, logits)

    u.topk(10, locs_src=logits_locs, locs_idx_dst=idx_dst, locs_val_dst=val_dst, k=3)

    idx = u.gather(10, idx_dst).view(np.float32).astype(np.int32)
    val = u.gather(10, val_dst).view(np.float32)
    print(f"[topk] idx={idx.tolist()} val={val.tolist()}")
    return idx.tolist() == [1, 3, 4] and np.allclose(val, [5.5, 5.5, 4.0], atol=1e-6)


def test_topp(u):
    u.array_init(11, 16, 4)
    logits = np.array([5.0, 4.0, 1.0, -1.0], dtype=np.float32)
    src = u.cache_locs(11, [0, 1, 2, 3])
    dst = u.cache_locs(11, [8, 9, 10, 11])
    u.scatter(11, src, logits)

    u.topp(11, locs_src=src, locs_dst=dst, n=4, temperature=1.0, top_p=0.9)

    probs = u.gather(11, dst).view(np.float32)
    expected = np.array([0.7310586, 0.26894143, 0.0, 0.0], dtype=np.float32)
    print(f"[topp] probs={probs.tolist()}")
    return np.allclose(probs, expected, atol=1e-5)


def test_resolve_idx_and_categorical(gpu, u):
    if not getattr(gpu, '_has_map_resolve_idx_dev', False):
        print("[resolve_idx/sample_categorical] skipped (backend primitive not available)")
        return True
    if not getattr(gpu, '_has_map_sample_categorical_dev', False):
        print("[resolve_idx/sample_categorical] skipped (backend primitive not available)")
        return True

    u.array_init(16, 32, 4)
    full = u.cache_locs(16, np.arange(32, dtype=np.uint32))
    vals = np.zeros(32, dtype=np.float32)
    vals[4:8] = np.array([11.0, 22.0, 33.0, 44.0], dtype=np.float32)
    vals[8:10] = np.array([2.0, 0.0], dtype=np.float32)
    vals[16:18] = np.array([0.25, 0.75], dtype=np.float32)
    u.scatter(16, full, vals)

    base_h, _ = gpu.upload_dev(np.array([4], dtype=np.uint32))
    sel = u.cache_locs(16, np.array([8, 9], dtype=np.uint32))
    resolved = u.cache_locs(16, np.array([12, 13], dtype=np.uint32))
    probs = u.cache_locs(16, np.array([16, 17], dtype=np.uint32))
    rand = u.cache_locs(16, np.array([20], dtype=np.uint32))
    dst = u.cache_locs(16, np.array([21], dtype=np.uint32))
    u.scatter(16, rand, np.array([0.6], dtype=np.float32))

    gpu.batch_begin()
    gpu.map_resolve_idx_dev(16, base_h, sel.handle, resolved.handle, 2)
    gpu.map_sample_categorical_dev(16, resolved.handle, probs.handle, rand.handle, dst.handle, 2)
    gpu.batch_end()

    resolved_vals = u.gather(16, resolved).view(np.float32).astype(np.int32)
    chosen = int(u.gather(16, dst).view(np.float32)[0])
    print(f"[resolve_idx/sample_categorical] resolved={resolved_vals.tolist()} chosen={chosen}")
    return resolved_vals.tolist() == [33, 11] and chosen == 11


def test_softmax_abs(gpu, u):
    u.array_init(3, 16, 4)
    all_locs = u.cache_locs(3, np.arange(16, dtype=np.uint32))
    vals = np.zeros(16, dtype=np.float32)
    vals[0:3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    vals[8:11] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    u.scatter(3, all_locs, vals)

    row_bases = np.array([0, 8], dtype=np.uint32)
    row_h, _ = gpu.upload_dev(row_bases)
    gpu.batch_begin()
    gpu.map_softmax_abs_dev(3, row_h, row_h, 3, 2)
    gpu.batch_end()

    probe = u.cache_locs(3, np.array([0, 1, 2, 8, 9, 10], dtype=np.uint32))
    out = u.gather(3, probe).view(np.float32)
    expected = np.array([
        0.09003057, 0.24472848, 0.66524094,
        0.21194157, 0.21194157, 0.57611686,
    ], dtype=np.float32)
    print(f"[softmax_abs] vals={out.tolist()}")
    return np.allclose(out, expected, atol=1e-5)


def test_attn_softmax_abs(gpu, u):
    u.array_init(6, 16, 4)
    all_locs = u.cache_locs(6, np.arange(16, dtype=np.uint32))
    vals = np.zeros(16, dtype=np.float32)
    vals[0:3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    vals[8:11] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    u.scatter(6, all_locs, vals)

    row_bases = np.array([0, 8], dtype=np.uint32)
    row_h, _ = gpu.upload_dev(row_bases)
    gpu.batch_begin()
    gpu.map_attn_softmax_abs_dev(6, row_h, row_h, 3, 2, 0.5, 1.0)
    gpu.batch_end()

    probe = u.cache_locs(6, np.array([0, 1, 2, 8, 9, 10], dtype=np.uint32))
    out = u.gather(6, probe).view(np.float32)

    def ref(row):
        s = row * 0.5
        s = np.tanh(s / 1.0) * 1.0
        s = s - s.max()
        e = np.exp(s).astype(np.float32)
        return e / e.sum()

    expected = np.concatenate([ref(np.array([1.0, 2.0, 3.0], dtype=np.float32)),
                               ref(np.array([0.0, 0.0, 1.0], dtype=np.float32))])
    print(f"[attn_softmax_abs] vals={out.tolist()}")
    return np.allclose(out, expected, atol=1e-5)


def test_q8_gather_scatter(gpu, u):
    gpu.map_create_typed(13, A.DTYPE_Q8, 1, 4, 4)
    gpu.set_qparams(13,
                    np.array([0.5], dtype=np.float32),
                    np.array([-64.0], dtype=np.float32))
    q8_locs = np.arange(4, dtype=np.uint32)
    gpu.scatter(13, q8_locs, np.array([124, 132, 128, 130], dtype=np.uint8))

    u.array_init(14, 4, 4)
    src = u.cache_locs(13, q8_locs)
    dst = u.cache_locs(14, q8_locs)

    gathered_h, ticket, _, _ = gpu.map_gather_dev(13, src.handle, 4)
    if ticket:
        gpu.synchronize(ticket)
    gpu.map_scatter_dev(14, dst.handle, 4, gathered_h)

    vals = u.gather(14, dst).view(np.float32)
    print(f"[q8_gather] vals={vals.tolist()}")
    return np.allclose(vals, [-2.0, 2.0, 0.0, 1.0], atol=1e-6)


def test_row_gather_xq8(gpu, u):
    gpu.map_create_typed(4, A.DTYPE_Q8, 1, 8, 4)
    gpu.set_qparams(4,
                    np.array([0.5, 0.5], dtype=np.float32),
                    np.array([-64.0, -64.0], dtype=np.float32))
    gpu.scatter(4, np.arange(8, dtype=np.uint32),
                np.array([124, 132, 128, 130, 140, 120, 128, 132], dtype=np.uint8))

    u.array_init(5, 4, 4)
    gpu.batch_begin()
    gpu.map_row_gather_xq8_dev(5, 4, 0, 0, 1, 4, 1.0)
    gpu.batch_end()

    vals = u.gather(5, u.cache_locs(5, np.arange(4, dtype=np.uint32))).view(np.float32)
    print(f"[row_gather_xq8] vals={vals.tolist()}")
    return np.allclose(vals, [6.0, -4.0, 0.0, 2.0], atol=1e-6)


def test_matvec_t_xq8(gpu, u):
    gpu.map_create_typed(7, A.DTYPE_Q8, 1, 8, 4)
    gpu.set_qparams(7,
                    np.array([0.5, 0.5], dtype=np.float32),
                    np.array([-64.0, -64.0], dtype=np.float32))
    raw = np.array([128, 130, 132, 134, 120, 124, 128, 132], dtype=np.uint8)
    gpu.scatter(7, np.arange(8, dtype=np.uint32), raw)

    u.array_init(8, 8, 4)
    vec = np.array([1.0, 2.0, -1.0, 0.5], dtype=np.float32)
    vec_locs = u.cache_locs(8, np.arange(4, dtype=np.uint32))
    out_locs = u.cache_locs(8, np.array([4, 5], dtype=np.uint32))
    u.scatter(8, vec_locs, vec)

    base_h, _ = gpu.upload_dev(np.array([0], dtype=np.uint32))
    gpu.batch_begin()
    gpu.map_matmul_t_xq8_dev(8, 7, vec_locs.handle, base_h, out_locs.handle, 1, 4, 2)
    gpu.batch_end()

    vals = u.gather(8, out_locs).view(np.float32)
    expected = np.array([1.5, -7.0], dtype=np.float32)
    print(f"[matvec_t_xq8] vals={vals.tolist()}")
    return np.allclose(vals, expected, atol=1e-5)


def test_matvec_topk_t_xq8(gpu, u):
    if not getattr(gpu, '_has_map_matvec_topk_t_xq8_dev', False):
        print("[matvec_topk_t_xq8] skipped (backend primitive not available)")
        return True

    gpu.map_create_typed(7, A.DTYPE_Q8, 1, 16, 4)
    gpu.set_qparams(7,
                    np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                    np.array([-128.0, -128.0, -128.0, -128.0], dtype=np.float32))
    # Rows: [1,0,0,0], [0,2,0,0], [0,0,3,0], [0,0,0,4]
    raw = np.array([
        129, 128, 128, 128,
        128, 130, 128, 128,
        128, 128, 131, 128,
        128, 128, 128, 132,
    ], dtype=np.uint8)
    gpu.scatter(7, np.arange(16, dtype=np.uint32), raw)

    u.array_init(8, 16, 4)
    vec = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    vec_locs = u.cache_locs(8, np.arange(4, dtype=np.uint32))
    idx_dst = u.cache_locs(8, np.array([4, 5], dtype=np.uint32))
    val_dst = u.cache_locs(8, np.array([8, 9], dtype=np.uint32))
    u.scatter(8, vec_locs, vec)
    penalty_h, _ = gpu.upload_dev(np.array([3], dtype=np.uint32))

    base_h, _ = gpu.upload_dev(np.array([0], dtype=np.uint32))
    gpu.batch_begin()
    gpu.map_matvec_topk_t_xq8_dev(
        8, 7,
        vec_locs.handle, base_h, penalty_h,
        idx_dst.handle, val_dst.handle,
        4, 4, 2,
        1, 2.0,
    )
    gpu.batch_end()

    idx = u.gather(8, idx_dst).view(np.float32).astype(np.int32)
    vals = u.gather(8, val_dst).view(np.float32)
    print(f"[matvec_topk_t_xq8] idx={idx.tolist()} vals={vals.tolist()}")
    return idx.tolist() == [2, 1] and np.allclose(vals, [3.0, 2.0], atol=1e-5)


def test_matvec_topk_t_xq4(gpu, u):
    if not getattr(gpu, '_has_map_matvec_topk_t_xq4_dev', False):
        print("[matvec_topk_t_xq4] skipped (backend primitive not available)")
        return True

    gpu.map_destroy(13)
    gpu.map_destroy(14)
    gpu.map_create_typed(13, A.DTYPE_Q4, 1, 16, 32)
    gpu.set_qparams(13,
                    np.array([4.0 / 15.0], dtype=np.float32),
                    np.array([0.0], dtype=np.float32))
    rows = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 4.0],
    ], dtype=np.float32)
    gpu.scatter(13, np.arange(16, dtype=np.uint32), rows.reshape(-1))

    u.array_init(14, 16, 4)
    vec = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    vec_locs = u.cache_locs(14, np.arange(4, dtype=np.uint32))
    idx_dst = u.cache_locs(14, np.array([4, 5], dtype=np.uint32))
    val_dst = u.cache_locs(14, np.array([8, 9], dtype=np.uint32))
    u.scatter(14, vec_locs, vec)
    penalty_h, _ = gpu.upload_dev(np.array([3], dtype=np.uint32))

    base_h, _ = gpu.upload_dev(np.array([0], dtype=np.uint32))
    gpu.batch_begin()
    gpu.map_matvec_topk_t_xq4_dev(
        14, 13,
        vec_locs.handle, base_h, penalty_h,
        idx_dst.handle, val_dst.handle,
        4, 4, 2,
        1, 2.0,
    )
    gpu.batch_end()

    idx = u.gather(14, idx_dst).view(np.float32).astype(np.int32)
    vals = u.gather(14, val_dst).view(np.float32)
    print(f"[matvec_topk_t_xq4] idx={idx.tolist()} vals={vals.tolist()}")
    return idx[0] == 2 and idx[1] in (1, 3) and np.allclose(vals, [3.0, 2.0], atol=0.15)


def test_matvec_rerank_t_xq8(gpu, u):
    if not getattr(gpu, '_has_map_matvec_rerank_t_xq8_dev', False):
        print("[matvec_rerank_t_xq8] skipped (backend primitive not available)")
        return True

    gpu.map_destroy(13)
    gpu.map_destroy(14)
    gpu.map_create_typed(13, A.DTYPE_Q8, 1, 16, 4)
    gpu.set_qparams(13,
                    np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                    np.array([-128.0, -128.0, -128.0, -128.0], dtype=np.float32))
    raw = np.array([
        129, 128, 128, 128,
        128, 130, 128, 128,
        128, 128, 131, 128,
        128, 128, 128, 132,
    ], dtype=np.uint8)
    gpu.scatter(13, np.arange(16, dtype=np.uint32), raw)

    u.array_init(14, 32, 4)
    vec = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    vec_locs = u.cache_locs(14, np.arange(4, dtype=np.uint32))
    partial_idx_locs = u.cache_locs(14, np.array([8, 9, 10, 11], dtype=np.uint32))
    sel_locs = u.cache_locs(14, np.array([12, 13], dtype=np.uint32))
    idx_dst = u.cache_locs(14, np.array([16, 17], dtype=np.uint32))
    val_dst = u.cache_locs(14, np.array([20, 21], dtype=np.uint32))
    u.scatter(14, vec_locs, vec)
    u.scatter(14, partial_idx_locs, np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32))
    u.scatter(14, sel_locs, np.array([3.0, 1.0], dtype=np.float32))

    base_h, _ = gpu.upload_dev(np.array([0], dtype=np.uint32))
    partial_base_h, _ = gpu.upload_dev(np.array([8], dtype=np.uint32))
    penalty_h, _ = gpu.upload_dev(np.array([3], dtype=np.uint32))
    gpu.batch_begin()
    gpu.map_matvec_rerank_t_xq8_dev(
        14, 13,
        vec_locs.handle, base_h,
        partial_base_h,
        sel_locs.handle, penalty_h,
        idx_dst.handle, val_dst.handle,
        4, 2,
        1, 2.0,
    )
    gpu.batch_end()

    idx = u.gather(14, idx_dst).view(np.float32).astype(np.int32)
    vals = u.gather(14, val_dst).view(np.float32)
    print(f"[matvec_rerank_t_xq8] idx={idx.tolist()} vals={vals.tolist()}")
    return idx.tolist() == [3, 1] and np.allclose(vals, [2.0, 2.0], atol=1e-5)


def test_matvec_t_xq4(gpu):
    ws_id = 15
    wt_id = 14
    M, K, N = 1, 8, 4
    a_data = np.ones((M, K), dtype=np.float32)
    b_data = np.eye(N, K, dtype=np.float32)

    gpu.map_create_typed(wt_id, A.DTYPE_Q4, 1, K * N, 32)
    gpu.set_qparams(wt_id,
                    np.array([1.0 / 15.0], dtype=np.float32),
                    np.array([0.0], dtype=np.float32))
    gpu.scatter(wt_id, np.arange(K * N, dtype=np.uint32), b_data.flatten())

    gpu.map_create(ws_id, 1, K + N + 4)
    gpu.scatter(ws_id, np.arange(K, dtype=np.uint32), a_data.flatten())

    a_h, _ = gpu.upload_dev(np.array([0], dtype=np.uint32))
    b_h, _ = gpu.upload_dev(np.array([0], dtype=np.uint32))
    c_h, _ = gpu.upload_dev(np.array([K], dtype=np.uint32))
    gpu.batch_begin()
    gpu.map_matmul_t_xq4_dev(ws_id, wt_id, a_h, b_h, c_h, 1, K, N)
    gpu.batch_end()

    vals = gpu.gather(ws_id, np.arange(K, K + N, dtype=np.uint32)).view(np.float32)
    expected = np.ones(N, dtype=np.float32)
    print(f"[matvec_t_xq4] vals={vals.tolist()}")
    return np.allclose(vals, expected, atol=0.05)


def test_batch_gather_scatter(gpu, u):
    u.array_init(0, 8, 4)
    u.array_init(1, 8, 4)
    src = u.cache_locs(0, np.arange(4, dtype=np.uint32))
    dst = u.cache_locs(1, np.arange(4, dtype=np.uint32))
    u.scatter(0, src, np.array([9.0, -2.0, 7.0, 1.5], dtype=np.float32))

    gpu.batch_begin()
    gathered_h, _, _, _ = gpu.map_gather_dev(0, src.handle, 4)
    gpu.map_scatter_dev(1, dst.handle, 4, gathered_h)
    gpu.batch_end()

    vals = u.gather(1, dst).view(np.float32)
    print(f"[batch_gather] vals={vals.tolist()}")
    return np.allclose(vals, [9.0, -2.0, 7.0, 1.5], atol=1e-6)


def test_row_copy_offset(gpu, u):
    u.array_init(2, 40, 4)
    all_locs = u.cache_locs(2, np.arange(40, dtype=np.uint32))
    data = np.zeros(40, dtype=np.float32)
    data[:8] = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    u.scatter(2, all_locs, data)

    spec = np.array([0, 3, 1, 5], dtype=np.uint32)
    spec_h, _ = gpu.upload_dev(spec)
    base_h, _ = gpu.upload_dev(np.array([0], dtype=np.uint32))

    gpu.batch_begin()
    gpu.map_row_copy_offset_dev(2, spec_h, base_h, 2, 2, 4)
    gpu.batch_end()

    dst = u.cache_locs(2, np.array(list(range(20, 24)) + list(range(28, 32)), dtype=np.uint32))
    vals = u.gather(2, dst).view(np.float32)
    print(f"[row_copy_offset] vals={vals.tolist()}")
    expected = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    return np.allclose(vals, expected, atol=1e-6)


def test_fusion_row_copy_then_matmul(gpu, u):
    gpu.fusion_enable(True)
    u.array_init(16, 32, 4)
    all_locs = u.cache_locs(16, np.arange(32, dtype=np.uint32))
    data = np.zeros(32, dtype=np.float32)
    data[:8] = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    data[8:12] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    u.scatter(16, all_locs, data)

    spec_h, _ = gpu.upload_dev(np.array([0, 0, 1, 1], dtype=np.uint32))
    src_base_h, _ = gpu.upload_dev(np.array([0], dtype=np.uint32))
    a_h, _ = gpu.upload_dev(np.array([8], dtype=np.uint32))
    b_h, _ = gpu.upload_dev(np.array([20], dtype=np.uint32))
    c_h, _ = gpu.upload_dev(np.array([12], dtype=np.uint32))

    gpu.batch_begin()
    gpu.map_row_copy_offset_dev(16, spec_h, src_base_h, 5, 2, 4)
    gpu.map_matmul_t_dev(16, a_h, b_h, c_h, 1, 4, 2)
    gpu.batch_end()

    vals = u.gather(16, np.array([12, 13], dtype=np.uint32)).view(np.float32)
    expected = np.array([10.0, 100.0], dtype=np.float32)
    print(f"[fusion_row_copy_then_matmul] vals={vals.tolist()}")
    return np.allclose(vals, expected, atol=1e-6)


def test_scalar_state(u):
    u.array_init(9, 8, 4)
    # current=3, next=0, one=1, limit=4, ge=0, done=0, out=0, eq=0
    state = np.array([3.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    state_locs = u.cache_locs(9, np.arange(state.size, dtype=np.uint32))
    u.scatter(9, state_locs, state)

    loc_current = u.cache_locs(9, [0])
    loc_next = u.cache_locs(9, [1])
    loc_one = u.cache_locs(9, [2])
    loc_limit = u.cache_locs(9, [3])
    loc_ge = u.cache_locs(9, [4])
    loc_done = u.cache_locs(9, [5])
    loc_out = u.cache_locs(9, [6])
    loc_eq = u.cache_locs(9, [7])

    u.scalar_copy(9, locs_src=loc_current, locs_dst=loc_next)
    u.scalar_increment(9, locs_src=loc_next, locs_one=loc_one, locs_dst=loc_next)
    u.scalar_eq(9, locs_a=loc_next, locs_b=loc_limit, locs_dst=loc_eq)
    u.scalar_ge(9, locs_a=loc_next, locs_b=loc_limit, locs_dst=loc_ge)
    u.scalar_or(9, locs_a=loc_ge, locs_b=loc_done, locs_dst=loc_out)

    out = u.gather(9, state_locs).view(np.float32)
    print(f"[state] {out.tolist()}")
    return (
        int(out[1]) == 4 and
        int(out[4]) == 1 and
        int(out[6]) == 1 and
        int(out[7]) == 1
    )


def test_fusion_batch_order(gpu, u):
    gpu.fusion_enable(True)
    u.array_init(12, 12, 4)
    full = u.cache_locs(12, np.arange(12, dtype=np.uint32))
    src = u.cache_locs(12, [0, 1, 2])
    dbl = u.cache_locs(12, [4, 5, 6])
    win = u.cache_locs(12, [8])
    neg = u.cache_locs(12, [9, 10, 11])
    u.scatter(12, full, np.array([1.0, 2.0, 3.0] + [0.0] * 9, dtype=np.float32))

    gpu.batch_begin()
    gpu.map_op2_dev(12, A.OP_ADD, src.handle, src.handle, dbl.handle, 3)
    gpu.map_argmax_dev(12, dbl.handle, win.handle, 3)
    gpu.batch_end()

    winner = int(u.gather(12, win).view(np.float32)[0])

    gpu.batch_begin()
    gpu.map_op1_dev(12, A.OP_NEG, src.handle, neg.handle, 3)
    gpu.batch_end()

    neg_vals = u.gather(12, neg).view(np.float32)
    print(f"[fusion] winner={winner} neg={neg_vals.tolist()}")
    return winner == 2 and np.allclose(neg_vals, [-1.0, -2.0, -3.0], atol=1e-6)


def main():
    gpu = A.init()
    try:
        gpu.set_dtype(A.DTYPE_F32)
        u = gpu.uucis
        ok_argmax = test_argmax(u)
        ok_topk = test_topk(u)
        ok_topp = test_topp(u)
        ok_resolve_sample = test_resolve_idx_and_categorical(gpu, u)
        ok_softmax_abs = test_softmax_abs(gpu, u)
        ok_attn_softmax_abs = test_attn_softmax_abs(gpu, u)
        ok_q8 = test_q8_gather_scatter(gpu, u)
        ok_row_gather = test_row_gather_xq8(gpu, u)
        ok_matvec_q8 = test_matvec_t_xq8(gpu, u)
        ok_matvec_topk_q8 = test_matvec_topk_t_xq8(gpu, u)
        ok_matvec_topk_q4 = test_matvec_topk_t_xq4(gpu, u)
        ok_matvec_rerank_q8 = test_matvec_rerank_t_xq8(gpu, u)
        ok_matvec_q4 = test_matvec_t_xq4(gpu)
        ok_batch_gather = test_batch_gather_scatter(gpu, u)
        ok_row_copy = test_row_copy_offset(gpu, u)
        ok_row_copy_fusion = test_fusion_row_copy_then_matmul(gpu, u)
        ok_state = test_scalar_state(u)
        ok_fusion = test_fusion_batch_order(gpu, u)
    finally:
        gpu.shutdown()

    ok = (
        ok_argmax and ok_topk and ok_topp and ok_resolve_sample and ok_softmax_abs and ok_attn_softmax_abs and
        ok_q8 and ok_row_gather and ok_matvec_q8 and ok_matvec_topk_q8 and
        ok_matvec_topk_q4 and ok_matvec_rerank_q8 and ok_matvec_q4 and
        ok_batch_gather and ok_row_copy and ok_row_copy_fusion and ok_state and ok_fusion
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
