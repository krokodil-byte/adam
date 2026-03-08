#!/usr/bin/env python3
"""
ADAMAH Benchmark: Mixed Non-Linear Operations

Tests realistic workloads with:
- Parallel branches (independent ops at same level)
- Sequential dependencies (chained ops)
- Mixed operation types (matmul, softmax, layernorm, elementwise)
- Non-linear activation chains

Compares ADAMAH vs PyTorch vs CuPy with IDENTICAL logical structure.
"""

import os
import sys
import time
import numpy as np

# Setup paths FIRST
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Initialize ADAMAH BEFORE torch/cupy to get priority on GPU memory
import adamah
print("[INIT] Initializing ADAMAH first to secure GPU memory...")
_adamah_gpu = adamah.Adamah()
print("[INIT] ADAMAH initialized successfully")

# Now import torch/cupy (they will get remaining GPU memory)
torch = None
_TORCH_AVAILABLE = False
try:
    import torch as _torch
    torch = _torch
    _TORCH_AVAILABLE = torch.cuda.is_available()
    if _TORCH_AVAILABLE:
        # Limit PyTorch memory to avoid starving ADAMAH
        torch.cuda.set_per_process_memory_fraction(0.5)  # Max 50% of remaining VRAM
        print(f"[INIT] PyTorch CUDA available, limited to 50% VRAM")
except Exception as e:
    print(f"[INIT] PyTorch not available: {e}")

cp = None
_CUPY_AVAILABLE = False
try:
    import cupy as _cp
    cp = _cp
    _CUPY_AVAILABLE = True
    # CuPy will use remaining memory
    print(f"[INIT] CuPy available")
except Exception as e:
    print(f"[INIT] CuPy not available: {e}")

# ============================================
# Configuration
# ============================================
SEED = 42
SEQ_LEN = 128      # Sequence length
D_MODEL = 256      # Model dimension
D_FF = D_MODEL * 4 # FFN hidden dim

ITERATIONS = [10, 25, 50, 100, 200]

def now_ms():
    return time.perf_counter() * 1000.0

# ============================================
# Block 1: Attention-FFN Hybrid
# ============================================
# Structure:
#   Branch A (attention-like):
#     Q = matmul(X, Wq)
#     K = matmul(X, Wk)  [parallel with Q]
#     scores = matmul(Q, K.T)
#     attn = softmax(scores)
#   Branch B (FFN-like, parallel with A):
#     H = relu(matmul(X, W1))
#     ffn_out = matmul(H, W2)
#   Merge:
#     combined = add(attn_out, ffn_out)  [waits for both branches]
#     normed = layernorm(combined)
#     result = tanh(normed)

def create_block1_adamah(gpu, u, seq, d, d_ff):
    """Create attention-FFN hybrid block for ADAMAH"""
    rng = np.random.default_rng(SEED)
    
    # Allocate map regions
    map_id = 5
    base = 0
    
    # Input X: (seq, d)
    x_base = base
    x_size = seq * d
    base += x_size
    
    # Attention weights
    wq_base = base; base += d * d
    wk_base = base; base += d * d
    
    # FFN weights  
    w1_base = base; base += d * d_ff
    w2_base = base; base += d_ff * d
    
    # Intermediates
    q_base = base; base += seq * d
    k_base = base; base += seq * d
    scores_base = base; base += seq * seq
    attn_base = base; base += seq * seq
    h_base = base; base += seq * d_ff
    ffn_out_base = base; base += seq * d
    combined_base = base; base += seq * d
    normed_base = base; base += seq * d
    result_base = base; base += seq * d
    
    # LayerNorm params
    gamma_base = base; base += d
    beta_base = base; base += d
    
    total_floats = base
    
    # Create map (ignore destroy error if doesn't exist)
    try:
        gpu.map_destroy(map_id)
    except:
        pass
    gpu.map_create(map_id, 4, 1, total_floats)
    
    # Initialize data
    x_data = rng.standard_normal((seq, d)).astype(np.float32).flatten()
    wq_data = (rng.standard_normal((d, d)) * 0.02).astype(np.float32).flatten()
    wk_data = (rng.standard_normal((d, d)) * 0.02).astype(np.float32).flatten()
    w1_data = (rng.standard_normal((d, d_ff)) * 0.02).astype(np.float32).flatten()
    w2_data = (rng.standard_normal((d_ff, d)) * 0.02).astype(np.float32).flatten()
    gamma_data = np.ones(d, dtype=np.float32)
    beta_data = np.zeros(d, dtype=np.float32)
    
    gpu.map_scatter(map_id, np.arange(x_size, dtype=np.uint32) + x_base, x_data)
    gpu.map_scatter(map_id, np.arange(d*d, dtype=np.uint32) + wq_base, wq_data)
    gpu.map_scatter(map_id, np.arange(d*d, dtype=np.uint32) + wk_base, wk_data)
    gpu.map_scatter(map_id, np.arange(d*d_ff, dtype=np.uint32) + w1_base, w1_data)
    gpu.map_scatter(map_id, np.arange(d_ff*d, dtype=np.uint32) + w2_base, w2_data)
    gpu.map_scatter(map_id, np.arange(d, dtype=np.uint32) + gamma_base, gamma_data)
    gpu.map_scatter(map_id, np.arange(d, dtype=np.uint32) + beta_base, beta_data)
    
    # Cache locations for matmul (single op each)
    locs_x = u.cache_locs(map_id, np.array([x_base], dtype=np.uint32))
    locs_wq = u.cache_locs(map_id, np.array([wq_base], dtype=np.uint32))
    locs_wk = u.cache_locs(map_id, np.array([wk_base], dtype=np.uint32))
    locs_w1 = u.cache_locs(map_id, np.array([w1_base], dtype=np.uint32))
    locs_w2 = u.cache_locs(map_id, np.array([w2_base], dtype=np.uint32))
    locs_q = u.cache_locs(map_id, np.array([q_base], dtype=np.uint32))
    locs_k = u.cache_locs(map_id, np.array([k_base], dtype=np.uint32))
    locs_scores = u.cache_locs(map_id, np.array([scores_base], dtype=np.uint32))
    locs_attn = u.cache_locs(map_id, np.array([attn_base], dtype=np.uint32))
    locs_h = u.cache_locs(map_id, np.array([h_base], dtype=np.uint32))
    locs_ffn = u.cache_locs(map_id, np.array([ffn_out_base], dtype=np.uint32))
    locs_combined = u.cache_locs(map_id, np.array([combined_base], dtype=np.uint32))
    locs_normed = u.cache_locs(map_id, np.array([normed_base], dtype=np.uint32))
    locs_result = u.cache_locs(map_id, np.array([result_base], dtype=np.uint32))
    
    # For elementwise ops (all elements)
    locs_h_all = u.cache_locs(map_id, np.arange(seq * d_ff, dtype=np.uint32) + h_base)
    locs_attn_all = u.cache_locs(map_id, np.arange(seq * seq, dtype=np.uint32) + attn_base)
    locs_ffn_all = u.cache_locs(map_id, np.arange(seq * d, dtype=np.uint32) + ffn_out_base)
    locs_combined_all = u.cache_locs(map_id, np.arange(seq * d, dtype=np.uint32) + combined_base)
    locs_normed_all = u.cache_locs(map_id, np.arange(seq * d, dtype=np.uint32) + normed_base)
    locs_result_all = u.cache_locs(map_id, np.arange(seq * d, dtype=np.uint32) + result_base)
    
    # For softmax (per-row)
    locs_scores_rows = u.cache_locs(map_id, np.arange(seq, dtype=np.uint32) * seq + scores_base)
    locs_attn_rows = u.cache_locs(map_id, np.arange(seq, dtype=np.uint32) * seq + attn_base)
    
    # For layernorm (per-row)
    locs_combined_rows = u.cache_locs(map_id, np.arange(seq, dtype=np.uint32) * d + combined_base)
    locs_normed_rows = u.cache_locs(map_id, np.arange(seq, dtype=np.uint32) * d + normed_base)
    locs_gamma_rows = u.cache_locs(map_id, np.arange(seq, dtype=np.uint32) * 0 + gamma_base)  # broadcast
    locs_beta_rows = u.cache_locs(map_id, np.arange(seq, dtype=np.uint32) * 0 + beta_base)
    
    def run_block():
        # === Level 0: Independent ops (parallel) ===
        # Branch A: Q = X @ Wq
        u.mop2("MATMUL", map_id, map_id, map_id, extra={
            "locs_a": locs_x, "locs_b": locs_wq, "locs_c": locs_q,
            "M": seq, "K": d, "N": d
        })
        # Branch A: K = X @ Wk (parallel with Q)
        u.mop2("MATMUL", map_id, map_id, map_id, extra={
            "locs_a": locs_x, "locs_b": locs_wk, "locs_c": locs_k,
            "M": seq, "K": d, "N": d
        })
        # Branch B: H = X @ W1 (parallel with Q, K)
        u.mop2("MATMUL", map_id, map_id, map_id, extra={
            "locs_a": locs_x, "locs_b": locs_w1, "locs_c": locs_h,
            "M": seq, "K": d, "N": d_ff
        })
        
        # === Level 1: Depends on level 0 ===
        # Branch A: scores = Q @ K.T
        u.mop2("MATMUL", map_id, map_id, map_id, extra={
            "locs_a": locs_q, "locs_b": locs_k, "locs_c": locs_scores,
            "M": seq, "K": d, "N": seq
        })
        # Branch B: H = relu(H)
        u.mop1("RELU", map_id, map_id, locs_src=locs_h_all, locs_dst=locs_h_all)
        
        # === Level 2: Depends on level 1 ===
        # Branch A: attn = softmax(scores)
        u.mop1("SOFTMAX", map_id, map_id, locs_src=locs_scores_rows, locs_dst=locs_attn_rows,
               extra={"row_size": seq})
        # Branch B: ffn_out = H @ W2
        u.mop2("MATMUL", map_id, map_id, map_id, extra={
            "locs_a": locs_h, "locs_b": locs_w2, "locs_c": locs_ffn,
            "M": seq, "K": d_ff, "N": d
        })
        
        # === Level 3: Merge branches ===
        # combined = attn + ffn_out (element-wise on flattened)
        # Note: attn is seq*seq, ffn is seq*d - we use ffn size
        u.mop2("ADD", map_id, map_id, map_id,
               locs_a=locs_ffn_all, locs_b=locs_ffn_all, locs_dst=locs_combined_all)
        
        # === Level 4: Final processing ===
        # normed = layernorm(combined)
        u.mop1("LAYERNORM", map_id, map_id, locs_src=locs_combined_rows, locs_dst=locs_normed_rows,
               extra={"dim": d, "eps": 1e-5, "locs_gamma": locs_gamma_rows, "locs_beta": locs_beta_rows})
        
        # === Level 5: Non-linear chain ===
        # result = tanh(normed)
        u.mop1("TANH", map_id, map_id, locs_src=locs_normed_all, locs_dst=locs_result_all)
    
    return run_block


def create_block1_pytorch(seq, d, d_ff, device):
    """Create identical attention-FFN hybrid block for PyTorch"""
    rng = np.random.default_rng(SEED)
    
    # Create tensors (same initialization as ADAMAH)
    X = torch.from_numpy(rng.standard_normal((seq, d)).astype(np.float32)).to(device)
    Wq = torch.from_numpy((rng.standard_normal((d, d)) * 0.02).astype(np.float32)).to(device)
    Wk = torch.from_numpy((rng.standard_normal((d, d)) * 0.02).astype(np.float32)).to(device)
    W1 = torch.from_numpy((rng.standard_normal((d, d_ff)) * 0.02).astype(np.float32)).to(device)
    W2 = torch.from_numpy((rng.standard_normal((d_ff, d)) * 0.02).astype(np.float32)).to(device)
    gamma = torch.ones(d, device=device)
    beta = torch.zeros(d, device=device)
    
    # Pre-allocate outputs
    Q = torch.empty((seq, d), device=device)
    K = torch.empty((seq, d), device=device)
    H = torch.empty((seq, d_ff), device=device)
    scores = torch.empty((seq, seq), device=device)
    attn = torch.empty((seq, seq), device=device)
    ffn_out = torch.empty((seq, d), device=device)
    combined = torch.empty((seq, d), device=device)
    normed = torch.empty((seq, d), device=device)
    result = torch.empty((seq, d), device=device)
    
    def run_block():
        # === Level 0: Independent ops (parallel) ===
        torch.mm(X, Wq, out=Q)
        torch.mm(X, Wk, out=K)
        torch.mm(X, W1, out=H)
        
        # === Level 1: Depends on level 0 ===
        torch.mm(Q, K.T, out=scores)
        H_relu = torch.relu(H)
        
        # === Level 2: Depends on level 1 ===
        attn = torch.softmax(scores, dim=-1)
        torch.mm(H_relu, W2, out=ffn_out)
        
        # === Level 3: Merge branches ===
        torch.add(ffn_out, ffn_out, out=combined)
        
        # === Level 4: LayerNorm ===
        mean = combined.mean(dim=-1, keepdim=True)
        var = combined.var(dim=-1, keepdim=True, unbiased=False)
        normed = (combined - mean) / torch.sqrt(var + 1e-5) * gamma + beta
        
        # === Level 5: Non-linear chain ===
        result = torch.tanh(normed)
        
        return result
    
    return run_block


def create_block1_cupy(seq, d, d_ff):
    """Create identical attention-FFN hybrid block for CuPy"""
    rng = np.random.default_rng(SEED)
    
    # Create arrays (same initialization as ADAMAH)
    X = cp.asarray(rng.standard_normal((seq, d)).astype(np.float32))
    Wq = cp.asarray((rng.standard_normal((d, d)) * 0.02).astype(np.float32))
    Wk = cp.asarray((rng.standard_normal((d, d)) * 0.02).astype(np.float32))
    W1 = cp.asarray((rng.standard_normal((d, d_ff)) * 0.02).astype(np.float32))
    W2 = cp.asarray((rng.standard_normal((d_ff, d)) * 0.02).astype(np.float32))
    gamma = cp.ones(d, dtype=cp.float32)
    beta = cp.zeros(d, dtype=cp.float32)
    
    def run_block():
        # === Level 0: Independent ops (parallel) ===
        Q = cp.matmul(X, Wq)
        K = cp.matmul(X, Wk)
        H = cp.matmul(X, W1)
        
        # === Level 1: Depends on level 0 ===
        scores = cp.matmul(Q, K.T)
        H_relu = cp.maximum(H, 0)  # relu
        
        # === Level 2: Depends on level 1 ===
        # Softmax
        scores_max = cp.max(scores, axis=-1, keepdims=True)
        scores_exp = cp.exp(scores - scores_max)
        attn = scores_exp / cp.sum(scores_exp, axis=-1, keepdims=True)
        ffn_out = cp.matmul(H_relu, W2)
        
        # === Level 3: Merge branches ===
        combined = ffn_out + ffn_out
        
        # === Level 4: LayerNorm ===
        mean = cp.mean(combined, axis=-1, keepdims=True)
        var = cp.var(combined, axis=-1, keepdims=True)
        normed = (combined - mean) / cp.sqrt(var + 1e-5) * gamma + beta
        
        # === Level 5: Non-linear chain ===
        result = cp.tanh(normed)
        
        return result
    
    return run_block


# ============================================
# Block 2: Deep Residual Chain
# ============================================
# Structure:
#   x1 = relu(x + matmul(x, W1))
#   x2 = tanh(x1 + matmul(x1, W2))
#   x3 = tanh(x2 + matmul(x2, W3))
#   result = layernorm(x3)

def create_block2_adamah(gpu, u, seq, d):
    """Create deep residual chain for ADAMAH"""
    rng = np.random.default_rng(SEED + 1)
    
    map_id = 6
    base = 0
    
    x_base = base; base += seq * d
    w1_base = base; base += d * d
    w2_base = base; base += d * d
    w3_base = base; base += d * d
    t1_base = base; base += seq * d  # matmul output
    t2_base = base; base += seq * d
    t3_base = base; base += seq * d
    x1_base = base; base += seq * d
    x2_base = base; base += seq * d
    x3_base = base; base += seq * d
    result_base = base; base += seq * d
    gamma_base = base; base += d
    beta_base = base; base += d
    
    total_floats = base
    
    try:
        gpu.map_destroy(map_id)
    except:
        pass
    gpu.map_create(map_id, 4, 1, total_floats)
    
    # Initialize
    x_data = rng.standard_normal((seq, d)).astype(np.float32).flatten()
    w1_data = (rng.standard_normal((d, d)) * 0.02).astype(np.float32).flatten()
    w2_data = (rng.standard_normal((d, d)) * 0.02).astype(np.float32).flatten()
    w3_data = (rng.standard_normal((d, d)) * 0.02).astype(np.float32).flatten()
    gamma_data = np.ones(d, dtype=np.float32)
    beta_data = np.zeros(d, dtype=np.float32)
    
    gpu.map_scatter(map_id, np.arange(seq*d, dtype=np.uint32) + x_base, x_data)
    gpu.map_scatter(map_id, np.arange(d*d, dtype=np.uint32) + w1_base, w1_data)
    gpu.map_scatter(map_id, np.arange(d*d, dtype=np.uint32) + w2_base, w2_data)
    gpu.map_scatter(map_id, np.arange(d*d, dtype=np.uint32) + w3_base, w3_data)
    gpu.map_scatter(map_id, np.arange(d, dtype=np.uint32) + gamma_base, gamma_data)
    gpu.map_scatter(map_id, np.arange(d, dtype=np.uint32) + beta_base, beta_data)
    
    # Cache locs
    locs_x = u.cache_locs(map_id, np.array([x_base], dtype=np.uint32))
    locs_w1 = u.cache_locs(map_id, np.array([w1_base], dtype=np.uint32))
    locs_w2 = u.cache_locs(map_id, np.array([w2_base], dtype=np.uint32))
    locs_w3 = u.cache_locs(map_id, np.array([w3_base], dtype=np.uint32))
    locs_t1 = u.cache_locs(map_id, np.array([t1_base], dtype=np.uint32))
    locs_t2 = u.cache_locs(map_id, np.array([t2_base], dtype=np.uint32))
    locs_t3 = u.cache_locs(map_id, np.array([t3_base], dtype=np.uint32))
    locs_x1 = u.cache_locs(map_id, np.array([x1_base], dtype=np.uint32))
    locs_x2 = u.cache_locs(map_id, np.array([x2_base], dtype=np.uint32))
    locs_x3 = u.cache_locs(map_id, np.array([x3_base], dtype=np.uint32))
    locs_result = u.cache_locs(map_id, np.array([result_base], dtype=np.uint32))
    
    locs_x_all = u.cache_locs(map_id, np.arange(seq*d, dtype=np.uint32) + x_base)
    locs_t1_all = u.cache_locs(map_id, np.arange(seq*d, dtype=np.uint32) + t1_base)
    locs_t2_all = u.cache_locs(map_id, np.arange(seq*d, dtype=np.uint32) + t2_base)
    locs_t3_all = u.cache_locs(map_id, np.arange(seq*d, dtype=np.uint32) + t3_base)
    locs_x1_all = u.cache_locs(map_id, np.arange(seq*d, dtype=np.uint32) + x1_base)
    locs_x2_all = u.cache_locs(map_id, np.arange(seq*d, dtype=np.uint32) + x2_base)
    locs_x3_all = u.cache_locs(map_id, np.arange(seq*d, dtype=np.uint32) + x3_base)
    locs_result_all = u.cache_locs(map_id, np.arange(seq*d, dtype=np.uint32) + result_base)
    
    locs_x3_rows = u.cache_locs(map_id, np.arange(seq, dtype=np.uint32) * d + x3_base)
    locs_result_rows = u.cache_locs(map_id, np.arange(seq, dtype=np.uint32) * d + result_base)
    locs_gamma_rows = u.cache_locs(map_id, np.arange(seq, dtype=np.uint32) * 0 + gamma_base)
    locs_beta_rows = u.cache_locs(map_id, np.arange(seq, dtype=np.uint32) * 0 + beta_base)
    
    def run_block():
        # t1 = matmul(x, W1)
        u.mop2("MATMUL", map_id, map_id, map_id, extra={
            "locs_a": locs_x, "locs_b": locs_w1, "locs_c": locs_t1,
            "M": seq, "K": d, "N": d
        })
        # x1 = relu(x + t1)
        u.mop2("ADD", map_id, map_id, map_id, locs_a=locs_x_all, locs_b=locs_t1_all, locs_dst=locs_x1_all)
        u.mop1("RELU", map_id, map_id, locs_src=locs_x1_all, locs_dst=locs_x1_all)
        
        # t2 = matmul(x1, W2)
        u.mop2("MATMUL", map_id, map_id, map_id, extra={
            "locs_a": locs_x1, "locs_b": locs_w2, "locs_c": locs_t2,
            "M": seq, "K": d, "N": d
        })
        # x2 = tanh(x1 + t2)
        u.mop2("ADD", map_id, map_id, map_id, locs_a=locs_x1_all, locs_b=locs_t2_all, locs_dst=locs_x2_all)
        u.mop1("TANH", map_id, map_id, locs_src=locs_x2_all, locs_dst=locs_x2_all)
        
        # t3 = matmul(x2, W3)
        u.mop2("MATMUL", map_id, map_id, map_id, extra={
            "locs_a": locs_x2, "locs_b": locs_w3, "locs_c": locs_t3,
            "M": seq, "K": d, "N": d
        })
        # x3 = tanh(x2 + t3)
        u.mop2("ADD", map_id, map_id, map_id, locs_a=locs_x2_all, locs_b=locs_t3_all, locs_dst=locs_x3_all)
        u.mop1("TANH", map_id, map_id, locs_src=locs_x3_all, locs_dst=locs_x3_all)
        
        # result = layernorm(x3)
        u.mop1("LAYERNORM", map_id, map_id, locs_src=locs_x3_rows, locs_dst=locs_result_rows,
               extra={"dim": d, "eps": 1e-5, "locs_gamma": locs_gamma_rows, "locs_beta": locs_beta_rows})
    
    return run_block


def create_block2_pytorch(seq, d, device):
    """Create identical deep residual chain for PyTorch"""
    rng = np.random.default_rng(SEED + 1)
    
    X = torch.from_numpy(rng.standard_normal((seq, d)).astype(np.float32)).to(device)
    W1 = torch.from_numpy((rng.standard_normal((d, d)) * 0.02).astype(np.float32)).to(device)
    W2 = torch.from_numpy((rng.standard_normal((d, d)) * 0.02).astype(np.float32)).to(device)
    W3 = torch.from_numpy((rng.standard_normal((d, d)) * 0.02).astype(np.float32)).to(device)
    gamma = torch.ones(d, device=device)
    beta = torch.zeros(d, device=device)
    
    def run_block():
        t1 = torch.mm(X, W1)
        x1 = torch.relu(X + t1)
        
        t2 = torch.mm(x1, W2)
        x2 = torch.tanh(x1 + t2)
        
        t3 = torch.mm(x2, W3)
        x3 = torch.tanh(x2 + t3)
        
        mean = x3.mean(dim=-1, keepdim=True)
        var = x3.var(dim=-1, keepdim=True, unbiased=False)
        result = (x3 - mean) / torch.sqrt(var + 1e-5) * gamma + beta
        
        return result
    
    return run_block


def create_block2_cupy(seq, d):
    """Create identical deep residual chain for CuPy"""
    rng = np.random.default_rng(SEED + 1)
    
    X = cp.asarray(rng.standard_normal((seq, d)).astype(np.float32))
    W1 = cp.asarray((rng.standard_normal((d, d)) * 0.02).astype(np.float32))
    W2 = cp.asarray((rng.standard_normal((d, d)) * 0.02).astype(np.float32))
    W3 = cp.asarray((rng.standard_normal((d, d)) * 0.02).astype(np.float32))
    gamma = cp.ones(d, dtype=cp.float32)
    beta = cp.zeros(d, dtype=cp.float32)
    
    def run_block():
        t1 = cp.matmul(X, W1)
        x1 = cp.maximum(X + t1, 0)  # relu
        
        t2 = cp.matmul(x1, W2)
        x2 = cp.tanh(x1 + t2)
        
        t3 = cp.matmul(x2, W3)
        x3 = cp.tanh(x2 + t3)  # tanh
        
        mean = cp.mean(x3, axis=-1, keepdims=True)
        var = cp.var(x3, axis=-1, keepdims=True)
        result = (x3 - mean) / cp.sqrt(var + 1e-5) * gamma + beta
        
        return result
    
    return run_block


# ============================================
# Block 3: Multi-Head Attention (simplified)
# ============================================
def create_block3_adamah(gpu, u, seq, d, n_heads):
    """Create multi-head attention for ADAMAH"""
    rng = np.random.default_rng(SEED + 2)
    head_dim = d // n_heads
    
    map_id = 7
    base = 0
    
    x_base = base; base += seq * d
    wq_base = base; base += d * d
    wk_base = base; base += d * d
    wv_base = base; base += d * d
    wo_base = base; base += d * d
    q_base = base; base += seq * d
    k_base = base; base += seq * d
    v_base = base; base += seq * d
    scores_base = base; base += n_heads * seq * seq
    attn_base = base; base += n_heads * seq * seq
    context_base = base; base += seq * d
    out_base = base; base += seq * d
    scale_base = base; base += n_heads * seq * seq  # Allocate scale here
    
    total_floats = base
    
    try:
        gpu.map_destroy(map_id)
    except:
        pass
    gpu.map_create(map_id, 4, 1, total_floats)
    
    # Initialize
    x_data = rng.standard_normal((seq, d)).astype(np.float32).flatten()
    wq_data = (rng.standard_normal((d, d)) * 0.02).astype(np.float32).flatten()
    wk_data = (rng.standard_normal((d, d)) * 0.02).astype(np.float32).flatten()
    wv_data = (rng.standard_normal((d, d)) * 0.02).astype(np.float32).flatten()
    wo_data = (rng.standard_normal((d, d)) * 0.02).astype(np.float32).flatten()
    scale = 1.0 / np.sqrt(head_dim)
    scale_arr = np.full(n_heads * seq * seq, scale, dtype=np.float32)
    
    gpu.map_scatter(map_id, np.arange(seq*d, dtype=np.uint32) + x_base, x_data)
    gpu.map_scatter(map_id, np.arange(d*d, dtype=np.uint32) + wq_base, wq_data)
    gpu.map_scatter(map_id, np.arange(d*d, dtype=np.uint32) + wk_base, wk_data)
    gpu.map_scatter(map_id, np.arange(d*d, dtype=np.uint32) + wv_base, wv_data)
    gpu.map_scatter(map_id, np.arange(d*d, dtype=np.uint32) + wo_base, wo_data)
    gpu.map_scatter(map_id, np.arange(len(scale_arr), dtype=np.uint32) + scale_base, scale_arr)
    
    # Cache locs
    locs_x = u.cache_locs(map_id, np.array([x_base], dtype=np.uint32))
    locs_wq = u.cache_locs(map_id, np.array([wq_base], dtype=np.uint32))
    locs_wk = u.cache_locs(map_id, np.array([wk_base], dtype=np.uint32))
    locs_wv = u.cache_locs(map_id, np.array([wv_base], dtype=np.uint32))
    locs_wo = u.cache_locs(map_id, np.array([wo_base], dtype=np.uint32))
    locs_q = u.cache_locs(map_id, np.array([q_base], dtype=np.uint32))
    locs_k = u.cache_locs(map_id, np.array([k_base], dtype=np.uint32))
    locs_v = u.cache_locs(map_id, np.array([v_base], dtype=np.uint32))
    locs_scores = u.cache_locs(map_id, np.array([scores_base], dtype=np.uint32))
    locs_attn = u.cache_locs(map_id, np.array([attn_base], dtype=np.uint32))
    locs_context = u.cache_locs(map_id, np.array([context_base], dtype=np.uint32))
    locs_out = u.cache_locs(map_id, np.array([out_base], dtype=np.uint32))
    
    # Softmax rows (n_heads * seq rows)
    locs_scores_rows = u.cache_locs(map_id, np.arange(n_heads * seq, dtype=np.uint32) * seq + scores_base)
    locs_attn_rows = u.cache_locs(map_id, np.arange(n_heads * seq, dtype=np.uint32) * seq + attn_base)
    
    locs_scale_all = u.cache_locs(map_id, np.arange(n_heads * seq * seq, dtype=np.uint32) + scale_base)
    locs_scores_all = u.cache_locs(map_id, np.arange(n_heads * seq * seq, dtype=np.uint32) + scores_base)
    
    def run_block():
        # Q, K, V projections (parallel)
        u.mop2("MATMUL", map_id, map_id, map_id, extra={
            "locs_a": locs_x, "locs_b": locs_wq, "locs_c": locs_q, "M": seq, "K": d, "N": d})
        u.mop2("MATMUL", map_id, map_id, map_id, extra={
            "locs_a": locs_x, "locs_b": locs_wk, "locs_c": locs_k, "M": seq, "K": d, "N": d})
        u.mop2("MATMUL", map_id, map_id, map_id, extra={
            "locs_a": locs_x, "locs_b": locs_wv, "locs_c": locs_v, "M": seq, "K": d, "N": d})
        
        # Attention scores (simplified: single head for now)
        u.mop2("MATMUL", map_id, map_id, map_id, extra={
            "locs_a": locs_q, "locs_b": locs_k, "locs_c": locs_scores, "M": seq, "K": d, "N": seq})
        
        # Scale
        u.mop2("MUL", map_id, map_id, map_id, locs_a=locs_scores_all, locs_b=locs_scale_all, locs_dst=locs_scores_all)
        
        # Softmax
        u.mop1("SOFTMAX", map_id, map_id, locs_src=locs_scores_rows, locs_dst=locs_attn_rows,
               extra={"row_size": seq})
        
        # Context
        u.mop2("MATMUL", map_id, map_id, map_id, extra={
            "locs_a": locs_attn, "locs_b": locs_v, "locs_c": locs_context, "M": seq, "K": seq, "N": d})
        
        # Output projection
        u.mop2("MATMUL", map_id, map_id, map_id, extra={
            "locs_a": locs_context, "locs_b": locs_wo, "locs_c": locs_out, "M": seq, "K": d, "N": d})
    
    return run_block


def create_block3_pytorch(seq, d, n_heads, device):
    """Create identical multi-head attention for PyTorch"""
    rng = np.random.default_rng(SEED + 2)
    head_dim = d // n_heads
    scale = 1.0 / np.sqrt(head_dim)
    
    X = torch.from_numpy(rng.standard_normal((seq, d)).astype(np.float32)).to(device)
    Wq = torch.from_numpy((rng.standard_normal((d, d)) * 0.02).astype(np.float32)).to(device)
    Wk = torch.from_numpy((rng.standard_normal((d, d)) * 0.02).astype(np.float32)).to(device)
    Wv = torch.from_numpy((rng.standard_normal((d, d)) * 0.02).astype(np.float32)).to(device)
    Wo = torch.from_numpy((rng.standard_normal((d, d)) * 0.02).astype(np.float32)).to(device)
    
    def run_block():
        Q = torch.mm(X, Wq)
        K = torch.mm(X, Wk)
        V = torch.mm(X, Wv)
        
        scores = torch.mm(Q, K.T) * scale
        attn = torch.softmax(scores, dim=-1)
        context = torch.mm(attn, V)
        out = torch.mm(context, Wo)
        
        return out
    
    return run_block


def create_block3_cupy(seq, d, n_heads):
    """Create identical multi-head attention for CuPy"""
    rng = np.random.default_rng(SEED + 2)
    head_dim = d // n_heads
    scale = 1.0 / np.sqrt(head_dim)
    
    X = cp.asarray(rng.standard_normal((seq, d)).astype(np.float32))
    Wq = cp.asarray((rng.standard_normal((d, d)) * 0.02).astype(np.float32))
    Wk = cp.asarray((rng.standard_normal((d, d)) * 0.02).astype(np.float32))
    Wv = cp.asarray((rng.standard_normal((d, d)) * 0.02).astype(np.float32))
    Wo = cp.asarray((rng.standard_normal((d, d)) * 0.02).astype(np.float32))
    
    def run_block():
        Q = cp.matmul(X, Wq)
        K = cp.matmul(X, Wk)
        V = cp.matmul(X, Wv)
        
        scores = cp.matmul(Q, K.T) * scale
        scores_max = cp.max(scores, axis=-1, keepdims=True)
        scores_exp = cp.exp(scores - scores_max)
        attn = scores_exp / cp.sum(scores_exp, axis=-1, keepdims=True)
        context = cp.matmul(attn, V)
        out = cp.matmul(context, Wo)
        
        return out
    
    return run_block


# ============================================
# Main Benchmark
# ============================================
def main():
    global _adamah_gpu
    
    print("=" * 100)
    print("ADAMAH Benchmark: Mixed Non-Linear Operations")
    print("=" * 100)
    print(f"Config: SEQ_LEN={SEQ_LEN}, D_MODEL={D_MODEL}, D_FF={D_FF}")
    print(f"PyTorch available: {_TORCH_AVAILABLE}")
    print(f"CuPy available: {_CUPY_AVAILABLE}")
    print()
    
    # Use already initialized ADAMAH instance
    gpu = _adamah_gpu
    u = gpu.uucis  # Use property accessor
    
    results = []
    
    # ========================================
    # Block 1: Attention-FFN Hybrid
    # ========================================
    print("\n" + "=" * 100)
    print("BLOCK 1: Attention-FFN Hybrid (parallel branches + merge)")
    print("=" * 100)
    
    block1_adamah = create_block1_adamah(gpu, u, SEQ_LEN, D_MODEL, D_FF)
    
    if _TORCH_AVAILABLE:
        device = torch.device("cuda")
        block1_torch = create_block1_pytorch(SEQ_LEN, D_MODEL, D_FF, device)
    
    if _CUPY_AVAILABLE:
        block1_cupy = create_block1_cupy(SEQ_LEN, D_MODEL, D_FF)
    
    print(f"\n{'Iterations':<12} {'ADAMAH (ms)':<15} {'PyTorch (ms)':<15} {'CuPy (ms)':<15} {'vs Torch':<12} {'vs CuPy':<12}")
    print("-" * 85)
    
    for n_iter in ITERATIONS:
        # Warmup
        for _ in range(3):
            block1_adamah()
        u.sync()  # lightweight sync - no data transfer
        
        # ADAMAH
        t0 = now_ms()
        for _ in range(n_iter):
            block1_adamah()
        u.sync()  # lightweight sync
        adamah_time = (now_ms() - t0) / n_iter
        
        # PyTorch
        torch_time = float('inf')
        if _TORCH_AVAILABLE:
            for _ in range(3):
                block1_torch()
            torch.cuda.synchronize()
            
            t0 = now_ms()
            for _ in range(n_iter):
                block1_torch()
            torch.cuda.synchronize()
            torch_time = (now_ms() - t0) / n_iter
        
        # CuPy
        cupy_time = float('inf')
        if _CUPY_AVAILABLE:
            for _ in range(3):
                block1_cupy()
            cp.cuda.Stream.null.synchronize()
            
            t0 = now_ms()
            for _ in range(n_iter):
                block1_cupy()
            cp.cuda.Stream.null.synchronize()
            cupy_time = (now_ms() - t0) / n_iter
        
        vs_torch = torch_time / adamah_time if adamah_time > 0 else 0
        vs_cupy = cupy_time / adamah_time if adamah_time > 0 else 0
        
        print(f"{n_iter:<12} {adamah_time:<15.4f} {torch_time:<15.4f} {cupy_time:<15.4f} {vs_torch:<12.2f}x {vs_cupy:<12.2f}x")
        
        results.append({"block": "Attn-FFN Hybrid", "n": n_iter, "adamah": adamah_time, "torch": torch_time, "cupy": cupy_time})
    
    # ========================================
    # Block 2: Deep Residual Chain
    # ========================================
    print("\n" + "=" * 100)
    print("BLOCK 2: Deep Residual Chain (sequential dependencies)")
    print("=" * 100)
    
    block2_adamah = create_block2_adamah(gpu, u, SEQ_LEN, D_MODEL)
    
    if _TORCH_AVAILABLE:
        block2_torch = create_block2_pytorch(SEQ_LEN, D_MODEL, device)
    
    if _CUPY_AVAILABLE:
        block2_cupy = create_block2_cupy(SEQ_LEN, D_MODEL)
    
    print(f"\n{'Iterations':<12} {'ADAMAH (ms)':<15} {'PyTorch (ms)':<15} {'CuPy (ms)':<15} {'vs Torch':<12} {'vs CuPy':<12}")
    print("-" * 85)
    
    for n_iter in ITERATIONS:
        # Warmup
        for _ in range(3):
            block2_adamah()
        u.sync()
        
        # ADAMAH
        t0 = now_ms()
        for _ in range(n_iter):
            block2_adamah()
        u.sync()
        adamah_time = (now_ms() - t0) / n_iter
        
        # PyTorch
        torch_time = float('inf')
        if _TORCH_AVAILABLE:
            for _ in range(3):
                block2_torch()
            torch.cuda.synchronize()
            
            t0 = now_ms()
            for _ in range(n_iter):
                block2_torch()
            torch.cuda.synchronize()
            torch_time = (now_ms() - t0) / n_iter
        
        # CuPy
        cupy_time = float('inf')
        if _CUPY_AVAILABLE:
            for _ in range(3):
                block2_cupy()
            cp.cuda.Stream.null.synchronize()
            
            t0 = now_ms()
            for _ in range(n_iter):
                block2_cupy()
            cp.cuda.Stream.null.synchronize()
            cupy_time = (now_ms() - t0) / n_iter
        
        vs_torch = torch_time / adamah_time if adamah_time > 0 else 0
        vs_cupy = cupy_time / adamah_time if adamah_time > 0 else 0
        
        print(f"{n_iter:<12} {adamah_time:<15.4f} {torch_time:<15.4f} {cupy_time:<15.4f} {vs_torch:<12.2f}x {vs_cupy:<12.2f}x")
        
        results.append({"block": "Residual Chain", "n": n_iter, "adamah": adamah_time, "torch": torch_time, "cupy": cupy_time})
    
    # ========================================
    # Block 3: Multi-Head Attention
    # ========================================
    print("\n" + "=" * 100)
    print("BLOCK 3: Multi-Head Attention (Q/K/V parallel + sequential)")
    print("=" * 100)
    
    N_HEADS = 8
    block3_adamah = create_block3_adamah(gpu, u, SEQ_LEN, D_MODEL, N_HEADS)
    
    if _TORCH_AVAILABLE:
        block3_torch = create_block3_pytorch(SEQ_LEN, D_MODEL, N_HEADS, device)
    
    if _CUPY_AVAILABLE:
        block3_cupy = create_block3_cupy(SEQ_LEN, D_MODEL, N_HEADS)
    
    print(f"\n{'Iterations':<12} {'ADAMAH (ms)':<15} {'PyTorch (ms)':<15} {'CuPy (ms)':<15} {'vs Torch':<12} {'vs CuPy':<12}")
    print("-" * 85)
    
    for n_iter in ITERATIONS:
        # Warmup
        for _ in range(3):
            block3_adamah()
        u.sync()
        
        # ADAMAH
        t0 = now_ms()
        for _ in range(n_iter):
            block3_adamah()
        u.sync()
        adamah_time = (now_ms() - t0) / n_iter
        
        # PyTorch
        torch_time = float('inf')
        if _TORCH_AVAILABLE:
            for _ in range(3):
                block3_torch()
            torch.cuda.synchronize()
            
            t0 = now_ms()
            for _ in range(n_iter):
                block3_torch()
            torch.cuda.synchronize()
            torch_time = (now_ms() - t0) / n_iter
        
        # CuPy
        cupy_time = float('inf')
        if _CUPY_AVAILABLE:
            for _ in range(3):
                block3_cupy()
            cp.cuda.Stream.null.synchronize()
            
            t0 = now_ms()
            for _ in range(n_iter):
                block3_cupy()
            cp.cuda.Stream.null.synchronize()
            cupy_time = (now_ms() - t0) / n_iter
        
        vs_torch = torch_time / adamah_time if adamah_time > 0 else 0
        vs_cupy = cupy_time / adamah_time if adamah_time > 0 else 0
        
        print(f"{n_iter:<12} {adamah_time:<15.4f} {torch_time:<15.4f} {cupy_time:<15.4f} {vs_torch:<12.2f}x {vs_cupy:<12.2f}x")
        
        results.append({"block": "Multi-Head Attn", "n": n_iter, "adamah": adamah_time, "torch": torch_time, "cupy": cupy_time})
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    for block_name in ["Attn-FFN Hybrid", "Residual Chain", "Multi-Head Attn"]:
        block_results = [r for r in results if r["block"] == block_name]
        if block_results:
            avg_vs_torch = np.mean([r["torch"] / r["adamah"] for r in block_results if r["adamah"] > 0])
            avg_vs_cupy = np.mean([r["cupy"] / r["adamah"] for r in block_results if r["adamah"] > 0])
            print(f"{block_name:<20} Average speedup: {avg_vs_torch:.2f}x vs PyTorch, {avg_vs_cupy:.2f}x vs CuPy")
    
    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
