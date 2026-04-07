# ADAMAH 5.2.0

**GPU compute on any hardware. No CUDA required.**

ADAMAH runs on NVIDIA, AMD, Intel, ARM, and Raspberry Pi via Vulkan. Supports float32, bfloat16, and int8 quantized operations. 2-4x faster than PyTorch CUDA on transformer workloads.

## Install

```bash
pip install .
```

`pip install .` remains the primary package install flow.

Requirements:
- Python 3.8+
- NumPy
- Vulkan drivers/runtime
- local toolchain for first native build

Typical Linux build dependencies:

```bash
sudo apt install gcc libvulkan-dev glslang-tools
```

Typical Windows build dependencies:

- Vulkan SDK
- MSVC or MinGW/clang

## Quick Start

```python
import adamah
import numpy as np

gpu = adamah.init()

# Create a GPU memory map
gpu.map_create(0, word_size=4, pack_size=1, n_packs=1024)

# Upload data
locs = np.arange(1024, dtype=np.uint32)
data = np.random.randn(1024).astype(np.float32)
gpu.scatter(0, locs, data)

# Download result
result = gpu.gather(0, locs)
```

## BFloat16

Half the memory, same compute precision. Ideal for LLM inference.

```python
gpu = adamah.init()
gpu.set_dtype(adamah.DTYPE_BF16)

# Create bf16 map — pack_size is logical elements, not bytes
gpu.map_create_typed(0, adamah.DTYPE_BF16, pack_size=768, n_packs=50000)

# scatter/gather auto-convert: CPU float32 <-> GPU bf16
gpu.scatter(0, locs, float32_data)
result = gpu.gather(0, locs)  # returns float32

# All ops work: matmul, softmax, layernorm, gelu, etc.
# Internally: load bf16 -> compute in f32 -> store bf16
```

## INT8 Quantized

Quarter the memory. For weight storage in inference.

```python
gpu.set_dtype(adamah.DTYPE_Q8)

gpu.map_create_typed(1, adamah.DTYPE_Q8,
                     pack_size=768, n_packs=50000,
                     group_size=128)

# Set quantization parameters (per-group scale and zero point)
scales = np.ones(n_groups, dtype=np.float32) * 0.01
zero_points = np.zeros(n_groups, dtype=np.float32)
gpu.set_qparams(1, scales, zero_points)

# scatter quantizes f32->int8, gather dequantizes int8->f32
gpu.scatter(1, locs, float_weights)
```

## Operations

### Unary (34 ops)
```python
gpu.sigmoid(map_id, src_h, dst_h, n)
gpu.gelu(map_id, src_h, dst_h, n)
gpu.relu(map_id, src_h, dst_h, n)
gpu.tanh(map_id, src_h, dst_h, n)
gpu.swish(map_id, src_h, dst_h, n)
gpu.exp(map_id, src_h, dst_h, n)
gpu.log(map_id, src_h, dst_h, n)
gpu.sqrt(map_id, src_h, dst_h, n)
# ... and 26 more

# Or by op code
gpu.map_op1_dev(map_id, adamah.OP_GELU, src_h, dst_h, n)
```

### Binary (20 ops)
```python
gpu.map_op2_dev(map_id, adamah.OP_ADD, a_h, b_h, dst_h, n)
# ADD, SUB, MUL, DIV, POW, MIN, MAX, MOD
# EQ, NE, LT, LE, GT, GE, AND, OR, XOR, ATAN2, STEP, SMOOTHSTEP
```

### Neural Network
```python
# Matrix multiply: C = A @ B
gpu.map_matmul_dev(map_id, a_h, b_h, c_h, M, K, N, n_ops)

# Softmax (per-row)
gpu.map_softmax_dev(map_id, src_h, dst_h, n_rows, row_size)

# Layer normalization
gpu.map_layernorm_dev(map_id, src_h, dst_h, gamma_h, beta_h, dim, n_rows, eps)

# Broadcast scalar ops
gpu.map_broadcast_dev(map_id, adamah.BROADCAST_MUL, src_h, scalar_h, dst_h, n)

# Reduce
gpu.map_reduce_dev(map_id, adamah.REDUCE_SUM, src_h, dst_h, n)
```

### Automatic Fusion
Operations are automatically batched and fused:
```python
gpu.map_op1_dev(0, adamah.OP_EXP, ...)
gpu.map_op1_dev(0, adamah.OP_TANH, ...)
gpu.map_op2_dev(0, adamah.OP_ADD, ...)
gpu.sync()  # single GPU dispatch for all
```

## Packaging

ADAMAH is now packaged both as:

- a normal `pip install .` Python package
- a source release folder/zip that can be uploaded directly to GitHub releases

The native library can be built during install or first-run bootstrap.

## Data Types

| Type | Constant | Memory | Use |
|------|----------|--------|-----|
| float32 | `adamah.DTYPE_F32` | 4 bytes/elem | Default, training |
| bfloat16 | `adamah.DTYPE_BF16` | 2 bytes/elem | Inference, fine-tuning |
| int8 | `adamah.DTYPE_Q8` | 1 byte/elem | Weight storage |

## Performance

| Workload | vs PyTorch CUDA | vs CuPy |
|----------|----------------|---------|
| Attention-FFN | **4x faster** | 20x |
| Residual Chain | **3.5x faster** | 17x |
| Multi-Head Attention | **2.5x faster** | 17x |

*RTX 3070, identical logical operations*

## License

CC-BY-NC 4.0 — Samuele Scuglia

**ADAMAH** — *The Ground for computation.*

## Authorship and AI Assistance

- Concept, architecture, and product direction: **Samuele Scuglia**.
- Implementation support used during development: **Claude**, **Gemini**,
  and **Codex**.
- These tools are used as engineering assistants only. They are **not
  co-authors** of the project.
