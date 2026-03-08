#!/usr/bin/env python3
"""
Generate 5 new transformer-specific shaders for all ADAMAH dtypes.

New ops:
  1. rmsnorm    - x / sqrt(mean(x²) + eps) * weight
  2. rope       - Rotary Positional Encoding (cos/sin rotation)
  3. matmul_t   - A @ B^T (B transposed) for attention scores
  4. row_copy   - Copy a single row by index (embedding lookup)
  5. fma        - Fused multiply-add: a * b + c element-wise

Generates for: f32, bf16, q4, q6, q8
"""
import os

# ============================================================
# dtype-specific load/store snippets
# ============================================================

F32_IO = {
    'buf_type': 'float',
    'load': 'map_data[{idx}]',
    'store': 'map_data[{idx}] = {val};',
    'extra_bindings': '',
    'extra_push': '',
}

BF16_IO = {
    'buf_type': 'uint',
    'load': 'bf16_load({idx})',
    'store': 'bf16_store({idx}, {val});',
    'extra_bindings': '',
    'extra_push': '',
    'helpers': r"""
float bf16_load(uint idx) {
    uint w = idx >> 1u; uint s = idx & 1u;
    return uintBitsToFloat(((map_data[w] >> (s*16u)) & 0xFFFFu) << 16u);
}
void bf16_store(uint idx, float val) {
    uint v16 = (floatBitsToUint(val) + 0x7FFFu + ((floatBitsToUint(val) >> 16u) & 1u)) >> 16u;
    uint w = idx >> 1u; uint s = idx & 1u;
    uint shift = s * 16u;
    atomicAnd(map_data[w], ~(0xFFFFu << shift));
    atomicOr(map_data[w], (v16 & 0xFFFFu) << shift);
}
""",
}

def make_quant_io(dt, bits, per_word, mask, max_val, signed=False, bias=0):
    sign_conv = f"float(int(raw) - {bias})" if signed else "float(raw)"
    if signed:
        requant = f"int qi = clamp(int(round(qf)) + {bias}, 0, {max_val});"
    else:
        requant = f"uint qi = uint(clamp(round(qf), 0.0, {max_val}.0));"
    return {
        'buf_type': 'uint',
        'extra_bindings_fn': lambda sb: f"""
layout(set = 0, binding = {sb}) buffer ScaleMap {{ float scale_data[]; }};
layout(set = 0, binding = {sb+1}) buffer ZPMap {{ float zp_data[]; }};""",
        'extra_push': 'uint group_size;',
        'helpers': f"""
float load_{dt}(uint idx) {{
    uint word_idx = idx / {per_word}u;
    uint sub_idx  = idx % {per_word}u;
    uint raw = (map_data[word_idx] >> (sub_idx * {bits}u)) & {mask};
    uint group = idx / pc.group_size;
    return {sign_conv} * scale_data[group] + zp_data[group];
}}
void store_{dt}(uint idx, float val) {{
    uint group = idx / pc.group_size;
    float qf = (val - zp_data[group]) / max(scale_data[group], 1e-10);
    {requant}
    uint word_idx = idx / {per_word}u;
    uint sub_idx  = idx % {per_word}u;
    uint shift = sub_idx * {bits}u;
    atomicAnd(map_data[word_idx], ~({mask} << shift));
    atomicOr(map_data[word_idx], (uint(qi) & {mask}) << shift);
}}
""",
        'load': f'load_{dt}({{idx}})',
        'store': f'store_{dt}({{idx}}, {{val}});',
    }

DTYPES = {
    'f32': F32_IO,
    'bf16': BF16_IO,
    'q4': make_quant_io('q4', 4, 8, '0xFu', 15),
    'q6': make_quant_io('q6', 6, 5, '0x3Fu', 63, signed=True, bias=32),
    'q8': make_quant_io('q8', 8, 4, '0xFFu', 255),
}

def L(dt, idx_expr):
    """Generate load expression."""
    return DTYPES[dt]['load'].format(idx=idx_expr)

def S(dt, idx_expr, val_expr):
    """Generate store statement."""
    return DTYPES[dt]['store'].format(idx=idx_expr, val=val_expr)

def helpers(dt):
    return DTYPES[dt].get('helpers', '')

def bindings(dt, start_binding):
    fn = DTYPES[dt].get('extra_bindings_fn')
    return fn(start_binding) if fn else ''

def push_extra(dt):
    return DTYPES[dt].get('extra_push', '')

# ============================================================
# 1. RMSNorm: x / sqrt(mean(x²) + eps) * weight
# ============================================================
def make_rmsnorm(dt):
    bt = DTYPES[dt]['buf_type']
    pe = push_extra(dt)
    # Binding layout depends on dtype
    if dt in ('f32',):
        b_scale = ''
        sb_w = 4
    elif dt == 'bf16':
        b_scale = ''
        sb_w = 4
    else:
        b_scale = bindings(dt, 4)
        sb_w = 6

    return f"""#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform PushConstants {{
    uint n_rows; uint dim; float eps; {pe}
}} pc;
layout(set = 0, binding = 0) buffer MapBuffer {{ {bt} map_data[]; }};
layout(set = 0, binding = 1) buffer SrcLocs   {{ uint locs_src[]; }};
layout(set = 0, binding = 2) buffer WtLocs    {{ uint locs_wt[]; }};
layout(set = 0, binding = 3) buffer DstLocs   {{ uint locs_dst[]; }};
{b_scale}
shared float sdata[256];
{helpers(dt)}
void main() {{
    uint tid = gl_LocalInvocationID.x;
    uint ri = gl_WorkGroupID.x;
    if (ri >= pc.n_rows) return;
    uint sb = locs_src[ri] * pc.dim;
    uint wb = locs_wt[ri] * pc.dim;
    uint db = locs_dst[ri] * pc.dim;
    // Sum of squares
    float ss = 0.0;
    for (uint i = tid; i < pc.dim; i += 256u) {{
        float x = {L(dt, 'sb+i')};
        ss += x * x;
    }}
    sdata[tid] = ss; barrier();
    for (uint s=128u; s>0u; s>>=1u) {{ if(tid<s) sdata[tid]+=sdata[tid+s]; barrier(); }}
    float inv_rms = 1.0 / sqrt(sdata[0] / float(pc.dim) + pc.eps); barrier();
    // Normalize and scale by weight
    for (uint i = tid; i < pc.dim; i += 256u) {{
        float x = {L(dt, 'sb+i')};
        float w = {L(dt, 'wb+i')};
        {S(dt, 'db+i', 'x * inv_rms * w')}
    }}
}}
"""

# ============================================================
# 2. RoPE: Rotary Positional Encoding
# ============================================================
def make_rope(dt):
    bt = DTYPES[dt]['buf_type']
    pe = push_extra(dt)
    b_scale = bindings(dt, 3) if dt not in ('f32', 'bf16') else ''

    return f"""#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform PushConstants {{
    uint n_tokens; uint n_heads; uint head_dim; uint pos_offset; float freq_base; {pe}
}} pc;
layout(set = 0, binding = 0) buffer MapBuffer {{ {bt} map_data[]; }};
layout(set = 0, binding = 1) buffer SrcLocs   {{ uint locs_src[]; }};
layout(set = 0, binding = 2) buffer DstLocs   {{ uint locs_dst[]; }};
{b_scale}
{helpers(dt)}
void main() {{
    uint tid = gl_GlobalInvocationID.x;
    uint total_pairs = pc.n_tokens * pc.n_heads * (pc.head_dim / 2u);
    if (tid >= total_pairs) return;

    uint pair = tid;
    uint half_dim = pc.head_dim / 2u;
    uint token = pair / (pc.n_heads * half_dim);
    uint rem = pair % (pc.n_heads * half_dim);
    uint head = rem / half_dim;
    uint d = rem % half_dim;

    uint pos = pc.pos_offset + token;
    float freq = 1.0 / pow(pc.freq_base, float(2u * d) / float(pc.head_dim));
    float angle = float(pos) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    uint stride = pc.n_heads * pc.head_dim;
    uint base_src = locs_src[token] * stride + head * pc.head_dim;
    uint base_dst = locs_dst[token] * stride + head * pc.head_dim;

    float x0 = {L(dt, 'base_src + d')};
    float x1 = {L(dt, 'base_src + d + half_dim')};

    {S(dt, 'base_dst + d',            'x0 * cos_a - x1 * sin_a')}
    {S(dt, 'base_dst + d + half_dim', 'x0 * sin_a + x1 * cos_a')}
}}
"""

# ============================================================
# 3. MatMul transposed: C = A @ B^T
# ============================================================
def make_matmul_t(dt):
    bt = DTYPES[dt]['buf_type']
    pe = push_extra(dt)
    b_scale = bindings(dt, 5) if dt not in ('f32', 'bf16') else ''

    return f"""#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(push_constant) uniform PushConstants {{
    uint M; uint K; uint N; uint n_ops; {pe}
}} pc;
layout(set = 0, binding = 0) buffer MapBuffer {{ {bt} map_data[]; }};
layout(set = 0, binding = 1) buffer LocsA {{ uint locs_a[]; }};
layout(set = 0, binding = 2) buffer LocsB {{ uint locs_b[]; }};
layout(set = 0, binding = 3) buffer LocsC {{ uint locs_c[]; }};
{b_scale}
{helpers(dt)}
void main() {{
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;
    uint op  = gl_GlobalInvocationID.z;
    if (row>=pc.M || col>=pc.N || op>=pc.n_ops) return;
    uint a_base = locs_a[op];
    uint b_base = locs_b[op];
    uint c_base = locs_c[op];
    float sum = 0.0;
    for (uint k = 0u; k < pc.K; k++) {{
        float a_val = {L(dt, 'a_base + row*pc.K + k')};
        float b_val = {L(dt, 'b_base + col*pc.K + k')};  // B transposed: B[col, k]
        sum += a_val * b_val;
    }}
    {S(dt, 'c_base + row*pc.N + col', 'sum')}
}}
"""

# ============================================================
# 4. Row copy: copy row[idx] from src to dst (embedding lookup)
# ============================================================
def make_row_copy(dt):
    bt = DTYPES[dt]['buf_type']
    pe = push_extra(dt)
    b_scale = bindings(dt, 3) if dt not in ('f32', 'bf16') else ''

    return f"""#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform PushConstants {{
    uint n_copies; uint row_size; {pe}
}} pc;
layout(set = 0, binding = 0) buffer MapBuffer  {{ {bt} map_data[]; }};
layout(set = 0, binding = 1) buffer CopySpec   {{ uint copy_spec[]; }};  // [src_row, dst_loc] pairs
layout(set = 0, binding = 2) buffer SrcBase    {{ uint src_base[]; }};   // base offset of embedding table
{b_scale}
{helpers(dt)}
void main() {{
    uint tid = gl_GlobalInvocationID.x;
    uint ci = tid / pc.row_size;
    uint ei = tid % pc.row_size;
    if (ci >= pc.n_copies) return;
    uint src_row = copy_spec[ci * 2u];
    uint dst_loc = copy_spec[ci * 2u + 1u];
    uint src_off = src_base[0] + src_row * pc.row_size + ei;
    uint dst_off = dst_loc * pc.row_size + ei;
    float val = {L(dt, 'src_off')};
    {S(dt, 'dst_off', 'val')}
}}
"""

# ============================================================
# 5. FMA: a * b + c element-wise
# ============================================================
def make_fma(dt):
    bt = DTYPES[dt]['buf_type']
    pe = push_extra(dt)
    b_scale = bindings(dt, 5) if dt not in ('f32', 'bf16') else ''

    return f"""#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform PushConstants {{
    uint n_locs; uint pack_size; {pe}
}} pc;
layout(set = 0, binding = 0) buffer MapBuffer {{ {bt} map_data[]; }};
layout(set = 0, binding = 1) buffer LocsA   {{ uint locs_a[]; }};
layout(set = 0, binding = 2) buffer LocsB   {{ uint locs_b[]; }};
layout(set = 0, binding = 3) buffer LocsC   {{ uint locs_c[]; }};
layout(set = 0, binding = 4) buffer LocsDst {{ uint locs_dst[]; }};
{b_scale}
{helpers(dt)}
void main() {{
    uint tid = gl_GlobalInvocationID.x;
    uint li = tid / pc.pack_size, ei = tid % pc.pack_size;
    if (li >= pc.n_locs) return;
    float a = {L(dt, 'locs_a[li]*pc.pack_size+ei')};
    float b = {L(dt, 'locs_b[li]*pc.pack_size+ei')};
    float c = {L(dt, 'locs_c[li]*pc.pack_size+ei')};
    {S(dt, 'locs_dst[li]*pc.pack_size+ei', 'a * b + c')}
}}
"""

# ============================================================
# Main
# ============================================================
GENERATORS = {
    'map_rmsnorm': make_rmsnorm,
    'map_rope': make_rope,
    'map_matmul_t': make_matmul_t,
    'map_row_copy': make_row_copy,
    'map_fma': make_fma,
}

def main():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'adamah', 'shaders', 'src')
    total = 0
    for dt in ('f32', 'bf16', 'q4', 'q6', 'q8'):
        out_dir = os.path.join(base, dt)
        os.makedirs(out_dir, exist_ok=True)
        print(f"=== {dt} ===")
        for name, gen_fn in GENERATORS.items():
            path = os.path.join(out_dir, f'{name}.comp')
            src = gen_fn(dt)
            with open(path, 'w') as f:
                f.write(src)
            total += 1
            print(f"  {name}.comp")
    print(f"\nGenerated {total} new shaders")

if __name__ == '__main__':
    main()
