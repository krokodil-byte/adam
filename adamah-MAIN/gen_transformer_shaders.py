#!/usr/bin/env python3
"""
Generate ALL dtype-variant shaders for ADAMAH from templates.

Shaders generated per dtype (f32, bf16, q4, q6, q8):
  rmsnorm, rope, matmul_t, row_copy, fma,
  op1, op2, broadcast, reduce, reduce_small,
  softmax, layernorm, matmul, scatter, gather

Each dtype only differs in load/store helpers; the core logic
is shared via L()/S() template functions.
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
# 6. Unary ops: elementwise unary with op_code switch
# ============================================================
# Guarded op table used for all non-f32 dtypes (defensive clamps)
UNARY_OPS_GUARDED = """\
    switch (pc.op_code) {
        case 0u:  r = -val; break;
        case 1u:  r = abs(val); break;
        case 2u:  r = sqrt(max(val,0.0)); break;
        case 3u:  r = exp(val); break;
        case 4u:  r = log(max(val,1e-7)); break;
        case 5u:  r = tanh(val); break;
        case 6u:  r = max(0.0, val); break;
        case 7u:  r = 0.5*val*(1.0+tanh(0.797884583*(val+0.0447149985*val*val*val))); break;
        case 8u:  r = sin(val); break;
        case 9u:  r = cos(val); break;
        case 10u: r = tan(val); break;
        case 11u: r = asin(clamp(val,-1.0,1.0)); break;
        case 12u: r = acos(clamp(val,-1.0,1.0)); break;
        case 13u: r = atan(val); break;
        case 14u: r = sinh(val); break;
        case 15u: r = cosh(val); break;
        case 16u: r = 1.0/(1.0+exp(-val)); break;
        case 17u: r = val/(1.0+exp(-val)); break;
        case 18u: { float sp=log(1.0+exp(val)); r=val*tanh(sp); break; }
        case 19u: { float t; if(val>0.0) t=val; else t=1.67325997*(exp(val)-1.0); r=t*1.05069995; break; }
        case 20u: r = (val>0.0)?val:exp(val)-1.0; break;
        case 21u: r = max(val,0.01*val); break;
        case 22u: r = ceil(val); break;
        case 23u: r = floor(val); break;
        case 24u: r = round(val); break;
        case 25u: r = (val>0.0)?1.0:((val<0.0)?-1.0:0.0); break;
        case 26u: r = 1.0/val; break;
        case 27u: r = val*val; break;
        case 28u: r = val*val*val; break;
        case 29u: r = log(1.0+exp(val)); break;
        case 30u: r = clamp(0.2*val+0.5,0.0,1.0); break;
        case 31u: r = val*clamp(0.2*val+0.5,0.0,1.0); break;
        case 32u: r = exp(val)-1.0; break;
        case 33u: r = log(1.0+val); break;
        default:  r = val; break;
    }"""

def make_op1(dt):
    bt = DTYPES[dt]['buf_type']
    pe = push_extra(dt)
    b_scale = bindings(dt, 3) if dt not in ('f32', 'bf16') else ''

    return f"""#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform PushConstants {{
    uint op_code; uint n_locs; uint pack_size; {pe}
}} pc;
layout(set = 0, binding = 0) buffer MapBuffer {{ {bt} map_data[]; }};
layout(set = 0, binding = 1) buffer SrcLocs   {{ uint locs_src[]; }};
layout(set = 0, binding = 2) buffer DstLocs   {{ uint locs_dst[]; }};
{b_scale}
{helpers(dt)}
void main() {{
    uint tid = gl_GlobalInvocationID.x;
    uint li = tid / pc.pack_size, ei = tid % pc.pack_size;
    if (li >= pc.n_locs) return;
    uint src = locs_src[li] * pc.pack_size + ei;
    uint dst = locs_dst[li] * pc.pack_size + ei;
    float val = {L(dt, 'src')};
    float r;
{UNARY_OPS_GUARDED}
    {S(dt, 'dst', 'r')}
}}
"""

# ============================================================
# 7. Binary ops: elementwise binary with op_code switch
# ============================================================
BINARY_OPS = """\
    switch (pc.op_code) {
        case 0u: r=a+b; break; case 1u: r=a-b; break;
        case 2u: r=a*b; break; case 3u: r=a/b; break;
        case 4u: r=pow(a,b); break; case 5u: r=min(a,b); break;
        case 6u: r=max(a,b); break; case 7u: r=mod(a,b); break;
        case 8u: r=(a==b)?1.0:0.0; break; case 9u: r=(a!=b)?1.0:0.0; break;
        case 10u: r=(a<b)?1.0:0.0; break; case 11u: r=(a<=b)?1.0:0.0; break;
        case 12u: r=(a>b)?1.0:0.0; break; case 13u: r=(a>=b)?1.0:0.0; break;
        case 14u: r=((a!=0.0)&&(b!=0.0))?1.0:0.0; break;
        case 15u: r=((a!=0.0)||(b!=0.0))?1.0:0.0; break;
        case 16u: r=((a!=0.0)!=(b!=0.0))?1.0:0.0; break;
        case 17u: r=atan(a,b); break;
        case 18u: r=step(a,b); break;
        case 19u: r=smoothstep(a,b,0.5); break;
        default: r=a; break;
    }"""

def make_op2(dt):
    bt = DTYPES[dt]['buf_type']
    pe = push_extra(dt)
    b_scale = bindings(dt, 4) if dt not in ('f32', 'bf16') else ''

    return f"""#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform PushConstants {{
    uint op_code; uint n_locs; uint pack_size; {pe}
}} pc;
layout(set = 0, binding = 0) buffer MapBuffer {{ {bt} map_data[]; }};
layout(set = 0, binding = 1) buffer LocsA     {{ uint locs_a[]; }};
layout(set = 0, binding = 2) buffer LocsB     {{ uint locs_b[]; }};
layout(set = 0, binding = 3) buffer LocsDst   {{ uint locs_dst[]; }};
{b_scale}
{helpers(dt)}
void main() {{
    uint tid = gl_GlobalInvocationID.x;
    uint li = tid / pc.pack_size, ei = tid % pc.pack_size;
    if (li >= pc.n_locs) return;
    float a = {L(dt, 'locs_a[li]*pc.pack_size+ei')};
    float b = {L(dt, 'locs_b[li]*pc.pack_size+ei')};
    float r;
{BINARY_OPS}
    {S(dt, 'locs_dst[li]*pc.pack_size+ei', 'r')}
}}
"""

# ============================================================
# 8. Broadcast: val op scalar
# ============================================================
def make_broadcast(dt):
    bt = DTYPES[dt]['buf_type']
    pe = push_extra(dt)
    b_scale = bindings(dt, 4) if dt not in ('f32', 'bf16') else ''
    # bf16 reads scalar at raw location; quant reads at loc*pack_size
    is_quant = dt not in ('f32', 'bf16')
    sc = 'locs_scalar[li]*pc.pack_size' if is_quant else 'locs_scalar[li]'

    return f"""#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform PushConstants {{
    uint op_code; uint n_locs; uint pack_size; {pe}
}} pc;
layout(set = 0, binding = 0) buffer MapBuffer   {{ {bt} map_data[]; }};
layout(set = 0, binding = 1) buffer SrcLocs     {{ uint locs_src[]; }};
layout(set = 0, binding = 2) buffer ScalarLocs  {{ uint locs_scalar[]; }};
layout(set = 0, binding = 3) buffer DstLocs     {{ uint locs_dst[]; }};
{b_scale}
{helpers(dt)}
void main() {{
    uint tid = gl_GlobalInvocationID.x;
    uint li = tid / pc.pack_size, ei = tid % pc.pack_size;
    if (li >= pc.n_locs) return;
    float val = {L(dt, 'locs_src[li]*pc.pack_size+ei')};
    float scalar = {L(dt, sc)};
    float r;
    if (pc.op_code==0u) r=val*scalar;
    else if (pc.op_code==1u) r=val/scalar;
    else if (pc.op_code==2u) r=val+scalar;
    else r=val-scalar;
    {S(dt, 'locs_dst[li]*pc.pack_size+ei', 'r')}
}}
"""

# ============================================================
# 9. Reduce: parallel sum/max/min
# ============================================================
def make_reduce(dt):
    bt = DTYPES[dt]['buf_type']
    pe = push_extra(dt)
    b_scale = bindings(dt, 3) if dt not in ('f32', 'bf16') else ''

    return f"""#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform PushConstants {{
    uint op_code; uint n_locs; uint pack_size; {pe}
}} pc;
layout(set = 0, binding = 0) buffer MapBuffer {{ {bt} map_data[]; }};
layout(set = 0, binding = 1) buffer SrcLocs   {{ uint locs_src[]; }};
layout(set = 0, binding = 2) buffer DstLocs   {{ uint locs_dst[]; }};
{b_scale}
shared float sdata[256];
{helpers(dt)}
void main() {{
    uint tid = gl_LocalInvocationID.x;
    uint li = gl_WorkGroupID.x;
    if (li >= pc.n_locs) return;
    uint base = locs_src[li] * pc.pack_size;
    float val = (pc.op_code==0u)?0.0:((pc.op_code==1u)?-1e38:1e38);
    for (uint i=tid; i<pc.pack_size; i+=256u) {{
        float x = {L(dt, 'base+i')};
        if (pc.op_code==0u) val+=x;
        else if (pc.op_code==1u) val=max(val,x);
        else val=min(val,x);
    }}
    sdata[tid]=val; barrier();
    for (uint s=128u; s>0u; s>>=1u) {{
        if (tid<s) {{
            if (pc.op_code==0u) sdata[tid]+=sdata[tid+s];
            else if (pc.op_code==1u) sdata[tid]=max(sdata[tid],sdata[tid+s]);
            else sdata[tid]=min(sdata[tid],sdata[tid+s]);
        }}
        barrier();
    }}
    if (tid==0u) {S(dt, 'locs_dst[li]', 'sdata[0]')}
}}
"""

# reduce_small is identical to reduce
make_reduce_small = make_reduce

# ============================================================
# 10. Softmax: exp-normalize per row
# ============================================================
def make_softmax(dt):
    bt = DTYPES[dt]['buf_type']
    pe = push_extra(dt)
    b_scale = bindings(dt, 3) if dt not in ('f32', 'bf16') else ''

    return f"""#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform PushConstants {{
    uint n_rows; uint row_size; {pe}
}} pc;
layout(set = 0, binding = 0) buffer MapBuffer {{ {bt} map_data[]; }};
layout(set = 0, binding = 1) buffer SrcLocs   {{ uint locs_src[]; }};
layout(set = 0, binding = 2) buffer DstLocs   {{ uint locs_dst[]; }};
{b_scale}
shared float sdata[256];
{helpers(dt)}
void main() {{
    uint tid = gl_LocalInvocationID.x;
    uint ri = gl_WorkGroupID.x;
    if (ri >= pc.n_rows) return;
    uint sb = locs_src[ri] * pc.row_size;
    uint db = locs_dst[ri] * pc.row_size;
    float mx = -1e38;
    for (uint i=tid; i<pc.row_size; i+=256u) mx=max(mx, {L(dt, 'sb+i')});
    sdata[tid]=mx; barrier();
    for (uint s=128u;s>0u;s>>=1u) {{ if(tid<s) sdata[tid]=max(sdata[tid],sdata[tid+s]); barrier(); }}
    float row_max=sdata[0]; barrier();
    float sm=0.0;
    for (uint i=tid; i<pc.row_size; i+=256u) {{
        float e=exp({L(dt, 'sb+i')}-row_max);
        {S(dt, 'db+i', 'e')}
        sm+=e;
    }}
    sdata[tid]=sm; barrier();
    for (uint s=128u;s>0u;s>>=1u) {{ if(tid<s) sdata[tid]+=sdata[tid+s]; barrier(); }}
    float inv=1.0/sdata[0]; barrier();
    for (uint i=tid; i<pc.row_size; i+=256u) {S(dt, 'db+i', f'{L(dt, "db+i")}*inv')}
}}
"""

# ============================================================
# 11. LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
# ============================================================
def make_layernorm(dt):
    bt = DTYPES[dt]['buf_type']
    pe = push_extra(dt)
    is_quant = dt not in ('f32', 'bf16')
    b_scale = bindings(dt, 5) if is_quant else ''
    # bf16 reads gamma/beta at raw location; quant multiplies by dim
    gb = 'locs_gamma[ri]*pc.dim' if is_quant else 'locs_gamma[ri]'
    bb = 'locs_beta[ri]*pc.dim' if is_quant else 'locs_beta[ri]'

    return f"""#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform PushConstants {{
    uint n_rows; uint dim; float eps; {pe}
}} pc;
layout(set = 0, binding = 0) buffer MapBuffer {{ {bt} map_data[]; }};
layout(set = 0, binding = 1) buffer SrcLocs   {{ uint locs_src[]; }};
layout(set = 0, binding = 2) buffer DstLocs   {{ uint locs_dst[]; }};
layout(set = 0, binding = 3) buffer GammaLocs {{ uint locs_gamma[]; }};
layout(set = 0, binding = 4) buffer BetaLocs  {{ uint locs_beta[]; }};
{b_scale}
shared float sdata[256];
{helpers(dt)}
void main() {{
    uint tid=gl_LocalInvocationID.x;
    uint ri=gl_WorkGroupID.x;
    if (ri>=pc.n_rows) return;
    uint sb=locs_src[ri]*pc.dim, db=locs_dst[ri]*pc.dim;
    uint gb={gb}, bb={bb};
    float sv=0.0;
    for (uint i=tid;i<pc.dim;i+=256u) sv+={L(dt, 'sb+i')};
    sdata[tid]=sv; barrier();
    for (uint s=128u;s>0u;s>>=1u) {{ if(tid<s) sdata[tid]+=sdata[tid+s]; barrier(); }}
    float mean=sdata[0]/float(pc.dim); barrier();
    float vv=0.0;
    for (uint i=tid;i<pc.dim;i+=256u) {{ float d={L(dt, 'sb+i')}-mean; vv+=d*d; }}
    sdata[tid]=vv; barrier();
    for (uint s=128u;s>0u;s>>=1u) {{ if(tid<s) sdata[tid]+=sdata[tid+s]; barrier(); }}
    float inv_std=1.0/sqrt(sdata[0]/float(pc.dim)+pc.eps); barrier();
    for (uint i=tid;i<pc.dim;i+=256u) {{
        float n=({L(dt, 'sb+i')}-mean)*inv_std;
        {S(dt, 'db+i', f'n*{L(dt, "gb+i")}+{L(dt, "bb+i")}')}
    }}
}}
"""

# ============================================================
# 12. MatMul (non-transposed): C = A @ B
# ============================================================
def make_matmul(dt):
    bt = DTYPES[dt]['buf_type']
    pe = push_extra(dt)
    b_scale = bindings(dt, 4) if dt not in ('f32', 'bf16') else ''

    return f"""#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(push_constant) uniform PushConstants {{
    uint M; uint K; uint N; uint n_ops; {pe}
}} pc;
layout(set = 0, binding = 0) buffer MapBuffer {{ {bt} map_data[]; }};
layout(set = 0, binding = 1) buffer LocsA     {{ uint locs_a[]; }};
layout(set = 0, binding = 2) buffer LocsB     {{ uint locs_b[]; }};
layout(set = 0, binding = 3) buffer LocsC     {{ uint locs_c[]; }};
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
    for (uint k=0u; k<pc.K; k++) {{
        sum += {L(dt, 'a_base + row*pc.K + k')} * {L(dt, 'b_base + k*pc.N + col')};
    }}
    {S(dt, 'c_base + row*pc.N + col', 'sum')}
}}
"""

# ============================================================
# 13. Scatter: write f32 src_data into packed dtype map
# ============================================================
# Scatter and gather have unique per-dtype buffer layouts
# (not using generic L/S since they bridge f32 ↔ packed dtype)

QUANT_CONFIGS = {
    'q4':  {'bits': 4, 'per_word': 8, 'mask': '0xFu',  'max_val': 15, 'signed': False, 'bias': 0},
    'q6':  {'bits': 6, 'per_word': 5, 'mask': '0x3Fu', 'max_val': 63, 'signed': True,  'bias': 32},
    'q8':  {'bits': 8, 'per_word': 4, 'mask': '0xFFu', 'max_val': 255,'signed': False, 'bias': 0},
}

def make_scatter(dt):
    if dt == 'f32':
        return f"""#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform PushConstants {{ uint n_locs; uint pack_size; }} pc;
layout(set = 0, binding = 0) buffer MapBuffer  {{ float map_data[]; }};
layout(set = 0, binding = 1) buffer SrcBuffer  {{ float src_data[]; }};
layout(set = 0, binding = 2) buffer LocsBuffer {{ uint locs[]; }};
void main() {{
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= pc.n_locs * pc.pack_size) return;
    uint li = tid / pc.pack_size, ei = tid % pc.pack_size;
    map_data[locs[li] * pc.pack_size + ei] = src_data[tid];
}}
"""
    if dt == 'bf16':
        return """#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform PushConstants { uint n_locs; uint pack_size; } pc;
layout(set = 0, binding = 0) buffer MapBuffer  { uint map_data[]; };
layout(set = 0, binding = 1) buffer SrcBuffer  { float src_data[]; };
layout(set = 0, binding = 2) buffer LocsBuffer { uint locs[]; };
void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= pc.n_locs * pc.pack_size) return;
    uint li = tid / pc.pack_size, ei = tid % pc.pack_size;
    uint logical = locs[li] * pc.pack_size + ei;
    float val = src_data[tid];
    uint bits = floatBitsToUint(val);
    uint v16 = (bits + 0x7FFFu + ((bits >> 16u) & 1u)) >> 16u;
    uint w = logical >> 1u; uint s = logical & 1u;
    uint shift = s * 16u;
    atomicAnd(map_data[w], ~(0xFFFFu << shift));
    atomicOr(map_data[w], (v16 & 0xFFFFu) << shift);
}
"""
    # Quantized types (q4/q6/q8) - use interleaved qparams
    c = QUANT_CONFIGS[dt]
    if c['signed']:
        requant = f"int qi = clamp(int(round(qf)) + {c['bias']}, 0, {c['max_val']});"
    else:
        requant = f"uint qi = uint(clamp(round(qf), 0.0, {c['max_val']}.0));"

    # q8 scatter historically uses separate scale_data/zp_data buffers
    if dt == 'q8':
        return f"""#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform PushConstants {{ uint n_locs; uint pack_size; uint group_size; }} pc;
layout(set = 0, binding = 0) buffer MapBuffer  {{ uint map_data[]; }};
layout(set = 0, binding = 1) buffer SrcBuffer  {{ float src_data[]; }};
layout(set = 0, binding = 2) buffer LocsBuffer {{ uint locs[]; }};
layout(set = 0, binding = 3) buffer ScaleMap   {{ float scale_data[]; }};
layout(set = 0, binding = 4) buffer ZPMap      {{ float zp_data[]; }};
void main() {{
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= pc.n_locs * pc.pack_size) return;
    uint li = tid / pc.pack_size, ei = tid % pc.pack_size;
    uint logical = locs[li] * pc.pack_size + ei;
    float val = src_data[tid];
    uint group = logical / pc.group_size;
    float scale = scale_data[group];
    float zp    = zp_data[group];
    float qf = (val - zp) / max(scale, 1e-10);
    {requant}
    uint word_idx = logical / {c['per_word']}u;
    uint sub_idx  = logical % {c['per_word']}u;
    uint shift = sub_idx * {c['bits']}u;
    atomicAnd(map_data[word_idx], ~({c['mask']} << shift));
    atomicOr(map_data[word_idx], (uint(qi) & {c['mask']}) << shift);
}}
"""
    # q4/q6 scatter uses interleaved qparams
    return f"""#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform PushConstants {{ uint n_locs; uint pack_size; uint group_size; }} pc;
layout(set = 0, binding = 0) buffer MapBuffer  {{ uint  map_data[]; }};
layout(set = 0, binding = 1) buffer SrcBuffer  {{ float src_data[]; }};
layout(set = 0, binding = 2) buffer LocsBuffer {{ uint  locs[]; }};
layout(set = 0, binding = 3) buffer QParams    {{ float qparams[]; }};
void main() {{
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= pc.n_locs * pc.pack_size) return;
    uint li = tid / pc.pack_size, ei = tid % pc.pack_size;
    uint logical = locs[li] * pc.pack_size + ei;
    float val = src_data[tid];
    uint group = logical / pc.group_size;
    float scale = qparams[group * 2u];
    float zp    = qparams[group * 2u + 1u];
    float qf = (val - zp) / max(scale, 1e-10);
    {requant}
    uint word_idx = logical / {c['per_word']}u;
    uint sub_idx  = logical % {c['per_word']}u;
    uint shift = sub_idx * {c['bits']}u;
    atomicAnd(map_data[word_idx], ~({c['mask']} << shift));
    atomicOr(map_data[word_idx], (uint(qi) & {c['mask']}) << shift);
}}
"""

# ============================================================
# 14. Gather: read packed dtype map into f32 dst_data
# ============================================================
def make_gather(dt):
    if dt == 'f32':
        return f"""#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform PushConstants {{ uint n_locs; uint pack_size; }} pc;
layout(set = 0, binding = 0) buffer MapBuffer  {{ float map_data[]; }};
layout(set = 0, binding = 1) buffer DstBuffer  {{ float dst_data[]; }};
layout(set = 0, binding = 2) buffer LocsBuffer {{ uint locs[]; }};
void main() {{
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= pc.n_locs * pc.pack_size) return;
    uint li = tid / pc.pack_size, ei = tid % pc.pack_size;
    dst_data[tid] = map_data[locs[li] * pc.pack_size + ei];
}}
"""
    if dt == 'bf16':
        return """#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform PushConstants { uint n_locs; uint pack_size; } pc;
layout(set = 0, binding = 0) buffer MapBuffer  { uint map_data[]; };
layout(set = 0, binding = 1) buffer DstBuffer  { float dst_data[]; };
layout(set = 0, binding = 2) buffer LocsBuffer { uint locs[]; };
void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= pc.n_locs * pc.pack_size) return;
    uint li = tid / pc.pack_size, ei = tid % pc.pack_size;
    uint logical = locs[li] * pc.pack_size + ei;
    uint w = logical >> 1u; uint s = logical & 1u;
    uint val16 = (map_data[w] >> (s * 16u)) & 0xFFFFu;
    dst_data[tid] = uintBitsToFloat(val16 << 16u);
}
"""
    # All quant types use interleaved qparams for gather
    c = QUANT_CONFIGS[dt]
    dequant = f"float(int(raw) - {c['bias']})" if c['signed'] else "float(raw)"
    return f"""#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform PushConstants {{ uint n_locs; uint pack_size; uint group_size; }} pc;
layout(set = 0, binding = 0) buffer MapBuffer  {{ uint  map_data[]; }};
layout(set = 0, binding = 1) buffer DstBuffer  {{ float dst_data[]; }};
layout(set = 0, binding = 2) buffer LocsBuffer {{ uint  locs[]; }};
layout(set = 0, binding = 3) buffer QParams    {{ float qparams[]; }};
void main() {{
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= pc.n_locs * pc.pack_size) return;
    uint li = tid / pc.pack_size, ei = tid % pc.pack_size;
    uint logical = locs[li] * pc.pack_size + ei;
    uint word_idx = logical / {c['per_word']}u;
    uint sub_idx  = logical % {c['per_word']}u;
    uint raw = (map_data[word_idx] >> (sub_idx * {c['bits']}u)) & {c['mask']};
    float ival = {dequant};
    uint group = logical / pc.group_size;
    dst_data[tid] = ival * qparams[group * 2u] + qparams[group * 2u + 1u];
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
    'map_op1': make_op1,
    'map_op2': make_op2,
    'map_broadcast': make_broadcast,
    'map_reduce': make_reduce,
    'map_reduce_small': make_reduce_small,
    'map_softmax': make_softmax,
    'map_layernorm': make_layernorm,
    'map_matmul': make_matmul,
    'map_scatter': make_scatter,
    'map_gather': make_gather,
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
