/*
 * ADAMAH v4.0 - Map-Centric GPU Compute
 *
 * Pure GPU operations on Memory Maps
 * scatter/gather for CPU I/O
 *
 * CC BY-NC 4.0 - Samuele Scuglia - 2026-01-18
 */

#ifdef __linux__
#define _GNU_SOURCE
#include <dlfcn.h>
#endif

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

// Error codes
#define ADAMAH_OK 0
#define ADAMAH_ERR_VULKAN -1
#define ADAMAH_ERR_MEMORY -2
#define ADAMAH_ERR_INVALID -3
#define ADAMAH_ERR_NOT_FOUND -4

#define MAX_MAPS 16
#define MAX_BUFS 64
#define MAX_RES 2048
#define MAX_FREE 4096
#define LOCAL_SIZE 256
#define CMD_RING 4
// Shared staging pool size: one pool for all map I/O (replaces per-map staging).
// 512 MB is enough to transfer any map in at most a few passes.
#define STAGING_POOL_BYTES (512ULL * 1024 * 1024)
#define HOT_POOL_BYTES_DEFAULT (512ull << 20)
#define COLD_POOL_BYTES_DEFAULT (512ull << 20)

#define RES_TYPE_CVAR 1
#define RES_TYPE_LOCS 2

// Data types
#define DTYPE_F32 0
#define DTYPE_BF16 1
#define DTYPE_Q8 2
#define DTYPE_Q4 3
#define DTYPE_Q6 4
#define DTYPE_COUNT 5

// Block sizes for quantized formats
#define Q4_BLOCK_ELEMS 256
#define Q4_BLOCK_BYTES 144 // 2+2+12+128
#define Q4_BLOCK_WORDS 36  // 144/4
#define Q6_BLOCK_ELEMS 256
#define Q6_BLOCK_BYTES 216 // 128+64+16+2 padded to 216
#define Q6_BLOCK_WORDS 54  // 216/4

// Fusion system constants - maximums, actual values set dynamically
#define FUSION_MAX_OPS_LIMIT 8192
#define FUSION_MAX_LEVELS 128
#define FUSION_MAX_LOCS 8192

// Operation types for fusion
#define FUSE_OP_UNARY 1
#define FUSE_OP_BINARY 2
#define FUSE_OP_MATMUL 3
#define FUSE_OP_REDUCE 4
#define FUSE_OP_SOFTMAX 5
#define FUSE_OP_LAYERNORM 6
#define FUSE_OP_BROADCAST 7

// Forward declarations
int map_destroy(uint32_t id);
int map_init_dtype(uint32_t id, uint32_t dtype, uint32_t pack_size,
                   uint32_t n_packs, uint32_t group_size);
int adamah_set_dtype(uint32_t dtype);
static int init_dtype_pipelines(uint32_t dtype);
int map_set_qparams(uint32_t map_id, const float *scales,
                    const float *zero_points, uint32_t n_groups);

// Fusion operation entry
typedef struct {
  int op_type;      // FUSE_OP_UNARY, FUSE_OP_BINARY, etc.
  uint32_t op_code; // specific op (EXP, ADD, etc.) or mode flag for fused ops
  uint32_t map_id;
  uint32_t locs_src;  // handle
  uint32_t locs_src2; // handle (for binary ops)
  uint32_t locs_dst;  // handle
  uint32_t n;         // number of elements
  int level;          // dependency level (computed)
  // Extra params for matmul, etc.
  uint32_t M, K, N;
  uint32_t locs_extra[4]; // for matmul: locs_a, locs_b, locs_c
  float scalar;           // for ops with scalar
  float eps;              // for layernorm
} FuseOp;

// GPU capabilities - set during init based on device properties
typedef struct {
  uint64_t vram_bytes;          // Total VRAM
  uint32_t max_workgroup_size;  // Max threads per workgroup
  uint32_t max_workgroups;      // Max workgroups per dispatch
  uint32_t max_descriptor_sets; // Max bound descriptor sets
  uint32_t max_compute_units;  // Approximate compute units (from subgroup size)
  uint32_t fusion_max_ops;     // Dynamic fusion queue size
  uint32_t optimal_batch_size; // Optimal ops per batch
} GpuCaps;

static GpuCaps gpu_caps = {0};

// Fusion context
typedef struct {
  FuseOp ops[FUSION_MAX_OPS_LIMIT];
  int n_ops;
  int loc_write_level[FUSION_MAX_LOCS]; // last write level for each loc handle
  int enabled;
  int max_level;
} FusionCtx;

static FusionCtx fusion = {0};

// Forward declarations
int adamah_fusion_flush(void);
static void fusion_reset(void);
static int fusion_flush_pending_for_immediate(void);

// ============================================
// Structures
// ============================================

// Internal GPU buffer (for locs, temp data)
typedef struct {
  char name[64];
  VkBuffer buf;
  VkDeviceMemory mem;
  void *ptr; // Mapped if HOST_VISIBLE
  VkDeviceSize bytes_capacity;
  uint32_t elem_size; // Bytes per element (for bookkeeping)
  int device_local;   // 1 = VRAM, 0 = HOST_VISIBLE
  VkBufferUsageFlags usage;
} GpuBuf;

static GpuBuf *get_or_create_buf_ex(const char *base_name, uint32_t n_elems,
                                    uint32_t elem_size, int device_local,
                                    VkBufferUsageFlags usage);

// Memory Map - the core data structure
typedef struct {
  int active;
  uint32_t word_size; // Bytes per word (4=f32, 2=bf16, 1=q8)
  uint32_t dtype;     // DTYPE_F32, DTYPE_BF16, DTYPE_Q8, DTYPE_Q4, DTYPE_Q6
  uint32_t pack_size; // Words per pack (logical elements)
  uint32_t n_packs;   // Number of packs
  uint64_t total_bytes;

  VkBuffer buf;
  VkDeviceMemory mem;

  // Staging for CPU transfer
  VkBuffer staging;
  VkDeviceMemory staging_mem;
  void *staging_ptr;
  VkMemoryPropertyFlags
      staging_mem_props; // Track memory properties for flush/invalidate

  // Quantization params (for DTYPE_Q8)
  uint32_t group_size; // Elements per quantization group
  VkBuffer qparam_buf; // GPU buffer for [scale, zp] per group
  VkDeviceMemory qparam_mem;
  void *qparam_staging_ptr;
  VkBuffer qparam_staging;
  VkDeviceMemory qparam_staging_mem;
  uint32_t n_groups; // Number of quantization groups
} Map;

// Compute pipeline
typedef struct {
  VkShaderModule shader;
  VkDescriptorSetLayout desc_layout;
  VkPipelineLayout pipe_layout;
  VkPipeline pipeline;
  VkDescriptorPool desc_pool;
  VkDescriptorSet desc_set;
} Pipeline;

typedef struct {
  VkDeviceSize offset;
  VkDeviceSize size;
} FreeSeg;

typedef struct {
  uint32_t active;
  uint32_t type;
  uint32_t size_bytes;
  VkDeviceSize alloc_size;
  VkDeviceSize hot_offset;
  VkDeviceSize cold_offset;
  uint64_t last_used;
  uint8_t hot_valid;
  uint8_t dirty;
  uint8_t pinned;
} ResEntry;

// Push constants
typedef struct {
  uint32_t op;
  uint32_t n;
} PushOp;
typedef struct {
  uint32_t op;
  uint32_t n;
  float scalar;
} PushOpS;

// Global context
static struct {
  int initialized;
  VkInstance instance;
  VkPhysicalDevice phys;
  VkDevice device;
  VkQueue queue;
  uint32_t queue_family;
  VkCommandPool cmd_pool;
  VkCommandBuffer cmd;
  VkFence fence;
  VkCommandBuffer cmd_ring[CMD_RING];
  VkFence fence_ring[CMD_RING];
  uint64_t submit_id_ring[CMD_RING];
  int in_flight_ring[CMD_RING];
  uint32_t cmd_ring_next;
  uint64_t submit_counter;
  uint64_t last_completed;
  uint32_t devbuf_counter;
  uint32_t num_buffer_recreates;
  uint32_t stage_upload_grow_events;
  uint32_t stage_download_grow_events;
  int pending_desc_reset;
  VkDeviceSize copy_align;
  VkDeviceSize storage_align;

  // Shared staging pool — one host-visible buffer reused by all scatter/gather.
  // Replaces per-map staging so maps of any size can be transferred in chunks.
  VkBuffer pool_buf;
  VkDeviceMemory pool_mem;
  void *pool_ptr;
  VkDeviceSize pool_size;
  VkMemoryPropertyFlags pool_mem_props;

  Map maps[MAX_MAPS];
  GpuBuf bufs[MAX_BUFS];
  int buf_count;

  GpuBuf *hot_pool;
  GpuBuf *cold_pool;
  VkDeviceSize hot_pool_bytes;
  VkDeviceSize cold_pool_bytes;
  VkDeviceSize cold_alloc;
  FreeSeg hot_free[MAX_FREE];
  uint32_t hot_free_count;
  ResEntry res[MAX_RES];
  uint32_t res_count;
  uint64_t res_tick;

  Pipeline unary_pipe;
  Pipeline binary_pipe;
  Pipeline matmul_pipe;
  Pipeline matmul_t_pipe;
  Pipeline rmsnorm_pipe;
  Pipeline rope_pipe;
  Pipeline row_copy_pipe;
  Pipeline fma_pipe;
  Pipeline reduce_pipe;
  Pipeline reduce_small_pipe;
  Pipeline broadcast_pipe;
  Pipeline softmax_pipe;
  Pipeline softmax_abs_pipe;
  Pipeline attn_softmax_abs_pipe;
  Pipeline layernorm_pipe;
  Pipeline unified_pipe;
  Pipeline scatter_pipe;
  Pipeline gather_pipe;
  Pipeline repeat_penalty_pipe;
  Pipeline argmax_pipe;
  Pipeline topk_pipe;
  Pipeline topp_pipe;

  // Cross-map (dual-buffer) pipelines
  Pipeline matmul_x_pipe;
  Pipeline matmul_t_x_pipe;
  Pipeline rmsnorm_x_pipe;

  // Cross-map F32×Q4 pipelines
  Pipeline matvec_topk_t_xq4_pipe;
  Pipeline matvec_t_xq4_pipe;
  Pipeline matmul_t_xq4_pipe;
  Pipeline matmul_xq4_pipe;

  // Cross-map F32×Q8 pipelines
  Pipeline matvec_topk_t_xq8_pipe;
  Pipeline matvec_rerank_t_xq8_pipe;
  Pipeline matvec_t_xq8_pipe;
  Pipeline matmul_t_xq8_pipe;
  Pipeline matmul_xq8_pipe;
  Pipeline row_gather_xq8_pipe;

  // Per-dtype pipeline sets (bf16, q8)
  // Index: 0=f32 (uses the pipes above), 1=bf16, 2=q8
  Pipeline dtype_scatter_pipe[DTYPE_COUNT];
  Pipeline dtype_gather_pipe[DTYPE_COUNT];
  Pipeline dtype_unary_pipe[DTYPE_COUNT];
  Pipeline dtype_binary_pipe[DTYPE_COUNT];
  Pipeline dtype_matmul_pipe[DTYPE_COUNT];
  Pipeline dtype_reduce_pipe[DTYPE_COUNT];
  Pipeline dtype_reduce_small_pipe[DTYPE_COUNT];
  Pipeline dtype_broadcast_pipe[DTYPE_COUNT];
  Pipeline dtype_softmax_pipe[DTYPE_COUNT];
  Pipeline dtype_layernorm_pipe[DTYPE_COUNT];
  int dtype_pipes_loaded[DTYPE_COUNT]; // Tracks which dtype sets are loaded

  uint32_t active_dtype; // Currently active dtype for new maps

  char shader_path[512];
} ctx = {0};

// ============================================
// Vulkan Helpers
// ============================================

static uint32_t find_mem_type(uint32_t bits, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(ctx.phys, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((bits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & props) == props)
      return i;
  return UINT32_MAX;
}

static uint32_t find_device_local(uint32_t bits) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(ctx.phys, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
    if (!(bits & (1u << i)))
      continue;
    VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
    if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
        !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
      return i;
  }
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
    if ((bits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
      return i;
  }
  return UINT32_MAX;
}

// Find optimal host-visible memory for staging buffers
// Universal Vulkan: HOST_CACHED (with flush) > HOST_COHERENT (fallback)
static uint32_t find_host_staging(uint32_t bits,
                                  VkMemoryPropertyFlags *props_out) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(ctx.phys, &mp);
  static int staging_logged = 0;

  // Try: HOST_VISIBLE + HOST_CACHED (fast memcpy, needs explicit
  // flush/invalidate)
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
    if (!(bits & (1u << i)))
      continue;
    VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
    if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
        (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) {
      *props_out = f;
      if (!staging_logged) {
        fprintf(stderr, "[ADAMAH] Staging: HOST_CACHED (fast memcpy with "
                        "explicit flush)\n");
        staging_logged = 1;
      }
      return i;
    }
  }

  // Fallback: HOST_VISIBLE + HOST_COHERENT (slower but no flush needed)
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
    if (!(bits & (1u << i)))
      continue;
    VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
    if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
        (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      *props_out = f;
      if (!staging_logged) {
        fprintf(stderr,
                "[ADAMAH] Staging: HOST_COHERENT (fallback, slower memcpy)\n");
        staging_logged = 1;
      }
      return i;
    }
  }

  return UINT32_MAX;
}

// Extended version that returns memory properties
static int create_buffer_ex(VkBuffer *buf, VkDeviceMemory *mem,
                            VkDeviceSize size, VkBufferUsageFlags usage,
                            int device_local,
                            VkMemoryPropertyFlags *props_out) {
  VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                            .size = size,
                            .usage = usage};
  if (vkCreateBuffer(ctx.device, &bci, NULL, buf) != VK_SUCCESS)
    return -1;

  VkMemoryRequirements reqs;
  vkGetBufferMemoryRequirements(ctx.device, *buf, &reqs);

  VkMemoryPropertyFlags props = 0;
  uint32_t mem_type;

  if (device_local) {
    mem_type = find_device_local(reqs.memoryTypeBits);
  } else {
    // For staging buffers, use optimized memory
    mem_type = find_host_staging(reqs.memoryTypeBits, &props);
  }

  if (mem_type == UINT32_MAX) {
    vkDestroyBuffer(ctx.device, *buf, NULL);
    return -1;
  }

  if (props_out)
    *props_out = props;

  VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                              .allocationSize = reqs.size,
                              .memoryTypeIndex = mem_type};
  if (vkAllocateMemory(ctx.device, &mai, NULL, mem) != VK_SUCCESS) {
    vkDestroyBuffer(ctx.device, *buf, NULL);
    return -1;
  }
  vkBindBufferMemory(ctx.device, *buf, *mem, 0);
  return 0;
}

static int create_buffer(VkBuffer *buf, VkDeviceMemory *mem, VkDeviceSize size,
                         VkBufferUsageFlags usage, int device_local) {
  return create_buffer_ex(buf, mem, size, usage, device_local, NULL);
}

// ============================================
// True Vulkan Batching - accumulate in single command buffer
// ============================================
static int batch_mode = 0;
static int cmd_recording = 0;    // Is command buffer currently recording?
static int batch_op_counter = 0; // Counter for unique buffer names in batch
static int cmd_recording_async = 0;
static int cmd_async_slot = -1;

#define HANDLE_FROM_BUF(b) ((uint32_t)((b) - ctx.bufs) + 1u)
#define BUF_FROM_HANDLE(h)                                                     \
  (((h) == 0 || (h) > (uint32_t)ctx.buf_count) ? NULL : &ctx.bufs[(h) - 1u])

static void reset_pipeline_desc_pool(Pipeline *p);
static int async_all_done(void);

static VkDescriptorSet alloc_desc_set(Pipeline *p) {
  VkDescriptorSet ds = p->desc_set;
  if (batch_mode) {
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = p->desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &p->desc_layout};
    if (vkAllocateDescriptorSets(ctx.device, &dsai, &ds) != VK_SUCCESS)
      return VK_NULL_HANDLE;
  }
  return ds;
}

static VkDescriptorSet alloc_desc_set_async(Pipeline *p) {
  VkDescriptorSet ds = VK_NULL_HANDLE;
  VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = p->desc_pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &p->desc_layout};
  if (vkAllocateDescriptorSets(ctx.device, &dsai, &ds) != VK_SUCCESS) {
    if (async_all_done()) {
      reset_pipeline_desc_pool(p);
      if (vkAllocateDescriptorSets(ctx.device, &dsai, &ds) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    } else {
      return VK_NULL_HANDLE;
    }
  }
  ctx.pending_desc_reset = 1;
  return ds;
}

static void cmd_barrier_after_dispatch(void) {
  if (!batch_mode)
    return;
  VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                         VK_ACCESS_SHADER_WRITE_BIT};
  vkCmdPipelineBarrier(ctx.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL,
                       0, NULL);
}

static void cmd_buffer_barrier(VkPipelineStageFlags src_stage,
                               VkAccessFlags src_access,
                               VkPipelineStageFlags dst_stage,
                               VkAccessFlags dst_access, VkBuffer *bufs,
                               uint32_t count) {
  if (count == 0)
    return;
  VkBufferMemoryBarrier barriers[4];
  if (count > 4)
    count = 4;
  for (uint32_t i = 0; i < count; i++) {
    barriers[i] = (VkBufferMemoryBarrier){
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = bufs[i],
        .offset = 0,
        .size = VK_WHOLE_SIZE};
  }
  vkCmdPipelineBarrier(ctx.cmd, src_stage, dst_stage, 0, 0, NULL, count,
                       barriers, 0, NULL);
}

static void cmd_begin(void);
static void cmd_submit(void);

static VkDeviceSize align_up(VkDeviceSize v, VkDeviceSize a) {
  if (a == 0)
    return v;
  return (v + a - 1) & ~(a - 1);
}

static VkDeviceSize align_storage(VkDeviceSize v) {
  return align_up(v, ctx.storage_align);
}

static void hot_free_insert(VkDeviceSize offset, VkDeviceSize size) {
  if (size == 0 || ctx.hot_free_count >= MAX_FREE)
    return;
  uint32_t idx = 0;
  while (idx < ctx.hot_free_count && ctx.hot_free[idx].offset < offset)
    idx++;
  for (uint32_t i = ctx.hot_free_count; i > idx; i--) {
    ctx.hot_free[i] = ctx.hot_free[i - 1];
  }
  ctx.hot_free[idx].offset = offset;
  ctx.hot_free[idx].size = size;
  ctx.hot_free_count++;

  // Coalesce with previous
  if (idx > 0) {
    FreeSeg *prev = &ctx.hot_free[idx - 1];
    FreeSeg *cur = &ctx.hot_free[idx];
    if (prev->offset + prev->size == cur->offset) {
      prev->size += cur->size;
      for (uint32_t i = idx; i + 1 < ctx.hot_free_count; i++) {
        ctx.hot_free[i] = ctx.hot_free[i + 1];
      }
      ctx.hot_free_count--;
      idx--;
    }
  }

  // Coalesce with next
  if (idx + 1 < ctx.hot_free_count) {
    FreeSeg *cur = &ctx.hot_free[idx];
    FreeSeg *next = &ctx.hot_free[idx + 1];
    if (cur->offset + cur->size == next->offset) {
      cur->size += next->size;
      for (uint32_t i = idx + 1; i + 1 < ctx.hot_free_count; i++) {
        ctx.hot_free[i] = ctx.hot_free[i + 1];
      }
      ctx.hot_free_count--;
    }
  }
}

static int hot_alloc(VkDeviceSize size, VkDeviceSize *out_offset) {
  size = align_storage(size);
  for (uint32_t i = 0; i < ctx.hot_free_count; i++) {
    FreeSeg *seg = &ctx.hot_free[i];
    if (seg->size >= size) {
      *out_offset = seg->offset;
      seg->offset += size;
      seg->size -= size;
      if (seg->size == 0) {
        for (uint32_t j = i; j + 1 < ctx.hot_free_count; j++) {
          ctx.hot_free[j] = ctx.hot_free[j + 1];
        }
        ctx.hot_free_count--;
      }
      return 0;
    }
  }
  return -1;
}

static ResEntry *res_get(uint32_t id) {
  if (id == 0 || id > MAX_RES)
    return NULL;
  ResEntry *r = &ctx.res[id - 1];
  if (!r->active)
    return NULL;
  return r;
}

static void res_pin(uint32_t id) {
  ResEntry *r = res_get(id);
  if (!r)
    return;
  if (r->pinned < 255)
    r->pinned++;
}

static void res_unpin(uint32_t id) {
  ResEntry *r = res_get(id);
  if (!r)
    return;
  if (r->pinned > 0)
    r->pinned--;
}

static void res_copy_buffers(VkBuffer src, VkDeviceSize src_off, VkBuffer dst,
                             VkDeviceSize dst_off, VkDeviceSize size,
                             int src_hot, int dst_hot) {
  VkDeviceSize copy_size = align_up(size, ctx.copy_align);
  if (copy_size == 0)
    return;
  cmd_begin();
  if (src_hot) {
    VkBuffer bufs[1] = {src};
    cmd_buffer_barrier(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT, bufs, 1);
  }
  VkBufferCopy copy = {
      .srcOffset = src_off, .dstOffset = dst_off, .size = copy_size};
  vkCmdCopyBuffer(ctx.cmd, src, dst, 1, &copy);
  if (dst_hot) {
    VkBuffer bufs[1] = {dst};
    cmd_buffer_barrier(
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, bufs, 1);
  }
  cmd_submit();
}

static int res_evict_one(void) {
  int victim = -1;
  uint64_t best = UINT64_MAX;
  for (uint32_t i = 0; i < MAX_RES; i++) {
    ResEntry *r = &ctx.res[i];
    if (!r->active || !r->hot_valid || r->pinned)
      continue;
    if (r->last_used < best) {
      best = r->last_used;
      victim = (int)i;
    }
  }
  if (victim < 0)
    return -1;
  ResEntry *r = &ctx.res[victim];
  if (r->dirty) {
    res_copy_buffers(ctx.hot_pool->buf, r->hot_offset, ctx.cold_pool->buf,
                     r->cold_offset, r->alloc_size, 1, 0);
    r->dirty = 0;
  }
  hot_free_insert(r->hot_offset, r->alloc_size);
  r->hot_valid = 0;
  return 0;
}

static int res_require_hot(uint32_t id, VkDeviceSize *out_offset) {
  ResEntry *r = res_get(id);
  if (!r)
    return -1;
  if (r->hot_valid) {
    r->last_used = ++ctx.res_tick;
    *out_offset = r->hot_offset;
    return 0;
  }

  while (hot_alloc(r->alloc_size, &r->hot_offset) != 0) {
    if (res_evict_one() != 0)
      return -1;
  }

  r->hot_valid = 1;
  r->last_used = ++ctx.res_tick;

  if (!r->dirty) {
    res_copy_buffers(ctx.cold_pool->buf, r->cold_offset, ctx.hot_pool->buf,
                     r->hot_offset, r->alloc_size, 0, 1);
  }
  *out_offset = r->hot_offset;
  return 0;
}

static int res_alloc(uint32_t type, uint32_t size_bytes, uint32_t *out_id) {
  if (size_bytes == 0)
    return -1;
  uint32_t id = 0;
  for (uint32_t i = 0; i < MAX_RES; i++) {
    if (!ctx.res[i].active) {
      id = i + 1;
      break;
    }
  }
  if (id == 0)
    return -1;
  ResEntry *r = &ctx.res[id - 1];
  memset(r, 0, sizeof(*r));
  r->active = 1;
  r->type = type;
  r->size_bytes = size_bytes;
  r->alloc_size = align_storage((VkDeviceSize)size_bytes);
  if (ctx.cold_alloc + r->alloc_size > ctx.cold_pool_bytes)
    return -1;
  r->cold_offset = ctx.cold_alloc;
  ctx.cold_alloc += r->alloc_size;
  r->hot_valid = 0;
  r->dirty = 0;
  r->pinned = 0;
  r->last_used = ++ctx.res_tick;
  ctx.res_count++;
  *out_id = id;
  return 0;
}

static int cache_init(VkDeviceSize hot_bytes, VkDeviceSize cold_bytes) {
  if (ctx.hot_pool && ctx.cold_pool)
    return 0;
  VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (hot_bytes == 0)
    hot_bytes = HOT_POOL_BYTES_DEFAULT;
  if (cold_bytes == 0)
    cold_bytes = COLD_POOL_BYTES_DEFAULT;
  if (hot_bytes > UINT32_MAX)
    hot_bytes = UINT32_MAX;
  if (cold_bytes > UINT32_MAX)
    cold_bytes = UINT32_MAX;

  ctx.hot_pool =
      get_or_create_buf_ex("_cache_hot", (uint32_t)hot_bytes, 1, 1, usage);
  ctx.cold_pool =
      get_or_create_buf_ex("_cache_cold", (uint32_t)cold_bytes, 1, 1, usage);
  if (!ctx.hot_pool || !ctx.cold_pool)
    return -1;

  ctx.hot_pool_bytes = ctx.hot_pool->bytes_capacity;
  ctx.cold_pool_bytes = ctx.cold_pool->bytes_capacity;
  ctx.cold_alloc = 0;
  ctx.hot_free_count = 0;
  hot_free_insert(0, ctx.hot_pool_bytes);
  memset(ctx.res, 0, sizeof(ctx.res));
  ctx.res_count = 0;
  ctx.res_tick = 0;
  return 0;
}

static int debug_enabled(void) {
  static int cached = -1;
  if (cached < 0) {
    const char *env = getenv("ADAMAH_DEBUG");
    cached = (env && env[0] && strcmp(env, "0") != 0) ? 1 : 0;
  }
  return cached;
}

static void debug_path(const char *op, const char *path) {
  if (debug_enabled()) {
    fprintf(stderr, "ADAMAH DEBUG: %s path: %s\n", op, path);
  }
}

static VkDeviceSize next_pow2(VkDeviceSize v) {
  if (v <= 1)
    return 1;
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
#if UINTPTR_MAX > 0xFFFFFFFFu
  v |= v >> 32;
#endif
  return v + 1;
}

static VkDeviceSize device_local_bucket(VkDeviceSize v) {
  VkDeviceSize size = 64 * 1024;
  while (size < v) {
    size *= 4;
    if (size == 0)
      break;
  }
  return size;
}

static int is_stage_upload_name(const char *name) {
  return strncmp(name, "_stage_upload", 13) == 0;
}

static int is_stage_download_name(const char *name) {
  return strncmp(name, "_stage_download", 15) == 0;
}

static void reset_pipeline_desc_pool(Pipeline *p) {
  if (!p->desc_pool)
    return;
  vkResetDescriptorPool(ctx.device, p->desc_pool, 0);
  VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = p->desc_pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &p->desc_layout};
  if (vkAllocateDescriptorSets(ctx.device, &dsai, &p->desc_set) != VK_SUCCESS) {
    p->desc_set = VK_NULL_HANDLE;
  }
}

static void update_async_completed(void) {
  for (int i = 0; i < CMD_RING; i++) {
    if (!ctx.in_flight_ring[i])
      continue;
    VkResult st = vkGetFenceStatus(ctx.device, ctx.fence_ring[i]);
    if (st == VK_SUCCESS) {
      ctx.in_flight_ring[i] = 0;
      if (ctx.submit_id_ring[i] > ctx.last_completed) {
        ctx.last_completed = ctx.submit_id_ring[i];
      }
    }
  }
}

static int async_all_done(void) {
  update_async_completed();
  return ctx.last_completed >= ctx.submit_counter;
}

static int cmd_begin_async(void) {
  if (cmd_recording_async)
    return 0;
  if (batch_mode)
    return -1;

  update_async_completed();

  int slot = -1;
  for (int i = 0; i < CMD_RING; i++) {
    int idx = (ctx.cmd_ring_next + i) % CMD_RING;
    if (!ctx.in_flight_ring[idx]) {
      slot = idx;
      break;
    }
  }

  if (slot < 0) {
    slot = ctx.cmd_ring_next % CMD_RING;
    vkWaitForFences(ctx.device, 1, &ctx.fence_ring[slot], VK_TRUE, UINT64_MAX);
    ctx.in_flight_ring[slot] = 0;
    if (ctx.submit_id_ring[slot] > ctx.last_completed) {
      ctx.last_completed = ctx.submit_id_ring[slot];
    }
  }

  ctx.cmd_ring_next = (uint32_t)((slot + 1) % CMD_RING);
  vkResetFences(ctx.device, 1, &ctx.fence_ring[slot]);
  vkResetCommandBuffer(ctx.cmd_ring[slot], 0);

  ctx.cmd = ctx.cmd_ring[slot];
  cmd_async_slot = slot;

  VkCommandBufferBeginInfo cbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  vkBeginCommandBuffer(ctx.cmd, &cbi);
  cmd_recording_async = 1;
  return 0;
}

static uint64_t cmd_submit_async(void) {
  if (!cmd_recording_async)
    return 0;
  vkEndCommandBuffer(ctx.cmd);
  VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                     .commandBufferCount = 1,
                     .pCommandBuffers = &ctx.cmd};
  vkQueueSubmit(ctx.queue, 1, &si, ctx.fence_ring[cmd_async_slot]);
  uint64_t ticket = ++ctx.submit_counter;
  ctx.submit_id_ring[cmd_async_slot] = ticket;
  ctx.in_flight_ring[cmd_async_slot] = 1;
  cmd_recording_async = 0;
  return ticket;
}

static void cmd_abort_async(void) {
  if (!cmd_recording_async)
    return;
  vkEndCommandBuffer(ctx.cmd);
  cmd_recording_async = 0;
}

static void cmd_begin(void) {
  if (batch_mode && cmd_recording) {
    // Already recording in batch mode, just continue adding commands
    return;
  }

  // Wait for previous work to complete
  vkWaitForFences(ctx.device, 1, &ctx.fence, VK_TRUE, UINT64_MAX);
  vkResetFences(ctx.device, 1, &ctx.fence);
  vkResetCommandBuffer(ctx.cmd, 0);

  VkCommandBufferBeginInfo cbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  vkBeginCommandBuffer(ctx.cmd, &cbi);
  cmd_recording = 1;
}

static void cmd_submit(void) {
  if (batch_mode) {
    // In batch mode, don't submit yet - keep accumulating
    // Increment counter for next op to use different buffers
    batch_op_counter++;
    return;
  }

  if (!cmd_recording)
    return;

  vkEndCommandBuffer(ctx.cmd);
  VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                     .commandBufferCount = 1,
                     .pCommandBuffers = &ctx.cmd};
  vkQueueSubmit(ctx.queue, 1, &si, ctx.fence);
  vkWaitForFences(ctx.device, 1, &ctx.fence, VK_TRUE, UINT64_MAX);
  cmd_recording = 0;
}

// Start batch mode - commands accumulated into single command buffer
void batch_begin(void) {
  // Ensure clean state
  if (cmd_recording) {
    vkEndCommandBuffer(ctx.cmd);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                       .commandBufferCount = 1,
                       .pCommandBuffers = &ctx.cmd};
    vkQueueSubmit(ctx.queue, 1, &si, ctx.fence);
    vkWaitForFences(ctx.device, 1, &ctx.fence, VK_TRUE, UINT64_MAX);
    cmd_recording = 0;
  }
  batch_mode = 1;
  batch_op_counter = 0; // Reset counter for new batch
}

// End batch mode - submit all accumulated commands
void batch_end(void) {
  if (!batch_mode)
    return;

  if (fusion.enabled && fusion.n_ops > 0) {
    adamah_fusion_flush();
  }

  batch_mode = 0;

  if (cmd_recording) {
    vkEndCommandBuffer(ctx.cmd);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                       .commandBufferCount = 1,
                       .pCommandBuffers = &ctx.cmd};
    vkQueueSubmit(ctx.queue, 1, &si, ctx.fence);
    vkWaitForFences(ctx.device, 1, &ctx.fence, VK_TRUE, UINT64_MAX);
    cmd_recording = 0;
  }

  // Batch completed, recycle descriptor sets for next batch
  reset_pipeline_desc_pool(&ctx.unary_pipe);
  reset_pipeline_desc_pool(&ctx.binary_pipe);
  reset_pipeline_desc_pool(&ctx.matmul_pipe);
  reset_pipeline_desc_pool(&ctx.matmul_t_pipe);
  reset_pipeline_desc_pool(&ctx.rmsnorm_pipe);
  reset_pipeline_desc_pool(&ctx.rope_pipe);
  reset_pipeline_desc_pool(&ctx.row_copy_pipe);
  reset_pipeline_desc_pool(&ctx.fma_pipe);
  reset_pipeline_desc_pool(&ctx.reduce_pipe);
  reset_pipeline_desc_pool(&ctx.reduce_small_pipe);
  reset_pipeline_desc_pool(&ctx.broadcast_pipe);
  reset_pipeline_desc_pool(&ctx.softmax_pipe);
  reset_pipeline_desc_pool(&ctx.softmax_abs_pipe);
  reset_pipeline_desc_pool(&ctx.attn_softmax_abs_pipe);
  reset_pipeline_desc_pool(&ctx.layernorm_pipe);
  reset_pipeline_desc_pool(&ctx.unified_pipe);
  reset_pipeline_desc_pool(&ctx.scatter_pipe);
  reset_pipeline_desc_pool(&ctx.gather_pipe);
  reset_pipeline_desc_pool(&ctx.repeat_penalty_pipe);
  reset_pipeline_desc_pool(&ctx.argmax_pipe);
  reset_pipeline_desc_pool(&ctx.topk_pipe);
  reset_pipeline_desc_pool(&ctx.topp_pipe);
  reset_pipeline_desc_pool(&ctx.matmul_x_pipe);
  reset_pipeline_desc_pool(&ctx.matmul_t_x_pipe);
  reset_pipeline_desc_pool(&ctx.rmsnorm_x_pipe);
  reset_pipeline_desc_pool(&ctx.matvec_topk_t_xq4_pipe);
  reset_pipeline_desc_pool(&ctx.matvec_t_xq4_pipe);
  reset_pipeline_desc_pool(&ctx.matmul_t_xq4_pipe);
  reset_pipeline_desc_pool(&ctx.matmul_xq4_pipe);
  reset_pipeline_desc_pool(&ctx.matvec_topk_t_xq8_pipe);
  reset_pipeline_desc_pool(&ctx.matvec_rerank_t_xq8_pipe);
  reset_pipeline_desc_pool(&ctx.matvec_t_xq8_pipe);
  reset_pipeline_desc_pool(&ctx.matmul_t_xq8_pipe);
  reset_pipeline_desc_pool(&ctx.matmul_xq8_pipe);
  reset_pipeline_desc_pool(&ctx.row_gather_xq8_pipe);

  batch_op_counter = 0; // Reset counter
}

// Sync: wait for all queued GPU work to finish
void adamah_sync(void) {
  if (!ctx.initialized)
    return;
  // Ensure async work completes and descriptor pools can be recycled
  for (int i = 0; i < CMD_RING; i++) {
    if (ctx.in_flight_ring[i]) {
      vkWaitForFences(ctx.device, 1, &ctx.fence_ring[i], VK_TRUE, UINT64_MAX);
      ctx.in_flight_ring[i] = 0;
      if (ctx.submit_id_ring[i] > ctx.last_completed) {
        ctx.last_completed = ctx.submit_id_ring[i];
      }
    }
  }
  if (ctx.pending_desc_reset) {
    reset_pipeline_desc_pool(&ctx.unary_pipe);
    reset_pipeline_desc_pool(&ctx.binary_pipe);
    reset_pipeline_desc_pool(&ctx.matmul_pipe);
    reset_pipeline_desc_pool(&ctx.reduce_pipe);
    reset_pipeline_desc_pool(&ctx.reduce_small_pipe);
    reset_pipeline_desc_pool(&ctx.broadcast_pipe);
    reset_pipeline_desc_pool(&ctx.softmax_pipe);
    reset_pipeline_desc_pool(&ctx.softmax_abs_pipe);
    reset_pipeline_desc_pool(&ctx.attn_softmax_abs_pipe);
    reset_pipeline_desc_pool(&ctx.layernorm_pipe);
    reset_pipeline_desc_pool(&ctx.unified_pipe);
    reset_pipeline_desc_pool(&ctx.scatter_pipe);
    reset_pipeline_desc_pool(&ctx.gather_pipe);
    reset_pipeline_desc_pool(&ctx.repeat_penalty_pipe);
    reset_pipeline_desc_pool(&ctx.argmax_pipe);
    reset_pipeline_desc_pool(&ctx.topk_pipe);
    reset_pipeline_desc_pool(&ctx.topp_pipe);
    reset_pipeline_desc_pool(&ctx.matvec_topk_t_xq4_pipe);
    reset_pipeline_desc_pool(&ctx.matvec_topk_t_xq8_pipe);
    reset_pipeline_desc_pool(&ctx.matvec_rerank_t_xq8_pipe);
    reset_pipeline_desc_pool(&ctx.row_gather_xq8_pipe);
    ctx.pending_desc_reset = 0;
  }
  vkDeviceWaitIdle(ctx.device);
}

void adamah_synchronize(uint64_t ticket) {
  if (!ctx.initialized)
    return;
  if (ticket == 0 || ticket <= ctx.last_completed)
    return;
  update_async_completed();
  if (ticket <= ctx.last_completed)
    return;
  for (int i = 0; i < CMD_RING; i++) {
    if (ctx.in_flight_ring[i] && ctx.submit_id_ring[i] <= ticket) {
      vkWaitForFences(ctx.device, 1, &ctx.fence_ring[i], VK_TRUE, UINT64_MAX);
      ctx.in_flight_ring[i] = 0;
      if (ctx.submit_id_ring[i] > ctx.last_completed) {
        ctx.last_completed = ctx.submit_id_ring[i];
      }
    }
  }
  if (ctx.pending_desc_reset && async_all_done()) {
    reset_pipeline_desc_pool(&ctx.unary_pipe);
    reset_pipeline_desc_pool(&ctx.binary_pipe);
    reset_pipeline_desc_pool(&ctx.matmul_pipe);
    reset_pipeline_desc_pool(&ctx.reduce_pipe);
    reset_pipeline_desc_pool(&ctx.reduce_small_pipe);
    reset_pipeline_desc_pool(&ctx.broadcast_pipe);
    reset_pipeline_desc_pool(&ctx.softmax_pipe);
    reset_pipeline_desc_pool(&ctx.softmax_abs_pipe);
    reset_pipeline_desc_pool(&ctx.attn_softmax_abs_pipe);
    reset_pipeline_desc_pool(&ctx.layernorm_pipe);
    reset_pipeline_desc_pool(&ctx.unified_pipe);
    reset_pipeline_desc_pool(&ctx.scatter_pipe);
    reset_pipeline_desc_pool(&ctx.gather_pipe);
    reset_pipeline_desc_pool(&ctx.repeat_penalty_pipe);
    reset_pipeline_desc_pool(&ctx.argmax_pipe);
    reset_pipeline_desc_pool(&ctx.topk_pipe);
    reset_pipeline_desc_pool(&ctx.topp_pipe);
    reset_pipeline_desc_pool(&ctx.matvec_topk_t_xq4_pipe);
    reset_pipeline_desc_pool(&ctx.matvec_topk_t_xq8_pipe);
    reset_pipeline_desc_pool(&ctx.matvec_rerank_t_xq8_pipe);
    reset_pipeline_desc_pool(&ctx.row_gather_xq8_pipe);
    ctx.pending_desc_reset = 0;
  }
}

void adamah_synchronize_all(void) {
  if (!ctx.initialized)
    return;

  // Flush pending fusion ops first
  if (fusion.enabled && fusion.n_ops > 0) {
    adamah_fusion_flush();
  }

  for (int i = 0; i < CMD_RING; i++) {
    if (ctx.in_flight_ring[i]) {
      vkWaitForFences(ctx.device, 1, &ctx.fence_ring[i], VK_TRUE, UINT64_MAX);
      ctx.in_flight_ring[i] = 0;
    }
  }
  ctx.last_completed = ctx.submit_counter;
  if (ctx.pending_desc_reset) {
    reset_pipeline_desc_pool(&ctx.unary_pipe);
    reset_pipeline_desc_pool(&ctx.binary_pipe);
    reset_pipeline_desc_pool(&ctx.matmul_pipe);
    reset_pipeline_desc_pool(&ctx.reduce_pipe);
    reset_pipeline_desc_pool(&ctx.reduce_small_pipe);
    reset_pipeline_desc_pool(&ctx.broadcast_pipe);
    reset_pipeline_desc_pool(&ctx.softmax_pipe);
    reset_pipeline_desc_pool(&ctx.softmax_abs_pipe);
    reset_pipeline_desc_pool(&ctx.attn_softmax_abs_pipe);
    reset_pipeline_desc_pool(&ctx.layernorm_pipe);
    reset_pipeline_desc_pool(&ctx.unified_pipe);
    reset_pipeline_desc_pool(&ctx.scatter_pipe);
    reset_pipeline_desc_pool(&ctx.gather_pipe);
    reset_pipeline_desc_pool(&ctx.repeat_penalty_pipe);
    reset_pipeline_desc_pool(&ctx.argmax_pipe);
    reset_pipeline_desc_pool(&ctx.topk_pipe);
    reset_pipeline_desc_pool(&ctx.topp_pipe);
    reset_pipeline_desc_pool(&ctx.matvec_topk_t_xq4_pipe);
    reset_pipeline_desc_pool(&ctx.matvec_topk_t_xq8_pipe);
    reset_pipeline_desc_pool(&ctx.matvec_rerank_t_xq8_pipe);
    reset_pipeline_desc_pool(&ctx.row_gather_xq8_pipe);
    ctx.pending_desc_reset = 0;
  }
}

void adamah_print_counters(void) {
  if (!debug_enabled())
    return;
  fprintf(stderr,
          "ADAMAH DEBUG: buffer_recreates=%u stage_upload_grows=%u "
          "stage_download_grows=%u\n",
          ctx.num_buffer_recreates, ctx.stage_upload_grow_events,
          ctx.stage_download_grow_events);
}

// ============================================
// Init / Shutdown
// ============================================

int adamah_init_ex(uint64_t hot_bytes, uint64_t cold_bytes) {
  if (ctx.initialized)
    return ADAMAH_OK;

  // Create instance
  VkApplicationInfo ai = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                          .pApplicationName = "ADAMAH",
                          .apiVersion = VK_API_VERSION_1_0};
  VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                              .pApplicationInfo = &ai};
  if (vkCreateInstance(&ici, NULL, &ctx.instance) != VK_SUCCESS)
    return ADAMAH_ERR_VULKAN;

  // Get physical device - prefer discrete GPU over integrated/software
  uint32_t dc = 0;
  vkEnumeratePhysicalDevices(ctx.instance, &dc, NULL);
  if (!dc)
    return ADAMAH_ERR_VULKAN;
  VkPhysicalDevice devs[8];
  vkEnumeratePhysicalDevices(ctx.instance, &dc, devs);

  // Debug: print all available devices
  fprintf(stderr, "[ADAMAH] Found %u Vulkan device(s):\n", dc);
  for (uint32_t i = 0; i < dc && i < 8; i++) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(devs[i], &p);
    const char *type_str = "OTHER";
    if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
      type_str = "DISCRETE_GPU";
    else if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
      type_str = "INTEGRATED_GPU";
    else if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
      type_str = "CPU";
    fprintf(stderr, "[ADAMAH]   [%u] %s (%s)\n", i, p.deviceName, type_str);
  }

  // Selection priority: DISCRETE_GPU > INTEGRATED_GPU > OTHER (skip CPU
  // renderer)
  ctx.phys = devs[0]; // Fallback
  int best_score = -1;

  for (uint32_t i = 0; i < dc && i < 8; i++) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(devs[i], &props);

    int score = 0;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
      score = 100;
    else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
      score = 50;
    else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
      score = -100; // Never use CPU renderer
    else
      score = 10;

    if (score > best_score) {
      ctx.phys = devs[i];
      best_score = score;
    }
  }

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(ctx.phys, &props);
  fprintf(stderr, "[ADAMAH] Selected: %s\n", props.deviceName);
  printf("ADAMAH v4: %s\n", props.deviceName);
  ctx.copy_align = props.limits.optimalBufferCopyOffsetAlignment;
  if (ctx.copy_align < 4)
    ctx.copy_align = 4;
  ctx.storage_align = props.limits.minStorageBufferOffsetAlignment;
  if (ctx.storage_align < ctx.copy_align)
    ctx.storage_align = ctx.copy_align;
  if (ctx.storage_align < 4)
    ctx.storage_align = 4;

  // Query GPU capabilities for dynamic tuning
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(ctx.phys, &mem_props);

  // Find total VRAM (device local heap)
  gpu_caps.vram_bytes = 0;
  for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
    if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      if (mem_props.memoryHeaps[i].size > gpu_caps.vram_bytes) {
        gpu_caps.vram_bytes = mem_props.memoryHeaps[i].size;
      }
    }
  }

  // Store compute limits from Vulkan
  gpu_caps.max_workgroup_size = props.limits.maxComputeWorkGroupSize[0];
  gpu_caps.max_workgroups = props.limits.maxComputeWorkGroupCount[0];
  gpu_caps.max_descriptor_sets = props.limits.maxBoundDescriptorSets;

  // Use maxComputeWorkGroupInvocations as primary indicator of GPU compute
  // power This is the max threads per workgroup - directly reflects GPU
  // architecture:
  // - Most desktop GPUs: 1024 (RTX 30xx, 40xx, RX 6000, 7000)
  // - Some mobile/integrated: 512 or 256
  // - High-end workstation: 1024-1536
  uint32_t max_invocations = props.limits.maxComputeWorkGroupInvocations;

  // Combine with workgroup count for total concurrent capacity estimate
  // maxComputeWorkGroupCount[0] is typically 65535 for modern GPUs
  // We use invocations as the scaling factor since it's more meaningful
  uint32_t compute_power = max_invocations; // Base: 256-1024 typically

  // Scale by VRAM as secondary indicator (more VRAM = usually more SMs)
  // This helps differentiate RTX 3070 (8GB) from RTX 3060 (12GB but fewer SMs)
  float vram_scale = 1.0f;
  if (gpu_caps.vram_bytes >= 16ull * 1024 * 1024 * 1024) {
    vram_scale = 2.0f; // 16GB+ = flagship class
  } else if (gpu_caps.vram_bytes >= 8ull * 1024 * 1024 * 1024) {
    vram_scale = 1.5f; // 8-16GB = high-end
  } else if (gpu_caps.vram_bytes >= 4ull * 1024 * 1024 * 1024) {
    vram_scale = 1.0f; // 4-8GB = mid-range
  } else {
    vram_scale = 0.5f; // <4GB = entry/integrated
  }

  gpu_caps.max_compute_units = (uint32_t)(compute_power * vram_scale);

  // Fusion queue size: scale with compute power
  // More powerful GPU = larger queue = better batching
  // Base: max_invocations * 4, scaled by VRAM
  gpu_caps.fusion_max_ops = (uint32_t)(max_invocations * 4 * vram_scale);
  if (gpu_caps.fusion_max_ops > FUSION_MAX_OPS_LIMIT) {
    gpu_caps.fusion_max_ops = FUSION_MAX_OPS_LIMIT;
  }
  if (gpu_caps.fusion_max_ops < 512) {
    gpu_caps.fusion_max_ops = 512; // Minimum for effective batching
  }

  // Optimal batch size: target good GPU utilization without overwhelming
  gpu_caps.optimal_batch_size = max_invocations / 2;
  if (gpu_caps.optimal_batch_size < 64)
    gpu_caps.optimal_batch_size = 64;
  if (gpu_caps.optimal_batch_size > 1024)
    gpu_caps.optimal_batch_size = 1024;

  fprintf(stderr,
          "[ADAMAH] GPU Caps: VRAM=%.1fGB, maxInvocations=%u, "
          "compute_units=%u, fusion_max_ops=%u, optimal_batch=%u\n",
          (double)gpu_caps.vram_bytes / (1024.0 * 1024.0 * 1024.0),
          max_invocations, gpu_caps.max_compute_units, gpu_caps.fusion_max_ops,
          gpu_caps.optimal_batch_size);

  // Find compute queue
  uint32_t qfc = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(ctx.phys, &qfc, NULL);
  VkQueueFamilyProperties qfp[8];
  vkGetPhysicalDeviceQueueFamilyProperties(ctx.phys, &qfc, qfp);
  for (uint32_t i = 0; i < qfc; i++) {
    if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      ctx.queue_family = i;
      break;
    }
  }

  // Create device
  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = {.sType =
                                     VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                 .queueFamilyIndex = ctx.queue_family,
                                 .queueCount = 1,
                                 .pQueuePriorities = &prio};
  VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                            .queueCreateInfoCount = 1,
                            .pQueueCreateInfos = &qci};
  if (vkCreateDevice(ctx.phys, &dci, NULL, &ctx.device) != VK_SUCCESS)
    return ADAMAH_ERR_VULKAN;
  vkGetDeviceQueue(ctx.device, ctx.queue_family, 0, &ctx.queue);

  // Command pool & buffer
  VkCommandPoolCreateInfo cpi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = ctx.queue_family};
  vkCreateCommandPool(ctx.device, &cpi, NULL, &ctx.cmd_pool);

  VkCommandBufferAllocateInfo cai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = ctx.cmd_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1};
  vkAllocateCommandBuffers(ctx.device, &cai, &ctx.cmd);

  VkCommandBufferAllocateInfo cai_ring = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = ctx.cmd_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = CMD_RING};
  vkAllocateCommandBuffers(ctx.device, &cai_ring, ctx.cmd_ring);

  VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                           .flags = VK_FENCE_CREATE_SIGNALED_BIT};
  vkCreateFence(ctx.device, &fci, NULL, &ctx.fence);
  for (int i = 0; i < CMD_RING; i++) {
    vkCreateFence(ctx.device, &fci, NULL, &ctx.fence_ring[i]);
    ctx.in_flight_ring[i] = 0;
    ctx.submit_id_ring[i] = 0;
  }
  ctx.cmd_ring_next = 0;
  ctx.submit_counter = 0;
  ctx.last_completed = 0;
  ctx.devbuf_counter = 0;
  ctx.pending_desc_reset = 0;

  if (cache_init(hot_bytes, cold_bytes) != 0) {
    return ADAMAH_ERR_MEMORY;
  }

  // Allocate shared staging pool (host-visible, used by all scatter/gather).
  ctx.pool_size = STAGING_POOL_BYTES;
  VkBufferUsageFlags pool_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (create_buffer_ex(&ctx.pool_buf, &ctx.pool_mem, ctx.pool_size,
                       pool_usage, 0, &ctx.pool_mem_props) != 0) {
    return ADAMAH_ERR_MEMORY;
  }
  vkMapMemory(ctx.device, ctx.pool_mem, 0, ctx.pool_size, 0, &ctx.pool_ptr);

  // Enable fusion by default and reset state
  fusion.enabled = 1;
  fusion_reset();

  ctx.initialized = 1;
  return ADAMAH_OK;
}

int adamah_init(void) {
  // Two-phase init: first query device, then allocate optimal pool sizes

  // Phase 1: Minimal Vulkan init to query device properties
  if (ctx.initialized)
    return ADAMAH_OK;

  // Create instance
  VkApplicationInfo ai = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                          .pApplicationName = "ADAMAH",
                          .apiVersion = VK_API_VERSION_1_0};
  VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                              .pApplicationInfo = &ai};
  if (vkCreateInstance(&ici, NULL, &ctx.instance) != VK_SUCCESS)
    return ADAMAH_ERR_VULKAN;

  // Get physical device
  uint32_t dc = 0;
  vkEnumeratePhysicalDevices(ctx.instance, &dc, NULL);
  if (!dc)
    return ADAMAH_ERR_VULKAN;
  VkPhysicalDevice devs[8];
  vkEnumeratePhysicalDevices(ctx.instance, &dc, devs);

  // Select best device (prefer discrete)
  ctx.phys = devs[0];
  for (uint32_t i = 0; i < dc && i < 8; i++) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(devs[i], &props);
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      ctx.phys = devs[i];
      break;
    }
  }

  // Query VRAM
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(ctx.phys, &mem_props);
  uint64_t vram = 0;
  for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
    if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      if (mem_props.memoryHeaps[i].size > vram) {
        vram = mem_props.memoryHeaps[i].size;
      }
    }
  }

  // Cleanup partial init - adamah_init_ex will redo this
  vkDestroyInstance(ctx.instance, NULL);
  ctx.instance = NULL;
  ctx.phys = NULL;

  // Phase 2: Calculate optimal pool sizes based on VRAM
  uint64_t hot, cold;
  if (vram >= 8ull * 1024 * 1024 * 1024) {
    // 8GB+ VRAM: use 25% hot, 12.5% cold
    hot = vram / 4;
    cold = vram / 8;
  } else if (vram >= 4ull * 1024 * 1024 * 1024) {
    // 4-8GB VRAM: use 20% hot, 10% cold
    hot = vram / 5;
    cold = vram / 10;
  } else {
    // <4GB: conservative
    hot = 512ull * 1024 * 1024;
    cold = 256ull * 1024 * 1024;
  }

  // Cap at reasonable maximums
  if (hot > 4ull * 1024 * 1024 * 1024)
    hot = 4ull * 1024 * 1024 * 1024;
  if (cold > 2ull * 1024 * 1024 * 1024)
    cold = 2ull * 1024 * 1024 * 1024;

  // Now do full init with optimal sizes
  return adamah_init_ex(hot, cold);
}

void adamah_shutdown(void) {
  if (!ctx.initialized)
    return;

  // Flush any pending fusion ops
  if (fusion.enabled && fusion.n_ops > 0) {
    adamah_fusion_flush();
  }

  vkDeviceWaitIdle(ctx.device);

  // Free shared staging pool
  if (ctx.pool_buf) {
    vkUnmapMemory(ctx.device, ctx.pool_mem);
    vkDestroyBuffer(ctx.device, ctx.pool_buf, NULL);
    vkFreeMemory(ctx.device, ctx.pool_mem, NULL);
    ctx.pool_buf = VK_NULL_HANDLE;
  }

  // Destroy maps
  for (int i = 0; i < MAX_MAPS; i++) {
    if (ctx.maps[i].active)
      map_destroy(i);
  }

  // Destroy buffers
  for (int i = 0; i < ctx.buf_count; i++) {
    if (ctx.bufs[i].ptr)
      vkUnmapMemory(ctx.device, ctx.bufs[i].mem);
    vkDestroyBuffer(ctx.device, ctx.bufs[i].buf, NULL);
    vkFreeMemory(ctx.device, ctx.bufs[i].mem, NULL);
  }

  if (ctx.fence)
    vkDestroyFence(ctx.device, ctx.fence, NULL);
  for (int i = 0; i < CMD_RING; i++) {
    if (ctx.fence_ring[i])
      vkDestroyFence(ctx.device, ctx.fence_ring[i], NULL);
  }
  if (ctx.cmd_pool)
    vkDestroyCommandPool(ctx.device, ctx.cmd_pool, NULL);
  if (ctx.device)
    vkDestroyDevice(ctx.device, NULL);
  if (ctx.instance)
    vkDestroyInstance(ctx.instance, NULL);

  memset(&ctx, 0, sizeof(ctx));
}

// ============================================
// Memory Maps
// ============================================

int map_init(uint32_t id, uint32_t word_size, uint32_t pack_size,
             uint32_t n_packs) {
  // Legacy: word_size=4 → f32, word_size=2 → bf16, word_size=1 → q8
  uint32_t dtype = DTYPE_F32;
  if (word_size == 2)
    dtype = DTYPE_BF16;
  else if (word_size == 1)
    dtype = DTYPE_Q8;
  return map_init_dtype(id, dtype, pack_size, n_packs, 128);
}

// New dtype-aware init
int map_init_dtype(uint32_t id, uint32_t dtype, uint32_t pack_size,
                   uint32_t n_packs, uint32_t group_size) {
  if (!ctx.initialized || id >= MAX_MAPS)
    return ADAMAH_ERR_INVALID;
  if (dtype >= DTYPE_COUNT)
    return ADAMAH_ERR_INVALID;
  if (ctx.maps[id].active)
    map_destroy(id);

  Map *m = &ctx.maps[id];
  m->dtype = dtype;
  m->pack_size = pack_size;
  m->n_packs = n_packs;
  m->group_size = (dtype == DTYPE_Q8 || dtype == DTYPE_Q4 || dtype == DTYPE_Q6) ? group_size : 0;

  // Set word_size based on dtype (logical, for metadata)
  switch (dtype) {
  case DTYPE_F32:
    m->word_size = 4;
    break;
  case DTYPE_BF16:
    m->word_size = 2;
    break;
  case DTYPE_Q8:
    m->word_size = 1;
    break;
  case DTYPE_Q4:
    m->word_size = 1;
    break; // ~0.5625 B/elem actual
  case DTYPE_Q6:
    m->word_size = 1;
    break; // ~0.84375 B/elem actual
  default:
    m->word_size = 4;
    break;
  }

  // Calculate total bytes for GPU storage
  uint64_t total_elements = (uint64_t)pack_size * n_packs;
  switch (dtype) {
  case DTYPE_F32:
    m->total_bytes = total_elements * 4;
    break;
  case DTYPE_BF16:
    // 2 elements per uint32
    m->total_bytes = ((total_elements + 1) / 2) * 4;
    break;
  case DTYPE_Q8:
    // 4 elements per uint32
    m->total_bytes = ((total_elements + 3) / 4) * 4;
    break;
  case DTYPE_Q4: {
    // 8 elements per uint32 (4 bits each)
    uint64_t n_words = (total_elements + 7) / 8;
    m->total_bytes = n_words * 4;
    break;
  }
  case DTYPE_Q6: {
    // 5 elements per uint32 (6 bits each, 2 unused)
    uint64_t n_words = (total_elements + 4) / 5;
    m->total_bytes = n_words * 4;
    break;
  }
  default:
    m->total_bytes = total_elements * 4;
    break;
  }

  // GPU buffer (DEVICE_LOCAL)
  VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (create_buffer(&m->buf, &m->mem, m->total_bytes, usage, 1) != 0)
    return ADAMAH_ERR_MEMORY;

  // No per-map staging buffer — all transfers go through the shared pool.
  m->staging = VK_NULL_HANDLE;
  m->staging_mem = VK_NULL_HANDLE;
  m->staging_ptr = NULL;
  m->staging_mem_props = 0;

  // Zero-init device buffer directly (no staging needed for fill).
  cmd_begin();
  vkCmdFillBuffer(ctx.cmd, m->buf, 0, VK_WHOLE_SIZE, 0);
  cmd_submit();

  // Allocate quantization params buffer for q8
  m->qparam_buf = VK_NULL_HANDLE;
  m->qparam_mem = VK_NULL_HANDLE;
  m->qparam_staging = VK_NULL_HANDLE;
  m->qparam_staging_mem = VK_NULL_HANDLE;
  m->qparam_staging_ptr = NULL;
  m->n_groups = 0;

  if ((dtype == DTYPE_Q8 || dtype == DTYPE_Q4 || dtype == DTYPE_Q6) &&
      group_size > 0) {
    uint64_t total_elems = (uint64_t)pack_size * n_packs;
    m->n_groups = (uint32_t)((total_elems + group_size - 1) / group_size);
    VkDeviceSize qparam_bytes = (VkDeviceSize)m->n_groups * 2 * sizeof(float);

    if (create_buffer(&m->qparam_buf, &m->qparam_mem, qparam_bytes, usage, 1) !=
        0) {
      vkDestroyBuffer(ctx.device, m->buf, NULL);
      vkFreeMemory(ctx.device, m->mem, NULL);
      return ADAMAH_ERR_MEMORY;
    }
    VkMemoryPropertyFlags qprops;
    if (create_buffer_ex(&m->qparam_staging, &m->qparam_staging_mem,
                         qparam_bytes, usage, 0, &qprops) != 0) {
      vkDestroyBuffer(ctx.device, m->qparam_buf, NULL);
      vkFreeMemory(ctx.device, m->qparam_mem, NULL);
      vkDestroyBuffer(ctx.device, m->buf, NULL);
      vkFreeMemory(ctx.device, m->mem, NULL);
      return ADAMAH_ERR_MEMORY;
    }
    vkMapMemory(ctx.device, m->qparam_staging_mem, 0, qparam_bytes, 0,
                &m->qparam_staging_ptr);
    // Init scale=1.0, zp=0.0
    float *qp = (float *)m->qparam_staging_ptr;
    for (uint32_t g = 0; g < m->n_groups; g++) {
      qp[g * 2] = 1.0f;
      qp[g * 2 + 1] = 0.0f;
    }
    cmd_begin();
    VkBufferCopy qcopy = {.size = qparam_bytes};
    vkCmdCopyBuffer(ctx.cmd, m->qparam_staging, m->qparam_buf, 1, &qcopy);
    cmd_submit();
  }

  m->active = 1;
  return ADAMAH_OK;
}

int map_destroy(uint32_t id) {
  if (id >= MAX_MAPS || !ctx.maps[id].active)
    return ADAMAH_ERR_INVALID;
  Map *m = &ctx.maps[id];

  if (m->staging != VK_NULL_HANDLE) {
    vkUnmapMemory(ctx.device, m->staging_mem);
    vkDestroyBuffer(ctx.device, m->staging, NULL);
    vkFreeMemory(ctx.device, m->staging_mem, NULL);
  }
  vkDestroyBuffer(ctx.device, m->buf, NULL);
  vkFreeMemory(ctx.device, m->mem, NULL);

  // Cleanup q8 params
  if (m->qparam_buf != VK_NULL_HANDLE) {
    if (m->qparam_staging_ptr)
      vkUnmapMemory(ctx.device, m->qparam_staging_mem);
    if (m->qparam_staging != VK_NULL_HANDLE) {
      vkDestroyBuffer(ctx.device, m->qparam_staging, NULL);
      vkFreeMemory(ctx.device, m->qparam_staging_mem, NULL);
    }
    vkDestroyBuffer(ctx.device, m->qparam_buf, NULL);
    vkFreeMemory(ctx.device, m->qparam_mem, NULL);
  }

  memset(m, 0, sizeof(Map));
  return ADAMAH_OK;
}

uint64_t map_size(uint32_t id) {
  if (id >= MAX_MAPS || !ctx.maps[id].active)
    return 0;
  return ctx.maps[id].n_packs;
}

// ============================================
// Scatter / Gather (CPU <-> Map)
// ============================================

static int locs_contiguous_in_range(const uint32_t *locs, uint32_t n_locs,
                                    uint32_t n_packs, uint32_t *start_out) {
  if (n_locs == 0)
    return 0;
  uint32_t start = locs[0];
  if (start >= n_packs)
    return 0;
  if ((uint64_t)start + (uint64_t)n_locs > (uint64_t)n_packs)
    return 0;
  for (uint32_t i = 1; i < n_locs; i++) {
    if (locs[i] != start + i)
      return 0;
  }
  if (start_out)
    *start_out = start;
  return 1;
}

// Forward declarations
static GpuBuf *get_or_create_buf_ex(const char *base_name, uint32_t n_elems,
                                    uint32_t elem_size, int device_local,
                                    VkBufferUsageFlags usage);
static GpuBuf *get_or_create_buf(const char *base_name, uint32_t n_elems,
                                 uint32_t elem_size);
static int init_pipelines(void);

// Scatter: write data to map at locations
// locs: array of pack indices (uint32)
// data: packed data (n_locs * pack_size * word_size bytes)
int map_scatter(uint32_t map_id, const uint32_t *locs, const void *data,
                uint32_t n_locs) {
  // Auto-flush pending fusion ops before writing new data
  if (fusion.enabled && fusion.n_ops > 0) {
    adamah_fusion_flush();
  }

  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return ADAMAH_ERR_INVALID;
  Map *m = &ctx.maps[map_id];
  uint32_t pack_bytes = m->word_size * m->pack_size;
  const uint32_t gpu_threshold = 64;

  if (n_locs == 0)
    return ADAMAH_OK;

  // Q4/Q6: the staging buffer stores nibbles/6-bit packed data (< 1 byte/elem),
  // so the generic CPU memcpy path (which uses element indices as byte offsets)
  // would overflow the buffer. These dtypes must always go through the GPU
  // quantization scatter shader which takes F32 input and packs nibbles.
  if (m->dtype == DTYPE_Q4 || m->dtype == DTYPE_Q6) {
    if (!ctx.unary_pipe.pipeline)
      init_pipelines();
    if (!ctx.dtype_pipes_loaded[m->dtype])
      init_dtype_pipelines(m->dtype);
    Pipeline *qpipe = &ctx.dtype_scatter_pipe[m->dtype];
    if (!qpipe->pipeline || !m->qparam_buf)
      return ADAMAH_ERR_INVALID;

    // F32 source data: 4 bytes per element
    VkDeviceSize f32_bytes  = (VkDeviceSize)n_locs * sizeof(float);
    VkDeviceSize locs_bytes = (VkDeviceSize)n_locs * sizeof(uint32_t);
    VkDeviceSize locs_off   = align_up(f32_bytes, ctx.copy_align);
    VkDeviceSize upload_sz  = locs_off + locs_bytes;

    VkBufferUsageFlags dev_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    GpuBuf *src_buf  = get_or_create_buf_ex("_q4_src",  n_locs, 4, 1, dev_usage);
    GpuBuf *locs_buf = get_or_create_buf_ex("_q4_locs", n_locs, 4, 1, dev_usage);
    if (!src_buf || !locs_buf)
      return ADAMAH_ERR_MEMORY;

    GpuBuf *stage = get_or_create_buf_ex(
        "_stage_upload", (uint32_t)upload_sz, 1, 0,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (!stage || !stage->ptr)
      return ADAMAH_ERR_MEMORY;

    memcpy(stage->ptr, data, (size_t)f32_bytes);
    memcpy((char *)stage->ptr + locs_off, locs, (size_t)locs_bytes);

    VkDescriptorBufferInfo infos[4] = {
        {.buffer = m->buf,        .range = VK_WHOLE_SIZE},
        {.buffer = src_buf->buf,  .range = VK_WHOLE_SIZE},
        {.buffer = locs_buf->buf, .range = VK_WHOLE_SIZE},
        {.buffer = m->qparam_buf, .range = VK_WHOLE_SIZE}};
    VkWriteDescriptorSet writes[4];
    VkDescriptorSet ds = alloc_desc_set(qpipe);
    if (!ds)
      return ADAMAH_ERR_MEMORY;
    for (int i = 0; i < 4; i++) {
      writes[i] = (VkWriteDescriptorSet){
          .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet          = ds,
          .dstBinding      = (uint32_t)i,
          .descriptorCount = 1,
          .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo     = &infos[i]};
    }
    vkUpdateDescriptorSets(ctx.device, 4, writes, 0, NULL);

    uint32_t push[3]      = {n_locs, m->pack_size, m->group_size};
    uint32_t total_threads = n_locs * m->pack_size;

    cmd_begin();
    VkBufferCopy copies[2] = {
        {.srcOffset = 0,        .dstOffset = 0, .size = f32_bytes},
        {.srcOffset = locs_off, .dstOffset = 0, .size = locs_bytes}};
    vkCmdCopyBuffer(ctx.cmd, stage->buf, src_buf->buf,  1, &copies[0]);
    vkCmdCopyBuffer(ctx.cmd, stage->buf, locs_buf->buf, 1, &copies[1]);
    VkBuffer trans_bufs[2] = {src_buf->buf, locs_buf->buf};
    cmd_buffer_barrier(
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        trans_bufs, 2);
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, qpipe->pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            qpipe->pipe_layout, 0, 1, &ds, 0, NULL);
    vkCmdPushConstants(ctx.cmd, qpipe->pipe_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push);
    vkCmdDispatch(ctx.cmd, (total_threads + 255) / 256, 1, 1);
    cmd_submit();
    return ADAMAH_OK;
  }

  // Fast path: contiguous locs - direct DMA
  uint32_t start = 0;
  if (locs_contiguous_in_range(locs, n_locs, m->n_packs, &start)) {
    debug_path("scatter", "contiguous");
    size_t off = (size_t)start * pack_bytes;
    size_t size = (size_t)n_locs * pack_bytes;

#ifdef ADAMAH_PROFILE
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
#endif

    // Chunk through the shared staging pool (handles maps larger than pool).
    size_t done = 0;
    while (done < size) {
      size_t chunk = (size - done < (size_t)ctx.pool_size)
                         ? (size - done)
                         : (size_t)ctx.pool_size;
      memcpy(ctx.pool_ptr, (const char *)data + done, chunk);
      if (!(ctx.pool_mem_props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        VkMappedMemoryRange flush_range = {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = ctx.pool_mem,
            .offset = 0,
            .size = chunk};
        vkFlushMappedMemoryRanges(ctx.device, 1, &flush_range);
      }
      cmd_begin();
      VkBufferCopy copy = {.srcOffset = 0, .dstOffset = off + done, .size = chunk};
      vkCmdCopyBuffer(ctx.cmd, ctx.pool_buf, m->buf, 1, &copy);
      cmd_submit();
      done += chunk;
    }

#ifdef ADAMAH_PROFILE
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total_ms =
        (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    double bandwidth_gbs = (size / 1e9) / (total_ms / 1000.0);
    fprintf(stderr,
            "[SCATTER] size=%zu bytes (%.2f MB), total=%.3fms, BW=%.2f GB/s\n",
            size, size / 1e6, total_ms, bandwidth_gbs);
#endif

    return ADAMAH_OK;
  }

  // GPU scatter path: use compute shader for sparse writes
  // This is faster than many small DMA copies for sparse data
  if (!ctx.scatter_pipe.pipeline)
    init_pipelines(); // Ensure pipeline is ready
  if (ctx.scatter_pipe.pipeline && n_locs >= gpu_threshold) {
    debug_path("scatter", "gpu_sparse");
    VkDeviceSize data_size = (VkDeviceSize)n_locs * pack_bytes;
    VkDeviceSize locs_size = (VkDeviceSize)n_locs * sizeof(uint32_t);
    VkDeviceSize locs_offset = align_up(data_size, ctx.copy_align);
    VkDeviceSize upload_size = locs_offset + locs_size;

    // Device-local buffers
    VkBufferUsageFlags dev_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    GpuBuf *src_buf = get_or_create_buf_ex(
        "_scatter_src", n_locs * m->pack_size, m->word_size, 1, dev_usage);
    GpuBuf *locs_buf =
        get_or_create_buf_ex("_scatter_locs", n_locs, 4, 1, dev_usage);
    if (!src_buf || !locs_buf)
      goto fallback;

    // Staging upload buffer
    GpuBuf *stage_up = get_or_create_buf_ex(
        "_stage_upload", (uint32_t)upload_size, 1, 0,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (!stage_up || !stage_up->ptr)
      goto fallback;

    memcpy((char *)stage_up->ptr, data, (size_t)data_size);
    memcpy((char *)stage_up->ptr + (size_t)locs_offset, locs,
           (size_t)locs_size);

    // Update descriptor set
    VkDescriptorBufferInfo buf_infos[3] = {
        {.buffer = m->buf, .range = VK_WHOLE_SIZE},
        {.buffer = src_buf->buf, .range = VK_WHOLE_SIZE},
        {.buffer = locs_buf->buf, .range = VK_WHOLE_SIZE}};
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
      writes[i] = (VkWriteDescriptorSet){
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = ctx.scatter_pipe.desc_set,
          .dstBinding = i,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buf_infos[i]};
    }
    vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);

    uint32_t push[2] = {n_locs, m->pack_size};
    uint32_t total_threads = n_locs * m->pack_size;

    cmd_begin();
    VkBufferCopy copies[2] = {
        {.srcOffset = 0, .dstOffset = 0, .size = data_size},
        {.srcOffset = locs_offset, .dstOffset = 0, .size = locs_size}};
    vkCmdCopyBuffer(ctx.cmd, stage_up->buf, src_buf->buf, 1, &copies[0]);
    vkCmdCopyBuffer(ctx.cmd, stage_up->buf, locs_buf->buf, 1, &copies[1]);
    VkBuffer trans_bufs[2] = {src_buf->buf, locs_buf->buf};
    cmd_buffer_barrier(
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, trans_bufs, 2);

    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      ctx.scatter_pipe.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx.scatter_pipe.pipe_layout, 0, 1,
                            &ctx.scatter_pipe.desc_set, 0, NULL);
    vkCmdPushConstants(ctx.cmd, ctx.scatter_pipe.pipe_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push);
    vkCmdDispatch(ctx.cmd, (total_threads + 255) / 256, 1, 1);
    cmd_submit();

    return ADAMAH_OK;
  }

fallback:
  // Pool-based fallback: batch elements through the shared pool.
  debug_path("scatter", "fallback");
  {
    const char *src = (const char *)data;
    VkBufferCopy *regions =
        (VkBufferCopy *)malloc(sizeof(VkBufferCopy) * n_locs);
    if (!regions)
      return ADAMAH_ERR_MEMORY;

    uint32_t pool_cap = (uint32_t)((size_t)ctx.pool_size / pack_bytes);
    if (pool_cap == 0)
      pool_cap = 1;

    uint32_t i = 0;
    while (i < n_locs) {
      uint32_t batch = (n_locs - i < pool_cap) ? (n_locs - i) : pool_cap;
      uint32_t rcount = 0;
      size_t pool_off = 0;

      for (uint32_t j = 0; j < batch; j++) {
        uint32_t loc = locs[i + j];
        if (loc >= m->n_packs)
          continue;
        memcpy((char *)ctx.pool_ptr + pool_off,
               src + (size_t)(i + j) * pack_bytes, pack_bytes);
        regions[rcount].srcOffset = pool_off;
        regions[rcount].dstOffset = (size_t)loc * pack_bytes;
        regions[rcount].size = pack_bytes;
        rcount++;
        pool_off += pack_bytes;
      }

      if (rcount > 0) {
        if (!(ctx.pool_mem_props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
          VkMappedMemoryRange flush_range = {
              .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
              .memory = ctx.pool_mem,
              .offset = 0,
              .size = pool_off};
          vkFlushMappedMemoryRanges(ctx.device, 1, &flush_range);
        }
        cmd_begin();
        vkCmdCopyBuffer(ctx.cmd, ctx.pool_buf, m->buf, rcount, regions);
        cmd_submit();
      }
      i += batch;
    }
    free(regions);
  }
  return ADAMAH_OK;
}

// Gather: read data from map at locations
int map_gather(uint32_t map_id, const uint32_t *locs, void *data,
               uint32_t n_locs) {
  // Auto-flush pending fusion ops before reading
  if (fusion.enabled && fusion.n_ops > 0) {
    adamah_fusion_flush();
  }

  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return ADAMAH_ERR_INVALID;
  Map *m = &ctx.maps[map_id];
  uint32_t pack_bytes = m->word_size * m->pack_size;
  const uint32_t gpu_threshold = 64;

  if (n_locs == 0)
    return ADAMAH_OK;

  // Fast path: contiguous locs - direct DMA
  uint32_t start = 0;
  if (locs_contiguous_in_range(locs, n_locs, m->n_packs, &start)) {
    debug_path("gather", "contiguous");
    size_t off = (size_t)start * pack_bytes;
    size_t size = (size_t)n_locs * pack_bytes;

#ifdef ADAMAH_PROFILE
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
#endif

    // Chunk through the shared staging pool (handles maps larger than pool).
    size_t done = 0;
    while (done < size) {
      size_t chunk = (size - done < (size_t)ctx.pool_size)
                         ? (size - done)
                         : (size_t)ctx.pool_size;
      cmd_begin();
      VkBufferCopy copy = {.srcOffset = off + done, .dstOffset = 0, .size = chunk};
      vkCmdCopyBuffer(ctx.cmd, m->buf, ctx.pool_buf, 1, &copy);
      cmd_submit();
      if (!(ctx.pool_mem_props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        VkMappedMemoryRange inv_range = {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = ctx.pool_mem,
            .offset = 0,
            .size = chunk};
        vkInvalidateMappedMemoryRanges(ctx.device, 1, &inv_range);
      }
      memcpy((char *)data + done, ctx.pool_ptr, chunk);
      done += chunk;
    }

#ifdef ADAMAH_PROFILE
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total_ms =
        (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    double bandwidth_gbs = (size / 1e9) / (total_ms / 1000.0);
    fprintf(stderr,
            "[GATHER] size=%zu bytes (%.2f MB), total=%.3fms, BW=%.2f GB/s\n",
            size, size / 1e6, total_ms, bandwidth_gbs);
#endif

    return ADAMAH_OK;
  }

  // GPU gather path: use compute shader for sparse reads
  if (!ctx.gather_pipe.pipeline)
    init_pipelines(); // Ensure pipeline is ready
  if (ctx.gather_pipe.pipeline && n_locs >= gpu_threshold) {
    debug_path("gather", "gpu_sparse");
    VkDeviceSize data_size = (VkDeviceSize)n_locs * pack_bytes;
    VkDeviceSize locs_size = (VkDeviceSize)n_locs * sizeof(uint32_t);

    VkBufferUsageFlags dev_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    GpuBuf *dst_buf = get_or_create_buf_ex("_gather_dst", n_locs * m->pack_size,
                                           m->word_size, 1, dev_usage);
    GpuBuf *locs_buf =
        get_or_create_buf_ex("_gather_locs", n_locs, 4, 1, dev_usage);
    if (!dst_buf || !locs_buf)
      goto fallback;

    GpuBuf *stage_up = get_or_create_buf_ex(
        "_stage_upload", n_locs * 4, 1, 0,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    GpuBuf *stage_down = get_or_create_buf_ex(
        "_stage_download", (uint32_t)data_size, 1, 0,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (!stage_up || !stage_down || !stage_up->ptr || !stage_down->ptr)
      goto fallback;

    memcpy(stage_up->ptr, locs, (size_t)locs_size);

    VkDescriptorBufferInfo buf_infos[3] = {
        {.buffer = m->buf, .range = VK_WHOLE_SIZE},
        {.buffer = dst_buf->buf, .range = VK_WHOLE_SIZE},
        {.buffer = locs_buf->buf, .range = VK_WHOLE_SIZE}};
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
      writes[i] = (VkWriteDescriptorSet){
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = ctx.gather_pipe.desc_set,
          .dstBinding = i,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buf_infos[i]};
    }
    vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);

    uint32_t push[2] = {n_locs, m->pack_size};
    uint32_t total_threads = n_locs * m->pack_size;

    cmd_begin();
    VkBufferCopy locs_copy = {
        .srcOffset = 0, .dstOffset = 0, .size = locs_size};
    vkCmdCopyBuffer(ctx.cmd, stage_up->buf, locs_buf->buf, 1, &locs_copy);
    VkBuffer trans_bufs[1] = {locs_buf->buf};
    cmd_buffer_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_ACCESS_SHADER_READ_BIT, trans_bufs, 1);

    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      ctx.gather_pipe.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx.gather_pipe.pipe_layout, 0, 1,
                            &ctx.gather_pipe.desc_set, 0, NULL);
    vkCmdPushConstants(ctx.cmd, ctx.gather_pipe.pipe_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push);
    vkCmdDispatch(ctx.cmd, (total_threads + 255) / 256, 1, 1);

    VkBuffer comp_bufs[1] = {dst_buf->buf};
    cmd_buffer_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_ACCESS_SHADER_WRITE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_ACCESS_TRANSFER_READ_BIT, comp_bufs, 1);
    VkBufferCopy dst_copy = {.srcOffset = 0, .dstOffset = 0, .size = data_size};
    vkCmdCopyBuffer(ctx.cmd, dst_buf->buf, stage_down->buf, 1, &dst_copy);
    cmd_submit();

    memcpy(data, stage_down->ptr, (size_t)data_size);

    return ADAMAH_OK;
  }

fallback:
  // Pool-based fallback: batch download through shared pool.
  debug_path("gather", "fallback");
  {
    char *dst = (char *)data;
    VkBufferCopy *regions =
        (VkBufferCopy *)malloc(sizeof(VkBufferCopy) * n_locs);
    if (!regions)
      return ADAMAH_ERR_MEMORY;

    uint32_t pool_cap = (uint32_t)((size_t)ctx.pool_size / pack_bytes);
    if (pool_cap == 0)
      pool_cap = 1;

    uint32_t i = 0;
    while (i < n_locs) {
      uint32_t batch = (n_locs - i < pool_cap) ? (n_locs - i) : pool_cap;
      uint32_t rcount = 0;

      for (uint32_t j = 0; j < batch; j++) {
        uint32_t loc = locs[i + j];
        if (loc >= m->n_packs)
          continue;
        regions[rcount].srcOffset = (size_t)loc * pack_bytes;
        regions[rcount].dstOffset = (size_t)rcount * pack_bytes;
        regions[rcount].size = pack_bytes;
        rcount++;
      }

      if (rcount > 0) {
        cmd_begin();
        vkCmdCopyBuffer(ctx.cmd, m->buf, ctx.pool_buf, rcount, regions);
        cmd_submit();
        if (!(ctx.pool_mem_props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
          VkMappedMemoryRange inv_range = {
              .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
              .memory = ctx.pool_mem,
              .offset = 0,
              .size = (size_t)rcount * pack_bytes};
          vkInvalidateMappedMemoryRanges(ctx.device, 1, &inv_range);
        }
        // Unpack pool slots (contiguous) to output positions.
        uint32_t pc = 0;
        for (uint32_t j = 0; j < batch; j++) {
          uint32_t loc = locs[i + j];
          if (loc >= m->n_packs)
            continue;
          memcpy(dst + (size_t)(i + j) * pack_bytes,
                 (char *)ctx.pool_ptr + (size_t)pc * pack_bytes, pack_bytes);
          pc++;
        }
      }
      i += batch;
    }
    free(regions);
  }
  return ADAMAH_OK;
}

// ============================================
// Device-only async sparse I/O
// ============================================

uint64_t map_upload_dev(uint32_t handle, const void *data, uint32_t n_bytes) {
  if (!data || n_bytes == 0)
    return 0;
  if (!ctx.hot_pool || !ctx.cold_pool) {
    if (debug_enabled())
      fprintf(stderr, "ADAMAH DEBUG: upload_dev missing cache pools\n");
    return 0;
  }

  uint32_t id = handle;
  if (id == 0) {
    if (res_alloc(RES_TYPE_CVAR, n_bytes, &id) != 0) {
      if (debug_enabled())
        fprintf(stderr,
                "ADAMAH DEBUG: upload_dev res_alloc failed (bytes=%u)\n",
                n_bytes);
      return 0;
    }
  }
  ResEntry *r = res_get(id);
  if (!r) {
    if (debug_enabled())
      fprintf(stderr, "ADAMAH DEBUG: upload_dev res_get failed (id=%u)\n", id);
    return 0;
  }
  if (r->size_bytes != n_bytes) {
    if (debug_enabled())
      fprintf(
          stderr,
          "ADAMAH DEBUG: upload_dev size mismatch (id=%u size=%zu bytes=%u)\n",
          id, (size_t)r->size_bytes, n_bytes);
    return 0;
  }

  VkDeviceSize hot_off = 0;
  r->dirty = 1;
  if (res_require_hot(id, &hot_off) != 0) {
    if (debug_enabled())
      fprintf(stderr,
              "ADAMAH DEBUG: upload_dev res_require_hot failed (id=%u)\n", id);
    return 0;
  }

  VkDeviceSize copy_size = align_up((VkDeviceSize)n_bytes, ctx.copy_align);
  if (copy_size > ctx.hot_pool->bytes_capacity)
    copy_size = (VkDeviceSize)n_bytes;
  GpuBuf *stage = get_or_create_buf_ex(
      "_stage_upload", (uint32_t)copy_size, 1, 0,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  if (!stage || !stage->ptr) {
    if (debug_enabled())
      fprintf(stderr, "ADAMAH DEBUG: upload_dev staging alloc failed\n");
    return 0;
  }

  memcpy(stage->ptr, data, n_bytes);
  if (copy_size > n_bytes) {
    memset((char *)stage->ptr + n_bytes, 0, (size_t)(copy_size - n_bytes));
  }

  cmd_begin();
  VkBufferCopy copy = {.srcOffset = 0, .dstOffset = hot_off, .size = copy_size};
  vkCmdCopyBuffer(ctx.cmd, stage->buf, ctx.hot_pool->buf, 1, &copy);
  VkBuffer trans_bufs[1] = {ctx.hot_pool->buf};
  cmd_buffer_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_ACCESS_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_SHADER_READ_BIT, trans_bufs, 1);
  cmd_submit();

  return (uint64_t)id;
}

int map_download_dev(uint32_t handle, void *data, uint32_t n_bytes) {
  if (!data || n_bytes == 0)
    return ADAMAH_ERR_INVALID;
  if (!ctx.hot_pool || !ctx.cold_pool)
    return ADAMAH_ERR_INVALID;
  ResEntry *r = res_get(handle);
  if (!r)
    return ADAMAH_ERR_INVALID;
  if (n_bytes > r->size_bytes)
    return ADAMAH_ERR_INVALID;

  VkDeviceSize hot_off = 0;
  if (res_require_hot(handle, &hot_off) != 0)
    return ADAMAH_ERR_INVALID;

  VkDeviceSize copy_size = align_up((VkDeviceSize)n_bytes, ctx.copy_align);
  GpuBuf *stage = get_or_create_buf_ex(
      "_stage_download", (uint32_t)copy_size, 1, 0,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  if (!stage || !stage->ptr)
    return ADAMAH_ERR_MEMORY;

  cmd_begin();
  VkBuffer comp_bufs[1] = {ctx.hot_pool->buf};
  cmd_buffer_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_ACCESS_TRANSFER_READ_BIT, comp_bufs, 1);
  VkBufferCopy copy = {.srcOffset = hot_off, .dstOffset = 0, .size = copy_size};
  vkCmdCopyBuffer(ctx.cmd, ctx.hot_pool->buf, stage->buf, 1, &copy);
  cmd_submit();

  memcpy(data, stage->ptr, n_bytes);
  return ADAMAH_OK;
}

uint64_t map_scatter_dev(uint32_t map_id, uint32_t locs_handle, uint32_t n_locs,
                         uint32_t src_handle) {
  if (fusion_flush_pending_for_immediate() != ADAMAH_OK)
    return 0;
  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return 0;
  if (n_locs == 0)
    return 0;
  Map *m = &ctx.maps[map_id];
  if (!ctx.scatter_pipe.pipeline)
    init_pipelines();
  if (!ctx.scatter_pipe.pipeline)
    return 0;

  ResEntry *src_res = res_get(src_handle);
  ResEntry *locs_res = res_get(locs_handle);
  if (!src_res || !locs_res)
    return 0;
  if ((VkDeviceSize)n_locs * 4 > locs_res->size_bytes)
    return 0;
  VkDeviceSize needed_src = (VkDeviceSize)n_locs * m->pack_size * m->word_size;
  if (needed_src > src_res->size_bytes)
    return 0;

  res_pin(src_handle);
  res_pin(locs_handle);
  VkDeviceSize src_off = 0;
  VkDeviceSize locs_off = 0;
  if (res_require_hot(src_handle, &src_off) != 0 ||
      res_require_hot(locs_handle, &locs_off) != 0) {
    res_unpin(src_handle);
    res_unpin(locs_handle);
    return 0;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.scatter_pipe);
  if (!ds)
    return 0;

  VkDescriptorBufferInfo buf_infos[3] = {
      {.buffer = m->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf,
       .offset = src_off,
       .range = src_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = locs_off,
       .range = locs_res->size_bytes}};
  VkWriteDescriptorSet writes[3];
  for (int i = 0; i < 3; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);

  uint32_t push[2] = {n_locs, m->pack_size};
  uint32_t total_threads = n_locs * m->pack_size;

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.scatter_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.scatter_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.scatter_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push);
  vkCmdDispatch(ctx.cmd, (total_threads + 255) / 256, 1, 1);

  VkBuffer comp_bufs[1] = {m->buf};
  cmd_buffer_barrier(
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, comp_bufs, 1);

  cmd_submit();
  res_unpin(src_handle);
  res_unpin(locs_handle);
  return 0;
}

uint64_t map_gather_dev(uint32_t map_id, uint32_t locs_handle,
                        uint32_t n_locs) {
  if (fusion_flush_pending_for_immediate() != ADAMAH_OK)
    return 0;
  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return 0;
  if (n_locs == 0)
    return 0;
  Map *m = &ctx.maps[map_id];
  Pipeline *gpipe = NULL;
  uint32_t out_word_size = 4; // device gather always materializes logical values as f32
  int needs_qparams = 0;

  if (!ctx.gather_pipe.pipeline)
    init_pipelines();
  if (m->dtype != DTYPE_F32 && !ctx.dtype_pipes_loaded[m->dtype]) {
    if (init_dtype_pipelines(m->dtype) != ADAMAH_OK)
      return 0;
  }

  if (m->dtype == DTYPE_F32) {
    gpipe = &ctx.gather_pipe;
  } else if (m->dtype == DTYPE_BF16) {
    gpipe = &ctx.dtype_gather_pipe[m->dtype];
  } else if (m->dtype == DTYPE_Q8 || m->dtype == DTYPE_Q4 ||
             m->dtype == DTYPE_Q6) {
    gpipe = &ctx.dtype_gather_pipe[m->dtype];
    needs_qparams = 1;
    if (m->qparam_buf == VK_NULL_HANDLE)
      return 0;
  }

  if (!gpipe || !gpipe->pipeline)
    return 0;

  ResEntry *locs_res = res_get(locs_handle);
  if (!locs_res)
    return 0;
  if ((VkDeviceSize)n_locs * 4 > locs_res->size_bytes)
    return 0;

  res_pin(locs_handle);
  VkDeviceSize locs_off = 0;
  if (res_require_hot(locs_handle, &locs_off) != 0) {
    res_unpin(locs_handle);
    return 0;
  }

  uint32_t dst_id = 0;
  uint32_t dst_bytes = n_locs * m->pack_size * out_word_size;
  if (res_alloc(RES_TYPE_CVAR, dst_bytes, &dst_id) != 0)
    return 0;
  ResEntry *dst_res = res_get(dst_id);
  if (!dst_res)
    return 0;
  dst_res->dirty = 1;

  res_pin(dst_id);
  VkDeviceSize dst_off = 0;
  if (res_require_hot(dst_id, &dst_off) != 0) {
    res_unpin(locs_handle);
    res_unpin(dst_id);
    return 0;
  }

  VkDescriptorSet ds = alloc_desc_set(gpipe);
  if (!ds)
    return 0;

  uint32_t push_small[2] = {n_locs, m->pack_size};
  uint32_t push_q[3] = {n_locs, m->pack_size, m->group_size};
  uint32_t total_threads = n_locs * m->pack_size;

  if (needs_qparams) {
    VkDescriptorBufferInfo buf_infos[4] = {
        {.buffer = m->buf, .range = VK_WHOLE_SIZE},
        {.buffer = ctx.hot_pool->buf,
         .offset = dst_off,
         .range = dst_res->size_bytes},
        {.buffer = ctx.hot_pool->buf,
         .offset = locs_off,
         .range = locs_res->size_bytes},
        {.buffer = m->qparam_buf, .range = VK_WHOLE_SIZE}};
    VkWriteDescriptorSet writes[4];
    for (int i = 0; i < 4; i++) {
      writes[i] = (VkWriteDescriptorSet){
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = ds,
          .dstBinding = i,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buf_infos[i]};
    }
    vkUpdateDescriptorSets(ctx.device, 4, writes, 0, NULL);
  } else {
    VkDescriptorBufferInfo buf_infos[3] = {
        {.buffer = m->buf, .range = VK_WHOLE_SIZE},
        {.buffer = ctx.hot_pool->buf,
         .offset = dst_off,
         .range = dst_res->size_bytes},
        {.buffer = ctx.hot_pool->buf,
         .offset = locs_off,
         .range = locs_res->size_bytes}};
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
      writes[i] = (VkWriteDescriptorSet){
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = ds,
          .dstBinding = i,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &buf_infos[i]};
    }
    vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);
  }

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    gpipe->pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          gpipe->pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, gpipe->pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     needs_qparams ? 12 : 8,
                     needs_qparams ? (void *)push_q : (void *)push_small);
  vkCmdDispatch(ctx.cmd, (total_threads + 255) / 256, 1, 1);

  VkBuffer comp_bufs[1] = {ctx.hot_pool->buf};
  cmd_buffer_barrier(
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, comp_bufs, 1);

  cmd_submit();
  res_unpin(locs_handle);
  res_unpin(dst_id);
  return (uint64_t)dst_id;
}

// Backward-compatible wrappers
int mscatter(uint32_t map_id, const uint32_t *locs, const void *data,
             uint32_t n_locs) {
  return map_scatter(map_id, locs, data, n_locs);
}

int mgather(uint32_t map_id, const uint32_t *locs, void *data,
            uint32_t n_locs) {
  return map_gather(map_id, locs, data, n_locs);
}

// ============================================
// ============================================
// Persistence
// ============================================

int map_save(uint32_t id, const char *path) {
  if (id >= MAX_MAPS || !ctx.maps[id].active)
    return ADAMAH_ERR_INVALID;
  Map *m = &ctx.maps[id];

  FILE *f = fopen(path, "wb");
  if (!f)
    return ADAMAH_ERR_INVALID;
  fwrite(&m->word_size, 4, 1, f);
  fwrite(&m->pack_size, 4, 1, f);
  fwrite(&m->n_packs, 4, 1, f);

  // Download in pool-sized chunks.
  size_t total = m->total_bytes;
  size_t done = 0;
  while (done < total) {
    size_t chunk = (total - done < (size_t)ctx.pool_size)
                       ? (total - done)
                       : (size_t)ctx.pool_size;
    cmd_begin();
    VkBufferCopy copy = {.srcOffset = done, .dstOffset = 0, .size = chunk};
    vkCmdCopyBuffer(ctx.cmd, m->buf, ctx.pool_buf, 1, &copy);
    cmd_submit();
    if (!(ctx.pool_mem_props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      VkMappedMemoryRange inv_range = {
          .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
          .memory = ctx.pool_mem,
          .offset = 0,
          .size = chunk};
      vkInvalidateMappedMemoryRanges(ctx.device, 1, &inv_range);
    }
    fwrite(ctx.pool_ptr, 1, chunk, f);
    done += chunk;
  }
  fclose(f);
  return ADAMAH_OK;
}

int map_load(uint32_t id, const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return ADAMAH_ERR_INVALID;

  uint32_t ws, ps, np;
  if (fread(&ws, 4, 1, f) != 1 || fread(&ps, 4, 1, f) != 1 ||
      fread(&np, 4, 1, f) != 1) {
    fclose(f);
    return ADAMAH_ERR_INVALID;
  }

  int ret = map_init(id, ws, ps, np);
  if (ret != ADAMAH_OK) {
    fclose(f);
    return ret;
  }

  Map *m = &ctx.maps[id];
  if (m->total_bytes > 0) {
    size_t got = fread(m->staging_ptr, 1, m->total_bytes, f);
    if (got != (size_t)m->total_bytes) {
      fclose(f);
      return ADAMAH_ERR_INVALID;
    }
  }
  fclose(f);

  // Upload
  cmd_begin();
  VkBufferCopy copy = {.size = m->total_bytes};
  vkCmdCopyBuffer(ctx.cmd, m->staging, m->buf, 1, &copy);
  cmd_submit();

  return ADAMAH_OK;
}

// ============================================
// Shader Loading & Pipeline Creation
// ============================================

static uint32_t *load_spv(const char *name, size_t *size) {
  char path[600];
  snprintf(path, sizeof(path), "%s/%s", ctx.shader_path, name);

  FILE *f = fopen(path, "rb");
  if (!f)
    return NULL;

  fseek(f, 0, SEEK_END);
  long end = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (end <= 0) {
    fclose(f);
    return NULL;
  }
  *size = (size_t)end;

  uint32_t *code = malloc(*size);
  if (!code) {
    fclose(f);
    return NULL;
  }
  size_t got = fread(code, 1, *size, f);
  if (got != *size) {
    free(code);
    fclose(f);
    return NULL;
  }
  fclose(f);
  return code;
}

static int create_pipeline(Pipeline *p, const char *shader_name,
                           int num_bindings, size_t push_size) {
  size_t code_size;
  uint32_t *code = load_spv(shader_name, &code_size);
  if (!code)
    return ADAMAH_ERR_INVALID;

  // Create shader module
  VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = code_size,
      .pCode = code};
  VkResult res = vkCreateShaderModule(ctx.device, &smci, NULL, &p->shader);
  free(code);
  if (res != VK_SUCCESS)
    return ADAMAH_ERR_VULKAN;

  // Descriptor set layout
  VkDescriptorSetLayoutBinding bindings[16]; // Enough for current custom pipelines
  if (num_bindings > 16)
    return ADAMAH_ERR_INVALID;
  for (int i = 0; i < num_bindings; i++) {
    bindings[i] = (VkDescriptorSetLayoutBinding){
        .binding = i,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
  }
  VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = num_bindings,
      .pBindings = bindings};
  vkCreateDescriptorSetLayout(ctx.device, &dslci, NULL, &p->desc_layout);

  // Pipeline layout
  VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                             .size = push_size};
  VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &p->desc_layout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pcr};
  vkCreatePipelineLayout(ctx.device, &plci, NULL, &p->pipe_layout);

  // Compute pipeline
  VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = p->shader,
                .pName = "main"},
      .layout = p->pipe_layout};
  vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpci, NULL,
                           &p->pipeline);

  // Descriptor pool
  const uint32_t max_sets = 8192;
  VkDescriptorPoolSize pool_size = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                    .descriptorCount =
                                        (uint32_t)num_bindings * max_sets};
  VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = max_sets,
      .poolSizeCount = 1,
      .pPoolSizes = &pool_size};
  vkCreateDescriptorPool(ctx.device, &dpci, NULL, &p->desc_pool);

  // Allocate descriptor set
  VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = p->desc_pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &p->desc_layout};
  vkAllocateDescriptorSets(ctx.device, &dsai, &p->desc_set);

  return ADAMAH_OK;
}

static void destroy_pipeline(Pipeline *p) {
  if (p->pipeline)
    vkDestroyPipeline(ctx.device, p->pipeline, NULL);
  if (p->pipe_layout)
    vkDestroyPipelineLayout(ctx.device, p->pipe_layout, NULL);
  if (p->desc_pool)
    vkDestroyDescriptorPool(ctx.device, p->desc_pool, NULL);
  if (p->desc_layout)
    vkDestroyDescriptorSetLayout(ctx.device, p->desc_layout, NULL);
  if (p->shader)
    vkDestroyShaderModule(ctx.device, p->shader, NULL);
  memset(p, 0, sizeof(Pipeline));
}

// Get directory of the shared library itself
static void get_lib_dir(char *buf, size_t size) {
#ifdef __linux__
  Dl_info info;
  if (dladdr((void *)get_lib_dir, &info) && info.dli_fname) {
    strncpy(buf, info.dli_fname, size - 1);
    buf[size - 1] = '\0';
    char *last_slash = strrchr(buf, '/');
    if (last_slash)
      *last_slash = '\0';
    else
      buf[0] = '\0';
  } else {
    buf[0] = '\0';
  }
#else
  buf[0] = '\0';
#endif
}

static int init_pipelines(void) {
  // Get library directory for relative shader lookup
  char lib_dir[512] = {0};
  get_lib_dir(lib_dir, sizeof(lib_dir));

  char lib_shaders[600] = {0};
  if (lib_dir[0]) {
    snprintf(lib_shaders, sizeof(lib_shaders), "%s/shaders", lib_dir);
  }

  // Find shader path - check paths in priority order:
  // 1. Compile-time define
  // 2. Relative to library location (for installed packages)
  // 3. ./adamah-MAIN/adamah/shaders (repo root in this workspace)
  // 4. ./adamah/shaders (installed/new structure)
  // 5. ./shaders (legacy)
  // 6. ./src/adamah/shaders (old structure)
  // 7. System path
  const char *paths[] = {
#ifdef SHADER_PATH
      SHADER_PATH,
#endif
      lib_shaders[0] ? lib_shaders : "",
      "./adamah-MAIN/adamah/shaders",
      "./adamah/shaders",
      "./shaders",
      "./src/adamah/shaders",
      "/usr/share/adamah/shaders",
      NULL};

  for (int i = 0; paths[i] != NULL; i++) {
    if (!paths[i][0])
      continue;
    char test[600];
    snprintf(test, sizeof(test), "%s/map_op1.spv", paths[i]);
    FILE *f = fopen(test, "rb");
    if (f) {
      fclose(f);
      strncpy(ctx.shader_path, paths[i], 511);
      break;
    }
  }

  if (!ctx.shader_path[0]) {
    fprintf(stderr, "ADAMAH: Cannot find shaders\n");
    fprintf(stderr, "Searched in:\n");
    for (int i = 0; paths[i] != NULL; i++) {
      if (!paths[i][0])
        continue;
      fprintf(stderr, "  - %s\n", paths[i]);
    }
    return ADAMAH_ERR_INVALID;
  }

  // Unary: op, n_locs, pack_size = 12 bytes, 3 bindings
  if (create_pipeline(&ctx.unary_pipe, "map_op1.spv", 3, 12) != 0)
    return ADAMAH_ERR_VULKAN;

  // Binary: op, n_locs, pack_size = 12 bytes, 4 bindings
  if (create_pipeline(&ctx.binary_pipe, "map_op2.spv", 4, 12) != 0)
    return ADAMAH_ERR_VULKAN;

  // Matmul: M, K, N, n_ops = 16 bytes, 4 bindings (map, locs_a, locs_b, locs_c)
  if (create_pipeline(&ctx.matmul_pipe, "map_matmul.spv", 4, 16) != 0) {
    fprintf(stderr, "ADAMAH: Warning - matmul shader not found\n");
    // Not fatal, continue without matmul
  }

  // Reduce: op, n_locs, pack_size = 12 bytes, 3 bindings
  if (create_pipeline(&ctx.reduce_pipe, "map_reduce.spv", 3, 12) != 0) {
    fprintf(stderr, "ADAMAH: Warning - reduce shader not found\n");
  }
  if (create_pipeline(&ctx.reduce_small_pipe, "map_reduce_small.spv", 3, 12) !=
      0) {
    fprintf(stderr, "ADAMAH: Warning - reduce_small shader not found\n");
  }

  // Broadcast: op, n_locs, pack_size = 12 bytes, 4 bindings
  if (create_pipeline(&ctx.broadcast_pipe, "map_broadcast.spv", 4, 12) != 0) {
    fprintf(stderr, "ADAMAH: Warning - broadcast shader not found\n");
  }

  // Softmax: n_rows, row_size = 8 bytes, 3 bindings
  if (create_pipeline(&ctx.softmax_pipe, "map_softmax.spv", 3, 8) != 0) {
    fprintf(stderr, "ADAMAH: Warning - softmax shader not found\n");
  }
  if (create_pipeline(&ctx.softmax_abs_pipe, "map_softmax_abs.spv", 3, 8) != 0) {
    fprintf(stderr, "ADAMAH: Warning - softmax_abs shader not found\n");
  }
  if (create_pipeline(&ctx.attn_softmax_abs_pipe, "map_attn_softmax_abs.spv", 3, 16) != 0) {
    fprintf(stderr, "ADAMAH: Warning - attn_softmax_abs shader not found\n");
  }

  // LayerNorm: n_rows, dim, eps = 12 bytes, 5 bindings
  if (create_pipeline(&ctx.layernorm_pipe, "map_layernorm.spv", 5, 12) != 0) {
    fprintf(stderr, "ADAMAH: Warning - layernorm shader not found\n");
  }

  // RMSNorm: n_rows, dim, eps = 12 bytes, 4 bindings (map, src, wt, dst)
  if (create_pipeline(&ctx.rmsnorm_pipe, "map_rmsnorm.spv", 4, 12) != 0) {
    fprintf(stderr, "ADAMAH: Warning - rmsnorm shader not found\n");
  }

  // RoPE: n_tokens, n_heads, head_dim, pos_offset, freq_base = 20 bytes, 3
  // bindings (map, src, dst)
  if (create_pipeline(&ctx.rope_pipe, "map_rope.spv", 3, 20) != 0) {
    fprintf(stderr, "ADAMAH: Warning - rope shader not found\n");
  }

  // MatMul Transposed: M, K, N, n_ops = 16 bytes, 4 bindings (map, a, b, c)
  if (create_pipeline(&ctx.matmul_t_pipe, "map_matmul_t.spv", 4, 16) != 0) {
    fprintf(stderr, "ADAMAH: Warning - matmul_t shader not found\n");
  }

  // Row Copy (embedding lookup): n_copies, row_size, dst_row_offset = 12 bytes,
  // 3 bindings (map, copy_spec, src_base)
  if (create_pipeline(&ctx.row_copy_pipe, "map_row_copy.spv", 3, 12) != 0) {
    fprintf(stderr, "ADAMAH: Warning - row_copy shader not found\n");
  }

  // FMA (fused multiply-add): n_locs, pack_size = 8 bytes, 5 bindings (map, a,
  // b, c, dst)
  if (create_pipeline(&ctx.fma_pipe, "map_fma.spv", 5, 8) != 0) {
    fprintf(stderr, "ADAMAH: Warning - fma shader not found\n");
  }

  // Unified FFN: BT, D, D4, apply_residual, phase = 20 bytes, 7 bindings
  if (create_pipeline(&ctx.unified_pipe, "unified.spv", 7, 20) != 0) {
    fprintf(stderr, "ADAMAH: Warning - unified shader not found\n");
  }

  // Scatter: n_locs, pack_size = 8 bytes, 3 bindings (map, src, locs)
  if (create_pipeline(&ctx.scatter_pipe, "map_scatter.spv", 3, 8) != 0) {
    fprintf(stderr, "ADAMAH: Warning - scatter shader not found\n");
  }

  // Gather: n_locs, pack_size = 8 bytes, 3 bindings (map, dst, locs)
  if (create_pipeline(&ctx.gather_pipe, "map_gather.spv", 3, 8) != 0) {
    fprintf(stderr, "ADAMAH: Warning - gather shader not found\n");
  }
  // Repeat penalty: n_ids, penalty = 8 bytes, 3 bindings (map, logits_locs, token_ids)
  if (create_pipeline(&ctx.repeat_penalty_pipe, "map_repeat_penalty.spv", 3, 8) != 0) {
    fprintf(stderr, "ADAMAH: Warning - repeat_penalty shader not found\n");
  }

  // Argmax: one workgroup scans the whole vector with a stride-256 loop.
  if (create_pipeline(&ctx.argmax_pipe, "map_argmax.spv", 3, 4) != 0) {
    fprintf(stderr, "ADAMAH: Warning - argmax shader not found\n");
  }
  // Top-k: repeated in-shader argmax passes for small shortlist sizes.
  if (create_pipeline(&ctx.topk_pipe, "map_topk.spv", 4, 8) != 0) {
    fprintf(stderr, "ADAMAH: Warning - topk shader not found\n");
  }
  // Top-p: softmax + cumulative cutoff on an already sorted shortlist.
  if (create_pipeline(&ctx.topp_pipe, "map_topp.spv", 3, 12) != 0) {
    fprintf(stderr, "ADAMAH: Warning - topp shader not found\n");
  }

  // Cross-map matmul: M,K,N,n_ops = 16 bytes, 5 bindings (act,wt,a,b,c)
  if (create_pipeline(&ctx.matmul_x_pipe, "map_matmul_x.spv", 5, 16) != 0) {
    fprintf(stderr, "ADAMAH: Warning - matmul_x shader not found\n");
  }
  if (create_pipeline(&ctx.matmul_t_x_pipe, "map_matmul_t_x.spv", 5, 16) != 0) {
    fprintf(stderr, "ADAMAH: Warning - matmul_t_x shader not found\n");
  }
  // Cross-map rmsnorm: n_rows,dim,eps = 12 bytes, 5 bindings
  // (act,wt,src,wt,dst)
  if (create_pipeline(&ctx.rmsnorm_x_pipe, "map_rmsnorm_x.spv", 5, 12) != 0) {
    fprintf(stderr, "ADAMAH: Warning - rmsnorm_x shader not found\n");
  }
  // Cross-map F32×Q4 matvec: K,N,group_size = 12 bytes, 6 bindings
  if (create_pipeline(&ctx.matvec_t_xq4_pipe, "map_matvec_t_xq4.spv", 6, 12) != 0) {
    fprintf(stderr, "ADAMAH: Warning - matvec_t_xq4 shader not found\n");
  }
  if (create_pipeline(&ctx.matvec_topk_t_xq4_pipe, "map_matvec_topk_t_xq4.spv", 8, 28) != 0) {
    fprintf(stderr, "ADAMAH: Warning - matvec_topk_t_xq4 shader not found\n");
  }
  // Cross-map F32×Q4 matmul: M,K,N,n_ops,group_size = 20 bytes, 6 bindings
  if (create_pipeline(&ctx.matmul_t_xq4_pipe, "map_matmul_t_xq4.spv", 6, 20) !=
      0) {
    fprintf(stderr, "ADAMAH: Warning - matmul_t_xq4 shader not found\n");
  }
  if (create_pipeline(&ctx.matmul_xq4_pipe, "map_matmul_xq4.spv", 6, 20) != 0) {
    fprintf(stderr, "ADAMAH: Warning - matmul_xq4 shader not found\n");
  }
  if (create_pipeline(&ctx.matvec_t_xq8_pipe, "map_matvec_t_xq8.spv", 6, 12) != 0) {
    fprintf(stderr, "ADAMAH: Warning - matvec_t_xq8 shader not found\n");
  }
  if (create_pipeline(&ctx.matvec_topk_t_xq8_pipe, "map_matvec_topk_t_xq8.spv", 8, 28) != 0) {
    fprintf(stderr, "ADAMAH: Warning - matvec_topk_t_xq8 shader not found\n");
  }
  if (create_pipeline(&ctx.matvec_rerank_t_xq8_pipe, "map_matvec_rerank_t_xq8.spv", 10, 20) != 0) {
    fprintf(stderr, "ADAMAH: Warning - matvec_rerank_t_xq8 shader not found\n");
  }
  if (create_pipeline(&ctx.matmul_t_xq8_pipe, "map_matmul_t_xq8.spv", 6, 20) != 0) {
    fprintf(stderr, "ADAMAH: Warning - matmul_t_xq8 shader not found\n");
  }
  if (create_pipeline(&ctx.matmul_xq8_pipe, "map_matmul_xq8.spv", 6, 20) != 0) {
    fprintf(stderr, "ADAMAH: Warning - matmul_xq8 shader not found\n");
  }
  if (create_pipeline(&ctx.row_gather_xq8_pipe, "map_row_gather_xq8.spv", 3, 24) != 0) {
    fprintf(stderr, "ADAMAH: Warning - row_gather_xq8 shader not found\n");
  }

  return ADAMAH_OK;
}

// Load dtype-specific pipelines from subdirectory
static int init_dtype_pipelines(uint32_t dtype) {
  if (dtype >= DTYPE_COUNT)
    return ADAMAH_ERR_INVALID;
  if (ctx.dtype_pipes_loaded[dtype])
    return ADAMAH_OK;

  // For f32, copy from the main (legacy) pipelines
  if (dtype == DTYPE_F32) {
    ctx.dtype_scatter_pipe[0] = ctx.scatter_pipe;
    ctx.dtype_gather_pipe[0] = ctx.gather_pipe;
    ctx.dtype_unary_pipe[0] = ctx.unary_pipe;
    ctx.dtype_binary_pipe[0] = ctx.binary_pipe;
    ctx.dtype_matmul_pipe[0] = ctx.matmul_pipe;
    ctx.dtype_reduce_pipe[0] = ctx.reduce_pipe;
    ctx.dtype_reduce_small_pipe[0] = ctx.reduce_small_pipe;
    ctx.dtype_broadcast_pipe[0] = ctx.broadcast_pipe;
    ctx.dtype_softmax_pipe[0] = ctx.softmax_pipe;
    ctx.dtype_layernorm_pipe[0] = ctx.layernorm_pipe;
    ctx.dtype_pipes_loaded[0] = 1;
    return ADAMAH_OK;
  }

  // Build shader subdir path
  const char *dtype_names[] = {"f32", "bf16", "q8", "q4", "q6"};
  char saved_path[512];
  strncpy(saved_path, ctx.shader_path, 511);
  saved_path[511] = '\0';

  char dtype_path[600];
  snprintf(dtype_path, sizeof(dtype_path), "%s/%s", saved_path,
           dtype_names[dtype]);

  // Check if dtype shaders exist
  char test[700];
  snprintf(test, sizeof(test), "%s/map_scatter.spv", dtype_path);
  FILE *f = fopen(test, "rb");
  if (!f) {
    fprintf(stderr, "ADAMAH: %s shaders not found at %s\n", dtype_names[dtype],
            dtype_path);
    return ADAMAH_ERR_NOT_FOUND;
  }
  fclose(f);

  // Temporarily switch shader_path for pipeline creation
  strncpy(ctx.shader_path, dtype_path, 511);

  int ok = 1;
  if (dtype == DTYPE_BF16) {
    // bf16 scatter/gather: push = 8 bytes, 3 bindings
    ok &= (create_pipeline(&ctx.dtype_scatter_pipe[dtype], "map_scatter.spv", 3,
                           8) == 0);
    ok &= (create_pipeline(&ctx.dtype_gather_pipe[dtype], "map_gather.spv", 3,
                           8) == 0);
    // bf16 unary/binary: same push sizes as f32
    ok &= (create_pipeline(&ctx.dtype_unary_pipe[dtype], "map_op1.spv", 3,
                           12) == 0);
    ok &= (create_pipeline(&ctx.dtype_binary_pipe[dtype], "map_op2.spv", 4,
                           12) == 0);
    ok &= (create_pipeline(&ctx.dtype_matmul_pipe[dtype], "map_matmul.spv", 4,
                           16) == 0);
    ok &= (create_pipeline(&ctx.dtype_reduce_pipe[dtype], "map_reduce.spv", 3,
                           12) == 0);
    ok &= (create_pipeline(&ctx.dtype_reduce_small_pipe[dtype],
                           "map_reduce_small.spv", 3, 12) == 0);
    ok &= (create_pipeline(&ctx.dtype_broadcast_pipe[dtype],
                           "map_broadcast.spv", 4, 12) == 0);
    ok &= (create_pipeline(&ctx.dtype_softmax_pipe[dtype], "map_softmax.spv", 3,
                           8) == 0);
    ok &= (create_pipeline(&ctx.dtype_layernorm_pipe[dtype],
                           "map_layernorm.spv", 5, 12) == 0);
  } else if (dtype == DTYPE_Q8) {
    // q8 scatter: push = 12 bytes (n_locs, pack_size, group_size), 4 bindings
    // (+qparams)
    ok &= (create_pipeline(&ctx.dtype_scatter_pipe[dtype], "map_scatter.spv", 4,
                           12) == 0);
    ok &= (create_pipeline(&ctx.dtype_gather_pipe[dtype], "map_gather.spv", 4,
                           12) == 0);
    // q8 matmul: push = 20 bytes (M,K,N,n_ops,group_size), 5 bindings
    // (+qparams)
    ok &= (create_pipeline(&ctx.dtype_matmul_pipe[dtype], "map_matmul.spv", 5,
                           20) == 0);
  } else if (dtype == DTYPE_Q4) {
    // q4 scatter: push = 12 bytes (n_locs, pack_size, group_size), 4 bindings (map, src, locs, qparams)
    ok &= (create_pipeline(&ctx.dtype_scatter_pipe[dtype], "map_scatter.spv", 4,
                           12) == 0);
    // q4 gather: push = 12 bytes (n_locs, pack_size, group_size), 4 bindings
    ok &= (create_pipeline(&ctx.dtype_gather_pipe[dtype], "map_gather.spv", 4,
                           12) == 0);
    // q4 matmul: push = 16 bytes (M,K,N,n_ops), 6 bindings
    ok &= (create_pipeline(&ctx.dtype_matmul_pipe[dtype], "map_matmul.spv", 6,
                           16) == 0);
  } else if (dtype == DTYPE_Q6) {
    // q6 scatter: push = 12 bytes (n_locs, pack_size, group_size), 4 bindings
    ok &= (create_pipeline(&ctx.dtype_scatter_pipe[dtype], "map_scatter.spv", 4,
                           12) == 0);
    // q6 gather: push = 12 bytes (n_locs, pack_size, group_size), 4 bindings
    ok &= (create_pipeline(&ctx.dtype_gather_pipe[dtype], "map_gather.spv", 4,
                           12) == 0);
    ok &= (create_pipeline(&ctx.dtype_matmul_pipe[dtype], "map_matmul.spv", 6,
                           16) == 0);
  }

  // Restore shader path
  strncpy(ctx.shader_path, saved_path, 511);

  if (!ok) {
    fprintf(stderr, "ADAMAH: Failed to load some %s pipelines\n",
            dtype_names[dtype]);
    return ADAMAH_ERR_VULKAN;
  }

  ctx.dtype_pipes_loaded[dtype] = 1;
  fprintf(stderr, "ADAMAH: Loaded %s shader pipelines\n", dtype_names[dtype]);
  return ADAMAH_OK;
}

// Public API: set active dtype and load pipelines
int adamah_set_dtype(uint32_t dtype) {
  if (!ctx.initialized)
    return ADAMAH_ERR_INVALID;
  if (dtype >= DTYPE_COUNT)
    return ADAMAH_ERR_INVALID;

  // Ensure base pipelines are loaded first (sets ctx.shader_path)
  if (!ctx.unary_pipe.pipeline) {
    int ret = init_pipelines();
    if (ret != ADAMAH_OK)
      return ret;
  }

  // Ensure f32 pipelines are loaded first (base requirement)
  if (!ctx.dtype_pipes_loaded[DTYPE_F32]) {
    init_dtype_pipelines(DTYPE_F32);
  }

  // Load dtype-specific pipelines if needed
  if (!ctx.dtype_pipes_loaded[dtype]) {
    int ret = init_dtype_pipelines(dtype);
    if (ret != ADAMAH_OK)
      return ret;
  }

  ctx.active_dtype = dtype;
  return ADAMAH_OK;
}

uint32_t adamah_get_dtype(void) { return ctx.active_dtype; }

// ============================================
// Quantization API
// ============================================

// Upload quantization parameters for a q8 map
int map_set_qparams(uint32_t map_id, const float *scales,
                    const float *zero_points, uint32_t n_groups) {
  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return ADAMAH_ERR_INVALID;
  Map *m = &ctx.maps[map_id];
  if (m->dtype != DTYPE_Q8 && m->dtype != DTYPE_Q4 && m->dtype != DTYPE_Q6)
    return ADAMAH_ERR_INVALID;
  if (n_groups > m->n_groups)
    return ADAMAH_ERR_INVALID;

  float *qp = (float *)m->qparam_staging_ptr;
  for (uint32_t g = 0; g < n_groups; g++) {
    qp[g * 2] = scales[g];
    qp[g * 2 + 1] = zero_points[g];
  }

  cmd_begin();
  VkBufferCopy copy = {.size = (VkDeviceSize)n_groups * 2 * sizeof(float)};
  vkCmdCopyBuffer(ctx.cmd, m->qparam_staging, m->qparam_buf, 1, &copy);
  cmd_submit();

  return ADAMAH_OK;
}

// Get map dtype
uint32_t map_get_dtype(uint32_t map_id) {
  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return DTYPE_F32;
  return ctx.maps[map_id].dtype;
}

// ============================================
// Fusion System - Automatic Operation Batching
// ============================================

// Enable/disable fusion
void adamah_fusion_enable(int enable) {
  if (!enable && fusion.n_ops > 0) {
    adamah_fusion_flush(); // flush pending ops before disabling
  }
  fusion.enabled = enable;
}

int adamah_fusion_is_enabled(void) { return fusion.enabled; }

// Reset fusion state
static void fusion_reset(void) {
  fusion.n_ops = 0;
  fusion.max_level = 0;
  memset(fusion.loc_write_level, -1, sizeof(fusion.loc_write_level));
}

static int fusion_flush_pending_for_immediate(void) {
  if (fusion.enabled && fusion.n_ops > 0) {
    return adamah_fusion_flush();
  }
  return ADAMAH_OK;
}

// Calculate level for an operation based on its source dependencies
static int fusion_calc_level(uint32_t locs_src, uint32_t locs_src2) {
  int level = 0;

  // Check first source
  if (locs_src < FUSION_MAX_LOCS && fusion.loc_write_level[locs_src] >= 0) {
    level = fusion.loc_write_level[locs_src] + 1;
  }

  // Check second source (for binary ops)
  if (locs_src2 != 0 && locs_src2 < FUSION_MAX_LOCS &&
      fusion.loc_write_level[locs_src2] >= 0) {
    int lvl2 = fusion.loc_write_level[locs_src2] + 1;
    if (lvl2 > level)
      level = lvl2;
  }

  return level;
}

// Mark destination as written at given level
static void fusion_mark_write(uint32_t locs_dst, int level) {
  if (locs_dst < FUSION_MAX_LOCS) {
    fusion.loc_write_level[locs_dst] = level;
  }
  if (level > fusion.max_level) {
    fusion.max_level = level;
  }
}

// Auto-flush if needed: ONLY when queue is full
// Level-based execution is handled inside adamah_fusion_flush()
static void fusion_auto_flush_if_needed(int new_level) {
  (void)new_level; // Level doesn't trigger flush - only used for ordering
                   // during execution

  // Flush only if queue is full (use dynamic limit based on GPU)
  uint32_t max_ops =
      gpu_caps.fusion_max_ops > 0 ? gpu_caps.fusion_max_ops : 512;
  if (fusion.n_ops >= (int)max_ops) {
    adamah_fusion_flush();
  }
}

// Queue a unary operation
static int fusion_queue_op1(uint32_t map_id, uint32_t op, uint32_t locs_src,
                            uint32_t locs_dst, uint32_t n) {
  int level = fusion_calc_level(locs_src, 0);
  fusion_auto_flush_if_needed(level);

  FuseOp *fop = &fusion.ops[fusion.n_ops++];
  memset(fop, 0, sizeof(*fop));
  fop->op_type = FUSE_OP_UNARY;
  fop->op_code = op;
  fop->map_id = map_id;
  fop->locs_src = locs_src;
  fop->locs_dst = locs_dst;
  fop->n = n;
  fop->level = level;
  fusion_mark_write(locs_dst, fop->level);

  return ADAMAH_OK;
}

// Queue a binary operation
static int fusion_queue_op2(uint32_t map_id, uint32_t op, uint32_t locs_a,
                            uint32_t locs_b, uint32_t locs_dst, uint32_t n) {
  int level = fusion_calc_level(locs_a, locs_b);
  fusion_auto_flush_if_needed(level);

  FuseOp *fop = &fusion.ops[fusion.n_ops++];
  memset(fop, 0, sizeof(*fop));
  fop->op_type = FUSE_OP_BINARY;
  fop->op_code = op;
  fop->map_id = map_id;
  fop->locs_src = locs_a;
  fop->locs_src2 = locs_b;
  fop->locs_dst = locs_dst;
  fop->n = n;
  fop->level = level;
  fusion_mark_write(locs_dst, fop->level);

  return ADAMAH_OK;
}

// Queue a matmul operation
static int fusion_queue_matmul(uint32_t map_id, uint32_t locs_a,
                               uint32_t locs_b, uint32_t locs_c, uint32_t M,
                               uint32_t K, uint32_t N, uint32_t n_ops) {
  int level = fusion_calc_level(locs_a, locs_b);
  fusion_auto_flush_if_needed(level);

  FuseOp *fop = &fusion.ops[fusion.n_ops++];
  memset(fop, 0, sizeof(*fop));
  fop->op_type = FUSE_OP_MATMUL;
  fop->map_id = map_id;
  fop->locs_extra[0] = locs_a;
  fop->locs_extra[1] = locs_b;
  fop->locs_extra[2] = locs_c;
  fop->M = M;
  fop->K = K;
  fop->N = N;
  fop->n = n_ops;
  fop->level = level;
  fusion_mark_write(locs_c, fop->level);

  return ADAMAH_OK;
}

// Queue a reduce operation
static int fusion_queue_reduce(uint32_t map_id, uint32_t op, uint32_t locs_src,
                               uint32_t locs_dst, uint32_t n) {
  int level = fusion_calc_level(locs_src, 0);
  fusion_auto_flush_if_needed(level);

  FuseOp *fop = &fusion.ops[fusion.n_ops++];
  memset(fop, 0, sizeof(*fop));
  fop->op_type = FUSE_OP_REDUCE;
  fop->op_code = op;
  fop->map_id = map_id;
  fop->locs_src = locs_src;
  fop->locs_dst = locs_dst;
  fop->n = n;
  fop->level = level;
  fusion_mark_write(locs_dst, fop->level);

  return ADAMAH_OK;
}

// Queue a softmax operation
static int fusion_queue_softmax(uint32_t map_id, uint32_t locs_src,
                                uint32_t locs_dst, uint32_t dim,
                                uint32_t n_rows, uint32_t abs_base) {
  int level = fusion_calc_level(locs_src, 0);
  fusion_auto_flush_if_needed(level);

  FuseOp *fop = &fusion.ops[fusion.n_ops++];
  memset(fop, 0, sizeof(*fop));
  fop->op_type = FUSE_OP_SOFTMAX;
  fop->op_code = abs_base;
  fop->map_id = map_id;
  fop->locs_src = locs_src;
  fop->locs_dst = locs_dst;
  fop->M = dim; // reuse M for dim
  fop->n = n_rows;
  fop->level = level;
  fusion_mark_write(locs_dst, fop->level);

  return ADAMAH_OK;
}

// Queue a layernorm operation
static int fusion_queue_layernorm(uint32_t map_id, uint32_t locs_src,
                                  uint32_t locs_dst, uint32_t locs_gamma,
                                  uint32_t locs_beta, uint32_t dim,
                                  uint32_t n_rows, float eps) {
  // Layernorm reads src, gamma, beta
  int level = fusion_calc_level(locs_src, 0);
  if (locs_gamma < FUSION_MAX_LOCS && fusion.loc_write_level[locs_gamma] >= 0) {
    int lvl = fusion.loc_write_level[locs_gamma] + 1;
    if (lvl > level)
      level = lvl;
  }
  if (locs_beta < FUSION_MAX_LOCS && fusion.loc_write_level[locs_beta] >= 0) {
    int lvl = fusion.loc_write_level[locs_beta] + 1;
    if (lvl > level)
      level = lvl;
  }
  fusion_auto_flush_if_needed(level);

  FuseOp *fop = &fusion.ops[fusion.n_ops++];
  memset(fop, 0, sizeof(*fop));
  fop->op_type = FUSE_OP_LAYERNORM;
  fop->map_id = map_id;
  fop->locs_src = locs_src;
  fop->locs_dst = locs_dst;
  fop->locs_extra[0] = locs_gamma;
  fop->locs_extra[1] = locs_beta;
  fop->M = dim;
  fop->n = n_rows;
  fop->eps = eps;
  fop->level = level;
  fusion_mark_write(locs_dst, fop->level);

  return ADAMAH_OK;
}

// Queue a broadcast operation
static int fusion_queue_broadcast(uint32_t map_id, uint32_t op, uint32_t locs_a,
                                  uint32_t locs_scalar, uint32_t locs_dst,
                                  uint32_t n) {
  int level = fusion_calc_level(locs_a, locs_scalar);
  fusion_auto_flush_if_needed(level);

  FuseOp *fop = &fusion.ops[fusion.n_ops++];
  memset(fop, 0, sizeof(*fop));
  fop->op_type = FUSE_OP_BROADCAST;
  fop->op_code = op;
  fop->map_id = map_id;
  fop->locs_src = locs_a;
  fop->locs_src2 = locs_scalar;
  fop->locs_dst = locs_dst;
  fop->n = n;
  fop->level = level;
  fusion_mark_write(locs_dst, fop->level);

  return ADAMAH_OK;
}

// Forward declarations for actual execution functions
static int exec_op1_internal(uint32_t map_id, uint32_t op, uint32_t locs_src,
                             uint32_t locs_dst, uint32_t n);
static int exec_op2_internal(uint32_t map_id, uint32_t op, uint32_t locs_a,
                             uint32_t locs_b, uint32_t locs_dst, uint32_t n);
static int exec_matmul_internal(uint32_t map_id, uint32_t locs_a,
                                uint32_t locs_b, uint32_t locs_c, uint32_t M,
                                uint32_t K, uint32_t N, uint32_t n_ops);
static int exec_reduce_internal(uint32_t map_id, uint32_t op, uint32_t locs_src,
                                uint32_t locs_dst, uint32_t n);
static int exec_softmax_internal(uint32_t map_id, uint32_t locs_src,
                                 uint32_t locs_dst, uint32_t dim,
                                 uint32_t n_rows);
static int exec_softmax_abs_internal(uint32_t map_id, uint32_t locs_src,
                                     uint32_t locs_dst, uint32_t dim,
                                     uint32_t n_rows);
static int exec_layernorm_internal(uint32_t map_id, uint32_t locs_src,
                                   uint32_t locs_dst, uint32_t locs_gamma,
                                   uint32_t locs_beta, uint32_t dim,
                                   uint32_t n_rows, float eps);
static int exec_broadcast_internal(uint32_t map_id, uint32_t op,
                                   uint32_t locs_a, uint32_t locs_scalar,
                                   uint32_t locs_dst, uint32_t n);

// Execute all queued operations, grouped by level
int adamah_fusion_flush(void) {
  if (fusion.n_ops == 0)
    return ADAMAH_OK;

  int result = ADAMAH_OK;
  int active_batch = batch_mode;

  if (!active_batch) {
    // Single batch for ALL levels
    batch_begin();
  }

  // Execute level by level with barriers between
  for (int level = 0; level <= fusion.max_level; level++) {
    // Execute all ops at this level
    for (int i = 0; i < fusion.n_ops; i++) {
      FuseOp *fop = &fusion.ops[i];
      if (fop->level != level)
        continue;

      int ret = ADAMAH_OK;
      switch (fop->op_type) {
      case FUSE_OP_UNARY:
        ret = exec_op1_internal(fop->map_id, fop->op_code, fop->locs_src,
                                fop->locs_dst, fop->n);
        break;
      case FUSE_OP_BINARY:
        ret = exec_op2_internal(fop->map_id, fop->op_code, fop->locs_src,
                                fop->locs_src2, fop->locs_dst, fop->n);
        break;
      case FUSE_OP_MATMUL:
        ret = exec_matmul_internal(fop->map_id, fop->locs_extra[0],
                                   fop->locs_extra[1], fop->locs_extra[2],
                                   fop->M, fop->K, fop->N, fop->n);
        break;
      case FUSE_OP_REDUCE:
        ret = exec_reduce_internal(fop->map_id, fop->op_code, fop->locs_src,
                                   fop->locs_dst, fop->n);
        break;
      case FUSE_OP_SOFTMAX:
        ret = fop->op_code
                  ? exec_softmax_abs_internal(fop->map_id, fop->locs_src,
                                              fop->locs_dst, fop->M, fop->n)
                  : exec_softmax_internal(fop->map_id, fop->locs_src,
                                          fop->locs_dst, fop->M, fop->n);
        break;
      case FUSE_OP_LAYERNORM:
        ret = exec_layernorm_internal(fop->map_id, fop->locs_src, fop->locs_dst,
                                      fop->locs_extra[0], fop->locs_extra[1],
                                      fop->M, fop->n, fop->eps);
        break;
      case FUSE_OP_BROADCAST:
        ret = exec_broadcast_internal(fop->map_id, fop->op_code, fop->locs_src,
                                      fop->locs_src2, fop->locs_dst, fop->n);
        break;
      }
      if (ret != ADAMAH_OK)
        result = ret;
    }

    // Barrier between levels (if not the last level)
    // The barrier is already added by cmd_barrier_after_dispatch() in each
    // exec_*_internal
  }

  if (!active_batch) {
    // Single submit for ALL ops across ALL levels
    batch_end();
  }

  // Reset fusion state
  fusion_reset();

  return result;
}

// Get fusion stats
void adamah_fusion_stats(int *n_ops, int *n_levels) {
  if (n_ops)
    *n_ops = fusion.n_ops;
  if (n_levels)
    *n_levels = fusion.max_level + 1;
}

// ============================================
// Internal GPU buffers for locations
// ============================================

static GpuBuf *get_or_create_buf_ex(const char *base_name, uint32_t n_elems,
                                    uint32_t elem_size, int device_local,
                                    VkBufferUsageFlags usage) {
  VkDeviceSize bytes_needed = (VkDeviceSize)n_elems * (VkDeviceSize)elem_size;
  if (bytes_needed == 0)
    bytes_needed = 1;
  // In batch mode, create unique buffer name for each op
  char name[64];
  if (batch_mode) {
    snprintf(name, sizeof(name), "%s_%d", base_name, batch_op_counter);
  } else {
    strncpy(name, base_name, 63);
    name[63] = 0;
  }

  VkDeviceSize aligned_needed = align_up(bytes_needed, ctx.copy_align);
  VkDeviceSize desired_capacity = bytes_needed;
  if (device_local) {
    desired_capacity = device_local_bucket(aligned_needed);
  } else if (is_stage_upload_name(name) || is_stage_download_name(name)) {
    desired_capacity = next_pow2(aligned_needed);
  }

  // Find existing
  for (int i = 0; i < ctx.buf_count; i++) {
    if (strcmp(ctx.bufs[i].name, name) == 0) {
      if (ctx.bufs[i].bytes_capacity >= desired_capacity &&
          ctx.bufs[i].device_local == device_local &&
          ctx.bufs[i].usage == usage) {
        return &ctx.bufs[i];
      }
      // Need to resize or mismatch - destroy old
      if (ctx.bufs[i].bytes_capacity < desired_capacity) {
        if (is_stage_upload_name(name))
          ctx.stage_upload_grow_events++;
        if (is_stage_download_name(name))
          ctx.stage_download_grow_events++;
      }
      ctx.num_buffer_recreates++;
      if (ctx.bufs[i].ptr)
        vkUnmapMemory(ctx.device, ctx.bufs[i].mem);
      vkDestroyBuffer(ctx.device, ctx.bufs[i].buf, NULL);
      vkFreeMemory(ctx.device, ctx.bufs[i].mem, NULL);
      ctx.bufs[i].buf = VK_NULL_HANDLE;
      ctx.bufs[i].ptr = NULL;
    }
  }

  // Find or create slot
  GpuBuf *b = NULL;
  for (int i = 0; i < ctx.buf_count; i++) {
    if (strcmp(ctx.bufs[i].name, name) == 0) {
      b = &ctx.bufs[i];
      break;
    }
  }
  if (!b && ctx.buf_count < MAX_BUFS) {
    b = &ctx.bufs[ctx.buf_count++];
  }
  if (!b)
    return NULL;

  strncpy(b->name, name, 63);
  b->bytes_capacity = desired_capacity;
  b->elem_size = elem_size;
  b->device_local = device_local;
  b->usage = usage;

  if (create_buffer(&b->buf, &b->mem, desired_capacity, usage, device_local) !=
      0)
    return NULL;
  if (!device_local) {
    vkMapMemory(ctx.device, b->mem, 0, desired_capacity, 0, &b->ptr);
  } else {
    b->ptr = NULL;
  }

  return b;
}

static GpuBuf *get_or_create_buf(const char *base_name, uint32_t n_elems,
                                 uint32_t elem_size) {
  VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  return get_or_create_buf_ex(base_name, n_elems, elem_size, 0, usage);
}

// ============================================
// Map Operations (Pure GPU)
// ============================================

int map_op1(uint32_t map_id, uint32_t op, const uint32_t *locs_src,
            const uint32_t *locs_dst, uint32_t n) {
  // Legacy path removed: use map_op1_dev with cached locs.
  (void)map_id;
  (void)op;
  (void)locs_src;
  (void)locs_dst;
  (void)n;
  return ADAMAH_ERR_INVALID;
}

// Internal execution - bypasses fusion
static int exec_op1_internal(uint32_t map_id, uint32_t op,
                             uint32_t locs_src_handle, uint32_t locs_dst_handle,
                             uint32_t n) {
  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return ADAMAH_ERR_INVALID;
  if (n == 0)
    return ADAMAH_OK;
  if (!ctx.unary_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;

  Map *m = &ctx.maps[map_id];
  ResEntry *src_res = res_get(locs_src_handle);
  ResEntry *dst_res = res_get(locs_dst_handle);
  if (!src_res || !dst_res)
    return ADAMAH_ERR_INVALID;
  if ((VkDeviceSize)n * 4 > src_res->size_bytes)
    return ADAMAH_ERR_INVALID;
  if ((VkDeviceSize)n * 4 > dst_res->size_bytes)
    return ADAMAH_ERR_INVALID;
  res_pin(locs_src_handle);
  res_pin(locs_dst_handle);
  VkDeviceSize src_off = 0;
  VkDeviceSize dst_off = 0;
  if (res_require_hot(locs_src_handle, &src_off) != 0 ||
      res_require_hot(locs_dst_handle, &dst_off) != 0) {
    res_unpin(locs_src_handle);
    res_unpin(locs_dst_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.unary_pipe);
  if (ds == VK_NULL_HANDLE)
    return ADAMAH_ERR_VULKAN;

  VkDescriptorBufferInfo buf_infos[3] = {
      {.buffer = m->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf,
       .offset = src_off,
       .range = src_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = dst_off,
       .range = dst_res->size_bytes}};
  VkWriteDescriptorSet writes[3];
  for (int i = 0; i < 3; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);

  uint32_t push[3] = {op, n, m->pack_size};
  uint32_t total_threads = n * m->pack_size;

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.unary_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.unary_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.unary_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push);
  vkCmdDispatch(ctx.cmd, (total_threads + 255) / 256, 1, 1);
  cmd_barrier_after_dispatch();
  cmd_submit();
  res_unpin(locs_src_handle);
  res_unpin(locs_dst_handle);

  return ADAMAH_OK;
}

// Public API - uses fusion if enabled
int map_op1_dev(uint32_t map_id, uint32_t op, uint32_t locs_src_handle,
                uint32_t locs_dst_handle, uint32_t n) {
  if (fusion.enabled) {
    return fusion_queue_op1(map_id, op, locs_src_handle, locs_dst_handle, n);
  }
  return exec_op1_internal(map_id, op, locs_src_handle, locs_dst_handle, n);
}

int map_op2(uint32_t map_id, uint32_t op, const uint32_t *locs_a,
            const uint32_t *locs_b, const uint32_t *locs_dst, uint32_t n) {
  // Legacy path removed: use map_op2_dev with cached locs.
  (void)map_id;
  (void)op;
  (void)locs_a;
  (void)locs_b;
  (void)locs_dst;
  (void)n;
  return ADAMAH_ERR_INVALID;
}

// Internal execution - bypasses fusion
static int exec_op2_internal(uint32_t map_id, uint32_t op,
                             uint32_t locs_a_handle, uint32_t locs_b_handle,
                             uint32_t locs_dst_handle, uint32_t n) {
  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return ADAMAH_ERR_INVALID;
  if (n == 0)
    return ADAMAH_OK;
  if (!ctx.binary_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;

  Map *m = &ctx.maps[map_id];
  ResEntry *a_res = res_get(locs_a_handle);
  ResEntry *b_res = res_get(locs_b_handle);
  ResEntry *dst_res = res_get(locs_dst_handle);
  if (!a_res || !b_res || !dst_res)
    return ADAMAH_ERR_INVALID;
  if ((VkDeviceSize)n * 4 > a_res->size_bytes)
    return ADAMAH_ERR_INVALID;
  if ((VkDeviceSize)n * 4 > b_res->size_bytes)
    return ADAMAH_ERR_INVALID;
  if ((VkDeviceSize)n * 4 > dst_res->size_bytes)
    return ADAMAH_ERR_INVALID;
  res_pin(locs_a_handle);
  res_pin(locs_b_handle);
  res_pin(locs_dst_handle);
  VkDeviceSize a_off = 0;
  VkDeviceSize b_off = 0;
  VkDeviceSize dst_off = 0;
  if (res_require_hot(locs_a_handle, &a_off) != 0 ||
      res_require_hot(locs_b_handle, &b_off) != 0 ||
      res_require_hot(locs_dst_handle, &dst_off) != 0) {
    res_unpin(locs_a_handle);
    res_unpin(locs_b_handle);
    res_unpin(locs_dst_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.binary_pipe);
  if (ds == VK_NULL_HANDLE)
    return ADAMAH_ERR_VULKAN;

  VkDescriptorBufferInfo buf_infos[4] = {
      {.buffer = m->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf,
       .offset = a_off,
       .range = a_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = b_off,
       .range = b_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = dst_off,
       .range = dst_res->size_bytes}};
  VkWriteDescriptorSet writes[4];
  for (int i = 0; i < 4; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 4, writes, 0, NULL);

  uint32_t push[3] = {op, n, m->pack_size};
  uint32_t total_threads = n * m->pack_size;

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.binary_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.binary_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.binary_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push);
  vkCmdDispatch(ctx.cmd, (total_threads + 255) / 256, 1, 1);
  cmd_barrier_after_dispatch();
  cmd_submit();
  res_unpin(locs_a_handle);
  res_unpin(locs_b_handle);
  res_unpin(locs_dst_handle);

  return ADAMAH_OK;
}

// Public API - uses fusion if enabled
int map_op2_dev(uint32_t map_id, uint32_t op, uint32_t locs_a_handle,
                uint32_t locs_b_handle, uint32_t locs_dst_handle, uint32_t n) {
  if (fusion.enabled) {
    return fusion_queue_op2(map_id, op, locs_a_handle, locs_b_handle,
                            locs_dst_handle, n);
  }
  return exec_op2_internal(map_id, op, locs_a_handle, locs_b_handle,
                           locs_dst_handle, n);
}

// ============================================
// Matrix Multiplication: C = A @ B
// A: (M x K), B: (K x N), C: (M x N)
// ============================================
int map_matmul(uint32_t map_id, const uint32_t *locs_a, const uint32_t *locs_b,
               const uint32_t *locs_c, uint32_t M, uint32_t K, uint32_t N,
               uint32_t n_ops) {
  // Legacy path removed: use map_matmul_dev with cached locs.
  (void)map_id;
  (void)locs_a;
  (void)locs_b;
  (void)locs_c;
  (void)M;
  (void)K;
  (void)N;
  (void)n_ops;
  return ADAMAH_ERR_INVALID;
}

int map_matmul_dev(uint32_t map_id, uint32_t locs_a_handle,
                   uint32_t locs_b_handle, uint32_t locs_c_handle, uint32_t M,
                   uint32_t K, uint32_t N, uint32_t n_ops) {
  if (fusion.enabled) {
    return fusion_queue_matmul(map_id, locs_a_handle, locs_b_handle,
                               locs_c_handle, M, K, N, n_ops);
  }
  return exec_matmul_internal(map_id, locs_a_handle, locs_b_handle,
                              locs_c_handle, M, K, N, n_ops);
}

// Internal execution - bypasses fusion
static int exec_matmul_internal(uint32_t map_id, uint32_t locs_a_handle,
                                uint32_t locs_b_handle, uint32_t locs_c_handle,
                                uint32_t M, uint32_t K, uint32_t N,
                                uint32_t n_ops) {
  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active) {
    if (debug_enabled())
      fprintf(stderr, "ADAMAH DEBUG: matmul invalid map_id=%u\n", map_id);
    return ADAMAH_ERR_INVALID;
  }
  if (n_ops == 0)
    return ADAMAH_OK;
  if (!ctx.matmul_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;
  if (!ctx.matmul_pipe.pipeline) {
    if (debug_enabled())
      fprintf(stderr, "ADAMAH DEBUG: matmul pipeline not available\n");
    return ADAMAH_ERR_INVALID;
  }

  Map *m = &ctx.maps[map_id];
  ResEntry *a_res = res_get(locs_a_handle);
  ResEntry *b_res = res_get(locs_b_handle);
  ResEntry *c_res = res_get(locs_c_handle);
  if (!a_res || !b_res || !c_res) {
    if (debug_enabled()) {
      fprintf(stderr, "ADAMAH DEBUG: matmul res missing a=%u b=%u c=%u\n",
              locs_a_handle, locs_b_handle, locs_c_handle);
    }
    return ADAMAH_ERR_INVALID;
  }
  if ((VkDeviceSize)n_ops * 4 > a_res->size_bytes ||
      (VkDeviceSize)n_ops * 4 > b_res->size_bytes ||
      (VkDeviceSize)n_ops * 4 > c_res->size_bytes) {
    if (debug_enabled()) {
      fprintf(stderr,
              "ADAMAH DEBUG: matmul locs size mismatch n_ops=%u "
              "sizes=%zu,%zu,%zu\n",
              n_ops, (size_t)a_res->size_bytes, (size_t)b_res->size_bytes,
              (size_t)c_res->size_bytes);
    }
    return ADAMAH_ERR_INVALID;
  }
  res_pin(locs_a_handle);
  res_pin(locs_b_handle);
  res_pin(locs_c_handle);
  VkDeviceSize a_off = 0;
  VkDeviceSize b_off = 0;
  VkDeviceSize c_off = 0;
  if (res_require_hot(locs_a_handle, &a_off) != 0 ||
      res_require_hot(locs_b_handle, &b_off) != 0 ||
      res_require_hot(locs_c_handle, &c_off) != 0) {
    if (debug_enabled())
      fprintf(stderr, "ADAMAH DEBUG: matmul res_require_hot failed\n");
    res_unpin(locs_a_handle);
    res_unpin(locs_b_handle);
    res_unpin(locs_c_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.matmul_pipe);
  if (ds == VK_NULL_HANDLE)
    return ADAMAH_ERR_VULKAN;

  VkDescriptorBufferInfo buf_infos[4] = {
      {.buffer = m->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf,
       .offset = a_off,
       .range = a_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = b_off,
       .range = b_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = c_off,
       .range = c_res->size_bytes}};
  VkWriteDescriptorSet writes[4];
  for (int i = 0; i < 4; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 4, writes, 0, NULL);

  uint32_t push[4] = {M, K, N, n_ops};
  uint32_t grid_x = (M + 15) / 16;
  uint32_t grid_y = (N + 15) / 16;

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.matmul_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.matmul_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.matmul_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, push);
  vkCmdDispatch(ctx.cmd, grid_x, grid_y, n_ops);
  cmd_barrier_after_dispatch();
  cmd_submit();
  res_unpin(locs_a_handle);
  res_unpin(locs_b_handle);
  res_unpin(locs_c_handle);

  return ADAMAH_OK;
}

// ============================================
// Reduce: sum/max/min along pack dimension
// ============================================
#define REDUCE_SUM 0
#define REDUCE_MAX 1
#define REDUCE_MIN 2

int map_reduce(uint32_t map_id, uint32_t op, const uint32_t *locs_src,
               const uint32_t *locs_dst, uint32_t n) {
  // Legacy path removed: use map_reduce_dev with cached locs.
  (void)map_id;
  (void)op;
  (void)locs_src;
  (void)locs_dst;
  (void)n;
  return ADAMAH_ERR_INVALID;
}

// Internal execution for reduce - bypasses fusion
static int exec_reduce_internal(uint32_t map_id, uint32_t op,
                                uint32_t locs_src_handle,
                                uint32_t locs_dst_handle, uint32_t n) {
  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return ADAMAH_ERR_INVALID;
  if (n == 0)
    return ADAMAH_OK;

  Map *m = &ctx.maps[map_id];
  if (m->pack_size == 1) {
    // Reduce of a single element is a copy; avoid 256-thread reduction.
    return exec_op1_internal(map_id, 255, locs_src_handle, locs_dst_handle, n);
  }
  if (!ctx.reduce_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;
  Pipeline *reduce_pipe = &ctx.reduce_pipe;
  if (m->pack_size <= 64 && ctx.reduce_small_pipe.pipeline) {
    reduce_pipe = &ctx.reduce_small_pipe;
  }
  if (!reduce_pipe->pipeline)
    return ADAMAH_ERR_INVALID;
  ResEntry *src_res = res_get(locs_src_handle);
  ResEntry *dst_res = res_get(locs_dst_handle);
  if (!src_res || !dst_res)
    return ADAMAH_ERR_INVALID;
  if ((VkDeviceSize)n * 4 > src_res->size_bytes)
    return ADAMAH_ERR_INVALID;
  if ((VkDeviceSize)n * 4 > dst_res->size_bytes)
    return ADAMAH_ERR_INVALID;
  res_pin(locs_src_handle);
  res_pin(locs_dst_handle);
  VkDeviceSize src_off = 0;
  VkDeviceSize dst_off = 0;
  if (res_require_hot(locs_src_handle, &src_off) != 0 ||
      res_require_hot(locs_dst_handle, &dst_off) != 0) {
    res_unpin(locs_src_handle);
    res_unpin(locs_dst_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(reduce_pipe);
  if (ds == VK_NULL_HANDLE)
    return ADAMAH_ERR_VULKAN;

  VkDescriptorBufferInfo buf_infos[3] = {
      {.buffer = m->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf,
       .offset = src_off,
       .range = src_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = dst_off,
       .range = dst_res->size_bytes}};
  VkWriteDescriptorSet writes[3];
  for (int i = 0; i < 3; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);

  uint32_t push[3] = {op, n, m->pack_size};

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    reduce_pipe->pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          reduce_pipe->pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, reduce_pipe->pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push);
  vkCmdDispatch(ctx.cmd, n, 1, 1);
  cmd_barrier_after_dispatch();
  cmd_submit();
  res_unpin(locs_src_handle);
  res_unpin(locs_dst_handle);

  return ADAMAH_OK;
}

// Public API - uses fusion if enabled
int map_reduce_dev(uint32_t map_id, uint32_t op, uint32_t locs_src_handle,
                   uint32_t locs_dst_handle, uint32_t n) {
  if (fusion.enabled) {
    return fusion_queue_reduce(map_id, op, locs_src_handle, locs_dst_handle, n);
  }
  return exec_reduce_internal(map_id, op, locs_src_handle, locs_dst_handle, n);
}

// ============================================
// Broadcast: element-wise op with scalar
// dst = src op scalar (scalar broadcast to all elements)
// ============================================
#define BROADCAST_MUL 0
#define BROADCAST_DIV 1
#define BROADCAST_ADD 2
#define BROADCAST_SUB 3

int map_broadcast(uint32_t map_id, uint32_t op, const uint32_t *locs_src,
                  const uint32_t *locs_scalar, const uint32_t *locs_dst,
                  uint32_t n) {
  // Legacy path removed: use map_broadcast_dev with cached locs.
  (void)map_id;
  (void)op;
  (void)locs_src;
  (void)locs_scalar;
  (void)locs_dst;
  (void)n;
  return ADAMAH_ERR_INVALID;
}

// Internal execution - bypasses fusion
static int exec_broadcast_internal(uint32_t map_id, uint32_t op,
                                   uint32_t locs_src_handle,
                                   uint32_t locs_scalar_handle,
                                   uint32_t locs_dst_handle, uint32_t n) {
  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return ADAMAH_ERR_INVALID;
  if (n == 0)
    return ADAMAH_OK;
  if (!ctx.broadcast_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;
  if (!ctx.broadcast_pipe.pipeline)
    return ADAMAH_ERR_INVALID;

  Map *m = &ctx.maps[map_id];
  ResEntry *src_res = res_get(locs_src_handle);
  ResEntry *scalar_res = res_get(locs_scalar_handle);
  ResEntry *dst_res = res_get(locs_dst_handle);
  if (!src_res || !scalar_res || !dst_res)
    return ADAMAH_ERR_INVALID;
  if ((VkDeviceSize)n * 4 > src_res->size_bytes)
    return ADAMAH_ERR_INVALID;
  if ((VkDeviceSize)n * 4 > scalar_res->size_bytes)
    return ADAMAH_ERR_INVALID;
  if ((VkDeviceSize)n * 4 > dst_res->size_bytes)
    return ADAMAH_ERR_INVALID;
  res_pin(locs_src_handle);
  res_pin(locs_scalar_handle);
  res_pin(locs_dst_handle);
  VkDeviceSize src_off = 0;
  VkDeviceSize scalar_off = 0;
  VkDeviceSize dst_off = 0;
  if (res_require_hot(locs_src_handle, &src_off) != 0 ||
      res_require_hot(locs_scalar_handle, &scalar_off) != 0 ||
      res_require_hot(locs_dst_handle, &dst_off) != 0) {
    res_unpin(locs_src_handle);
    res_unpin(locs_scalar_handle);
    res_unpin(locs_dst_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.broadcast_pipe);
  if (ds == VK_NULL_HANDLE)
    return ADAMAH_ERR_VULKAN;

  VkDescriptorBufferInfo buf_infos[4] = {
      {.buffer = m->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf,
       .offset = src_off,
       .range = src_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = scalar_off,
       .range = scalar_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = dst_off,
       .range = dst_res->size_bytes}};
  VkWriteDescriptorSet writes[4];
  for (int i = 0; i < 4; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 4, writes, 0, NULL);

  uint32_t push[3] = {op, n, m->pack_size};
  uint32_t total_threads = n * m->pack_size;

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.broadcast_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.broadcast_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.broadcast_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push);
  vkCmdDispatch(ctx.cmd, (total_threads + 255) / 256, 1, 1);
  cmd_barrier_after_dispatch();
  cmd_submit();
  res_unpin(locs_src_handle);
  res_unpin(locs_scalar_handle);
  res_unpin(locs_dst_handle);

  return ADAMAH_OK;
}

// Public API - uses fusion if enabled
int map_broadcast_dev(uint32_t map_id, uint32_t op, uint32_t locs_src_handle,
                      uint32_t locs_scalar_handle, uint32_t locs_dst_handle,
                      uint32_t n) {
  if (fusion.enabled) {
    return fusion_queue_broadcast(map_id, op, locs_src_handle,
                                  locs_scalar_handle, locs_dst_handle, n);
  }
  return exec_broadcast_internal(map_id, op, locs_src_handle,
                                 locs_scalar_handle, locs_dst_handle, n);
}

// ============================================
// Softmax: fused max-subtract-exp-sum-normalize
// ============================================
int map_softmax(uint32_t map_id, const uint32_t *locs_src,
                const uint32_t *locs_dst, uint32_t n_rows, uint32_t row_size) {
  // Legacy path removed: use map_softmax_dev with cached locs.
  (void)map_id;
  (void)locs_src;
  (void)locs_dst;
  (void)n_rows;
  (void)row_size;
  return ADAMAH_ERR_INVALID;
}

// Internal execution - bypasses fusion
static int exec_softmax_internal(uint32_t map_id, uint32_t locs_src_handle,
                                 uint32_t locs_dst_handle, uint32_t row_size,
                                 uint32_t n_rows) {
  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return ADAMAH_ERR_INVALID;
  if (n_rows == 0)
    return ADAMAH_OK;
  if (!ctx.softmax_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;
  if (!ctx.softmax_pipe.pipeline)
    return ADAMAH_ERR_INVALID;

  Map *m = &ctx.maps[map_id];
  ResEntry *src_res = res_get(locs_src_handle);
  ResEntry *dst_res = res_get(locs_dst_handle);
  if (!src_res || !dst_res)
    return ADAMAH_ERR_INVALID;
  if ((VkDeviceSize)n_rows * 4 > src_res->size_bytes)
    return ADAMAH_ERR_INVALID;
  if ((VkDeviceSize)n_rows * 4 > dst_res->size_bytes)
    return ADAMAH_ERR_INVALID;
  res_pin(locs_src_handle);
  res_pin(locs_dst_handle);
  VkDeviceSize src_off = 0;
  VkDeviceSize dst_off = 0;
  if (res_require_hot(locs_src_handle, &src_off) != 0 ||
      res_require_hot(locs_dst_handle, &dst_off) != 0) {
    res_unpin(locs_src_handle);
    res_unpin(locs_dst_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.softmax_pipe);
  if (ds == VK_NULL_HANDLE)
    return ADAMAH_ERR_VULKAN;

  VkDescriptorBufferInfo buf_infos[3] = {
      {.buffer = m->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf,
       .offset = src_off,
       .range = src_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = dst_off,
       .range = dst_res->size_bytes}};
  VkWriteDescriptorSet writes[3];
  for (int i = 0; i < 3; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);

  uint32_t push[2] = {n_rows, row_size};

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.softmax_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.softmax_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.softmax_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push);
  vkCmdDispatch(ctx.cmd, n_rows, 1, 1);
  cmd_barrier_after_dispatch();
  cmd_submit();
  res_unpin(locs_src_handle);
  res_unpin(locs_dst_handle);

  return ADAMAH_OK;
}

static int exec_softmax_abs_internal(uint32_t map_id, uint32_t locs_src_handle,
                                     uint32_t locs_dst_handle,
                                     uint32_t row_size, uint32_t n_rows) {
  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return ADAMAH_ERR_INVALID;
  if (n_rows == 0)
    return ADAMAH_OK;
  if (!ctx.softmax_abs_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;
  if (!ctx.softmax_abs_pipe.pipeline)
    return ADAMAH_ERR_INVALID;

  Map *m = &ctx.maps[map_id];
  ResEntry *src_res = res_get(locs_src_handle);
  ResEntry *dst_res = res_get(locs_dst_handle);
  if (!src_res || !dst_res)
    return ADAMAH_ERR_INVALID;
  if ((VkDeviceSize)n_rows * 4 > src_res->size_bytes)
    return ADAMAH_ERR_INVALID;
  if ((VkDeviceSize)n_rows * 4 > dst_res->size_bytes)
    return ADAMAH_ERR_INVALID;
  res_pin(locs_src_handle);
  res_pin(locs_dst_handle);
  VkDeviceSize src_off = 0;
  VkDeviceSize dst_off = 0;
  if (res_require_hot(locs_src_handle, &src_off) != 0 ||
      res_require_hot(locs_dst_handle, &dst_off) != 0) {
    res_unpin(locs_src_handle);
    res_unpin(locs_dst_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.softmax_abs_pipe);
  if (ds == VK_NULL_HANDLE)
    return ADAMAH_ERR_VULKAN;

  VkDescriptorBufferInfo buf_infos[3] = {
      {.buffer = m->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf,
       .offset = src_off,
       .range = src_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = dst_off,
       .range = dst_res->size_bytes}};
  VkWriteDescriptorSet writes[3];
  for (int i = 0; i < 3; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);

  uint32_t push[2] = {n_rows, row_size};

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.softmax_abs_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.softmax_abs_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.softmax_abs_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push);
  vkCmdDispatch(ctx.cmd, n_rows, 1, 1);
  cmd_barrier_after_dispatch();
  cmd_submit();
  res_unpin(locs_src_handle);
  res_unpin(locs_dst_handle);

  return ADAMAH_OK;
}

// Public API - uses fusion if enabled
int map_softmax_dev(uint32_t map_id, uint32_t locs_src_handle,
                    uint32_t locs_dst_handle, uint32_t n_rows,
                    uint32_t row_size) {
  if (fusion.enabled) {
    return fusion_queue_softmax(map_id, locs_src_handle, locs_dst_handle,
                                row_size, n_rows, 0);
  }
  return exec_softmax_internal(map_id, locs_src_handle, locs_dst_handle,
                               row_size, n_rows);
}

int map_softmax_abs_dev(uint32_t map_id, uint32_t locs_src_handle,
                        uint32_t locs_dst_handle, uint32_t n_rows,
                        uint32_t row_size) {
  if (fusion.enabled) {
    return fusion_queue_softmax(map_id, locs_src_handle, locs_dst_handle,
                                row_size, n_rows, 1);
  }
  return exec_softmax_abs_internal(map_id, locs_src_handle, locs_dst_handle,
                                   row_size, n_rows);
}

int map_attn_softmax_abs_dev(uint32_t map_id, uint32_t locs_src_handle,
                             uint32_t locs_dst_handle, uint32_t n_rows,
                             uint32_t row_size, float scale, float cap) {
  int fret = fusion_flush_pending_for_immediate();
  if (fret != ADAMAH_OK)
    return fret;
  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return ADAMAH_ERR_INVALID;
  if (n_rows == 0)
    return ADAMAH_OK;
  if (!ctx.attn_softmax_abs_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;
  if (!ctx.attn_softmax_abs_pipe.pipeline)
    return ADAMAH_ERR_INVALID;

  Map *m = &ctx.maps[map_id];
  ResEntry *src_res = res_get(locs_src_handle);
  ResEntry *dst_res = res_get(locs_dst_handle);
  if (!src_res || !dst_res)
    return ADAMAH_ERR_INVALID;
  if ((VkDeviceSize)n_rows * 4 > src_res->size_bytes)
    return ADAMAH_ERR_INVALID;
  if ((VkDeviceSize)n_rows * 4 > dst_res->size_bytes)
    return ADAMAH_ERR_INVALID;

  res_pin(locs_src_handle);
  res_pin(locs_dst_handle);
  VkDeviceSize src_off = 0;
  VkDeviceSize dst_off = 0;
  if (res_require_hot(locs_src_handle, &src_off) != 0 ||
      res_require_hot(locs_dst_handle, &dst_off) != 0) {
    res_unpin(locs_src_handle);
    res_unpin(locs_dst_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.attn_softmax_abs_pipe);
  if (ds == VK_NULL_HANDLE)
    return ADAMAH_ERR_VULKAN;

  VkDescriptorBufferInfo buf_infos[3] = {
      {.buffer = m->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf, .offset = src_off, .range = src_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = dst_off, .range = dst_res->size_bytes},
  };
  VkWriteDescriptorSet writes[3];
  for (int i = 0; i < 3; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds, .dstBinding = i, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);

  uint32_t push[4];
  push[0] = n_rows;
  push[1] = row_size;
  memcpy(&push[2], &scale, 4);
  memcpy(&push[3], &cap, 4);

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.attn_softmax_abs_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.attn_softmax_abs_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.attn_softmax_abs_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, push);
  vkCmdDispatch(ctx.cmd, n_rows, 1, 1);
  cmd_barrier_after_dispatch();
  cmd_submit();
  res_unpin(locs_src_handle);
  res_unpin(locs_dst_handle);
  return ADAMAH_OK;
}

// ============================================
// Repeat penalty: in-place adjustment of selected logits
// Applies penalty to map[locs_src[token_ids[i]]] for i in [0..n_ids).
// ============================================
int map_repeat_penalty_dev(uint32_t map_id, uint32_t locs_src_handle,
                           uint32_t token_ids_handle, uint32_t n_ids,
                           float penalty) {
  int fret = fusion_flush_pending_for_immediate();
  if (fret != ADAMAH_OK)
    return fret;
  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return ADAMAH_ERR_INVALID;
  if (n_ids == 0)
    return ADAMAH_OK;
  if (n_ids > 64)
    return ADAMAH_ERR_INVALID;
  if (!ctx.repeat_penalty_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;
  if (!ctx.repeat_penalty_pipe.pipeline)
    return ADAMAH_ERR_INVALID;

  Map *m = &ctx.maps[map_id];
  if (m->dtype != DTYPE_F32 || m->pack_size != 1)
    return ADAMAH_ERR_INVALID;

  ResEntry *src_res = res_get(locs_src_handle);
  ResEntry *tok_res = res_get(token_ids_handle);
  if (!src_res || !tok_res)
    return ADAMAH_ERR_INVALID;
  if ((VkDeviceSize)n_ids * 4 > tok_res->size_bytes)
    return ADAMAH_ERR_INVALID;

  res_pin(locs_src_handle);
  res_pin(token_ids_handle);
  VkDeviceSize src_off = 0;
  VkDeviceSize tok_off = 0;
  if (res_require_hot(locs_src_handle, &src_off) != 0 ||
      res_require_hot(token_ids_handle, &tok_off) != 0) {
    res_unpin(locs_src_handle);
    res_unpin(token_ids_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.repeat_penalty_pipe);
  if (ds == VK_NULL_HANDLE) {
    res_unpin(locs_src_handle);
    res_unpin(token_ids_handle);
    return ADAMAH_ERR_VULKAN;
  }

  VkDescriptorBufferInfo buf_infos[3] = {
      {.buffer = m->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf, .offset = src_off, .range = src_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = tok_off, .range = tok_res->size_bytes},
  };
  VkWriteDescriptorSet writes[3];
  for (int i = 0; i < 3; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);

  typedef struct {
    uint32_t n_ids;
    float penalty;
  } RepeatPenaltyPush;
  RepeatPenaltyPush push = {.n_ids = n_ids, .penalty = penalty};

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.repeat_penalty_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.repeat_penalty_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.repeat_penalty_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &push);
  vkCmdDispatch(ctx.cmd, (n_ids + 63) / 64, 1, 1);
  cmd_barrier_after_dispatch();
  cmd_submit();
  res_unpin(locs_src_handle);
  res_unpin(token_ids_handle);
  return ADAMAH_OK;
}

// ============================================
// Argmax: reduce a 1D scalar locs list to the winning index
// Writes the winning source index [0..n-1] as float32 into locs_dst[0].
// ============================================
int map_argmax_dev(uint32_t map_id, uint32_t locs_src_handle,
                   uint32_t locs_dst_handle, uint32_t n) {
  int fret = fusion_flush_pending_for_immediate();
  if (fret != ADAMAH_OK)
    return fret;
  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return ADAMAH_ERR_INVALID;
  if (n == 0)
    return ADAMAH_OK;
  if (!ctx.argmax_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;
  if (!ctx.argmax_pipe.pipeline) {
    if (debug_enabled()) {
      fprintf(stderr, "ADAMAH DEBUG: argmax missing pipeline\n");
    }
    return ADAMAH_ERR_INVALID;
  }

  Map *m = &ctx.maps[map_id];
  if (m->dtype != DTYPE_F32 || m->pack_size != 1) {
    if (debug_enabled()) {
      fprintf(stderr, "ADAMAH DEBUG: argmax unsupported dtype=%u pack=%u\n",
              m->dtype, m->pack_size);
    }
    return ADAMAH_ERR_INVALID;
  }

  ResEntry *src_res = res_get(locs_src_handle);
  ResEntry *dst_res = res_get(locs_dst_handle);
  if (!src_res || !dst_res) {
    if (debug_enabled()) {
      fprintf(stderr, "ADAMAH DEBUG: argmax bad resources src=%p dst=%p\n",
              (void *)src_res, (void *)dst_res);
    }
    return ADAMAH_ERR_INVALID;
  }
  if ((VkDeviceSize)n * 4 > src_res->size_bytes || dst_res->size_bytes < 4) {
    if (debug_enabled()) {
      fprintf(stderr,
              "ADAMAH DEBUG: argmax bad sizes n=%u src_bytes=%u dst_bytes=%u\n",
              n, src_res->size_bytes, dst_res->size_bytes);
    }
    return ADAMAH_ERR_INVALID;
  }

  res_pin(locs_src_handle);
  res_pin(locs_dst_handle);
  VkDeviceSize src_off = 0;
  VkDeviceSize dst_off = 0;
  if (res_require_hot(locs_src_handle, &src_off) != 0 ||
      res_require_hot(locs_dst_handle, &dst_off) != 0) {
    if (debug_enabled()) {
      fprintf(stderr, "ADAMAH DEBUG: argmax res_require_hot failed\n");
    }
    res_unpin(locs_src_handle);
    res_unpin(locs_dst_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.argmax_pipe);
  if (ds == VK_NULL_HANDLE) {
    res_unpin(locs_src_handle);
    res_unpin(locs_dst_handle);
    return ADAMAH_ERR_VULKAN;
  }

  VkDescriptorBufferInfo buf_infos[3] = {
      {.buffer = m->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf, .offset = src_off, .range = src_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = dst_off, .range = dst_res->size_bytes},
  };
  VkWriteDescriptorSet writes[3];
  for (int i = 0; i < 3; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.argmax_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.argmax_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.argmax_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &n);
  vkCmdDispatch(ctx.cmd, 1, 1, 1);
  cmd_barrier_after_dispatch();
  cmd_submit();
  res_unpin(locs_src_handle);
  res_unpin(locs_dst_handle);
  return ADAMAH_OK;
}

// ============================================
// Top-k: select the k best source indices and values from a scalar locs list
// Writes source indices [0..n-1] to locs_idx_dst[0..k-1] and corresponding
// values to locs_val_dst[0..k-1], both as float32.
// ============================================
int map_topk_dev(uint32_t map_id, uint32_t locs_src_handle,
                 uint32_t locs_idx_dst_handle, uint32_t locs_val_dst_handle,
                 uint32_t n, uint32_t k) {
  int fret = fusion_flush_pending_for_immediate();
  if (fret != ADAMAH_OK)
    return fret;
  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return ADAMAH_ERR_INVALID;
  if (n == 0 || k == 0)
    return ADAMAH_OK;
  if (k > 64)
    return ADAMAH_ERR_INVALID;
  if (!ctx.topk_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;
  if (!ctx.topk_pipe.pipeline) {
    if (debug_enabled()) {
      fprintf(stderr, "ADAMAH DEBUG: topk missing pipeline\n");
    }
    return ADAMAH_ERR_INVALID;
  }

  Map *m = &ctx.maps[map_id];
  if (m->dtype != DTYPE_F32 || m->pack_size != 1) {
    if (debug_enabled()) {
      fprintf(stderr, "ADAMAH DEBUG: topk unsupported dtype=%u pack=%u\n",
              m->dtype, m->pack_size);
    }
    return ADAMAH_ERR_INVALID;
  }

  ResEntry *src_res = res_get(locs_src_handle);
  ResEntry *idx_res = res_get(locs_idx_dst_handle);
  ResEntry *val_res = res_get(locs_val_dst_handle);
  if (!src_res || !idx_res || !val_res) {
    if (debug_enabled()) {
      fprintf(stderr, "ADAMAH DEBUG: topk bad resources src=%p idx=%p val=%p\n",
              (void *)src_res, (void *)idx_res, (void *)val_res);
    }
    return ADAMAH_ERR_INVALID;
  }
  if ((VkDeviceSize)n * 4 > src_res->size_bytes ||
      (VkDeviceSize)k * 4 > idx_res->size_bytes ||
      (VkDeviceSize)k * 4 > val_res->size_bytes) {
    if (debug_enabled()) {
      fprintf(stderr,
              "ADAMAH DEBUG: topk bad sizes n=%u k=%u src_bytes=%u idx_bytes=%u val_bytes=%u\n",
              n, k, src_res->size_bytes, idx_res->size_bytes, val_res->size_bytes);
    }
    return ADAMAH_ERR_INVALID;
  }

  res_pin(locs_src_handle);
  res_pin(locs_idx_dst_handle);
  res_pin(locs_val_dst_handle);
  VkDeviceSize src_off = 0;
  VkDeviceSize idx_off = 0;
  VkDeviceSize val_off = 0;
  if (res_require_hot(locs_src_handle, &src_off) != 0 ||
      res_require_hot(locs_idx_dst_handle, &idx_off) != 0 ||
      res_require_hot(locs_val_dst_handle, &val_off) != 0) {
    if (debug_enabled()) {
      fprintf(stderr, "ADAMAH DEBUG: topk res_require_hot failed\n");
    }
    res_unpin(locs_src_handle);
    res_unpin(locs_idx_dst_handle);
    res_unpin(locs_val_dst_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.topk_pipe);
  if (ds == VK_NULL_HANDLE) {
    res_unpin(locs_src_handle);
    res_unpin(locs_idx_dst_handle);
    res_unpin(locs_val_dst_handle);
    return ADAMAH_ERR_VULKAN;
  }

  VkDescriptorBufferInfo buf_infos[4] = {
      {.buffer = m->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf, .offset = src_off, .range = src_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = idx_off, .range = idx_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = val_off, .range = val_res->size_bytes},
  };
  VkWriteDescriptorSet writes[4];
  for (int i = 0; i < 4; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 4, writes, 0, NULL);

  uint32_t push[2] = {n, k};

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.topk_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.topk_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.topk_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push);
  vkCmdDispatch(ctx.cmd, 1, 1, 1);
  cmd_barrier_after_dispatch();
  cmd_submit();
  res_unpin(locs_src_handle);
  res_unpin(locs_idx_dst_handle);
  res_unpin(locs_val_dst_handle);
  return ADAMAH_OK;
}

// ============================================
// Top-p: normalize a sorted shortlist and zero out entries beyond cumulative p
// Writes normalized probabilities into locs_dst[0..n-1].
// ============================================
int map_topp_dev(uint32_t map_id, uint32_t locs_src_handle,
                 uint32_t locs_dst_handle, uint32_t n, float temperature,
                 float top_p) {
  int fret = fusion_flush_pending_for_immediate();
  if (fret != ADAMAH_OK)
    return fret;
  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return ADAMAH_ERR_INVALID;
  if (n == 0)
    return ADAMAH_OK;
  if (n > 64)
    return ADAMAH_ERR_INVALID;
  if (!ctx.topp_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;
  if (!ctx.topp_pipe.pipeline) {
    if (debug_enabled()) {
      fprintf(stderr, "ADAMAH DEBUG: topp missing pipeline\n");
    }
    return ADAMAH_ERR_INVALID;
  }

  Map *m = &ctx.maps[map_id];
  if (m->dtype != DTYPE_F32 || m->pack_size != 1) {
    if (debug_enabled()) {
      fprintf(stderr, "ADAMAH DEBUG: topp unsupported dtype=%u pack=%u\n",
              m->dtype, m->pack_size);
    }
    return ADAMAH_ERR_INVALID;
  }

  ResEntry *src_res = res_get(locs_src_handle);
  ResEntry *dst_res = res_get(locs_dst_handle);
  if (!src_res || !dst_res) {
    if (debug_enabled()) {
      fprintf(stderr, "ADAMAH DEBUG: topp bad resources src=%p dst=%p\n",
              (void *)src_res, (void *)dst_res);
    }
    return ADAMAH_ERR_INVALID;
  }
  if ((VkDeviceSize)n * 4 > src_res->size_bytes ||
      (VkDeviceSize)n * 4 > dst_res->size_bytes) {
    if (debug_enabled()) {
      fprintf(stderr,
              "ADAMAH DEBUG: topp bad sizes n=%u src_bytes=%u dst_bytes=%u\n",
              n, src_res->size_bytes, dst_res->size_bytes);
    }
    return ADAMAH_ERR_INVALID;
  }

  res_pin(locs_src_handle);
  res_pin(locs_dst_handle);
  VkDeviceSize src_off = 0;
  VkDeviceSize dst_off = 0;
  if (res_require_hot(locs_src_handle, &src_off) != 0 ||
      res_require_hot(locs_dst_handle, &dst_off) != 0) {
    if (debug_enabled()) {
      fprintf(stderr, "ADAMAH DEBUG: topp res_require_hot failed\n");
    }
    res_unpin(locs_src_handle);
    res_unpin(locs_dst_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.topp_pipe);
  if (ds == VK_NULL_HANDLE) {
    res_unpin(locs_src_handle);
    res_unpin(locs_dst_handle);
    return ADAMAH_ERR_VULKAN;
  }

  VkDescriptorBufferInfo buf_infos[3] = {
      {.buffer = m->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf, .offset = src_off, .range = src_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = dst_off, .range = dst_res->size_bytes},
  };
  VkWriteDescriptorSet writes[3];
  for (int i = 0; i < 3; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);

  typedef struct {
    uint32_t n_locs;
    float temperature;
    float top_p;
  } TopPPush;
  TopPPush push = {.n_locs = n, .temperature = temperature, .top_p = top_p};

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.topp_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.topp_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.topp_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &push);
  vkCmdDispatch(ctx.cmd, 1, 1, 1);
  cmd_barrier_after_dispatch();
  cmd_submit();
  res_unpin(locs_src_handle);
  res_unpin(locs_dst_handle);
  return ADAMAH_OK;
}

// ============================================
// LayerNorm: fused mean-var-normalize-scale-shift
// ============================================
int map_layernorm(uint32_t map_id, const uint32_t *locs_src,
                  const uint32_t *locs_dst, const uint32_t *locs_gamma,
                  const uint32_t *locs_beta, uint32_t n_rows, uint32_t dim,
                  float eps) {
  // Legacy path removed: use map_layernorm_dev with cached locs.
  (void)map_id;
  (void)locs_src;
  (void)locs_dst;
  (void)locs_gamma;
  (void)locs_beta;
  (void)n_rows;
  (void)dim;
  (void)eps;
  return ADAMAH_ERR_INVALID;
}

// Internal execution - bypasses fusion
static int exec_layernorm_internal(uint32_t map_id, uint32_t locs_src_handle,
                                   uint32_t locs_dst_handle,
                                   uint32_t locs_gamma_handle,
                                   uint32_t locs_beta_handle, uint32_t dim,
                                   uint32_t n_rows, float eps) {
  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return ADAMAH_ERR_INVALID;
  if (n_rows == 0)
    return ADAMAH_OK;
  if (!ctx.layernorm_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;
  if (!ctx.layernorm_pipe.pipeline)
    return ADAMAH_ERR_INVALID;

  Map *m = &ctx.maps[map_id];
  ResEntry *src_res = res_get(locs_src_handle);
  ResEntry *dst_res = res_get(locs_dst_handle);
  ResEntry *gamma_res = res_get(locs_gamma_handle);
  ResEntry *beta_res = res_get(locs_beta_handle);
  if (!src_res || !dst_res || !gamma_res || !beta_res)
    return ADAMAH_ERR_INVALID;
  if ((VkDeviceSize)n_rows * 4 > src_res->size_bytes)
    return ADAMAH_ERR_INVALID;
  if ((VkDeviceSize)n_rows * 4 > dst_res->size_bytes)
    return ADAMAH_ERR_INVALID;
  if ((VkDeviceSize)n_rows * 4 > gamma_res->size_bytes)
    return ADAMAH_ERR_INVALID;
  if ((VkDeviceSize)n_rows * 4 > beta_res->size_bytes)
    return ADAMAH_ERR_INVALID;
  res_pin(locs_src_handle);
  res_pin(locs_dst_handle);
  res_pin(locs_gamma_handle);
  res_pin(locs_beta_handle);
  VkDeviceSize src_off = 0;
  VkDeviceSize dst_off = 0;
  VkDeviceSize gamma_off = 0;
  VkDeviceSize beta_off = 0;
  if (res_require_hot(locs_src_handle, &src_off) != 0 ||
      res_require_hot(locs_dst_handle, &dst_off) != 0 ||
      res_require_hot(locs_gamma_handle, &gamma_off) != 0 ||
      res_require_hot(locs_beta_handle, &beta_off) != 0) {
    res_unpin(locs_src_handle);
    res_unpin(locs_dst_handle);
    res_unpin(locs_gamma_handle);
    res_unpin(locs_beta_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.layernorm_pipe);
  if (ds == VK_NULL_HANDLE)
    return ADAMAH_ERR_VULKAN;

  VkDescriptorBufferInfo buf_infos[5] = {
      {.buffer = m->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf,
       .offset = src_off,
       .range = src_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = dst_off,
       .range = dst_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = gamma_off,
       .range = gamma_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = beta_off,
       .range = beta_res->size_bytes}};
  VkWriteDescriptorSet writes[5];
  for (int i = 0; i < 5; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 5, writes, 0, NULL);

  uint32_t push[3];
  push[0] = n_rows;
  push[1] = dim;
  memcpy(&push[2], &eps, 4);

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.layernorm_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.layernorm_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.layernorm_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push);
  vkCmdDispatch(ctx.cmd, n_rows, 1, 1);
  cmd_barrier_after_dispatch();
  cmd_submit();
  res_unpin(locs_src_handle);
  res_unpin(locs_dst_handle);
  res_unpin(locs_gamma_handle);
  res_unpin(locs_beta_handle);

  return ADAMAH_OK;
}

// Public API - uses fusion if enabled
int map_layernorm_dev(uint32_t map_id, uint32_t locs_src_handle,
                      uint32_t locs_dst_handle, uint32_t locs_gamma_handle,
                      uint32_t locs_beta_handle, uint32_t n_rows, uint32_t dim,
                      float eps) {
  if (fusion.enabled) {
    return fusion_queue_layernorm(map_id, locs_src_handle, locs_dst_handle,
                                  locs_gamma_handle, locs_beta_handle, dim,
                                  n_rows, eps);
  }
  return exec_layernorm_internal(map_id, locs_src_handle, locs_dst_handle,
                                 locs_gamma_handle, locs_beta_handle, dim,
                                 n_rows, eps);
}

// ============================================
// Unified FFN (MLP) - 2-phase fused shader
// ============================================
int adamah_fused_ffn(uint32_t map_id, uint32_t loc_out, uint32_t loc_x,
                     uint32_t loc_w1, uint32_t loc_b1, uint32_t loc_w2,
                     uint32_t loc_b2, uint32_t BT, uint32_t D,
                     uint32_t apply_residual) {
  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return ADAMAH_ERR_INVALID;
  if (!ctx.unified_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;
  if (!ctx.unified_pipe.pipeline)
    return ADAMAH_ERR_INVALID;

  Map *m = &ctx.maps[map_id];
  uint32_t D4 = D * 4;

  GpuBuf *h_buf = get_or_create_buf("_ffn_h", BT * D4, 4);
  if (!h_buf)
    return ADAMAH_ERR_MEMORY;

  VkDescriptorSet ds = alloc_desc_set(&ctx.unified_pipe);
  if (ds == VK_NULL_HANDLE)
    return ADAMAH_ERR_VULKAN;

  VkDeviceSize x_bytes = (VkDeviceSize)BT * D * 4;
  VkDeviceSize w1_bytes = (VkDeviceSize)D * D4 * 4;
  VkDeviceSize b1_bytes = (VkDeviceSize)D4 * 4;
  VkDeviceSize w2_bytes = (VkDeviceSize)D4 * D * 4;
  VkDeviceSize b2_bytes = (VkDeviceSize)D * 4;
  VkDeviceSize y_bytes = (VkDeviceSize)BT * D * 4;
  VkDeviceSize h_bytes = (VkDeviceSize)BT * D4 * 4;

  VkDescriptorBufferInfo buf_infos[7] = {
      {.buffer = m->buf, .offset = (VkDeviceSize)loc_x * 4, .range = x_bytes},
      {.buffer = m->buf, .offset = (VkDeviceSize)loc_w1 * 4, .range = w1_bytes},
      {.buffer = m->buf, .offset = (VkDeviceSize)loc_b1 * 4, .range = b1_bytes},
      {.buffer = m->buf, .offset = (VkDeviceSize)loc_w2 * 4, .range = w2_bytes},
      {.buffer = m->buf, .offset = (VkDeviceSize)loc_b2 * 4, .range = b2_bytes},
      {.buffer = m->buf, .offset = (VkDeviceSize)loc_out * 4, .range = y_bytes},
      {.buffer = h_buf->buf, .offset = 0, .range = h_bytes}};
  VkWriteDescriptorSet writes[7];
  for (int i = 0; i < 7; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 7, writes, 0, NULL);

  uint32_t push_phase0[5] = {BT, D, D4, apply_residual, 0};
  uint32_t push_phase1[5] = {BT, D, D4, apply_residual, 1};

  uint32_t gx_h = (D4 + 15) / 16;
  uint32_t gy_h = (BT + 15) / 16;
  uint32_t gx_y = (D + 15) / 16;
  uint32_t gy_y = gy_h;

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.unified_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.unified_pipe.pipe_layout, 0, 1, &ds, 0, NULL);

  // Phase 0: H = GELU(X*W1 + b1)
  vkCmdPushConstants(ctx.cmd, ctx.unified_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, push_phase0);
  vkCmdDispatch(ctx.cmd, gx_h, gy_h, 1);

  // Barrier between phases
  VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
  vkCmdPipelineBarrier(ctx.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL,
                       0, NULL);

  // Phase 1: Y = H*W2 + b2 (+ residual)
  vkCmdPushConstants(ctx.cmd, ctx.unified_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, push_phase1);
  vkCmdDispatch(ctx.cmd, gx_y, gy_y, 1);
  cmd_submit();

  return ADAMAH_OK;
}

// ============================================
// RMSNorm: x / sqrt(mean(x²) + eps) * weight
// Used by Gemma/LLaMA instead of LayerNorm
// ============================================
int map_rmsnorm_dev(uint32_t map_id, uint32_t locs_src_handle,
                    uint32_t locs_wt_handle, uint32_t locs_dst_handle,
                    uint32_t n_rows, uint32_t dim, float eps) {
  int fret = fusion_flush_pending_for_immediate();
  if (fret != ADAMAH_OK)
    return fret;
  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return ADAMAH_ERR_INVALID;
  if (n_rows == 0)
    return ADAMAH_OK;
  if (!ctx.rmsnorm_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;
  if (!ctx.rmsnorm_pipe.pipeline)
    return ADAMAH_ERR_INVALID;

  Map *m = &ctx.maps[map_id];
  ResEntry *src_res = res_get(locs_src_handle);
  ResEntry *wt_res = res_get(locs_wt_handle);
  ResEntry *dst_res = res_get(locs_dst_handle);
  if (!src_res || !wt_res || !dst_res)
    return ADAMAH_ERR_INVALID;

  res_pin(locs_src_handle);
  res_pin(locs_wt_handle);
  res_pin(locs_dst_handle);
  VkDeviceSize src_off = 0, wt_off = 0, dst_off = 0;
  if (res_require_hot(locs_src_handle, &src_off) != 0 ||
      res_require_hot(locs_wt_handle, &wt_off) != 0 ||
      res_require_hot(locs_dst_handle, &dst_off) != 0) {
    res_unpin(locs_src_handle);
    res_unpin(locs_wt_handle);
    res_unpin(locs_dst_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.rmsnorm_pipe);
  if (ds == VK_NULL_HANDLE)
    return ADAMAH_ERR_VULKAN;

  VkDescriptorBufferInfo buf_infos[4] = {
      {.buffer = m->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf,
       .offset = src_off,
       .range = src_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = wt_off,
       .range = wt_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = dst_off,
       .range = dst_res->size_bytes}};
  VkWriteDescriptorSet writes[4];
  for (int i = 0; i < 4; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 4, writes, 0, NULL);

  uint32_t push[3];
  push[0] = n_rows;
  push[1] = dim;
  memcpy(&push[2], &eps, 4);

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.rmsnorm_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.rmsnorm_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.rmsnorm_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push);
  vkCmdDispatch(ctx.cmd, n_rows, 1, 1);
  cmd_barrier_after_dispatch();
  cmd_submit();
  res_unpin(locs_src_handle);
  res_unpin(locs_wt_handle);
  res_unpin(locs_dst_handle);
  return ADAMAH_OK;
}

// ============================================
// RoPE: Rotary Positional Encoding
// Standard for Gemma/LLaMA/GPT-NeoX
// ============================================
int map_rope_dev(uint32_t map_id, uint32_t locs_src_handle,
                 uint32_t locs_dst_handle, uint32_t n_tokens, uint32_t n_heads,
                 uint32_t head_dim, uint32_t pos_offset, float freq_base) {
  int fret = fusion_flush_pending_for_immediate();
  if (fret != ADAMAH_OK)
    return fret;
  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return ADAMAH_ERR_INVALID;
  if (n_tokens == 0)
    return ADAMAH_OK;
  if (!ctx.rope_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;
  if (!ctx.rope_pipe.pipeline)
    return ADAMAH_ERR_INVALID;

  Map *m = &ctx.maps[map_id];
  ResEntry *src_res = res_get(locs_src_handle);
  ResEntry *dst_res = res_get(locs_dst_handle);
  if (!src_res || !dst_res)
    return ADAMAH_ERR_INVALID;

  res_pin(locs_src_handle);
  res_pin(locs_dst_handle);
  VkDeviceSize src_off = 0, dst_off = 0;
  if (res_require_hot(locs_src_handle, &src_off) != 0 ||
      res_require_hot(locs_dst_handle, &dst_off) != 0) {
    res_unpin(locs_src_handle);
    res_unpin(locs_dst_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.rope_pipe);
  if (ds == VK_NULL_HANDLE)
    return ADAMAH_ERR_VULKAN;

  VkDescriptorBufferInfo buf_infos[3] = {
      {.buffer = m->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf,
       .offset = src_off,
       .range = src_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = dst_off,
       .range = dst_res->size_bytes}};
  VkWriteDescriptorSet writes[3];
  for (int i = 0; i < 3; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);

  uint32_t push[5];
  push[0] = n_tokens;
  push[1] = n_heads;
  push[2] = head_dim;
  push[3] = pos_offset;
  memcpy(&push[4], &freq_base, 4);

  uint32_t total_pairs = n_tokens * n_heads * (head_dim / 2);
  uint32_t n_groups = (total_pairs + 255) / 256;

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.rope_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.rope_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.rope_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, push);
  vkCmdDispatch(ctx.cmd, n_groups, 1, 1);
  cmd_barrier_after_dispatch();
  cmd_submit();
  res_unpin(locs_src_handle);
  res_unpin(locs_dst_handle);
  return ADAMAH_OK;
}

// ============================================
// MatMul Transposed: C = A @ B^T
// B stored as (N, K) — each col of B^T is a row
// ============================================
int map_matmul_t_dev(uint32_t map_id, uint32_t locs_a_handle,
                     uint32_t locs_b_handle, uint32_t locs_c_handle, uint32_t M,
                     uint32_t K, uint32_t N, uint32_t n_ops) {
  int fret = fusion_flush_pending_for_immediate();
  if (fret != ADAMAH_OK)
    return fret;
  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return ADAMAH_ERR_INVALID;
  if (n_ops == 0)
    return ADAMAH_OK;
  if (!ctx.matmul_t_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;
  if (!ctx.matmul_t_pipe.pipeline)
    return ADAMAH_ERR_INVALID;

  Map *m = &ctx.maps[map_id];
  ResEntry *a_res = res_get(locs_a_handle);
  ResEntry *b_res = res_get(locs_b_handle);
  ResEntry *c_res = res_get(locs_c_handle);
  if (!a_res || !b_res || !c_res)
    return ADAMAH_ERR_INVALID;

  res_pin(locs_a_handle);
  res_pin(locs_b_handle);
  res_pin(locs_c_handle);
  VkDeviceSize a_off = 0, b_off = 0, c_off = 0;
  if (res_require_hot(locs_a_handle, &a_off) != 0 ||
      res_require_hot(locs_b_handle, &b_off) != 0 ||
      res_require_hot(locs_c_handle, &c_off) != 0) {
    res_unpin(locs_a_handle);
    res_unpin(locs_b_handle);
    res_unpin(locs_c_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.matmul_t_pipe);
  if (ds == VK_NULL_HANDLE)
    return ADAMAH_ERR_VULKAN;

  VkDescriptorBufferInfo buf_infos[4] = {
      {.buffer = m->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf,
       .offset = a_off,
       .range = a_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = b_off,
       .range = b_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = c_off,
       .range = c_res->size_bytes}};
  VkWriteDescriptorSet writes[4];
  for (int i = 0; i < 4; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 4, writes, 0, NULL);

  uint32_t push[4] = {M, K, N, n_ops};
  uint32_t grid_x = (M + 15) / 16;
  uint32_t grid_y = (N + 15) / 16;

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.matmul_t_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.matmul_t_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.matmul_t_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, push);
  vkCmdDispatch(ctx.cmd, grid_x, grid_y, n_ops);
  cmd_barrier_after_dispatch();
  cmd_submit();
  res_unpin(locs_a_handle);
  res_unpin(locs_b_handle);
  res_unpin(locs_c_handle);
  return ADAMAH_OK;
}

// ============================================
// Row Copy: embedding table lookup
// copy_spec = [src_row_idx, dst_row_base] pairs
// dst_row_offset is added at dispatch time
// ============================================
int map_row_copy_offset_dev(uint32_t map_id, uint32_t copy_spec_handle,
                            uint32_t src_base_handle,
                            uint32_t dst_row_offset,
                            uint32_t n_copies, uint32_t row_size) {
  int fret = fusion_flush_pending_for_immediate();
  if (fret != ADAMAH_OK)
    return fret;
  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return ADAMAH_ERR_INVALID;
  if (n_copies == 0)
    return ADAMAH_OK;
  if (!ctx.row_copy_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;
  if (!ctx.row_copy_pipe.pipeline)
    return ADAMAH_ERR_INVALID;

  Map *m = &ctx.maps[map_id];
  ResEntry *spec_res = res_get(copy_spec_handle);
  ResEntry *base_res = res_get(src_base_handle);
  if (!spec_res || !base_res)
    return ADAMAH_ERR_INVALID;

  res_pin(copy_spec_handle);
  res_pin(src_base_handle);
  VkDeviceSize spec_off = 0, base_off = 0;
  if (res_require_hot(copy_spec_handle, &spec_off) != 0 ||
      res_require_hot(src_base_handle, &base_off) != 0) {
    res_unpin(copy_spec_handle);
    res_unpin(src_base_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.row_copy_pipe);
  if (ds == VK_NULL_HANDLE)
    return ADAMAH_ERR_VULKAN;

  VkDescriptorBufferInfo buf_infos[3] = {
      {.buffer = m->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf,
       .offset = spec_off,
       .range = spec_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = base_off,
       .range = base_res->size_bytes}};
  VkWriteDescriptorSet writes[3];
  for (int i = 0; i < 3; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);

  uint32_t push[3] = {n_copies, row_size, dst_row_offset};
  uint32_t total_threads = n_copies * row_size;
  uint32_t n_groups = (total_threads + 255) / 256;

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.row_copy_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.row_copy_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.row_copy_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push);
  vkCmdDispatch(ctx.cmd, n_groups, 1, 1);
  cmd_barrier_after_dispatch();
  cmd_submit();
  res_unpin(copy_spec_handle);
  res_unpin(src_base_handle);
  return ADAMAH_OK;
}

int map_row_copy_dev(uint32_t map_id, uint32_t copy_spec_handle,
                     uint32_t src_base_handle, uint32_t n_copies,
                     uint32_t row_size) {
  return map_row_copy_offset_dev(map_id, copy_spec_handle, src_base_handle, 0,
                                 n_copies, row_size);
}

// ============================================
// FMA: dst = a * b + c (element-wise)
// For residual connections, scale+bias
// ============================================
int map_fma_dev(uint32_t map_id, uint32_t locs_a_handle, uint32_t locs_b_handle,
                uint32_t locs_c_handle, uint32_t locs_dst_handle,
                uint32_t n_locs) {
  int fret = fusion_flush_pending_for_immediate();
  if (fret != ADAMAH_OK)
    return fret;
  if (map_id >= MAX_MAPS || !ctx.maps[map_id].active)
    return ADAMAH_ERR_INVALID;
  if (n_locs == 0)
    return ADAMAH_OK;
  if (!ctx.fma_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;
  if (!ctx.fma_pipe.pipeline)
    return ADAMAH_ERR_INVALID;

  Map *m = &ctx.maps[map_id];
  ResEntry *a_res = res_get(locs_a_handle);
  ResEntry *b_res = res_get(locs_b_handle);
  ResEntry *c_res = res_get(locs_c_handle);
  ResEntry *dst_res = res_get(locs_dst_handle);
  if (!a_res || !b_res || !c_res || !dst_res)
    return ADAMAH_ERR_INVALID;

  res_pin(locs_a_handle);
  res_pin(locs_b_handle);
  res_pin(locs_c_handle);
  res_pin(locs_dst_handle);
  VkDeviceSize a_off = 0, b_off = 0, c_off = 0, dst_off = 0;
  if (res_require_hot(locs_a_handle, &a_off) != 0 ||
      res_require_hot(locs_b_handle, &b_off) != 0 ||
      res_require_hot(locs_c_handle, &c_off) != 0 ||
      res_require_hot(locs_dst_handle, &dst_off) != 0) {
    res_unpin(locs_a_handle);
    res_unpin(locs_b_handle);
    res_unpin(locs_c_handle);
    res_unpin(locs_dst_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.fma_pipe);
  if (ds == VK_NULL_HANDLE)
    return ADAMAH_ERR_VULKAN;

  VkDescriptorBufferInfo buf_infos[5] = {
      {.buffer = m->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf,
       .offset = a_off,
       .range = a_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = b_off,
       .range = b_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = c_off,
       .range = c_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = dst_off,
       .range = dst_res->size_bytes}};
  VkWriteDescriptorSet writes[5];
  for (int i = 0; i < 5; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 5, writes, 0, NULL);

  uint32_t push[2] = {n_locs, m->pack_size};
  uint32_t total_threads = n_locs * m->pack_size;
  uint32_t n_groups_dispatch = (total_threads + 255) / 256;

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.fma_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.fma_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.fma_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push);
  vkCmdDispatch(ctx.cmd, n_groups_dispatch, 1, 1);
  cmd_barrier_after_dispatch();
  cmd_submit();
  res_unpin(locs_a_handle);
  res_unpin(locs_b_handle);
  res_unpin(locs_c_handle);
  res_unpin(locs_dst_handle);
  return ADAMAH_OK;
}

// ============================================
// Cross-map MatMul Transposed: C = A @ B^T
// A from map_act, B from map_wt, C to map_act
// ============================================
int map_matmul_t_x_dev(uint32_t map_act_id, uint32_t map_wt_id,
                       uint32_t locs_a_handle, uint32_t locs_b_handle,
                       uint32_t locs_c_handle, uint32_t M, uint32_t K,
                       uint32_t N, uint32_t n_ops) {
  int fret = fusion_flush_pending_for_immediate();
  if (fret != ADAMAH_OK)
    return fret;
  if (map_act_id >= MAX_MAPS || !ctx.maps[map_act_id].active)
    return ADAMAH_ERR_INVALID;
  if (map_wt_id >= MAX_MAPS || !ctx.maps[map_wt_id].active)
    return ADAMAH_ERR_INVALID;
  if (n_ops == 0)
    return ADAMAH_OK;
  if (!ctx.matmul_t_x_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;
  if (!ctx.matmul_t_x_pipe.pipeline)
    return ADAMAH_ERR_INVALID;

  Map *m_act = &ctx.maps[map_act_id];
  Map *m_wt = &ctx.maps[map_wt_id];
  ResEntry *a_res = res_get(locs_a_handle);
  ResEntry *b_res = res_get(locs_b_handle);
  ResEntry *c_res = res_get(locs_c_handle);
  if (!a_res || !b_res || !c_res)
    return ADAMAH_ERR_INVALID;

  res_pin(locs_a_handle);
  res_pin(locs_b_handle);
  res_pin(locs_c_handle);
  VkDeviceSize a_off = 0, b_off = 0, c_off = 0;
  if (res_require_hot(locs_a_handle, &a_off) != 0 ||
      res_require_hot(locs_b_handle, &b_off) != 0 ||
      res_require_hot(locs_c_handle, &c_off) != 0) {
    res_unpin(locs_a_handle);
    res_unpin(locs_b_handle);
    res_unpin(locs_c_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.matmul_t_x_pipe);
  if (ds == VK_NULL_HANDLE)
    return ADAMAH_ERR_VULKAN;

  VkDescriptorBufferInfo buf_infos[5] = {
      {.buffer = m_act->buf, .range = VK_WHOLE_SIZE},
      {.buffer = m_wt->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf,
       .offset = a_off,
       .range = a_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = b_off,
       .range = b_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = c_off,
       .range = c_res->size_bytes}};
  VkWriteDescriptorSet writes[5];
  for (int i = 0; i < 5; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 5, writes, 0, NULL);

  uint32_t push[4] = {M, K, N, n_ops};
  uint32_t grid_x = (M + 15) / 16;
  uint32_t grid_y = (N + 15) / 16;

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.matmul_t_x_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.matmul_t_x_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.matmul_t_x_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, push);
  vkCmdDispatch(ctx.cmd, grid_x, grid_y, n_ops);
  cmd_barrier_after_dispatch();
  cmd_submit();
  res_unpin(locs_a_handle);
  res_unpin(locs_b_handle);
  res_unpin(locs_c_handle);
  return ADAMAH_OK;
}

// ============================================
// Cross-map MatMul: C = A @ B
// A from map_act, B from map_wt, C to map_act
// ============================================
int map_matmul_x_dev(uint32_t map_act_id, uint32_t map_wt_id,
                     uint32_t locs_a_handle, uint32_t locs_b_handle,
                     uint32_t locs_c_handle, uint32_t M, uint32_t K, uint32_t N,
                     uint32_t n_ops) {
  int fret = fusion_flush_pending_for_immediate();
  if (fret != ADAMAH_OK)
    return fret;
  if (map_act_id >= MAX_MAPS || !ctx.maps[map_act_id].active)
    return ADAMAH_ERR_INVALID;
  if (map_wt_id >= MAX_MAPS || !ctx.maps[map_wt_id].active)
    return ADAMAH_ERR_INVALID;
  if (n_ops == 0)
    return ADAMAH_OK;
  if (!ctx.matmul_x_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;
  if (!ctx.matmul_x_pipe.pipeline)
    return ADAMAH_ERR_INVALID;

  Map *m_act = &ctx.maps[map_act_id];
  Map *m_wt = &ctx.maps[map_wt_id];
  ResEntry *a_res = res_get(locs_a_handle);
  ResEntry *b_res = res_get(locs_b_handle);
  ResEntry *c_res = res_get(locs_c_handle);
  if (!a_res || !b_res || !c_res)
    return ADAMAH_ERR_INVALID;

  res_pin(locs_a_handle);
  res_pin(locs_b_handle);
  res_pin(locs_c_handle);
  VkDeviceSize a_off = 0, b_off = 0, c_off = 0;
  if (res_require_hot(locs_a_handle, &a_off) != 0 ||
      res_require_hot(locs_b_handle, &b_off) != 0 ||
      res_require_hot(locs_c_handle, &c_off) != 0) {
    res_unpin(locs_a_handle);
    res_unpin(locs_b_handle);
    res_unpin(locs_c_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.matmul_x_pipe);
  if (ds == VK_NULL_HANDLE)
    return ADAMAH_ERR_VULKAN;

  VkDescriptorBufferInfo buf_infos[5] = {
      {.buffer = m_act->buf, .range = VK_WHOLE_SIZE},
      {.buffer = m_wt->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf,
       .offset = a_off,
       .range = a_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = b_off,
       .range = b_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = c_off,
       .range = c_res->size_bytes}};
  VkWriteDescriptorSet writes[5];
  for (int i = 0; i < 5; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 5, writes, 0, NULL);

  uint32_t push[4] = {M, K, N, n_ops};
  uint32_t grid_x = (M + 15) / 16;
  uint32_t grid_y = (N + 15) / 16;

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.matmul_x_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.matmul_x_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.matmul_x_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, push);
  vkCmdDispatch(ctx.cmd, grid_x, grid_y, n_ops);
  cmd_barrier_after_dispatch();
  cmd_submit();
  res_unpin(locs_a_handle);
  res_unpin(locs_b_handle);
  res_unpin(locs_c_handle);
  return ADAMAH_OK;
}

// ============================================
// Cross-map RMSNorm
// Activations from map_act, weights from map_wt
// ============================================
int map_rmsnorm_x_dev(uint32_t map_act_id, uint32_t map_wt_id,
                      uint32_t locs_src_handle, uint32_t locs_wt_handle,
                      uint32_t locs_dst_handle, uint32_t n_rows, uint32_t dim,
                      float eps) {
  int fret = fusion_flush_pending_for_immediate();
  if (fret != ADAMAH_OK)
    return fret;
  if (map_act_id >= MAX_MAPS || !ctx.maps[map_act_id].active)
    return ADAMAH_ERR_INVALID;
  if (map_wt_id >= MAX_MAPS || !ctx.maps[map_wt_id].active)
    return ADAMAH_ERR_INVALID;
  if (n_rows == 0)
    return ADAMAH_OK;
  if (!ctx.rmsnorm_x_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;
  if (!ctx.rmsnorm_x_pipe.pipeline)
    return ADAMAH_ERR_INVALID;

  Map *m_act = &ctx.maps[map_act_id];
  Map *m_wt = &ctx.maps[map_wt_id];
  ResEntry *src_res = res_get(locs_src_handle);
  ResEntry *wt_res = res_get(locs_wt_handle);
  ResEntry *dst_res = res_get(locs_dst_handle);
  if (!src_res || !wt_res || !dst_res)
    return ADAMAH_ERR_INVALID;

  res_pin(locs_src_handle);
  res_pin(locs_wt_handle);
  res_pin(locs_dst_handle);
  VkDeviceSize src_off = 0, wt_off = 0, dst_off = 0;
  if (res_require_hot(locs_src_handle, &src_off) != 0 ||
      res_require_hot(locs_wt_handle, &wt_off) != 0 ||
      res_require_hot(locs_dst_handle, &dst_off) != 0) {
    res_unpin(locs_src_handle);
    res_unpin(locs_wt_handle);
    res_unpin(locs_dst_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.rmsnorm_x_pipe);
  if (ds == VK_NULL_HANDLE)
    return ADAMAH_ERR_VULKAN;

  VkDescriptorBufferInfo buf_infos[5] = {
      {.buffer = m_act->buf, .range = VK_WHOLE_SIZE},
      {.buffer = m_wt->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf,
       .offset = src_off,
       .range = src_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = wt_off,
       .range = wt_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = dst_off,
       .range = dst_res->size_bytes}};
  VkWriteDescriptorSet writes[5];
  for (int i = 0; i < 5; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 5, writes, 0, NULL);

  uint32_t push[3];
  push[0] = n_rows;
  push[1] = dim;
  memcpy(&push[2], &eps, 4);

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.rmsnorm_x_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.rmsnorm_x_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.rmsnorm_x_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push);
  vkCmdDispatch(ctx.cmd, n_rows, 1, 1);
  cmd_barrier_after_dispatch();
  cmd_submit();
  res_unpin(locs_src_handle);
  res_unpin(locs_wt_handle);
  res_unpin(locs_dst_handle);
  return ADAMAH_OK;
}

// ============================================
// Cross-map MatMul F32×Q4: C = A @ B^T
// A from map_act (F32), B from map_wt (Q4), C to map_act (F32)
// ============================================
static int exec_matvec_t_xq_internal(Pipeline *pipe, uint32_t map_act_id,
                                     uint32_t map_wt_id,
                                     uint32_t locs_a_handle,
                                     uint32_t locs_b_handle,
                                     uint32_t locs_c_handle, uint32_t K,
                                     uint32_t N) {
  const uint32_t rows_per_group = 1;
  if (map_act_id >= MAX_MAPS || !ctx.maps[map_act_id].active)
    return ADAMAH_ERR_INVALID;
  if (map_wt_id >= MAX_MAPS || !ctx.maps[map_wt_id].active)
    return ADAMAH_ERR_INVALID;
  if (N == 0)
    return ADAMAH_OK;

  Map *m_act = &ctx.maps[map_act_id];
  Map *m_wt = &ctx.maps[map_wt_id];
  ResEntry *a_res = res_get(locs_a_handle);
  ResEntry *b_res = res_get(locs_b_handle);
  ResEntry *c_res = res_get(locs_c_handle);
  if (!a_res || !b_res || !c_res)
    return ADAMAH_ERR_INVALID;
  if (m_wt->qparam_buf == VK_NULL_HANDLE)
    return ADAMAH_ERR_INVALID;

  res_pin(locs_a_handle);
  res_pin(locs_b_handle);
  res_pin(locs_c_handle);

  VkDeviceSize a_off = 0, b_off = 0, c_off = 0;
  if (res_require_hot(locs_a_handle, &a_off) != 0 ||
      res_require_hot(locs_b_handle, &b_off) != 0 ||
      res_require_hot(locs_c_handle, &c_off) != 0) {
    res_unpin(locs_a_handle);
    res_unpin(locs_b_handle);
    res_unpin(locs_c_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(pipe);
  if (ds == VK_NULL_HANDLE) {
    res_unpin(locs_a_handle);
    res_unpin(locs_b_handle);
    res_unpin(locs_c_handle);
    return ADAMAH_ERR_VULKAN;
  }

  VkDescriptorBufferInfo buf_infos[6] = {
      {.buffer = m_act->buf, .range = VK_WHOLE_SIZE},
      {.buffer = m_wt->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf, .offset = a_off, .range = a_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = b_off, .range = b_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = c_off, .range = c_res->size_bytes},
      {.buffer = m_wt->qparam_buf, .range = VK_WHOLE_SIZE},
  };
  VkWriteDescriptorSet writes[6];
  for (int i = 0; i < 6; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 6, writes, 0, NULL);

  uint32_t push[3] = {K, N, m_wt->group_size};
  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipe->pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, pipe->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     12, push);
  vkCmdDispatch(ctx.cmd, (N + rows_per_group - 1) / rows_per_group, 1, 1);
  cmd_barrier_after_dispatch();
  cmd_submit();

  res_unpin(locs_a_handle);
  res_unpin(locs_b_handle);
  res_unpin(locs_c_handle);
  return ADAMAH_OK;
}

int map_matmul_t_xq4_dev(uint32_t map_act_id, uint32_t map_wt_id,
                         uint32_t locs_a_handle, uint32_t locs_b_handle,
                         uint32_t locs_c_handle, uint32_t M, uint32_t K,
                         uint32_t N, uint32_t n_ops) {
  int fret = fusion_flush_pending_for_immediate();
  if (fret != ADAMAH_OK)
    return fret;
  if (map_act_id >= MAX_MAPS || !ctx.maps[map_act_id].active)
    return ADAMAH_ERR_INVALID;
  if (map_wt_id >= MAX_MAPS || !ctx.maps[map_wt_id].active)
    return ADAMAH_ERR_INVALID;
  if (n_ops == 0)
    return ADAMAH_OK;
  if (M == 1 && n_ops == 1) {
    if (!ctx.matvec_t_xq4_pipe.pipeline && init_pipelines() != 0)
      return ADAMAH_ERR_VULKAN;
    if (!ctx.matvec_t_xq4_pipe.pipeline)
      return ADAMAH_ERR_INVALID;
    return exec_matvec_t_xq_internal(&ctx.matvec_t_xq4_pipe, map_act_id,
                                     map_wt_id, locs_a_handle, locs_b_handle,
                                     locs_c_handle, K, N);
  }
  if (!ctx.matmul_t_xq4_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;
  if (!ctx.matmul_t_xq4_pipe.pipeline)
    return ADAMAH_ERR_INVALID;

  Map *m_act = &ctx.maps[map_act_id];
  Map *m_wt = &ctx.maps[map_wt_id];
  ResEntry *a_res = res_get(locs_a_handle);
  ResEntry *b_res = res_get(locs_b_handle);
  ResEntry *c_res = res_get(locs_c_handle);
  if (!a_res || !b_res || !c_res)
    return ADAMAH_ERR_INVALID;
  if (m_wt->qparam_buf == VK_NULL_HANDLE)
    return ADAMAH_ERR_INVALID;

  res_pin(locs_a_handle);
  res_pin(locs_b_handle);
  res_pin(locs_c_handle);
  VkDeviceSize a_off = 0, b_off = 0, c_off = 0;
  if (res_require_hot(locs_a_handle, &a_off) != 0 ||
      res_require_hot(locs_b_handle, &b_off) != 0 ||
      res_require_hot(locs_c_handle, &c_off) != 0) {
    res_unpin(locs_a_handle);
    res_unpin(locs_b_handle);
    res_unpin(locs_c_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.matmul_t_xq4_pipe);
  if (ds == VK_NULL_HANDLE)
    return ADAMAH_ERR_VULKAN;

  VkDescriptorBufferInfo buf_infos[6] = {
      {.buffer = m_act->buf, .range = VK_WHOLE_SIZE},
      {.buffer = m_wt->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf,
       .offset = a_off,
       .range = a_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = b_off,
       .range = b_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = c_off,
       .range = c_res->size_bytes},
      {.buffer = m_wt->qparam_buf, .range = VK_WHOLE_SIZE}};
  VkWriteDescriptorSet writes[6];
  for (int i = 0; i < 6; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 6, writes, 0, NULL);

  uint32_t push[5] = {M, K, N, n_ops, m_wt->group_size};
  uint32_t grid_x = (M + 15) / 16;
  uint32_t grid_y = (N + 15) / 16;

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.matmul_t_xq4_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.matmul_t_xq4_pipe.pipe_layout, 0, 1, &ds, 0,
                          NULL);
  vkCmdPushConstants(ctx.cmd, ctx.matmul_t_xq4_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, push);
  vkCmdDispatch(ctx.cmd, grid_x, grid_y, n_ops);
  cmd_barrier_after_dispatch();
  cmd_submit();
  res_unpin(locs_a_handle);
  res_unpin(locs_b_handle);
  res_unpin(locs_c_handle);
  return ADAMAH_OK;
}

// ============================================
// Cross-map MatMul F32×Q4: C = A @ B (non-transposed)
// A from map_act (F32), B from map_wt (Q4), C to map_act (F32)
// ============================================
int map_matmul_xq4_dev(uint32_t map_act_id, uint32_t map_wt_id,
                       uint32_t locs_a_handle, uint32_t locs_b_handle,
                       uint32_t locs_c_handle, uint32_t M, uint32_t K,
                       uint32_t N, uint32_t n_ops) {
  int fret = fusion_flush_pending_for_immediate();
  if (fret != ADAMAH_OK)
    return fret;
  if (map_act_id >= MAX_MAPS || !ctx.maps[map_act_id].active)
    return ADAMAH_ERR_INVALID;
  if (map_wt_id >= MAX_MAPS || !ctx.maps[map_wt_id].active)
    return ADAMAH_ERR_INVALID;
  if (n_ops == 0)
    return ADAMAH_OK;
  if (!ctx.matmul_xq4_pipe.pipeline && init_pipelines() != 0)
    return ADAMAH_ERR_VULKAN;
  if (!ctx.matmul_xq4_pipe.pipeline)
    return ADAMAH_ERR_INVALID;

  Map *m_act = &ctx.maps[map_act_id];
  Map *m_wt = &ctx.maps[map_wt_id];
  ResEntry *a_res = res_get(locs_a_handle);
  ResEntry *b_res = res_get(locs_b_handle);
  ResEntry *c_res = res_get(locs_c_handle);
  if (!a_res || !b_res || !c_res)
    return ADAMAH_ERR_INVALID;
  if (m_wt->qparam_buf == VK_NULL_HANDLE)
    return ADAMAH_ERR_INVALID;

  res_pin(locs_a_handle);
  res_pin(locs_b_handle);
  res_pin(locs_c_handle);
  VkDeviceSize a_off = 0, b_off = 0, c_off = 0;
  if (res_require_hot(locs_a_handle, &a_off) != 0 ||
      res_require_hot(locs_b_handle, &b_off) != 0 ||
      res_require_hot(locs_c_handle, &c_off) != 0) {
    res_unpin(locs_a_handle);
    res_unpin(locs_b_handle);
    res_unpin(locs_c_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.matmul_xq4_pipe);
  if (ds == VK_NULL_HANDLE)
    return ADAMAH_ERR_VULKAN;

  VkDescriptorBufferInfo buf_infos[6] = {
      {.buffer = m_act->buf, .range = VK_WHOLE_SIZE},
      {.buffer = m_wt->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf,
       .offset = a_off,
       .range = a_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = b_off,
       .range = b_res->size_bytes},
      {.buffer = ctx.hot_pool->buf,
       .offset = c_off,
       .range = c_res->size_bytes},
      {.buffer = m_wt->qparam_buf, .range = VK_WHOLE_SIZE}};
  VkWriteDescriptorSet writes[6];
  for (int i = 0; i < 6; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 6, writes, 0, NULL);

  uint32_t push[5] = {M, K, N, n_ops, m_wt->group_size};
  uint32_t grid_x = (M + 15) / 16;
  uint32_t grid_y = (N + 15) / 16;

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    ctx.matmul_xq4_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.matmul_xq4_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.matmul_xq4_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, push);
  vkCmdDispatch(ctx.cmd, grid_x, grid_y, n_ops);
  cmd_barrier_after_dispatch();
  cmd_submit();
  res_unpin(locs_a_handle);
  res_unpin(locs_b_handle);
  res_unpin(locs_c_handle);
  return ADAMAH_OK;
}

// ============================================
// Cross-map MatVec+TopK F32xQ4: partial top-k directly from A @ B^T
// A from map_act (single row), B from map_wt (Q4), output partial idx/val to map_act.
// ============================================
int map_matvec_topk_t_xq4_ex_dev(uint32_t map_act_id, uint32_t map_wt_id,
                                 uint32_t locs_a_handle, uint32_t locs_b_handle,
                                 uint32_t penalty_ids_handle,
                                 uint32_t locs_idx_dst_handle,
                                 uint32_t locs_val_dst_handle,
                                 uint32_t K, uint32_t N, uint32_t k,
                                 uint32_t n_penalty, float repeat_penalty,
                                 uint32_t rows_per_group) {
  int fret = fusion_flush_pending_for_immediate();
  if (fret != ADAMAH_OK) return fret;
  if (map_act_id >= MAX_MAPS || !ctx.maps[map_act_id].active) return ADAMAH_ERR_INVALID;
  if (map_wt_id >= MAX_MAPS || !ctx.maps[map_wt_id].active) return ADAMAH_ERR_INVALID;
  if (N == 0 || k == 0) return ADAMAH_OK;
  if (k > 64 || n_penalty > 64) return ADAMAH_ERR_INVALID;
  if (rows_per_group == 0) return ADAMAH_ERR_INVALID;
  if (!ctx.matvec_topk_t_xq4_pipe.pipeline && init_pipelines() != 0) return ADAMAH_ERR_VULKAN;
  if (!ctx.matvec_topk_t_xq4_pipe.pipeline) return ADAMAH_ERR_INVALID;

  Map *m_act = &ctx.maps[map_act_id];
  Map *m_wt = &ctx.maps[map_wt_id];
  ResEntry *a_res = res_get(locs_a_handle);
  ResEntry *b_res = res_get(locs_b_handle);
  ResEntry *p_res = res_get(penalty_ids_handle);
  ResEntry *i_res = res_get(locs_idx_dst_handle);
  ResEntry *v_res = res_get(locs_val_dst_handle);
  if (!a_res || !b_res || !p_res || !i_res || !v_res) return ADAMAH_ERR_INVALID;
  if (m_wt->qparam_buf == VK_NULL_HANDLE) return ADAMAH_ERR_INVALID;

  res_pin(locs_a_handle);
  res_pin(locs_b_handle);
  res_pin(penalty_ids_handle);
  res_pin(locs_idx_dst_handle);
  res_pin(locs_val_dst_handle);

  VkDeviceSize a_off = 0, b_off = 0, p_off = 0, i_off = 0, v_off = 0;
  if (res_require_hot(locs_a_handle, &a_off) != 0 ||
      res_require_hot(locs_b_handle, &b_off) != 0 ||
      res_require_hot(penalty_ids_handle, &p_off) != 0 ||
      res_require_hot(locs_idx_dst_handle, &i_off) != 0 ||
      res_require_hot(locs_val_dst_handle, &v_off) != 0) {
    res_unpin(locs_a_handle); res_unpin(locs_b_handle); res_unpin(penalty_ids_handle);
    res_unpin(locs_idx_dst_handle); res_unpin(locs_val_dst_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.matvec_topk_t_xq4_pipe);
  if (ds == VK_NULL_HANDLE) {
    res_unpin(locs_a_handle); res_unpin(locs_b_handle); res_unpin(penalty_ids_handle);
    res_unpin(locs_idx_dst_handle); res_unpin(locs_val_dst_handle);
    return ADAMAH_ERR_VULKAN;
  }

  VkDescriptorBufferInfo buf_infos[8] = {
      {.buffer = m_act->buf, .range = VK_WHOLE_SIZE},
      {.buffer = m_wt->buf,  .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf, .offset = a_off, .range = a_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = b_off, .range = b_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = p_off, .range = p_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = i_off, .range = i_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = v_off, .range = v_res->size_bytes},
      {.buffer = m_wt->qparam_buf,  .range = VK_WHOLE_SIZE},
  };
  VkWriteDescriptorSet writes[8];
  for (int i = 0; i < 8; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 8, writes, 0, NULL);

  typedef struct {
    uint32_t K;
    uint32_t N;
    uint32_t k;
    uint32_t group_size;
    uint32_t n_penalty;
    float repeat_penalty;
    uint32_t rows_per_group;
  } MatvecTopKPush;
  MatvecTopKPush push = {
      K, N, k, m_wt->group_size, n_penalty, repeat_penalty, rows_per_group};

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.matvec_topk_t_xq4_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.matvec_topk_t_xq4_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.matvec_topk_t_xq4_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 28, &push);
  vkCmdDispatch(ctx.cmd, (N + rows_per_group - 1) / rows_per_group, 1, 1);
  cmd_barrier_after_dispatch();
  cmd_submit();

  res_unpin(locs_a_handle); res_unpin(locs_b_handle); res_unpin(penalty_ids_handle);
  res_unpin(locs_idx_dst_handle); res_unpin(locs_val_dst_handle);
  return ADAMAH_OK;
}

int map_matvec_topk_t_xq4_dev(uint32_t map_act_id, uint32_t map_wt_id,
                              uint32_t locs_a_handle, uint32_t locs_b_handle,
                              uint32_t penalty_ids_handle,
                              uint32_t locs_idx_dst_handle,
                              uint32_t locs_val_dst_handle,
                              uint32_t K, uint32_t N, uint32_t k,
                              uint32_t n_penalty, float repeat_penalty) {
  return map_matvec_topk_t_xq4_ex_dev(
      map_act_id, map_wt_id,
      locs_a_handle, locs_b_handle, penalty_ids_handle,
      locs_idx_dst_handle, locs_val_dst_handle,
      K, N, k, n_penalty, repeat_penalty, 256);
}

// ============================================
// Cross-map MatMul F32xQ8: C = A @ B^T
// A from map_act (F32), B from map_wt (Q8), C to map_act (F32)
// ============================================
int map_matmul_t_xq8_dev(uint32_t map_act_id, uint32_t map_wt_id,
                         uint32_t locs_a_handle, uint32_t locs_b_handle,
                         uint32_t locs_c_handle, uint32_t M, uint32_t K,
                         uint32_t N, uint32_t n_ops) {
  int fret = fusion_flush_pending_for_immediate();
  if (fret != ADAMAH_OK) return fret;
  if (map_act_id >= MAX_MAPS || !ctx.maps[map_act_id].active) return ADAMAH_ERR_INVALID;
  if (map_wt_id  >= MAX_MAPS || !ctx.maps[map_wt_id].active)  return ADAMAH_ERR_INVALID;
  if (n_ops == 0) return ADAMAH_OK;
  if (M == 1 && n_ops == 1) {
    if (!ctx.matvec_t_xq8_pipe.pipeline && init_pipelines() != 0) return ADAMAH_ERR_VULKAN;
    if (!ctx.matvec_t_xq8_pipe.pipeline) return ADAMAH_ERR_INVALID;
    return exec_matvec_t_xq_internal(&ctx.matvec_t_xq8_pipe, map_act_id, map_wt_id,
                                     locs_a_handle, locs_b_handle, locs_c_handle, K, N);
  }
  if (!ctx.matmul_t_xq8_pipe.pipeline && init_pipelines() != 0) return ADAMAH_ERR_VULKAN;
  if (!ctx.matmul_t_xq8_pipe.pipeline) return ADAMAH_ERR_INVALID;
  Map *m_act = &ctx.maps[map_act_id]; Map *m_wt = &ctx.maps[map_wt_id];
  ResEntry *a_res = res_get(locs_a_handle);
  ResEntry *b_res = res_get(locs_b_handle);
  ResEntry *c_res = res_get(locs_c_handle);
  if (!a_res || !b_res || !c_res) return ADAMAH_ERR_INVALID;
  if (m_wt->qparam_buf == VK_NULL_HANDLE) return ADAMAH_ERR_INVALID;
  res_pin(locs_a_handle); res_pin(locs_b_handle); res_pin(locs_c_handle);
  VkDeviceSize a_off = 0, b_off = 0, c_off = 0;
  if (res_require_hot(locs_a_handle, &a_off) != 0 ||
      res_require_hot(locs_b_handle, &b_off) != 0 ||
      res_require_hot(locs_c_handle, &c_off) != 0) {
    res_unpin(locs_a_handle); res_unpin(locs_b_handle); res_unpin(locs_c_handle);
    return ADAMAH_ERR_INVALID;
  }
  VkDescriptorSet ds = alloc_desc_set(&ctx.matmul_t_xq8_pipe);
  if (ds == VK_NULL_HANDLE) return ADAMAH_ERR_VULKAN;
  VkDescriptorBufferInfo buf_infos[6] = {
      {.buffer = m_act->buf, .range = VK_WHOLE_SIZE},
      {.buffer = m_wt->buf,  .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf, .offset = a_off, .range = a_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = b_off, .range = b_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = c_off, .range = c_res->size_bytes},
      {.buffer = m_wt->qparam_buf,  .range = VK_WHOLE_SIZE}};
  VkWriteDescriptorSet writes[6];
  for (int i = 0; i < 6; i++) {
    writes[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds, .dstBinding = i, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 6, writes, 0, NULL);
  uint32_t push[5] = {M, K, N, n_ops, m_wt->group_size};
  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.matmul_t_xq8_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.matmul_t_xq8_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.matmul_t_xq8_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, push);
  vkCmdDispatch(ctx.cmd, (M+15)/16, (N+15)/16, n_ops);
  cmd_barrier_after_dispatch(); cmd_submit();
  res_unpin(locs_a_handle); res_unpin(locs_b_handle); res_unpin(locs_c_handle);
  return ADAMAH_OK;
}

// ============================================
// Cross-map MatVec+TopK F32xQ8: partial top-k directly from A @ B^T
// A from map_act (single row), B from map_wt (Q8), output partial idx/val to map_act.
// ============================================
int map_matvec_topk_t_xq8_ex_dev(uint32_t map_act_id, uint32_t map_wt_id,
                                 uint32_t locs_a_handle, uint32_t locs_b_handle,
                                 uint32_t penalty_ids_handle,
                                 uint32_t locs_idx_dst_handle,
                                 uint32_t locs_val_dst_handle,
                                 uint32_t K, uint32_t N, uint32_t k,
                                 uint32_t n_penalty, float repeat_penalty,
                                 uint32_t rows_per_group) {
  int fret = fusion_flush_pending_for_immediate();
  if (fret != ADAMAH_OK) return fret;
  if (map_act_id >= MAX_MAPS || !ctx.maps[map_act_id].active) return ADAMAH_ERR_INVALID;
  if (map_wt_id  >= MAX_MAPS || !ctx.maps[map_wt_id].active)  return ADAMAH_ERR_INVALID;
  if (N == 0 || k == 0) return ADAMAH_OK;
  if (k > 64 || n_penalty > 64) return ADAMAH_ERR_INVALID;
  if (rows_per_group == 0) return ADAMAH_ERR_INVALID;
  if (!ctx.matvec_topk_t_xq8_pipe.pipeline && init_pipelines() != 0) return ADAMAH_ERR_VULKAN;
  if (!ctx.matvec_topk_t_xq8_pipe.pipeline) return ADAMAH_ERR_INVALID;

  Map *m_act = &ctx.maps[map_act_id];
  Map *m_wt = &ctx.maps[map_wt_id];
  ResEntry *a_res = res_get(locs_a_handle);
  ResEntry *b_res = res_get(locs_b_handle);
  ResEntry *p_res = res_get(penalty_ids_handle);
  ResEntry *i_res = res_get(locs_idx_dst_handle);
  ResEntry *v_res = res_get(locs_val_dst_handle);
  if (!a_res || !b_res || !p_res || !i_res || !v_res) return ADAMAH_ERR_INVALID;
  if (m_wt->qparam_buf == VK_NULL_HANDLE) return ADAMAH_ERR_INVALID;

  res_pin(locs_a_handle);
  res_pin(locs_b_handle);
  res_pin(penalty_ids_handle);
  res_pin(locs_idx_dst_handle);
  res_pin(locs_val_dst_handle);

  VkDeviceSize a_off = 0, b_off = 0, p_off = 0, i_off = 0, v_off = 0;
  if (res_require_hot(locs_a_handle, &a_off) != 0 ||
      res_require_hot(locs_b_handle, &b_off) != 0 ||
      res_require_hot(penalty_ids_handle, &p_off) != 0 ||
      res_require_hot(locs_idx_dst_handle, &i_off) != 0 ||
      res_require_hot(locs_val_dst_handle, &v_off) != 0) {
    res_unpin(locs_a_handle); res_unpin(locs_b_handle); res_unpin(penalty_ids_handle);
    res_unpin(locs_idx_dst_handle); res_unpin(locs_val_dst_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.matvec_topk_t_xq8_pipe);
  if (ds == VK_NULL_HANDLE) {
    res_unpin(locs_a_handle); res_unpin(locs_b_handle); res_unpin(penalty_ids_handle);
    res_unpin(locs_idx_dst_handle); res_unpin(locs_val_dst_handle);
    return ADAMAH_ERR_VULKAN;
  }

  VkDescriptorBufferInfo buf_infos[8] = {
      {.buffer = m_act->buf, .range = VK_WHOLE_SIZE},
      {.buffer = m_wt->buf,  .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf, .offset = a_off, .range = a_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = b_off, .range = b_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = p_off, .range = p_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = i_off, .range = i_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = v_off, .range = v_res->size_bytes},
      {.buffer = m_wt->qparam_buf,  .range = VK_WHOLE_SIZE},
  };
  VkWriteDescriptorSet writes[8];
  for (int i = 0; i < 8; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 8, writes, 0, NULL);

  typedef struct {
    uint32_t K;
    uint32_t N;
    uint32_t k;
    uint32_t group_size;
    uint32_t n_penalty;
    float repeat_penalty;
    uint32_t rows_per_group;
  } MatvecTopKPush;
  MatvecTopKPush push = {
      K, N, k, m_wt->group_size, n_penalty, repeat_penalty, rows_per_group};

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.matvec_topk_t_xq8_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.matvec_topk_t_xq8_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.matvec_topk_t_xq8_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 28, &push);
  vkCmdDispatch(ctx.cmd, (N + rows_per_group - 1) / rows_per_group, 1, 1);
  cmd_barrier_after_dispatch();
  cmd_submit();

  res_unpin(locs_a_handle); res_unpin(locs_b_handle); res_unpin(penalty_ids_handle);
  res_unpin(locs_idx_dst_handle); res_unpin(locs_val_dst_handle);
  return ADAMAH_OK;
}

int map_matvec_topk_t_xq8_dev(uint32_t map_act_id, uint32_t map_wt_id,
                              uint32_t locs_a_handle, uint32_t locs_b_handle,
                              uint32_t penalty_ids_handle,
                              uint32_t locs_idx_dst_handle,
                              uint32_t locs_val_dst_handle,
                              uint32_t K, uint32_t N, uint32_t k,
                              uint32_t n_penalty, float repeat_penalty) {
  return map_matvec_topk_t_xq8_ex_dev(
      map_act_id, map_wt_id,
      locs_a_handle, locs_b_handle, penalty_ids_handle,
      locs_idx_dst_handle, locs_val_dst_handle,
      K, N, k, n_penalty, repeat_penalty, 256);
}

// ============================================
// Cross-map MatVec rerank F32xQ8: exact logits for shortlisted token ids
// A from map_act (single row), B from map_wt (Q8), shortlisted token ids live in map_act.
// ============================================
int map_matvec_rerank_t_xq8_dev(uint32_t map_act_id, uint32_t map_wt_id,
                                uint32_t locs_a_handle, uint32_t locs_b_handle,
                                uint32_t partial_idx_base_handle,
                                uint32_t sel_locs_handle,
                                uint32_t penalty_ids_handle,
                                uint32_t locs_idx_dst_handle,
                                uint32_t locs_val_dst_handle,
                                uint32_t K, uint32_t n_ids,
                                uint32_t n_penalty, float repeat_penalty) {
  int fret = fusion_flush_pending_for_immediate();
  if (fret != ADAMAH_OK) return fret;
  if (map_act_id >= MAX_MAPS || !ctx.maps[map_act_id].active) return ADAMAH_ERR_INVALID;
  if (map_wt_id >= MAX_MAPS || !ctx.maps[map_wt_id].active) return ADAMAH_ERR_INVALID;
  if (n_ids == 0) return ADAMAH_OK;
  if (n_ids > 64 || n_penalty > 64) return ADAMAH_ERR_INVALID;
  if (!ctx.matvec_rerank_t_xq8_pipe.pipeline && init_pipelines() != 0) return ADAMAH_ERR_VULKAN;
  if (!ctx.matvec_rerank_t_xq8_pipe.pipeline) return ADAMAH_ERR_INVALID;

  Map *m_act = &ctx.maps[map_act_id];
  Map *m_wt = &ctx.maps[map_wt_id];
  ResEntry *a_res = res_get(locs_a_handle);
  ResEntry *b_res = res_get(locs_b_handle);
  ResEntry *pb_res = res_get(partial_idx_base_handle);
  ResEntry *s_res = res_get(sel_locs_handle);
  ResEntry *p_res = res_get(penalty_ids_handle);
  ResEntry *i_res = res_get(locs_idx_dst_handle);
  ResEntry *v_res = res_get(locs_val_dst_handle);
  if (!a_res || !b_res || !pb_res || !s_res || !p_res || !i_res || !v_res) return ADAMAH_ERR_INVALID;
  if (m_wt->qparam_buf == VK_NULL_HANDLE) return ADAMAH_ERR_INVALID;

  res_pin(locs_a_handle);
  res_pin(locs_b_handle);
  res_pin(partial_idx_base_handle);
  res_pin(sel_locs_handle);
  res_pin(penalty_ids_handle);
  res_pin(locs_idx_dst_handle);
  res_pin(locs_val_dst_handle);

  VkDeviceSize a_off = 0, b_off = 0, pb_off = 0, s_off = 0, p_off = 0, i_off = 0, v_off = 0;
  if (res_require_hot(locs_a_handle, &a_off) != 0 ||
      res_require_hot(locs_b_handle, &b_off) != 0 ||
      res_require_hot(partial_idx_base_handle, &pb_off) != 0 ||
      res_require_hot(sel_locs_handle, &s_off) != 0 ||
      res_require_hot(penalty_ids_handle, &p_off) != 0 ||
      res_require_hot(locs_idx_dst_handle, &i_off) != 0 ||
      res_require_hot(locs_val_dst_handle, &v_off) != 0) {
    res_unpin(locs_a_handle); res_unpin(locs_b_handle); res_unpin(partial_idx_base_handle);
    res_unpin(sel_locs_handle); res_unpin(penalty_ids_handle);
    res_unpin(locs_idx_dst_handle); res_unpin(locs_val_dst_handle);
    return ADAMAH_ERR_INVALID;
  }

  VkDescriptorSet ds = alloc_desc_set(&ctx.matvec_rerank_t_xq8_pipe);
  if (ds == VK_NULL_HANDLE) {
    res_unpin(locs_a_handle); res_unpin(locs_b_handle); res_unpin(partial_idx_base_handle);
    res_unpin(sel_locs_handle); res_unpin(penalty_ids_handle);
    res_unpin(locs_idx_dst_handle); res_unpin(locs_val_dst_handle);
    return ADAMAH_ERR_VULKAN;
  }

  VkDescriptorBufferInfo buf_infos[10] = {
      {.buffer = m_act->buf, .range = VK_WHOLE_SIZE},
      {.buffer = m_wt->buf, .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf, .offset = a_off, .range = a_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = b_off, .range = b_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = pb_off, .range = pb_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = s_off, .range = s_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = p_off, .range = p_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = i_off, .range = i_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = v_off, .range = v_res->size_bytes},
      {.buffer = m_wt->qparam_buf, .range = VK_WHOLE_SIZE},
  };
  VkWriteDescriptorSet writes[10];
  for (int i = 0; i < 10; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 10, writes, 0, NULL);

  typedef struct {
    uint32_t K;
    uint32_t n_ids;
    uint32_t group_size;
    uint32_t n_penalty;
    float repeat_penalty;
  } MatvecRerankPush;
  MatvecRerankPush push = {K, n_ids, m_wt->group_size, n_penalty, repeat_penalty};

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.matvec_rerank_t_xq8_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.matvec_rerank_t_xq8_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.matvec_rerank_t_xq8_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, &push);
  vkCmdDispatch(ctx.cmd, n_ids, 1, 1);
  cmd_barrier_after_dispatch();
  cmd_submit();

  res_unpin(locs_a_handle); res_unpin(locs_b_handle); res_unpin(partial_idx_base_handle);
  res_unpin(sel_locs_handle); res_unpin(penalty_ids_handle);
  res_unpin(locs_idx_dst_handle); res_unpin(locs_val_dst_handle);
  return ADAMAH_OK;
}

// ============================================
// Cross-map MatMul F32xQ8: C = A @ B (non-transposed)
// A from map_act (F32), B from map_wt (Q8), C to map_act (F32)
// ============================================
int map_matmul_xq8_dev(uint32_t map_act_id, uint32_t map_wt_id,
                       uint32_t locs_a_handle, uint32_t locs_b_handle,
                       uint32_t locs_c_handle, uint32_t M, uint32_t K,
                       uint32_t N, uint32_t n_ops) {
  int fret = fusion_flush_pending_for_immediate();
  if (fret != ADAMAH_OK) return fret;
  if (map_act_id >= MAX_MAPS || !ctx.maps[map_act_id].active) return ADAMAH_ERR_INVALID;
  if (map_wt_id  >= MAX_MAPS || !ctx.maps[map_wt_id].active)  return ADAMAH_ERR_INVALID;
  if (n_ops == 0) return ADAMAH_OK;
  if (!ctx.matmul_xq8_pipe.pipeline && init_pipelines() != 0) return ADAMAH_ERR_VULKAN;
  if (!ctx.matmul_xq8_pipe.pipeline) return ADAMAH_ERR_INVALID;
  Map *m_act = &ctx.maps[map_act_id]; Map *m_wt = &ctx.maps[map_wt_id];
  ResEntry *a_res = res_get(locs_a_handle);
  ResEntry *b_res = res_get(locs_b_handle);
  ResEntry *c_res = res_get(locs_c_handle);
  if (!a_res || !b_res || !c_res) return ADAMAH_ERR_INVALID;
  if (m_wt->qparam_buf == VK_NULL_HANDLE) return ADAMAH_ERR_INVALID;
  res_pin(locs_a_handle); res_pin(locs_b_handle); res_pin(locs_c_handle);
  VkDeviceSize a_off = 0, b_off = 0, c_off = 0;
  if (res_require_hot(locs_a_handle, &a_off) != 0 ||
      res_require_hot(locs_b_handle, &b_off) != 0 ||
      res_require_hot(locs_c_handle, &c_off) != 0) {
    res_unpin(locs_a_handle); res_unpin(locs_b_handle); res_unpin(locs_c_handle);
    return ADAMAH_ERR_INVALID;
  }
  VkDescriptorSet ds = alloc_desc_set(&ctx.matmul_xq8_pipe);
  if (ds == VK_NULL_HANDLE) return ADAMAH_ERR_VULKAN;
  VkDescriptorBufferInfo buf_infos[6] = {
      {.buffer = m_act->buf, .range = VK_WHOLE_SIZE},
      {.buffer = m_wt->buf,  .range = VK_WHOLE_SIZE},
      {.buffer = ctx.hot_pool->buf, .offset = a_off, .range = a_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = b_off, .range = b_res->size_bytes},
      {.buffer = ctx.hot_pool->buf, .offset = c_off, .range = c_res->size_bytes},
      {.buffer = m_wt->qparam_buf,  .range = VK_WHOLE_SIZE}};
  VkWriteDescriptorSet writes[6];
  for (int i = 0; i < 6; i++) {
    writes[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds, .dstBinding = i, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 6, writes, 0, NULL);
  uint32_t push[5] = {M, K, N, n_ops, m_wt->group_size};
  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.matmul_xq8_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.matmul_xq8_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.matmul_xq8_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, push);
  vkCmdDispatch(ctx.cmd, (M+15)/16, (N+15)/16, n_ops);
  cmd_barrier_after_dispatch(); cmd_submit();
  res_unpin(locs_a_handle); res_unpin(locs_b_handle); res_unpin(locs_c_handle);
  return ADAMAH_OK;
}

// ============================================
// Cross-map contiguous row gather F32 <- Q8
// Writes one Q8 row directly into a contiguous F32 slice in map_act.
// ============================================
int map_row_gather_xq8_dev(uint32_t map_act_id, uint32_t map_wt_id,
                           uint32_t src_base, uint32_t dst_base,
                           uint32_t row_idx, uint32_t row_size, float scale) {
  int fret = fusion_flush_pending_for_immediate();
  if (fret != ADAMAH_OK) return fret;
  if (map_act_id >= MAX_MAPS || !ctx.maps[map_act_id].active) return ADAMAH_ERR_INVALID;
  if (map_wt_id  >= MAX_MAPS || !ctx.maps[map_wt_id].active)  return ADAMAH_ERR_INVALID;
  if (row_size == 0) return ADAMAH_OK;
  if (!ctx.row_gather_xq8_pipe.pipeline && init_pipelines() != 0) return ADAMAH_ERR_VULKAN;
  if (!ctx.row_gather_xq8_pipe.pipeline) return ADAMAH_ERR_INVALID;

  Map *m_act = &ctx.maps[map_act_id];
  Map *m_wt = &ctx.maps[map_wt_id];
  if (m_act->dtype != DTYPE_F32 || m_act->pack_size != 1) return ADAMAH_ERR_INVALID;
  if (m_wt->dtype != DTYPE_Q8 || m_wt->qparam_buf == VK_NULL_HANDLE) return ADAMAH_ERR_INVALID;

  VkDescriptorSet ds = alloc_desc_set(&ctx.row_gather_xq8_pipe);
  if (ds == VK_NULL_HANDLE) return ADAMAH_ERR_VULKAN;

  VkDescriptorBufferInfo buf_infos[3] = {
      {.buffer = m_act->buf,       .range = VK_WHOLE_SIZE},
      {.buffer = m_wt->buf,        .range = VK_WHOLE_SIZE},
      {.buffer = m_wt->qparam_buf, .range = VK_WHOLE_SIZE},
  };
  VkWriteDescriptorSet writes[3];
  for (int i = 0; i < 3; i++) {
    writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds, .dstBinding = i, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buf_infos[i]};
  }
  vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);

  uint32_t push[6];
  push[0] = src_base;
  push[1] = dst_base;
  push[2] = row_idx;
  push[3] = row_size;
  push[4] = m_wt->group_size;
  memcpy(&push[5], &scale, 4);

  cmd_begin();
  vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.row_gather_xq8_pipe.pipeline);
  vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ctx.row_gather_xq8_pipe.pipe_layout, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(ctx.cmd, ctx.row_gather_xq8_pipe.pipe_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, 24, push);
  vkCmdDispatch(ctx.cmd, (row_size + 255) / 256, 1, 1);
  cmd_barrier_after_dispatch();
  cmd_submit();
  return ADAMAH_OK;
}
