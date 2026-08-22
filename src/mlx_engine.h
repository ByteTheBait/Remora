#pragma once

// MLX GPU compute engine for Remora.
//
// This is a drop-in alternative to the ggml-backed ComputeEngine. It runs the
// Mamba selective-scan forward pass on Apple Silicon's GPU (Metal) via Apple's
// MLX C++ API (libmlx.dylib), streaming the same per-layer shard files the
// ggml path consumes.
//
// Design notes (verified against the MLX C++ API on an Apple M2):
//   * MLX C++ matmuls on the GPU (Device::gpu) read back correctly as F32.
//   * FP16 GPU readback is unreliable in this MLX build, so MLX weights are
//     decoded host-side F16 -> F32 once, then uploaded to the GPU as F32.
//   * The sequential selective-scan loop (which has no MLX primitive) runs on
//     the host against the F32 hidden/SSM state; the heavy matmuls (in_proj,
//     x_proj, dt_proj, out_proj) run on the GPU.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <mlx/stream.h>   // mlx::core::Stream

#include "rnn_kernel.h"   // RuntimeContext, LayerDesc, ModelConfig

namespace remora {

// Forward-declared pimpl holding persistent GPU arrays (the MLX C++ types
// live in the .cpp only, so non-MLX builds can still include this header).
struct MlxGpuCache;

// ---------------------------------------------------------------------------
// MlxEngine: owns a persistent MLX GPU backend + per-layer F32 weight cache.
//
// The weights are memory-mapped and byte-identical across every token; only
// the mmap base changes. We decode each block's F16 weights to F32 once and
// cache them, then re-point to the current shard base on each prepare (the
// same strategy as the ggml LayerCache).
// ---------------------------------------------------------------------------
class MlxEngine {
public:
    MlxEngine();
    ~MlxEngine();

    MlxEngine(const MlxEngine&) = delete;
    MlxEngine& operator=(const MlxEngine&) = delete;

    // True once the MLX GPU backend is ready.
    bool ok() const;

    // Cap the number of layer-weight blocks resident at once (GPU arrays +
    // host decode). Set to ~buffer_layers+1 to make peak RSS track the
    // stream window instead of the whole model. Evicted blocks re-decode on
    // next access. Call before the first forward.
    void set_max_cached_blocks(std::size_t n) { max_cached_blocks_ = n; }

    // Human-readable backend name (e.g. "MLX GPU (Metal)").
    const char* backend_name() const;

    // Run one block's forward pass over ctx.hidden_state, updating the SSM /
    // conv state in place. Uses the GPU for the matmuls. Returns the number
    // of weight bytes consumed (the shard size).
    std::size_t run_layer_forward(RuntimeContext& ctx, const LayerContext& layer);

    // Batch-parallel forward: process a batch of *independent* token contexts
    // through ONE streamed layer buffer. Every ctx shares the same
    // layer.weights_base / layer.desc (read-only weights, mapped once), so the
    // layer is resident in a single spot; each context is advanced on its own
    // thread. Only safe when the contexts are independent (different
    // sequences / a batch) — the Mamba SSM recurrence is sequential within one
    // stream, so this does NOT parallelize the tokens of a single stream.
    // Returns the number of weight bytes consumed (the shard size).
    std::size_t run_layer_forward_batch(std::vector<RuntimeContext*>& ctxs,
                                        const LayerContext& layer);

    // Embed token `id` into ctx.hidden_state from the global embedding table
    // (`backbone.embeddings.weight`, F16, [vocab, d_model]).
    bool embed(RuntimeContext& ctx, int32_t id,
               const LayerDesc& global_desc, const void* global_base,
               std::size_t global_size);

    // Final RMSNorm + lm_head: compute logits [vocab] for the current
    // hidden_state. Uses the tied `backbone.embeddings.weight` as the lm_head
    // and `backbone.norm_f.weight` as the final norm.
    bool compute_logits(RuntimeContext& ctx, std::vector<float>& logits,
                        const LayerDesc& global_desc, const void* global_base,
                        std::size_t global_size);

private:
    // A decoded weight tensor for one block. `data` holds F32 values for F16
    // tensors, or the raw packed uint32 bytes (reinterpreted as float) for U32
    // quantized tensors.
    struct CachedWeight {
        std::string name;           // full tensor name from the sidecar
        std::string dtype = "F16";  // "F16" or "U32"
        std::vector<float> data;    // F32 values (F16) or packed U32 bytes (U32)
        int64_t n_elements = 0;
    };

    // One block's cached decoded weights.
    struct BlockCache {
        int block_id = -1;
        std::vector<CachedWeight> weights;
    };

    // Locate a cached weight whose name ends with `suffix`.
    // Returns pointer to decoded F32 data + element count, or nullptr.
    const float* find_weight(const BlockCache& blk, const std::string& suffix,
                             int64_t* n_elements_out = nullptr) const;

    // Locate a cached U32 packed weight by suffix; returns raw uint32 pointer.
    const uint32_t* find_u32(const BlockCache& blk, const std::string& suffix) const;

    // Run one token's forward pass against the shared per-block weights.
    // `gpu_blk` is a `MlxGpuCache::BlockGpu*` (defined in the .cpp only, so it
    // travels as void* here). The weights are read-only and shared across all
    // batch threads; only `ctx` is thread-private. `stream` is the caller's
    // own MLX stream. `gpu_lock` serializes the fast GPU matmul evals (Metal
    // is NOT thread-safe for concurrent eval even with separate streams), so
    // only the host-side selective-scan / conv / gating overlap across batch
    // threads — which is the real Mamba bottleneck.
    void compute_token(RuntimeContext& ctx, const LayerDesc& desc,
                       const BlockCache& blk, void* gpu_blk,
                       const mlx::core::Stream& stream, std::mutex& gpu_lock);

    // Decode a block's F16 weights from the shard into the cache.
    bool prepare_block(const LayerDesc& desc, const void* shard_base,
                       std::size_t shard_size);

    // Bounded-cache eviction (streaming): cap the number of blocks whose
    // weights are resident at once (GPU arrays + host decode). Evicts the
    // least-recently-used block so peak memory tracks the stream window
    // (~buffer_layers) instead of the whole model. Evicted blocks are
    // re-decoded on next access.
    void touch_block(int block_id);
    void evict_if_needed();
    std::size_t count_resident() const;

    std::vector<uint64_t> recency_;   // indexed by block_id
    uint64_t tick_ = 0;
    std::size_t max_cached_blocks_ = 48;   // override via set_max_cached_blocks

    // Cache the global.bin embedding table and final norm once, keyed by the
    // mmap base pointer (global.bin is stable across the whole run).
    const float* global_tensor(const LayerDesc& gdesc, const void* base,
                               std::size_t size, const std::string& suffix,
                               std::vector<float>& out, std::size_t* n_out);

    std::vector<BlockCache> blocks_;   // indexed by block_id
    // Persistent GPU arrays for the global (embedding / final norm) weights.
    std::unique_ptr<MlxGpuCache> gpu_cache_;
    // Cached decoded global tensors (embeddings, final norm) + their byte size.
    const void* global_cache_base_ = nullptr;
    std::size_t global_cache_size_ = 0;
    std::vector<float> cached_embeddings_;
    std::vector<float> cached_norm_f_;
    bool ok_ = false;
};

}  // namespace remora
