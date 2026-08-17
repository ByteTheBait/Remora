#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#ifdef REMORA_HAVE_GGML
#include <ggml.h>
#include <ggml-backend.h>
#endif

namespace remora {

// Forward declaration so run_layer_forward can accept a ComputeEngine& even
// when REMORA_HAVE_GGML is disabled (the full definition lives under the
// guard below).
class ComputeEngine;

// ---------------------------------------------------------------------------
// Model configuration (parsed from manifest.json).
// ---------------------------------------------------------------------------
struct ModelConfig {
    std::string architecture = "mamba";
    int n_layer    = 0;
    int d_model    = 0;
    int d_inner    = 0;   // = expand factor * d_model for SSM blocks
    int d_state    = 16;
    int d_conv     = 4;
    int dt_rank    = 128;
    int vocab_size = 0;
    int context_length = 2048;
    bool tied_embeddings = true;
};

// ---------------------------------------------------------------------------
// Per-layer descriptor (parsed from blk.N.meta.json).
//
// One Mamba block contains 10 parameters (attn_norm, ssm_a, ssm_d,
// ssm_conv1d.{bias,weight}, ssm_dt.{bias,weight}, ssm_in.weight,
// ssm_out.weight, ssm_x.weight). The sidecar gives us their (name, shape,
// dtype, offset, size) within the shard file. The C++ side maps each one to
// a ggml_tensor pointing into the memory-mapped shard.
// ---------------------------------------------------------------------------
struct TensorDesc {
    std::string name;
    std::vector<int64_t> shape;
    std::string dtype;        // "F32", "Q4_K", "Q6_K", "Q5_0", ...
    std::size_t offset = 0;   // byte offset within the shard
    std::size_t size = 0;     // byte size within the shard
};

struct LayerDesc {
    int block_id = -1;
    std::vector<TensorDesc> tensors;
};

// ---------------------------------------------------------------------------
// LayerContext: passed into run_layer_forward.
//
// `weights_base` points to the memory-mapped shard; `weights_size` is its
// total size. `desc` is the parsed sidecar that tells us how to carve the
// shard into individual ggml_tensors.
// ---------------------------------------------------------------------------
struct LayerContext {
    const void* weights_base = nullptr;
    std::size_t weights_size = 0;
    const LayerDesc* desc = nullptr;
};

// ---------------------------------------------------------------------------
// RuntimeContext: holds the per-token persistent state.
//
// `hidden_state` is the (d_model,) residual stream — the "current token's
// activation" as it flows through the network.
//
// `ssm_state` holds per-layer SSM hidden state (one (d_state, d_inner)
// tensor per Mamba layer). This is the O(1)-per-layer state that lets us
// stream layer weights without losing recurrence.
//
// `conv_state` holds per-layer causal-conv1d history (one (d_conv-1, d_inner)
// tensor per Mamba block) — required because conv1d has a (d_conv-1)-token
// receptive field and we process one token at a time in the streaming loop.
// ---------------------------------------------------------------------------
struct RuntimeContext {
    ModelConfig cfg;

    // Per-token activations: the residual stream and per-layer SSM/conv state.
    std::vector<float> hidden_state;          // [d_model]
    std::vector<float> ssm_state;             // [n_layer * d_state * d_inner]
    std::vector<float> conv_state;            // [n_layer * (d_conv - 1) * d_inner]

    // Step counter (number of tokens processed so far).
    int64_t step = 0;
};

#ifdef REMORA_HAVE_GGML

// ---------------------------------------------------------------------------
// ComputeEngine: owns a single persistent backend list + scheduler.
//
// This is the heart of the performance fix. The previous kernel created a
// fresh ggml backend and scheduler on *every matmul*, per block, per token
// — hundreds of expensive backend init / buffer alloc / sched teardown
// cycles per forward pass. ComputeEngine initializes the backends once
// (GPU-first: Metal, then CPU, then BLAS) and reuses one `ggml_backend_sched`
// across every block and every token. The scheduler lazily allocates its
// compute buffers on first use and reuses them thereafter, so we avoid
// per-call alloc/free churn and automatically offload matmuls to the GPU
// when the backend supports it.
// ---------------------------------------------------------------------------
class ComputeEngine {
public:
    ComputeEngine();
    ~ComputeEngine();

    ComputeEngine(const ComputeEngine&) = delete;
    ComputeEngine& operator=(const ComputeEngine&) = delete;

    // True once the backends + scheduler are ready.
    bool ok() const { return sched_ != nullptr; }

    // Name of the primary (first) backend, e.g. "Metal" / "CPU" / "BLAS".
    const char* primary_backend_name() const;

    // Run a pre-built forward graph on the persistent scheduler.
    // Allocates the scheduler's compute buffers on first call, reuses them
    // on subsequent calls. Returns false on compute failure.
    bool compute(struct ggml_cgraph* gf);

    // Backend for allocating activation tensors. Falls back to CPU.
    ggml_backend_t backend() const { return backends_.empty() ? nullptr : backends_.front(); }
    // Buffer type for activation tensors (default type of `backend()`).
    ggml_backend_buffer_type_t buft() const { return bufts_.empty() ? nullptr : bufts_.front(); }

private:
    std::vector<ggml_backend_t> backends_;
    std::vector<ggml_backend_buffer_type_t> bufts_;
    ggml_backend_sched_t sched_ = nullptr;
};

#endif  // REMORA_HAVE_GGML

// ---------------------------------------------------------------------------
// GlobalContext: the non-repeating tensors (embeddings, output norm, lm_head).
//
// The sharder writes these into `global.bin` (with `global.bin.meta.json`
// sidecar). For a tied-embedding Mamba model the token embedding table is
// used both as the input embedding AND as the lm_head output matrix.
// ---------------------------------------------------------------------------
struct GlobalContext {
    ModelConfig cfg;

    const void* base = nullptr;         // memory-mapped global.bin base
    std::size_t size = 0;               // total bytes
    const LayerDesc* desc = nullptr;    // parsed sidecar (tensor locs)

    // Locate the offset of a tensor by name within global.bin, or -1.
    int64_t tensor_offset(const std::string& name) const;
    // Locate the dtype of a tensor by name.
    std::string tensor_dtype(const std::string& name) const;
    // Load token `id`'s embedding row [d_model] into ctx.hidden_state.
    bool embed(RuntimeContext& ctx, int32_t id, ComputeEngine& engine) const;
    // Final RMSNorm + lm_head: compute logits [vocab] for the current
    // hidden_state. Returns false on failure.
    bool compute_logits(RuntimeContext& ctx, std::vector<float>& logits,
                        ComputeEngine& engine) const;
};

// Run one block's forward pass over the current hidden_state and update
// ctx in place. Uses `engine` for all ggml compute. Returns the number of
// weight bytes consumed (or, in ggml mode, the size of the shard file).
std::size_t run_layer_forward(RuntimeContext& ctx, const LayerContext& layer,
                              ComputeEngine& engine);

#ifdef REMORA_HAVE_GGML
// ---------------------------------------------------------------------------
// Helpers used by main.cpp to set up ggml contexts from shards.
// ---------------------------------------------------------------------------

// Map a dtype string from the sidecar JSON to a ggml_type enum.
// Returns GGML_TYPE_COUNT on unknown / unsupported dtypes.
ggml_type remora_dtype_from_string(const std::string& s);

// Compute the byte size of a tensor with the given ggml_type and shape.
// Returns 0 on error.
std::size_t remora_tensor_nbytes(ggml_type type,
                                 const std::vector<int64_t>& shape);

// Build a ggml_context that owns a copy of one block's tensors, given a
// memory-mapped shard and the parsed sidecar. Returns nullptr on failure.
// The caller is responsible for ggml_free()-ing the returned context.
struct ggml_context* remora_build_block_ctx(
    const void* shard_base,
    std::size_t shard_size,
    const LayerDesc& desc);

#endif  // REMORA_HAVE_GGML

}  // namespace remora