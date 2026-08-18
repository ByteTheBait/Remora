#include "rnn_kernel.h"

#include <cmath>
#include <cstdio>
#include <cstring>

#ifdef REMORA_HAVE_GGML
#include <ggml.h>
#include <ggml-backend.h>
#endif

namespace remora {

// clangd's "unused include" diagnostic can't see through the
// REMORA_HAVE_GGML guard around the std::sqrt / std::exp calls below,
// so we keep one unconditional reference to <cmath> in scope to silence
// the false positive.
inline double remora_clangd_cmath_keep(double x) {
    return std::sqrt(x);
}

// ---------------------------------------------------------------------------
// Placeholder fallback (when ggml is unavailable)
// ---------------------------------------------------------------------------
#ifndef REMORA_HAVE_GGML

std::size_t run_layer_forward(RuntimeContext& ctx, const LayerContext& layer,
                              ComputeEngine& /*engine*/) {
    // Pretend we ran a real forward pass: the residual stream is left
    // untouched but the layer is "consumed". This is for build smoke-tests
    // and for benchmarking the I/O scaffolding without real math.
    return layer.weights_size;
}

#else  // REMORA_HAVE_GGML

// ---------------------------------------------------------------------------
// ggml dtype <-> string mapping
// ---------------------------------------------------------------------------

ggml_type remora_dtype_from_string(const std::string& s) {
    if (s == "F32")  return GGML_TYPE_F32;
    if (s == "F16")  return GGML_TYPE_F16;
    if (s == "BF16") return GGML_TYPE_BF16;
    if (s == "Q4_0") return GGML_TYPE_Q4_0;
    if (s == "Q4_1") return GGML_TYPE_Q4_1;
    if (s == "Q5_0") return GGML_TYPE_Q5_0;
    if (s == "Q5_1") return GGML_TYPE_Q5_1;
    if (s == "Q8_0") return GGML_TYPE_Q8_0;
    if (s == "Q8_1") return GGML_TYPE_Q8_1;
    if (s == "Q2_K") return GGML_TYPE_Q2_K;
    if (s == "Q3_K") return GGML_TYPE_Q3_K;
    if (s == "Q4_K") return GGML_TYPE_Q4_K;
    if (s == "Q5_K") return GGML_TYPE_Q5_K;
    if (s == "Q6_K") return GGML_TYPE_Q6_K;
    if (s == "Q8_K") return GGML_TYPE_Q8_K;
    return GGML_TYPE_COUNT;
}

std::size_t remora_tensor_nbytes(ggml_type type,
                                 const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    int64_t blck = ggml_blck_size(type);
    if (blck == 0) return 0;
    int64_t padded = ((n + blck - 1) / blck) * blck;
    return static_cast<std::size_t>(padded / blck *
                                    ggml_type_size(static_cast<enum ggml_type>(type)));
}

// ---------------------------------------------------------------------------
// ComputeEngine: persistent backends + scheduler (GPU-first, CPU fallback).
// ---------------------------------------------------------------------------

// Helper to strip a leading "blk.N." prefix from a sidecar tensor name.
static std::string strip_blk_prefix(const std::string& name) {
    if (name.rfind("blk.", 0) != 0) return name;
    const auto dot = name.find('.', 4);
    if (dot != std::string::npos) return name.substr(dot + 1);
    return name;
}

ComputeEngine::ComputeEngine() {
    // This runtime's Mamba forward reads and writes activation tensors
    // directly through their host `->data` pointers: the hand-rolled
    // conv1d / selective-scan / silu loops and the memcpy's that stage
    // activations between ggml graphs. That direct access is only valid
    // if every tensor in the compute graph lives in host-visible memory.
    //
    // The Metal backend on the ggml releases we target only exposes its
    // tensor API on M5/A19+ silicon (it reports "has tensor = false" on
    // an Apple M2 and fails to compile its shaders when forced), and a
    // BLAS/ACCEL primary backend leaves matmul *result* tensors in device
    // memory where `->data` is not meaningful. A GPU path would require
    // explicit staging with ggml_backend_tensor_get/set, which is a
    // separate refactor.
    //
    // For a correct baseline we therefore run on the CPU backend only.
    // The scheduler also asserts that its *last* backend is CPU, so a
    // single CPU backend trivially satisfies that invariant.
    ggml_backend_t cpu =
        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu) {
        // Note: we deliberately do NOT raise the CPU thread count here.
        // The matmuls are tiny (single-token, [4096,2048]@[2048]), so
        // thread-sync overhead dominates; the ggml default of 4 threads
        // measured ~4 tok/s while forcing 8 threads collapsed to ~0.4
        // tok/s. Leave the backend at its default.
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(cpu);
        if (buft) {
            backends_.push_back(cpu);
            bufts_.push_back(buft);
        } else {
            ggml_backend_free(cpu);
        }
    }

    if (backends_.empty()) {
        std::fprintf(stderr, "[remora] ComputeEngine: no ggml backends available\n");
        return;
    }

    // One scheduler for the whole run. Graph size is generous so the
    // scheduler can hold the full Mamba forward graph without realloc.
    sched_ = ggml_backend_sched_new(backends_.data(), bufts_.data(),
                                    (int) backends_.size(),
                                    16384, /*parallel=*/false, /*op_offload=*/true);
    if (!sched_) {
        std::fprintf(stderr, "[remora] ComputeEngine: failed to create scheduler\n");
    }
}

ComputeEngine::~ComputeEngine() {
    if (sched_) ggml_backend_sched_free(sched_);
    for (auto b : backends_) ggml_backend_free(b);
}

const char* ComputeEngine::primary_backend_name() const {
    if (backends_.empty()) return "none";
    return ggml_backend_name(backends_.front());
}

bool ComputeEngine::compute(struct ggml_cgraph* gf) {
    if (!sched_ || !gf) return false;
    // The scheduler asserts !is_alloc before allocating a new graph, so we
    // must reset it between graphs. ggml_backend_sched_reset clears the
    // previous allocation/assignment state and lets the next graph be
    // allocated and computed.
    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) return false;
    if (ggml_backend_sched_graph_compute(sched_, gf) != GGML_STATUS_SUCCESS) return false;
    return true;
}

// ---------------------------------------------------------------------------
// LayerCache: build each block's ggml contexts once and reuse them.
//
// The weights are memory-mapped and byte-identical across every token; only
// the mmap base pointer changes. Rebuilding the ggml_context (10 tensors, a
// fresh CPU backend, and ~376 KB of F32 weight copies) on every layer, every
// token is the dominant cost of the forward pass. LayerCache builds each
// block's contexts once and reuses them, re-pointing only the aliased
// quantized weight data pointers to the current shard base.
// ---------------------------------------------------------------------------

LayerCache::~LayerCache() {
    for (auto& e : entries_) {
        if (e.act_ctx) ggml_free(e.act_ctx);
        if (e.blk_ctx) ggml_free(e.blk_ctx);
    }
}

bool LayerCache::prepare(const LayerDesc& desc, const void* shard_base,
                         std::size_t shard_size, ComputeEngine& engine) {
    // Fast path: this block was already built. Just re-point the aliased
    // quantized weights to the current shard base.
    if (desc.block_id >= 0 && desc.block_id < (int)entries_.size()) {
        Entry& e = entries_[desc.block_id];
        if (e.blk_ctx && e.act_ctx) {
            for (auto& q : e.quantized) {
                const void* p = static_cast<const uint8_t*>(shard_base) + q.offset;
                q.tensor->data = const_cast<void*>(p);
            }
            cur_ = &e;
            return true;
        }
    }

    // Cold path: build this block's contexts and store them.
    if (desc.block_id < 0) return false;
    if (desc.block_id >= (int)entries_.size()) {
        entries_.resize(desc.block_id + 1);
    }
    Entry& e = entries_[desc.block_id];
    e.block_id = desc.block_id;

    // --- Weight context: aliases quantized weights, copies F32 weights. ---
    constexpr std::size_t kMetaBytes = 64 * 1024;
    struct ggml_init_params p = { .mem_size = kMetaBytes,
                                   .mem_buffer = nullptr,
                                   .no_alloc = true };
    e.blk_ctx = ggml_init(p);
    if (!e.blk_ctx) return false;

    // First pass: create every tensor with the right shape + dtype.
    for (const TensorDesc& td : desc.tensors) {
        if (td.shape.empty()) continue;
        ggml_type t = remora_dtype_from_string(td.dtype);
        if (t == GGML_TYPE_COUNT) {
            std::fprintf(stderr,
                "[remora] unknown dtype '%s' for tensor '%s'\n",
                td.dtype.c_str(), td.name.c_str());
            continue;
        }
        int n_dims = static_cast<int>(td.shape.size());
        std::vector<int64_t> ne(td.shape.begin(), td.shape.end());
        ggml_tensor* t_tensor = ggml_new_tensor(e.blk_ctx, t, n_dims, ne.data());
        if (!t_tensor) {
            std::fprintf(stderr,
                "[remora] ggml_new_tensor failed for '%s'\n", td.name.c_str());
            continue;
        }
        std::string short_name = strip_blk_prefix(td.name);
        ggml_set_name(t_tensor, short_name.c_str());
    }

    // Allocate the F32 weight copies from a CPU buffer type.
    ggml_backend_t cpu =
        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        std::fprintf(stderr, "[remora] failed to init CPU backend\n");
        ggml_free(e.blk_ctx); e.blk_ctx = nullptr;
        return false;
    }
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_get_default_buffer_type(cpu);
    ggml_backend_alloc_ctx_tensors_from_buft(e.blk_ctx, cpu_buft);

    for (const TensorDesc& td : desc.tensors) {
        if (td.shape.empty()) continue;
        std::string short_name = strip_blk_prefix(td.name);
        ggml_tensor* t_tensor = ggml_get_tensor(e.blk_ctx, short_name.c_str());
        if (!t_tensor) continue;

        const std::size_t want = ggml_nbytes(t_tensor);
        if (td.offset + want > shard_size) {
            std::fprintf(stderr,
                "[remora] tensor '%s' exceeds shard (%zu + %zu > %zu)\n",
                td.name.c_str(), td.offset, want, shard_size);
            continue;
        }
        const void* src = static_cast<const uint8_t*>(shard_base) + td.offset;

        if (ggml_is_quantized(t_tensor->type)) {
            // Quantized: alias (don't copy) — the dequant path is read-only.
            // Record the offset so we can re-point it on later prepares.
            t_tensor->data = const_cast<void*>(src);
            e.quantized.push_back({ t_tensor, td.offset });
        } else {
            // Plain (F32): copy so the tensor is self-contained.
            std::memcpy(t_tensor->data, src, want);
        }
    }
    ggml_backend_free(cpu);

    // --- Activation context: 14 tensors, allocated from the engine. ---
    constexpr std::size_t kActBytes = 8 * 1024 * 1024;
    struct ggml_init_params ap = { .mem_size = kActBytes,
                                   .mem_buffer = nullptr,
                                   .no_alloc = true };
    e.act_ctx = ggml_init(ap);
    if (!e.act_ctx) {
        ggml_free(e.blk_ctx); e.blk_ctx = nullptr;
        return false;
    }

    // We need the config dims; they're not in the LayerDesc, so derive from
    // the weight tensor shapes.
    ggml_tensor* w_in = ggml_get_tensor(e.blk_ctx, "ssm_in.weight");
    ggml_tensor* w_out = ggml_get_tensor(e.blk_ctx, "ssm_out.weight");
    ggml_tensor* w_x = ggml_get_tensor(e.blk_ctx, "ssm_x.weight");
    ggml_tensor* w_dt = ggml_get_tensor(e.blk_ctx, "ssm_dt.weight");
    ggml_tensor* w_conv = ggml_get_tensor(e.blk_ctx, "ssm_conv1d.weight");
    if (!w_in || !w_out || !w_x || !w_dt || !w_conv) {
        std::fprintf(stderr, "[remora] cache: missing weight tensors in block %d\n",
                     desc.block_id);
        ggml_free(e.act_ctx); e.act_ctx = nullptr;
        ggml_free(e.blk_ctx); e.blk_ctx = nullptr;
        return false;
    }
    // ggml stores ne[0] as the fastest-varying dim. For these weight tensors:
    //   ssm_in.weight  [d_model, 2*d_inner]  -> ne[0]=d_model, ne[1]=2*d_inner
    //   ssm_out.weight [d_inner, d_model]    -> ne[0]=d_inner, ne[1]=d_model
    //   ssm_dt.weight  [dt_rank, d_inner]    -> ne[0]=dt_rank, ne[1]=d_inner
    //   ssm_x.weight   [d_inner, dt_rank+2*d_state] -> ne[0]=d_inner, ne[1]=...
    const int64_t d_inner = w_in->ne[1] / 2;      // 2*d_inner
    const int64_t d_model = w_out->ne[1];
    const int64_t dt_rank = w_dt->ne[0];
    const int64_t d_state = (w_x->ne[1] - dt_rank) / 2;  // dt_rank + 2*d_state

    ggml_set_name(ggml_new_tensor_1d(e.act_ctx, GGML_TYPE_F32, d_model), "y_norm");
    ggml_set_name(ggml_new_tensor_1d(e.act_ctx, GGML_TYPE_F32, 2 * d_inner), "xz");
    ggml_set_name(ggml_new_tensor_1d(e.act_ctx, GGML_TYPE_F32, d_inner), "x_pre");
    ggml_set_name(ggml_new_tensor_1d(e.act_ctx, GGML_TYPE_F32, d_inner), "z_pre");
    ggml_set_name(ggml_new_tensor_1d(e.act_ctx, GGML_TYPE_F32, d_inner), "x_conv");
    ggml_set_name(ggml_new_tensor_1d(e.act_ctx, GGML_TYPE_F32, d_inner), "x_act");
    ggml_set_name(ggml_new_tensor_1d(e.act_ctx, GGML_TYPE_F32, dt_rank + 2*d_state), "x_dtbc");
    ggml_set_name(ggml_new_tensor_1d(e.act_ctx, GGML_TYPE_F32, dt_rank), "dt_pre");
    ggml_set_name(ggml_new_tensor_1d(e.act_ctx, GGML_TYPE_F32, d_state), "B_pre");
    ggml_set_name(ggml_new_tensor_1d(e.act_ctx, GGML_TYPE_F32, d_state), "C_pre");
    ggml_set_name(ggml_new_tensor_1d(e.act_ctx, GGML_TYPE_F32, d_inner), "dt_soft");
    ggml_set_name(ggml_new_tensor_1d(e.act_ctx, GGML_TYPE_F32, d_inner), "y_ssm");
    ggml_set_name(ggml_new_tensor_1d(e.act_ctx, GGML_TYPE_F32, d_inner), "y_gated");
    ggml_set_name(ggml_new_tensor_1d(e.act_ctx, GGML_TYPE_F32, d_model), "y_out");

    if (!ggml_backend_alloc_ctx_tensors_from_buft(e.act_ctx, engine.buft())) {
        std::fprintf(stderr,
            "[remora] cache: failed to allocate activation tensors (block %d)\n",
            desc.block_id);
        ggml_free(e.act_ctx); e.act_ctx = nullptr;
        ggml_free(e.blk_ctx); e.blk_ctx = nullptr;
        return false;
    }

    cur_ = &e;
    return true;
}

// ---------------------------------------------------------------------------
// Mamba block forward pass (one token at a time).
//
// The math is the canonical Mamba-1 selective scan (Gu & Dao 2023).
// We use ggml for the matmuls / elementwise ops and a hand-rolled CPU
// loop for the sequential scan (since ggml has no built-in selective-scan
// op and the scan is inherently sequential over time).
//
// Per-call cost: ~6 matmuls + 1 conv1d + 1 scan + a handful of silus.
// ---------------------------------------------------------------------------

namespace {

// Apply a 1D causal convolution with kernel size d_conv. The conv has
// weights of shape (d_conv, d_inner) and bias of shape (d_inner,). We
// maintain a (d_conv-1, d_inner) history buffer passed in via `conv_hist`.
// `out` is a pre-allocated [d_inner] buffer that the result is written into.
void mamba_conv1d(
        struct ggml_tensor* out,          // [d_inner] pre-allocated output
        struct ggml_tensor* x,            // [d_inner], current token input
        struct ggml_tensor* weight,       // [d_conv, d_inner]
        struct ggml_tensor* bias,         // [d_inner]
        float* conv_hist)                 // [(d_conv-1) * d_inner] scratch
{
    const int64_t d_inner = x->ne[0];
    const int64_t d_conv  = weight->ne[0];   // shape [d_conv, d_inner]
    const float* W = static_cast<const float*>(weight->data);
    const float* B = static_cast<const float*>(bias->data);
    const float* X = static_cast<const float*>(x->data);

    float* Y = static_cast<float*>(out->data);
    for (int64_t i = 0; i < d_inner; ++i) {
        float y = B[i];
        for (int64_t k = 0; k < d_conv - 1; ++k) {
            y += W[k * d_inner + i] * conv_hist[k * d_inner + i];
        }
        y += W[(d_conv - 1) * d_inner + i] * X[i];
        Y[i] = y;
    }
}

// Update the conv history: shift left by one (drop oldest), append x.
void mamba_conv_shift(float* conv_hist, const float* x, int64_t d_inner, int64_t d_conv) {
    for (int64_t k = 0; k < d_conv - 2; ++k) {
        std::memcpy(conv_hist + k * d_inner,
                    conv_hist + (k + 1) * d_inner,
                    d_inner * sizeof(float));
    }
    std::memcpy(conv_hist + (d_conv - 2) * d_inner, x, d_inner * sizeof(float));
}

// One block forward.
std::size_t run_mamba_block(
        RuntimeContext& ctx,
        const LayerDesc& desc,
        const void* shard_base,
        std::size_t shard_size,
        ComputeEngine& engine,
        LayerCache& cache)
{
    // Build (or fetch) the cached ggml contexts for this block. The weights
    // are byte-identical across tokens; only the mmap base changes, so we
    // re-point the aliased quantized weights and reuse everything else.
    if (!cache.prepare(desc, shard_base, shard_size, engine)) return 0;

    ggml_context* blk_ctx = cache.block_ctx();
    ggml_context* act     = cache.act_ctx();

    // Fresh scratch context for the transient graph tensors (matmul results,
    // elementwise intermediates, and the cgraph). The persistent activation
    // tensors live in the cached `act` context; the transient tensors must
    // NOT accumulate there across calls or the 8 MB buffer overflows.
    constexpr std::size_t kScratchBytes = 4 * 1024 * 1024;
    struct ggml_init_params sp = { .mem_size = kScratchBytes,
                                   .mem_buffer = nullptr,
                                   .no_alloc = true };
    ggml_context* scratch = ggml_init(sp);
    if (!scratch) return 0;

    // Locate the parameter tensors by name.
    ggml_tensor* w_norm    = ggml_get_tensor(blk_ctx, "attn_norm.weight");
    ggml_tensor* w_in_proj = ggml_get_tensor(blk_ctx, "ssm_in.weight");
    ggml_tensor* w_conv1d  = ggml_get_tensor(blk_ctx, "ssm_conv1d.weight");
    ggml_tensor* b_conv1d  = ggml_get_tensor(blk_ctx, "ssm_conv1d.bias");
    ggml_tensor* w_dt_proj = ggml_get_tensor(blk_ctx, "ssm_dt.weight");
    ggml_tensor* b_dt_proj = ggml_get_tensor(blk_ctx, "ssm_dt.bias");
    ggml_tensor* w_ssm_x   = ggml_get_tensor(blk_ctx, "ssm_x.weight");
    ggml_tensor* ssm_a     = ggml_get_tensor(blk_ctx, "ssm_a");
    ggml_tensor* ssm_d     = ggml_get_tensor(blk_ctx, "ssm_d");
    ggml_tensor* w_out_proj= ggml_get_tensor(blk_ctx, "ssm_out.weight");
    if (!w_norm || !w_in_proj || !w_conv1d || !b_conv1d || !w_dt_proj ||
        !b_dt_proj || !w_ssm_x || !ssm_a || !ssm_d || !w_out_proj) {
        std::fprintf(stderr,
            "[remora] missing required tensors in block %d\n", desc.block_id);
        return 0;
    }

    const int64_t d_model = ctx.cfg.d_model;
    const int64_t d_inner = ctx.cfg.d_inner;
    const int64_t d_state = ctx.cfg.d_state;
    const int64_t d_conv  = ctx.cfg.d_conv;
    const int64_t dt_rank = ctx.cfg.dt_rank;

    // Activation tensors, fetched from the cached (pre-allocated) context.
    ggml_tensor* y_norm = ggml_get_tensor(act, "y_norm");
    ggml_tensor* xz     = ggml_get_tensor(act, "xz");
    ggml_tensor* x_pre  = ggml_get_tensor(act, "x_pre");
    ggml_tensor* z_pre  = ggml_get_tensor(act, "z_pre");
    ggml_tensor* x_conv = ggml_get_tensor(act, "x_conv");
    ggml_tensor* x_act  = ggml_get_tensor(act, "x_act");
    ggml_tensor* x_dtbc = ggml_get_tensor(act, "x_dtbc");
    ggml_tensor* dt_pre = ggml_get_tensor(act, "dt_pre");
    ggml_tensor* B_pre  = ggml_get_tensor(act, "B_pre");
    ggml_tensor* C_pre  = ggml_get_tensor(act, "C_pre");
    ggml_tensor* dt_soft= ggml_get_tensor(act, "dt_soft");
    ggml_tensor* y_ssm  = ggml_get_tensor(act, "y_ssm");
    ggml_tensor* y_gated= ggml_get_tensor(act, "y_gated");
    ggml_tensor* y_out  = ggml_get_tensor(act, "y_out");

    // Copy the residual stream into y_norm. y_norm has a valid host
    // `->data` (allocated from the engine's CPU buft in the cache).
    std::memcpy(y_norm->data, ctx.hidden_state.data(), d_model * sizeof(float));

    // 1. RMSNorm.
    {
        const float eps = 1e-5f;
        const float* x = static_cast<const float*>(y_norm->data);
        const float* w = static_cast<const float*>(w_norm->data);
        float* y = static_cast<float*>(y_norm->data);
        float mean_sq = 0.0f;
        for (int64_t i = 0; i < d_model; ++i) mean_sq += x[i] * x[i];
        mean_sq /= d_model;
        const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
        for (int64_t i = 0; i < d_model; ++i) y[i] = (x[i] * inv_rms) * w[i];
    }

    // 2. in_proj: xz = ssm_in @ y_norm -> [2*d_inner].
    {
        ggml_tensor* in_xz = ggml_mul_mat(scratch, w_in_proj, y_norm);
        ggml_cgraph* gf = ggml_new_graph(scratch);
        ggml_build_forward_expand(gf, in_xz);
        if (!engine.compute(gf)) { ggml_free(scratch); return 0; }
        std::memcpy(xz->data, in_xz->data, 2 * d_inner * sizeof(float));
    }

    // Split xz into x and z.
    {
        const float* X = static_cast<const float*>(xz->data);
        float* x_out = static_cast<float*>(x_pre->data);
        float* z_out = static_cast<float*>(z_pre->data);
        std::memcpy(x_out, X, d_inner * sizeof(float));
        std::memcpy(z_out, X + d_inner, d_inner * sizeof(float));
    }

    // 3. causal conv1d on x.
    {
        float* hist = ctx.conv_state.data() +
                      desc.block_id * (d_conv - 1) * d_inner;
        mamba_conv1d(x_conv, x_pre, w_conv1d, b_conv1d, hist);
        mamba_conv_shift(hist,
                         static_cast<const float*>(x_pre->data),
                         d_inner, d_conv);
    }

    // 4. silu(x_conv) -> x_act.
    {
        const float* x = static_cast<const float*>(x_conv->data);
        float* y = static_cast<float*>(x_act->data);
        for (int64_t i = 0; i < d_inner; ++i) {
            y[i] = x[i] / (1.0f + std::exp(-x[i]));
        }
    }

    // 5. ssm_x @ x_act -> x_db [dt_rank + 2*d_state]; split into dt/B/C.
    {
        ggml_tensor* xbc = ggml_mul_mat(scratch, w_ssm_x, x_act);
        ggml_cgraph* gf = ggml_new_graph(scratch);
        ggml_build_forward_expand(gf, xbc);
        if (!engine.compute(gf)) { ggml_free(scratch); return 0; }
        std::memcpy(x_dtbc->data, xbc->data,
                    (dt_rank + 2*d_state) * sizeof(float));
    }
    {
        const float* X = static_cast<const float*>(x_dtbc->data);
        float* dt_out = static_cast<float*>(dt_pre->data);
        float* b_out = static_cast<float*>(B_pre->data);
        float* c_out = static_cast<float*>(C_pre->data);
        std::memcpy(dt_out, X, dt_rank * sizeof(float));
        std::memcpy(b_out, X + dt_rank, d_state * sizeof(float));
        std::memcpy(c_out, X + dt_rank + d_state, d_state * sizeof(float));
    }

    // 6. dt_proj: dt = ssm_dt @ dt_pre + ssm_dt_b; softplus -> [d_inner].
    {
        ggml_tensor* dt_lin = ggml_mul_mat(scratch, w_dt_proj, dt_pre);
        ggml_tensor* dt_bias = ggml_add(scratch, dt_lin, b_dt_proj);
        ggml_tensor* dt_sp = ggml_softplus(scratch, dt_bias);
        ggml_cgraph* gf = ggml_new_graph(scratch);
        ggml_build_forward_expand(gf, dt_sp);
        if (!engine.compute(gf)) { ggml_free(scratch); return 0; }
        std::memcpy(dt_soft->data, dt_sp->data, d_inner * sizeof(float));
    }

    // 7. Selective scan (one step).
    {
        const float* a_in  = static_cast<const float*>(ssm_a->data);   // [d_state, d_inner]
        const float* d_in  = static_cast<const float*>(ssm_d->data);   // [d_inner]
        const float* dt    = static_cast<const float*>(dt_soft->data); // [d_inner]
        const float* x_in  = static_cast<const float*>(x_act->data);  // [d_inner]
        const float* b_in  = static_cast<const float*>(B_pre->data);  // [d_state]
        const float* c_in  = static_cast<const float*>(C_pre->data);  // [d_state]
        float*       h     = ctx.ssm_state.data() +
                              desc.block_id * d_state * d_inner;
        float*       y     = static_cast<float*>(y_ssm->data);

        for (int64_t i = 0; i < d_inner; ++i) {
            float yi = d_in[i] * x_in[i];
            for (int64_t s = 0; s < d_state; ++s) {
                const float a_neg = -a_in[s * d_inner + i];
                const float A_bar = std::exp(dt[i] * a_neg);
                const float B_bar = dt[i] * b_in[s];
                const float h_new = A_bar * h[s * d_inner + i] +
                                    B_bar * x_in[i];
                h[s * d_inner + i] = h_new;
                yi += c_in[s] * h_new;
            }
            y[i] = yi;
        }
    }

    // 8. silu(z) * y_ssm -> y_gated.
    {
        const float* z = static_cast<const float*>(z_pre->data);
        const float* ys = static_cast<const float*>(y_ssm->data);
        float* y = static_cast<float*>(y_gated->data);
        for (int64_t i = 0; i < d_inner; ++i) {
            const float sz = z[i] / (1.0f + std::exp(-z[i]));
            y[i] = sz * ys[i];
        }
    }

    // 9. out_proj: y_out = ssm_out @ y_gated -> [d_model].
    {
        ggml_tensor* yo = ggml_mul_mat(scratch, w_out_proj, y_gated);
        ggml_cgraph* gf = ggml_new_graph(scratch);
        ggml_build_forward_expand(gf, yo);
        if (!engine.compute(gf)) { ggml_free(scratch); return 0; }
        std::memcpy(y_out->data, yo->data, d_model * sizeof(float));
    }

    // 10. Residual: hidden_state += y_out.
    {
        const float* yo = static_cast<const float*>(y_out->data);
        float* h = ctx.hidden_state.data();
        for (int64_t i = 0; i < d_model; ++i) h[i] += yo[i];
    }

    ggml_free(scratch);
    return shard_size;
}

}  // namespace

// ---------------------------------------------------------------------------
// GlobalContext: embeddings, output norm, lm_head.
// ---------------------------------------------------------------------------

int64_t GlobalContext::tensor_offset(const std::string& name) const {
    if (!desc) return -1;
    for (const auto& td : desc->tensors) {
        if (td.name == name) return static_cast<int64_t>(td.offset);
    }
    return -1;
}

std::string GlobalContext::tensor_dtype(const std::string& name) const {
    if (!desc) return {};
    for (const auto& td : desc->tensors) {
        if (td.name == name) return td.dtype;
    }
    return {};
}

bool GlobalContext::embed(RuntimeContext& ctx, int32_t id,
                          ComputeEngine& engine) const {
    const int64_t off = tensor_offset("token_embd.weight");
    if (off < 0) {
        std::fprintf(stderr, "[remora] global.bin has no token_embd.weight\n");
        return false;
    }
    ggml_type t = remora_dtype_from_string(tensor_dtype("token_embd.weight"));
    if (t == GGML_TYPE_COUNT) return false;

    const int64_t d_model = cfg.d_model;
    const int64_t n_vocab = cfg.vocab_size;

    // Weight context: aliases token_embd.weight into global.bin (read-only
    // dequant path for quantized embeddings — no copy).
    struct ggml_init_params wp = {
        .mem_size = 64 * 1024, .mem_buffer = nullptr, .no_alloc = true };
    struct ggml_context* wctx = ggml_init(wp);
    if (!wctx) return false;
    ggml_tensor* w = ggml_new_tensor_2d(wctx, t, d_model, n_vocab);
    ggml_set_name(w, "token_embd.weight");
    w->data = const_cast<void*>(
        static_cast<const void*>(static_cast<const char*>(base) + off));

    // Activation context for the tiny extraction graph.
    struct ggml_init_params ap = {
        .mem_size = 1 << 20, .mem_buffer = nullptr, .no_alloc = true };
    struct ggml_context* act = ggml_init(ap);
    if (!act) { ggml_free(wctx); return false; }

    // ids is a leaf tensor we fill directly, so it must have a host pointer.
    ggml_tensor* ids = ggml_new_tensor_1d(act, GGML_TYPE_I32, 1);
    if (!ggml_backend_alloc_ctx_tensors_from_buft(act, engine.buft())) {
        std::fprintf(stderr, "[remora] embed: failed to allocate ids tensor\n");
        ggml_free(act); ggml_free(wctx);
        return false;
    }
    static_cast<int32_t*>(ids->data)[0] = id;

    ggml_tensor* emb = ggml_get_rows(act, w, ids);  // [d_model]
    ggml_cgraph* gf = ggml_new_graph(act);
    ggml_build_forward_expand(gf, emb);

    if (!engine.compute(gf)) { ggml_free(act); ggml_free(wctx); return false; }
    ggml_backend_tensor_get(emb, ctx.hidden_state.data(), 0, d_model * sizeof(float));

    ggml_free(act);
    ggml_free(wctx);
    return true;
}

bool GlobalContext::compute_logits(RuntimeContext& ctx,
                                   std::vector<float>& logits,
                                   ComputeEngine& engine) const {
    const int64_t off_head = tensor_offset("token_embd.weight");
    const int64_t off_norm = tensor_offset("output_norm.weight");
    if (off_head < 0 || off_norm < 0) {
        std::fprintf(stderr,
            "[remora] global.bin missing token_embd.weight or output_norm.weight\n");
        return false;
    }
    ggml_type th = remora_dtype_from_string(tensor_dtype("token_embd.weight"));
    if (th == GGML_TYPE_COUNT) return false;

    const int64_t d_model = cfg.d_model;
    const int64_t n_vocab = cfg.vocab_size;

    // 1. Final RMSNorm with output_norm.weight.
    const float eps = 1e-5f;
    const float* wn = static_cast<const float*>(
        static_cast<const void*>(static_cast<const char*>(base) + off_norm));
    const float* x = ctx.hidden_state.data();
    std::vector<float> normed(d_model);
    float mean_sq = 0.0f;
    for (int64_t i = 0; i < d_model; ++i) mean_sq += x[i] * x[i];
    mean_sq /= d_model;
    const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
    for (int64_t i = 0; i < d_model; ++i) normed[i] = x[i] * inv_rms * wn[i];

    // 2. lm_head = tied token_embd.weight, aliased into global.bin.
    struct ggml_init_params wp = {
        .mem_size = 64 * 1024, .mem_buffer = nullptr, .no_alloc = true };
    struct ggml_context* wctx = ggml_init(wp);
    if (!wctx) return false;
    ggml_tensor* w = ggml_new_tensor_2d(wctx, th, d_model, n_vocab);
    ggml_set_name(w, "token_embd.weight");
    w->data = const_cast<void*>(
        static_cast<const void*>(static_cast<const char*>(base) + off_head));

    // 3. logits = lm_head @ normed  ->  [n_vocab]
    struct ggml_init_params ap = {
        .mem_size = 4 << 20, .mem_buffer = nullptr, .no_alloc = true };
    struct ggml_context* act = ggml_init(ap);
    if (!act) { ggml_free(wctx); return false; }

    // hv is a leaf tensor we fill directly, so it must have a host pointer.
    ggml_tensor* hv = ggml_new_tensor_1d(act, GGML_TYPE_F32, d_model);
    if (!ggml_backend_alloc_ctx_tensors_from_buft(act, engine.buft())) {
        std::fprintf(stderr, "[remora] compute_logits: failed to allocate hv\n");
        ggml_free(act); ggml_free(wctx);
        return false;
    }
    std::memcpy(hv->data, normed.data(), d_model * sizeof(float));

    ggml_tensor* lg = ggml_mul_mat(act, w, hv);  // [n_vocab]
    ggml_cgraph* gf = ggml_new_graph(act);
    ggml_build_forward_expand(gf, lg);

    if (!engine.compute(gf)) { ggml_free(act); ggml_free(wctx); return false; }

    logits.resize(n_vocab);
    ggml_backend_tensor_get(lg, logits.data(), 0, n_vocab * sizeof(float));

    ggml_free(act);
    ggml_free(wctx);
    return true;
}

// Public dispatch.
std::size_t run_layer_forward(RuntimeContext& ctx, const LayerContext& layer,
                              ComputeEngine& engine, LayerCache& cache) {
    if (!layer.desc) return 0;
    // For now we only handle the Mamba architecture.
    return run_mamba_block(ctx, *layer.desc,
                           layer.weights_base, layer.weights_size, engine, cache);
}

#endif  // REMORA_HAVE_GGML

}  // namespace remora
