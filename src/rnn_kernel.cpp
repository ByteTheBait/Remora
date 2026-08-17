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

std::size_t run_layer_forward(RuntimeContext& ctx, const LayerContext& layer) {
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
// Build a ggml_context that owns one block's worth of tensors. Each tensor's
// `.data` pointer is wired to point into the memory-mapped shard file at the
// recorded offset. The caller takes ownership of the returned context.
//
// Allocation strategy: a small no-alloc context for the metadata, plus
// `ggml_backend_alloc_ctx_tensors_from_buft` to allocate the actual
// per-tensor storage from the CPU backend buffer. For *quantized* tensors
// we then overwrite `.data` with a pointer into the shard. For F32 tensors
// we copy the bytes in (since the rest of the network assumes ownership).
// ---------------------------------------------------------------------------

struct ggml_context* remora_build_block_ctx(
    const void* shard_base,
    std::size_t shard_size,
    const LayerDesc& desc)
{
    // We need enough room for the tensor metadata: ~ sizeof(ggml_tensor)
    // per tensor, plus workspace. 64 KiB is comfortable for ~10 tensors.
    constexpr std::size_t kMetaBytes = 64 * 1024;
    struct ggml_init_params p = { .mem_size = kMetaBytes,
                                   .mem_buffer = nullptr,
                                   .no_alloc = true };
    struct ggml_context* ctx = ggml_init(p);
    if (!ctx) return nullptr;

    // First pass: create every tensor with the right shape + dtype.
    // We can't allocate data yet because we'll need to overwrite the
    // data pointer for quantized tensors. So just record names.
    for (const TensorDesc& td : desc.tensors) {
        if (td.shape.empty()) continue;
        ggml_type t = remora_dtype_from_string(td.dtype);
        if (t == GGML_TYPE_COUNT) {
            std::fprintf(stderr,
                "[remora] unknown dtype '%s' for tensor '%s'\n",
                td.dtype.c_str(), td.name.c_str());
            continue;
        }
        // ggml_new_tensor takes an int64_t array of sizes, with the
        // innermost dimension first. Our sidecar shape follows numpy /
        // gguf convention, which already does this.
        int n_dims = static_cast<int>(td.shape.size());
        std::vector<int64_t> ne(td.shape.begin(), td.shape.end());
        ggml_tensor* t_tensor = ggml_new_tensor(ctx, t, n_dims, ne.data());
        if (!t_tensor) {
            std::fprintf(stderr,
                "[remora] ggml_new_tensor failed for '%s'\n", td.name.c_str());
            continue;
        }
        // Name the tensor so graph debugging works.
        ggml_set_name(t_tensor, td.name.c_str());
    }

    // Allocate data buffers via the CPU backend.
    ggml_backend_t cpu =
        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        std::fprintf(stderr,
            "[remora] failed to init CPU backend (no libggml-cpu?)\n");
        ggml_free(ctx);
        return nullptr;
    }
    ggml_backend_buffer_type_t buft =
        ggml_backend_get_default_buffer_type(cpu);
    ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);

    // For each tensor, point its data at the shard at the recorded offset.
    // For F32 tensors we *copy* because the network will overwrite them
    // during forward pass. For quantized tensors we *alias* (the dequant
    // path is read-only) so we don't pay 16+ MB of allocation per block.
    for (const TensorDesc& td : desc.tensors) {
        if (td.shape.empty()) continue;
        // Find the named tensor in the context.
        ggml_tensor* t_tensor = ggml_get_tensor(ctx, td.name.c_str());
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
            t_tensor->data = const_cast<void*>(src);
        } else {
            // Plain (F32): copy so the tensor is self-contained.
            std::memcpy(t_tensor->data, src, want);
        }
    }

    ggml_backend_free(cpu);
    return ctx;
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
//
// This is a single-token forward; for a sequence, the caller loops over
// tokens and updates conv_hist in place.
struct ggml_tensor* mamba_conv1d(
        struct ggml_context* ctx,
        struct ggml_tensor* x,           // [d_inner], current token input
        struct ggml_tensor* weight,      // [d_conv, d_inner]
        struct ggml_tensor* bias,        // [d_inner]
        float* conv_hist)                // [(d_conv-1) * d_inner] scratch
{
    // y[i] = bias[i] + sum_{k=0..d_conv-1} weight[k, i] * input[k, i]
    //   where input[d_conv-1] = x (current) and input[k<d_conv-1] = conv_hist[k]
    //
    // For d_conv = 4: y = bias + W[0]*hist[0] + W[1]*hist[1] + W[2]*hist[2] + W[3]*x
    //
    // We implement this as a per-channel loop (d_inner = 4096 here, very
    // small) and produce a ggml_tensor for the result.

    const int64_t d_inner = x->ne[0];
    const int64_t d_conv  = weight->ne[1];
    const float* W = static_cast<const float*>(weight->data);
    const float* B = static_cast<const float*>(bias->data);
    const float* X = static_cast<const float*>(x->data);

    auto* out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_inner);
    float* Y = static_cast<float*>(out->data);
    for (int64_t i = 0; i < d_inner; ++i) {
        float y = B[i];
        // k = 0..d_conv-2 reads from history
        for (int64_t k = 0; k < d_conv - 1; ++k) {
            y += W[k * d_inner + i] * conv_hist[k * d_inner + i];
        }
        // k = d_conv-1 reads from current input
        y += W[(d_conv - 1) * d_inner + i] * X[i];
        Y[i] = y;
    }
    return out;
}

// Update the conv history: shift left by one (drop oldest), append x.
void mamba_conv_shift(float* conv_hist, const float* x, int64_t d_inner, int64_t d_conv) {
    // hist is (d_conv-1, d_inner) row-major; hist[k] = input at t - (d_conv-1-k)
    // After this step: new hist[0] = old hist[1], ..., new hist[d_conv-2] = x
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
        std::size_t shard_size)
{
    ggml_context* blk_ctx = remora_build_block_ctx(shard_base, shard_size, desc);
    if (!blk_ctx) return 0;

    ggml_backend_t cpu =
        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(cpu);

    // Allocate activations: a working context big enough for ~10 tensors.
    constexpr std::size_t kActBytes = 4 * 1024 * 1024;
    struct ggml_init_params ap = { .mem_size = kActBytes,
                                   .mem_buffer = nullptr,
                                   .no_alloc = true };
    ggml_context* act = ggml_init(ap);

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
        ggml_free(act); ggml_free(blk_ctx); ggml_backend_free(cpu);
        return 0;
    }

    const int64_t d_model = ctx.cfg.d_model;
    const int64_t d_inner = ctx.cfg.d_inner;
    const int64_t d_state = ctx.cfg.d_state;
    const int64_t d_conv  = ctx.cfg.d_conv;
    const int64_t dt_rank = ctx.cfg.dt_rank;

    // 1. Build activations and copy in the residual stream.
    ggml_tensor* y_norm = ggml_new_tensor_1d(act, GGML_TYPE_F32, d_model);
    ggml_tensor* xz     = ggml_new_tensor_1d(act, GGML_TYPE_F32, 2 * d_inner);
    ggml_tensor* x_pre  = ggml_new_tensor_1d(act, GGML_TYPE_F32, d_inner);
    ggml_tensor* z_pre  = ggml_new_tensor_1d(act, GGML_TYPE_F32, d_inner);
    ggml_tensor* x_conv = ggml_new_tensor_1d(act, GGML_TYPE_F32, d_inner);
    ggml_tensor* x_act  = ggml_new_tensor_1d(act, GGML_TYPE_F32, d_inner);
    ggml_tensor* x_dtbc = ggml_new_tensor_1d(act, GGML_TYPE_F32, dt_rank + 2*d_state);
    ggml_tensor* dt_pre = ggml_new_tensor_1d(act, GGML_TYPE_F32, dt_rank);
    ggml_tensor* B_pre  = ggml_new_tensor_1d(act, GGML_TYPE_F32, d_state);
    ggml_tensor* C_pre  = ggml_new_tensor_1d(act, GGML_TYPE_F32, d_state);
    ggml_tensor* dt_soft= ggml_new_tensor_1d(act, GGML_TYPE_F32, d_inner);
    ggml_tensor* y_ssm  = ggml_new_tensor_1d(act, GGML_TYPE_F32, d_inner);
    ggml_tensor* y_gated= ggml_new_tensor_1d(act, GGML_TYPE_F32, d_inner);
    ggml_tensor* y_out  = ggml_new_tensor_1d(act, GGML_TYPE_F32, d_model);

    ggml_backend_alloc_ctx_tensors_from_buft(act, buft);
    std::memcpy(y_norm->data, ctx.hidden_state.data(), d_model * sizeof(float));

    // 2. RMSNorm (manual; ggml_rms_norm adds eps+scale under the hood but
    //    we have a learned weight, so build it explicitly).
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

    // 3. in_proj: xz = ssm_in @ y_norm -> [2*d_inner].
    {
        ggml_tensor* in_xz = ggml_mul_mat(act, w_in_proj, y_norm);
        ggml_cgraph* gf = ggml_new_graph(act);
        ggml_build_forward_expand(gf, in_xz);
        ggml_backend_t backends[] = { cpu };
        ggml_backend_buffer_type_t bufts[] = { buft };
        ggml_backend_sched_t sched =
            ggml_backend_sched_new(backends, bufts, 1, 4096 * 4096, false, false);
        ggml_backend_sched_alloc_graph(sched, gf);
        ggml_backend_sched_graph_compute(sched, gf);
        std::memcpy(xz->data, in_xz->data, 2 * d_inner * sizeof(float));
        ggml_backend_sched_free(sched);
    }

    // Split xz into x and z.
    {
        const float* X = static_cast<const float*>(xz->data);
        float* x_out = static_cast<float*>(x_pre->data);
        float* z_out = static_cast<float*>(z_pre->data);
        std::memcpy(x_out, X, d_inner * sizeof(float));
        std::memcpy(z_out, X + d_inner, d_inner * sizeof(float));
    }

    // 4. causal conv1d on x.
    {
        float* hist = ctx.conv_state.data() +
                      desc.block_id * (d_conv - 1) * d_inner;
        ggml_tensor* xc = mamba_conv1d(act, x_pre, w_conv1d, b_conv1d, hist);
        std::memcpy(x_conv->data, xc->data, d_inner * sizeof(float));
        mamba_conv_shift(hist,
                         static_cast<const float*>(x_pre->data),
                         d_inner, d_conv);
    }

    // 5. silu(x_conv) -> x_act.
    {
        const float* x = static_cast<const float*>(x_conv->data);
        float* y = static_cast<float*>(x_act->data);
        for (int64_t i = 0; i < d_inner; ++i) {
            y[i] = x[i] / (1.0f + std::exp(-x[i]));
        }
    }

    // 6. ssm_x @ x_act -> [dt_rank + 2*d_state], then dt_proj -> dt.
    {
        ggml_tensor* xbc = ggml_mul_mat(act, w_ssm_x, x_act);
        ggml_tensor* dt_proj = ggml_mul_mat(act, w_dt_proj, x_act);
        ggml_tensor* dt_sp = ggml_softplus(act, dt_proj);
        ggml_cgraph* gf = ggml_new_graph(act);
        ggml_build_forward_expand(gf, xbc);
        ggml_build_forward_expand(gf, dt_sp);
        ggml_backend_t backends[] = { cpu };
        ggml_backend_buffer_type_t bufts[] = { buft };
        ggml_backend_sched_t sched =
            ggml_backend_sched_new(backends, bufts, 1, 4096 * 4096, false, false);
        ggml_backend_sched_alloc_graph(sched, gf);
        ggml_backend_sched_graph_compute(sched, gf);
        std::memcpy(x_dtbc->data, xbc->data,
                    (dt_rank + 2*d_state) * sizeof(float));
        std::memcpy(dt_soft->data, dt_sp->data, d_inner * sizeof(float));
        ggml_backend_sched_free(sched);
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

    // 7. Selective scan (one step).
    //
    // For each inner channel i (d_inner = 4096 of them, all parallel in time):
    //   A = -exp(ssm_a[:, i])   (shape [d_state])
    //   dt = dt_soft[i]         (scalar)
    //   B = B_pre               (shape [d_state])
    //   C = C_pre               (shape [d_state])
    //   A_bar = exp(dt * A)
    //   B_bar = dt * B
    //   h[:, i] = A_bar * h[:, i] + outer(B_bar, x_act[i])
    //   y_ssm[i] = dot(C, h[:, i]) + ssm_d[i] * x_act[i]
    //
    // State h has shape (d_state, d_inner). We keep it across tokens in
    // ctx.ssm_state.
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
                const float a_neg = -a_in[s * d_inner + i];   // -A_log[i, s] in canonical Mamba
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
        ggml_tensor* yo = ggml_mul_mat(act, w_out_proj, y_gated);
        ggml_cgraph* gf = ggml_new_graph(act);
        ggml_build_forward_expand(gf, yo);
        ggml_backend_t backends[] = { cpu };
        ggml_backend_buffer_type_t bufts[] = { buft };
        ggml_backend_sched_t sched =
            ggml_backend_sched_new(backends, bufts, 1, 4096 * 4096, false, false);
        ggml_backend_sched_alloc_graph(sched, gf);
        ggml_backend_sched_graph_compute(sched, gf);
        std::memcpy(y_out->data, yo->data, d_model * sizeof(float));
        ggml_backend_sched_free(sched);
    }

    // 10. Residual: hidden_state += y_out.
    {
        const float* yo = static_cast<const float*>(y_out->data);
        float* h = ctx.hidden_state.data();
        for (int64_t i = 0; i < d_model; ++i) h[i] += yo[i];
    }

    ggml_free(act);
    ggml_free(blk_ctx);
    ggml_backend_free(cpu);

    return shard_size;
}

}  // namespace

// Public dispatch.
std::size_t run_layer_forward(RuntimeContext& ctx, const LayerContext& layer) {
    if (!layer.desc) return 0;
    // For now we only handle the Mamba architecture.
    return run_mamba_block(ctx, *layer.desc,
                           layer.weights_base, layer.weights_size);
}

#endif  // REMORA_HAVE_GGML

}  // namespace remora