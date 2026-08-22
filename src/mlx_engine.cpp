#include "mlx_engine.h"

#include <cstring>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

// MLX C++ API
#include <mlx/mlx.h>

namespace remora {

// ---------------------------------------------------------------------------
// FP16 -> FP32 decode (host-side). MLX GPU F16 readback is unreliable in the
// shipped libmlx build, so we decode weights to F32 on the host once, then
// upload the F32 arrays to the GPU.
// ---------------------------------------------------------------------------
static float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t frac = h & 0x3ff;
    float v;
    if (exp == 0) {
        v = frac * 0.00006103515625f;                       // subnormal
    } else if (exp == 31) {
        v = frac ? NAN : INFINITY;
    } else {
        v = (1.0f + frac / 1024.0f) * std::ldexp(1.0f, (int)exp - 15);
    }
    return sign ? -v : v;
}

// ---------------------------------------------------------------------------
// MLX API surface used below.
// ---------------------------------------------------------------------------
using mlx::core::array;
using mlx::core::Device;
using mlx::core::Dtype;
using mlx::core::float32;
using mlx::core::uint32;
using mlx::core::matmul;
using mlx::core::add;
using mlx::core::exp;
using mlx::core::log1p;
using mlx::core::reshape;
using mlx::core::quantized_matmul;
using mlx::core::dequantize;
using mlx::core::synchronize;
using mlx::core::new_stream;
using mlx::core::default_stream;
using mlx::core::Stream;
using mlx::core::fast::rms_norm;

// Read an F32 MLX array's host-visible data into `out`. MLX ops are lazy, so
// we must `eval()` first to force the (GPU) compute and materialize the buffer.
static void read_f32(const array& a, std::vector<float>& out) {
    const_cast<array&>(a).eval();
    out.resize(static_cast<size_t>(a.size()));
    const float* p = static_cast<const float*>(
        a.data_shared_ptr()->buffer.raw_ptr());
    std::memcpy(out.data(), p, out.size() * sizeof(float));
}

// ---------------------------------------------------------------------------
// MlxGpuCache: persistent GPU arrays.
//
// Quantized (4-bit affine) blocks run their four projection matmuls entirely
// on the GPU via `quantized_matmul` (U32 packed weights + F16 scales/biases),
// so we never materialize the full F32 weight matrices. The small per-block
// tensors (norm, conv, A_log, D) are F16 and decoded to F32 host-side once.
// ---------------------------------------------------------------------------
struct MlxGpuCache {
    // lm_head for the *unquantized* F16 path: [vocab, d_model] F32.
    std::optional<array> lm_head;
    bool lm_head_valid = false;
    int64_t lm_head_vocab = 0, lm_head_d_model = 0;

    // Quantized lm_head (tied embeddings): raw host copies + GPU arrays.
    std::vector<uint32_t> q_lm_w_host;   // [vocab, d_model/8] packed
    std::vector<float> q_lm_s_host;      // [vocab, d_model/64]
    std::vector<float> q_lm_b_host;      // [vocab, d_model/64]
    int64_t q_lm_vocab = 0, q_lm_d_model = 0;
    bool q_lm_valid = false;

    void set_lm_head(const float* W, int64_t vocab, int64_t d_model) {
        lm_head.emplace(array(W, {(int)vocab, (int)d_model}, float32));
        lm_head_valid = true;
        lm_head_vocab = vocab;
        lm_head_d_model = d_model;
    }

    // Per-block persistent GPU weight arrays. Uploaded once per block on first
    // touch, then reused across every token.
    struct BlockGpu {
        bool quantized = false;             // true if this block is 4-bit affine
        // Quantized projections (U32 packed weight + F16 scales/biases).
        std::optional<array> q_w_in, q_s_in, q_b_in;    // in_proj
        std::optional<array> q_w_x,  q_s_x,  q_b_x;     // x_proj
        std::optional<array> q_w_dt, q_s_dt, q_b_dt;    // dt_proj
        std::optional<array> q_w_out, q_s_out, q_b_out; // out_proj
        // Unquantized fallback F32 projection weights.
        std::optional<array> w_in, w_x, w_dt, b_dt, w_out;
        // Shared small per-block tensors (F16 -> F32).
        std::optional<array> w_norm;
        int64_t d_model = 0, d_inner = 0, dt_rank = 0, d_state = 0, d_conv = 0;
    };
    std::vector<std::optional<BlockGpu>> blocks;   // indexed by block_id
};

// ---------------------------------------------------------------------------
// MlxEngine
// ---------------------------------------------------------------------------

MlxEngine::MlxEngine() {
    if (!mlx::core::is_available(Device::gpu)) {
        std::fprintf(stderr,
            "[mlx] MLX GPU backend unavailable (no Metal on this device)\n");
        ok_ = false;
        return;
    }
    ok_ = true;
    gpu_cache_ = std::make_unique<MlxGpuCache>();
}

MlxEngine::~MlxEngine() = default;

bool MlxEngine::ok() const { return ok_; }

const char* MlxEngine::backend_name() const { return "MLX GPU (Metal)"; }

// Locate a cached weight by suffix; returns a pointer into its data buffer.
// For F16 tensors `data` holds decoded F32 values. For U32 tensors `data`
// holds the raw packed uint32 bytes (bitcast to float), so reinterpret_cast
// the pointer to uint32_t* to read the packed values.
const float* MlxEngine::find_weight(const BlockCache& blk,
                                    const std::string& suffix,
                                    int64_t* n_elements_out) const {
    for (const auto& w : blk.weights) {
        if (w.name.size() >= suffix.size() &&
            w.name.compare(w.name.size() - suffix.size(), suffix.size(),
                           suffix) == 0) {
            if (n_elements_out) *n_elements_out = w.n_elements;
            return w.data.data();
        }
    }
    return nullptr;
}

const uint32_t* MlxEngine::find_u32(const BlockCache& blk,
                                    const std::string& suffix) const {
    for (const auto& w : blk.weights) {
        if (w.dtype == "U32" &&
            w.name.size() >= suffix.size() &&
            w.name.compare(w.name.size() - suffix.size(), suffix.size(),
                           suffix) == 0) {
            return reinterpret_cast<const uint32_t*>(w.data.data());
        }
    }
    return nullptr;
}

bool MlxEngine::prepare_block(const LayerDesc& desc, const void* shard_base,
                              std::size_t shard_size) {
    if (desc.block_id >= 0 && desc.block_id < (int)blocks_.size()) {
        BlockCache& blk = blocks_[desc.block_id];
        if (!blk.weights.empty()) return true;   // already decoded this run
    }

    if (desc.block_id < 0) return false;
    if (desc.block_id >= (int)blocks_.size()) blocks_.resize(desc.block_id + 1);
    BlockCache& blk = blocks_[desc.block_id];
    blk.block_id = desc.block_id;
    blk.weights.clear();

    const uint8_t* base = static_cast<const uint8_t*>(shard_base);
    for (const TensorDesc& td : desc.tensors) {
        if (td.offset + td.size > shard_size) continue;
        CachedWeight cw;
        cw.name = td.name;
        cw.n_elements = td.n_elements > 0 ? td.n_elements : 1;
        if (td.dtype == "U32") {
            // Raw packed 4-bit data (8 int4 per uint32). Keep the raw bytes
            // in the float buffer (bitcast) so `find_u32` can reinterpret it.
            cw.dtype = "U32";
            cw.data.resize(cw.n_elements);
            std::memcpy(cw.data.data(), base + td.offset,
                        (size_t)cw.n_elements * sizeof(uint32_t));
        } else if (td.dtype == "F16") {
            cw.dtype = "F16";
            cw.data.resize(cw.n_elements);
            for (int64_t i = 0; i < cw.n_elements; ++i) {
                const uint16_t h =
                    *reinterpret_cast<const uint16_t*>(base + td.offset + i * 2);
                cw.data[i] = half_to_float(h);
            }
        } else {
            continue;   // unsupported dtype for this block
        }
        blk.weights.push_back(std::move(cw));
    }
    return !blk.weights.empty();
}

// Mark a block as recently used.
void MlxEngine::touch_block(int block_id) {
    if (block_id < 0) return;
    if ((int)recency_.size() <= block_id) recency_.resize(block_id + 1, 0);
    recency_[block_id] = ++tick_;
}

// Count resident blocks (host decode + GPU arrays present).
std::size_t MlxEngine::count_resident() const {
    std::size_t n = 0;
    for (std::size_t i = 0; i < blocks_.size(); ++i) {
        bool host = !blocks_[i].weights.empty();
        bool gpu_ = i < gpu_cache_->blocks.size() && gpu_cache_->blocks[i].has_value();
        if (host || gpu_) ++n;
    }
    return n;
}

// Evict the least-recently-used resident block to keep peak RSS bounded to
// ~max_cached_blocks_ instead of the whole model.
void MlxEngine::evict_if_needed() {
    if (max_cached_blocks_ == 0) return;
    while (count_resident() > max_cached_blocks_) {
        // Find the LRU resident block.
        int victim = -1;
        uint64_t oldest = ~0ull;
        for (std::size_t i = 0; i < blocks_.size(); ++i) {
            bool resident = !blocks_[i].weights.empty() ||
                (i < gpu_cache_->blocks.size() && gpu_cache_->blocks[i].has_value());
            if (!resident) continue;
            uint64_t age = (i < recency_.size()) ? recency_[i] : 0;
            if (age < oldest) { oldest = age; victim = (int)i; }
        }
        if (victim < 0) break;
        // Free the GPU arrays (reset the BlockGpu).
        if ((size_t)victim < gpu_cache_->blocks.size())
            gpu_cache_->blocks[victim].reset();
        // Free the host decode.
        blocks_[victim].weights.clear();
        blocks_[victim].weights.shrink_to_fit();
    }
}

// ---------------------------------------------------------------------------
// Global tensor access (global.bin). Tensors are looked up by name suffix.
// ---------------------------------------------------------------------------
namespace {

const TensorDesc* find_tdesc(const LayerDesc& gdesc, const std::string& suffix) {
    for (const auto& td : gdesc.tensors) {
        if (td.name.size() >= suffix.size() &&
            td.name.compare(td.name.size() - suffix.size(), suffix.size(),
                            suffix) == 0)
            return &td;
    }
    return nullptr;
}

const float* decode_f16_tensor(const LayerDesc& gdesc, const void* base,
                               std::size_t size, const std::string& suffix,
                               std::vector<float>& out) {
    const TensorDesc* td = find_tdesc(gdesc, suffix);
    if (!td || td->dtype != "F16") return nullptr;
    const uint8_t* b = static_cast<const uint8_t*>(base);
    const int64_t n = td->n_elements > 0 ? td->n_elements : 1;
    out.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        if (td->offset + (i + 1) * 2 > size) break;
        const uint16_t h = *reinterpret_cast<const uint16_t*>(b + td->offset + i * 2);
        out[i] = half_to_float(h);
    }
    return out.data();
}

bool copy_u32_tensor(const LayerDesc& gdesc, const void* base,
                     std::size_t size, const std::string& suffix,
                     std::vector<uint32_t>& out) {
    const TensorDesc* td = find_tdesc(gdesc, suffix);
    if (!td || td->dtype != "U32") return false;
    const uint8_t* b = static_cast<const uint8_t*>(base);
    const int64_t n = td->n_elements > 0 ? td->n_elements : 1;
    out.resize(static_cast<size_t>(n));
    if (td->offset + (size_t)n * 4 > size) return false;
    std::memcpy(out.data(), b + td->offset, (size_t)n * sizeof(uint32_t));
    return true;
}

}  // namespace

// Cached host decode of the global (unquantized F16) embedding / final norm.
const float* MlxEngine::global_tensor(const LayerDesc& gdesc, const void* base,
                                      std::size_t size,
                                      const std::string& suffix,
                                      std::vector<float>& out,
                                      std::size_t* n_out) {
    if (global_cache_base_ != base || global_cache_size_ != size) {
        global_cache_base_ = base;
        global_cache_size_ = size;
        cached_embeddings_.clear();
        cached_norm_f_.clear();
    }
    if (suffix == ".embeddings.weight" && !cached_embeddings_.empty()) {
        if (n_out) *n_out = cached_embeddings_.size();
        return cached_embeddings_.data();
    }
    if (suffix == ".norm_f.weight" && !cached_norm_f_.empty()) {
        if (n_out) *n_out = cached_norm_f_.size();
        return cached_norm_f_.data();
    }
    const float* p = decode_f16_tensor(gdesc, base, size, suffix, out);
    if (!p) return nullptr;
    if (suffix == ".embeddings.weight") {
        cached_embeddings_ = std::move(out);
        if (n_out) *n_out = cached_embeddings_.size();
        return cached_embeddings_.data();
    }
    if (suffix == ".norm_f.weight") {
        cached_norm_f_ = std::move(out);
        if (n_out) *n_out = cached_norm_f_.size();
        return cached_norm_f_.data();
    }
    if (n_out) *n_out = out.size();
    return p;
}

bool MlxEngine::embed(RuntimeContext& ctx, int32_t id,
                      const LayerDesc& global_desc, const void* global_base,
                      std::size_t global_size) {
    const int64_t d_model = ctx.cfg.d_model;

    // Quantized embedding (U32 packed). Dequantize the single row on the GPU.
    const TensorDesc* wt = find_tdesc(global_desc, ".embeddings.weight");
    if (wt && wt->dtype == "U32") {
        if (!gpu_cache_->q_lm_valid) {
            std::vector<uint32_t> w; std::vector<float> s, b;
            if (!copy_u32_tensor(global_desc, global_base, global_size,
                                 ".embeddings.weight", w)) return false;
            if (!decode_f16_tensor(global_desc, global_base, global_size,
                                   ".embeddings.scales", s)) return false;
            if (!decode_f16_tensor(global_desc, global_base, global_size,
                                   ".embeddings.biases", b)) return false;
            gpu_cache_->q_lm_w_host = std::move(w);
            gpu_cache_->q_lm_s_host = std::move(s);
            gpu_cache_->q_lm_b_host = std::move(b);
            gpu_cache_->q_lm_vocab = ctx.cfg.vocab_size;
            gpu_cache_->q_lm_d_model = d_model;
            gpu_cache_->q_lm_valid = true;
        }
        const int64_t vocab = gpu_cache_->q_lm_vocab;
        const int64_t in8 = d_model / 8, n_groups = d_model / 64;
        if (id < 0 || id >= vocab) return false;
        const Device gpu = Device::gpu;
        array wrow = array(&gpu_cache_->q_lm_w_host[(size_t)id * in8],
                           {1, (int)in8}, uint32);
        array srow = array(&gpu_cache_->q_lm_s_host[(size_t)id * n_groups],
                           {1, (int)n_groups}, float32);
        array brow = array(&gpu_cache_->q_lm_b_host[(size_t)id * n_groups],
                           {1, (int)n_groups}, float32);
        array emb = dequantize(wrow, srow, brow, 64, 4, "affine", gpu);  // [1, d_model]
        std::vector<float> row(d_model);
        read_f32(emb, row);
        std::memcpy(ctx.hidden_state.data(), row.data(), d_model * sizeof(float));
        return true;
    }

    // Unquantized F16 path.
    std::vector<float> emb_table;
    const float* W = global_tensor(global_desc, global_base,
                                   global_size, ".embeddings.weight",
                                   emb_table, nullptr);
    if (!W || id < 0 || id >= ctx.cfg.vocab_size) return false;
    std::memcpy(ctx.hidden_state.data(), W + (int64_t)id * d_model,
                d_model * sizeof(float));
    return true;
}

bool MlxEngine::compute_logits(RuntimeContext& ctx,
                               std::vector<float>& logits,
                               const LayerDesc& global_desc,
                               const void* global_base,
                               std::size_t global_size) {
    const int64_t d_model = ctx.cfg.d_model;
    const int64_t n_vocab = ctx.cfg.vocab_size;

    // 1. Final RMSNorm (always F16 in both layouts).
    std::vector<float> norm_w;
    const float* wn = global_tensor(global_desc, global_base,
                                    global_size, ".norm_f.weight",
                                    norm_w, nullptr);
    std::vector<float> normed(d_model);
    const float* x = ctx.hidden_state.data();
    float mean_sq = 0.0f;
    for (int64_t i = 0; i < d_model; ++i) mean_sq += x[i] * x[i];
    mean_sq /= d_model;
    const float inv_rms = 1.0f / std::sqrt(mean_sq + 1e-5f);
    for (int64_t i = 0; i < d_model; ++i)
        normed[i] = x[i] * inv_rms * (wn ? wn[i] : 1.0f);

    const Device gpu = Device::gpu;
    array hv = array(normed.data(), {1, (int)d_model}, float32);

    // 2. lm_head.
    const TensorDesc* wt = find_tdesc(global_desc, ".embeddings.weight");
    if (wt && wt->dtype == "U32") {
        // Quantized lm_head (tied embeddings).
        if (!gpu_cache_->q_lm_valid ||
            gpu_cache_->q_lm_vocab != n_vocab ||
            gpu_cache_->q_lm_d_model != d_model) {
            if (!copy_u32_tensor(global_desc, global_base, global_size,
                                 ".embeddings.weight", gpu_cache_->q_lm_w_host))
                return false;
            if (!decode_f16_tensor(global_desc, global_base, global_size,
                                   ".embeddings.scales", gpu_cache_->q_lm_s_host))
                return false;
            if (!decode_f16_tensor(global_desc, global_base, global_size,
                                   ".embeddings.biases", gpu_cache_->q_lm_b_host))
                return false;
            gpu_cache_->q_lm_vocab = n_vocab;
            gpu_cache_->q_lm_d_model = d_model;
            gpu_cache_->q_lm_valid = true;
        }
        array qw = array(gpu_cache_->q_lm_w_host.data(),
                         {(int)n_vocab, (int)(d_model / 8)}, uint32);
        array qs = array(gpu_cache_->q_lm_s_host.data(),
                         {(int)n_vocab, (int)(d_model / 64)}, float32);
        array qb = array(gpu_cache_->q_lm_b_host.data(),
                         {(int)n_vocab, (int)(d_model / 64)}, float32);
        array lg = quantized_matmul(hv, qw, qs, qb, /*transpose=*/true,
                                    64, 4, "affine", gpu);   // [1, vocab]
        read_f32(reshape(lg, {(int)n_vocab}), logits);
        return logits.size() == (size_t)n_vocab;
    }

    // Unquantized F16 lm_head.
    std::vector<float> emb;
    const float* W = global_tensor(global_desc, global_base,
                                   global_size, ".embeddings.weight",
                                   emb, nullptr);
    if (!W) return false;
    if (!gpu_cache_->lm_head_valid ||
        gpu_cache_->lm_head_vocab != n_vocab ||
        gpu_cache_->lm_head_d_model != d_model) {
        gpu_cache_->set_lm_head(W, n_vocab, d_model);
    }
    array lg = matmul(*gpu_cache_->lm_head, hv, gpu);   // [vocab, 1]
    read_f32(reshape(lg, {(int)n_vocab}), logits);
    return logits.size() == (size_t)n_vocab;
}

std::size_t MlxEngine::run_layer_forward(RuntimeContext& ctx,
                                         const LayerContext& layer) {
    if (!layer.desc) return 0;
    const LayerDesc& desc = *layer.desc;

    if (!prepare_block(desc, layer.weights_base, layer.weights_size)) return 0;
    touch_block(desc.block_id);
    evict_if_needed();
    BlockCache& blk = blocks_[desc.block_id];

    const int64_t d_model = ctx.cfg.d_model;
    const int64_t d_inner = ctx.cfg.d_inner;
    const int64_t d_state = ctx.cfg.d_state;
    const int64_t d_conv  = ctx.cfg.d_conv;
    const int64_t dt_rank = ctx.cfg.dt_rank;

    // --- Persistent per-block GPU arrays (uploaded once, reused every token) ---
    if (desc.block_id >= (int)gpu_cache_->blocks.size())
        gpu_cache_->blocks.resize(desc.block_id + 1);
    auto& bg = gpu_cache_->blocks[desc.block_id];
    if (!bg) bg.emplace();
    MlxGpuCache::BlockGpu& g = *bg;

    const Device gpu = Device::gpu;

    // Build this block's GPU arrays on first touch.
    if (!g.w_norm) {
        g.d_model = d_model; g.d_inner = d_inner; g.dt_rank = dt_rank;
        g.d_state = d_state; g.d_conv = d_conv;

        const float* w_norm = find_weight(blk, ".norm.weight", nullptr);
        if (!w_norm) return 0;
        g.w_norm.emplace(array(w_norm, {(int)d_model}, float32));

        // Detect quantization: U32 packed `.mixer.in_proj.weight` present.
        const uint32_t* in_u32 = find_u32(blk, ".mixer.in_proj.weight");
        if (in_u32) {
            g.quantized = true;
            const uint32_t* s_in = find_u32(blk, ".mixer.in_proj.weight");
            (void)s_in;
            // in_proj
            g.q_w_in.emplace(array(find_u32(blk, ".mixer.in_proj.weight"),
                                   {(int)(2*d_inner), (int)(d_model/8)}, uint32));
            g.q_s_in.emplace(array(find_weight(blk, ".mixer.in_proj.scales"),
                                   {(int)(2*d_inner), (int)(d_model/64)}, float32));
            g.q_b_in.emplace(array(find_weight(blk, ".mixer.in_proj.biases"),
                                   {(int)(2*d_inner), (int)(d_model/64)}, float32));
            // x_proj
            g.q_w_x.emplace(array(find_u32(blk, ".mixer.x_proj.weight"),
                                   {(int)(dt_rank+2*d_state), (int)(d_inner/8)}, uint32));
            g.q_s_x.emplace(array(find_weight(blk, ".mixer.x_proj.scales"),
                                   {(int)(dt_rank+2*d_state), (int)(d_inner/64)}, float32));
            g.q_b_x.emplace(array(find_weight(blk, ".mixer.x_proj.biases"),
                                   {(int)(dt_rank+2*d_state), (int)(d_inner/64)}, float32));
            // dt_proj
            g.q_w_dt.emplace(array(find_u32(blk, ".mixer.dt_proj.weight"),
                                   {(int)d_inner, (int)(dt_rank/8)}, uint32));
            g.q_s_dt.emplace(array(find_weight(blk, ".mixer.dt_proj.scales"),
                                   {(int)d_inner, (int)(dt_rank/64)}, float32));
            g.q_b_dt.emplace(array(find_weight(blk, ".mixer.dt_proj.biases"),
                                   {(int)d_inner, (int)(dt_rank/64)}, float32));
            g.w_dt.emplace(array(find_weight(blk, ".mixer.dt_proj.bias"),
                                   {(int)d_inner}, float32));  // plain bias
            // out_proj
            g.q_w_out.emplace(array(find_u32(blk, ".mixer.out_proj.weight"),
                                    {(int)d_model, (int)(d_inner/8)}, uint32));
            g.q_s_out.emplace(array(find_weight(blk, ".mixer.out_proj.scales"),
                                    {(int)d_model, (int)(d_inner/64)}, float32));
            g.q_b_out.emplace(array(find_weight(blk, ".mixer.out_proj.biases"),
                                    {(int)d_model, (int)(d_inner/64)}, float32));
        } else {
            // Unquantized F16 path.
            g.quantized = false;
            const float* w_in  = find_weight(blk, ".mixer.in_proj.weight", nullptr);
            const float* w_x   = find_weight(blk, ".mixer.x_proj.weight", nullptr);
            const float* w_dt  = find_weight(blk, ".mixer.dt_proj.weight", nullptr);
            const float* b_dt  = find_weight(blk, ".mixer.dt_proj.bias", nullptr);
            const float* w_out = find_weight(blk, ".mixer.out_proj.weight", nullptr);
            if (!w_in || !w_x || !w_dt || !b_dt || !w_out) return 0;
            g.w_in.emplace(array(w_in, {(int)(2*d_inner), (int)d_model}, float32));
            g.w_x.emplace(array(w_x, {(int)(dt_rank+2*d_state), (int)d_inner}, float32));
            g.w_dt.emplace(array(w_dt, {(int)d_inner, (int)dt_rank}, float32));
            g.b_dt.emplace(array(b_dt, {(int)d_inner}, float32));
            g.w_out.emplace(array(w_out, {(int)d_model, (int)d_inner}, float32));
        }
    }

    array h = array(ctx.hidden_state.data(), {(int)d_model}, float32);

    // 1. RMSNorm.
    array y_norm = rms_norm(h, *g.w_norm, 1e-5f, gpu);

    // 2. in_proj: xz = W_in @ y_norm -> [2*d_inner]
    std::optional<array> xz_arr;
    if (g.quantized)
        xz_arr.emplace(quantized_matmul(reshape(y_norm, {1, (int)d_model}),
                                  *g.q_w_in, *g.q_s_in, *g.q_b_in,
                                  true, 64, 4, "affine", gpu));   // [1, 2*d_inner]
    else
        xz_arr.emplace(matmul(*g.w_in, y_norm, gpu));                    // [2*d_inner]
    std::vector<float> xz_host;
    read_f32(*xz_arr, xz_host);
    if (xz_host.size() != (size_t)(2*d_inner)) {
        std::fprintf(stderr, "[mlx] in_proj returned %zu, want %zu\n",
                     xz_host.size(), (size_t)(2*d_inner));
        return 0;
    }

    std::vector<float> x(d_inner), z(d_inner);
    std::memcpy(x.data(), xz_host.data(), d_inner * sizeof(float));
    std::memcpy(z.data(), xz_host.data() + d_inner, d_inner * sizeof(float));

    // 3. causal conv1d on x (host).
    const float* b_conv = find_weight(blk, ".mixer.conv1d.bias", nullptr);
    const float* w_c = find_weight(blk, ".mixer.conv1d.weight", nullptr);
    if (!b_conv || !w_c) return 0;
    float* conv_hist = ctx.conv_state.data() +
                       desc.block_id * (d_conv - 1) * d_inner;
    std::vector<float> x_conv(d_inner);
    for (int64_t i = 0; i < d_inner; ++i) {
        float y = b_conv[i];
        for (int64_t k = 0; k < d_conv - 1; ++k)
            y += w_c[k * d_inner + i] * conv_hist[k * d_inner + i];
        y += w_c[(d_conv - 1) * d_inner + i] * x[i];
        x_conv[i] = y;
    }
    for (int64_t k = 0; k < d_conv - 2; ++k)
        std::memcpy(conv_hist + k * d_inner, conv_hist + (k + 1) * d_inner,
                    d_inner * sizeof(float));
    std::memcpy(conv_hist + (d_conv - 2) * d_inner, x.data(),
                d_inner * sizeof(float));

    // 4. silu(x_conv) -> x_act (host).
    std::vector<float> x_act(d_inner);
    for (int64_t i = 0; i < d_inner; ++i)
        x_act[i] = x_conv[i] / (1.0f + std::exp(-x_conv[i]));

    // 5. x_proj: xbc = W_x @ x_act -> [dt_rank + 2*d_state]
    std::optional<array> xbc_arr;
    if (g.quantized)
        xbc_arr.emplace(quantized_matmul(
            array(x_act.data(), {1, (int)d_inner}, float32),
            *g.q_w_x, *g.q_s_x, *g.q_b_x, true, 64, 4, "affine", gpu));
    else
        xbc_arr.emplace(matmul(*g.w_x,
                         array(x_act.data(), {(int)d_inner}, float32), gpu));
    std::vector<float> xbc_host;
    read_f32(*xbc_arr, xbc_host);
    if (xbc_host.size() != (size_t)(dt_rank + 2*d_state)) return 0;

    std::vector<float> dt_pre(dt_rank), B(d_state), C(d_state);
    std::memcpy(dt_pre.data(), xbc_host.data(), dt_rank * sizeof(float));
    std::memcpy(B.data(), xbc_host.data() + dt_rank, d_state * sizeof(float));
    std::memcpy(C.data(), xbc_host.data() + dt_rank + d_state,
                d_state * sizeof(float));

    // 6. dt_proj: dt = softplus(W_dt @ dt_pre + b_dt)
    std::optional<array> dt_lin;
    if (g.quantized)
        dt_lin.emplace(add(quantized_matmul(
                         array(dt_pre.data(), {1, (int)dt_rank}),
                         *g.q_w_dt, *g.q_s_dt, *g.q_b_dt, true, 64, 4, "affine", gpu),
                     *g.w_dt));
    else
        dt_lin.emplace(add(matmul(*g.w_dt,
                            array(dt_pre.data(), {(int)dt_rank}, float32), gpu),
                     *g.b_dt));
    array dt_soft = log1p(exp(*dt_lin));
    std::vector<float> dt(d_inner);
    read_f32(reshape(dt_soft, {(int)d_inner}), dt);

    // 7. Selective scan (one step, host sequential loop).
    const float* a_log = find_weight(blk, ".mixer.A_log", nullptr);
    const float* ssm_d = find_weight(blk, ".mixer.D", nullptr);
    if (!a_log || !ssm_d) return 0;
    float* h_ssm = ctx.ssm_state.data() +
                   desc.block_id * d_state * d_inner;
    std::vector<float> y_ssm(d_inner);
    for (int64_t i = 0; i < d_inner; ++i) {
        float yi = ssm_d[i] * x_act[i];
        for (int64_t s = 0; s < d_state; ++s) {
            const float a_neg = -a_log[s * d_inner + i];
            const float A_bar = std::exp(dt[i] * a_neg);
            const float B_bar = dt[i] * B[s];
            const float h_new = A_bar * h_ssm[s * d_inner + i] +
                                B_bar * x_act[i];
            h_ssm[s * d_inner + i] = h_new;
            yi += C[s] * h_new;
        }
        y_ssm[i] = yi;
    }

    // 8. silu(z) * y_ssm -> y_gated (host).
    std::vector<float> y_gated(d_inner);
    for (int64_t i = 0; i < d_inner; ++i) {
        const float sz = z[i] / (1.0f + std::exp(-z[i]));
        y_gated[i] = sz * y_ssm[i];
    }

    // 9. out_proj: y_out = W_out @ y_gated -> [d_model]
    std::optional<array> yo_arr;
    if (g.quantized)
        yo_arr.emplace(quantized_matmul(
            array(y_gated.data(), {1, (int)d_inner}),
            *g.q_w_out, *g.q_s_out, *g.q_b_out, true, 64, 4, "affine", gpu));
    else
        yo_arr.emplace(matmul(*g.w_out,
                        array(y_gated.data(), {(int)d_inner}, float32), gpu));
    std::vector<float> y_out(d_model);
    read_f32(*yo_arr, y_out);
    if (y_out.size() != (size_t)d_model) return 0;

    // 10. Residual: hidden_state += y_out.
    for (int64_t i = 0; i < d_model; ++i)
        ctx.hidden_state[i] += y_out[i];

    return layer.weights_size;
}

// ---------------------------------------------------------------------------
// Per-token compute against the shared per-block weights. The weights (blk
// host decode + g GPU arrays) are read-only and shared across all batch
// threads; only `ctx` is thread-private.
// ---------------------------------------------------------------------------
void MlxEngine::compute_token(RuntimeContext& ctx, const LayerDesc& desc,
                              const BlockCache& blk, void* gpu_blk,
                              const mlx::core::Stream& stream,
                              std::mutex& gpu_lock) {
    MlxGpuCache::BlockGpu& g = *static_cast<MlxGpuCache::BlockGpu*>(gpu_blk);
    const int64_t d_model = ctx.cfg.d_model;
    const int64_t d_inner = ctx.cfg.d_inner;
    const int64_t d_state = ctx.cfg.d_state;
    const int64_t d_conv  = ctx.cfg.d_conv;
    const int64_t dt_rank = ctx.cfg.dt_rank;

    // The heavy matmuls + their GPU eval/readback are serialized under the
    // mutex (Metal is NOT thread-safe for concurrent eval), but the host-side
    // selective-scan / conv / gating below run WITHOUT the lock, so across a
    // batch these overlap across threads — that is the real Mamba bottleneck.

    std::vector<float> x, z, x_act, dt, y_ssm, y_gated, y_out;
    x.resize(d_inner); z.resize(d_inner); x_act.resize(d_inner);
    dt.resize(d_inner); y_ssm.resize(d_inner); y_gated.resize(d_inner);

    // --- Phase A (GPU, locked): RMSNorm + in_proj -> x,z ------------------
    {
        std::lock_guard<std::mutex> lk(gpu_lock);
        const Device gpu = Device::gpu;
        array h = array(ctx.hidden_state.data(), {(int)d_model}, float32);
        array y_norm = rms_norm(h, *g.w_norm, 1e-5f, stream);
        std::optional<array> xz_arr;
        if (g.quantized)
            xz_arr.emplace(quantized_matmul(reshape(y_norm, {1, (int)d_model}, stream),
                                      *g.q_w_in, *g.q_s_in, *g.q_b_in,
                                      true, 64, 4, "affine", stream));
        else
            xz_arr.emplace(matmul(*g.w_in, y_norm, stream));
        std::vector<float> xz_host;
        read_f32(*xz_arr, xz_host);
        std::memcpy(x.data(), xz_host.data(), d_inner * sizeof(float));
        std::memcpy(z.data(), xz_host.data() + d_inner, d_inner * sizeof(float));
    }

    // --- Host: causal conv1d + silu -> x_act (overlaps across threads) -----
    const float* b_conv = find_weight(blk, ".mixer.conv1d.bias", nullptr);
    const float* w_c = find_weight(blk, ".mixer.conv1d.weight", nullptr);
    float* conv_hist = ctx.conv_state.data() +
                       desc.block_id * (d_conv - 1) * d_inner;
    for (int64_t i = 0; i < d_inner; ++i) {
        float y = b_conv[i];
        for (int64_t k = 0; k < d_conv - 1; ++k)
            y += w_c[k * d_inner + i] * conv_hist[k * d_inner + i];
        y += w_c[(d_conv - 1) * d_inner + i] * x[i];
        x_act[i] = y / (1.0f + std::exp(-y));   // silu
    }
    for (int64_t k = 0; k < d_conv - 2; ++k)
        std::memcpy(conv_hist + k * d_inner, conv_hist + (k + 1) * d_inner,
                    d_inner * sizeof(float));
    std::memcpy(conv_hist + (d_conv - 2) * d_inner, x.data(),
                d_inner * sizeof(float));

    // --- Phase B (GPU, locked): x_proj + dt_proj -> B, C, dt ---------------
    std::vector<float> B(d_state), C(d_state);
    std::vector<float> dt_pre(dt_rank);
    {
        std::lock_guard<std::mutex> lk(gpu_lock);
        const Device gpu = Device::gpu;
        std::optional<array> xbc_arr;
        if (g.quantized)
            xbc_arr.emplace(quantized_matmul(
                array(x_act.data(), {1, (int)d_inner}, float32),
                *g.q_w_x, *g.q_s_x, *g.q_b_x, true, 64, 4, "affine", stream));
        else
            xbc_arr.emplace(matmul(*g.w_x,
                             array(x_act.data(), {(int)d_inner}, float32), stream));
        std::vector<float> xbc_host;
        read_f32(*xbc_arr, xbc_host);
        std::memcpy(dt_pre.data(), xbc_host.data(), dt_rank * sizeof(float));
        std::memcpy(B.data(), xbc_host.data() + dt_rank, d_state * sizeof(float));
        std::memcpy(C.data(), xbc_host.data() + dt_rank + d_state,
                    d_state * sizeof(float));

        std::optional<array> dt_lin;
        if (g.quantized)
            dt_lin.emplace(add(quantized_matmul(
                             array(dt_pre.data(), {1, (int)dt_rank}),
                             *g.q_w_dt, *g.q_s_dt, *g.q_b_dt, true, 64, 4, "affine", stream),
                         *g.w_dt, stream));
        else
            dt_lin.emplace(add(matmul(*g.w_dt,
                                array(dt_pre.data(), {(int)dt_rank}, float32), stream),
                         *g.b_dt, stream));
        array dt_soft = log1p(exp(*dt_lin, stream), stream);
        read_f32(reshape(dt_soft, {(int)d_inner}, stream), dt);
    }

    // --- Host: selective scan (overlaps across threads) ---------------------
    const float* a_log = find_weight(blk, ".mixer.A_log", nullptr);
    const float* ssm_d = find_weight(blk, ".mixer.D", nullptr);
    float* h_ssm = ctx.ssm_state.data() +
                   desc.block_id * d_state * d_inner;
    for (int64_t i = 0; i < d_inner; ++i) {
        float yi = ssm_d[i] * x_act[i];
        for (int64_t s = 0; s < d_state; ++s) {
            const float a_neg = -a_log[s * d_inner + i];
            const float A_bar = std::exp(dt[i] * a_neg);
            const float B_bar = dt[i] * B[s];
            const float h_new = A_bar * h_ssm[s * d_inner + i] +
                                B_bar * x_act[i];
            h_ssm[s * d_inner + i] = h_new;
            yi += C[s] * h_new;
        }
        y_ssm[i] = yi;
    }

    // --- Host: silu(z) * y_ssm -> y_gated (overlaps) ------------------------
    for (int64_t i = 0; i < d_inner; ++i) {
        const float sz = z[i] / (1.0f + std::exp(-z[i]));
        y_gated[i] = sz * y_ssm[i];
    }

    // --- Phase C (GPU, locked): out_proj -> y_out, residual -----------------
    {
        std::lock_guard<std::mutex> lk(gpu_lock);
        const Device gpu = Device::gpu;
        std::optional<array> yo_arr;
        if (g.quantized)
            yo_arr.emplace(quantized_matmul(
                array(y_gated.data(), {1, (int)d_inner}),
                *g.q_w_out, *g.q_s_out, *g.q_b_out, true, 64, 4, "affine", stream));
        else
            yo_arr.emplace(matmul(*g.w_out,
                            array(y_gated.data(), {(int)d_inner}, float32), stream));
        read_f32(*yo_arr, y_out);
        for (int64_t i = 0; i < d_model; ++i)
            ctx.hidden_state[i] += y_out[i];
    }
}

std::size_t MlxEngine::run_layer_forward_batch(std::vector<RuntimeContext*>& ctxs,
                                               const LayerContext& layer) {
    if (!layer.desc || ctxs.empty()) return 0;
    const LayerDesc& desc = *layer.desc;

    // The layer weights are decoded + uploaded ONCE, shared read-only by all
    // batch threads via the single streamed-in layer buffer.
    if (!prepare_block(desc, layer.weights_base, layer.weights_size)) return 0;
    touch_block(desc.block_id);
    evict_if_needed();
    BlockCache& blk = blocks_[desc.block_id];

    if (desc.block_id >= (int)gpu_cache_->blocks.size())
        gpu_cache_->blocks.resize(desc.block_id + 1);
    auto& bg = gpu_cache_->blocks[desc.block_id];
    if (!bg) bg.emplace();
    MlxGpuCache::BlockGpu& g = *bg;

    // Build this block's GPU arrays on first touch (single-threaded, before
    // the workers start).
    if (!g.w_norm) {
        g.d_model = ctxs[0]->cfg.d_model;
        g.d_inner = ctxs[0]->cfg.d_inner;
        g.dt_rank = ctxs[0]->cfg.dt_rank;
        g.d_state = ctxs[0]->cfg.d_state;
        g.d_conv  = ctxs[0]->cfg.d_conv;

        const float* w_norm = find_weight(blk, ".norm.weight", nullptr);
        if (!w_norm) return 0;
        g.w_norm.emplace(array(w_norm, {(int)g.d_model}, float32));

        const uint32_t* in_u32 = find_u32(blk, ".mixer.in_proj.weight");
        if (in_u32) {
            g.quantized = true;
            g.q_w_in.emplace(array(find_u32(blk, ".mixer.in_proj.weight"),
                                   {(int)(2*g.d_inner), (int)(g.d_model/8)}, uint32));
            g.q_s_in.emplace(array(find_weight(blk, ".mixer.in_proj.scales"),
                                   {(int)(2*g.d_inner), (int)(g.d_model/64)}, float32));
            g.q_b_in.emplace(array(find_weight(blk, ".mixer.in_proj.biases"),
                                   {(int)(2*g.d_inner), (int)(g.d_model/64)}, float32));
            g.q_w_x.emplace(array(find_u32(blk, ".mixer.x_proj.weight"),
                                   {(int)(g.dt_rank+2*g.d_state), (int)(g.d_inner/8)}, uint32));
            g.q_s_x.emplace(array(find_weight(blk, ".mixer.x_proj.scales"),
                                   {(int)(g.dt_rank+2*g.d_state), (int)(g.d_inner/64)}, float32));
            g.q_b_x.emplace(array(find_weight(blk, ".mixer.x_proj.biases"),
                                   {(int)(g.dt_rank+2*g.d_state), (int)(g.d_inner/64)}, float32));
            g.q_w_dt.emplace(array(find_u32(blk, ".mixer.dt_proj.weight"),
                                   {(int)g.d_inner, (int)(g.dt_rank/8)}, uint32));
            g.q_s_dt.emplace(array(find_weight(blk, ".mixer.dt_proj.scales"),
                                   {(int)g.d_inner, (int)(g.dt_rank/64)}, float32));
            g.q_b_dt.emplace(array(find_weight(blk, ".mixer.dt_proj.biases"),
                                   {(int)g.d_inner, (int)(g.dt_rank/64)}, float32));
            g.w_dt.emplace(array(find_weight(blk, ".mixer.dt_proj.bias"),
                                   {(int)g.d_inner}, float32));
            g.q_w_out.emplace(array(find_u32(blk, ".mixer.out_proj.weight"),
                                    {(int)g.d_model, (int)(g.d_inner/8)}, uint32));
            g.q_s_out.emplace(array(find_weight(blk, ".mixer.out_proj.scales"),
                                    {(int)g.d_model, (int)(g.d_inner/64)}, float32));
            g.q_b_out.emplace(array(find_weight(blk, ".mixer.out_proj.biases"),
                                    {(int)g.d_model, (int)(g.d_inner/64)}, float32));
        } else {
            g.quantized = false;
            const float* w_in  = find_weight(blk, ".mixer.in_proj.weight", nullptr);
            const float* w_x   = find_weight(blk, ".mixer.x_proj.weight", nullptr);
            const float* w_dt  = find_weight(blk, ".mixer.dt_proj.weight", nullptr);
            const float* b_dt  = find_weight(blk, ".mixer.dt_proj.bias", nullptr);
            const float* w_out = find_weight(blk, ".mixer.out_proj.weight", nullptr);
            if (!w_in || !w_x || !w_dt || !b_dt || !w_out) return 0;
            g.w_in.emplace(array(w_in, {(int)(2*g.d_inner), (int)g.d_model}, float32));
            g.w_x.emplace(array(w_x, {(int)(g.dt_rank+2*g.d_state), (int)g.d_inner}, float32));
            g.w_dt.emplace(array(w_dt, {(int)g.d_inner, (int)g.dt_rank}, float32));
            g.b_dt.emplace(array(b_dt, {(int)g.d_inner}, float32));
            g.w_out.emplace(array(w_out, {(int)g.d_model, (int)g.d_inner}, float32));
        }
    }

    // Launch one worker per independent context. All share `blk` + `g`
    // (read-only), each owns its own hidden/ssm/conv state. The fast GPU
    // matmul evals are serialized under `gpu_lock` (MLX Metal eval is not
    // thread-safe, even for concurrent command-buffer access), so all workers
    // share the single default GPU stream; the host-side selective-scan /
    // conv / gating run unlocked and overlap across threads (the real Mamba
    // bottleneck). No per-thread streams: creating them concurrently races
    // with the global Metal device state.
    const size_t n = ctxs.size();
    std::mutex gpu_lock;
    Stream shared = default_stream(Device::gpu);
    if (n == 1) {
        compute_token(*ctxs[0], desc, blk, &g, shared, gpu_lock);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(n);
        for (auto* c : ctxs) {
            workers.emplace_back([this, c, &desc, &blk, &g, &gpu_lock, &shared]() {
                compute_token(*c, desc, blk, &g, shared, gpu_lock);
            });
        }
        for (auto& t : workers) t.join();
    }
    return layer.weights_size;
}

}  // namespace remora
