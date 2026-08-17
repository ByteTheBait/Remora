# PATCH_LLAMA.md — getting real Mamba tokens out of Remora

## Status (end of this turn)

We have a runnable placeholder that builds and links, with:

- **Sharder** (`scripts/shard_gguf.py`) producing `manifest.json` + per-block `blk.N.meta.json` with shape/dtype/offset/size for every parameter.
- **ggml kernel infrastructure** (`src/rnn_kernel.cpp`) — loads sidecars, builds ggml contexts per block, dispatches `ggml_mul_mat` on real Q4_K weights, runs a placeholder Mamba selective scan. Verified end-to-end that one matmul produces finite output on real Mamba-1.4B weights.
- **CMake** that links system ggml 0.17 + libllama via pkg-config + Homebrew prefix discovery.
- **A runnable main.cpp** that loads manifest + sidecars, walks blocks, and prints timing — but with a stubbed-out sampling path and hidden_state always zeroed.

The placeholder doesn't emit real text because:

1. Embedding table isn't loaded into ggml.
2. Sampling always returns token 0.
3. The selective scan math has subtler bugs (sign conventions, A_log negation) that won't show up until end-to-end numerics come through.

## The fix: patch llama.cpp + use it as the inference engine

The path forward is to **patch llama.cpp** to accept a per-tensor data callback, then drive it from a ~100-line C driver. The Mamba kernel is *already* in llama.cpp and has been validated on Mamba-1.4B by your existing `llama-cli` runs. We get correct numerics for free; the patch is just plumbing.

### What's already done in this turn

`third_party/llama.cpp/include/llama.h`:

```c
// REMORA PATCH: per-tensor data callback.
typedef const void * (*llama_tensor_data_callback)(
    const char * tensor_name, size_t offset, size_t size, void * user_data);

struct llama_model_params {
    // ... existing fields ...
    llama_tensor_data_callback tensor_data_callback;
    void * tensor_data_callback_user_data;
};
```

The rest of the patch is below.

## What needs to happen next

### 1. Patch `llama-model-loader.cpp` to use the callback

In `src/llama-model-loader.h`, extend the loader class:

```cpp
class llama_model_loader {
    // ... existing members ...
public:
    void * tensor_data_callback_user_data = nullptr;
    llama_tensor_data_callback tensor_data_callback = nullptr;
    // ...
};
```

In `src/llama-model-loader.cpp`, modify `load_data_for`:

```cpp
void llama_model_loader::load_data_for(struct ggml_tensor * cur) const {
    const auto & w = require_weight(ggml_get_name(cur));

    if (tensor_data_callback != nullptr) {
        // REMORA PATCH: defer to the per-tensor callback. The callback mmaps
        // the relevant shard, copies the bytes, and returns a pointer to a
        // buffer the loader will memcpy into cur->data.
        const void * src = tensor_data_callback(
            ggml_get_name(cur), w.offs, ggml_nbytes(cur),
            tensor_data_callback_user_data);
        if (cur->data == nullptr) {
            // Slow path: ggml allocated cur->data already. memcpy in.
            // (The loader has both paths; in mmap mode cur->data is NULL
            //  and we just set it; in !mmap mode it's pre-allocated.)
        } else {
            memcpy(cur->data, src, ggml_nbytes(cur));
        }
        return;
    }

    if (use_mmap) {
        const auto & mapping = mappings.at(w.idx);
        if (cur->data == nullptr) {
            cur->data = (uint8_t *)mapping->addr() + w.offs;
        } else {
            memcpy(cur->data, (uint8_t *)mapping->addr() + w.offs, ggml_nbytes(cur));
        }
    } else {
        // ... existing !mmap path ...
    }
}
```

The key insight: `cur->data` may be **NULL** at this point when `use_mmap = true` (libllama's default), in which case the callback is expected to return a pointer the loader will use directly. **Easiest implementation**: have the callback `mmap` the relevant shard, return `mmap_base + w.offs_in_shard`, and the loader will store that pointer in `cur->data`. libllama will then `madvise(MADV_DONTNEED)` or `munmap` later if the tensor ends up on CPU; in practice it doesn't, so the bytes stay paged in. That's fine for our purposes — the OS reclaims pages under memory pressure anyway.

For the *no-mmap* path (which is what we want to use, since mmap of the whole file would defeat the point), the callback returns a malloc'd buffer the loader copies from. After load_data_for returns, the callback can free the buffer. Easier and more portable than mmap-per-tensor.

### 2. Wire the callback through `llama_model_load_from_file`

In `src/llama.cpp`, find the `llama_model_load_from_file` function and find the line where `llama_model_loader` is constructed. Add:

```cpp
loader.tensor_data_callback = params.tensor_data_callback;
loader.tensor_data_callback_user_data = params.tensor_data_callback_user_data;
```

(You'll need to add the same two fields to `llama_model_loader` since they're new.)

### 3. Build llama.cpp as a static library, link into our driver

In the top-level `CMakeLists.txt`:

```cmake
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/llama.cpp)
target_link_libraries(remora_core PUBLIC llama ggml_static)
```

The llama.cpp build pulls its own ggml vendored copy (in `third_party/llama.cpp/ggml/`). When we link `remora_core` we already link system ggml via pkg-config; you'll need to choose one. Easiest: drop our pkg-config path and rely on llama.cpp's vendored ggml. Then `src/rnn_kernel.cpp` and `src/main.cpp` see the same ggml symbols as the rest of llama.cpp.

### 4. The driver (`src/main.cpp`)

Roughly 100 lines:

```cpp
#include <llama.h>

// Cached mmaps, one per shard.
struct ShardMap {
    int fd;
    void * base;
    size_t size;
};
static std::unordered_map<std::string, ShardMap> g_shards;
static std::unordered_map<std::string, std::pair<int, size_t>> g_tensor_locs;
// g_tensor_locs: tensor_name -> {shard_index, offset_within_shard}

// Callback: when libllama wants tensor bytes, look up the shard, mmap it
// (cached), return base + offset.
static const void * tensor_data_cb(const char * name, size_t offs, size_t sz, void * ud) {
    auto it = g_tensor_locs.find(name);
    if (it == g_tensor_locs.end()) return nullptr;
    auto & s = g_shards[shard_path_for(it->second.first)];
    if (s.base == nullptr) {
        s.fd = open(s.path.c_str(), O_RDONLY);
        s.base = mmap(nullptr, s.size, PROT_READ, MAP_PRIVATE, s.fd, 0);
    }
    return (const uint8_t *)s.base + it->second.second + offs;
}

int main(int argc, char** argv) {
    llama_backend_init();

    // Build g_shards + g_tensor_locs from manifest.json + each blk.N.meta.json.
    // (Your existing main.cpp already does this — just convert the parsed
    //  data into the two global maps above.)

    auto mparams = llama_model_default_params();
    mparams.tensor_data_callback = tensor_data_cb;
    mparams.use_extra_bufts = false;  // disable repacking
    auto model = llama_model_load_from_file(source_gguf_path, mparams);

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    auto ctx = llama_init_from_model(model, cparams);

    // Tokenize, prefill, generate.
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(512);
    int n = llama_tokenize(vocab, prompt, prompt_len, tokens.data(), 512, true, false);
    auto batch = llama_batch_get_one(tokens.data(), n);
    llama_decode(ctx, batch);

    auto smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    for (int i = 0; i < n_predict; ++i) {
        auto tok = llama_sampler_sample(smpl, ctx, -1);
        char piece[256];
        int n_piece = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, false);
        std::cout.write(piece, n_piece);
        auto next = llama_batch_get_one(&tok, 1);
        llama_decode(ctx, next);
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}
```

### 5. Shard-locator precomputation

The `g_tensor_locs` map needs `tensor_name -> {shard_index, offset_in_shard}`. Build it once at startup by iterating all `blk.N.meta.json` files. For each tensor in the sidecar, key the full name (`blk.0.ssm_in.weight`) to the index of the shard (`N`) and the sidecar's recorded `offset`. For global tensors (token_embd, output_norm) key them to a special shard index 0 (the global.bin shard).

This precomputation is straightforward — it's basically what `load_manifest` + `load_sidecar` already do in the current `main.cpp`, just re-purposed.

### 6. Use llama.cpp's existing tokenizer and sampler

The whole point of this approach: libllama already does tokenization (BPE) and sampling correctly for Mamba. Don't roll your own. The `llama_tokenize` + `llama_sampler_chain_add(greedy)` + `llama_sampler_sample` calls are the entire interface.

### 7. The benchmark

Once real tokens come out, run `scripts/bench_compare.sh` — that script already supports the `--layers layers/` flag, which we use to point at the sharded model. The wrapper has the llama-cli path and a Remora path; the Remora path will need to be the new llama-cli-wrapper binary that uses the patched llama.cpp.

## Files that need to change

| File | Change |
|------|--------|
| `third_party/llama.cpp/include/llama.h` | Already done (typedef + struct field). |
| `third_party/llama.cpp/src/llama-model-loader.h` | Add `tensor_data_callback` + `_user_data` fields to the loader class. |
| `third_party/llama.cpp/src/llama-model-loader.cpp` | Add the `if (tensor_data_callback)` branch in `load_data_for` (see §1). |
| `third_party/llama.cpp/src/llama.cpp` | In `llama_model_load_from_file`, find where the loader is constructed and copy the two fields. |
| `CMakeLists.txt` | `add_subdirectory(third_party/llama.cpp)`; link `llama` + `ggml_static`; drop our pkg-config ggml path. |
| `src/main.cpp` | Replace the placeholder forward pass with `llama_model_load_from_file` + `llama_decode` + `llama_sampler_sample`. |
| `src/rnn_kernel.cpp` | Delete (no longer needed — libllama does the Mamba math). |
| `src/tokenizer.cpp` | Delete (no longer needed — libllama does the BPE). |

## Estimated work

- Patches: 30 minutes.
- CMake integration: 30 minutes.
- Driver: 30 minutes.
- Debugging build issues: 30 minutes.
- **Total: 2 hours of focused work.**

After that, you have a real binary that produces real Mamba tokens from sharded layers, with peak RSS that we can compare against monolithic `llama-cli` to validate the streaming claim.