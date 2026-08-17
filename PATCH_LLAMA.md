# PATCH_LLAMA.md — getting real Mamba tokens out of Remora

## Status (end of this turn)

The plan in this document is now obsolete in important ways. The actual
implementation that worked is **much simpler** than the design below
describes. Keep this file for historical context, but read the new section
"Final design" first.

## Final design (use this)

The vendored llama.cpp b10180 already exposes an `llama_model_init_from_user`
API that does exactly what we need: it reads GGUF metadata (header only) from
a `gguf_context *`, then calls a per-tensor `set_tensor_data` callback for
each weight. The callback is responsible for filling `tensor->data` from
whatever source we want. **No patches to llama.cpp are required.**

### Pieces

1. **Sharder** (`scripts/shard_gguf.py`) emits per-block sidecars
   `layers/blk.N.meta.json` and `layers/global.bin.meta.json`. Each tensor
   entry now records BOTH:
   - `offset` — shard-relative byte offset (for tools that read shards
     directly)
   - `file_offset` — absolute byte offset within the original GGUF's
     tensor data section (i.e., what `gguf_get_tensor_offset()` returns).
     This is what the libllama callback uses.

   The sharder also writes `manifest.json` (architecture, hyperparams,
   tokenizer) but that's not used by the libllama path — libllama parses
   all of that from the GGUF header itself.

2. **Driver** (`src/main_llama.cpp`) is the new libllama-based binary:
   - `gguf_init_from_file("mamba-1.4b-hf.Q4_K_M.gguf",
       {.no_alloc = true, .ctx = nullptr})` to read only the header.
   - `gguf_init_from_file` with `no_alloc=true` leaves `tensor->data == NULL`
     on every tensor in the metadata. **Critical:** this is what makes the
     callback path get invoked at all (otherwise libllama would mmap the
     file and ignore our callback).
   - `llama_model_init_from_user(gguf_ctx, set_tensor_data_cb, &cb_ud,
       params)` — libllama builds the model graph, allocating backend
     buffers for every weight. For each weight it calls our callback.
   - Our `set_tensor_data_cb(struct ggml_tensor *t, void *ud)` looks up
     `t->name` in a precomputed map (built at startup from the sidecars
     and the original GGUF) → resolves to `(shard_path, shard_offset)` →
     `mmap`s the shard (cached) → `memcpy`s `ggml_nbytes(t)` bytes from
     `shard_base + shard_offset` into `t->data`.
   - `llama_init_from_model(...)` then `llama_decode(...)` per token.
   - `llama_tokenize`, `llama_sampler_*`, `llama_token_to_piece` give us
     real text out, just like `llama-cli`.

3. **CMake** links the system ggml 0.17 via `pkg-config` (NOT the vendored
   ggml — `add_subdirectory(third_party/llama.cpp)` pulls its own ggml
   symbols, and we only need libllama). `target_link_libraries(remora
   PRIVATE llama ggml::ggml)`.

### Why this is simpler than the original plan

The original plan was to add a new `tensor_data_callback` field to
`llama_model_params` and patch `llama-model-loader.cpp` to use it. But
`llama_model_init_from_user` already exists and gives us the same hook
through `llama_model_loader::set_tensor_data`. The only difference is
the entry point (`init_from_user` instead of `load_from_file`), and that
we pass `no_alloc=true` to skip the file mmap. **Zero patches to
llama.cpp source.** The vendored copy is unmodified except for
`scripts/shard_gguf.py` (our own script) and `.clangd` (so clangd finds
the vendored ggml when indexing `llama.h`).

### Edge cases / known issues

- The original GGUF must be present on disk during model load (used only
  for header parsing — its tensor data section is never read).
- `llama_model_init_from_user` requires `gguf_init_from_file` to have run
  first; libllama takes ownership of the `gguf_context *` and frees it
  after model creation.
- All `set_tensor_data` calls happen during `llama_model_init_from_user`,
  so we mmap every shard up-front. For Mamba-1.4B that's 48 blocks of
  ~15 MB each = ~720 MB resident, plus the 80 MB embedding. Acceptable
  for the benchmark; streaming/eviction is a separate concern.
- The vendored ggml's `init_from_user` path **does not** trigger
  Metal/CPU backend writes — libllama handles allocation and we just
  memcpy into the pre-allocated buffer. Performance is identical to the
  mmap path.

---

## Historical design (no longer applicable, kept for context)

> *The plan below was written before we discovered `llama_model_init_from_user`.
> The "tensor_data_callback" typedef and `llama_model_params` fields it
> describes have been removed from `llama.h` (revert commit). All the
> weight-loading work is now done via the existing public API.*

The original plan was to patch `third_party/llama.cpp/include/llama.h`
to add a `llama_tensor_data_callback` typedef and `tensor_data_callback`
field on `llama_model_params`, then plumb them through
`llama_model_loader` to a new branch in `load_data_for`. We then
intended to call `llama_model_load_from_file` with the callback set
and pass the (mostly empty) original GGUF path.

The problem with that design is that `llama_model_load_from_file` does
its own `gguf_init_from_file` and an `init_mappings()` (file mmap)
regardless. To use the callback path you have to override
`load_data_for` *and* disable the file mapping — which means
`init_from_user` is cleaner because it has `LLAMA_LOAD_MODE_NONE` and
`use_extra_bufts=false` baked in.

So the new design uses `init_from_user` directly, with
`gguf_init_from_file` called first to get a `gguf_context *`. The
existing `set_tensor_data` field on `llama_model_loader` (a callback
the loader invokes for every tensor when `files.empty()`) is our hook
into the weight-loading path.

No source patches to llama.cpp. No header changes. No typedefs to
register. The patch commit was reverted.
