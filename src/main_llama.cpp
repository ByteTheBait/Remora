// Remora libllama driver.
//
// Drives the vendored llama.cpp's `llama_model_init_from_user` API with a
// per-tensor data callback that resolves each tensor to a sharded layer
// file. This gets us real Mamba inference (with libllama's correct
// numerics, tokenizer, and sampler) while reading weights from the
// sharded layout that `scripts/shard_gguf.py` produces — so peak RSS
// can be measured against the monolithic `llama-cli` baseline.
//
// The flow:
//
//   1. Parse `layers/manifest.json` to confirm we're looking at a Mamba
//      model and to learn the n_layer count (used at startup only —
//      libllama re-derives everything from the GGUF header).
//
//   2. Walk every `layers/blk.N.meta.json` and `layers/global.bin.meta.json`
//      to build a `tensor_name -> (shard_path, file_offset)` map. The
//      `file_offset` is the absolute byte offset within the original
//      GGUF's tensor data section (matches what `gguf_get_tensor_offset`
//      returns).
//
//   3. `gguf_init_from_file("mamba-1.4b-hf.Q4_K_M.gguf", {.no_alloc=true,
//       .ctx=nullptr})` to read the GGUF header. The original GGUF stays
//      on disk; its tensor data section is never read.
//
//   4. `llama_model_init_from_user(gguf_ctx, set_tensor_data_cb, &ud, params)`
//      builds the model and calls our callback for every weight. The
//      callback mmaps the relevant shard (cached) and `memcpy`s bytes
//      into `tensor->data` using `ggml_backend_tensor_set`.
//
//   5. `llama_init_from_model(...)` + `llama_decode(...)` per token.
//      `llama_tokenize` + `llama_sampler_*` + `llama_token_to_piece` give
//      real text out, just like `llama-cli`.
//
// Usage:
//   remora_llama --gguf <model.gguf> --layers <dir> \
//                [--prompt "..."] [--tokens N] [--ctx N]

#include <llama.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unordered_map>
#include <unistd.h>
#include <vector>

// ---------------------------------------------------------------------------
// Tiny JSON parser sufficient for the sharder sidecars.
//
// We only need to read flat string->value maps and arrays; the sharder
// writes a tiny manifest + per-block sidecars and that's all we consume.
// This is not a general JSON parser — if the sharder changes format this
// breaks. We deliberately keep it dependency-free.
// ---------------------------------------------------------------------------

namespace remora_json {

// Skip whitespace and a single optional value (used by arrays).
struct Cursor {
    const char* p;
    const char* end;
    void skip_ws() { while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p; }
    bool match(char c) { skip_ws(); if (p < end && *p == c) { ++p; return true; } return false; }
    bool peek(char c)   { skip_ws(); return p < end && *p == c; }
};

// Read a JSON string starting at *p (which must point at the opening
// quote). Advances *p past the closing quote. Returns the unescaped
// string (UTF-8 bytes are passed through unchanged).
static std::string read_string(Cursor& c) {
    c.skip_ws();
    if (c.p >= c.end || *c.p != '"') return {};
    ++c.p;
    std::string out;
    while (c.p < c.end && *c.p != '"') {
        if (*c.p == '\\' && c.p + 1 < c.end) {
            char esc = c.p[1];
            ++c.p;
            switch (esc) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/');  break;
                case 'n':  out.push_back('\n'); break;
                case 't':  out.push_back('\t'); break;
                case 'r':  out.push_back('\r'); break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                default:   out.push_back(esc);  break;  // tolerate unknown escapes
            }
            ++c.p;
        } else {
            out.push_back(*c.p++);
        }
    }
    if (c.p < c.end && *c.p == '"') ++c.p;
    return out;
}

// Read a JSON number (int or float). Returns as double for simplicity.
static double read_number(Cursor& c) {
    c.skip_ws();
    const char* start = c.p;
    if (c.p < c.end && (*c.p == '-' || *c.p == '+')) ++c.p;
    while (c.p < c.end && (std::isdigit(static_cast<unsigned char>(*c.p)) ||
                           *c.p == '.' || *c.p == 'e' || *c.p == 'E' ||
                           *c.p == '-' || *c.p == '+')) ++c.p;
    return std::strtod(std::string(start, c.p).c_str(), nullptr);
}

// Read a value, returning the string if it's a string, or the number as
// a string in the out string. Used for fields we just want to log.
static bool read_string_or_number_into(Cursor& c, std::string& out) {
    c.skip_ws();
    if (c.p >= c.end) return false;
    if (*c.p == '"') { out = read_string(c); return true; }
    if (*c.p == '-' || std::isdigit(static_cast<unsigned char>(*c.p))) {
        char buf[64];
        double v = read_number(c);
        std::snprintf(buf, sizeof(buf), "%g", v);
        out = buf;
        return true;
    }
    if (c.p + 4 <= c.end && std::strncmp(c.p, "true", 4) == 0) { c.p += 4; out = "true";  return true; }
    if (c.p + 5 <= c.end && std::strncmp(c.p, "false", 5) == 0) { c.p += 5; out = "false"; return true; }
    if (c.p + 4 <= c.end && std::strncmp(c.p, "null", 4) == 0) { c.p += 4; out = "null";  return true; }
    return false;
}

// Read an array of strings (skips non-string elements). Currently unused
// but kept for future use.
#if 0
static std::vector<std::string> read_string_array(Cursor& c) {
    std::vector<std::string> out;
    c.skip_ws();
    if (c.p >= c.end || *c.p != '[') return out;
    ++c.p;
    while (true) {
        c.skip_ws();
        if (c.p >= c.end) break;
        if (*c.p == ']') { ++c.p; break; }
        if (*c.p == ',') { ++c.p; continue; }
        out.push_back(read_string(c));
    }
    return out;
}
#endif

}  // namespace remora_json

// ---------------------------------------------------------------------------
// Shard index: name -> (shard_path, file_offset, size).
// ---------------------------------------------------------------------------

struct TensorLoc {
    std::string shard_path;
    uint64_t    shard_offset = 0;  // byte offset within the shard
    uint64_t    size         = 0;  // tensor size in bytes
};

struct ShardMmap {
    int         fd = -1;
    void*       base = nullptr;
    uint64_t    size = 0;
};

// Userdata handed to the libllama callback. We index into the global map
// keyed by tensor name; on first access we open + mmap the shard.
struct CallbackState {
    std::unordered_map<std::string, TensorLoc> tensor_locs;
    std::unordered_map<std::string, ShardMmap> shards;
    // shard_path -> file_offset of the first tensor placed in that shard.
    // Used to translate absolute file_offset to shard-relative offset.
    std::unordered_map<std::string, uint64_t> shard_bases;
    uint64_t total_bytes_moved = 0;
    uint64_t total_tensors = 0;
};

// ---------------------------------------------------------------------------
// Sidecar readers.
// ---------------------------------------------------------------------------

static bool parse_sidecar(const std::string& path, const std::string& shard_path,
                          CallbackState& state) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "remora: cannot open %s: %s\n", path.c_str(), std::strerror(errno));
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz <= 0) { std::fclose(f); return false; }
    std::string buf(static_cast<size_t>(sz), '\0');
    if (std::fread(buf.data(), 1, sz, f) != static_cast<size_t>(sz)) {
        std::fclose(f); return false;
    }
    std::fclose(f);

    remora_json::Cursor c{buf.data(), buf.data() + buf.size()};
    if (!c.match('{')) return false;

    while (true) {
        c.skip_ws();
        if (c.p >= c.end) break;
        if (*c.p == '}') { ++c.p; break; }
        if (*c.p == ',') { ++c.p; continue; }
        std::string key = remora_json::read_string(c);
        if (!c.match(':')) return false;
        if (key == "tensors") {
            c.skip_ws();
            if (c.p >= c.end || *c.p != '[') return false;
            ++c.p;
            while (true) {
                c.skip_ws();
                if (c.p >= c.end) break;
                if (*c.p == ']') { ++c.p; break; }
                if (*c.p == ',') { ++c.p; continue; }
                if (!c.match('{')) return false;
                std::string current_name;
                TensorLoc   current_loc;
                current_loc.shard_path = shard_path;
                while (true) {
                    c.skip_ws();
                    if (c.p >= c.end) break;
                    if (*c.p == '}') { ++c.p; break; }
                    if (*c.p == ',') { ++c.p; continue; }
                    std::string field = remora_json::read_string(c);
                    if (!c.match(':')) return false;
                    if (field == "name") {
                        current_name = remora_json::read_string(c);
                    } else if (field == "offset") {
                        // In-shard offset (what we want for byte delivery).
                        current_loc.shard_offset = (uint64_t) remora_json::read_number(c);
                    } else if (field == "file_offset") {
                        // Absolute offset in the original GGUF; we ignore
                        // this in favour of `offset` (the sharder wrote
                        // tensors in some order that doesn't match GGUF
                        // order, so file_offset is not a simple offset +
                        // cum-size sequence).
                        (void) remora_json::read_number(c);
                    } else if (field == "size") {
                        current_loc.size = (uint64_t) remora_json::read_number(c);
                    } else if (field == "dtype" || field == "shape") {
                        // `dtype` is a string, `shape` is an array. Both
                        // are ignored by the loader, so handle each.
                        std::string tmp;
                        if (!remora_json::read_string_or_number_into(c, tmp)) {
                            // Likely an array; skip one level of brackets.
                            if (c.peek('[')) {
                                int depth = 0;
                                do {
                                    if (*c.p == '[') ++depth;
                                    else if (*c.p == ']') --depth;
                                    ++c.p;
                                } while (c.p < c.end && depth > 0);
                            } else break;
                        }
                    } else if (field == "n_elements") {
                        (void) remora_json::read_number(c);
                    } else {
                        // skip unknown value
                        std::string tmp;
                        if (!remora_json::read_string_or_number_into(c, tmp)) {
                            // array? skip one level
                            if (c.peek('[')) {
                                int depth = 0;
                                do {
                                    if (*c.p == '[') ++depth;
                                    else if (*c.p == ']') --depth;
                                    ++c.p;
                                } while (c.p < c.end && depth > 0);
                            } else break;
                        }
                    }
                }
                if (!current_name.empty()) {
                    state.tensor_locs[current_name] = current_loc;
                }
            }
        } else {
            std::string tmp;
            (void) remora_json::read_string_or_number_into(c, tmp);
        }
    }
    return true;
}

// Forward declaration: get_shard_base_offset is defined further below.
static uint64_t get_shard_base_offset(const std::string& meta_path);

static bool load_index(const std::string& layers_dir, CallbackState& state) {
    // Per-block sidecars.
    for (int i = 0; ; ++i) {
        std::string meta = layers_dir + "/blk." + std::to_string(i) + ".meta.json";
        std::string shard = layers_dir + "/blk." + std::to_string(i);
        struct stat st;
        if (stat(shard.c_str(), &st) != 0) break;
        if (!parse_sidecar(meta, shard, state)) {
            std::fprintf(stderr, "remora: failed to parse %s\n", meta.c_str());
            return false;
        }
        // Record the absolute file_offset of the first tensor placed in
        // this shard. The sharder writes tensors in gguf order, so the
        // first entry's file_offset is the minimum file_offset in this
        // shard. We re-parse the sidecar just to find it.
        state.shard_bases[shard] = get_shard_base_offset(meta);
    }
    // Global sidecar.
    std::string meta = layers_dir + "/global.bin.meta.json";
    std::string shard = layers_dir + "/global.bin";
    struct stat st;
    if (stat(shard.c_str(), &st) == 0) {
        if (!parse_sidecar(meta, shard, state)) {
            std::fprintf(stderr, "remora: failed to parse %s\n", meta.c_str());
            return false;
        }
        state.shard_bases[shard] = get_shard_base_offset(meta);
    }
    std::fprintf(stderr, "remora: indexed %zu tensors across %zu shards\n",
                 state.tensor_locs.size(), state.shard_bases.size());
    return !state.tensor_locs.empty();
}

// ---------------------------------------------------------------------------
// libllama callback: fill tensor->data from the appropriate shard.
// ---------------------------------------------------------------------------
//
// Design note: the sharder writes per-block shard files (blk.N, global.bin)
// whose contents are the per-block tensor data concatenated. The sidecar
// entries record both `offset` (shard-relative) and `file_offset` (absolute
// offset in the original GGUF's tensor data section).
//
// For the libllama callback, we want **shard-relative** offsets. The
// translation is: shard_offset = file_offset - (file_offset of the first
// tensor assigned to that shard). The first tensor's file_offset is the
// offset of the block's data section in the original GGUF.
//
// We compute this translation at index-build time and stash the result in
// `TensorLoc::size` repurpose as `shard_offset` — but to keep the type
// stable, we store both: `file_offset` (absolute) plus compute the delta at
// lookup time.
//
// Simpler approach: store the file_offset of the FIRST tensor in each
// shard at index build time, then subtract.

static ShardMmap* get_or_mmap_shard(CallbackState& state, const std::string& path) {
    auto it = state.shards.find(path);
    if (it != state.shards.end()) return &it->second;
    ShardMmap sm;
    sm.fd = ::open(path.c_str(), O_RDONLY);
    if (sm.fd < 0) {
        std::fprintf(stderr, "remora: cannot open %s: %s\n", path.c_str(), std::strerror(errno));
        return nullptr;
    }
    struct stat st;
    if (::fstat(sm.fd, &st) != 0) {
        std::fprintf(stderr, "remora: fstat %s failed: %s\n", path.c_str(), std::strerror(errno));
        ::close(sm.fd); return nullptr;
    }
    sm.size = (uint64_t) st.st_size;
    sm.base = ::mmap(nullptr, sm.size, PROT_READ, MAP_PRIVATE, sm.fd, 0);
    if (sm.base == MAP_FAILED) {
        std::fprintf(stderr, "remora: mmap %s failed: %s\n", path.c_str(), std::strerror(errno));
        sm.base = nullptr; ::close(sm.fd); return nullptr;
    }
    // Advise sequential read; the tensor is read once and then never again.
    ::madvise(sm.base, sm.size, MADV_SEQUENTIAL);
    auto [ins, _] = state.shards.emplace(path, sm);
    return &ins->second;
}

// Returns the file_offset of the first tensor placed in the given shard.
// We track this at index build time by looking at the sidecar's first
// tensor entry.
static uint64_t get_shard_base_offset(const std::string& meta_path) {
    FILE* f = std::fopen(meta_path.c_str(), "rb");
    if (!f) return 0;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz <= 0) { std::fclose(f); return 0; }
    std::string buf(static_cast<size_t>(sz), '\0');
    std::fread(buf.data(), 1, sz, f);
    std::fclose(f);
    // The tensors are sorted by absolute file_offset (because the sharder
    // reads them in gguf order). The first tensor's file_offset is the
    // minimum one for this shard.
    remora_json::Cursor c{buf.data(), buf.data() + buf.size()};
    if (!c.match('{')) return 0;
    // We don't need to parse the whole thing; just scan for the first
    // "file_offset" or "offset" value. For simplicity, we'll re-parse.
    // (The sidecar format is small enough that this is fine.)
    uint64_t min_off = UINT64_MAX;
    bool seen_any = false;
    while (true) {
        c.skip_ws();
        if (c.p >= c.end) break;
        if (*c.p == '}') break;
        if (*c.p == ',') { ++c.p; continue; }
        std::string key = remora_json::read_string(c);
        if (!c.match(':')) break;
        if (key == "tensors") {
            c.skip_ws();
            if (c.p >= c.end || *c.p != '[') return 0;
            ++c.p;
            while (true) {
                c.skip_ws();
                if (c.p >= c.end) break;
                if (*c.p == ']') { ++c.p; break; }
                if (*c.p == ',') { ++c.p; continue; }
                if (!c.match('{')) return 0;
                uint64_t this_off = 0;
                while (true) {
                    c.skip_ws();
                    if (c.p >= c.end) break;
                    if (*c.p == '}') { ++c.p; break; }
                    if (*c.p == ',') { ++c.p; continue; }
                    std::string field = remora_json::read_string(c);
                    if (!c.match(':')) break;
                    if (field == "file_offset" || field == "offset") {
                        this_off = (uint64_t) remora_json::read_number(c);
                    } else {
                        std::string tmp;
                        if (!remora_json::read_string_or_number_into(c, tmp)) {
                            if (c.peek('[')) {
                                int depth = 0;
                                do { if (*c.p == '[') ++depth; else if (*c.p == ']') --depth; ++c.p; }
                                while (c.p < c.end && depth > 0);
                            } else break;
                        }
                    }
                }
                if (seen_any == false || this_off < min_off) {
                    min_off = this_off;
                    seen_any = true;
                }
            }
        } else {
            std::string tmp;
            if (!remora_json::read_string_or_number_into(c, tmp)) {
                if (c.peek('[')) {
                    int depth = 0;
                    do { if (*c.p == '[') ++depth; else if (*c.p == ']') --depth; ++c.p; }
                    while (c.p < c.end && depth > 0);
                } else break;
            }
        }
    }
    return seen_any ? min_off : 0;
}

static void set_tensor_data_cb(struct ggml_tensor* tensor, void* userdata) {
    auto* state = static_cast<CallbackState*>(userdata);
    if (tensor == nullptr) return;
    const char* name = ggml_get_name(tensor);
    if (name == nullptr) name = "";
    auto it = state->tensor_locs.find(name);
    if (it == state->tensor_locs.end()) {
        // Optional tensor (e.g. blk.0.ssm_in.scale) that doesn't exist in
        // the source GGUF. The loader allocated it speculatively with
        // TENSOR_NOT_REQUIRED; leave its data zeroed.
        if (state->total_tensors < 8) {
            std::fprintf(stderr, "remora: skipping optional tensor '%s'\n", name);
        }
        return;
    }
    const TensorLoc& loc = it->second;
    ShardMmap* sm = get_or_mmap_shard(*state, loc.shard_path);
    if (sm == nullptr || sm->base == nullptr) return;
    const uint64_t nbytes = ggml_nbytes(tensor);
    // The sharder writes each tensor's sidecar entry with the byte offset
    // within the shard (`offset` field), so we can read directly from
    // that offset without any translation.
    const uint64_t shard_offset = loc.shard_offset;
    if (shard_offset + nbytes > sm->size) {
        std::fprintf(stderr, "remora: tensor '%s' overflows shard (shard_off=%llu size=%llu shard=%llu)\n",
                     name, (unsigned long long) shard_offset,
                     (unsigned long long) nbytes, (unsigned long long) sm->size);
        return;
    }
    const void* src = static_cast<const uint8_t*>(sm->base) + shard_offset;
    ggml_backend_tensor_set(tensor, src, 0, nbytes);
    state->total_bytes_moved += nbytes;
    state->total_tensors += 1;
}

// Unmap + close every shard we still have in `state.shards`. After all
// `set_tensor_data_cb` calls have returned, libllama has copied each
// tensor's bytes into a backend buffer (CPU host pointer or Metal
// shared pointer); the source mmap is dead weight in our address space.
// Freeing it is the whole point of sharding — peak RSS drops from
// ~730 MB resident (the shards) + ~1 GB (backend buffers) down to just
// the backend buffers.
//
// We use `madvise(MADV_DONTNEED)` rather than `munmap` for two reasons:
//   1. The kernel keeps the VMA alive but drops the resident pages;
//      a future page fault would re-fault from disk (we don't expect any
//      unless libllama re-reads weights for some reason).
//   2. The fd is kept open until `~ShardMmap` runs, so a re-fault is
//      cheap. We follow up with `munmap` + `close` to release the VMA
//      and the fd entirely.
static void release_shards(CallbackState& state) {
    uint64_t total_released = 0;
    for (auto& [path, sm] : state.shards) {
        if (sm.base != nullptr && sm.base != MAP_FAILED) {
            ::madvise(sm.base, sm.size, MADV_DONTNEED);
            ::munmap(sm.base, sm.size);
            total_released += sm.size;
        }
        if (sm.fd >= 0) {
            ::close(sm.fd);
        }
        sm.base = nullptr;
        sm.fd = -1;
    }
    state.shards.clear();
    std::fprintf(stderr, "remora: released %llu MB of shard mmap after load\n",
                 (unsigned long long) (total_released / (1024 * 1024)));
}

// ---------------------------------------------------------------------------
// CLI + main.
// ---------------------------------------------------------------------------

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Remora libllama driver — runs Mamba through sharded layer files.\n"
        "\n"
        "Usage:\n"
        "  %s --gguf <model.gguf> --layers <dir> [options]\n"
        "\n"
        "Required:\n"
        "  --gguf <file>     Path to the original GGUF (used for header only).\n"
        "  --layers <dir>    Directory containing blk.N(.meta.json) and\n"
        "                    global.bin(.meta.json) produced by shard_gguf.py.\n"
        "\n"
        "Options:\n"
        "  --prompt <text>   Prompt to feed the model (default: \"Hello, my name is\").\n"
        "  --tokens  <N>     Tokens to generate after the prompt (default: 32).\n"
        "  --ctx <N>         Context size (default: 512).\n"
        "  --n-gpu-layers N  Layers to offload to GPU (default: 0).\n"
        "  --seed <N>        Sampler seed (default: 0 = time-based).\n",
        prog);
}

static std::string arg_value(int argc, char** argv, const std::string& flag, const std::string& def = {}) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (flag == argv[i]) return argv[i + 1];
    }
    return def;
}

static int arg_value_int(int argc, char** argv, const std::string& flag, int def) {
    std::string v = arg_value(argc, argv, flag, "");
    if (v.empty()) return def;
    try { return std::stoi(v); } catch (...) { return def; }
}

int main(int argc, char** argv) try {
    if (argc < 5) { print_usage(argv[0]); return 1; }
    std::string gguf_path       = arg_value(argc, argv, "--gguf");
    std::string layers_dir      = arg_value(argc, argv, "--layers");
    std::string prompt          = arg_value(argc, argv, "--prompt", "Hello, my name is");
    int n_predict               = arg_value_int(argc, argv, "--tokens", 32);
    int n_ctx                   = arg_value_int(argc, argv, "--ctx", 512);
    int n_gpu_layers            = arg_value_int(argc, argv, "--n-gpu-layers", 0);
    // --seed is accepted for CLI parity with llama-cli/llama-bench. The
    // greedy sampler is deterministic and the dist sampler would consume
    // it; we currently use greedy only.
    uint32_t seed               = (uint32_t) arg_value_int(argc, argv, "--seed", 0);
    (void) seed;

    if (gguf_path.empty() || layers_dir.empty()) {
        print_usage(argv[0]); return 1;
    }

    // 1. Build the shard index.
    CallbackState state;
    if (!load_index(layers_dir, state)) {
        std::fprintf(stderr, "remora: failed to load shard index from %s\n", layers_dir.c_str());
        return 1;
    }

    // 2. Init llama + backends.
    llama_backend_init();
    ggml_backend_load_all();  // ensures CPU/Metal/BLAS are visible

    // 3. Open the GGUF header (no_alloc=true: skip the giant data malloc).
    struct gguf_init_params gguf_params = {};
    gguf_params.no_alloc = true;
    gguf_params.ctx      = nullptr;
    struct gguf_context* gguf_ctx = gguf_init_from_file(gguf_path.c_str(), gguf_params);
    if (!gguf_ctx) {
        std::fprintf(stderr, "remora: gguf_init_from_file(%s) failed\n", gguf_path.c_str());
        return 1;
    }

    // 4. Build the model via per-tensor callback.
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    mparams.use_extra_bufts = false;
    auto* model = llama_model_init_from_user(gguf_ctx, set_tensor_data_cb, &state, mparams);
    if (!model) {
        std::fprintf(stderr, "remora: llama_model_init_from_user failed\n");
        return 1;
    }
    std::fprintf(stderr, "remora: loaded %llu tensors (%llu MB) from %zu shards\n",
                 (unsigned long long) state.total_tensors,
                 (unsigned long long) (state.total_bytes_moved / (1024 * 1024)),
                 state.shards.size());

    // Each weight has now been copied into a backend buffer (CPU host ptr
    // or Metal shared memory). The source mmaps are dead weight — drop
    // them so peak RSS reflects the libllama backend, not the shards we
    // streamed from.
    release_shards(state);

    // 5. Build the context.
    auto cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) n_ctx;
    cparams.n_threads = 0;          // default to system thread count
    cparams.n_threads_batch = 0;
    auto* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "remora: llama_init_from_model failed\n");
        return 1;
    }

    // 6. Tokenize the prompt.
    const llama_vocab* vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(n_ctx);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                                  tokens.data(), (int32_t) tokens.size(), true, false);
    if (n_tokens < 0) {
        std::fprintf(stderr, "remora: tokenize failed (%d)\n", n_tokens);
        return 1;
    }
    std::fprintf(stderr, "remora: prompt=%d tokens\n", n_tokens);

    // 7. Prefill.
    auto batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "remora: prefill decode failed\n");
        return 1;
    }

    // 8. Build a sampler chain.
    // Mamba's GGUF declares eos_token_id=0 (which is also BOS), so greedy
    // sampling tends to predict 0 forever. Use a small temperature to
    // escape the EOS trap.
    auto smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

    // 9. Generate tokens.
    auto t0 = std::chrono::steady_clock::now();
    std::string out;
    out.reserve(n_predict * 8);
    int n_emitted = 0;
    for (int i = 0; i < n_predict * 4 && n_emitted < n_predict; ++i) {
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        // Mamba's GGUF declares eos_token_id=0 (which is also BOS), so
        // greedy sampling tends to predict 0 forever. We use a small
        // temperature + dist sampler (configured above) to escape the
        // EOS trap. The EOG check is still useful to break out of long
        // runs that happen to land on a real EOS, so we keep it.
        if (n_emitted > 0 && llama_vocab_is_eog(vocab, id)) break;
        char piece[64];
        int n_piece = llama_token_to_piece(vocab, id, piece, sizeof(piece), 0, false);
        if (n_piece > 0) {
            out.append(piece, n_piece);
            n_emitted += 1;
        }
        auto next = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx, next) != 0) {
            std::fprintf(stderr, "remora: decode failed at token %d\n", i);
            break;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::fprintf(stderr, "remora: %.3fs for %d tokens (%.2f tok/s)\n",
                 secs, n_predict, secs > 0 ? n_predict / secs : 0.0);

    // 10. Emit the output.
    std::fwrite(out.data(), 1, out.size(), stdout);
    std::fflush(stdout);

    // 11. Cleanup.
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "remora: fatal: %s\n", e.what());
    return 2;
}
