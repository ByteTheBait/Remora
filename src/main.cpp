// Remora - Layer-by-Layer RNN MoE on consumer hardware.
//
// Drives the layer-streamer and dispatches each sharded block to the
// Mamba-1 selective-scan kernel. The forward pass walks every block in
// order, streaming weights from disk and reusing per-layer SSM/conv state.
//
// Usage:
//   remora --layers layers/ [--prompt "..."] [--tokens N] [--temperature T]
//
//   --layers <dir>   directory containing sharded layer files (blk.0, ...)
//                    plus manifest.json and global.bin
//   --prompt <text>  text to feed into the model (default: "Hello, my name is")
//   --tokens  <N>    number of tokens to generate after the prompt
//   --temperature T  sampling temperature (0 = greedy; default 0)
//   --routing <f>    JSON routing table (kept for back-compat with the old
//                    placeholder CLI; the router isn't used by the real
//                    kernel path)

#include "layer_streamer.h"
#include "rnn_kernel.h"
#include "tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#ifdef REMORA_HAVE_GGML
#include <ggml.h>
#include <gguf.h>
#include <ggml-backend.h>
#endif

namespace fs = std::filesystem;

namespace {

void print_usage(const char* prog) {
    std::printf(
        "Remora - Layer-by-Layer RNN MoE on consumer hardware\n"
        "\n"
        "Usage:\n"
        "  %s --layers <dir> [--prompt <text>] [--tokens N]\n"
        "                       [--temperature T] [--buffer-layers N] [--routing <f>]\n"
        "\n"
        "Options:\n"
        "  --layers <dir>    Directory containing sharded layer files\n"
        "                    (blk.0, blk.1, ...), manifest.json, global.bin.\n"
        "  --prompt <text>   Prompt text to feed the model.\n"
        "                    Default: \"Hello, my name is\"\n"
        "  --tokens  <N>     Tokens to generate after the prompt (default 32).\n"
        "  --temperature T   Sampling temperature, 0 = greedy (default 0.7).\n"
        "  --buffer-layers N Layers to keep mmap'd ahead of the running layer\n"
        "                    as streaming lookahead. Only the running layer + N\n"
        "                    are resident at any moment. Default: 1.\n"
        "  --routing <f>     Routing table (unused by the ggml kernel; kept\n"
        "                    for back-compat with the placeholder CLI).\n",
        prog);
}

std::string arg_value(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (flag == argv[i]) return argv[i + 1];
    }
    return {};
}

bool has_flag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (flag == argv[i]) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Minimal JSON reader (sufficient for the schema we emit).
//
// We only need to read flat string->value maps and arrays of strings; the
// sharder writes a tiny manifest + per-block sidecars and that's all we
// consume. nlohmann::json would be nicer but adds a dependency.
//
// This parser handles:
//   - "key": <number>, true/false/null
//   - "key": "string"        (with \" and \\ escapes)
//   - "key": [ "a", "b", ... ]
// It is NOT a general JSON parser; if the sharder changes format this breaks.
// ---------------------------------------------------------------------------

struct JsonSidecarTensor {
    std::string name;
    std::vector<int64_t> shape;
    std::string dtype;
    std::size_t offset = 0;
    std::size_t size = 0;
    int64_t n_elements = 0;
};

struct JsonSidecar {
    std::vector<JsonSidecarTensor> tensors;
};

struct JsonManifest {
    std::string architecture = "mamba";
    int n_layer = 0;
    int d_model = 0;
    int d_inner = 0;
    int d_state = 16;
    int d_conv = 4;
    int dt_rank = 128;
    int vocab_size = 0;
    int context_length = 2048;
    bool tied_embeddings = true;
    int bos_token_id = 0;
    int eos_token_id = 0;
    std::vector<std::string> vocab;
    std::vector<std::string> merges;
};

class JsonReader {
public:
    explicit JsonReader(const std::string& s) : s_(s) {}

    void skip_ws() {
        while (p_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[p_])))
            ++p_;
    }

    bool eat(char c) {
        skip_ws();
        if (p_ < s_.size() && s_[p_] == c) { ++p_; return true; }
        return false;
    }

    // Parse a quoted string starting at current position.
    bool parse_string(std::string& out) {
        skip_ws();
        if (p_ >= s_.size() || s_[p_] != '"') return false;
        ++p_;
        out.clear();
        while (p_ < s_.size() && s_[p_] != '"') {
            if (s_[p_] == '\\' && p_ + 1 < s_.size()) {
                char esc = s_[p_ + 1];
                switch (esc) {
                    case 'n': out.push_back('\n'); break;
                    case 't': out.push_back('\t'); break;
                    case 'r': out.push_back('\r'); break;
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/':  out.push_back('/'); break;
                    default:   out.push_back(esc); break;
                }
                p_ += 2;
            } else {
                out.push_back(s_[p_]);
                ++p_;
            }
        }
        if (p_ >= s_.size()) return false;
        ++p_;  // closing quote
        return true;
    }

    // Try to parse a number starting at current position.
    bool parse_number(std::string& out) {
        skip_ws();
        size_t start = p_;
        if (p_ < s_.size() && (s_[p_] == '-' || s_[p_] == '+')) ++p_;
        while (p_ < s_.size() &&
               (std::isdigit(static_cast<unsigned char>(s_[p_])) ||
                s_[p_] == '.' || s_[p_] == 'e' || s_[p_] == 'E' ||
                s_[p_] == '-' || s_[p_] == '+')) {
            ++p_;
        }
        if (p_ == start) return false;
        out = s_.substr(start, p_ - start);
        return true;
    }

    // Skip an arbitrary JSON value (object, array, number, string, true/false/null).
    void skip_value() {
        skip_ws();
        if (p_ >= s_.size()) return;
        if (s_[p_] == '"') {
            std::string t; parse_string(t); return;
        }
        if (s_[p_] == '{') { skip_object(); return; }
        if (s_[p_] == '[') { skip_array();  return; }
        // number / true / false / null
        while (p_ < s_.size() && s_[p_] != ',' && s_[p_] != '}' && s_[p_] != ']') {
            ++p_;
        }
    }

    void skip_object() {
        if (!eat('{')) return;
        while (true) {
            skip_ws();
            if (p_ < s_.size() && s_[p_] == '}') { ++p_; return; }
            std::string k; if (!parse_string(k)) return;
            if (!eat(':')) return;
            skip_value();
            skip_ws();
            if (p_ < s_.size() && s_[p_] == ',') { ++p_; continue; }
            if (p_ < s_.size() && s_[p_] == '}') { ++p_; return; }
        }
    }

    void skip_array() {
        if (!eat('[')) return;
        while (true) {
            skip_ws();
            if (p_ < s_.size() && s_[p_] == ']') { ++p_; return; }
            skip_value();
            skip_ws();
            if (p_ < s_.size() && s_[p_] == ',') { ++p_; continue; }
            if (p_ < s_.size() && s_[p_] == ']') { ++p_; return; }
        }
    }

    // Find the value associated with `key` in the current object.
    bool find_key(const std::string& key) {
        skip_ws();
        if (!eat('{')) return false;
        while (true) {
            skip_ws();
            if (p_ < s_.size() && s_[p_] == '}') { ++p_; return false; }
            std::string k;
            if (!parse_string(k)) return false;
            if (!eat(':')) return false;
            if (k == key) return true;
            skip_value();
            skip_ws();
            if (p_ < s_.size() && s_[p_] == ',') { ++p_; continue; }
            if (p_ < s_.size() && s_[p_] == '}') { ++p_; return false; }
        }
    }

    bool end_of_object() {
        skip_ws();
        return p_ < s_.size() && s_[p_] == '}';
    }

    bool next_field() {
        skip_ws();
        if (p_ < s_.size() && s_[p_] == ',') { ++p_; return true; }
        return false;
    }

    const std::string& str() const { return s_; }
    size_t pos() const { return p_; }

private:
    std::string s_;
    size_t p_ = 0;
};

bool load_sidecar(const std::string& path, JsonSidecar& out) {
    std::ifstream f(path);
    if (!f) return false;
    std::string text((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
    JsonReader r(text);
    if (!r.find_key("tensors")) return false;
    if (!r.eat('[')) return false;
    while (true) {
        r.skip_ws();
        if (r.pos() >= r.str().size() || r.str()[r.pos()] == ']') { r.eat(']'); break; }
        if (!r.eat('{')) return false;
        JsonSidecarTensor t;
        while (true) {
            r.skip_ws();
            if (r.pos() < r.str().size() && r.str()[r.pos()] == '}') { r.eat('}'); break; }
            std::string k; if (!r.parse_string(k)) return false;
            if (!r.eat(':')) return false;
            if (k == "name") {
                r.parse_string(t.name);
            } else if (k == "shape") {
                if (!r.eat('[')) return false;
                while (true) {
                    r.skip_ws();
                    if (r.pos() < r.str().size() && r.str()[r.pos()] == ']') { r.eat(']'); break; }
                    std::string num; if (!r.parse_number(num)) return false;
                    t.shape.push_back(std::stoll(num));
                    r.skip_ws();
                    if (r.pos() < r.str().size() && r.str()[r.pos()] == ',') { r.eat(','); continue; }
                }
            } else if (k == "dtype") {
                r.parse_string(t.dtype);
            } else if (k == "offset") {
                std::string num; r.parse_number(num); t.offset = std::stoull(num);
            } else if (k == "size") {
                std::string num; r.parse_number(num); t.size = std::stoull(num);
            } else if (k == "n_elements") {
                std::string num; r.parse_number(num); t.n_elements = std::stoll(num);
            } else {
                r.skip_value();
            }
            r.skip_ws();
            if (r.pos() < r.str().size() && r.str()[r.pos()] == ',') { r.eat(','); continue; }
        }
        out.tensors.push_back(std::move(t));
        r.skip_ws();
        if (r.pos() < r.str().size() && r.str()[r.pos()] == ',') { r.eat(','); continue; }
    }
    return true;
}

bool load_manifest(const std::string& path, JsonManifest& out) {
    std::ifstream f(path);
    if (!f) return false;
    std::string text((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
    JsonReader r(text);
    auto read_int = [&](const std::string& k, int& v) {
        // We rewind to start by re-finding the key.
        // Simple approach: walk the JSON sequentially, find "key", parse num.
        // For nested objects we just call skip_value on miss.
        // (We re-create a reader each call.)
        JsonReader rr(text);
        if (!rr.find_key(k)) return;
        std::string n; if (!rr.parse_number(n)) return;
        v = std::stoi(n);
    };
    auto read_str = [&](const std::string& k, std::string& v) {
        JsonReader rr(text);
        if (!rr.find_key(k)) return;
        rr.parse_string(v);
    };
    auto read_bool = [&](const std::string& k, bool& v) {
        JsonReader rr(text);
        if (!rr.find_key(k)) return;
        rr.skip_ws();
        if (rr.pos() + 4 <= rr.str().size() &&
            rr.str().compare(rr.pos(), 4, "true") == 0) { v = true; return; }
        if (rr.pos() + 5 <= rr.str().size() &&
            rr.str().compare(rr.pos(), 5, "false") == 0) { v = false; return; }
        std::string n; rr.parse_number(n);
        v = (n == "1");
    };

    read_str("architecture", out.architecture);
    read_int("n_layer",        out.n_layer);
    read_int("d_model",        out.d_model);
    read_int("d_inner",        out.d_inner);
    read_int("d_state",        out.d_state);
    read_int("d_conv",         out.d_conv);
    read_int("dt_rank",        out.dt_rank);
    read_int("vocab_size",     out.vocab_size);
    read_int("context_length", out.context_length);
    read_bool("tied_embeddings", out.tied_embeddings);

    // Tokenizer sub-object.
    {
        JsonReader rr(text);
        if (rr.find_key("tokenizer")) {
            // Walk tokenizer sub-object fields.
            JsonReader sub(text.substr(rr.pos()));
            sub.skip_ws();
            sub.skip_value();  // skip the tokenizer object
            // Better: scan tokenizer keys manually.
        }
    }
    {
        // Robust parse: walk the full text and pick out
        // "tokenizer": { ... } content.
        auto pos = text.find("\"tokenizer\"");
        if (pos != std::string::npos) {
            JsonReader rr(text.substr(pos));
            if (rr.find_key("tokenizer")) {
                if (rr.eat('{')) {
                    while (true) {
                        rr.skip_ws();
                        if (rr.pos() < rr.str().size() &&
                            rr.str()[rr.pos()] == '}') { rr.eat('}'); break; }
                        std::string k; if (!rr.parse_string(k)) break;
                        if (!rr.eat(':')) break;
                        if (k == "model") {
                            rr.parse_string(out.architecture);  // not used here
                        } else if (k == "pre") {
                            std::string t; rr.parse_string(t);
                        } else if (k == "bos_token_id") {
                            std::string n; rr.parse_number(n); out.bos_token_id = std::stoi(n);
                        } else if (k == "eos_token_id") {
                            std::string n; rr.parse_number(n); out.eos_token_id = std::stoi(n);
                        } else {
                            rr.skip_value();
                        }
                        rr.skip_ws();
                        if (rr.pos() < rr.str().size() &&
                            rr.str()[rr.pos()] == ',') { rr.eat(','); continue; }
                    }
                }
            }
        }
    }
    return true;
}

// Load vocab + merges from a freshly-opened GGUF context (we use gguf.h).
// For now, we expect them to live in the manifest. If not, we read them
// straight from the GGUF file using the C gguf API.
struct GGUFLoader {
    int vocab_size = 0;
    int bos_token_id = 0;
    int eos_token_id = 0;
    std::vector<std::string> vocab;
    std::vector<std::string> merges;
};

bool load_gguf_tokenizer(const std::string& gguf_path, GGUFLoader& out) {
#ifdef REMORA_HAVE_GGML
    struct gguf_init_params params = {
        .no_alloc = true,
        .ctx = nullptr,
    };
    struct gguf_context* ctx =
        gguf_init_from_file(gguf_path.c_str(), params);
    if (!ctx) return false;

    const int64_t n_kv = gguf_get_n_kv(ctx);

    // Find vocab / merges / special tokens by name.
    for (int64_t i = 0; i < n_kv; ++i) {
        const char* k = gguf_get_key(ctx, i);
        enum gguf_type t = gguf_get_kv_type(ctx, i);
        if (std::strcmp(k, "tokenizer.ggml.tokens") == 0 && t == GGUF_TYPE_ARRAY) {
            const int64_t n = gguf_get_arr_n(ctx, i);
            out.vocab.reserve(n);
            for (int64_t j = 0; j < n; ++j) {
                out.vocab.push_back(gguf_get_arr_str(ctx, i, j));
            }
            out.vocab_size = static_cast<int>(n);
        } else if (std::strcmp(k, "tokenizer.ggml.merges") == 0 && t == GGUF_TYPE_ARRAY) {
            const int64_t n = gguf_get_arr_n(ctx, i);
            out.merges.reserve(n);
            for (int64_t j = 0; j < n; ++j) {
                out.merges.push_back(gguf_get_arr_str(ctx, i, j));
            }
        } else if (std::strcmp(k, "tokenizer.ggml.bos_token_id") == 0) {
            out.bos_token_id = static_cast<int>(gguf_get_val_u32(ctx, i));
        } else if (std::strcmp(k, "tokenizer.ggml.eos_token_id") == 0) {
            out.eos_token_id = static_cast<int>(gguf_get_val_u32(ctx, i));
        }
    }

    gguf_free(ctx);
    return !out.vocab.empty();
#else
    (void)gguf_path; (void)out;
    return false;
#endif
}

}  // namespace

int main(int argc, char** argv) {
    if (has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
        print_usage(argv[0]);
        return 0;
    }

#ifdef REMORA_HAVE_GGML
    // Load all available ggml backends (CPU, Metal, BLAS) from the directory
    // we baked into the rpath at build time. Without this, the registry is
    // empty and `ggml_backend_init_by_type` returns NULL.
    ggml_backend_load_all();
#endif

    const std::string layers_dir  = arg_value(argc, argv, "--layers");
    const std::string prompt_text = arg_value(argc, argv, "--prompt");
    const std::string routing     = arg_value(argc, argv, "--routing");
    const int n_predict           = std::atoi(arg_value(argc, argv, "--tokens").c_str());
    const double temperature      = std::atof(arg_value(argc, argv, "--temperature").c_str());
    const int buffer_layers       = std::atoi(arg_value(argc, argv, "--buffer-layers").c_str());

    if (layers_dir.empty()) {
        std::fprintf(stderr, "[remora] --layers <dir> is required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    // Routing table is ignored by the ggml kernel but kept for CLI compat.
    (void)routing;

    // ---- Load manifest ----------------------------------------------------
    JsonManifest cfg;
    {
        std::string mp = layers_dir + "/manifest.json";
        if (!load_manifest(mp, cfg)) {
            std::fprintf(stderr, "[remora] failed to load %s\n", mp.c_str());
            return 1;
        }
    }
    std::printf("[remora] %s: d_model=%d d_inner=%d d_state=%d d_conv=%d dt_rank=%d "
                "vocab=%d n_layer=%d tied=%s\n",
                cfg.architecture.c_str(),
                cfg.d_model, cfg.d_inner, cfg.d_state, cfg.d_conv, cfg.dt_rank,
                cfg.vocab_size, cfg.n_layer,
                cfg.tied_embeddings ? "yes" : "no");

    // ---- Load per-block sidecars in numeric order -------------------------
    std::vector<std::pair<int, JsonSidecar>> indexed;
    for (const auto& entry : fs::directory_iterator(layers_dir)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("blk.", 0) != 0) continue;
        if (name.find(".meta.json") != std::string::npos) continue;
        int idx = std::atoi(name.c_str() + 4);
        JsonSidecar sc;
        std::string meta = entry.path().string() + ".meta.json";
        if (!load_sidecar(meta, sc)) {
            std::fprintf(stderr, "[remora] failed to parse %s\n", meta.c_str());
            return 1;
        }
        indexed.emplace_back(idx, std::move(sc));
    }
    std::stable_sort(indexed.begin(), indexed.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });
    if (indexed.empty()) {
        std::fprintf(stderr, "[remora] no blk.* sidecars in %s\n",
                     layers_dir.c_str());
        return 1;
    }

    // Convert to remora::LayerDesc[].
    std::vector<remora::LayerDesc> descs;
    descs.reserve(indexed.size());
    for (auto& pair_item : indexed) {
        int idx = pair_item.first;
        JsonSidecar& sc = pair_item.second;
        remora::LayerDesc d;
        d.block_id = idx;
        for (const auto& t : sc.tensors) {
            remora::TensorDesc td;
            td.name  = t.name;
            td.shape = t.shape;
            td.dtype = t.dtype;
            td.offset = t.offset;
            td.size   = t.size;
            d.tensors.push_back(std::move(td));
        }
        descs.push_back(std::move(d));
    }
    const std::size_t n_layer = descs.size();

    // ---- Memory-map each block shard (lazily on demand) -------------------
    // We let the LayerStreamer own the mappings; main.cpp just dispatches.

    // ---- Set up runtime state --------------------------------------------
    remora::RuntimeContext ctx;
    ctx.cfg = remora::ModelConfig{};
    ctx.cfg.architecture       = cfg.architecture;
    ctx.cfg.n_layer            = cfg.n_layer ? cfg.n_layer : static_cast<int>(n_layer);
    ctx.cfg.d_model            = cfg.d_model;
    ctx.cfg.d_inner            = cfg.d_inner;
    ctx.cfg.d_state            = cfg.d_state;
    ctx.cfg.d_conv             = cfg.d_conv;
    ctx.cfg.dt_rank            = cfg.dt_rank;
    ctx.cfg.vocab_size         = cfg.vocab_size;
    ctx.cfg.context_length     = cfg.context_length;
    ctx.cfg.tied_embeddings    = cfg.tied_embeddings;
    ctx.hidden_state.assign(cfg.d_model, 0.0f);
    ctx.ssm_state.assign(static_cast<std::size_t>(cfg.n_layer) *
                         cfg.d_state * cfg.d_inner, 0.0f);
    ctx.conv_state.assign(static_cast<std::size_t>(cfg.n_layer) *
                          (cfg.d_conv - 1) * cfg.d_inner, 0.0f);

    // ---- Build ordered layer path list -----------------------------------
    std::vector<std::string> layer_paths;
    layer_paths.reserve(indexed.size());
    for (auto& [idx, sc] : indexed) {
        std::string p = layers_dir + "/blk." + std::to_string(idx);
        layer_paths.push_back(std::move(p));
    }

    remora::LayerStreamer streamer(buffer_layers);
    streamer.set_layer_paths(layer_paths);
    if (!streamer.start()) {
        std::fprintf(stderr, "[remora] failed to start layer streamer\n");
        return 1;
    }

    // ---- Pretokenize the prompt and run prefill ---------------------------
    //
    // We need vocab + merges from the original GGUF for the tokenizer.
    // Try to locate the source GGUF next to the layers dir (the sharder
    // doesn't write it; user must keep it around).
    std::string gguf_path = layers_dir + "/../mamba-1.4b-hf.Q4_K_M.gguf";
    if (!fs::exists(gguf_path)) {
        // fall back: search one level up
        gguf_path = "";
        for (const auto& e : fs::directory_iterator(layers_dir + "/..")) {
            if (e.path().extension() == ".gguf") {
                gguf_path = e.path().string();
                break;
            }
        }
    }

    GGUFLoader tk;
    bool have_tk = false;
    if (!gguf_path.empty()) {
        have_tk = load_gguf_tokenizer(gguf_path, tk);
    }
    if (!have_tk) {
        std::fprintf(stderr,
            "[remora] could not load tokenizer from %s\n",
            gguf_path.empty() ? "<none>" : gguf_path.c_str());
        return 1;
    }

    remora::BPETokenizer tokenizer;
    tokenizer.load(tk.vocab, tk.merges, tk.bos_token_id, tk.eos_token_id);
    const std::string text = prompt_text.empty()
        ? std::string("Hello, my name is") : prompt_text;
    auto prompt_ids = tokenizer.encode(text);
    if (prompt_ids.empty()) {
        std::fprintf(stderr, "[remora] prompt tokenized to empty sequence\n");
        return 1;
    }
    std::printf("[remora] prompt (%zu tokens): %s\n",
                prompt_ids.size(), text.c_str());

    // ---- Load global.bin (embeddings + output norm, aliased into mmap) ----
    std::string global_path = layers_dir + "/global.bin";
    remora::MappedFile global_file;
    if (!global_file.open(global_path)) {
        std::fprintf(stderr, "[remora] failed to mmap %s\n", global_path.c_str());
        return 1;
    }
    remora::LayerDesc global_desc;
    {
        JsonSidecar sc;
        std::string meta = global_path + ".meta.json";
        if (!load_sidecar(meta, sc)) {
            std::fprintf(stderr, "[remora] failed to parse %s\n", meta.c_str());
            return 1;
        }
        global_desc.block_id = -1;
        for (const auto& t : sc.tensors) {
            remora::TensorDesc td;
            td.name  = t.name;
            td.shape = t.shape;
            td.dtype = t.dtype;
            td.offset = t.offset;
            td.size   = t.size;
            global_desc.tensors.push_back(std::move(td));
        }
    }
    remora::GlobalContext gctx;
    gctx.cfg  = ctx.cfg;
    gctx.base = global_file.data();
    gctx.size = global_file.size();
    gctx.desc = &global_desc;

    // We implement a single-token forward at a time. The sharded kernel
    // reads its input from ctx.hidden_state; the streamer is reset and
    // re-swept for every token so the sliding window stays bounded.
    //
    // One persistent ComputeEngine backs every layer and every token: it
    // owns a single GPU-first ggml backend list + scheduler, so the layer
    // matmuls reuse the same buffers instead of re-init'ing a backend per
    // call (the previous design, which was pathologically slow).
    remora::ComputeEngine engine;
    if (!engine.ok()) {
        std::fprintf(stderr, "[remora] failed to init compute engine\n");
        return 1;
    }
    std::fprintf(stderr, "[remora] compute engine ready (primary backend: %s)\n",
                 engine.primary_backend_name());

    // Per-layer ggml context cache. The weights are byte-identical across
    // tokens; only the mmap base changes. Building each block's contexts
    // once and reusing them across tokens avoids re-initializing a CPU
    // backend and re-copying ~376 KB of F32 weights per layer per token.
    remora::LayerCache layer_cache;

    auto t0 = std::chrono::steady_clock::now();

    // ---- Prefill: run prompt tokens, updating recurrent state ------------
    for (int32_t tok : prompt_ids) {
        // Embed the token into hidden_state (residual stream).
        if (!gctx.embed(ctx, tok, engine)) return 1;

        if (!streamer.reset()) {
            std::fprintf(stderr, "[remora] failed to reset streamer\n");
            return 1;
        }
        do {
            const remora::LayerBuffer& buf = streamer.active();
            remora::LayerContext layer;
            layer.weights_base  = buf.data;
            layer.weights_size  = buf.size;
            layer.desc          = &descs[streamer.current_index()];
            run_layer_forward(ctx, layer, engine, layer_cache);
        } while (streamer.advance());
        ctx.step++;
    }

    // ---- Generate `n_predict` more tokens --------------------------------
    int32_t last_tok = prompt_ids.back();
    std::string generated;
    auto t_first = std::chrono::steady_clock::time_point{};
    bool first_tok_recorded = false;
    std::vector<float> logits;
    for (int step = 0; step < n_predict; ++step) {
        // Embed the previous token into hidden_state.
        if (!gctx.embed(ctx, last_tok, engine)) return 1;

        if (!streamer.reset()) {
            std::fprintf(stderr, "[remora] failed to reset streamer\n");
            return 1;
        }
        do {
            const remora::LayerBuffer& buf = streamer.active();
            remora::LayerContext layer;
            layer.weights_base  = buf.data;
            layer.weights_size  = buf.size;
            layer.desc          = &descs[streamer.current_index()];
            run_layer_forward(ctx, layer, engine, layer_cache);
        } while (streamer.advance());

        // Final RMSNorm + lm_head -> logits.
        if (!gctx.compute_logits(ctx, logits, engine)) return 1;
        ctx.step++;

        // Sample the next token.
        if (temperature <= 0.0f || cfg.tied_embeddings == false) {
            // Greedy: argmax.
            int argmax = 0;
            for (size_t i = 1; i < logits.size(); ++i) {
                if (logits[i] > logits[argmax]) argmax = static_cast<int>(i);
            }
            last_tok = argmax;
        } else {
            // Temperature sampling: softmax over logits/temp, then sample.
            std::vector<double> probs(logits.size());
            double max_l = logits[0];
            for (auto v : logits) max_l = std::max<double>(max_l, v);
            double sum = 0.0;
            for (size_t i = 0; i < logits.size(); ++i) {
                probs[i] = std::exp((logits[i] - max_l) / temperature);
                sum += probs[i];
            }
            for (auto& p : probs) p /= sum;
            // Simple stochastic sampling via random_device.
            static std::mt19937_64 rng(std::random_device{}());
            std::uniform_real_distribution<double> uni(0.0, 1.0);
            double r = uni(rng);
            double acc = 0.0;
            last_tok = static_cast<int>(logits.size() - 1);
            for (size_t i = 0; i < probs.size(); ++i) {
                acc += probs[i];
                if (r <= acc) { last_tok = static_cast<int>(i); break; }
            }
        }

        std::string piece = tokenizer.decode_token(last_tok);
        generated += piece;
        if (!first_tok_recorded) {
            t_first = std::chrono::steady_clock::now();
            first_tok_recorded = true;
        }
    }
    auto t_end = std::chrono::steady_clock::now();

    auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                       t_end - t0).count();
    auto ttft_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                       t_first - t0).count();
    const double gen_tok_per_s = n_predict > 0
        ? (1000.0 * n_predict) / std::max<long>(wall_ms, 1)
        : 0.0;

    std::printf("[remora] generated %d tokens in %lld ms (ttft %lld ms)\n",
                n_predict,
                static_cast<long long>(wall_ms),
                static_cast<long long>(ttft_ms));
    std::printf("[remora] gen tok/s: %.2f\n", gen_tok_per_s);
    std::printf("[remora] output: %s\n", generated.c_str());
    return 0;
}