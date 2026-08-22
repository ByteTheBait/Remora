// Remora MLX GPU driver.
//
// Runs the Mamba selective-scan forward pass on Apple Silicon's GPU (Metal)
// via the MLX C++ API (libmlx.dylib), streaming the same per-layer shard
// files that `scripts/shard_mlx.py` produces (F16 weights).
//
// This is the "no ggml" GPU path. ggml's Metal tensor API is disabled on
// pre-M5 / pre-A19 silicon (it reports `has tensor = false` on an Apple M2),
// but MLX's GPU backend works on all Apple Silicon. So on a Mac this driver
// gives real GPU acceleration that the ggml-based `remora` binary cannot.
//
// The tokenizer comes from the source MLX model dir's tokenizer.json (the
// same one `mlx_lm` uses), parsed into the Remora BPE tokenizer.
//
// Usage:
//   remora_mlx --layers <dir> [--tokenizer <mlx_model_dir>]
//              [--prompt "..."] [--tokens N] [--temperature T]

#include "layer_streamer.h"
#include "mlx_engine.h"
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

namespace fs = std::filesystem;

namespace {

void print_usage(const char* prog) {
    std::printf(
        "Remora MLX GPU driver — runs the layer-by-layer Mamba forward on\n"
        "Apple Silicon's GPU (Metal) via the MLX C++ API.\n"
        "\n"
        "Usage:\n"
        "  %s --layers <dir> [--tokenizer <dir>] [--prompt <text>]\n"
        "                     [--tokens N] [--temperature T] [--buffer-layers N]\n"
        "                     [--batch N]\n"
        "\n"
        "Options:\n"
        "  --layers <dir>     Directory containing MLX sharded layer files\n"
        "                     (blk.0, blk.1, ...), manifest.json, global.bin\n"
        "                     produced by scripts/shard_mlx.py.\n"
        "  --tokenizer <dir>  MLX model dir with tokenizer.json (default:\n"
        "                     layers dir /../mlx_model).\n"
        "  --prompt <text>    Prompt text (default: \"Hello, my name is\").\n"
        "  --tokens  <N>      Tokens to generate (default: 32).\n"
        "  --temperature T    Sampling temperature, 0 = greedy (default 0.7).\n"
        "  --buffer-layers N  Streaming lookahead (default: 1).\n"
        "  --batch N          Run N independent token streams in parallel per\n"
        "                     layer, sharing the single streamed-in layer\n"
        "                     buffer (read-only weights, one resident copy).\n"
        "                     Independent only — not the tokens of one stream.\n",
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
// Minimal JSON reader (mirrors main.cpp's; enough for the sharder sidecars,
// manifest, and the MLX tokenizer.json vocab/merges).
// ---------------------------------------------------------------------------
struct JsonSidecarTensor {
    std::string name;
    std::vector<int64_t> shape;
    std::string dtype;
    std::size_t offset = 0;
    std::size_t size = 0;
    int64_t n_elements = 0;
};
struct JsonSidecar { std::vector<JsonSidecarTensor> tensors; };

struct JsonManifest {
    std::string architecture = "mamba";
    int n_layer = 0, d_model = 0, d_inner = 0, d_state = 16, d_conv = 4;
    int dt_rank = 128, vocab_size = 0, context_length = 2048;
    bool tied_embeddings = true;
    int bos_token_id = 0, eos_token_id = 0;
};

class JsonReader {
public:
    explicit JsonReader(const std::string& s) : s_(s) {}
    void skip_ws() {
        while (p_ < s_.size() &&
               std::isspace(static_cast<unsigned char>(s_[p_]))) ++p_;
    }
    bool eat(char c) {
        skip_ws();
        if (p_ < s_.size() && s_[p_] == c) { ++p_; return true; }
        return false;
    }
    bool parse_string(std::string& out) {
        skip_ws();
        if (p_ >= s_.size() || s_[p_] != '"') return false;
        ++p_; out.clear();
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
            } else { out.push_back(s_[p_]); ++p_; }
        }
        if (p_ >= s_.size()) return false;
        ++p_;
        return true;
    }
    bool parse_number(std::string& out) {
        skip_ws();
        size_t start = p_;
        if (p_ < s_.size() && (s_[p_] == '-' || s_[p_] == '+')) ++p_;
        while (p_ < s_.size() &&
               (std::isdigit(static_cast<unsigned char>(s_[p_])) ||
                s_[p_] == '.' || s_[p_] == 'e' || s_[p_] == 'E' ||
                s_[p_] == '-' || s_[p_] == '+')) ++p_;
        if (p_ == start) return false;
        out = s_.substr(start, p_ - start);
        return true;
    }
    void skip_value() {
        skip_ws();
        if (p_ >= s_.size()) return;
        if (s_[p_] == '"') { std::string t; parse_string(t); return; }
        if (s_[p_] == '{') { skip_object(); return; }
        if (s_[p_] == '[') { skip_array(); return; }
        while (p_ < s_.size() && s_[p_] != ',' &&
               s_[p_] != '}' && s_[p_] != ']') ++p_;
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
            if (k == "name") r.parse_string(t.name);
            else if (k == "shape") {
                if (!r.eat('[')) return false;
                while (true) {
                    r.skip_ws();
                    if (r.pos() < r.str().size() && r.str()[r.pos()] == ']') { r.eat(']'); break; }
                    std::string num; if (!r.parse_number(num)) return false;
                    t.shape.push_back(std::stoll(num));
                    r.skip_ws();
                    if (r.pos() < r.str().size() && r.str()[r.pos()] == ',') { r.eat(','); continue; }
                }
            } else if (k == "dtype") r.parse_string(t.dtype);
            else if (k == "offset") { std::string n; r.parse_number(n); t.offset = std::stoull(n); }
            else if (k == "size") { std::string n; r.parse_number(n); t.size = std::stoull(n); }
            else if (k == "n_elements") { std::string n; r.parse_number(n); t.n_elements = std::stoll(n); }
            else r.skip_value();
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
    auto read_int = [&](const std::string& k, int& v) {
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
        std::string n; rr.parse_number(n); v = (n == "1");
    };
    read_str("architecture", out.architecture);
    read_int("n_layer", out.n_layer);
    read_int("d_model", out.d_model);
    read_int("d_inner", out.d_inner);
    read_int("d_state", out.d_state);
    read_int("d_conv", out.d_conv);
    read_int("dt_rank", out.dt_rank);
    read_int("vocab_size", out.vocab_size);
    read_int("context_length", out.context_length);
    read_bool("tied_embeddings", out.tied_embeddings);
    // bos/eos live inside the nested "tokenizer" object.
    {
        JsonReader rr(text);
        if (rr.find_key("tokenizer") && rr.eat('{')) {
            std::string b, e;
            while (true) {
                rr.skip_ws();
                if (rr.pos() >= rr.str().size() || rr.str()[rr.pos()] == '}') { rr.eat('}'); break; }
                std::string tokkey;
                if (!rr.parse_string(tokkey)) break;
                if (!rr.eat(':')) break;
                if (tokkey == "bos_token_id") rr.parse_number(b);
                else if (tokkey == "eos_token_id") rr.parse_number(e);
                else rr.skip_value();
                rr.skip_ws();
                if (rr.pos() < rr.str().size() && rr.str()[rr.pos()] == ',') { rr.eat(','); continue; }
            }
            if (!b.empty()) out.bos_token_id = std::stoi(b);
            if (!e.empty()) out.eos_token_id = std::stoi(e);
        }
    }
    return true;
}

// Load vocab + merges from the MLX model's tokenizer.json.
bool load_tokenizer_json(const std::string& path, remora::BPETokenizer& tokenizer,
                         int bos_id, int eos_id) {
    std::ifstream f(path);
    if (!f) return false;
    std::string text((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    JsonReader r(text);
    // Navigate to "model" object, then "vocab" (dict) and "merges" (array).
    std::vector<std::string> vocab;
    std::vector<std::string> merges;

    // Find the "model" object.
    if (!r.find_key("model")) return false;
    if (!r.eat('{')) return false;
    std::string model_key;
    while (true) {
        r.skip_ws();
        if (r.pos() < r.str().size() && r.str()[r.pos()] == '}') { r.eat('}'); break; }
        if (!r.parse_string(model_key)) break;
        if (!r.eat(':')) break;
        if (model_key == "vocab") {
            if (!r.eat('{')) break;
            while (true) {
                r.skip_ws();
                if (r.pos() < r.str().size() && r.str()[r.pos()] == '}') { r.eat('}'); break; }
                std::string tok; if (!r.parse_string(tok)) break;
                if (!r.eat(':')) break;
                std::string idnum; r.parse_number(idnum);
                vocab.push_back(tok);
                r.skip_ws();
                if (r.pos() < r.str().size() && r.str()[r.pos()] == ',') { r.eat(','); continue; }
            }
        } else if (model_key == "merges") {
            if (!r.eat('[')) break;
            while (true) {
                r.skip_ws();
                if (r.pos() < r.str().size() && r.str()[r.pos()] == ']') { r.eat(']'); break; }
                // Each merge is either a string "a b" or a 2-array [a,b].
                if (r.pos() < r.str().size() && r.str()[r.pos()] == '"') {
                    std::string m; r.parse_string(m);
                    merges.push_back(m);
                } else if (r.pos() < r.str().size() && r.str()[r.pos()] == '[') {
                    r.eat('[');
                    std::string a, b;
                    if (!r.parse_string(a)) { r.skip_ws(); r.skip_value(); }
                    r.skip_ws();
                    if (r.pos() < r.str().size() && r.str()[r.pos()] == ',') r.eat(',');
                    if (!r.parse_string(b)) { r.skip_ws(); r.skip_value(); }
                    r.skip_ws();
                    if (r.pos() < r.str().size() && r.str()[r.pos()] == ']') r.eat(']');
                    merges.push_back(a + " " + b);
                }
                r.skip_ws();
                if (r.pos() < r.str().size() && r.str()[r.pos()] == ',') { r.eat(','); continue; }
            }
        } else {
            r.skip_value();
        }
        r.skip_ws();
        if (r.pos() < r.str().size() && r.str()[r.pos()] == ',') { r.eat(','); continue; }
        if (r.pos() < r.str().size() && r.str()[r.pos()] == '}') { r.eat('}'); break; }
    }

    if (vocab.empty()) return false;
    tokenizer.load(vocab, merges, bos_id, eos_id);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);   // line-buffer so prints flush to pipes
    if (has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
        print_usage(argv[0]);
        return 0;
    }

    const std::string layers_dir  = arg_value(argc, argv, "--layers");
    const std::string tokenizer_dir = arg_value(argc, argv, "--tokenizer");
    const std::string prompt_text = arg_value(argc, argv, "--prompt");
    const int n_predict           = std::atoi(arg_value(argc, argv, "--tokens").c_str());
    const double temperature      = std::atof(arg_value(argc, argv, "--temperature").c_str());
    const int buffer_layers       = std::atoi(arg_value(argc, argv, "--buffer-layers").c_str());
    const int batch_size          = std::atoi(arg_value(argc, argv, "--batch").c_str());
    // Number of independent token streams to run in parallel per layer. Only
    // valid for independent sequences — NOT the tokens of a single stream.
    const int n_streams = batch_size > 0 ? batch_size : 1;

    if (layers_dir.empty()) {
        std::fprintf(stderr, "[remora-mlx] --layers <dir> is required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    // ---- Load manifest ----------------------------------------------------
    JsonManifest cfg;
    if (!load_manifest(layers_dir + "/manifest.json", cfg)) {
        std::fprintf(stderr, "[remora-mlx] failed to load manifest.json\n");
        return 1;
    }
    std::printf("[remora-mlx] %s: d_model=%d d_inner=%d d_state=%d d_conv=%d "
                "dt_rank=%d vocab=%d n_layer=%d tied=%s\n",
                cfg.architecture.c_str(), cfg.d_model, cfg.d_inner, cfg.d_state,
                cfg.d_conv, cfg.dt_rank, cfg.vocab_size, cfg.n_layer,
                cfg.tied_embeddings ? "yes" : "no");

    // ---- Load per-block sidecars ------------------------------------------
    std::vector<std::pair<int, JsonSidecar>> indexed;
    for (const auto& entry : fs::directory_iterator(layers_dir)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("blk.", 0) != 0) continue;
        if (name.find(".meta.json") != std::string::npos) continue;
        int idx = std::atoi(name.c_str() + 4);
        JsonSidecar sc;
        if (!load_sidecar(entry.path().string() + ".meta.json", sc)) {
            std::fprintf(stderr, "[remora-mlx] failed to parse %s\n",
                         (entry.path().string() + ".meta.json").c_str());
            return 1;
        }
        indexed.emplace_back(idx, std::move(sc));
    }
    std::stable_sort(indexed.begin(), indexed.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });
    if (indexed.empty()) {
        std::fprintf(stderr, "[remora-mlx] no blk.* sidecars in %s\n",
                     layers_dir.c_str());
        return 1;
    }

    std::vector<remora::LayerDesc> descs;
    descs.reserve(indexed.size());
    for (auto& pair_item : indexed) {
        remora::LayerDesc d;
        d.block_id = pair_item.first;
        for (const auto& t : pair_item.second.tensors) {
            remora::TensorDesc td;
            td.name = t.name; td.shape = t.shape; td.dtype = t.dtype;
            td.offset = t.offset; td.size = t.size;
            td.n_elements = t.n_elements;
            d.tensors.push_back(std::move(td));
        }
        descs.push_back(std::move(d));
    }
    const std::size_t n_layer = descs.size();

    std::printf("[remora-mlx] %zu blocks found\n", descs.size());
    if (descs.empty()) return 1;

    // ---- Set up runtime state --------------------------------------------
    remora::RuntimeContext ctx;
    ctx.cfg = remora::ModelConfig{};
    ctx.cfg.architecture       = cfg.architecture;
    ctx.cfg.n_layer            = cfg.n_layer ? cfg.n_layer : (int)n_layer;
    ctx.cfg.d_model            = cfg.d_model;
    ctx.cfg.d_inner            = cfg.d_inner;
    ctx.cfg.d_state            = cfg.d_state;
    ctx.cfg.d_conv             = cfg.d_conv;
    ctx.cfg.dt_rank            = cfg.dt_rank;
    ctx.cfg.vocab_size         = cfg.vocab_size;
    ctx.cfg.context_length     = cfg.context_length;
    ctx.cfg.tied_embeddings    = cfg.tied_embeddings;
    ctx.hidden_state.assign(cfg.d_model, 0.0f);
    ctx.ssm_state.assign((size_t)cfg.n_layer * cfg.d_state * cfg.d_inner, 0.0f);
    ctx.conv_state.assign((size_t)cfg.n_layer * (cfg.d_conv - 1) * cfg.d_inner, 0.0f);

    // ---- MLX GPU engine ----------------------------------------------------
    remora::MlxEngine engine;
    if (!engine.ok()) {
        std::fprintf(stderr, "[remora-mlx] failed to init MLX GPU engine\n");
        return 1;
    }
    std::fprintf(stderr, "[remora-mlx] compute engine ready (backend: %s)\n",
                 engine.backend_name());
    // Bound the MLX weight cache to the stream window so peak RSS tracks
    // ~(buffer_layers + 1) blocks instead of the whole model.
    engine.set_max_cached_blocks((std::size_t)(buffer_layers + 1));

    // ---- Streamer ---------------------------------------------------------
    std::vector<std::string> layer_paths;
    layer_paths.reserve(indexed.size());
    for (auto& [idx, sc] : indexed)
        layer_paths.push_back(layers_dir + "/blk." + std::to_string(idx));
    remora::LayerStreamer streamer(buffer_layers);
    streamer.set_layer_paths(layer_paths);
    if (!streamer.start()) {
        std::fprintf(stderr, "[remora-mlx] failed to start layer streamer\n");
        return 1;
    }

    // ---- Tokenizer ---------------------------------------------------------
    std::string tk_path = tokenizer_dir.empty()
        ? layers_dir + "/../mlx_model/tokenizer.json" : tokenizer_dir + "/tokenizer.json";
    remora::BPETokenizer tokenizer;
    if (!load_tokenizer_json(tk_path, tokenizer, cfg.bos_token_id, cfg.eos_token_id)) {
        std::fprintf(stderr, "[remora-mlx] failed to load tokenizer from %s\n",
                     tk_path.c_str());
        return 1;
    }
    const std::string text = prompt_text.empty()
        ? std::string("Hello, my name is") : prompt_text;
    auto prompt_ids = tokenizer.encode(text);
    if (prompt_ids.empty()) {
        std::fprintf(stderr, "[remora-mlx] prompt tokenized to empty sequence\n");
        return 1;
    }
    std::printf("[remora-mlx] prompt (%zu tokens): %s\n", prompt_ids.size(),
                text.c_str());

    // ---- global.bin ---------------------------------------------------------
    std::string global_path = layers_dir + "/global.bin";
    remora::MappedFile global_file;
    if (!global_file.open(global_path)) {
        std::fprintf(stderr, "[remora-mlx] failed to mmap %s\n", global_path.c_str());
        return 1;
    }
    remora::LayerDesc global_desc;
    {
        JsonSidecar sc;
        if (!load_sidecar(global_path + ".meta.json", sc)) {
            std::fprintf(stderr, "[remora-mlx] failed to parse global.bin.meta.json\n");
            return 1;
        }
        global_desc.block_id = -1;
        for (const auto& t : sc.tensors) {
            remora::TensorDesc td;
            td.name = t.name; td.shape = t.shape; td.dtype = t.dtype;
            td.offset = t.offset; td.size = t.size;
            td.n_elements = t.n_elements;
            global_desc.tensors.push_back(std::move(td));
        }
    }

    auto t0 = std::chrono::steady_clock::now();

    // ------------------------------------------------------------------
    // Batch mode: run `batch_size` independent token streams in parallel,
    // all sharing the single streamed-in layer buffer (read-only weights).
    // Each stream owns its own hidden / ssm / conv state. This is only
    // valid for independent sequences — NOT the tokens of one stream.
    // ------------------------------------------------------------------
    std::vector<remora::RuntimeContext> batch(n_streams);
    for (auto& b : batch) {
        b.cfg = ctx.cfg;
        b.hidden_state.assign(cfg.d_model, 0.0f);
        b.ssm_state.assign((size_t)cfg.n_layer * cfg.d_state * cfg.d_inner, 0.0f);
        b.conv_state.assign((size_t)cfg.n_layer * (cfg.d_conv - 1) * cfg.d_inner, 0.0f);
    }
    auto& cur = batch[0];   // single-stream path (n_streams==1) uses this

    // ---- Prefill -----------------------------------------------------------
    for (int32_t tok : prompt_ids) {
        for (auto& b : batch) {
            if (!engine.embed(b, tok, global_desc, global_file.data(),
                              global_file.size())) return 1;
        }
        if (!streamer.reset()) { std::fprintf(stderr, "[remora-mlx] streamer reset failed\n"); return 1; }
        do {
            const remora::LayerBuffer& buf = streamer.active();
            remora::LayerContext layer;
            layer.weights_base = buf.data;
            layer.weights_size = buf.size;
            layer.desc         = &descs[streamer.current_index()];
            if (n_streams > 1) {
                std::vector<remora::RuntimeContext*> ptrs;
                ptrs.reserve(batch.size());
                for (auto& b : batch) ptrs.push_back(&b);
                engine.run_layer_forward_batch(ptrs, layer);
            } else {
                engine.run_layer_forward(cur, layer);
            }
        } while (streamer.advance());
        for (auto& b : batch) b.step++;
    }

    // ---- Generate ----------------------------------------------------------
    std::vector<int32_t> last_toks(n_streams, prompt_ids.back());
    std::vector<std::string> generated_out(n_streams);
    std::vector<std::vector<float>> batch_logits(n_streams);
    auto t_first = std::chrono::steady_clock::time_point{};
    bool first_tok_recorded = false;
    for (int step = 0; step < n_predict; ++step) {
        for (int s = 0; s < n_streams; ++s) {
            if (!engine.embed(batch[s], last_toks[s], global_desc,
                              global_file.data(), global_file.size())) return 1;
        }
        if (!streamer.reset()) { std::fprintf(stderr, "[remora-mlx] streamer reset failed\n"); return 1; }
        do {
            const remora::LayerBuffer& buf = streamer.active();
            remora::LayerContext layer;
            layer.weights_base = buf.data;
            layer.weights_size = buf.size;
            layer.desc         = &descs[streamer.current_index()];
            if (n_streams > 1) {
                std::vector<remora::RuntimeContext*> ptrs;
                ptrs.reserve(batch.size());
                for (auto& b : batch) ptrs.push_back(&b);
                engine.run_layer_forward_batch(ptrs, layer);
            } else {
                engine.run_layer_forward(batch[0], layer);
            }
        } while (streamer.advance());

        for (int s = 0; s < n_streams; ++s) {
            if (!engine.compute_logits(batch[s], batch_logits[s], global_desc,
                                       global_file.data(), global_file.size())) return 1;
            batch[s].step++;
        }

        for (int s = 0; s < n_streams; ++s) {
            int& last_tok = last_toks[s];
            auto& lg = batch_logits[s];
            if (temperature <= 0.0f) {
                int argmax = 0;
                for (size_t i = 1; i < lg.size(); ++i)
                    if (lg[i] > lg[argmax]) argmax = (int)i;
                last_tok = argmax;
            } else {
                std::vector<double> probs(lg.size());
                double max_l = lg[0];
                for (auto v : lg) max_l = std::max<double>(max_l, v);
                double sum = 0.0;
                for (size_t i = 0; i < lg.size(); ++i) {
                    probs[i] = std::exp((lg[i] - max_l) / temperature);
                    sum += probs[i];
                }
                for (auto& p : probs) p /= sum;
                static std::mt19937_64 rng(std::random_device{}());
                std::uniform_real_distribution<double> uni(0.0, 1.0);
                double r = uni(rng), acc = 0.0;
                last_tok = (int)lg.size() - 1;
                for (size_t i = 0; i < probs.size(); ++i) {
                    acc += probs[i];
                    if (r <= acc) { last_tok = (int)i; break; }
                }
            }
            generated_out[s] += tokenizer.decode_token(last_tok);
            if (!first_tok_recorded) { t_first = std::chrono::steady_clock::now(); first_tok_recorded = true; }
        }
    }
    auto t_end = std::chrono::steady_clock::now();

    auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t0).count();
    auto ttft_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_first - t0).count();
    const double gen_tok_per_s = n_predict > 0
        ? (1000.0 * n_predict) / std::max<long>(wall_ms, 1) : 0.0;

    std::printf("[remora-mlx] %d stream(s) x %d tokens in %lld ms (ttft %lld ms)\n",
                n_streams, n_predict, (long long)wall_ms, (long long)ttft_ms);
    std::printf("[remora-mlx] gen tok/s: %.2f\n", gen_tok_per_s);
    for (int s = 0; s < n_streams; ++s)
        std::printf("[remora-mlx] stream %d output: %s\n", s, generated_out[s].c_str());
    return 0;
}
