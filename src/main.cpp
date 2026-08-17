#include "draft_router.h"
#include "layer_streamer.h"
#include "rnn_kernel.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

void print_usage(const char* prog) {
    std::printf(
        "Remora - Layer-by-Layer RNN MoE on consumer hardware\n"
        "\n"
        "Usage:\n"
        "  %s --routing <routing.json> [--draft <draft.gguf>] [--layers <dir>] [--tokens N]\n"
        "\n"
        "Options:\n"
        "  --routing <file>   JSON routing table (required)\n"
        "  --draft <file>     draft model path (optional, placeholder)\n"
        "  --layers <dir>     directory containing sharded layer files (blk.0, blk.1, ...)\n"
        "  --tokens <N>       number of tokens to simulate (default 8)\n",
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

}  // namespace

int main(int argc, char** argv) {
    if (has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
        print_usage(argv[0]);
        return 0;
    }

    const std::string routing_path = arg_value(argc, argv, "--routing");
    const std::string draft_path = arg_value(argc, argv, "--draft");
    const std::string layers_dir = arg_value(argc, argv, "--layers");
    const int tokens = std::atoi(arg_value(argc, argv, "--tokens").c_str());

    if (routing_path.empty()) {
        std::fprintf(stderr, "[remora] --routing is required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    // ---- Phase 2: draft model + router --------------------------------
    remora::DraftRouter router;
    if (!router.load_routing(routing_path)) return 1;
    if (!draft_path.empty()) {
        if (!router.load_draft_model(draft_path)) {
            std::fprintf(stderr, "[remora] failed to load draft model\n");
            return 1;
        }
        std::printf("[remora] draft model loaded: %s\n", draft_path.c_str());
    }

    // Simulate a token stream; the draft model classifies the task and the
    // router selects an expert pool.
    std::vector<int32_t> token_stream;
    for (int i = 0; i < tokens; ++i) token_stream.push_back(i % 3);
    const std::string pool = router.route(token_stream);
    std::printf("[remora] draft router selected expert pool: %s\n", pool.c_str());

    // ---- Phase 1: layer streaming + RNN execution loop ----------------
    // Build the ordered layer path list. In a real run these come from the
    // sharded expert pool produced by scripts/shard_gguf.py.
    std::vector<std::string> layer_paths;
    bool synthetic = layers_dir.empty();
    if (!synthetic) {
        for (int i = 0; i < 4; ++i) {
            layer_paths.push_back(layers_dir + "/blk." + std::to_string(i));
        }
    } else {
        // Fallback: no layers on disk; run the loop with empty buffers to
        // demonstrate the state-compaction mechanics.
        std::printf("[remora] no --layers given; running with synthetic layers\n");
        for (int i = 0; i < 4; ++i) layer_paths.push_back("");
    }

    remora::LayerStreamer streamer(/*buffer_capacity=*/1u << 20);
    streamer.set_layer_paths(layer_paths);

    remora::RuntimeContext ctx;
    ctx.state_dim = 64;
    ctx.hidden_state.assign(64, 0.0f);
    ctx.layer_count = layer_paths.size();

    if (!synthetic && !streamer.start()) {
        std::fprintf(stderr, "[remora] failed to start layer streamer\n");
        return 1;
    }

    std::printf("[remora] executing %zu layers (O(1) state, double-buffered I/O)\n",
                streamer.layer_count());

    std::size_t total_bytes = 0;
    if (synthetic) {
        // Synthetic run: no real files, just exercise the state recurrence.
        for (std::size_t i = 0; i < layer_paths.size(); ++i) {
            remora::LayerContext layer;
            layer.weights = nullptr;
            layer.weights_size = 0;
            layer.state = ctx.hidden_state;
            total_bytes += run_layer_forward(ctx, layer);
            std::printf("[remora]   layer %zu/%zu done (synthetic)\n", i + 1,
                        layer_paths.size());
        }
    } else {
        do {
            const remora::LayerBuffer& buf = streamer.active();
            remora::LayerContext layer;
            layer.weights = buf.data;
            layer.weights_size = buf.size;
            layer.state = ctx.hidden_state;  // output of layer N feeds layer N+1

            total_bytes += run_layer_forward(ctx, layer);
            std::printf("[remora]   layer %zu/%zu done (weights %zu bytes)\n",
                        streamer.current_index() + 1, streamer.layer_count(), buf.size);
        } while (streamer.advance());
    }

    std::printf("[remora] finished: %zu layers, %zu bytes of weights streamed\n",
                streamer.layer_count(), total_bytes);
    return 0;
}
