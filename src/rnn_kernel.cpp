#include "rnn_kernel.h"

#include <cstring>

namespace remora {

// Minimal placeholder forward pass.
//
// In the full implementation this calls into GGML custom kernels for
// Mamba / RWKV-style layers (selective scan, gated linear recurrences).
// The key property preserved here is the O(1) state compaction: the hidden
// state is a fixed-size vector that is transformed in place, never growing
// with sequence length.
std::size_t run_layer_forward(RuntimeContext& ctx, const LayerContext& layer) {
    if (ctx.hidden_state.empty() && layer.state.size() > 0) {
        ctx.hidden_state = layer.state;
        ctx.state_dim = layer.state.size();
    }

    // Simulate a stateful recurrence: mix the incoming layer state into the
    // persistent hidden state. This is where a real kernel would run the
    // selective-scan / linear-recurrence math.
    const std::size_t n = ctx.hidden_state.size();
    const float* in = layer.state.empty() ? nullptr : layer.state.data();
    for (std::size_t i = 0; i < n; ++i) {
        float x = in ? in[i] : 0.0f;
        // Simple leaky recurrence: h = 0.9 * h + 0.1 * x
        ctx.hidden_state[i] = 0.9f * ctx.hidden_state[i] + 0.1f * x;
    }

    return layer.weights_size;
}

}  // namespace remora
