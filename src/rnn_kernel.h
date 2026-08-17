#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace remora {

// A single RNN layer's forward pass. In a real build this dispatches to
// GGML kernels (see rnn_kernel.cpp). Here we keep a minimal, dependency-free
// contract so the execution loop can be exercised end-to-end.
struct LayerContext {
    // Hidden state vector (the RNN state compaction). O(1) in sequence
    // length: it never grows with the number of tokens processed.
    std::vector<float> state;

    // Layer weights, memory-mapped from the sharded payload.
    const void* weights = nullptr;
    std::size_t weights_size = 0;
};

// Global runtime context holding the persistent RNN hidden state.
struct RuntimeContext {
    std::size_t state_dim = 0;
    std::vector<float> hidden_state;
    std::size_t layer_count = 0;
    std::size_t current_layer = 0;
};

// Run one forward pass for the given layer, updating the shared hidden state.
// Returns the number of bytes of weights consumed.
std::size_t run_layer_forward(RuntimeContext& ctx, const LayerContext& layer);

}  // namespace remora
