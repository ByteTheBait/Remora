#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace remora {

// A single routing rule: a task class maps to an expert model pool.
struct RoutingRule {
    std::string task;      // e.g. "code", "math", "chat"
    std::string expert_id; // e.g. "expert_code_7b"
    std::string pool_path; // path to the sharded expert layers
};

// The draft model router.
//
// A tiny (~300M) draft model stays permanently resident in memory. It runs
// cheaply on every token and emits a task classification, which is mapped
// through a JSON routing table to an expert ID. The expert's layers are then
// streamed from disk by the LayerStreamer.
class DraftRouter {
public:
    DraftRouter() = default;

    // Load the routing table from a JSON file (see examples/basic_routing.json).
    bool load_routing(const std::string& json_path);

    // Run the embedded draft model over the given token ids and return the
    // selected expert pool path. Falls back to the default rule if none match.
    std::string route(const std::vector<int32_t>& tokens);

    // The draft model itself. In a full build this hosts a GGML runner for a
    // 300M model. Here we keep a lightweight, dependency-free classifier.
    bool load_draft_model(const std::string& model_path);

    const std::vector<RoutingRule>& rules() const { return rules_; }

private:
    std::vector<RoutingRule> rules_;
    std::string default_pool_;
    bool draft_loaded_ = false;
};

}  // namespace remora
