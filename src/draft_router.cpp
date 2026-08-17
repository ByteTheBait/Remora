#include "draft_router.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

namespace remora {

namespace {
// Minimal JSON value parser sufficient for the routing table format:
//   { "default": "expert_chat_7b",
//     "rules": [ {"task":"code","expert":"expert_code_7b","pool":"layers/expert_code_7b"} ] }
// This keeps the router dependency-free; a production build can swap in a
// full JSON library.
class MiniJson {
public:
    explicit MiniJson(const std::string& s) : s_(s), i_(0) {}

    bool parse() {
        // Skip a UTF-8 BOM and any leading null bytes.
        while (i_ < s_.size() &&
               ((unsigned char)s_[i_] == 0xEF || (unsigned char)s_[i_] == 0x00)) {
            ++i_;
        }
        skip_ws();
        if (peek() != '{') return false;
        ++i_;
        while (true) {
            skip_ws();
            if (peek() == '}') { ++i_; break; }
            std::string key = parse_string();
            if (key.empty()) return false;
            skip_ws();
            if (peek() != ':') return false;
            ++i_;
            skip_ws();
            if (key == "default") {
                default_rule = parse_string();
            } else if (key == "rules") {
                if (!parse_rules()) return false;
            } else {
                skip_value();
            }
            skip_ws();
            if (peek() == ',') { ++i_; continue; }
            if (peek() == '}') { ++i_; break; }
            return false;
        }
        return true;
    }

    std::string default_rule;
    std::vector<RoutingRule> rules;

private:
    void skip_ws() { while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) ++i_; }
    char peek() { return i_ < s_.size() ? s_[i_] : '\0'; }

    std::string parse_string() {
        if (peek() != '"') return {};
        ++i_;
        std::string out;
        while (i_ < s_.size() && s_[i_] != '"') {
            if (s_[i_] == '\\' && i_ + 1 < s_.size()) { ++i_; }
            out.push_back(s_[i_++]);
        }
        if (peek() == '"') ++i_;
        return out;
    }

    bool parse_rules() {
        if (peek() != '[') return false;
        ++i_;
        while (true) {
            skip_ws();
            if (peek() == ']') { ++i_; break; }
            if (peek() != '{') return false;
            ++i_;  // consume '{'
            RoutingRule r;
            while (true) {
                skip_ws();
                if (peek() == '}') { ++i_; break; }
                std::string key = parse_string();
                if (key.empty()) return false;
                skip_ws();
                if (peek() != ':') return false;
                ++i_;
                skip_ws();
                std::string val = parse_string();
                if (key == "task") r.task = val;
                else if (key == "expert") r.expert_id = val;
                else if (key == "pool") r.pool_path = val;
                skip_ws();
                if (peek() == ',') { ++i_; continue; }
                if (peek() == '}') { ++i_; break; }
                return false;
            }
            rules.push_back(std::move(r));
            skip_ws();
            if (peek() == ',') { ++i_; continue; }
            if (peek() == ']') { ++i_; break; }
            return false;
        }
        return true;
    }

    void skip_value() {
        if (peek() == '"') { parse_string(); return; }
        if (peek() == '{' || peek() == '[') {
            int depth = 0;
            do {
                if (peek() == '{' || peek() == '[') ++depth;
                else if (peek() == '}' || peek() == ']') --depth;
                ++i_;
            } while (depth > 0 && i_ < s_.size());
            return;
        }
        while (i_ < s_.size() && peek() != ',' && peek() != '}') ++i_;
    }

    const std::string& s_;
    std::size_t i_;
};
}  // namespace

bool DraftRouter::load_routing(const std::string& json_path) {
    std::ifstream f(json_path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[remora] cannot open routing file %s\n", json_path.c_str());
        return false;
    }
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    MiniJson json(content);
    if (!json.parse()) {
        std::fprintf(stderr, "[remora] failed to parse routing file %s\n", json_path.c_str());
        return false;
    }
    rules_ = std::move(json.rules);
    default_pool_ = json.default_rule;
    return true;
}

bool DraftRouter::load_draft_model(const std::string& model_path) {
    // In a full build this hosts a GGML runner for the ~300M draft model.
    // Here we just record that a model was provided.
    draft_loaded_ = !model_path.empty();
    return draft_loaded_;
}

std::string DraftRouter::route(const std::vector<int32_t>& tokens) {
    // Task classification from the draft model's logits. The real
    // implementation reads the top-k logits and maps them to a task class.
    // Here we use a trivial heuristic on the token stream so the pipeline
    // runs end-to-end without a heavyweight dependency.
    std::string task = "chat";
    for (int32_t t : tokens) {
        if (t == 0) { task = "code"; break; }      // placeholder: token 0 => code
        if (t == 1) { task = "math"; break; }      // placeholder: token 1 => math
    }
    for (const auto& r : rules_) {
        if (r.task == task) return r.pool_path;
    }
    return default_pool_;
}

}  // namespace remora
