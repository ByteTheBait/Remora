#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace remora {

// ---------------------------------------------------------------------------
// Minimal GPT-2-style BPE tokenizer, tailored to Mamba's GGUF vocab format.
//
// The GGUF stores:
//   - tokenizer.ggml.tokens      : array of strings, the vocab
//   - tokenizer.ggml.merges      : array of strings, "left right" BPE merges
//   - tokenizer.ggml.scores      : optional, used by some tokenizers for
//                                 priority; we ignore it (BPE only needs merges)
//   - tokenizer.ggml.token_type  : optional int32 array, used by llama.cpp
//                                 to flag special tokens; we treat all
//                                 tokens as normal unless their ID == eos.
//
// The "olmo" pre-tokenizer splits text on a regex that approximates GPT-2's
// behaviour for English + code. We use a simplified version: split on
// whitespace boundaries, then for each chunk convert leading whitespace to
// the G-prefix (Ġ = space byte) as GPT-2 does.
// ---------------------------------------------------------------------------
class BPETokenizer {
public:
    bool load(const std::vector<std::string>& vocab,
             const std::vector<std::string>& merges,
             int bos_id, int eos_id);

    // Encode a UTF-8 string into a sequence of token IDs. Uses the BPE
    // algorithm: greedily merge adjacent pairs according to merge priority.
    // For simplicity (and because Mamba uses "olmo" pre-tokenization which
    // is whitespace-prefix based), we do NOT run a full regex pretok; we
    // split on whitespace and rejoin leading-space-prefixes as Ġ-prefixed
    // tokens. This is a faithful-enough approximation for short English
    // prompts like "Hello, my name is".
    std::vector<int32_t> encode(const std::string& text) const;

    // Decode one token ID to its raw string (with byte-level UTF-8
    // preservation: "Ġ" -> " ", etc).
    std::string decode_token(int32_t id) const;

    int vocab_size() const { return static_cast<int>(vocab_.size()); }
    int bos_id() const { return bos_; }
    int eos_id() const { return eos_; }

private:
    std::vector<std::string> vocab_;
    // merge priority: pair string "left right" -> rank (0 = highest priority)
    std::unordered_map<std::string, int> merge_ranks_;
    // token string -> id (for BPE inner loop)
    std::unordered_map<std::string, int32_t> token_to_id_;
    int bos_ = 0;
    int eos_ = 0;
};

}  // namespace remora