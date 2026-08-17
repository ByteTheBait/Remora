#include "tokenizer.h"

#include <cstdio>
#include <cstring>

namespace remora {

namespace {

// BPE inner loop: take a list of single-byte tokens and repeatedly merge the
// adjacent pair with the lowest rank (highest priority). This is the standard
// Sennrich-style BPE algorithm.
//
// `pieces` is mutated in place. We never insert/remove from the front, so we
// just keep scanning forward until no more merges are possible.
void bpe_merge(std::vector<std::string>& pieces,
               const std::unordered_map<std::string, int>& ranks) {
    while (pieces.size() >= 2) {
        int best_rank = std::numeric_limits<int>::max();
        int best_idx = -1;
        for (size_t i = 0; i + 1 < pieces.size(); ++i) {
            auto it = ranks.find(pieces[i] + " " + pieces[i + 1]);
            if (it != ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = static_cast<int>(i);
            }
        }
        if (best_idx < 0) break;
        pieces[best_idx] = pieces[best_idx] + pieces[best_idx + 1];
        pieces.erase(pieces.begin() + best_idx + 1);
    }
}

// Splits `text` into segments where leading whitespace (and its GPT-2 Ġ
// rewriting) lives in the prefix. e.g. "Hello,  my" -> ["Hello,", "ĠĠ", "my"].
std::vector<std::string> pretok_olmo(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    enum class Kind { None, Word, Space };
    Kind k = Kind::None;
    auto flush = [&]() {
        if (!cur.empty()) {
            out.push_back(std::move(cur));
            cur.clear();
        }
    };
    for (char c : text) {
        if (c == ' ') {
            if (k == Kind::Word) { flush(); }
            // UTF-8 encoding of U+0120 ("Ġ") is 0xC4 0xA0.
            cur.push_back('\xC4');
            cur.push_back('\xA0');
            k = Kind::Space;
        } else {
            if (k == Kind::Space) { flush(); }
            cur.push_back(c);
            k = Kind::Word;
        }
    }
    flush();
    return out;
}

}  // namespace

bool BPETokenizer::load(const std::vector<std::string>& vocab,
                        const std::vector<std::string>& merges,
                        int bos_id, int eos_id) {
    vocab_ = vocab;
    bos_ = bos_id;
    eos_ = eos_id;
    token_to_id_.reserve(vocab_.size() * 2);
    for (size_t i = 0; i < vocab_.size(); ++i) {
        token_to_id_[vocab_[i]] = static_cast<int32_t>(i);
    }
    merge_ranks_.reserve(merges.size() * 2);
    for (size_t i = 0; i < merges.size(); ++i) {
        merge_ranks_[merges[i]] = static_cast<int>(i);
    }
    return true;
}

std::vector<int32_t> BPETokenizer::encode(const std::string& text) const {
    std::vector<int32_t> ids;
    auto segs = pretok_olmo(text);
    for (const auto& seg : segs) {
        // Initial pieces: one byte per Unicode codepoint (simplified: split
        // by UTF-8 byte for non-ASCII safety).
        std::vector<std::string> pieces;
        for (size_t i = 0; i < seg.size(); ) {
            unsigned char c = static_cast<unsigned char>(seg[i]);
            int n = 1;
            if      (c < 0x80) n = 1;
            else if ((c & 0xE0) == 0xC0) n = 2;
            else if ((c & 0xF0) == 0xE0) n = 3;
            else if ((c & 0xF8) == 0xF0) n = 4;
            pieces.emplace_back(seg.substr(i, n));
            i += n;
        }
        bpe_merge(pieces, merge_ranks_);
        for (const auto& p : pieces) {
            auto it = token_to_id_.find(p);
            if (it != token_to_id_.end()) {
                ids.push_back(it->second);
            } else {
                // Unknown byte-level chunk: emit a byte-fallback token if
                // present, otherwise skip. The vocab typically has individual
                // bytes 0..255 as fallback tokens.
                std::fprintf(stderr,
                    "[remora] warn: no vocab entry for byte '%s'\n", p.c_str());
            }
        }
    }
    return ids;
}

std::string BPETokenizer::decode_token(int32_t id) const {
    if (id < 0 || id >= static_cast<int32_t>(vocab_.size())) return {};
    std::string s = vocab_[id];
    // Reverse the GPT-2 byte-level encoding: "Ġ" (0xC4 0xA0) -> ' '.
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c == 0xC4 && i + 1 < s.size() &&
            static_cast<unsigned char>(s[i + 1]) == 0xA0) {
            out.push_back(' ');
            i += 2;
        } else if (c < 0x80) {
            out.push_back(s[i]);
            ++i;
        } else {
            // Pass through multi-byte UTF-8 unchanged.
            int n = 1;
            if      ((c & 0xE0) == 0xC0) n = 2;
            else if ((c & 0xF0) == 0xE0) n = 3;
            else if ((c & 0xF8) == 0xF0) n = 4;
            out.append(s, i, n);
            i += n;
        }
    }
    return out;
}

}  // namespace remora