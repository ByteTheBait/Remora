#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace remora {

// A single sharded layer payload, memory-mapped from disk.
struct LayerBuffer {
    std::string path;          // source file on disk
    std::size_t size = 0;      // payload size in bytes
    void* data = nullptr;      // mapped base pointer
    bool resident = false;     // whether the mapping is currently active
};

// Cross-platform memory mapping abstraction.
//   - POSIX (Linux / macOS): mmap + madvise
//   - Windows: CreateFileMapping + MapViewOfFile
class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    MappedFile(MappedFile&& other) noexcept { *this = std::move(other); }
    MappedFile& operator=(MappedFile&& other) noexcept {
        if (this != &other) {
            close();
            data_ = other.data_;
            size_ = other.size_;
#ifdef _WIN32
            file_handle_ = other.file_handle_;
            map_handle_ = other.map_handle_;
#endif
            other.data_ = nullptr;
            other.size_ = 0;
#ifdef _WIN32
            other.file_handle_ = nullptr;
            other.map_handle_ = nullptr;
#endif
        }
        return *this;
    }

    // Map the file at `path` read-only. Returns false on failure.
    bool open(const std::string& path);

    // Release the mapping (unmap + close).
    void close();

    void* data() const { return data_; }
    std::size_t size() const { return size_; }
    bool is_open() const { return data_ != nullptr; }

private:
    void* data_ = nullptr;
    std::size_t size_ = 0;
#ifdef _WIN32
    void* file_handle_ = nullptr;
    void* map_handle_ = nullptr;
#endif
};

// Double-buffered asynchronous layer loader.
//
// Two VRAM/CPU buffers (A and B) are alternated so that while the compute
// engine runs a forward pass on buffer A, the I/O layer streams the next
// layer into buffer B. This hides disk latency behind compute.
class LayerStreamer {
public:
    explicit LayerStreamer(std::size_t buffer_capacity);
    ~LayerStreamer();

    // Register the ordered list of layer files to stream.
    void set_layer_paths(std::vector<std::string> paths);

    // Begin streaming the first layer into the active buffer.
    bool start();

    // Advance to the next layer. The buffer that was just consumed is
    // immediately reused for the following layer's read.
    // Returns false when there are no more layers.
    bool advance();

    // Pointer to the currently active (ready-to-compute) buffer.
    const LayerBuffer& active() const { return active_; }

    std::size_t layer_count() const { return paths_.size(); }
    std::size_t current_index() const { return current_; }

private:
    bool load_into(LayerBuffer& buf, std::size_t index);

    std::vector<std::string> paths_;
    std::vector<MappedFile> maps_;
    LayerBuffer buffer_a_;
    LayerBuffer buffer_b_;
    LayerBuffer active_;
    std::size_t current_ = 0;
    bool started_ = false;
};

}  // namespace remora
