#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
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

// True sliding-window layer streamer.
//
// Only `buffer_layers + 1` layers are memory-mapped at any moment: the
// layer currently being computed plus the next `buffer_layers` layers as
// lookahead (hiding disk latency). As the running layer finishes it is
// immediately unmapped and the layer `buffer_layers` positions ahead is
// streamed in. This keeps peak RSS bounded to ~(buffer_layers + 1) layer
// shards regardless of model depth.
//
//   --buffer-layers 0  : only the running layer is resident
//   --buffer-layers 1  : running + 1 lookahead (default)
//   --buffer-layers N  : running + N lookahead
class LayerStreamer {
public:
    explicit LayerStreamer(int buffer_layers);
    ~LayerStreamer();

    // Register the ordered list of layer files to stream.
    void set_layer_paths(std::vector<std::string> paths);

    // Begin streaming: map the first layer and the next buffer_layers.
    bool start();

    // Advance the running layer by one. Unmaps the layer that just
    // finished and streams in the next lookahead layer.
    // Returns false when there are no more layers.
    bool advance();

    // Reset the streamer back to the first layer so a fresh forward pass
    // can sweep all layers again. Unmaps everything and re-maps the first
    // (buffer_layers + 1) layers.
    bool reset();

    // Pointer to the currently active (ready-to-compute) buffer.
    const LayerBuffer& active() const { return resident_[current_]; }

    std::size_t layer_count() const { return paths_.size(); }
    std::size_t current_index() const { return current_; }

    // Number of layers currently resident (for logging / verification).
    std::size_t resident_count() const;

    int buffer_layers() const { return buffer_layers_; }

private:
    // Map a single layer index. Returns false on failure.
    bool map_layer(std::size_t index);
    // Unmap a single layer index.
    void unmap_layer(std::size_t index);

    std::vector<std::string> paths_;
    std::vector<MappedFile> maps_;
    std::vector<LayerBuffer> resident_;
    int buffer_layers_;
    std::size_t current_ = 0;
    bool started_ = false;
};

}  // namespace remora
