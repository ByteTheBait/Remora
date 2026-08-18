#include "layer_streamer.h"

#include <cstdio>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace remora {

// ---------------------------------------------------------------------------
// MappedFile
// ---------------------------------------------------------------------------

MappedFile::~MappedFile() { close(); }

bool MappedFile::open(const std::string& path) {
    close();

#ifdef _WIN32
    HANDLE file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                              nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        std::fprintf(stderr, "[remora] failed to open %s\n", path.c_str());
        return false;
    }
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(file, &sz)) {
        CloseHandle(file);
        return false;
    }
    HANDLE map = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (map == nullptr) {
        CloseHandle(file);
        return false;
    }
    void* base = MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0);
    if (base == nullptr) {
        CloseHandle(map);
        CloseHandle(file);
        return false;
    }
    file_handle_ = file;
    map_handle_ = map;
    data_ = base;
    size_ = static_cast<std::size_t>(sz.QuadPart);
    return true;
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "[remora] failed to open %s\n", path.c_str());
        return false;
    }
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        return false;
    }
    if (st.st_size == 0) {
        ::close(fd);
        return false;
    }
    void* base = ::mmap(nullptr, static_cast<std::size_t>(st.st_size),
                        PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) {
        std::fprintf(stderr, "[remora] mmap failed for %s\n", path.c_str());
        return false;
    }
    // Advise the kernel that we will read this sequentially and only once.
    ::madvise(base, static_cast<std::size_t>(st.st_size), MADV_SEQUENTIAL);
    data_ = base;
    size_ = static_cast<std::size_t>(st.st_size);
    return true;
#endif
}

void MappedFile::close() {
    if (data_ == nullptr) return;
#ifdef _WIN32
    UnmapViewOfFile(data_);
    if (map_handle_) CloseHandle(map_handle_);
    if (file_handle_) CloseHandle(file_handle_);
    map_handle_ = nullptr;
    file_handle_ = nullptr;
#else
    ::munmap(data_, size_);
#endif
    data_ = nullptr;
    size_ = 0;
}

// ---------------------------------------------------------------------------
// LayerStreamer — true sliding window
// ---------------------------------------------------------------------------

LayerStreamer::LayerStreamer(int buffer_layers)
    : buffer_layers_(buffer_layers < 0 ? 0 : buffer_layers) {}

LayerStreamer::~LayerStreamer() = default;

void LayerStreamer::set_layer_paths(std::vector<std::string> paths) {
    paths_ = std::move(paths);
    maps_.clear();
    maps_.resize(paths_.size());
    resident_.clear();
    resident_.resize(paths_.size());
    current_ = 0;
    started_ = false;
}

bool LayerStreamer::map_layer(std::size_t index) {
    if (index >= paths_.size()) return false;
    if (maps_[index].is_open()) return true;  // already mapped
    if (!maps_[index].open(paths_[index])) return false;
    resident_[index].path = paths_[index];
    resident_[index].size = maps_[index].size();
    resident_[index].data = maps_[index].data();
    resident_[index].resident = true;
    return true;
}

void LayerStreamer::unmap_layer(std::size_t index) {
    if (index >= maps_.size()) return;
    maps_[index].close();
    resident_[index].data = nullptr;
    resident_[index].size = 0;
    resident_[index].resident = false;
}

bool LayerStreamer::start() {
    if (paths_.empty()) return false;
    // Map the first layer and the next `buffer_layers` lookahead.
    const std::size_t n_initial =
        std::min(static_cast<std::size_t>(buffer_layers_) + 1, paths_.size());
    for (std::size_t i = 0; i < n_initial; ++i) {
        if (!map_layer(i)) return false;
    }
    current_ = 0;
    started_ = true;
    return true;
}

bool LayerStreamer::reset() {
    if (paths_.empty()) return false;
    // Unmap everything and re-map the initial window.
    for (std::size_t i = 0; i < maps_.size(); ++i) {
        unmap_layer(i);
    }
    const std::size_t n_initial =
        std::min(static_cast<std::size_t>(buffer_layers_) + 1, paths_.size());
    for (std::size_t i = 0; i < n_initial; ++i) {
        if (!map_layer(i)) return false;
    }
    current_ = 0;
    started_ = true;
    return true;
}

bool LayerStreamer::advance() {
    if (!started_) return false;
    const std::size_t next = current_ + 1;
    if (next >= paths_.size()) return false;

    // Unmap the layer that just finished.
    unmap_layer(current_);

    // Stream in the layer `buffer_layers` positions ahead of the new
    // running layer (if any). This is what keeps the window at
    // (buffer_layers + 1) resident layers.
    const std::size_t lookahead = next + static_cast<std::size_t>(buffer_layers_);
    if (lookahead < paths_.size()) {
        if (!map_layer(lookahead)) return false;
    }

    current_ = next;
    return true;
}

std::size_t LayerStreamer::resident_count() const {
    std::size_t n = 0;
    for (const auto& r : resident_) {
        if (r.resident) ++n;
    }
    return n;
}

}  // namespace remora
