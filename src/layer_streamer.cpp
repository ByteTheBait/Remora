#include "layer_streamer.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

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
// LayerStreamer
// ---------------------------------------------------------------------------

LayerStreamer::LayerStreamer(std::size_t /*buffer_capacity*/) {}

LayerStreamer::~LayerStreamer() = default;

void LayerStreamer::set_layer_paths(std::vector<std::string> paths) {
    paths_ = std::move(paths);
    maps_.clear();
    maps_.resize(paths_.size());
    current_ = 0;
    started_ = false;
}

bool LayerStreamer::load_into(LayerBuffer& buf, std::size_t index) {
    if (index >= paths_.size()) return false;
    if (!maps_[index].open(paths_[index])) return false;
    buf.path = paths_[index];
    buf.size = maps_[index].size();
    buf.data = maps_[index].data();
    buf.resident = true;
    return true;
}

bool LayerStreamer::start() {
    if (paths_.empty()) return false;
    // Preload the first layer into buffer A.
    if (!load_into(buffer_a_, 0)) return false;
    active_ = buffer_a_;
    current_ = 0;
    started_ = true;
    return true;
}

bool LayerStreamer::advance() {
    if (!started_) return false;
    std::size_t next = current_ + 1;
    if (next >= paths_.size()) return false;

    // The buffer that just finished computing is reused for the next read.
    // If the active buffer is A, stream into B and vice versa.
    LayerBuffer& free_buf =
        (active_.data == buffer_a_.data) ? buffer_b_ : buffer_a_;

    if (free_buf.resident) {
        // Release the previous mapping for this slot.
        std::size_t old_idx = (free_buf.data == buffer_a_.data) ? current_ : current_;
        (void)old_idx;
        free_buf.resident = false;
    }

    if (!load_into(free_buf, next)) return false;
    active_ = free_buf;
    current_ = next;
    return true;
}

}  // namespace remora
