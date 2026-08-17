cmake_minimum_required(VERSION 3.16)
project(Remora LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)

# Optional: link against a local ggml / llama.cpp build in third_party.
# When GGML_DIR is provided, the RNN kernels dispatch to real GGML math.
set(GGML_DIR "" CACHE PATH "Path to a ggml source tree (optional)")
if(GGML_DIR)
    add_subdirectory(${GGML_DIR} third_party/ggml)
    set(REMORA_HAVE_GGML ON)
else()
    set(REMORA_HAVE_GGML OFF)
endif()

add_library(remora_core STATIC
    src/layer_streamer.cpp
    src/draft_router.cpp
    src/rnn_kernel.cpp
)
target_include_directories(remora_core PUBLIC src)
if(REMORA_HAVE_GGML)
    target_link_libraries(remora_core PUBLIC ggml)
    target_compile_definitions(remora_core PUBLIC REMORA_HAVE_GGML=1)
endif()

add_executable(remora src/main.cpp)
target_link_libraries(remora PRIVATE remora_core)

install(TARGETS remora RUNTIME DESTINATION bin)
