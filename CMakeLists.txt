cmake_minimum_required(VERSION 3.16)
project(Remora LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)

# ---------------------------------------------------------------------------
# ggml discovery
# ---------------------------------------------------------------------------
#
# Remora's RNN kernel dispatches to real GGML math (Mamba selective-scan,
# RWKV recurrences, RMS norm, etc). On macOS dev machines the simplest
# option is Homebrew's `ggml` + `llama.cpp` packages, which provide both
# the ggml headers and the ggml CPU/Metal backends (the CPU backend dylib
# lives under $(brew --cellar ggml)/libexec/ and is auto-loaded by
# `ggml_backend_load_all()` at startup).
#
# Discovery order:
#   1. -DGGML_DIR=<path>  : vendored ggml source tree (build via cmake).
#      The build will then auto-detect CPU/Metal backends via
#      `ggml_backend_load_all()` if dylibs sit next to the executable.
#   2. Homebrew `ggml` (pkg-config). On macOS we *also* link `llama` and
#      `ggml-base` from the llama.cpp bottle, because that's where the
#      ggml CPU/Metal op implementations actually live (ggml 0.17 ships
#      op implementations in the CPU backend dylib; the base ggml dylib
#      is mostly a backend loader).
#   3. (none)            : placeholder RNN kernel, no real math.
# ---------------------------------------------------------------------------

set(REMORA_HAVE_GGML OFF)

# We need at least the ggml .pc on macOS to pull in headers + base linkage.
find_package(PkgConfig QUIET)
set(GGML_HAVE_PKGCONFIG OFF)
if(PkgConfig_FOUND)
    pkg_check_modules(GGML IMPORTED_TARGET ggml)
    if(GGML_FOUND)
        set(GGML_HAVE_PKGCONFIG ON)
    endif()
endif()

if(GGML_DIR)
    message(STATUS "[remora] building ggml from ${GGML_DIR}")
    add_subdirectory(${GGML_DIR} third_party/ggml EXCLUDE_FROM_ALL)
    set(REMORA_HAVE_GGML ON)
elseif(GGML_HAVE_PKGCONFIG)
    message(STATUS "[remora] using system ggml ${GGML_VERSION} via pkg-config")
    set(REMORA_HAVE_GGML ON)
endif()

if(NOT REMORA_HAVE_GGML)
    message(WARNING
        "[remora] ggml not found. Building with placeholder RNN kernel only.\n"
        "         Install Homebrew 'ggml' or pass -DGGML_DIR=<path>.")
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------

add_library(remora_core STATIC
    src/layer_streamer.cpp
    src/draft_router.cpp
    src/rnn_kernel.cpp
    src/tokenizer.cpp
)
target_include_directories(remora_core PUBLIC src)

if(REMORA_HAVE_GGML)
    if(GGML_DIR)
        target_link_libraries(remora_core PUBLIC ggml)
    else()
        # System ggml on Homebrew: headers + libggml come from the ggml
        # pkgconfig. The actual ggml CPU/Metal ops live in libllama (from
        # the llama.cpp bottle) and libggml-base (from ggml itself).
        target_link_libraries(remora_core PUBLIC PkgConfig::GGML ggml-base llama)
        # On macOS Homebrew installs to /opt/homebrew/lib; ensure -l llama
        # and -l ggml-base can be found by the linker.
        if(APPLE)
            execute_process(
                COMMAND brew --prefix llama.cpp
                OUTPUT_VARIABLE REMORA_LLAMA_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _brew_rc)
            if(_brew_rc EQUAL 0)
                target_link_directories(remora_core PUBLIC
                    "${REMORA_LLAMA_PREFIX}/lib"
                    "${GGML_LIBDIR}")
            endif()
        endif()
    endif()
    target_compile_definitions(remora_core PUBLIC REMORA_HAVE_GGML=1)
endif()

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------

add_executable(remora src/main.cpp)
target_link_libraries(remora PRIVATE remora_core)

# Make sure the runtime can find the ggml CPU backend dylib. Without this,
# ggml_backend_load_all() finds zero backends and every op aborts.
if(REMORA_HAVE_GGML AND NOT GGML_DIR)
    if(APPLE)
        # Homebrew ggml ships its CPU backend under $(brew --cellar)/libexec/.
        # Add that directory to the rpath so the dynamic loader can find the
        # dylib at runtime.
        execute_process(
            COMMAND brew --cellar ggml
            OUTPUT_VARIABLE REMORA_GGML_CELLAR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _brew_rc
        )
        if(_brew_rc EQUAL 0)
            file(GLOB REMORA_GGML_LIBEXEC_DIRS
                LIST_DIRECTORIES true
                "${REMORA_GGML_CELLAR}/*/libexec")
            foreach(dir IN LISTS REMORA_GGML_LIBEXEC_DIRS)
                set_target_properties(remora PROPERTIES
                    BUILD_RPATH "${dir};${BUILD_RPATH}"
                    INSTALL_RPATH "${dir};${INSTALL_RPATH}")
            endforeach()
        endif()
    endif()
endif()

install(TARGETS remora RUNTIME DESTINATION bin)

# ---------------------------------------------------------------------------
# remora_llama — libllama-based driver that runs the model from sharded
# layer files using llama_model_init_from_user + a per-tensor callback.
# ---------------------------------------------------------------------------
#
# We don't build the vendored llama.cpp source — that's expensive and
# pointless since Homebrew ships libllama at /opt/homebrew/Cellar/llama.cpp/
# (same version as our vendored copy, b10180). We just consume the public
# C API through <llama.h> and link against -lllama.
#
# pkg-config llama gives us the include path + -lggml -lggml-base -lllama
# in one place. We require pkg-config to be available AND the Homebrew
# llama.cpp bottle to be installed. If either is missing, this target is
# skipped with a warning so the existing `remora` binary still builds.
# ---------------------------------------------------------------------------

if(PkgConfig_FOUND)
    pkg_check_modules(LLAMA IMPORTED_TARGET llama)
endif()
if(LLAMA_FOUND)
    message(STATUS "[remora] using system llama.cpp ${LLAMA_VERSION} via pkg-config")
    add_executable(remora_llama src/main_llama.cpp)
    # Homebrew's llama.cpp pkg-config is incomplete — it lists -lggml but
    # doesn't propagate the ggml include path. Add it explicitly so that
    # <llama.h> can find <ggml.h>.
    target_include_directories(remora_llama PRIVATE
        ${LLAMA_INCLUDE_DIRS}
        ${GGML_INCLUDE_DIRS})
    target_link_libraries(remora_llama PRIVATE PkgConfig::LLAMA)
    target_compile_options(remora_llama PRIVATE -Wall -Wextra)
    set_target_properties(remora_llama PROPERTIES
        BUILD_RPATH "${LLAMA_LIBRARY_DIRS};${BUILD_RPATH}"
        INSTALL_RPATH "${LLAMA_LIBRARY_DIRS};${INSTALL_RPATH}")
    install(TARGETS remora_llama RUNTIME DESTINATION bin)
else()
    message(WARNING
        "[remora] libllama not found via pkg-config. Skipping remora_llama.\n"
        "         Install Homebrew 'llama.cpp' to enable the libllama driver.")
endif()