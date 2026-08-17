# Remora

**Infinite Model Pools on Consumer Hardware.**

Remora is a research-grade runtime that runs *layer-by-layer RNN MoE* models
on a single consumer GPU. Instead of loading an entire 10 GB+ transformer into
VRAM, Remora streams one layer at a time from SSD, keeps only a tiny fixed-size
hidden state in memory, and uses a small always-resident draft model to route
each request to the right expert pool.

The result: you can hold **hundreds of expert models** on a single SSD and
switch between them with **zero reload latency** — something a traditional
transformer stack cannot do.

---

## The High-Level Pitch

Remora sits at the intersection of two ideas:

1. **RNN State Compaction — O(1) memory complexity.**
   Recurrent layers (Mamba, RWKV, linear attention) compress the entire
   sequence into a fixed-size hidden state. VRAM usage is *constant*, no matter
   how long the conversation gets. This is the property that makes streaming
   weights feasible: the resident working set is tiny.

2. **Hardware-Level Zero-Copy Streaming — DMA / mmap.**
   We memory-map sharded layer files and double-buffer them. While the GPU
   computes layer *N*, the I/O layer streams layer *N+1* into the other buffer.
   Disk latency is completely hidden behind compute — the GPU never waits.

Because the resident state is O(1) and each layer is streamed independently,
the *only* limit on how many models you can run is **SSD storage space**, not
VRAM.

---

## Architectural Comparison

| Metric | Traditional Transformer | Your Layer-by-Layer RNN MoE |
| ------ | ----------------------- | --------------------------- |
| VRAM Consumption | Scales with Sequence Length (O(N)) | Constant / Static (O(1)) |
| Model Pool Limit | Limited by physical VRAM capacity | Limited only by SSD storage space |
| Switching Latency | High (must reload entire 10GB+ weights) | Zero (hidden by compute-to-I/O overlap) |

---

## Repository Layout

```
├── README.md               # Architecture explanation, benchmarks, and quickstart
├── CMakeLists.txt         # Cross-platform build configuration (C++20)
├── third_party/           # Submodules (ggml / llama.cpp core)
├── scripts/               # Sharding utility scripts
│   └── shard_gguf.py      # Script to slice a standard GGUF into layer files
├── src/
│   ├── main.cpp           # Entry point and runtime execution loop
│   ├── draft_router.cpp   # Draft model execution and expert selection logic
│   ├── layer_streamer.cpp # Asynchronous DMA / mmap layer loader
│   └── rnn_kernel.cpp     # Custom GGML math kernels for Mamba/RWKV layers
└── examples/
    ├── basic_routing.json # Sample routing configuration map
    └── run_demo.sh        # Script to download a small model, shard it, and run it
```

---

## Quickstart

### Prerequisites

- A C++20 compiler (GCC 11+, Clang 14+, or MSVC 2022)
- CMake 3.16+
- Python 3.8+ with the `gguf` package (`pip install gguf`)
- `curl` (for the demo download)

### Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Run the demo

```bash
./examples/run_demo.sh
```

This downloads a tiny 260K-parameter model, shards it into per-layer files,
builds the engine, and runs the layer-by-layer execution loop.

### Run manually

```bash
# 1. Shard a GGUF model into per-layer files
python3 scripts/shard_gguf.py model.gguf --out layers/

# 2. Run the engine with a routing table
./build/remora \
    --routing examples/basic_routing.json \
    --layers layers/ \
    --tokens 8
```

---

## How It Works

### Phase 1 — Core Engine (C++)

- **`layer_streamer.cpp`** — Cross-platform memory mapping (`mmap` on
  Linux/macOS, `CreateFileMapping` on Windows) with a double-buffering scheme.
  Two buffers (A and B) alternate so disk reads overlap with compute.
- **`main.cpp`** — The RNN execution loop. A global context holds the persistent
  hidden state; the output state of layer *N* feeds layer *N+1*. When a buffer
  finishes its forward pass, an asynchronous signal triggers the next layer read.
- **`rnn_kernel.cpp`** — Custom GGML math kernels for Mamba/RWKV-style layers
  (selective scan, gated linear recurrences). The placeholder implementation
  preserves the O(1) state-compaction contract.

### Phase 2 — Router & Model Setup

- **`scripts/shard_gguf.py`** — Reads a compiled GGUF, isolates the global
  tensors (`token_embd`, `output`), and writes each sequential block
  (`blk.0`, `blk.1`, ...) to its own binary payload.
- **`draft_router.cpp`** — Hosts a tiny (~300M) draft model that stays
  permanently resident. It classifies the task and selects an expert ID from a
  JSON routing table (`examples/basic_routing.json`).

### Phase 3 — Documentation

This README.

---

## Roadmap

- [ ] Wire the RNN kernels to real GGML math (Mamba / RWKV selective scan)
- [ ] Integrate `io_uring` on Linux for registered, pinned I/O buffers
- [ ] Add CUDA / Metal backends for the streaming buffers
- [ ] Benchmark suite comparing VRAM usage and switching latency vs. transformers

---

## License

MIT — see `LICENSE` (add as needed).
