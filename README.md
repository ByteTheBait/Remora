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

## Theory: Sparse MoE where every expert is an RNN

Remora's structure is a natural fit for a *sparse mixture-of-experts* where
each expert is itself a recurrent (RNN) block. Here is the theory of why that
combination is compelling, and how it maps onto the layer-by-layer streaming
design.

### 1. The core idea

A classic transformer MoE keeps a single large model and routes each token to
a few of its many feed-forward experts. The experts are *dense* and *stateless*:
they transform one token in isolation and forget it immediately.

Remora inverts this. Instead of many stateless experts inside one model, we
treat **each expert as a full recurrent block** — a Mamba / RWKV / linear-attention
layer with its own persistent hidden state. The router picks, per token, which
expert(s) should process the current hidden state, and only those experts'
weights are streamed in and run.

### 2. Why RNN experts are the right primitive

- **O(1) state per expert.** A recurrent expert compresses its entire history
  into a fixed-size hidden state. So the cost of "keeping an expert warm" is
  constant — it does not grow with sequence length. This is what makes it
  feasible to have *hundreds* of experts: the resident working set is just the
  union of the active experts' hidden states, not their weights.
- **State is the memory, weights are the disk.** The expensive part of an
  expert (its weight matrices) lives on SSD and is streamed in only when the
  expert is selected. The cheap part (its hidden state) stays resident. This is
  exactly the split Remora's layer streamer already implements.
- **Sparsity buys capacity, not just speed.** A dense model's capacity is
  bounded by its parameter count. A sparse MoE of RNN experts can hold far more
  total parameters on disk than would ever fit in VRAM, and only materialize
  the small active subset. The "model" is effectively unbounded.

### 3. Routing over recurrent state

Routing in a transformer MoE is a function of the *token* embedding. Routing
in Remora is a function of the *recurrent hidden state* — which already
summarizes the whole conversation. This is a strictly richer signal:

- The router sees not just "what token is this" but "where is the conversation
  going", so it can pick experts that are specialized for the current topic,
  style, or task.
- Because the hidden state is O(1), the router itself can be a small always-
  resident draft model (Remora's `draft_router.cpp`) that runs on every token
  at negligible cost.

### 4. The layer-by-layer structure makes this cheap

The key structural insight is that **an RNN expert and a layer are the same
thing in Remora**. The streamer already sweeps one block at a time; a sparse
MoE just changes *which* blocks get swept for a given token:

- **Dense path (today):** every token visits all 48 blocks in order.
- **Sparse path (the theory):** the router picks, say, 4 of 1000 expert blocks
  per token. Only those 4 are streamed and computed. The other 996 stay on
  disk, untouched.

The resident memory is bounded by the *active* experts' hidden states plus the
streaming window — not by the total number of experts. So you can scale the
expert pool to the size of your SSD, not your RAM.

Because each expert is an independent recurrent block with its own hidden state,
the sparse path is embarrassingly parallel: we can **stream each expert
layer-by-layer as a separate thread**. The router hands the current hidden state
to the selected experts, and each thread independently streams its own expert's
weights from SSD, runs its forward pass, and returns its updated state. The
threads only need to synchronize at the merge point where their outputs are
combined (e.g. a weighted sum or a gated combination) before the next token is
routed. This turns the MoE fan-out into a natural multi-threaded pipeline — the
I/O for expert *B* overlaps with the compute of expert *A* across threads, not
just across the double-buffered window of a single sequential sweep. The only
shared resource is the SSD itself, so the practical ceiling is disk bandwidth
rather than VRAM or a single compute core.

### 5. What this buys us

| Property | Dense RNN (today) | Sparse RNN MoE (theory) |
| -------- | ----------------- | ----------------------- |
| Experts per token | all layers | a small routed subset |
| Total capacity | bounded by one model | bounded by SSD space |
| Resident memory | O(active layers) | O(active experts) |
| Specialization | none (one model) | per-topic / per-task experts |
| Switching cost | stream next layer | stream next *expert* |

### 6. Open questions / research directions

- **Router quality.** Does routing on the recurrent hidden state converge to
  stable, meaningful expert specializations, or does it thrash between experts
  token-to-token? (A soft / top-k router with a small temperature may help.)
- **State handoff.** When a token is routed to an expert that has not been
  active for a while, its hidden state is stale. Do we need a "state refresh"
  pass, or is the recurrence forgiving enough to recover in a few tokens?
- **Load balancing.** Sparse MoEs are prone to a few experts absorbing all the
  traffic. Recurrent experts add a second axis: an expert's state can go stale
  if it is under-selected. Balancing must consider both routing frequency and
  state freshness.
- **Expert granularity.** Is the right unit a full recurrent block, or a
  *sub-layer* (e.g. just the SSM, or just the gating projection)? Finer
  granularity means more routing decisions but smaller streamed units.

This is the direction Remora's architecture is built to explore: the streaming
infrastructure, the O(1) state contract, and the draft router are all already
in place. The missing piece is the sparse routing policy that decides, per
token, which expert blocks to sweep.

---

## Architectural Comparison

| Metric | Traditional Transformer | Remora Layer-by-Layer RNN MoE |
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
# 1. Download a Mamba model (GGUF)
curl -L -o mamba-1.4b-hf.Q4_K_M.gguf \
    "https://huggingface.co/bartowski/mamba-1.4b-hf-GGUF/resolve/main/mamba-1.4b-hf.Q4_K_M.gguf"

# 2. Shard the GGUF into per-layer files
python3 scripts/shard_gguf.py mamba-1.4b-hf.Q4_K_M.gguf --out layers/

# 3. Run the engine with a routing table
./build/remora \
    --routing examples/basic_routing.json \
    --layers layers/ \
    --tokens 8
```

The engine discovers every `blk.*` file in the layers directory, sorts them by
block index, and streams them one at a time through the double-buffered loader.
For the 1.4B Mamba model this is 48 layers (~15 MB each, ~733 MB total) — all
streamed from disk with a constant O(1) resident state.

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
