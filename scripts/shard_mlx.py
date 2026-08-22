#!/usr/bin/env python3
"""Slice an MLX model (produced by scripts/convert_mlx.py) into per-layer
binary payloads with sidecars, in the same format as scripts/shard_gguf.py.

For each block we emit:

    layers/blk.N            raw tensor payloads concatenated (back-compat)
    layers/blk.N.meta.json  sidecar with per-tensor name/shape/dtype + offsets
    layers/global.bin       global tensor payloads concatenated
    layers/global.meta.json sidecar for globals
    layers/manifest.json    model architecture + tokenizer summary

The sidecar is what the C++ runtime reads to reconstruct ggml_tensor objects
from the raw byte stream. Without it the runtime has no way to know tensor
shapes, dtypes, or quantization parameters, so ggml cannot dequantize them.

Usage:
    python3 scripts/shard_mlx.py mlx_model/ --out layers/
"""

import argparse
import json
import os
import sys

try:
    import mlx.core as mx
except ImportError:
    sys.stderr.write(
        "error: the 'mlx' package is required. Install with:\n"
        "    pip install mlx\n"
    )
    sys.exit(1)


def _dtype_name(dtype) -> str:
    """Map an mlx dtype to a canonical ggml-compatible string name.

    The C++ runtime (rnn_kernel.cpp) maps dtype strings via
    `remora_dtype_from_string`, which expects names like "F32", "F16",
    "BF16", "Q4_K", etc. MLX dtypes stringify as "mlx.core.float16", so we
    translate them here to keep MLX shards byte-compatible with the C++
    streamer.
    """
    s = str(dtype)
    mapping = {
        "mlx.core.float32": "F32",
        "mlx.core.float16": "F16",
        "mlx.core.bfloat16": "BF16",
        "mlx.core.int8": "I8",
        "mlx.core.int16": "I16",
        "mlx.core.int32": "I32",
        "mlx.core.int64": "I64",
        "mlx.core.uint8": "U8",
        "mlx.core.uint16": "U16",
        "mlx.core.uint32": "U32",
        "mlx.core.uint64": "U64",
    }
    return mapping.get(s, s)


def _write_shard(out_dir: str, name: str, tensors: list) -> str:
    """Write a shard (raw bytes) and its sidecar; return the meta path.

    Each tensor entry records both:
      - ``offset``: shard-relative byte offset (where the bytes live in this
        shard file).
      - ``file_offset``: absolute byte offset within the *original* MLX
        safetensors file's tensor data section.
    """
    raw_path = os.path.join(out_dir, name)
    meta_path = os.path.join(out_dir, name + ".meta.json")
    sidecar = {"name": name, "tensors": []}
    offset = 0
    with open(raw_path, "wb") as f:
        for t in tensors:
            arr = t["data"]
            if hasattr(arr, "tobytes"):
                data = arr.tobytes()
            else:
                # MLX arrays don't expose .tobytes(); go through numpy.
                import numpy as np
                data = np.asarray(arr).tobytes()
            sidecar["tensors"].append({
                "name":         t["name"],
                "shape":        [int(x) for x in t["shape"]],
                "dtype":        _dtype_name(t["dtype"]),
                "offset":       offset,
                "file_offset":  t.get("file_offset", 0),
                "size":         len(data),
                "n_elements":   int(t["n_elements"]),
            })
            f.write(data)
            offset += len(data)
    with open(meta_path, "w") as mf:
        json.dump(sidecar, mf, indent=2)
    return meta_path


def _build_manifest(config: dict, global_tensors: list, block_tensors: dict) -> dict:
    """Top-level metadata: architecture + tokenizer summary."""
    arch = config.get("model_type", "mamba")
    n_layer = config.get("num_hidden_layers", 0) or len(block_tensors)
    d_model = config.get("hidden_size", 0) or config.get("d_model", 0)
    d_inner = config.get("intermediate_size", 0) or config.get("d_inner", 0)
    d_state = config.get("state_size", 16) or config.get("d_state", 16)
    d_conv = config.get("conv_kernel", 4) or config.get("d_conv", 4)
    dt_rank = config.get("time_step_rank", 128) or config.get("dt_rank", 128)
    vocab_size = config.get("vocab_size", 0)
    context_length = config.get("max_position_embeddings", 0) or config.get("context_length", 2048)
    tie_word_embeddings = config.get("tie_word_embeddings", True)

    has_output = any(t["name"].endswith("lm_head.weight") for t in global_tensors)
    has_embd = any(t["name"].endswith("embeddings.weight") for t in global_tensors)

    return {
        "architecture":      arch,
        "n_layer":           n_layer,
        "d_model":           d_model,
        "d_inner":           d_inner,
        "d_state":           d_state,
        "d_conv":            d_conv,
        "dt_rank":           dt_rank,
        "vocab_size":        vocab_size,
        "context_length":    context_length,
        "tied_embeddings":   (not has_output) and has_embd,
        "tokenizer": {
            "model":         "mlx",
            "pre":           None,
            "bos_token_id":  config.get("bos_token_id"),
            "eos_token_id":  config.get("eos_token_id"),
        },
    }


def _load_weights(model_dir: str) -> dict:
    """Load all safetensors weights from the MLX model directory."""
    import glob
    weights = {}
    for wf in sorted(glob.glob(os.path.join(model_dir, "model*.safetensors"))):
        w = mx.load(wf)
        weights.update(w)
    return weights


def _classify_tensor(name: str):
    """Classify an MLX weight name into (block_id, is_global).

    MLX Mamba weight names look like:
        backbone.embeddings.weight          -> global
        backbone.layers.0.norm.weight       -> block 0
        backbone.layers.0.mixer.in_proj.weight -> block 0
        backbone.layers.0.mixer.conv1d.weight  -> block 0
        backbone.layers.0.mixer.conv1d.bias     -> block 0
        backbone.layers.0.mixer.x_proj.weight   -> block 0
        backbone.layers.0.mixer.dt_proj.weight -> block 0
        backbone.layers.0.mixer.dt_proj.bias    -> block 0
        backbone.layers.0.mixer.out_proj.weight -> block 0
        backbone.norm_f.weight              -> global
        lm_head.weight                      -> global
    """
    parts = name.split(".")
    # Find the layer index
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1]), False
    return None, True  # global


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="path to the MLX model directory")
    parser.add_argument("--out", default="layers", help="output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load config
    config_path = os.path.join(args.model, "config.json")
    if not os.path.exists(config_path):
        sys.stderr.write(f"error: {config_path} not found\n")
        return 1
    with open(config_path) as f:
        config = json.load(f)

    # Load weights
    weights = _load_weights(args.model)
    if not weights:
        sys.stderr.write(f"error: no safetensors weights found in {args.model}\n")
        return 1

    global_tensors = []
    block_tensors: dict[int, list] = {}

    for name, arr in weights.items():
        block_id, is_global = _classify_tensor(name)
        tensor = {
            "name": name,
            "shape": list(arr.shape),
            "dtype": arr.dtype,
            "data": arr,
            "n_elements": arr.size,
            "file_offset": 0,
        }
        if is_global:
            global_tensors.append(tensor)
        else:
            block_tensors.setdefault(block_id, []).append(tensor)

    # Per-block shards + sidecars.
    for block_id in sorted(block_tensors):
        raw_path = os.path.join(args.out, f"blk.{block_id}")
        meta_path = _write_shard(args.out, f"blk.{block_id}", block_tensors[block_id])
        print(f"wrote {raw_path} ({len(block_tensors[block_id])} tensors) + {meta_path}")

    # Global shard + sidecar.
    if global_tensors:
        meta_path = _write_shard(args.out, "global.bin", global_tensors)
        print(f"wrote global.bin ({len(global_tensors)} tensors) + {meta_path}")

    # Top-level manifest.
    manifest = _build_manifest(config, global_tensors, block_tensors)
    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w") as mf:
        json.dump(manifest, mf, indent=2)
    print(f"wrote {manifest_path}")

    print(
        f"done: {len(block_tensors)} blocks sharded into {args.out} "
        f"({len(global_tensors)} globals, {len(weights)} total tensors)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
