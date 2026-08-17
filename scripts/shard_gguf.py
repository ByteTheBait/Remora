#!/usr/bin/env python3
"""Slice a standard GGUF model into per-layer binary payloads with sidecars.

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
    python3 scripts/shard_gguf.py model.gguf --out layers/
"""

import argparse
import json
import os
import sys

try:
    from gguf import GGUFReader
except ImportError:
    sys.stderr.write(
        "error: the 'gguf' package is required. Install with:\n"
        "    pip install gguf\n"
    )
    sys.exit(1)


def _gguf_type_name(t) -> str:
    """Map gguf.tensor_type (enum) to the canonical ggml_type name string."""
    name = getattr(t, "name", None)
    if name:
        return str(name)
    return str(t)


def _kv_str(reader, key: str):
    if key not in reader.fields:
        return None
    f = reader.fields[key]
    contents = f.contents() if callable(f.contents) else f.contents
    if isinstance(contents, (bytes, bytearray)):
        return contents.decode("utf-8", errors="replace")
    return contents


def _kv_int(reader, key: str):
    if key not in reader.fields:
        return None
    f = reader.fields[key]
    contents = f.contents() if callable(f.contents) else f.contents
    try:
        return int(contents)
    except (TypeError, ValueError):
        return None


def _kv_array(reader, key: str):
    """Read an ARRAY-typed KV into a Python list (or None)."""
    if key not in reader.fields:
        return None
    f = reader.fields[key]
    contents = f.contents() if callable(f.contents) else f.contents
    if contents is None:
        return None
    try:
        return list(contents)
    except TypeError:
        return None


def _write_shard(out_dir: str, name: str, tensors: list) -> str:
    """Write a shard (raw bytes) and its sidecar; return the meta path.

    Each tensor entry records both:
      - ``offset``: shard-relative byte offset (where the bytes live in this
        shard file). Used if a runtime reads the shard directly.
      - ``file_offset``: absolute byte offset within the *original* GGUF
        file's tensor data section (i.e. ``gguf_get_tensor_offset``).
        Used by the libllama-based driver, which passes the original GGUF
        path to ``gguf_init_from_file`` for header parsing and then resolves
        tensor bytes through a per-tensor callback keyed on this offset.
    """
    raw_path = os.path.join(out_dir, name)
    meta_path = os.path.join(out_dir, name + ".meta.json")
    sidecar = {"name": name, "tensors": []}
    offset = 0
    with open(raw_path, "wb") as f:
        for t in tensors:
            data = t.data.tobytes()
            sidecar["tensors"].append({
                "name":         t.name,
                "shape":        [int(x) for x in t.shape],
                "dtype":        _gguf_type_name(t.tensor_type),
                "offset":       offset,
                "file_offset":  int(t.data_offset),
                "size":         len(data),
                "n_elements":   int(t.n_elements),
            })
            f.write(data)
            offset += len(data)
    with open(meta_path, "w") as mf:
        json.dump(sidecar, mf, indent=2)
    return meta_path


def _build_manifest(reader, global_tensors, block_tensors) -> dict:
    """Top-level metadata: architecture + tokenizer summary."""
    arch = _kv_str(reader, "general.architecture") or "mamba"
    n_layer       = _kv_int(reader, f"{arch}.block_count")         or len(block_tensors)
    d_model       = _kv_int(reader, f"{arch}.embedding_length")    or 0

    # The Mamba GGUF dialect names differ from transformer KVs.
    # (For Mamba: feed_forward_length is 0, the real dim is ssm.inner_size.)
    d_inner       = (_kv_int(reader, f"{arch}.ssm.inner_size")
                     or _kv_int(reader, f"{arch}.feed_forward_length")
                     or 0)
    d_state       = _kv_int(reader, f"{arch}.ssm.state_size")     or 16
    d_conv        = _kv_int(reader, f"{arch}.ssm.conv_kernel")    or 4
    dt_rank       = _kv_int(reader, f"{arch}.ssm.time_step_rank") or 128

    # Vocab size: explicit KV first, fall back to length of tokenizer.ggml.tokens.
    vocab_size = _kv_int(reader, f"{arch}.vocab_size")
    if not vocab_size:
        toks = _kv_array(reader, "tokenizer.ggml.tokens")
        if toks is not None:
            vocab_size = len(toks)

    context_length = _kv_int(reader, f"{arch}.context_length")     or 2048

    has_output = any(t.name == "output.weight"      for t in global_tensors)
    has_embd   = any(t.name == "token_embd.weight"  for t in global_tensors)

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
            "model":         _kv_str(reader, "tokenizer.ggml.model"),
            "pre":           _kv_str(reader, "tokenizer.ggml.pre"),
            "bos_token_id":  _kv_int(reader, "tokenizer.ggml.bos_token_id"),
            "eos_token_id":  _kv_int(reader, "tokenizer.ggml.eos_token_id"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="path to the source .gguf model")
    parser.add_argument("--out", default="layers", help="output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    reader = GGUFReader(args.model)
    global_tensors = []
    block_tensors: dict[int, list] = {}

    for tensor in reader.tensors:
        name = tensor.name
        if name.startswith("blk."):
            try:
                block_id = int(name.split(".")[1])
            except (IndexError, ValueError):
                continue
            block_tensors.setdefault(block_id, []).append(tensor)
        else:
            global_tensors.append(tensor)

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
    manifest = _build_manifest(reader, global_tensors, block_tensors)
    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w") as mf:
        json.dump(manifest, mf, indent=2)
    print(f"wrote {manifest_path}")

    print(
        f"done: {len(block_tensors)} blocks sharded into {args.out} "
        f"({len(global_tensors)} globals, {len(reader.tensors)} total tensors)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())