#!/usr/bin/env python3
"""Slice a standard GGUF model into per-layer binary payloads.

Remora streams layers one at a time from disk, so a monolithic GGUF must be
split into individual files: one for the global tensors (token_embd, output)
and one per sequential block (blk.0, blk.1, ...).

This script reads a GGUF using the `gguf` package, isolates the global layers,
and writes each block's raw tensor payload to its own file under an output
directory. The resulting layout is consumed by the C++ LayerStreamer.

Usage:
    python3 scripts/shard_gguf.py model.gguf --out layers/
"""

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="path to the source .gguf model")
    parser.add_argument("--out", default="layers", help="output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    reader = GGUFReader(args.model)
    global_tensors = []
    block_tensors = {}

    for tensor in reader.tensors:
        name = tensor.name
        if name.startswith("blk."):
            # blk.N.<tensor> -> group by block index N
            try:
                block_id = int(name.split(".")[1])
            except (IndexError, ValueError):
                continue
            block_tensors.setdefault(block_id, []).append(tensor)
        else:
            global_tensors.append(tensor)

    # Write global tensors (token_embd, output, norm, etc.) to one file.
    global_path = os.path.join(args.out, "global.bin")
    with open(global_path, "wb") as f:
        for t in global_tensors:
            f.write(t.data.tobytes())
    print(f"wrote {global_path} ({len(global_tensors)} global tensors)")

    # Write each block's tensors to its own file.
    for block_id in sorted(block_tensors):
        path = os.path.join(args.out, f"blk.{block_id}")
        with open(path, "wb") as f:
            for t in block_tensors[block_id]:
                f.write(t.data.tobytes())
        print(f"wrote {path} ({len(block_tensors[block_id])} tensors)")

    print(f"done: {len(block_tensors)} blocks sharded into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
