#!/usr/bin/env python3
"""Convert a Hugging Face model to MLX format using mlx_lm.convert.

This is a thin CLI wrapper around `mlx_lm.convert` so Remora can produce
MLX-compatible model weights from a standard Hugging Face checkpoint. The
resulting directory (default `mlx_model/`) contains:

    config.json                 model architecture + hyperparameters
    model.safetensors           weights (or model-00001-of-NNNNN.safetensors)
    tokenizer.json / tokenizer.model
    README.md                   model card

That directory is what `scripts/shard_mlx.py` consumes to produce the
per-layer shards that Remora's layer-by-layer runtime streams from disk.

Usage:
    python3 scripts/convert_mlx.py --hf-path state-spaces/mamba-1.4b-hf \
        --mlx-path mlx_model [--quantize] [--q-bits 4] [--dtype float16]

Options:
    --hf-path <path|repo>   Hugging Face model path or repo ID.
    --mlx-path <dir>        Output directory for the MLX model (default: mlx_model).
    -q, --quantize          Quantize the model weights (default: off).
    --q-bits <N>            Quantization bits (default: 4).
    --q-group-size <N>      Quantization group size (default: 64).
    --dtype <t>             Save dtype: float16 | bfloat16 | float32.
                            Defaults to the model's torch_dtype.
    --dequantize            Dequantize an already-quantized model.
    --trust-remote-code     Trust remote code when loading the tokenizer.
"""

import argparse
import os
import sys


def _ensure_safetensors(hf_path: str) -> str:
    """If the HF model only ships .bin weights, convert them to safetensors
    in a temp dir so mlx_lm.convert can load them.

    mlx_lm.convert requires safetensors weights. Many Mamba checkpoints
    (state-spaces/mamba-*) only ship pytorch_model.bin. We load those with
    torch and re-save as safetensors so the MLX converter can proceed.
    """
    from huggingface_hub import snapshot_download

    local = snapshot_download(hf_path)
    # mlx_lm globs for `model*.safetensors`; only those count as "ready".
    has_safetensors = any(
        f.startswith("model") and f.endswith(".safetensors")
        for f in os.listdir(local)
    )
    if has_safetensors:
        return local

    bin_files = [f for f in os.listdir(local) if f.endswith(".bin")]
    if not bin_files:
        sys.stderr.write(
            f"error: no .safetensors or .bin weights found in {local}\n"
        )
        sys.exit(1)

    import torch
    from safetensors.torch import save_file

    print(f"[convert_mlx] Converting {len(bin_files)} .bin file(s) to safetensors")
    for bf in bin_files:
        bin_path = os.path.join(local, bf)
        # mlx_lm globs for `model*.safetensors`, so name the output accordingly.
        st_path = os.path.join(local, "model.safetensors")
        state = torch.load(bin_path, map_location="cpu")
        # torch.load may return a dict directly or a checkpoint wrapper.
        if not isinstance(state, dict):
            state = state.get("state_dict", {})
        # Tied embeddings (e.g. Mamba's backbone.embedding.weight == lm_head.weight)
        # share memory in the .bin. safetensors refuses shared tensors, so we
        # save a deep copy of each tensor to break the aliasing.
        state = {k: v.clone() for k, v in state.items()}
        save_file(state, st_path)
        print(f"[convert_mlx]   {bf} -> model.safetensors")

    return local


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True,
                        help="Hugging Face model path or repo ID")
    parser.add_argument("--mlx-path", default="mlx_model",
                        help="Output directory for the MLX model")
    parser.add_argument("-q", "--quantize", action="store_true",
                        help="Quantize the model weights")
    parser.add_argument("--q-bits", type=int, default=4,
                        help="Quantization bits (default: 4)")
    parser.add_argument("--q-group-size", type=int, default=64,
                        help="Quantization group size (default: 64)")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"],
                        default=None,
                        help="Save dtype (defaults to model's torch_dtype)")
    parser.add_argument("--dequantize", action="store_true",
                        help="Dequantize an already-quantized model")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Trust remote code when loading the tokenizer")
    args = parser.parse_args()

    try:
        from mlx_lm.convert import convert
    except ImportError:
        sys.stderr.write(
            "error: mlx-lm is required. Install with:\n"
            "    pip install mlx-lm\n"
        )
        return 1

    print(f"[convert_mlx] Converting {args.hf_path} -> {args.mlx_path}")
    if args.quantize:
        print(f"[convert_mlx] Quantizing to {args.q_bits} bits, "
              f"group size {args.q_group_size}")

    # mlx_lm.convert needs safetensors; convert .bin checkpoints if needed.
    hf_path = _ensure_safetensors(args.hf_path)

    convert(
        hf_path=hf_path,
        mlx_path=args.mlx_path,
        quantize=args.quantize,
        q_group_size=args.q_group_size,
        q_bits=args.q_bits,
        dtype=args.dtype,
        dequantize=args.dequantize,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"[convert_mlx] Done. MLX model saved to {args.mlx_path}")
    print(f"[convert_mlx] Next: python3 scripts/shard_mlx.py {args.mlx_path} --out layers/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
