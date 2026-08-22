#!/usr/bin/env bash
# Remora MLX demo: convert a HF model to MLX, shard it, and run it.
#
# This demonstrates the MLX-compatible path:
#   1. convert_mlx.py  — Hugging Face -> MLX (via mlx_lm.convert)
#   2. shard_mlx.py    — MLX model -> per-layer shards (same format as shard_gguf.py)
#   3. run_mlx.py      — run the MLX model natively on Apple Silicon (Metal)
#
# Usage:
#   ./examples/run_mlx_demo.sh [--hf-path state-spaces/mamba-130m-hf]
#
# Requires: pip install mlx mlx-lm torch safetensors
set -euo pipefail

# Prefer the project venv python if present (it has mlx / mlx-lm installed).
if [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi

HF_PATH="${HF_PATH:-state-spaces/mamba-130m-hf}"
MLX_DIR="${MLX_DIR:-mlx_model}"
OUT="${OUT:-layers_mlx}"
TOKENS="${TOKENS:-8}"
PROMPT="${PROMPT:-Hello, my name is}"

echo "==> Converting $HF_PATH to MLX"
"$PY" scripts/convert_mlx.py \
    --hf-path "$HF_PATH" \
    --mlx-path "$MLX_DIR" \
    --dtype float16

echo "==> Sharding MLX model into per-layer files"
"$PY" scripts/shard_mlx.py "$MLX_DIR" --out "$OUT"

echo "==> Running the MLX model"
"$PY" scripts/run_mlx.py \
    --model "$MLX_DIR" \
    --prompt "$PROMPT" \
    --tokens "$TOKENS" \
    --temp 0.0

echo "==> Done"
