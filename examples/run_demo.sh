#!/usr/bin/env bash
# Remora demo: download a small model, shard it, and run the engine.
set -euo pipefail

MODEL_URL="${MODEL_URL:-https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf}"
MODEL="stories260K.gguf"
OUT="layers"

echo "==> Downloading small model"
if [ ! -f "$MODEL" ]; then
    curl -L -o "$MODEL" "$MODEL_URL"
fi

echo "==> Sharding GGUF into per-layer files"
python3 scripts/shard_gguf.py "$MODEL" --out "$OUT"

echo "==> Building Remora"
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

echo "==> Running the engine"
./build/remora \
    --routing examples/basic_routing.json \
    --layers "$OUT" \
    --tokens 8

echo "==> Done"
