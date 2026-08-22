#!/usr/bin/env bash
# scripts/bench_mlx_compare.sh
#
# Side-by-side benchmark: MLX (Apple Silicon) vs the existing Remora /
# llama.cpp results. This runs the MLX-native inference path (via
# scripts/bench_mlx.py) on a converted MLX model and prints a comparison
# table against the numbers already captured in bench_results.txt.
#
# Usage:
#   ./scripts/bench_mlx_compare.sh [--model mlx_model] [--tokens 64]
#
# Requires: mlx, mlx-lm installed (pip install mlx mlx-lm), and a converted
# MLX model (see scripts/convert_mlx.py).

set -euo pipefail

cd "$(dirname "$0")/.."

# Prefer the project venv python if present.
if [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi

MODEL="${MODEL:-mlx_model}"
TOKENS="${TOKENS:-64}"
PROMPT="${PROMPT:-Hello, my name is}"
TEMP="${TEMP:-0.0}"
OUT_FILE="${OUT_FILE:-bench_mlx_results.txt}"

log() { printf "[bench-mlx] %s\n" "$*"; }

[[ -d "$MODEL" && -f "$MODEL/config.json" ]] \
    || { echo "missing $MODEL (run scripts/convert_mlx.py first)"; exit 1; }

log "model:   $MODEL"
log "tokens:  $TOKENS"
log "prompt:  $PROMPT"
log "temp:    $TEMP"
echo

# ---------------------------------------------------------------------------
# Run the MLX benchmark
# ---------------------------------------------------------------------------
log "===== MLX native (Apple Silicon / Metal) ====="
MLX_OUT=$("$PY" scripts/bench_mlx.py \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --tokens "$TOKENS" \
    --temp "$TEMP" 2>&1 | grep -v urllib3)

echo "$MLX_OUT"

# Parse the RESULT line: gen_tps=... prompt_tps=... peak_gb=... load_s=...
MLX_RESULT=$(echo "$MLX_OUT" | grep -E '^\[bench_mlx\] RESULT' | tail -1)
MLX_GEN_TPS=$(echo "$MLX_RESULT" | sed -nE 's/.*gen_tps=([0-9.]+).*/\1/p')
MLX_PROMPT_TPS=$(echo "$MLX_RESULT" | sed -nE 's/.*prompt_tps=([0-9.]+).*/\1/p')
MLX_PEAK_GB=$(echo "$MLX_RESULT" | sed -nE 's/.*peak_gb=([0-9.]+).*/\1/p')
MLX_LOAD_S=$(echo "$MLX_RESULT" | sed -nE 's/.*load_s=([0-9.]+).*/\1/p')
MLX_PEAK_MIB=$(awk -v g="$MLX_PEAK_GB" 'BEGIN{printf "%.1f", g*1024}')
echo

# ---------------------------------------------------------------------------
# Pull the existing Remora / llama.cpp numbers from bench_results.txt
# ---------------------------------------------------------------------------
log "===== Comparison vs existing bench_results.txt ====="
if [[ -f bench_results.txt ]]; then
    # The summary table in bench_results.txt has rows like:
    #   "tokens/sec"          | 10.08 | 26.96 | N/A
    #   "peak RSS (MiB)"      | 198.8 | 1212.4 | 873.2
    # We extract the Remora (col 2) and llama.cpp (col 4) values.
    REMORA_TPS=$(grep -E 'tokens/sec' bench_results.txt | head -1 \
        | awk -F'|' '{gsub(/ /,"",$2); print $2}')
    REMORA_PEAK=$(grep -E 'peak RSS' bench_results.txt | head -1 \
        | awk -F'|' '{gsub(/ /,"",$2); print $2}')
    LLAMA_TPS=$(grep -E 'tokens/sec' bench_results.txt | head -1 \
        | awk -F'|' '{gsub(/ /,"",$4); print $4}')
    LLAMA_PEAK=$(grep -E 'peak RSS' bench_results.txt | head -1 \
        | awk -F'|' '{gsub(/ /,"",$4); print $4}')
    # llama.cpp's real gen tok/s lives in the inline speed line of its log
    # (e.g. "Generation: 52.8 t/s"); the summary table often shows N/A.
    if [[ -f llama.stdout.log ]]; then
        LLAMA_INLINE=$(grep -oE 'Generation:[[:space:]]*[0-9.]+[[:space:]]*t/s' \
            llama.stdout.log 2>/dev/null | tail -1 \
            | sed -nE 's/.*Generation:[[:space:]]*([0-9.]+).*/\1/p' || true)
        [[ -n "$LLAMA_INLINE" ]] && LLAMA_TPS="$LLAMA_INLINE"
    fi
else
    REMORA_TPS="n/a"; REMORA_PEAK="n/a"; LLAMA_TPS="n/a"; LLAMA_PEAK="n/a"
fi

{
    echo "============================================================"
    echo "Mamba 1.4B  —  MLX vs Remora vs llama.cpp"
    echo "============================================================"
    printf "%-28s | %-20s | %-20s | %-20s\n" "metric" "MLX (Metal)" "Remora (C++)" "llama.cpp"
    echo "----------------------------+----------------------+----------------------+----------------------"
    printf "%-28s | %-20s | %-20s | %-20s\n" "model"        "$MODEL"              "mamba-1.4b Q4_K_M"    "mamba-1.4b Q4_K_M"
    printf "%-28s | %-20s | %-20s | %-20s\n" "weights"      "float16 (MLX)"       "sharded GGUF"         "monolithic GGUF"
    printf "%-28s | %-20s | %-20s | %-20s\n" "gen tok/s"    "${MLX_GEN_TPS:-n/a}" "${REMORA_TPS:-n/a}"   "${LLAMA_TPS:-n/a}"
    printf "%-28s | %-20s | %-20s | %-20s\n" "prompt tok/s" "${MLX_PROMPT_TPS:-n/a}" "n/a"                "n/a"
    printf "%-28s | %-20s | %-20s | %-20s\n" "peak mem (MiB)" "${MLX_PEAK_MIB:-n/a}" "${REMORA_PEAK:-n/a}" "${LLAMA_PEAK:-n/a}"
    printf "%-28s | %-20s | %-20s | %-20s\n" "load time (s)" "${MLX_LOAD_S:-n/a}"  "n/a"                 "n/a"
    echo
    echo "Notes:"
    echo "  * MLX runs the converted float16 model natively on Apple Silicon"
    echo "    (Metal backend) via scripts/bench_mlx.py. It forces exactly"
    echo "    $TOKENS tokens so base models that emit EOS immediately still"
    echo "    get a clean tokens/sec number."
    echo "  * Remora / llama.cpp numbers come from bench_results.txt (the"
    echo "    existing GGUF-based comparison)."
    echo "  * MLX peak memory is reported by mlx (mx.get_peak_memory);"
    echo "    Remora / llama.cpp peak RSS is sampled from ps at 20 Hz."
    echo
} | tee "$OUT_FILE"

log "wrote $OUT_FILE"
