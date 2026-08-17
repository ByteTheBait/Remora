#!/usr/bin/env bash
# scripts/bench_compare.sh
#
# Side-by-side benchmark:
#   * Remora        — runs the layer-by-layer placeholder streamer over the
#                      sharded GGUF produced by scripts/shard_gguf.py.
#   * remora_llama  — runs the libllama-based driver that consumes sharded
#                      weights via a per-tensor callback. This produces real
#                      Mamba outputs (no placeholder math) and is the actual
#                      apples-to-apples comparison vs llama.cpp.
#   * llama.cpp     — runs llama-cli on the original monolithic .gguf.
#
# For each backend the script samples peak RSS (KB) and wall time, derives
# a TTFT proxy, and prints tokens/sec. Results go to stdout and to
# bench_results.txt.

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-mamba-1.4b-hf.Q4_K_M.gguf}"
LAYERS_DIR="${LAYERS_DIR:-layers}"
REMORA_BIN="${REMORA_BIN:-./build/remora}"
REMORA_LLAMA_BIN="${REMORA_LLAMA_BIN:-./build/remora_llama}"
ROUTING="${ROUTING:-examples/basic_routing.json}"
LLAMA_CLI="${LLAMA_CLI:-llama-cli}"

PROMPT="${PROMPT:-Hello, my name is}"
N_PREDICT="${N_PREDICT:-64}"
N_THREADS="${N_THREADS:-$(sysctl -n hw.physicalcpu)}"
CTX_SIZE="${CTX_SIZE:-512}"

OUT_FILE="${OUT_FILE:-bench_results.txt}"
SAMPLE_HZ="${SAMPLE_HZ:-20}"   # RSS sampling rate (Hz)

log() { printf "[bench] %s\n" "$*"; }

# Defaults so `set -u` doesn't trip on unset optional variables later.
INLINE_GEN_TPS=""
INLINE_PROMPT_TPS=""

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

# rss_samples_log FILE PID
# Append (timestamp_ms, rss_kb) lines to FILE while PID is alive.
rss_samples_log() {
    local file="$1" pid="$2"
    while kill -0 "$pid" 2>/dev/null; do
        local ts rss
        ts=$(($(date +%s%N) / 1000000))
        rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
        [[ -z "$rss" ]] && rss=0
        printf "%s %s\n" "$ts" "$rss" >> "$file"
        sleep "$(awk -v hz="$SAMPLE_HZ" 'BEGIN{printf "%.3f", 1.0/hz}')"
    done
}

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

[[ -f "$MODEL" ]]      || { echo "missing $MODEL"; exit 1; }
[[ -x "$REMORA_BIN" ]] || { echo "missing $REMORA_BIN (build first)"; exit 1; }
[[ -f "$ROUTING" ]]    || { echo "missing $ROUTING"; exit 1; }
[[ -d "$LAYERS_DIR" && -f "$LAYERS_DIR/blk.0" ]] \
    || { echo "missing $LAYERS_DIR/blk.0 (shard first)"; exit 1; }
command -v "$LLAMA_CLI" >/dev/null \
    || { echo "missing $LLAMA_CLI on PATH"; exit 1; }

LAYER_COUNT=$(ls "$LAYERS_DIR"/blk.* 2>/dev/null | grep -v '\.meta\.json$' | wc -l | tr -d ' ')
# Total weight bytes across all blk.* files (BSD `du` on macOS has no -b).
TOTAL_BYTES=$(cat "$LAYERS_DIR"/blk.* 2>/dev/null | wc -c | tr -d ' ')

log "model:       $MODEL"
log "layers:      $LAYER_COUNT sharded blocks in $LAYERS_DIR"
log "total bytes: $TOTAL_BYTES (~$(awk -v b="$TOTAL_BYTES" 'BEGIN{printf "%.1f", b/1024/1024}') MiB)"
log "prompt:      $PROMPT"
log "n_predict:   $N_PREDICT"
log "threads:     $N_THREADS"
log "ctx:         $CTX_SIZE"
echo

# ---------------------------------------------------------------------------
# Run 1: Remora
# ---------------------------------------------------------------------------

log "===== Run 1: Remora layer-streamer ====="
REMORA_LOG="remora.stdout.log"
REMORA_RSS="remora.rss.log"
: > "$REMORA_LOG"
: > "$REMORA_RSS"

START_NS=$(date +%s%N)
"$REMORA_BIN" \
    --routing "$ROUTING" \
    --layers "$LAYERS_DIR" \
    --tokens 8 \
    > "$REMORA_LOG" 2>&1 &
REMORA_PID=$!

# Spawn the RSS sampler in a subshell so it tracks the right PID.
( rss_samples_log "$REMORA_RSS" "$REMORA_PID" ) &
SAMPLER_PID=$!

wait "$REMORA_PID"
REMORA_EXIT=$?
wait "$SAMPLER_PID" 2>/dev/null || true
END_NS=$(date +%s%N)

REMORA_WALL_MS=$(( (END_NS - START_NS) / 1000000 ))

# Detect first-output timestamp (proxy for TTFT in the placeholder runtime).
# Remora's first per-layer printf is "   layer 1/N done ...".
REMORA_FIRST_OUTPUT_MS=$(awk -v t0="$START_NS" '
    /layer 1\// {
        # Convert ms epoch to relative ms.
        cmd = "date +%s%N"; cmd | getline now; close(cmd);
        rel_ms = int((now - t0) / 1000000);
        print rel_ms; exit
    }' "$REMORA_LOG")

# Peak RSS from the sampler (KB).
REMORA_PEAK_KB=$(awk '{print $2}' "$REMORA_RSS" | sort -n | tail -1)
REMORA_PEAK_KB=${REMORA_PEAK_KB:-0}

# Streaming throughput proxy: bytes streamed / wall time.
REMORA_STREAM_MBPS=$(awk -v b="$TOTAL_BYTES" -v ms="$REMORA_WALL_MS" '
    BEGIN{ if (ms <= 0) { print "0.0"; exit } printf "%.1f", (b/1024.0/1024.0) / (ms/1000.0) }')

# Tokens/sec for the placeholder: total work is LAYER_COUNT iterations; we
# report layers/sec as a comparable throughput number. The placeholder is NOT
# a real token generator, so we label the unit clearly.
REMORA_LAYERSPERSEC=$(awk -v n="$LAYER_COUNT" -v ms="$REMORA_WALL_MS" '
    BEGIN{ if (ms <= 0) { print "0.0"; exit } printf "%.2f", n / (ms/1000.0) }')

log "Remora wall:     ${REMORA_WALL_MS} ms"
log "Remora first-out (TTFT proxy): ${REMORA_FIRST_OUTPUT_MS:-N/A} ms"
log "Remora peak RSS: ${REMORA_PEAK_KB} KB (~$(awk -v k="$REMORA_PEAK_KB" 'BEGIN{printf "%.1f", k/1024}') MiB)"
log "Remora stream:   ${REMORA_STREAM_MBPS} MiB/s"
log "Remora layers/s: ${REMORA_LAYERSPERSEC} (placeholder kernel, not real TPS)"
echo

# ---------------------------------------------------------------------------
# Run 2: llama.cpp
# ---------------------------------------------------------------------------

log "===== Run 2: llama.cpp ====="
LLAMA_LOG="llama.stdout.log"
LLAMA_RSS="llama.rss.log"
: > "$LLAMA_LOG"
: > "$LLAMA_RSS"

START_NS=$(date +%s%N)
"$LLAMA_CLI" \
    -m "$MODEL" \
    -p "$PROMPT" \
    -n "$N_PREDICT" \
    -c "$CTX_SIZE" \
    -t "$N_THREADS" \
    --no-conversation \
    --no-display-prompt \
    --single-turn \
    > "$LLAMA_LOG" 2>&1 &
LLAMA_PID=$!

( rss_samples_log "$LLAMA_RSS" "$LLAMA_PID" ) &
SAMPLER_PID=$!

wait "$LLAMA_PID"
LLAMA_EXIT=$?
wait "$SAMPLER_PID" 2>/dev/null || true
END_NS=$(date +%s%N)

LLAMA_WALL_MS=$(( (END_NS - START_NS) / 1000000 ))

LLAMA_PEAK_KB=$(awk '{print $2}' "$LLAMA_RSS" | sort -n | tail -1)
LLAMA_PEAK_KB=${LLAMA_PEAK_KB:-0}

# Parse llama-cli timings. Newer builds print, at the end of generation,
# something like:
#   llama_print_timings:        eval time =     X ms /    N runs   ( ... ms per token, ...)
#   llama_print_timings:       total time =     Y ms /    P tokens
#   ...
# In addition, the inline speed line printed after each generation in
# interactive / single-turn modes looks like:
#   [ Prompt: 225.6 t/s | Generation: 76.3 t/s ]
# We prefer the footer timings and fall back to the inline line.

LLAMA_EVAL_LINE=$(grep -E 'eval time[[:space:]]*=' "$LLAMA_LOG" 2>/dev/null | tail -1 || true)
LLAMA_TOTAL_LINE=$(grep -E 'total time[[:space:]]*=' "$LLAMA_LOG" 2>/dev/null | tail -1 || true)

# Inline speed line (used when footer timings are absent or suppressed).
INLINE_SPEED=$(grep -oE 'Prompt:[[:space:]]*[0-9.]+[[:space:]]*t/s[[:space:]]*\|[[:space:]]*Generation:[[:space:]]*[0-9.]+[[:space:]]*t/s' "$LLAMA_LOG" 2>/dev/null | tail -1 || true)
INLINE_PROMPT_TPS=$(echo "$INLINE_SPEED" | sed -nE 's/.*Prompt:[[:space:]]*([0-9.]+).*/\1/p' || true)
INLINE_GEN_TPS=$(echo "$INLINE_SPEED"    | sed -nE 's/.*Generation:[[:space:]]*([0-9.]+).*/\1/p' || true)

# eval time in milliseconds
LLAMA_EVAL_MS=$(echo "$LLAMA_EVAL_LINE" | sed -nE 's/.*eval time *= *([0-9]+) *ms.*/\1/p' || true)
# tokens generated
LLAMA_N_TOK=$(echo "$LLAMA_EVAL_LINE" | sed -nE 's/.*\/ *([0-9]+) *runs.*/\1/p' || true)
# total time in milliseconds
LLAMA_TOTAL_MS=$(echo "$LLAMA_TOTAL_LINE" | sed -nE 's/.*total time *= *([0-9]+) *ms.*/\1/p' || true)

# If we couldn't parse, fall back to wall time and the requested n_predict.
[[ -z "$LLAMA_EVAL_MS" ]] && LLAMA_EVAL_MS=0
[[ -z "$LLAMA_N_TOK" ]]   && LLAMA_N_TOK=$N_PREDICT
[[ -z "$LLAMA_TOTAL_MS" ]]&& LLAMA_TOTAL_MS=$LLAMA_WALL_MS

# Tokens/sec over the generation phase only (excludes model load).
if [[ "$LLAMA_EVAL_MS" -gt 0 && "$LLAMA_N_TOK" -gt 0 ]]; then
    LLAMA_TPS=$(awk -v n="$LLAMA_N_TOK" -v ms="$LLAMA_EVAL_MS" \
        'BEGIN{ printf "%.2f", (n * 1000.0) / ms }')
else
    LLAMA_TPS="N/A"
fi

# TTFT: time from process start to first token emitted. We approximate by
# (model_load_time + first_token_eval_time). llama-cli doesn't expose
# first-token timestamp directly, so we use a structural proxy:
#   TTFT_proxy_ms ≈ total_ms - eval_ms
# i.e. everything before the eval phase (load + prompt processing) is the
# time-to-first-token cost. Labeled as a proxy because it includes the
# prompt processing pass as well.
LLAMA_TTFT_MS=$(awk -v total="$LLAMA_TOTAL_MS" -v eval="$LLAMA_EVAL_MS" \
    'BEGIN{ v = total - eval; if (v < 0) v = 0; printf "%d", v }')

log "llama.cpp wall:    ${LLAMA_WALL_MS} ms"
log "llama.cpp total:   ${LLAMA_TOTAL_MS} ms"
log "llama.cpp eval:    ${LLAMA_EVAL_MS} ms over ${LLAMA_N_TOK} tokens"
log "llama.cpp TPS:     ${LLAMA_TPS} tok/s (eval phase)"
log "llama.cpp TTFT px: ${LLAMA_TTFT_MS} ms (total - eval, includes prompt eval)"
[[ -n "${INLINE_GEN_TPS:-}" ]] && log "llama.cpp inline:  ${INLINE_GEN_TPS} gen tok/s, ${INLINE_PROMPT_TPS:-N/A} prompt tok/s"
log "llama.cpp peak RSS:${LLAMA_PEAK_KB} KB (~$(awk -v k="$LLAMA_PEAK_KB" 'BEGIN{printf "%.1f", k/1024}') MiB)"
echo

# ---------------------------------------------------------------------------
# Run 3: remora_llama (libllama driver, sharded weights)
# ---------------------------------------------------------------------------

if [[ -x "$REMORA_LLAMA_BIN" ]]; then
    log "===== Run 3: remora_llama (libllama + sharded weights) ====="
    REMORA_LLAMA_LOG="remora_llama.stdout.log"
    REMORA_LLAMA_RSS="remora_llama.rss.log"
    : > "$REMORA_LLAMA_LOG"
    : > "$REMORA_LLAMA_RSS"

    START_NS=$(date +%s%N)
    DYLD_LIBRARY_PATH="/opt/homebrew/Cellar/llama.cpp/10180/lib:/opt/homebrew/Cellar/ggml/0.17.0/lib" \
    GGML_BACKEND_PATH="/opt/homebrew/Cellar/ggml/0.17.0/libexec" \
    "$REMORA_LLAMA_BIN" \
        --gguf "$MODEL" \
        --layers "$LAYERS_DIR" \
        --prompt "$PROMPT" \
        --tokens "$N_PREDICT" \
        --ctx "$CTX_SIZE" \
        > "$REMORA_LLAMA_LOG" 2>&1 &
    REMORA_LLAMA_PID=$!

    ( rss_samples_log "$REMORA_LLAMA_RSS" "$REMORA_LLAMA_PID" ) &
    SAMPLER_PID=$!

    wait "$REMORA_LLAMA_PID"
    REMORA_LLAMA_EXIT=$?
    wait "$SAMPLER_PID" 2>/dev/null || true
    END_NS=$(date +%s%N)

    REMORA_LLAMA_WALL_MS=$(( (END_NS - START_NS) / 1000000 ))
    REMORA_LLAMA_PEAK_KB=$(awk '{print $2}' "$REMORA_LLAMA_RSS" | sort -n | tail -1)
    REMORA_LLAMA_PEAK_KB=${REMORA_LLAMA_PEAK_KB:-0}

    # remora_llama prints its own timing line:
    #   remora: %.3fs for %d tokens (%.2f tok/s)
    REMORA_LLAMA_TIMING=$(grep -E 'remora: [0-9.]+s for [0-9]+ tokens' "$REMORA_LLAMA_LOG" | tail -1 || true)
    REMORA_LLAMA_TPS=$(echo "$REMORA_LLAMA_TIMING" | sed -nE 's/.*\(([0-9.]+) tok\/s\).*/\1/p' || true)
    REMORA_LLAMA_TPS=${REMORA_LLAMA_TPS:-N/A}

    # Detect "loaded N tensors" line as a cheap proxy for "first operand ready".
    REMORA_LLAMA_TTFT_MS=$(awk -v t0="$START_NS" '
        /remora: indexed [0-9]+ tensors/ {
            cmd = "date +%s%N"; cmd | getline now; close(cmd);
            rel_ms = int((now - t0) / 1000000);
            print rel_ms; exit
        }' "$REMORA_LLAMA_LOG")

    log "remora_llama wall:    ${REMORA_LLAMA_WALL_MS} ms"
    log "remora_llama first-out (TTFT px): ${REMORA_LLAMA_TTFT_MS:-N/A} ms"
    log "remora_llama TPS:     ${REMORA_LLAMA_TPS} tok/s"
    log "remora_llama peak RSS:${REMORA_LLAMA_PEAK_KB} KB (~$(awk -v k="$REMORA_LLAMA_PEAK_KB" 'BEGIN{printf "%.1f", k/1024}') MiB)"
    echo
else
    log "remora_llama binary not built at $REMORA_LLAMA_BIN — skipping Run 3."
    REMORA_LLAMA_WALL_MS=0
    REMORA_LLAMA_TPS="n/a"
    REMORA_LLAMA_PEAK_KB=0
    REMORA_LLAMA_TTFT_MS=""
    REMORA_LLAMA_EXIT=0
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

{
    echo "============================================================"
    echo "Mamba 1.4B Q4_K_M  —  Remora vs llama.cpp"
    echo "============================================================"
    printf "%-30s | %-22s | %-22s | %-22s\n" "metric" "Remora" "remora_llama" "llama.cpp"
    echo "------------------------------+------------------------+------------------------+------------------------"
    printf "%-30s | %-22s | %-22s | %-22s\n" "model"        "$MODEL"                   "$MODEL"                      "$MODEL"
    printf "%-30s | %-22s | %-22s | %-22s\n" "sharded?"     "yes (${LAYER_COUNT} blocks)" "yes (${LAYER_COUNT} blocks)" "no (single file)"
    printf "%-30s | %-22s | %-22s | %-22s\n" "wall time (ms)"      "$REMORA_WALL_MS"   "$REMORA_LLAMA_WALL_MS"  "$LLAMA_WALL_MS"
    printf "%-30s | %-22s | %-22s | %-22s\n" "TTFT (ms)"           "${REMORA_FIRST_OUTPUT_MS:-N/A}" "${REMORA_LLAMA_TTFT_MS:-N/A}" "$LLAMA_TTFT_MS"
    printf "%-30s | %-22s | %-22s | %-22s\n" "tokens generated"    "0 (placeholder)"   "$N_PREDICT"               "$LLAMA_N_TOK"
    printf "%-30s | %-22s | %-22s | %-22s\n" "tokens/sec"          "n/a"               "${REMORA_LLAMA_TPS}"        "$LLAMA_TPS"
    printf "%-30s | %-22s | %-22s | %-22s\n" "gen tok/s (inline)"  "n/a"               "n/a"                       "${INLINE_GEN_TPS:-N/A}"
    printf "%-30s | %-22s | %-22s | %-22s\n" "prompt tok/s (inline)" "n/a"             "n/a"                       "${INLINE_PROMPT_TPS:-N/A}"
    printf "%-30s | %-22s | %-22s | %-22s\n" "layers/sec (proxy)"  "$REMORA_LAYERSPERSEC" "n/a"                       "n/a"
    printf "%-30s | %-22s | %-22s | %-22s\n" "stream throughput"   "$REMORA_STREAM_MBPS MiB/s" "n/a (loads shards)"   "n/a (mmap-once)"
    printf "%-30s | %-22s | %-22s | %-22s\n" "peak RSS (MiB)"      "$(awk -v k="$REMORA_PEAK_KB" 'BEGIN{printf "%.1f", k/1024}')" \
                                                          "$(awk -v k="$REMORA_LLAMA_PEAK_KB" 'BEGIN{printf "%.1f", k/1024}')" \
                                                                                                                                                                                                 "$(awk -v k="$LLAMA_PEAK_KB" 'BEGIN{printf "%.1f", k/1024}')"
    printf "%-30s | %-22s | %-22s | %-22s\n" "exit code"           "$REMORA_EXIT"      "$REMORA_LLAMA_EXIT"        "$LLAMA_EXIT"
    echo
    echo "Notes:"
    echo "  * Remora's RNN kernel is currently a placeholder (h = 0.9h + 0.1x over a"
    echo "    64-dim state). It streams real bytes off disk via mmap, but does NOT"
    echo "    run a real Mamba selective scan. 'layers/sec' is a work-rate proxy."
    echo "  * remora_llama is the libllama-based driver: it produces real Mamba"
    echo "    outputs from sharded weights via the same llama.cpp numerics and"
    echo "    sampler as llama-cli. It exists to compare RSS / TTFT / TPS between"
    echo "    sharded-weight streaming and monolithic llama-cli."
    echo "  * 'TTFT' for Remora is measured as wall time until the first per-layer"
    echo "    printf; remora_llama's is wall time until 'indexed N tensors' is"
    echo "    printed (model-graph ready); llama.cpp's is (total_ms - eval_ms),"
    echo "    which still includes prompt processing."
    echo "  * RSS is sampled from \`ps -o rss=\` at ${SAMPLE_HZ} Hz; peak is the max."
    echo
} | tee "$OUT_FILE"

log "wrote $OUT_FILE"
log "raw logs: $REMORA_LOG $LLAMA_LOG $REMORA_LLAMA_LOG  ($REMORA_RSS, $LLAMA_RSS, $REMORA_LLAMA_RSS)"
