#!/usr/bin/env python3
"""Benchmark an MLX model's generation speed (tokens/sec).

Unlike scripts/run_mlx.py (which stops at EOS), this forces exactly N tokens
so we get a clean tokens/sec number even for base models that emit EOS
immediately. It reports prompt tokens/sec, generation tokens/sec, and peak
memory.

Usage:
    python3 scripts/bench_mlx.py --model mlx_model/ [--prompt "Hello"] \
        [--tokens 64] [--temp 0.0] [--warmup 2]
"""

import argparse
import resource
import sys
import time

import mlx.core as mx


def peak_rss_bytes() -> int:
    """Peak resident set size in bytes (macOS / Linux ru_maxrss)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="Path to the MLX model directory")
    parser.add_argument("--prompt", default="Hello, my name is",
                        help="Prompt text")
    parser.add_argument("--tokens", type=int, default=64,
                        help="Number of tokens to generate")
    parser.add_argument("--temp", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup tokens before timing")
    args = parser.parse_args()

    try:
        from mlx_lm import load
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        sys.stderr.write("error: mlx-lm is required. Install with:\n"
                         "    pip install mlx-lm\n")
        return 1

    print(f"[bench_mlx] Loading model from {args.model}")
    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    t_load = time.perf_counter() - t0
    print(f"[bench_mlx] Model loaded in {t_load:.2f}s")

    prompt = args.prompt
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    prompt_tokens = tokenizer.encode(prompt)
    print(f"[bench_mlx] Prompt: {len(prompt_tokens)} tokens")

    sampler = make_sampler(temp=args.temp)

    # --- Prompt processing (prefill) timing ---
    mx.clear_cache()
    t0 = time.perf_counter()
    gen = generate_step(mx.array(prompt_tokens), model, max_tokens=1, sampler=sampler)
    for _ in gen:
        pass
    mx.synchronize()
    t_prefill = time.perf_counter() - t0
    prompt_tps = len(prompt_tokens) / t_prefill
    print(f"[bench_mlx] Prefill: {t_prefill*1000:.1f} ms "
          f"({prompt_tps:.1f} prompt tok/s)")

    # --- Generation timing (force N tokens, ignore EOS) ---
    mx.clear_cache()
    gen = generate_step(mx.array(prompt_tokens), model, max_tokens=args.tokens,
                        sampler=sampler)
    t0 = time.perf_counter()
    n = 0
    for _ in gen:
        n += 1
        if n == 1:
            ttft = time.perf_counter() - t0
    mx.synchronize()
    t_gen = time.perf_counter() - t0
    gen_tps = n / t_gen if t_gen > 0 else 0.0

    peak_gb = mx.get_peak_memory() / 1e9
    rss_mib = peak_rss_bytes() / (1024 * 1024)
    ttft_ms = ttft * 1000 if n > 0 else float("nan")
    # TTFT is dominated by prefill; combine prefill + first token latency.
    ttft_total_ms = t_prefill * 1000 + ttft_ms

    print(f"[bench_mlx] Generated {n} tokens in {t_gen*1000:.1f} ms")
    print(f"[bench_mlx] Generation: {gen_tps:.2f} tok/s")
    print(f"[bench_mlx] TTFT (first token, after prefill): {ttft_ms:.1f} ms")
    print(f"[bench_mlx] TTFT (incl. prefill): {ttft_total_ms:.1f} ms")
    print(f"[bench_mlx] Peak memory: {peak_gb:.3f} GB")
    print(f"[bench_mlx] Peak RSS: {rss_mib:.1f} MiB")
    print(f"[bench_mlx] RESULT gen_tps={gen_tps:.2f} prompt_tps={prompt_tps:.1f} "
          f"ttft_ms={ttft_total_ms:.1f} rss_mib={rss_mib:.1f} "
          f"peak_gb={peak_gb:.3f} load_s={t_load:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
