#!/usr/bin/env python3
"""Run an MLX model (produced by scripts/convert_mlx.py) with mlx_lm.

This is the MLX-native inference path for Remora. It loads the converted
MLX model and runs generation on Apple Silicon via the Metal backend. It
complements the C++ layer-by-layer streamer: the C++ path streams sharded
GGUF weights from disk, while this path runs the same model natively on
the MLX runtime (which is what `mlx_lm.convert` produces).

Usage:
    python3 scripts/run_mlx.py --model mlx_model/ [--prompt "Hello"] \
        [--tokens 32] [--temp 0.7] [--top-p 0.9]

Options:
    --model <dir>    Path to the MLX model directory (from convert_mlx.py).
    --prompt <text>  Prompt text (default: "Hello, my name is").
    --tokens <N>     Max tokens to generate (default: 32).
    --temp <T>       Sampling temperature (default: 0.7).
    --top-p <P>      Top-p sampling (default: 1.0).
    --seed <N>       PRNG seed (default: none).
    --verbose        Print timing / memory stats.
"""

import argparse
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="Path to the MLX model directory")
    parser.add_argument("--prompt", default="Hello, my name is",
                        help="Prompt text")
    parser.add_argument("--tokens", type=int, default=32,
                        help="Max tokens to generate")
    parser.add_argument("--temp", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p sampling")
    parser.add_argument("--seed", type=int, default=None,
                        help="PRNG seed")
    parser.add_argument("--verbose", action="store_true",
                        help="Print timing / memory stats")
    args = parser.parse_args()

    try:
        import mlx.core as mx
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        sys.stderr.write(
            "error: mlx-lm is required. Install with:\n"
            "    pip install mlx-lm\n"
        )
        return 1

    if args.seed is not None:
        mx.random.seed(args.seed)

    print(f"[run_mlx] Loading model from {args.model}")
    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    t_load = time.perf_counter() - t0
    print(f"[run_mlx] Model loaded in {t_load:.2f}s")

    # Apply chat template if the tokenizer has one.
    prompt = args.prompt
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

    sampler = make_sampler(temp=args.temp, top_p=args.top_p)

    print(f"[run_mlx] Generating up to {args.tokens} tokens...")
    t0 = time.perf_counter()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=args.tokens,
        sampler=sampler,
        verbose=args.verbose,
    )
    t_gen = time.perf_counter() - t0

    print(f"[run_mlx] Generated in {t_gen:.2f}s")
    if response:
        print(f"[run_mlx] Output: {response}")
    else:
        print("[run_mlx] No text generated (model may be a base model "
              "that emits EOS immediately, or the prompt was empty).")

    if args.verbose:
        import mlx.core as mx
        peak = mx.get_peak_memory() / 1e9
        print(f"[run_mlx] Peak memory: {peak:.3f} GB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
