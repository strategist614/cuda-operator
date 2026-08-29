#!/usr/bin/env python3

import argparse
import subprocess

SHAPES = [
    (32, 1024, 1),
    (32, 4096, 2),
    (32, 4096, 4),
    (64, 16384, 8),
    (64, 16384, 16),

    (128, 65536, 1),
    (128, 65536, 2),
    (128, 65536, 4),
    (128, 65536, 8),
    (128, 65536, 16),
]

def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bin",
        default="./build/topk_bench"
    )

    parser.add_argument(
        "--cache",
        default="results/topk_cache_v5.csv"
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=50
    )

    parser.add_argument(
        "--groups",
        type=int,
        default=5
    )

    parser.add_argument(
        "--cache-hit-pass",
        action="store_true"
    )

    args = parser.parse_args()

    for B, N, K in SHAPES:
        print("=" * 72, flush=True)
        print(
            f"AUTOTUNE B={B} N={N} K={K}",
            flush=True
        )

        run(
            [
                args.bin,
                str(B),
                str(N),
                str(K),
                "--retune",
                "--cache",
                args.cache,
                "--repeat",
                str(args.repeat),
                "--groups",
                str(args.groups),
            ]
        )

        if args.cache_hit_pass:
            print(
                f"CACHE-HIT B={B} N={N} K={K}",
                flush=True
            )

            run(
                [
                    args.bin,
                    str(B),
                    str(N),
                    str(K),
                    "--cache",
                    args.cache,
                    "--repeat",
                    str(args.repeat),
                    "--groups",
                    str(args.groups),
                ]
            )

if __name__ == "__main__":
    main()
