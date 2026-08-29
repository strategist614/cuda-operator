#!/usr/bin/env python3

import argparse
import subprocess

DEFAULT_SHAPES = [
    (32, 1024, 1),
    (32, 1024, 8),
    (32, 4096, 8),
    (64, 16384, 16),
    (128, 65536, 16),
    (128, 65536, 32),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", default="./build/topk_bench")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--groups", type=int, default=3)
    args = parser.parse_args()

    for B, N, K in DEFAULT_SHAPES:
        print("=" * 72, flush=True)
        print(f"B={B} N={N} K={K}", flush=True)

        subprocess.run(
            [
                args.bin,
                str(B),
                str(N),
                str(K),
                "--repeat",
                str(args.repeat),
                "--groups",
                str(args.groups),
            ],
            check=True,
        )

if __name__ == "__main__":
    main()
