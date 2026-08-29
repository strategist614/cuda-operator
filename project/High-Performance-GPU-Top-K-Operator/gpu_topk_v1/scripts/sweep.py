#!/usr/bin/env python3

import argparse
import subprocess

SHAPES = [
    (32, 1024, 1),
    (32, 1024, 4),
    (32, 4096, 8),
    (64, 16384, 8),
    (64, 16384, 16),
    (128, 65536, 8),
    (128, 65536, 16),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", default="./build/topk_bench")
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--groups", type=int, default=5)
    args = parser.parse_args()

    for B, N, K in SHAPES:
        print("=" * 72, flush=True)
        print(f"B={B} N={N} K={K}", flush=True)

        subprocess.run(
            [
                args.bin,
                str(B),
                str(N),
                str(K),
                "--kernel",
                "register",
                "--repeat",
                str(args.repeat),
                "--groups",
                str(args.groups),
            ],
            check=True,
        )

if __name__ == "__main__":
    main()
