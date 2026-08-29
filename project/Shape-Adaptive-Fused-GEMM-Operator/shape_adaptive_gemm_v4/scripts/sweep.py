#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

SHAPES = [
    (8, 4096, 4096),
    (16, 4096, 4096),
    (32, 4096, 4096),
    (64, 4096, 4096),
    (128, 4096, 4096),
    (256, 4096, 4096),
    (512, 4096, 4096),
    (1024, 4096, 4096),

    (4096, 8, 4096),
    (4096, 16, 4096),
    (4096, 32, 4096),
    (4096, 64, 4096),

    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
]

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bin",
        default="./build/shape_gemm"
    )

    parser.add_argument(
        "--cache",
        default="results/gemm_cache_v4.csv"
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

    args = parser.parse_args()

    Path(
        args.cache
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    for M, N, K in SHAPES:
        print(
            "=" * 72,
            flush=True
        )

        print(
            f"Shape {M} x {N} x {K}",
            flush=True
        )

        subprocess.run(
            [
                args.bin,
                str(M),
                str(N),
                str(K),
                "--retune",
                "--cache",
                args.cache,
                "--repeat",
                str(args.repeat),
                "--groups",
                str(args.groups),
                "--no-cublas",
            ],
            check=True
        )

if __name__ == "__main__":
    main()
