import argparse
import math

import torch
import triton

from gemm import triton_gemm
from gemm_optimized import gemm_optimized_kernel, triton_gemm_optimized


def benchmark(
    name: str,
    function,
    m: int,
    n: int,
    k: int,
    warmup: int,
    repetitions: int,
) -> tuple[float, float]:
    average_ms = triton.testing.do_bench(
        function,
        warmup=warmup,
        rep=repetitions,
    )
    gflops = 2.0 * m * n * k / (average_ms * 1.0e6)
    print(f"provider={name} average_ms={average_ms:.8f} gflops={gflops:.4f}")
    return average_ms, gflops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark baseline/optimized Triton GEMM against PyTorch."
    )
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()
    if min(args.m, args.n, args.k) <= 0:
        parser.error("--m, --n, and --k must be positive")
    if args.warmup < 0 or args.repetitions <= 0:
        parser.error("--warmup must be non-negative and --repetitions positive")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA-capable GPU is required")
    supports_optimized = torch.cuda.get_device_capability()[0] >= 8

    torch.manual_seed(20260825)
    scale = 1.0 / math.sqrt(args.k)
    a = scale * torch.randn((args.m, args.k), device="cuda", dtype=torch.float16)
    b = torch.randn((args.k, args.n), device="cuda", dtype=torch.float16)

    baseline_output = triton_gemm(a, b)
    torch_output = torch.matmul(a, b)
    torch.testing.assert_close(
        baseline_output, torch_output, atol=2.0e-2, rtol=2.0e-2
    )
    if supports_optimized:
        optimized_output = triton_gemm_optimized(a, b)
        torch.testing.assert_close(
            optimized_output, torch_output, atol=2.0e-2, rtol=2.0e-2
        )
    torch.cuda.synchronize()

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"shape={args.m}x{args.n}x{args.k} dtype=fp16")
    print(f"warmup={args.warmup} repetitions={args.repetitions}")
    baseline_ms, _ = benchmark(
        "triton_baseline",
        lambda: triton_gemm(a, b),
        args.m,
        args.n,
        args.k,
        args.warmup,
        args.repetitions,
    )
    optimized_ms = None
    if supports_optimized:
        optimized_ms, _ = benchmark(
            "triton_optimized",
            lambda: triton_gemm_optimized(a, b),
            args.m,
            args.n,
            args.k,
            args.warmup,
            args.repetitions,
        )
    else:
        print("provider=triton_optimized status=SKIP reason=requires_sm80_plus")
    torch_ms, _ = benchmark(
        "pytorch",
        lambda: torch.matmul(a, b),
        args.m,
        args.n,
        args.k,
        args.warmup,
        args.repetitions,
    )

    if optimized_ms is not None:
        print(f"optimized_vs_baseline={baseline_ms / optimized_ms:.4f}x")
        print(f"optimized_vs_pytorch={torch_ms / optimized_ms:.4f}x")
        print(f"best_config={gemm_optimized_kernel.best_config}")
    print(f"baseline_vs_pytorch={torch_ms / baseline_ms:.4f}x")
    print("status=PASS")


if __name__ == "__main__":
    main()
