import argparse
import os
from pathlib import Path
from typing import Callable


os.environ.setdefault(
    "TORCH_EXTENSIONS_DIR",
    "/tmp/cuda_attention_torch_extensions",
)

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load


DTYPES = {
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def build_extension(verbose: bool):
    root = Path(__file__).resolve().parent
    return load(
        name="cuda_attention_samples",
        sources=[
            str(root / "binding.cpp"),
            str(root / "attention_cuda.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=verbose,
    )


def benchmark_cuda_events(
    function: Callable[[], torch.Tensor],
    warmup: int,
    repetitions: int,
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        function()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / repetitions


def attention_gflops(
    batch: int,
    heads: int,
    sequence: int,
    head_dim: int,
    average_ms: float,
) -> float:
    # QK^T and PV each perform approximately 2 * B * H * S^2 * D FLOPs.
    operations = 4.0 * batch * heads * sequence * sequence * head_dim
    return operations / (average_ms * 1.0e6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark basic CUDA attention, CUDA FlashAttention, and PyTorch SDPA."
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=DTYPES, default="fp16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    if min(args.batch, args.heads, args.sequence, args.head_dim) <= 0:
        parser.error("all dimensions must be positive")
    if args.head_dim > 256:
        parser.error("--head-dim must not exceed 256")
    if args.warmup < 0 or args.repetitions <= 0:
        parser.error("--warmup must be non-negative and --repetitions positive")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA-capable GPU is required")

    extension = build_extension(args.verbose_build)
    dtype = DTYPES[args.dtype]
    shape = (args.batch, args.heads, args.sequence, args.head_dim)
    torch.manual_seed(20260826)
    query = 0.5 * torch.randn(shape, device="cuda", dtype=dtype)
    key = 0.5 * torch.randn(shape, device="cuda", dtype=dtype)
    value = torch.randn(shape, device="cuda", dtype=dtype)

    with torch.no_grad():
        basic_output = extension.basic_attention(query, key, value)
        flash_output = extension.flash_attention(query, key, value)
        torch_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )

    tolerance = 2.0e-4 if dtype == torch.float32 else 3.0e-3
    torch.testing.assert_close(
        basic_output,
        torch_output,
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        flash_output,
        torch_output,
        atol=tolerance,
        rtol=tolerance,
    )
    basic_error = (basic_output.float() - torch_output.float()).abs().max().item()
    flash_error = (flash_output.float() - torch_output.float()).abs().max().item()

    basic_function = lambda: extension.basic_attention(query, key, value)
    flash_function = lambda: extension.flash_attention(query, key, value)
    torch_function = lambda: F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=False,
    )

    with torch.no_grad():
        basic_ms = benchmark_cuda_events(
            basic_function, args.warmup, args.repetitions
        )
        flash_ms = benchmark_cuda_events(
            flash_function, args.warmup, args.repetitions
        )
        torch_ms = benchmark_cuda_events(
            torch_function, args.warmup, args.repetitions
        )

    basic_gflops = attention_gflops(*shape, basic_ms)
    flash_gflops = attention_gflops(*shape, flash_ms)
    torch_gflops = attention_gflops(*shape, torch_ms)
    flash_provider = (
        "cuda_flash_wmma"
        if dtype == torch.float16
        and args.sequence % 16 == 0
        and args.head_dim % 16 == 0
        and torch.cuda.get_device_capability()[0] >= 7
        else "cuda_flash_online_fallback"
    )
    score_mib = (
        args.batch
        * args.heads
        * args.sequence
        * args.sequence
        * torch.tensor([], dtype=torch.float32).element_size()
        / (1024.0 * 1024.0)
    )

    print(f"device={torch.cuda.get_device_name(0)}")
    print(
        f"shape={args.batch}x{args.heads}x{args.sequence}x{args.head_dim} "
        f"dtype={args.dtype}"
    )
    print(f"warmup={args.warmup} repetitions={args.repetitions}")
    print(f"basic_score_intermediate_mib={score_mib:.4f}")
    print(
        f"provider=cuda_basic average_ms={basic_ms:.8f} "
        f"gflops={basic_gflops:.4f} max_abs_error={basic_error:.8f} status=PASS"
    )
    print(
        f"provider={flash_provider} average_ms={flash_ms:.8f} "
        f"gflops={flash_gflops:.4f} max_abs_error={flash_error:.8f} status=PASS"
    )
    print(
        f"provider=torch_sdpa average_ms={torch_ms:.8f} "
        f"gflops={torch_gflops:.4f} status=PASS"
    )
    print(f"flash_vs_basic={basic_ms / flash_ms:.4f}x")
    print(f"flash_vs_torch={torch_ms / flash_ms:.4f}x")
    print("status=PASS")


if __name__ == "__main__":
    main()
