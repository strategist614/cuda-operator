import argparse
from collections.abc import Callable

import torch
import torch.nn.functional as F
import triton

from layernorm import triton_layer_norm
from rmsnorm import triton_rms_norm
from softmax import triton_softmax


DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def effective_bandwidth_gbps(x: torch.Tensor, average_ms: float) -> float:
    minimum_bytes = 2 * x.numel() * x.element_size()
    return minimum_bytes / (average_ms * 1.0e6)


def benchmark_pair(
    operator: str,
    dtype_name: str,
    x: torch.Tensor,
    triton_function: Callable[[], torch.Tensor],
    torch_function: Callable[[], torch.Tensor],
    atol: float,
    rtol: float,
    warmup: int,
    repetitions: int,
) -> None:
    triton_output = triton_function()
    torch_output = torch_function()
    torch.testing.assert_close(triton_output, torch_output, atol=atol, rtol=rtol)
    max_abs_error = (
        (triton_output.float() - torch_output.float()).abs().max().item()
    )
    torch.cuda.synchronize()

    triton_ms = triton.testing.do_bench(
        triton_function,
        warmup=warmup,
        rep=repetitions,
    )
    torch_ms = triton.testing.do_bench(
        torch_function,
        warmup=warmup,
        rep=repetitions,
    )

    triton_bandwidth = effective_bandwidth_gbps(x, triton_ms)
    torch_bandwidth = effective_bandwidth_gbps(x, torch_ms)
    speedup = torch_ms / triton_ms
    print(
        f"operator={operator} dtype={dtype_name} "
        f"triton_ms={triton_ms:.8f} torch_ms={torch_ms:.8f} "
        f"speedup={speedup:.4f}x "
        f"triton_effective_gbps={triton_bandwidth:.4f} "
        f"torch_effective_gbps={torch_bandwidth:.4f} "
        f"max_abs_error={max_abs_error:.8f} status=PASS"
    )


def run_dtype(
    rows: int,
    hidden: int,
    dtype_name: str,
    warmup: int,
    repetitions: int,
) -> None:
    dtype = DTYPES[dtype_name]
    x = torch.randn((rows, hidden), device="cuda", dtype=dtype)
    weight = torch.randn(hidden, device="cuda", dtype=dtype)
    bias = torch.randn(hidden, device="cuda", dtype=dtype)

    norm_tolerance = 1.0e-4 if dtype == torch.float32 else 3.0e-3
    softmax_tolerance = 1.0e-5 if dtype == torch.float32 else 1.0e-3

    benchmark_pair(
        "layernorm",
        dtype_name,
        x,
        lambda: triton_layer_norm(x, weight, bias),
        lambda: F.layer_norm(x, (hidden,), weight, bias, eps=1.0e-5),
        norm_tolerance,
        norm_tolerance,
        warmup,
        repetitions,
    )
    benchmark_pair(
        "rmsnorm",
        dtype_name,
        x,
        lambda: triton_rms_norm(x, weight),
        lambda: F.rms_norm(x, (hidden,), weight, eps=1.0e-5),
        norm_tolerance,
        norm_tolerance,
        warmup,
        repetitions,
    )
    benchmark_pair(
        "softmax",
        dtype_name,
        x,
        lambda: triton_softmax(x),
        lambda: torch.softmax(x, dim=-1),
        softmax_tolerance,
        softmax_tolerance,
        warmup,
        repetitions,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Triton normalization kernels against PyTorch."
    )
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument(
        "--dtype",
        choices=("fp32", "fp16", "bf16", "all"),
        default="all",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()

    if args.rows <= 0 or args.hidden <= 0:
        parser.error("--rows and --hidden must be positive")
    if args.hidden > 65536:
        parser.error("--hidden must not exceed 65536")
    if args.warmup < 0 or args.repetitions <= 0:
        parser.error("--warmup must be non-negative and --repetitions positive")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA-capable GPU is required")

    torch.manual_seed(20260825)
    capability = torch.cuda.get_device_capability()
    dtype_names = ["fp32", "fp16"] if args.dtype == "all" else [args.dtype]
    if args.dtype == "all" and capability[0] >= 8:
        dtype_names.append("bf16")
    if "bf16" in dtype_names and capability[0] < 8:
        raise RuntimeError("BF16 benchmark requires compute capability 8.0 or newer")

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"shape={args.rows}x{args.hidden}")
    print(f"warmup={args.warmup} repetitions={args.repetitions}")
    for dtype_name in dtype_names:
        run_dtype(
            args.rows,
            args.hidden,
            dtype_name,
            args.warmup,
            args.repetitions,
        )


if __name__ == "__main__":
    main()
