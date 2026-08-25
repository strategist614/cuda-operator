import math

import torch
import triton
import triton.language as tl


SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}


def cuda_configs() -> list[triton.Config]:
    return [
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 8,
                "PIPELINE_STAGES": 2,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 8,
                "PIPELINE_STAGES": 3,
            },
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 8,
                "PIPELINE_STAGES": 2,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 8,
                "PIPELINE_STAGES": 2,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 8,
                "PIPELINE_STAGES": 2,
            },
            num_warps=8,
            num_stages=2,
        ),
    ]


@triton.autotune(configs=cuda_configs(), key=["m", "n", "k"])
@triton.jit
def gemm_optimized_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(m, BLOCK_M)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    programs_per_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // programs_per_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_in_group = pid % programs_per_group
    pid_m = first_pid_m + pid_in_group % group_size_m
    pid_n = pid_in_group // group_size_m

    offsets_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % m
    offsets_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % n
    offsets_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    b_ptrs = b_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in tl.range(
        0,
        tl.cdiv(k, BLOCK_K),
        num_stages=PIPELINE_STAGES,
    ):
        current_k = k_start * BLOCK_K + offsets_k
        a = tl.load(
            a_ptrs,
            mask=current_k[None, :] < k,
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=current_k[:, None] < k,
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offsets_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offsets_cm[:, None] * stride_cm + offsets_cn[None, :] * stride_cn
    c_mask = (offsets_cm[:, None] < m) & (offsets_cn[None, :] < n)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _validate_inputs(a: torch.Tensor, b: torch.Tensor) -> None:
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("a and b must be CUDA tensors")
    if a.device != b.device:
        raise ValueError("a and b must be on the same CUDA device")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be two-dimensional")
    if a.shape[1] != b.shape[0]:
        raise ValueError("a.shape[1] must equal b.shape[0]")
    if a.numel() == 0 or b.numel() == 0:
        raise ValueError("empty matrices are not supported")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("a and b must be contiguous row-major tensors")
    if a.dtype != b.dtype or a.dtype not in SUPPORTED_DTYPES:
        raise TypeError("optimized GEMM requires matching float16 or bfloat16 inputs")
    if a.dtype == torch.bfloat16 and torch.cuda.get_device_capability(a.device)[0] < 8:
        raise RuntimeError("bfloat16 Tensor Core GEMM requires compute capability 8.0+")
    if torch.cuda.get_device_capability(a.device)[0] < 8:
        raise RuntimeError(
            "optimized Tensor Core + pipelined GEMM requires compute capability 8.0+"
        )


def _launch_optimized(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
):
    m, k = a.shape
    _, n = b.shape
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    return gemm_optimized_kernel[grid](
        a,
        b,
        output,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        output.stride(1),
    )


def triton_gemm_optimized(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Tensor Core GEMM with grouped ordering and autotuned pipelining."""
    _validate_inputs(a, b)
    output = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    _launch_optimized(a, b, output)
    return output


def _test_case(m: int, n: int, k: int, dtype: torch.dtype) -> None:
    scale = 1.0 / math.sqrt(k)
    a = scale * torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n), device="cuda", dtype=dtype)
    expected = torch.matmul(a, b)
    actual = triton_gemm_optimized(a, b)
    torch.testing.assert_close(actual, expected, atol=2.0e-2, rtol=2.0e-2)
    max_abs_error = (actual.float() - expected.float()).abs().max().item()
    best_config = gemm_optimized_kernel.best_config
    print(
        f"kernel=triton_gemm_optimized shape={m}x{n}x{k} dtype={dtype} "
        f"max_abs_error={max_abs_error:.8f} "
        f"config={best_config} status=PASS"
    )


def main() -> None:
    if torch.cuda.get_device_capability()[0] < 8:
        print(f"device={torch.cuda.get_device_name(0)}")
        print("required_compute_capability=8.0+")
        print("runtime_status=SKIP")
        print("run='python gemm_sm86_compile_check.py' for offline PTX verification")
        return

    torch.manual_seed(20260825)
    _test_case(257, 259, 384, torch.float16)
    _test_case(512, 512, 512, torch.float16)
    print("runtime_status=PASS")


if __name__ == "__main__":
    main()
