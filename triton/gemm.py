import math

import torch
import triton
import triton.language as tl


SUPPORTED_DTYPES = {torch.float16, torch.float32}


@triton.jit
def gemm_kernel(
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
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    b_ptrs = b_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(k, BLOCK_K)):
        current_k = k_start * BLOCK_K + offsets_k
        a = tl.load(
            a_ptrs,
            mask=(offsets_m[:, None] < m) & (current_k[None, :] < k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(current_k[:, None] < k) & (offsets_n[None, :] < n),
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    c_mask = (offsets_m[:, None] < m) & (offsets_n[None, :] < n)
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
        raise TypeError("a and b must have the same float16 or float32 dtype")


def triton_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply two contiguous row-major CUDA matrices with a fixed tile."""
    _validate_inputs(a, b)
    m, k = a.shape
    _, n = b.shape
    output = torch.empty((m, n), device=a.device, dtype=a.dtype)

    block_m = 32
    block_n = 32
    block_k = 32
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    gemm_kernel[grid](
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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=1,
    )
    return output


def _test_case(m: int, n: int, k: int, dtype: torch.dtype) -> None:
    scale = 1.0 / math.sqrt(k)
    a = scale * torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n), device="cuda", dtype=dtype)
    expected = torch.matmul(a, b)
    actual = triton_gemm(a, b)
    tolerance = 2.0e-4 if dtype == torch.float32 else 2.0e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    max_abs_error = (actual.float() - expected.float()).abs().max().item()
    print(
        f"kernel=triton_gemm shape={m}x{n}x{k} dtype={dtype} "
        f"max_abs_error={max_abs_error:.8f} status=PASS"
    )


def main() -> None:
    torch.manual_seed(20260825)
    _test_case(257, 259, 384, torch.float32)
    _test_case(257, 259, 384, torch.float16)
    _test_case(512, 512, 512, torch.float16)


if __name__ == "__main__":
    main()
