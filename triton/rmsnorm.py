import torch
import triton
import triton.language as tl


MAX_FUSED_SIZE = 65536
SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def rms_norm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    n_cols,
    EPSILON: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    row_start = row * n_cols

    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) / n_cols
    inverse_rms = tl.rsqrt(mean_square + EPSILON)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    output = x * inverse_rms * weight
    tl.store(output_ptr + row_start + cols, output, mask=mask)


def _validate_inputs(x: torch.Tensor, weight: torch.Tensor) -> None:
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.ndim == 0 or x.numel() == 0:
        raise ValueError("x must have at least one non-empty dimension")
    if x.dtype not in SUPPORTED_DTYPES:
        raise TypeError("x must have dtype float16, bfloat16, or float32")

    n_cols = x.shape[-1]
    if not weight.is_cuda or weight.device != x.device:
        raise ValueError("weight must be on the same CUDA device as x")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous")
    if weight.ndim != 1 or weight.numel() != n_cols:
        raise ValueError(f"weight must have shape ({n_cols},)")
    if weight.dtype != x.dtype:
        raise TypeError("weight must have the same dtype as x")


def triton_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1.0e-5,
) -> torch.Tensor:
    """Apply RMSNorm over the last dimension of a contiguous CUDA tensor."""
    _validate_inputs(x, weight)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    n_cols = x.shape[-1]
    block_size = triton.next_power_of_2(n_cols)
    if block_size > MAX_FUSED_SIZE:
        raise ValueError(
            f"last dimension is too large for this fused kernel: {n_cols} > "
            f"{MAX_FUSED_SIZE}"
        )

    n_rows = x.numel() // n_cols
    output = torch.empty_like(x)
    num_warps = min(max(block_size // 256, 1), 8)
    rms_norm_kernel[(n_rows,)](
        x,
        weight,
        output,
        n_cols,
        EPSILON=epsilon,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output


def _rms_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    x_float = x.float()
    inverse_rms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + epsilon)
    return (x_float * inverse_rms * weight.float()).to(x.dtype)


def _test_case(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)

    expected = _rms_norm_reference(x, weight, 1.0e-5)
    actual = triton_rms_norm(x, weight)
    tolerance = 1.0e-4 if dtype == torch.float32 else 3.0e-3
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    max_abs_error = (actual.float() - expected.float()).abs().max().item()
    print(
        f"kernel=triton_rms_norm shape={shape} dtype={dtype} "
        f"max_abs_error={max_abs_error:.8f} status=PASS"
    )


def main() -> None:
    torch.manual_seed(20260825)
    dtypes = [torch.float32, torch.float16]
    if torch.cuda.get_device_capability()[0] >= 8:
        dtypes.append(torch.bfloat16)

    for dtype in dtypes:
        _test_case((257, 1000), dtype)
        _test_case((4, 8, 1024), dtype)


if __name__ == "__main__":
    main()
