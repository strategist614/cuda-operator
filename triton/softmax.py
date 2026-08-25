import torch
import triton
import triton.language as tl


MAX_FUSED_SIZE = 65536
SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def softmax_kernel(
    x_ptr,
    output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    row_start = row * n_cols

    x = tl.load(x_ptr + row_start + cols, mask=mask, other=-float("inf")).to(
        tl.float32
    )
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    output = numerator / denominator
    tl.store(output_ptr + row_start + cols, output, mask=mask)


def _validate_input(x: torch.Tensor) -> None:
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.ndim == 0 or x.numel() == 0:
        raise ValueError("x must have at least one non-empty dimension")
    if x.dtype not in SUPPORTED_DTYPES:
        raise TypeError("x must have dtype float16, bfloat16, or float32")


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Apply a numerically stable Softmax over the last dimension."""
    _validate_input(x)

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
    softmax_kernel[(n_rows,)](
        x,
        output,
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output


def _test_case(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    x = 2.0 * torch.randn(shape, device="cuda", dtype=dtype)

    expected = torch.softmax(x, dim=-1)
    actual = triton_softmax(x)
    tolerance = 1.0e-5 if dtype == torch.float32 else 1.0e-3
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    max_abs_error = (actual.float() - expected.float()).abs().max().item()
    row_sum_error = (actual.float().sum(dim=-1) - 1.0).abs().max().item()
    print(
        f"kernel=triton_softmax shape={shape} dtype={dtype} "
        f"max_abs_error={max_abs_error:.8f} "
        f"max_row_sum_error={row_sum_error:.8f} status=PASS"
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
