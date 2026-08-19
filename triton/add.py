import torch
import triton
import triton.language as tl


# ============================================================
# 1. Triton Kernel
# ============================================================

@triton.jit
def add_kernel(
    x_ptr,                      # x tensor 的起始地址
    y_ptr,                      # y tensor 的起始地址
    output_ptr,                 # output tensor 的起始地址
    n_elements,                 # 元素总数
    BLOCK_SIZE: tl.constexpr,   # 每个 Triton program 处理多少个元素
):
    # 类似 CUDA 中的 blockIdx.x
    pid = tl.program_id(axis=0)

    # 当前 program 负责的数据范围
    #
    # 例如：
    # BLOCK_SIZE = 256
    #
    # pid = 0:
    # offsets = [0, 1, ..., 255]
    #
    # pid = 1:
    # offsets = [256, 257, ..., 511]
    #
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 防止最后一个 block 越界
    mask = offsets < n_elements

    # 从 global memory 读取
    x = tl.load(
        x_ptr + offsets,
        mask=mask,
    )

    y = tl.load(
        y_ptr + offsets,
        mask=mask,
    )

    # elementwise add
    output = x + y

    # 写回 global memory
    tl.store(
        output_ptr + offsets,
        output,
        mask=mask,
    )


# ============================================================
# 2. Python Wrapper
# ============================================================

def triton_add(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda
    assert y.is_cuda

    assert x.shape == y.shape
    assert x.is_contiguous()
    assert y.is_contiguous()

    # 创建输出 tensor
    output = torch.empty_like(x)

    n_elements = x.numel()

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------
    #
    # Triton 会根据 BLOCK_SIZE 决定需要启动多少个 program。
    #
    # 比如：
    #
    # n_elements = 1000
    # BLOCK_SIZE = 256
    #
    # grid = ceil(1000 / 256)
    #      = 4
    #
    # 所以启动 4 个 Triton programs
    #
    grid = lambda meta: (
        triton.cdiv(
            n_elements,
            meta["BLOCK_SIZE"],
        ),
    )

    # launch kernel
    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=256,
    )

    return output


# ============================================================
# 3. Test
# ============================================================

def main():
    torch.manual_seed(0)

    # 故意不用 256 的整数倍
    # 用来测试最后一个 program 的 mask
    n = 1000

    x = torch.randn(
        n,
        device="cuda",
        dtype=torch.float32,
    )

    y = torch.randn(
        n,
        device="cuda",
        dtype=torch.float32,
    )

    # PyTorch reference
    torch_output = x + y

    # Triton
    triton_output = triton_add(x, y)

    print("x:")
    print(x[:10])

    print("\ny:")
    print(y[:10])

    print("\nPyTorch:")
    print(torch_output[:10])

    print("\nTriton:")
    print(triton_output[:10])

    # correctness check
    torch.testing.assert_close(
        triton_output,
        torch_output,
    )

    print("\n✅ Triton result matches PyTorch")


if __name__ == "__main__":
    main()