import torch
import triton
import triton.language as tl


@triton.jit
def sigmoid_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr, ):
    pid = tl.program_id(axis=0)

    offsets = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    y = 1.0 / (1.0 + tl.exp(-x))

    tl.store(y_ptr + offsets, y, mask=mask)

def triton_sigmoid(x:torch.Tensor):
    assert x.is_cuda
    assert x.is_contiguous()

    y = torch.empty_like(x)

    n_elements = x.numel()

    BLOCK_SIZE = 256

    grid = (triton.cdiv(n_elements, BLOCK_SIZE), )

    sigmoid_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE, )
    
    return y

def main():
    torch.manual_seed(0)

    x=torch.randn(1000, device="cuda", dtype=torch.float32)

    y_ref = torch.sigmoid(x)

    y_triton = triton_sigmoid(x)

    print("x:")
    print(x[:10])

    print("\nPyTorch:")
    print(y_ref[:10])

    print("\nTriton:")
    print(y_triton[:10])

    torch.testing.assert_close(
        y_triton,
        y_ref,
    )

    print("\nresult correct")


if __name__ == "__main__":
    main()





