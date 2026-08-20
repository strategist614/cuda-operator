import torch
import triton
import triton.language as tl


@triton.jit
def sigmoid_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr, ):
    