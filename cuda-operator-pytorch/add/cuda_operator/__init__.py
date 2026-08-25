import torch

# 必须先加载 C++ 扩展，触发 TORCH_LIBRARY 注册
from . import _C  # noqa: F401


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.ops.cuda_operator.vector_add(a, b)


# torch.compile / FakeTensor 需要知道输出的元信息
@torch.library.register_fake("cuda_operator::vector_add")
def _vector_add_fake(a, b):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == b.dtype)
    torch._check(a.device == b.device)

    return torch.empty_like(
        a,
        memory_format=torch.contiguous_format,
    )


# vector_add(a, b) 对 a 和 b 的梯度都等于 grad_output
def _setup_context(ctx, inputs, output):
    pass


def _backward(ctx, grad_output):
    return grad_output, grad_output


torch.library.register_autograd(
    "cuda_operator::vector_add",
    _backward,
    setup_context=_setup_context,
)


__all__ = ["vector_add"]