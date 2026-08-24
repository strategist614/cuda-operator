from pathlib import Path

import torch
from torch.utils.cpp_extension import load

root = Path(__file__).parent

my_operator = load(
    name="my_operator_cuda",
    sources=[
        str(root / "binding.cpp"),
        str(root / "my_kernel.cu"),
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)

a = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
b = torch.randn_like(a)

reference = a + b
output = my_operator.forward(a, b)

torch.testing.assert_close(output, reference)

print("结果正确")
print(output)