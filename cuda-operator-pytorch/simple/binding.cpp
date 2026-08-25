#include <torch/extension.h>

at::Tensor add_cuda(at::Tensor a, at::Tensor b);

at::Tensor add(at::Tensor a, at::Tensor b)
{
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.device() == b.device(),
                "a and b must be on the same GPU");
    TORCH_CHECK(a.scalar_type() == at::kFloat,
                "a must be float32");
    TORCH_CHECK(b.scalar_type() == at::kFloat,
                "b must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(),
                "a and b must have the same shape");

    return add_cuda(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    module.def("forward", &add, "Optimized CUDA add");
}