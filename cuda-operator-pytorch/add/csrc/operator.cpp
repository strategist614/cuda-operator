#include <torch/extension.h>

at::Tensor vector_add_cuda(
    const at::Tensor& a,
    const at::Tensor& b
);

at::Tensor vector_add_cpu(
    const at::Tensor& a,
    const at::Tensor& b)
{
    TORCH_CHECK(!a.is_cuda(), "a must be a CPU tensor");
    TORCH_CHECK(!b.is_cuda(), "b must be a CPU tensor");

    TORCH_CHECK(
        a.device() == b.device(),
        "a and b must be on the same device"
    );

    TORCH_CHECK(
        a.scalar_type() == b.scalar_type(),
        "a and b must have the same dtype"
    );

    TORCH_CHECK(
        a.sizes() == b.sizes(),
        "a and b must have the same shape"
    );

    return at::add(a, b);
}

// 定义算子 schema
TORCH_LIBRARY(cuda_operator, module)
{
    module.def(
        "vector_add(Tensor a, Tensor b) -> Tensor"
    );
}

// 注册 CPU 实现
TORCH_LIBRARY_IMPL(cuda_operator, CPU, module)
{
    module.impl(
        "vector_add",
        TORCH_FN(vector_add_cpu)
    );
}

// 注册 CUDA 实现
TORCH_LIBRARY_IMPL(cuda_operator, CUDA, module)
{
    module.impl(
        "vector_add",
        TORCH_FN(vector_add_cuda)
    );
}

// 创建可被 Python import 的空扩展模块。
// import 时，上面的 TORCH_LIBRARY 静态注册会执行。
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
}