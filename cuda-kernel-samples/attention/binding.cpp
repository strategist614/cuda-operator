#include <torch/extension.h>


at::Tensor basic_attention_cuda(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value);

at::Tensor flash_attention_cuda(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value);


void check_inputs(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value)
{
    TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
    TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
    TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
    TORCH_CHECK(
        query.device() == key.device() && query.device() == value.device(),
        "query, key, and value must be on the same CUDA device");
    TORCH_CHECK(
        query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
        "query, key, and value must have shape [batch, heads, sequence, dim]");
    TORCH_CHECK(
        query.sizes() == key.sizes() && query.sizes() == value.sizes(),
        "query, key, and value must have identical shapes");
    TORCH_CHECK(
        query.scalar_type() == key.scalar_type() &&
            query.scalar_type() == value.scalar_type(),
        "query, key, and value must have the same dtype");
    TORCH_CHECK(
        query.scalar_type() == at::kHalf || query.scalar_type() == at::kFloat,
        "only float16 and float32 are supported");
    TORCH_CHECK(
        query.is_contiguous() && key.is_contiguous() && value.is_contiguous(),
        "query, key, and value must be contiguous");
    TORCH_CHECK(query.size(2) > 0, "sequence length must be positive");
    TORCH_CHECK(
        query.size(3) > 0 && query.size(3) <= 256,
        "head dimension must be in [1, 256]");
}


at::Tensor basic_attention(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value)
{
    check_inputs(query, key, value);
    return basic_attention_cuda(query, key, value);
}


at::Tensor flash_attention(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value)
{
    check_inputs(query, key, value);
    return flash_attention_cuda(query, key, value);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    module.def(
        "basic_attention",
        &basic_attention,
        "Basic three-stage attention (CUDA)");
    module.def(
        "flash_attention",
        &flash_attention,
        "Fused online-softmax attention (CUDA)");
}
