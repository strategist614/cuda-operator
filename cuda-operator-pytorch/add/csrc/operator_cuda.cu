#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void vector_add_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ output,
    int64_t numel)
{
    const int64_t index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index < numel) {
        output[index] = a[index] + b[index];
    }
}

at::Tensor vector_add_cuda(
    const at::Tensor& a,
    const at::Tensor& b)
{
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");

    TORCH_CHECK(
        a.device() == b.device(),
        "a and b must be on the same CUDA device"
    );

    TORCH_CHECK(
        a.scalar_type() == b.scalar_type(),
        "a and b must have the same dtype"
    );

    TORCH_CHECK(
        a.sizes() == b.sizes(),
        "a and b must have the same shape"
    );

    TORCH_CHECK(
        a.scalar_type() == at::kFloat ||
        a.scalar_type() == at::kDouble ||
        a.scalar_type() == at::kHalf,
        "only float16, float32 and float64 are supported"
    );

    // 保证 kernel 在输入 Tensor 所在设备运行
    const c10::cuda::CUDAGuard device_guard(a.device());

    // 当前 kernel 按连续内存寻址
    const auto a_contiguous = a.contiguous();
    const auto b_contiguous = b.contiguous();

    auto output = at::empty_like(a_contiguous);

    const int64_t numel = a_contiguous.numel();

    if (numel == 0) {
        return output;
    }

    constexpr int threads = 256;
    const int blocks =
        static_cast<int>((numel + threads - 1) / threads);

    // 必须接入 PyTorch 当前 stream
    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(a.get_device()).stream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        a_contiguous.scalar_type(),
        "vector_add_cuda",
        [&] {
            vector_add_kernel<scalar_t>
                <<<blocks, threads, 0, stream>>>(
                    a_contiguous.data_ptr<scalar_t>(),
                    b_contiguous.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    numel
                );
        }
    );

    // 检查 kernel 启动错误，不进行全局同步
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}