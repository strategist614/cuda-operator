#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

__global__ void add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    int64_t n)
{
    int64_t index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index < n) {
        output[index] = a[index] + b[index];
    }
}

at::Tensor add_cuda(at::Tensor a, at::Tensor b)
{
    // 确保 kernel 在输入 Tensor 所在 GPU 上启动
    c10::cuda::CUDAGuard device_guard(a.device());

    // 如果你的 kernel 只支持连续内存，需要这样处理
    auto a_contiguous = a.contiguous();
    auto b_contiguous = b.contiguous();

    auto output = at::empty_like(a_contiguous);
    int64_t n = a_contiguous.numel();

    if (n == 0) {
        return output;
    }

    constexpr int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);

    // 一定使用 PyTorch 当前 CUDA stream
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(a.get_device()).stream();

    add_kernel<<<blocks, threads, 0, stream>>>(
        a_contiguous.data_ptr<float>(),
        b_contiguous.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}