#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA Error: "                       \
                      << cudaGetErrorString(err)               \
                      << " at " << __FILE__                    \
                      << ":" << __LINE__                       \
                      << std::endl;                            \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)


// ============================================================
// Warp Reduce Max
// ============================================================

__device__ float warpReduceMax(
    float value,
    unsigned mask) {

    for (int offset = 16; offset > 0; offset >>= 1) {

        float other =
            __shfl_down_sync(
                mask,
                value,
                offset
            );

        value = fmaxf(value, other);
    }

    return value;
}


// ============================================================
// Warp Reduce Sum
// ============================================================

__device__ float warpReduceSum(
    float value,
    unsigned mask) {

    for (int offset = 16; offset > 0; offset >>= 1) {

        float other =
            __shfl_down_sync(
                mask,
                value,
                offset
            );

        value += other;
    }

    return value;
}


// ============================================================
// Softmax V1
//
// 限制：
//   N <= 32
//
// 一个 warp 处理整条一维向量
// 一个线程负责一个元素
// ============================================================

__global__ void softmaxV1(
    const float* input,
    float* output,
    int n) {

    const int tid = threadIdx.x;

    // --------------------------------------------------------
    // 哪些线程是真正参与计算的线程
    //
    // n = 10 时：
    // lane 0~9 有效
    // lane 10~31 无效
    // --------------------------------------------------------
    // 生成一个“哪些线程有效”的32位名单
    // 根据 n 的大小，mask 的低 n 位为 1，高位为 0
    // 也就是说当前的 tid < n 的线程才是有效的线程 也就是true
    const unsigned mask =
        __ballot_sync(
            0xffffffff,
            tid < n
        );

    // --------------------------------------------------------
    // Step 1:
    // 每个线程读取一个元素
    // --------------------------------------------------------

    float value = -INFINITY;

    if (tid < n) {
        value = input[tid];
    }

    // --------------------------------------------------------
    // Step 2:
    // Warp Reduce Max
    // --------------------------------------------------------

    float max_val =
        warpReduceMax(
            value,
            mask
        );

    // reduction 后，lane 0 拥有最终 max
    //
    // 需要广播给整个 warp
    max_val =
        __shfl_sync(
            mask,
            max_val,
            0
        );

    // --------------------------------------------------------
    // Step 3:
    // exp(x - max)
    // --------------------------------------------------------

    float exp_value = 0.0f;

    if (tid < n) {

        exp_value =
            expf(
                input[tid] - max_val
            );
    }

    // --------------------------------------------------------
    // Step 4:
    // Warp Reduce Sum
    // --------------------------------------------------------

    float sum =
        warpReduceSum(
            exp_value,
            mask
        );

    // lane 0 得到最终 sum
    // 再广播给整个 warp
    sum =
        __shfl_sync(
            mask,
            sum,
            0
        );

    // --------------------------------------------------------
    // Step 5:
    // Normalize
    // --------------------------------------------------------

    if (tid < n) {

        output[tid] =
            exp_value / sum;
    }
}


// ============================================================
// CPU reference
// ============================================================

void softmaxCPU(
    const std::vector<float>& input,
    std::vector<float>& output) {

    const int n =
        static_cast<int>(input.size());

    output.resize(n);

    float max_val = -INFINITY;

    for (int i = 0; i < n; ++i) {
        max_val =
            std::max(
                max_val,
                input[i]
            );
    }

    float sum = 0.0f;

    for (int i = 0; i < n; ++i) {

        output[i] =
            std::exp(
                input[i] - max_val
            );

        sum += output[i];
    }

    for (int i = 0; i < n; ++i) {
        output[i] /= sum;
    }
}


// ============================================================
// main
// ============================================================

int main() {

    // --------------------------------------------------------
    // 输入
    // --------------------------------------------------------

    std::vector<float> h_input = {
        1.0f,
        2.0f,
        3.0f,
        4.0f,
        5.0f,
        6.0f,
        7.0f,
        8.0f
    };

    const int n =
        static_cast<int>(
            h_input.size()
        );

    if (n > 32) {

        std::cerr
            << "Softmax V1 only supports n <= 32"
            << std::endl;

        return 1;
    }

    std::vector<float> h_output(n);

    // --------------------------------------------------------
    // GPU memory
    // --------------------------------------------------------

    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            n * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_output,
            n * sizeof(float)
        )
    );

    // --------------------------------------------------------
    // CPU -> GPU
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            n * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    // --------------------------------------------------------
    // Launch
    //
    // 一个 block
    // 一个 warp
    // 32 threads
    // --------------------------------------------------------

    constexpr int THREADS = 32;

    softmaxV1<<<1, THREADS>>>(
        d_input,
        d_output,
        n
    );

    CUDA_CHECK(
        cudaGetLastError()
    );

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );

    // --------------------------------------------------------
    // GPU -> CPU
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            h_output.data(),
            d_output,
            n * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    // --------------------------------------------------------
    // CUDA output
    // --------------------------------------------------------

    std::cout << "CUDA Softmax V1:\n";

    for (float x : h_output) {
        std::cout << x << " ";
    }

    std::cout << "\n\n";

    // --------------------------------------------------------
    // CPU reference
    // --------------------------------------------------------

    std::vector<float> cpu_output;

    softmaxCPU(
        h_input,
        cpu_output
    );

    std::cout << "CPU reference:\n";

    for (float x : cpu_output) {
        std::cout << x << " ";
    }

    std::cout << '\n';

    // --------------------------------------------------------
    // Free
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaFree(d_input)
    );

    CUDA_CHECK(
        cudaFree(d_output)
    );

    return 0;
}
