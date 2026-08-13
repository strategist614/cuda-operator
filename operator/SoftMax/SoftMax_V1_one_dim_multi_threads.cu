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
//
// 一个 warp = 32 threads
// 使用 shuffle 在 warp 内求最大值
// ============================================================

__device__ __forceinline__
float warpReduceMax(float value) {

    constexpr unsigned FULL_MASK = 0xffffffff;

    for (int offset = 16; offset > 0; offset >>= 1) {
        // 取得寄存器中的浮点数
        float other =
            __shfl_down_sync(
                FULL_MASK,
                value,
                offset
            );
        // 取最大值
        value = fmaxf(value, other);
    }

    return value;
}


// ============================================================
// Warp Reduce Sum
// ============================================================

__device__ __forceinline__
float warpReduceSum(float value) {

    constexpr unsigned FULL_MASK = 0xffffffff;

    for (int offset = 16; offset > 0; offset >>= 1) {
        // 取得寄存器中的浮点数 再相加
        value +=
            __shfl_down_sync(
                FULL_MASK,
                value,
                offset
            );
    }

    return value;
}


// ============================================================
// Block Reduce Max
//
// 256 threads
// = 8 warps
//
// 第一步：
// 每个 warp 内部 reduce
//
// 第二步：
// 8 个 warp 的结果放 shared memory
//
// 第三步：
// warp 0 再 reduce 一次
// ============================================================

__device__ float blockReduceMax(float value) {

    // 256 / 32 = 8
    __shared__ float warp_results[8];

    const int tid =
        threadIdx.x;

    const int lane =
        tid % 32;

    const int warp_id =
        tid / 32;

    // --------------------------------------------------------
    // Step 1
    // 每个 warp 自己求最大值
    // --------------------------------------------------------

    value = warpReduceMax(value);

    // reduction 之后
    // 每个 warp 的 lane 0 保存该 warp 最大值
    if (lane == 0) {
        warp_results[warp_id] = value;
    }

    __syncthreads();


    // --------------------------------------------------------
    // Step 2
    // warp 0 读取 8 个 warp 的结果
    // --------------------------------------------------------

    float block_max = -INFINITY;

    if (warp_id == 0) {

        if (lane < 8) {
            block_max = warp_results[lane];
        }

        // warp 0 对这 8 个值再做一次 reduce
        block_max =
            warpReduceMax(block_max);

        // lane 0 拿到整个 block 的最大值
        if (lane == 0) {
            warp_results[0] = block_max;
        }
    }

    __syncthreads();


    // 所有线程都能拿到最终结果
    return warp_results[0];
}


// ============================================================
// Block Reduce Sum
// ============================================================

__device__ float blockReduceSum(float value) {

    __shared__ float warp_results[8];

    const int tid =
        threadIdx.x;

    const int lane =
        tid % 32;

    const int warp_id =
        tid / 32;


    // --------------------------------------------------------
    // Step 1
    // 每个 warp 自己求 sum
    // --------------------------------------------------------

    value = warpReduceSum(value);

    if (lane == 0) {
        warp_results[warp_id] = value;
    }

    __syncthreads();


    // --------------------------------------------------------
    // Step 2
    // warp 0 对 8 个结果继续求 sum
    // --------------------------------------------------------

    float block_sum = 0.0f;

    if (warp_id == 0) {

        if (lane < 8) {
            block_sum = warp_results[lane];
        }

        block_sum =
            warpReduceSum(block_sum);

        if (lane == 0) {
            warp_results[0] = block_sum;
        }
    }

    __syncthreads();


    return warp_results[0];
}


// ============================================================
// Softmax V2
//
// 一个 block
// 256 threads
// 8 warps
//
// 可以处理 N > 256
//
// 每个 thread 可以负责多个元素
// ============================================================

__global__ void softmaxV2(
    const float* input,
    float* output,
    int n) {

    const int tid =
        threadIdx.x;

    const int stride =
        blockDim.x;


    // ========================================================
    // Step 1
    // 每个线程先找自己负责数据中的最大值
    // ========================================================

    float local_max = -INFINITY;

    for (int i = tid;
         i < n;
         i += stride) {

        local_max =
            fmaxf(
                local_max,
                input[i] // global memory read
            );
    }


    // ========================================================
    // Step 2
    // 256 threads 一起 Reduce Max
    // ========================================================

    float max_val =
        blockReduceMax(local_max);


    // ========================================================
    // Step 3
    // exp(x - max)
    //
    // 每个线程计算自己的 local_sum
    // ========================================================

    float local_sum = 0.0f;

    for (int i = tid;
         i < n;
         i += stride) {

        float exp_value =
            expf(
                input[i] - max_val // global memory read
            );
        output[i] =
            exp_value; // global memory write

        local_sum +=
            exp_value;
    }


    // ========================================================
    // Step 4
    // 256 threads 一起 Reduce Sum
    // ========================================================

    float sum =
        blockReduceSum(local_sum);


    // ========================================================
    // Step 5
    // Normalize
    // ========================================================

    float inv_sum =
        1.0f / sum;

    for (int i = tid;
         i < n;
         i += stride) {

        output[i] *=
            inv_sum; // global memory write
    }
}


// ============================================================
// CPU reference
// ============================================================

void softmaxCPU(
    const std::vector<float>& input,
    std::vector<float>& output) {

    int n =
        static_cast<int>(
            input.size()
        );

    output.resize(n);


    // max
    float max_val =
        -INFINITY;

    for (int i = 0; i < n; ++i) {

        max_val =
            std::max(
                max_val,
                input[i]
            );
    }


    // exp + sum
    float sum =
        0.0f;

    for (int i = 0; i < n; ++i) {

        output[i] =
            std::exp(
                input[i] - max_val
            );

        sum +=
            output[i];
    }


    // normalize
    for (int i = 0; i < n; ++i) {

        output[i] /=
            sum;
    }
}


// ============================================================
// main
// ============================================================

int main() {

    // ========================================================
    // 创建 N = 1024 的输入
    // ========================================================

    constexpr int N = 1024;

    std::vector<float> h_input(N);

    for (int i = 0; i < N; ++i) {

        h_input[i] =
            static_cast<float>(i % 100);
    }


    std::vector<float>
        h_output(N);


    // ========================================================
    // GPU memory
    // ========================================================

    float* d_input =
        nullptr;

    float* d_output =
        nullptr;


    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            N * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_output,
            N * sizeof(float)
        )
    );


    // ========================================================
    // CPU -> GPU
    // ========================================================

    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            N * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    // ========================================================
    // Launch
    //
    // 1 block
    // 256 threads
    //
    // 256 threads = 8 warps
    // ========================================================

    constexpr int THREADS =
        256;


    softmaxV2<<<1, THREADS>>>(
        d_input,
        d_output,
        N
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    // ========================================================
    // GPU -> CPU
    // ========================================================

    CUDA_CHECK(
        cudaMemcpy(
            h_output.data(),
            d_output,
            N * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    // ========================================================
    // CPU reference
    // ========================================================

    std::vector<float>
        cpu_output;

    softmaxCPU(
        h_input,
        cpu_output
    );


    // ========================================================
    // Compare
    // ========================================================

    float max_error =
        0.0f;

    for (int i = 0; i < N; ++i) {

        float error =
            std::fabs(
                h_output[i]
                - cpu_output[i]
            );

        max_error =
            std::max(
                max_error,
                error
            );
    }


    std::cout
        << "Max error: "
        << max_error
        << '\n';


    std::cout
        << "First 10 CUDA results:\n";

    for (int i = 0; i < 10; ++i) {

        std::cout
            << h_output[i]
            << " ";
    }

    std::cout << '\n';


    // ========================================================
    // Free
    // ========================================================

    CUDA_CHECK(
        cudaFree(d_input)
    );

    CUDA_CHECK(
        cudaFree(d_output)
    );


    return 0;
}
