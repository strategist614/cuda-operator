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

constexpr int WARP_SIZE  = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_WARPS  = BLOCK_SIZE / WARP_SIZE;
constexpr int MAX_N      = 1024;


// ============================================================
// Warp Reduce Max
// ============================================================

__device__ __forceinline__
float warpReduceMax(float value) {

    constexpr unsigned FULL_MASK = 0xffffffff;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {

        value = fmaxf(
            value,
            __shfl_down_sync(
                FULL_MASK,
                value,
                offset
            )
        );
    }

    return value;
}


// ============================================================
// Warp Reduce Sum
// ============================================================

__device__ __forceinline__
float warpReduceSum(float value) {

    constexpr unsigned FULL_MASK = 0xffffffff;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {

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
// ============================================================

__device__ __forceinline__
float blockReduceMax(
    float value,
    float* shared) {

    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;

    // 每个 warp 内求最大值
    value = warpReduceMax(value);

    // 每个 warp 的 lane 0 写结果
    if (lane == 0) {
        shared[warp_id] = value;
    }

    __syncthreads();

    // warp 0 对 8 个 warp 的结果继续 reduction
    float block_value = -INFINITY;

    if (warp_id == 0) {

        if (lane < NUM_WARPS) {
            block_value = shared[lane];
        }

        block_value =
            warpReduceMax(block_value);

        if (lane == 0) {
            shared[0] = block_value;
        }
    }

    __syncthreads();

    return shared[0];
}


// ============================================================
// Block Reduce Sum
// ============================================================

__device__ __forceinline__
float blockReduceSum(
    float value,
    float* shared) {

    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;

    // 每个 warp 内求和
    value = warpReduceSum(value);

    if (lane == 0) {
        shared[warp_id] = value;
    }

    __syncthreads();

    float block_value = 0.0f;

    if (warp_id == 0) {

        if (lane < NUM_WARPS) {
            block_value = shared[lane];
        }

        block_value =
            warpReduceSum(block_value);

        if (lane == 0) {
            shared[0] = block_value;
        }
    }

    __syncthreads();

    return shared[0];
}


// ============================================================
// Softmax V3
//
// 256 threads
// 每个线程最多处理 4 个元素
//
// 不再使用：
// float values[4];
//
// 改成：
// float v0, v1, v2, v3;
//
// 让编译器更容易把它们分配到寄存器
// ============================================================

__global__ void softmaxV3(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n) {

    const int tid = threadIdx.x;

    __shared__ float shared[NUM_WARPS];


    // ========================================================
    // Step 1
    // 计算当前线程负责的 4 个 index
    //
    // thread 0:
    //   i0 = 0
    //   i1 = 256
    //   i2 = 512
    //   i3 = 768
    //
    // thread 1:
    //   i0 = 1
    //   i1 = 257
    //   ...
    // ========================================================

    const int i0 = tid;
    const int i1 = tid + BLOCK_SIZE;
    const int i2 = tid + BLOCK_SIZE * 2;
    const int i3 = tid + BLOCK_SIZE * 3;


    // ========================================================
    // Step 2
    // Load input -> scalar registers
    // ========================================================
    // 这里是用单个local变量 更容易被编译器分配到寄存器
    float v0 = -INFINITY;
    float v1 = -INFINITY;
    float v2 = -INFINITY;
    float v3 = -INFINITY;


    if (i0 < n) {
        v0 = input[i0];
    }

    if (i1 < n) {
        v1 = input[i1];
    }

    if (i2 < n) {
        v2 = input[i2];
    }

    if (i3 < n) {
        v3 = input[i3];
    }


    // ========================================================
    // Step 3
    // 当前线程自己的 local max
    // ========================================================

    float local_max = v0;

    local_max = fmaxf(local_max, v1);
    local_max = fmaxf(local_max, v2);
    local_max = fmaxf(local_max, v3);


    // ========================================================
    // Step 4
    // 256 threads -> block max
    // ========================================================

    const float max_val =
        blockReduceMax(
            local_max,
            shared
        );


    // ========================================================
    // Step 5
    // exp
    //
    // 用 e0 e1 e2 e3 保存 exp 结果
    //
    // 同样希望它们保留在寄存器
    // ========================================================

    float e0 = 0.0f;
    float e1 = 0.0f;
    float e2 = 0.0f;
    float e3 = 0.0f;


    if (i0 < n) {
        e0 = expf(v0 - max_val);
    }

    if (i1 < n) {
        e1 = expf(v1 - max_val);
    }

    if (i2 < n) {
        e2 = expf(v2 - max_val);
    }

    if (i3 < n) {
        e3 = expf(v3 - max_val);
    }


    // ========================================================
    // Step 6
    // 当前线程自己的 local sum
    // ========================================================

    const float local_sum =
        e0 + e1 + e2 + e3;


    // ========================================================
    // Step 7
    // 256 threads -> block sum
    // ========================================================

    const float sum =
        blockReduceSum(
            local_sum,
            shared
        );


    // ========================================================
    // Step 8
    // Normalize
    // ========================================================

    const float inv_sum =
        1.0f / sum;


    // ========================================================
    // Step 9
    // 最终才写 Global Memory
    // ========================================================

    if (i0 < n) {
        output[i0] =
            e0 * inv_sum;
    }

    if (i1 < n) {
        output[i1] =
            e1 * inv_sum;
    }

    if (i2 < n) {
        output[i2] =
            e2 * inv_sum;
    }

    if (i3 < n) {
        output[i3] =
            e3 * inv_sum;
    }
}


// ============================================================
// CPU Reference
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

    const float inv_sum =
        1.0f / sum;

    for (int i = 0; i < n; ++i) {
        output[i] *= inv_sum;
    }
}


// ============================================================
// main
// ============================================================

int main() {

    constexpr int N = 1024;

    if (N > MAX_N) {

        std::cerr
            << "V3 supports N <= "
            << MAX_N
            << std::endl;

        return 1;
    }


    // --------------------------------------------------------
    // Host input
    // --------------------------------------------------------

    std::vector<float> h_input(N);

    for (int i = 0; i < N; ++i) {

        h_input[i] =
            static_cast<float>(i % 100)
            / 10.0f;
    }

    std::vector<float> h_output(N);


    // --------------------------------------------------------
    // Device memory
    // --------------------------------------------------------

    float* d_input  = nullptr;
    float* d_output = nullptr;

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


    // --------------------------------------------------------
    // CPU -> GPU
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            N * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    // --------------------------------------------------------
    // Launch
    //
    // 1 block
    // 256 threads
    // --------------------------------------------------------

    softmaxV3<<<1, BLOCK_SIZE>>>(
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


    // --------------------------------------------------------
    // GPU -> CPU
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            h_output.data(),
            d_output,
            N * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    // --------------------------------------------------------
    // CPU reference
    // --------------------------------------------------------

    std::vector<float> cpu_output;

    softmaxCPU(
        h_input,
        cpu_output
    );


    // --------------------------------------------------------
    // Verify
    // --------------------------------------------------------

    float max_error = 0.0f;
    float softmax_sum = 0.0f;

    for (int i = 0; i < N; ++i) {

        const float error =
            std::fabs(
                h_output[i]
                - cpu_output[i]
            );

        max_error =
            std::max(
                max_error,
                error
            );

        softmax_sum +=
            h_output[i];
    }


    std::cout
        << "N = "
        << N
        << '\n';

    std::cout
        << "Threads = "
        << BLOCK_SIZE
        << '\n';

    std::cout
        << "Softmax sum = "
        << softmax_sum
        << '\n';

    std::cout
        << "Max error = "
        << max_error
        << '\n';


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