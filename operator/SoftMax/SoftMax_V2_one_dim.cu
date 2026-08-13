#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

// ============================================================
// CUDA Error Check
// ============================================================

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA Error: "                       \
                      << cudaGetErrorString(err)               \
                      << " at "                                \
                      << __FILE__                              \
                      << ":"                                   \
                      << __LINE__                              \
                      << std::endl;                            \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)


// ============================================================
// Constants
// ============================================================

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

// 每个线程最多处理 4 个元素
constexpr int ITEMS_PER_THREAD = 4;

// 最大支持：
// 256 threads * 4 elements = 1024 elements
constexpr int MAX_N =
    BLOCK_SIZE * ITEMS_PER_THREAD;


// ============================================================
// Warp Reduce Max
//
// 一个 warp 内部：
// 32 -> 16 -> 8 -> 4 -> 2 -> 1
//
// 最终 lane 0 获得 max
// ============================================================

__device__ __forceinline__
float warpReduceMax(float value) {

    constexpr unsigned FULL_MASK =
        0xffffffff;

    for (int offset = 16;
         offset > 0;
         offset >>= 1) {

        float other =
            __shfl_down_sync(
                FULL_MASK,
                value,
                offset
            );

        value =
            fmaxf(
                value,
                other
            );
    }

    return value;
}


// ============================================================
// Warp Reduce Sum
// ============================================================

__device__ __forceinline__
float warpReduceSum(float value) {

    constexpr unsigned FULL_MASK =
        0xffffffff;

    for (int offset = 16;
         offset > 0;
         offset >>= 1) {

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
// 第一级：
// 每个 warp 自己 reduce
//
// 256 values
//     ↓
// 8 warp results
//
// 第二级：
// warp 0 对这 8 个结果继续 reduce
//
// 8 values
//     ↓
// 1 block result
// ============================================================

__device__ __forceinline__
float blockReduceMax(
    float value,
    float* shared) {

    const int tid =
        threadIdx.x;

    const int lane =
        tid & 31;

    const int warp_id =
        tid >> 5;


    // ========================================================
    // Level 1
    // warp 内 reduction
    // ========================================================

    value =
        warpReduceMax(value);


    // 每个 warp 的 lane 0
    // 把结果写入 shared memory
    if (lane == 0) {

        shared[warp_id] =
            value;
    }


    // 等待 8 个 warp 都写完
    __syncthreads();


    // ========================================================
    // Level 2
    // warp 0 对 8 个结果继续 reduction
    // ========================================================

    float block_value =
        -INFINITY;


    if (warp_id == 0) {

        // lane 0~7:
        // 读取 8 个 warp 的结果
        //
        // lane 8~31:
        // 保持 -INFINITY
        if (lane < NUM_WARPS) {

            block_value =
                shared[lane];
        }


        block_value =
            warpReduceMax(
                block_value
            );


        // warp 0 lane 0
        // 获得最终 block max
        if (lane == 0) {

            shared[0] =
                block_value;
        }
    }


    __syncthreads();


    // 所有线程读取最终结果
    return shared[0];
}


// ============================================================
// Block Reduce Sum
// ============================================================

__device__ __forceinline__
float blockReduceSum(
    float value,
    float* shared) {

    const int tid =
        threadIdx.x;

    const int lane =
        tid & 31;

    const int warp_id =
        tid >> 5;


    // ========================================================
    // Level 1
    // warp reduce sum
    // ========================================================

    value =
        warpReduceSum(value);


    if (lane == 0) {

        shared[warp_id] =
            value;
    }


    __syncthreads();


    // ========================================================
    // Level 2
    // warp 0 reduce 8 个结果
    // ========================================================

    float block_value =
        0.0f;


    if (warp_id == 0) {

        if (lane < NUM_WARPS) {

            block_value =
                shared[lane];
        }


        block_value =
            warpReduceSum(
                block_value
            );


        if (lane == 0) {

            shared[0] =
                block_value;
        }
    }


    __syncthreads();


    return shared[0];
}


// ============================================================
// Softmax V3
//
// V3 核心优化：
//
// 1. 256 threads
// 2. 每个线程处理多个元素
// 3. values[] 缓存在寄存器
// 4. exp 结果不写 global memory
// 5. warp shuffle reduction
// 6. shared memory 只保存 warp reduction 结果
// ============================================================

__global__ void softmaxV3(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n) {

    const int tid =
        threadIdx.x;


    // ========================================================
    // Shared Memory
    //
    // 只需要存 8 个 warp 的结果
    // ========================================================

    __shared__ float shared[NUM_WARPS];


    // ========================================================
    // Register Cache
    //
    // 每个线程缓存最多 4 个数据
    //
    // 例如：
    //
    // thread 0:
    // values[0] = input[0]
    // values[1] = input[256]
    // values[2] = input[512]
    // values[3] = input[768]
    // ========================================================
    // 注意这里寄存器不一定是百分百存放成功的
    float values[ITEMS_PER_THREAD];


    // ========================================================
    // Step 1
    // Load Input + Local Max
    // ========================================================

    float local_max =
        -INFINITY;


    #pragma unroll
    for (int k = 0;
         k < ITEMS_PER_THREAD;
         ++k) {

        const int index =
            tid +
            k * BLOCK_SIZE;


        float value =
            -INFINITY;


        if (index < n) {

            value =
                input[index];
        }


        // 暂存在寄存器
        values[k] =
            value;


        // 每个线程先计算自己的 local max
        local_max =
            fmaxf(
                local_max,
                value
            );
    }


    // ========================================================
    // Step 2
    // Block Reduce Max
    //
    // 256 个 local_max
    //        ↓
    // 8 个 warp max
    //        ↓
    // 1 个 block max
    // ========================================================

    const float max_val =
        blockReduceMax(
            local_max,
            shared
        );


    // ========================================================
    // Step 3
    // Exp + Local Sum
    //
    // 重要：
    //
    // exp 结果继续放在 values[]
    //
    // V2:
    //
    // exp
    //  ↓
    // output[i]
    //  ↓
    // global memory
    //
    // V3:
    //
    // exp
    //  ↓
    // values[k]
    //  ↓
    // register
    // ========================================================

    float local_sum =
        0.0f;

    // 这里是循环优化 手动展开
    #pragma unroll
    for (int k = 0;
         k < ITEMS_PER_THREAD;
         ++k) {

        const int index =
            tid +
            k * BLOCK_SIZE;


        if (index < n) {

            const float exp_value =
                expf(
                    values[k]
                    - max_val
                );


            // exp 结果覆盖原来的 input
            // 继续缓存在寄存器里
            values[k] =
                exp_value;


            local_sum +=
                exp_value;
        }
    }


    // ========================================================
    // Step 4
    // Block Reduce Sum
    //
    // 每个 thread:
    // local_sum
    //
    // 256 local sums
    //      ↓
    // 8 warp sums
    //      ↓
    // total sum
    // ========================================================

    const float sum =
        blockReduceSum(
            local_sum,
            shared
        );


    // ========================================================
    // Step 5
    // Normalize
    // ========================================================

    const float inv_sum =
        1.0f / sum;


    // ========================================================
    // Step 6
    // 最终写回 Global Memory
    //
    // 到这里才写 output
    // ========================================================

    #pragma unroll
    for (int k = 0;
         k < ITEMS_PER_THREAD;
         ++k) {

        const int index =
            tid +
            k * BLOCK_SIZE;


        if (index < n) {

            output[index] =
                values[k]
                * inv_sum;
        }
    }
}


// ============================================================
// CPU Reference Softmax
// ============================================================

void softmaxCPU(
    const std::vector<float>& input,
    std::vector<float>& output) {

    const int n =
        static_cast<int>(
            input.size()
        );


    output.resize(n);


    // ========================================================
    // Max
    // ========================================================

    float max_val =
        -INFINITY;


    for (int i = 0;
         i < n;
         ++i) {

        max_val =
            std::max(
                max_val,
                input[i]
            );
    }


    // ========================================================
    // Exp + Sum
    // ========================================================

    float sum =
        0.0f;


    for (int i = 0;
         i < n;
         ++i) {

        output[i] =
            std::exp(
                input[i]
                - max_val
            );


        sum +=
            output[i];
    }


    // ========================================================
    // Normalize
    // ========================================================

    const float inv_sum =
        1.0f / sum;


    for (int i = 0;
         i < n;
         ++i) {

        output[i] *=
            inv_sum;
    }
}


// ============================================================
// main
// ============================================================

int main() {

    // ========================================================
    // Input
    //
    // 测试：
    // N = 1024
    //
    // 256 threads
    // 每个 thread 处理 4 个元素
    // ========================================================

    constexpr int N =
        1024;


    if (N > MAX_N) {

        std::cerr
            << "V3 only supports N <= "
            << MAX_N
            << std::endl;

        return 1;
    }


    // ========================================================
    // Host Input
    // ========================================================

    std::vector<float>
        h_input(N);


    for (int i = 0;
         i < N;
         ++i) {

        // 构造一些测试数据
        h_input[i] =
            static_cast<float>(
                i % 100
            ) / 10.0f;
    }


    std::vector<float>
        h_output(N);


    // ========================================================
    // Device Memory
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
    // Host -> Device
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
    // Launch Softmax V3
    //
    // 1 block
    // 256 threads
    // ========================================================

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


    // ========================================================
    // Device -> Host
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
    // CPU Reference
    // ========================================================

    std::vector<float>
        cpu_output;


    softmaxCPU(
        h_input,
        cpu_output
    );


    // ========================================================
    // Verify
    // ========================================================

    float max_error =
        0.0f;


    float cuda_sum =
        0.0f;


    for (int i = 0;
         i < N;
         ++i) {

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


        cuda_sum +=
            h_output[i];
    }


    // ========================================================
    // Print Result
    // ========================================================

    std::cout
        << "N = "
        << N
        << '\n';


    std::cout
        << "Threads = "
        << BLOCK_SIZE
        << '\n';


    std::cout
        << "Warps = "
        << NUM_WARPS
        << '\n';


    std::cout
        << "Items per thread = "
        << ITEMS_PER_THREAD
        << '\n';


    std::cout
        << "Softmax sum = "
        << cuda_sum
        << '\n';


    std::cout
        << "Max error = "
        << max_error
        << '\n';


    std::cout
        << "\nFirst 10 CUDA results:\n";


    for (int i = 0;
         i < 10;
         ++i) {

        std::cout
            << h_output[i]
            << " ";
    }


    std::cout
        << "\n\nFirst 10 CPU results:\n";


    for (int i = 0;
         i < 10;
         ++i) {

        std::cout
            << cpu_output[i]
            << " ";
    }


    std::cout
        << '\n';


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