#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>


// ============================================================
// CUDA error check
// ============================================================

#define CHECK_CUDA(call)                                                    \
do {                                                                        \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)              \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
        std::exit(1);                                                       \
    }                                                                       \
} while (0)


// ============================================================
// CPU reference
// ============================================================

void rmsnorm_cpu(
    const std::vector<float>& x,
    std::vector<float>& y,
    const std::vector<float>& gamma,
    int rows,
    int cols,
    float eps
) {
    for (int r = 0; r < rows; ++r) {

        // --------------------------------
        // Step 1: sum(x^2)
        // --------------------------------

        float sum_sq = 0.0f;

        for (int c = 0; c < cols; ++c) {
            float v = x[r * cols + c];

            sum_sq += v * v;
        }


        // --------------------------------
        // Step 2: RMS
        // --------------------------------

        float mean_sq =
            sum_sq / static_cast<float>(cols);


        float rstd =
            1.0f / std::sqrt(mean_sq + eps);


        // --------------------------------
        // Step 3: normalize
        // --------------------------------

        for (int c = 0; c < cols; ++c) {

            y[r * cols + c] =
                x[r * cols + c]
                *
                rstd
                *
                gamma[c];
        }
    }
}


// ============================================================
// Warp reduction
//
// 一个 warp = 32 threads
//
// 每一个 thread 有一个 val
//
// 最终：
// lane 0 得到整个 warp 的 sum
// ============================================================

__device__ __forceinline__
float warp_reduce_sum(float val)
{
    // offset:
    //
    // 16
    // 8
    // 4
    // 2
    // 1

    for (
        int offset = warpSize / 2;
        offset > 0;
        offset >>= 1
    ) {

        val +=
            __shfl_down_sync(
                0xffffffffu,
                val,
                offset
            );
    }

    return val;
}


// ============================================================
// Block reduction
//
// blockDim.x = 256
//
// 256 threads
//      ↓
// 8 warps
//      ↓
// 每个 warp 内 shuffle
//      ↓
// 8 个 warp sum
//      ↓
// shared memory
//      ↓
// warp 0 再 shuffle
//      ↓
// block sum
// ============================================================

__device__ __forceinline__
float block_reduce_sum(
    float val,
    float* shared
) {
    int tid =
        threadIdx.x;


    // 当前 thread 在 warp 内的位置
    //
    // 0 ~ 31
    int lane =
        tid % warpSize;


    // 当前 thread 属于哪个 warp
    //
    // block = 256:
    //
    // warp_id = 0 ~ 7
    int warp_id =
        tid / warpSize;


    // ========================================================
    // Step 1:
    // 每个 warp 自己先 reduction
    // ========================================================

    val =
        warp_reduce_sum(val);


    // ========================================================
    // Step 2:
    // 每个 warp 的 lane 0 保存结果
    // ========================================================

    if (lane == 0) {

        shared[warp_id] =
            val;
    }


    // 不同 warp 之间需要同步
    __syncthreads();


    // block 中 warp 数量
    int num_warps =
        (
            blockDim.x
            +
            warpSize
            -
            1
        )
        /
        warpSize;


    // ========================================================
    // Step 3:
    // Warp 0 读取每个 warp 的结果
    // ========================================================

    float block_sum =
        0.0f;


    if (warp_id == 0) {

        // block = 256 时：
        //
        // lane 0 -> shared[0]
        // lane 1 -> shared[1]
        // ...
        // lane 7 -> shared[7]
        //
        // lane 8~31 -> 0

        if (lane < num_warps) {

            block_sum =
                shared[lane];
        }


        // Warp 0 再做一次 reduction
        block_sum =
            warp_reduce_sum(
                block_sum
            );


        // lane 0 最终拿到整个 block 的 sum
        if (lane == 0) {

            shared[0] =
                block_sum;
        }
    }


    // 确保 shared[0] 已经准备好
    __syncthreads();


    // 所有 thread 都返回一样的结果
    return shared[0];
}


// ============================================================
// RMSNorm V1
//
// V0:
// shared memory tree reduction
//
// V1:
// warp shuffle reduction
//
// 注意：
// 当前 x 仍然从 global memory 读取两遍
// ============================================================

__global__ void rmsnorm_v1_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float* __restrict__ y,
    int rows,
    int cols,
    float eps
) {
    extern __shared__ float shared[];


    int row =
        blockIdx.x;

    int tid =
        threadIdx.x;


    if (row >= rows) {
        return;
    }


    // ========================================================
    // 当前 block 负责一整行
    // ========================================================

    const float* x_row =
        x
        +
        static_cast<size_t>(row)
        *
        cols;


    float* y_row =
        y
        +
        static_cast<size_t>(row)
        *
        cols;


    // ========================================================
    // Step 1:
    //
    // 每个线程计算自己的局部 sum(x^2)
    //
    // cols = 1024
    // block = 256
    //
    // 每个线程处理 4 个元素
    // ========================================================

    float sum_sq =
        0.0f;


    for (
        int i = tid;
        i < cols;
        i += blockDim.x
    ) {

        float v =
            x_row[i];


        sum_sq +=
            v * v;
    }


    // ========================================================
    // Step 2:
    //
    // V1 核心优化
    //
    // 不再做 shared-memory tree reduction
    //
    // 改成 warp shuffle block reduction
    // ========================================================

    float total_sum_sq =
        block_reduce_sum(
            sum_sq,
            shared
        );


    // ========================================================
    // Step 3:
    // mean(x^2)
    // ========================================================

    float mean_sq =
        total_sum_sq
        /
        static_cast<float>(cols);


    float rstd =
        rsqrtf(
            mean_sq + eps
        );


    // ========================================================
    // Step 4:
    //
    // normalize + gamma
    //
    // 注意：
    // 这里又读了一次 x
    //
    // 所以 V1 中：
    //
    // x 第一次读 -> sum(x^2)
    // x 第二次读 -> output
    //
    // 下一版 V2 才解决这个问题
    // ========================================================

    for (
        int i = tid;
        i < cols;
        i += blockDim.x
    ) {

        float v =
            x_row[i];


        y_row[i] =
            v
            *
            rstd
            *
            gamma[i];
    }
}


// ============================================================
// Launcher
// ============================================================

void launch_rmsnorm_v1(
    const float* d_x,
    const float* d_gamma,
    float* d_y,
    int rows,
    int cols,
    float eps
) {
    constexpr int block =
        256;


    int grid =
        rows;


    // ========================================================
    // V0:
    //
    // shared_mem =
    // 256 * sizeof(float)
    //
    //
    // V1:
    //
    // 只需要保存每个 warp 的结果
    //
    // 256 threads / 32
    // = 8 warps
    //
    // 只需要:
    //
    // 8 * sizeof(float)
    // = 32 bytes
    // ========================================================

    int num_warps =
        (
            block
            +
            warpSize
            -
            1
        )
        /
        warpSize;


    size_t shared_mem =
        static_cast<size_t>(
            num_warps
        )
        *
        sizeof(float);


    rmsnorm_v1_kernel<<<
        grid,
        block,
        shared_mem
    >>>(
        d_x,
        d_gamma,
        d_y,
        rows,
        cols,
        eps
    );


    CHECK_CUDA(
        cudaGetLastError()
    );
}


// ============================================================
// main
// ============================================================

int main()
{
    // Transformer:
    //
    // rows = batch_size * seq_len
    // cols = hidden_size

    int rows =
        4096;

    int cols =
        1024;


    float eps =
        1e-5f;


    int warmup =
        10;

    int repeat =
        100;


    // ========================================================
    // Size
    // ========================================================

    size_t numel =
        static_cast<size_t>(
            rows
        )
        *
        cols;


    size_t bytes_x =
        numel
        *
        sizeof(float);


    size_t bytes_y =
        numel
        *
        sizeof(float);


    size_t bytes_gamma =
        static_cast<size_t>(
            cols
        )
        *
        sizeof(float);


    std::cout
        << "rows = "
        << rows
        << ", cols = "
        << cols
        << std::endl;


    std::cout
        << "warmup = "
        << warmup
        << ", repeat = "
        << repeat
        << std::endl;


    // ========================================================
    // Host
    // ========================================================

    std::vector<float> h_x(
        numel
    );


    std::vector<float> h_y(
        numel
    );


    std::vector<float> h_y_ref(
        numel
    );


    std::vector<float> h_gamma(
        cols
    );


    // ========================================================
    // Initialize input
    // ========================================================

    for (
        size_t i = 0;
        i < numel;
        ++i
    ) {

        int value =
            static_cast<int>(
                i % 127
            )
            -
            63;


        h_x[i] =
            static_cast<float>(
                value
            )
            *
            0.01f;
    }


    for (
        int i = 0;
        i < cols;
        ++i
    ) {

        h_gamma[i] =
            1.0f
            +
            0.001f
            *
            static_cast<float>(
                i % 13
            );
    }


    // ========================================================
    // Device memory
    // ========================================================

    float* d_x =
        nullptr;

    float* d_y =
        nullptr;

    float* d_gamma =
        nullptr;


    CHECK_CUDA(
        cudaMalloc(
            &d_x,
            bytes_x
        )
    );


    CHECK_CUDA(
        cudaMalloc(
            &d_y,
            bytes_y
        )
    );


    CHECK_CUDA(
        cudaMalloc(
            &d_gamma,
            bytes_gamma
        )
    );


    // ========================================================
    // Host -> Device
    // ========================================================

    CHECK_CUDA(
        cudaMemcpy(
            d_x,
            h_x.data(),
            bytes_x,
            cudaMemcpyHostToDevice
        )
    );


    CHECK_CUDA(
        cudaMemcpy(
            d_gamma,
            h_gamma.data(),
            bytes_gamma,
            cudaMemcpyHostToDevice
        )
    );


    // ========================================================
    // Warmup
    // ========================================================

    for (
        int i = 0;
        i < warmup;
        ++i
    ) {

        launch_rmsnorm_v1(
            d_x,
            d_gamma,
            d_y,
            rows,
            cols,
            eps
        );
    }


    CHECK_CUDA(
        cudaDeviceSynchronize()
    );


    // ========================================================
    // Benchmark
    // ========================================================

    cudaEvent_t start;
    cudaEvent_t stop;


    CHECK_CUDA(
        cudaEventCreate(
            &start
        )
    );


    CHECK_CUDA(
        cudaEventCreate(
            &stop
        )
    );


    CHECK_CUDA(
        cudaEventRecord(
            start
        )
    );


    for (
        int i = 0;
        i < repeat;
        ++i
    ) {

        launch_rmsnorm_v1(
            d_x,
            d_gamma,
            d_y,
            rows,
            cols,
            eps
        );
    }


    CHECK_CUDA(
        cudaEventRecord(
            stop
        )
    );


    CHECK_CUDA(
        cudaEventSynchronize(
            stop
        )
    );


    float elapsed_ms =
        0.0f;


    CHECK_CUDA(
        cudaEventElapsedTime(
            &elapsed_ms,
            start,
            stop
        )
    );


    float avg_ms =
        elapsed_ms
        /
        static_cast<float>(
            repeat
        );


    std::cout
        << "Total kernel time: "
        << elapsed_ms
        << " ms"
        << std::endl;


    std::cout
        << "V1 Average kernel time: "
        << avg_ms
        << " ms"
        << std::endl;


    // ========================================================
    // Effective bandwidth
    //
    // 当前 V1：
    //
    // x 第一次 read    -> 1
    // x 第二次 read    -> 1
    // gamma read       -> 1
    // y write          -> 1
    //
    // 总计粗略：
    //
    // 4 * numel * sizeof(float)
    //
    // 注意：
    // V1 和 V0 的 global memory 流量基本没有改变
    //
    // V1 优化的是 reduction
    // ========================================================

    double bytes_per_iter =
        4.0
        *
        static_cast<double>(
            numel
        )
        *
        sizeof(float);


    double gb_per_iter =
        bytes_per_iter
        /
        1e9;


    double bandwidth_gb_s =
        gb_per_iter
        /
        (
            avg_ms
            /
            1000.0
        );


    std::cout
        << "Approx effective bandwidth: "
        << bandwidth_gb_s
        << " GB/s"
        << std::endl;


    // ========================================================
    // Device -> Host
    // ========================================================

    CHECK_CUDA(
        cudaMemcpy(
            h_y.data(),
            d_y,
            bytes_y,
            cudaMemcpyDeviceToHost
        )
    );


    // ========================================================
    // CPU reference
    // ========================================================

    rmsnorm_cpu(
        h_x,
        h_y_ref,
        h_gamma,
        rows,
        cols,
        eps
    );


    // ========================================================
    // Correctness
    // ========================================================

    float max_err =
        0.0f;


    for (
        size_t i = 0;
        i < numel;
        ++i
    ) {

        float err =
            std::abs(
                h_y[i]
                -
                h_y_ref[i]
            );


        max_err =
            std::max(
                max_err,
                err
            );
    }


    std::cout
        << "Max error: "
        << max_err
        << std::endl;


    // ========================================================
    // Cleanup
    // ========================================================

    CHECK_CUDA(
        cudaEventDestroy(
            start
        )
    );


    CHECK_CUDA(
        cudaEventDestroy(
            stop
        )
    );


    CHECK_CUDA(
        cudaFree(
            d_x
        )
    );


    CHECK_CUDA(
        cudaFree(
            d_y
        )
    );


    CHECK_CUDA(
        cudaFree(
            d_gamma
        )
    );


    return 0;
}