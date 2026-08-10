#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

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

        // ====================================
        // Step 1: mean(x^2)
        // ====================================

        float sum_sq = 0.0f;

        for (int c = 0; c < cols; ++c) {
            float v = x[r * cols + c];
            sum_sq += v * v;
        }

        float mean_sq =
            sum_sq / static_cast<float>(cols);


        // ====================================
        // Step 2: reciprocal RMS
        // ====================================

        float rstd =
            1.0f / std::sqrt(mean_sq + eps);


        // ====================================
        // Step 3: normalize + gamma
        // ====================================

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
// RMSNorm V0
//
// 设计：
//
// 1 block = 1 row
//
// cols = 1024
// blockDim.x = 256
//
// 所以每个线程大约处理 4 个元素
//
// thread 0:
// 0, 256, 512, 768
//
// thread 1:
// 1, 257, 513, 769
//
// ...
//
// 使用 shared memory 做 naive reduction
// ============================================================

__global__ void rmsnorm_v0_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float* __restrict__ y,
    int rows,
    int cols,
    float eps
) {
    extern __shared__ float sdata[];

    int row =
        blockIdx.x;

    int tid =
        threadIdx.x;


    if (row >= rows) {
        return;
    }


    // 当前 block 负责第 row 行
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
    // 每个线程计算自己负责元素的 sum(x^2)
    // ========================================================

    float sum_sq =
        0.0f;


    for (int i = tid;
         i < cols;
         i += blockDim.x) {

        float v =
            x_row[i];

        sum_sq +=
            v * v;
    }


    // 每个线程把自己的局部 sum_sq
    // 放进 shared memory
    sdata[tid] =
        sum_sq;


    __syncthreads();


    // ========================================================
    // Step 2:
    // shared memory tree reduction
    //
    // 256
    // ↓
    // 128
    // ↓
    // 64
    // ↓
    // ...
    // ↓
    // 1
    // ========================================================

    for (
        int stride = blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {

        if (tid < stride) {

            sdata[tid] +=
                sdata[
                    tid + stride
                ];
        }


        __syncthreads();
    }


    // ========================================================
    // Step 3:
    // sdata[0] 就是整行 sum(x^2)
    // ========================================================

    float mean_sq =
        sdata[0]
        /
        static_cast<float>(cols);


    float rstd =
        rsqrtf(
            mean_sq + eps
        );


    // ========================================================
    // Step 4:
    // normalize + gamma
    //
    // RMSNorm:
    //
    // y = x / rms * gamma
    //
    // 不需要减 mean
    // ========================================================

    for (int i = tid;
         i < cols;
         i += blockDim.x) {

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

void launch_rmsnorm_v0(
    const float* d_x,
    const float* d_gamma,
    float* d_y,
    int rows,
    int cols,
    float eps
) {
    int block =
        256;

    int grid =
        rows;


    // 每个线程一个 float
    //
    // 256 * 4 bytes
    // = 1024 bytes
    size_t shared_mem =
        block
        *
        sizeof(float);


    rmsnorm_v0_kernel<<<
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
    // 模拟 Transformer:
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


    size_t numel =
        static_cast<size_t>(rows)
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
        static_cast<size_t>(cols)
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
    // Host memory
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

        launch_rmsnorm_v0(
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

        launch_rmsnorm_v0(
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
        << "V0 Average kernel time: "
        << avg_ms
        << " ms"
        << std::endl;


    // ========================================================
    // 粗略估算 effective bandwidth
    //
    // V0:
    //
    // 第一次:
    // 读 x 求 sum(x^2)       -> 1x
    //
    // 第二次:
    // 读 x 做 output         -> 1x
    //
    // 读 gamma              -> 1x
    //
    // 写 y                  -> 1x
    //
    // 总共粗略:
    //
    // 4 * numel * sizeof(float)
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
    // Copy result back
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
    // Correctness check
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