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

void layernorm_cpu(
    const std::vector<float>& x,
    std::vector<float>& y,
    const std::vector<float>& gamma,
    const std::vector<float>& beta,
    int rows,
    int cols,
    float eps
) {
    for (int r = 0; r < rows; r++) {

        float mean = 0.0f;

        for (int c = 0; c < cols; c++) {
            mean += x[r * cols + c];
        }

        mean /= static_cast<float>(cols);


        float var = 0.0f;

        for (int c = 0; c < cols; c++) {
            float diff = x[r * cols + c] - mean;
            var += diff * diff;
        }

        var /= static_cast<float>(cols);


        float rstd =
            1.0f / std::sqrt(var + eps);


        for (int c = 0; c < cols; c++) {

            float norm =
                (x[r * cols + c] - mean) * rstd;

            y[r * cols + c] =
                norm * gamma[c] + beta[c];
        }
    }
}


// ============================================================
// Warp reduction
// ============================================================

__device__ __forceinline__
float warp_reduce_sum(float val)
{
    for (int offset = 16;
         offset > 0;
         offset >>= 1) {

        val += __shfl_down_sync(
            0xffffffff,
            val,
            offset
        );
    }

    return val;
}


// ============================================================
// Block reduction
// ============================================================

__device__ __forceinline__
float block_reduce_sum(
    float val,
    float* shared
) {
    int tid = threadIdx.x;

    int lane =
        tid % warpSize;

    int warp_id =
        tid / warpSize;


    // 每个 warp 自己求和
    val =
        warp_reduce_sum(val);


    // 每个 warp 的 lane 0 保存结果
    if (lane == 0) {
        shared[warp_id] = val;
    }

    __syncthreads();


    int num_warps =
        blockDim.x / warpSize;


    float block_sum = 0.0f;


    // 第一个 warp 再把 8 个 warp sum 加起来
    if (warp_id == 0) {

        if (lane < num_warps) {
            block_sum =
                shared[lane];
        }

        block_sum =
            warp_reduce_sum(block_sum);

        if (lane == 0) {
            shared[0] =
                block_sum;
        }
    }


    __syncthreads();


    return shared[0];
}


// ============================================================
// LayerNorm V2
//
// 条件：
// cols = 1024
// block = 256
//
// 每个线程负责 4 个 float
// ============================================================

__global__ void layernorm_v2_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
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


    const float* x_row =
        x + static_cast<size_t>(row) * cols;

    float* y_row =
        y + static_cast<size_t>(row) * cols;


    // ========================================================
    // V2:
    // global memory -> register
    //
    // 每个 x 只从 global memory 读取一次
    // ========================================================

    int i0 = tid;
    int i1 = tid + blockDim.x;
    int i2 = tid + blockDim.x * 2;
    int i3 = tid + blockDim.x * 3;


    float v0 = x_row[i0];
    float v1 = x_row[i1];
    float v2 = x_row[i2];
    float v3 = x_row[i3];


    // ========================================================
    // Step 1: mean
    // ========================================================

    float sum =
        v0 + v1 + v2 + v3;


    float total_sum =
        block_reduce_sum(
            sum,
            shared
        );


    float mean =
        total_sum /
        static_cast<float>(cols);


    // ========================================================
    // Step 2: variance
    //
    // 不再访问 global memory x
    // ========================================================

    float d0 =
        v0 - mean;

    float d1 =
        v1 - mean;

    float d2 =
        v2 - mean;

    float d3 =
        v3 - mean;


    float var_sum =
        d0 * d0 +
        d1 * d1 +
        d2 * d2 +
        d3 * d3;


    float total_var_sum =
        block_reduce_sum(
            var_sum,
            shared
        );


    float var =
        total_var_sum /
        static_cast<float>(cols);


    float rstd =
        rsqrtf(var + eps);


    // ========================================================
    // Step 3: normalize + affine
    //
    // 继续复用寄存器中的 v0~v3
    // ========================================================

    y_row[i0] =
        (v0 - mean)
        *
        rstd
        *
        gamma[i0]
        +
        beta[i0];


    y_row[i1] =
        (v1 - mean)
        *
        rstd
        *
        gamma[i1]
        +
        beta[i1];


    y_row[i2] =
        (v2 - mean)
        *
        rstd
        *
        gamma[i2]
        +
        beta[i2];


    y_row[i3] =
        (v3 - mean)
        *
        rstd
        *
        gamma[i3]
        +
        beta[i3];
}


// ============================================================
// Launcher
// ============================================================

void launch_layernorm_v2(
    const float* d_x,
    const float* d_gamma,
    const float* d_beta,
    float* d_y,
    int rows,
    int cols,
    float eps
) {
    constexpr int block = 256;

    int grid =
        rows;


    // 当前这个 V2 是针对 cols=1024 写的
    if (cols != 1024) {
        std::cerr
            << "V2 currently requires cols = 1024"
            << std::endl;

        std::exit(1);
    }


    int num_warps =
        block / 32;


    size_t shared_mem =
        num_warps
        *
        sizeof(float);


    layernorm_v2_kernel<<<
        grid,
        block,
        shared_mem
    >>>(
        d_x,
        d_gamma,
        d_beta,
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
    int rows = 4096;
    int cols = 1024;

    float eps = 1e-5f;

    int warmup = 10;
    int repeat = 100;


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
        cols
        *
        sizeof(float);

    size_t bytes_beta =
        cols
        *
        sizeof(float);


    std::vector<float> h_x(numel);
    std::vector<float> h_y(numel);
    std::vector<float> h_y_ref(numel);

    std::vector<float> h_gamma(cols);
    std::vector<float> h_beta(cols);


    // ========================================================
    // Initialize
    // ========================================================

    for (size_t i = 0;
         i < numel;
         i++) {

        int value =
            static_cast<int>(i % 127)
            -
            63;

        h_x[i] =
            static_cast<float>(value)
            *
            0.01f;
    }


    for (int i = 0;
         i < cols;
         i++) {

        h_gamma[i] =
            1.0f
            +
            0.001f
            *
            static_cast<float>(i % 13);


        h_beta[i] =
            0.01f
            *
            static_cast<float>(i % 7);
    }


    // ========================================================
    // Device memory
    // ========================================================

    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_gamma = nullptr;
    float* d_beta = nullptr;


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

    CHECK_CUDA(
        cudaMalloc(
            &d_beta,
            bytes_beta
        )
    );


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


    CHECK_CUDA(
        cudaMemcpy(
            d_beta,
            h_beta.data(),
            bytes_beta,
            cudaMemcpyHostToDevice
        )
    );


    // ========================================================
    // Warmup
    // ========================================================

    for (int i = 0;
         i < warmup;
         i++) {

        launch_layernorm_v2(
            d_x,
            d_gamma,
            d_beta,
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
        cudaEventCreate(&start)
    );

    CHECK_CUDA(
        cudaEventCreate(&stop)
    );


    CHECK_CUDA(
        cudaEventRecord(start)
    );


    for (int i = 0;
         i < repeat;
         i++) {

        launch_layernorm_v2(
            d_x,
            d_gamma,
            d_beta,
            d_y,
            rows,
            cols,
            eps
        );
    }


    CHECK_CUDA(
        cudaEventRecord(stop)
    );


    CHECK_CUDA(
        cudaEventSynchronize(stop)
    );


    float elapsed_ms = 0.0f;


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
        repeat;


    std::cout
        << "V2 Average kernel time: "
        << avg_ms
        << " ms"
        << std::endl;


    // ========================================================
    // V2 bandwidth estimate
    //
    // x:      1 read
    // gamma:  1 read
    // beta:   1 read
    // y:      1 write
    //
    // 粗略变成 4 * numel * sizeof(float)
    //
    // 注意依然只是 effective bandwidth 的粗略估算
    // ========================================================

    double bytes_per_iter =
        4.0
        *
        static_cast<double>(numel)
        *
        sizeof(float);


    double gb_per_iter =
        bytes_per_iter
        /
        1e9;


    double bandwidth_gb_s =
        gb_per_iter
        /
        (avg_ms / 1000.0);


    std::cout
        << "Approx effective bandwidth: "
        << bandwidth_gb_s
        << " GB/s"
        << std::endl;


    // ========================================================
    // Correctness
    // ========================================================

    CHECK_CUDA(
        cudaMemcpy(
            h_y.data(),
            d_y,
            bytes_y,
            cudaMemcpyDeviceToHost
        )
    );


    layernorm_cpu(
        h_x,
        h_y_ref,
        h_gamma,
        h_beta,
        rows,
        cols,
        eps
    );


    float max_err =
        0.0f;


    for (size_t i = 0;
         i < numel;
         i++) {

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
        cudaEventDestroy(start)
    );

    CHECK_CUDA(
        cudaEventDestroy(stop)
    );


    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_gamma));
    CHECK_CUDA(cudaFree(d_beta));


    return 0;
}