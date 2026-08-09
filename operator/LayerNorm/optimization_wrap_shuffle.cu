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

        // -------------------------
        // mean
        // -------------------------

        float mean = 0.0f;

        for (int c = 0; c < cols; c++) {
            mean += x[r * cols + c];
        }

        mean /= static_cast<float>(cols);


        // -------------------------
        // variance
        // -------------------------

        float var = 0.0f;

        for (int c = 0; c < cols; c++) {
            float diff = x[r * cols + c] - mean;
            var += diff * diff;
        }

        var /= static_cast<float>(cols);


        // -------------------------
        // normalize
        // -------------------------

        float rstd = 1.0f / std::sqrt(var + eps);

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
//
// 一个 warp = 32 threads
//
// 输入：
// 每个 thread 有一个 val
//
// 输出：
// lane 0 得到整个 warp 的 sum
// ============================================================

__device__ __forceinline__
float warp_reduce_sum(float val)
{
    // warpSize = 32
    //
    // offset:
    // 16 -> 8 -> 4 -> 2 -> 1

    for (int offset = warpSize / 2;
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
//
// 比如 blockDim.x = 256
//
// 256 threads
//      ↓
// 8 warps
//      ↓
// 每个 warp shuffle reduction
//      ↓
// 8 个 warp sum
//      ↓
// shared memory
//      ↓
// warp 0 再 reduction
//      ↓
// block sum
// ============================================================

__device__ __forceinline__
float block_reduce_sum(
    float val,
    float* shared
) {
    int tid = threadIdx.x;

    // 当前线程在 warp 内的编号
    //
    // 0 ~ 31
    int lane = tid % warpSize;

    // 当前线程属于哪个 warp
    //
    // block = 256 时:
    //
    // warp_id = 0 ~ 7
    int warp_id = tid / warpSize;


    // ========================================================
    // Step 1
    // 每个 warp 自己 reduction
    // ========================================================

    val = warp_reduce_sum(val);


    // ========================================================
    // Step 2
    // 每个 warp 的 lane 0 保存 warp sum
    // ========================================================

    if (lane == 0) {
        shared[warp_id] = val;
    }

    __syncthreads();


    // block 中有多少个 warp
    int num_warps =
        (blockDim.x + warpSize - 1) / warpSize;


    // ========================================================
    // Step 3
    // 让第一个 warp 对所有 warp sum 再做 reduction
    // ========================================================

    float block_sum = 0.0f;

    if (warp_id == 0) {

        // 前 num_warps 个 lane
        // 读取 shared memory 中的 warp sum
        if (lane < num_warps) {
            block_sum = shared[lane];
        }

        // 其他 lane 的 block_sum = 0
        //
        // 然后整个 warp 再做一次 reduction
        block_sum = warp_reduce_sum(block_sum);


        // lane 0 得到最终 block sum
        if (lane == 0) {
            shared[0] = block_sum;
        }
    }

    __syncthreads();


    // 所有线程都可以读取最终结果
    return shared[0];
}


// ============================================================
// LayerNorm V1
//
// 优化点：
//
// naive:
//
// shared memory tree reduction
// +
// 很多 __syncthreads()
//
// V1:
//
// warp shuffle
// +
// 少量 shared memory
// +
// 少量 __syncthreads()
//
// 注意：
// 目前 x 仍然读 3 遍
// 这一版只优化 reduction
// ============================================================

__global__ void layernorm_v1_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int rows,
    int cols,
    float eps
) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;


    if (row >= rows) {
        return;
    }


    // 当前 block 负责一整行
    //
    // x_row 指向当前行开头
    const float* x_row =
        x + static_cast<size_t>(row) * cols;

    float* y_row =
        y + static_cast<size_t>(row) * cols;


    // ========================================================
    // Step 1: mean
    // ========================================================

    float sum = 0.0f;


    // cols = 1024
    // blockDim.x = 256
    //
    // thread 0:
    // 0, 256, 512, 768
    //
    // thread 1:
    // 1, 257, 513, 769
    //
    // ...
    //
    // thread 255:
    // 255, 511, 767, 1023

    for (int i = tid;
         i < cols;
         i += blockDim.x) {

        sum += x_row[i];
    }


    // 256 个线程的局部 sum
    // reduction 成一个总 sum
    float total_sum =
        block_reduce_sum(sum, shared);


    float mean =
        total_sum / static_cast<float>(cols);


    // ========================================================
    // Step 2: variance
    // ========================================================

    float var_sum = 0.0f;


    for (int i = tid;
         i < cols;
         i += blockDim.x) {

        float diff =
            x_row[i] - mean;

        var_sum +=
            diff * diff;
    }


    float total_var_sum =
        block_reduce_sum(var_sum, shared);


    float var =
        total_var_sum / static_cast<float>(cols);


    float rstd =
        rsqrtf(var + eps);


    // ========================================================
    // Step 3: normalize + affine
    // ========================================================

    for (int i = tid;
         i < cols;
         i += blockDim.x) {

        float v =
            x_row[i];

        float norm =
            (v - mean) * rstd;


        y_row[i] =
            norm * gamma[i] + beta[i];
    }
}


// ============================================================
// Kernel launcher
// ============================================================

void launch_layernorm_v1(
    const float* d_x,
    const float* d_gamma,
    const float* d_beta,
    float* d_y,
    int rows,
    int cols,
    float eps
) {
    int block = 256;

    int grid = rows;


    // block = 256
    //
    // 256 / 32 = 8 warps
    //
    // shared memory 实际只需要保存
    // 8 个 warp sum

    int num_warps =
        (block + 31) / 32;

    size_t shared_mem =
        num_warps * sizeof(float);


    layernorm_v1_kernel<<<
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


    CHECK_CUDA(cudaGetLastError());
}


// ============================================================
// main
// ============================================================

int main()
{
    // Transformer LayerNorm:
    //
    // rows = batch_size * seq_len
    // cols = hidden_size

    int rows = 4096;
    int cols = 1024;

    float eps = 1e-5f;

    int warmup = 10;
    int repeat = 100;


    size_t numel =
        static_cast<size_t>(rows) * cols;


    size_t bytes_x =
        numel * sizeof(float);

    size_t bytes_y =
        numel * sizeof(float);

    size_t bytes_gamma =
        static_cast<size_t>(cols) * sizeof(float);

    size_t bytes_beta =
        static_cast<size_t>(cols) * sizeof(float);


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

    std::vector<float> h_x(numel);

    std::vector<float> h_y(numel);

    std::vector<float> h_y_ref(numel);

    std::vector<float> h_gamma(cols);

    std::vector<float> h_beta(cols);


    // ========================================================
    // Initialize input
    // ========================================================

    for (size_t i = 0;
         i < numel;
         i++) {

        // 注意：
        //
        // i 是 size_t，无符号数
        //
        // 所以必须先转 int
        //
        // 否则:
        //
        // (i % 127) - 63
        //
        // 可能发生 unsigned underflow

        int value =
            static_cast<int>(i % 127) - 63;


        h_x[i] =
            static_cast<float>(value) * 0.01f;
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

        launch_layernorm_v1(
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

        launch_layernorm_v1(
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
        elapsed_ms /
        static_cast<float>(repeat);


    std::cout
        << "Total kernel time: "
        << elapsed_ms
        << " ms"
        << std::endl;


    std::cout
        << "Average kernel time: "
        << avg_ms
        << " ms"
        << std::endl;


    // ========================================================
    // Approx effective bandwidth
    //
    // 当前版本仍然：
    //
    // mean:       读 x
    // variance:   读 x
    // normalize:  读 x
    // gamma:      读
    // beta:       读
    // y:          写
    //
    // 粗略按 6 次 float 流量计算
    // ========================================================

    double bytes_per_iter =
        6.0
        *
        static_cast<double>(numel)
        *
        sizeof(float);


    double gb_per_iter =
        bytes_per_iter / 1e9;


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

    layernorm_cpu(
        h_x,
        h_y_ref,
        h_gamma,
        h_beta,
        rows,
        cols,
        eps
    );


    // ========================================================
    // Correctness check
    // ========================================================

    float max_err = 0.0f;


    for (size_t i = 0;
         i < numel;
         i++) {

        float err =
            std::abs(
                h_y[i] -
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


    CHECK_CUDA(
        cudaFree(d_x)
    );


    CHECK_CUDA(
        cudaFree(d_y)
    );


    CHECK_CUDA(
        cudaFree(d_gamma)
    );


    CHECK_CUDA(
        cudaFree(d_beta)
    );


    return 0;
}