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

        // ====================================
        // Step 1: sum(x^2)
        // ====================================

        float sum_sq = 0.0f;

        for (int c = 0; c < cols; ++c) {

            float v =
                x[r * cols + c];

            sum_sq +=
                v * v;
        }


        // ====================================
        // Step 2: RMS
        // ====================================

        float mean_sq =
            sum_sq
            /
            static_cast<float>(cols);


        float rstd =
            1.0f
            /
            std::sqrt(mean_sq + eps);


        // ====================================
        // Step 3: normalize
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
// Warp reduction
//
// 一个 warp = 32 threads
//
// 最终 lane 0 得到整个 warp 的 sum
// ============================================================

__device__ __forceinline__
float warp_reduce_sum(float val)
{
    for (
        int offset = 16;
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
// 256 threads
//
//      ↓
//
// 8 warps
//
//      ↓
//
// 每个 warp 内部 shuffle
//
//      ↓
//
// 8 个 warp sum
//
//      ↓
//
// shared memory
//
//      ↓
//
// warp 0 再 shuffle
//
//      ↓
//
// 整个 block 的 sum
// ============================================================

__device__ __forceinline__
float block_reduce_sum(
    float val,
    float* shared
) {
    int tid =
        threadIdx.x;


    // 当前 thread 在 warp 中的位置
    //
    // 0 ~ 31
    int lane =
        tid & 31;


    // 当前属于哪个 warp
    //
    // tid / 32
    int warp_id =
        tid >> 5;


    // ========================================================
    // Step 1:
    // warp 内 reduction
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


    __syncthreads();


    int num_warps =
        (
            blockDim.x + 31
        )
        /
        32;


    // ========================================================
    // Step 3:
    // warp 0 对这些 warp sum 再 reduction
    // ========================================================

    float block_sum =
        0.0f;


    if (warp_id == 0) {

        if (lane < num_warps) {

            block_sum =
                shared[lane];
        }


        block_sum =
            warp_reduce_sum(
                block_sum
            );


        if (lane == 0) {

            shared[0] =
                block_sum;
        }
    }


    __syncthreads();


    return shared[0];
}


// ============================================================
// RMSNorm V3
//
// V0:
// shared memory reduction
//
// V1:
// warp shuffle reduction
//
// V2:
// x 缓存在 register
//
// V3:
// float4 vectorized load/store
//
//
// 当前专门针对：
//
// cols  = 1024
// block = 256
//
// 1024 float
// =
// 256 float4
//
// 所以：
// 每个 thread 正好负责一个 float4
// ============================================================

__global__ void rmsnorm_v3_kernel(
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
    // 当前行
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
    // V3:
    //
    // float* -> float4*
    //
    // 每个 float4 = 4 个 float = 16 bytes
    // ========================================================

    const float4* x4 =
        reinterpret_cast<const float4*>(
            x_row
        );


    const float4* gamma4 =
        reinterpret_cast<const float4*>(
            gamma
        );


    float4* y4 =
        reinterpret_cast<float4*>(
            y_row
        );


    // ========================================================
    // Step 1:
    //
    // 一个线程一次读取一个 float4
    //
    // thread 0:
    // x[0], x[1], x[2], x[3]
    //
    // thread 1:
    // x[4], x[5], x[6], x[7]
    //
    // ...
    //
    // thread 255:
    // x[1020], x[1021], x[1022], x[1023]
    //
    //
    // v 会保存在寄存器中
    // ========================================================

    float4 v =
        x4[tid];


    // ========================================================
    // Step 2:
    //
    // 每个线程计算自己 4 个元素的 sum(x^2)
    // ========================================================

    float sum_sq =
        v.x * v.x
        +
        v.y * v.y
        +
        v.z * v.z
        +
        v.w * v.w;


    // ========================================================
    // Step 3:
    //
    // block reduction
    //
    // 得到整行所有 1024 个元素的 sum(x^2)
    // ========================================================

    float total_sum_sq =
        block_reduce_sum(
            sum_sq,
            shared
        );


    // ========================================================
    // Step 4:
    //
    // mean(x^2)
    //
    // RMS = sqrt(mean(x^2) + eps)
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
    // Step 5:
    //
    // gamma 也一次读取一个 float4
    // ========================================================

    float4 g =
        gamma4[tid];


    // ========================================================
    // Step 6:
    //
    // normalize
    //
    // 注意：
    //
    // 这里没有再次读取 x
    //
    // 直接复用寄存器里的：
    //
    // v.x
    // v.y
    // v.z
    // v.w
    // ========================================================

    float4 out;


    out.x =
        v.x
        *
        rstd
        *
        g.x;


    out.y =
        v.y
        *
        rstd
        *
        g.y;


    out.z =
        v.z
        *
        rstd
        *
        g.z;


    out.w =
        v.w
        *
        rstd
        *
        g.w;


    // ========================================================
    // Step 7:
    //
    // 一次 float4 store
    //
    // 一次写 16 bytes
    // ========================================================

    y4[tid] =
        out;
}


// ============================================================
// Launcher
// ============================================================

void launch_rmsnorm_v3(
    const float* d_x,
    const float* d_gamma,
    float* d_y,
    int rows,
    int cols,
    float eps
) {
    constexpr int block =
        256;


    // ========================================================
    // 当前 V3 是针对 cols = 1024 特化的
    //
    // 因为：
    //
    // 1024 / 4
    // =
    // 256 float4
    //
    // 正好：
    //
    // 256 threads
    // 每个线程一个 float4
    // ========================================================

    if (cols != 1024) {

        std::cerr
            << "RMSNorm V3 currently requires cols = 1024"
            << std::endl;

        std::exit(1);
    }


    int grid =
        rows;


    // 256 / 32 = 8 warps
    constexpr int num_warps =
        block / 32;


    // 只需要保存 8 个 warp sum
    //
    // 8 * 4 bytes
    // =
    // 32 bytes shared memory
    size_t shared_mem =
        num_warps
        *
        sizeof(float);


    rmsnorm_v3_kernel<<<
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
    // 模拟 Transformer
    //
    // rows:
    // batch * sequence
    //
    // cols:
    // hidden_size

    constexpr int rows =
        4096;

    constexpr int cols =
        1024;


    float eps =
        1e-5f;


    int warmup =
        10;

    int repeat =
        100;


    // ========================================================
    // Tensor size
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
    // Initialize x
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


    // ========================================================
    // Initialize gamma
    // ========================================================

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
    // Device
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

        launch_rmsnorm_v3(
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

        launch_rmsnorm_v3(
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
        << "V3 Average kernel time: "
        << avg_ms
        << " ms"
        << std::endl;


    // ========================================================
    // Effective bandwidth
    //
    // V3:
    //
    // x:
    // read 1 次
    //
    // gamma:
    // read 1 次
    //
    // y:
    // write 1 次
    //
    //
    // 所以粗略是：
    //
    // 3 * numel * sizeof(float)
    //
    //
    // 注意：
    // float4 并没有减少“数据量”
    //
    // 只是把访问方式从标量访问
    // 变成向量化访问
    // ========================================================

    double bytes_per_iter =
        3.0
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