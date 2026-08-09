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
    for (int r = 0; r < rows; ++r) {

        float mean = 0.0f;

        for (int c = 0; c < cols; ++c) {
            mean += x[r * cols + c];
        }

        mean /= static_cast<float>(cols);


        float var = 0.0f;

        for (int c = 0; c < cols; ++c) {

            float diff =
                x[r * cols + c] - mean;

            var += diff * diff;
        }

        var /= static_cast<float>(cols);


        float rstd =
            1.0f / std::sqrt(var + eps);


        for (int c = 0; c < cols; ++c) {

            float norm =
                (x[r * cols + c] - mean)
                *
                rstd;

            y[r * cols + c] =
                norm * gamma[c]
                +
                beta[c];
        }
    }
}


// ============================================================
// Welford
// ============================================================

struct WelfordData {
    float mean;
    float m2;
    int count;
};


// 加入一个元素
__device__ __forceinline__
WelfordData welford_update(
    WelfordData a,
    float x
) {
    a.count += 1;

    float delta =
        x - a.mean;

    a.mean +=
        delta /
        static_cast<float>(a.count);

    float delta2 =
        x - a.mean;

    a.m2 +=
        delta * delta2;

    return a;
}


// 合并两个 Welford 状态
__device__ __forceinline__
WelfordData welford_combine(
    WelfordData a,
    WelfordData b
) {
    if (a.count == 0) {
        return b;
    }

    if (b.count == 0) {
        return a;
    }


    float count_a =
        static_cast<float>(a.count);

    float count_b =
        static_cast<float>(b.count);

    float total_count =
        count_a + count_b;


    float delta =
        b.mean - a.mean;


    WelfordData out;

    out.count =
        a.count + b.count;


    out.mean =
        a.mean
        +
        delta
        *
        (count_b / total_count);


    out.m2 =
        a.m2
        +
        b.m2
        +
        delta
        *
        delta
        *
        (
            count_a
            *
            count_b
            /
            total_count
        );


    return out;
}


// ============================================================
// Warp Welford Reduction
// ============================================================

__device__ __forceinline__
WelfordData warp_reduce_welford(
    WelfordData val
) {
    constexpr unsigned mask =
        0xffffffffu;

    int lane =
        threadIdx.x & 31;


    for (
        int offset = 16;
        offset > 0;
        offset >>= 1
    ) {

        WelfordData other;


        other.mean =
            __shfl_down_sync(
                mask,
                val.mean,
                offset
            );


        other.m2 =
            __shfl_down_sync(
                mask,
                val.m2,
                offset
            );


        other.count =
            __shfl_down_sync(
                mask,
                val.count,
                offset
            );


        if (lane + offset < 32) {

            val =
                welford_combine(
                    val,
                    other
                );
        }
    }


    return val;
}


// ============================================================
// Block Welford Reduction
// ============================================================

__device__ __forceinline__
WelfordData block_reduce_welford(
    WelfordData val,
    WelfordData* shared
) {
    int tid =
        threadIdx.x;

    int lane =
        tid & 31;

    int warp_id =
        tid >> 5;


    // -------------------------
    // warp 内 reduction
    // -------------------------

    val =
        warp_reduce_welford(val);


    // 每个 warp 的 lane 0
    // 保存一个结果
    if (lane == 0) {
        shared[warp_id] = val;
    }


    __syncthreads();


    int num_warps =
        (blockDim.x + 31) / 32;


    // -------------------------
    // warp 0 再 reduction
    // -------------------------

    if (warp_id == 0) {

        WelfordData block_val;

        block_val.mean = 0.0f;
        block_val.m2 = 0.0f;
        block_val.count = 0;


        if (lane < num_warps) {
            block_val =
                shared[lane];
        }


        block_val =
            warp_reduce_welford(
                block_val
            );


        if (lane == 0) {
            shared[0] =
                block_val;
        }
    }


    __syncthreads();


    return shared[0];
}


// ============================================================
// LayerNorm V4
//
// V1:
// warp shuffle reduction
//
// V2:
// x 只读一次，缓存在 register
//
// V3:
// Welford 一次 reduction 得到 mean + variance
//
// V4:
// float4 vectorized load/store
//
// 当前 specialized:
// cols = 1024
// block = 256
//
// 1024 / 4 = 256 float4
//
// 所以刚好：
// 一个 thread 负责一个 float4
// ============================================================

__global__ void layernorm_v4_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int rows,
    int cols,
    float eps
) {
    extern __shared__ WelfordData shared[];


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
    // V4 核心
    //
    // 把 float* 转成 float4*
    // ========================================================

    const float4* x4 =
        reinterpret_cast<const float4*>(
            x_row
        );


    const float4* gamma4 =
        reinterpret_cast<const float4*>(
            gamma
        );


    const float4* beta4 =
        reinterpret_cast<const float4*>(
            beta
        );


    float4* y4 =
        reinterpret_cast<float4*>(
            y_row
        );


    // ========================================================
    // Step 1
    //
    // 一次 16-byte load
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
    // x[1020] ... x[1023]
    // ========================================================

    float4 v =
        x4[tid];


    // v.x
    // v.y
    // v.z
    // v.w
    //
    // 后面全部复用这些寄存器


    // ========================================================
    // Step 2
    // local Welford
    // ========================================================

    WelfordData local;

    local.mean = 0.0f;
    local.m2 = 0.0f;
    local.count = 0;


    local =
        welford_update(
            local,
            v.x
        );


    local =
        welford_update(
            local,
            v.y
        );


    local =
        welford_update(
            local,
            v.z
        );


    local =
        welford_update(
            local,
            v.w
        );


    // ========================================================
    // Step 3
    // 一个 block 一次 Welford reduction
    // ========================================================

    WelfordData stats =
        block_reduce_welford(
            local,
            shared
        );


    float mean =
        stats.mean;


    float var =
        stats.m2
        /
        static_cast<float>(
            stats.count
        );


    float rstd =
        rsqrtf(
            var + eps
        );


    // ========================================================
    // Step 4
    //
    // gamma/beta 也使用 float4
    // ========================================================

    float4 g =
        gamma4[tid];


    float4 b =
        beta4[tid];


    // ========================================================
    // Step 5
    // normalize + affine
    // ========================================================

    float4 out;


    out.x =
        (v.x - mean)
        *
        rstd
        *
        g.x
        +
        b.x;


    out.y =
        (v.y - mean)
        *
        rstd
        *
        g.y
        +
        b.y;


    out.z =
        (v.z - mean)
        *
        rstd
        *
        g.z
        +
        b.z;


    out.w =
        (v.w - mean)
        *
        rstd
        *
        g.w
        +
        b.w;


    // ========================================================
    // 一次 16-byte store
    // ========================================================

    y4[tid] =
        out;
}


// ============================================================
// Launcher
// ============================================================

void launch_layernorm_v4(
    const float* d_x,
    const float* d_gamma,
    const float* d_beta,
    float* d_y,
    int rows,
    int cols,
    float eps
) {
    constexpr int block =
        256;


    // 当前版本专门针对：
    //
    // cols = 1024
    //
    // 1024 floats
    // =
    // 256 float4
    //
    // =
    // 每线程正好一个 float4
    if (cols != 1024) {

        std::cerr
            << "V4 currently requires cols = 1024"
            << std::endl;

        std::exit(1);
    }


    int grid =
        rows;


    int num_warps =
        block / 32;


    size_t shared_mem =
        num_warps
        *
        sizeof(WelfordData);


    layernorm_v4_kernel<<<
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


    size_t numel =
        static_cast<size_t>(rows)
        *
        cols;


    size_t bytes =
        numel
        *
        sizeof(float);


    size_t param_bytes =
        static_cast<size_t>(cols)
        *
        sizeof(float);


    // ========================================================
    // Host
    // ========================================================

    std::vector<float> h_x(numel);

    std::vector<float> h_y(numel);

    std::vector<float> h_ref(numel);

    std::vector<float> h_gamma(cols);

    std::vector<float> h_beta(cols);


    // ========================================================
    // Init
    // ========================================================

    for (size_t i = 0;
         i < numel;
         ++i) {

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
         ++i) {

        h_gamma[i] =
            1.0f
            +
            0.001f
            *
            static_cast<float>(
                i % 13
            );


        h_beta[i] =
            0.01f
            *
            static_cast<float>(
                i % 7
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

    float* d_beta =
        nullptr;


    CHECK_CUDA(
        cudaMalloc(
            &d_x,
            bytes
        )
    );


    CHECK_CUDA(
        cudaMalloc(
            &d_y,
            bytes
        )
    );


    CHECK_CUDA(
        cudaMalloc(
            &d_gamma,
            param_bytes
        )
    );


    CHECK_CUDA(
        cudaMalloc(
            &d_beta,
            param_bytes
        )
    );


    // ========================================================
    // H2D
    // ========================================================

    CHECK_CUDA(
        cudaMemcpy(
            d_x,
            h_x.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );


    CHECK_CUDA(
        cudaMemcpy(
            d_gamma,
            h_gamma.data(),
            param_bytes,
            cudaMemcpyHostToDevice
        )
    );


    CHECK_CUDA(
        cudaMemcpy(
            d_beta,
            h_beta.data(),
            param_bytes,
            cudaMemcpyHostToDevice
        )
    );


    // ========================================================
    // Warmup
    // ========================================================

    for (int i = 0;
         i < warmup;
         ++i) {

        launch_layernorm_v4(
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


    for (int i = 0;
         i < repeat;
         ++i) {

        launch_layernorm_v4(
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
        << "V4 average kernel time: "
        << avg_ms
        << " ms"
        << std::endl;


    // ========================================================
    // Effective bandwidth
    //
    // x      read 1
    // gamma  read 1
    // beta   read 1
    // y      write 1
    //
    // 总字节数和 V3 基本一样
    //
    // float4 优化的是访问方式，
    // 不是减少总数据量
    // ========================================================

    double bytes_per_iter =
        4.0
        *
        static_cast<double>(numel)
        *
        sizeof(float);


    double bandwidth =
        (
            bytes_per_iter / 1e9
        )
        /
        (
            avg_ms / 1000.0
        );


    std::cout
        << "Approx effective bandwidth: "
        << bandwidth
        << " GB/s"
        << std::endl;


    // ========================================================
    // Correctness
    // ========================================================

    CHECK_CUDA(
        cudaMemcpy(
            h_y.data(),
            d_y,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );


    layernorm_cpu(
        h_x,
        h_ref,
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
         ++i) {

        float err =
            std::abs(
                h_y[i]
                -
                h_ref[i]
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
    // cleanup
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


    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_gamma));
    CHECK_CUDA(cudaFree(d_beta));


    return 0;
}