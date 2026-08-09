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
            float diff = x[r * cols + c] - mean;
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
                norm * gamma[c] + beta[c];
        }
    }
}


// ============================================================
// Welford state
// ============================================================

struct WelfordData
{
    float mean;
    float m2;
    int count;
};


// ============================================================
// 给 Welford state 加入一个新元素
// ============================================================

__device__ __forceinline__
WelfordData welford_update(
    WelfordData a,
    float x
) {
    a.count += 1;

    float delta =
        x - a.mean;

    a.mean +=
        delta / static_cast<float>(a.count);

    float delta2 =
        x - a.mean;

    a.m2 +=
        delta * delta2;

    return a;
}


// ============================================================
// 合并两个 Welford state
//
// A:
// count_a, mean_a, m2_a
//
// B:
// count_b, mean_b, m2_b
//
//            ↓
//
// 合并后的 A+B
// ============================================================

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

    float count =
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
        (count_b / count);


    out.m2 =
        a.m2
        +
        b.m2
        +
        delta
        *
        delta
        *
        (count_a * count_b / count);


    return out;
}


// ============================================================
// Warp Welford reduction
//
// 32 threads
//      ↓
// 一个 WelfordData
// ============================================================

__device__ __forceinline__
WelfordData warp_reduce_welford(
    WelfordData val
) {
    unsigned mask = 0xffffffffu;

    int lane =
        threadIdx.x % warpSize;


    for (int offset = warpSize / 2;
         offset > 0;
         offset >>= 1) {

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


        // 避免读取 warp 外不存在的数据
        if (lane + offset < warpSize) {
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
// Block Welford reduction
//
// 256 threads
//
// ↓
//
// 8 warp results
//
// ↓ shared memory
//
// warp 0 再 reduction
//
// ↓
//
// 整行统计结果
// ============================================================

__device__ __forceinline__
WelfordData block_reduce_welford(
    WelfordData val,
    WelfordData* shared
) {
    int tid =
        threadIdx.x;

    int lane =
        tid % warpSize;

    int warp_id =
        tid / warpSize;


    // --------------------------------------------------------
    // 每个 warp 内部 reduction
    // --------------------------------------------------------

    val =
        warp_reduce_welford(val);


    // --------------------------------------------------------
    // 每个 warp 的 lane 0 保存结果
    // --------------------------------------------------------

    if (lane == 0) {
        shared[warp_id] =
            val;
    }


    __syncthreads();


    int num_warps =
        (blockDim.x + warpSize - 1)
        /
        warpSize;


    // --------------------------------------------------------
    // Warp 0 对 8 个 warp result 再做 reduction
    // --------------------------------------------------------

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
// LayerNorm V3
//
// V1:
// warp shuffle
//
// V2:
// x 只读一次并放寄存器
//
// V3:
// Welford 一次 reduction 得到 mean + variance
//
// 当前针对：
// cols = 1024
// block = 256
// ============================================================

__global__ void layernorm_v3_kernel(
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
    // Step 1
    //
    // global memory -> registers
    //
    // 每个 x 只读取一次
    // ========================================================

    int i0 =
        tid;

    int i1 =
        tid + blockDim.x;

    int i2 =
        tid + blockDim.x * 2;

    int i3 =
        tid + blockDim.x * 3;


    float v0 =
        x_row[i0];

    float v1 =
        x_row[i1];

    float v2 =
        x_row[i2];

    float v3 =
        x_row[i3];


    // ========================================================
    // Step 2
    //
    // 每个线程自己对 4 个数据做 Welford
    // ========================================================

    WelfordData local;

    local.mean = 0.0f;
    local.m2 = 0.0f;
    local.count = 0;


    local =
        welford_update(
            local,
            v0
        );

    local =
        welford_update(
            local,
            v1
        );

    local =
        welford_update(
            local,
            v2
        );

    local =
        welford_update(
            local,
            v3
        );


    // ========================================================
    // Step 3
    //
    // 整个 block 一次 Welford reduction
    //
    // 同时得到 mean 和 variance 信息
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
    // normalize + affine
    //
    // 继续复用寄存器里的 v0~v3
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

void launch_layernorm_v3(
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


    if (cols != 1024) {

        std::cerr
            << "V3 currently requires cols = 1024"
            << std::endl;

        std::exit(1);
    }


    int grid =
        rows;


    int num_warps =
        block / warpSize;


    // 每个 warp 保存一个 WelfordData
    size_t shared_mem =
        num_warps
        *
        sizeof(WelfordData);


    layernorm_v3_kernel<<<
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

        launch_layernorm_v3(
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

        launch_layernorm_v3(
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
        << "V3 average kernel time: "
        << avg_ms
        << " ms"
        << std::endl;


    // ========================================================
    // Copy result
    // ========================================================

    CHECK_CUDA(
        cudaMemcpy(
            h_y.data(),
            d_y,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );


    // ========================================================
    // CPU reference
    // ========================================================

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


    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_gamma));
    CHECK_CUDA(cudaFree(d_beta));


    return 0;
}