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
            std::cerr                                        \
                << "CUDA Error: "                             \
                << cudaGetErrorString(err)                    \
                << " at "                                     \
                << __FILE__                                   \
                << ":"                                        \
                << __LINE__                                   \
                << std::endl;                                 \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)


// ============================================================
// QK^T Tile Size
// ============================================================

constexpr int QK_TILE = 16;


// ============================================================
// Softmax
//
// 一个 block 处理一整行
// ============================================================

constexpr int SOFTMAX_THREADS = 256;


// ============================================================
// Warp Reduction: Max
// ============================================================

__device__ __forceinline__
float warpReduceMax(float value)
{
    for (
        int offset = 16;
        offset > 0;
        offset >>= 1
    ) {
        value =
            fmaxf(
                value,
                __shfl_down_sync(
                    0xffffffff,
                    value,
                    offset
                )
            );
    }

    return value;
}


// ============================================================
// Warp Reduction: Sum
// ============================================================

__device__ __forceinline__
float warpReduceSum(float value)
{
    for (
        int offset = 16;
        offset > 0;
        offset >>= 1
    ) {
        value +=
            __shfl_down_sync(
                0xffffffff,
                value,
                offset
            );
    }

    return value;
}


// ============================================================
// Kernel 1
//
// S = Q * K^T / sqrt(D)
//
// V0:
// 每个 thread 直接从 Global Memory
// 读取整个 Q row 和 K row
//
// V1:
// 把 D 方向切成 TILE
//
// Global
//   ↓
// Shared
//   ↓
// FMA
//
// ============================================================

__global__
void qkTiledKernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ S,
    int N,
    int D)
{
    // ========================================================
    // 当前 thread 计算:
    //
    // S[row][col]
    // ========================================================

    const int tx =
        threadIdx.x;

    const int ty =
        threadIdx.y;


    const int row =
        blockIdx.y * QK_TILE
        +
        ty;


    const int col =
        blockIdx.x * QK_TILE
        +
        tx;


    // ========================================================
    // Shared Q Tile
    //
    // QTile:
    //
    // [16 rows][16 D]
    // ========================================================

    __shared__
    float QTile[QK_TILE][QK_TILE];


    // ========================================================
    // K 在 Shared Memory 中转置存
    //
    // Global K:
    //
    //      [K row][D]
    //
    // Shared:
    //
    //      [D][K row]
    //
    //
    // +1 padding:
    //
    //      16 -> 17
    //
    // 减少 shared bank conflict
    // ========================================================

    __shared__
    float KTileT[QK_TILE][QK_TILE + 1];


    float sum =
        0.0f;


    // ========================================================
    // D direction tiling
    //
    // 假设:
    //
    // D = 64
    //
    // 那么:
    //
    // tile0 = D 0~15
    // tile1 = D 16~31
    // tile2 = D 32~47
    // tile3 = D 48~63
    // ========================================================

    for (
        int d0 = 0;
        d0 < D;
        d0 += QK_TILE
    ) {

        // ====================================================
        // Global Q -> Shared Q
        // ====================================================

        const int q_d =
            d0 + tx;


        if (
            row < N
            &&
            q_d < D
        ) {
            QTile[ty][tx] =
                Q[
                    row * D
                    +
                    q_d
                ];
        }
        else {
            QTile[ty][tx] =
                0.0f;
        }


        // ====================================================
        // Global K -> Shared K
        //
        // 注意:
        //
        // KTileT[tx][ty]
        //
        // 是转置写入
        // ====================================================

        const int k_row =
            blockIdx.x * QK_TILE
            +
            ty;


        const int k_d =
            d0 + tx;


        if (
            k_row < N
            &&
            k_d < D
        ) {

            KTileT[tx][ty] =
                K[
                    k_row * D
                    +
                    k_d
                ];
        }
        else {

            KTileT[tx][ty] =
                0.0f;
        }


        // ====================================================
        // 等整个 block 把 Q/K tile 搬完
        // ====================================================

        __syncthreads();


        // ====================================================
        // Compute
        //
        // 当前 thread:
        //
        // Q[row, d0:d0+16]
        //
        // dot
        //
        // K[col, d0:d0+16]
        // ====================================================

#pragma unroll
        for (
            int k = 0;
            k < QK_TILE;
            ++k
        ) {

            const float q =
                QTile[
                    ty
                ][
                    k
                ];


            const float kval =
                KTileT[
                    k
                ][
                    tx
                ];


            sum +=
                q * kval;
        }


        // ====================================================
        // 下一轮要覆盖 Shared Memory
        // ====================================================

        __syncthreads();
    }


    // ========================================================
    // Scale
    //
    // S = QK^T / sqrt(D)
    // ========================================================

    if (
        row < N
        &&
        col < N
    ) {

        const float scale =
            rsqrtf(
                static_cast<float>(D)
            );


        S[
            row * N
            +
            col
        ] =
            sum * scale;
    }
}


// ============================================================
// Kernel 2
//
// Block Softmax
//
// 一个 Block
//
//      ↓
//
// 处理 S 的一整行
//
// 256 threads 合作
//
// ============================================================

__global__
void softmaxBlockKernel(
    const float* __restrict__ S,
    float* __restrict__ P,
    int N)
{
    // ========================================================
    // 一个 block 对应一行
    // ========================================================

    const int row =
        blockIdx.x;


    const int tid =
        threadIdx.x;


    const int lane_id =
        tid % 32;


    const int warp_id =
        tid / 32;


    const int num_warps =
        blockDim.x / 32;


    // ========================================================
    // 最多支持 32 warps
    // ========================================================

    __shared__
    float warp_values[32];


    __shared__
    float block_max;


    __shared__
    float block_sum;


    // ========================================================
    // STEP 1
    //
    // 每个 thread 找自己的 local max
    //
    // 比如 N=1024
    //
    // thread0:
    //
    // col 0
    // col 256
    // col 512
    // col 768
    //
    // ========================================================

    float local_max =
        -INFINITY;


    for (
        int col = tid;
        col < N;
        col += blockDim.x
    ) {

        local_max =
            fmaxf(
                local_max,
                S[
                    row * N
                    +
                    col
                ]
            );
    }


    // ========================================================
    // STEP 2
    //
    // Warp 内 reduction
    //
    // 32 threads
    //
    // ↓
    //
    // 一个 warp max
    // ========================================================

    local_max =
        warpReduceMax(
            local_max
        );


    // ========================================================
    // 每个 warp 的 lane0
    // 把结果放到 Shared
    // ========================================================

    if (
        lane_id == 0
    ) {

        warp_values[
            warp_id
        ] =
            local_max;
    }


    __syncthreads();


    // ========================================================
    // STEP 3
    //
    // Warp0 再把所有 warp 的结果 reduction
    // ========================================================

    if (
        warp_id == 0
    ) {

        float value =
            (
                lane_id < num_warps
            )
            ?
            warp_values[lane_id]
            :
            -INFINITY;


        value =
            warpReduceMax(
                value
            );


        if (
            lane_id == 0
        ) {

            block_max =
                value;
        }
    }


    __syncthreads();


    // ========================================================
    // 现在所有 thread 都可以读:
    //
    // block_max
    // ========================================================

    const float max_value =
        block_max;


    // ========================================================
    // STEP 4
    //
    // exp(x - max)
    //
    // 每个 thread 负责自己的元素
    //
    // 同时计算 local sum
    // ========================================================

    float local_sum =
        0.0f;


    for (
        int col = tid;
        col < N;
        col += blockDim.x
    ) {

        const float e =
            expf(
                S[
                    row * N
                    +
                    col
                ]
                -
                max_value
            );


        // ====================================================
        // 先把 exp 写到 P
        //
        // 后面再 normalize
        // ====================================================

        P[
            row * N
            +
            col
        ] =
            e;


        local_sum +=
            e;
    }


    // ========================================================
    // STEP 5
    //
    // Warp Sum Reduction
    // ========================================================

    local_sum =
        warpReduceSum(
            local_sum
        );


    if (
        lane_id == 0
    ) {

        warp_values[
            warp_id
        ] =
            local_sum;
    }


    __syncthreads();


    // ========================================================
    // STEP 6
    //
    // Warp0:
    //
    // warp sums
    //
    // ↓
    //
    // block sum
    // ========================================================

    if (
        warp_id == 0
    ) {

        float value =
            (
                lane_id < num_warps
            )
            ?
            warp_values[lane_id]
            :
            0.0f;


        value =
            warpReduceSum(
                value
            );


        if (
            lane_id == 0
        ) {

            block_sum =
                value;
        }
    }


    __syncthreads();


    // ========================================================
    // STEP 7
    //
    // Normalize
    // ========================================================

    const float sum_value =
        block_sum;


    for (
        int col = tid;
        col < N;
        col += blockDim.x
    ) {

        P[
            row * N
            +
            col
        ]
        /=
        sum_value;
    }
}


// ============================================================
// Kernel 3
//
// O = P * V
//
// V1 这里故意暂时保持 naive
//
// 一个 thread 一个 O[row][d]
// ============================================================

__global__
void pvNaiveKernel(
    const float* __restrict__ P,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N,
    int D)
{
    const int d =
        blockIdx.x * blockDim.x
        +
        threadIdx.x;


    const int row =
        blockIdx.y * blockDim.y
        +
        threadIdx.y;


    if (
        row >= N
        ||
        d >= D
    ) {
        return;
    }


    float sum =
        0.0f;


    for (
        int j = 0;
        j < N;
        ++j
    ) {

        sum +=
            P[
                row * N
                +
                j
            ]
            *
            V[
                j * D
                +
                d
            ];
    }


    O[
        row * D
        +
        d
    ] =
        sum;
}


// ============================================================
// CPU Reference
// ============================================================

void attentionCPU(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    std::vector<float>& O,
    int N,
    int D)
{
    std::vector<float>
        S(N * N);


    std::vector<float>
        P(N * N);


    const float scale =
        1.0f
        /
        std::sqrt(
            static_cast<float>(D)
        );


    // ========================================================
    // QK^T
    // ========================================================

    for (
        int i = 0;
        i < N;
        ++i
    ) {

        for (
            int j = 0;
            j < N;
            ++j
        ) {

            float sum =
                0.0f;


            for (
                int d = 0;
                d < D;
                ++d
            ) {

                sum +=
                    Q[
                        i * D
                        +
                        d
                    ]
                    *
                    K[
                        j * D
                        +
                        d
                    ];
            }


            S[
                i * N
                +
                j
            ] =
                sum * scale;
        }
    }


    // ========================================================
    // Softmax
    // ========================================================

    for (
        int row = 0;
        row < N;
        ++row
    ) {

        float max_value =
            -INFINITY;


        for (
            int col = 0;
            col < N;
            ++col
        ) {

            max_value =
                std::max(
                    max_value,
                    S[
                        row * N
                        +
                        col
                    ]
                );
        }


        float sum =
            0.0f;


        for (
            int col = 0;
            col < N;
            ++col
        ) {

            const float e =
                std::exp(
                    S[
                        row * N
                        +
                        col
                    ]
                    -
                    max_value
                );


            P[
                row * N
                +
                col
            ] =
                e;


            sum +=
                e;
        }


        for (
            int col = 0;
            col < N;
            ++col
        ) {

            P[
                row * N
                +
                col
            ]
            /=
            sum;
        }
    }


    // ========================================================
    // PV
    // ========================================================

    O.resize(
        N * D
    );


    for (
        int row = 0;
        row < N;
        ++row
    ) {

        for (
            int d = 0;
            d < D;
            ++d
        ) {

            float sum =
                0.0f;


            for (
                int j = 0;
                j < N;
                ++j
            ) {

                sum +=
                    P[
                        row * N
                        +
                        j
                    ]
                    *
                    V[
                        j * D
                        +
                        d
                    ];
            }


            O[
                row * D
                +
                d
            ] =
                sum;
        }
    }
}


// ============================================================
// Main
// ============================================================

int main()
{
    // ========================================================
    // Single-Head Attention
    //
    // N = sequence length
    // D = head dimension
    // ========================================================

    constexpr int N =
        512;


    constexpr int D =
        64;


    std::vector<float>
        h_Q(N * D);


    std::vector<float>
        h_K(N * D);


    std::vector<float>
        h_V(N * D);


    std::vector<float>
        h_O(N * D);


    // ========================================================
    // Initialize
    // ========================================================

    for (
        int i = 0;
        i < N * D;
        ++i
    ) {

        h_Q[i] =
            static_cast<float>(
                (i % 17) - 8
            )
            /
            16.0f;


        h_K[i] =
            static_cast<float>(
                (i % 13) - 6
            )
            /
            16.0f;


        h_V[i] =
            static_cast<float>(
                (i % 11) - 5
            )
            /
            8.0f;
    }


    // ========================================================
    // Device memory
    // ========================================================

    float* d_Q = nullptr;
    float* d_K = nullptr;
    float* d_V = nullptr;

    float* d_S = nullptr;
    float* d_P = nullptr;
    float* d_O = nullptr;


    CUDA_CHECK(
        cudaMalloc(
            &d_Q,
            N * D * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_K,
            N * D * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_V,
            N * D * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_S,
            N * N * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_P,
            N * N * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_O,
            N * D * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_Q,
            h_Q.data(),
            N * D * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_K,
            h_K.data(),
            N * D * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_V,
            h_V.data(),
            N * D * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    // ========================================================
    // QK launch config
    // ========================================================

    dim3 qk_block(
        QK_TILE,
        QK_TILE
    );


    dim3 qk_grid(
        (N + QK_TILE - 1)
        /
        QK_TILE,

        (N + QK_TILE - 1)
        /
        QK_TILE
    );


    // ========================================================
    // Softmax:
    //
    // 一个 block 对应一个 row
    // ========================================================

    dim3 softmax_grid(
        N
    );


    dim3 softmax_block(
        SOFTMAX_THREADS
    );


    // ========================================================
    // PV
    // ========================================================

    dim3 pv_block(
        16,
        16
    );


    dim3 pv_grid(
        (D + pv_block.x - 1)
        /
        pv_block.x,

        (N + pv_block.y - 1)
        /
        pv_block.y
    );


    // ========================================================
    // Warmup
    // ========================================================

    for (
        int i = 0;
        i < 10;
        ++i
    ) {

        qkTiledKernel<<<
            qk_grid,
            qk_block
        >>>(
            d_Q,
            d_K,
            d_S,
            N,
            D
        );


        softmaxBlockKernel<<<
            softmax_grid,
            softmax_block
        >>>(
            d_S,
            d_P,
            N
        );


        pvNaiveKernel<<<
            pv_grid,
            pv_block
        >>>(
            d_P,
            d_V,
            d_O,
            N,
            D
        );
    }


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    // ========================================================
    // Benchmark
    // ========================================================

    cudaEvent_t start;
    cudaEvent_t stop;


    CUDA_CHECK(
        cudaEventCreate(
            &start
        )
    );


    CUDA_CHECK(
        cudaEventCreate(
            &stop
        )
    );


    constexpr int ITERATIONS =
        100;


    CUDA_CHECK(
        cudaEventRecord(
            start
        )
    );


    for (
        int i = 0;
        i < ITERATIONS;
        ++i
    ) {

        qkTiledKernel<<<
            qk_grid,
            qk_block
        >>>(
            d_Q,
            d_K,
            d_S,
            N,
            D
        );


        softmaxBlockKernel<<<
            softmax_grid,
            softmax_block
        >>>(
            d_S,
            d_P,
            N
        );


        pvNaiveKernel<<<
            pv_grid,
            pv_block
        >>>(
            d_P,
            d_V,
            d_O,
            N,
            D
        );
    }


    CUDA_CHECK(
        cudaEventRecord(
            stop
        )
    );


    CUDA_CHECK(
        cudaEventSynchronize(
            stop
        )
    );


    float total_ms =
        0.0f;


    CUDA_CHECK(
        cudaEventElapsedTime(
            &total_ms,
            start,
            stop
        )
    );


    const float avg_ms =
        total_ms
        /
        ITERATIONS;


    // ========================================================
    // Copy result
    // ========================================================

    CUDA_CHECK(
        cudaMemcpy(
            h_O.data(),
            d_O,
            N * D * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    // ========================================================
    // CPU Reference
    // ========================================================

    std::vector<float>
        h_ref;


    attentionCPU(
        h_Q,
        h_K,
        h_V,
        h_ref,
        N,
        D
    );


    // ========================================================
    // Verify
    // ========================================================

    float max_error =
        0.0f;


    for (
        int i = 0;
        i < N * D;
        ++i
    ) {

        max_error =
            std::max(
                max_error,
                std::fabs(
                    h_O[i]
                    -
                    h_ref[i]
                )
            );
    }


    // ========================================================
    // Device info
    // ========================================================

    cudaDeviceProp prop;


    CUDA_CHECK(
        cudaGetDeviceProperties(
            &prop,
            0
        )
    );


    std::cout
        << "GPU: "
        << prop.name
        << '\n';


    std::cout
        << "N = "
        << N
        << '\n';


    std::cout
        << "D = "
        << D
        << '\n';


    std::cout
        << "QK Tile = "
        << QK_TILE
        << " x "
        << QK_TILE
        << '\n';


    std::cout
        << "Softmax Threads = "
        << SOFTMAX_THREADS
        << '\n';


    std::cout
        << "Attention V1 Average Time = "
        << avg_ms
        << " ms\n";


    std::cout
        << "Max Error = "
        << max_error
        << '\n';


    std::cout
        << "First output row:\n";


    for (
        int d = 0;
        d < std::min(D, 8);
        ++d
    ) {

        std::cout
            << h_O[d]
            << " ";
    }


    std::cout
        << '\n';


    // ========================================================
    // Cleanup
    // ========================================================

    CUDA_CHECK(
        cudaEventDestroy(
            start
        )
    );


    CUDA_CHECK(
        cudaEventDestroy(
            stop
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_Q
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_K
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_V
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_S
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_P
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_O
        )
    );


    return 0;
}