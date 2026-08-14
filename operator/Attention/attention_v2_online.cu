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
            std::cerr                                         \
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
// Teaching configuration
//
// 一个 block 处理一个 Query row
//
// 每次处理:
//
//      128 个 K/V token
//
// block:
//
//      128 threads
//
// 当前教学版:
//
//      D <= 128
// ============================================================

constexpr int TILE_KEYS =
    128;

constexpr int BLOCK_THREADS =
    128;

constexpr int MAX_HEAD_DIM =
    128;

constexpr int WARP_SIZE =
    32;


// ============================================================
// Warp Reduce Max
// ============================================================

__device__ __forceinline__
float warpReduceMax(float value)
{
#pragma unroll
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
// Warp Reduce Sum
// ============================================================

__device__ __forceinline__
float warpReduceSum(float value)
{
#pragma unroll
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
// Block Reduce Max
//
// 先:
//
//      Warp Reduction
//
// 再:
//
//      Warp0 Reduction
//
// shared:
//
//      存每个 warp 的结果
// ============================================================

__device__ __forceinline__
float blockReduceMax(
    float value,
    float* warp_values)
{
    const int lane_id =
        threadIdx.x % WARP_SIZE;

    const int warp_id =
        threadIdx.x / WARP_SIZE;

    const int num_warps =
        blockDim.x / WARP_SIZE;


    // ========================================================
    // Warp 内 reduction
    // ========================================================

    value =
        warpReduceMax(
            value
        );


    // ========================================================
    // 每个 warp 的 lane0
    //
    // 把 warp max 写 Shared
    // ========================================================

    if (
        lane_id == 0
    ) {

        warp_values[
            warp_id
        ] =
            value;
    }


    __syncthreads();


    // ========================================================
    // Warp0 再 reduction
    // ========================================================

    if (
        warp_id == 0
    ) {

        float warp_value =
            (
                lane_id < num_warps
            )
            ?
            warp_values[lane_id]
            :
            -INFINITY;


        warp_value =
            warpReduceMax(
                warp_value
            );


        if (
            lane_id == 0
        ) {

            warp_values[0] =
                warp_value;
        }
    }


    __syncthreads();


    return warp_values[0];
}


// ============================================================
// Block Reduce Sum
// ============================================================

__device__ __forceinline__
float blockReduceSum(
    float value,
    float* warp_values)
{
    const int lane_id =
        threadIdx.x % WARP_SIZE;

    const int warp_id =
        threadIdx.x / WARP_SIZE;

    const int num_warps =
        blockDim.x / WARP_SIZE;


    value =
        warpReduceSum(
            value
        );


    if (
        lane_id == 0
    ) {

        warp_values[
            warp_id
        ] =
            value;
    }


    __syncthreads();


    if (
        warp_id == 0
    ) {

        float warp_value =
            (
                lane_id < num_warps
            )
            ?
            warp_values[lane_id]
            :
            0.0f;


        warp_value =
            warpReduceSum(
                warp_value
            );


        if (
            lane_id == 0
        ) {

            warp_values[0] =
                warp_value;
        }
    }


    __syncthreads();


    return warp_values[0];
}


// ============================================================
// Attention V2
//
// Online Softmax Attention
//
// 一个 block:
//
//      负责一个 Q row
//
// K / V:
//
//      按 TILE_KEYS 分块
//
// 不生成:
//
//      S[N,N]
//
// 不生成:
//
//      P[N,N]
//
// ============================================================

__global__
void onlineAttentionKernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N,
    int D)
{
    // ========================================================
    // 一个 block 对应:
    //
    //      一个 Query token
    // ========================================================

    const int query_row =
        blockIdx.x;


    const int tid =
        threadIdx.x;


    if (
        query_row >= N
    ) {
        return;
    }


    // ========================================================
    // 教学版限制
    // ========================================================

    if (
        D > MAX_HEAD_DIM
    ) {
        return;
    }


    // ========================================================
    // Shared Q
    //
    // 当前 Q row:
    //
    //      Q[query_row][:]
    //
    // 只加载一次
    //
    // 后面所有 K tile 都复用
    // ========================================================

    __shared__
    float Qs[MAX_HEAD_DIM];


    // ========================================================
    // 当前 K tile 对应的 scores
    //
    // 最多 128 个
    // ========================================================

    __shared__
    float scores[TILE_KEYS];


    // ========================================================
    // Online Softmax weights
    //
    // weight[j]
    //
    // =
    //
    // exp(score[j] - m_new)
    // ========================================================

    __shared__
    float weights[TILE_KEYS];


    // ========================================================
    // Block reduction 临时空间
    //
    // 128 threads
    //
    // =
    //
    // 4 warps
    //
    // 这里开 32 只是写起来方便
    // ========================================================

    __shared__
    float warp_values[32];


    // ========================================================
    // Online Softmax State
    // ========================================================

    __shared__
    float m_shared;

    __shared__
    float l_shared;


    // ========================================================
    // 当前 tile 的状态
    // ========================================================

    __shared__
    float tile_max_shared;

    __shared__
    float tile_sum_shared;

    __shared__
    float new_m_shared;

    __shared__
    float alpha_shared;


    // ========================================================
    // STEP 0
    //
    // Global Q -> Shared Q
    //
    // Q row 会被所有 K tile 重复使用
    // ========================================================

    for (
        int d = tid;
        d < D;
        d += blockDim.x
    ) {

        Qs[d] =
            Q[
                query_row * D
                +
                d
            ];
    }


    // ========================================================
    // 初始化 Online Softmax
    //
    // 什么都没见过:
    //
    // m = -inf
    // l = 0
    // ========================================================

    if (
        tid == 0
    ) {

        m_shared =
            -INFINITY;

        l_shared =
            0.0f;
    }


    // ========================================================
    // 每个 thread 如果 tid < D
    //
    // 就负责一个 output dimension
    //
    // 并把 output numerator
    //
    // 一直保存在自己的 register 中
    // ========================================================

    float o_acc =
        0.0f;


    __syncthreads();


    const float scale =
        rsqrtf(
            static_cast<float>(D)
        );


    // ========================================================
    // 开始扫 K / V tiles
    //
    // N = 512
    //
    // TILE_KEYS = 128
    //
    // 就会:
    //
    // tile0: key   0~127
    // tile1: key 128~255
    // tile2: key 256~383
    // tile3: key 384~511
    // ========================================================

    for (
        int key_start = 0;
        key_start < N;
        key_start += TILE_KEYS
    ) {

        const int key_index =
            key_start
            +
            tid;


        // ====================================================
        // STEP 1
        //
        // 每个 thread 算一个 attention score
        //
        // score:
        //
        // Q[query_row]
        //
        // dot
        //
        // K[key_index]
        // ====================================================

        float score =
            -INFINITY;


        if (
            key_index < N
        ) {

            float dot =
                0.0f;


#pragma unroll 4
            for (
                int d = 0;
                d < D;
                ++d
            ) {

                dot +=
                    Qs[d]
                    *
                    K[
                        key_index * D
                        +
                        d
                    ];
            }


            score =
                dot * scale;
        }


        // ====================================================
        // score 存到 Shared
        //
        // 后面算 V contribution 时会用到
        // ====================================================

        scores[tid] =
            score;


        // ====================================================
        // STEP 2
        //
        // 求当前 tile 最大值
        //
        // 使用:
        //
        // Warp Reduction
        // +
        // Block Reduction
        // ====================================================

        const float tile_max =
            blockReduceMax(
                score,
                warp_values
            );


        if (
            tid == 0
        ) {

            tile_max_shared =
                tile_max;


            // =================================================
            // Online Softmax:
            //
            // m_new
            //
            // =
            //
            // max(m_old, m_tile)
            // =================================================

            const float old_m =
                m_shared;


            const float new_m =
                fmaxf(
                    old_m,
                    tile_max
                );


            new_m_shared =
                new_m;


            // =================================================
            // 旧结果要缩放:
            //
            // alpha
            //
            // =
            //
            // exp(m_old - m_new)
            //
            //
            // 第一个 tile:
            //
            // old_m = -inf
            //
            // alpha = 0
            // =================================================

            if (
                isinf(old_m)
                &&
                old_m < 0.0f
            ) {

                alpha_shared =
                    0.0f;
            }
            else {

                alpha_shared =
                    expf(
                        old_m
                        -
                        new_m
                    );
            }
        }


        __syncthreads();


        // ====================================================
        // STEP 3
        //
        // 当前 tile:
        //
        // weight
        //
        // =
        //
        // exp(score - m_new)
        //
        // ====================================================

        float weight =
            0.0f;


        if (
            key_index < N
        ) {

            weight =
                expf(
                    score
                    -
                    new_m_shared
                );
        }


        weights[tid] =
            weight;


        // ====================================================
        // 求当前 tile 的:
        //
        // sum exp(score - m_new)
        // ====================================================

        const float tile_sum =
            blockReduceSum(
                weight,
                warp_values
            );


        if (
            tid == 0
        ) {

            tile_sum_shared =
                tile_sum;
        }


        __syncthreads();


        // ====================================================
        // STEP 4
        //
        // 更新 output numerator
        //
        //
        // o_new
        //
        // =
        //
        // alpha * o_old
        //
        // +
        //
        // sum_j
        //
        // weight_j * V_j
        //
        //
        // tid < D:
        //
        // 每个 thread 负责一个 output dimension
        // ====================================================

        if (
            tid < D
        ) {

            float tile_o =
                0.0f;


            // ================================================
            // 当前 tile:
            //
            // sum_j
            //
            // weight[j]
            // *
            // V[j][tid]
            // ================================================

            for (
                int j = 0;
                j < TILE_KEYS;
                ++j
            ) {

                const int v_row =
                    key_start
                    +
                    j;


                if (
                    v_row < N
                ) {

                    tile_o +=
                        weights[j]
                        *
                        V[
                            v_row * D
                            +
                            tid
                        ];
                }
            }


            // ================================================
            // Online update
            //
            // 注意:
            //
            // o_acc 是 register
            // ================================================

            o_acc =
                o_acc
                *
                alpha_shared
                +
                tile_o;
        }


        // ====================================================
        // 所有 output dimension
        // 都完成当前 tile 更新后
        // ====================================================

        __syncthreads();


        // ====================================================
        // STEP 5
        //
        // 更新 l
        //
        //
        // l_new
        //
        // =
        //
        // alpha*l_old
        //
        // +
        //
        // tile_sum
        // ====================================================

        if (
            tid == 0
        ) {

            const float new_l =
                l_shared
                *
                alpha_shared
                +
                tile_sum_shared;


            // =================================================
            // 更新状态
            // =================================================

            m_shared =
                new_m_shared;


            l_shared =
                new_l;
        }


        __syncthreads();
    }


    // ========================================================
    // 所有 K/V tiles 处理结束
    //
    // 最终:
    //
    // O
    //
    // =
    //
    // o_acc / l
    // ========================================================

    if (
        tid < D
    ) {

        O[
            query_row * D
            +
            tid
        ] =
            o_acc
            /
            l_shared;
    }
}


// ============================================================
// CPU Reference Attention
//
// 标准:
//
// S = QK^T / sqrt(D)
// P = Softmax(S)
// O = PV
//
// 用来验证 Online 版本
// ============================================================

void attentionCPU(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    std::vector<float>& O,
    int N,
    int D)
{
    O.resize(
        N * D
    );


    const float scale =
        1.0f
        /
        std::sqrt(
            static_cast<float>(D)
        );


    // ========================================================
    // 一行一行算
    // ========================================================

    for (
        int q = 0;
        q < N;
        ++q
    ) {

        std::vector<float>
            scores(N);


        // ====================================================
        // QK^T
        // ====================================================

        for (
            int k = 0;
            k < N;
            ++k
        ) {

            float dot =
                0.0f;


            for (
                int d = 0;
                d < D;
                ++d
            ) {

                dot +=
                    Q[
                        q * D
                        +
                        d
                    ]
                    *
                    K[
                        k * D
                        +
                        d
                    ];
            }


            scores[k] =
                dot * scale;
        }


        // ====================================================
        // Stable Softmax
        // ====================================================

        float max_value =
            -INFINITY;


        for (
            int k = 0;
            k < N;
            ++k
        ) {

            max_value =
                std::max(
                    max_value,
                    scores[k]
                );
        }


        float sum =
            0.0f;


        std::vector<float>
            probs(N);


        for (
            int k = 0;
            k < N;
            ++k
        ) {

            probs[k] =
                std::exp(
                    scores[k]
                    -
                    max_value
                );


            sum +=
                probs[k];
        }


        for (
            int k = 0;
            k < N;
            ++k
        ) {

            probs[k] /=
                sum;
        }


        // ====================================================
        // PV
        // ====================================================

        for (
            int d = 0;
            d < D;
            ++d
        ) {

            float value =
                0.0f;


            for (
                int k = 0;
                k < N;
                ++k
            ) {

                value +=
                    probs[k]
                    *
                    V[
                        k * D
                        +
                        d
                    ];
            }


            O[
                q * D
                +
                d
            ] =
                value;
        }
    }
}


// ============================================================
// Main
// ============================================================

int main()
{
    // ========================================================
    // Single Head Attention
    // ========================================================

    constexpr int N =
        512;


    constexpr int D =
        64;


    static_assert(
        D <= MAX_HEAD_DIM,
        "D must be <= MAX_HEAD_DIM"
    );


    // ========================================================
    // Host Data
    // ========================================================

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
    // Device pointers
    //
    // 注意:
    //
    // 已经没有:
    //
    // d_S
    // d_P
    // ========================================================

    float* d_Q =
        nullptr;


    float* d_K =
        nullptr;


    float* d_V =
        nullptr;


    float* d_O =
        nullptr;


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
            &d_O,
            N * D * sizeof(float)
        )
    );


    // ========================================================
    // H -> D
    // ========================================================

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
    // 一个 block:
    //
    // 一个 Q row
    //
    // 所以:
    //
    // grid = N blocks
    // ========================================================

    dim3 grid(
        N
    );


    dim3 block(
        BLOCK_THREADS
    );


    // ========================================================
    // Warmup
    // ========================================================

    for (
        int i = 0;
        i < 10;
        ++i
    ) {

        onlineAttentionKernel<<<
            grid,
            block
        >>>(
            d_Q,
            d_K,
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

        onlineAttentionKernel<<<
            grid,
            block
        >>>(
            d_Q,
            d_K,
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
    // Device Info
    // ========================================================

    cudaDeviceProp prop;


    CUDA_CHECK(
        cudaGetDeviceProperties(
            &prop,
            0
        )
    );


    // ========================================================
    // Print
    // ========================================================

    std::cout
        << "GPU: "
        << prop.name
        << '\n';


    std::cout
        << "Compute Capability: "
        << prop.major
        << "."
        << prop.minor
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
        << "K/V Tile Size = "
        << TILE_KEYS
        << '\n';


    std::cout
        << "Attention V2 Online Average Time = "
        << avg_ms
        << " ms\n";


    std::cout
        << "Max Error = "
        << max_error
        << '\n';


    std::cout
        << "First Output Row:\n";


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
            d_O
        )
    );


    return 0;
}