#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>


// ============================================================
// CUDA CHECK
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
// GEMM V9 Configuration
//
// Block Tile:
//
//      128 x 64
//
// K Tile:
//
//      16
//
// Warp Tile:
//
//      32 x 32
//
// Thread Tile:
//
//      8 x 4
//
// 8 warps = 256 threads
// ============================================================

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 16;

constexpr int WARP_M = 32;
constexpr int WARP_N = 32;

constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;

constexpr int NUM_WARPS =
    WARPS_M * WARPS_N;

constexpr int THREAD_M = 8;
constexpr int THREAD_N = 4;

constexpr int WARP_SIZE = 32;

constexpr int THREADS_PER_BLOCK =
    NUM_WARPS * WARP_SIZE;

constexpr int VEC_SIZE = 4;


// ============================================================
// Shared Memory Transpose Padding
//
// 原来:
//
// A tile = [128][16]
//
// 转置后:
//
// A^T tile = [16][128]
//
// 加 padding:
//
//      [16][129]
// ============================================================

constexpr int A_TRANSPOSE_STRIDE =
    BLOCK_M + 1;


// ============================================================
// Softmax
// ============================================================

constexpr int SOFTMAX_THREADS = 256;


// ============================================================
// float4 Safe Load
// ============================================================

__device__ __forceinline__
float4 loadFloat4Safe(
    const float* ptr,
    int row,
    int col,
    int rows,
    int cols,
    int ld)
{
    float4 result =
        make_float4(
            0.0f,
            0.0f,
            0.0f,
            0.0f
        );

    if (row >= rows) {
        return result;
    }

    const int index =
        row * ld + col;

    if (
        col + 3 < cols
        &&
        index % 4 == 0
    ) {
        return
            *reinterpret_cast<const float4*>(
                ptr + index
            );
    }

    if (col + 0 < cols) {
        result.x = ptr[index + 0];
    }

    if (col + 1 < cols) {
        result.y = ptr[index + 1];
    }

    if (col + 2 < cols) {
        result.z = ptr[index + 2];
    }

    if (col + 3 < cols) {
        result.w = ptr[index + 3];
    }

    return result;
}


// ============================================================
// float4 Safe Store
// ============================================================

__device__ __forceinline__
void storeFloat4Safe(
    float* ptr,
    int row,
    int col,
    int rows,
    int cols,
    int ld,
    float v0,
    float v1,
    float v2,
    float v3)
{
    if (row >= rows) {
        return;
    }

    const int index =
        row * ld + col;

    if (
        col + 3 < cols
        &&
        index % 4 == 0
    ) {
        *reinterpret_cast<float4*>(
            ptr + index
        ) =
            make_float4(
                v0,
                v1,
                v2,
                v3
            );

        return;
    }

    if (col + 0 < cols) {
        ptr[index + 0] = v0;
    }

    if (col + 1 < cols) {
        ptr[index + 1] = v1;
    }

    if (col + 2 < cols) {
        ptr[index + 2] = v2;
    }

    if (col + 3 < cols) {
        ptr[index + 3] = v3;
    }
}


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
// Kernel 1
//
// S = Q * K^T / sqrt(D)
//
// Q: [N, D]
// K: [N, D]
//
// S: [N, N]
//
// ============================================================
//
// 注意:
//
// 普通 GEMM 是:
//
// A[M,K] × B[K,N]
//
// 这里:
//
// Q[N,D] × K^T[D,N]
//
// 但是我们根本不创建 K^T。
//
// 直接读取:
//
// K[col][d]
//
// 就等价于:
//
// K^T[d][col]
//
// ============================================================

__global__
void qkV9Kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ S,
    int N,
    int D)
{
    const int tid =
        threadIdx.x;


    // ========================================================
    // Warp
    // ========================================================

    const int warp_id =
        tid / WARP_SIZE;

    const int lane_id =
        tid % WARP_SIZE;


    // ========================================================
    // Q Shared Layout
    //
    // Q logical tile:
    //
    //      [128][16]
    //
    // Shared:
    //
    //      [16][129]
    //
    // 转置 + padding
    // ========================================================

    __shared__ __align__(16)
    float QsT[
        BLOCK_K
    ][
        A_TRANSPOSE_STRIDE
    ];


    // ========================================================
    // K
    //
    // 数学上我们需要 K^T:
    //
    // [D][N]
    //
    // 所以 Shared 直接存成:
    //
    // [K tile][N tile]
    //
    //      [16][64]
    // ========================================================

    __shared__ __align__(16)
    float Ks[
        BLOCK_K
    ][
        BLOCK_N
    ];


    // ========================================================
    // 当前 block 负责的 S Tile
    //
    // 128 x 64
    // ========================================================

    const int block_row =
        blockIdx.y * BLOCK_M;

    const int block_col =
        blockIdx.x * BLOCK_N;


    // ========================================================
    // Block -> Warp
    // ========================================================

    const int warp_m =
        warp_id / WARPS_N;

    const int warp_n =
        warp_id % WARPS_N;


    const int warp_row =
        warp_m * WARP_M;

    const int warp_col =
        warp_n * WARP_N;


    // ========================================================
    // Warp -> Lane
    //
    // 32 lanes
    //
    // 排成:
    //
    //      4 x 8
    //
    // 每 lane:
    //
    //      8 x 4
    // ========================================================

    constexpr int LANES_N =
        WARP_N / THREAD_N;

    // 32 / 4 = 8


    const int lane_m =
        lane_id / LANES_N;

    const int lane_n =
        lane_id % LANES_N;


    // ========================================================
    // 当前 thread 的 output tile
    //
    // 在 block 内的位置
    // ========================================================

    const int thread_row =
        warp_row
        +
        lane_m * THREAD_M;


    const int thread_col =
        warp_col
        +
        lane_n * THREAD_N;


    // ========================================================
    // Register Accumulator
    //
    // 一个 thread:
    //
    //      8 x 4
    //
    // = 32 个 S
    // ========================================================

    float acc[THREAD_M][THREAD_N];


#pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {

#pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {

            acc[i][j] =
                0.0f;
        }
    }


    // ========================================================
    // D direction tile
    // ========================================================

    const int num_k_tiles =
        (D + BLOCK_K - 1)
        /
        BLOCK_K;


    // ========================================================
    // Q Tile:
    //
    // 128 x 16
    //
    // 2048 float
    //
    // = 512 float4
    //
    // 256 threads
    //
    // 每个 thread 搬两个 float4
    // ========================================================

    constexpr int Q_VECS_PER_ROW =
        BLOCK_K / VEC_SIZE;

    // 4


    constexpr int Q_TOTAL_VECS =
        BLOCK_M * Q_VECS_PER_ROW;

    // 512


    // ========================================================
    // K Tile:
    //
    // 64 rows of K
    //
    // 每 row 取 16 D
    //
    // =
    //
    // 64 x 16
    //
    // =
    //
    // 1024 floats
    //
    // = 256 float4
    //
    // 所以每个 thread 搬一个 float4
    // ========================================================

    constexpr int K_VECS_PER_ROW =
        BLOCK_K / VEC_SIZE;

    // 4


    // ========================================================
    // Main D Tile Loop
    // ========================================================

    for (
        int tile = 0;
        tile < num_k_tiles;
        ++tile
    ) {

        const int d0 =
            tile * BLOCK_K;


        // ====================================================
        // STEP 1
        //
        // Global Q
        //
        //      ↓
        //
        // Transposed Shared Q
        //
        // ====================================================

        for (
            int vec_index = tid;
            vec_index < Q_TOTAL_VECS;
            vec_index += THREADS_PER_BLOCK
        ) {

            // ------------------------------------------------
            // 当前 Q tile 的 row
            // ------------------------------------------------

            const int q_row =
                vec_index
                /
                Q_VECS_PER_ROW;


            // ------------------------------------------------
            // row 内第几个 float4
            // ------------------------------------------------

            const int q_vec =
                vec_index
                %
                Q_VECS_PER_ROW;


            const int q_d =
                q_vec * VEC_SIZE;


            // ------------------------------------------------
            // Global coordinate
            // ------------------------------------------------

            const int global_row =
                block_row
                +
                q_row;


            const int global_d =
                d0
                +
                q_d;


            // ------------------------------------------------
            // Global contiguous float4
            // ------------------------------------------------

            const float4 value =
                loadFloat4Safe(
                    Q,

                    global_row,
                    global_d,

                    N,
                    D,
                    D
                );


            // ------------------------------------------------
            // 转置写 Shared
            //
            // Global:
            //
            // Q[row][d]
            //
            // Shared:
            //
            // QsT[d][row]
            // ------------------------------------------------

            QsT[q_d + 0][q_row] =
                value.x;

            QsT[q_d + 1][q_row] =
                value.y;

            QsT[q_d + 2][q_row] =
                value.z;

            QsT[q_d + 3][q_row] =
                value.w;
        }


        // ====================================================
        // STEP 2
        //
        // Global K
        //
        //      ↓
        //
        // Shared K^T Layout
        //
        // ====================================================
        //
        // tid 0~255
        //
        // 每个 K row 有:
        //
        //      16 / 4 = 4 float4
        //
        // 所以:
        //
        // k_row = tid / 4
        //
        //      0~63
        //
        // ====================================================

        const int k_row =
            tid / K_VECS_PER_ROW;


        const int k_vec =
            tid % K_VECS_PER_ROW;


        const int k_d =
            k_vec * VEC_SIZE;


        const int global_k_row =
            block_col
            +
            k_row;


        const int global_k_d =
            d0
            +
            k_d;


        const float4 kval =
            loadFloat4Safe(
                K,

                global_k_row,
                global_k_d,

                N,
                D,
                D
            );


        // ====================================================
        // K 原来:
        //
        // K[token][d]
        //
        //
        // 数学上 QK^T 需要:
        //
        // K^T[d][token]
        //
        //
        // 所以 Shared 直接写成:
        //
        // Ks[d][token]
        // ====================================================

        Ks[k_d + 0][k_row] =
            kval.x;

        Ks[k_d + 1][k_row] =
            kval.y;

        Ks[k_d + 2][k_row] =
            kval.z;

        Ks[k_d + 3][k_row] =
            kval.w;


        // ====================================================
        // Tile Ready
        // ====================================================

        __syncthreads();


        // ====================================================
        // COMPUTE
        //
        // 一个 thread:
        //
        // 8 Q rows
        //
        // x
        //
        // 4 K rows
        //
        // =
        //
        // 8 x 4 S
        // ====================================================

#pragma unroll
        for (
            int k = 0;
            k < BLOCK_K;
            ++k
        ) {

            float q_frag[THREAD_M];

            float k_frag[THREAD_N];


            // ================================================
            // Shared Q -> Registers
            // ================================================

#pragma unroll
            for (
                int i = 0;
                i < THREAD_M;
                ++i
            ) {

                q_frag[i] =
                    QsT[
                        k
                    ][
                        thread_row + i
                    ];
            }


            // ================================================
            // Shared K -> Registers
            // ================================================

#pragma unroll
            for (
                int j = 0;
                j < THREAD_N;
                ++j
            ) {

                k_frag[j] =
                    Ks[
                        k
                    ][
                        thread_col + j
                    ];
            }


            // ================================================
            // Register Outer Product
            //
            // 8 x 4
            //
            // = 32 FMA
            // ================================================

#pragma unroll
            for (
                int i = 0;
                i < THREAD_M;
                ++i
            ) {

#pragma unroll
                for (
                    int j = 0;
                    j < THREAD_N;
                    ++j
                ) {

                    acc[i][j]
                        +=
                        q_frag[i]
                        *
                        k_frag[j];
                }
            }
        }


        // ====================================================
        // 下一 tile 覆盖 Shared Memory 前
        // ====================================================

        __syncthreads();
    }


    // ========================================================
    // Scale
    //
    // 1 / sqrt(D)
    // ========================================================

    const float scale =
        rsqrtf(
            static_cast<float>(D)
        );


    // ========================================================
    // Register -> Global S
    //
    // 每一行有 4 个连续值
    // ========================================================

#pragma unroll
    for (
        int i = 0;
        i < THREAD_M;
        ++i
    ) {

        const int global_row =
            block_row
            +
            thread_row
            +
            i;


        const int global_col =
            block_col
            +
            thread_col;


        storeFloat4Safe(
            S,

            global_row,
            global_col,

            N,
            N,
            N,

            acc[i][0] * scale,
            acc[i][1] * scale,
            acc[i][2] * scale,
            acc[i][3] * scale
        );
    }
}


// ============================================================
// Kernel 2
//
// P = Softmax(S)
//
// 一个 Block 处理 S 的一整行
//
// 使用:
//
// Thread local reduction
// ↓
// Warp __shfl_down_sync
// ↓
// Block reduction
// ============================================================

__global__
void softmaxV1Kernel(
    const float* __restrict__ S,
    float* __restrict__ P,
    int N)
{
    const int row =
        blockIdx.x;


    const int tid =
        threadIdx.x;


    const int lane_id =
        tid % WARP_SIZE;


    const int warp_id =
        tid / WARP_SIZE;


    const int num_warps =
        blockDim.x / WARP_SIZE;


    __shared__
    float warp_values[32];


    __shared__
    float block_max;


    __shared__
    float block_sum;


    // ========================================================
    // Local Max
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
    // Warp Max
    // ========================================================

    local_max =
        warpReduceMax(
            local_max
        );


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
    // Warp0 reduce warp results
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


    const float max_value =
        block_max;


    // ========================================================
    // exp + local sum
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
    // Warp Sum
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
    // Warp0 -> Block Sum
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


    const float sum_value =
        block_sum;


    // ========================================================
    // Normalize
    // ========================================================

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
// 这里就是标准 GEMM:
//
// P:
//
//      [N, N]
//
// V:
//
//      [N, D]
//
// O:
//
//      [N, D]
//
// 所以这里可以非常直接使用 GEMM V9。
// ============================================================

__global__
void pvV9Kernel(
    const float* __restrict__ P,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N,
    int D)
{
    const int tid =
        threadIdx.x;


    const int warp_id =
        tid / WARP_SIZE;


    const int lane_id =
        tid % WARP_SIZE;


    // ========================================================
    // P tile:
    //
    // logical:
    //
    //      [128][16]
    //
    // Shared:
    //
    //      [16][129]
    //
    // ========================================================

    __shared__ __align__(16)
    float PsT[
        BLOCK_K
    ][
        A_TRANSPOSE_STRIDE
    ];


    // ========================================================
    // V tile:
    //
    //      [16][64]
    // ========================================================

    __shared__ __align__(16)
    float Vs[
        BLOCK_K
    ][
        BLOCK_N
    ];


    // ========================================================
    // O Block Tile
    // ========================================================

    const int block_row =
        blockIdx.y * BLOCK_M;


    const int block_col =
        blockIdx.x * BLOCK_N;


    // ========================================================
    // Warp Tile
    // ========================================================

    const int warp_m =
        warp_id / WARPS_N;


    const int warp_n =
        warp_id % WARPS_N;


    const int warp_row =
        warp_m * WARP_M;


    const int warp_col =
        warp_n * WARP_N;


    // ========================================================
    // Lane Tile
    // ========================================================

    constexpr int LANES_N =
        WARP_N / THREAD_N;


    const int lane_m =
        lane_id / LANES_N;


    const int lane_n =
        lane_id % LANES_N;


    // ========================================================
    // Thread Tile
    // ========================================================

    const int thread_row =
        warp_row
        +
        lane_m * THREAD_M;


    const int thread_col =
        warp_col
        +
        lane_n * THREAD_N;


    // ========================================================
    // Register Accumulator
    // ========================================================

    float acc[THREAD_M][THREAD_N];


#pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {

#pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {

            acc[i][j] =
                0.0f;
        }
    }


    // ========================================================
    // P's K dimension = N
    // ========================================================

    const int num_k_tiles =
        (N + BLOCK_K - 1)
        /
        BLOCK_K;


    // ========================================================
    // P:
    //
    // 128 x 16
    //
    // = 512 float4
    // ========================================================

    constexpr int P_VECS_PER_ROW =
        BLOCK_K / VEC_SIZE;


    constexpr int P_TOTAL_VECS =
        BLOCK_M
        *
        P_VECS_PER_ROW;


    // ========================================================
    // V:
    //
    // 16 x 64
    //
    // = 256 float4
    // ========================================================

    constexpr int V_VECS_PER_ROW =
        BLOCK_N / VEC_SIZE;


    // ========================================================
    // Main GEMM loop
    // ========================================================

    for (
        int tile = 0;
        tile < num_k_tiles;
        ++tile
    ) {

        const int k0 =
            tile * BLOCK_K;


        // ====================================================
        // Global P -> Transposed Shared P
        // ====================================================

        for (
            int vec_index = tid;
            vec_index < P_TOTAL_VECS;
            vec_index += THREADS_PER_BLOCK
        ) {

            const int p_row =
                vec_index
                /
                P_VECS_PER_ROW;


            const int p_vec =
                vec_index
                %
                P_VECS_PER_ROW;


            const int p_k =
                p_vec * VEC_SIZE;


            const int global_row =
                block_row
                +
                p_row;


            const int global_col =
                k0
                +
                p_k;


            const float4 value =
                loadFloat4Safe(
                    P,

                    global_row,
                    global_col,

                    N,
                    N,
                    N
                );


            PsT[p_k + 0][p_row] =
                value.x;

            PsT[p_k + 1][p_row] =
                value.y;

            PsT[p_k + 2][p_row] =
                value.z;

            PsT[p_k + 3][p_row] =
                value.w;
        }


        // ====================================================
        // Global V -> Shared V
        //
        // V tile:
        //
        // 16 x 64
        //
        // 256 float4
        //
        // 一个 thread 一个 float4
        // ====================================================

        const int v_k =
            tid
            /
            V_VECS_PER_ROW;


        const int v_vec =
            tid
            %
            V_VECS_PER_ROW;


        const int v_col =
            v_vec
            *
            VEC_SIZE;


        const int global_v_row =
            k0
            +
            v_k;


        const int global_v_col =
            block_col
            +
            v_col;


        const float4 value =
            loadFloat4Safe(
                V,

                global_v_row,
                global_v_col,

                N,
                D,
                D
            );


        Vs[v_k][v_col + 0] =
            value.x;

        Vs[v_k][v_col + 1] =
            value.y;

        Vs[v_k][v_col + 2] =
            value.z;

        Vs[v_k][v_col + 3] =
            value.w;


        __syncthreads();


        // ====================================================
        // Compute
        // ====================================================

#pragma unroll
        for (
            int k = 0;
            k < BLOCK_K;
            ++k
        ) {

            float p_frag[THREAD_M];

            float v_frag[THREAD_N];


#pragma unroll
            for (
                int i = 0;
                i < THREAD_M;
                ++i
            ) {

                p_frag[i] =
                    PsT[
                        k
                    ][
                        thread_row + i
                    ];
            }


#pragma unroll
            for (
                int j = 0;
                j < THREAD_N;
                ++j
            ) {

                v_frag[j] =
                    Vs[
                        k
                    ][
                        thread_col + j
                    ];
            }


#pragma unroll
            for (
                int i = 0;
                i < THREAD_M;
                ++i
            ) {

#pragma unroll
                for (
                    int j = 0;
                    j < THREAD_N;
                    ++j
                ) {

                    acc[i][j]
                        +=
                        p_frag[i]
                        *
                        v_frag[j];
                }
            }
        }


        __syncthreads();
    }


    // ========================================================
    // Register -> O
    // ========================================================

#pragma unroll
    for (
        int i = 0;
        i < THREAD_M;
        ++i
    ) {

        const int global_row =
            block_row
            +
            thread_row
            +
            i;


        const int global_col =
            block_col
            +
            thread_col;


        storeFloat4Safe(
            O,

            global_row,
            global_col,

            N,
            D,
            D,

            acc[i][0],
            acc[i][1],
            acc[i][2],
            acc[i][3]
        );
    }
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
    // Q K^T
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
                sum
                *
                scale;
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
    // P V
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
    // Single Head Attention
    //
    // N:
    //
    //      sequence length
    //
    // D:
    //
    //      head dimension
    // ========================================================

    constexpr int N = 512;

    constexpr int D = 64;


    // ========================================================
    // Host
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
    // Init
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
    // Device
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
    // QK GEMM config
    //
    // 256 threads
    // ========================================================

    dim3 gemm_block(
        THREADS_PER_BLOCK
    );


    // ========================================================
    // QK output:
    //
    // S = [N,N]
    //
    // block tile:
    //
    // 128 x 64
    // ========================================================

    dim3 qk_grid(
        (N + BLOCK_N - 1)
            /
            BLOCK_N,

        (N + BLOCK_M - 1)
            /
            BLOCK_M
    );


    // ========================================================
    // Softmax
    //
    // 一个 block 一行
    // ========================================================

    dim3 softmax_grid(
        N
    );


    dim3 softmax_block(
        SOFTMAX_THREADS
    );


    // ========================================================
    // PV output:
    //
    // O = [N,D]
    // ========================================================

    dim3 pv_grid(
        (D + BLOCK_N - 1)
            /
            BLOCK_N,

        (N + BLOCK_M - 1)
            /
            BLOCK_M
    );


    // ========================================================
    // Warmup
    // ========================================================

    for (
        int i = 0;
        i < 10;
        ++i
    ) {

        qkV9Kernel<<<
            qk_grid,
            gemm_block
        >>>(
            d_Q,
            d_K,
            d_S,
            N,
            D
        );


        softmaxV1Kernel<<<
            softmax_grid,
            softmax_block
        >>>(
            d_S,
            d_P,
            N
        );


        pvV9Kernel<<<
            pv_grid,
            gemm_block
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

        qkV9Kernel<<<
            qk_grid,
            gemm_block
        >>>(
            d_Q,
            d_K,
            d_S,
            N,
            D
        );


        softmaxV1Kernel<<<
            softmax_grid,
            softmax_block
        >>>(
            d_S,
            d_P,
            N
        );


        pvV9Kernel<<<
            pv_grid,
            gemm_block
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
    // QK and PV matmul FLOPs:
    //
    // QK:
    //
    //      2*N*N*D
    //
    // PV:
    //
    //      2*N*N*D
    //
    // total:
    //
    //      4*N*N*D
    //
    // 不包括 Softmax FLOPs
    // ========================================================

    const double matmul_flops =
        4.0
        *
        static_cast<double>(N)
        *
        static_cast<double>(N)
        *
        static_cast<double>(D);


    const double effective_gflops =
        matmul_flops
        /
        (
            avg_ms
            *
            1.0e6
        );


    // ========================================================
    // Copy O
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
        << "GEMM Block Tile = "
        << BLOCK_M
        << " x "
        << BLOCK_N
        << '\n';


    std::cout
        << "Warp Tile = "
        << WARP_M
        << " x "
        << WARP_N
        << '\n';


    std::cout
        << "Thread Tile = "
        << THREAD_M
        << " x "
        << THREAD_N
        << '\n';


    std::cout
        << "Average Attention Time = "
        << avg_ms
        << " ms\n";


    std::cout
        << "Matmul-only Effective Performance = "
        << effective_gflops
        << " GFLOPS\n";


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