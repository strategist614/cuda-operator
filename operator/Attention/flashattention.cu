#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>


#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr                                         \
                << "CUDA error: "                             \
                << cudaGetErrorString(err)                    \
                << " at " << __FILE__                         \
                << ":" << __LINE__                            \
                << std::endl;                                 \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)


// ============================================================
// FlashAttention configuration
//
// Teaching + optimization version for:
//
//     sm_75 / Turing
//
// Fixed:
//
//     head_dim = 64
//
// Block:
//
//     16 query rows
//     32 key/value rows
//
// Threads:
//
//     256 = 8 warps
//
// Warp:
//
//     2 query rows
//     32 keys
//
// Each lane:
//
//     QK:
//         2 query rows x 1 key
//
//     PV:
//         2 query rows x 2 output dims
//
// ============================================================

constexpr int HEAD_DIM = 64;

constexpr int BLOCK_Q  = 16;
constexpr int BLOCK_KV = 32;

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 8;

constexpr int THREADS =
    NUM_WARPS * WARP_SIZE;


// ============================================================
// float4
// ============================================================

constexpr int VEC = 4;

constexpr int VECS_PER_ROW =
    HEAD_DIM / VEC;          // 16


// ============================================================
// Shared memory padding
// ============================================================

constexpr int Q_STRIDE =
    BLOCK_Q + 1;             // 17

constexpr int K_STRIDE =
    BLOCK_KV + 1;            // 33

constexpr int W_STRIDE =
    BLOCK_KV + 1;            // 33


// ============================================================
// Every thread loads:
//
// K tile:
// 32 * 64 floats
// = 2048 floats
// = 512 float4
//
// 512 / 256
// = 2 float4 per thread
// ============================================================

constexpr int KV_TOTAL_VECS =
    BLOCK_KV * VECS_PER_ROW;

constexpr int KV_VECS_PER_THREAD =
    KV_TOTAL_VECS / THREADS;


// ============================================================
// Warp reduction
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
// FlashAttention Forward
//
// Q, K, V:
//     [B, H, N, D]
//
// O:
//     [B, H, N, D]
//
// LSE:
//     [B, H, N]
//
// ============================================================

template <bool CAUSAL>
__global__
__launch_bounds__(THREADS, 1)
void flashAttentionForward(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ LSE,
    int B,
    int H,
    int N)
{
    const int tid =
        threadIdx.x;

    const int warp_id =
        tid / WARP_SIZE;

    const int lane_id =
        tid % WARP_SIZE;


    // ========================================================
    // Which batch/head/query block?
    // ========================================================

    const int q_block =
        blockIdx.x;

    const int head =
        blockIdx.y;

    const int batch =
        blockIdx.z;


    const int q_start =
        q_block * BLOCK_Q;


    if (
        q_start >= N
    ) {
        return;
    }


    // ========================================================
    // Base pointer
    //
    // [B,H,N,D]
    // ========================================================

    const size_t bh_offset =
        (
            static_cast<size_t>(batch)
            *
            H
            +
            head
        )
        *
        N
        *
        HEAD_DIM;


    const float* Qbh =
        Q + bh_offset;

    const float* Kbh =
        K + bh_offset;

    const float* Vbh =
        V + bh_offset;

    float* Obh =
        O + bh_offset;


    const size_t lse_offset =
        (
            static_cast<size_t>(batch)
            *
            H
            +
            head
        )
        *
        N;


    float* LSEbh =
        LSE + lse_offset;


    // ========================================================
    // Shared Q
    //
    // Global:
    //
    //     Q[q][d]
    //
    // Shared:
    //
    //     QsT[d][q]
    //
    // [64][17]
    // ========================================================

    __shared__ __align__(16)
    float QsT[
        HEAD_DIM
    ][
        Q_STRIDE
    ];


    // ========================================================
    // K Double Buffer
    //
    // Global:
    //
    //     K[key][d]
    //
    // Shared:
    //
    //     KsT[buffer][d][key]
    //
    // ========================================================

    __shared__ __align__(16)
    float KsT[
        2
    ][
        HEAD_DIM
    ][
        K_STRIDE
    ];


    // ========================================================
    // V Double Buffer
    //
    // V stays row-major
    //
    // Vs[buffer][key][d]
    // ========================================================

    __shared__ __align__(16)
    float Vs[
        2
    ][
        BLOCK_KV
    ][
        HEAD_DIM
    ];


    // ========================================================
    // Softmax weights for current tile
    //
    // Only exists in Shared.
    //
    // Never Global.
    // ========================================================

    __shared__ __align__(16)
    float Ws[
        BLOCK_Q
    ][
        W_STRIDE
    ];


    // ========================================================
    // STEP 1
    //
    // Load Q Tile
    //
    // BLOCK_Q x D
    //
    // 16 x 64
    //
    // = 1024 floats
    // = 256 float4
    //
    // exactly one float4/thread
    // ========================================================

    const int q_local =
        tid / VECS_PER_ROW;

    const int q_vec =
        tid % VECS_PER_ROW;

    const int q_d =
        q_vec * VEC;


    const int global_q =
        q_start + q_local;


    float4 q4 =
        make_float4(
            0.0f,
            0.0f,
            0.0f,
            0.0f
        );


    if (
        global_q < N
    ) {
        q4 =
            *reinterpret_cast<
                const float4*
            >(
                Qbh
                +
                global_q * HEAD_DIM
                +
                q_d
            );
    }


    // ========================================================
    // Global coalesced
    //
    // ->
    //
    // Shared transpose
    // ========================================================

    QsT[q_d + 0][q_local] =
        q4.x;

    QsT[q_d + 1][q_local] =
        q4.y;

    QsT[q_d + 2][q_local] =
        q4.z;

    QsT[q_d + 3][q_local] =
        q4.w;


    // ========================================================
    // STEP 2
    //
    // Load first K/V tile into buffer 0
    // ========================================================

#pragma unroll
    for (
        int iter = 0;
        iter < KV_VECS_PER_THREAD;
        ++iter
    ) {

        const int vec_index =
            tid
            +
            iter * THREADS;


        const int local_key =
            vec_index
            /
            VECS_PER_ROW;


        const int vec_id =
            vec_index
            %
            VECS_PER_ROW;


        const int d =
            vec_id * VEC;


        const int global_key =
            local_key;


        float4 k4 =
            make_float4(
                0.0f,
                0.0f,
                0.0f,
                0.0f
            );


        float4 v4 =
            make_float4(
                0.0f,
                0.0f,
                0.0f,
                0.0f
            );


        if (
            global_key < N
        ) {

            k4 =
                *reinterpret_cast<
                    const float4*
                >(
                    Kbh
                    +
                    global_key * HEAD_DIM
                    +
                    d
                );


            v4 =
                *reinterpret_cast<
                    const float4*
                >(
                    Vbh
                    +
                    global_key * HEAD_DIM
                    +
                    d
                );
        }


        // ====================================================
        // K transpose into shared
        // ====================================================

        KsT[0][d + 0][local_key] =
            k4.x;

        KsT[0][d + 1][local_key] =
            k4.y;

        KsT[0][d + 2][local_key] =
            k4.z;

        KsT[0][d + 3][local_key] =
            k4.w;


        // ====================================================
        // V row-major
        // ====================================================

        Vs[0][local_key][d + 0] =
            v4.x;

        Vs[0][local_key][d + 1] =
            v4.y;

        Vs[0][local_key][d + 2] =
            v4.z;

        Vs[0][local_key][d + 3] =
            v4.w;
    }


    __syncthreads();


    // ========================================================
    // Warp owns:
    //
    // 2 Query rows
    //
    // warp0 -> q0,q1
    // warp1 -> q2,q3
    // ...
    // ========================================================

    const int warp_q0 =
        warp_id * 2;

    const int warp_q1 =
        warp_q0 + 1;


    const int global_q0 =
        q_start + warp_q0;

    const int global_q1 =
        q_start + warp_q1;


    const bool q0_valid =
        global_q0 < N;

    const bool q1_valid =
        global_q1 < N;


    // ========================================================
    // Each lane owns one key
    //
    // lane0  -> key0
    // lane1  -> key1
    // ...
    // lane31 -> key31
    // ========================================================

    const int local_key =
        lane_id;


    // ========================================================
    // Each lane also owns 2 output dimensions
    //
    // lane0:
    //      d0
    //      d32
    //
    // lane1:
    //      d1
    //      d33
    //
    // ...
    // ========================================================

    const int out_d0 =
        lane_id;

    const int out_d1 =
        lane_id + 32;


    // ========================================================
    // Online Softmax state
    //
    // kept in registers
    // ========================================================

    float m0 =
        -INFINITY;

    float l0 =
        0.0f;


    float m1 =
        -INFINITY;

    float l1 =
        0.0f;


    // ========================================================
    // Output accumulators
    //
    //             d0      d1
    //
    // q0          o00     o01
    //
    // q1          o10     o11
    //
    // kept in registers for entire sequence
    // ========================================================

    float o00 = 0.0f;
    float o01 = 0.0f;

    float o10 = 0.0f;
    float o11 = 0.0f;


    const float scale =
        rsqrtf(
            static_cast<float>(
                HEAD_DIM
            )
        );


    // ========================================================
    // Number of K/V tiles
    // ========================================================

    const int num_tiles =
        (
            N
            +
            BLOCK_KV
            -
            1
        )
        /
        BLOCK_KV;


    // ========================================================
    // Main FlashAttention Loop
    // ========================================================

    for (
        int tile = 0;
        tile < num_tiles;
        ++tile
    ) {

        const int current_buffer =
            tile & 1;


        const int next_buffer =
            current_buffer ^ 1;


        const int key_start =
            tile * BLOCK_KV;


        const int next_key_start =
            key_start
            +
            BLOCK_KV;


        const bool has_next =
            tile + 1 < num_tiles;


        // ====================================================
        // SOFTWARE PREFETCH
        //
        // Load NEXT K/V tile
        //
        // Global -> Register
        //
        // while current Shared tile still exists.
        //
        // On sm75 there is no cp.async.
        // ====================================================

        float4 next_k[
            KV_VECS_PER_THREAD
        ];


        float4 next_v[
            KV_VECS_PER_THREAD
        ];


#pragma unroll
        for (
            int iter = 0;
            iter < KV_VECS_PER_THREAD;
            ++iter
        ) {

            next_k[iter] =
                make_float4(
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f
                );


            next_v[iter] =
                make_float4(
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f
                );


            if (
                has_next
            ) {

                const int vec_index =
                    tid
                    +
                    iter * THREADS;


                const int next_local_key =
                    vec_index
                    /
                    VECS_PER_ROW;


                const int vec_id =
                    vec_index
                    %
                    VECS_PER_ROW;


                const int d =
                    vec_id * VEC;


                const int global_key =
                    next_key_start
                    +
                    next_local_key;


                if (
                    global_key < N
                ) {

                    next_k[iter] =
                        *reinterpret_cast<
                            const float4*
                        >(
                            Kbh
                            +
                            global_key
                                *
                                HEAD_DIM
                            +
                            d
                        );


                    next_v[iter] =
                        *reinterpret_cast<
                            const float4*
                        >(
                            Vbh
                            +
                            global_key
                                *
                                HEAD_DIM
                            +
                            d
                        );
                }
            }
        }


        // ====================================================
        // PART A
        //
        // Q Tile x K Tile^T
        //
        // Warp:
        //
        // 2 Query x 32 Keys
        //
        // Thread:
        //
        // 2 Query x 1 Key
        // ====================================================

        float score0 =
            0.0f;

        float score1 =
            0.0f;


#pragma unroll 4
        for (
            int d = 0;
            d < HEAD_DIM;
            ++d
        ) {

            const float q0 =
                QsT[
                    d
                ][
                    warp_q0
                ];


            const float q1 =
                QsT[
                    d
                ][
                    warp_q1
                ];


            const float kval =
                KsT[
                    current_buffer
                ][
                    d
                ][
                    local_key
                ];


            score0 +=
                q0 * kval;


            score1 +=
                q1 * kval;
        }


        const int global_key =
            key_start
            +
            local_key;


        // ====================================================
        // Validity + causal mask
        // ====================================================

        const bool valid0 =
            q0_valid
            &&
            global_key < N
            &&
            (
                !CAUSAL
                ||
                global_key <= global_q0
            );


        const bool valid1 =
            q1_valid
            &&
            global_key < N
            &&
            (
                !CAUSAL
                ||
                global_key <= global_q1
            );


        if (
            valid0
        ) {
            score0 *= scale;
        }
        else {
            score0 =
                -INFINITY;
        }


        if (
            valid1
        ) {
            score1 *= scale;
        }
        else {
            score1 =
                -INFINITY;
        }


        // ====================================================
        // PART B
        //
        // Current tile row max
        // ====================================================

        float tile_m0 =
            warpReduceMax(
                score0
            );


        float tile_m1 =
            warpReduceMax(
                score1
            );


        // ====================================================
        // reduction result is in lane0
        //
        // broadcast to all lanes
        // ====================================================

        tile_m0 =
            __shfl_sync(
                0xffffffff,
                tile_m0,
                0
            );


        tile_m1 =
            __shfl_sync(
                0xffffffff,
                tile_m1,
                0
            );


        // ====================================================
        // Online max
        //
        // m_new =
        //
        // max(m_old, tile_max)
        // ====================================================

        const float new_m0 =
            q0_valid
            ?
            fmaxf(
                m0,
                tile_m0
            )
            :
            -INFINITY;


        const float new_m1 =
            q1_valid
            ?
            fmaxf(
                m1,
                tile_m1
            )
            :
            -INFINITY;


        // ====================================================
        // Rescale old state
        //
        // alpha =
        //
        // exp(m_old - m_new)
        // ====================================================

        const float alpha0 =
            (
                !q0_valid
                ||
                tile == 0
            )
            ?
            0.0f
            :
            expf(
                m0
                -
                new_m0
            );


        const float alpha1 =
            (
                !q1_valid
                ||
                tile == 0
            )
            ?
            0.0f
            :
            expf(
                m1
                -
                new_m1
            );


        // ====================================================
        // Current tile unnormalized weights
        //
        // exp(score - new_m)
        // ====================================================

        const float weight0 =
            valid0
            ?
            expf(
                score0
                -
                new_m0
            )
            :
            0.0f;


        const float weight1 =
            valid1
            ?
            expf(
                score1
                -
                new_m1
            )
            :
            0.0f;


        // ====================================================
        // Tile denominator
        // ====================================================

        float tile_l0 =
            warpReduceSum(
                weight0
            );


        float tile_l1 =
            warpReduceSum(
                weight1
            );


        tile_l0 =
            __shfl_sync(
                0xffffffff,
                tile_l0,
                0
            );


        tile_l1 =
            __shfl_sync(
                0xffffffff,
                tile_l1,
                0
            );


        // ====================================================
        // Online denominator update
        //
        // l_new =
        //
        // alpha * l_old
        //
        // +
        //
        // tile_sum
        // ====================================================

        const float new_l0 =
            alpha0 * l0
            +
            tile_l0;


        const float new_l1 =
            alpha1 * l1
            +
            tile_l1;


        // ====================================================
        // Store current tile P into Shared
        //
        // P never goes to Global Memory
        // ====================================================

        Ws[
            warp_q0
        ][
            local_key
        ] =
            weight0;


        Ws[
            warp_q1
        ][
            local_key
        ] =
            weight1;


        // ====================================================
        // Each warp owns its own two rows
        // ====================================================

        __syncwarp();


        // ====================================================
        // PART C
        //
        // P Tile x V Tile
        //
        // Output warp tile:
        //
        // 2 x 64
        //
        // Thread:
        //
        // 2 Query x 2 dimensions
        // ====================================================

        float tile_o00 =
            0.0f;

        float tile_o01 =
            0.0f;


        float tile_o10 =
            0.0f;

        float tile_o11 =
            0.0f;


#pragma unroll 4
        for (
            int k = 0;
            k < BLOCK_KV;
            ++k
        ) {

            // ================================================
            // Same P weight broadcast
            // across warp
            // ================================================

            const float p0 =
                Ws[
                    warp_q0
                ][
                    k
                ];


            const float p1 =
                Ws[
                    warp_q1
                ][
                    k
                ];


            // ================================================
            // V
            //
            // lanes 0~31:
            //
            // d0 contiguous
            //
            // d32~63 contiguous
            // ================================================

            const float v0 =
                Vs[
                    current_buffer
                ][
                    k
                ][
                    out_d0
                ];


            const float v1 =
                Vs[
                    current_buffer
                ][
                    k
                ][
                    out_d1
                ];


            // ================================================
            // Register outer product
            // ================================================

            tile_o00 +=
                p0 * v0;

            tile_o01 +=
                p0 * v1;


            tile_o10 +=
                p1 * v0;

            tile_o11 +=
                p1 * v1;
        }


        // ====================================================
        // Online output update
        //
        // O_new =
        //
        // alpha * O_old
        //
        // +
        //
        // P_tile V_tile
        // ====================================================

        o00 =
            alpha0 * o00
            +
            tile_o00;


        o01 =
            alpha0 * o01
            +
            tile_o01;


        o10 =
            alpha1 * o10
            +
            tile_o10;


        o11 =
            alpha1 * o11
            +
            tile_o11;


        // ====================================================
        // Update online softmax state
        // ====================================================

        m0 =
            new_m0;

        l0 =
            new_l0;


        m1 =
            new_m1;

        l1 =
            new_l1;


        // ====================================================
        // PART D
        //
        // Register-prefetched NEXT K/V
        //
        // ->
        //
        // inactive Shared buffer
        //
        // This is the software double buffer.
        // ====================================================

        if (
            has_next
        ) {

#pragma unroll
            for (
                int iter = 0;
                iter < KV_VECS_PER_THREAD;
                ++iter
            ) {

                const int vec_index =
                    tid
                    +
                    iter * THREADS;


                const int next_local_key =
                    vec_index
                    /
                    VECS_PER_ROW;


                const int vec_id =
                    vec_index
                    %
                    VECS_PER_ROW;


                const int d =
                    vec_id * VEC;


                const float4 k4 =
                    next_k[iter];


                const float4 v4 =
                    next_v[iter];


                // ============================================
                // K:
                //
                // Register -> Shared transpose
                // ============================================

                KsT[
                    next_buffer
                ][
                    d + 0
                ][
                    next_local_key
                ] =
                    k4.x;


                KsT[
                    next_buffer
                ][
                    d + 1
                ][
                    next_local_key
                ] =
                    k4.y;


                KsT[
                    next_buffer
                ][
                    d + 2
                ][
                    next_local_key
                ] =
                    k4.z;


                KsT[
                    next_buffer
                ][
                    d + 3
                ][
                    next_local_key
                ] =
                    k4.w;


                // ============================================
                // V
                // ============================================

                Vs[
                    next_buffer
                ][
                    next_local_key
                ][
                    d + 0
                ] =
                    v4.x;


                Vs[
                    next_buffer
                ][
                    next_local_key
                ][
                    d + 1
                ] =
                    v4.y;


                Vs[
                    next_buffer
                ][
                    next_local_key
                ][
                    d + 2
                ] =
                    v4.z;


                Vs[
                    next_buffer
                ][
                    next_local_key
                ][
                    d + 3
                ] =
                    v4.w;
            }


            // ================================================
            // Next Shared tile must be fully ready
            // before next iteration.
            // ================================================

            __syncthreads();
        }
    }


    // ========================================================
    // Final normalization
    //
    // O =
    //
    // numerator / denominator
    // ========================================================

    if (
        q0_valid
    ) {

        const float inv_l0 =
            1.0f / l0;


        Obh[
            global_q0 * HEAD_DIM
            +
            out_d0
        ] =
            o00 * inv_l0;


        Obh[
            global_q0 * HEAD_DIM
            +
            out_d1
        ] =
            o01 * inv_l0;


        // ====================================================
        // Save LogSumExp
        //
        // only lane0
        //
        // useful for backward later
        // ====================================================

        if (
            lane_id == 0
        ) {

            LSEbh[
                global_q0
            ] =
                m0
                +
                logf(l0);
        }
    }


    if (
        q1_valid
    ) {

        const float inv_l1 =
            1.0f / l1;


        Obh[
            global_q1 * HEAD_DIM
            +
            out_d0
        ] =
            o10 * inv_l1;


        Obh[
            global_q1 * HEAD_DIM
            +
            out_d1
        ] =
            o11 * inv_l1;


        if (
            lane_id == 0
        ) {

            LSEbh[
                global_q1
            ] =
                m1
                +
                logf(l1);
        }
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
    int B,
    int H,
    int N,
    bool causal)
{
    const size_t total =
        static_cast<size_t>(B)
        *
        H
        *
        N
        *
        HEAD_DIM;


    O.assign(
        total,
        0.0f
    );


    const float scale =
        1.0f
        /
        std::sqrt(
            static_cast<float>(
                HEAD_DIM
            )
        );


    for (
        int b = 0;
        b < B;
        ++b
    ) {

        for (
            int h = 0;
            h < H;
            ++h
        ) {

            const size_t base =
                (
                    static_cast<size_t>(b)
                    *
                    H
                    +
                    h
                )
                *
                N
                *
                HEAD_DIM;


            std::vector<float>
                scores(N);


            std::vector<float>
                probs(N);


            for (
                int q = 0;
                q < N;
                ++q
            ) {

                // ============================================
                // Q K^T
                // ============================================

                for (
                    int k = 0;
                    k < N;
                    ++k
                ) {

                    if (
                        causal
                        &&
                        k > q
                    ) {

                        scores[k] =
                            -INFINITY;

                        continue;
                    }


                    float dot =
                        0.0f;


                    for (
                        int d = 0;
                        d < HEAD_DIM;
                        ++d
                    ) {

                        dot +=
                            Q[
                                base
                                +
                                q
                                    *
                                    HEAD_DIM
                                +
                                d
                            ]
                            *
                            K[
                                base
                                +
                                k
                                    *
                                    HEAD_DIM
                                +
                                d
                            ];
                    }


                    scores[k] =
                        dot
                        *
                        scale;
                }


                // ============================================
                // Softmax max
                // ============================================

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


                // ============================================
                // Softmax sum
                // ============================================

                float sum =
                    0.0f;


                for (
                    int k = 0;
                    k < N;
                    ++k
                ) {

                    if (
                        std::isinf(
                            scores[k]
                        )
                        &&
                        scores[k] < 0.0f
                    ) {

                        probs[k] =
                            0.0f;
                    }
                    else {

                        probs[k] =
                            std::exp(
                                scores[k]
                                -
                                max_value
                            );
                    }


                    sum +=
                        probs[k];
                }


                // ============================================
                // PV
                // ============================================

                for (
                    int d = 0;
                    d < HEAD_DIM;
                    ++d
                ) {

                    float result =
                        0.0f;


                    for (
                        int k = 0;
                        k < N;
                        ++k
                    ) {

                        result +=
                            (
                                probs[k]
                                /
                                sum
                            )
                            *
                            V[
                                base
                                +
                                k
                                    *
                                    HEAD_DIM
                                +
                                d
                            ];
                    }


                    O[
                        base
                        +
                        q
                            *
                            HEAD_DIM
                        +
                        d
                    ] =
                        result;
                }
            }
        }
    }
}


// ============================================================
// Main
// ============================================================

int main()
{
    // ========================================================
    // Change this:
    //
    // false = normal attention
    // true  = causal attention
    // ========================================================

    constexpr bool CAUSAL =
        false;


    constexpr int B =
        1;


    constexpr int H =
        2;


    constexpr int N =
        256;


    const size_t elements =
        static_cast<size_t>(B)
        *
        H
        *
        N
        *
        HEAD_DIM;


    const size_t bytes =
        elements
        *
        sizeof(float);


    const size_t lse_elements =
        static_cast<size_t>(B)
        *
        H
        *
        N;


    // ========================================================
    // Host
    // ========================================================

    std::vector<float>
        h_Q(elements);


    std::vector<float>
        h_K(elements);


    std::vector<float>
        h_V(elements);


    std::vector<float>
        h_O(elements);


    std::vector<float>
        h_ref;


    // ========================================================
    // Initialize
    // ========================================================

    for (
        size_t i = 0;
        i < elements;
        ++i
    ) {

        h_Q[i] =
            static_cast<float>(
                static_cast<int>(
                    i % 17
                )
                -
                8
            )
            /
            16.0f;


        h_K[i] =
            static_cast<float>(
                static_cast<int>(
                    i % 13
                )
                -
                6
            )
            /
            16.0f;


        h_V[i] =
            static_cast<float>(
                static_cast<int>(
                    i % 11
                )
                -
                5
            )
            /
            8.0f;
    }


    // ========================================================
    // Device
    // ========================================================

    float* d_Q =
        nullptr;

    float* d_K =
        nullptr;

    float* d_V =
        nullptr;

    float* d_O =
        nullptr;

    float* d_LSE =
        nullptr;


    CUDA_CHECK(
        cudaMalloc(
            &d_Q,
            bytes
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_K,
            bytes
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_V,
            bytes
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_O,
            bytes
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_LSE,
            lse_elements
                *
                sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_Q,
            h_Q.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_K,
            h_K.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_V,
            h_V.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );


    // ========================================================
    // Grid
    //
    // x:
    // query tiles
    //
    // y:
    // heads
    //
    // z:
    // batch
    // ========================================================

    dim3 block(
        THREADS
    );


    dim3 grid(
        (
            N
            +
            BLOCK_Q
            -
            1
        )
        /
        BLOCK_Q,

        H,

        B
    );


    // ========================================================
    // Warmup
    // ========================================================

    for (
        int i = 0;
        i < 10;
        ++i
    ) {

        flashAttentionForward<
            CAUSAL
        ><<<
            grid,
            block
        >>>(
            d_Q,
            d_K,
            d_V,
            d_O,
            d_LSE,
            B,
            H,
            N
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

        flashAttentionForward<
            CAUSAL
        ><<<
            grid,
            block
        >>>(
            d_Q,
            d_K,
            d_V,
            d_O,
            d_LSE,
            B,
            H,
            N
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
            bytes,
            cudaMemcpyDeviceToHost
        )
    );


    // ========================================================
    // CPU reference
    // ========================================================

    attentionCPU(
        h_Q,
        h_K,
        h_V,
        h_ref,
        B,
        H,
        N,
        CAUSAL
    );


    // ========================================================
    // Verify
    // ========================================================

    float max_error =
        0.0f;


    for (
        size_t i = 0;
        i < elements;
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
    // GPU info
    // ========================================================

    cudaDeviceProp prop;


    CUDA_CHECK(
        cudaGetDeviceProperties(
            &prop,
            0
        )
    );


    // ========================================================
    // Output
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
        << "B = "
        << B
        << '\n';


    std::cout
        << "H = "
        << H
        << '\n';


    std::cout
        << "N = "
        << N
        << '\n';


    std::cout
        << "D = "
        << HEAD_DIM
        << '\n';


    std::cout
        << "Causal = "
        << (
            CAUSAL
            ?
            "true"
            :
            "false"
        )
        << '\n';


    std::cout
        << "Q Tile = "
        << BLOCK_Q
        << '\n';


    std::cout
        << "KV Tile = "
        << BLOCK_KV
        << '\n';


    std::cout
        << "Average Time = "
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
        d < 8;
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


    CUDA_CHECK(
        cudaFree(
            d_LSE
        )
    );


    return 0;
}