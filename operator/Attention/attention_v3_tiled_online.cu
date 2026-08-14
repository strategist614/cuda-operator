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
// Fixed head dimension for this teaching version
// ============================================================

constexpr int HEAD_DIM = 64;


// ============================================================
// Attention Tile
//
// One block:
//
//     16 Query rows
//
// ×
//
//     64 Key rows
//
// Score Block Tile:
//
//     16 x 64
// ============================================================

constexpr int BLOCK_Q = 16;
constexpr int BLOCK_KV = 64;


// ============================================================
// 256 threads
//
// 8 warps
// ============================================================

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 8;

constexpr int THREADS =
    NUM_WARPS * WARP_SIZE;


// ============================================================
// Warp Tile
//
// One warp:
//
//     2 Query rows
//
// ×
//
//     64 Key rows
//
// =
//
//     2 x 64 Score Tile
// ============================================================

constexpr int WARP_Q = 2;


// ============================================================
// Thread Tile
//
// One lane:
//
//     2 Query rows
//
// ×
//
//     2 Keys
//
// =
//
//     2 x 2
//
// 4 score accumulators
// ============================================================

constexpr int THREAD_Q = 2;
constexpr int THREAD_K = 2;


// ============================================================
// Padding
// ============================================================

constexpr int Q_STRIDE =
    BLOCK_Q + 1;        // 17

constexpr int K_STRIDE =
    BLOCK_KV + 1;       // 65

constexpr int W_STRIDE =
    BLOCK_KV + 1;       // 65


// ============================================================
// float4 constants
// ============================================================

constexpr int VEC = 4;

constexpr int VECS_PER_ROW =
    HEAD_DIM / VEC;     // 16


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
// Attention V3
//
// Features:
//
// 1. Block Tiling
// 2. Warp Tiling
// 3. Thread/Register Tiling
// 4. Shared Memory layout
// 5. float4 cooperative load
// 6. Warp Reduction
// 7. Online Softmax
// 8. QK + Softmax + PV Fusion
//
// No full S[N,N]
// No full P[N,N]
//
// ============================================================

__global__
void attentionV3Kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N)
{
    const int tid =
        threadIdx.x;

    const int warp_id =
        tid / WARP_SIZE;

    const int lane_id =
        tid % WARP_SIZE;


    // ========================================================
    // Block owns 16 query rows
    // ========================================================

    const int block_q_start =
        blockIdx.x * BLOCK_Q;


    // Entire block is outside N
    if (
        block_q_start >= N
    ) {
        return;
    }


    // ========================================================
    // Shared Q
    //
    // Logical:
    //
    //     [16][64]
    //
    // Shared transposed:
    //
    //     [64][17]
    //
    // QsT[d][q]
    // ========================================================

    __shared__ __align__(16)
    float QsT[
        HEAD_DIM
    ][
        Q_STRIDE
    ];


    // ========================================================
    // Shared K
    //
    // Global:
    //
    //     K[key][d]
    //
    // Shared transposed:
    //
    //     KsT[d][key]
    //
    // [64][65]
    // ========================================================

    __shared__ __align__(16)
    float KsT[
        HEAD_DIM
    ][
        K_STRIDE
    ];


    // ========================================================
    // Shared V
    //
    // Keep row-major:
    //
    //     Vs[key][d]
    //
    // [64][64]
    // ========================================================

    __shared__ __align__(16)
    float Vs[
        BLOCK_KV
    ][
        HEAD_DIM
    ];


    // ========================================================
    // Current tile softmax weights
    //
    // [16][65]
    //
    // Padding prevents ugly bank patterns.
    // ========================================================

    __shared__ __align__(16)
    float Ws[
        BLOCK_Q
    ][
        W_STRIDE
    ];


    // ========================================================
    // STEP 0
    //
    // Load Q Block Tile:
    //
    //     16 x 64
    //
    // = 1024 floats
    //
    // = 256 float4
    //
    // Exactly:
    //
    //     one float4 per thread
    // ========================================================

    const int q_local =
        tid / VECS_PER_ROW;

    const int q_vec =
        tid % VECS_PER_ROW;

    const int q_d =
        q_vec * VEC;

    const int global_q =
        block_q_start
        +
        q_local;


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
            *reinterpret_cast<const float4*>(
                Q
                +
                global_q * HEAD_DIM
                +
                q_d
            );
    }


    // ========================================================
    // Global:
    //
    // Q[q][d]
    //
    // ->
    //
    // Shared:
    //
    // QsT[d][q]
    // ========================================================

    QsT[q_d + 0][q_local] =
        q4.x;

    QsT[q_d + 1][q_local] =
        q4.y;

    QsT[q_d + 2][q_local] =
        q4.z;

    QsT[q_d + 3][q_local] =
        q4.w;


    __syncthreads();


    // ========================================================
    // Warp mapping
    //
    // warp0 -> query 0,1
    // warp1 -> query 2,3
    // ...
    // warp7 -> query 14,15
    // ========================================================

    const int warp_q0 =
        warp_id * WARP_Q;

    const int warp_q1 =
        warp_q0 + 1;


    const int global_q0 =
        block_q_start
        +
        warp_q0;

    const int global_q1 =
        block_q_start
        +
        warp_q1;


    const bool q0_valid =
        global_q0 < N;

    const bool q1_valid =
        global_q1 < N;


    // ========================================================
    // Thread Key mapping
    //
    // lane0:
    //
    //     key0
    //     key32
    //
    // lane1:
    //
    //     key1
    //     key33
    //
    // ...
    //
    // lane31:
    //
    //     key31
    //     key63
    //
    // ========================================================

    const int local_k0 =
        lane_id;

    const int local_k1 =
        lane_id + 32;


    // ========================================================
    // Output dimension mapping
    //
    // Same lane mapping is useful for PV:
    //
    // lane0 -> d0, d32
    // lane1 -> d1, d33
    //
    // etc.
    // ========================================================

    const int d0 =
        lane_id;

    const int d1 =
        lane_id + 32;


    // ========================================================
    // Online Softmax State
    //
    // One warp owns 2 query rows.
    //
    // Every lane keeps copies of:
    //
    // m0, l0
    // m1, l1
    //
    // in registers.
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
    // Output Numerator Accumulators
    //
    // Thread Tile:
    //
    //             d0      d1
    //
    // query0     o00     o01
    //
    // query1     o10     o11
    //
    //
    // All kept in registers across ALL K/V tiles.
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
    // Loop over sequence dimension
    //
    // N = 512:
    //
    // tile0 -> 0~63
    // tile1 -> 64~127
    // ...
    // ========================================================

    for (
        int key_start = 0;
        key_start < N;
        key_start += BLOCK_KV
    ) {

        // ====================================================
        // Global -> Shared K / V
        //
        // Each K tile:
        //
        //     64 x 64
        //
        // = 4096 floats
        //
        // = 1024 float4
        //
        // 256 threads:
        //
        //     each thread loads 4 float4
        // ====================================================

        constexpr int TOTAL_KV_VECS =
            BLOCK_KV
            *
            VECS_PER_ROW;


        for (
            int vec_index = tid;
            vec_index < TOTAL_KV_VECS;
            vec_index += THREADS
        ) {

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
                key_start
                +
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
                    *reinterpret_cast<const float4*>(
                        K
                        +
                        global_key * HEAD_DIM
                        +
                        d
                    );


                v4 =
                    *reinterpret_cast<const float4*>(
                        V
                        +
                        global_key * HEAD_DIM
                        +
                        d
                    );
            }


            // ================================================
            // K:
            //
            // Global row-major
            //
            // ->
            //
            // Shared transpose
            // ================================================

            KsT[d + 0][local_key] =
                k4.x;

            KsT[d + 1][local_key] =
                k4.y;

            KsT[d + 2][local_key] =
                k4.z;

            KsT[d + 3][local_key] =
                k4.w;


            // ================================================
            // V stays row-major
            // ================================================

            Vs[local_key][d + 0] =
                v4.x;

            Vs[local_key][d + 1] =
                v4.y;

            Vs[local_key][d + 2] =
                v4.z;

            Vs[local_key][d + 3] =
                v4.w;
        }


        // ====================================================
        // K/V Tile Ready
        // ====================================================

        __syncthreads();


        // ====================================================
        // PART A
        //
        // Q Tile x K Tile^T
        //
        //
        // Block Tile:
        //
        //     16 x 64
        //
        // Warp Tile:
        //
        //     2 x 64
        //
        // Thread Tile:
        //
        //     2 x 2
        // ====================================================

        float s00 =
            0.0f;

        float s01 =
            0.0f;

        float s10 =
            0.0f;

        float s11 =
            0.0f;


        // ====================================================
        // Reduction dimension = D = 64
        //
        // Every iteration:
        //
        // 2 Q registers
        // x
        // 2 K registers
        //
        // ->
        //
        // 4 FMA
        // ====================================================

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


            const float k0 =
                KsT[
                    d
                ][
                    local_k0
                ];


            const float k1 =
                KsT[
                    d
                ][
                    local_k1
                ];


            // ================================================
            // 2 x 2 Register Outer Product
            // ================================================

            s00 +=
                q0 * k0;

            s01 +=
                q0 * k1;

            s10 +=
                q1 * k0;

            s11 +=
                q1 * k1;
        }


        // ====================================================
        // Global key positions
        // ====================================================

        const int global_k0 =
            key_start
            +
            local_k0;


        const int global_k1 =
            key_start
            +
            local_k1;


        // ====================================================
        // Scale and mask out invalid positions
        // ====================================================

        if (
            q0_valid
            &&
            global_k0 < N
        ) {
            s00 *= scale;
        }
        else {
            s00 =
                -INFINITY;
        }


        if (
            q0_valid
            &&
            global_k1 < N
        ) {
            s01 *= scale;
        }
        else {
            s01 =
                -INFINITY;
        }


        if (
            q1_valid
            &&
            global_k0 < N
        ) {
            s10 *= scale;
        }
        else {
            s10 =
                -INFINITY;
        }


        if (
            q1_valid
            &&
            global_k1 < N
        ) {
            s11 *= scale;
        }
        else {
            s11 =
                -INFINITY;
        }


        // ====================================================
        // PART B
        //
        // Online Softmax
        //
        // Each lane has 2 scores per query.
        //
        // First find tile max.
        // ====================================================

        float lane_max0 =
            fmaxf(
                s00,
                s01
            );


        float lane_max1 =
            fmaxf(
                s10,
                s11
            );


        // ====================================================
        // Warp Reduction
        //
        // One warp owns the entire row:
        //
        // 32 lanes
        // x
        // 2 scores/lane
        //
        // = 64 scores
        // ====================================================

        float tile_m0 =
            warpReduceMax(
                lane_max0
            );


        float tile_m1 =
            warpReduceMax(
                lane_max1
            );


        // ====================================================
        // warpReduce result is complete in lane0.
        //
        // Broadcast lane0 -> all lanes.
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
        // Online max:
        //
        // m_new = max(m_old, m_tile)
        // ====================================================

        const float new_m0 =
            fmaxf(
                m0,
                tile_m0
            );


        const float new_m1 =
            fmaxf(
                m1,
                tile_m1
            );


        // ====================================================
        // Rescale old state:
        //
        // alpha =
        //
        // exp(m_old - m_new)
        //
        // first tile:
        //
        // m_old = -inf
        //
        // alpha = 0
        // ====================================================

        const float alpha0 =
            (
                m0 == -INFINITY
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
                m1 == -INFINITY
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
        // Current tile weights:
        //
        // exp(score - new_m)
        // ====================================================

        float w00 =
            0.0f;

        float w01 =
            0.0f;

        float w10 =
            0.0f;

        float w11 =
            0.0f;


        if (
            q0_valid
            &&
            global_k0 < N
        ) {
            w00 =
                expf(
                    s00
                    -
                    new_m0
                );
        }


        if (
            q0_valid
            &&
            global_k1 < N
        ) {
            w01 =
                expf(
                    s01
                    -
                    new_m0
                );
        }


        if (
            q1_valid
            &&
            global_k0 < N
        ) {
            w10 =
                expf(
                    s10
                    -
                    new_m1
                );
        }


        if (
            q1_valid
            &&
            global_k1 < N
        ) {
            w11 =
                expf(
                    s11
                    -
                    new_m1
                );
        }


        // ====================================================
        // Tile denominator
        //
        // One lane:
        //
        // two weights
        // ====================================================

        float lane_sum0 =
            w00 + w01;


        float lane_sum1 =
            w10 + w11;


        float tile_l0 =
            warpReduceSum(
                lane_sum0
            );


        float tile_l1 =
            warpReduceSum(
                lane_sum1
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
        // Online denominator update:
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

        const float new_l0 =
            alpha0 * l0
            +
            tile_l0;


        const float new_l1 =
            alpha1 * l1
            +
            tile_l1;


        // ====================================================
        // Store current weights in Shared.
        //
        // Each lane writes:
        //
        // query0,key0
        // query0,key1
        // query1,key0
        // query1,key1
        // ====================================================

        Ws[
            warp_q0
        ][
            local_k0
        ] =
            w00;


        Ws[
            warp_q0
        ][
            local_k1
        ] =
            w01;


        Ws[
            warp_q1
        ][
            local_k0
        ] =
            w10;


        Ws[
            warp_q1
        ][
            local_k1
        ] =
            w11;


        // ====================================================
        // Each warp only reads the two rows it just wrote.
        //
        // warp sync is enough for W.
        // ====================================================

        __syncwarp();


        // ====================================================
        // PART C
        //
        // P_tile x V_tile
        //
        // But P_tile never exists in Global Memory.
        //
        //
        // Output Block Tile:
        //
        //     16 x 64
        //
        // Warp:
        //
        //     2 x 64
        //
        // Thread:
        //
        //     2 x 2
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
            // Two softmax weights
            //
            // Same weight is broadcast to all lanes in warp.
            // ================================================

            const float pw0 =
                Ws[
                    warp_q0
                ][
                    k
                ];


            const float pw1 =
                Ws[
                    warp_q1
                ][
                    k
                ];


            // ================================================
            // Two V dimensions
            //
            // Across lanes:
            //
            // d0 = 0..31
            //
            // contiguous shared access
            //
            // d1 = 32..63
            //
            // contiguous shared access
            // ================================================

            const float v0 =
                Vs[
                    k
                ][
                    d0
                ];


            const float v1 =
                Vs[
                    k
                ][
                    d1
                ];


            // ================================================
            // 2 x 2 outer product again
            // ================================================

            tile_o00 +=
                pw0 * v0;

            tile_o01 +=
                pw0 * v1;

            tile_o10 +=
                pw1 * v0;

            tile_o11 +=
                pw1 * v1;
        }


        // ====================================================
        // Online Output update:
        //
        // O_acc_new
        //
        // =
        //
        // alpha * O_acc_old
        //
        // +
        //
        // current_tile_weighted_V
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
        // Update Online Softmax state
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
        // IMPORTANT
        //
        // Before ANY warp/thread starts overwriting KsT / Vs
        // for next tile, every warp must finish using current V.
        // ====================================================

        __syncthreads();
    }


    // ========================================================
    // All K/V tiles complete.
    //
    // Final:
    //
    // O = numerator / denominator
    // ========================================================

    if (
        q0_valid
    ) {

        O[
            global_q0 * HEAD_DIM
            +
            d0
        ] =
            o00 / l0;


        O[
            global_q0 * HEAD_DIM
            +
            d1
        ] =
            o01 / l0;
    }


    if (
        q1_valid
    ) {

        O[
            global_q1 * HEAD_DIM
            +
            d0
        ] =
            o10 / l1;


        O[
            global_q1 * HEAD_DIM
            +
            d1
        ] =
            o11 / l1;
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
    int N)
{
    O.resize(
        N * HEAD_DIM
    );


    const float scale =
        1.0f
        /
        std::sqrt(
            static_cast<float>(
                HEAD_DIM
            )
        );


    std::vector<float>
        scores(N);


    std::vector<float>
        probs(N);


    for (
        int q = 0;
        q < N;
        ++q
    ) {

        // ====================================================
        // Q K^T
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
                d < HEAD_DIM;
                ++d
            ) {

                dot +=
                    Q[
                        q * HEAD_DIM
                        +
                        d
                    ]
                    *
                    K[
                        k * HEAD_DIM
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
            d < HEAD_DIM;
            ++d
        ) {

            float out =
                0.0f;


            for (
                int k = 0;
                k < N;
                ++k
            ) {

                out +=
                    probs[k]
                    *
                    V[
                        k * HEAD_DIM
                        +
                        d
                    ];
            }


            O[
                q * HEAD_DIM
                +
                d
            ] =
                out;
        }
    }
}


// ============================================================
// Main
// ============================================================

int main()
{
    constexpr int N =
        512;


    // ========================================================
    // Host
    // ========================================================

    std::vector<float>
        h_Q(
            N * HEAD_DIM
        );


    std::vector<float>
        h_K(
            N * HEAD_DIM
        );


    std::vector<float>
        h_V(
            N * HEAD_DIM
        );


    std::vector<float>
        h_O(
            N * HEAD_DIM
        );


    // ========================================================
    // Init
    // ========================================================

    for (
        int i = 0;
        i < N * HEAD_DIM;
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
    //
    // NOTICE:
    //
    // No d_S
    // No d_P
    // ========================================================

    float* d_Q =
        nullptr;

    float* d_K =
        nullptr;

    float* d_V =
        nullptr;

    float* d_O =
        nullptr;


    const size_t bytes =
        static_cast<size_t>(N)
        *
        HEAD_DIM
        *
        sizeof(float);


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
    // One block = 16 query rows
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
        BLOCK_Q
    );


    // ========================================================
    // Warmup
    // ========================================================

    for (
        int i = 0;
        i < 10;
        ++i
    ) {

        attentionV3Kernel<<<
            grid,
            block
        >>>(
            d_Q,
            d_K,
            d_V,
            d_O,
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

        attentionV3Kernel<<<
            grid,
            block
        >>>(
            d_Q,
            d_K,
            d_V,
            d_O,
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
    // Copy output
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

    std::vector<float>
        h_ref;


    attentionCPU(
        h_Q,
        h_K,
        h_V,
        h_ref,
        N
    );


    // ========================================================
    // Verify
    // ========================================================

    float max_error =
        0.0f;


    for (
        int i = 0;
        i < N * HEAD_DIM;
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
        << "Head Dim = "
        << HEAD_DIM
        << '\n';


    std::cout
        << "Score Block Tile = "
        << BLOCK_Q
        << " x "
        << BLOCK_KV
        << '\n';


    std::cout
        << "Score Warp Tile = "
        << WARP_Q
        << " x "
        << BLOCK_KV
        << '\n';


    std::cout
        << "Thread Tile = "
        << THREAD_Q
        << " x "
        << THREAD_K
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


    return 0;
}