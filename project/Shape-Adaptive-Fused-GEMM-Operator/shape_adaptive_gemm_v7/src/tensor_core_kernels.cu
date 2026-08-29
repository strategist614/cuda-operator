#include "tensor_core.h"
#include "common.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>
#include <vector>

using namespace nvcuda;

__device__ __forceinline__
int4 tc_load_half8(
    const half* ptr
) {
    return
        *reinterpret_cast<
            const int4*
        >(ptr);
}

template<
    int BM,
    int BN,
    int BK,
    int THREADS,
    int A_CHUNKS,
    int B_CHUNKS
>
__device__ __forceinline__
void tc_prefetch_tile(
    const half* A,
    const half* B,
    int4 (&a_prefetch)[A_CHUNKS],
    int4 (&b_prefetch)[B_CHUNKS],
    int tid,
    int block_row,
    int block_col,
    int bk,
    int K,
    int N
) {
    constexpr int VEC_ELEMS =
        8;

    constexpr int A_VEC_COLS =
        BK / VEC_ELEMS;

    constexpr int B_VEC_COLS =
        BN / VEC_ELEMS;

    constexpr int A_VECS =
        BM * A_VEC_COLS;

    constexpr int B_VECS =
        BK * B_VEC_COLS;

#pragma unroll
    for (
        int q = 0;
        q < A_CHUNKS;
        ++q
    ) {
        const int idx =
            tid
            +
            q * THREADS;

        if (
            idx < A_VECS
        ) {
            const int r =
                idx
                /
                A_VEC_COLS;

            const int c8 =
                idx
                %
                A_VEC_COLS;

            const half* ptr =
                A
                +
                static_cast<size_t>(
                    block_row + r
                )
                *
                K
                +
                bk
                +
                c8 * VEC_ELEMS;

            a_prefetch[q] =
                tc_load_half8(
                    ptr
                );
        }
    }

#pragma unroll
    for (
        int q = 0;
        q < B_CHUNKS;
        ++q
    ) {
        const int idx =
            tid
            +
            q * THREADS;

        if (
            idx < B_VECS
        ) {
            const int r =
                idx
                /
                B_VEC_COLS;

            const int c8 =
                idx
                %
                B_VEC_COLS;

            const half* ptr =
                B
                +
                static_cast<size_t>(
                    bk + r
                )
                *
                N
                +
                block_col
                +
                c8 * VEC_ELEMS;

            b_prefetch[q] =
                tc_load_half8(
                    ptr
                );
        }
    }
}

template<
    int BM,
    int BN,
    int BK,
    int THREADS,
    int A_CHUNKS,
    int B_CHUNKS
>
__device__ __forceinline__
void tc_commit_tile(
    half (&As)[2][BM][BK],
    half (&Bs)[2][BK][BN],
    const int4 (&a_prefetch)[A_CHUNKS],
    const int4 (&b_prefetch)[B_CHUNKS],
    int tid,
    int stage
) {
    constexpr int VEC_ELEMS =
        8;

    constexpr int A_VEC_COLS =
        BK / VEC_ELEMS;

    constexpr int B_VEC_COLS =
        BN / VEC_ELEMS;

    constexpr int A_VECS =
        BM * A_VEC_COLS;

    constexpr int B_VECS =
        BK * B_VEC_COLS;

#pragma unroll
    for (
        int q = 0;
        q < A_CHUNKS;
        ++q
    ) {
        const int idx =
            tid
            +
            q * THREADS;

        if (
            idx < A_VECS
        ) {
            const int r =
                idx
                /
                A_VEC_COLS;

            const int c8 =
                idx
                %
                A_VEC_COLS;

            int4* dst =
                reinterpret_cast<int4*>(
                    &As[
                        stage
                    ][
                        r
                    ][
                        c8 * VEC_ELEMS
                    ]
                );

            *dst =
                a_prefetch[q];
        }
    }

#pragma unroll
    for (
        int q = 0;
        q < B_CHUNKS;
        ++q
    ) {
        const int idx =
            tid
            +
            q * THREADS;

        if (
            idx < B_VECS
        ) {
            const int r =
                idx
                /
                B_VEC_COLS;

            const int c8 =
                idx
                %
                B_VEC_COLS;

            int4* dst =
                reinterpret_cast<int4*>(
                    &Bs[
                        stage
                    ][
                        r
                    ][
                        c8 * VEC_ELEMS
                    ]
                );

            *dst =
                b_prefetch[q];
        }
    }
}

template<
    int BM,
    int BN,
    int BK,
    int WM,
    int WN
>
__global__
void tensor_core_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    constexpr int MMA_M =
        16;

    constexpr int MMA_N =
        16;

    constexpr int MMA_K =
        16;

    static_assert(
        BM % WM == 0,
        "BM must divide WM."
    );

    static_assert(
        BN % WN == 0,
        "BN must divide WN."
    );

    static_assert(
        WM % MMA_M == 0,
        "WM must be multiple of 16."
    );

    static_assert(
        WN % MMA_N == 0,
        "WN must be multiple of 16."
    );

    static_assert(
        BK % MMA_K == 0,
        "BK must be multiple of 16."
    );

    static_assert(
        BK % 8 == 0,
        "BK must support half8 vector loads."
    );

    static_assert(
        BN % 8 == 0,
        "BN must support half8 vector loads."
    );

    constexpr int WARPS_M =
        BM / WM;

    constexpr int WARPS_N =
        BN / WN;

    constexpr int WARPS =
        WARPS_M
        *
        WARPS_N;

    constexpr int THREADS =
        WARPS
        *
        32;

    constexpr int WARP_M_TILES =
        WM / MMA_M;

    constexpr int WARP_N_TILES =
        WN / MMA_N;

    constexpr int A_VECS =
        BM
        *
        (BK / 8);

    constexpr int B_VECS =
        BK
        *
        (BN / 8);

    constexpr int A_CHUNKS =
        (
            A_VECS
            +
            THREADS
            -
            1
        )
        /
        THREADS;

    constexpr int B_CHUNKS =
        (
            B_VECS
            +
            THREADS
            -
            1
        )
        /
        THREADS;

    __shared__ __align__(32)
    half As[2][BM][BK];

    __shared__ __align__(32)
    half Bs[2][BK][BN];

    const int tid =
        threadIdx.x;

    const int warp_id =
        tid
        >>
        5;

    const int warp_row =
        warp_id
        /
        WARPS_N;

    const int warp_col =
        warp_id
        %
        WARPS_N;

    const int block_row =
        blockIdx.y
        *
        BM;

    const int block_col =
        blockIdx.x
        *
        BN;

    wmma::fragment<
        wmma::accumulator,
        MMA_M,
        MMA_N,
        MMA_K,
        float
    >
    c_frag[
        WARP_M_TILES
    ][
        WARP_N_TILES
    ];

#pragma unroll
    for (
        int mi = 0;
        mi < WARP_M_TILES;
        ++mi
    ) {
#pragma unroll
        for (
            int ni = 0;
            ni < WARP_N_TILES;
            ++ni
        ) {
            wmma::fill_fragment(
                c_frag[mi][ni],
                0.0f
            );
        }
    }

    int4 a_prefetch[
        A_CHUNKS
    ];

    int4 b_prefetch[
        B_CHUNKS
    ];

    /*
     * Prologue:
     * Global -> register -> shared stage 0.
     */
    tc_prefetch_tile<
        BM,
        BN,
        BK,
        THREADS,
        A_CHUNKS,
        B_CHUNKS
    >(
        A,
        B,
        a_prefetch,
        b_prefetch,
        tid,
        block_row,
        block_col,
        0,
        K,
        N
    );

    tc_commit_tile<
        BM,
        BN,
        BK,
        THREADS,
        A_CHUNKS,
        B_CHUNKS
    >(
        As,
        Bs,
        a_prefetch,
        b_prefetch,
        tid,
        0
    );

    __syncthreads();

    int stage =
        0;

    for (
        int bk = 0;
        bk < K;
        bk += BK
    ) {
        const int next_bk =
            bk + BK;

        const bool has_next =
            next_bk < K;

        /*
         * Software prefetch next K tile to registers.
         * SM75 has no cp.async.
         */
        if (
            has_next
        ) {
            tc_prefetch_tile<
                BM,
                BN,
                BK,
                THREADS,
                A_CHUNKS,
                B_CHUNKS
            >(
                A,
                B,
                a_prefetch,
                b_prefetch,
                tid,
                block_row,
                block_col,
                next_bk,
                K,
                N
            );
        }

#pragma unroll
        for (
            int kk = 0;
            kk < BK;
            kk += MMA_K
        ) {
            wmma::fragment<
                wmma::matrix_a,
                MMA_M,
                MMA_N,
                MMA_K,
                half,
                wmma::row_major
            >
            a_frag[
                WARP_M_TILES
            ];

            wmma::fragment<
                wmma::matrix_b,
                MMA_M,
                MMA_N,
                MMA_K,
                half,
                wmma::row_major
            >
            b_frag[
                WARP_N_TILES
            ];

#pragma unroll
            for (
                int mi = 0;
                mi < WARP_M_TILES;
                ++mi
            ) {
                const int a_row =
                    warp_row * WM
                    +
                    mi * MMA_M;

                wmma::load_matrix_sync(
                    a_frag[mi],
                    &As[
                        stage
                    ][
                        a_row
                    ][
                        kk
                    ],
                    BK
                );
            }

#pragma unroll
            for (
                int ni = 0;
                ni < WARP_N_TILES;
                ++ni
            ) {
                const int b_col =
                    warp_col * WN
                    +
                    ni * MMA_N;

                wmma::load_matrix_sync(
                    b_frag[ni],
                    &Bs[
                        stage
                    ][
                        kk
                    ][
                        b_col
                    ],
                    BN
                );
            }

#pragma unroll
            for (
                int mi = 0;
                mi < WARP_M_TILES;
                ++mi
            ) {
#pragma unroll
                for (
                    int ni = 0;
                    ni < WARP_N_TILES;
                    ++ni
                ) {
                    wmma::mma_sync(
                        c_frag[mi][ni],
                        a_frag[mi],
                        b_frag[ni],
                        c_frag[mi][ni]
                    );
                }
            }
        }

        if (
            has_next
        ) {
            const int next_stage =
                stage
                ^
                1;

            tc_commit_tile<
                BM,
                BN,
                BK,
                THREADS,
                A_CHUNKS,
                B_CHUNKS
            >(
                As,
                Bs,
                a_prefetch,
                b_prefetch,
                tid,
                next_stage
            );

            __syncthreads();

            stage =
                next_stage;
        }
    }

#pragma unroll
    for (
        int mi = 0;
        mi < WARP_M_TILES;
        ++mi
    ) {
#pragma unroll
        for (
            int ni = 0;
            ni < WARP_N_TILES;
            ++ni
        ) {
            const int row =
                block_row
                +
                warp_row * WM
                +
                mi * MMA_M;

            const int col =
                block_col
                +
                warp_col * WN
                +
                ni * MMA_N;

            float* out =
                C
                +
                static_cast<size_t>(
                    row
                )
                *
                N
                +
                col;

            wmma::store_matrix_sync(
                out,
                c_frag[mi][ni],
                N,
                wmma::mem_row_major
            );
        }
    }
}

template<
    int BM,
    int BN,
    int BK,
    int WM,
    int WN
>
void tc_launcher(
    const half* A,
    const half* B,
    float* C,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    constexpr int THREADS =
        (BM / WM)
        *
        (BN / WN)
        *
        32;

    static_assert(
        THREADS <= 1024,
        "Too many threads."
    );

    dim3 block(
        THREADS
    );

    dim3 grid(
        (N + BN - 1)
        /
        BN,
        (M + BM - 1)
        /
        BM
    );

    tensor_core_gemm_kernel<
        BM,
        BN,
        BK,
        WM,
        WN
    >
    <<<grid, block, 0, stream>>>(
        A,
        B,
        C,
        M,
        N,
        K
    );
}


/*
 * ============================================================
 * V7 Tensor Core data-feeding path
 * ============================================================
 *
 * V6 already pipelines:
 *   Global -> Register -> double-buffered Shared
 *
 * V7 additionally pipelines:
 *   Shared -> WMMA Fragment registers
 *
 * and uses WMMA-compatible padded shared-memory strides:
 *   A stride = BK + 8 half
 *   B stride = BN + 8 half
 *
 * Padding is deliberately 8 half values (=16 bytes), rather than
 * one element. This keeps WMMA leading dimensions valid and keeps
 * every cooperative vector store naturally aligned.
 *
 * This is NOT CUTLASS's full XOR-permuted TensorOp layout. It is a
 * conservative SM75-compatible stepping stone that can be measured
 * against the V6 baseline without pretending to implement ldmatrix
 * iterators that are not actually present here.
 */

template<
    int BM,
    int BN,
    int BK,
    int THREADS,
    int A_CHUNKS,
    int B_CHUNKS,
    int PAD
>
__device__ __forceinline__
void tc_commit_tile_padded(
    half (&As)[2][BM][BK + PAD],
    half (&Bs)[2][BK][BN + PAD],
    const int4 (&a_prefetch)[A_CHUNKS],
    const int4 (&b_prefetch)[B_CHUNKS],
    int tid,
    int stage
) {
    constexpr int VEC_ELEMS = 8;

    constexpr int A_VEC_COLS =
        BK / VEC_ELEMS;

    constexpr int B_VEC_COLS =
        BN / VEC_ELEMS;

    constexpr int A_VECS =
        BM * A_VEC_COLS;

    constexpr int B_VECS =
        BK * B_VEC_COLS;

#pragma unroll
    for (int q = 0; q < A_CHUNKS; ++q) {
        const int idx =
            tid + q * THREADS;

        if (idx < A_VECS) {
            const int r =
                idx / A_VEC_COLS;

            const int c8 =
                idx % A_VEC_COLS;

            int4* dst =
                reinterpret_cast<int4*>(
                    &As[stage][r][c8 * VEC_ELEMS]
                );

            *dst =
                a_prefetch[q];
        }
    }

#pragma unroll
    for (int q = 0; q < B_CHUNKS; ++q) {
        const int idx =
            tid + q * THREADS;

        if (idx < B_VECS) {
            const int r =
                idx / B_VEC_COLS;

            const int c8 =
                idx % B_VEC_COLS;

            int4* dst =
                reinterpret_cast<int4*>(
                    &Bs[stage][r][c8 * VEC_ELEMS]
                );

            *dst =
                b_prefetch[q];
        }
    }
}

template<
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    int PAD
>
__global__
void tensor_core_fragpipe_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 16;
    constexpr int MMA_K = 16;

    static_assert(
        BM % WM == 0,
        "BM must be divisible by WM."
    );

    static_assert(
        BN % WN == 0,
        "BN must be divisible by WN."
    );

    static_assert(
        WM % MMA_M == 0,
        "WM must be a multiple of 16."
    );

    static_assert(
        WN % MMA_N == 0,
        "WN must be a multiple of 16."
    );

    static_assert(
        BK % MMA_K == 0,
        "BK must be a multiple of 16."
    );

    static_assert(
        PAD % 8 == 0,
        "WMMA-compatible padding must be a multiple of 8 half values."
    );

    constexpr int A_STRIDE =
        BK + PAD;

    constexpr int B_STRIDE =
        BN + PAD;

    static_assert(
        A_STRIDE % 8 == 0,
        "A WMMA leading dimension must be a multiple of 8 half values."
    );

    static_assert(
        B_STRIDE % 8 == 0,
        "B WMMA leading dimension must be a multiple of 8 half values."
    );

    constexpr int WARPS_M =
        BM / WM;

    constexpr int WARPS_N =
        BN / WN;

    constexpr int WARPS =
        WARPS_M * WARPS_N;

    constexpr int THREADS =
        WARPS * 32;

    constexpr int WARP_M_TILES =
        WM / MMA_M;

    constexpr int WARP_N_TILES =
        WN / MMA_N;

    constexpr int A_VECS =
        BM * (BK / 8);

    constexpr int B_VECS =
        BK * (BN / 8);

    constexpr int A_CHUNKS =
        (A_VECS + THREADS - 1)
        /
        THREADS;

    constexpr int B_CHUNKS =
        (B_VECS + THREADS - 1)
        /
        THREADS;

    __shared__ __align__(32)
    half As[2][BM][A_STRIDE];

    __shared__ __align__(32)
    half Bs[2][BK][B_STRIDE];

    const int tid =
        threadIdx.x;

    const int warp_id =
        tid >> 5;

    const int warp_row =
        warp_id / WARPS_N;

    const int warp_col =
        warp_id % WARPS_N;

    const int block_row =
        blockIdx.y * BM;

    const int block_col =
        blockIdx.x * BN;

    using AFrag =
        wmma::fragment<
            wmma::matrix_a,
            MMA_M,
            MMA_N,
            MMA_K,
            half,
            wmma::row_major
        >;

    using BFrag =
        wmma::fragment<
            wmma::matrix_b,
            MMA_M,
            MMA_N,
            MMA_K,
            half,
            wmma::row_major
        >;

    using CFrag =
        wmma::fragment<
            wmma::accumulator,
            MMA_M,
            MMA_N,
            MMA_K,
            float
        >;

    CFrag c_frag[
        WARP_M_TILES
    ][
        WARP_N_TILES
    ];

#pragma unroll
    for (int mi = 0; mi < WARP_M_TILES; ++mi) {
#pragma unroll
        for (int ni = 0; ni < WARP_N_TILES; ++ni) {
            wmma::fill_fragment(
                c_frag[mi][ni],
                0.0f
            );
        }
    }

    int4 a_prefetch[
        A_CHUNKS
    ];

    int4 b_prefetch[
        B_CHUNKS
    ];

    /*
     * CTA-level pipeline prologue:
     * Global -> Registers -> Shared stage 0.
     */
    tc_prefetch_tile<
        BM,
        BN,
        BK,
        THREADS,
        A_CHUNKS,
        B_CHUNKS
    >(
        A,
        B,
        a_prefetch,
        b_prefetch,
        tid,
        block_row,
        block_col,
        0,
        K,
        N
    );

    tc_commit_tile_padded<
        BM,
        BN,
        BK,
        THREADS,
        A_CHUNKS,
        B_CHUNKS,
        PAD
    >(
        As,
        Bs,
        a_prefetch,
        b_prefetch,
        tid,
        0
    );

    __syncthreads();

    int shared_stage =
        0;

    for (
        int bk = 0;
        bk < K;
        bk += BK
    ) {
        const int next_bk =
            bk + BK;

        const bool has_next_cta_tile =
            next_bk < K;

        /*
         * Start the next Global -> Register load before the current
         * Tensor Core work.
         */
        if (has_next_cta_tile) {
            tc_prefetch_tile<
                BM,
                BN,
                BK,
                THREADS,
                A_CHUNKS,
                B_CHUNKS
            >(
                A,
                B,
                a_prefetch,
                b_prefetch,
                tid,
                block_row,
                block_col,
                next_bk,
                K,
                N
            );
        }

        /*
         * Second pipeline level:
         *
         * Shared -> Fragment slot 0
         * then, before MMA(slot 0), preload Shared -> Fragment slot 1.
         *
         * The compiler/scheduler can overlap independent shared-memory
         * operand movement with Tensor Core issue where dependencies permit.
         */
        AFrag a_frag[
            2
        ][
            WARP_M_TILES
        ];

        BFrag b_frag[
            2
        ][
            WARP_N_TILES
        ];

        int fragment_stage =
            0;

#pragma unroll
        for (int mi = 0; mi < WARP_M_TILES; ++mi) {
            const int a_row =
                warp_row * WM
                +
                mi * MMA_M;

            wmma::load_matrix_sync(
                a_frag[0][mi],
                &As[
                    shared_stage
                ][
                    a_row
                ][
                    0
                ],
                A_STRIDE
            );
        }

#pragma unroll
        for (int ni = 0; ni < WARP_N_TILES; ++ni) {
            const int b_col =
                warp_col * WN
                +
                ni * MMA_N;

            wmma::load_matrix_sync(
                b_frag[0][ni],
                &Bs[
                    shared_stage
                ][
                    0
                ][
                    b_col
                ],
                B_STRIDE
            );
        }

#pragma unroll
        for (
            int kk = 0;
            kk < BK;
            kk += MMA_K
        ) {
            const int next_kk =
                kk + MMA_K;

            const bool has_next_fragment =
                next_kk < BK;

            const int next_fragment_stage =
                fragment_stage ^ 1;

            if (has_next_fragment) {
#pragma unroll
                for (int mi = 0; mi < WARP_M_TILES; ++mi) {
                    const int a_row =
                        warp_row * WM
                        +
                        mi * MMA_M;

                    wmma::load_matrix_sync(
                        a_frag[next_fragment_stage][mi],
                        &As[
                            shared_stage
                        ][
                            a_row
                        ][
                            next_kk
                        ],
                        A_STRIDE
                    );
                }

#pragma unroll
                for (int ni = 0; ni < WARP_N_TILES; ++ni) {
                    const int b_col =
                        warp_col * WN
                        +
                        ni * MMA_N;

                    wmma::load_matrix_sync(
                        b_frag[next_fragment_stage][ni],
                        &Bs[
                            shared_stage
                        ][
                            next_kk
                        ][
                            b_col
                        ],
                        B_STRIDE
                    );
                }
            }

#pragma unroll
            for (int mi = 0; mi < WARP_M_TILES; ++mi) {
#pragma unroll
                for (int ni = 0; ni < WARP_N_TILES; ++ni) {
                    wmma::mma_sync(
                        c_frag[mi][ni],
                        a_frag[fragment_stage][mi],
                        b_frag[fragment_stage][ni],
                        c_frag[mi][ni]
                    );
                }
            }

            if (has_next_fragment) {
                fragment_stage =
                    next_fragment_stage;
            }
        }

        if (has_next_cta_tile) {
            const int next_shared_stage =
                shared_stage ^ 1;

            tc_commit_tile_padded<
                BM,
                BN,
                BK,
                THREADS,
                A_CHUNKS,
                B_CHUNKS,
                PAD
            >(
                As,
                Bs,
                a_prefetch,
                b_prefetch,
                tid,
                next_shared_stage
            );

            __syncthreads();

            shared_stage =
                next_shared_stage;
        }
    }

#pragma unroll
    for (int mi = 0; mi < WARP_M_TILES; ++mi) {
#pragma unroll
        for (int ni = 0; ni < WARP_N_TILES; ++ni) {
            const int row =
                block_row
                +
                warp_row * WM
                +
                mi * MMA_M;

            const int col =
                block_col
                +
                warp_col * WN
                +
                ni * MMA_N;

            float* out =
                C
                +
                static_cast<size_t>(row)
                *
                N
                +
                col;

            wmma::store_matrix_sync(
                out,
                c_frag[mi][ni],
                N,
                wmma::mem_row_major
            );
        }
    }
}

template<
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    int PAD
>
void tc_fragpipe_launcher(
    const half* A,
    const half* B,
    float* C,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    constexpr int THREADS =
        (BM / WM)
        *
        (BN / WN)
        *
        32;

    static_assert(
        THREADS <= 1024,
        "Too many threads."
    );

    dim3 block(
        THREADS
    );

    dim3 grid(
        (N + BN - 1)
        /
        BN,
        (M + BM - 1)
        /
        BM
    );

    tensor_core_fragpipe_kernel<
        BM,
        BN,
        BK,
        WM,
        WN,
        PAD
    >
    <<<grid, block, 0, stream>>>(
        A,
        B,
        C,
        M,
        N,
        K
    );
}

#define REG_TC_V6(NAME, BM_, BN_, BK_, WM_, WN_) \
    { \
        NAME, \
        BM_, BN_, BK_, \
        WM_, WN_, \
        16, 16, 16, \
        (BM_ / WM_) * (BN_ / WN_) * 32, \
        2, \
        1, \
        0, \
        2 * (BM_ * BK_ + BK_ * BN_) * (int)sizeof(half), \
        TensorCorePath::V6_WMMA, \
        &tc_launcher<BM_, BN_, BK_, WM_, WN_> \
    }

#define REG_TC_V7(NAME, BM_, BN_, BK_, WM_, WN_, PAD_) \
    { \
        NAME, \
        BM_, BN_, BK_, \
        WM_, WN_, \
        16, 16, 16, \
        (BM_ / WM_) * (BN_ / WN_) * 32, \
        2, \
        2, \
        PAD_, \
        2 * (BM_ * (BK_ + PAD_) + BK_ * (BN_ + PAD_)) * (int)sizeof(half), \
        TensorCorePath::FRAGPIPE_PADDED, \
        &tc_fragpipe_launcher<BM_, BN_, BK_, WM_, WN_, PAD_> \
    }

const std::vector<TensorCoreConfig>&
get_tensor_core_registry() {
    static const std::vector<TensorCoreConfig>
    registry = {
        /*
         * V6 baselines kept for direct A/B tests.
         */
        REG_TC_V6(
            "tcv6_m64_n64_k16_w32x32",
            64, 64, 16,
            32, 32
        ),

        REG_TC_V6(
            "tcv6_m128_n64_k16_w64x32",
            128, 64, 16,
            64, 32
        ),

        REG_TC_V6(
            "tcv6_m128_n64_k32_w64x32",
            128, 64, 32,
            64, 32
        ),

        REG_TC_V6(
            "tcv6_m64_n128_k32_w32x64",
            64, 128, 32,
            32, 64
        ),

        REG_TC_V6(
            "tcv6_m128_n128_k32_w64x64",
            128, 128, 32,
            64, 64
        ),

        /*
         * V7 fragment-pipeline + padded-SMEM candidates.
         *
         * PAD=8 half values = 16 bytes.
         */
        REG_TC_V7(
            "tcv7_m64_n64_k16_w32x32_fp",
            64, 64, 16,
            32, 32,
            8
        ),

        REG_TC_V7(
            "tcv7_m64_n64_k32_w32x32_fp",
            64, 64, 32,
            32, 32,
            8
        ),

        REG_TC_V7(
            "tcv7_m64_n64_k64_w32x32_fp",
            64, 64, 64,
            32, 32,
            8
        ),

        {
            "tcv7_m128_n64_k16_w64x32_frag",
            128, 64, 16,
            64, 32,
            16, 16, 16,
            (128 / 64) * (64 / 32) * 32,
            2,
            2,
            0,
            2 * (128 * 16 + 16 * 64) * (int)sizeof(half),
            TensorCorePath::FRAGPIPE,
            &tc_fragpipe_launcher<128, 64, 16, 64, 32, 0>
        },

        REG_TC_V7(
            "tcv7_m128_n64_k16_w64x32_fp",
            128, 64, 16,
            64, 32,
            8
        ),

        REG_TC_V7(
            "tcv7_m128_n64_k32_w64x32_fp",
            128, 64, 32,
            64, 32,
            8
        ),

        REG_TC_V7(
            "tcv7_m128_n64_k48_w64x32_fp",
            128, 64, 48,
            64, 32,
            8
        ),

        REG_TC_V7(
            "tcv7_m128_n64_k32_w32x32_fp",
            128, 64, 32,
            32, 32,
            8
        ),

        REG_TC_V7(
            "tcv7_m64_n128_k16_w32x64_fp",
            64, 128, 16,
            32, 64,
            8
        ),

        REG_TC_V7(
            "tcv7_m64_n128_k32_w32x64_fp",
            64, 128, 32,
            32, 64,
            8
        ),

        REG_TC_V7(
            "tcv7_m128_n128_k16_w64x64_fp",
            128, 128, 16,
            64, 64,
            8
        ),

        REG_TC_V7(
            "tcv7_m128_n128_k32_w64x64_fp",
            128, 128, 32,
            64, 64,
            8
        ),

        /*
         * Wider N tiles broaden the search space and explicitly test
         * grid-parallelism vs reuse on small-M / large-N workloads.
         */
        REG_TC_V7(
            "tcv7_m64_n256_k16_w32x64_fp",
            64, 256, 16,
            32, 64,
            8
        ),

        REG_TC_V7(
            "tcv7_m64_n256_k32_w32x64_fp",
            64, 256, 32,
            32, 64,
            8
        ),

        REG_TC_V7(
            "tcv7_m128_n256_k16_w64x64_fp",
            128, 256, 16,
            64, 64,
            8
        )
    };

    return registry;
}

#undef REG_TC_V6
#undef REG_TC_V7


const TensorCoreConfig*
find_tensor_core_kernel(
    const std::string& name
) {
    for (
        const auto& kernel :
        get_tensor_core_registry()
    ) {
        if (
            name ==
            kernel.name
        ) {
            return
                &kernel;
        }
    }

    return nullptr;
}

const char*
tensor_core_path_name(
    TensorCorePath path
) {
    switch (path) {
        case TensorCorePath::V6_WMMA:
            return "v6_wmma";

        case TensorCorePath::FRAGPIPE:
            return "fragpipe";

        case TensorCorePath::FRAGPIPE_PADDED:
            return "fragpipe_padded";

        default:
            return "unknown";
    }
}

bool tensor_core_problem_compatible(
    const TensorCoreConfig& kernel,
    const half* A,
    const half* B,
    int M,
    int N,
    int K
) {
    const auto a_addr =
        reinterpret_cast<std::uintptr_t>(
            A
        );

    const auto b_addr =
        reinterpret_cast<std::uintptr_t>(
            B
        );

    const bool aligned =
        ((a_addr & 0xF) == 0)
        &&
        ((b_addr & 0xF) == 0);

    /*
     * V6 Tensor Core fast path deliberately requires exact
     * 16-element granularity. FP32 SIMT remains the general path.
     */
    const bool shape_ok =
        (M % kernel.BM == 0)
        &&
        (N % kernel.BN == 0)
        &&
        (K % kernel.BK == 0)
        &&
        (M % 16 == 0)
        &&
        (N % 16 == 0)
        &&
        (K % 16 == 0);

    return
        aligned
        &&
        shape_ok;
}

void launch_tensor_core_kernel(
    const TensorCoreConfig& config,
    const half* A,
    const half* B,
    float* C,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    config.launcher(
        A,
        B,
        C,
        M,
        N,
        K,
        stream
    );

    CUDA_CHECK(
        cudaGetLastError()
    );
}
