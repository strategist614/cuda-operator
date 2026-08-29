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

#define REG_TC(NAME, BM_, BN_, BK_, WM_, WN_) \
    { \
        NAME, \
        BM_, BN_, BK_, \
        WM_, WN_, \
        16, 16, 16, \
        (BM_ / WM_) * (BN_ / WN_) * 32, \
        2, \
        2 * (BM_ * BK_ + BK_ * BN_) * (int)sizeof(half), \
        &tc_launcher<BM_, BN_, BK_, WM_, WN_> \
    }

const std::vector<TensorCoreConfig>&
get_tensor_core_registry() {
    static const std::vector<TensorCoreConfig>
    registry = {
        REG_TC(
            "tc_m64_n64_k16_w32x32",
            64, 64, 16,
            32, 32
        ),

        REG_TC(
            "tc_m64_n64_k32_w32x32",
            64, 64, 32,
            32, 32
        ),

        REG_TC(
            "tc_m128_n64_k16_w64x32",
            128, 64, 16,
            64, 32
        ),

        REG_TC(
            "tc_m128_n64_k32_w64x32",
            128, 64, 32,
            64, 32
        ),

        REG_TC(
            "tc_m128_n64_k32_w32x32",
            128, 64, 32,
            32, 32
        ),

        REG_TC(
            "tc_m64_n128_k16_w32x64",
            64, 128, 16,
            32, 64
        ),

        REG_TC(
            "tc_m64_n128_k32_w32x64",
            64, 128, 32,
            32, 64
        ),

        REG_TC(
            "tc_m128_n128_k16_w64x64",
            128, 128, 16,
            64, 64
        ),

        REG_TC(
            "tc_m128_n128_k32_w64x64",
            128, 128, 32,
            64, 64
        ),

        REG_TC(
            "tc_m128_n128_k32_w32x64",
            128, 128, 32,
            32, 64
        )
    };

    return registry;
}

#undef REG_TC

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
