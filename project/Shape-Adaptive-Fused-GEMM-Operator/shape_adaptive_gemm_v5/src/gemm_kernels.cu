#include "gemm.h"
#include "common.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <vector>

__device__ __forceinline__
float silu_device(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__
float4 zero_float4() {
    return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

__device__ __forceinline__
float4 load_float4_safe(
    const float* base,
    int row,
    int ld,
    int col,
    int rows,
    int cols
) {
    float4 v = zero_float4();

    if (row >= rows || col >= cols) {
        return v;
    }

    if (col + 3 < cols) {
        const float* ptr =
            base
            +
            static_cast<size_t>(row) * ld
            +
            col;

        return *reinterpret_cast<const float4*>(ptr);
    }

    if (col + 0 < cols) {
        v.x = base[static_cast<size_t>(row) * ld + col + 0];
    }
    if (col + 1 < cols) {
        v.y = base[static_cast<size_t>(row) * ld + col + 1];
    }
    if (col + 2 < cols) {
        v.z = base[static_cast<size_t>(row) * ld + col + 2];
    }
    if (col + 3 < cols) {
        v.w = base[static_cast<size_t>(row) * ld + col + 3];
    }

    return v;
}

template<int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__
void compute_tile(
    float (&acc)[TM][TN],
    const float (&As)[BM][BK],
    const float (&Bs)[BK][BN],
    int thread_row,
    int thread_col
) {
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
        float a_frag[TM];
        float b_frag[TN];

#pragma unroll
        for (int i = 0; i < TM; ++i) {
            a_frag[i] =
                As[thread_row * TM + i][kk];
        }

#pragma unroll
        for (int j = 0; j < TN; ++j) {
            b_frag[j] =
                Bs[kk][thread_col * TN + j];
        }

#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                acc[i][j] =
                    fmaf(
                        a_frag[i],
                        b_frag[j],
                        acc[i][j]
                    );
            }
        }
    }
}

template<int BM, int BN, int BK, int TM, int TN, bool VEC4>
__global__
void gemm_basic_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const float* __restrict__ bias,
    int M,
    int N,
    int K,
    EpilogueType epilogue
) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    constexpr int THREAD_COLS = BN / TN;

    const int tid = threadIdx.x;
    const int thread_row = tid / THREAD_COLS;
    const int thread_col = tid % THREAD_COLS;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[TM][TN] = {};

    for (int bk = 0; bk < K; bk += BK) {
        if constexpr (VEC4) {
            constexpr int A_VEC_COLS = BK / 4;
            constexpr int B_VEC_COLS = BN / 4;
            constexpr int A_VECS = BM * A_VEC_COLS;
            constexpr int B_VECS = BK * B_VEC_COLS;

            for (int idx = tid; idx < A_VECS; idx += blockDim.x) {
                const int r = idx / A_VEC_COLS;
                const int c4 = idx % A_VEC_COLS;

                const float4 v =
                    load_float4_safe(
                        A,
                        block_row + r,
                        K,
                        bk + c4 * 4,
                        M,
                        K
                    );

                As[r][c4 * 4 + 0] = v.x;
                As[r][c4 * 4 + 1] = v.y;
                As[r][c4 * 4 + 2] = v.z;
                As[r][c4 * 4 + 3] = v.w;
            }

            for (int idx = tid; idx < B_VECS; idx += blockDim.x) {
                const int r = idx / B_VEC_COLS;
                const int c4 = idx % B_VEC_COLS;

                const float4 v =
                    load_float4_safe(
                        B,
                        bk + r,
                        N,
                        block_col + c4 * 4,
                        K,
                        N
                    );

                Bs[r][c4 * 4 + 0] = v.x;
                Bs[r][c4 * 4 + 1] = v.y;
                Bs[r][c4 * 4 + 2] = v.z;
                Bs[r][c4 * 4 + 3] = v.w;
            }
        } else {
            for (int idx = tid; idx < BM * BK; idx += blockDim.x) {
                const int r = idx / BK;
                const int c = idx % BK;

                const int gr = block_row + r;
                const int gc = bk + c;

                As[r][c] =
                    (gr < M && gc < K)
                    ? A[static_cast<size_t>(gr) * K + gc]
                    : 0.0f;
            }

            for (int idx = tid; idx < BK * BN; idx += blockDim.x) {
                const int r = idx / BN;
                const int c = idx % BN;

                const int gr = bk + r;
                const int gc = block_col + c;

                Bs[r][c] =
                    (gr < K && gc < N)
                    ? B[static_cast<size_t>(gr) * N + gc]
                    : 0.0f;
            }
        }

        __syncthreads();

        compute_tile<BM, BN, BK, TM, TN>(
            acc,
            As,
            Bs,
            thread_row,
            thread_col
        );

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int row =
                block_row + thread_row * TM + i;

            const int col =
                block_col + thread_col * TN + j;

            if (row < M && col < N) {
                float value = acc[i][j];

                if (epilogue == EpilogueType::BIAS ||
                    epilogue == EpilogueType::BIAS_SILU) {
                    value += bias[col];
                }

                if (epilogue == EpilogueType::BIAS_SILU) {
                    value = silu_device(value);
                }

                C[static_cast<size_t>(row) * N + col] = value;
            }
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
void pipeline_prefetch_tile(
    const float* A,
    const float* B,
    float4 (&a_prefetch)[A_CHUNKS],
    float4 (&b_prefetch)[B_CHUNKS],
    int tid,
    int block_row,
    int block_col,
    int bk,
    int M,
    int N,
    int K
) {
    constexpr int A_VEC_COLS = BK / 4;
    constexpr int B_VEC_COLS = BN / 4;
    constexpr int A_VECS = BM * A_VEC_COLS;
    constexpr int B_VECS = BK * B_VEC_COLS;

#pragma unroll
    for (int q = 0; q < A_CHUNKS; ++q) {
        const int idx = tid + q * THREADS;

        if (idx < A_VECS) {
            const int r = idx / A_VEC_COLS;
            const int c4 = idx % A_VEC_COLS;

            a_prefetch[q] =
                load_float4_safe(
                    A,
                    block_row + r,
                    K,
                    bk + c4 * 4,
                    M,
                    K
                );
        } else {
            a_prefetch[q] = zero_float4();
        }
    }

#pragma unroll
    for (int q = 0; q < B_CHUNKS; ++q) {
        const int idx = tid + q * THREADS;

        if (idx < B_VECS) {
            const int r = idx / B_VEC_COLS;
            const int c4 = idx % B_VEC_COLS;

            b_prefetch[q] =
                load_float4_safe(
                    B,
                    bk + r,
                    N,
                    block_col + c4 * 4,
                    K,
                    N
                );
        } else {
            b_prefetch[q] = zero_float4();
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
void pipeline_commit_tile(
    float (&As)[2][BM][BK],
    float (&Bs)[2][BK][BN],
    const float4 (&a_prefetch)[A_CHUNKS],
    const float4 (&b_prefetch)[B_CHUNKS],
    int tid,
    int stage
) {
    constexpr int A_VEC_COLS = BK / 4;
    constexpr int B_VEC_COLS = BN / 4;
    constexpr int A_VECS = BM * A_VEC_COLS;
    constexpr int B_VECS = BK * B_VEC_COLS;

#pragma unroll
    for (int q = 0; q < A_CHUNKS; ++q) {
        const int idx = tid + q * THREADS;

        if (idx < A_VECS) {
            const int r = idx / A_VEC_COLS;
            const int c4 = idx % A_VEC_COLS;
            const float4 v = a_prefetch[q];

            As[stage][r][c4 * 4 + 0] = v.x;
            As[stage][r][c4 * 4 + 1] = v.y;
            As[stage][r][c4 * 4 + 2] = v.z;
            As[stage][r][c4 * 4 + 3] = v.w;
        }
    }

#pragma unroll
    for (int q = 0; q < B_CHUNKS; ++q) {
        const int idx = tid + q * THREADS;

        if (idx < B_VECS) {
            const int r = idx / B_VEC_COLS;
            const int c4 = idx % B_VEC_COLS;
            const float4 v = b_prefetch[q];

            Bs[stage][r][c4 * 4 + 0] = v.x;
            Bs[stage][r][c4 * 4 + 1] = v.y;
            Bs[stage][r][c4 * 4 + 2] = v.z;
            Bs[stage][r][c4 * 4 + 3] = v.w;
        }
    }
}

template<int BM, int BN, int BK, int TM, int TN>
__global__
void gemm_pipeline_vec4_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const float* __restrict__ bias,
    int M,
    int N,
    int K,
    EpilogueType epilogue
) {
    static_assert(BK % 4 == 0, "BK must be divisible by 4.");
    static_assert(BN % 4 == 0, "BN must be divisible by 4.");

    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];

    constexpr int THREADS =
        (BM / TM) * (BN / TN);

    constexpr int THREAD_COLS =
        BN / TN;

    constexpr int A_VEC_COLS =
        BK / 4;

    constexpr int B_VEC_COLS =
        BN / 4;

    constexpr int A_VECS =
        BM * A_VEC_COLS;

    constexpr int B_VECS =
        BK * B_VEC_COLS;

    constexpr int A_CHUNKS =
        (A_VECS + THREADS - 1) / THREADS;

    constexpr int B_CHUNKS =
        (B_VECS + THREADS - 1) / THREADS;

    const int tid = threadIdx.x;
    const int thread_row = tid / THREAD_COLS;
    const int thread_col = tid % THREAD_COLS;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[TM][TN] = {};

    float4 a_prefetch[A_CHUNKS];
    float4 b_prefetch[B_CHUNKS];

    /*
     * Prologue.
     */
    pipeline_prefetch_tile<
        BM, BN, BK,
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
        M,
        N,
        K
    );

    pipeline_commit_tile<
        BM, BN, BK,
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

    int stage = 0;

    for (int bk = 0; bk < K; bk += BK) {
        const int next_bk = bk + BK;
        const bool has_next = next_bk < K;

        /*
         * Global -> register prefetch of next tile.
         *
         * These loads are independent of the current tile's shared-memory
         * reads/FMA chain, giving the scheduler an opportunity to hide part
         * of their latency behind current-tile computation.
         */
        if (has_next) {
            pipeline_prefetch_tile<
                BM, BN, BK,
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
                M,
                N,
                K
            );
        }

        compute_tile<BM, BN, BK, TM, TN>(
            acc,
            As[stage],
            Bs[stage],
            thread_row,
            thread_col
        );

        if (has_next) {
            const int next_stage = stage ^ 1;

            pipeline_commit_tile<
                BM, BN, BK,
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

            stage = next_stage;
        }
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int row =
                block_row + thread_row * TM + i;

            const int col =
                block_col + thread_col * TN + j;

            if (row < M && col < N) {
                float value = acc[i][j];

                if (epilogue == EpilogueType::BIAS ||
                    epilogue == EpilogueType::BIAS_SILU) {
                    value += bias[col];
                }

                if (epilogue == EpilogueType::BIAS_SILU) {
                    value = silu_device(value);
                }

                C[static_cast<size_t>(row) * N + col] = value;
            }
        }
    }
}

template<
    int BM,
    int BN,
    int BK,
    int TM,
    int TN,
    bool VEC4
>
void basic_launcher(
    const float* A,
    const float* B,
    float* C,
    const float* bias,
    int M,
    int N,
    int K,
    EpilogueType epilogue,
    cudaStream_t stream
) {
    constexpr int THREADS =
        (BM / TM)
        *
        (BN / TN);

    dim3 block(THREADS);

    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    gemm_basic_kernel<
        BM,
        BN,
        BK,
        TM,
        TN,
        VEC4
    >
    <<<grid, block, 0, stream>>>(
        A,
        B,
        C,
        bias,
        M,
        N,
        K,
        epilogue
    );
}

template<
    int BM,
    int BN,
    int BK,
    int TM,
    int TN
>
void pipeline_launcher(
    const float* A,
    const float* B,
    float* C,
    const float* bias,
    int M,
    int N,
    int K,
    EpilogueType epilogue,
    cudaStream_t stream
) {
    constexpr int THREADS =
        (BM / TM)
        *
        (BN / TN);

    dim3 block(THREADS);

    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    gemm_pipeline_vec4_kernel<
        BM,
        BN,
        BK,
        TM,
        TN
    >
    <<<grid, block, 0, stream>>>(
        A,
        B,
        C,
        bias,
        M,
        N,
        K,
        epilogue
    );
}


/*
 * ============================================================
 * V5: Warp-level tiled, padded shared-memory software pipeline
 * ============================================================
 *
 * Each warp owns one WM x WN output tile.
 *
 * Example for the V4 winner:
 *
 *   BM=128 BN=64 BK=16
 *   WM=32  WN=32
 *   TM=8   TN=4
 *
 * 8 warps/block:
 *
 *   block 128x64
 *     -> 4 warp rows x 2 warp cols
 *     -> each warp computes 32x32
 *
 * Within a warp:
 *
 *   WM/TM = 4 lane rows
 *   WN/TN = 8 lane cols
 *   4 x 8 = 32 lanes
 *
 * Shared-memory row strides are padded by one FP32 element.
 */

template<
    int BM,
    int BN,
    int BK,
    int TM,
    int TN,
    int WM,
    int WN
>
__device__ __forceinline__
void compute_tile_warp_padded(
    float (&acc)[TM][TN],
    const float (&As)[BM][BK + 1],
    const float (&Bs)[BK][BN + 1],
    int warp_row,
    int warp_col,
    int lane_row,
    int lane_col
) {
    const int thread_tile_row =
        warp_row * WM
        +
        lane_row * TM;

    const int thread_tile_col =
        warp_col * WN
        +
        lane_col * TN;

#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
        float a_frag[TM];
        float b_frag[TN];

#pragma unroll
        for (int i = 0; i < TM; ++i) {
            a_frag[i] =
                As[
                    thread_tile_row + i
                ][kk];
        }

#pragma unroll
        for (int j = 0; j < TN; ++j) {
            b_frag[j] =
                Bs[
                    kk
                ][
                    thread_tile_col + j
                ];
        }

#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                acc[i][j] =
                    fmaf(
                        a_frag[i],
                        b_frag[j],
                        acc[i][j]
                    );
            }
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
void pipeline_commit_tile_padded(
    float (&As)[2][BM][BK + 1],
    float (&Bs)[2][BK][BN + 1],
    const float4 (&a_prefetch)[A_CHUNKS],
    const float4 (&b_prefetch)[B_CHUNKS],
    int tid,
    int stage
) {
    constexpr int A_VEC_COLS =
        BK / 4;

    constexpr int B_VEC_COLS =
        BN / 4;

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

            const int c4 =
                idx % A_VEC_COLS;

            const float4 v =
                a_prefetch[q];

            As[stage][r][c4 * 4 + 0] = v.x;
            As[stage][r][c4 * 4 + 1] = v.y;
            As[stage][r][c4 * 4 + 2] = v.z;
            As[stage][r][c4 * 4 + 3] = v.w;
        }
    }

#pragma unroll
    for (int q = 0; q < B_CHUNKS; ++q) {
        const int idx =
            tid + q * THREADS;

        if (idx < B_VECS) {
            const int r =
                idx / B_VEC_COLS;

            const int c4 =
                idx % B_VEC_COLS;

            const float4 v =
                b_prefetch[q];

            Bs[stage][r][c4 * 4 + 0] = v.x;
            Bs[stage][r][c4 * 4 + 1] = v.y;
            Bs[stage][r][c4 * 4 + 2] = v.z;
            Bs[stage][r][c4 * 4 + 3] = v.w;
        }
    }
}

template<
    int BM,
    int BN,
    int BK,
    int TM,
    int TN,
    int WM,
    int WN
>
__global__
void gemm_warp_pipeline_vec4_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const float* __restrict__ bias,
    int M,
    int N,
    int K,
    EpilogueType epilogue
) {
    static_assert(
        BM % WM == 0,
        "BM must be divisible by WM."
    );

    static_assert(
        BN % WN == 0,
        "BN must be divisible by WN."
    );

    static_assert(
        WM % TM == 0,
        "WM must be divisible by TM."
    );

    static_assert(
        WN % TN == 0,
        "WN must be divisible by TN."
    );

    static_assert(
        (WM / TM) * (WN / TN) == 32,
        "One warp must exactly cover one warp tile."
    );

    static_assert(
        BK % 4 == 0,
        "BK must be divisible by 4."
    );

    static_assert(
        BN % 4 == 0,
        "BN must be divisible by 4."
    );

    constexpr int WARPS_M =
        BM / WM;

    constexpr int WARPS_N =
        BN / WN;

    constexpr int WARPS =
        WARPS_M * WARPS_N;

    constexpr int THREADS =
        WARPS * 32;

    constexpr int LANE_COLS =
        WN / TN;

    constexpr int A_VEC_COLS =
        BK / 4;

    constexpr int B_VEC_COLS =
        BN / 4;

    constexpr int A_VECS =
        BM * A_VEC_COLS;

    constexpr int B_VECS =
        BK * B_VEC_COLS;

    constexpr int A_CHUNKS =
        (A_VECS + THREADS - 1)
        /
        THREADS;

    constexpr int B_CHUNKS =
        (B_VECS + THREADS - 1)
        /
        THREADS;

    __shared__ float
    As[2][BM][BK + 1];

    __shared__ float
    Bs[2][BK][BN + 1];

    const int tid =
        threadIdx.x;

    const int warp_id =
        tid >> 5;

    const int lane_id =
        tid & 31;

    const int warp_row =
        warp_id / WARPS_N;

    const int warp_col =
        warp_id % WARPS_N;

    const int lane_row =
        lane_id / LANE_COLS;

    const int lane_col =
        lane_id % LANE_COLS;

    const int thread_tile_row =
        warp_row * WM
        +
        lane_row * TM;

    const int thread_tile_col =
        warp_col * WN
        +
        lane_col * TN;

    const int block_row =
        blockIdx.y * BM;

    const int block_col =
        blockIdx.x * BN;

    float acc[TM][TN] = {};

    float4 a_prefetch[A_CHUNKS];
    float4 b_prefetch[B_CHUNKS];

    pipeline_prefetch_tile<
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
        M,
        N,
        K
    );

    pipeline_commit_tile_padded<
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

        if (has_next) {
            pipeline_prefetch_tile<
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
                M,
                N,
                K
            );
        }

        compute_tile_warp_padded<
            BM,
            BN,
            BK,
            TM,
            TN,
            WM,
            WN
        >(
            acc,
            As[stage],
            Bs[stage],
            warp_row,
            warp_col,
            lane_row,
            lane_col
        );

        if (has_next) {
            const int next_stage =
                stage ^ 1;

            pipeline_commit_tile_padded<
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
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int row =
                block_row
                +
                thread_tile_row
                +
                i;

            const int col =
                block_col
                +
                thread_tile_col
                +
                j;

            if (
                row < M
                &&
                col < N
            ) {
                float value =
                    acc[i][j];

                if (
                    epilogue ==
                        EpilogueType::BIAS
                    ||
                    epilogue ==
                        EpilogueType::BIAS_SILU
                ) {
                    value +=
                        bias[col];
                }

                if (
                    epilogue ==
                    EpilogueType::BIAS_SILU
                ) {
                    value =
                        silu_device(
                            value
                        );
                }

                C[
                    static_cast<size_t>(row)
                    *
                    N
                    +
                    col
                ] =
                    value;
            }
        }
    }
}

template<
    int BM,
    int BN,
    int BK,
    int TM,
    int TN,
    int WM,
    int WN
>
void warp_pipeline_launcher(
    const float* A,
    const float* B,
    float* C,
    const float* bias,
    int M,
    int N,
    int K,
    EpilogueType epilogue,
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
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    gemm_warp_pipeline_vec4_kernel<
        BM,
        BN,
        BK,
        TM,
        TN,
        WM,
        WN
    >
    <<<grid, block, 0, stream>>>(
        A,
        B,
        C,
        bias,
        M,
        N,
        K,
        epilogue
    );
}

#define REG_TRIPLE(NAME, BM_, BN_, BK_, TM_, TN_) \
    { \
        NAME "_scalar", \
        BM_, BN_, BK_, TM_, TN_, \
        0, 0, \
        (BM_ / TM_) * (BN_ / TN_), \
        (BM_ * BK_ + BK_ * BN_) * (int)sizeof(float), \
        KernelPath::SCALAR, \
        &basic_launcher<BM_, BN_, BK_, TM_, TN_, false> \
    }, \
    { \
        NAME "_vec4", \
        BM_, BN_, BK_, TM_, TN_, \
        0, 0, \
        (BM_ / TM_) * (BN_ / TN_), \
        (BM_ * BK_ + BK_ * BN_) * (int)sizeof(float), \
        KernelPath::VEC4, \
        &basic_launcher<BM_, BN_, BK_, TM_, TN_, true> \
    }, \
    { \
        NAME "_pipe", \
        BM_, BN_, BK_, TM_, TN_, \
        0, 0, \
        (BM_ / TM_) * (BN_ / TN_), \
        2 * (BM_ * BK_ + BK_ * BN_) * (int)sizeof(float), \
        KernelPath::PIPE_VEC4, \
        &pipeline_launcher<BM_, BN_, BK_, TM_, TN_> \
    }

#define REG_WARP_PIPE(NAME, BM_, BN_, BK_, TM_, TN_, WM_, WN_) \
    { \
        NAME "_warp", \
        BM_, BN_, BK_, TM_, TN_, \
        WM_, WN_, \
        (BM_ / WM_) * (BN_ / WN_) * 32, \
        2 * (BM_ * (BK_ + 1) + BK_ * (BN_ + 1)) * (int)sizeof(float), \
        KernelPath::WARP_PIPE_VEC4, \
        &warp_pipeline_launcher<BM_, BN_, BK_, TM_, TN_, WM_, WN_> \
    }

const std::vector<KernelConfig>&
get_kernel_registry() {
    static const std::vector<KernelConfig>
    registry = {
        REG_TRIPLE("m16_n64_k16_t1x4",   16,  64, 16, 1, 4),
        REG_TRIPLE("m32_n64_k16_t2x4",   32,  64, 16, 2, 4),
        REG_TRIPLE("m64_n64_k16_t4x4",   64,  64, 16, 4, 4),
        REG_TRIPLE("m64_n128_k16_t4x8",  64, 128, 16, 4, 8),
        REG_TRIPLE("m128_n64_k16_t8x4", 128,  64, 16, 8, 4),
        REG_TRIPLE("m128_n128_k8_t8x8", 128, 128,  8, 8, 8),
        REG_TRIPLE("m64_n16_k16_t4x1",   64,  16, 16, 4, 1),
        REG_TRIPLE("m64_n32_k16_t4x2",   64,  32, 16, 4, 2),
        REG_TRIPLE("m32_n128_k8_t2x8",   32, 128,  8, 2, 8),

        /*
         * V5 warp-level candidates.
         *
         * Only the most relevant large/regular configs are added first,
         * keeping the search space interpretable.
         */
        REG_WARP_PIPE(
            "m64_n64_k16_t4x4",
            64, 64, 16,
            4, 4,
            16, 32
        ),

        REG_WARP_PIPE(
            "m64_n128_k16_t4x8",
            64, 128, 16,
            4, 8,
            16, 64
        ),

        REG_WARP_PIPE(
            "m128_n64_k16_t8x4",
            128, 64, 16,
            8, 4,
            32, 32
        ),

        REG_WARP_PIPE(
            "m128_n128_k8_t8x8",
            128, 128, 8,
            8, 8,
            32, 64
        )
    };

    return registry;
}

#undef REG_TRIPLE
#undef REG_WARP_PIPE

const KernelConfig*
find_kernel(
    const std::string& name
) {
    for (
        const auto& kernel :
        get_kernel_registry()
    ) {
        if (name == kernel.name) {
            return &kernel;
        }
    }

    return nullptr;
}

const char*
kernel_path_name(
    KernelPath path
) {
    switch (path) {
        case KernelPath::SCALAR:
            return "scalar";

        case KernelPath::VEC4:
            return "vec4";

        case KernelPath::PIPE_VEC4:
            return "pipe";

        case KernelPath::WARP_PIPE_VEC4:
            return "warp";

        default:
            return "unknown";
    }
}

bool kernel_problem_compatible(
    const KernelConfig& kernel,
    const float* A,
    const float* B,
    int M,
    int N,
    int K
) {
    (void)M;

    if (kernel.path == KernelPath::SCALAR) {
        return true;
    }

    const auto a_addr =
        reinterpret_cast<std::uintptr_t>(A);

    const auto b_addr =
        reinterpret_cast<std::uintptr_t>(B);

    const bool base_aligned =
        ((a_addr & 0xF) == 0)
        &&
        ((b_addr & 0xF) == 0);

    const bool stride_aligned =
        (K % 4 == 0)
        &&
        (N % 4 == 0);

    return
        base_aligned
        &&
        stride_aligned;
}

void launch_kernel(
    const KernelConfig& config,
    const float* A,
    const float* B,
    float* C,
    const float* bias,
    int M,
    int N,
    int K,
    EpilogueType epilogue,
    cudaStream_t stream
) {
    config.launcher(
        A,
        B,
        C,
        bias,
        M,
        N,
        K,
        epilogue,
        stream
    );

    CUDA_CHECK(
        cudaGetLastError()
    );
}
