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

template<
    int BM,
    int BN,
    int BK,
    int TM,
    int TN,
    bool VEC4
>
__global__
void gemm_kernel(
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

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    constexpr int THREAD_COLS =
        BN / TN;

    const int tid =
        threadIdx.x;

    const int thread_row =
        tid / THREAD_COLS;

    const int thread_col =
        tid % THREAD_COLS;

    const int block_row =
        blockIdx.y * BM;

    const int block_col =
        blockIdx.x * BN;

    float acc[TM][TN];

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (
        int bk = 0;
        bk < K;
        bk += BK
    ) {
        /*
         * ====================================================
         * Global -> Shared
         *
         * VEC4=false:
         *   scalar FP32 loads.
         *
         * VEC4=true:
         *   each logical load moves 4 adjacent FP32 values
         *   with one float4 global load on the fast path.
         *
         * Tail rows/K tiles still use zero fill / scalar-safe
         * handling, so correctness is preserved.
         * ====================================================
         */

        if constexpr (VEC4) {
            constexpr int A_VEC_COLS =
                BK / 4;

            constexpr int A_VECS =
                BM * A_VEC_COLS;

            for (
                int idx = tid;
                idx < A_VECS;
                idx += blockDim.x
            ) {
                const int r =
                    idx / A_VEC_COLS;

                const int c4 =
                    idx % A_VEC_COLS;

                const int gr =
                    block_row + r;

                const int gc =
                    bk + c4 * 4;

                float4 v{
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f
                };

                if (
                    gr < M &&
                    gc + 3 < K
                ) {
                    const float* ptr =
                        A
                        +
                        static_cast<size_t>(gr) * K
                        +
                        gc;

                    v =
                        *reinterpret_cast<
                            const float4*
                        >(ptr);
                } else if (
                    gr < M &&
                    gc < K
                ) {
                    if (gc + 0 < K) {
                        v.x =
                            A[
                                static_cast<size_t>(gr) * K
                                + gc + 0
                            ];
                    }

                    if (gc + 1 < K) {
                        v.y =
                            A[
                                static_cast<size_t>(gr) * K
                                + gc + 1
                            ];
                    }

                    if (gc + 2 < K) {
                        v.z =
                            A[
                                static_cast<size_t>(gr) * K
                                + gc + 2
                            ];
                    }

                    if (gc + 3 < K) {
                        v.w =
                            A[
                                static_cast<size_t>(gr) * K
                                + gc + 3
                            ];
                    }
                }

                As[r][c4 * 4 + 0] = v.x;
                As[r][c4 * 4 + 1] = v.y;
                As[r][c4 * 4 + 2] = v.z;
                As[r][c4 * 4 + 3] = v.w;
            }

            constexpr int B_VEC_COLS =
                BN / 4;

            constexpr int B_VECS =
                BK * B_VEC_COLS;

            for (
                int idx = tid;
                idx < B_VECS;
                idx += blockDim.x
            ) {
                const int r =
                    idx / B_VEC_COLS;

                const int c4 =
                    idx % B_VEC_COLS;

                const int gr =
                    bk + r;

                const int gc =
                    block_col + c4 * 4;

                float4 v{
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f
                };

                if (
                    gr < K &&
                    gc + 3 < N
                ) {
                    const float* ptr =
                        B
                        +
                        static_cast<size_t>(gr) * N
                        +
                        gc;

                    v =
                        *reinterpret_cast<
                            const float4*
                        >(ptr);
                } else if (
                    gr < K &&
                    gc < N
                ) {
                    if (gc + 0 < N) {
                        v.x =
                            B[
                                static_cast<size_t>(gr) * N
                                + gc + 0
                            ];
                    }

                    if (gc + 1 < N) {
                        v.y =
                            B[
                                static_cast<size_t>(gr) * N
                                + gc + 1
                            ];
                    }

                    if (gc + 2 < N) {
                        v.z =
                            B[
                                static_cast<size_t>(gr) * N
                                + gc + 2
                            ];
                    }

                    if (gc + 3 < N) {
                        v.w =
                            B[
                                static_cast<size_t>(gr) * N
                                + gc + 3
                            ];
                    }
                }

                Bs[r][c4 * 4 + 0] = v.x;
                Bs[r][c4 * 4 + 1] = v.y;
                Bs[r][c4 * 4 + 2] = v.z;
                Bs[r][c4 * 4 + 3] = v.w;
            }

        } else {
            for (
                int idx = tid;
                idx < BM * BK;
                idx += blockDim.x
            ) {
                const int r =
                    idx / BK;

                const int c =
                    idx % BK;

                const int gr =
                    block_row + r;

                const int gc =
                    bk + c;

                As[r][c] =
                    (gr < M && gc < K)
                    ?
                    A[
                        static_cast<size_t>(gr) * K
                        + gc
                    ]
                    :
                    0.0f;
            }

            for (
                int idx = tid;
                idx < BK * BN;
                idx += blockDim.x
            ) {
                const int r =
                    idx / BN;

                const int c =
                    idx % BN;

                const int gr =
                    bk + r;

                const int gc =
                    block_col + c;

                Bs[r][c] =
                    (gr < K && gc < N)
                    ?
                    B[
                        static_cast<size_t>(gr) * N
                        + gc
                    ]
                    :
                    0.0f;
            }
        }

        __syncthreads();

#pragma unroll
        for (
            int kk = 0;
            kk < BK;
            ++kk
        ) {
            float a_frag[TM];
            float b_frag[TN];

#pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_frag[i] =
                    As[
                        thread_row * TM + i
                    ][kk];
            }

#pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_frag[j] =
                    Bs[
                        kk
                    ][
                        thread_col * TN + j
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

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int row =
                block_row
                +
                thread_row * TM
                +
                i;

            const int col =
                block_col
                +
                thread_col * TN
                +
                j;

            if (
                row < M &&
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
                        silu_device(value);
                }

                C[
                    static_cast<size_t>(row) * N
                    +
                    col
                ] = value;
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
void kernel_launcher(
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
    static_assert(
        BM % TM == 0,
        "BM must be divisible by TM."
    );

    static_assert(
        BN % TN == 0,
        "BN must be divisible by TN."
    );

    constexpr int THREADS =
        (BM / TM)
        *
        (BN / TN);

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

    gemm_kernel<
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

#define REGISTER_PAIR(NAME, BM_, BN_, BK_, TM_, TN_) \
    { \
        NAME "_scalar", \
        BM_, BN_, BK_, TM_, TN_, \
        (BM_ / TM_) * (BN_ / TN_), \
        (BM_ * BK_ + BK_ * BN_) * (int)sizeof(float), \
        MemoryPath::SCALAR, \
        &kernel_launcher<BM_, BN_, BK_, TM_, TN_, false> \
    }, \
    { \
        NAME "_vec4", \
        BM_, BN_, BK_, TM_, TN_, \
        (BM_ / TM_) * (BN_ / TN_), \
        (BM_ * BK_ + BK_ * BN_) * (int)sizeof(float), \
        MemoryPath::VEC4, \
        &kernel_launcher<BM_, BN_, BK_, TM_, TN_, true> \
    }

const std::vector<KernelConfig>&
get_kernel_registry() {
    static const std::vector<KernelConfig>
    registry = {
        REGISTER_PAIR(
            "m16_n64_k16_t1x4",
            16, 64, 16, 1, 4
        ),

        REGISTER_PAIR(
            "m32_n64_k16_t2x4",
            32, 64, 16, 2, 4
        ),

        REGISTER_PAIR(
            "m64_n64_k16_t4x4",
            64, 64, 16, 4, 4
        ),

        REGISTER_PAIR(
            "m64_n128_k16_t4x8",
            64, 128, 16, 4, 8
        ),

        REGISTER_PAIR(
            "m128_n64_k16_t8x4",
            128, 64, 16, 8, 4
        ),

        REGISTER_PAIR(
            "m128_n128_k8_t8x8",
            128, 128, 8, 8, 8
        ),

        REGISTER_PAIR(
            "m64_n16_k16_t4x1",
            64, 16, 16, 4, 1
        ),

        REGISTER_PAIR(
            "m64_n32_k16_t4x2",
            64, 32, 16, 4, 2
        ),

        REGISTER_PAIR(
            "m32_n128_k8_t2x8",
            32, 128, 8, 2, 8
        )
    };

    return registry;
}

#undef REGISTER_PAIR

const KernelConfig*
find_kernel(
    const std::string& name
) {
    for (
        const auto& kernel :
        get_kernel_registry()
    ) {
        if (
            name ==
            kernel.name
        ) {
            return &kernel;
        }
    }

    return nullptr;
}

const char*
memory_path_name(
    MemoryPath path
) {
    switch (path) {
        case MemoryPath::SCALAR:
            return "scalar";

        case MemoryPath::VEC4:
            return "vec4";

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

    if (
        kernel.memory_path ==
        MemoryPath::SCALAR
    ) {
        return true;
    }

    /*
     * cudaMalloc allocations are naturally highly aligned,
     * but explicitly check the base addresses and row strides.
     *
     * A row address:
     *   A + row*K
     *
     * B row address:
     *   B + row*N
     *
     * For every row to preserve 16-byte alignment,
     * K and N must both be multiples of 4 FP32 elements.
     */
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
