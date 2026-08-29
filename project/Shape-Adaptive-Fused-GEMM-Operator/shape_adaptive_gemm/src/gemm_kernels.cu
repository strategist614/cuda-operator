#include "gemm.h"
#include "common.h"
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float silu_device(float x) {
    return x / (1.0f + expf(-x));
}

template<int BM, int BN, int BK, int TM, int TN>
__global__ void tiled_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const float* __restrict__ bias,
    int M, int N, int K,
    EpilogueType epilogue
) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    constexpr int THREAD_ROWS = BM / TM;
    constexpr int THREAD_COLS = BN / TN;

    const int tid = threadIdx.x;
    const int thread_row = tid / THREAD_COLS;
    const int thread_col = tid % THREAD_COLS;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[TM][TN];

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    const int threads = blockDim.x;

    for (int bk = 0; bk < K; bk += BK) {
        for (int idx = tid; idx < BM * BK; idx += threads) {
            const int r = idx / BK;
            const int c = idx % BK;

            const int gr = block_row + r;
            const int gc = bk + c;

            As[r][c] = (gr < M && gc < K)
                     ? A[static_cast<size_t>(gr) * K + gc]
                     : 0.0f;
        }

        for (int idx = tid; idx < BK * BN; idx += threads) {
            const int r = idx / BN;
            const int c = idx % BN;

            const int gr = bk + r;
            const int gc = block_col + c;

            Bs[r][c] = (gr < K && gc < N)
                     ? B[static_cast<size_t>(gr) * N + gc]
                     : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a_frag[TM];
            float b_frag[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_frag[i] = As[thread_row * TM + i][kk];
            }

            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_frag[j] = Bs[kk][thread_col * TN + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int row = block_row + thread_row * TM + i;
            const int col = block_col + thread_col * TN + j;

            if (row < M && col < N) {
                float v = acc[i][j];

                if (epilogue == EpilogueType::BIAS ||
                    epilogue == EpilogueType::BIAS_SILU) {
                    v += bias[col];
                }

                if (epilogue == EpilogueType::BIAS_SILU) {
                    v = silu_device(v);
                }

                C[static_cast<size_t>(row) * N + col] = v;
            }
        }
    }
}

template<int BM, int BN, int BK, int TM, int TN>
static void launch_config(
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
    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(BN % TN == 0, "BN must be divisible by TN");

    constexpr int THREADS = (BM / TM) * (BN / TN);
    static_assert(THREADS <= 1024, "too many threads");

    dim3 block(THREADS);
    dim3 grid((N + BN - 1) / BN,
              (M + BM - 1) / BM);

    tiled_gemm_kernel<BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(
            A, B, C, bias, M, N, K, epilogue
        );
}

KernelType select_kernel(int M, int N, int K) {
    (void)K;

    if (M <= 32) {
        return KernelType::SMALL_M;
    }

    if (N <= 32) {
        return KernelType::SKINNY_N;
    }

    return KernelType::REGULAR;
}

std::string kernel_name(KernelType type) {
    switch (type) {
        case KernelType::SMALL_M:  return "small_m";
        case KernelType::SKINNY_N: return "skinny_n";
        case KernelType::REGULAR:  return "regular";
        default:                   return "unknown";
    }
}

void launch_gemm_forced(
    KernelType type,
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
    switch (type) {
        case KernelType::SMALL_M:
            // 256 threads, good for small-M shapes.
            launch_config<16, 64, 16, 1, 4>(
                A, B, C, bias, M, N, K, epilogue, stream
            );
            break;

        case KernelType::SKINNY_N:
            // 256 threads, emphasizes M direction when N is narrow.
            launch_config<64, 16, 16, 4, 1>(
                A, B, C, bias, M, N, K, epilogue, stream
            );
            break;

        case KernelType::REGULAR:
        default:
            // 256 threads, each thread computes a 4x4 register tile.
            launch_config<64, 64, 16, 4, 4>(
                A, B, C, bias, M, N, K, epilogue, stream
            );
            break;
    }

    CUDA_CHECK(cudaGetLastError());
}

void launch_gemm(
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
    launch_gemm_forced(
        select_kernel(M, N, K),
        A, B, C, bias,
        M, N, K,
        epilogue,
        stream
    );
}
