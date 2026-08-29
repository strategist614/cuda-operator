#include "gemm.h"
#include "common.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

__device__ __forceinline__ float silu_device(float x) {
    return x / (1.0f + expf(-x));
}

template<int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const float* __restrict__ bias,
    int M, int N, int K,
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

    float acc[TM][TN];

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int bk = 0; bk < K; bk += BK) {
        for (int idx = tid; idx < BM * BK; idx += blockDim.x) {
            int r = idx / BK;
            int c = idx % BK;

            int gr = block_row + r;
            int gc = bk + c;

            As[r][c] =
                (gr < M && gc < K)
                ? A[static_cast<size_t>(gr) * K + gc]
                : 0.0f;
        }

        for (int idx = tid; idx < BK * BN; idx += blockDim.x) {
            int r = idx / BN;
            int c = idx % BN;

            int gr = bk + r;
            int gc = block_col + c;

            Bs[r][c] =
                (gr < K && gc < N)
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
            int row = block_row + thread_row * TM + i;
            int col = block_col + thread_col * TN + j;

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

template<int BM, int BN, int BK, int TM, int TN>
void kernel_launcher(
    const float* A,
    const float* B,
    float* C,
    const float* bias,
    int M, int N, int K,
    EpilogueType epilogue,
    cudaStream_t stream
) {
    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(BN % TN == 0, "BN must be divisible by TN");

    constexpr int THREADS = (BM / TM) * (BN / TN);
    static_assert(THREADS <= 1024, "Too many threads");

    dim3 block(THREADS);
    dim3 grid((N + BN - 1) / BN,
              (M + BM - 1) / BM);

    gemm_kernel<BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(
            A, B, C, bias, M, N, K, epilogue
        );
}

const std::vector<KernelConfig>& get_kernel_registry() {
    static const std::vector<KernelConfig> registry = {
        {
            "m16_n64_k16_t1x4",
            16, 64, 16, 1, 4, 256,
            (16 * 16 + 16 * 64) * sizeof(float),
            &kernel_launcher<16, 64, 16, 1, 4>
        },
        {
            "m32_n64_k16_t2x4",
            32, 64, 16, 2, 4, 256,
            (32 * 16 + 16 * 64) * sizeof(float),
            &kernel_launcher<32, 64, 16, 2, 4>
        },
        {
            "m64_n64_k16_t4x4",
            64, 64, 16, 4, 4, 256,
            (64 * 16 + 16 * 64) * sizeof(float),
            &kernel_launcher<64, 64, 16, 4, 4>
        },
        {
            "m64_n128_k16_t4x8",
            64, 128, 16, 4, 8, 256,
            (64 * 16 + 16 * 128) * sizeof(float),
            &kernel_launcher<64, 128, 16, 4, 8>
        },
        {
            "m128_n64_k16_t8x4",
            128, 64, 16, 8, 4, 256,
            (128 * 16 + 16 * 64) * sizeof(float),
            &kernel_launcher<128, 64, 16, 8, 4>
        },
        {
            "m128_n128_k8_t8x8",
            128, 128, 8, 8, 8, 256,
            (128 * 8 + 8 * 128) * sizeof(float),
            &kernel_launcher<128, 128, 8, 8, 8>
        },
        {
            "m64_n16_k16_t4x1",
            64, 16, 16, 4, 1, 256,
            (64 * 16 + 16 * 16) * sizeof(float),
            &kernel_launcher<64, 16, 16, 4, 1>
        },
        {
            "m64_n32_k16_t4x2",
            64, 32, 16, 4, 2, 256,
            (64 * 16 + 16 * 32) * sizeof(float),
            &kernel_launcher<64, 32, 16, 4, 2>
        },
        {
            "m32_n128_k8_t2x8",
            32, 128, 8, 2, 8, 256,
            (32 * 8 + 8 * 128) * sizeof(float),
            &kernel_launcher<32, 128, 8, 2, 8>
        }
    };

    return registry;
}

const KernelConfig* find_kernel(const std::string& name) {
    const auto& registry = get_kernel_registry();
    for (const auto& kernel : registry) {
        if (name == kernel.name) {
            return &kernel;
        }
    }
    return nullptr;
}

const KernelConfig& heuristic_select_kernel(int M, int N, int K) {
    (void)K;
    const auto& registry = get_kernel_registry();

    if (M <= 16) return registry[0];
    if (M <= 32) return registry[1];
    if (N <= 16) return registry[6];
    if (N <= 32) return registry[7];

    return registry[2];
}

void launch_kernel(
    const KernelConfig& config,
    const float* A,
    const float* B,
    float* C,
    const float* bias,
    int M, int N, int K,
    EpilogueType epilogue,
    cudaStream_t stream
) {
    config.launcher(
        A, B, C, bias,
        M, N, K,
        epilogue,
        stream
    );

    CUDA_CHECK(cudaGetLastError());
}
