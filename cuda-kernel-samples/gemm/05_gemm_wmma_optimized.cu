// nvcc -O3 -std=c++17 -arch=sm_75 05_gemm_wmma_optimized.cu -o 05_gemm_wmma_optimized
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error_ = (call);                                            \
        if (error_ != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(error_)          \
                      << " at " << __FILE__ << ':' << __LINE__ << '\n';        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

using namespace nvcuda;

constexpr int M = 512;
constexpr int N = 512;
constexpr int K = 512;
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int BM = 64, BN = 64, BK = 16;
constexpr int WARPS_M = 4;
constexpr int WARP_GROUPS_N = 2;
constexpr int N_TILES_PER_WARP = 2;
constexpr int WARPS = WARPS_M * WARP_GROUPS_N;
constexpr int THREADS = WARPS * 32;
constexpr int HALF_PER_INT4 = sizeof(int4) / sizeof(half);
constexpr int WARMUP = 5;
constexpr int REPEATS = 20;

static_assert(BM == WARPS_M * WMMA_M);
static_assert(BN == WARP_GROUPS_N * N_TILES_PER_WARP * WMMA_N);
static_assert(BK == WMMA_K);
static_assert(THREADS == 256);
static_assert(BK % HALF_PER_INT4 == 0 && BN % HALF_PER_INT4 == 0);

__global__ void gemm_wmma_optimized(const half* __restrict__ A,
                                    const half* __restrict__ B,
                                    float* __restrict__ C,
                                    int m, int n, int k_size) {
    __shared__ __align__(16) half As[BM][BK];
    __shared__ __align__(16) half Bs[BK][BN];
    // Only edge tiles use this buffer. Each warp owns one 16 x 16 region.
    __shared__ __align__(32) float edge_tiles[WARPS][WMMA_M * WMMA_N];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int warp_row = warp_id / WARP_GROUPS_N;
    const int warp_col_group = warp_id % WARP_GROUPS_N;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int warp_m = warp_row * WMMA_M;
    const int warp_n_base =
        warp_col_group * N_TILES_PER_WARP * WMMA_N;

    wmma::fragment<wmma::accumulator,
                   WMMA_M, WMMA_N, WMMA_K, float>
        c_frag[N_TILES_PER_WARP];

#pragma unroll
    for (int tile_n = 0; tile_n < N_TILES_PER_WARP; ++tile_n) {
        wmma::fill_fragment(c_frag[tile_n], 0.0f);
    }

    for (int bk = 0; bk < k_size; bk += BK) {
        const bool full_tile =
            block_row + BM <= m && block_col + BN <= n &&
            bk + BK <= k_size && k_size % HALF_PER_INT4 == 0 &&
            n % HALF_PER_INT4 == 0;

        if (full_tile) {
            // int4 moves eight half values (16 bytes) per instruction.
            constexpr int A_VECS_PER_ROW = BK / HALF_PER_INT4;
            constexpr int A_VECS = BM * A_VECS_PER_ROW;
            for (int i = tid; i < A_VECS; i += blockDim.x) {
                const int r = i / A_VECS_PER_ROW;
                const int c = (i % A_VECS_PER_ROW) * HALF_PER_INT4;
                *reinterpret_cast<int4*>(&As[r][c]) =
                    *reinterpret_cast<const int4*>(
                        &A[(block_row + r) * k_size + bk + c]);
            }

            constexpr int B_VECS_PER_ROW = BN / HALF_PER_INT4;
            constexpr int B_VECS = BK * B_VECS_PER_ROW;
            for (int i = tid; i < B_VECS; i += blockDim.x) {
                const int r = i / B_VECS_PER_ROW;
                const int c = (i % B_VECS_PER_ROW) * HALF_PER_INT4;
                *reinterpret_cast<int4*>(&Bs[r][c]) =
                    *reinterpret_cast<const int4*>(
                        &B[(bk + r) * n + block_col + c]);
            }
        } else {
            for (int i = tid; i < BM * BK; i += blockDim.x) {
                const int r = i / BK;
                const int c = i % BK;
                const int global_row = block_row + r;
                const int global_col = bk + c;
                As[r][c] = (global_row < m && global_col < k_size)
                               ? A[global_row * k_size + global_col]
                               : __float2half(0.0f);
            }
            for (int i = tid; i < BK * BN; i += blockDim.x) {
                const int r = i / BN;
                const int c = i % BN;
                const int global_row = bk + r;
                const int global_col = block_col + c;
                Bs[r][c] = (global_row < k_size && global_col < n)
                               ? B[global_row * n + global_col]
                               : __float2half(0.0f);
            }
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a,
                       WMMA_M, WMMA_N, WMMA_K,
                       half, wmma::row_major>
            a_frag;
        wmma::load_matrix_sync(a_frag, &As[warp_m][0], BK);

#pragma unroll
        for (int tile_n = 0; tile_n < N_TILES_PER_WARP; ++tile_n) {
            wmma::fragment<wmma::matrix_b,
                           WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::row_major>
                b_frag;
            const int warp_n = warp_n_base + tile_n * WMMA_N;
            wmma::load_matrix_sync(b_frag, &Bs[0][warp_n], BN);
            wmma::mma_sync(c_frag[tile_n], a_frag, b_frag,
                           c_frag[tile_n]);
        }

        // All warps must finish reading before the next K tile overwrites As/Bs.
        __syncthreads();
    }

#pragma unroll
    for (int tile_n = 0; tile_n < N_TILES_PER_WARP; ++tile_n) {
        const int global_row = block_row + warp_m;
        const int global_col =
            block_col + warp_n_base + tile_n * WMMA_N;
        const bool full_output_tile =
            global_row + WMMA_M <= m && global_col + WMMA_N <= n &&
            n % 8 == 0;

        if (full_output_tile) {
            // The common path avoids the 64 x 64 shared-memory C staging in 04.
            wmma::store_matrix_sync(
                &C[global_row * n + global_col], c_frag[tile_n], n,
                wmma::mem_row_major);
        } else {
            float* tile = edge_tiles[warp_id];
            wmma::store_matrix_sync(tile, c_frag[tile_n], WMMA_N,
                                    wmma::mem_row_major);
            __syncwarp();

            for (int i = lane; i < WMMA_M * WMMA_N; i += 32) {
                const int r = i / WMMA_N;
                const int c = i % WMMA_N;
                if (global_row + r < m && global_col + c < n) {
                    C[(global_row + r) * n + global_col + c] = tile[i];
                }
            }
            // A warp reuses its scratch region for the next output tile.
            __syncwarp();
        }
    }
}

void fill_inputs(std::vector<float>& A, std::vector<float>& B) {
    uint32_t state = 20260822u;
    auto next_value = [&state]() {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>(
                   static_cast<int>((state >> 8) % 2001u) - 1000) /
               1000.0f;
    };
    for (float& value : A) value = next_value();
    for (float& value : B) value = next_value();
}

std::vector<double> cpu_reference(const std::vector<float>& A,
                                  const std::vector<float>& B) {
    std::vector<double> reference(M * N, 0.0);
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += static_cast<double>(A[row * K + k]) *
                       static_cast<double>(B[k * N + col]);
            }
            reference[row * N + col] = sum;
        }
    }
    return reference;
}

bool report(const std::vector<float>& output,
            const std::vector<double>& reference, float average_ms) {
    double max_abs_error = 0.0;
    double max_reference = 0.0;
    double output_checksum = 0.0;
    double reference_checksum = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        max_abs_error = std::max(
            max_abs_error,
            std::abs(static_cast<double>(output[i]) - reference[i]));
        max_reference = std::max(max_reference, std::abs(reference[i]));
        output_checksum += output[i];
        reference_checksum += reference[i];
    }
    const double relative_linf =
        max_abs_error / std::max(max_reference, 1.0e-30);
    const double gflops =
        (2.0 * M * N * K) / (average_ms * 1.0e6);

    std::cout << std::fixed << std::setprecision(8)
              << "kernel=wmma_optimized_fp16_acc_fp32\n"
              << "shape=" << M << 'x' << N << 'x' << K << '\n'
              << "max_abs_error=" << max_abs_error << '\n'
              << "relative_linf=" << relative_linf << '\n'
              << "reference_checksum=" << reference_checksum << '\n'
              << "output_checksum=" << output_checksum << '\n'
              << "average_ms=" << average_ms << '\n'
              << "gflops=" << gflops << '\n'
              << "first_values=";
    for (int i = 0; i < 8; ++i) {
        std::cout << (i ? "," : "") << output[i];
    }
    std::cout << '\n'
              << "status="
              << (max_abs_error <= 5.0e-2 ? "PASS" : "FAIL") << '\n';
    return max_abs_error <= 5.0e-2;
}

int main() {
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N);
    fill_inputs(h_A, h_B);

    std::vector<half> h_A_half(M * K), h_B_half(K * N);
    for (size_t i = 0; i < h_A.size(); ++i) {
        h_A_half[i] = __float2half_rn(h_A[i]);
    }
    for (size_t i = 0; i < h_B.size(); ++i) {
        h_B_half[i] = __float2half_rn(h_B[i]);
    }

    half *d_A = nullptr, *d_B = nullptr;
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, h_A_half.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, h_B_half.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, h_C.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A_half.data(),
                          h_A_half.size() * sizeof(half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_half.data(),
                          h_B_half.size() * sizeof(half),
                          cudaMemcpyHostToDevice));

    const dim3 block(THREADS);
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    for (int i = 0; i < WARMUP; ++i) {
        gemm_wmma_optimized<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < REPEATS; ++i) {
        gemm_wmma_optimized<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Reference uses the original FP32 inputs, so the reported error includes
    // the FP16 input quantization used by the Tensor Core path.
    const std::vector<double> reference = cpu_reference(h_A, h_B);
    const bool passed = report(h_C, reference, total_ms / REPEATS);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
