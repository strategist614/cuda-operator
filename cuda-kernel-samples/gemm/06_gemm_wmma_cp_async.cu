// nvcc -O3 -std=c++17 -arch=sm_86 06_gemm_wmma_cp_async.cu -o 06_gemm_wmma_cp_async
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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#error "06_gemm_wmma_cp_async requires sm_80 or newer"
#endif

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
constexpr int BM = 64, BN = 64, BK = 32;
constexpr int STAGES = 2;
constexpr int SKEW_HALF = 8;
constexpr int AS_LD = BK + SKEW_HALF;
constexpr int BS_LD = BN + SKEW_HALF;
constexpr int WARPS_M = 4;
constexpr int WARP_GROUPS_N = 2;
constexpr int N_TILES_PER_WARP = 2;
constexpr int WARPS = WARPS_M * WARP_GROUPS_N;
constexpr int THREADS = WARPS * 32;
constexpr int HALF_PER_CP_ASYNC = 16 / sizeof(half);
constexpr int WARMUP = 5;
constexpr int REPEATS = 20;

static_assert(BM == WARPS_M * WMMA_M);
static_assert(BN == WARP_GROUPS_N * N_TILES_PER_WARP * WMMA_N);
static_assert(BK % WMMA_K == 0);
static_assert(THREADS == 256);
static_assert(BM * (BK / HALF_PER_CP_ASYNC) == THREADS);
static_assert(BK * (BN / HALF_PER_CP_ASYNC) == THREADS);
static_assert(AS_LD % 8 == 0 && BS_LD % 8 == 0);

__device__ __forceinline__ void cp_async_16(void* shared_dst,
                                            const void* global_src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned shared_addr =
        static_cast<unsigned>(__cvta_generic_to_shared(shared_dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n" ::
            "r"(shared_addr), "l"(global_src) : "memory");
#endif
}

__device__ __forceinline__ void cp_async_commit_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}

template <int MATRIX_N, int MATRIX_K>
__device__ __forceinline__ void copy_stage_async(
    half* shared_a,
    half* shared_b,
    const half* __restrict__ A,
    const half* __restrict__ B,
    int block_row,
    int block_col,
    int bk,
    int tid) {
    constexpr int A_VECS_PER_ROW = BK / HALF_PER_CP_ASYNC;
    const int a_row = tid / A_VECS_PER_ROW;
    const int a_col = (tid % A_VECS_PER_ROW) * HALF_PER_CP_ASYNC;
    cp_async_16(
        &shared_a[a_row * AS_LD + a_col],
        &A[(block_row + a_row) * MATRIX_K + bk + a_col]);

    constexpr int B_VECS_PER_ROW = BN / HALF_PER_CP_ASYNC;
    const int b_row = tid / B_VECS_PER_ROW;
    const int b_col = (tid % B_VECS_PER_ROW) * HALF_PER_CP_ASYNC;
    cp_async_16(
        &shared_b[b_row * BS_LD + b_col],
        &B[(bk + b_row) * MATRIX_N + block_col + b_col]);

    cp_async_commit_group();
}

template <int MATRIX_M, int MATRIX_N, int MATRIX_K>
__global__ void gemm_wmma_cp_async(const half* __restrict__ A,
                                   const half* __restrict__ B,
                                   float* __restrict__ C) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    static_assert(MATRIX_M % BM == 0,
                  "M must be a multiple of the block tile height");
    static_assert(MATRIX_N % BN == 0,
                  "N must be a multiple of the block tile width");
    static_assert(MATRIX_K % BK == 0,
                  "K must be a multiple of the K-stage width");
    static_assert(MATRIX_N % 8 == 0,
                  "N must preserve WMMA store alignment");

    __shared__ __align__(32) half As[STAGES][BM][AS_LD];
    __shared__ __align__(32) half Bs[STAGES][BK][BS_LD];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
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

    // Prologue: stage 0 must be visible before the first WMMA operation.
    copy_stage_async<MATRIX_N, MATRIX_K>(
        &As[0][0][0], &Bs[0][0][0], A, B,
        block_row, block_col, 0, tid);
    cp_async_wait_all();
    __syncthreads();

    int read_stage = 0;

#pragma unroll 1
    for (int bk = 0; bk < MATRIX_K; bk += BK) {
        const int next_bk = bk + BK;
        const int write_stage = read_stage ^ 1;
        const bool has_next = next_bk < MATRIX_K;

        // The copy runs asynchronously while this warp consumes read_stage.
        if (has_next) {
            copy_stage_async<MATRIX_N, MATRIX_K>(
                &As[write_stage][0][0], &Bs[write_stage][0][0], A, B,
                block_row, block_col, next_bk, tid);
        }

#pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a,
                           WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::row_major>
                a_frag;
            wmma::load_matrix_sync(
                a_frag, &As[read_stage][warp_m][kk], AS_LD);

#pragma unroll
            for (int tile_n = 0; tile_n < N_TILES_PER_WARP; ++tile_n) {
                wmma::fragment<wmma::matrix_b,
                               WMMA_M, WMMA_N, WMMA_K,
                               half, wmma::row_major>
                    b_frag;
                const int warp_n =
                    warp_n_base + tile_n * WMMA_N;
                wmma::load_matrix_sync(
                    b_frag, &Bs[read_stage][kk][warp_n], BS_LD);
                wmma::mma_sync(c_frag[tile_n], a_frag, b_frag,
                               c_frag[tile_n]);
            }
        }

        if (has_next) {
            // Each thread waits for its committed group; the block barrier
            // makes the completed stage visible to every compute warp.
            cp_async_wait_all();
            __syncthreads();
            read_stage = write_stage;
        }
    }

#pragma unroll
    for (int tile_n = 0; tile_n < N_TILES_PER_WARP; ++tile_n) {
        const int global_row = block_row + warp_m;
        const int global_col =
            block_col + warp_n_base + tile_n * WMMA_N;
        wmma::store_matrix_sync(
            &C[global_row * MATRIX_N + global_col],
            c_frag[tile_n], MATRIX_N, wmma::mem_row_major);
    }
#endif
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
              << "kernel=wmma_cp_async_fp16_acc_fp32\n"
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
    const dim3 grid(N / BN, M / BM);
    for (int i = 0; i < WARMUP; ++i) {
        gemm_wmma_cp_async<M, N, K><<<grid, block>>>(d_A, d_B, d_C);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < REPEATS; ++i) {
        gemm_wmma_cp_async<M, N, K><<<grid, block>>>(d_A, d_B, d_C);
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
