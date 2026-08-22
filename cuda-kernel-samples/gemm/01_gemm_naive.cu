// nvcc -O3 -std=c++17 -arch=sm_75 01_gemm_naive.cu -o 01_gemm_naive
#include <cuda_runtime.h>

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

constexpr int M = 512;
constexpr int N = 512;
constexpr int K = 512;
constexpr int WARMUP = 5;
constexpr int REPEATS = 20;

__global__ void gemm_naive(const float* A, const float* B, float* C,
                           int m, int n, int k_size) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;

    float sum = 0.0f;
    for (int k = 0; k < k_size; ++k) {
        sum += A[row * k_size + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
}

void fill_inputs(std::vector<float>& A, std::vector<float>& B) {
    uint32_t state = 20260822u;
    auto next_value = [&state]() {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>(static_cast<int>((state >> 8) % 2001u) - 1000)
               / 1000.0f;
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
        max_abs_error = std::max(max_abs_error,
                                 std::abs(static_cast<double>(output[i]) - reference[i]));
        max_reference = std::max(max_reference, std::abs(reference[i]));
        output_checksum += output[i];
        reference_checksum += reference[i];
    }
    const double relative_linf = max_abs_error / std::max(max_reference, 1.0e-30);
    const double gflops = (2.0 * M * N * K) / (average_ms * 1.0e6);

    std::cout << std::fixed << std::setprecision(8)
              << "kernel=naive_fp32\n"
              << "shape=" << M << 'x' << N << 'x' << K << '\n'
              << "max_abs_error=" << max_abs_error << '\n'
              << "relative_linf=" << relative_linf << '\n'
              << "reference_checksum=" << reference_checksum << '\n'
              << "output_checksum=" << output_checksum << '\n'
              << "average_ms=" << average_ms << '\n'
              << "gflops=" << gflops << '\n'
              << "first_values=";
    for (int i = 0; i < 8; ++i) std::cout << (i ? "," : "") << output[i];
    std::cout << '\n'
              << "status=" << (max_abs_error <= 1.0e-3 ? "PASS" : "FAIL") << '\n';
    return max_abs_error <= 1.0e-3;
}

int main() {
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N);
    fill_inputs(h_A, h_B);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, h_B.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, h_C.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 block(16, 16);
    const dim3 grid((N + block.x - 1) / block.x,
                    (M + block.y - 1) / block.y);
    for (int i = 0; i < WARMUP; ++i) gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < REPEATS; ++i) gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost));

    const std::vector<double> reference = cpu_reference(h_A, h_B);
    const bool passed = report(h_C, reference, total_ms / REPEATS);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
