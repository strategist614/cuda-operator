#include "autotuner.h"
#include "common.h"
#include "gemm.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

static float max_error(
    const std::vector<float>& a,
    const std::vector<float>& b
) {
    float error = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        error = std::max(
            error,
            std::abs(a[i] - b[i])
        );
    }
    return error;
}

static void cublas_gemm(
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N, M, K,
            &alpha,
            B, N,
            A, K,
            &beta,
            C, N
        )
    );
}

static float benchmark_cublas(
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    int warmup = 10,
    int repeat = 50
) {
    for (int i = 0; i < warmup; ++i) {
        cublas_gemm(
            handle,
            A, B, C,
            M, N, K
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        cublas_gemm(
            handle,
            A, B, C,
            M, N, K
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(
        cudaEventElapsedTime(
            &total_ms,
            start,
            stop
        )
    );

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / repeat;
}

static double get_tflops(
    int M, int N, int K,
    float ms
) {
    const double ops =
        2.0 *
        static_cast<double>(M) *
        static_cast<double>(N) *
        static_cast<double>(K);

    return ops / (ms * 1e-3) / 1e12;
}

static void print_registry() {
    const auto& registry = get_kernel_registry();

    std::cout
        << "\nKernel Registry\n"
        << "=========================================================\n";

    for (const auto& k : registry) {
        std::cout
            << std::left
            << std::setw(28)
            << k.name
            << "BM=" << std::setw(4) << k.BM
            << " BN=" << std::setw(4) << k.BN
            << " BK=" << std::setw(4) << k.BK
            << " TM=" << k.TM
            << " TN=" << k.TN
            << "\n";
    }
}

int main(
    int argc,
    char** argv
) {
    if (argc < 4) {
        std::cout
            << "Usage:\n"
            << "./shape_gemm M N K\n";
        return 0;
    }

    const int M = std::atoi(argv[1]);
    const int N = std::atoi(argv[2]);
    const int K = std::atoi(argv[3]);

    if (M <= 0 || N <= 0 || K <= 0) {
        std::cerr << "M/N/K must be positive.\n";
        return 1;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout
        << "GPU: " << prop.name << "\n"
        << "Shape: "
        << M << " x "
        << N << " x "
        << K << "\n";

    print_registry();

    const size_t size_A =
        static_cast<size_t>(M) * K;

    const size_t size_B =
        static_cast<size_t>(K) * N;

    const size_t size_C =
        static_cast<size_t>(M) * N;

    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float>
        dist(-0.1f, 0.1f);

    for (auto& x : h_A) {
        x = dist(rng);
    }

    for (auto& x : h_B) {
        x = dist(rng);
    }

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    float* d_ref = nullptr;
    float* d_bias = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_A,
            size_A * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_B,
            size_B * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_C,
            size_C * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_ref,
            size_C * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_bias,
            static_cast<size_t>(N) * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_A,
            h_A.data(),
            size_A * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_B,
            h_B.data(),
            size_B * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemset(
            d_bias,
            0,
            static_cast<size_t>(N) * sizeof(float)
        )
    );

    cublasHandle_t handle{};
    CUBLAS_CHECK(cublasCreate(&handle));

    TuneResult result =
        autotune_gemm(
            d_A, d_B, d_C, d_bias,
            M, N, K,
            EpilogueType::NONE,
            10,
            50
        );

    launch_kernel(
        *result.kernel,
        d_A, d_B, d_C, d_bias,
        M, N, K,
        EpilogueType::NONE
    );

    cublas_gemm(
        handle,
        d_A, d_B, d_ref,
        M, N, K
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_C(size_C);
    std::vector<float> h_ref(size_C);

    CUDA_CHECK(
        cudaMemcpy(
            h_C.data(),
            d_C,
            size_C * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            h_ref.data(),
            d_ref,
            size_C * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    const float error =
        max_error(
            h_C,
            h_ref
        );

    std::cout
        << "\nCorrectness\n"
        << "Max error = "
        << error
        << "\n";

    const float cublas_ms =
        benchmark_cublas(
            handle,
            d_A, d_B, d_ref,
            M, N, K
        );

    const double cublas_perf =
        get_tflops(
            M, N, K,
            cublas_ms
        );

    std::cout
        << "\ncuBLAS\n"
        << "latency = "
        << cublas_ms
        << " ms\n"
        << "TFLOPS = "
        << cublas_perf
        << "\n";

    std::cout
        << "\nRelative performance = "
        << result.tflops
           / cublas_perf
           * 100.0
        << "%\n";

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_ref));
    CUDA_CHECK(cudaFree(d_bias));

    return 0;
}
