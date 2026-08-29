#include "benchmark.h"
#include "common.h"

static void cublas_gemm(
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Row-major C = A * B is computed as column-major C^T = B^T * A^T.
    CUBLAS_CHECK(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            M,
            K,
            &alpha,
            B,
            N,
            A,
            K,
            &beta,
            C,
            N
        )
    );
}

float benchmark_custom(
    KernelType type,
    const float* A,
    const float* B,
    float* C,
    const float* bias,
    int M,
    int N,
    int K,
    EpilogueType epilogue,
    int warmup,
    int repeat
) {
    for (int i = 0; i < warmup; ++i) {
        launch_gemm_forced(type, A, B, C, bias, M, N, K, epilogue);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) {
        launch_gemm_forced(type, A, B, C, bias, M, N, K, epilogue);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / repeat;
}

float benchmark_dispatch(
    const float* A,
    const float* B,
    float* C,
    const float* bias,
    int M,
    int N,
    int K,
    EpilogueType epilogue,
    int warmup,
    int repeat
) {
    KernelType type = select_kernel(M, N, K);
    return benchmark_custom(
        type, A, B, C, bias, M, N, K, epilogue, warmup, repeat
    );
}

float benchmark_cublas(
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    int warmup,
    int repeat
) {
    for (int i = 0; i < warmup; ++i) {
        cublas_gemm(handle, A, B, C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) {
        cublas_gemm(handle, A, B, C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / repeat;
}

double calculate_tflops(int M, int N, int K, float ms) {
    const double flops =
        2.0 * static_cast<double>(M) *
        static_cast<double>(N) *
        static_cast<double>(K);

    return flops / (ms * 1e-3) / 1e12;
}
