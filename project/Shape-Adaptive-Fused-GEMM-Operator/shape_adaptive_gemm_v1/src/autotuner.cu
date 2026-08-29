#include "autotuner.h"
#include "common.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>

static double calc_tflops(
    int M, int N, int K,
    float ms
) {
    const double operations =
        2.0 *
        static_cast<double>(M) *
        static_cast<double>(N) *
        static_cast<double>(K);

    return operations / (ms * 1e-3) / 1e12;
}

bool kernel_supported_on_device(
    const KernelConfig& kernel
) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    if (kernel.threads > prop.maxThreadsPerBlock) {
        return false;
    }

    if (kernel.shared_memory_bytes > prop.sharedMemPerBlock) {
        return false;
    }

    return true;
}

float benchmark_kernel(
    const KernelConfig& kernel,
    const float* A,
    const float* B,
    float* C,
    const float* bias,
    int M, int N, int K,
    EpilogueType epilogue,
    int warmup,
    int repeat
) {
    for (int i = 0; i < warmup; ++i) {
        launch_kernel(
            kernel,
            A, B, C, bias,
            M, N, K,
            epilogue
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        launch_kernel(
            kernel,
            A, B, C, bias,
            M, N, K,
            epilogue
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / repeat;
}

TuneResult autotune_gemm(
    const float* A,
    const float* B,
    float* C,
    const float* bias,
    int M, int N, int K,
    EpilogueType epilogue,
    int warmup,
    int repeat
) {
    const auto& registry = get_kernel_registry();

    const KernelConfig* best_kernel = nullptr;
    float best_latency =
        std::numeric_limits<float>::max();

    std::cout
        << "\n============================================\n"
        << "Autotuning GEMM\n"
        << "M=" << M
        << " N=" << N
        << " K=" << K
        << "\n============================================\n";

    for (const auto& kernel : registry) {
        if (!kernel_supported_on_device(kernel)) {
            std::cout
                << std::left
                << std::setw(28)
                << kernel.name
                << "SKIP\n";
            continue;
        }

        const float latency =
            benchmark_kernel(
                kernel,
                A, B, C, bias,
                M, N, K,
                epilogue,
                warmup,
                repeat
            );

        const double perf =
            calc_tflops(
                M, N, K,
                latency
            );

        std::cout
            << std::left
            << std::setw(28)
            << kernel.name
            << " latency = "
            << std::setw(10)
            << latency
            << " ms"
            << "  TFLOPS = "
            << perf
            << "\n";

        if (latency < best_latency) {
            best_latency = latency;
            best_kernel = &kernel;
        }
    }

    if (best_kernel == nullptr) {
        std::cerr << "No valid GEMM kernel found.\n";
        std::exit(EXIT_FAILURE);
    }

    TuneResult result{
        best_kernel,
        best_latency,
        calc_tflops(M, N, K, best_latency)
    };

    std::cout
        << "\nBest kernel:\n"
        << "  " << result.kernel->name << "\n"
        << "  latency = " << result.latency_ms << " ms\n"
        << "  TFLOPS = " << result.tflops << "\n";

    return result;
}
