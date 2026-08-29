#include "autotuner.h"
#include "common.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

double gemm_tflops(
    int M,
    int N,
    int K,
    float ms
) {
    const double operations =
        2.0
        *
        static_cast<double>(M)
        *
        static_cast<double>(N)
        *
        static_cast<double>(K);

    return
        operations
        /
        (ms * 1e-3)
        /
        1e12;
}

bool kernel_supported_on_device(
    const KernelConfig& kernel
) {
    int device = 0;

    CUDA_CHECK(
        cudaGetDevice(&device)
    );

    cudaDeviceProp prop{};

    CUDA_CHECK(
        cudaGetDeviceProperties(
            &prop,
            device
        )
    );

    if (
        kernel.threads
        >
        prop.maxThreadsPerBlock
    ) {
        return false;
    }

    if (
        kernel.shared_memory_bytes
        >
        prop.sharedMemPerBlock
    ) {
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
    int M,
    int N,
    int K,
    EpilogueType epilogue,
    int warmup,
    int repeat,
    int groups
) {
    if (
        !kernel_problem_compatible(
            kernel,
            A,
            B,
            M,
            N,
            K
        )
    ) {
        return
            std::numeric_limits<float>::infinity();
    }

    for (
        int i = 0;
        i < warmup;
        ++i
    ) {
        launch_kernel(
            kernel,
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

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );

    std::vector<float> samples;

    samples.reserve(
        groups
    );

    for (
        int g = 0;
        g < groups;
        ++g
    ) {
        cudaEvent_t start{};
        cudaEvent_t stop{};

        CUDA_CHECK(
            cudaEventCreate(
                &start
            )
        );

        CUDA_CHECK(
            cudaEventCreate(
                &stop
            )
        );

        CUDA_CHECK(
            cudaEventRecord(
                start
            )
        );

        for (
            int i = 0;
            i < repeat;
            ++i
        ) {
            launch_kernel(
                kernel,
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

        CUDA_CHECK(
            cudaEventRecord(
                stop
            )
        );

        CUDA_CHECK(
            cudaEventSynchronize(
                stop
            )
        );

        float total_ms =
            0.0f;

        CUDA_CHECK(
            cudaEventElapsedTime(
                &total_ms,
                start,
                stop
            )
        );

        CUDA_CHECK(
            cudaEventDestroy(
                start
            )
        );

        CUDA_CHECK(
            cudaEventDestroy(
                stop
            )
        );

        samples.push_back(
            total_ms
            /
            repeat
        );
    }

    std::sort(
        samples.begin(),
        samples.end()
    );

    return
        samples[
            samples.size() / 2
        ];
}

TuneResult autotune_gemm(
    const float* A,
    const float* B,
    float* C,
    const float* bias,
    int M,
    int N,
    int K,
    EpilogueType epilogue,
    int warmup,
    int repeat,
    int groups
) {
    const KernelConfig*
    best_kernel = nullptr;

    float best_latency =
        std::numeric_limits<float>::max();

    std::cout
        << "\n============================================\n"
        << "Autotuning GEMM V4\n"
        << "M=" << M
        << " N=" << N
        << " K=" << K
        << "\n"
        << "benchmark="
        << groups
        << " groups x "
        << repeat
        << " launches, median selected\n"
        << "============================================\n";

    for (
        const auto& kernel :
        get_kernel_registry()
    ) {
        if (
            !kernel_supported_on_device(
                kernel
            )
        ) {
            std::cout
                << std::left
                << std::setw(34)
                << kernel.name
                << "SKIP(device)\n";

            continue;
        }

        if (
            !kernel_problem_compatible(
                kernel,
                A,
                B,
                M,
                N,
                K
            )
        ) {
            std::cout
                << std::left
                << std::setw(34)
                << kernel.name
                << "SKIP(alignment/stride)\n";

            continue;
        }

        const float latency =
            benchmark_kernel(
                kernel,
                A,
                B,
                C,
                bias,
                M,
                N,
                K,
                epilogue,
                warmup,
                repeat,
                groups
            );

        const double perf =
            gemm_tflops(
                M,
                N,
                K,
                latency
            );

        std::cout
            << std::left
            << std::setw(34)
            << kernel.name
            << " path="
            << std::setw(6)
            << kernel_path_name(
                kernel.path
            )
            << " latency="
            << std::setw(10)
            << latency
            << " ms"
            << " TFLOPS="
            << perf
            << "\n";

        if (
            latency
            <
            best_latency
        ) {
            best_latency =
                latency;

            best_kernel =
                &kernel;
        }
    }

    if (
        best_kernel ==
        nullptr
    ) {
        std::cerr
            << "No valid GEMM kernel found.\n";

        std::exit(
            EXIT_FAILURE
        );
    }

    TuneResult result{
        best_kernel,
        best_latency,
        gemm_tflops(
            M,
            N,
            K,
            best_latency
        )
    };

    std::cout
        << "\nBest kernel:\n"
        << "  "
        << result.kernel->name
        << "\n"
        << "  memory path = "
        << kernel_path_name(
            result.kernel->path
        )
        << "\n"
        << "  latency = "
        << result.latency_ms
        << " ms\n"
        << "  TFLOPS = "
        << result.tflops
        << "\n";

    return result;
}
