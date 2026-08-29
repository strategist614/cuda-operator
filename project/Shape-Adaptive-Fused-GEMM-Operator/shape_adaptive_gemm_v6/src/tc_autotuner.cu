#include "tc_autotuner.h"
#include "common.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

double tensor_core_tflops(
    int M,
    int N,
    int K,
    float ms
) {
    const double ops =
        2.0
        *
        static_cast<double>(M)
        *
        static_cast<double>(N)
        *
        static_cast<double>(K);

    return
        ops
        /
        (ms * 1e-3)
        /
        1e12;
}

bool tensor_core_kernel_supported_on_device(
    const TensorCoreConfig& kernel
) {
    int device = 0;

    CUDA_CHECK(
        cudaGetDevice(
            &device
        )
    );

    cudaDeviceProp prop{};

    CUDA_CHECK(
        cudaGetDeviceProperties(
            &prop,
            device
        )
    );

    /*
     * WMMA Tensor Core FP16 requires Volta or newer.
     * This project is primarily tuned for SM75.
     */
    if (
        prop.major < 7
    ) {
        return false;
    }

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

float benchmark_tensor_core_kernel(
    const TensorCoreConfig& kernel,
    const half* A,
    const half* B,
    float* C,
    int M,
    int N,
    int K,
    int warmup,
    int repeat,
    int groups
) {
    if (
        !tensor_core_problem_compatible(
            kernel,
            A,
            B,
            M,
            N,
            K
        )
    ) {
        return
            std::numeric_limits<
                float
            >::infinity();
    }

    for (
        int i = 0;
        i < warmup;
        ++i
    ) {
        launch_tensor_core_kernel(
            kernel,
            A,
            B,
            C,
            M,
            N,
            K
        );
    }

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );

    std::vector<float>
    samples;

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
            launch_tensor_core_kernel(
                kernel,
                A,
                B,
                C,
                M,
                N,
                K
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

TensorCoreTuneResult
autotune_tensor_core_gemm(
    const half* A,
    const half* B,
    float* C,
    int M,
    int N,
    int K,
    int warmup,
    int repeat,
    int groups
) {
    const TensorCoreConfig*
    best_kernel =
        nullptr;

    float best_latency =
        std::numeric_limits<
            float
        >::max();

    std::cout
        << "\n============================================\n"
        << "Autotuning GEMM V6 Tensor Core Family\n"
        << "dtype=FP16 input / FP32 accumulate\n"
        << "M="
        << M
        << " N="
        << N
        << " K="
        << K
        << "\n"
        << "benchmark="
        << groups
        << " groups x "
        << repeat
        << " launches, median selected\n"
        << "============================================\n";

    for (
        const auto& kernel :
        get_tensor_core_registry()
    ) {
        if (
            !tensor_core_kernel_supported_on_device(
                kernel
            )
        ) {
            std::cout
                << std::left
                << std::setw(38)
                << kernel.name
                << "SKIP(device/resources)\n";

            continue;
        }

        if (
            !tensor_core_problem_compatible(
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
                << std::setw(38)
                << kernel.name
                << "SKIP(shape/alignment)\n";

            continue;
        }

        const float latency =
            benchmark_tensor_core_kernel(
                kernel,
                A,
                B,
                C,
                M,
                N,
                K,
                warmup,
                repeat,
                groups
            );

        const double perf =
            tensor_core_tflops(
                M,
                N,
                K,
                latency
            );

        std::cout
            << std::left
            << std::setw(38)
            << kernel.name
            << " CTA="
            << kernel.BM
            << "x"
            << kernel.BN
            << "x"
            << kernel.BK
            << " warp="
            << kernel.WM
            << "x"
            << kernel.WN
            << " latency="
            << std::setw(9)
            << latency
            << " ms TFLOPS="
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
            << "No compatible Tensor Core kernel found.\n"
            << "V6 TC fast path requires the shape to exactly tile "
            << "at least one registered CTA configuration.\n";

        std::exit(
            EXIT_FAILURE
        );
    }

    TensorCoreTuneResult result{
        best_kernel,
        best_latency,
        tensor_core_tflops(
            M,
            N,
            K,
            best_latency
        )
    };

    std::cout
        << "\nBest Tensor Core kernel:\n"
        << "  "
        << result.kernel->name
        << "\n"
        << "  latency = "
        << result.latency_ms
        << " ms\n"
        << "  TFLOPS = "
        << result.tflops
        << "\n";

    return result;
}
