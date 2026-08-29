#include "topk_autotuner.h"
#include "common.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

double topk_gelem_per_s(
    int batch,
    int n,
    float latency_ms
) {
    return
        static_cast<double>(batch)
        *
        n
        /
        (latency_ms * 1e-3)
        /
        1e9;
}

float benchmark_topk_kernel(
    const TopKKernelConfig& kernel,
    const float* input,
    float* output_values,
    int* output_indices,
    int batch,
    int n,
    int k,
    int warmup,
    int repeat,
    int groups
) {
    for (int i = 0; i < warmup; ++i) {
        kernel.launcher(
            input,
            output_values,
            output_indices,
            batch,
            n,
            k,
            nullptr
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

    for (int g = 0; g < groups; ++g) {
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

        for (int i = 0; i < repeat; ++i) {
            kernel.launcher(
                input,
                output_values,
                output_indices,
                batch,
                n,
                k,
                nullptr
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

TopKTuneResult autotune_topk(
    const float* input,
    float* output_values,
    int* output_indices,
    int batch,
    int n,
    int k,
    int warmup,
    int repeat,
    int groups
) {
    const TopKKernelConfig*
    best_kernel =
        nullptr;

    float best_latency =
        std::numeric_limits<float>::max();

    std::cout
        << "\n============================================\n"
        << "Autotuning GPU Top-K V5\n"
        << "B=" << batch
        << " N=" << n
        << " K=" << k
        << "\n"
        << "benchmark="
        << groups
        << " groups x "
        << repeat
        << " launches, median selected\n"
        << "============================================\n";

    for (
        const auto& kernel :
        get_topk_registry()
    ) {
        if (
            !kernel.autotune_candidate
            ||
            !kernel.supports
            ||
            !kernel.supports(
                batch,
                n,
                k
            )
        ) {
            continue;
        }

        const float latency =
            benchmark_topk_kernel(
                kernel,
                input,
                output_values,
                output_indices,
                batch,
                n,
                k,
                warmup,
                repeat,
                groups
            );

        const double gelem =
            topk_gelem_per_s(
                batch,
                n,
                latency
            );

        std::cout
            << std::left
            << std::setw(28)
            << kernel.name
            << " family="
            << std::setw(34)
            << kernel.family
            << " latency="
            << std::setw(10)
            << latency
            << " ms"
            << " GElem/s="
            << gelem
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

    if (!best_kernel) {
        std::cerr
            << "No compatible V5 autotune candidate for "
            << "B=" << batch
            << " N=" << n
            << " K=" << k
            << "\n";

        std::exit(EXIT_FAILURE);
    }

    TopKTuneResult result;

    result.kernel =
        best_kernel;

    result.latency_ms =
        best_latency;

    result.gelem_per_s =
        topk_gelem_per_s(
            batch,
            n,
            best_latency
        );

    std::cout
        << "\nBest Top-K kernel:\n"
        << "  "
        << result.kernel->name
        << "\n"
        << "  family = "
        << result.kernel->family
        << "\n"
        << "  latency = "
        << result.latency_ms
        << " ms\n"
        << "  GElem/s = "
        << result.gelem_per_s
        << "\n";

    return result;
}
