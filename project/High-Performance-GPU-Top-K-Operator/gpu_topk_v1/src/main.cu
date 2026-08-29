#include "common.h"
#include "topk.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct Pair {
    float value;
    int index;
};

static bool pair_better(
    const Pair& a,
    const Pair& b
) {
    if (a.value != b.value) {
        return a.value > b.value;
    }

    return a.index < b.index;
}

static void cpu_topk_reference(
    const std::vector<float>& input,
    std::vector<float>& out_values,
    std::vector<int>& out_indices,
    int batch,
    int n,
    int k
) {
    out_values.resize(
        static_cast<size_t>(batch) * k
    );

    out_indices.resize(
        static_cast<size_t>(batch) * k
    );

    std::vector<Pair> row;
    row.reserve(n);

    for (int b = 0; b < batch; ++b) {
        row.clear();

        const size_t base =
            static_cast<size_t>(b) * n;

        for (int i = 0; i < n; ++i) {
            row.push_back(
                Pair{
                    input[base + i],
                    i
                }
            );
        }

        std::partial_sort(
            row.begin(),
            row.begin() + k,
            row.end(),
            pair_better
        );

        for (int j = 0; j < k; ++j) {
            out_values[
                static_cast<size_t>(b) * k + j
            ] =
                row[j].value;

            out_indices[
                static_cast<size_t>(b) * k + j
            ] =
                row[j].index;
        }
    }
}

using LaunchFn = void (*)(
    const float*,
    float*,
    int*,
    int,
    int,
    int,
    cudaStream_t
);

static float benchmark_kernel(
    LaunchFn launch,
    const float* d_input,
    float* d_values,
    int* d_indices,
    int batch,
    int n,
    int k,
    int warmup,
    int repeat,
    int groups
) {
    for (int i = 0; i < warmup; ++i) {
        launch(
            d_input,
            d_values,
            d_indices,
            batch,
            n,
            k,
            nullptr
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> samples;
    samples.reserve(groups);

    for (int g = 0; g < groups; ++g) {
        cudaEvent_t start{};
        cudaEvent_t stop{};

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < repeat; ++i) {
            launch(
                d_input,
                d_values,
                d_indices,
                batch,
                n,
                k,
                nullptr
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

        samples.push_back(
            total_ms / repeat
        );
    }

    std::sort(
        samples.begin(),
        samples.end()
    );

    return samples[samples.size() / 2];
}

static bool check_correctness(
    const std::vector<float>& gpu_values,
    const std::vector<int>& gpu_indices,
    const std::vector<float>& cpu_values,
    const std::vector<int>& cpu_indices,
    int k,
    float& max_abs_error
) {
    max_abs_error = 0.0f;

    for (size_t i = 0; i < gpu_values.size(); ++i) {
        const float err =
            std::abs(
                gpu_values[i]
                -
                cpu_values[i]
            );

        max_abs_error =
            std::max(
                max_abs_error,
                err
            );

        if (
            err > 1e-6f
            ||
            gpu_indices[i] != cpu_indices[i]
        ) {
            const int row =
                static_cast<int>(i / k);

            const int rank =
                static_cast<int>(i % k);

            std::cerr
                << "Mismatch at row="
                << row
                << " rank="
                << rank
                << "\n"
                << "CPU: value="
                << cpu_values[i]
                << " index="
                << cpu_indices[i]
                << "\n"
                << "GPU: value="
                << gpu_values[i]
                << " index="
                << gpu_indices[i]
                << "\n";

            return false;
        }
    }

    return true;
}

static double normalized_input_bandwidth_gbps(
    int batch,
    int n,
    float ms
) {
    const double bytes =
        static_cast<double>(batch)
        *
        n
        *
        sizeof(float);

    return
        bytes
        /
        (ms * 1e-3)
        /
        1e9;
}

static void usage(
    const char* prog
) {
    std::cout
        << "Usage:\n"
        << "  "
        << prog
        << " B N K [options]\n\n"
        << "Options:\n"
        << "  --kernel register|naive   default: register\n"
        << "  --compare-naive           also benchmark V0 naive baseline\n"
        << "  --warmup W                default: 5\n"
        << "  --repeat R                default: 50 for V1\n"
        << "  --groups G                default: 5\n"
        << "  --seed S                  default: 123\n"
        << "  --profile-once            launch selected kernel once and exit\n\n"
        << "V1 register path supports 1 <= K <= 16.\n";
}

int main(
    int argc,
    char** argv
) {
    if (argc < 4) {
        usage(argv[0]);
        return 1;
    }

    const int batch = std::atoi(argv[1]);
    const int n = std::atoi(argv[2]);
    const int k = std::atoi(argv[3]);

    std::string kernel = "register";
    bool compare_naive = false;
    bool profile_once = false;

    int warmup = 5;
    int repeat = 50;
    int groups = 5;
    int seed = 123;

    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];

        if (
            arg == "--kernel"
            &&
            i + 1 < argc
        ) {
            kernel = argv[++i];

        } else if (
            arg == "--compare-naive"
        ) {
            compare_naive = true;

        } else if (
            arg == "--warmup"
            &&
            i + 1 < argc
        ) {
            warmup = std::atoi(argv[++i]);

        } else if (
            arg == "--repeat"
            &&
            i + 1 < argc
        ) {
            repeat = std::atoi(argv[++i]);

        } else if (
            arg == "--groups"
            &&
            i + 1 < argc
        ) {
            groups = std::atoi(argv[++i]);

        } else if (
            arg == "--seed"
            &&
            i + 1 < argc
        ) {
            seed = std::atoi(argv[++i]);

        } else if (
            arg == "--profile-once"
        ) {
            profile_once = true;
        }
    }

    if (
        batch <= 0
        ||
        n <= 0
        ||
        k <= 0
        ||
        k > n
    ) {
        std::cerr
            << "Require B>0, N>0, 0<K<=N.\n";

        return 2;
    }

    LaunchFn selected = nullptr;
    std::string implementation;

    if (kernel == "register") {
        if (!topk_register_v1_supported(k)) {
            std::cerr
                << "V1 register kernel supports 1 <= K <= 16.\n";

            return 3;
        }

        selected = launch_topk_register_v1;
        implementation =
            "register_local_topk_block_merge_v1";

    } else if (kernel == "naive") {
        selected = launch_topk_naive;
        implementation = "naive_cuda_v0";

    } else {
        std::cerr
            << "Unknown --kernel "
            << kernel
            << "\n";

        return 4;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout
        << "GPU: "
        << prop.name
        << " (SM "
        << prop.major
        << "."
        << prop.minor
        << ")\n"
        << "Top-K shape: B="
        << batch
        << " N="
        << n
        << " K="
        << k
        << "\n"
        << "dtype: fp32\n"
        << "Implementation: "
        << implementation
        << "\n"
        << "Benchmark: "
        << groups
        << " groups x "
        << repeat
        << " launches, median selected\n";

    const size_t input_count =
        static_cast<size_t>(batch) * n;

    const size_t output_count =
        static_cast<size_t>(batch) * k;

    std::vector<float> h_input(input_count);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float>
    dist(-1.0f, 1.0f);

    for (auto& x : h_input) {
        x = dist(rng);
    }

    /*
     * Deterministic edge values exercise ranking and tie breaking.
     */
    if (n >= 4) {
        for (int b = 0; b < batch; ++b) {
            const size_t base =
                static_cast<size_t>(b) * n;

            h_input[base + 0] = 10.0f;
            h_input[base + 1] = 10.0f;
            h_input[base + 2] = 9.0f;
            h_input[base + 3] = -10.0f;
        }
    }

    std::vector<float> cpu_values;
    std::vector<int> cpu_indices;

    const auto cpu_start =
        std::chrono::high_resolution_clock::now();

    cpu_topk_reference(
        h_input,
        cpu_values,
        cpu_indices,
        batch,
        n,
        k
    );

    const auto cpu_stop =
        std::chrono::high_resolution_clock::now();

    const double cpu_ms =
        std::chrono::duration<double, std::milli>(
            cpu_stop - cpu_start
        ).count();

    float* d_input = nullptr;
    float* d_values = nullptr;
    int* d_indices = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            input_count * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_values,
            output_count * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_indices,
            output_count * sizeof(int)
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            input_count * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    if (profile_once) {
        selected(
            d_input,
            d_values,
            d_indices,
            batch,
            n,
            k,
            nullptr
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        std::cout
            << "Profile launch complete.\n";

        return 0;
    }

    selected(
        d_input,
        d_values,
        d_indices,
        batch,
        n,
        k,
        nullptr
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_values(output_count);
    std::vector<int> gpu_indices(output_count);

    CUDA_CHECK(
        cudaMemcpy(
            gpu_values.data(),
            d_values,
            output_count * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            gpu_indices.data(),
            d_indices,
            output_count * sizeof(int),
            cudaMemcpyDeviceToHost
        )
    );

    float max_abs_error = 0.0f;

    const bool correct =
        check_correctness(
            gpu_values,
            gpu_indices,
            cpu_values,
            cpu_indices,
            k,
            max_abs_error
        );

    std::cout
        << "\nCorrectness\n"
        << "CPU reference: partial_sort\n"
        << "Max abs value error = "
        << max_abs_error
        << "\n"
        << "Value + index match = "
        << (correct ? "PASS" : "FAIL")
        << "\n";

    const float latency_ms =
        benchmark_kernel(
            selected,
            d_input,
            d_values,
            d_indices,
            batch,
            n,
            k,
            warmup,
            repeat,
            groups
        );

    const double input_bw =
        normalized_input_bandwidth_gbps(
            batch,
            n,
            latency_ms
        );

    const double elements_per_second =
        static_cast<double>(batch)
        *
        n
        /
        (latency_ms * 1e-3);

    std::cout
        << "\nPerformance\n"
        << "CPU reference latency = "
        << cpu_ms
        << " ms\n"
        << "GPU V1 latency = "
        << latency_ms
        << " ms\n"
        << "Input elements/s = "
        << elements_per_second / 1e9
        << " GElem/s\n"
        << "Normalized input bandwidth = "
        << input_bw
        << " GB/s\n";

    if (
        compare_naive
        &&
        kernel != "naive"
    ) {
        /*
         * V0 is extremely slow. Use a deliberately smaller timing sample
         * so --compare-naive remains practical while still giving a
         * same-process A/B reference.
         */
        const float naive_ms =
            benchmark_kernel(
                launch_topk_naive,
                d_input,
                d_values,
                d_indices,
                batch,
                n,
                k,
                1,
                3,
                3
            );

        std::cout
            << "\nV0 comparison\n"
            << "Naive latency = "
            << naive_ms
            << " ms\n"
            << "V1 speedup over V0 = "
            << naive_ms / latency_ms
            << "x\n";
    }

    std::cout
        << "\nFirst row Top-K preview\n";

    const int preview =
        std::min(k, 8);

    for (int i = 0; i < preview; ++i) {
        std::cout
            << "rank "
            << i
            << ": value="
            << gpu_values[i]
            << " index="
            << gpu_indices[i]
            << "\n";
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_indices));

    return correct ? 0 : 5;
}
