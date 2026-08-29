#include "common.h"
#include "topk.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <utility>
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
                Pair{input[base + i], i}
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
            ] = row[j].value;

            out_indices[
                static_cast<size_t>(b) * k + j
            ] = row[j].index;
        }
    }
}

static float benchmark_naive(
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
        launch_topk_naive(
            d_input,
            d_values,
            d_indices,
            batch,
            n,
            k
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
            launch_topk_naive(
                d_input,
                d_values,
                d_indices,
                batch,
                n,
                k
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

        samples.push_back(total_ms / repeat);
    }

    std::sort(
        samples.begin(),
        samples.end()
    );

    return samples[samples.size() / 2];
}

static double effective_input_bandwidth_gbps(
    int batch,
    int n,
    float latency_ms
) {
    /*
     * This is intentionally called "input bandwidth", not true achieved
     * DRAM bandwidth. V0 rereads the row K times, so it is not a perfect
     * traffic model. It is only a stable normalization metric across
     * future versions.
     */
    const double bytes =
        static_cast<double>(batch)
        *
        n
        *
        sizeof(float);

    return bytes / (latency_ms * 1e-3) / 1e9;
}

static void print_usage(
    const char* prog
) {
    std::cout
        << "Usage:\n"
        << "  " << prog
        << " B N K [--warmup W] [--repeat R] [--groups G] [--seed S]\n\n"
        << "Example:\n"
        << "  " << prog
        << " 128 65536 16\n";
}

int main(
    int argc,
    char** argv
) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    const int batch =
        std::atoi(argv[1]);

    const int n =
        std::atoi(argv[2]);

    const int k =
        std::atoi(argv[3]);

    int warmup = 5;
    int repeat = 20;
    int groups = 5;
    int seed = 123;

    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else if (arg == "--repeat" && i + 1 < argc) {
            repeat = std::atoi(argv[++i]);
        } else if (arg == "--groups" && i + 1 < argc) {
            groups = std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::atoi(argv[++i]);
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
            << "Invalid shape: require B>0, N>0, 0<K<=N.\n";

        return 2;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout
        << "GPU: " << prop.name
        << " (SM " << prop.major
        << "." << prop.minor << ")\n"
        << "Top-K shape: B=" << batch
        << " N=" << n
        << " K=" << k << "\n"
        << "dtype: fp32\n"
        << "Implementation: naive_cuda_v0\n"
        << "Benchmark: "
        << groups << " groups x "
        << repeat << " launches, median selected\n";

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
     * Inject deterministic values so correctness tests also cover
     * obvious extremes and equal-value tie breaking.
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

    launch_topk_naive(
        d_input,
        d_values,
        d_indices,
        batch,
        n,
        k
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

    bool correct = true;
    float max_abs_error = 0.0f;
    size_t first_bad = output_count;

    for (size_t i = 0; i < output_count; ++i) {
        const float error =
            std::abs(
                gpu_values[i]
                -
                cpu_values[i]
            );

        max_abs_error =
            std::max(
                max_abs_error,
                error
            );

        if (
            error > 1e-6f
            ||
            gpu_indices[i] != cpu_indices[i]
        ) {
            correct = false;
            first_bad = i;
            break;
        }
    }

    std::cout
        << "\nCorrectness\n"
        << "CPU reference: partial_sort\n"
        << "Max abs value error = "
        << max_abs_error << "\n"
        << "Value + index match = "
        << (correct ? "PASS" : "FAIL")
        << "\n";

    if (!correct) {
        const int row =
            static_cast<int>(first_bad / k);

        const int rank =
            static_cast<int>(first_bad % k);

        std::cout
            << "First mismatch: row="
            << row
            << " rank="
            << rank
            << "\n"
            << "CPU: value="
            << cpu_values[first_bad]
            << " index="
            << cpu_indices[first_bad]
            << "\n"
            << "GPU: value="
            << gpu_values[first_bad]
            << " index="
            << gpu_indices[first_bad]
            << "\n";
    }

    const float gpu_ms =
        benchmark_naive(
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
        effective_input_bandwidth_gbps(
            batch,
            n,
            gpu_ms
        );

    const double elems_per_second =
        static_cast<double>(batch)
        *
        n
        /
        (gpu_ms * 1e-3);

    std::cout
        << "\nPerformance\n"
        << "CPU reference latency = "
        << cpu_ms
        << " ms\n"
        << "GPU naive latency = "
        << gpu_ms
        << " ms\n"
        << "Input elements/s = "
        << elems_per_second / 1e9
        << " GElem/s\n"
        << "Normalized input bandwidth = "
        << input_bw
        << " GB/s\n";

    std::cout
        << "\nFirst row Top-K preview\n";

    const int preview =
        std::min(k, 8);

    for (int i = 0; i < preview; ++i) {
        std::cout
            << "rank " << i
            << ": value="
            << gpu_values[i]
            << " index="
            << gpu_indices[i]
            << "\n";
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_indices));

    return correct ? 0 : 3;
}
