#include "common.h"
#include "topk_autotuner.h"
#include "topk_cache.h"
#include "topk_registry.h"

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

struct Options {
    int batch = 0;
    int n = 0;
    int k = 0;

    std::string cache_path =
        "results/topk_cache_v5.csv";

    std::string forced_kernel;

    bool retune =
        false;

    bool list_kernels =
        false;

    bool profile_once =
        false;

    int warmup =
        5;

    int repeat =
        50;

    int groups =
        5;

    int seed =
        123;
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

static bool check_correctness(
    const std::vector<float>& gpu_values,
    const std::vector<int>& gpu_indices,
    const std::vector<float>& cpu_values,
    const std::vector<int>& cpu_indices,
    int k,
    float& max_abs_error
) {
    max_abs_error =
        0.0f;

    for (
        size_t i = 0;
        i < gpu_values.size();
        ++i
    ) {
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
            gpu_indices[i]
                !=
            cpu_indices[i]
        ) {
            const int row =
                static_cast<int>(
                    i / k
                );

            const int rank =
                static_cast<int>(
                    i % k
                );

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

static void print_registry() {
    std::cout
        << "\nTop-K V5 Kernel Registry\n"
        << "==========================================================================\n";

    for (
        const auto& kernel :
        get_topk_registry()
    ) {
        std::cout
            << std::left
            << std::setw(24)
            << kernel.name
            << " family="
            << std::setw(34)
            << kernel.family
            << " K=["
            << kernel.min_k
            << ","
            << kernel.max_k
            << "]"
            << " specialized="
            << (
                kernel.specialized_k
                ?
                "yes"
                :
                "no"
            )
            << " autotune="
            << (
                kernel.autotune_candidate
                ?
                "yes"
                :
                "no"
            )
            << "\n";
    }
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
        << "  --retune              Ignore cache and autotune again\n"
        << "  --cache PATH          Cache CSV (default results/topk_cache_v5.csv)\n"
        << "  --kernel NAME         Force one registry kernel\n"
        << "  --list-kernels        Print registry and exit\n"
        << "  --profile-once        Launch selected kernel once and exit\n"
        << "  --warmup N            default 5\n"
        << "  --repeat N            default 50\n"
        << "  --groups N            default 5\n"
        << "  --seed N              default 123\n\n"
        << "Examples:\n"
        << "  "
        << prog
        << " 128 65536 4 --retune\n"
        << "  "
        << prog
        << " 128 65536 8\n"
        << "  "
        << prog
        << " 128 65536 8 --kernel batch_v3\n";
}

static Options parse_options(
    int argc,
    char** argv
) {
    Options opt;

    if (
        argc >= 4
        &&
        argv[1][0] != '-'
    ) {
        opt.batch =
            std::atoi(
                argv[1]
            );

        opt.n =
            std::atoi(
                argv[2]
            );

        opt.k =
            std::atoi(
                argv[3]
            );
    }

    for (
        int i = 1;
        i < argc;
        ++i
    ) {
        const std::string arg =
            argv[i];

        if (
            arg == "--retune"
        ) {
            opt.retune =
                true;

        } else if (
            arg == "--cache"
            &&
            i + 1 < argc
        ) {
            opt.cache_path =
                argv[++i];

        } else if (
            arg == "--kernel"
            &&
            i + 1 < argc
        ) {
            opt.forced_kernel =
                argv[++i];

        } else if (
            arg == "--list-kernels"
        ) {
            opt.list_kernels =
                true;

        } else if (
            arg == "--profile-once"
        ) {
            opt.profile_once =
                true;

        } else if (
            arg == "--warmup"
            &&
            i + 1 < argc
        ) {
            opt.warmup =
                std::atoi(
                    argv[++i]
                );

        } else if (
            arg == "--repeat"
            &&
            i + 1 < argc
        ) {
            opt.repeat =
                std::atoi(
                    argv[++i]
                );

        } else if (
            arg == "--groups"
            &&
            i + 1 < argc
        ) {
            opt.groups =
                std::atoi(
                    argv[++i]
                );

        } else if (
            arg == "--seed"
            &&
            i + 1 < argc
        ) {
            opt.seed =
                std::atoi(
                    argv[++i]
                );
        }
    }

    return opt;
}

int main(
    int argc,
    char** argv
) {
    const Options opt =
        parse_options(
            argc,
            argv
        );

    if (opt.list_kernels) {
        print_registry();
        return 0;
    }

    if (
        opt.batch <= 0
        ||
        opt.n <= 0
        ||
        opt.k <= 0
        ||
        opt.k > opt.n
    ) {
        usage(argv[0]);

        std::cerr
            << "\nRequire B>0, N>0, 0<K<=N.\n";

        return 1;
    }

    if (
        opt.k > 16
    ) {
        std::cerr
            << "V5 small-K library currently supports K <= 16.\n";

        return 2;
    }

    cudaDeviceProp prop{};

    CUDA_CHECK(
        cudaGetDeviceProperties(
            &prop,
            0
        )
    );

    std::cout
        << "GPU: "
        << prop.name
        << " (SM "
        << prop.major
        << "."
        << prop.minor
        << ")\n"
        << "Top-K shape: B="
        << opt.batch
        << " N="
        << opt.n
        << " K="
        << opt.k
        << "\n"
        << "dtype: fp32\n"
        << "Cache: "
        << opt.cache_path
        << "\n";

    const size_t input_count =
        static_cast<size_t>(
            opt.batch
        )
        *
        opt.n;

    const size_t output_count =
        static_cast<size_t>(
            opt.batch
        )
        *
        opt.k;

    std::vector<float>
    h_input(
        input_count
    );

    std::mt19937 rng(
        opt.seed
    );

    std::uniform_real_distribution<float>
    dist(
        -1.0f,
        1.0f
    );

    for (
        auto& x :
        h_input
    ) {
        x =
            dist(rng);
    }

    /*
     * Deterministic tie/extreme values.
     */
    if (opt.n >= 4) {
        for (
            int b = 0;
            b < opt.batch;
            ++b
        ) {
            const size_t base =
                static_cast<size_t>(b)
                *
                opt.n;

            h_input[base + 0] =
                10.0f;

            h_input[base + 1] =
                10.0f;

            h_input[base + 2] =
                9.0f;

            h_input[base + 3] =
                -10.0f;
        }
    }

    float* d_input =
        nullptr;

    float* d_values =
        nullptr;

    int* d_indices =
        nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            input_count
            *
            sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_values,
            output_count
            *
            sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_indices,
            output_count
            *
            sizeof(int)
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            input_count
            *
            sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    const TopKKernelConfig*
    selected =
        nullptr;

    float selected_latency =
        0.0f;

    double selected_gelem =
        0.0;

    TopKTuneCache cache(
        opt.cache_path
    );

    if (
        !opt.forced_kernel.empty()
    ) {
        selected =
            find_topk_kernel(
                opt.forced_kernel
            );

        if (!selected) {
            std::cerr
                << "Unknown kernel: "
                << opt.forced_kernel
                << "\n";

            print_registry();

            return 3;
        }

        if (
            !selected->supports
            ||
            !selected->supports(
                opt.batch,
                opt.n,
                opt.k
            )
        ) {
            std::cerr
                << "Forced kernel "
                << selected->name
                << " does not support K="
                << opt.k
                << "\n";

            return 4;
        }

        std::cout
            << "Mode: forced kernel\n";

    } else {
        TopKCacheRecord record;

        if (
            !opt.retune
            &&
            cache.lookup(
                prop.name,
                prop.major,
                prop.minor,
                "fp32",
                opt.batch,
                opt.n,
                opt.k,
                record
            )
        ) {
            const TopKKernelConfig*
            cached =
                find_topk_kernel(
                    record.kernel
                );

            if (
                cached
                &&
                cached->supports
                &&
                cached->supports(
                    opt.batch,
                    opt.n,
                    opt.k
                )
            ) {
                selected =
                    cached;

                selected_latency =
                    record.latency_ms;

                selected_gelem =
                    record.gelem_per_s;

                std::cout
                    << "Mode: cache hit\n"
                    << "Cached kernel: "
                    << record.kernel
                    << "\n"
                    << "Cached latency: "
                    << selected_latency
                    << " ms\n"
                    << "Cached GElem/s: "
                    << selected_gelem
                    << "\n";
            }
        }

        if (!selected) {
            std::cout
                << "Mode: cache miss -> autotune\n";

            const TopKTuneResult tuned =
                autotune_topk(
                    d_input,
                    d_values,
                    d_indices,
                    opt.batch,
                    opt.n,
                    opt.k,
                    opt.warmup,
                    opt.repeat,
                    opt.groups
                );

            selected =
                tuned.kernel;

            selected_latency =
                tuned.latency_ms;

            selected_gelem =
                tuned.gelem_per_s;

            TopKCacheRecord record;

            record.gpu =
                prop.name;

            record.cc_major =
                prop.major;

            record.cc_minor =
                prop.minor;

            record.dtype =
                "fp32";

            record.batch =
                opt.batch;

            record.n =
                opt.n;

            record.k =
                opt.k;

            record.kernel =
                selected->name;

            record.latency_ms =
                selected_latency;

            record.gelem_per_s =
                selected_gelem;

            cache.upsert(
                record
            );

            cache.save();

            std::cout
                << "Saved tune result to: "
                << cache.path()
                << "\n";
        }
    }

    std::cout
        << "\nSelected kernel\n"
        << "  name = "
        << selected->name
        << "\n"
        << "  family = "
        << selected->family
        << "\n";

    if (opt.profile_once) {
        selected->launcher(
            d_input,
            d_values,
            d_indices,
            opt.batch,
            opt.n,
            opt.k,
            nullptr
        );

        CUDA_CHECK(
            cudaDeviceSynchronize()
        );

        std::cout
            << "Profile launch complete.\n";

        return 0;
    }

    if (
        !opt.forced_kernel.empty()
        ||
        selected_latency == 0.0f
    ) {
        selected_latency =
            benchmark_topk_kernel(
                *selected,
                d_input,
                d_values,
                d_indices,
                opt.batch,
                opt.n,
                opt.k,
                opt.warmup,
                opt.repeat,
                opt.groups
            );

        selected_gelem =
            topk_gelem_per_s(
                opt.batch,
                opt.n,
                selected_latency
            );
    }

    /*
     * One exact selected-kernel launch for correctness.
     */
    selected->launcher(
        d_input,
        d_values,
        d_indices,
        opt.batch,
        opt.n,
        opt.k,
        nullptr
    );

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );

    std::vector<float>
    gpu_values(
        output_count
    );

    std::vector<int>
    gpu_indices(
        output_count
    );

    CUDA_CHECK(
        cudaMemcpy(
            gpu_values.data(),
            d_values,
            output_count
            *
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            gpu_indices.data(),
            d_indices,
            output_count
            *
            sizeof(int),
            cudaMemcpyDeviceToHost
        )
    );

    std::vector<float>
    cpu_values;

    std::vector<int>
    cpu_indices;

    const auto cpu_start =
        std::chrono::high_resolution_clock::now();

    cpu_topk_reference(
        h_input,
        cpu_values,
        cpu_indices,
        opt.batch,
        opt.n,
        opt.k
    );

    const auto cpu_stop =
        std::chrono::high_resolution_clock::now();

    const double cpu_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            cpu_stop
            -
            cpu_start
        ).count();

    float max_abs_error =
        0.0f;

    const bool correct =
        check_correctness(
            gpu_values,
            gpu_indices,
            cpu_values,
            cpu_indices,
            opt.k,
            max_abs_error
        );

    const double normalized_bw =
        static_cast<double>(
            opt.batch
        )
        *
        opt.n
        *
        sizeof(float)
        /
        (selected_latency * 1e-3)
        /
        1e9;

    std::cout
        << "\nCorrectness\n"
        << "CPU reference: partial_sort\n"
        << "Max abs value error = "
        << max_abs_error
        << "\n"
        << "Value + index match = "
        << (
            correct
            ?
            "PASS"
            :
            "FAIL"
        )
        << "\n";

    std::cout
        << "\nPerformance\n"
        << "CPU reference latency = "
        << cpu_ms
        << " ms\n"
        << "GPU selected latency = "
        << selected_latency
        << " ms\n"
        << "Input elements/s = "
        << selected_gelem
        << " GElem/s\n"
        << "Normalized input bandwidth = "
        << normalized_bw
        << " GB/s\n";

    std::cout
        << "\nFirst row Top-K preview\n";

    const int preview =
        std::min(
            opt.k,
            8
        );

    for (
        int i = 0;
        i < preview;
        ++i
    ) {
        std::cout
            << "rank "
            << i
            << ": value="
            << gpu_values[i]
            << " index="
            << gpu_indices[i]
            << "\n";
    }

    CUDA_CHECK(
        cudaFree(
            d_input
        )
    );

    CUDA_CHECK(
        cudaFree(
            d_values
        )
    );

    CUDA_CHECK(
        cudaFree(
            d_indices
        )
    );

    return
        correct
        ?
        0
        :
        5;
}
