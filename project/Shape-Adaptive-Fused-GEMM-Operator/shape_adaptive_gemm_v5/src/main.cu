#include "autotuner.h"
#include "cache.h"
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
#include <string>
#include <vector>

struct Options {
    int M = 0;
    int N = 0;
    int K = 0;

    std::string cache_path =
        "results/gemm_cache_v5.csv";

    std::string forced_kernel;

    bool retune = false;
    bool list_kernels = false;
    bool no_cublas = false;
    bool profile_once = false;

    int warmup = 10;
    int repeat = 50;
    int groups = 5;
};

static void usage(
    const char* prog
) {
    std::cout
        << "Usage:\n"
        << "  "
        << prog
        << " M N K [options]\n\n"
        << "Options:\n"
        << "  --retune          Ignore cache and autotune again\n"
        << "  --cache PATH      Tune-cache CSV path\n"
        << "  --kernel NAME     Force one registered kernel\n"
        << "  --profile-once    Launch selected/forced kernel once and exit\n"
        << "  --list-kernels    Print registry and exit\n"
        << "  --warmup N        Warmup launches (default 10)\n"
        << "  --repeat N        Timed launches/group (default 50)\n"
        << "  --groups N        Benchmark groups (default 5)\n"
        << "  --no-cublas       Skip cuBLAS performance baseline\n\n"
        << "Examples:\n"
        << "  "
        << prog
        << " 128 4096 4096\n"
        << "  "
        << prog
        << " 128 4096 4096 --retune\n"
        << "  "
        << prog
        << " 128 4096 4096 --kernel m128_n64_k16_t8x4_warp\n"
        << "  "
        << prog
        << " 128 4096 4096 --kernel m128_n64_k16_t8x4_warp --profile-once\n";
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
        opt.M =
            std::atoi(
                argv[1]
            );

        opt.N =
            std::atoi(
                argv[2]
            );

        opt.K =
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
            opt.retune = true;
        } else if (
            arg == "--list-kernels"
        ) {
            opt.list_kernels = true;
        } else if (
            arg == "--no-cublas"
        ) {
            opt.no_cublas = true;
        } else if (
            arg == "--profile-once"
        ) {
            opt.profile_once = true;
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
        }
    }

    return opt;
}

static void print_registry() {
    std::cout
        << "\n# Kernel Registry\n\n";

    for (
        const auto& k :
        get_kernel_registry()
    ) {
        std::cout
            << std::left
            << std::setw(34)
            << k.name
            << " BM="
            << std::setw(4)
            << k.BM
            << " BN="
            << std::setw(4)
            << k.BN
            << " BK="
            << std::setw(4)
            << k.BK
            << " TM="
            << k.TM
            << " TN="
            << k.TN
            << " WM="
            << std::setw(4)
            << k.WM
            << " WN="
            << std::setw(4)
            << k.WN
            << " path="
            << kernel_path_name(
                k.path
            )
            << " threads="
            << k.threads
            << " smem="
            << k.shared_memory_bytes
            << "\n";
    }
}

static float max_error(
    const std::vector<float>& a,
    const std::vector<float>& b
) {
    float error =
        0.0f;

    for (
        size_t i = 0;
        i < a.size();
        ++i
    ) {
        error =
            std::max(
                error,
                std::abs(
                    a[i]
                    -
                    b[i]
                )
            );
    }

    return error;
}

static void cublas_gemm(
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    const float alpha =
        1.0f;

    const float beta =
        0.0f;

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

static float benchmark_cublas(
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    int warmup,
    int repeat,
    int groups
) {
    for (
        int i = 0;
        i < warmup;
        ++i
    ) {
        cublas_gemm(
            handle,
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
            cublas_gemm(
                handle,
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

int main(
    int argc,
    char** argv
) {
    const Options opt =
        parse_options(
            argc,
            argv
        );

    if (
        opt.list_kernels
    ) {
        print_registry();

        return 0;
    }

    if (
        opt.M <= 0
        ||
        opt.N <= 0
        ||
        opt.K <= 0
    ) {
        usage(
            argv[0]
        );

        return 1;
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
        << "Shape: "
        << opt.M
        << " x "
        << opt.N
        << " x "
        << opt.K
        << "\n"
        << "Cache: "
        << opt.cache_path
        << "\n";

    const size_t size_A =
        static_cast<size_t>(
            opt.M
        )
        *
        opt.K;

    const size_t size_B =
        static_cast<size_t>(
            opt.K
        )
        *
        opt.N;

    const size_t size_C =
        static_cast<size_t>(
            opt.M
        )
        *
        opt.N;

    std::vector<float>
    h_A(
        size_A
    );

    std::vector<float>
    h_B(
        size_B
    );

    std::mt19937
    rng(
        123
    );

    std::uniform_real_distribution<float>
    dist(
        -0.1f,
        0.1f
    );

    for (
        auto& x :
        h_A
    ) {
        x =
            dist(
                rng
            );
    }

    for (
        auto& x :
        h_B
    ) {
        x =
            dist(
                rng
            );
    }

    float* d_A =
        nullptr;

    float* d_B =
        nullptr;

    float* d_C =
        nullptr;

    float* d_ref =
        nullptr;

    float* d_bias =
        nullptr;

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
            static_cast<size_t>(
                opt.N
            )
            *
            sizeof(float)
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
            static_cast<size_t>(
                opt.N
            )
            *
            sizeof(float)
        )
    );

    const KernelConfig*
    selected =
        nullptr;

    float selected_latency =
        0.0f;

    double selected_tflops =
        0.0;

    if (
        !opt.forced_kernel.empty()
    ) {
        selected =
            find_kernel(
                opt.forced_kernel
            );

        if (
            !selected
        ) {
            std::cerr
                << "Unknown kernel: "
                << opt.forced_kernel
                << "\n";

            print_registry();

            return 2;
        }

        if (
            !kernel_problem_compatible(
                *selected,
                d_A,
                d_B,
                opt.M,
                opt.N,
                opt.K
            )
        ) {
            std::cerr
                << "Forced kernel is incompatible with this problem's "
                << "alignment/stride requirements.\n";

            return 3;
        }

        std::cout
            << "Mode: forced kernel\n";

    } else {
        TuneCache cache(
            opt.cache_path
        );

        CacheRecord cached;

        if (
            !opt.retune
            &&
            cache.lookup(
                prop.name,
                prop.major,
                prop.minor,
                opt.M,
                opt.N,
                opt.K,
                EpilogueType::NONE,
                cached
            )
        ) {
            selected =
                find_kernel(
                    cached.kernel
                );

            if (
                selected
                &&
                kernel_problem_compatible(
                    *selected,
                    d_A,
                    d_B,
                    opt.M,
                    opt.N,
                    opt.K
                )
            ) {
                selected_latency =
                    cached.latency_ms;

                selected_tflops =
                    cached.tflops;

                std::cout
                    << "Mode: cache hit\n"
                    << "Cached kernel: "
                    << cached.kernel
                    << "\n"
                    << "Cached path: "
                    << kernel_path_name(
                        selected->path
                    )
                    << "\n"
                    << "Cached latency: "
                    << cached.latency_ms
                    << " ms\n"
                    << "Cached TFLOPS: "
                    << cached.tflops
                    << "\n";
            }
        }

        if (
            !selected
        ) {
            std::cout
                << "Mode: cache miss -> autotune\n";

            TuneResult tuned =
                autotune_gemm(
                    d_A,
                    d_B,
                    d_C,
                    d_bias,
                    opt.M,
                    opt.N,
                    opt.K,
                    EpilogueType::NONE,
                    opt.warmup,
                    opt.repeat,
                    opt.groups
                );

            selected =
                tuned.kernel;

            selected_latency =
                tuned.latency_ms;

            selected_tflops =
                tuned.tflops;

            CacheRecord record;

            record.gpu =
                prop.name;

            record.cc_major =
                prop.major;

            record.cc_minor =
                prop.minor;

            record.M =
                opt.M;

            record.N =
                opt.N;

            record.K =
                opt.K;

            record.epilogue =
                static_cast<int>(
                    EpilogueType::NONE
                );

            record.kernel =
                selected->name;

            record.latency_ms =
                selected_latency;

            record.tflops =
                selected_tflops;

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

    if (
        opt.profile_once
    ) {
        std::cout
            << "Profile kernel: "
            << selected->name
            << "\n"
            << "Memory path: "
            << kernel_path_name(
                selected->path
            )
            << "\n";

        launch_kernel(
            *selected,
            d_A,
            d_B,
            d_C,
            d_bias,
            opt.M,
            opt.N,
            opt.K,
            EpilogueType::NONE
        );

        CUDA_CHECK(
            cudaDeviceSynchronize()
        );

        CUDA_CHECK(
            cudaFree(
                d_A
            )
        );

        CUDA_CHECK(
            cudaFree(
                d_B
            )
        );

        CUDA_CHECK(
            cudaFree(
                d_C
            )
        );

        CUDA_CHECK(
            cudaFree(
                d_ref
            )
        );

        CUDA_CHECK(
            cudaFree(
                d_bias
            )
        );

        return 0;
    }

    if (
        !opt.forced_kernel.empty()
        ||
        selected_latency == 0.0f
    ) {
        selected_latency =
            benchmark_kernel(
                *selected,
                d_A,
                d_B,
                d_C,
                d_bias,
                opt.M,
                opt.N,
                opt.K,
                EpilogueType::NONE,
                opt.warmup,
                opt.repeat,
                opt.groups
            );

        selected_tflops =
            gemm_tflops(
                opt.M,
                opt.N,
                opt.K,
                selected_latency
            );
    }

    std::cout
        << "\nSelected kernel\n"
        << "  "
        << selected->name
        << "\n"
        << "  path = "
        << kernel_path_name(
            selected->path
        )
        << "\n"
        << "  latency = "
        << selected_latency
        << " ms\n"
        << "  TFLOPS = "
        << selected_tflops
        << "\n";

    cublasHandle_t handle{};

    CUBLAS_CHECK(
        cublasCreate(
            &handle
        )
    );

    launch_kernel(
        *selected,
        d_A,
        d_B,
        d_C,
        d_bias,
        opt.M,
        opt.N,
        opt.K,
        EpilogueType::NONE
    );

    cublas_gemm(
        handle,
        d_A,
        d_B,
        d_ref,
        opt.M,
        opt.N,
        opt.K
    );

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );

    std::vector<float>
    h_C(
        size_C
    );

    std::vector<float>
    h_ref(
        size_C
    );

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

    std::cout
        << "\nCorrectness\n"
        << "Max abs error = "
        << max_error(
            h_C,
            h_ref
        )
        << "\n";

    if (
        !opt.no_cublas
    ) {
        const float cublas_ms =
            benchmark_cublas(
                handle,
                d_A,
                d_B,
                d_ref,
                opt.M,
                opt.N,
                opt.K,
                opt.warmup,
                opt.repeat,
                opt.groups
            );

        const double cublas_perf =
            gemm_tflops(
                opt.M,
                opt.N,
                opt.K,
                cublas_ms
            );

        std::cout
            << "\ncuBLAS\n"
            << "latency = "
            << cublas_ms
            << " ms\n"
            << "TFLOPS = "
            << cublas_perf
            << "\n"
            << "Relative performance = "
            << selected_tflops
               /
               cublas_perf
               *
               100.0
            << "%\n";
    }

    CUBLAS_CHECK(
        cublasDestroy(
            handle
        )
    );

    CUDA_CHECK(
        cudaFree(
            d_A
        )
    );

    CUDA_CHECK(
        cudaFree(
            d_B
        )
    );

    CUDA_CHECK(
        cudaFree(
            d_C
        )
    );

    CUDA_CHECK(
        cudaFree(
            d_ref
        )
    );

    CUDA_CHECK(
        cudaFree(
            d_bias
        )
    );

    return 0;
}
