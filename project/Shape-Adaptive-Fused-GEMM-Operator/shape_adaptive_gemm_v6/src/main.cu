#include "autotuner.h"
#include "cache.h"
#include "common.h"
#include "gemm.h"
#include "tc_autotuner.h"
#include "tensor_core.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

struct Options {
    int M = 0;
    int N = 0;
    int K = 0;

    std::string dtype =
        "fp16";

    std::string cache_path =
        "results/gemm_cache_v6.csv";

    std::string forced_kernel;

    bool retune =
        false;

    bool list_kernels =
        false;

    bool no_cublas =
        false;

    bool profile_once =
        false;

    int warmup =
        10;

    int repeat =
        50;

    int groups =
        5;
};

static void usage(
    const char* prog
) {
    std::cout
        << "Usage:\n"
        << "  "
        << prog
        << " M N K [options]\n\n"
        << "V6 defaults to FP16 Tensor Core family.\n\n"
        << "Options:\n"
        << "  --dtype fp16|fp32   fp16=Tensor Core, fp32=V5 SIMT family\n"
        << "  --retune            Ignore cache and autotune again\n"
        << "  --cache PATH        Unified V6 cache CSV\n"
        << "  --kernel NAME       Force one kernel in selected dtype family\n"
        << "  --profile-once      Launch forced/selected kernel once and exit\n"
        << "  --list-kernels      Print selected family's registry and exit\n"
        << "  --warmup N          Warmup launches (default 10)\n"
        << "  --repeat N          Timed launches/group (default 50)\n"
        << "  --groups N          Benchmark groups, median selected (default 5)\n"
        << "  --no-cublas         Skip cuBLAS performance baseline\n\n"
        << "Examples:\n"
        << "  "
        << prog
        << " 128 4096 4096 --retune\n"
        << "  "
        << prog
        << " 128 4096 4096 --dtype fp32 --retune\n"
        << "  "
        << prog
        << " 128 4096 4096 --kernel tc_m128_n128_k32_w64x64\n";
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
            arg == "--dtype"
            &&
            i + 1 < argc
        ) {
            opt.dtype =
                argv[++i];

        } else if (
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
            arg == "--profile-once"
        ) {
            opt.profile_once =
                true;

        } else if (
            arg == "--list-kernels"
        ) {
            opt.list_kernels =
                true;

        } else if (
            arg == "--no-cublas"
        ) {
            opt.no_cublas =
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
        }
    }

    return opt;
}

static void print_fp32_registry() {
    std::cout
        << "\nFP32 SIMT Kernel Registry\n"
        << "============================================================\n";

    for (
        const auto& k :
        get_kernel_registry()
    ) {
        std::cout
            << std::left
            << std::setw(36)
            << k.name
            << " CTA="
            << k.BM
            << "x"
            << k.BN
            << "x"
            << k.BK
            << " thread="
            << k.TM
            << "x"
            << k.TN
            << " warp="
            << k.WM
            << "x"
            << k.WN
            << " path="
            << kernel_path_name(
                k.path
            )
            << "\n";
    }
}

static void print_fp16_registry() {
    std::cout
        << "\nFP16 Tensor Core Kernel Registry\n"
        << "============================================================\n";

    for (
        const auto& k :
        get_tensor_core_registry()
    ) {
        std::cout
            << std::left
            << std::setw(38)
            << k.name
            << " CTA="
            << k.BM
            << "x"
            << k.BN
            << "x"
            << k.BK
            << " warp="
            << k.WM
            << "x"
            << k.WN
            << " MMA="
            << k.mma_m
            << "x"
            << k.mma_n
            << "x"
            << k.mma_k
            << " threads="
            << k.threads
            << " smem="
            << k.shared_memory_bytes
            << "\n";
    }
}

static double gemm_perf(
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

template<typename LaunchFn>
static float benchmark_groups(
    LaunchFn&& launch,
    int warmup,
    int repeat,
    int groups
) {
    for (
        int i = 0;
        i < warmup;
        ++i
    ) {
        launch();
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
            launch();
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

static void cublas_fp32_gemm(
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

static void cublas_tensor_core_gemm(
    cublasHandle_t handle,
    const half* A,
    const half* B,
    float* C,
    int M,
    int N,
    int K
) {
    const float alpha =
        1.0f;

    const float beta =
        0.0f;

    /*
     * Row-major trick:
     *
     * C = A B
     * =>
     * C^T = B^T A^T
     *
     * Inputs are FP16, accumulation/output is FP32.
     *
     * On Turing we request the Tensor-Op algorithm explicitly.
     */
    CUBLAS_CHECK(
        cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            M,
            K,
            &alpha,
            B,
            CUDA_R_16F,
            N,
            A,
            CUDA_R_16F,
            K,
            &beta,
            C,
            CUDA_R_32F,
            N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        )
    );
}

static float max_abs_error(
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

    return
        error;
}

static float max_rel_error(
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
        const float denom =
            std::max(
                std::abs(
                    b[i]
                ),
                1e-5f
            );

        error =
            std::max(
                error,
                std::abs(
                    a[i]
                    -
                    b[i]
                )
                /
                denom
            );
    }

    return
        error;
}

static bool allclose_fp16(
    const std::vector<float>& a,
    const std::vector<float>& b
) {
    constexpr float atol =
        1e-2f;

    constexpr float rtol =
        1e-2f;

    for (
        size_t i = 0;
        i < a.size();
        ++i
    ) {
        const float diff =
            std::abs(
                a[i]
                -
                b[i]
            );

        const float tol =
            atol
            +
            rtol
            *
            std::abs(
                b[i]
            );

        if (
            diff
            >
            tol
        ) {
            return false;
        }
    }

    return true;
}

static int run_fp32(
    const Options& opt,
    const cudaDeviceProp& prop
) {
    std::cout
        << "Family: FP32 SIMT (V5-compatible)\n";

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

    std::mt19937 rng(
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
            size_A
            *
            sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_B,
            size_B
            *
            sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_C,
            size_C
            *
            sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_ref,
            size_C
            *
            sizeof(float)
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
            size_A
            *
            sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_B,
            h_B.data(),
            size_B
            *
            sizeof(float),
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

    float latency =
        0.0f;

    double perf =
        0.0;

    TuneCache cache(
        opt.cache_path
    );

    if (
        !opt.forced_kernel.empty()
    ) {
        selected =
            find_kernel(
                opt.forced_kernel
            );

        if (!selected) {
            std::cerr
                << "Unknown FP32 kernel: "
                << opt.forced_kernel
                << "\n";

            print_fp32_registry();

            return 2;
        }

        std::cout
            << "Mode: forced kernel\n";

    } else {
        CacheRecord record;

        if (
            !opt.retune
            &&
            cache.lookup(
                prop.name,
                prop.major,
                prop.minor,
                "fp32_simt",
                opt.M,
                opt.N,
                opt.K,
                0,
                record
            )
        ) {
            selected =
                find_kernel(
                    record.kernel
                );

            if (selected) {
                latency =
                    record.latency_ms;

                perf =
                    record.tflops;

                std::cout
                    << "Mode: cache hit\n"
                    << "Cached kernel: "
                    << record.kernel
                    << "\n";
            }
        }

        if (!selected) {
            std::cout
                << "Mode: cache miss -> FP32 autotune\n";

            const TuneResult tuned =
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

            latency =
                tuned.latency_ms;

            perf =
                tuned.tflops;

            CacheRecord record;

            record.gpu =
                prop.name;

            record.cc_major =
                prop.major;

            record.cc_minor =
                prop.minor;

            record.dtype =
                "fp32_simt";

            record.M =
                opt.M;

            record.N =
                opt.N;

            record.K =
                opt.K;

            record.epilogue =
                0;

            record.kernel =
                selected->name;

            record.latency_ms =
                latency;

            record.tflops =
                perf;

            cache.upsert(
                record
            );

            cache.save();
        }
    }

    if (
        opt.profile_once
    ) {
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

        std::cout
            << "Profile kernel: "
            << selected->name
            << "\n";

        return 0;
    }

    if (
        !opt.forced_kernel.empty()
        ||
        latency == 0.0f
    ) {
        latency =
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

        perf =
            gemm_tflops(
                opt.M,
                opt.N,
                opt.K,
                latency
            );
    }

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

    cublas_fp32_gemm(
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
            size_C
            *
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            h_ref.data(),
            d_ref,
            size_C
            *
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    std::cout
        << "\nSelected FP32 kernel\n"
        << "  "
        << selected->name
        << "\n"
        << "  latency = "
        << latency
        << " ms\n"
        << "  TFLOPS = "
        << perf
        << "\n"
        << "Correctness max abs error = "
        << max_abs_error(
            h_C,
            h_ref
        )
        << "\n";

    if (
        !opt.no_cublas
    ) {
        const float cublas_ms =
            benchmark_groups(
                [&]() {
                    cublas_fp32_gemm(
                        handle,
                        d_A,
                        d_B,
                        d_ref,
                        opt.M,
                        opt.N,
                        opt.K
                    );
                },
                opt.warmup,
                opt.repeat,
                opt.groups
            );

        const double cublas_perf =
            gemm_perf(
                opt.M,
                opt.N,
                opt.K,
                cublas_ms
            );

        std::cout
            << "\ncuBLAS FP32\n"
            << "latency = "
            << cublas_ms
            << " ms\n"
            << "TFLOPS = "
            << cublas_perf
            << "\n"
            << "Relative performance = "
            << perf
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

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_ref));
    CUDA_CHECK(cudaFree(d_bias));

    return 0;
}

static int run_fp16(
    const Options& opt,
    const cudaDeviceProp& prop
) {
    std::cout
        << "Family: FP16 Tensor Core / FP32 accumulate\n";

    if (
        prop.major < 7
    ) {
        std::cerr
            << "Tensor Core WMMA requires compute capability 7.0+.\n";

        return 4;
    }

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

    std::vector<half>
    h_A(
        size_A
    );

    std::vector<half>
    h_B(
        size_B
    );

    std::mt19937 rng(
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
            __float2half_rn(
                dist(
                    rng
                )
            );
    }

    for (
        auto& x :
        h_B
    ) {
        x =
            __float2half_rn(
                dist(
                    rng
                )
            );
    }

    half* d_A =
        nullptr;

    half* d_B =
        nullptr;

    float* d_C =
        nullptr;

    float* d_ref =
        nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_A,
            size_A
            *
            sizeof(half)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_B,
            size_B
            *
            sizeof(half)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_C,
            size_C
            *
            sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_ref,
            size_C
            *
            sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_A,
            h_A.data(),
            size_A
            *
            sizeof(half),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_B,
            h_B.data(),
            size_B
            *
            sizeof(half),
            cudaMemcpyHostToDevice
        )
    );

    const TensorCoreConfig*
    selected =
        nullptr;

    float latency =
        0.0f;

    double perf =
        0.0;

    TuneCache cache(
        opt.cache_path
    );

    if (
        !opt.forced_kernel.empty()
    ) {
        selected =
            find_tensor_core_kernel(
                opt.forced_kernel
            );

        if (!selected) {
            std::cerr
                << "Unknown Tensor Core kernel: "
                << opt.forced_kernel
                << "\n";

            print_fp16_registry();

            return 2;
        }

        if (
            !tensor_core_problem_compatible(
                *selected,
                d_A,
                d_B,
                opt.M,
                opt.N,
                opt.K
            )
        ) {
            std::cerr
                << "Forced Tensor Core kernel is incompatible with this shape.\n";

            return 3;
        }

        std::cout
            << "Mode: forced Tensor Core kernel\n";

    } else {
        CacheRecord record;

        if (
            !opt.retune
            &&
            cache.lookup(
                prop.name,
                prop.major,
                prop.minor,
                "fp16_tc",
                opt.M,
                opt.N,
                opt.K,
                0,
                record
            )
        ) {
            selected =
                find_tensor_core_kernel(
                    record.kernel
                );

            if (
                selected
                &&
                tensor_core_problem_compatible(
                    *selected,
                    d_A,
                    d_B,
                    opt.M,
                    opt.N,
                    opt.K
                )
            ) {
                latency =
                    record.latency_ms;

                perf =
                    record.tflops;

                std::cout
                    << "Mode: cache hit\n"
                    << "Cached Tensor Core kernel: "
                    << record.kernel
                    << "\n"
                    << "Cached latency: "
                    << latency
                    << " ms\n"
                    << "Cached TFLOPS: "
                    << perf
                    << "\n";
            }
        }

        if (!selected) {
            std::cout
                << "Mode: cache miss -> Tensor Core autotune\n";

            const TensorCoreTuneResult tuned =
                autotune_tensor_core_gemm(
                    d_A,
                    d_B,
                    d_C,
                    opt.M,
                    opt.N,
                    opt.K,
                    opt.warmup,
                    opt.repeat,
                    opt.groups
                );

            selected =
                tuned.kernel;

            latency =
                tuned.latency_ms;

            perf =
                tuned.tflops;

            CacheRecord record;

            record.gpu =
                prop.name;

            record.cc_major =
                prop.major;

            record.cc_minor =
                prop.minor;

            record.dtype =
                "fp16_tc";

            record.M =
                opt.M;

            record.N =
                opt.N;

            record.K =
                opt.K;

            record.epilogue =
                0;

            record.kernel =
                selected->name;

            record.latency_ms =
                latency;

            record.tflops =
                perf;

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
        launch_tensor_core_kernel(
            *selected,
            d_A,
            d_B,
            d_C,
            opt.M,
            opt.N,
            opt.K
        );

        CUDA_CHECK(
            cudaDeviceSynchronize()
        );

        std::cout
            << "Profile Tensor Core kernel: "
            << selected->name
            << "\n";

        return 0;
    }

    if (
        !opt.forced_kernel.empty()
        ||
        latency == 0.0f
    ) {
        latency =
            benchmark_tensor_core_kernel(
                *selected,
                d_A,
                d_B,
                d_C,
                opt.M,
                opt.N,
                opt.K,
                opt.warmup,
                opt.repeat,
                opt.groups
            );

        perf =
            tensor_core_tflops(
                opt.M,
                opt.N,
                opt.K,
                latency
            );
    }

    cublasHandle_t handle{};

    CUBLAS_CHECK(
        cublasCreate(
            &handle
        )
    );

    launch_tensor_core_kernel(
        *selected,
        d_A,
        d_B,
        d_C,
        opt.M,
        opt.N,
        opt.K
    );

    cublas_tensor_core_gemm(
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
            size_C
            *
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            h_ref.data(),
            d_ref,
            size_C
            *
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    const float abs_err =
        max_abs_error(
            h_C,
            h_ref
        );

    const float rel_err =
        max_rel_error(
            h_C,
            h_ref
        );

    const bool close =
        allclose_fp16(
            h_C,
            h_ref
        );

    std::cout
        << "\nSelected Tensor Core kernel\n"
        << "  "
        << selected->name
        << "\n"
        << "  CTA = "
        << selected->BM
        << "x"
        << selected->BN
        << "x"
        << selected->BK
        << "\n"
        << "  Warp = "
        << selected->WM
        << "x"
        << selected->WN
        << "\n"
        << "  MMA = 16x16x16\n"
        << "  latency = "
        << latency
        << " ms\n"
        << "  TFLOPS = "
        << perf
        << "\n";

    std::cout
        << "\nCorrectness vs cuBLAS FP16->FP32\n"
        << "Max abs error = "
        << abs_err
        << "\n"
        << "Max relative error = "
        << rel_err
        << "\n"
        << "allclose(atol=1e-2, rtol=1e-2) = "
        << (
            close
            ?
            "PASS"
            :
            "FAIL"
        )
        << "\n";

    if (
        !opt.no_cublas
    ) {
        const float cublas_ms =
            benchmark_groups(
                [&]() {
                    cublas_tensor_core_gemm(
                        handle,
                        d_A,
                        d_B,
                        d_ref,
                        opt.M,
                        opt.N,
                        opt.K
                    );
                },
                opt.warmup,
                opt.repeat,
                opt.groups
            );

        const double cublas_perf =
            gemm_perf(
                opt.M,
                opt.N,
                opt.K,
                cublas_ms
            );

        std::cout
            << "\ncuBLAS Tensor Core baseline\n"
            << "FP16 input / FP32 accumulate+output\n"
            << "latency = "
            << cublas_ms
            << " ms\n"
            << "TFLOPS = "
            << cublas_perf
            << "\n"
            << "Relative performance = "
            << perf
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

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_ref));

    return
        close
        ?
        0
        :
        5;
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
        opt.dtype != "fp16"
        &&
        opt.dtype != "fp32"
    ) {
        std::cerr
            << "--dtype must be fp16 or fp32.\n";

        return 1;
    }

    if (
        opt.list_kernels
    ) {
        if (
            opt.dtype == "fp16"
        ) {
            print_fp16_registry();
        } else {
            print_fp32_registry();
        }

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
        << "dtype: "
        << opt.dtype
        << "\n"
        << "Cache: "
        << opt.cache_path
        << "\n";

    if (
        opt.dtype == "fp16"
    ) {
        return
            run_fp16(
                opt,
                prop
            );
    }

    return
        run_fp32(
            opt,
            prop
        );
}
