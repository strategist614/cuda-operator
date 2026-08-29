#include "benchmark.h"
#include "common.h"
#include "gemm.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static float max_abs_error(
    const std::vector<float>& a,
    const std::vector<float>& b
) {
    float err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        err = std::max(err, std::abs(a[i] - b[i]));
    }
    return err;
}

static void usage(const char* prog) {
    std::cout
        << "Usage:\n"
        << "  " << prog << " M N K [--all-kernels] [--bias] [--silu]\n\n"
        << "Examples:\n"
        << "  " << prog << " 128 4096 4096\n"
        << "  " << prog << " 16 4096 4096 --all-kernels\n"
        << "  " << prog << " 512 512 512 --bias\n"
        << "  " << prog << " 512 512 512 --silu\n";
}

int main(int argc, char** argv) {
    if (argc < 4) {
        usage(argv[0]);
        return 0;
    }

    const int M = std::stoi(argv[1]);
    const int N = std::stoi(argv[2]);
    const int K = std::stoi(argv[3]);

    bool all_kernels = false;
    EpilogueType epilogue = EpilogueType::NONE;

    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--all-kernels") {
            all_kernels = true;
        } else if (arg == "--bias") {
            epilogue = EpilogueType::BIAS;
        } else if (arg == "--silu") {
            epilogue = EpilogueType::BIAS_SILU;
        }
    }

    if (M <= 0 || N <= 0 || K <= 0) {
        std::cerr << "M/N/K must be positive.\n";
        return 1;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Shape: M=" << M << " N=" << N << " K=" << K << "\n";

    const size_t size_A = static_cast<size_t>(M) * K;
    const size_t size_B = static_cast<size_t>(K) * N;
    const size_t size_C = static_cast<size_t>(M) * N;

    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    std::vector<float> h_bias(N, 0.0f);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    for (auto& x : h_A) x = dist(rng);
    for (auto& x : h_B) x = dist(rng);
    for (auto& x : h_bias) x = dist(rng);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    float *d_ref = nullptr, *d_bias = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref, size_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, static_cast<size_t>(N) * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        d_A, h_A.data(), size_A * sizeof(float), cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        d_B, h_B.data(), size_B * sizeof(float), cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        d_bias, h_bias.data(), static_cast<size_t>(N) * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    cublasHandle_t handle{};
    CUBLAS_CHECK(cublasCreate(&handle));

    // Correctness check only for plain GEMM because cuBLAS baseline here
    // does not apply the fused epilogue.
    if (epilogue == EpilogueType::NONE) {
        launch_gemm(d_A, d_B, d_C, d_bias, M, N, K, EpilogueType::NONE);

        // Compute reference using cuBLAS.
        const float alpha = 1.0f;
        const float beta  = 0.0f;

        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, N,
            d_A, K,
            &beta,
            d_ref, N
        ));

        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_C(size_C);
        std::vector<float> h_ref(size_C);

        CUDA_CHECK(cudaMemcpy(
            h_C.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost
        ));
        CUDA_CHECK(cudaMemcpy(
            h_ref.data(), d_ref, size_C * sizeof(float), cudaMemcpyDeviceToHost
        ));

        const float err = max_abs_error(h_C, h_ref);
        std::cout << "Max abs error: " << err << "\n";
    }

    const int warmup = 10;
    const int repeat = 100;

    KernelType selected = select_kernel(M, N, K);
    std::cout << "Dispatcher selected: " << kernel_name(selected) << "\n\n";

    auto print_result = [&](const std::string& name, float ms) {
        std::cout
            << std::left << std::setw(14) << name
            << " latency=" << std::setw(10) << ms << " ms"
            << " TFLOPS=" << calculate_tflops(M, N, K, ms)
            << "\n";
    };

    if (all_kernels) {
        print_result(
            "small_m",
            benchmark_custom(
                KernelType::SMALL_M,
                d_A, d_B, d_C, d_bias,
                M, N, K, epilogue,
                warmup, repeat
            )
        );

        print_result(
            "regular",
            benchmark_custom(
                KernelType::REGULAR,
                d_A, d_B, d_C, d_bias,
                M, N, K, epilogue,
                warmup, repeat
            )
        );

        print_result(
            "skinny_n",
            benchmark_custom(
                KernelType::SKINNY_N,
                d_A, d_B, d_C, d_bias,
                M, N, K, epilogue,
                warmup, repeat
            )
        );
    } else {
        print_result(
            "dispatch",
            benchmark_dispatch(
                d_A, d_B, d_C, d_bias,
                M, N, K, epilogue,
                warmup, repeat
            )
        );
    }

    if (epilogue == EpilogueType::NONE) {
        print_result(
            "cuBLAS",
            benchmark_cublas(
                handle,
                d_A, d_B, d_ref,
                M, N, K,
                warmup, repeat
            )
        );
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_ref));
    CUDA_CHECK(cudaFree(d_bias));

    return 0;
}
