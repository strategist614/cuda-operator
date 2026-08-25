#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace norm_test {

constexpr int ROWS = 4096;
constexpr int HIDDEN = 1024;
constexpr int THREADS = 256;
constexpr int WARMUP = 10;
constexpr int REPEATS = 100;
constexpr float EPSILON = 1.0e-5f;
constexpr float ERROR_TOLERANCE = 1.0e-4f;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t error = (call);                                      \
        if (error != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(error)           \
                      << " at " << __FILE__ << ':' << __LINE__ << '\n';       \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

inline void initialize(std::vector<float>& x,
                       std::vector<float>& gamma,
                       std::vector<float>& beta) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float index = static_cast<float>(i);
        x[i] = 0.80f * std::sin(index * 0.0013f) +
               0.20f * std::cos(index * 0.0007f);
    }

    for (int col = 0; col < HIDDEN; ++col) {
        const float index = static_cast<float>(col);
        gamma[col] = 0.75f + 0.25f * std::cos(index * 0.017f);
        beta[col] = 0.10f * std::sin(index * 0.013f);
    }
}

inline void layernorm_reference(const std::vector<float>& x,
                                const std::vector<float>& gamma,
                                const std::vector<float>& beta,
                                std::vector<float>& y) {
    for (int row = 0; row < ROWS; ++row) {
        const std::size_t offset = static_cast<std::size_t>(row) * HIDDEN;
        double sum = 0.0;
        double square_sum = 0.0;

        for (int col = 0; col < HIDDEN; ++col) {
            const double value = static_cast<double>(x[offset + col]);
            sum += value;
            square_sum += value * value;
        }

        const double mean = sum / HIDDEN;
        const double variance =
            std::max(square_sum / HIDDEN - mean * mean, 0.0);
        const double inverse_std = 1.0 / std::sqrt(variance + EPSILON);

        for (int col = 0; col < HIDDEN; ++col) {
            const double normalized =
                (static_cast<double>(x[offset + col]) - mean) * inverse_std;
            y[offset + col] = static_cast<float>(
                normalized * static_cast<double>(gamma[col]) +
                static_cast<double>(beta[col]));
        }
    }
}

inline void rmsnorm_reference(const std::vector<float>& x,
                              const std::vector<float>& gamma,
                              const std::vector<float>&,
                              std::vector<float>& y) {
    for (int row = 0; row < ROWS; ++row) {
        const std::size_t offset = static_cast<std::size_t>(row) * HIDDEN;
        double square_sum = 0.0;

        for (int col = 0; col < HIDDEN; ++col) {
            const double value = static_cast<double>(x[offset + col]);
            square_sum += value * value;
        }

        const double inverse_rms =
            1.0 / std::sqrt(square_sum / HIDDEN + EPSILON);

        for (int col = 0; col < HIDDEN; ++col) {
            y[offset + col] = static_cast<float>(
                static_cast<double>(x[offset + col]) * inverse_rms *
                static_cast<double>(gamma[col]));
        }
    }
}

using ReferenceFunction = void (*)(const std::vector<float>&,
                                   const std::vector<float>&,
                                   const std::vector<float>&,
                                   std::vector<float>&);

template <typename LaunchFunction>
int run(const char* kernel_name,
        int logical_float_accesses_per_element,
        ReferenceFunction reference,
        LaunchFunction launch) {
    static_assert(HIDDEN % 4 == 0,
                  "optimized kernels require a float4-aligned hidden size");

    const std::size_t element_count =
        static_cast<std::size_t>(ROWS) * HIDDEN;
    const std::size_t tensor_bytes = element_count * sizeof(float);
    const std::size_t parameter_bytes = HIDDEN * sizeof(float);

    std::vector<float> h_x(element_count);
    std::vector<float> h_gamma(HIDDEN);
    std::vector<float> h_beta(HIDDEN);
    std::vector<float> h_reference(element_count);
    std::vector<float> h_output(element_count);

    initialize(h_x, h_gamma, h_beta);
    reference(h_x, h_gamma, h_beta, h_reference);

    float* d_x = nullptr;
    float* d_gamma = nullptr;
    float* d_beta = nullptr;
    float* d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_gamma, parameter_bytes));
    CUDA_CHECK(cudaMalloc(&d_beta, parameter_bytes));
    CUDA_CHECK(cudaMalloc(&d_y, tensor_bytes));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), tensor_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), parameter_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), parameter_bytes,
                          cudaMemcpyHostToDevice));

    for (int iteration = 0; iteration < WARMUP; ++iteration) {
        launch(d_x, d_gamma, d_beta, d_y);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < REPEATS; ++iteration) {
        launch(d_x, d_gamma, d_beta, d_y);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const double average_ms = static_cast<double>(total_ms) / REPEATS;

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_y, tensor_bytes,
                          cudaMemcpyDeviceToHost));

    double max_abs_error = 0.0;
    double max_reference = 0.0;
    double reference_checksum = 0.0;
    double output_checksum = 0.0;
    for (std::size_t i = 0; i < element_count; ++i) {
        max_abs_error = std::max(
            max_abs_error,
            std::abs(static_cast<double>(h_output[i]) - h_reference[i]));
        max_reference = std::max(
            max_reference, std::abs(static_cast<double>(h_reference[i])));
        reference_checksum += h_reference[i];
        output_checksum += h_output[i];
    }

    const double relative_linf =
        max_abs_error / std::max(max_reference, 1.0e-12);
    const double logical_bytes =
        static_cast<double>(element_count) *
        logical_float_accesses_per_element * sizeof(float);
    const double effective_bandwidth_gbps = logical_bytes / (average_ms * 1.0e6);
    const bool passed = max_abs_error <= ERROR_TOLERANCE;

    std::cout << std::fixed << std::setprecision(8)
              << "kernel=" << kernel_name << '\n'
              << "shape=" << ROWS << 'x' << HIDDEN << '\n'
              << "epsilon=" << EPSILON << '\n'
              << "max_abs_error=" << max_abs_error << '\n'
              << "relative_linf=" << relative_linf << '\n'
              << "reference_checksum=" << reference_checksum << '\n'
              << "output_checksum=" << output_checksum << '\n'
              << "average_ms=" << average_ms << '\n'
              << "effective_bandwidth_gbps=" << effective_bandwidth_gbps
              << '\n'
              << "first_values=";
    for (int i = 0; i < 8; ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        std::cout << h_output[i];
    }
    std::cout << '\n' << "status=" << (passed ? "PASS" : "FAIL") << '\n';

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_y));
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace norm_test
