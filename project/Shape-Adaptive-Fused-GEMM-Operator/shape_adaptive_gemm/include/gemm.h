#pragma once
#include <cuda_runtime.h>
#include <string>

enum class EpilogueType {
    NONE = 0,
    BIAS = 1,
    BIAS_SILU = 2
};

enum class KernelType {
    SMALL_M = 0,
    REGULAR = 1,
    SKINNY_N = 2
};

KernelType select_kernel(int M, int N, int K);
std::string kernel_name(KernelType type);

void launch_gemm(
    const float* A,
    const float* B,
    float* C,
    const float* bias,
    int M,
    int N,
    int K,
    EpilogueType epilogue,
    cudaStream_t stream = nullptr
);

void launch_gemm_forced(
    KernelType type,
    const float* A,
    const float* B,
    float* C,
    const float* bias,
    int M,
    int N,
    int K,
    EpilogueType epilogue,
    cudaStream_t stream = nullptr
);
