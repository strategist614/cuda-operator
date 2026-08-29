#pragma once
#include <cuda_runtime.h>
#include <string>
#include <vector>

enum class EpilogueType {
    NONE = 0,
    BIAS = 1,
    BIAS_SILU = 2
};

using GemmLauncher = void (*)(
    const float* A,
    const float* B,
    float* C,
    const float* bias,
    int M, int N, int K,
    EpilogueType epilogue,
    cudaStream_t stream
);

struct KernelConfig {
    const char* name;
    int BM, BN, BK;
    int TM, TN;
    int threads;
    int shared_memory_bytes;
    GemmLauncher launcher;
};

const std::vector<KernelConfig>& get_kernel_registry();
const KernelConfig* find_kernel(const std::string& name);
const KernelConfig& heuristic_select_kernel(int M, int N, int K);

void launch_kernel(
    const KernelConfig& config,
    const float* A,
    const float* B,
    float* C,
    const float* bias,
    int M, int N, int K,
    EpilogueType epilogue,
    cudaStream_t stream = nullptr
);
