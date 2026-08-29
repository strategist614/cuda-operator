#pragma once
#include <cuda_runtime.h>
#include <string>
#include <vector>

enum class EpilogueType {
    NONE = 0,
    BIAS = 1,
    BIAS_SILU = 2
};

enum class KernelPath {
    SCALAR = 0,
    VEC4 = 1,
    PIPE_VEC4 = 2,
    WARP_PIPE_VEC4 = 3
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

    // 0/0 means the kernel does not use an explicit warp tile.
    int WM, WN;

    int threads;
    int shared_memory_bytes;

    KernelPath path;
    GemmLauncher launcher;
};

const std::vector<KernelConfig>& get_kernel_registry();
const KernelConfig* find_kernel(const std::string& name);
const char* kernel_path_name(KernelPath path);

bool kernel_problem_compatible(
    const KernelConfig& kernel,
    const float* A,
    const float* B,
    int M, int N, int K
);

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
