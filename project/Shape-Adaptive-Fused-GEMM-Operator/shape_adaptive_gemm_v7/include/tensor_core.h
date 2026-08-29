#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

enum class TensorCorePath {
    V6_WMMA = 0,
    FRAGPIPE = 1,
    FRAGPIPE_PADDED = 2
};

using TensorCoreLauncher = void (*)(
    const half* A,
    const half* B,
    float* C,
    int M,
    int N,
    int K,
    cudaStream_t stream
);

struct TensorCoreConfig {
    const char* name;

    int BM;
    int BN;
    int BK;

    int WM;
    int WN;

    int mma_m;
    int mma_n;
    int mma_k;

    int threads;

    int shared_stages;
    int fragment_stages;
    int smem_padding;

    int shared_memory_bytes;

    TensorCorePath path;

    TensorCoreLauncher launcher;
};

const std::vector<TensorCoreConfig>&
get_tensor_core_registry();

const TensorCoreConfig*
find_tensor_core_kernel(
    const std::string& name
);

const char*
tensor_core_path_name(
    TensorCorePath path
);

bool tensor_core_problem_compatible(
    const TensorCoreConfig& kernel,
    const half* A,
    const half* B,
    int M,
    int N,
    int K
);

void launch_tensor_core_kernel(
    const TensorCoreConfig& config,
    const half* A,
    const half* B,
    float* C,
    int M,
    int N,
    int K,
    cudaStream_t stream = nullptr
);
