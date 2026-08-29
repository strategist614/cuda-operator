#pragma once

#include <cuda_runtime.h>

#include <string>
#include <vector>

using TopKLauncher = void (*)(
    const float* input,
    float* output_values,
    int* output_indices,
    int batch,
    int n,
    int k,
    cudaStream_t stream
);

struct TopKKernelConfig {
    const char* name;
    const char* family;

    int min_k;
    int max_k;

    bool specialized_k;
    bool autotune_candidate;

    TopKLauncher launcher;

    bool (*supports)(int batch, int n, int k);
};

const std::vector<TopKKernelConfig>&
get_topk_registry();

const TopKKernelConfig*
find_topk_kernel(
    const std::string& name
);

std::vector<const TopKKernelConfig*>
get_compatible_topk_kernels(
    int batch,
    int n,
    int k
);
