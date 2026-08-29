#pragma once

#include <cuda_runtime.h>

void launch_topk_naive(
    const float* input,
    float* output_values,
    int* output_indices,
    int batch,
    int n,
    int k,
    cudaStream_t stream = nullptr
);
