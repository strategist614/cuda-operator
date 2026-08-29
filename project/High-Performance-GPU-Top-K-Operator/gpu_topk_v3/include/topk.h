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

bool topk_register_v1_supported(int k);

void launch_topk_register_v1(
    const float* input,
    float* output_values,
    int* output_indices,
    int batch,
    int n,
    int k,
    cudaStream_t stream = nullptr
);

bool topk_warp_v2_supported(int k);

void launch_topk_warp_v2(
    const float* input,
    float* output_values,
    int* output_indices,
    int batch,
    int n,
    int k,
    cudaStream_t stream = nullptr
);

bool topk_batch_v3_supported(int k);

void launch_topk_batch_v3(
    const float* input,
    float* output_values,
    int* output_indices,
    int batch,
    int n,
    int k,
    cudaStream_t stream = nullptr
);
