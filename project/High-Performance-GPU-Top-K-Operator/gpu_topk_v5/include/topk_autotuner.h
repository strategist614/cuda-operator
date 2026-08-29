#pragma once

#include "topk_registry.h"

struct TopKTuneResult {
    const TopKKernelConfig* kernel = nullptr;
    float latency_ms = 0.0f;
    double gelem_per_s = 0.0;
};

double topk_gelem_per_s(
    int batch,
    int n,
    float latency_ms
);

float benchmark_topk_kernel(
    const TopKKernelConfig& kernel,
    const float* input,
    float* output_values,
    int* output_indices,
    int batch,
    int n,
    int k,
    int warmup = 5,
    int repeat = 50,
    int groups = 5
);

TopKTuneResult autotune_topk(
    const float* input,
    float* output_values,
    int* output_indices,
    int batch,
    int n,
    int k,
    int warmup = 5,
    int repeat = 50,
    int groups = 5
);
