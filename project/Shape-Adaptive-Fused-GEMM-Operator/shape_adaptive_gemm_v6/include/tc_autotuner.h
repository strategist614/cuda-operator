#pragma once

#include "tensor_core.h"

struct TensorCoreTuneResult {
    const TensorCoreConfig* kernel = nullptr;
    float latency_ms = 0.0f;
    double tflops = 0.0;
};

bool tensor_core_kernel_supported_on_device(
    const TensorCoreConfig& kernel
);

float benchmark_tensor_core_kernel(
    const TensorCoreConfig& kernel,
    const half* A,
    const half* B,
    float* C,
    int M,
    int N,
    int K,
    int warmup = 10,
    int repeat = 50,
    int groups = 5
);

TensorCoreTuneResult autotune_tensor_core_gemm(
    const half* A,
    const half* B,
    float* C,
    int M,
    int N,
    int K,
    int warmup = 10,
    int repeat = 50,
    int groups = 5
);

double tensor_core_tflops(
    int M,
    int N,
    int K,
    float ms
);
