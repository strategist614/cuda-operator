#pragma once
#include "gemm.h"

struct TuneResult {
    const KernelConfig* kernel;
    float latency_ms;
    double tflops;
};

bool kernel_supported_on_device(const KernelConfig& kernel);

float benchmark_kernel(
    const KernelConfig& kernel,
    const float* A,
    const float* B,
    float* C,
    const float* bias,
    int M, int N, int K,
    EpilogueType epilogue,
    int warmup = 10,
    int repeat = 50
);

TuneResult autotune_gemm(
    const float* A,
    const float* B,
    float* C,
    const float* bias,
    int M, int N, int K,
    EpilogueType epilogue,
    int warmup = 10,
    int repeat = 50
);
