#pragma once
#include "gemm.h"

struct TuneResult {
    const KernelConfig* kernel = nullptr;
    float latency_ms = 0.0f;
    double tflops = 0.0;
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
    int repeat = 50,
    int groups = 5
);

TuneResult autotune_gemm(
    const float* A,
    const float* B,
    float* C,
    const float* bias,
    int M, int N, int K,
    EpilogueType epilogue,
    int warmup = 10,
    int repeat = 50,
    int groups = 5
);

double gemm_tflops(int M, int N, int K, float ms);
