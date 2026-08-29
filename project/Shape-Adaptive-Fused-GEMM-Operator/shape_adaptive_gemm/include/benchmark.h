#pragma once
#include "gemm.h"
#include <cublas_v2.h>

float benchmark_custom(
    KernelType type,
    const float* A,
    const float* B,
    float* C,
    const float* bias,
    int M,
    int N,
    int K,
    EpilogueType epilogue,
    int warmup,
    int repeat
);

float benchmark_dispatch(
    const float* A,
    const float* B,
    float* C,
    const float* bias,
    int M,
    int N,
    int K,
    EpilogueType epilogue,
    int warmup,
    int repeat
);

float benchmark_cublas(
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    int warmup,
    int repeat
);

double calculate_tflops(int M, int N, int K, float ms);
