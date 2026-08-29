#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                     \
do {                                                                         \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
        std::cerr << "CUDA error: " << cudaGetErrorString(err__)             \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n";          \
        std::exit(EXIT_FAILURE);                                             \
    }                                                                        \
} while (0)

#define CUBLAS_CHECK(call)                                                   \
do {                                                                         \
    cublasStatus_t st__ = (call);                                            \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                     \
        std::cerr << "cuBLAS error: " << static_cast<int>(st__)              \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n";          \
        std::exit(EXIT_FAILURE);                                             \
    }                                                                        \
} while (0)
