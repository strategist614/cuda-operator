#pragma once

#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err__) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)
