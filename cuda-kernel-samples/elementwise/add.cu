#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// ============================================================

#define checkCudaErrors(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr                                         \
                << "CUDA Error: "                             \
                << cudaGetErrorString(err)                    \
                << " at "                                     \
                << __FILE__                                   \
                << ":"                                        \
                << __LINE__                                   \
                << std::endl;                                 \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)
// 1. 向上取整
#define CEIL(a, b) (((a) + (b) - 1) / (b))

// 2. FLOAT4，用于向量化访存
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])


// CUDA kernel
__global__ void add(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}


int main()
{
    const int N = 1024 * 1024;
    const size_t bytes = N * sizeof(float);

    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    checkCudaErrors(cudaMalloc(&d_a, bytes));
    checkCudaErrors(cudaMalloc(&d_b, bytes));
    checkCudaErrors(cudaMalloc(&d_c, bytes));

    checkCudaErrors(
        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice)
    );

    checkCudaErrors(
        cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice)
    );


    int blocksize = 256;
    int gridsize = CEIL(N, blocksize);

    add<<<gridsize, blocksize>>>(d_a, d_b, d_c, N);

    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(
        cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost)
    );

    bool success = true;

    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];

        if (fabs(h_c[i] - expected) > 1e-5f) {
            std::cout << "Error at index " << i
                      << ": expected = " << expected
                      << ", result = " << h_c[i]
                      << std::endl;

            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "CUDA vector add success!" << std::endl;
    }

    for (int i = 0; i < 10; ++i) {
        std::cout
            << h_a[i]
            << " + "
            << h_b[i]
            << " = "
            << h_c[i]
            << std::endl;
    }

    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;


    return 0;
}