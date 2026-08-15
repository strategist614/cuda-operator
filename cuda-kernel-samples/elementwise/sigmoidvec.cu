#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

using namespace std;

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

__global__ void add_float4(float *a, float *b, int N){
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if(idx + 3 < N){
       float4 a4 = FLOAT4(a[idx]);
        float4 b4;
        b4.x = 1.0 / (1.0f+ expf(-a4.x));
        b4.y = 1.0 / (1.0f + expf(-a4.y));
        b4.z = 1.0 / (1.0f + expf(-a4.z));
        b4.w = 1.0 / (1.0f + expf(-a4.w));

        FLOAT4(b[idx]) = b4;
    }
}

int main()
{
    int N = 1024;
    const size_t bytes = N * sizeof(float);
    float *h_a = new float[N];
    float *h_b = new float[N];
    for (int i = 0; i < N; i++) {
       h_a[i] = static_cast<float>(i);
    }
    float *d_a = nullptr;
    float *d_b = nullptr;
    checkCudaErrors(cudaMalloc(&d_a, bytes));
    checkCudaErrors(cudaMalloc(&d_b, bytes));

    checkCudaErrors(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    int blocksize = 256;
    int num = CEIL(N, 4);
    int gridsize = CEIL(num, blocksize);
    add_float4<<<gridsize, blocksize>>>(d_a, d_b, N);


    checkCudaErrors(cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost));

    for(int i = 0;i < 10;i ++) cout << h_b[i] << ' ';

    return 0;
}