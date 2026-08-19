#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>

using namespace std;

#define checkCudaErrors(call)                                  \
    do {                                                       \
        cudaError_t err = (call);                              \
        if (err != cudaSuccess) {                              \
            std::cerr << "CUDA Error: "                        \
                      << cudaGetErrorString(err)                \
                      << " at "                                 \
                      << __FILE__                               \
                      << ":"                                    \
                      << __LINE__                               \
                      << std::endl;                             \
            std::exit(EXIT_FAILURE);                           \
        }                                                      \
    } while (0)

__global__ void transpose(float *d_in, float *d_out, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        d_out[col * m + row] = d_in[row * n + col];
    }
}

int main()
{
    int N = 1024;   // 列数
    int M = 1024;   // 行数

    const size_t bytes = M * N * sizeof(float);

    // 1. CPU 内存
    float *h_in  = new float[M * N];
    float *h_out = new float[M * N];

    // 初始化输入矩阵
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            h_in[row * N + col] = row * N + col;
        }
    }

    // 2. GPU 内存
    float *d_in = nullptr;
    float *d_out = nullptr;

    checkCudaErrors(cudaMalloc(&d_in, bytes));
    checkCudaErrors(cudaMalloc(&d_out, bytes));

    // 3. CPU -> GPU
    checkCudaErrors(cudaMemcpy(
        d_in,
        h_in,
        bytes,
        cudaMemcpyHostToDevice
    ));

    // 4. 设置线程块和网格
    dim3 block(16, 16);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    // 5. 启动 kernel
    transpose<<<grid, block>>>(d_in, d_out, M, N);

    // 检查 kernel launch 是否出错
    checkCudaErrors(cudaGetLastError());

    // 等待 GPU 执行完成
    checkCudaErrors(cudaDeviceSynchronize());

    // 6. GPU -> CPU
    checkCudaErrors(cudaMemcpy(
        h_out,
        d_out,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    // 7. 验证结果
    bool correct = true;

    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {

            // 输入 A[row][col]
            float input = h_in[row * N + col];

            // 转置后应该在 B[col][row]
            float output = h_out[col * M + row];

            if (input != output) {
                correct = false;

                cout << "Error at row = " << row
                     << ", col = " << col
                     << ", input = " << input
                     << ", output = " << output
                     << endl;

                break;
            }
        }

        if (!correct) {
            break;
        }
    }

    if (correct) {
        cout << "Transpose correct!" << endl;
    }

    // 8. 释放内存
    delete[] h_in;
    delete[] h_out;

    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));

    return 0;
}