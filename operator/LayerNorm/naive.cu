#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while (0)

void layernorm_cpu(
    const std::vector<float>& x,
    std::vector<float>& y,
    const std::vector<float>& gamma,
    const std::vector<float>& beta,
    int rows,
    int cols,
    float eps
) {
    for (int r = 0; r < rows; r++) {
        float mean = 0.0f;

        for (int c = 0; c < cols; c++) {
            mean += x[r * cols + c];
        }
        mean /= cols;

        float var = 0.0f;

        for (int c = 0; c < cols; c++) {
            float v = x[r * cols + c] - mean;
            var += v * v;
        }
        var /= cols;

        float rstd = 1.0f / std::sqrt(var + eps);

        for (int c = 0; c < cols; c++) {
            float norm = (x[r * cols + c] - mean) * rstd;
            y[r * cols + c] = norm * gamma[c] + beta[c];
        }
    }
}

__global__ void layernorm_naive_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int rows,
    int cols,
    float eps
){
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if(row >= rows) return;

    const float* x_row = x + row * cols;
    float* y_row = y + row * cols;

    float sum = 0.0f;
    // blockDim.x 表示一个 block 中的线程数，tid 表示当前线程在 block 中的索引
    for(int i = tid; i < cols; i += blockDim.x){
        sum += x_row[i];
    }   

    sdata[tid] = sum;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid < stride){
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float mean = sdata[0] / cols;

    float var_sum = 0.0f;
    for(int i = tid; i < cols; i += blockDim.x){
        float v = x_row[i] - mean;
        var_sum += v * v;
    }

    sdata[tid] = var_sum;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid < stride){
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float var = sdata[0] / cols;
    float rstd = sqrtf(var + eps);

    for(int i = tid; i < cols; i += blcokDim.x){
        float v = x_row[i];
        float norm = (v - mean) * rstd;
        y_row[i] = norm * gamma[i] + beta[i];
    }
}

void layernorm_cuda(
    const float* d_x,
    const float* d_gamma,
    const float* d_beta,
    float* d_y,
    int rows,
    int cols,
    float eps
) {
    int block = 256;
    int grid = rows;

    size_t shared_mem = block * sizeof(float);

    layernorm_naive_kernel<<<grid, block, shared_mem>>>(
        d_x, d_gamma, d_beta, d_y,
        rows, cols,
        eps
    );
}

int main() {
    int rows = 4;
    int cols = 8;

    std::vector<float> h_x(rows * cols);
    std::vector<float> h_y(rows * cols);
    std::vector<float> h_y_ref(rows * cols);

    std::vector<float> gamma(cols, 1.0f);
    std::vector<float> beta(cols, 0.0f);

    for (int i = 0; i < rows * cols; i++) {
        h_x[i] = (float)(i % 10);
    }

    float *d_x, *d_y, *d_gamma, *d_beta;

    CHECK_CUDA(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, h_y.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gamma, cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_beta, cols * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gamma, gamma.data(), cols * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_beta, beta.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    layernorm_cuda(d_x, d_gamma, d_beta, d_y, rows, cols, 1e-5f);

    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(float), cudaMemcpyDeviceToHost));

    layernorm_cpu(h_x, h_y_ref, gamma, beta, rows, cols, 1e-5f);

    float max_err = 0.0f;

    for (int i = 0; i < rows * cols; i++) {
        max_err = std::max(max_err, std::abs(h_y[i] - h_y_ref[i]));
    }

    std::cout << "Max error: " << max_err << std::endl;

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_gamma);
    cudaFree(d_beta);

    return 0;
}