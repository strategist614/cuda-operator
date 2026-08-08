#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#define CHECK_CUDA(call)                                                        \
do {                                                                            \
    cudaError_t err = call;                                                     \
    if (err != cudaSuccess) {                                                   \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                  \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;       \
        exit(1);                                                                \
    }                                                                           \
} while (0)

void layernorm_cpu(
    const std::vector<float>& x,
    std::vector<float>& y,
    const std::vector<float>& gamma, // 缩放系数
    const std::vector<float>& beta, // 偏移系数
    int rows,
    int cols,
    float eps
) {
    for (int r = 0; r < rows; r++) {
        float mean = 0.0f;

        for (int c = 0; c < cols; c++) {
            mean += x[r * cols + c];
        }
        mean /= static_cast<float>(cols);

        float var = 0.0f;

        for (int c = 0; c < cols; c++) {
            float diff = x[r * cols + c] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(cols);

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
) {
    extern __shared__ float sdata[]; // shared memory

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    const float* x_row = x + row * cols; // 二维排布 定位到当前的输入元素
    float* y_row = y + row * cols; // 定位到当前的输出元素

    // =========================
    // Step 1: mean
    // =========================
    float sum = 0.0f;
    // 这里是多个线程并行计算每一行的均值, 每个线程负责计算一部分元素的和，实际上是跳着计算的
    for (int i = tid; i < cols; i += blockDim.x) {
        sum += x_row[i];
    }
    // 放到共享内存中, 以便后续的归约操作
    sdata[tid] = sum;
    // 要进行同步 不然 sdata 的值可能还没写完就被其他线程读取了
    __syncthreads();
    // 归约操作, 计算总和
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float mean = sdata[0] / static_cast<float>(cols);

    // =========================
    // Step 2: variance
    // =========================
    float var_sum = 0.0f;
    // 针对每个线程负责的元素计算方差
    for (int i = tid; i < cols; i += blockDim.x) {
        float diff = x_row[i] - mean;
        var_sum += diff * diff;
    }

    sdata[tid] = var_sum;
    __syncthreads();
    // 归约操作, 计算总和
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    // 计算出来的方差是所有线程计算的方差和除以列数
    float var = sdata[0] / static_cast<float>(cols);
    // 计算标准差的倒数, 用于归一化
    float rstd = rsqrtf(var + eps);

    // =========================
    // Step 3: normalize + affine
    // =========================
    // 针对每个线程负责的元素进行归一化和仿射变换
    for (int i = tid; i < cols; i += blockDim.x) {
        float v = x_row[i];
        float norm = (v - mean) * rstd;
        y_row[i] = norm * gamma[i] + beta[i];
    }
}

void launch_layernorm_naive(
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
    size_t shared_mem = block * sizeof(float); // 启动 kernel 时分配的共享内存大小 256 * sizeof(float) = 1024 bytes
    // 4096个 blcok 每个block 256个线程 
    // 256个线程负责1024个元素的计算 每个线程负责 1024/256 = 4 个元素

    // 只传入一个 block=256 那么视作 (256, 1, 1) 的线程块, grid=4096 视作 (4096, 1, 1) 的网格
    layernorm_naive_kernel<<<grid, block, shared_mem>>>(
        d_x,
        d_gamma,
        d_beta,
        d_y,
        rows,
        cols,
        eps
    );

    CHECK_CUDA(cudaGetLastError());
}

int main() {
    // 模拟 Transformer 里的 LayerNorm:
    // rows = batch_size * seq_len
    // cols = hidden_size
    int rows = 4096;
    int cols = 1024;

    float eps = 1e-5f;

    int warmup = 10;
    int repeat = 100;

    size_t numel = static_cast<size_t>(rows) * cols;
    size_t bytes_x = numel * sizeof(float);
    size_t bytes_y = numel * sizeof(float);
    size_t bytes_gamma = cols * sizeof(float);
    size_t bytes_beta = cols * sizeof(float);

    std::cout << "rows = " << rows << ", cols = " << cols << std::endl;
    std::cout << "warmup = " << warmup << ", repeat = " << repeat << std::endl;

    std::vector<float> h_x(numel);
    std::vector<float> h_y(numel);
    std::vector<float> h_y_ref(numel);

    std::vector<float> h_gamma(cols);
    std::vector<float> h_beta(cols);

    for (size_t i = 0; i < numel; i++) {
        h_x[i] = static_cast<float>((i % 127) - 63) * 0.01f;
    }
    
    for (int i = 0; i < cols; i++) {
        h_gamma[i] = 1.0f + 0.001f * static_cast<float>(i % 13);
        h_beta[i] = 0.01f * static_cast<float>(i % 7);
    }

    float *d_x = nullptr;
    float *d_y = nullptr;
    float *d_gamma = nullptr;
    float *d_beta = nullptr;

    CHECK_CUDA(cudaMalloc(&d_x, bytes_x));
    CHECK_CUDA(cudaMalloc(&d_y, bytes_y));
    CHECK_CUDA(cudaMalloc(&d_gamma, bytes_gamma));
    CHECK_CUDA(cudaMalloc(&d_beta, bytes_beta));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), bytes_x, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gamma, h_gamma.data(), bytes_gamma, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_beta, h_beta.data(), bytes_beta, cudaMemcpyHostToDevice));

    // =========================
    // Warmup
    // =========================
    for (int i = 0; i < warmup; i++) {
        launch_layernorm_naive(d_x, d_gamma, d_beta, d_y, rows, cols, eps);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // =========================
    // Benchmark
    // =========================
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < repeat; i++) {
        launch_layernorm_naive(d_x, d_gamma, d_beta, d_y, rows, cols, eps);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    float avg_ms = elapsed_ms / repeat;

    std::cout << "Total kernel time: " << elapsed_ms << " ms" << std::endl;
    std::cout << "Average kernel time: " << avg_ms << " ms" << std::endl;

    // 粗略估算 naive LayerNorm 每次访问的数据量:
    // 读 x 求 mean:       1 * x
    // 读 x 求 variance:   1 * x
    // 读 x normalize:     1 * x
    // 读 gamma + beta:    2 * numel
    // 写 y:               1 * y
    //
    // 总计约 6 * numel * sizeof(float)
    double bytes_per_iter = 6.0 * static_cast<double>(numel) * sizeof(float);
    double gb_per_iter = bytes_per_iter / 1e9;
    double bandwidth_gb_s = gb_per_iter / (avg_ms / 1000.0);

    std::cout << "Approx effective bandwidth: "
              << bandwidth_gb_s << " GB/s" << std::endl;

    // =========================
    // Correctness check
    // =========================
    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, bytes_y, cudaMemcpyDeviceToHost));

    layernorm_cpu(h_x, h_y_ref, h_gamma, h_beta, rows, cols, eps);

    float max_err = 0.0f;
    for (size_t i = 0; i < numel; i++) {
        max_err = std::max(max_err, std::abs(h_y[i] - h_y_ref[i]));
    }

    std::cout << "Max error: " << max_err << std::endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_gamma));
    CHECK_CUDA(cudaFree(d_beta));

    return 0;
}
