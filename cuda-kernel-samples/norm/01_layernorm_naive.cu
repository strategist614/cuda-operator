#include "norm_test_utils.cuh"

__global__ void layernorm_naive(const float* __restrict__ x,
                                const float* __restrict__ gamma,
                                const float* __restrict__ beta,
                                float* __restrict__ y,
                                int hidden_size,
                                float epsilon) {
    extern __shared__ float shared[];
    float* shared_sum = shared;
    float* shared_square_sum = shared + blockDim.x;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int row_offset = row * hidden_size;

    float local_sum = 0.0f;
    float local_square_sum = 0.0f;
    for (int col = tid; col < hidden_size; col += blockDim.x) {
        const float value = x[row_offset + col];
        local_sum += value;
        local_square_sum = fmaf(value, value, local_square_sum);
    }

    shared_sum[tid] = local_sum;
    shared_square_sum[tid] = local_square_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_square_sum[tid] += shared_square_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float mean = shared_sum[0] / hidden_size;
        const float variance =
            fmaxf(shared_square_sum[0] / hidden_size - mean * mean, 0.0f);
        shared_sum[0] = mean;
        shared_square_sum[0] = rsqrtf(variance + epsilon);
    }
    __syncthreads();

    const float mean = shared_sum[0];
    const float inverse_std = shared_square_sum[0];
    for (int col = tid; col < hidden_size; col += blockDim.x) {
        const float normalized = (x[row_offset + col] - mean) * inverse_std;
        y[row_offset + col] = normalized * gamma[col] + beta[col];
    }
}

int main() {
    const std::size_t shared_bytes =
        2 * norm_test::THREADS * sizeof(float);

    auto launch = [shared_bytes](const float* x,
                                 const float* gamma,
                                 const float* beta,
                                 float* y) {
        layernorm_naive<<<norm_test::ROWS, norm_test::THREADS, shared_bytes>>>(
            x, gamma, beta, y, norm_test::HIDDEN, norm_test::EPSILON);
    };

    return norm_test::run("layernorm_naive_fp32", 2,
                          norm_test::layernorm_reference, launch);
}
