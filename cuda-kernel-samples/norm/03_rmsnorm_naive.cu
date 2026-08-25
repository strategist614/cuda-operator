#include "norm_test_utils.cuh"

__global__ void rmsnorm_naive(const float* __restrict__ x,
                              const float* __restrict__ gamma,
                              float* __restrict__ y,
                              int hidden_size,
                              float epsilon) {
    extern __shared__ float shared_square_sum[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int row_offset = row * hidden_size;

    float local_square_sum = 0.0f;
    for (int col = tid; col < hidden_size; col += blockDim.x) {
        const float value = x[row_offset + col];
        local_square_sum = fmaf(value, value, local_square_sum);
    }

    shared_square_sum[tid] = local_square_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_square_sum[tid] += shared_square_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        shared_square_sum[0] =
            rsqrtf(shared_square_sum[0] / hidden_size + epsilon);
    }
    __syncthreads();

    const float inverse_rms = shared_square_sum[0];
    for (int col = tid; col < hidden_size; col += blockDim.x) {
        y[row_offset + col] = x[row_offset + col] * inverse_rms * gamma[col];
    }
}

int main() {
    const std::size_t shared_bytes = norm_test::THREADS * sizeof(float);

    auto launch = [shared_bytes](const float* x,
                                 const float* gamma,
                                 const float*,
                                 float* y) {
        rmsnorm_naive<<<norm_test::ROWS, norm_test::THREADS, shared_bytes>>>(
            x, gamma, y, norm_test::HIDDEN, norm_test::EPSILON);
    };

    return norm_test::run("rmsnorm_naive_fp32", 2,
                          norm_test::rmsnorm_reference, launch);
}
