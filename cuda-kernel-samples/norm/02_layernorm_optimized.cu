#include "norm_test_utils.cuh"

__device__ __forceinline__ void warp_reduce_pair(float& sum,
                                                  float& square_sum) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        square_sum += __shfl_down_sync(0xffffffff, square_sum, offset);
    }
}

__global__ void layernorm_float4(const float* __restrict__ x,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ y,
                                 int hidden_size,
                                 float epsilon) {
    __shared__ float warp_sum[32];
    __shared__ float warp_square_sum[32];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int warp_count = blockDim.x >> 5;
    const int hidden4 = hidden_size >> 2;
    const int row4_offset = blockIdx.x * hidden4;

    const float4* x4 = reinterpret_cast<const float4*>(x);
    float local_sum = 0.0f;
    float local_square_sum = 0.0f;
    for (int col4 = tid; col4 < hidden4; col4 += blockDim.x) {
        const float4 value = x4[row4_offset + col4];
        local_sum += value.x + value.y + value.z + value.w;
        local_square_sum = fmaf(value.x, value.x, local_square_sum);
        local_square_sum = fmaf(value.y, value.y, local_square_sum);
        local_square_sum = fmaf(value.z, value.z, local_square_sum);
        local_square_sum = fmaf(value.w, value.w, local_square_sum);
    }

    warp_reduce_pair(local_sum, local_square_sum);
    if (lane == 0) {
        warp_sum[warp_id] = local_sum;
        warp_square_sum[warp_id] = local_square_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_sum = lane < warp_count ? warp_sum[lane] : 0.0f;
        local_square_sum =
            lane < warp_count ? warp_square_sum[lane] : 0.0f;
        warp_reduce_pair(local_sum, local_square_sum);

        if (lane == 0) {
            const float mean = local_sum / hidden_size;
            const float variance =
                fmaxf(local_square_sum / hidden_size - mean * mean, 0.0f);
            warp_sum[0] = mean;
            warp_square_sum[0] = rsqrtf(variance + epsilon);
        }
    }
    __syncthreads();

    const float mean = warp_sum[0];
    const float inverse_std = warp_square_sum[0];
    const float4* gamma4 = reinterpret_cast<const float4*>(gamma);
    const float4* beta4 = reinterpret_cast<const float4*>(beta);
    float4* y4 = reinterpret_cast<float4*>(y);

    for (int col4 = tid; col4 < hidden4; col4 += blockDim.x) {
        const float4 value = x4[row4_offset + col4];
        const float4 scale = gamma4[col4];
        const float4 bias = beta4[col4];
        float4 output;
        output.x = (value.x - mean) * inverse_std * scale.x + bias.x;
        output.y = (value.y - mean) * inverse_std * scale.y + bias.y;
        output.z = (value.z - mean) * inverse_std * scale.z + bias.z;
        output.w = (value.w - mean) * inverse_std * scale.w + bias.w;
        y4[row4_offset + col4] = output;
    }
}

int main() {
    auto launch = [](const float* x,
                     const float* gamma,
                     const float* beta,
                     float* y) {
        layernorm_float4<<<norm_test::ROWS, norm_test::THREADS>>>(
            x, gamma, beta, y, norm_test::HIDDEN, norm_test::EPSILON);
    };

    return norm_test::run("layernorm_float4_warp_fp32", 2,
                          norm_test::layernorm_reference, launch);
}
