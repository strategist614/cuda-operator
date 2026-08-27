#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>


namespace {

constexpr int ELEMENTWISE_THREADS = 256;
constexpr int SOFTMAX_THREADS = 256;
constexpr int WARP_SIZE = 32;
constexpr int WMMA_TILE = 16;


__device__ __forceinline__ float warp_reduce_sum(float value)
{
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}


__device__ __forceinline__ float warp_reduce_max(float value)
{
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        value = fmaxf(
            value,
            __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}


__device__ __forceinline__ float block_reduce_sum(
    float value,
    float* warp_values)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    const int warp_count = blockDim.x / WARP_SIZE;

    value = warp_reduce_sum(value);
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        value = lane < warp_count ? warp_values[lane] : 0.0f;
        value = warp_reduce_sum(value);
        if (lane == 0) {
            warp_values[0] = value;
        }
    }
    __syncthreads();
    return warp_values[0];
}


__device__ __forceinline__ float block_reduce_max(
    float value,
    float* warp_values)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    const int warp_count = blockDim.x / WARP_SIZE;

    value = warp_reduce_max(value);
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        value = lane < warp_count ? warp_values[lane] : -INFINITY;
        value = warp_reduce_max(value);
        if (lane == 0) {
            warp_values[0] = value;
        }
    }
    __syncthreads();
    return warp_values[0];
}


template <typename scalar_t>
__global__ void attention_scores_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    float* __restrict__ scores,
    int sequence,
    int head_dim,
    int64_t score_count)
{
    const int64_t index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= score_count) {
        return;
    }

    const int key_index = index % sequence;
    const int64_t query_row = index / sequence;
    const int query_index = query_row % sequence;
    const int64_t batch_head = query_row / sequence;
    const int64_t query_offset =
        (batch_head * sequence + query_index) * head_dim;
    const int64_t key_offset =
        (batch_head * sequence + key_index) * head_dim;

    float dot = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
        dot = fmaf(
            static_cast<float>(query[query_offset + dim]),
            static_cast<float>(key[key_offset + dim]),
            dot);
    }
    scores[index] = dot * rsqrtf(static_cast<float>(head_dim));
}


__global__ void softmax_inplace_kernel(
    float* __restrict__ scores,
    int sequence)
{
    __shared__ float warp_values[32];

    const int row = blockIdx.x;
    const int row_offset = row * sequence;
    float local_max = -INFINITY;
    for (int col = threadIdx.x; col < sequence; col += blockDim.x) {
        local_max = fmaxf(local_max, scores[row_offset + col]);
    }
    const float row_max = block_reduce_max(local_max, warp_values);

    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < sequence; col += blockDim.x) {
        local_sum += __expf(scores[row_offset + col] - row_max);
    }
    const float row_sum = block_reduce_sum(local_sum, warp_values);

    for (int col = threadIdx.x; col < sequence; col += blockDim.x) {
        scores[row_offset + col] =
            __expf(scores[row_offset + col] - row_max) / row_sum;
    }
}


template <typename scalar_t>
__global__ void attention_output_kernel(
    const float* __restrict__ probabilities,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ output,
    int sequence,
    int head_dim,
    int64_t output_count)
{
    const int64_t index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= output_count) {
        return;
    }

    const int dim = index % head_dim;
    const int64_t query_row = index / head_dim;
    const int query_index = query_row % sequence;
    const int64_t batch_head = query_row / sequence;
    const int64_t probability_offset =
        (batch_head * sequence + query_index) * sequence;
    const int64_t value_offset = batch_head * sequence * head_dim + dim;

    float accumulator = 0.0f;
    for (int key_index = 0; key_index < sequence; ++key_index) {
        accumulator = fmaf(
            probabilities[probability_offset + key_index],
            static_cast<float>(
                value[value_offset + static_cast<int64_t>(key_index) * head_dim]),
            accumulator);
    }
    output[index] = static_cast<scalar_t>(accumulator);
}


__global__ void flash_attention_wmma_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key,
    const half* __restrict__ value,
    half* __restrict__ output,
    int sequence,
    int head_dim)
{
    namespace wmma = nvcuda::wmma;
    extern __shared__ __align__(16) unsigned char shared_storage[];

    float* scores = reinterpret_cast<float*>(shared_storage);
    float* product = scores + WMMA_TILE * WMMA_TILE;
    float* output_accumulator = product + WMMA_TILE * WMMA_TILE;
    float* row_scale = output_accumulator + WMMA_TILE * head_dim;
    half* probabilities = reinterpret_cast<half*>(row_scale + WMMA_TILE);

    const int lane = threadIdx.x;
    const int query_tile_count = sequence / WMMA_TILE;
    const int query_tile = blockIdx.x % query_tile_count;
    const int batch_head = blockIdx.x / query_tile_count;
    const int query_start = query_tile * WMMA_TILE;
    const int64_t batch_head_offset =
        static_cast<int64_t>(batch_head) * sequence * head_dim;
    const half* query_tile_pointer =
        query + batch_head_offset + query_start * head_dim;
    const float score_scale = rsqrtf(static_cast<float>(head_dim));

    for (int index = lane; index < WMMA_TILE * head_dim; index += WARP_SIZE) {
        output_accumulator[index] = 0.0f;
    }
    __syncwarp();

    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (int key_start = 0; key_start < sequence; key_start += WMMA_TILE) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> score_fragment;
        wmma::fill_fragment(score_fragment, 0.0f);

        for (int dim = 0; dim < head_dim; dim += WMMA_TILE) {
            wmma::fragment<
                wmma::matrix_a,
                16,
                16,
                16,
                half,
                wmma::row_major>
                query_fragment;
            wmma::fragment<
                wmma::matrix_b,
                16,
                16,
                16,
                half,
                wmma::col_major>
                key_fragment;

            wmma::load_matrix_sync(
                query_fragment,
                query_tile_pointer + dim,
                head_dim);
            wmma::load_matrix_sync(
                key_fragment,
                key + batch_head_offset + key_start * head_dim + dim,
                head_dim);
            wmma::mma_sync(
                score_fragment,
                query_fragment,
                key_fragment,
                score_fragment);
        }
        wmma::store_matrix_sync(
            scores,
            score_fragment,
            WMMA_TILE,
            wmma::mem_row_major);
        __syncwarp();

        if (lane < WMMA_TILE) {
            const int row_offset = lane * WMMA_TILE;
            float tile_max = -INFINITY;
#pragma unroll
            for (int col = 0; col < WMMA_TILE; ++col) {
                const float scaled_score = scores[row_offset + col] * score_scale;
                scores[row_offset + col] = scaled_score;
                tile_max = fmaxf(tile_max, scaled_score);
            }

            const float new_max = fmaxf(running_max, tile_max);
            const float alpha = __expf(running_max - new_max);
            float tile_sum = 0.0f;
#pragma unroll
            for (int col = 0; col < WMMA_TILE; ++col) {
                const float probability =
                    __expf(scores[row_offset + col] - new_max);
                probabilities[row_offset + col] = __float2half_rn(probability);
                tile_sum += probability;
            }

            running_sum = alpha * running_sum + tile_sum;
            running_max = new_max;
            row_scale[lane] = alpha;
        }
        __syncwarp();

        for (int dim = 0; dim < head_dim; dim += WMMA_TILE) {
            wmma::fragment<
                wmma::matrix_a,
                16,
                16,
                16,
                half,
                wmma::row_major>
                probability_fragment;
            wmma::fragment<
                wmma::matrix_b,
                16,
                16,
                16,
                half,
                wmma::row_major>
                value_fragment;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float>
                product_fragment;

            wmma::load_matrix_sync(
                probability_fragment,
                probabilities,
                WMMA_TILE);
            wmma::load_matrix_sync(
                value_fragment,
                value + batch_head_offset + key_start * head_dim + dim,
                head_dim);
            wmma::fill_fragment(product_fragment, 0.0f);
            wmma::mma_sync(
                product_fragment,
                probability_fragment,
                value_fragment,
                product_fragment);
            wmma::store_matrix_sync(
                product,
                product_fragment,
                WMMA_TILE,
                wmma::mem_row_major);
            __syncwarp();

            for (int index = lane;
                 index < WMMA_TILE * WMMA_TILE;
                 index += WARP_SIZE) {
                const int row = index / WMMA_TILE;
                const int col = index % WMMA_TILE;
                const int output_index = row * head_dim + dim + col;
                output_accumulator[output_index] =
                    row_scale[row] * output_accumulator[output_index] +
                    product[index];
            }
            __syncwarp();
        }
    }

    if (lane < WMMA_TILE) {
        row_scale[lane] = running_sum;
    }
    __syncwarp();

    for (int index = lane; index < WMMA_TILE * head_dim; index += WARP_SIZE) {
        const int row = index / head_dim;
        const int dim = index % head_dim;
        output[
            batch_head_offset +
            static_cast<int64_t>(query_start + row) * head_dim + dim] =
            __float2half_rn(output_accumulator[index] / row_scale[row]);
    }
}


template <typename scalar_t>
__global__ void flash_attention_online_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ output,
    int sequence,
    int head_dim)
{
    __shared__ float warp_values[32];
    __shared__ float online_state[3];

    const int query_row = blockIdx.x;
    const int query_index = query_row % sequence;
    const int batch_head = query_row / sequence;
    const int dim = threadIdx.x;
    const bool active_dim = dim < head_dim;
    const int64_t query_offset =
        (static_cast<int64_t>(batch_head) * sequence + query_index) * head_dim;

    float output_accumulator = 0.0f;
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    for (int key_index = 0; key_index < sequence; ++key_index) {
        const int64_t key_offset =
            (static_cast<int64_t>(batch_head) * sequence + key_index) * head_dim;
        float partial_dot = 0.0f;
        if (active_dim) {
            partial_dot =
                static_cast<float>(query[query_offset + dim]) *
                static_cast<float>(key[key_offset + dim]);
        }
        const float score =
            block_reduce_sum(partial_dot, warp_values) * scale;

        if (threadIdx.x == 0) {
            const float new_max = fmaxf(running_max, score);
            const float alpha = __expf(running_max - new_max);
            const float beta = __expf(score - new_max);
            running_sum = running_sum * alpha + beta;
            running_max = new_max;
            online_state[0] = alpha;
            online_state[1] = beta;
            online_state[2] = running_sum;
        }
        __syncthreads();

        if (active_dim) {
            output_accumulator =
                output_accumulator * online_state[0] +
                online_state[1] * static_cast<float>(value[key_offset + dim]);
        }
        __syncthreads();
    }

    if (active_dim) {
        output[query_offset + dim] =
            static_cast<scalar_t>(output_accumulator / online_state[2]);
    }
}


int next_power_of_two_threads(int value)
{
    int threads = WARP_SIZE;
    while (threads < value) {
        threads <<= 1;
    }
    return threads;
}

}  // namespace


at::Tensor basic_attention_cuda(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value)
{
    c10::cuda::CUDAGuard device_guard(query.device());
    const int64_t batch = query.size(0);
    const int64_t heads = query.size(1);
    const int sequence = static_cast<int>(query.size(2));
    const int head_dim = static_cast<int>(query.size(3));
    const int64_t rows = batch * heads * sequence;
    const int64_t score_count = rows * sequence;
    const int64_t output_count = rows * head_dim;

    auto scores = at::empty(
        {batch, heads, sequence, sequence},
        query.options().dtype(at::kFloat));
    auto output = at::empty_like(query);
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(query.get_device()).stream();

    const int score_blocks = static_cast<int>(
        (score_count + ELEMENTWISE_THREADS - 1) / ELEMENTWISE_THREADS);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        query.scalar_type(), "attention_scores_cuda", [&] {
            attention_scores_kernel<scalar_t>
                <<<score_blocks, ELEMENTWISE_THREADS, 0, stream>>>(
                    query.data_ptr<scalar_t>(),
                    key.data_ptr<scalar_t>(),
                    scores.data_ptr<float>(),
                    sequence,
                    head_dim,
                    score_count);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    softmax_inplace_kernel<<<rows, SOFTMAX_THREADS, 0, stream>>>(
        scores.data_ptr<float>(), sequence);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int output_blocks = static_cast<int>(
        (output_count + ELEMENTWISE_THREADS - 1) / ELEMENTWISE_THREADS);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        query.scalar_type(), "attention_output_cuda", [&] {
            attention_output_kernel<scalar_t>
                <<<output_blocks, ELEMENTWISE_THREADS, 0, stream>>>(
                    scores.data_ptr<float>(),
                    value.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    sequence,
                    head_dim,
                    output_count);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


at::Tensor flash_attention_cuda(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value)
{
    c10::cuda::CUDAGuard device_guard(query.device());
    const int64_t batch = query.size(0);
    const int64_t heads = query.size(1);
    const int sequence = static_cast<int>(query.size(2));
    const int head_dim = static_cast<int>(query.size(3));
    const int64_t rows = batch * heads * sequence;
    auto output = at::empty_like(query);
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(query.get_device()).stream();
    const cudaDeviceProp* device_properties =
        at::cuda::getDeviceProperties(query.get_device());

    const bool use_wmma =
        query.scalar_type() == at::kHalf &&
        sequence % WMMA_TILE == 0 &&
        head_dim % WMMA_TILE == 0 &&
        device_properties->major >= 7;
    if (use_wmma) {
        const int query_tiles = sequence / WMMA_TILE;
        const int blocks = static_cast<int>(batch * heads * query_tiles);
        const size_t shared_bytes =
            2 * WMMA_TILE * WMMA_TILE * sizeof(float) +
            WMMA_TILE * head_dim * sizeof(float) +
            WMMA_TILE * sizeof(float) +
            WMMA_TILE * WMMA_TILE * sizeof(half);
        flash_attention_wmma_kernel<<<
            blocks,
            WARP_SIZE,
            shared_bytes,
            stream>>>(
                reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(key.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(value.data_ptr<at::Half>()),
                reinterpret_cast<half*>(output.data_ptr<at::Half>()),
                sequence,
                head_dim);
    } else {
        const int threads = next_power_of_two_threads(head_dim);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            query.scalar_type(), "flash_attention_online_cuda", [&] {
                flash_attention_online_kernel<scalar_t>
                    <<<rows, threads, 0, stream>>>(
                        query.data_ptr<scalar_t>(),
                        key.data_ptr<scalar_t>(),
                        value.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        sequence,
                        head_dim);
            });
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
