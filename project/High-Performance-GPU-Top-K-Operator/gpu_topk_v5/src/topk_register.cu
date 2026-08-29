#include "topk.h"
#include "common.h"

#include <cuda_runtime.h>
#include <float.h>

namespace {

constexpr int THREADS = 256;
constexpr int MAX_K = 16;

__device__ __forceinline__
bool better_pair(
    float lhs_value,
    int lhs_index,
    float rhs_value,
    int rhs_index
) {
    if (lhs_value != rhs_value) {
        return lhs_value > rhs_value;
    }

    if (lhs_index < 0) {
        return false;
    }

    if (rhs_index < 0) {
        return true;
    }

    return lhs_index < rhs_index;
}

__global__
void topk_register_v1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output_values,
    int* __restrict__ output_indices,
    int batch,
    int n,
    int k
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= batch) {
        return;
    }

    const float* row_ptr =
        input + static_cast<size_t>(row) * n;

    /*
     * Each thread keeps its own sorted local Top-K in registers
     * (the compiler may spill some of this depending on K/resources).
     *
     * local_values[0] is best, local_values[k-1] is current threshold.
     */
    float local_values[MAX_K];
    int local_indices[MAX_K];

#pragma unroll
    for (int i = 0; i < MAX_K; ++i) {
        local_values[i] = -FLT_MAX;
        local_indices[i] = -1;
    }

    /*
     * Parallel strided scan.
     *
     * At a given loop iteration, neighboring lanes access neighboring
     * columns, giving naturally coalesced global loads.
     */
    for (
        int col = tid;
        col < n;
        col += THREADS
    ) {
        const float value = row_ptr[col];

        if (
            better_pair(
                value,
                col,
                local_values[k - 1],
                local_indices[k - 1]
            )
        ) {
            int pos = k - 1;

            while (
                pos > 0
                &&
                better_pair(
                    value,
                    col,
                    local_values[pos - 1],
                    local_indices[pos - 1]
                )
            ) {
                local_values[pos] =
                    local_values[pos - 1];

                local_indices[pos] =
                    local_indices[pos - 1];

                --pos;
            }

            local_values[pos] = value;
            local_indices[pos] = col;
        }
    }

    /*
     * Each thread now owns one sorted candidate list.
     *
     * We perform a k-way merge of 256 lists.
     * For each output rank every thread proposes the current head of
     * its local list, then the whole block reduces to the best proposal.
     *
     * This costs K block reductions rather than K rescans of N.
     */
    __shared__ float s_values[THREADS];
    __shared__ int s_indices[THREADS];
    __shared__ int s_owners[THREADS];
    __shared__ int s_cursors[THREADS];

    s_cursors[tid] = 0;
    __syncthreads();

    for (int rank = 0; rank < k; ++rank) {
        const int cursor =
            s_cursors[tid];

        if (cursor < k) {
            s_values[tid] =
                local_values[cursor];

            s_indices[tid] =
                local_indices[cursor];

            s_owners[tid] =
                tid;
        } else {
            s_values[tid] =
                -FLT_MAX;

            s_indices[tid] =
                -1;

            s_owners[tid] =
                tid;
        }

        __syncthreads();

        /*
         * Block reduction over (value, original index, owner thread).
         */
        for (
            int offset = THREADS / 2;
            offset > 0;
            offset >>= 1
        ) {
            if (tid < offset) {
                const int rhs =
                    tid + offset;

                if (
                    better_pair(
                        s_values[rhs],
                        s_indices[rhs],
                        s_values[tid],
                        s_indices[tid]
                    )
                ) {
                    s_values[tid] =
                        s_values[rhs];

                    s_indices[tid] =
                        s_indices[rhs];

                    s_owners[tid] =
                        s_owners[rhs];
                }
            }

            __syncthreads();
        }

        if (tid == 0) {
            const int winner_owner =
                s_owners[0];

            output_values[
                static_cast<size_t>(row) * k + rank
            ] =
                s_values[0];

            output_indices[
                static_cast<size_t>(row) * k + rank
            ] =
                s_indices[0];

            s_cursors[winner_owner] += 1;
        }

        __syncthreads();
    }
}

} // namespace

bool topk_register_v1_supported(
    int k
) {
    return k >= 1 && k <= MAX_K;
}

void launch_topk_register_v1(
    const float* input,
    float* output_values,
    int* output_indices,
    int batch,
    int n,
    int k,
    cudaStream_t stream
) {
    if (!topk_register_v1_supported(k)) {
        std::cerr
            << "topk_register_v1 supports 1 <= K <= "
            << MAX_K
            << ", got K="
            << k
            << "\n";

        std::exit(EXIT_FAILURE);
    }

    dim3 grid(batch);
    dim3 block(THREADS);

    topk_register_v1_kernel
    <<<grid, block, 0, stream>>>(
        input,
        output_values,
        output_indices,
        batch,
        n,
        k
    );

    CUDA_CHECK(cudaGetLastError());
}
