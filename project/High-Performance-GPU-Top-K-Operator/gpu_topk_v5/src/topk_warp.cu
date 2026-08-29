#include "topk.h"
#include "common.h"

#include <cuda_runtime.h>
#include <float.h>
#include <cstdlib>
#include <iostream>

namespace {

constexpr int THREADS = 256;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = THREADS / WARP_SIZE;
constexpr int MAX_K = 16;
constexpr unsigned FULL_MASK = 0xffffffffu;

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

/*
 * Reduce one candidate per lane to the best candidate in the warp.
 *
 * The tuple is:
 *   (value, original_index, owner_lane)
 *
 * After the reduction, lane 0 owns the winner.
 */
__device__ __forceinline__
void warp_reduce_best(
    float& value,
    int& index,
    int& owner
) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float rhs_value =
            __shfl_down_sync(
                FULL_MASK,
                value,
                offset
            );

        const int rhs_index =
            __shfl_down_sync(
                FULL_MASK,
                index,
                offset
            );

        const int rhs_owner =
            __shfl_down_sync(
                FULL_MASK,
                owner,
                offset
            );

        if (
            better_pair(
                rhs_value,
                rhs_index,
                value,
                index
            )
        ) {
            value = rhs_value;
            index = rhs_index;
            owner = rhs_owner;
        }
    }
}

__global__
void topk_warp_v2_kernel(
    const float* __restrict__ input,
    float* __restrict__ output_values,
    int* __restrict__ output_indices,
    int batch,
    int n,
    int k
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp_id = tid / WARP_SIZE;

    if (row >= batch) {
        return;
    }

    const float* row_ptr =
        input
        +
        static_cast<size_t>(row) * n;

    /*
     * Same strong part as V1:
     * every thread scans a coalesced strided slice and keeps an exact
     * sorted local Top-K candidate list.
     */
    float local_values[MAX_K];
    int local_indices[MAX_K];

#pragma unroll
    for (int i = 0; i < MAX_K; ++i) {
        local_values[i] = -FLT_MAX;
        local_indices[i] = -1;
    }

    for (
        int col = tid;
        col < n;
        col += THREADS
    ) {
        const float value =
            row_ptr[col];

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
     * Stage 1: each warp merges 32 sorted local candidate lists.
     *
     * V1 did a 256-thread block reduction for every output rank.
     * V2 uses register-to-register shuffle communication inside a warp.
     */
    __shared__ float warp_values[NUM_WARPS][MAX_K];
    __shared__ int warp_indices[NUM_WARPS][MAX_K];

    int local_cursor = 0;

    for (int rank = 0; rank < k; ++rank) {
        float candidate_value =
            local_cursor < k
            ? local_values[local_cursor]
            : -FLT_MAX;

        int candidate_index =
            local_cursor < k
            ? local_indices[local_cursor]
            : -1;

        int candidate_owner =
            lane;

        warp_reduce_best(
            candidate_value,
            candidate_index,
            candidate_owner
        );

        const int winner_lane =
            __shfl_sync(
                FULL_MASK,
                candidate_owner,
                0
            );

        if (lane == 0) {
            warp_values[warp_id][rank] =
                candidate_value;

            warp_indices[warp_id][rank] =
                candidate_index;
        }

        if (lane == winner_lane) {
            ++local_cursor;
        }
    }

    /*
     * All 8 warp-level Top-K lists must be visible before the final merge.
     * This is the only block-wide barrier in the merge path.
     */
    __syncthreads();

    /*
     * Stage 2: warp 0 merges only NUM_WARPS (=8) sorted lists.
     *
     * lanes 0..7 each represent one warp result list.
     * lanes 8..31 participate in the shuffle network with invalid values.
     *
     * No more __syncthreads() are needed.
     */
    if (warp_id == 0) {
        int warp_cursor = 0;

        for (int rank = 0; rank < k; ++rank) {
            float candidate_value =
                -FLT_MAX;

            int candidate_index =
                -1;

            int candidate_owner =
                lane;

            if (
                lane < NUM_WARPS
                &&
                warp_cursor < k
            ) {
                candidate_value =
                    warp_values[lane][warp_cursor];

                candidate_index =
                    warp_indices[lane][warp_cursor];
            }

            warp_reduce_best(
                candidate_value,
                candidate_index,
                candidate_owner
            );

            const int winner_lane =
                __shfl_sync(
                    FULL_MASK,
                    candidate_owner,
                    0
                );

            if (lane == 0) {
                output_values[
                    static_cast<size_t>(row) * k + rank
                ] =
                    candidate_value;

                output_indices[
                    static_cast<size_t>(row) * k + rank
                ] =
                    candidate_index;
            }

            if (
                lane < NUM_WARPS
                &&
                lane == winner_lane
            ) {
                ++warp_cursor;
            }
        }
    }
}

} // namespace

bool topk_warp_v2_supported(
    int k
) {
    return k >= 1 && k <= MAX_K;
}

void launch_topk_warp_v2(
    const float* input,
    float* output_values,
    int* output_indices,
    int batch,
    int n,
    int k,
    cudaStream_t stream
) {
    if (!topk_warp_v2_supported(k)) {
        std::cerr
            << "topk_warp_v2 supports 1 <= K <= "
            << MAX_K
            << ", got K="
            << k
            << "\n";

        std::exit(EXIT_FAILURE);
    }

    dim3 grid(batch);
    dim3 block(THREADS);

    topk_warp_v2_kernel
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
