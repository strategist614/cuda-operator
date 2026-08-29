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

__device__ __forceinline__
void select_pair(
    float& value,
    int& index,
    float other_value,
    int other_index,
    bool keep_better
) {
    if (keep_better) {
        if (
            better_pair(
                other_value,
                other_index,
                value,
                index
            )
        ) {
            value = other_value;
            index = other_index;
        }
    } else {
        if (
            better_pair(
                value,
                index,
                other_value,
                other_index
            )
        ) {
            value = other_value;
            index = other_index;
        }
    }
}

/*
 * Full 32-lane bitonic sort, descending by:
 *   value desc, index asc.
 *
 * Each lane contributes exactly one pair.
 */
__device__ __forceinline__
void warp_sort32_desc(
    float& value,
    int& index,
    int lane
) {
#pragma unroll
    for (int size = 2; size <= 32; size <<= 1) {
#pragma unroll
        for (
            int stride = size >> 1;
            stride > 0;
            stride >>= 1
        ) {
            const float other_value =
                __shfl_xor_sync(
                    FULL_MASK,
                    value,
                    stride
                );

            const int other_index =
                __shfl_xor_sync(
                    FULL_MASK,
                    index,
                    stride
                );

            const bool lower_lane =
                (lane & stride) == 0;

            /*
             * At the final size=32 stage the whole warp is sorted
             * descending. Earlier stages create alternating bitonic runs.
             */
            const bool descending_run =
                (lane & size) == 0;

            const bool keep_better =
                lower_lane == descending_run;

            select_pair(
                value,
                index,
                other_value,
                other_index,
                keep_better
            );
        }
    }
}

/*
 * Merge two already-sorted Top-16 lists:
 *
 * lanes 0..15 : list A descending
 * lanes16..31 : reversed list B, therefore ascending
 *
 * The combined sequence is bitonic. Five compare-exchange stages are
 * enough to obtain a fully sorted descending 32-element sequence.
 */
__device__ __forceinline__
void warp_bitonic_merge32_desc(
    float& value,
    int& index,
    int lane
) {
#pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
        const float other_value =
            __shfl_xor_sync(
                FULL_MASK,
                value,
                stride
            );

        const int other_index =
            __shfl_xor_sync(
                FULL_MASK,
                index,
                stride
            );

        const bool keep_better =
            (lane & stride) == 0;

        select_pair(
            value,
            index,
            other_value,
            other_index,
            keep_better
        );
    }
}

/*
 * Compute one warp maximum candidate.
 * Result is meaningful in lane 0.
 */
__device__ __forceinline__
void warp_reduce_best(
    float& value,
    int& index
) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_value =
            __shfl_down_sync(
                FULL_MASK,
                value,
                offset
            );

        const int other_index =
            __shfl_down_sync(
                FULL_MASK,
                index,
                offset
            );

        if (
            better_pair(
                other_value,
                other_index,
                value,
                index
            )
        ) {
            value = other_value;
            index = other_index;
        }
    }
}

__global__
void topk_batch_v3_kernel(
    const float* __restrict__ input,
    float* __restrict__ output_values,
    int* __restrict__ output_indices,
    int batch,
    int n,
    int k
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    if (row >= batch) {
        return;
    }

    const float* row_ptr =
        input
        +
        static_cast<size_t>(row) * n;

    /*
     * V3 key idea:
     *
     * One warp owns ONE Top-16 list.
     * The list is distributed across lanes 0..15.
     *
     * No per-thread float[16] / int[16] arrays.
     */
    float warp_top_value =
        -FLT_MAX;

    int warp_top_index =
        -1;

    /*
     * Warp i consumes disjoint 32-element batches:
     *
     * warp0: [0..31], [256..287], ...
     * warp1: [32..63], [288..319], ...
     * ...
     *
     * Across the block all input elements are covered exactly once.
     */
    for (
        int base = warp_id * WARP_SIZE;
        base < n;
        base += NUM_WARPS * WARP_SIZE
    ) {
        const int col =
            base + lane;

        float batch_value =
            -FLT_MAX;

        int batch_index =
            -1;

        if (col < n) {
            batch_value =
                row_ptr[col];

            batch_index =
                col;
        }

        /*
         * Cheap threshold test first.
         *
         * Only 5 shuffle stages are needed to know whether the best
         * element in this batch can beat the current warp Top-16 threshold.
         */
        float batch_best_value =
            batch_value;

        int batch_best_index =
            batch_index;

        warp_reduce_best(
            batch_best_value,
            batch_best_index
        );

        batch_best_value =
            __shfl_sync(
                FULL_MASK,
                batch_best_value,
                0
            );

        batch_best_index =
            __shfl_sync(
                FULL_MASK,
                batch_best_index,
                0
            );

        const float threshold_value =
            __shfl_sync(
                FULL_MASK,
                warp_top_value,
                MAX_K - 1
            );

        const int threshold_index =
            __shfl_sync(
                FULL_MASK,
                warp_top_index,
                MAX_K - 1
            );

        const bool should_merge =
            better_pair(
                batch_best_value,
                batch_best_index,
                threshold_value,
                threshold_index
            );

        /*
         * Uniform warp branch.
         * Once threshold becomes strong, most random batches are rejected
         * here and avoid the expensive sort+merge path.
         */
        if (!should_merge) {
            continue;
        }

        /*
         * Sort the 32 new values. Only the best 16 can possibly affect
         * the warp's global Top-16.
         */
        warp_sort32_desc(
            batch_value,
            batch_index,
            lane
        );

        /*
         * Build a bitonic 32-element sequence:
         *
         * lanes 0..15  = current Top-16 descending
         * lanes16..31  = new batch Top-16 reversed (ascending)
         */
        /*
         * IMPORTANT:
         * Every lane named in FULL_MASK must execute the shuffle.
         * Do NOT put __shfl_sync(FULL_MASK, ...) only in lanes 16..31.
         *
         * Compute the reversed batch value in all 32 lanes first,
         * then choose which half participates in the bitonic sequence.
         */
        const int src_lane =
            31 - lane;

        const float reversed_batch_value =
            __shfl_sync(
                FULL_MASK,
                batch_value,
                src_lane
            );

        const int reversed_batch_index =
            __shfl_sync(
                FULL_MASK,
                batch_index,
                src_lane
            );

        float merge_value;
        int merge_index;

        if (lane < MAX_K) {
            merge_value =
                warp_top_value;

            merge_index =
                warp_top_index;
        } else {
            merge_value =
                reversed_batch_value;

            merge_index =
                reversed_batch_index;
        }

        warp_bitonic_merge32_desc(
            merge_value,
            merge_index,
            lane
        );

        if (lane < MAX_K) {
            warp_top_value =
                merge_value;

            warp_top_index =
                merge_index;
        }
    }

    /*
     * Each warp now owns an exact Top-16 for its disjoint subset.
     * Hand off 8 sorted lists to shared memory.
     */
    __shared__ float warp_values[NUM_WARPS][MAX_K];
    __shared__ int warp_indices[NUM_WARPS][MAX_K];

    if (lane < MAX_K) {
        warp_values[warp_id][lane] =
            warp_top_value;

        warp_indices[warp_id][lane] =
            warp_top_index;
    }

    __syncthreads();

    /*
     * Warp 0 merges the eight warp Top-16 lists.
     *
     * This is only 8 bitonic merges and uses no additional block barrier.
     */
    if (warp_id == 0) {
        float block_top_value =
            -FLT_MAX;

        int block_top_index =
            -1;

#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            float merge_value;
            int merge_index;

            if (lane < MAX_K) {
                merge_value =
                    block_top_value;

                merge_index =
                    block_top_index;
            } else {
                const int list_pos =
                    31 - lane;

                merge_value =
                    warp_values[w][list_pos];

                merge_index =
                    warp_indices[w][list_pos];
            }

            warp_bitonic_merge32_desc(
                merge_value,
                merge_index,
                lane
            );

            if (lane < MAX_K) {
                block_top_value =
                    merge_value;

                block_top_index =
                    merge_index;
            }
        }

        if (lane < k) {
            output_values[
                static_cast<size_t>(row) * k + lane
            ] =
                block_top_value;

            output_indices[
                static_cast<size_t>(row) * k + lane
            ] =
                block_top_index;
        }
    }
}

} // namespace

bool topk_batch_v3_supported(
    int k
) {
    return k >= 1 && k <= MAX_K;
}

void launch_topk_batch_v3(
    const float* input,
    float* output_values,
    int* output_indices,
    int batch,
    int n,
    int k,
    cudaStream_t stream
) {
    if (!topk_batch_v3_supported(k)) {
        std::cerr
            << "topk_batch_v3 supports 1 <= K <= "
            << MAX_K
            << ", got K="
            << k
            << "\n";

        std::exit(EXIT_FAILURE);
    }

    dim3 grid(batch);
    dim3 block(THREADS);

    topk_batch_v3_kernel
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
