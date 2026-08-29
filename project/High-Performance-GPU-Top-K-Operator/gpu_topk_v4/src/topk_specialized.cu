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

/*
 * Full 32-lane descending sort for a newly loaded batch.
 * Same stable comparator as V3.
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

template<int K>
__device__ __forceinline__
constexpr unsigned active_mask_2k() {
    static_assert(
        K == 1 || K == 2 || K == 4 || K == 8 || K == 16,
        "V4 specialized K must be power-of-two <= 16."
    );

    if constexpr (K == 16) {
        return 0xffffffffu;
    } else {
        return (1u << (2 * K)) - 1u;
    }
}

/*
 * Merge two sorted Top-K lists into Top-K.
 *
 * Layout before merge:
 *
 * lanes [0, K):
 *   current Top-K descending
 *
 * lanes [K, 2K):
 *   new Top-K reversed, therefore ascending
 *
 * This is a bitonic sequence of length 2K.
 *
 * Unlike V3, which always pays five 32-lane merge stages,
 * V4 only pays log2(2K) stages:
 *
 * K=1  -> 1 stage
 * K=2  -> 2 stages
 * K=4  -> 3 stages
 * K=8  -> 4 stages
 * K=16 -> 5 stages
 */
template<int K>
__device__ __forceinline__
void warp_merge_2k_desc(
    float& value,
    int& index,
    int lane
) {
    constexpr unsigned MASK =
        active_mask_2k<K>();

#pragma unroll
    for (
        int stride = K;
        stride > 0;
        stride >>= 1
    ) {
        if (lane < 2 * K) {
            const float other_value =
                __shfl_xor_sync(
                    MASK,
                    value,
                    stride
                );

            const int other_index =
                __shfl_xor_sync(
                    MASK,
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
}

template<int K>
__global__
void topk_specialized_v4_kernel(
    const float* __restrict__ input,
    float* __restrict__ output_values,
    int* __restrict__ output_indices,
    int batch,
    int n
) {
    static_assert(
        K == 1 || K == 2 || K == 4 || K == 8 || K == 16,
        "Unsupported specialized K."
    );

    const int row =
        blockIdx.x;

    const int tid =
        threadIdx.x;

    const int lane =
        tid & 31;

    const int warp_id =
        tid >> 5;

    if (row >= batch) {
        return;
    }

    const float* row_ptr =
        input
        +
        static_cast<size_t>(row) * n;

    /*
     * V4's central difference from V3:
     *
     * V3 always stores Top-16 in lanes 0..15.
     *
     * V4 compile-time specializes the actual queue length:
     *
     * K=1  -> lane 0
     * K=2  -> lanes 0..1
     * K=4  -> lanes 0..3
     * K=8  -> lanes 0..7
     * K=16 -> lanes 0..15
     */
    float warp_top_value =
        -FLT_MAX;

    int warp_top_index =
        -1;

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
         * Cheap uniform threshold reject.
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
                K - 1
            );

        const int threshold_index =
            __shfl_sync(
                FULL_MASK,
                warp_top_index,
                K - 1
            );

        const bool should_merge =
            better_pair(
                batch_best_value,
                batch_best_index,
                threshold_value,
                threshold_index
            );

        if (!should_merge) {
            continue;
        }

        /*
         * Sort new 32-element batch once.
         * lanes 0..K-1 become exact Batch Top-K.
         */
        warp_sort32_desc(
            batch_value,
            batch_index,
            lane
        );

        constexpr unsigned MASK =
            active_mask_2k<K>();

        /*
         * All lanes in MASK execute the shuffle.
         * This avoids the undefined behavior that was fixed in V3.
         */
        float reversed_batch_value =
            -FLT_MAX;

        int reversed_batch_index =
            -1;

        if (lane < 2 * K) {
            const int src_lane =
                2 * K - 1 - lane;

            reversed_batch_value =
                __shfl_sync(
                    MASK,
                    batch_value,
                    src_lane
                );

            reversed_batch_index =
                __shfl_sync(
                    MASK,
                    batch_index,
                    src_lane
                );
        }

        float merge_value =
            -FLT_MAX;

        int merge_index =
            -1;

        if (lane < K) {
            merge_value =
                warp_top_value;

            merge_index =
                warp_top_index;

        } else if (lane < 2 * K) {
            merge_value =
                reversed_batch_value;

            merge_index =
                reversed_batch_index;
        }

        warp_merge_2k_desc<K>(
            merge_value,
            merge_index,
            lane
        );

        if (lane < K) {
            warp_top_value =
                merge_value;

            warp_top_index =
                merge_index;
        }
    }

    /*
     * Each warp owns exact Top-K for its disjoint subset.
     */
    __shared__ float
    warp_values[NUM_WARPS][16];

    __shared__ int
    warp_indices[NUM_WARPS][16];

    if (lane < K) {
        warp_values[warp_id][lane] =
            warp_top_value;

        warp_indices[warp_id][lane] =
            warp_top_index;
    }

    __syncthreads();

    /*
     * Final block merge:
     * only warp0 participates.
     *
     * Again use K-specialized 2K merge instead of V3's fixed Top-16.
     */
    if (warp_id == 0) {
        float block_top_value =
            -FLT_MAX;

        int block_top_index =
            -1;

        constexpr unsigned MASK =
            active_mask_2k<K>();

#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            float reversed_list_value =
                -FLT_MAX;

            int reversed_list_index =
                -1;

            if (lane < 2 * K) {
                const int src_lane =
                    2 * K - 1 - lane;

                /*
                 * Load the K-element warp list through lanes 0..K-1,
                 * then reverse it into lanes K..2K-1 using shuffle.
                 */
                float list_value =
                    -FLT_MAX;

                int list_index =
                    -1;

                if (lane < K) {
                    list_value =
                        warp_values[w][lane];

                    list_index =
                        warp_indices[w][lane];
                }

                reversed_list_value =
                    __shfl_sync(
                        MASK,
                        list_value,
                        src_lane
                    );

                reversed_list_index =
                    __shfl_sync(
                        MASK,
                        list_index,
                        src_lane
                    );
            }

            float merge_value =
                -FLT_MAX;

            int merge_index =
                -1;

            if (lane < K) {
                merge_value =
                    block_top_value;

                merge_index =
                    block_top_index;

            } else if (lane < 2 * K) {
                merge_value =
                    reversed_list_value;

                merge_index =
                    reversed_list_index;
            }

            warp_merge_2k_desc<K>(
                merge_value,
                merge_index,
                lane
            );

            if (lane < K) {
                block_top_value =
                    merge_value;

                block_top_index =
                    merge_index;
            }
        }

        if (lane < K) {
            output_values[
                static_cast<size_t>(row) * K + lane
            ] =
                block_top_value;

            output_indices[
                static_cast<size_t>(row) * K + lane
            ] =
                block_top_index;
        }
    }
}

template<int K>
void launch_specialized(
    const float* input,
    float* output_values,
    int* output_indices,
    int batch,
    int n,
    cudaStream_t stream
) {
    dim3 grid(batch);
    dim3 block(THREADS);

    topk_specialized_v4_kernel<K>
    <<<grid, block, 0, stream>>>(
        input,
        output_values,
        output_indices,
        batch,
        n
    );

    CUDA_CHECK(cudaGetLastError());
}

} // namespace

bool topk_specialized_v4_supported(
    int k
) {
    return
        k == 1
        ||
        k == 2
        ||
        k == 4
        ||
        k == 8
        ||
        k == 16;
}

const char* topk_specialized_v4_name(
    int k
) {
    switch (k) {
        case 1:
            return "warpselect_k1_v4";

        case 2:
            return "warpselect_k2_v4";

        case 4:
            return "warpselect_k4_v4";

        case 8:
            return "warpselect_k8_v4";

        case 16:
            return "warpselect_k16_v4";

        default:
            return "unsupported_v4_k";
    }
}

void launch_topk_specialized_v4(
    const float* input,
    float* output_values,
    int* output_indices,
    int batch,
    int n,
    int k,
    cudaStream_t stream
) {
    switch (k) {
        case 1:
            launch_specialized<1>(
                input,
                output_values,
                output_indices,
                batch,
                n,
                stream
            );
            break;

        case 2:
            launch_specialized<2>(
                input,
                output_values,
                output_indices,
                batch,
                n,
                stream
            );
            break;

        case 4:
            launch_specialized<4>(
                input,
                output_values,
                output_indices,
                batch,
                n,
                stream
            );
            break;

        case 8:
            launch_specialized<8>(
                input,
                output_values,
                output_indices,
                batch,
                n,
                stream
            );
            break;

        case 16:
            launch_specialized<16>(
                input,
                output_values,
                output_indices,
                batch,
                n,
                stream
            );
            break;

        default:
            std::cerr
                << "V4 specialized WarpSelect supports K in "
                << "{1,2,4,8,16}, got K="
                << k
                << "\n";

            std::exit(EXIT_FAILURE);
    }
}
