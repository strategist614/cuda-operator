#include "topk.h"
#include "common.h"

#include <cuda_runtime.h>
#include <float.h>

/*
 * V0 naive GPU baseline.
 *
 * One block processes one row, but only thread 0 performs selection.
 * Each Top-K rank rescans the full row and skips indices selected before.
 *
 * Complexity per row: O(K * N).
 *
 * This is intentionally inefficient. V1 will parallelize the scan and
 * maintain thread-local candidates in registers.
 */
__global__
void topk_naive_kernel(
    const float* __restrict__ input,
    float* __restrict__ output_values,
    int* __restrict__ output_indices,
    int batch,
    int n,
    int k
) {
    const int row = blockIdx.x;

    if (row >= batch || threadIdx.x != 0) {
        return;
    }

    const float* row_ptr =
        input + static_cast<size_t>(row) * n;

    float* out_values =
        output_values + static_cast<size_t>(row) * k;

    int* out_indices =
        output_indices + static_cast<size_t>(row) * k;

    for (int rank = 0; rank < k; ++rank) {
        float best_value = -FLT_MAX;
        int best_index = -1;

        for (int col = 0; col < n; ++col) {
            bool already_selected = false;

            for (int prev = 0; prev < rank; ++prev) {
                if (out_indices[prev] == col) {
                    already_selected = true;
                    break;
                }
            }

            if (already_selected) {
                continue;
            }

            const float value = row_ptr[col];

            if (
                value > best_value
                ||
                (
                    value == best_value
                    &&
                    (best_index < 0 || col < best_index)
                )
            ) {
                best_value = value;
                best_index = col;
            }
        }

        out_values[rank] = best_value;
        out_indices[rank] = best_index;
    }
}

void launch_topk_naive(
    const float* input,
    float* output_values,
    int* output_indices,
    int batch,
    int n,
    int k,
    cudaStream_t stream
) {
    dim3 grid(batch);
    dim3 block(128);

    topk_naive_kernel<<<grid, block, 0, stream>>>(
        input,
        output_values,
        output_indices,
        batch,
        n,
        k
    );

    CUDA_CHECK(cudaGetLastError());
}
