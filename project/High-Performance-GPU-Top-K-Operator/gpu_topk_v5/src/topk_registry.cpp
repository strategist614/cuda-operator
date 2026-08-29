#include "topk_registry.h"
#include "topk.h"

#include <vector>

namespace {

bool supports_v1(
    int,
    int,
    int k
) {
    return topk_register_v1_supported(k);
}

bool supports_v2(
    int,
    int,
    int k
) {
    return topk_warp_v2_supported(k);
}

bool supports_v3(
    int,
    int,
    int k
) {
    return topk_batch_v3_supported(k);
}

template<int K>
bool supports_exact_k(
    int,
    int,
    int k
) {
    return k == K;
}

template<int K>
void launch_v4_exact_k(
    const float* input,
    float* output_values,
    int* output_indices,
    int batch,
    int n,
    int,
    cudaStream_t stream
) {
    launch_topk_specialized_v4(
        input,
        output_values,
        output_indices,
        batch,
        n,
        K,
        stream
    );
}

} // namespace

const std::vector<TopKKernelConfig>&
get_topk_registry() {
    static const std::vector<TopKKernelConfig>
    registry = {
        /*
         * Historical diagnostic kernels.
         * Kept available for forced A/B profiling, but excluded from
         * default V5 autotuning because V3/V4 dominate them on the
         * current small-K regime.
         */
        {
            "register_v1",
            "v1_register_block_merge",
            1,
            16,
            false,
            false,
            &launch_topk_register_v1,
            &supports_v1
        },

        {
            "warp_v2",
            "v2_register_warp_merge",
            1,
            16,
            false,
            false,
            &launch_topk_warp_v2,
            &supports_v2
        },

        /*
         * V3 generic exact small-K WarpSelect.
         */
        {
            "batch_v3",
            "v3_warp_batch_threshold_bitonic",
            1,
            16,
            false,
            true,
            &launch_topk_batch_v3,
            &supports_v3
        },

        /*
         * V4 compile-time K-specialized kernels.
         */
        {
            "warpselect_k1_v4",
            "v4_k_specialized",
            1,
            1,
            true,
            true,
            &launch_v4_exact_k<1>,
            &supports_exact_k<1>
        },

        {
            "warpselect_k2_v4",
            "v4_k_specialized",
            2,
            2,
            true,
            true,
            &launch_v4_exact_k<2>,
            &supports_exact_k<2>
        },

        {
            "warpselect_k4_v4",
            "v4_k_specialized",
            4,
            4,
            true,
            true,
            &launch_v4_exact_k<4>,
            &supports_exact_k<4>
        },

        {
            "warpselect_k8_v4",
            "v4_k_specialized",
            8,
            8,
            true,
            true,
            &launch_v4_exact_k<8>,
            &supports_exact_k<8>
        },

        {
            "warpselect_k16_v4",
            "v4_k_specialized",
            16,
            16,
            true,
            true,
            &launch_v4_exact_k<16>,
            &supports_exact_k<16>
        }
    };

    return registry;
}

const TopKKernelConfig*
find_topk_kernel(
    const std::string& name
) {
    for (
        const auto& kernel :
        get_topk_registry()
    ) {
        if (name == kernel.name) {
            return &kernel;
        }
    }

    return nullptr;
}

std::vector<const TopKKernelConfig*>
get_compatible_topk_kernels(
    int batch,
    int n,
    int k
) {
    std::vector<const TopKKernelConfig*>
    out;

    for (
        const auto& kernel :
        get_topk_registry()
    ) {
        if (
            kernel.supports
            &&
            kernel.supports(
                batch,
                n,
                k
            )
        ) {
            out.push_back(
                &kernel
            );
        }
    }

    return out;
}
