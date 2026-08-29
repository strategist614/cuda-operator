#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)

__device__ __forceinline__
void mma_sm75_m16n8k8_f32_f16_f16_f32(
    float &d0,
    float &d1,
    float &d2,
    float &d3,
    uint32_t a0,
    uint32_t a1,
    uint32_t b0,
    float c0,
    float c1,
    float c2,
    float c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5}, "
        "{%6}, "
        "{%7,%8,%9,%10};\n"
        : "=f"(d0),
          "=f"(d1),
          "=f"(d2),
          "=f"(d3)
        : "r"(a0),
          "r"(a1),
          "r"(b0),
          "f"(c0),
          "f"(c1),
          "f"(c2),
          "f"(c3)
    );
}

#endif
