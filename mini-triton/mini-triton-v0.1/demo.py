import ctypes

import numpy as np

from mini_triton import compile_kernel
from cuda.bindings import driver as cuda


SOURCE = """
def add_kernel(X, Y, Z, N):
    pid = program_id(0)

    offs = pid * BLOCK + arange(BLOCK)

    mask = offs < N

    x = load(X, offs, mask)
    y = load(Y, offs, mask)

    z = x + y

    store(Z, offs, z, mask)
"""


def check(result):
    """
    cuda.bindings 通常返回：

        (error,)
    或
        (error, value)
    """

    err, *values = result

    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(
            f"CUDA error: {err}"
        )

    if not values:
        return None

    if len(values) == 1:
        return values[0]

    return tuple(values)


def run():
    BLOCK = 256
    N = 1000

    # ========================================================
    # compile
    # ========================================================

    ir, ptx = compile_kernel(
        SOURCE,
        block=BLOCK,
    )

    print("======== Mini Triton IR ========")
    print(ir)

    print()
    print("======== PTX ========")
    print(ptx)

    # ========================================================
    # CPU data
    # ========================================================

    x = np.arange(
        N,
        dtype=np.float32,
    )

    y = np.arange(
        N,
        dtype=np.float32,
    ) * 2

    z = np.empty_like(x)

    # ========================================================
    # CUDA initialization
    # ========================================================

    check(
        cuda.cuInit(0)
    )

    device = check(
        cuda.cuDeviceGet(0)
    )

    context = check(
        cuda.cuDevicePrimaryCtxRetain(
            device
        )
    )

    check(
        cuda.cuCtxSetCurrent(
            context
        )
    )

    # ========================================================
    # PTX -> CUDA module
    # ========================================================

    # cuModuleLoadData expects module data.
    # PTX should be null terminated.
    ptx_image = np.char.array(
        ptx.encode("utf-8") + b"\0"
    )

    module = check(
        cuda.cuModuleLoadData(
            ptx_image
        )
    )

    kernel = check(
        cuda.cuModuleGetFunction(
            module,
            b"add_kernel",
        )
    )

    # ========================================================
    # Allocate GPU memory
    # ========================================================

    dx = check(
        cuda.cuMemAlloc(
            x.nbytes
        )
    )

    dy = check(
        cuda.cuMemAlloc(
            y.nbytes
        )
    )

    dz = check(
        cuda.cuMemAlloc(
            z.nbytes
        )
    )

    try:

        # ====================================================
        # CPU -> GPU
        # ====================================================

        check(
            cuda.cuMemcpyHtoD(
                dx,
                x.ctypes.data,
                x.nbytes,
            )
        )

        check(
            cuda.cuMemcpyHtoD(
                dy,
                y.ctypes.data,
                y.nbytes,
            )
        )

        # ====================================================
        # Launch
        # ====================================================

        grid = (
            N + BLOCK - 1
        ) // BLOCK

        kernel_values = (
            dx,
            dy,
            dz,
            N,
        )

        kernel_types = (
            None,
            None,
            None,
            ctypes.c_uint32,
        )

        check(
            cuda.cuLaunchKernel(
                kernel,

                # grid
                grid,
                1,
                1,

                # block
                BLOCK,
                1,
                1,

                # shared memory
                0,

                # stream
                0,

                # kernel params
                (
                    kernel_values,
                    kernel_types,
                ),

                # extra
                0,
            )
        )

        check(
            cuda.cuCtxSynchronize()
        )

        # ====================================================
        # GPU -> CPU
        # ====================================================

        check(
            cuda.cuMemcpyDtoH(
                z.ctypes.data,
                dz,
                z.nbytes,
            )
        )

    finally:

        check(
            cuda.cuMemFree(dx)
        )

        check(
            cuda.cuMemFree(dy)
        )

        check(
            cuda.cuMemFree(dz)
        )

        check(
            cuda.cuModuleUnload(
                module
            )
        )

        check(
            cuda.cuDevicePrimaryCtxRelease(
                device
            )
        )

    # ========================================================
    # Verify
    # ========================================================

    expected = x + y

    print()
    print("X:", x[:10])
    print("Y:", y[:10])
    print("Z:", z[:10])

    print(
        "correct:",
        np.allclose(
            z,
            expected,
        )
    )


if __name__ == "__main__":
    run()