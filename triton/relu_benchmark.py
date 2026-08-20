import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets = (
        pid * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )

    mask = offsets < n_elements

    x = tl.load(
        x_ptr + offsets,
        mask=mask,
    )

    y = tl.maximum(x, 0.0)

    tl.store(
        y_ptr + offsets,
        y,
        mask=mask,
    )


def main():

    N = 16 * 1024 * 1024

    x = torch.randn(
        N,
        device="cuda",
        dtype=torch.float32,
    )

    y = torch.empty_like(x)

    for block_size in [
        64,
        128,
        256,
        512,
        1024,
    ]:

        grid = (
            triton.cdiv(N, block_size),
        )

        # 第一次 launch，同时拿到编译结果
        kernel = relu_kernel[grid](
            x,
            y,
            N,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )

        kernel._init_handles()

        # benchmark
        ms = triton.testing.do_bench(
            lambda: relu_kernel[grid](
                x,
                y,
                N,
                BLOCK_SIZE=block_size,
                num_warps=4,
            )
        )

        bytes_moved = (
            2
            * N
            * x.element_size()
        )

        gbps = (
            bytes_moved
            / (ms * 1e-3)
            / 1e9
        )

        print(
            f"BLOCK={block_size:4d} "
            f"regs={kernel.n_regs:3d} "
            f"time={ms:.4f} ms "
            f"bw={gbps:.2f} GB/s"
        )

        # 只对 256 看汇编
        if block_size == 256:

            print(
                "\nASM keys:",
                kernel.asm.keys(),
            )

            if "ptx" in kernel.asm:
                with open(
                    "relu_256.ptx",
                    "w",
                ) as f:
                    f.write(
                        kernel.asm["ptx"]
                    )

            sass = kernel.asm["sass"]

            with open(
                "relu_256.sass",
                "w",
            ) as f:
                f.write(sass)

            print(
                "\nGlobal memory instructions:"
            )

            for line in sass.splitlines():
                if (
                    "LDG" in line
                    or "STG" in line
                ):
                    print(line)


if __name__ == "__main__":
    main()