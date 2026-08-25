import os


os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_sm86_compile_cache")

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource

from gemm_optimized import gemm_optimized_kernel


def compile_sm86_gemm():
    function = gemm_optimized_kernel.fn
    constants = {
        "stride_ak": 1,
        "stride_bn": 1,
        "stride_cn": 1,
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_K": 32,
        "GROUP_SIZE_M": 8,
        "PIPELINE_STAGES": 3,
    }
    signature = {
        name: (
            "constexpr"
            if name in constants
            else "*fp16"
            if name in ("a_ptr", "b_ptr", "c_ptr")
            else "i32"
        )
        for name in function.arg_names
    }
    divisible_by_16 = {
        "a_ptr",
        "b_ptr",
        "c_ptr",
        "m",
        "n",
        "k",
        "stride_am",
        "stride_bk",
        "stride_cm",
    }
    attributes = {
        (index,): [["tt.divisibility", 16]]
        for index, name in enumerate(function.arg_names)
        if name in divisible_by_16
    }
    source = ASTSource(
        function,
        signature,
        constexprs=constants,
        attrs=attributes,
    )
    return triton.compile(
        source,
        target=GPUTarget("cuda", 86, 32),
        options={"num_warps": 4, "num_stages": 3},
    )


def main() -> None:
    compiled = compile_sm86_gemm()
    ptx = compiled.asm["ptx"]
    mma_count = ptx.count("mma.sync")
    async_copy_count = ptx.count("cp.async")
    has_commit = "cp.async.commit_group" in ptx
    has_wait = "cp.async.wait_group" in ptx
    shared_bytes = compiled.metadata.shared

    tensor_core_passed = mma_count > 0
    double_buffer_passed = (
        async_copy_count > 0 and has_commit and has_wait and shared_bytes >= 16384
    )
    passed = tensor_core_passed and double_buffer_passed

    print(f"target={compiled.metadata.arch}")
    print(f"num_warps={compiled.metadata.num_warps}")
    print(f"num_stages={compiled.metadata.num_stages}")
    print(f"shared_bytes={shared_bytes}")
    print(f"mma_sync_count={mma_count}")
    print(f"cp_async_count={async_copy_count}")
    print(f"cp_async_commit_group={has_commit}")
    print(f"cp_async_wait_group={has_wait}")
    print(f"tensor_core_status={'PASS' if tensor_core_passed else 'FAIL'}")
    print(f"double_buffer_status={'PASS' if double_buffer_passed else 'FAIL'}")
    print(f"status={'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
