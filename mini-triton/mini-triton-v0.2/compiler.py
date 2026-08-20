
from frontend import Frontend
from backend import PTXBackend
from pathlib import Path

def compile_kernel(src,name):
    ir=Frontend().compile(src)

    Path("output").mkdir(exist_ok=True)

    Path(f"output/{name}.tir").write_text(
        ir.dump()
    )

    ptx=PTXBackend().emit(ir)

    Path(f"output/{name}.ptx").write_text(
        ptx
    )

    return ir,ptx
