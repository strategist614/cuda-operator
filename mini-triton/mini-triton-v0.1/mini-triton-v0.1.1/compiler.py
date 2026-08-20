from frontend import Frontend
from backend import PTXBackend
from pathlib import Path

def compile_kernel(src,name="kernel"):
    ir=Frontend().compile(src)

    Path("output").mkdir(exist_ok=True)

    Path("output/"+name+".mir").write_text(
        ir.dump()
    )

    ptx=PTXBackend().emit(ir)

    Path("output/"+name+".ptx").write_text(
        ptx
    )

    return ir,ptx
