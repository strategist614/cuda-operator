from frontend import Frontend
from backend import PTXBackend
from pathlib import Path

def compile_kernel(source,name="add_kernel"):
    ir = Frontend().compile(source)

    Path("output").mkdir(exist_ok=True)

    Path(f"output/{name}.mir").write_text(
        ir.dump()
    )

    ptx = PTXBackend().emit(ir)

    Path(f"output/{name}.ptx").write_text(
        ptx
    )

    return ir, ptx
