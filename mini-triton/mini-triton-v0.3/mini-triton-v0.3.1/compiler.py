from frontend import Frontend
from passes import LayoutLoweringPass,ThreadLoweringPass
from backend import PTXBackend

def compile_kernel(src):
    ir=Frontend().compile(src)
    LayoutLoweringPass().run(ir)
    ThreadLoweringPass().run(ir)
    return ir,PTXBackend().emit(ir)
