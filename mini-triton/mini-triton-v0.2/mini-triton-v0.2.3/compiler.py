from frontend import Frontend
from passes import TypeShapePass,SimplifyPass
from backend import PTXBackend

def compile_kernel(src):
    ir=Frontend().compile(src)
    TypeShapePass().run(ir)
    SimplifyPass().run(ir)
    return ir, PTXBackend().emit(ir)
