
from frontend import Frontend
from passes import RegisterLoweringPass
from backend import PTXBackend


def compile_kernel(src):

    ir=Frontend().compile(src)

    RegisterLoweringPass().run(ir)

    return ir, PTXBackend().emit(ir)
