from frontend import Frontend
from passes import LayoutPass, ThreadMappingPass
from backend import PTXBackend


def compile_kernel(src):

    ir = Frontend().compile(src)

    LayoutPass().run(ir)

    ThreadMappingPass().run(ir)

    ptx = PTXBackend().emit(ir)

    return ir, ptx