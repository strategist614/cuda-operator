
from frontend import Frontend
from passes import (
    LayoutLoweringPass,
    ThreadLoweringPass,
    AddressLoweringPass
)
from backend import PTXBackend


def compile_kernel(src):

    ir=Frontend().compile(src)

    LayoutLoweringPass().run(ir)

    ThreadLoweringPass().run(ir)

    AddressLoweringPass().run(ir)

    return ir,PTXBackend().emit(ir)
