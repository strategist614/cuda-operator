
from frontend import Frontend
from passes import (
    LayoutPass,
    ThreadPass,
    AddressPass,
    RegisterAllocator
)
from backend import PTXBackend


def compile_kernel(src):

    ir=Frontend().compile(src)

    LayoutPass().run(ir)
    ThreadPass().run(ir)
    AddressPass().run(ir)
    RegisterAllocator().run(ir)

    return ir,PTXBackend().emit(ir)
