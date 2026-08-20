try:
    from cuda.bindings import driver as cuda
except ImportError:
    cuda=None

def load_ptx(ptx):
    if cuda is None:
        raise RuntimeError("install cuda-python")

    cuda.cuInit(0)
    dev=cuda.cuDeviceGet(0)[1]
    ctx=cuda.cuDevicePrimaryCtxRetain(dev)[1]
    cuda.cuCtxSetCurrent(ctx)

    module=cuda.cuModuleLoadData(ptx.encode())[1]
    return module

def get_kernel(module,name):
    return cuda.cuModuleGetFunction(
        module,
        name.encode()
    )[1]
