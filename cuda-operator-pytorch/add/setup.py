from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


root = Path(__file__).parent.resolve()

setup(
    name="cuda-operator",
    version="0.1.0",

    packages=find_packages(),

    ext_modules=[
        CUDAExtension(
            name="cuda_operator._C",
            sources=[
                str(root / "csrc" / "operator.cpp"),
                str(root / "csrc" / "operator_cuda.cu"),
            ],
            extra_compile_args={
                "cxx": [
                    "-O3",
                ],
                "nvcc": [
                    "-O3",
                    "-lineinfo",
                ],
            },
        )
    ],

    cmdclass={
        "build_ext": BuildExtension,
    },

    zip_safe=False,
)