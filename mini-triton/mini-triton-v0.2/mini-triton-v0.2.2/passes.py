
from tensor_types import TensorType, PointerType


class TypeCheckPass:

    def run(self, func):

        for op in func.ops:

            if op.opcode == "tensor_add":

                a = op.args[0]
                b = op.args[1]

                if a.ty.shape != b.ty.shape:
                    raise TypeError(
                        "tensor shape mismatch"
                    )


                if a.ty.dtype != b.ty.dtype:
                    raise TypeError(
                        "tensor dtype mismatch"
                    )


            if op.opcode == "tensor_load":

                ptr = op.args[0]
                offset = op.args[1]

                if not isinstance(ptr.ty, PointerType):
                    raise TypeError(
                        "load requires pointer"
                    )


                if not isinstance(offset.ty, TensorType):
                    raise TypeError(
                        "offset must be tensor"
                    )


        return func
