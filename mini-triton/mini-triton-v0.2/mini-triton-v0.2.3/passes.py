from tensor_types import TensorType, PointerType

class TypeShapePass:
    def run(self, ir):
        for op in ir.ops:
            if op.opcode == "tensor_add":
                a,b=op.args
                if a.ty.shape != b.ty.shape:
                    raise TypeError("shape mismatch")
                if a.ty.dtype != b.ty.dtype:
                    raise TypeError("dtype mismatch")
        return ir


class SimplifyPass:
    def run(self, ir):
        return ir
