
class PTXBackend:
    def emit(self,ir):
        out=[
            "// Mini Triton v0.2.1",
            "// Tensor IR"
        ]

        for op in ir.ops:
            out.append("// "+op.opcode)

        return "\n".join(out)
