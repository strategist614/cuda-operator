
class PTXBackend:

    def emit(self,ir):

        out=[
            "// Mini Triton v0.2.2",
            "// after type checking"
        ]

        for op in ir.ops:
            out.append("// "+op.opcode)

        return "\n".join(out)
