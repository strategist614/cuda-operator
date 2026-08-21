
class PTXBackend:

    def emit(self,ir):

        out=[
            "// Mini Triton v0.3",
            "// Layout lowered"
        ]

        for op in ir.ops:
            out.append(
                "// "+str(op)
            )

        return "\n".join(out)
