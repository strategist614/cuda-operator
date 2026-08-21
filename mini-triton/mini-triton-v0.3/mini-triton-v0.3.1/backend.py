class PTXBackend:
    def emit(self,ir):
        return "\n".join(
            ["// Mini Triton v0.3.1",
             "// "+str(ir.launch)] +
            ["// "+str(x) for x in ir.ops]
        )
