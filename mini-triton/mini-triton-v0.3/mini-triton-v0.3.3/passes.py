
from ir import Op
from register_ir import Register


class RegisterLoweringPass:

    def run(self, ir):

        ir.ops.append(
            Op(
                "scalarize_tensor",
                None,
                ("tensor<256xf32>",
                 "per-thread values")
            )
        )

        ir.ops.append(
            Op(
                "allocate_register",
                None,
                (
                    Register("%f0","f32"),
                    Register("%f1","f32")
                )
            )
        )

        return ir
