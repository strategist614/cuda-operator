
from ir import Op
from thread_ir import ThreadMapping,LaunchConfig


class LayoutLoweringPass:

    def run(self,ir):

        ir.mapping=ThreadMapping(
            128,
            2
        )

        ir.launch=LaunchConfig(
            128,
            4
        )

        return ir



class AddressLoweringPass:

    def run(self,ir):

        ir.ops.append(
            Op(
                "element_to_thread",
                None,
                (
                    "element_id / 2",
                    "thread_id"
                )
            )
        )

        ir.ops.append(
            Op(
                "address_calculation",
                None,
                (
                    "offset=thread_id*2"
                )
            )
        )

        return ir



class ThreadLoweringPass:

    def run(self,ir):

        ir.ops.append(
            Op(
                "thread_id",
                "%tid",
                ()
            )
        )

        return ir
