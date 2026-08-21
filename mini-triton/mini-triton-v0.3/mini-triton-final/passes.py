
from ir import Op,Value
from layout import BlockedLayout


class LayoutPass:

    def run(self,ir):

        ir.layout = BlockedLayout(
            warps=4,
            threads_per_warp=32,
            elements_per_thread=2
        )

        ir.ops.append(
            Op(
                "layout",
                None,
                (str(ir.layout),)
            )
        )

        return ir



class ThreadPass:

    def run(self,ir):

        ir.ops.append(
            Op(
                "thread_id",
                Value("%tid","i32"),
                ()
            )
        )

        ir.ops.append(
            Op(
                "lane_id",
                Value("%lane","i32"),
                ()
            )
        )

        return ir



class AddressPass:

    def run(self,ir):

        ir.ops.append(
            Op(
                "address",
                Value("%addr","ptr"),
                ("tid * elements_per_thread",)
            )
        )

        return ir



class RegisterAllocator:

    def run(self,ir):

        registers=[]

        for op in ir.ops:
            if op.result:
                reg=f"%r{len(registers)}"
                registers.append(
                    (op.result.name,reg)
                )

        ir.registers=registers

        ir.ops.append(
            Op(
                "register_allocate",
                None,
                tuple(registers)
            )
        )

        return ir
