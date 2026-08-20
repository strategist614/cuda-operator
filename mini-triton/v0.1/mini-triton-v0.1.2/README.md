# Mini Triton v0.1.2

A teaching GPU DSL compiler.

Pipeline:

Mini Triton DSL
    -> Python AST
    -> Mini SSA IR
    -> PTX

Run:

python demo.py

Generated files:
output/add_kernel.mir
output/add_kernel.ptx
