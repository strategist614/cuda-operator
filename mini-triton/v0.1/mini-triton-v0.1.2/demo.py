from compiler import compile_kernel

source = """
def add_kernel(X,Y,Z,N):
    pid = program_id(0)
    x = arange(256)
    y = pid + x
    z = load(X,y)
    store(Z,y,z)
"""

ir, ptx = compile_kernel(source)

print("===== IR =====")
print(ir.dump())

print()
print("===== PTX =====")
print(ptx)

print()
print("Generated:")
print("output/add_kernel.mir")
print("output/add_kernel.ptx")
