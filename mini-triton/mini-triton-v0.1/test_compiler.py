from mini_triton import compile_kernel


SOURCE = """
def add_kernel(X, Y, Z, N):
    pid = program_id(0)

    offs = pid * BLOCK + arange(BLOCK)

    mask = offs < N

    x = load(X, offs, mask)
    y = load(Y, offs, mask)

    z = x + y

    store(Z, offs, z, mask)
"""


ir, ptx = compile_kernel(
    SOURCE,
    block=256,
)

print("======== IR ========")
print(ir)

print()
print("======== PTX ========")
print(ptx)