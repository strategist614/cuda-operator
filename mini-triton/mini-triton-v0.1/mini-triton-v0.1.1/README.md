# Mini Triton v0.1.1

这是一个教学型 GPU DSL 编译器骨架，通过 Python AST 把少量 Triton 风格语法转换为自定义 Mini IR，再生成简化的 PTX 文本。本版本还增加了基础 CUDA Driver API 封装，为后续加载 PTX 和获取 kernel function 做准备。

目录名为 `mini-triton-v0.1.1`，旧 README 曾标记为 `v0.1.3`；本文以实际目录版本 v0.1.1 为准。

当前可稳定演示的流程是 DSL → AST → Mini IR → PTX 文本。`runtime.py` 提供 PTX module 加载接口，但 `demo.py` 并未调用它，而且当前 backend 生成的 PTX 缺少 kernel 参数、寄存器声明和真实 load/store 指令，因此还不能完成 GPU 上的向量加法。

## 当前编译流程

```text
Python DSL source
        ↓ ast.parse()
Python AST
        ↓ Frontend
Mini SSA-like IR
        ├── dump → output/<name>.mir
        ↓ PTXBackend
简化 PTX 文本 → output/<name>.ptx
```

规划中的完整流程是：

```text
DSL → AST → Mini IR → valid PTX
                         ↓ CUDA Driver JIT
                        SASS → GPU
```

后半段目前只有 `load_ptx()` 和 `get_kernel()` 的接口骨架，尚未在 demo 中打通。

## 文件结构

| 文件或目录 | 作用 |
| --- | --- |
| `ir.py` | 定义 `Value`、`Op` 和 `FunctionIR` |
| `frontend.py` | 将受限 Python AST lowering 为 Mini IR |
| `backend.py` | 将部分 IR opcode 映射成简化 PTX 文本 |
| `compiler.py` | 串联 frontend/backend，并保存 `.mir`、`.ptx` |
| `runtime.py` | 初始化 CUDA Driver、加载 PTX module、获取 kernel function |
| `demo.py` | 编译并打印一个 load/store 示例，不启动 GPU |
| `output/add_kernel.mir` | demo 生成的 IR 快照 |
| `output/add_kernel.ptx` | demo 生成的简化 PTX 快照 |

`__pycache__/` 是 Python 自动生成的字节码缓存，不属于编译器实现。

## Demo DSL

`demo.py` 编译：

```python
def add_kernel(X, Y, Z, N):
    pid = program_id(0)
    x = arange(256)
    y = pid + x
    z = load(X, y)
    store(Z, y, z)
```

这段代码生成 program id、lane id、整数索引、load 和 store operation。虽然函数名是 `add_kernel`，它没有读取参数 Y，也没有执行 `X + Y`；实际表达的是把 `X[y]` 读取后写入 `Z[y]`。

索引 `pid + lane_id` 也只是演示写法。常见的一维索引应该是：

```text
offset = pid * BLOCK + lane_id
```

否则相邻 program 处理的地址范围会大量重叠。此外，当前 DSL 没有 `offset < N` mask，无法安全处理尾部线程。

## Mini IR

### Value

```python
@dataclass
class Value:
    name: str
    ty: str
```

`Value` 表示一个 SSA-like 中间值。Frontend 使用 `%0`、`%1`、`%2` 依次命名，当前类型字符串包括 `i32` 和 `f32`。

### Op

```python
@dataclass
class Op:
    opcode: str
    result: Value | None
    args: tuple
```

`program_id`、`lane_id`、`add`、`mul` 和 `load` 会产生结果；`store` 没有结果。

### FunctionIR

`FunctionIR` 只保存函数名与 operation 列表。`dump()` 生成项目自定义的可读文本，不是 MLIR、LLVM IR 或官方 Triton IR。

demo 当前生成：

```text
func @add_kernel
  %0 = program_id ()
  %1 = lane_id ()
  %2 = add (Value(name='%0', ty='i32'), Value(name='%1', ty='i32'))
  %3 = load ('X', Value(name='%2', ty='i32'))
  store ('Z', Value(name='%2', ty='i32'), Value(name='%3', ty='f32'))
```

函数参数 X/Y/Z/N 没有进入 FunctionIR 的参数列表，而是在表达式中以普通字符串出现。

## Frontend

`Frontend.compile()` 使用 `ast.parse()`，选择第一个 `FunctionDef` 作为 kernel；找不到函数时抛出 `kernel function not found`。

当前支持：

| Python/DSL | Mini IR |
| --- | --- |
| 简单变量赋值 | 将结果保存到 symbol table |
| 整数字面量 | Python int |
| `a + b` | `add : i32` |
| `a * b` | `mul : i32` |
| `program_id(...)` | `program_id : i32` |
| `arange(...)` | `lane_id : i32` |
| `load(...)` | `load : f32` |
| `store(...)` | 无结果的 `store` |

未在 symbol table 中出现的变量会直接以变量名字符串返回，因此 X、Y、Z、N 不需要显式注册。这个设计简单，但不能检查未定义变量或区分 kernel 参数与拼写错误。

Frontend 只接受 `ast.Name` 形式的简单函数调用，不支持属性调用、控制流、比较、mask、减法、除法和其他 Python 语法。

## PTX Backend

backend 输出固定 header：

```text
.version 7.0
.target sm_70
.address_size 64
```

当前 opcode 映射：

| Mini IR | PTX 文本 |
| --- | --- |
| `program_id` | `mov.u32 %r1, %ctaid.x;` |
| `lane_id` | `mov.u32 %r2, %tid.x;` |
| `add` | `add.u32 %r3, %r1, %r2;` |
| `mul` | `mul.lo.u32 %r4, %r1, %r2;` |
| `load` | `// ld.global.f32` 注释 |
| `store` | `// st.global.f32` 注释 |

backend 使用固定寄存器，不读取 operation 的真实 operands 或 result 映射。因此多个 add/mul 也会重复覆盖同一寄存器，无法正确表达一般 IR。

当前输出还存在以下不可执行点：

- `.entry add_kernel()` 没有 X/Y/Z/N 参数 ABI。
- 使用 `%r1` 等寄存器前没有 `.reg` 声明。
- load/store 只是注释，没有 pointer 地址计算和 global-memory 指令。
- 没有 predicate 或边界 mask。

所以 `output/add_kernel.ptx` 是 codegen 草稿，而不是可执行 kernel。

## Compiler API

```python
from compiler import compile_kernel

ir, ptx = compile_kernel(src, name="add_kernel")
```

`compile_kernel()` 会：

1. 通过 `Frontend().compile(src)` 构建 IR。
2. 在当前工作目录创建 `output/`。
3. 把 `ir.dump()` 写入 `output/<name>.mir`。
4. 通过 `PTXBackend().emit(ir)` 生成 PTX 文本。
5. 把 PTX 写入 `output/<name>.ptx`。
6. 返回 `(ir, ptx)`。

`name` 只控制输出文件名，PTX entry 使用 DSL 中的函数名。输出路径相对于执行命令时的当前目录。

## Runtime API

`runtime.py` 尝试导入：

```python
from cuda.bindings import driver as cuda
```

它提供两个函数：

- `load_ptx(ptx)`：初始化 CUDA、取得 device 0、保留 primary context、设置当前 context，然后通过 `cuModuleLoadData()` 加载 PTX。
- `get_kernel(module, name)`：通过 `cuModuleGetFunction()` 获取 kernel handle。

如果没有安装 cuda-python，会抛出 `RuntimeError("install cuda-python")`。

当前 runtime 只是最小骨架：它没有统一检查 CUDA 返回码，没有分配/拷贝显存，没有 launch kernel，也没有释放 module 或 primary context。由于 backend PTX 尚不完整，直接调用 `load_ptx()` 也可能在 Driver JIT 阶段失败。

## 运行编译演示

只运行 DSL → IR → PTX 文本不需要 GPU，也不需要 cuda-python：

```bash
cd mini-triton/mini-triton-v0.1/mini-triton-v0.1.1
python demo.py
```

如果系统没有 `python` 命令，可使用当前 Conda 环境：

```bash
conda run -n main python demo.py
```

程序会打印 IR 和 PTX，并更新：

```text
output/add_kernel.mir
output/add_kernel.ptx
```

demo 不会调用 `runtime.py`，不会执行 GPU kernel，也不会进行数值正确性检查。

## 当前限制

- FunctionIR 没有参数类型或 kernel ABI。
- 所有 add/mul 固定产生 i32，load 固定产生 f32。
- 未知变量会变成字符串，无法发现参数拼写错误。
- `program_id` 和 `arange` 不校验参数。
- 不支持 mask、比较、控制流、多维索引和 tensor block 语义。
- Frontend 实例重复编译时不会重置 ops、env 和 SSA id。
- Backend 使用固定寄存器且忽略真实 SSA 映射。
- PTX 缺少寄存器声明、参数和真实 load/store。
- Runtime 不处理大多数 CUDA 错误，也没有 launch、内存管理与资源释放。
- 没有单元测试、GPU correctness test 或 benchmark。

## 后续建议

1. 把函数参数与类型加入 FunctionIR，并生成 PTX `.param` ABI。
2. 为每个 SSA value 分配与类型匹配的 PTX 寄存器。
3. 修正 demo 索引为 `pid * BLOCK + lane_id`，增加 N 和 mask。
4. 实现 pointer 地址计算及 predicated `ld.global.f32`/`st.global.f32`。
5. 为 runtime 增加 CUDA 错误检查、内存管理、kernel launch 和资源释放。
6. 添加 frontend 负例测试、PTX snapshot 测试和 GPU 数值验证。
