# Mini Triton v0.2.1

Mini Triton v0.2.1 是一个教学型 GPU DSL 编译器原型。本版本开始在 IR 中保留 Triton 风格的整块 tensor 语义：`arange()` 创建索引 tensor，`load()` 返回数据 tensor，`x + y` 表示 tensor 加法，`store()` 写回整个 tensor。

目录名是 `mini-triton-v0.2`，代码和生成文件中的版本标识为 `v0.2.1`。

当前目标是展示 Python DSL 到 Tensor IR 的 lowering，并将编译产物写入文件。PTX backend 仍是占位实现，只输出 operation 名称的注释，尚不能生成可加载执行的 GPU kernel。

## 相比 v0.1 的变化

- 从函数签名读取 kernel 参数，并把它们保存到 `FunctionIR.args`。
- 新增 `PointerType`，用于表示 X、Y、Z 等指针参数。
- 新增 `TensorType`，同时记录元素类型和 tensor shape。
- `arange(size)` 生成 `tensor<[size]xi32>` 索引值。
- `load(ptr, offsets)` 生成与 offsets shape 相同的 FP32 tensor。
- 普通 `+` 被 lowering 为 `tensor_add`，结果继承输入 tensor 类型。
- `store(ptr, offsets, value)` 被保留为无返回值的 `tensor_store`。
- IR dump 会同时打印函数参数、tensor 类型和 operation。

## 编译流程

```text
Python DSL source
        ↓ ast.parse()
Python AST
        ↓ Frontend
Tensor IR
        ├── dump → output/<name>.tir
        ↓ PTXBackend
占位 PTX 文本 → output/<name>.ptx
```

当前没有 optimizer/pass pipeline，也没有 runtime：

```text
Frontend().compile(src)
        ↓
FunctionIR
        ↓
PTXBackend().emit(ir)
```

## 文件结构

| 文件或目录 | 作用 |
| --- | --- |
| `tensor_types.py` | 定义 tensor 与 pointer 类型 |
| `ir.py` | 定义 `Value`、`Op`、`FunctionIR` |
| `frontend.py` | 将受限 Python AST lowering 为 Tensor IR |
| `backend.py` | 遍历 IR；目前只生成 opcode 注释 |
| `compiler.py` | 编译入口，并保存 `.tir` 与 `.ptx` 文件 |
| `demo.py` | 构造并编译一个 256 元素的 tensor add kernel |
| `output/add_kernel.tir` | demo 生成的 Tensor IR 快照 |
| `output/add_kernel.ptx` | demo 生成的占位 PTX 快照 |

`__pycache__/` 是 Python 自动生成的字节码缓存，不属于编译器实现。

## Demo DSL

`demo.py` 编译下面的源码：

```python
def add_kernel(X, Y, Z):
    offsets = arange(256)
    x = load(X, offsets)
    y = load(Y, offsets)
    z = x + y
    store(Z, offsets, z)
```

这段 DSL 描述一个长度为 256 的向量加法：

```text
Z[offsets] = X[offsets] + Y[offsets]
offsets = [0, 1, ..., 255]
```

与真正的 Triton kernel 相比，当前 DSL 没有 `program_id`，因此只描述从 offset 0 开始的一个 tensor block；它还没有 N 和 mask，无法处理任意长度或尾部越界。

## 类型系统

### PointerType

`PointerType` 当前只表示一个通用的 `ptr`：

```python
@dataclass
class PointerType:
    dtype: str = "ptr"
```

Frontend 会把函数签名中的每个参数都定义为 pointer，包括 X、Y、Z。当前还不能表达 `ptr<f32>`、地址空间、const 属性或标量参数。

### TensorType

`TensorType` 记录元素类型与 shape：

```python
@dataclass
class TensorType:
    dtype: str
    shape: list
```

例如：

```text
tensor<[256]xi32>  # 256 个 i32 offset
tensor<[256]xf32>  # 256 个 FP32 数据元素
```

这是 IR 层的逻辑类型，当前 backend 还没有决定它应如何映射到 CUDA thread、warp、寄存器或向量指令。

## IR 结构

### Value

`Value` 包含 SSA-like 名称和类型。Frontend 依次生成 `%0`、`%1`、`%2` 等名称：

```python
@dataclass
class Value:
    name: str
    ty: object
```

### Op

`Op` 保存 opcode、可选结果和 operands：

```python
@dataclass
class Op:
    opcode: str
    result: object
    args: tuple
```

`arange`、`tensor_load`、`tensor_add` 会生成结果；`tensor_store` 没有返回值。

### FunctionIR

`FunctionIR` 保存函数名、参数和 operation 列表。`dump()` 生成易读的 `.tir` 文本，但这不是 MLIR/Triton IR 的标准序列化格式。

demo 生成的 Tensor IR 为：

```text
func @add_kernel
  arg X:ptr
  arg Y:ptr
  arg Z:ptr
  %0:tensor<[256]xi32> = arange 256
  %1:tensor<[256]xf32> = tensor_load X:ptr %0:tensor<[256]xi32>
  %2:tensor<[256]xf32> = tensor_load Y:ptr %0:tensor<[256]xi32>
  %3:tensor<[256]xf32> = tensor_add %1:tensor<[256]xf32> %2:tensor<[256]xf32>
  tensor_store Z:ptr %0:tensor<[256]xi32> %3:tensor<[256]xf32>
```

## Frontend lowering

`Frontend.compile()` 使用 `ast.parse()` 解析源码，并选择第一个 `FunctionDef`。它遍历函数参数，把参数注册到 symbol table，然后逐条编译函数体。

当前支持：

| Python/DSL 写法 | Tensor IR |
| --- | --- |
| 函数参数 | `Value(name, PointerType())` |
| `offsets = arange(size)` | `arange -> tensor<[size]xi32>` |
| `x = load(ptr, offsets)` | `tensor_load -> tensor<offsets.shape × f32>` |
| `z = x + y` | `tensor_add`，结果类型继承左操作数 |
| `store(ptr, offsets, value)` | `tensor_store` |
| 简单赋值 | 将结果写入 frontend symbol table |

只有 `ast.Add` 会被处理，其他二元运算最终会落入不支持路径。Frontend 也没有检查 tensor shape、dtype 或 pointer/value operand 是否兼容。

## Backend 当前状态

`PTXBackend.emit()` 目前只输出版本注释和每条 IR 的 opcode：

```text
// Mini Triton v0.2.1
// Tensor IR
// arange
// tensor_load
// tensor_load
// tensor_add
// tensor_store
```

这个文件虽然使用 `.ptx` 扩展名，但不包含 PTX header、kernel entry、参数 ABI、寄存器、地址计算或任何可执行指令，不能交给 CUDA Driver API 加载。

要生成真实 PTX，backend 还需要决定 tensor operation 到 CUDA 执行模型的映射，例如：

- `arange(256)` 如何分配给 thread/warp。
- pointer 参数如何生成 `.param .u64` 和地址寄存器。
- tensor load/store 如何生成合并访存和尾部 predicate。
- tensor add 如何针对每个 lane 生成 FP32 指令。
- 多个 program 如何利用 `blockIdx` 覆盖完整输入。

## Compiler API

公开编译入口是：

```python
from compiler import compile_kernel

ir, ptx = compile_kernel(src, "add_kernel")
```

`compile_kernel()` 执行以下步骤：

1. 通过 `Frontend().compile(src)` 构建 Tensor IR。
2. 在当前工作目录创建 `output/`。
3. 将 `ir.dump()` 写入 `output/<name>.tir`。
4. 调用 `PTXBackend().emit(ir)` 生成占位 PTX。
5. 将结果写入 `output/<name>.ptx`。
6. 返回 `(ir, ptx)`。

传入的 `name` 控制输出文件名，IR 中的函数名仍来自 DSL 源码。输出路径相对于执行命令时的当前工作目录。

## 运行

本版本只使用 Python 标准库，不需要 Triton、PyTorch、CUDA Toolkit 或 NVIDIA GPU。建议从版本目录运行：

```bash
cd mini-triton/mini-triton-v0.2
python demo.py
```

如果系统 Python 不在 PATH 中，可使用仓库当前 Conda 环境：

```bash
conda run -n main python demo.py
```

程序会打印 Tensor IR 与占位 PTX，并更新：

```text
output/add_kernel.tir
output/add_kernel.ptx
```

## 当前限制

- 仅支持第一个 Python 函数；找不到函数时没有明确错误信息。
- 所有函数参数都被视为无元素类型的 pointer，尚不支持标量参数。
- 仅支持简单赋值、表达式语句、`arange`、`load`、`store` 和 tensor 加法。
- `arange()` 参数必须是 AST 整数字面量，不能使用变量或表达式。
- `load()` 固定返回 FP32 tensor。
- `tensor_add` 直接继承左操作数类型，没有 shape/dtype 校验。
- 没有 program id、grid、mask、broadcast、reduce、多维 tensor 或控制流。
- `Frontend` 实例重复编译时不会清空 ops、args、env 和 SSA id。
- Backend 只是 opcode dump，尚未生成真实 PTX。
- 没有优化 pass、runtime、数值正确性测试和性能 benchmark。

## 下一步建议

1. 增加 `program_id(axis)`、`BLOCK` 和 `offsets < N` mask。
2. 为参数增加 `ptr<dtype>` 和标量类型，并执行 shape/dtype 校验。
3. 明确 tensor block 到 CUDA threads/warps 的布局规则。
4. 实现真实 PTX kernel ABI、寄存器分配和 pointer 地址计算。
5. Lower `tensor_load`、`tensor_add`、`tensor_store` 为可执行 PTX。
6. 加入 CUDA Driver runtime，运行并验证任意长度的向量加法。
7. 补充 frontend 单元测试、非法 DSL 测试和编译快照测试。
