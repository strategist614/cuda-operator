# Mini Triton v0.1.2

Mini Triton v0.1.2 是一个教学型 GPU DSL 编译器骨架。它使用 Python AST 解析一小部分 Triton 风格语法，构建简单的 SSA-like IR，并生成 PTX 文本。

这个版本的重点是把 frontend、IR、pass、backend 和 compiler driver 拆分为独立模块，展示编译器的基本分层。目前生成的 PTX 只用于观察 lowering 过程，还不能作为完整 CUDA kernel 加载执行。

## 编译流程

```text
Python DSL source
        ↓ ast.parse()
Python AST
        ↓ Frontend
Mini IR（Value / Op / FunctionIR）
        ↓ ConstantFoldPass（当前为空实现）
PTXBackend
        ↓
PTX 文本与 .mir/.ptx 文件
```

当前 `compiler.py` 尚未把 `ConstantFoldPass` 接入实际 pipeline，所以真实调用链为：

```text
source → Frontend().compile() → ir.dump()
                              → PTXBackend().emit()
```

## 目录结构

| 文件或目录 | 作用 |
| --- | --- |
| `ir.py` | 定义 `Value`、`Op`、`FunctionIR` 三种 IR 数据结构 |
| `frontend.py` | 将 Python AST 转换为 Mini IR |
| `passes.py` | 预留优化 pass；当前 constant folding 直接返回原 IR |
| `backend.py` | 将 Mini IR 转换成简化 PTX 文本 |
| `compiler.py` | 串联 frontend 与 backend，并写出 `.mir`、`.ptx` 文件 |
| `demo.py` | 编译一个 load/store 示例并打印生成结果 |
| `output/add_kernel.mir` | demo 生成的 IR 快照 |
| `output/add_kernel.ptx` | demo 生成的 PTX 快照 |

`__pycache__/` 是 Python 自动生成的字节码缓存，不属于编译器逻辑。

## 示例 DSL

`demo.py` 中的输入源码为：

```python
def add_kernel(X, Y, Z, N):
    pid = program_id(0)
    x = arange(256)
    y = pid + x
    z = load(X, y)
    store(Z, y, z)
```

它表达的流程是：取得 program id 和 lane id，计算索引，从 X 加载一个 FP32 值，再把该值存入 Z。

需要注意，这段 demo 名为 `add_kernel`，但当前并没有读取 Y 或执行两个输入 tensor 的加法；它实际演示的是索引、load 和 store 的 IR 构建。另外，索引写成了 `pid + lane_id`，并不是常见的 `pid * BLOCK + lane_id`，不同 program 的地址区间会发生重叠。

## IR 设计

### Value

`Value` 表示一个带名称和类型的中间值：

```python
@dataclass
class Value:
    name: str
    ty: str
```

Frontend 使用 `%0`、`%1`、`%2` 等名称模拟 SSA value。目前使用的类型包括 `i32`、`f32` 和 `pred`。

### Op

`Op` 表示一条 IR operation：

```python
@dataclass
class Op:
    opcode: str
    result: Value | None
    args: tuple
```

有返回值的 operation 会持有一个 `Value`；`store` 等无结果 operation 的 `result` 为 `None`。

### FunctionIR

`FunctionIR` 保存函数名和 operation 列表。`dump()` 将它们格式化为可读文本，但当前 dump 格式不是 MLIR，也没有声明函数参数与返回类型。

demo 当前生成的 IR 为：

```text
func @add_kernel
  %0 = program_id ()
  %1 = lane_id ()
  %2 = add (Value(name='%0', ty='i32'), Value(name='%1', ty='i32'))
  %3 = load ('X', Value(name='%2', ty='i32'))
  store ('Z', Value(name='%2', ty='i32'), Value(name='%3', ty='f32'))
```

## Frontend

`Frontend.compile()` 调用 `ast.parse(source)`，直接取源码中的第一个顶层节点作为 kernel 函数。它用 `env` 保存变量到常量、参数名或 IR value 的映射。

预置符号如下：

```text
X, Y, Z, N
BLOCK = 256（默认值，可通过 Frontend(block=...) 修改）
```

当前可识别的语法：

| DSL/Python 语法 | 生成的 IR |
| --- | --- |
| `program_id(...)` | `program_id : i32` |
| `arange(...)` | `lane_id : i32` |
| `a * b` | `mul : i32` |
| `a + b` | `add : i32` |
| 比较表达式 | `cmp_lt : pred` |
| `load(...)` | `load : f32` |
| `store(...)` | 无返回值的 `store` |
| 简单赋值 | 把表达式结果写入 symbol table |

当前 frontend 没有严格验证函数数量、函数签名、调用参数数量和 AST 节点形态。例如所有比较运算都会被 lowering 成 `cmp_lt`，而 `program_id`、`arange` 的实参也不会被校验。

## Pass

`passes.py` 定义了 `ConstantFoldPass`：

```python
class ConstantFoldPass:
    def run(self, ir):
        return ir
```

它目前是占位实现，既没有执行常量折叠，也没有在 `compile_kernel()` 中被调用。后续可以把 pass pipeline 放在 frontend 和 backend 之间，并加入常量折叠、死代码消除及类型检查。

## PTX Backend

`PTXBackend.emit()` 默认生成以下 PTX header：

```text
.version 7.0
.target sm_70
.address_size 64
```

部分 operation 的映射如下：

| IR opcode | 当前 PTX 输出 |
| --- | --- |
| `program_id` | `mov.u32 %r1, %ctaid.x;` |
| `lane_id` | `mov.u32 %r2, %tid.x;` |
| `add` | `add.u32 %r3, %r1, %r2;` |
| `mul` | `mul.lo.u32 %r4, %r1, %r2;` |
| `load` | `// ld.global.f32` 注释 |
| `store` | `// st.global.f32` 注释 |

backend 当前使用固定寄存器，而不是根据每个 `Value` 建立寄存器映射；同时 kernel entry 是空参数列表，load/store 也没有生成地址计算和真实指令。因此输出适合查看框架流程，但不能正确表示任意 IR，更不能直接完成 demo 中的数据读写。

## Compiler Driver

公开入口位于 `compiler.py`：

```python
from compiler import compile_kernel

ir, ptx = compile_kernel(source, name="add_kernel")
```

`compile_kernel()` 会：

1. 调用 `Frontend().compile(source)` 生成 IR。
2. 在当前工作目录创建 `output/`。
3. 将 `ir.dump()` 写入 `output/<name>.mir`。
4. 调用 `PTXBackend().emit(ir)` 生成 PTX。
5. 将 PTX 写入 `output/<name>.ptx`。
6. 返回 `(ir, ptx)`。

`name` 参数只控制输出文件名；PTX entry 名仍来自 DSL 源码中的函数名。输出路径相对于运行命令时的当前目录，而不是相对于 `compiler.py`。

## 运行方式

本版本只依赖 Python 标准库，推荐从当前目录运行：

```bash
cd mini-triton/v0.1/mini-triton-v0.1.2
python demo.py
```

程序会打印 IR、PTX 和生成文件路径，并写出：

```text
output/add_kernel.mir
output/add_kernel.ptx
```

它不会初始化 CUDA、加载 PTX、启动 GPU kernel 或检查数值结果，因此运行 demo 不要求 NVIDIA GPU。

## 当前限制

- 只读取源码的第一个顶层 AST 节点，没有检查它是否确实为函数。
- Kernel 参数固定假设为 `X/Y/Z/N`，没有从函数签名构造 ABI。
- 表达式类型基本写死：加法和乘法输出 `i32`，load 输出 `f32`。
- 不支持减法、除法、循环、条件分支、return、多维索引和函数组合。
- 所有比较都按小于号处理，没有检查真实比较运算符。
- `Frontend` 实例重复调用 `compile()` 时不会清空 `ops` 和 SSA id。
- Constant folding pass 是空实现，并且尚未接入编译流程。
- Backend 固定寄存器，不跟踪 SSA operand，也没有寄存器声明。
- PTX entry 没有 kernel 参数，load/store 只有注释。
- 没有 CUDA runtime、正确性测试、异常测试或性能 benchmark。

## 后续建议

建议按以下顺序继续完善：

1. 从函数签名生成参数 IR 和 PTX `.param` ABI。
2. 为 SSA value 建立类型可靠的 PTX 寄存器映射。
3. 实现 FP32 指针地址计算、masked load/store 和 predicate。
4. 修正 demo 索引为 `pid * BLOCK + lane_id`，并加入 `offset < N` mask。
5. 把 pass manager 接入 `compile_kernel()`，实现真正的 constant folding。
6. 增加 frontend 语法校验、单元测试和错误用例。
7. 最后再接入 CUDA Driver API，验证生成 PTX 的数值正确性。
