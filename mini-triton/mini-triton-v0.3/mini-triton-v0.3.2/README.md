# Mini Triton v0.3.2

Mini Triton v0.3.2 在固定线程映射的基础上继续向地址生成推进。它为 256 个逻辑元素指定 128 个线程、每线程 2 个元素，并通过 AddressLoweringPass 添加 element→thread 与地址计算 operation；backend 还输出几条 PTX 风格的线程索引、乘法和 global load 指令。

这些指令目前只是 codegen 模拟，不构成完整 PTX module。实现没有 kernel ABI、寄存器声明、真实 base address、第二个元素、Y 输入、加法、store 或边界保护，因此不能执行 demo 描述的向量加法。

## 本版本新增内容

- `ThreadMapping.element_owner()`：查询某个逻辑元素由哪个线程负责。
- `AddressLoweringPass`：追加 element→thread 和 address calculation IR。
- Backend 开始输出 PTX 风格的 `%tid.x`、整数乘法和 `ld.global.f32`。
- Pipeline 明确分为 layout、thread、address、backend 四个 lowering 阶段。

## 相比 v0.3.1 的变化

| 项目 | v0.3.1 | v0.3.2 |
| --- | --- | --- |
| Thread mapping API | `element_to_thread(e)` | `element_owner(element)` |
| LayoutLoweringPass | 附加 mapping/launch，并追加 thread_mapping op | 只附加 mapping/launch 属性 |
| ThreadLoweringPass | 追加 thread_id 和 lane_id | 只追加 thread_id |
| Address lowering | 无 | 新增 AddressLoweringPass |
| Backend | 全部输出为注释 | 额外输出三行 PTX 风格文本 |
| 实际 GPU codegen | 不支持 | 仍不支持 |

v0.3.2 增加了地址 lowering，但移除了 v0.3.1 的 lane_id 和显式 thread_mapping operation。`ir.mapping` 仍然存在，只是不会出现在 `ir.dump()` 中。

## 编译流程

```text
Python DSL
    ↓ Frontend
Tensor IR（固定 256 元素）
    ↓ LayoutLoweringPass
ThreadMapping(128, 2) + LaunchConfig(128, 4)
    ↓ ThreadLoweringPass
thread_id IR
    ↓ AddressLoweringPass
element_to_thread + address_calculation IR
    ↓ PTXBackend
IR 注释 + PTX 地址生成模拟
```

实际 compiler pipeline：

```python
ir = Frontend().compile(src)
LayoutLoweringPass().run(ir)
ThreadLoweringPass().run(ir)
AddressLoweringPass().run(ir)
ptx = PTXBackend().emit(ir)
```

## 文件结构

| 文件 | 作用 |
| --- | --- |
| `ir.py` | 定义 `Value`、`Op`、`FunctionIR` |
| `thread_ir.py` | 定义 `ThreadMapping` 与 `LaunchConfig` |
| `frontend.py` | 把受限 Python AST lowering 为固定 tensor IR |
| `passes.py` | 实现 layout、thread 和 address lowering |
| `backend.py` | 打印 IR，并追加 PTX 风格地址生成文本 |
| `compiler.py` | 按顺序运行三个 pass 和 backend |
| `demo.py` | 编译并打印一个 256 元素 tensor add 示例 |

`__pycache__/` 是 Python 自动生成的字节码缓存，不属于编译器源码。

## Demo DSL

```python
def add_kernel(X, Y, Z):
    offsets = arange(256)
    x = load(X, offsets)
    y = load(Y, offsets)
    z = x + y
    store(Z, offsets, z)
```

目标语义为：

```text
Z[0:256] = X[0:256] + Y[0:256]
```

当前 DSL 没有 program id、N 或 mask，所以只表示一个固定 tensor block，不能覆盖任意长度输入或安全处理尾部。

## Frontend

Frontend 解析第一个 Python 函数，将参数 X、Y、Z 注册为 `ptr` 类型的 Value，然后编译简单赋值和表达式语句。

当前 lowering：

| Python/DSL | IR |
| --- | --- |
| 函数参数 | `Value(name, "ptr")` |
| `arange(...)` | `make_tensor : tensor<256>` |
| `load(ptr, offsets)` | `load : tensor` |
| 任意二元表达式 | `tensor_add : tensor` |
| `store(ptr, offsets, value)` | 无结果的 `store` |

`arange()` 的实参被忽略，shape 固定为 256。Frontend 对所有 `ast.BinOp` 都生成 tensor_add，也没有 dtype、shape 或 operand 检查。

## ThreadMapping

本版本固定：

```text
threads             = 128
elements_per_thread = 2
total elements      = 256
```

`element_owner(element)` 的实现是：

```python
return element // self.elements_per_thread
```

所以：

```text
element 0, 1     → thread 0
element 2, 3     → thread 1
...
element 254, 255 → thread 127
```

该方法没有校验 element 范围。负数或大于 255 的输入仍会返回一个整数线程编号。

## LaunchConfig

LayoutLoweringPass 动态附加：

```python
ir.mapping = ThreadMapping(128, 2)
ir.launch = LaunchConfig(128, 4)
```

launch 字符串为：

```text
threads=128, warps=4
```

128 个线程对应 4 个 32-thread warp，但当前代码没有验证 `threads_per_block == warps * 32`，也没有把 launch config 传给任何 runtime。

## ThreadLoweringPass

该 pass 在原 tensor operations 之后追加：

```text
%tid = thread_id
```

`%tid` 是普通字符串，而不是 SSA allocator 生成的 Value。它也没有替换 tensor load/store 中的 offsets，因此 thread id 暂时只作为后续 backend 模拟的概念输入。

## AddressLoweringPass

该 pass 继续追加两条无返回值 operation。设计意图是：

```text
element_to_thread element_id / 2 thread_id
address_calculation offset=thread_id*2
```

它表达的意图是：

1. 每两个逻辑元素分配给同一线程。
2. 线程 `tid` 的第一个逻辑元素索引是 `tid * 2`。

但 operation operands 只是字符串，没有连接到 `%tid` Value，也没有生成第二个元素 `tid * 2 + 1`。Pass 只在函数尾部添加描述，没有把原来的 tensor load/add/store 重写成逐线程操作。

另外，源码中的 address calculation 参数写成：

```python
("offset=thread_id*2")
```

这仍然是一个字符串，并不是单元素 tuple；正确的单元素 tuple 需要尾逗号：

```python
("offset=thread_id*2",)
```

由于 `Op.__str__()` 会遍历 `args`，当前字符串被按字符遍历，实际 dump 会显示为 `o f f s e t ...`。

## Address IR 输出

demo 的 `ir.dump()` 类似：

```text
func @add_kernel
  arg X:ptr
  arg Y:ptr
  arg Z:ptr
  %0:tensor<256> = make_tensor 256
  %1:tensor = load X:ptr %0:tensor<256>
  %2:tensor = load Y:ptr %0:tensor<256>
  %3:tensor = tensor_add %1:tensor %2:tensor
  store Z:ptr %0:tensor<256> %3:tensor
  %tid = thread_id
  element_to_thread element_id / 2 thread_id
  address_calculation o f f s e t = t h r e a d _ i d * 2
```

这些新增 operation 出现在 store 之后，说明它们目前是 lowering 说明，而不是可按顺序执行的低层 IR。

## PTX 地址生成模拟

backend 在 IR 注释后追加：

```text
mov.u32 %r1, %tid.x;
mul.lo.u32 %r2, %r1, 2;
ld.global.f32 %f1, [addr+%r2];
```

三行文本表达：读取 CUDA thread id、乘以每线程元素数 2、使用结果构造 load 地址。

但对于 FP32，`ld.global` 地址是字节地址。若 `%r2 = thread_id * 2` 表示元素索引，还需要乘以 `sizeof(float) = 4`：

```text
first_element = thread_id * 2
byte_offset   = first_element * 4
              = thread_id * 8
```

此外，一个线程负责两个元素，完整实现还应处理 `first_element + 1`。当前 backend 只模拟一次 X-like load，没有区分 X/Y/Z base pointer，也没有 add 或 store。

## 为什么当前输出不是可执行 PTX

- 缺少 `.version`、`.target` 和 `.address_size`。
- 缺少 `.entry` kernel 和参数 ABI。
- `%r1`、`%r2`、`%f1` 没有寄存器声明。
- `addr` 未定义，也没有从 X 参数加载 base pointer。
- 地址偏移没有按 FP32 转换成字节。
- 每线程第二个元素没有处理。
- 没有加载 Y、执行加法或写入 Z。
- 没有 N、predicate 或尾部 mask。

因此 backend 返回的是 PTX codegen 草图，不能交给 CUDA Driver API 编译执行。

## Compiler API

```python
from compiler import compile_kernel

ir, ptx = compile_kernel(src)
print(ir.dump())
print(ptx)
```

`compile_kernel()` 只返回内存中的 IR 与字符串，不创建 output 目录，也不写文件。backend 假设 LayoutLoweringPass 已运行并创建 `ir.launch`。

## 运行

本版本只依赖 Python 标准库，不要求 GPU：

```bash
cd mini-triton/mini-triton-v0.3/mini-triton-v0.3.2
python demo.py
```

或者：

```bash
conda run -n main python demo.py
```

demo 会打印 Address IR 和 PTX 风格模拟文本，不会启动 CUDA kernel 或验证数值结果。

## 当前限制

- Tensor shape、dtype、mapping 和 launch 全部硬编码。
- 所有 binary operation 都被当成 tensor add。
- 没有 program id、grid、N、mask、多维 tensor 或控制流。
- Thread/Address operation 使用字符串而不是正规 SSA Value。
- address_calculation 的单参数缺少 tuple 尾逗号，dump 会逐字符展开。
- Lowering pass 只在原 store 后追加描述，没有真正重写 tensor IR。
- element_owner 没有边界检查。
- 地址计算遗漏 FP32 字节缩放和每线程第二个元素。
- Backend 只实现一次不完整 load，没有 Y/add/store。
- 输出不是合法 PTX，没有 runtime、正确性测试或 benchmark。

## 下一步建议

1. 将 thread id、element id、address 定义为有类型的 SSA Value。
2. 把 tensor load/add/store 重写为每线程两个元素的低层 operations。
3. 正确生成 `thread_id * elements_per_thread * sizeof(dtype)` 字节偏移。
4. 为第二个元素生成 `base + byte_offset + sizeof(dtype)`。
5. 加入 X/Y/Z 参数 ABI、寄存器声明、load/add/store。
6. 增加 program id、N 和 predicated tail handling。
7. 验证 mapping、launch、shape 和 dtype 的一致性。
8. 接入 CUDA Driver runtime，并用 CPU reference 验证结果。
