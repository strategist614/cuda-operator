# High-Performance GPU Top-K V2

V2 在 V1 的并行扫描和 thread-local register Top-K 基础上，继续优化 **候选合并阶段**。

V1：

```text
256 thread local Top-K lists
          ↓
每个输出 rank
做一次 256-way shared-memory block reduction
          ↓
大量 __syncthreads()
```

V2：

```text
256 thread local Top-K lists
          ↓
8 warps
          ↓
warp shuffle merge
          ↓
8 warp Top-K lists
          ↓
warp0 做 8-way final merge
          ↓
final Top-K
```

---

## 1. V1 baseline

此前实测：

```text
RTX 2080
B=128
N=65536
K=16
FP32

V0:
525.86 ms

V1:
4.11993 ms

V1 speedup over V0:
~127.6x
```

V1 最大的新瓶颈已经不再是输入扫描，而是：

```text
shared-memory block merge
+
block-wide barriers
```

---

# 2. V2 保留 V1 的 parallel scan

输入扫描完全不变：

```text
256 threads / row
```

访问：

```text
thread 0:
0,256,512,...

thread 1:
1,257,513,...

...
```

同一个 warp 访问连续 input，因此 global load 可以 coalesce。

---

# 3. Thread-local Top-K 仍然在 registers

每个 thread：

```cpp
float local_values[16];
int local_indices[16];
```

仍然维护精确 local Top-K。

这一部分和 V1 一致，所以 V1/V2 的性能差主要来自：

```text
merge strategy
```

而不是 global scan。

---

# 4. V1 merge 为什么重

V1 每输出一个 rank：

```text
256 threads
   ↓
shared memory
   ↓
offset 128
__syncthreads

offset 64
__syncthreads

offset 32
__syncthreads

...

offset 1
__syncthreads
```

对于：

```text
K=16
```

大约需要很多轮：

```text
shared communication
+
block barrier
```

---

# 5. V2 Stage 1：Warp-level merge

256 threads：

```text
8 warps
```

每个 warp 内有：

```text
32 sorted local Top-K lists
```

每个 lane 有一个：

```text
local_cursor
```

每个 rank：

```text
32 lanes
各拿自己的当前 candidate
        ↓
__shfl_down_sync
        ↓
warp reduce best
        ↓
winner lane cursor++
```

数据交换：

```text
Register
   ↓
shuffle
   ↓
Register
```

不再经过 shared memory。

---

# 6. 为什么 warp merge 是 exact Top-K

每个 lane 的 local Top-K 已经排序：

```text
lane0:
a0 >= a1 >= ...

lane1:
b0 >= b1 >= ...

...
```

现在做的本质上就是：

```text
32-way merge of sorted lists
```

每轮取：

```text
32 个 list heads 中最大的一个
```

然后只推进 winner list。

重复 K 次：

```text
得到这个 warp 的 exact Top-K
```

不是 approximate selection。

---

# 7. V2 Stage 2：只 merge 8 个 warp lists

8 个 warp 都产生自己的：

```text
warp Top-K
```

写到很小的 shared memory：

```text
8 × K values
8 × K indices
```

然后只需要：

```text
1 次 __syncthreads()
```

保证 8 个 warp 结果可见。

之后：

```text
warp 0
```

负责 final merge。

lane：

```text
lane0 -> warp list 0
lane1 -> warp list 1
...
lane7 -> warp list 7
```

然后再次通过：

```cpp
__shfl_down_sync
```

做 8-way k-way merge。

---

# 8. 通信结构变化

V1：

```text
256 candidates
    ↓
shared memory
    ↓
block reduction
    ↓
barrier x many
```

V2：

```text
每组 32 candidates
    ↓
warp shuffle
    ↓
8 warp results
    ↓
one shared-memory handoff
    ↓
warp shuffle final merge
```

所以 V2 的核心优化就是：

> **把绝大多数 block-wide communication 降成 warp-local register communication。**

---

# 9. 编译

```bash
rm -rf build
mkdir build
cd build

cmake ..
cmake --build . -j

cd ..
```

默认目标：

```text
SM75 / RTX 2080
```

---

# 10. 第一条测试

```bash
./build/topk_bench \
  128 65536 16
```

默认：

```text
--kernel warp
```

即 V2。

应该看到：

```text
Implementation:
register_local_topk_warp_merge_v2

Correctness:
PASS
```

---

# 11. 直接 V1/V2 A/B

```bash
./build/topk_bench \
  128 65536 16 \
  --compare-v1
```

会输出：

```text
GPU selected latency = ...

V1 comparison
V1 register+block-merge latency = ...
V2 speedup over V1 = ...x
```

也可以分别运行：

```bash
./build/topk_bench \
  128 65536 16 \
  --kernel register
```

和：

```bash
./build/topk_bench \
  128 65536 16 \
  --kernel warp
```

---

# 12. NCU

```bash
NCU=/usr/local/cuda-13.2/bin/ncu \
./scripts/profile_v1_vs_v2.sh
```

最值得比较：

```text
Barrier stalls

Shared Memory throughput

MIO throttle

Eligible Warps

Active Warps

Registers / Thread

Achieved Occupancy
```

预期 V2：

```text
block-wide barrier cost ↓
shared merge traffic ↓
warp register communication ↑
```

---

# 13. V2 仍然可能有什么瓶颈

V2 只优化了 merge。

输入扫描阶段仍然是：

```text
每 thread local sorted Top-K insertion
```

对于 K=16：

```text
candidate insertion
可能形成很长的 register dependency chain
```

所以如果 V2 相比 V1 没有巨大提升，不代表 shuffle 无效。

可能说明：

```text
V1 的主要时间
其实已经花在 local candidate insertion
```

而不是 merge。

这正是 profiler / A-B benchmark 要验证的。

---

# 14. 当前 kernel family

V2 包仍保留三个版本：

```text
--kernel naive

V0:
serial repeated selection
```

```text
--kernel register

V1:
parallel scan
+
register local Top-K
+
block-wide shared merge
```

```text
--kernel warp

V2:
parallel scan
+
register local Top-K
+
warp shuffle merge
+
8-way hierarchical merge
```

方便做严格对照。

---

# 15. 下一步 V3

如果 V2 证明 merge 已经不是主要瓶颈，那么 V3 就不该继续优化 shuffle。

V3 应该针对：

```text
thread-local candidate queue
```

做优化，例如：

```text
smaller per-thread candidate capacity
+
warp-level cooperative selection
```

或者：

```text
multi-value / ILP scan
```

目标是降低：

```text
register pressure
+
serial insertion dependency
```

也就是开始真正优化 selection 本体，而不只是通信。
