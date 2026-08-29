# High-Performance GPU Top-K V1

V1 的目标是把 V0 最大的问题直接解决：

```text
V0:
one block / row
128 threads
but only thread 0 works

每一个 Top-K rank
都重新扫描 N
```

升级成：

```text
V1:
one block / row
256 active threads

parallel strided scan
        ↓
thread-local register Top-K
        ↓
block-level k-way merge
        ↓
final Top-K
```

目标 workload：

```text
B=128
N=65536
K=16
FP32
RTX 2080 / SM75
```

---

## 1. V0 baseline

此前实测：

```text
CPU partial_sort:
74.27 ms

CUDA V0:
525.86 ms

Correctness:
PASS
```

V0 的主要问题：

```text
1. 只有一个 thread 工作
2. global loads 没有利用 warp coalescing
3. 每一个 rank 都重新扫描整行
4. selected-index 检查进一步增加复杂度
```

---

# 2. V1：256 threads parallel scan

V1 一个 block 负责一行：

```text
256 threads
```

访问：

```text
thread 0:
0, 256, 512, ...

thread 1:
1, 257, 513, ...

...

thread 255:
255, 511, 767, ...
```

所以同一个 warp 的第一次访问：

```text
lane 0  -> input[0]
lane 1  -> input[1]
...
lane 31 -> input[31]
```

形成连续/coalesced global loads。

对于：

```text
N=65536
```

每个 thread 大约扫描：

```text
65536 / 256
=
256 elements
```

而不是一个线程扫描整个 65536。

---

# 3. Thread-local Register Top-K

每个 thread：

```cpp
float local_values[16];
int local_indices[16];
```

只保留自己处理元素中的 Top-K。

例如：

```text
thread 7 scans:

x7
x263
x519
...
```

扫描过程中：

```text
new value
   ↓
compare with current local threshold
   ↓
too small → discard

or

insert into local sorted candidate queue
```

因此输入：

```text
N elements
```

只扫描一次。

不是 V0 的：

```text
rank 0 → scan N
rank 1 → scan N
...
rank K-1 → scan N
```

---

# 4. 为什么每个 thread 保留 local K 能保证正确

假设：

```text
global K = 16
```

如果某个元素甚至不在它所属 thread 的 local Top-16 中：

```text
至少已经有 16 个
同一 thread 的元素比它更好
```

那么全局至少也有 16 个元素比它更好。

所以它不可能进入 global Top-16。

因此：

```text
global Top-K
一定包含在
所有 thread local Top-K 的并集里
```

这个结论保证 V1 merge 是 exact Top-K，不是近似算法。

---

# 5. Block-level k-way merge

256 个 thread 扫描结束：

```text
thread 0:
[v00, v01, ... v0K]

thread 1:
[v10, v11, ... v1K]

...

thread 255:
...
```

每个 local list 已经从大到小排序。

我们不需要把：

```text
256 × K
```

个元素完整排序。

而是做一个 k-way merge。

每个 thread 有一个：

```text
cursor
```

最开始：

```text
cursor[tid] = 0
```

每一轮：

```text
256 threads
分别提出当前 local head

        ↓

block reduction

        ↓

选出 global best

        ↓

winner thread cursor++

        ↓

下一轮
```

只需要做 K 次 block reduction。

---

# 6. 算法复杂度变化

V0 大致：

```text
serial scan
+
repeated scan
+
selected-index search
```

非常差。

V1：

```text
Parallel scan:
O(N * K / THREADS)

+

Merge:
O(K * log THREADS)
```

在当前：

```text
THREADS=256
K<=16
```

情况下，block merge 的规模远小于扫描 N。

---

# 7. 当前支持范围

V1 register kernel：

```text
1 <= K <= 16
```

这是刻意的。

因为：

```text
K ↑
local_values[K]
local_indices[K]
```

会增加：

```text
register pressure
local memory spilling
```

Top-K 项目后面很重要的一条主线就是：

```text
small K
→ register strategy

medium K
→ different block/warp strategy

large K
→ radix / bucket strategy
```

所以 V1 不应该假装一个 kernel 能解决所有 K。

---

# 8. 编译

```bash
rm -rf build
mkdir build
cd build

cmake ..
cmake --build . -j

cd ..
```

默认：

```text
SM75
```

---

# 9. 第一条运行命令

直接跑当前 baseline shape：

```bash
./build/topk_bench \
  128 65536 16
```

默认使用：

```text
register_local_topk_block_merge_v1
```

程序输出：

```text
Correctness
Value + index match = PASS

Performance
GPU V1 latency = ...
Normalized input bandwidth = ...
```

---

# 10. 同进程比较 V0

如果想直接看 speedup：

```bash
./build/topk_bench \
  128 65536 16 \
  --compare-naive
```

V0 非常慢，所以程序对 naive baseline 使用较少的 benchmark repeats。

最后会输出：

```text
V0 comparison

Naive latency = ...
V1 speedup over V0 = ...x
```

---

# 11. Force V0

仍然保留 V0 kernel：

```bash
./build/topk_bench \
  128 65536 16 \
  --kernel naive
```

V1：

```bash
./build/topk_bench \
  128 65536 16 \
  --kernel register
```

---

# 12. NCU

```bash
NCU=/usr/local/cuda-13.2/bin/ncu \
./scripts/profile_v0_vs_v1.sh
```

重点比较：

```text
V0:
active lanes 极低
serial memory access
low SM utilization

V1:
active lanes ↑
coalesced global access
SM throughput ↑
memory throughput ↑
```

V1 同时可能出现新的瓶颈：

```text
register pressure

local candidate insertion dependency

__syncthreads

block reduction

shared-memory traffic
```

这就是 V2 要继续解决的问题。

---

# 13. Sweep

```bash
python3 scripts/sweep.py
```

测试：

```text
K = 1 / 4 / 8 / 16
```

以及不同 N。

重点观察：

```text
K ↑
latency 如何变化？
```

因为 thread-local insertion cost 和 block merge cost 都和 K 有关。

---

# 14. V1 后下一步

V1 merge 仍然大量使用：

```text
shared memory
+
__syncthreads
```

每一个 Top-K rank 做：

```text
256-way block reduction
```

V2 要开始利用：

```text
warp shuffle
```

即：

```text
Thread local candidates
        ↓
Warp-level merge
        ↓
Warp Top-K
        ↓
少量 warp results
        ↓
Block final merge
```

目标是减少：

```text
shared-memory communication
+
block-wide barriers
```

开始真正进入：

```text
WarpSelect
```

方向。
