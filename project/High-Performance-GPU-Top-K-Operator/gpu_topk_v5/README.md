# High-Performance GPU Top-K V5

V5 把前面的 V3/V4 kernel 实验升级成一个真正的：

```text
Shape/K-Adaptive GPU Top-K Library
```

核心结构：

```text
Input Shape
(B, N, K)
    ↓
Kernel Registry
    ↓
Autotuner
    ↓
Performance Cache
    ↓
Runtime Dispatcher
    ↓
Exact Top-K
```

目标环境：

```text
GPU: RTX 2080
Architecture: SM75
dtype: FP32
Current K range: 1..16
```

---

# 1. 为什么需要 V5

V4 sweep 已经证明：

```text
最佳 kernel 不是 K 的简单函数
```

例如实测：

```text
B=128 N=65536 K=1
V4 specialized:
0.316 ms
V3:
0.468 ms
→ V4 wins
```

```text
B=128 N=65536 K=8
V4 specialized:
0.538 ms
V3:
0.476 ms
→ V3 wins
```

甚至：

```text
K=4
```

在不同 N 上会反转：

```text
B=32 N=4096 K=4
→ V3 wins

B=128 N=65536 K=4
→ V4 wins
```

所以不能再硬编码：

```text
K → implementation
```

而应该：

```text
(GPU, B, N, K)
        ↓
benchmark candidate kernels
        ↓
choose fastest
```

---

# 2. Kernel Registry

查看：

```bash
./build/topk_bench --list-kernels
```

V5 Registry 包含：

```text
register_v1
warp_v2

batch_v3

warpselect_k1_v4
warpselect_k2_v4
warpselect_k4_v4
warpselect_k8_v4
warpselect_k16_v4
```

其中：

```text
V1/V2
```

保留用于：

```text
forced benchmark
NCU profiling
historical A/B
```

但默认 autotune 不再测试它们。

原因是：

```text
V1/V2 ≈ 4 ms
V3/V4 ≈ 0.3~0.5 ms
```

继续每次 autotune V1/V2 只会浪费 tuning 时间。

---

# 3. Default Autotune Candidate Set

如果：

```text
K=1
```

候选：

```text
batch_v3
warpselect_k1_v4
```

如果：

```text
K=4
```

候选：

```text
batch_v3
warpselect_k4_v4
```

如果：

```text
K=8
```

候选：

```text
batch_v3
warpselect_k8_v4
```

如果 K 不是：

```text
1/2/4/8/16
```

但仍：

```text
K <= 16
```

则目前只有：

```text
batch_v3
```

作为高性能 exact fallback。

---

# 4. Cache Key

V5 cache：

```text
results/topk_cache_v5.csv
```

key：

```text
GPU name
Compute Capability
dtype
B
N
K
```

value：

```text
best kernel
latency
GElem/s
```

例如：

```text
RTX2080,7,5,fp32,128,65536,4,warpselect_k4_v4,...
RTX2080,7,5,fp32,128,65536,8,batch_v3,...
```

---

# 5. 第一次运行

编译：

```bash
rm -rf build
mkdir build
cd build

cmake ..
cmake --build . -j

cd ..
```

然后：

```bash
./build/topk_bench \
  128 65536 4 \
  --retune
```

应该看到：

```text
Mode: cache miss -> autotune

Autotuning GPU Top-K V5

batch_v3
warpselect_k4_v4

Best Top-K kernel:
...
```

然后保存：

```text
results/topk_cache_v5.csv
```

---

# 6. 第二次运行

直接：

```bash
./build/topk_bench \
  128 65536 4
```

应该：

```text
Mode: cache hit

Cached kernel:
...

Selected kernel:
...
```

不再重新 autotune。

---

# 7. Force Kernel

例如：

```bash
./build/topk_bench \
  128 65536 8 \
  --kernel batch_v3
```

或者：

```bash
./build/topk_bench \
  128 65536 8 \
  --kernel warpselect_k8_v4
```

历史版本也可以：

```bash
./build/topk_bench \
  128 65536 16 \
  --kernel warp_v2
```

---

# 8. Profiling

单 launch：

```bash
NCU=/usr/local/cuda-13.2/bin/ncu \
KERNEL=batch_v3 \
K=8 \
./scripts/profile_selected.sh
```

或者：

```bash
NCU=/usr/local/cuda-13.2/bin/ncu \
KERNEL=warpselect_k4_v4 \
K=4 \
./scripts/profile_selected.sh
```

---

# 9. Sweep + Fill Cache

```bash
python3 scripts/sweep.py
```

它会对多组：

```text
(B,N,K)
```

执行：

```text
--retune
```

并填充：

```text
results/topk_cache_v5.csv
```

如果还想紧接着验证 cache hit：

```bash
python3 scripts/sweep.py \
  --cache-hit-pass
```

---

# 10. Current Project Evolution

实测核心路径：

```text
V0
serial repeated selection
525.86 ms

↓

V1
parallel scan
+ per-thread register Top-K
4.12 ms

↓

V2
warp merge
4.09 ms
only ~0.8% improvement

↓

re-diagnose bottleneck

↓

V3
warp-cooperative batch selection
+ threshold reject
+ bitonic merge
0.476 ms

↓

V4
K-specialized WarpSelect family
different K/N produce different winners

↓

V5
Registry
+ Autotuner
+ Performance Cache
+ Runtime Dispatcher
```

---

# 11. V5 项目 Thesis

V5 最重要的结论是：

```text
Top-K performance
is a function of:

N
K
batch
GPU architecture
```

不是：

```text
one algorithm fits all
```

因此最终架构：

```text
Top-K Request
(B,N,K)
    ↓
Performance DB lookup
    ↓
cache hit?
   /     \
 yes      no
  ↓        ↓
dispatch   autotune
            ↓
          save DB
            ↓
          dispatch
```

这和成熟算子库的核心思想一致：

```text
多个 kernel family
+
shape-aware selection
```

---

# 12. 当前最值得跑的命令

先：

```bash
./build/topk_bench \
  128 65536 1 \
  --retune
```

再：

```bash
./build/topk_bench \
  128 65536 4 \
  --retune
```

再：

```bash
./build/topk_bench \
  128 65536 8 \
  --retune
```

预期根据此前 V4 数据：

```text
K=1
→ likely warpselect_k1_v4

K=4
→ likely warpselect_k4_v4

K=8
→ likely batch_v3
```

但 V5 不硬编码这个结论。

它会真正 benchmark 后选择。

---

# 13. 下一阶段

V5 完成后，项目已经是一个 small-K Top-K library。

下一阶段再考虑：

```text
K > 16
```

这时不应该继续无限扩 register/bitonic queue。

更合理的是新增：

```text
radix selection
bucket / threshold selection
hierarchical large-K family
```

然后它们继续挂进同一个：

```text
Registry
Autotuner
Dispatcher
```

框架里。
