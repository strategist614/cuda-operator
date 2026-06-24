## transpose

#### CUDA内存
```c++
threads(TILE_DIM, BLOCK_ROWS)
```
这里是 `threads(32, 16)`
表示 `threadIdx.y` 是固定的 然后
`threadIdx.x = 0,1,2..31`，所以`xIndex`连续在变化，`yIndex`保持不变。这个也符合行优先的规则

> 同一个 warp 里面，这些线程在同一条写指令上，要写的地址是不是挨在一起。只要目标地址是连续的，那对于 global memory 来说，他就是最快的

#### tile转置
确定最后输出的位置，保证将`tile`里面的值连续的写入`global memory`里面。例如：
```text
A:
a00  a01  a02  a03
a10  a11  a12  a13
a20  a21  a22  a23
a30  a31  a32  a33

B:

a00  a10  a20  a30
a01  a11  a21  a31
a02  a12  a22  a32
a03  a13  a23  a33

分成4个tile
左下角的tile:
A10:
a20  a21
a30  a31

在A的位置：
blockIdx.x = 0
blockIdx.y = 1
转置后 去B的右上角：
B 右上角:

a20  a30
a21  a31
所以这个tile有四个线程：
T00: threadIdx.x = 0, threadIdx.y = 0
T10: threadIdx.x = 1, threadIdx.y = 0

T01: threadIdx.x = 0, threadIdx.y = 1
T11: threadIdx.x = 1, threadIdx.y = 1

thread 布局:

T00  T10
T01  T11

读入：
T00 读 A[2][0] = a20，放到 tile[0][0]
T10 读 A[2][1] = a21，放到 tile[0][1]

T01 读 A[3][0] = a30，放到 tile[1][0]
T11 读 A[3][1] = a31，放到 tile[1][1]

矩阵是按照行优先去存储的，A[2][0]，A[2][1]在内存是连续的。

输出：
输出位置计算：
xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
yIndex = blockIdx.x * TILE_DIM + threadIdx.y;

当前tile：
blockIdx.x = 0
blockIdx.y = 1
TILE_DIM = 2

xIndex = 1 * 2 + threadIdx.x = 2 + threadIdx.x
yIndex = 0 * 2 + threadIdx.y = threadIdx.y

T00: threadIdx.x=0, threadIdx.y=0
xIndex = 2 + 0 = 2
yIndex = 0
写 B[0][2]

T10: threadIdx.x=1, threadIdx.y=0
xIndex = 2 + 1 = 3
yIndex = 0
写 B[0][3]
所以T00，T10写的地址是B[0][2],B[0][3]，属于同一行相邻位置，连续性的写入。

线程      前面读入 tile                 后面写到 B      后面从 tile 取值

T00       tile[0][0] = a20              B[0][2]         tile[0][0] = a20
T10       tile[0][1] = a21              B[0][3]         tile[1][0] = a30

T01       tile[1][0] = a30              B[1][2]         tile[0][1] = a21
T11       tile[1][1] = a31              B[1][3]         tile[1][1] = a31
```

硬件级视角：
```text
一个 warp 执行同一条 store 指令
warp 里的 32 个 lane 同时给出 32 个地址
硬件检查这些地址能不能合并
```
#### tile内部转置
完成真正的转置过程