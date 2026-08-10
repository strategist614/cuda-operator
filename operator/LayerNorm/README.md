## LayerNorm

### 安装 nsys
```
apt update
apt install -y --no-install-recommends gnupg lsb-release ca-certificates wget software-properties-common

echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" \
  > /etc/apt/sources.list.d/nvidia-devtools.list

apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

apt update
apt install -y nsight-systems-cli
```

```
输入矩阵
4096 rows × 1024 cols

             GPU Grid
────────────────────────────────

Block 0     → 负责 row 0
Block 1     → 负责 row 1
Block 2     → 负责 row 2
...
Block 4095  → 负责 row 4095


每一个 Block:

256 threads
│
├─ thread 0   → col 0,256,512,768
├─ thread 1   → col 1,257,513,769
├─ thread 2   → col 2,258,514,770
│
│      ...
│
└─ thread 255 → col 255,511,767,1023
```

### 优化思路

* 在 `reduction` 那里 每次都需要 `__syncthreads()` 有很多次 `block synchronization`
  所以优化思路是 `warp` 内不需要 `shared memory reduction`
  我现在是 `256` 个 `threads` 一个 `warp` 有 `32` 个 `threads` 所以有 `8` 个 `wraps` 
  `__shfl_down_sync()` 在 `wrap` 中能直接做加法。
  
  ```c++
  // warp 的 thread 0 得到 32 个线程的总和
  __inline__ __device__
  float warp_reduce_sum(float val)
  {
      for (int offset = 16; offset > 0; offset >>= 1) {
          val += __shfl_down_sync(0xffffffff, val, offset);
      }

      return val;
  }
  ```
  这些数据相加的操作主要是在寄存器之间交换

* 在 `V1` 的 `warp shuffle reduction` 基础上，让每个线程把自己负责的 `4` 个 `x` 一开始读进寄存器，后面 `mean、variance、normalize` 都复用，不再重复访问 `x_row`
  ```c++
    __global__ void layernorm_v2_kernel(
      const float* __restrict__ x,
      const float* __restrict__ gamma,
      const float* __restrict__ beta,
      float* __restrict__ y,
      int rows,
      int cols,
      float eps
  ) {
      extern __shared__ float shared[];

      int row = blockIdx.x;
      int tid = threadIdx.x;

      if (row >= rows) {
          return;
      }

      const float* x_row =
          x + static_cast<size_t>(row) * cols;

      float* y_row =
          y + static_cast<size_t>(row) * cols;

      // ========================================================
      // V2 核心优化：
      // 每个线程一次性把自己负责的 4 个 x 读进寄存器
      // ========================================================

      float v0 = x_row[tid];
      float v1 = x_row[tid + 256];
      float v2 = x_row[tid + 512];
      float v3 = x_row[tid + 768];


      // ========================================================
      // Step 1: mean
      // 不再读取 x_row
      // ========================================================

      float sum =
          v0 + v1 + v2 + v3;

      float total_sum =
          block_reduce_sum(sum, shared);

      float mean =
          total_sum / static_cast<float>(cols);


      // ========================================================
      // Step 2: variance
      // 继续使用寄存器里的 v0~v3
      // ========================================================

      float d0 = v0 - mean;
      float d1 = v1 - mean;
      float d2 = v2 - mean;
      float d3 = v3 - mean;

      float var_sum =
          d0 * d0 +
          d1 * d1 +
          d2 * d2 +
          d3 * d3;

      float total_var_sum =
          block_reduce_sum(var_sum, shared);

      float var =
          total_var_sum / static_cast<float>(cols);

      float rstd =
          rsqrtf(var + eps);


      // ========================================================
      // Step 3: normalize + affine
      // 仍然使用 v0~v3
      // ========================================================

      int i0 = tid;
      int i1 = tid + 256;
      int i2 = tid + 512;
      int i3 = tid + 768;

      y_row[i0] =
          (v0 - mean) * rstd * gamma[i0] + beta[i0];

      y_row[i1] =
          (v1 - mean) * rstd * gamma[i1] + beta[i1];

      y_row[i2] =
          (v2 - mean) * rstd * gamma[i2] + beta[i2];

      y_row[i3] =
          (v3 - mean) * rstd * gamma[i3] + beta[i3];
  }
  ```
  这里如果存的数字太多了的话 会导致寄存器不够用 所以不能无止境的取用
* 现在有一种方法使得不要整体得到 `mean` 就能计算 `variance` 就是 `welford` 方法，它的方法是动态维护一个状态
  每次有新的数字加入就进行计算 
* 每次读入的是四个 `x_row` 可以用 `float4` 一次就读入 `4` 个浮点数