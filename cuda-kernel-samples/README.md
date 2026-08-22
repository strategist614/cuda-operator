# CUDA Kernel Samples

本目录包含一组可独立编译、运行的 CUDA kernel 学习示例，覆盖逐元素计算、归约、矩阵乘法和矩阵转置。大多数 `.cu` 文件自带 `main()`、输入构造以及基础的结果输出或正确性验证，适合按单文件阅读和实验。

## 目录概览

| 目录 | 示例 | 主要学习内容 |
| --- | --- | --- |
| [`elementwise/`](elementwise/README.md) | Add、ReLU、Sigmoid 的标量版与 `float4` 向量版 | 线程索引、逐元素计算、向量化访存和尾部处理 |
| [`reduce/`](reduce/README.md) | Sum、warp shuffle reduction、Softmax | 线程内归约、warp shuffle、跨 warp 合并和数值稳定性 |
| [`gemm/`](gemm/README.md) | Naive、shared-memory tile、warp/register tile、WMMA | GEMM 分块、数据复用、寄存器累加和 Tensor Core |
| [`transpose/`](transpose/README.md) | Naive 矩阵转置 | 二维 grid/block、行列索引与非合并写入基线 |

GEMM 示例当前包含 4 个完整程序，优化路线为：

```text
Naive FP32
    -> Shared-memory tiling FP32
    -> Warp/register tiling FP32
    -> WMMA Tensor Core（FP16 输入、FP32 累加）
```

四个 GEMM kernel 已在 `512 x 512 x 512` 矩阵上通过各自的 CPU reference 验证。具体实现参数、误差、运行时间和 GFLOPS 见 [`gemm/README.md`](gemm/README.md)。

## 环境要求

- 支持 CUDA 的 NVIDIA GPU。
- CUDA Toolkit，以及可直接调用的 `nvcc`。
- WMMA 示例需要支持 Tensor Core 的 GPU，并使用与实际 GPU compute capability 对应的 `-arch=sm_XX`。

当前 GEMM 源码中的示例编译目标是 `sm_75`。其他架构应调整该参数，例如使用 `nvidia-smi` 确认 GPU 型号后查询对应的 compute capability。

## 编译与运行

在 `cuda-kernel-samples` 目录中，可以单独编译一个基础示例：

```bash
nvcc -O3 -std=c++17 elementwise/add.cu -o elementwise/add
./elementwise/add
```

编译并运行 4 个 GEMM 示例：

```bash
nvcc -O3 -std=c++17 -arch=sm_75 gemm/01_gemm_naive.cu -o gemm/01_gemm_naive
nvcc -O3 -std=c++17 -arch=sm_75 gemm/02_gemm_shared_tile.cu -o gemm/02_gemm_shared_tile
nvcc -O3 -std=c++17 -arch=sm_75 gemm/03_gemm_register_tile.cu -o gemm/03_gemm_register_tile
nvcc -O3 -std=c++17 -arch=sm_75 gemm/04_gemm_wmma.cu -o gemm/04_gemm_wmma

./gemm/01_gemm_naive
./gemm/02_gemm_shared_tile
./gemm/03_gemm_register_tile
./gemm/04_gemm_wmma
```

其他示例也可以使用相同方式单独编译。需要调试时，可去掉 `-O3` 并加入 `-lineinfo`；需要检查运行时错误或访存问题时，可配合 NVIDIA Compute Sanitizer。

## 阅读与实验建议

建议按照以下顺序阅读：

1. 从 `elementwise/add.cu` 熟悉线程索引和边界判断。
2. 通过 `float4` 版本观察向量化访存及对齐要求。
3. 阅读 reduction 示例，理解线程内、warp 内和 block 内的数据合并。
4. 对比 4 个 GEMM 版本，观察 shared memory、寄存器分块和 Tensor Core 带来的变化。
5. 最后从 naive transpose 延伸到 shared-memory tiled transpose。

这些代码以学习和实验为目的，不是生产级算子库。修改输入尺寸、block 配置或 tile 参数后，应重新检查边界条件、数值误差、shared-memory/寄存器占用以及 kernel launch 错误。各示例的具体限制见对应子目录 README。

目录中无扩展名的同名文件通常是本地编译产物；重新编译会覆盖它们。
