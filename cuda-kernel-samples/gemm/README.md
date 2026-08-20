# GEMM Sample

`gemm.cu` 用于练习行主序矩阵乘 `C[M,N] = A[M,K] × B[K,N]`，文件中包含 naive、block tile 和更细粒度 tile 的实现草稿。

优化路线包括：让相邻线程连续读取 B、把 A/B tile 缓存到 shared memory、让每线程用寄存器累计多个输出，以及循环展开。

当前源码仍处于草稿状态，存在重复宏定义、函数参数分隔符和变量拼写等编译问题，不能直接作为可运行基准。建议先单独修复并验证 `gemm_naive`，再逐个启用 tiled kernel，并使用 CPU reference 检查所有边界尺寸。
