# Separable Convolution

本目录计划练习二维可分离卷积：把一个二维卷积拆成水平方向和垂直方向的两个一维卷积，以减少计算量，并利用 shared memory 复用包含 halo 的输入 tile。

建议实现顺序：CPU reference、naive CUDA、行卷积 shared-memory tile、列卷积 shared-memory tile，最后加入正确性与性能对比。

当前目录尚无源码，是待实现的练习占位。
