cmake_minimum_required(VERSION 3.20)

project(
    shape_adaptive_gemm_v2
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(
    shape_gemm
    src/main.cu
    src/gemm_kernels.cu
    src/autotuner.cu
    src/cache.cpp
)

target_include_directories(
    shape_gemm
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
    shape_gemm
    PRIVATE
    CUDA::cudart
    CUDA::cublas
)

target_compile_options(
    shape_gemm
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(
    shape_gemm
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
