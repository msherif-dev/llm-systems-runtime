cmake_minimum_required(VERSION 3.18)

project(
    GPUMemoryPool
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# Library
# ============================================================

add_library(
    gpu_memory_pool
    memory-pool/src/gpu_memory_pool.cu
)

target_include_directories(
    gpu_memory_pool
    PUBLIC
    ${PROJECT_SOURCE_DIR}/memory-pool/include
    ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(
    gpu_memory_pool
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================
# Tests
# ============================================================

add_executable(
    test_pool
    memory-pool/tests/test_basic.cu
)

target_link_libraries(
    test_pool
    PRIVATE
    gpu_memory_pool
)

set_target_properties(
    test_pool
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================
# Benchmark
# ============================================================

add_executable(
    benchmark_pool
    benchmarks/benchmark_pool.cu
)

target_link_libraries(
    benchmark_pool
    PRIVATE
    gpu_memory_pool
)

set_target_properties(
    benchmark_pool
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)