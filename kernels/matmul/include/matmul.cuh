#pragma once

#include <cuda_runtime.h>

constexpr int TILE_SIZE = 16;

__global__
void matmul_naive(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
);

__global__
void matmul_tiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
);