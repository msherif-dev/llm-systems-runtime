#pragma once

#include <cuda_runtime.h>

// Naive matrix multiplication:
//
// A: M x K
// B: K x N
// C: M x N
//
// Each CUDA thread computes exactly one element C[row][col].
__global__
void matmul_naive(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
);