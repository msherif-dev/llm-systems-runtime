#include "matmul.cuh"

__global__ void matmul_naive(

    const float* A ,
    const float* B ,
    float* C ,
    int M,
    int N,
    int K
){
    int row = blockIdx.y * blockDim.y + threadIdx.y ; 
    int col  = blockIdx.x * blockDim.x + threadIdx.x ; 

    if (row >= M || col >= N){
        return;
    }

    float sum = 0.0f;

    for (int i = 0; i < K; ++i)
    {
        float a = A[row * K + i];
        float b = B[i * N + col];

        sum += a * b;
    }


    C[row * N + col] = sum;
    
}

