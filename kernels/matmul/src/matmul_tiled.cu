#include "matmul.cuh"


__global__
void matmul_tiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
)
{


    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row =
        blockIdx.y * TILE_SIZE +
        threadIdx.y;

    int col =
        blockIdx.x * TILE_SIZE +
        threadIdx.x;

    float sum = 0.0f;


    int num_tiles =
        (K + TILE_SIZE - 1) / TILE_SIZE;


    for (int tile = 0; tile < num_tiles; ++tile)
    {

        int A_col =
            tile * TILE_SIZE +
            threadIdx.x;

        int B_row =
            tile * TILE_SIZE +
            threadIdx.y;


        if (row < M && A_col < K)
        {
            tile_A[threadIdx.y][threadIdx.x] =
                A[row * K + A_col];
        }
        else
        {
            tile_A[threadIdx.y][threadIdx.x] =
                0.0f;
        }


        if (B_row < K && col < N)
        {
            tile_B[threadIdx.y][threadIdx.x] =
                B[B_row * N + col];
        }
        else
        {
            tile_B[threadIdx.y][threadIdx.x] =
                0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
        {
            sum +=
                tile_A[threadIdx.y][i] *
                tile_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}