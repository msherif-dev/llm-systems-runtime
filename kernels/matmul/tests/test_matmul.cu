#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "matmul.cuh"


#define CUDA_CHECK(call)                                             \
do                                                                    \
{                                                                     \
    cudaError_t error = (call);                                       \
                                                                      \
    if (error != cudaSuccess)                                         \
    {                                                                 \
        std::cerr                                                     \
            << "CUDA error: "                                         \
            << cudaGetErrorString(error)                              \
            << " at "                                                  \
            << __FILE__                                                \
            << ":"                                                      \
            << __LINE__                                                \
            << std::endl;                                             \
        std::exit(EXIT_FAILURE);                                      \
    }                                                                 \
} while (0)


void matmul_cpu(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M,
    int N,
    int K
)
{
    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float sum = 0.0f;

            for (int i = 0; i < K; ++i)
            {
                sum +=
                    A[row * K + i] *
                    B[i * N + col];
            }

            C[row * N + col] = sum;
        }
    }
}


bool compare(
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    float tolerance = 1e-4f
)
{
    if (expected.size() != actual.size())
        return false;

    for (size_t i = 0; i < expected.size(); ++i)
    {
        float diff =
            std::fabs(
                expected[i] -
                actual[i]
            );

        if (diff > tolerance)
        {
            std::cerr
                << "Mismatch at "
                << i
                << ": expected="
                << expected[i]
                << ", actual="
                << actual[i]
                << ", diff="
                << diff
                << std::endl;

            return false;
        }
    }

    return true;
}


void run_test(
    int M,
    int N,
    int K
)
{
    std::cout
        << "\nTesting "
        << M << "x" << K
        << " * "
        << K << "x" << N
        << std::endl;


    size_t count_A =
        static_cast<size_t>(M) * K;

    size_t count_B =
        static_cast<size_t>(K) * N;

    size_t count_C =
        static_cast<size_t>(M) * N;


    size_t bytes_A =
        count_A * sizeof(float);

    size_t bytes_B =
        count_B * sizeof(float);

    size_t bytes_C =
        count_C * sizeof(float);


    std::vector<float> h_A(count_A);
    std::vector<float> h_B(count_B);

    std::vector<float> h_reference(
        count_C,
        0.0f
    );

    std::vector<float> h_naive(
        count_C,
        0.0f
    );

    std::vector<float> h_tiled(
        count_C,
        0.0f
    );


    // --------------------------------------------------------
    // Initialize deterministic data
    // --------------------------------------------------------

    for (size_t i = 0; i < count_A; ++i)
    {
        h_A[i] =
            static_cast<float>(
                (i % 7) + 1
            );
    }

    for (size_t i = 0; i < count_B; ++i)
    {
        h_B[i] =
            static_cast<float>(
                (i % 5) + 1
            );
    }


    // --------------------------------------------------------
    // CPU reference
    // --------------------------------------------------------

    matmul_cpu(
        h_A,
        h_B,
        h_reference,
        M,
        N,
        K
    );


    // --------------------------------------------------------
    // Device memory
    // --------------------------------------------------------

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C_naive = nullptr;
    float* d_C_tiled = nullptr;


    CUDA_CHECK(
        cudaMalloc(
            &d_A,
            bytes_A
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_B,
            bytes_B
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_C_naive,
            bytes_C
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_C_tiled,
            bytes_C
        )
    );


    // --------------------------------------------------------
    // Copy input
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            d_A,
            h_A.data(),
            bytes_A,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_B,
            h_B.data(),
            bytes_B,
            cudaMemcpyHostToDevice
        )
    );


    // --------------------------------------------------------
    // Configuration
    // --------------------------------------------------------

    dim3 block(
        TILE_SIZE,
        TILE_SIZE
    );

    dim3 grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );


    // --------------------------------------------------------
    // Naive
    // --------------------------------------------------------

    matmul_naive<<<grid, block>>>(
        d_A,
        d_B,
        d_C_naive,
        M,
        N,
        K
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    // --------------------------------------------------------
    // Tiled
    // --------------------------------------------------------

    matmul_tiled<<<grid, block>>>(
        d_A,
        d_B,
        d_C_tiled,
        M,
        N,
        K
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    // --------------------------------------------------------
    // Copy results
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            h_naive.data(),
            d_C_naive,
            bytes_C,
            cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            h_tiled.data(),
            d_C_tiled,
            bytes_C,
            cudaMemcpyDeviceToHost
        )
    );


    // --------------------------------------------------------
    // Validate Naive
    // --------------------------------------------------------

    if (!compare(h_reference, h_naive))
    {
        std::cerr
            << "Naive kernel FAILED"
            << std::endl;

        std::exit(EXIT_FAILURE);
    }

    std::cout
        << "Naive: PASS"
        << std::endl;


    // --------------------------------------------------------
    // Validate Tiled
    // --------------------------------------------------------

    if (!compare(h_reference, h_tiled))
    {
        std::cerr
            << "Tiled kernel FAILED"
            << std::endl;

        std::exit(EXIT_FAILURE);
    }

    std::cout
        << "Tiled: PASS"
        << std::endl;


    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_naive));
    CUDA_CHECK(cudaFree(d_C_tiled));
}


int main()
{
    std::cout
        << "========================================\n"
        << " CUDA MatMul Correctness Test\n"
        << "========================================\n";


    // Square
    run_test(
        256,
        256,
        256
    );


    // Non-square
    run_test(
        512,
        256,
        1024
    );


    // Non-square
    run_test(
        513,
        257,
        777
    );


    // Small edge case
    run_test(
        17,
        19,
        23
    );


    std::cout
        << "\n========================================\n"
        << " All tests passed!\n"
        << "========================================\n";


    return EXIT_SUCCESS;
}