#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "matmul.cuh"


// ============================================================
// CUDA error checking
// ============================================================

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
                                                                      \
        std::exit(EXIT_FAILURE);                                      \
    }                                                                 \
} while (0)


// ============================================================
// CPU reference implementation
// ============================================================

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


// ============================================================
// Compare matrices
// ============================================================

bool compare_matrices(
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    float tolerance = 1e-4f
)
{
    if (expected.size() != actual.size())
        return false;

    for (size_t i = 0; i < expected.size(); ++i)
    {
        float difference =
            std::fabs(expected[i] - actual[i]);

        if (difference > tolerance)
        {
            std::cerr
                << "Mismatch at index "
                << i
                << ": expected="
                << expected[i]
                << ", actual="
                << actual[i]
                << ", difference="
                << difference
                << std::endl;

            return false;
        }
    }

    return true;
}


// ============================================================
// Print matrix
// ============================================================

void print_matrix(
    const std::vector<float>& matrix,
    int rows,
    int cols
)
{
    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            std::cout
                << matrix[row * cols + col]
                << " ";
        }

        std::cout << '\n';
    }
}


// ============================================================
// Test
// ============================================================

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


    // --------------------------------------------------------
    // Host matrices
    // --------------------------------------------------------

    size_t size_A =
        static_cast<size_t>(M) * K;

    size_t size_B =
        static_cast<size_t>(K) * N;

    size_t size_C =
        static_cast<size_t>(M) * N;


    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    std::vector<float> h_C(size_C, 0.0f);

    std::vector<float> h_reference(size_C, 0.0f);


    // --------------------------------------------------------
    // Initialize data
    // --------------------------------------------------------

    for (size_t i = 0; i < size_A; ++i)
    {
        h_A[i] =
            static_cast<float>((i % 7) + 1);
    }

    for (size_t i = 0; i < size_B; ++i)
    {
        h_B[i] =
            static_cast<float>((i % 5) + 1);
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
    float* d_C = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_A,
            size_A * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_B,
            size_B * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_C,
            size_C * sizeof(float)
        )
    );


    // --------------------------------------------------------
    // Copy Host -> Device
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            d_A,
            h_A.data(),
            size_A * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_B,
            h_B.data(),
            size_B * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    // --------------------------------------------------------
    // Configure CUDA execution
    // --------------------------------------------------------

    dim3 block(16, 16);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );


    std::cout
        << "Block: "
        << block.x
        << "x"
        << block.y
        << std::endl;

    std::cout
        << "Grid: "
        << grid.x
        << "x"
        << grid.y
        << std::endl;


    // --------------------------------------------------------
    // Launch kernel
    // --------------------------------------------------------

    matmul_naive<<<grid, block>>>(
        d_A,
        d_B,
        d_C,
        M,
        N,
        K
    );


    // --------------------------------------------------------
    // Check kernel launch
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaGetLastError()
    );


    // --------------------------------------------------------
    // Wait for kernel to finish
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    // --------------------------------------------------------
    // Copy Device -> Host
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            h_C.data(),
            d_C,
            size_C * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    // --------------------------------------------------------
    // Validate
    // --------------------------------------------------------

    bool passed =
        compare_matrices(
            h_reference,
            h_C
        );


    if (passed)
    {
        std::cout
            << "PASS"
            << std::endl;
    }
    else
    {
        std::cout
            << "FAIL"
            << std::endl;

        std::cout
            << "\nExpected:\n";

        print_matrix(
            h_reference,
            M,
            N
        );

        std::cout
            << "\nActual:\n";

        print_matrix(
            h_C,
            M,
            N
        );

        std::exit(EXIT_FAILURE);
    }


    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}


// ============================================================
// Main
// ============================================================

int main()
{
    std::cout
        << "========================================\n"
        << " Naive CUDA Matrix Multiplication Test\n"
        << "========================================\n";


    // Small square matrix
    run_test(
        4,
        4,
        4
    );


    // Non-square matrix
    //
    // A = 4 x 3
    // B = 3 x 2
    // C = 4 x 2

    run_test(
        4,
        2,
        3
    );


    // Another non-square case

    run_test(
        7,
        5,
        11
    );


    std::cout
        << "\n========================================\n"
        << " All tests passed!\n"
        << "========================================\n";

    return EXIT_SUCCESS;
}