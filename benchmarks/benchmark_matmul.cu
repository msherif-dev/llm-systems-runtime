#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "matmul.cuh"


// Cuda Error Check
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

// Benchmark for one matrix 

void benchmark(
    int M,
    int N,
    int K,
    int warmup_iterations,
    int benchmark_iterations
)
{
    std::cout
        << "\n========================================\n";

    std::cout
        << "Matrix: "
        << M << " x " << K
        << " * "
        << K << " x " << N
        << std::endl;


    // --------------------------------------------------------
    // Allocate device memory
    // --------------------------------------------------------

    size_t size_A =
        static_cast<size_t>(M) *
        K *
        sizeof(float);

    size_t size_B =
        static_cast<size_t>(K) *
        N *
        sizeof(float);

    size_t size_C =
        static_cast<size_t>(M) *
        N *
        sizeof(float);


    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;


    CUDA_CHECK(
        cudaMalloc(
            &d_A,
            size_A
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_B,
            size_B
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_C,
            size_C
        )
    );

    // Initialize GPU memory

    CUDA_CHECK(
        cudaMemset(
            d_A,
            1,
            size_A
        )
    );

    CUDA_CHECK(
        cudaMemset(
            d_B,
            1,
            size_B
        )
    );

    CUDA_CHECK(
        cudaMemset(
            d_C,
            0,
            size_C
        )
    );

    // CONFIGE THE KERNAL 

    dim3 block(16, 16);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );


    std::cout
        << "Grid: "
        << grid.x
        << " x "
        << grid.y
        << std::endl;

    std::cout
        << "Block: "
        << block.x
        << " x "
        << block.y
        << std::endl;

    // Warmup

    for (int i = 0; i < warmup_iterations; ++i)
    {
        matmul_naive<<<grid, block>>>(
            d_A,
            d_B,
            d_C,
            M,
            N,
            K
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

   // CUDA events

   cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(
        cudaEventCreate(&start)
    );

    CUDA_CHECK(
        cudaEventCreate(&stop)
    );

    CUDA_CHECK(
        cudaEventRecord(start)
    );

    for (int i = 0; i < benchmark_iterations; ++i)
    {
        matmul_naive<<<grid, block>>>(
            d_A,
            d_B,
            d_C,
            M,
            N,
            K
        );
    }


    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(
        cudaEventRecord(stop)
    );

    CUDA_CHECK(
        cudaEventSynchronize(stop)
    );

    float total_ms = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &total_ms,
            start,
            stop
        )
    );


    float average_ms =
        total_ms /
        benchmark_iterations;

    
    // Calculate FLOPs

    double operations =
        2.0 *
        static_cast<double>(M) *
        static_cast<double>(N) *
        static_cast<double>(K);


    double seconds =
        average_ms / 1000.0;


    double gflops =
        operations /
        seconds /
        1e9;

    // Print Result 

    std::cout
        << std::fixed
        << std::setprecision(3);

    std::cout
        << "Average kernel time: "
        << average_ms
        << " ms"
        << std::endl;

    std::cout
        << "Performance: "
        << gflops
        << " GFLOPS"
        << std::endl;

    // Cleanup
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaEventDestroy(start)
    );

    CUDA_CHECK(
        cudaEventDestroy(stop)
    );

    CUDA_CHECK(
        cudaFree(d_A)
    );

    CUDA_CHECK(
        cudaFree(d_B)
    );

    CUDA_CHECK(
        cudaFree(d_C)
    );
}

int main()
{
    std::cout
        << "========================================\n"
        << " Naive CUDA MatMul Benchmark\n"
        << "========================================\n";


    constexpr int WARMUP = 10;
    constexpr int ITERATIONS = 100;


    // --------------------------------------------------------
    // 256 x 256
    // --------------------------------------------------------

    benchmark(
        256,
        256,
        256,
        WARMUP,
        ITERATIONS
    );


    // --------------------------------------------------------
    // 512 x 512
    // --------------------------------------------------------

    benchmark(
        512,
        512,
        512,
        WARMUP,
        ITERATIONS
    );


    // --------------------------------------------------------
    // 1024 x 1024
    // --------------------------------------------------------

    benchmark(
        1024,
        1024,
        1024,
        WARMUP,
        ITERATIONS
    );


    std::cout
        << "\n========================================\n"
        << " Benchmark complete\n"
        << "========================================\n";


    return EXIT_SUCCESS;
}