#include "matmul.cuh"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>


#define CUDA_CHECK(call)                                             \
do {                                                                 \
    cudaError_t err = call;                                          \
    if (err != cudaSuccess) {                                        \
        std::cerr << "CUDA Error: "                                     \
                  << cudaGetErrorString(err)                         \
                  << " at " << __FILE__ << ":" << __LINE__           \
                  << std::endl;                                      \
        std::exit(EXIT_FAILURE);                                     \
    }                                                                \
} while (0)



struct Result
{
    float ms;
    double gflops;
};



void launch_naive(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
)
{
    dim3 block(
        TILE_SIZE,
        TILE_SIZE
    );


    dim3 grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );


    matmul_naive<<<grid, block>>>(
        A,
        B,
        C,
        M,
        N,
        K
    );
}



void launch_tiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
)
{
    dim3 block(
        TILE_SIZE,
        TILE_SIZE
    );


    dim3 grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );


    matmul_tiled<<<grid, block>>>(
        A,
        B,
        C,
        M,
        N,
        K
    );
}



Result benchmark(
    void (*launch)(
        const float*,
        const float*,
        float*,
        int,
        int,
        int
    ),
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
)
{
    constexpr int WARMUP = 10;
    constexpr int ITERATIONS = 100;


    // --------------------------------------------------------
    // Warmup
    // --------------------------------------------------------

    for (int i = 0; i < WARMUP; ++i)
    {
        launch(
            A,
            B,
            C,
            M,
            N,
            K
        );
    }


    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    // --------------------------------------------------------
    // Events
    // --------------------------------------------------------

    cudaEvent_t start;
    cudaEvent_t stop;


    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));


    CUDA_CHECK(cudaEventRecord(start));


    for (int i = 0; i < ITERATIONS; ++i)
    {
        launch(
            A,
            B,
            C,
            M,
            N,
            K
        );
    }


    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));


    float total_ms = 0.0f;


    CUDA_CHECK(
        cudaEventElapsedTime(
            &total_ms,
            start,
            stop
        )
    );


    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));


    float avg_ms =
        total_ms / ITERATIONS;


    // --------------------------------------------------------
    // GFLOPS
    // --------------------------------------------------------

    double operations =
        2.0 *
        static_cast<double>(M) *
        static_cast<double>(N) *
        static_cast<double>(K);


    double seconds =
        avg_ms / 1000.0;


    double gflops =
        operations /
        seconds /
        1e9;


    return {
        avg_ms,
        gflops
    };
}



void run(int size)
{
    int M = size;
    int N = size;
    int K = size;


    size_t bytes_A =
        static_cast<size_t>(M) *
        K *
        sizeof(float);


    size_t bytes_B =
        static_cast<size_t>(K) *
        N *
        sizeof(float);


    size_t bytes_C =
        static_cast<size_t>(M) *
        N *
        sizeof(float);


    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;


    CUDA_CHECK(cudaMalloc(
        &d_A,
        bytes_A
    ));


    CUDA_CHECK(cudaMalloc(
        &d_B,
        bytes_B
    ));


    CUDA_CHECK(cudaMalloc(
        &d_C,
        bytes_C
    ));


    CUDA_CHECK(cudaMemset(
        d_A,
        0,
        bytes_A
    ));


    CUDA_CHECK(cudaMemset(
        d_B,
        0,
        bytes_B
    ));


    CUDA_CHECK(cudaMemset(
        d_C,
        0,
        bytes_C
    ));


    Result naive =
        benchmark(
            launch_naive,
            d_A,
            d_B,
            d_C,
            M,
            N,
            K
        );


    Result tiled =
        benchmark(
            launch_tiled,
            d_A,
            d_B,
            d_C,
            M,
            N,
            K
        );


    // --------------------------------------------------------
    // Comparison
    // --------------------------------------------------------

    double speedup =
        naive.ms /
        tiled.ms;


    double improvement =
        (
            naive.ms -
            tiled.ms
        )
        /
        naive.ms
        * 100.0;


    std::cout << "\n";
    std::cout
        << "============================================\n";

    std::cout
        << "Matrix: "
        << size
        << " x "
        << size
        << "\n";

    std::cout
        << "============================================\n";


    std::cout
        << std::left
        << std::setw(15)
        << "Kernel"
        << std::right
        << std::setw(15)
        << "Time (ms)"
        << std::setw(15)
        << "GFLOPS"
        << "\n";


    std::cout
        << "--------------------------------------------\n";


    std::cout
        << std::left
        << std::setw(15)
        << "Naive"
        << std::right
        << std::setw(15)
        << std::fixed
        << std::setprecision(4)
        << naive.ms
        << std::setw(15)
        << std::setprecision(2)
        << naive.gflops
        << "\n";


    std::cout
        << std::left
        << std::setw(15)
        << "Tiled"
        << std::right
        << std::setw(15)
        << std::fixed
        << std::setprecision(4)
        << tiled.ms
        << std::setw(15)
        << std::setprecision(2)
        << tiled.gflops
        << "\n";


    std::cout
        << "\nTiled Speedup: "
        << std::fixed
        << std::setprecision(2)
        << speedup
        << "x\n";


    std::cout
        << "Tiled Improvement: "
        << std::fixed
        << std::setprecision(2)
        << improvement
        << "%\n";


    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}



int main()
{
    CUDA_CHECK(cudaSetDevice(0));


    std::cout
        << "============================================\n"
        << "CUDA MatMul Benchmark\n"
        << "Naive vs Tiled\n"
        << "============================================\n";


    run(256);
    run(512);
    run(1024);


    return 0;
}