#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                      \
do                                                           \
{                                                            \
    cudaError_t error = call;                                \
    if (error != cudaSuccess)                                \
    {                                                        \
        std::cerr << "CUDA Error: "                         \
                  << cudaGetErrorString(error)              \
                  << " at " << __FILE__                     \
                  << ":" << __LINE__ << std::endl;          \
        return 1;                                            \
    }                                                        \
} while (0)


__global__ void vectorAdd(
    const float* A,
    const float* B,
    float* C,
    int N
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Prevent out-of-bounds access
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

// Computes transfer bandwidth in GB/s given bytes moved and elapsed time in ms
float computeBandwidthGBps(size_t bytes, float ms)
{
    // bytes -> GB, ms -> s
    return (static_cast<float>(bytes) / 1e9f) / (ms / 1000.0f);
}

int main(){

    const int N = 1000000 ; 
    const size_t size = N * sizeof(float) ;

    // host 

    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    for (int i = 0; i < N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // drvice 

    float* d_A;
    float* d_B;
    float* d_C;

    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    //  Cuda Event 

    cudaEvent_t start ; 
    cudaEvent_t stop ;
    
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));



    // copy CPU -->> GPU 

    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpy(
        d_A,
        h_A,
        size,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_B,
        h_B,
        size,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float h2d_ms;

    CUDA_CHECK(cudaEventElapsedTime(
        &h2d_ms,
        start,
        stop
    ));

    // Bandwidth for H2D: two arrays (A and B) were transferred
    float h2d_gbps = computeBandwidthGBps(size * 2, h2d_ms);


    // kERNAL CONFUGURATION 

    int threadsPerBlock = 512;

    int blocksPerGrid =
        (N + threadsPerBlock - 1)
        / threadsPerBlock;

    std::cout << "Blocks: "
              << blocksPerGrid << std::endl;

    std::cout << "Threads per block: "
              << threadsPerBlock << std::endl;      


    // Run Kernal 

    CUDA_CHECK(cudaEventRecord(start));

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(
        d_A,
        d_B,
        d_C,
        N
    );

    // check kernal launch 
    CUDA_CHECK(cudaEventRecord(stop));
    // wait for GPU 
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaGetLastError());

    float kernel_ms;

    CUDA_CHECK(cudaEventElapsedTime(
        &kernel_ms,
        start,
        stop
    ));
    
    // copy GPU -->> CPU 

    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpy(
        h_C,
        d_C,
        size,
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float d2h_ms;

    CUDA_CHECK(cudaEventElapsedTime(
        &d2h_ms,
        start,
        stop
    ));

    // Bandwidth for D2H: one array (C) was transferred back
    float d2h_gbps = computeBandwidthGBps(size, d2h_ms);


    // check Result 
    bool correct = true;

    for (int i = 0; i < N; i++)
    {
        float expected = h_A[i] + h_B[i];

        if (h_C[i] != expected)
        {
            std::cout
                << "Error at index "
                << i
                << std::endl;

            correct = false;
            break;
        }
    }

    if (correct)
    {
        std::cout << "Result: CORRECT" << std::endl;
    }

    // print result 

    float total_ms = h2d_ms + kernel_ms + d2h_ms;

    std::cout << "\n========== CUDA Benchmark ==========\n";

    std::cout << "N: "
              << N
              << '\n';

    std::cout << "Threads per block: "
              << threadsPerBlock
              << '\n';

    std::cout << "Blocks: "
              << blocksPerGrid
              << '\n';

    std::cout << "\n";

    std::cout << "Host -> Device: "
              << h2d_ms
              << " ms  ("
              << h2d_gbps
              << " GB/s)\n";

    std::cout << "Kernel:          "
              << kernel_ms
              << " ms\n";

    std::cout << "Device -> Host: "
              << d2h_ms
              << " ms  ("
              << d2h_gbps
              << " GB/s)\n";

    std::cout << "------------------------------------\n";

    std::cout << "Total:           "
              << total_ms
              << " ms\n";

    std::cout << "\nResult: "
              << (correct ? "CORRECT" : "WRONG")
              << '\n';

    // Free Memory >  >  >  

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
