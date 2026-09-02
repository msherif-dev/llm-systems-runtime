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

int main(){

    const int N = 1000 ; 
    const size_t size = N * sizeof(float) ;

    // host 

    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    for(int i = 0 ; i < N ; i++){

        h_A[i] =static_cast<float>(i) ;
        h_B[i] = static_cast<float>(i * 2);

    }

    // drvice 

    float* d_A;
    float* d_B;
    float* d_C;

    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));


    // copy CPU -->> GPU 

    CUDA_CHECK(
        cudaMemcpy(
            d_A,
            h_A,
            size,
            cudaMemcpyHostToDevice
        )
    );
    CUDA_CHECK(
        cudaMemcpy(
            d_B ,
            h_B ,
            size ,
            cudaMemcpyHostToDevice
        )
    );

    // kERNAL CONFUGURATION 

    int threadsPerBlock = 256;

    int blocksPerGrid =
        (N + threadsPerBlock - 1)
        / threadsPerBlock;

    std::cout << "Blocks: "
              << blocksPerGrid << std::endl;

    std::cout << "Threads per block: "
              << threadsPerBlock << std::endl;      


    // Run Kernal 

    vectorAdd<<<blocksPerGrid,threadsPerBlock>>>(
        d_A,
        d_B,
        d_C,
        N
    );

    // Check kernel launch
    CUDA_CHECK(cudaGetLastError());

    // Wait for GPU
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // copy GPU -->> CPU 

    CUDA_CHECK(
        cudaMemcpy(
            h_C,
            d_C,
            size,
            cudaMemcpyDeviceToHost
        )
    );


    // Result 
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

    // Free device Memory 

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}