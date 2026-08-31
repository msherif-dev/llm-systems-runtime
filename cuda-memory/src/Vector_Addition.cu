#include <iostream>
#include <cuda_runtime.h>

__global__  void vec_add(const float *A , const float *B , float *C){

    int i = blockIdx.x * blockDim.x + threadIdx.x ;

    C[i] = A[i] + B[i] ;
}

int main(){

    const int N = 10 ; 
    const size_t size = N * sizeof(float) ;

    // host 

    float h_A[N];
    float h_B[N];
    float h_C[N];

    for(int i = 0 ; i < N ; i++){

        h_A[i] = i ;
        h_B[i] = i*2;

    }

    // drvice 

    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc((void**)&d_A , size);
    cudaMalloc((void**)&d_B , size);
    cudaMalloc((void**)&d_C , size);

    // copy CPU -->> GPU 

    cudaMemcpy(d_A ,h_A ,size ,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B ,h_B ,size ,cudaMemcpyHostToDevice);

    // Run Kernal 

    vec_add<<<1 , N>>>(d_A, d_B, d_C) ;
    cudaDeviceSynchronize();

    // copy GPU -->> CPU 

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Show result 
    for (int i = 0; i < N; i++)
    {
        std::cout << h_A[i]
                  << " + "
                  << h_B[i]
                  << " = "
                  << h_C[i]
                  << '\n';}

    // Free Memory 

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0 ;
}