#include <cuda_runtime.h>

#include <iostream>
using namespace std;

int main(){

    void* ptr = nullptr ;

    size_t size = 100 * 1024 * 1024 ;

    cudaError_t error = cudaMalloc(&ptr , size) ;

    if (error != cudaSuccess){

        cerr << "cudaMalloc failed: "
             << cudaGetErrorString(error)
             << "\n" ;

        return 1 ; 

    }

    std::cout << "GPU memory allocated successfully\n";
    std::cout << "Pointer: " << ptr << '\n';
    std::cout << "Size: " << size << " bytes\n";

    cudaFree(ptr);

    return 0;
}