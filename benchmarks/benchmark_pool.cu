#include "gpu_memory_pool.cuh"

#include <cuda_runtime.h>

#include <string>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>


// configuration 

constexpr std::size_t NUM_ALLOCATIONS = 1000 ; 
constexpr std::size_t ALLOCATION_SIZE = 1024 * 64 ; // 64 KB
constexpr std::size_t TOTAL_SIZE = NUM_ALLOCATIONS * ALLOCATION_SIZE  ; // 64MB


// Handel Cuda Error 

void check_cuda (
    cudaError_t result ,
    const char* operation 
){
    if (result != cudaSuccess){
        throw std::runtime_error(
            std::string(operation)
            + ": "
            + cudaGetErrorString(result)
        );
    }
}

// warmup_cuda 

void warmup_cuda(){

    void* ptr = nullptr ; 

    check_cuda(
        cudaMalloc(&ptr , 1024 ) , 
        "Warmup cudaMalloc"
    );

    check_cuda(
        cudaFree(ptr) , 
        "Warmup cudaFree"
    );

    check_cuda(
        cudaDeviceSynchronize(),
        "Warmup cudaDeviceSynchronize "
    ); 
}

// Benchmark result

struct BenchmarkResult
{
    double total_ms ;

    double average_us ;

    std::size_t cuda_malloc_calls;

    std::size_t cuda_free_calls;

    std::size_t allocations;
};


// Benchmark A
// Direct cudaMalloc

BenchmarkResult benchmark_direct() {
    std::vector<void*> pointers ;
    pointers.reserve(NUM_ALLOCATIONS);

    auto start =  std::chrono::steady_clock::now();

    // allcator 
    for(std::size_t i = 0 ; i < NUM_ALLOCATIONS ; ++i){
        void* ptr = nullptr ;
        check_cuda(cudaMalloc(&ptr , ALLOCATION_SIZE) , "cuadMalloc");
        pointers.push_back(ptr) ;
    }

    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // free
    for (void* ptr : pointers) {
        check_cuda(cudaFree(ptr), "cudaFree");
    }

    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    auto end = std::chrono::steady_clock::now();

    const double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    const double average_us = (total_ms * 1000.0) / static_cast<double>(NUM_ALLOCATIONS);

    return { total_ms, average_us, NUM_ALLOCATIONS, NUM_ALLOCATIONS, NUM_ALLOCATIONS };

}

// benchmark B

BenchmarkResult benchmark_pool(){
    GPUMemoryPool pool(256);
    pool.init(TOTAL_SIZE);

    std::vector<void*> pointers;
    pointers.reserve(NUM_ALLOCATIONS);

    auto start = std::chrono::steady_clock::now();

    // Allocate from pool
    for (std::size_t i = 0; i < NUM_ALLOCATIONS; ++i) {
        void* ptr = pool.allocate(ALLOCATION_SIZE);
        pointers.push_back(ptr);
    }

    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    auto end = std::chrono::steady_clock::now();

    const double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    const double average_us = (total_ms * 1000.0) / static_cast<double>(NUM_ALLOCATIONS);

    pool.destroy();

    return { total_ms, average_us, 1, 1, NUM_ALLOCATIONS };
}


// Print result

void print_result(
    const char* name,
    const BenchmarkResult& result
) {

    std::cout << "\n";
    std::cout << "----------------------------------------\n";
    std::cout << name << "\n";
    std::cout << "----------------------------------------\n";

    std::cout
        << "Allocations       : "
        << result.allocations
        << "\n";

    std::cout
        << "cudaMalloc calls  : "
        << result.cuda_malloc_calls
        << "\n";

    std::cout
        << "cudaFree calls    : "
        << result.cuda_free_calls
        << "\n";

    std::cout
        << std::fixed
        << std::setprecision(3);

    std::cout
        << "Total time        : "
        << result.total_ms
        << " ms\n";

    std::cout
        << "Average allocation: "
        << result.average_us
        << " us\n";
}

// Main

int main() {

    try {

        std::cout << "\n";
        std::cout
            << "========================================\n";

        std::cout
            << " GPU Memory Pool Benchmark\n";

        std::cout
            << "========================================\n";


        std::cout << "\nConfiguration:\n";

        std::cout
            << "Allocations : "
            << NUM_ALLOCATIONS
            << "\n";

        std::cout
            << "Allocation  : "
            << ALLOCATION_SIZE / 1024
            << " KB\n";

        std::cout
            << "Pool size   : "
            << TOTAL_SIZE / (1024 * 1024)
            << " MB\n";


        // Warm-up

        warmup_cuda();


        // Run benchmarks

        const auto direct =
            benchmark_direct();

        const auto pool =
            benchmark_pool();


        // Print results

        print_result(
            "Direct cudaMalloc",
            direct
        );

        print_result(
            "Memory Pool",
            pool
        );


        // Speedup
        const double speedup =
            direct.total_ms
            / pool.total_ms;


        std::cout << "\n";
        std::cout
            << "========================================\n";

        std::cout
            << " Performance Summary\n";

        std::cout
            << "========================================\n";


        std::cout
            << std::fixed
            << std::setprecision(2);


        std::cout
            << "Direct / Pool speedup : "
            << speedup
            << "x\n";


        std::cout
            << "cudaMalloc reduction  : "
            << direct.cuda_malloc_calls
            << " -> "
            << pool.cuda_malloc_calls
            << "\n";


        std::cout
            << "cudaFree reduction    : "
            << direct.cuda_free_calls
            << " -> "
            << pool.cuda_free_calls
            << "\n";


        std::cout << "\n";


        return 0;

    }
    catch (const std::exception& e) {

        std::cerr
            << "\nBenchmark failed: "
            << e.what()
            << "\n";

        return 1;
    }
}