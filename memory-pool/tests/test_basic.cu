#include "gpu_memory_pool.cuh"

#include <iostream>


int main()
{
    constexpr std::size_t MB =
        1024 * 1024;


    std::cout
        << "Creating GPU memory pool...\n";


    GPUMemoryPool pool;


    // Allocate 100 MB once.
    pool.init(100 * MB);


    std::cout
        << "Pool created: 100 MB\n";


    // Multiple allocations.
    void* p1 =
        pool.allocate(10 * MB);

    void* p2 =
        pool.allocate(20 * MB);

    void* p3 =
        pool.allocate(5 * MB);


    std::cout
        << "p1: " << p1 << '\n';

    std::cout
        << "p2: " << p2 << '\n';

    std::cout
        << "p3: " << p3 << '\n';


    // Verify pointers are different.
    if (p1 == p2 ||
        p1 == p3 ||
        p2 == p3)
    {
        std::cerr
            << "ERROR: allocations overlap!\n";

        pool.destroy();

        return 1;
    }


    std::cout
        << "All pointers are different.\n";


    pool.destroy();


    std::cout
        << "Pool destroyed.\n";


    return 0;
}