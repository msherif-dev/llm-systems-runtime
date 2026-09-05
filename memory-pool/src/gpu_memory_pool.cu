#include "gpu_memory_pool.cuh"

#include <cstdint>
#include <limits>
#include <stdexcept>


namespace {

std::size_t align_up(
    std::size_t value,
    std::size_t alignment
) {
    if (alignment == 0) {
        throw std::invalid_argument(
            "Alignment must be greater than zero."
        );
    }

    const std::size_t remainder = value % alignment;

    if (remainder == 0) {
        return value;
    }

    const std::size_t padding = alignment - remainder;

    if (value >
        std::numeric_limits<std::size_t>::max() - padding) {

        throw std::overflow_error(
            "Alignment calculation overflow."
        );
    }

    return value + padding;
}

}

GPUMemoryPool::GPUMemoryPool(std::size_t alignment)
    : base_ptr(nullptr),
      pool_size(0),
      offset(0),
      alignment(alignment)
{
    if (alignment == 0) {
        throw std::invalid_argument(
            "Alignment must be greater than zero."
        );
    }
}



GPUMemoryPool::~GPUMemoryPool() {
    destroy();
}


void GPUMemoryPool::init(std::size_t size) {

    if (size == 0) {
        throw std::invalid_argument(
            "Pool size must be greater than zero."
        );
    }

    if (base_ptr != nullptr) {
        throw std::runtime_error(
            "Memory pool is already initialized."
        );
    }

    cudaError_t result = cudaMalloc(
        &base_ptr,
        size
    );

    if (result != cudaSuccess) {

        base_ptr = nullptr;

        throw std::runtime_error(
            cudaGetErrorString(result)
        );
    }

    pool_size = size;
    offset = 0;
}



void* GPUMemoryPool::allocate(std::size_t size) {

    if (base_ptr == nullptr) {
        throw std::runtime_error(
            "Memory pool is not initialized."
        );
    }

    if (size == 0) {
        throw std::invalid_argument(
            "Allocation size must be greater than zero."
        );
    }


    std::size_t aligned_offset =
        align_up(offset, alignment);


    if (aligned_offset > pool_size) {

        throw std::bad_alloc();
    }



    if (size > pool_size - aligned_offset) {

        throw std::bad_alloc();
    }



    auto* base =
        static_cast<std::uint8_t*>(base_ptr);


    void* result =
        static_cast<void*>(
            base + aligned_offset
        );

    offset = aligned_offset + size;

    return result;
}


void GPUMemoryPool::destroy() {

    if (base_ptr != nullptr) {

        cudaError_t result =
            cudaFree(base_ptr);

        if (result != cudaSuccess) {

            throw std::runtime_error(
                cudaGetErrorString(result)
            );
        }
    }

    base_ptr = nullptr;
    pool_size = 0;
    offset = 0;
}


std::size_t GPUMemoryPool::used() const {

    return offset;
}


std::size_t GPUMemoryPool::remaining() const {

    if (offset >= pool_size) {
        return 0;
    }

    return pool_size - offset;
}


std::size_t GPUMemoryPool::capacity() const {

    return pool_size;
}


void* GPUMemoryPool::base() const {

    return base_ptr;
}