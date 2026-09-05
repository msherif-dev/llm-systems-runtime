#pragma once 

#include<cstddef>

class GPUMemoryPool
{
    private:
        void *base_ptr ;
        std::size_t pool_size ; 
        std::size_t offset ; 
        std::size_t alignment;

    public:
        explicit GPUMemoryPool(std::size_t alignment = 256);

        ~GPUMemoryPool();


        void init(size_t size);
        void* allocate(size_t size) ;

        void destroy();
        std::size_t used() const;
        std::size_t remaining() const;
        std::size_t capacity() const;
        void* base() const;
};

