
#include "gpu_memory_pool.cuh"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>


// ============================================================
// Helper
// ============================================================

template <typename Exception, typename Function>
void expect_exception(Function&& function) {

    bool thrown = false;

    try {

        function();

    }
    catch (const Exception&) {

        thrown = true;
    }

    assert(thrown);
}


// ============================================================
// Test 1
// Basic allocation
// ============================================================

void test_basic_allocation() {

    std::cout << "[TEST] Basic allocation\n";

    GPUMemoryPool pool(256);

    pool.init(4096);

    void* p1 = pool.allocate(100);
    void* p2 = pool.allocate(100);
    void* p3 = pool.allocate(100);

    assert(p1 != nullptr);
    assert(p2 != nullptr);
    assert(p3 != nullptr);

    assert(p1 != p2);
    assert(p2 != p3);
    assert(p1 != p3);

    pool.destroy();

    std::cout << "  PASSED\n";
}


// ============================================================
// Test 2
// Zero-size pool
// ============================================================

void test_zero_pool() {

    std::cout << "[TEST] Zero-size pool\n";

    GPUMemoryPool pool(256);

    expect_exception<std::invalid_argument>(
        [&]() {
            pool.init(0);
        }
    );

    std::cout << "  PASSED\n";
}


// ============================================================
// Test 3
// Zero allocation
// ============================================================

void test_zero_allocation() {

    std::cout << "[TEST] Zero allocation\n";

    GPUMemoryPool pool(256);

    pool.init(4096);

    expect_exception<std::invalid_argument>(
        [&]() {
            pool.allocate(0);
        }
    );

    pool.destroy();

    std::cout << "  PASSED\n";
}


// ============================================================
// Test 4
// Allocate before initialization
// ============================================================

void test_allocate_before_init() {

    std::cout
        << "[TEST] Allocate before init\n";

    GPUMemoryPool pool(256);

    expect_exception<std::runtime_error>(
        [&]() {
            pool.allocate(100);
        }
    );

    std::cout << "  PASSED\n";
}


// ============================================================
// Test 5
// Double initialization
// ============================================================

void test_double_init() {

    std::cout << "[TEST] Double init\n";

    GPUMemoryPool pool(256);

    pool.init(4096);

    expect_exception<std::runtime_error>(
        [&]() {
            pool.init(4096);
        }
    );

    pool.destroy();

    std::cout << "  PASSED\n";
}


// ============================================================
// Test 6
// Allocation larger than pool
// ============================================================

void test_allocation_too_large() {

    std::cout
        << "[TEST] Allocation larger than pool\n";

    GPUMemoryPool pool(256);

    pool.init(1024);

    expect_exception<std::bad_alloc>(
        [&]() {
            pool.allocate(2048);
        }
    );

    pool.destroy();

    std::cout << "  PASSED\n";
}


// ============================================================
// Test 7
// Allocation larger than remaining
// ============================================================

void test_allocation_larger_than_remaining() {

    std::cout
        << "[TEST] Allocation larger than remaining\n";

    GPUMemoryPool pool(256);

    pool.init(1024);

    pool.allocate(512);

    assert(pool.remaining() == 512);

    expect_exception<std::bad_alloc>(
        [&]() {
            pool.allocate(513);
        }
    );

    pool.destroy();

    std::cout << "  PASSED\n";
}


// ============================================================
// Test 8
// Exact remaining allocation
// ============================================================

void test_exact_remaining() {

    std::cout
        << "[TEST] Exact remaining allocation\n";

    GPUMemoryPool pool(256);

    pool.init(1024);

    pool.allocate(512);

    assert(pool.remaining() == 512);

    pool.allocate(512);

    assert(pool.used() == 1024);

    assert(pool.remaining() == 0);

    pool.destroy();

    std::cout << "  PASSED\n";
}


// ============================================================
// Test 9
// Allocate after pool is full
// ============================================================

void test_allocate_after_full() {

    std::cout
        << "[TEST] Allocate after full\n";

    GPUMemoryPool pool(256);

    pool.init(1024);

    pool.allocate(1024);

    assert(pool.remaining() == 0);

    expect_exception<std::bad_alloc>(
        [&]() {
            pool.allocate(1);
        }
    );

    pool.destroy();

    std::cout << "  PASSED\n";
}


// ============================================================
// Test 10
// Alignment
// ============================================================

void test_alignment() {

    std::cout << "[TEST] Alignment\n";

    GPUMemoryPool pool(256);

    pool.init(4096);

    void* p1 = pool.allocate(100);
    void* p2 = pool.allocate(100);
    void* p3 = pool.allocate(100);


    const auto addr1 =
        reinterpret_cast<std::uintptr_t>(p1);

    const auto addr2 =
        reinterpret_cast<std::uintptr_t>(p2);

    const auto addr3 =
        reinterpret_cast<std::uintptr_t>(p3);


    assert(addr1 % 256 == 0);
    assert(addr2 % 256 == 0);
    assert(addr3 % 256 == 0);

    pool.destroy();

    std::cout << "  PASSED\n";
}


// ============================================================
// Test 11
// Pointer range
// ============================================================

void test_pointer_range() {

    std::cout
        << "[TEST] Pointer range\n";

    constexpr std::size_t POOL_SIZE = 4096;

    GPUMemoryPool pool(256);

    pool.init(POOL_SIZE);


    void* p1 = pool.allocate(100);
    void* p2 = pool.allocate(200);
    void* p3 = pool.allocate(300);


    const auto base =
        reinterpret_cast<std::uintptr_t>(
            pool.base()
        );


    const auto end =
        base + POOL_SIZE;


    const auto addr1 =
        reinterpret_cast<std::uintptr_t>(p1);

    const auto addr2 =
        reinterpret_cast<std::uintptr_t>(p2);

    const auto addr3 =
        reinterpret_cast<std::uintptr_t>(p3);


    assert(addr1 >= base);
    assert(addr1 < end);

    assert(addr2 >= base);
    assert(addr2 < end);

    assert(addr3 >= base);
    assert(addr3 < end);


    pool.destroy();

    std::cout << "  PASSED\n";
}


// ============================================================
// Test 12
// Accounting invariant
// ============================================================

void test_accounting() {

    std::cout
        << "[TEST] Accounting invariant\n";

    GPUMemoryPool pool(256);

    pool.init(8192);


    assert(
        pool.used() +
        pool.remaining()
        ==
        pool.capacity()
    );


    pool.allocate(100);

    assert(
        pool.used() +
        pool.remaining()
        ==
        pool.capacity()
    );


    pool.allocate(200);

    assert(
        pool.used() +
        pool.remaining()
        ==
        pool.capacity()
    );


    pool.allocate(500);

    assert(
        pool.used() +
        pool.remaining()
        ==
        pool.capacity()
    );


    pool.destroy();

    std::cout << "  PASSED\n";
}


// ============================================================
// Test 13
// Destroy resets state
// ============================================================

void test_destroy_resets_state() {

    std::cout
        << "[TEST] Destroy resets state\n";

    GPUMemoryPool pool(256);

    pool.init(4096);

    pool.allocate(500);

    pool.destroy();


    assert(pool.base() == nullptr);

    assert(pool.capacity() == 0);

    assert(pool.used() == 0);

    assert(pool.remaining() == 0);


    std::cout << "  PASSED\n";
}


// ============================================================
// Test 14
// Allocate after destroy
// ============================================================

void test_allocate_after_destroy() {

    std::cout
        << "[TEST] Allocate after destroy\n";

    GPUMemoryPool pool(256);

    pool.init(4096);

    pool.destroy();


    expect_exception<std::runtime_error>(
        [&]() {
            pool.allocate(100);
        }
    );


    std::cout << "  PASSED\n";
}


// ============================================================
// Test 15
// Destroy twice
// ============================================================

void test_destroy_twice() {

    std::cout
        << "[TEST] Destroy twice\n";

    GPUMemoryPool pool(256);

    pool.init(4096);

    pool.destroy();

    // Should be safe.
    pool.destroy();

    assert(pool.base() == nullptr);
    assert(pool.capacity() == 0);
    assert(pool.used() == 0);

    std::cout << "  PASSED\n";
}


// ============================================================
// Test 16
// Invalid alignment
// ============================================================

void test_invalid_alignment() {

    std::cout
        << "[TEST] Invalid alignment\n";

    expect_exception<std::invalid_argument>(
        []() {
            GPUMemoryPool pool(0);
        }
    );

    std::cout << "  PASSED\n";
}


// ============================================================
// Test 17
// Pool exactly one allocation
// ============================================================

void test_exact_pool_allocation() {

    std::cout
        << "[TEST] Exact pool allocation\n";

    GPUMemoryPool pool(256);

    pool.init(1024);

    void* ptr = pool.allocate(1024);

    assert(ptr != nullptr);

    assert(pool.used() == 1024);

    assert(pool.remaining() == 0);

    pool.destroy();

    std::cout << "  PASSED\n";
}


// ============================================================
// Main
// ============================================================

int main() {

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << " GPU Memory Pool - Day 3 Tests\n";
    std::cout << "========================================\n\n";


    test_basic_allocation();

    test_zero_pool();

    test_zero_allocation();

    test_allocate_before_init();

    test_double_init();

    test_allocation_too_large();

    test_allocation_larger_than_remaining();

    test_exact_remaining();

    test_allocate_after_full();

    test_alignment();

    test_pointer_range();

    test_accounting();

    test_destroy_resets_state();

    test_allocate_after_destroy();

    test_destroy_twice();

    test_invalid_alignment();

    test_exact_pool_allocation();


    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << " ALL DAY 3 TESTS PASSED\n";
    std::cout << "========================================\n";


    return 0;
}