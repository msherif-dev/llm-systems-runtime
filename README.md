# LLM Systems Runtime 

A high-performance C++/CUDA system runtime focused on GPU memory management, custom CUDA kernels, and low-level latency optimization for LLM inference primitives.

---

##  Release Updates

### **v0.2: CUDA Memory Operations & Bandwidth Profiling** *(Current)*
Focuses on safe CUDA memory handling routines, explicit Host-Device allocation strategies, and precise throughput measurements.

####  Benchmark Results ($N = 1,000,000$ elements, 512 threads/block)
| Operation | Latency (ms) | Measured Bandwidth |
| :--- | :--- | :--- |
| **Host $\rightarrow$ Device Transfer** | `1.125 ms` | **7.11 GB/s** |
| **Kernel Execution Time** | `1.551 ms` | N/A |
| **Device $\rightarrow$ Host Transfer** | `2.423 ms` | **1.65 GB/s** |

> **Key Takeaway:** Demonstrates the impact of PCIe transfer bottlenecks on un-pinned host memory allocations and reveals H2D/D2H bandwidth asymmetry during runtime execution.

---

##  Repository Layout

```text
llm-systems-runtime/
├── cuda-memory/
│   └── src/
│       ├── vector_add_safe.cu            # Robust error-checked vector addition
│       └── vector_add_safe_benchmark.cu  # Memory bandwidth & latency profiler
├── docs/                                 # Architecture notes
└── README.md
