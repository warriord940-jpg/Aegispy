AegisPy: Parallel Heterogeneous Execution & Smart Caching Engine
​AegisPy is a lightweight Python systems-level parallel execution framework designed to automatically analyze, route, and execute computational tasks across heterogeneous hardware architectures (Ryzen Multicore CPUs and NVIDIA RTX CUDA streams) while leveraging an intelligent caching layer (smartrun) to eliminate redundant computation overhead.
​Key Features
​Automated AST Hardware Routing: Parses Python Abstract Syntax Trees to automatically detect GPU-specific commands (such as .to(), cuda(), etc.) and route tensor operations directly to the NVIDIA RTX 2050 GPU streams, while routing standard data and math workloads to the Ryzen multicore CPU pool.
​Concurrent Thread Pool Pipeline: Utilizes Python's concurrent thread architecture to execute CPU and GPU workloads simultaneously, bypassing sequential bottlenecks.
​Instant Cache Short-Circuit (smartrun): Caches processed execution states using optimized storage, reducing subsequent execution times to near-zero (0.01s) on warm runs.
​CLI Payload Handling: Seamlessly processes input arguments and interactive payloads (factorial_user.py) within automated batch pipelines.
​Repository Structure
├── aegispy_core.py          # Core AST parser and hardware routing engine
├── sequential_core.py       # Sequential execution and caching baseline engine
├── testrun.py               # Heterogeneous (CPU + GPU) concurrent test suite
├── factorial.py             # CPU-bound factorial benchmark script
├── factorial_user.py        # CLI interactive payload benchmark script
├── run.bat                  # Batch script for AegisPy parallel execution
└── seq.bat                  # Batch script for sequential cached execution
Getting Started & Execution
​1. Running the AegisPy Parallel Engine
​To execute a script or a multi-file suite concurrently across the Ryzen CPU pool and NVIDIA RTX 2050 GPU:
.\run.bat testrun.py factorial.py factorial_user.py 5
2. Running the Sequential Cached Baseline
​To execute the same workload sequentially with disk caching enabled:
.\seq.bat testrun.py factorial.py factorial_user.py 5
Benchmark Performance
Execution Engine         Initial Run (Cache Miss)  Re-Run (Cache Hit / Short-Circuit)  Core Architecture & Hardware Routing
Pure Python                       0.72s                         0.77s                   Standard sequential execution with zero caching or parallel optimization.
Sequential Cached Engine          0.75s                         0.01s                   Sequential file loading with memory storage and disk retrieval.
AegisPy Parallel Engine           0.66s                         0.01s                   Concurrent ThreadPool execution + Dynamic AST CPU/GPU routing + Instant                                                                                            cache short-circuit.
