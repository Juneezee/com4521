| FINAL MARK | 96 | 100 |
| --- | ---: | ---: |
| **Part 1: Code** | | |
| <ul><li>Is it functionally correct? Have you managed to use GPUs for all aspects of the simulation without having to read back data to and from the device each iteration? Has iterative improvement of the code yielded a sufficiently optimised final GPU implementations? Does the code make good use of memory bandwidth? Does the implementation avoid race conditions in calculating the Histogram/Activity Map? Does the GPU code use effective caching to reduce the number of global memory loads? Are there any compiler warnings or dangerous memory accesses (beyond the bounds of the memory allocated)? Does your program free any memory which is allocated?</li><li>A low mark suggests that you need to think more carefully about what is required to write and optimise CUDA code; A mark in the middle of the range suggests that you have some understanding of how to use CUDA. A high mark suggests you are well on your way to understanding how to use CUDA to produce efficient optimised code</li></ul> | 47 | 50 |
| **Functionally Correct** | 10 | 10 |
| <ul><li>Correct force calculation</li><li>Correct activity map</li></ul> | | |
| **Use of GPU** | 10 | 10 |
| <ul><li>All aspects of the simulation were implemented on the GPU</li><li>No data movement between host and device each iteration</li></ul> | | |
| **Iterative Improvement** | 10 | 10 |
| <ul><li>Implementation of the simulation parallelised each body. Parallelising over bodies provides opportunity to use gpu caching effectively</li><li>Good grid and block size selection</li><li>Fast performance</li></ul> | | |
| **Memory Bandwidth** | 7.5 | 10 |
| <ul><li>No pointer swapping. Consider using double buffering and pointer swaps to reduce memory copies and memory movement. It is better to use 2 buffers, input and output, you can perform all the calculations in one kernel and swap the pointers before starting the new iteration</li><li>Coalesced memory access</li></ul> | | |
| **Race Conditions** | 10 | 10 |
| <ul><li>Correct use of double buffered data. You did not use double buffered data, but your code does not have a race condition in the force calculation since you use 2 kernels</li><li>No race condition in the activity map</li></ul> | | |
| **GPU Caching** | 10 | 10 |
| <ul><li>Correct implementation of shared memory. This is the fastest approach for the simulation</li><li>No bank conflicts when accessing memory</li></ul> | | |
| **Warnings & Dangerous Memory Access** | 8 | 10 |
| <ul><li>No obvious memory leaks</li><li>All the allocated memory was freed correctly</li><li>No compiler warnings</li><li>Use of uninitialised variables.CUDA Memory Checker throws OutOfRangeStore exception when assigning the local_sum to d_force_sum; you need to check I < c_N</li><li>Correct use of syncthreads() where needed</li></ul> | | |
| **Well Written Code** | 10 | 10 |
| <ul><li>Well commented code</li><li>Good variable naming</li><li>Correct use of data types</li><li>Good use of methods</li></ul> | | |
| **Part 2: Documentation** | 45 | 50 |
| <ul><li>Is there a description of the technique and its implementation? Have appropriate investigations been made into using a good memory access pattern and suitable caching technique? Are good explanations given for the benchmarking results?; Does your document describe optimisations to your code and show the impact of these? Is there benchmarking and discussion about the performance difference between all three version of the code? </li><li>A low mark suggests that you need to think more carefully about what is required to document development and optimisation. A mark in the middle of the range suggests that you have some understanding of the process and some consideration of attention to detail. A high mark suggests you are well on your way to understanding how to document the optimisation of code you develop</li></ul> | | |
| **Description of Implementation** | 10 | 10 |
| Good explanations as to the methods implemented; Good justification for the parallelisation approach used; Included timings to aid the justification | | |
| **Memory Access Pattern** | 9 | 10 |
| Good examination of the different types of caches available. Good theoretic explanations and conclusions; Considerable discussion about AOS vs SOA memory implications; Benchmarking and profiling is very thorough; Good selection of N values for the benchmarking | | |
| **Optimisations** | 9 | 10 |
| Good iterative improvement loops showing progressive changes; Report shows understanding of limiting factors and justification as to the work carried out; Great use of the profiler to show areas for improvement | | |
| **Benchmarking and Discussion** | 10 | 10 |
| No explicit comparison of three versions; Report only examines the CUDA version, CPU and OpenMP discussion is missing; Suitable timings over a range of variables to prove timings; Good use of code snippets to highlight the modifications done | | |

**Overall**
---

**What is good about this work?**
- Correct force calculation
- Correct activity map
- Fast performance
- All aspects of the simulation were implemented on the GPU
- No data movement between host and device each iteration
- Correct implementation of shared memory. This is the fastest approach for the simulation
- No bank conflicts when accessing memory. No unnecessary memory data movement from global memory within loops
- Well structure report covering almost all of the requirements

**What needs to be done to make it better?**
- No pointer swapping. Consider using double buffering and pointer swaps to reduce memory copies and memory movement
- CUDA Memory Checker throws OutOfRangeStore exception when assigning the local_sum to d_force_sum; you need to check I < c_N
- Report only examines the CUDA version, CPU and OpenMP discussion is missing
