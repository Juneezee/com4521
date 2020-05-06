| FINAL MARK | 96 | 100 |
| --- | ---: | ---: |
| **Part 1: Code** | | |
| <ul><li> Is it functionally correct?; Has iterative improvement produced a sufficiently optimised final program?; Does the code make good use of OpenMP?; Is memory handled appropriately and efficienty?; Are there compiler warnings?; Is the structure clear and easy to parse?</li><li>A low mark suggests that you need to think more carefully about what is required to write and optimise C and OpenMP code; A mark in the middle of the range suggests that you have some understanding of how to use C and OpenMP. A high mark suggests you are well on your way to understanding how to use C and OpenMP to produce efficient optimised code</li></ul> | 48 | 50 |
| **Functionally Correct** | 9 | 10 |
| <ul><li>Incorrect force calculation CPU. Your force calculation is incorrect on the CPU. Check the mathematics to ensure you have implemented the requirements in the handout.You need to calculate all the forces/accelerations first and then update the position and velocity of the bodies</li><li>Correct activity map CPU</li><li> Incorrect force calculation OpenMP. Your force calculation is incorrect in the OpenMP version . This may be caused by an incorrect CPU calculation. You should check the mathematics to ensure you have implemented the requirements in the handout </li><li> Correct activity map OpenMP</li><li>Simulation runs correctly with random data</li><li>Correct output format</li><li>Program takes the expected arguments</li><li>Code compiles</li><li>Program does not crash</li><li>Correct file reading</li></ul> | | |
| **Iterative Improvement** | 10 | 10 |
| <ul><li>Clear iterative improvement</li><li>Parallelised outer loop</li><li>Fast performance</li></ul> | | |
| **Memory Bandwidth** | 8 | 10 |
| <ul><li>No coalesced memory access. The memory access could be improved by using a different data structure. Use coalesced memory access by using a structure of arrays (SoA) rather than array of structure (AoS)</li><li>No unnecessary data movement</li><li>Correct memory allocation</li></ul> | | |
| **OpenMP Race Conditions** | 10 | 10 |
| <ul><li>No race condition in force calculation</li><li>No race condition in activity map</li></ul> | | |
| **OpenMP Variable Scoping** | 10 | 10 |
| <ul><li>Correct use of explicit scoping</li><li>No unnecessary shared variables</li></ul> | | |
| **Memory Access & Warnings** | 10 | 10 |
| <ul><li>Code has no memory leaks</li><li>The allocated memory is freed appropriately</li><li>Compiler does not throw warnings</li><li>No use of uninitialised variables</li></ul> | | |
| **Robust Input Handling** | 10 | 10 |
| <ul><li>Code deals with badly formatted files</li><li>Code deals with different number of bodies</li><li>Code reads comment lines</li><li>Code deals with empty values</li><li>Program handles input errors and exits</li></ul>
| **Well Written Code** | 10 | 10 |
| <ul><li>Well commented code</li><li>Good variable naming</li><li>Correct use of data types</li><li>Good use of methods</li><li>Good code structure</li></ul> | | |
| **Part 2: Documentation** | 48 | 50 |
| <ul><li>Is there a description of the implemntation? Is there a justifcation of the final implementation, including any steps taken to reach that conclusion? Description of the general testing process? Has performance throughout development been analysed – is there evidence of this?</li><li>A low mark suggests that you need to think more carefully about what is required to document development and optimisation. A mark in the middle of the range suggests that you have some understanding of the process and some consideration of attention to detail. A high mark suggests you are well on your way to understanding how to document the optimisation of code you develop</li></ul> | | |
| **Description of Implementation** | 10 | 10 |
| Very well structured report; Good use of tables with timings, graphs and code snippets; Explanations were thorough and coupled to solid theory; | | |
| **Description of OpenMP Implementation (Force calculation)** | 9 | 10 |
| Detailed examination of inner/outer loop parallelising with measurements to back up; Very thorough scheduling examination; Discussed atomics vs critical vs reduction | | |
| **Description of OpenMP Implementation (Activity map)** | 9 | 10 |
| Detailed examination of inner/outer loop parallelising with measurements to back up; Very thorough scheduling examination; Little comparison of atomic or critical sections | | |
| **Analysis of Optimisation** | 10 | 10 |
| Good structure of iterative improvements; Optimisations done and results were clear; Decisions were clearly influenced by well found data; Comprehensive explanation of why changes affect the performance; Efficient use of the profiler to find places to optimise | | |

**Overall**
---

**What is good about this work?**
- Fast performance
- No race condition in force calculation
- No race condition in activity map
- Code deals with badly formatted files
- Good structure of iterative improvements
- Optimisations done and results were clear
- Decisions were clearly influenced by well found data
- Comprehensive explanation of why changes affect the performance
- Efficient use of the profiler to find places to optimise

**What needs to be done to make it better?**
- You need to calculate all the forces/accelerations first and then update the position and velocity of the bodies;
- No coalesced memory access
- The memory access could be improved by using a different data structure
