
## Files
- **[sgemm.cpp](llamafile/sgemm.cpp.driver.md)**: The `sgemm.cpp` file in the `llama.cpp` codebase implements multithreaded CPU matrix multiplication for the common contiguous use case `C = Aᵀ * B`, utilizing various vectorized arithmetic operations and optimizations for different CPU architectures to enhance performance.
- **[sgemm.h](llamafile/sgemm.h.driver.md)**: The `sgemm.h` file declares the `llamafile_sgemm` function, which performs a single-precision general matrix multiplication operation.
