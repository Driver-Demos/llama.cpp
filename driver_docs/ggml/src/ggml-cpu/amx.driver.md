
## Files
- **[amx.cpp](amx/amx.cpp.driver.md)**: The `amx.cpp` file in the `llama.cpp` codebase implements AMX-specific tensor operations and buffer management for the GGML backend, including initialization, memory allocation, and operation support checks.
- **[amx.h](amx/amx.h.driver.md)**: The `amx.h` file in the `llama.cpp` codebase is an internal header that declares a function for determining the buffer type for AMX when certain CPU features are available.
- **[common.h](amx/common.h.driver.md)**: The `common.h` file in the `llama.cpp` codebase provides utility functions and definitions for parallel processing and AMX support, including templates for dividing tasks and checking quantized types with AMX kernel compatibility.
- **[mmq.cpp](amx/mmq.cpp.driver.md)**: The `mmq.cpp` file in the `llama.cpp` codebase provides implementations for matrix multiplication using Advanced Matrix Extensions (AMX) and AVX512 instructions, supporting various quantized data types and optimizing for different hardware configurations.
- **[mmq.h](amx/mmq.h.driver.md)**: The `mmq.h` file in the `llama.cpp` codebase defines functions for managing tensor operations and memory allocation specific to the AMX backend, including functions for determining desired workspace size, allocation size, weight conversion, and matrix multiplication.
