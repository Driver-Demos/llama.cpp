## Folders
- **[kompute-shaders](ggml-kompute/kompute-shaders.driver.md)**: The `kompute-shaders` folder in the `llama.cpp` codebase contains a collection of GLSL compute shaders designed for various tensor operations, including arithmetic, activation functions, normalization, and matrix manipulations, optimized for GPU execution.

## Files
- **[CMakeLists.txt](ggml-kompute/CMakeLists.txt.driver.md)**: The `CMakeLists.txt` file in the `llama.cpp/ggml/src/ggml-kompute` directory configures the build process for the `ggml-kompute` library, including finding Vulkan components, compiling shader files, and managing dependencies for shader generation.
- **[ggml-kompute.cpp](ggml-kompute/ggml-kompute.cpp.driver.md)**: The `ggml-kompute.cpp` file in the `llama.cpp` codebase provides an implementation for managing and executing tensor operations using Vulkan and the Kompute library, including functions for device management, memory allocation, and various tensor operations like addition, multiplication, and activation functions.
