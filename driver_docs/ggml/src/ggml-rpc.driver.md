
## Files
- **[CMakeLists.txt](ggml-rpc/CMakeLists.txt.driver.md)**: The `CMakeLists.txt` file in `llama.cpp/ggml/src/ggml-rpc` configures the build system to use the RPC backend and conditionally links the `ws2_32` library on Windows.
- **[ggml-rpc.cpp](ggml-rpc/ggml-rpc.cpp.driver.md)**: The `ggml-rpc.cpp` file in the `llama.cpp` codebase implements a remote procedure call (RPC) system for managing tensor operations and memory buffers across different platforms, including functions for socket communication, tensor serialization, and server-client interactions.
