
## Files
- **[CMakeLists.txt](rpc/CMakeLists.txt.driver.md)**: The `CMakeLists.txt` file in `llama.cpp/tools/rpc` configures the build system to compile the `rpc-server` executable with C++17 standard and links it with the `ggml` library.
- **[README.md](rpc/README.md.driver.md)**: The `README.md` file in the `llama.cpp/tools/rpc` directory provides an overview and usage instructions for setting up and running the `rpc-server` to enable distributed LLM inference using the `ggml` backend on remote hosts, with a focus on building, configuring, and running the server securely.
- **[rpc-server.cpp](rpc/rpc-server.cpp.driver.md)**: The `rpc-server.cpp` file in the `llama.cpp` codebase implements an RPC server that initializes a backend, parses command-line parameters, manages cache directories, and starts the server using the specified host and port.
