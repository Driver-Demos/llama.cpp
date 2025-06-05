
## Files
- **[CMakeLists.txt](gguf-split/CMakeLists.txt.driver.md)**: The `CMakeLists.txt` file in `llama.cpp/tools/gguf-split` configures the build system to compile and install the `llama-gguf-split` executable with C++17 standard and links it with the `common`, `llama`, and thread libraries.
- **[gguf-split.cpp](gguf-split/gguf-split.cpp.driver.md)**: The `gguf-split.cpp` file in the `llama.cpp` codebase provides functionality to split or merge GGUF files based on specified parameters, including options for splitting by tensor count or file size.
- **[README.md](gguf-split/README.md.driver.md)**: The `README.md` file in the `llama.cpp/tools/gguf-split` directory provides instructions for using a command-line interface to split and merge GGUF files, detailing options such as maximum size and number of tensors per split.
- **[tests.sh](gguf-split/tests.sh.driver.md)**: The `tests.sh` file is a bash script used to test the splitting and merging of GGUF models in the `llama.cpp` codebase, verifying the functionality of the `llama-gguf-split` and `llama-cli` tools.
