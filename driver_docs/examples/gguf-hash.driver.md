## Folders
- **[deps](gguf-hash/deps.driver.md)**: The `deps` folder in the `llama.cpp` codebase contains subfolders for various hashing and bit manipulation libraries, including `rotate-bits` for bit rotation macros, `sha1` and `sha256` for SHA-1 and SHA-256 hashing algorithms, and `xxhash` for fast non-cryptographic hashing.

## Files
- **[CMakeLists.txt](gguf-hash/CMakeLists.txt.driver.md)**: The `CMakeLists.txt` file in the `llama.cpp/examples/gguf-hash` directory configures the build process for the `llama-gguf-hash` executable, specifying dependencies on `xxhash`, `sha1`, and `sha256` libraries, and setting the C++ standard to C++17.
- **[gguf-hash.cpp](gguf-hash/gguf-hash.cpp.driver.md)**: The `gguf-hash.cpp` file in the `llama.cpp` codebase provides functionality for generating and verifying various types of hashes (xxh64, sha1, sha256, and uuid) for GGUF files, with options for checking against a manifest file.
- **[README.md](gguf-hash/README.md.driver.md)**: The `README.md` file in the `llama.cpp/examples/gguf-hash` directory provides instructions and details for using the `llama-gguf-hash` command-line tool, which hashes GGUF files to detect differences at both the model and tensor levels, offering various hashing options and verification capabilities.
