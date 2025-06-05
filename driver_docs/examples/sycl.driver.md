
## Files
- **[build.sh](sycl/build.sh.driver.md)**: The `build.sh` file in the `llama.cpp` codebase is a shell script for setting up and building SYCL-based examples using Intel's compilers, with options for FP16 and FP32 configurations.
- **[CMakeLists.txt](sycl/CMakeLists.txt.driver.md)**: The `CMakeLists.txt` file in `llama.cpp/examples/sycl` configures the build process for the `llama-ls-sycl-device` executable, specifying source files, installation rules, library dependencies, and C++ standard requirements.
- **[ls-sycl-device.cpp](sycl/ls-sycl-device.cpp.driver.md)**: The `ls-sycl-device.cpp` file in the `llama.cpp` codebase is a simple program that prints available SYCL devices using the `ggml_backend_sycl_print_sycl_devices` function.
- **[README.md](sycl/README.md.driver.md)**: The `README.md` file in `llama.cpp/examples/sycl` provides instructions and details for using the `llama.cpp` example program with SYCL on Intel GPUs, including a tool to list SYCL devices and their specifications.
- **[run-llama2.sh](sycl/run-llama2.sh.driver.md)**: The `run-llama2.sh` file is a shell script for executing the Llama 2 model using Intel's oneAPI with SYCL support, allowing for single or multiple GPU configurations.
- **[run-llama3.sh](sycl/run-llama3.sh.driver.md)**: The `run-llama3.sh` file is a shell script for executing the Llama model using Intel's oneAPI with SYCL support, allowing for GPU offloading and device selection.
- **[win-build-sycl.bat](sycl/win-build-sycl.bat.driver.md)**: The `win-build-sycl.bat` file is a batch script for setting up and building the SYCL example in the `llama.cpp` project on Windows, utilizing Intel's oneAPI and CMake with options for FP16 and FP32 configurations.
- **[win-run-llama2.bat](sycl/win-run-llama2.bat.driver.md)**: The `win-run-llama2.bat` file is a batch script for running the `llama-cli.exe` executable with specific parameters to process a text input using the Llama 2 model on a Windows system.
- **[win-run-llama3.bat](sycl/win-run-llama3.bat.driver.md)**: The `win-run-llama3.bat` file is a batch script for running the Llama CLI with a specific model and input on a Windows system using Intel's oneAPI environment.
