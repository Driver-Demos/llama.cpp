# Purpose
This source code file is a header file that provides a comprehensive set of definitions and utilities for managing CUDA and HIP (Heterogeneous-Compute Interface for Portability) operations within a software project. The file includes various preprocessor directives, type definitions, and function declarations that facilitate the integration and execution of GPU-accelerated computations. It is designed to support multiple GPU architectures, including NVIDIA's CUDA, AMD's HIP, and MUSA (Moore Threads Unified Software Architecture), by defining macros and functions that abstract the differences between these platforms.

The file defines a series of macros for handling different GPU compute capabilities, such as those for NVIDIA's Pascal, Volta, Turing, and Ampere architectures, as well as AMD's GCN, Vega, and RDNA architectures. These macros are used to determine the availability of specific hardware features, such as FP16 (half-precision floating-point) support and tensor core instructions, which are crucial for optimizing performance on different GPU models. Additionally, the file includes error-checking macros and functions to handle CUDA and cuBLAS (CUDA Basic Linear Algebra Subprograms) errors, ensuring robust error management during GPU operations.

Furthermore, the file provides template structures and functions for handling various data types and operations, such as dequantization and warp-level reductions, which are essential for efficient parallel processing on GPUs. It also defines a set of device and backend context structures that manage device-specific resources, such as streams, cuBLAS handles, and memory pools, enabling efficient resource allocation and management across multiple GPUs. Overall, this header file serves as a foundational component for building GPU-accelerated applications, offering a unified interface for handling diverse GPU architectures and optimizing computational performance.
# Imports and Dependencies

---
- `ggml.h`
- `ggml-cuda.h`
- `cstdint`
- `memory`
- `cstdio`
- `array`
- `cassert`
- `cfloat`
- `string`
- `vector`
- `vendors/hip.h`
- `vendors/musa.h`
- `vendors/cuda.h`


# Global Variables

---
### kvalues\_iq4nl
- **Type**: `int8_t[16]`
- **Description**: `kvalues_iq4nl` is a global constant array of 16 elements of type `int8_t`. It contains a sequence of integer values ranging from -127 to 113, which are likely used for quantization or dequantization purposes in CUDA or HIP-based computations.
- **Use**: This array is used as a lookup table for quantization or dequantization operations in GPU computations.


# Data Structures

---
### ggml\_cuda\_device\_info
- **Type**: `struct`
- **Members**:
    - `device_count`: The number of CUDA devices available.
    - `devices`: An array of `cuda_device_info` structures for each CUDA device.
    - `default_tensor_split`: An array of floats representing the default tensor split across devices.
- **Description**: The `ggml_cuda_device_info` structure is designed to store information about CUDA devices available in the system. It includes a count of the devices and an array of `cuda_device_info` structures, each containing details such as compute capability, number of streaming multiprocessors, shared memory per block, and other device-specific attributes. Additionally, it holds an array for default tensor split configuration across the devices.


---
### ggml\_cuda\_pool
- **Type**: `struct`
- **Members**:
    - `alloc`: A pure virtual function to allocate memory of a given size and return a pointer to it.
    - `free`: A pure virtual function to free the memory pointed to by a given pointer.
- **Description**: The `ggml_cuda_pool` is an abstract base class that defines an interface for memory management in CUDA environments. It provides two pure virtual functions, `alloc` and `free`, which must be implemented by derived classes to handle memory allocation and deallocation, respectively. This structure is designed to manage memory resources efficiently on CUDA devices, ensuring that memory is allocated and freed as needed for CUDA operations.


---
### ggml\_cuda\_pool\_alloc
- **Type**: `template<typename T> struct`
- **Members**:
    - `pool`: A pointer to a ggml_cuda_pool object used for memory allocation.
    - `ptr`: A pointer to the allocated memory of type T.
    - `actual_size`: The actual size of the allocated memory in bytes.
- **Description**: The `ggml_cuda_pool_alloc` is a template structure designed to manage memory allocation for CUDA operations using a specified memory pool. It holds a pointer to a `ggml_cuda_pool` for managing allocations, a pointer to the allocated memory of type T, and the actual size of the allocated memory. The structure provides constructors for initializing with a pool and optionally allocating memory, a destructor to free allocated memory, and methods to allocate memory and retrieve the pointer to the allocated memory. It ensures that memory is properly managed and released, preventing memory leaks in CUDA applications.


---
### ggml\_tensor\_extra\_gpu
- **Type**: `struct`
- **Members**:
    - `data_device`: An array of pointers, one for each device, used for split tensors.
    - `events`: A 2D array of CUDA events for synchronizing multiple GPUs, with dimensions for devices and streams.
- **Description**: The `ggml_tensor_extra_gpu` structure is designed to facilitate the management of tensor data across multiple GPU devices in a CUDA environment. It contains an array of device pointers, `data_device`, which allows for the distribution of tensor data across different GPUs, enabling parallel processing. Additionally, the `events` member is a 2D array of CUDA events that are used to synchronize operations across multiple GPUs and streams, ensuring that data dependencies are respected and operations are executed in the correct order. This structure is crucial for efficient GPU utilization and performance in multi-GPU setups.


---
### ggml\_graph\_node\_properties
- **Type**: `struct`
- **Members**:
    - `node_address`: Pointer to the memory address of the node.
    - `node_op`: Operation type associated with the node.
    - `ne`: Array representing the number of elements in each dimension.
    - `nb`: Array representing the number of bytes in each dimension.
    - `src_address`: Array of pointers to the source addresses for the node.
    - `op_params`: Array of operation parameters for the node.
- **Description**: The `ggml_graph_node_properties` structure is designed to encapsulate the properties of a node within a computational graph, particularly in the context of GPU operations. It includes information about the node's memory address, the operation it performs, and its dimensional properties. Additionally, it holds pointers to source addresses and operation-specific parameters, making it a comprehensive representation of a node's configuration and requirements in a graph-based computation environment.


---
### ggml\_cuda\_graph
- **Type**: `struct`
- **Members**:
    - `graph`: A CUDA graph object used for capturing and executing a sequence of CUDA operations.
    - `instance`: An executable instance of the CUDA graph.
    - `num_nodes`: The number of nodes in the CUDA graph.
    - `nodes`: A vector containing the nodes of the CUDA graph.
    - `params`: A vector containing the parameters for CUDA kernel nodes.
    - `disable_due_to_gpu_arch`: A flag indicating if the graph is disabled due to incompatible GPU architecture.
    - `disable_due_to_too_many_updates`: A flag indicating if the graph is disabled due to excessive updates.
    - `disable_due_to_failed_graph_capture`: A flag indicating if the graph is disabled due to a failed graph capture.
    - `number_consecutive_updates`: The number of consecutive updates made to the graph.
    - `ggml_graph_properties`: A vector containing properties of the graph nodes.
    - `use_cpy_indirection`: A flag indicating if copy indirection is used in the graph.
    - `cpy_dest_ptrs`: A vector of destination pointers for copy operations.
    - `dest_ptrs_d`: A device pointer to an array of destination pointers.
    - `dest_ptrs_size`: The size of the destination pointers array.
    - `graph_cpynode_index`: The index of the copy node within the graph.
- **Description**: The `ggml_cuda_graph` structure is designed to manage and execute a sequence of CUDA operations using CUDA Graphs, which can optimize performance by reducing the overhead of launching multiple kernels. It includes members for storing the graph and its executable instance, as well as vectors for nodes and kernel parameters. The structure also contains flags to disable the graph under certain conditions, such as incompatible GPU architecture or excessive updates, and manages properties related to graph nodes and copy operations.


---
### ggml\_backend\_cuda\_context
- **Type**: `struct`
- **Members**:
    - `device`: The CUDA device identifier associated with this context.
    - `name`: A string representing the name of the CUDA context, typically including the device number.
    - `copy_event`: A CUDA event used for synchronizing copy operations.
    - `streams`: An array of CUDA streams for each device, used to manage concurrent operations.
    - `cublas_handles`: An array of cuBLAS handles for each device, used for performing BLAS operations on the GPU.
    - `cuda_graph`: A unique pointer to a CUDA graph, which is used for capturing and executing a sequence of CUDA operations.
    - `pools`: An array of unique pointers to memory pools for each device, used for efficient memory management.
- **Description**: The `ggml_backend_cuda_context` structure is designed to manage and encapsulate the resources and operations associated with a specific CUDA device in a multi-device environment. It includes device-specific identifiers, CUDA streams for managing concurrent operations, cuBLAS handles for performing linear algebra operations, and a CUDA graph for capturing and executing sequences of operations. Additionally, it manages memory pools for efficient memory allocation and deallocation on each device. This structure is essential for optimizing performance and resource management in GPU-accelerated applications.


# Functions

---
### ggml\_cuda\_has\_arch\_impl
The `ggml_cuda_has_arch_impl` function checks if a given CUDA architecture is supported by the compiled code.
- **Inputs**:
    - `arch`: An integer representing the CUDA architecture to check against the compiled architectures.
    - `first`: An integer representing the first architecture in the list of compiled architectures.
    - `rest`: A variadic template parameter representing the rest of the compiled architectures.
- **Control Flow**:
    - The function is defined as a constexpr, meaning it is evaluated at compile time.
    - It checks if the given architecture matches the first architecture in the list.
    - If it matches, it returns true.
    - If it does not match, it recursively calls itself with the rest of the architectures.
    - If no match is found, it eventually returns false.
- **Output**: A boolean value indicating whether the given architecture is supported by the compiled code.


---
### ggml\_cuda\_has\_arch
The `ggml_cuda_has_arch` function checks if a given CUDA architecture is supported by the compiled code.
- **Inputs**:
    - `arch`: An integer representing the CUDA architecture to check for support.
- **Control Flow**:
    - The function calls `ggml_cuda_has_arch_impl` with the provided `arch` and the list of compiled architectures `__CUDA_ARCH_LIST__`.
    - The `ggml_cuda_has_arch_impl` function checks if the `arch` matches any of the architectures in the list using recursion.
- **Output**: A boolean value indicating whether the specified CUDA architecture is supported by the compiled code.


---
### ggml\_cuda\_highest\_compiled\_arch\_impl
The function `ggml_cuda_highest_compiled_arch_impl` determines the highest CUDA architecture version that the code was compiled for, which is less than or equal to a specified architecture version.
- **Inputs**:
    - `arch`: An integer representing the target CUDA architecture version to check against.
    - `cur`: An integer representing the current highest compiled architecture version found so far.
    - `first`: An integer representing the first architecture version in the list of architectures to check.
    - `rest`: A variadic list of integers representing the remaining architecture versions to check.
- **Control Flow**:
    - If `cur` is 0, the function aborts with an error message indicating no compatible CUDA architecture was compiled.
    - If `first` is less than or equal to `arch` and greater than `cur`, the function recursively calls itself with `first` as the new `cur`.
    - Otherwise, the function recursively calls itself with the current `cur` and the rest of the architecture versions.
- **Output**: The function returns an integer representing the highest compiled CUDA architecture version that is less than or equal to the specified `arch`.


---
### ggml\_cuda\_highest\_compiled\_arch
The function `ggml_cuda_highest_compiled_arch` determines the highest CUDA architecture version that the code was compiled for, up to a specified architecture version.
- **Inputs**:
    - `arch`: An integer representing the maximum CUDA architecture version to consider.
- **Control Flow**:
    - The function checks if `__CUDA_ARCH_LIST__` is defined, which indicates that multiple CUDA architectures were specified during compilation.
    - If `__CUDA_ARCH_LIST__` is defined, it uses a recursive template function `ggml_cuda_highest_compiled_arch_impl` to iterate over the list of architectures and find the highest one that is less than or equal to the specified `arch`.
    - If `__CUDA_ARCH_LIST__` is not defined, the function simply returns the input `arch` as the highest compiled architecture.
- **Output**: An integer representing the highest CUDA architecture version that the code was compiled for, up to the specified `arch`.


---
### ggml\_cuda\_error
The `ggml_cuda_error` function is a noreturn function that handles CUDA errors by printing an error message and terminating the program.
- **Inputs**:
    - `stmt`: A string representing the CUDA statement that caused the error.
    - `func`: A string representing the name of the function where the error occurred.
    - `file`: A string representing the name of the file where the error occurred.
    - `line`: An integer representing the line number in the file where the error occurred.
    - `msg`: A string containing a custom error message describing the error.
- **Control Flow**:
    - The function is marked with the `[[noreturn]]` attribute, indicating it does not return to the caller.
    - It is intended to be called when a CUDA error is detected, typically through a macro that checks CUDA function return values.
    - The function prints an error message to the standard output, including the file name, line number, function name, and a custom error message.
    - The function then terminates the program, ensuring that execution does not continue after a critical error.
- **Output**: This function does not return any value as it is designed to terminate the program upon execution.


---
### cublas\_get\_error\_str
The `cublas_get_error_str` function returns a string representation of a cuBLAS error code.
- **Inputs**:
    - `err`: A `cublasStatus_t` error code representing the cuBLAS error.
- **Control Flow**:
    - The function checks if the CUDART_VERSION is 12000 or higher, or if GGML_USE_MUSA is defined.
    - If the condition is true, it returns the error string using `cublasGetStatusString(err)`.
    - If the condition is false, it uses a switch statement to match the error code to a predefined string representation of the error.
    - If the error code does not match any predefined cases, it returns 'unknown error'.
- **Output**: A constant character pointer to a string describing the cuBLAS error.


---
### cu\_get\_error\_str
The `cu_get_error_str` function retrieves a human-readable error string for a given CUDA error code.
- **Inputs**:
    - `err`: A `CUresult` type representing the CUDA error code for which the error string is to be retrieved.
- **Control Flow**:
    - The function calls `cuGetErrorString` with the provided error code `err` and a pointer to a `const char *` to store the error string.
    - The function returns the error string obtained from `cuGetErrorString`.
- **Output**: A `const char *` representing the human-readable error string corresponding to the given CUDA error code.


---
### fp16\_available
The `fp16_available` function checks if the highest compiled CUDA architecture for a given compute capability (cc) is at least Pascal, indicating support for FP16 operations.
- **Inputs**:
    - `cc`: An integer representing the compute capability of a CUDA device.
- **Control Flow**:
    - The function calls `ggml_cuda_highest_compiled_arch` with the input `cc` to determine the highest compiled architecture.
    - It compares the result with `GGML_CUDA_CC_PASCAL` to check if the architecture is at least Pascal.
    - The function returns `true` if the architecture is at least Pascal, otherwise it returns `false`.
- **Output**: A boolean value indicating whether FP16 operations are available for the given compute capability.


---
### fast\_fp16\_available
The `fast_fp16_available` function checks if fast FP16 operations are available on a given GPU architecture.
- **Inputs**:
    - `cc`: An integer representing the compute capability of the GPU.
- **Control Flow**:
    - The function first checks if the GPU is an NVIDIA GPU and if FP16 is available, excluding compute capability 610.
    - If the above condition is met, it returns true.
    - If the GPU is an AMD GPU, it also returns true.
    - If neither condition is met, it returns false.
- **Output**: A boolean value indicating whether fast FP16 operations are available for the given compute capability.


---
### fast\_fp16\_hardware\_available
The function `fast_fp16_hardware_available` checks if fast FP16 hardware support is available for a given compute capability.
- **Inputs**:
    - `cc`: An integer representing the compute capability of the GPU.
- **Control Flow**:
    - The function checks if the compute capability (cc) is for an NVIDIA GPU and is greater than or equal to the Pascal architecture (600) and not equal to 610.
    - Alternatively, it checks if the compute capability is for an AMD GPU.
- **Output**: A boolean value indicating whether fast FP16 hardware support is available for the given compute capability.


---
### fp16\_mma\_available
The `fp16_mma_available` function checks if FP16 matrix-multiply-accumulate (MMA) instructions are available for a given compute capability (cc).
- **Inputs**:
    - `cc`: An integer representing the compute capability of the GPU.
- **Control Flow**:
    - The function first checks if the code is compiled with HIP and for AMD platforms without ROCWMMA FATTN support, returning false if true.
    - If not, it checks if the compute capability is for NVIDIA and if the highest compiled architecture is at least Volta, returning true if so.
    - It also checks if the compute capability is for CDNA, RDNA3, or RDNA4 architectures, returning true if any of these conditions are met.
- **Output**: A boolean value indicating whether FP16 MMA instructions are available for the given compute capability.


---
### fp16\_mma\_hardware\_available
The function `fp16_mma_hardware_available` checks if FP16 matrix-multiply-accumulate (MMA) hardware is available for a given compute capability (cc).
- **Inputs**:
    - `cc`: An integer representing the compute capability of the GPU.
- **Control Flow**:
    - The function checks if the compute capability (cc) belongs to NVIDIA and is greater than or equal to Volta (700).
    - If the above condition is true, it returns true, indicating FP16 MMA hardware is available.
    - If the compute capability (cc) belongs to AMD and is part of the CDNA, RDNA3, or RDNA4 architectures, it also returns true.
    - If none of the above conditions are met, it returns false, indicating FP16 MMA hardware is not available.
- **Output**: A boolean value indicating whether FP16 MMA hardware is available for the specified compute capability.


---
### new\_mma\_available
The `new_mma_available` function checks if the new matrix multiply-accumulate (MMA) instructions are available for a given NVIDIA GPU compute capability.
- **Inputs**:
    - `cc`: An integer representing the compute capability of the NVIDIA GPU.
- **Control Flow**:
    - The function checks if the compute capability (cc) is for an NVIDIA GPU using the macro `GGML_CUDA_CC_IS_NVIDIA`.
    - It then checks if the highest compiled architecture for the given compute capability is greater than or equal to `GGML_CUDA_CC_TURING`.
    - If both conditions are met, the function returns true, indicating that new MMA instructions are available.
- **Output**: A boolean value indicating whether new MMA instructions are available for the given compute capability.


---
### cp\_async\_available
The `cp_async_available` function checks if the `cp.async` feature is available for a given CUDA compute capability.
- **Inputs**:
    - `cc`: An integer representing the CUDA compute capability of the device.
- **Control Flow**:
    - The function checks if the compute capability (cc) is less than the offset for AMD devices (GGML_CUDA_CC_OFFSET_AMD).
    - It then checks if the highest compiled architecture for the given compute capability is greater than or equal to the Ampere architecture (GGML_CUDA_CC_AMPERE).
    - If both conditions are met, the function returns true, indicating that `cp.async` is available; otherwise, it returns false.
- **Output**: A boolean value indicating whether the `cp.async` feature is available for the given compute capability.


---
### ggml\_cuda\_get\_physical\_warp\_size
The function `ggml_cuda_get_physical_warp_size` returns the physical warp size for CUDA or HIP platforms.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the code is being compiled for the HIP platform with AMD GPU support.
    - If the HIP platform with AMD GPU is detected, it returns the wavefront size specific to AMD GPUs.
    - If not, it defaults to returning a warp size of 32, which is standard for NVIDIA CUDA platforms.
- **Output**: The function returns an integer representing the physical warp size, which is 32 for CUDA and the wavefront size for AMD HIP.


---
### no\_device\_code
The `no_device_code` function is a CUDA or HIP device function that prints an error message indicating that a kernel has no compatible device code for a given architecture and then triggers a trap to halt execution.
- **Inputs**:
    - `file_name`: A constant character pointer representing the name of the file where the error occurred.
    - `line`: An integer representing the line number in the file where the error occurred.
    - `function_name`: A constant character pointer representing the name of the function where the error occurred.
    - `arch`: An integer representing the architecture version for which the device code is not available.
    - `arch_list`: A constant character pointer representing the list of architectures for which the code was compiled.
- **Control Flow**:
    - The function checks if the code is being compiled for HIP and AMD platforms or CUDA.
    - It prints an error message using `printf`, indicating the file, line, function, and architecture details.
    - For CUDA, it also includes the list of architectures for which the code was compiled.
    - The function then calls `__trap()` to halt execution, indicating a critical error.
    - If compiled with MUSA, it uses `__builtin_unreachable()` to indicate that the code should not be reached.
- **Output**: The function does not return any value as it is marked with `[[noreturn]]`, indicating it will not return to the caller.


---
### warp\_reduce\_sum
The `warp_reduce_sum` function performs a warp-level reduction to compute the sum of elements across threads within a warp in CUDA.
- **Inputs**:
    - `x`: An integer, float, float2, or half2 value representing the data to be summed across the warp.
- **Control Flow**:
    - The function checks if the architecture supports the `__reduce_add_sync` intrinsic for efficient reduction; if so, it uses this intrinsic to perform the sum.
    - If the architecture does not support `__reduce_add_sync`, the function uses a loop with `__shfl_xor_sync` to perform the reduction by iteratively summing values from different threads within the warp.
    - The loop iterates, halving the offset each time, until the offset is zero, effectively summing all values within the warp.
- **Output**: The function returns the sum of the input values across all threads in the warp.


---
### ggml\_cuda\_hmax
The `ggml_cuda_hmax` function computes the maximum of two half-precision floating-point numbers on CUDA-enabled devices.
- **Inputs**:
    - `a`: A half-precision floating-point number (half) to be compared.
    - `b`: Another half-precision floating-point number (half) to be compared.
- **Control Flow**:
    - Check if FP16 (half-precision floating-point) operations are available.
    - If using HIP on AMD platform and CUDA version is less than 11.7, convert both half-precision numbers to float, compute the maximum, and convert back to half.
    - Otherwise, use the intrinsic `__hmax` function to compute the maximum of the two half-precision numbers.
    - If FP16 is not available, trigger a device code error and return the first input as a fallback.
- **Output**: Returns the maximum of the two input half-precision floating-point numbers.


---
### ggml\_cuda\_hmax2
The `ggml_cuda_hmax2` function computes the element-wise maximum of two `half2` data types on CUDA-enabled devices.
- **Inputs**:
    - `a`: A `half2` data type representing the first input vector.
    - `b`: A `half2` data type representing the second input vector.
- **Control Flow**:
    - The function checks if the HIP platform is used and the HIP version is 5.7 or higher, in which case it uses the `__hmax` function for each component of the `half2` type.
    - If the HIP platform is not used and the CUDA runtime version is 11.7 or higher, it uses the `__hmax2` intrinsic to compute the maximum of the two `half2` inputs.
    - If neither of the above conditions are met, it manually computes the maximum for each component of the `half2` type by converting them to `float`, using `fmaxf`, and then converting back to `half`.
    - If none of the conditions are met, it calls `NO_DEVICE_CODE` to handle the case where no device code is available.
- **Output**: A `half2` data type representing the element-wise maximum of the two input `half2` vectors.


---
### ggml\_cuda\_dp4a
The `ggml_cuda_dp4a` function performs a byte-wise dot product of two 32-bit integers and adds the result to a third integer, optimized for different GPU architectures.
- **Inputs**:
    - `a`: A 32-bit integer representing the first operand for the dot product.
    - `b`: A 32-bit integer representing the second operand for the dot product.
    - `c`: A 32-bit integer to which the result of the dot product will be added.
- **Control Flow**:
    - Check if the code is compiled for AMD's HIP platform and specific architectures, then use AMD-specific intrinsics or inline assembly for the dot product.
    - If not on AMD, check if the CUDA architecture supports the `__dp4a` intrinsic, and use it if available.
    - If neither condition is met, manually compute the dot product using byte-wise multiplication and addition.
- **Output**: Returns an integer which is the result of the byte-wise dot product of `a` and `b`, added to `c`.


---
### get\_alibi\_slope
The `get_alibi_slope` function calculates a slope value based on the maximum bias, head index, number of heads, and two base values.
- **Inputs**:
    - `max_bias`: A float representing the maximum bias value, which influences the slope calculation.
    - `h`: An unsigned integer representing the current head index.
    - `n_head_log2`: An unsigned integer representing the logarithm base 2 of the number of heads.
    - `m0`: A float representing the base value used when the head index is less than the logarithm of the number of heads.
    - `m1`: A float representing the base value used when the head index is greater than or equal to the logarithm of the number of heads.
- **Control Flow**:
    - Check if max_bias is less than or equal to 0.0f; if true, return 1.0f as the slope.
    - Determine the base value to use: if the head index h is less than n_head_log2, use m0; otherwise, use m1.
    - Calculate the exponent: if h is less than n_head_log2, set exph to h + 1; otherwise, set exph to 2*(h - n_head_log2) + 1.
    - Return the result of raising the base to the power of exph using the powf function.
- **Output**: The function returns a float representing the calculated slope value based on the input parameters.


---
### ggml\_cuda\_info
The `ggml_cuda_info` function retrieves and returns information about the CUDA devices available on the system.
- **Inputs**: None
- **Control Flow**:
    - The function is defined to return a constant reference to a `ggml_cuda_device_info` structure.
    - The `ggml_cuda_device_info` structure contains information about the number of CUDA devices and details for each device, such as compute capability, number of streaming multiprocessors, shared memory per block, and other device-specific attributes.
    - The function does not take any parameters and is likely implemented elsewhere to populate and return the `ggml_cuda_device_info` structure.
- **Output**: A constant reference to a `ggml_cuda_device_info` structure containing details about the CUDA devices.


---
### ggml\_cuda\_set\_device
The `ggml_cuda_set_device` function sets the current CUDA device to the specified device index.
- **Inputs**:
    - `device`: An integer representing the index of the CUDA device to be set as the current device.
- **Control Flow**:
    - The function takes an integer input representing the device index.
    - It calls the CUDA API function to set the current device to the specified index.
- **Output**: The function does not return any value.


---
### ggml\_cuda\_get\_device
The `ggml_cuda_get_device` function retrieves the current CUDA device being used.
- **Inputs**: None
- **Control Flow**:
    - The function does not take any input parameters.
    - It directly returns the current CUDA device index.
- **Output**: The function returns an integer representing the current CUDA device index.


---
### ggml\_backend\_cuda\_context::stream
The `ggml_backend_cuda_context::stream` function retrieves or creates a CUDA stream for a specified device and stream index, or defaults to the primary stream for the current device.
- **Inputs**:
    - `device`: An integer representing the device index for which the CUDA stream is to be retrieved or created.
    - `stream`: An integer representing the stream index for the specified device.
- **Control Flow**:
    - Checks if the CUDA stream for the specified device and stream index is already created.
    - If not created, sets the current device to the specified device using `ggml_cuda_set_device`.
    - Creates a new CUDA stream with non-blocking flags using `cudaStreamCreateWithFlags`.
    - Returns the CUDA stream for the specified device and stream index.
- **Output**: Returns a `cudaStream_t` object representing the CUDA stream for the specified device and stream index.


---
### ggml\_backend\_cuda\_context::cublas\_handle
The `cublas_handle` function in the `ggml_backend_cuda_context` class initializes and returns a cuBLAS handle for a specified CUDA device, ensuring that the handle is created and configured if it does not already exist.
- **Inputs**:
    - `device`: An integer representing the CUDA device for which the cuBLAS handle is requested.
- **Control Flow**:
    - Check if the cuBLAS handle for the specified device is null.
    - If the handle is null, set the current CUDA device to the specified device using `ggml_cuda_set_device`.
    - Create a new cuBLAS handle using `cublasCreate` and store it in the `cublas_handles` array for the specified device.
    - Set the math mode of the cuBLAS handle to `CUBLAS_TF32_TENSOR_OP_MATH` using `cublasSetMathMode`.
    - Return the cuBLAS handle for the specified device.
- **Output**: Returns a `cublasHandle_t`, which is a handle to the cuBLAS library context for the specified CUDA device.


---
### ggml\_backend\_cuda\_context::pool
The `ggml_backend_cuda_context::pool` function manages memory allocation and deallocation for CUDA devices using a pool allocator.
- **Inputs**: None
- **Control Flow**:
    - The function checks if a memory pool for the specified device exists.
    - If the pool does not exist, it creates a new pool for the device using `new_pool_for_device`.
    - The function returns a reference to the memory pool for the specified device.
- **Output**: A reference to a `ggml_cuda_pool` object for the specified CUDA device.


