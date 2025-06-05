# Purpose
This C header file serves as a compatibility layer between CUDA and HIP (Heterogeneous-Compute Interface for Portability), specifically targeting the AMD platform. It provides a series of macro definitions that map CUDA API calls and data types to their HIP equivalents. This allows code originally written for CUDA to be compiled and executed on AMD GPUs using the HIP runtime. The file includes conditional compilation directives to ensure compatibility with different versions of the HIP platform and specific GPU architectures, such as GCN, CDNA, RDNA1, RDNA2, RDNA3, and RDNA4. It also includes inline device functions for vector operations, which are optimized for execution on the GPU.

The file is structured to facilitate the transition of CUDA-based applications to the HIP environment by redefining CUDA-specific constructs to their HIP counterparts. This includes data types, memory management functions, and BLAS (Basic Linear Algebra Subprograms) operations. The use of `#pragma once` ensures that the file is included only once per compilation unit, preventing redefinition errors. The file is not an executable but rather a header file intended to be included in other source files, providing a seamless interface for developers to write code that can run on both NVIDIA and AMD hardware with minimal changes.
# Imports and Dependencies

---
- `hip/hip_runtime.h`
- `hipblas/hipblas.h`
- `hip/hip_fp16.h`
- `hip/hip_bfloat16.h`
- `rocblas/rocblas.h`


# Data Structures

---
### half2\_b32\_t
- **Type**: `union`
- **Members**:
    - `val`: A member of type `half2` that represents a 16-bit floating-point vector.
    - `b32`: An integer member that provides a 32-bit representation of the `half2` value.
- **Description**: The `half2_b32_t` is a union data structure that allows for the representation of a `half2` type, which is a 16-bit floating-point vector, as a 32-bit integer. This union facilitates operations that require both the floating-point and integer representations of the data, enabling efficient manipulation and conversion between these two forms. It is particularly useful in GPU programming where such conversions are common for performance optimization.


# Functions

---
### \_\_vsubss4<!-- {{#callable:__vsubss4}} -->
The `__vsubss4` function performs element-wise saturated subtraction on two 32-bit integers interpreted as vectors of four 8-bit integers.
- **Inputs**:
    - `a`: A 32-bit integer representing a vector of four 8-bit integers.
    - `b`: A 32-bit integer representing a vector of four 8-bit integers.
- **Control Flow**:
    - The function begins by casting the input integers `a` and `b` to vectors of four 8-bit integers (`int8x4_t`).
    - It checks if the built-in function `__builtin_elementwise_sub_sat` is available.
    - If available, it uses `__builtin_elementwise_sub_sat` to perform element-wise saturated subtraction on the vectors `va` and `vb`, storing the result in `c`.
    - If the built-in function is not available, it manually performs element-wise subtraction in a loop for each of the four elements.
    - During manual subtraction, it checks if the result exceeds the 8-bit integer limits and clamps the result to the maximum or minimum 8-bit integer value if necessary.
    - Finally, it casts the resulting vector `c` back to an integer and returns it.
- **Output**: A 32-bit integer representing the result of element-wise saturated subtraction of the input vectors.


---
### \_\_vsub4<!-- {{#callable:__vsub4}} -->
The `__vsub4` function performs a vectorized subtraction of two 32-bit integers, treating them as vectors of four 8-bit integers, and returns the result.
- **Inputs**:
    - `a`: A 32-bit integer representing a vector of four 8-bit integers.
    - `b`: A 32-bit integer representing a vector of four 8-bit integers.
- **Control Flow**:
    - The function calls [`__vsubss4`](#__vsubss4) with the inputs `a` and `b`.
    - [`__vsubss4`](#__vsubss4) interprets `a` and `b` as vectors of four 8-bit integers each.
    - It performs element-wise subtraction of the vectors, with saturation to handle overflow and underflow.
    - The result is reinterpreted as a 32-bit integer and returned.
- **Output**: A 32-bit integer representing the result of the element-wise saturated subtraction of the input vectors.
- **Functions called**:
    - [`__vsubss4`](#__vsubss4)


---
### \_\_vcmpeq4<!-- {{#callable:__vcmpeq4}} -->
The `__vcmpeq4` function compares two 32-bit unsigned integers as vectors of four 8-bit unsigned integers and returns a 32-bit unsigned integer where each byte is 0xFF if the corresponding bytes in the inputs are equal, otherwise 0x00.
- **Inputs**:
    - `a`: A 32-bit unsigned integer to be compared, interpreted as a vector of four 8-bit unsigned integers.
    - `b`: A 32-bit unsigned integer to be compared, interpreted as a vector of four 8-bit unsigned integers.
- **Control Flow**:
    - The function begins by casting the input integers `a` and `b` to `uint8x4_t` types, which are vectors of four 8-bit unsigned integers.
    - An uninitialized 32-bit unsigned integer `c` is declared, and a reference `vc` to it is created as a `uint8x4_t` type.
    - A loop iterates over the four elements of the vectors `va` and `vb`.
    - For each element, it checks if the corresponding elements in `va` and `vb` are equal.
    - If they are equal, the corresponding element in `vc` is set to 0xFF; otherwise, it is set to 0x00.
    - The loop is unrolled to optimize performance.
    - Finally, the function returns the integer `c`, which now contains the comparison results.
- **Output**: A 32-bit unsigned integer where each byte is 0xFF if the corresponding bytes in the inputs are equal, otherwise 0x00.


---
### \_\_vcmpne4<!-- {{#callable:__vcmpne4}} -->
The `__vcmpne4` function compares two 32-bit unsigned integers as vectors of four 8-bit unsigned integers and returns a 32-bit unsigned integer where each byte is 0xff if the corresponding bytes in the inputs are not equal, and 0x00 otherwise.
- **Inputs**:
    - `a`: A 32-bit unsigned integer to be compared, interpreted as a vector of four 8-bit unsigned integers.
    - `b`: A 32-bit unsigned integer to be compared, interpreted as a vector of four 8-bit unsigned integers.
- **Control Flow**:
    - The function begins by casting the input integers `a` and `b` to `uint8x4_t` types, which are vectors of four 8-bit unsigned integers.
    - A local variable `c` is declared to store the result, and it is also cast to a `uint8x4_t` type `vc`.
    - A loop iterates over the four elements of the vectors `va` and `vb`.
    - For each element, it checks if the corresponding elements of `va` and `vb` are equal.
    - If they are equal, the corresponding element in `vc` is set to 0x00; otherwise, it is set to 0xff.
    - The loop is unrolled to optimize performance.
    - Finally, the function returns the integer `c`, which now contains the comparison results.
- **Output**: A 32-bit unsigned integer where each byte is 0xff if the corresponding bytes in the inputs are not equal, and 0x00 otherwise.


---
### \_\_shfl\_xor<!-- {{#callable:__shfl_xor}} -->
The `__shfl_xor` function performs a bitwise XOR shuffle operation on a `half2` data type across threads in a warp.
- **Inputs**:
    - `var`: A `half2` variable representing the data to be shuffled.
    - `laneMask`: An integer mask used to determine which lanes to XOR with.
    - `width`: An integer specifying the width of the warp, which is the number of threads participating in the shuffle operation.
- **Control Flow**:
    - The function defines a union `half2_b32_t` to facilitate conversion between `half2` and `int` types.
    - The input `half2` variable `var` is stored in the union `tmp` as `val`.
    - The `int` representation of `var` (`tmp.b32`) is shuffled using the `__shfl_xor` intrinsic function with the specified `laneMask` and `width`.
    - The shuffled `int` value is converted back to `half2` and returned as the result.
- **Output**: The function returns a `half2` value that is the result of the XOR shuffle operation across the specified lanes in the warp.


