# Purpose
This C header file is an internal component of the GGML library, specifically designed for CPU-based operations. It provides a collection of macros, inline functions, and type definitions that facilitate vectorized computations across various CPU architectures, including ARM NEON, x86 with AVX/AVX2/AVX512, and others like RISC-V and Power9. The file includes compatibility layers and workarounds for different compilers and platforms, such as MSVC and MinGW, ensuring that the library can leverage hardware-specific optimizations for performance improvements. The header is not intended to be a public API but rather a utility to support the internal workings of the GGML library's CPU computations.

The file defines several inline functions for vector operations, such as addition, multiplication, and loading of vector data, which are crucial for high-performance numerical computations. It also includes conditional compilation directives to handle different CPU features and instruction sets, ensuring that the code can be compiled and run efficiently on a wide range of hardware. The use of `#pragma once` indicates that this header is designed to be included only once per compilation unit, preventing multiple inclusions. Overall, this file is a specialized component that enhances the GGML library's ability to perform efficient, parallel computations on CPUs by abstracting the complexities of SIMD (Single Instruction, Multiple Data) operations across different architectures.
# Imports and Dependencies

---
- `ggml.h`
- `ggml-impl.h`
- `stdlib.h`
- `stdbool.h`
- `string.h`
- `math.h`
- `sys/prctl.h`
- `wasm_simd128.h`
- `altivec.h`
- `intrin.h`
- `immintrin.h`
- `riscv_vector.h`
- `lasxintrin.h`
- `lsxintrin.h`
- `vecintrin.h`


# Data Structures

---
### ggml\_compute\_params
- **Type**: `struct`
- **Members**:
    - `ith`: Represents the index of the current thread.
    - `nth`: Indicates the total number of threads available.
    - `wsize`: Specifies the size of the work buffer shared among all threads.
    - `wdata`: Points to the work buffer data used by all threads.
    - `threadpool`: Points to a thread pool structure for managing threads.
- **Description**: The `ggml_compute_params` structure is designed to manage and coordinate multi-threaded computations. It includes fields for tracking the current thread index (`ith`) and the total number of threads (`nth`), which are essential for parallel processing. The structure also provides a work buffer (`wdata`) of a specified size (`wsize`) that is shared among all threads, facilitating efficient data handling during computations. Additionally, it includes a pointer to a `ggml_threadpool` structure, which is used to manage the pool of threads, ensuring that resources are allocated and utilized effectively during parallel operations.


---
### ggml\_int16x8x2\_t
- **Type**: `struct`
- **Members**:
    - `val`: An array of two int16x8_t vectors.
- **Description**: The `ggml_int16x8x2_t` structure is a compound data type that encapsulates an array of two `int16x8_t` vectors. This structure is likely used to handle operations on 16-bit integer vectors, providing a convenient way to manage and manipulate two such vectors simultaneously. It is part of a larger set of vector operations, possibly for SIMD (Single Instruction, Multiple Data) processing, which is common in performance-critical applications such as graphics or scientific computations.


---
### ggml\_uint8x16x2\_t
- **Type**: `struct`
- **Members**:
    - `val`: An array of two 128-bit vectors of unsigned 8-bit integers.
- **Description**: The `ggml_uint8x16x2_t` structure is a compound data type that encapsulates two 128-bit vectors, each containing 16 unsigned 8-bit integers. This structure is typically used in SIMD (Single Instruction, Multiple Data) operations to efficiently process data in parallel, leveraging the capabilities of modern processors to handle multiple data points simultaneously. The `val` array holds these two vectors, allowing for operations that require handling of 32 bytes of data at once.


---
### ggml\_uint8x16x4\_t
- **Type**: `struct`
- **Members**:
    - `val`: An array of four 128-bit vectors of unsigned 8-bit integers.
- **Description**: The `ggml_uint8x16x4_t` structure is designed to hold four 128-bit vectors, each containing 16 unsigned 8-bit integers. This structure is useful for operations that require processing multiple vectors of 8-bit data simultaneously, leveraging SIMD (Single Instruction, Multiple Data) capabilities for efficient computation. It is typically used in environments where vectorized operations on byte data are needed, such as image processing or data compression tasks.


---
### ggml\_int8x16x2\_t
- **Type**: `struct`
- **Members**:
    - `val`: An array of two int8x16_t elements, representing 128-bit vectors of 8-bit integers.
- **Description**: The `ggml_int8x16x2_t` structure is a compound data type designed to hold two 128-bit vectors of 8-bit integers, encapsulated in an array. This structure is likely used in scenarios where operations on 128-bit integer vectors are required, such as SIMD (Single Instruction, Multiple Data) operations, which are common in performance-critical applications like graphics processing or scientific computations. The use of two vectors allows for efficient parallel processing of data.


---
### ggml\_int8x16x4\_t
- **Type**: `struct`
- **Members**:
    - `val`: An array of four int8x16_t vectors.
- **Description**: The `ggml_int8x16x4_t` structure is a compound data type that encapsulates an array of four `int8x16_t` vectors. This structure is likely used to handle operations on multiple 128-bit vectors of 8-bit integers, which can be useful in SIMD (Single Instruction, Multiple Data) operations for efficient parallel processing. The structure provides a convenient way to manage and pass around these vectors as a single entity.


# Functions

---
### vaddlvq\_s16<!-- {{#callable:vaddlvq_s16}} -->
The `vaddlvq_s16` function performs a horizontal addition of all elements in a 16-bit integer vector and returns the result as a 32-bit integer.
- **Inputs**:
    - `v`: A vector of eight 16-bit integers (`int16x8_t`).
- **Control Flow**:
    - The function first applies `vpaddlq_s16` to the input vector `v`, which performs a pairwise addition of adjacent elements, resulting in a vector of four 32-bit integers.
    - The resulting vector is then passed to `vpaddlq_s32`, which again performs a pairwise addition, resulting in a vector of two 64-bit integers.
    - The `vreinterpretq_s32_s64` function is used to reinterpret the 64-bit integer vector as a 32-bit integer vector.
    - Finally, the function retrieves the first and third elements of the 32-bit vector using `vgetq_lane_s32` and returns their sum.
- **Output**: A 32-bit integer representing the sum of all elements in the input vector.


---
### vpaddq\_s16<!-- {{#callable:vpaddq_s16}} -->
The `vpaddq_s16` function performs pairwise addition of two 16-bit integer vectors and combines the results into a single vector.
- **Inputs**:
    - `a`: A 128-bit vector of eight 16-bit integers (int16x8_t).
    - `b`: A 128-bit vector of eight 16-bit integers (int16x8_t).
- **Control Flow**:
    - Extracts the lower and upper halves of vector 'a' using `vget_low_s16` and `vget_high_s16`, respectively.
    - Performs pairwise addition on the extracted halves of 'a' using `vpadd_s16`, resulting in a 64-bit vector 'a0'.
    - Extracts the lower and upper halves of vector 'b' using `vget_low_s16` and `vget_high_s16`, respectively.
    - Performs pairwise addition on the extracted halves of 'b' using `vpadd_s16`, resulting in a 64-bit vector 'b0'.
    - Combines the results 'a0' and 'b0' into a single 128-bit vector using `vcombine_s16`.
- **Output**: A 128-bit vector of eight 16-bit integers (int16x8_t) containing the pairwise addition results of the input vectors.


---
### vpaddq\_s32<!-- {{#callable:vpaddq_s32}} -->
The `vpaddq_s32` function performs a pairwise addition of two 128-bit vectors of 32-bit integers and combines the results into a single 128-bit vector.
- **Inputs**:
    - `a`: A 128-bit vector of four 32-bit integers.
    - `b`: A 128-bit vector of four 32-bit integers.
- **Control Flow**:
    - Extracts the lower and upper halves of vector 'a' using `vget_low_s32` and `vget_high_s32` respectively.
    - Performs pairwise addition on the extracted halves of 'a' using `vpadd_s32`, resulting in a 64-bit vector 'a0'.
    - Extracts the lower and upper halves of vector 'b' using `vget_low_s32` and `vget_high_s32` respectively.
    - Performs pairwise addition on the extracted halves of 'b' using `vpadd_s32`, resulting in a 64-bit vector 'b0'.
    - Combines the results 'a0' and 'b0' into a single 128-bit vector using `vcombine_s32`.
- **Output**: A 128-bit vector of four 32-bit integers, where each pair of integers from the input vectors 'a' and 'b' has been added together.


---
### vaddvq\_s32<!-- {{#callable:vaddvq_s32}} -->
The function `vaddvq_s32` computes the sum of all elements in a 4-element vector of 32-bit integers.
- **Inputs**:
    - `v`: A 4-element vector of 32-bit integers (int32x4_t).
- **Control Flow**:
    - Retrieve the first element of the vector using `vgetq_lane_s32(v, 0)`.
    - Retrieve the second element of the vector using `vgetq_lane_s32(v, 1)`.
    - Retrieve the third element of the vector using `vgetq_lane_s32(v, 2)`.
    - Retrieve the fourth element of the vector using `vgetq_lane_s32(v, 3)`.
    - Sum all four retrieved elements together.
- **Output**: The function returns the sum of the four 32-bit integer elements in the input vector.


---
### vaddvq\_f32<!-- {{#callable:vaddvq_f32}} -->
The function `vaddvq_f32` computes the sum of all elements in a 4-element vector of single-precision floating-point numbers.
- **Inputs**:
    - `v`: A vector of type `float32x4_t` containing four single-precision floating-point numbers.
- **Control Flow**:
    - The function retrieves the first element of the vector `v` using `vgetq_lane_f32(v, 0)`.
    - It retrieves the second element of the vector `v` using `vgetq_lane_f32(v, 1)`.
    - It retrieves the third element of the vector `v` using `vgetq_lane_f32(v, 2)`.
    - It retrieves the fourth element of the vector `v` using `vgetq_lane_f32(v, 3)`.
    - The function adds all four retrieved elements together and returns the result.
- **Output**: A single `float` representing the sum of the four elements in the input vector.


---
### vmaxvq\_f32<!-- {{#callable:vmaxvq_f32}} -->
The `vmaxvq_f32` function computes the maximum value from a 4-element vector of 32-bit floating-point numbers.
- **Inputs**:
    - `v`: A vector of four 32-bit floating-point numbers (float32x4_t).
- **Control Flow**:
    - Retrieve the first element of the vector using `vgetq_lane_f32(v, 0)`.
    - Retrieve the second element of the vector using `vgetq_lane_f32(v, 1)`.
    - Compute the maximum of the first two elements using `MAX`.
    - Retrieve the third element of the vector using `vgetq_lane_f32(v, 2)`.
    - Retrieve the fourth element of the vector using `vgetq_lane_f32(v, 3)`.
    - Compute the maximum of the third and fourth elements using `MAX`.
    - Compute the maximum of the two previously computed maximums to get the overall maximum value.
- **Output**: The function returns the maximum value among the four elements of the input vector as a float.


---
### vcvtnq\_s32\_f32<!-- {{#callable:vcvtnq_s32_f32}} -->
The function `vcvtnq_s32_f32` converts a vector of four 32-bit floating-point numbers to a vector of four 32-bit integers by rounding each floating-point number to the nearest integer.
- **Inputs**:
    - `v`: A vector of four 32-bit floating-point numbers (float32x4_t).
- **Control Flow**:
    - Initialize an integer vector `res` to store the results.
    - For each element in the input vector `v`, retrieve the floating-point value using `vgetq_lane_f32`.
    - Round each retrieved floating-point value to the nearest integer using `roundf`.
    - Store the rounded integer value in the corresponding position in the result vector `res`.
    - Return the result vector `res`.
- **Output**: A vector of four 32-bit integers (int32x4_t) where each element is the rounded integer value of the corresponding element in the input vector.


---
### vzip1\_u8<!-- {{#callable:vzip1_u8}} -->
The `vzip1_u8` function interleaves the first four elements of two 8-element vectors of unsigned 8-bit integers.
- **Inputs**:
    - `a`: An 8-element vector of unsigned 8-bit integers.
    - `b`: Another 8-element vector of unsigned 8-bit integers.
- **Control Flow**:
    - Initialize a result vector `res` of type `uint8x8_t`.
    - Assign the first element of `res` to the first element of `a`.
    - Assign the second element of `res` to the first element of `b`.
    - Assign the third element of `res` to the second element of `a`.
    - Assign the fourth element of `res` to the second element of `b`.
    - Assign the fifth element of `res` to the third element of `a`.
    - Assign the sixth element of `res` to the third element of `b`.
    - Assign the seventh element of `res` to the fourth element of `a`.
    - Assign the eighth element of `res` to the fourth element of `b`.
    - Return the interleaved result vector `res`.
- **Output**: A vector of type `uint8x8_t` containing the interleaved first four elements of the input vectors `a` and `b`.


---
### vzip2\_u8<!-- {{#callable:vzip2_u8}} -->
The `vzip2_u8` function interleaves the upper halves of two 8-element vectors of unsigned 8-bit integers.
- **Inputs**:
    - `a`: An 8-element vector of unsigned 8-bit integers.
    - `b`: Another 8-element vector of unsigned 8-bit integers.
- **Control Flow**:
    - Initialize a result vector `res` of type `uint8x8_t`.
    - Assign the 5th to 8th elements of vector `a` to the even indices of `res`.
    - Assign the 5th to 8th elements of vector `b` to the odd indices of `res`.
    - Return the interleaved result vector `res`.
- **Output**: An 8-element vector of unsigned 8-bit integers, containing interleaved elements from the upper halves of the input vectors.


---
### ggml\_vld1q\_s16\_x2<!-- {{#callable:ggml_vld1q_s16_x2}} -->
The function `ggml_vld1q_s16_x2` loads two vectors of 8 int16_t elements each from a given memory address into a structure containing two int16x8_t vectors.
- **Inputs**:
    - `ptr`: A pointer to an array of int16_t elements from which the function will load data.
- **Control Flow**:
    - Declare a variable `res` of type `ggml_int16x8x2_t` to store the result.
    - Load the first 8 int16_t elements from the memory location pointed to by `ptr` into `res.val[0]` using `vld1q_s16`.
    - Load the next 8 int16_t elements from the memory location pointed to by `ptr + 8` into `res.val[1]` using `vld1q_s16`.
    - Return the `res` structure containing the two loaded vectors.
- **Output**: A `ggml_int16x8x2_t` structure containing two int16x8_t vectors loaded from the specified memory location.


---
### ggml\_vld1q\_u8\_x2<!-- {{#callable:ggml_vld1q_u8_x2}} -->
The function `ggml_vld1q_u8_x2` loads two 128-bit vectors of unsigned 8-bit integers from a given memory address.
- **Inputs**:
    - `ptr`: A pointer to a memory location containing unsigned 8-bit integers from which two 128-bit vectors will be loaded.
- **Control Flow**:
    - Declare a variable `res` of type `ggml_uint8x16x2_t` to store the result.
    - Load the first 128-bit vector from the memory location pointed to by `ptr` into `res.val[0]` using `vld1q_u8`.
    - Load the second 128-bit vector from the memory location `ptr + 16` into `res.val[1]` using `vld1q_u8`.
    - Return the `res` structure containing the two loaded vectors.
- **Output**: A `ggml_uint8x16x2_t` structure containing two 128-bit vectors of unsigned 8-bit integers loaded from the specified memory location.


---
### ggml\_vld1q\_u8\_x4<!-- {{#callable:ggml_vld1q_u8_x4}} -->
The function `ggml_vld1q_u8_x4` loads four 128-bit vectors of unsigned 8-bit integers from a given memory address.
- **Inputs**:
    - `ptr`: A pointer to a memory location containing unsigned 8-bit integers from which the function will load data.
- **Control Flow**:
    - Declare a variable `res` of type `ggml_uint8x16x4_t` to store the result.
    - Load the first 128-bit vector from the memory location pointed to by `ptr` and store it in `res.val[0]`.
    - Load the second 128-bit vector from the memory location `ptr + 16` and store it in `res.val[1]`.
    - Load the third 128-bit vector from the memory location `ptr + 32` and store it in `res.val[2]`.
    - Load the fourth 128-bit vector from the memory location `ptr + 48` and store it in `res.val[3]`.
    - Return the `res` structure containing the four loaded vectors.
- **Output**: A `ggml_uint8x16x4_t` structure containing four 128-bit vectors of unsigned 8-bit integers loaded from the specified memory location.


---
### ggml\_vld1q\_s8\_x2<!-- {{#callable:ggml_vld1q_s8_x2}} -->
The function `ggml_vld1q_s8_x2` loads two 16-element vectors of 8-bit signed integers from a given memory address into a structure containing two such vectors.
- **Inputs**:
    - `ptr`: A pointer to the starting memory address from which two 16-element vectors of 8-bit signed integers will be loaded.
- **Control Flow**:
    - Declare a variable `res` of type `ggml_int8x16x2_t` to store the result.
    - Load the first 16-element vector of 8-bit signed integers from the memory address pointed to by `ptr` into `res.val[0]` using `vld1q_s8`.
    - Load the second 16-element vector of 8-bit signed integers from the memory address `ptr + 16` into `res.val[1]` using `vld1q_s8`.
    - Return the `res` structure containing the two loaded vectors.
- **Output**: A `ggml_int8x16x2_t` structure containing two 16-element vectors of 8-bit signed integers loaded from the specified memory address.


---
### ggml\_vld1q\_s8\_x4<!-- {{#callable:ggml_vld1q_s8_x4}} -->
The function `ggml_vld1q_s8_x4` loads four 16-element vectors of 8-bit signed integers from a given memory address into a `ggml_int8x16x4_t` structure.
- **Inputs**:
    - `ptr`: A pointer to the starting memory address of an array of 8-bit signed integers from which the vectors will be loaded.
- **Control Flow**:
    - Declare a variable `res` of type `ggml_int8x16x4_t` to store the result.
    - Load the first 16 elements from the memory address pointed to by `ptr` into `res.val[0]` using `vld1q_s8`.
    - Load the next 16 elements from the memory address (offset by 16) into `res.val[1]`.
    - Load the next 16 elements from the memory address (offset by 32) into `res.val[2]`.
    - Load the next 16 elements from the memory address (offset by 48) into `res.val[3]`.
    - Return the `res` structure containing the four loaded vectors.
- **Output**: A `ggml_int8x16x4_t` structure containing four 16-element vectors of 8-bit signed integers loaded from the specified memory address.


---
### ggml\_vqtbl1q\_s8<!-- {{#callable:ggml_vqtbl1q_s8}} -->
The function `ggml_vqtbl1q_s8` performs a table lookup operation on a vector of 16 signed 8-bit integers using another vector of 16 unsigned 8-bit integers as indices.
- **Inputs**:
    - `a`: A vector of 16 signed 8-bit integers (int8x16_t) that serves as the lookup table.
    - `b`: A vector of 16 unsigned 8-bit integers (uint8x16_t) that provides the indices for the lookup operation.
- **Control Flow**:
    - Initialize a result vector `res` of type int8x16_t.
    - For each index from 0 to 15, assign `res[i]` the value of `a` at the position specified by `b[i]`.
    - Return the result vector `res`.
- **Output**: A vector of 16 signed 8-bit integers (int8x16_t) containing the values from `a` at the indices specified by `b`.


---
### ggml\_vqtbl1q\_u8<!-- {{#callable:ggml_vqtbl1q_u8}} -->
The function `ggml_vqtbl1q_u8` performs a table lookup operation on two 128-bit vectors of unsigned 8-bit integers, using the second vector as an index to reorder the elements of the first vector.
- **Inputs**:
    - `a`: A 128-bit vector of unsigned 8-bit integers, serving as the lookup table.
    - `b`: A 128-bit vector of unsigned 8-bit integers, serving as the index vector for the lookup operation.
- **Control Flow**:
    - Initialize a 128-bit vector `res` to store the result.
    - For each index from 0 to 15, assign `res[i]` the value of `a[b[i]]`, effectively using `b` as an index to reorder elements of `a`.
    - Return the resulting vector `res`.
- **Output**: A 128-bit vector of unsigned 8-bit integers, where each element is selected from `a` based on the corresponding index in `b`.


---
### ggml\_vdotq\_s32<!-- {{#callable:ggml_vdotq_s32}} -->
The `ggml_vdotq_s32` function computes the dot product of two 16-element vectors of 8-bit integers, accumulates the result into a 4-element vector of 32-bit integers, and returns the updated accumulator.
- **Inputs**:
    - `acc`: A 4-element vector of 32-bit integers (int32x4_t) that serves as the initial accumulator for the dot product result.
    - `a`: A 16-element vector of 8-bit integers (int8x16_t) representing the first operand of the dot product.
    - `b`: A 16-element vector of 8-bit integers (int8x16_t) representing the second operand of the dot product.
- **Control Flow**:
    - Extracts the lower 8 elements of vectors 'a' and 'b' and multiplies them, storing the result in a 16-element vector of 16-bit integers 'p0'.
    - Extracts the higher 8 elements of vectors 'a' and 'b' and multiplies them, storing the result in a 16-element vector of 16-bit integers 'p1'.
    - Performs pairwise addition of adjacent elements in 'p0' and 'p1', reducing them to 8-element vectors of 32-bit integers.
    - Adds the results of the pairwise additions to the initial accumulator 'acc'.
    - Returns the updated accumulator.
- **Output**: A 4-element vector of 32-bit integers (int32x4_t) representing the accumulated dot product result.


---
### ggml\_vec\_xl\_u8x2<!-- {{#callable:ggml_vec_xl_u8x2}} -->
The function `ggml_vec_xl_u8x2` loads two 16-byte vectors from a given pointer to an array of unsigned 8-bit integers.
- **Inputs**:
    - `ptr`: A pointer to an array of unsigned 8-bit integers from which two 16-byte vectors will be loaded.
- **Control Flow**:
    - Declare a variable `res` of type `ggml_uint8x16x2_t`.
    - Load the first 16-byte vector from the memory location pointed to by `ptr` into `res.val[0]` using `vec_xl` with an offset of 0.
    - Load the second 16-byte vector from the memory location pointed to by `ptr` into `res.val[1]` using `vec_xl` with an offset of 16.
    - Return the `res` structure containing the two loaded vectors.
- **Output**: A `ggml_uint8x16x2_t` structure containing two 16-byte vectors loaded from the specified memory location.


---
### ggml\_vec\_xl\_u8x4<!-- {{#callable:ggml_vec_xl_u8x4}} -->
The function `ggml_vec_xl_u8x4` loads four 16-byte vectors from a given pointer to an array of unsigned 8-bit integers.
- **Inputs**:
    - `ptr`: A pointer to an array of unsigned 8-bit integers from which the vectors will be loaded.
- **Control Flow**:
    - Declare a variable `res` of type `ggml_uint8x16x4_t` to store the result.
    - Load the first 16-byte vector from the memory location pointed to by `ptr` into `res.val[0]`.
    - Load the second 16-byte vector from the memory location `ptr + 16` into `res.val[1]`.
    - Load the third 16-byte vector from the memory location `ptr + 32` into `res.val[2]`.
    - Load the fourth 16-byte vector from the memory location `ptr + 48` into `res.val[3]`.
    - Return the `res` structure containing the four loaded vectors.
- **Output**: A `ggml_uint8x16x4_t` structure containing four 16-byte vectors loaded from the specified memory location.


---
### ggml\_vec\_xl\_s8x4<!-- {{#callable:ggml_vec_xl_s8x4}} -->
The function `ggml_vec_xl_s8x4` loads four 16-byte vectors from a given pointer to int8_t data, each offset by 16 bytes, into a `ggml_int8x16x4_t` structure.
- **Inputs**:
    - `ptr`: A pointer to an array of int8_t data from which the vectors will be loaded.
- **Control Flow**:
    - Initialize a `ggml_int8x16x4_t` structure named `res`.
    - Load the first 16-byte vector from `ptr` into `res.val[0]` using `vec_xl` with an offset of 0 bytes.
    - Load the second 16-byte vector from `ptr` into `res.val[1]` using `vec_xl` with an offset of 16 bytes.
    - Load the third 16-byte vector from `ptr` into `res.val[2]` using `vec_xl` with an offset of 32 bytes.
    - Load the fourth 16-byte vector from `ptr` into `res.val[3]` using `vec_xl` with an offset of 48 bytes.
    - Return the `res` structure containing the four loaded vectors.
- **Output**: A `ggml_int8x16x4_t` structure containing four 16-byte vectors loaded from the specified memory location.


---
### ggml\_vec\_xl\_s16x2<!-- {{#callable:ggml_vec_xl_s16x2}} -->
The function `ggml_vec_xl_s16x2` loads two 128-bit vectors from a given pointer to 16-bit integers and returns them as a structure containing two 128-bit integer vectors.
- **Inputs**:
    - `ptr`: A pointer to an array of 16-bit integers from which the vectors will be loaded.
- **Control Flow**:
    - Declare a variable `res` of type `ggml_int16x8x2_t` to store the result.
    - Load the first 128-bit vector from the memory location pointed to by `ptr` into `res.val[0]` using `vec_xl` with an offset of 0.
    - Load the second 128-bit vector from the memory location pointed to by `ptr` into `res.val[1]` using `vec_xl` with an offset of 16.
    - Return the `res` structure containing the two loaded vectors.
- **Output**: A `ggml_int16x8x2_t` structure containing two 128-bit vectors loaded from the specified memory location.


---
### ggml\_vec\_tbl<!-- {{#callable:ggml_vec_tbl}} -->
The `ggml_vec_tbl` function performs a table lookup operation on a vector of 16 signed 8-bit integers using indices specified by another vector of 16 unsigned 8-bit integers.
- **Inputs**:
    - `a`: A vector of 16 signed 8-bit integers (int8x16_t) that serves as the lookup table.
    - `b`: A vector of 16 unsigned 8-bit integers (uint8x16_t) that specifies the indices for the lookup operation.
- **Control Flow**:
    - Initialize a result vector `res` of type int8x16_t.
    - For each index from 0 to 15, assign `res[i]` the value of `a[b[i]]`, effectively performing a lookup in vector `a` using the index specified by `b[i]`.
    - Return the result vector `res`.
- **Output**: A vector of 16 signed 8-bit integers (int8x16_t) containing the results of the lookup operation.


---
### vec\_padd\_s16<!-- {{#callable:vec_padd_s16}} -->
The `vec_padd_s16` function performs a vectorized addition of two 16-bit integer vectors by packing and permuting them before summing.
- **Inputs**:
    - `a`: A vector of eight 16-bit integers (int16x8_t).
    - `b`: Another vector of eight 16-bit integers (int16x8_t).
- **Control Flow**:
    - Define a constant permutation mask `v_maske` to rearrange elements from the input vectors.
    - Pack the input vectors `a` and `b` into a single vector `v_abo` using `vec_pack`.
    - Permute the elements of `a` and `b` using `vec_perm` with the mask `v_maske` to create `v_abe`.
    - Add the vectors `v_abo` and `v_abe` together and return the result.
- **Output**: The function returns a vector of eight 16-bit integers (int16x8_t) which is the result of the vectorized addition.


---
### ggml\_vec\_dot<!-- {{#callable:ggml_vec_dot}} -->
The `ggml_vec_dot` function computes the dot product of two 128-bit integer vectors and accumulates the result into a 128-bit integer accumulator.
- **Inputs**:
    - `acc`: A 128-bit integer vector (int32x4_t) that serves as the accumulator for the dot product result.
    - `a`: A 128-bit integer vector (int8x16_t) representing the first operand in the dot product.
    - `b`: A 128-bit integer vector (int8x16_t) representing the second operand in the dot product.
- **Control Flow**:
    - Compute the element-wise multiplication of the high and low parts of vectors 'a' and 'b' using `vec_mule` and `vec_mulo`, resulting in a 16-bit integer vector 'p'.
    - Unpack the high and low parts of 'p' into separate 32-bit integer vectors using `vec_unpackh` and `vec_unpackl`.
    - Add the unpacked high and low parts together.
    - Add the result to the accumulator 'acc' and return the updated accumulator.
- **Output**: A 128-bit integer vector (int32x4_t) representing the accumulated dot product result.


---
### \_\_lsx\_vreplfr2vr\_s<!-- {{#callable:__lsx_vreplfr2vr_s}} -->
The function `__lsx_vreplfr2vr_s` replicates a single float value across all elements of a 128-bit vector and returns it as an `__m128` type.
- **Inputs**:
    - `val`: A single float value to be replicated across the vector.
- **Control Flow**:
    - The function takes a float input `val`.
    - It initializes a vector `v4f32 res` with four elements, all set to `val`.
    - The vector `res` is then cast to an `__m128` type and returned.
- **Output**: A 128-bit vector (`__m128`) with all elements set to the input float value.


---
### \_\_lasx\_xvreplfr2vr\_s<!-- {{#callable:__lasx_xvreplfr2vr_s}} -->
The function `__lasx_xvreplfr2vr_s` creates a 256-bit vector with all elements set to the same float value.
- **Inputs**:
    - `val`: A float value that will be replicated across all elements of the vector.
- **Control Flow**:
    - The function takes a single float input `val`.
    - It initializes a vector `res` of type `v8f32` with all eight elements set to `val`.
    - The vector `res` is then cast to a `__m256` type and returned.
- **Output**: A 256-bit vector (`__m256`) with all elements set to the input float value.


# Function Declarations (Public API)

---
### ggml\_barrier<!-- {{#callable_declaration:ggml_barrier}} -->
Synchronizes threads in a thread pool at a barrier point.
- **Description**: Use this function to synchronize all threads in a given thread pool at a specific point in the execution. It ensures that all threads reach the barrier before any of them can proceed, effectively pausing execution until all threads have caught up. This function should be called when you need to ensure that all threads have completed a certain task before moving on to the next phase. It is important to note that this function is a no-op if there is only one thread in the pool.
- **Inputs**:
    - `tp`: A pointer to a `ggml_threadpool` structure representing the thread pool whose threads need to be synchronized. This pointer must not be null, and the thread pool should be properly initialized before calling this function.
- **Output**: None
- **See also**: [`ggml_barrier`](ggml-cpu.c.driver.md#ggml_barrier)  (Implementation)


