# Purpose
This C source code file provides a collection of vectorized functions for performing fundamental mathematical operations on arrays of various data types, including `float`, `int8_t`, `int16_t`, `int32_t`, `ggml_fp16_t`, and `ggml_bf16_t`. The file is structured to support operations such as dot products, element-wise addition, subtraction, multiplication, division, and various activation functions like GELU, SiLU, and ReLU. It also includes functions for computing mathematical transformations like square root, logarithm, sine, cosine, and exponential functions. The code is optimized for performance using SIMD (Single Instruction, Multiple Data) instructions, and it includes conditional compilation to leverage hardware-specific optimizations, such as those provided by the Accelerate framework on macOS or ARM NEON and Intel AVX instructions.

The file is intended to be included in other C or C++ projects, as indicated by the use of `#pragma once` for header inclusion protection and the `extern "C"` block for C++ compatibility. It defines a set of public APIs that can be used to perform high-performance vectorized computations, making it suitable for applications in machine learning, scientific computing, or any domain requiring efficient numerical processing. The presence of precomputed tables for GELU functions and the use of macros for unrolling loops further emphasize the focus on optimizing computational efficiency. The file does not contain a `main` function, indicating that it is not an executable but rather a library to be integrated into larger software systems.
# Imports and Dependencies

---
- `ggml-impl.h`
- `simd-mappings.h`
- `ggml.h`
- `ggml-cpu.h`
- `Accelerate/Accelerate.h`


# Functions

---
### ggml\_vec\_set\_i8<!-- {{#callable:ggml_vec_set_i8}} -->
Sets all elements of a vector to a specified 8-bit integer value.
- **Inputs**:
    - `n`: The number of elements in the vector.
    - `x`: A pointer to the vector of type `int8_t` that will be modified.
    - `v`: The value of type `int8_t` to set each element of the vector to.
- **Control Flow**:
    - A for loop iterates from 0 to n-1.
    - In each iteration, the current element of the vector pointed to by x is set to the value v.
- **Output**: The function does not return a value; it modifies the vector in place.


---
### ggml\_vec\_set\_i16<!-- {{#callable:ggml_vec_set_i16}} -->
Sets all elements of a vector of `int16_t` type to a specified value.
- **Inputs**:
    - `n`: The number of elements in the vector to be set.
    - `x`: A pointer to the array of `int16_t` values that will be modified.
    - `v`: The value of type `int16_t` that will be assigned to each element of the vector.
- **Control Flow**:
    - The function uses a for loop that iterates from 0 to n-1.
    - In each iteration, the current index of the vector is set to the value v.
- **Output**: The function does not return a value; it modifies the input vector in place.


---
### ggml\_vec\_set\_i32<!-- {{#callable:ggml_vec_set_i32}} -->
Sets all elements of a vector to a specified integer value.
- **Inputs**:
    - `n`: The number of elements in the vector.
    - `x`: A pointer to the array of `int32_t` values that will be modified.
    - `v`: The integer value to set each element of the vector to.
- **Control Flow**:
    - A for loop iterates from 0 to n-1.
    - In each iteration, the current element of the vector pointed to by x is set to the value v.
- **Output**: The function does not return a value; it modifies the input vector in place.


---
### ggml\_vec\_cpy\_i32<!-- {{#callable:ggml_vec_cpy_i32}} -->
Copies `n` elements from one integer array to another.
- **Inputs**:
    - `n`: The number of elements to copy from the source array.
    - `y`: Pointer to the destination array where the elements will be copied.
    - `x`: Pointer to the source array from which the elements will be copied.
- **Control Flow**:
    - A for loop iterates from 0 to `n`.
    - In each iteration, the element at index `i` from the source array `x` is copied to the destination array `y`.
- **Output**: The function does not return a value; it modifies the destination array `y` in place.


---
### ggml\_vec\_set\_f16<!-- {{#callable:ggml_vec_set_f16}} -->
Sets all elements of a vector of `ggml_fp16_t` type to a specified value.
- **Inputs**:
    - `n`: The number of elements in the vector.
    - `x`: A pointer to the vector of type `ggml_fp16_t` that will be modified.
    - `v`: The value of type `ggml_fp16_t` to set each element of the vector to.
- **Control Flow**:
    - A for loop iterates from 0 to n-1.
    - In each iteration, the current element of the vector pointed to by x is set to the value v.
- **Output**: The function does not return a value; it modifies the input vector in place.


---
### ggml\_vec\_set\_bf16<!-- {{#callable:ggml_vec_set_bf16}} -->
Sets all elements of a vector to a specified `ggml_bf16_t` value.
- **Inputs**:
    - `n`: The number of elements in the vector.
    - `x`: A pointer to the vector of type `ggml_bf16_t` that will be modified.
    - `v`: The value of type `ggml_bf16_t` to set each element of the vector to.
- **Control Flow**:
    - A for loop iterates from 0 to n-1.
    - In each iteration, the current element of the vector `x[i]` is set to the value `v`.
- **Output**: The function does not return a value; it modifies the vector `x` in place.


---
### ggml\_vec\_add\_f32<!-- {{#callable:ggml_vec_add_f32}} -->
Adds two vectors of single-precision floating-point numbers element-wise.
- **Inputs**:
    - `n`: The number of elements in the vectors.
    - `z`: Pointer to the output vector where the result will be stored.
    - `x`: Pointer to the first input vector.
    - `y`: Pointer to the second input vector.
- **Control Flow**:
    - A for loop iterates from 0 to n-1.
    - In each iteration, the corresponding elements of vectors `x` and `y` are added together and stored in vector `z`.
- **Output**: The function does not return a value; instead, it modifies the output vector `z` in place with the sum of the input vectors `x` and `y`.


---
### ggml\_vec\_add\_f16<!-- {{#callable:ggml_vec_add_f16}} -->
Adds two vectors of half-precision floating-point numbers element-wise.
- **Inputs**:
    - `n`: The number of elements in the input vectors.
    - `z`: Pointer to the output vector where the result will be stored.
    - `x`: Pointer to the first input vector of half-precision floating-point numbers.
    - `y`: Pointer to the second input vector of half-precision floating-point numbers.
- **Control Flow**:
    - A for loop iterates from 0 to n, processing each element of the input vectors.
    - Within the loop, each element from vectors `x` and `y` is converted from half-precision to single-precision using `GGML_FP16_TO_FP32`.
    - The converted values are added together, and the result is converted back to half-precision using `GGML_FP32_TO_FP16` before storing it in the output vector `z`.
- **Output**: The function does not return a value; instead, it populates the output vector `z` with the sum of the corresponding elements from vectors `x` and `y`.


---
### ggml\_vec\_add1\_f32<!-- {{#callable:ggml_vec_add1_f32}} -->
Adds a constant value to each element of a vector.
- **Inputs**:
    - `n`: The number of elements in the vector.
    - `z`: Pointer to the output vector where the results will be stored.
    - `x`: Pointer to the input vector whose elements will be added to the constant.
    - `v`: The constant value to be added to each element of the input vector.
- **Control Flow**:
    - The function iterates over each element of the input vector `x` from index 0 to `n-1`.
    - For each index `i`, it computes the sum of `x[i]` and the constant `v`, storing the result in `z[i]`.
- **Output**: The function does not return a value; instead, it modifies the output vector `z` in place, containing the results of the addition.


---
### ggml\_vec\_acc\_f32<!-- {{#callable:ggml_vec_acc_f32}} -->
Accumulates the values of one vector into another vector element-wise.
- **Inputs**:
    - `n`: An integer representing the number of elements in the vectors.
    - `y`: A pointer to the destination vector where the accumulated results will be stored.
    - `x`: A pointer to the source vector whose values will be added to the destination vector.
- **Control Flow**:
    - The function uses a for loop that iterates from 0 to n-1.
    - In each iteration, the value from the source vector `x` is added to the corresponding element in the destination vector `y`.
- **Output**: The function does not return a value; instead, it modifies the destination vector `y` in place, resulting in `y[i]` containing the sum of its original value and the corresponding value from `x`.


---
### ggml\_vec\_acc1\_f32<!-- {{#callable:ggml_vec_acc1_f32}} -->
Accumulates a constant value to each element of a float array.
- **Inputs**:
    - `n`: The number of elements in the array to be updated.
    - `y`: A pointer to the float array that will be updated.
    - `v`: The float value to be added to each element of the array.
- **Control Flow**:
    - A for loop iterates from 0 to n-1.
    - In each iteration, the value v is added to the current element y[i].
- **Output**: The function does not return a value; it modifies the input array y in place.


---
### ggml\_vec\_sub\_f32<!-- {{#callable:ggml_vec_sub_f32}} -->
Subtracts two vectors of single-precision floating-point numbers element-wise.
- **Inputs**:
    - `n`: The number of elements in the vectors to be subtracted.
    - `z`: Pointer to the output vector where the result of the subtraction will be stored.
    - `x`: Pointer to the first input vector from which elements will be subtracted.
    - `y`: Pointer to the second input vector whose elements will be subtracted from the first vector.
- **Control Flow**:
    - A for loop iterates from 0 to n-1, where n is the number of elements in the vectors.
    - During each iteration, the corresponding elements from vectors `x` and `y` are subtracted, and the result is stored in the output vector `z`.
- **Output**: The function does not return a value; instead, it modifies the output vector `z` in place with the results of the element-wise subtraction.


---
### ggml\_vec\_sub\_f16<!-- {{#callable:ggml_vec_sub_f16}} -->
Subtracts two vectors of `ggml_fp16_t` type and stores the result in a third vector.
- **Inputs**:
    - `n`: An integer representing the number of elements in the input vectors.
    - `z`: A pointer to the output vector where the result of the subtraction will be stored.
    - `x`: A pointer to the first input vector of type `ggml_fp16_t`.
    - `y`: A pointer to the second input vector of type `ggml_fp16_t`.
- **Control Flow**:
    - The function iterates over each element of the input vectors from index 0 to n-1.
    - For each index i, it converts the elements of `x` and `y` from `ggml_fp16_t` to `float`, performs the subtraction, and then converts the result back to `ggml_fp16_t` before storing it in `z`.
- **Output**: The function does not return a value; instead, it modifies the output vector `z` in place with the results of the subtraction of corresponding elements from vectors `x` and `y`.


---
### ggml\_vec\_set\_f32<!-- {{#callable:ggml_vec_set_f32}} -->
Sets all elements of a float vector to a specified value.
- **Inputs**:
    - `n`: The number of elements in the vector.
    - `x`: A pointer to the float array (vector) that will be modified.
    - `v`: The float value to set each element of the vector to.
- **Control Flow**:
    - A for loop iterates from 0 to n-1.
    - In each iteration, the current element of the vector x is set to the value v.
- **Output**: The function does not return a value; it modifies the input vector x in place.


---
### ggml\_vec\_cpy\_f32<!-- {{#callable:ggml_vec_cpy_f32}} -->
Copies `n` elements from the source array `x` to the destination array `y`.
- **Inputs**:
    - `n`: An integer representing the number of elements to copy.
    - `y`: A pointer to the destination array where the elements will be copied.
    - `x`: A pointer to the source array from which the elements will be copied.
- **Control Flow**:
    - The function uses a for loop that iterates from 0 to `n`.
    - In each iteration, it assigns the value from the source array `x` at index `i` to the destination array `y` at the same index.
- **Output**: The function does not return a value; it modifies the destination array `y` in place.


---
### ggml\_vec\_neg\_f32<!-- {{#callable:ggml_vec_neg_f32}} -->
The `ggml_vec_neg_f32` function negates each element of a float array.
- **Inputs**:
    - `n`: An integer representing the number of elements in the input array.
    - `y`: A pointer to the output array where the negated values will be stored.
    - `x`: A pointer to the input array containing the values to be negated.
- **Control Flow**:
    - The function uses a for loop that iterates from 0 to n-1.
    - In each iteration, it assigns the negated value of the corresponding element from the input array `x` to the output array `y`.
- **Output**: The function does not return a value; instead, it modifies the output array `y` in place with the negated values of the input array `x`.


---
### ggml\_vec\_neg\_f16<!-- {{#callable:ggml_vec_neg_f16}} -->
The `ggml_vec_neg_f16` function negates each element of a vector of half-precision floating-point numbers.
- **Inputs**:
    - `n`: An integer representing the number of elements in the input vector.
    - `y`: A pointer to an array of `ggml_fp16_t` where the negated results will be stored.
    - `x`: A pointer to an array of `ggml_fp16_t` containing the input values to be negated.
- **Control Flow**:
    - The function iterates over each element of the input vector `x` from index 0 to `n-1`.
    - For each element, it converts the half-precision floating-point value to single-precision, negates it, and then converts it back to half-precision before storing it in the output vector `y`.
- **Output**: The function does not return a value; instead, it modifies the output array `y` in place to contain the negated values of the input array `x`.


---
### ggml\_vec\_mul\_f32<!-- {{#callable:ggml_vec_mul_f32}} -->
Multiplies two vectors element-wise and stores the result in a third vector.
- **Inputs**:
    - `n`: The number of elements in the input vectors.
    - `z`: Pointer to the output vector where the result of the multiplication will be stored.
    - `x`: Pointer to the first input vector.
    - `y`: Pointer to the second input vector.
- **Control Flow**:
    - The function uses a for loop that iterates from 0 to n-1.
    - In each iteration, it multiplies the corresponding elements of vectors `x` and `y` and stores the result in vector `z`.
- **Output**: The function does not return a value; instead, it modifies the output vector `z` in place with the results of the element-wise multiplication.


---
### ggml\_vec\_mul\_f16<!-- {{#callable:ggml_vec_mul_f16}} -->
Multiplies two vectors of `ggml_fp16_t` type element-wise and stores the result in a third vector.
- **Inputs**:
    - `n`: The number of elements in the input vectors.
    - `z`: A pointer to the output vector where the results of the multiplication will be stored.
    - `x`: A pointer to the first input vector of type `ggml_fp16_t`.
    - `y`: A pointer to the second input vector of type `ggml_fp16_t`.
- **Control Flow**:
    - A for loop iterates from 0 to n, processing each element of the input vectors.
    - Within the loop, each element of the output vector `z` is computed by converting the corresponding elements of `x` and `y` to `float`, multiplying them, and then converting the result back to `ggml_fp16_t`.
- **Output**: The function does not return a value; instead, it populates the output vector `z` with the results of the element-wise multiplication of `x` and `y`.


---
### ggml\_vec\_div\_f32<!-- {{#callable:ggml_vec_div_f32}} -->
Performs element-wise division of two float arrays.
- **Inputs**:
    - `n`: The number of elements in the input arrays.
    - `z`: Pointer to the output array where the results of the division will be stored.
    - `x`: Pointer to the first input array containing the dividend values.
    - `y`: Pointer to the second input array containing the divisor values.
- **Control Flow**:
    - The function uses a for loop that iterates from 0 to n-1.
    - In each iteration, it divides the corresponding elements of arrays `x` and `y` and stores the result in the array `z`.
- **Output**: The function does not return a value; instead, it modifies the output array `z` in place with the results of the division.


---
### ggml\_vec\_div\_f16<!-- {{#callable:ggml_vec_div_f16}} -->
Divides two vectors element-wise and stores the result in a third vector.
- **Inputs**:
    - `n`: The number of elements in the vectors.
    - `z`: Pointer to the output vector where the results of the division will be stored.
    - `x`: Pointer to the first input vector (numerator) of type `ggml_fp16_t`.
    - `y`: Pointer to the second input vector (denominator) of type `ggml_fp16_t`.
- **Control Flow**:
    - A for loop iterates from 0 to n, processing each element of the input vectors.
    - For each index i, the function converts the `ggml_fp16_t` values from `x` and `y` to `float`, performs the division, and converts the result back to `ggml_fp16_t` before storing it in `z`.
- **Output**: The function does not return a value; instead, it modifies the output vector `z` in place with the results of the element-wise division of `x` by `y`.


---
### ggml\_vec\_dot\_f16\_unroll<!-- {{#callable:ggml_vec_dot_f16_unroll}} -->
Computes the dot product of multiple vectors using half-precision floating-point values with optional SIMD optimizations.
- **Inputs**:
    - `n`: The number of elements in the vectors to be processed.
    - `xs`: The byte stride between consecutive elements in the input vectors.
    - `s`: An array to store the resulting dot products.
    - `xv`: A pointer to the input vectors, which are stored in half-precision format.
    - `y`: A pointer to the second input vector, also in half-precision format.
- **Control Flow**:
    - Initialize an array `sumf` to accumulate the results of the dot products.
    - Set up pointers to the input vectors based on the provided stride `xs`.
    - If SIMD optimizations are enabled, process the vectors in chunks, using vectorized operations to compute the dot products.
    - For each chunk, load the corresponding elements from the input vectors, perform the multiply-accumulate operation, and store the intermediate results.
    - After processing the main chunks, handle any remaining elements that do not fit into the SIMD processing loop.
    - Finally, store the accumulated results into the output array `s`.
- **Output**: The function does not return a value; instead, it populates the output array `s` with the computed dot products for each vector.


---
### ggml\_vec\_mad\_f32<!-- {{#callable:ggml_vec_mad_f32}} -->
Performs a multiply-accumulate operation on two vectors with a scalar multiplier.
- **Inputs**:
    - `n`: The number of elements in the vectors.
    - `y`: Pointer to the output vector where results will be stored.
    - `x`: Pointer to the input vector that will be multiplied.
    - `v`: The scalar value to multiply with each element of vector `x`.
- **Control Flow**:
    - The function checks if SIMD (Single Instruction, Multiple Data) is enabled.
    - If SIMD is enabled and ARM SVE (Scalable Vector Extension) is supported, it calculates the number of elements that can be processed in parallel and performs the multiply-accumulate operation in chunks.
    - For each chunk, it loads elements from vectors `x` and `y`, performs the multiply-accumulate operation, and stores the result back into vector `y`.
    - After processing the main chunks, it handles any remaining elements that were not processed in the SIMD loop.
    - If SIMD is not enabled, it falls back to a scalar implementation that processes each element in a simple loop.
- **Output**: The output vector `y` is updated with the results of the operation, where each element is computed as y[i] += x[i] * v for all i from 0 to n-1.
- **Functions called**:
    - [`ggml_cpu_get_sve_cnt`](ggml-cpu.c.driver.md#ggml_cpu_get_sve_cnt)


---
### ggml\_vec\_mad\_f16<!-- {{#callable:ggml_vec_mad_f16}} -->
The `ggml_vec_mad_f16` function performs a multiply-accumulate operation on two vectors of half-precision floating-point numbers.
- **Inputs**:
    - `n`: An integer representing the number of elements in the vectors.
    - `y`: A pointer to the output vector where the results will be stored.
    - `x`: A pointer to the input vector that will be multiplied.
    - `v`: A float value that will be multiplied with each element of the input vector `x`.
- **Control Flow**:
    - The function checks if SIMD (Single Instruction, Multiple Data) is enabled.
    - If SIMD is enabled, it calculates the number of elements that can be processed in parallel and initializes vector registers.
    - It then enters a loop to process the vectors in chunks, loading elements from `x` and `y`, performing the multiply-accumulate operation, and storing the results back to `y`.
    - After processing the main chunks, it handles any remaining elements that could not be processed in the SIMD loop by performing the operation in a scalar manner.
    - If SIMD is not enabled, it simply iterates through each element of the vectors and performs the multiply-accumulate operation in a scalar manner.
- **Output**: The function modifies the output vector `y` in place, where each element is updated to be the result of the operation: y[i] = y[i] + (x[i] * v) for each element i.


---
### ggml\_vec\_mad\_f32\_unroll<!-- {{#callable:ggml_vec_mad_f32_unroll}} -->
Performs a vectorized multiply-accumulate operation on two input vectors and adds the result to an output vector.
- **Inputs**:
    - `n`: The number of elements to process in the vectors.
    - `xs`: The byte stride for the first input vector `x`.
    - `vs`: The byte stride for the second input vector `v`.
    - `y`: A pointer to the output vector where the results will be accumulated.
    - `xv`: A pointer to the first input vector `x`.
    - `vv`: A pointer to the second input vector `v`.
- **Control Flow**:
    - Initialize arrays of pointers for the input vectors `x` and `v` based on the provided strides.
    - Check if SIMD (Single Instruction, Multiple Data) is enabled.
    - If SIMD is enabled and ARM SVE (Scalable Vector Extension) is supported, perform a scalar route to the scalar implementation.
    - If SIMD is enabled but SVE is not supported, process the vectors in chunks using vectorized operations, handling leftovers after the main loop.
    - If SIMD is not enabled, perform the operation in a simple loop for each element.
- **Output**: The function modifies the output vector `y` in place, accumulating the results of the multiply-accumulate operations.


---
### ggml\_vec\_scale\_f32<!-- {{#callable:ggml_vec_scale_f32}} -->
Scales a vector of floats by a given scalar value.
- **Inputs**:
    - `n`: The number of elements in the vector to be scaled.
    - `y`: A pointer to the float array that will be scaled.
    - `v`: The scalar value by which to scale each element of the vector.
- **Control Flow**:
    - The function checks if the `GGML_USE_ACCELERATE` macro is defined to use the Accelerate framework for vectorized operations.
    - If `GGML_USE_ACCELERATE` is not defined, it checks if `GGML_SIMD` is defined to utilize SIMD instructions for optimization.
    - If `__ARM_FEATURE_SVE` is defined, it uses SVE (Scalable Vector Extension) instructions to process multiple elements in parallel.
    - If SVE is not available, it falls back to a different SIMD approach using predefined vector operations.
    - If neither SIMD nor Accelerate is available, it defaults to a simple scalar multiplication in a loop for each element.
- **Output**: The function modifies the input array `y` in place, scaling each element by the scalar value `v`.
- **Functions called**:
    - [`ggml_cpu_get_sve_cnt`](ggml-cpu.c.driver.md#ggml_cpu_get_sve_cnt)


---
### ggml\_vec\_scale\_f16<!-- {{#callable:ggml_vec_scale_f16}} -->
Scales a vector of `ggml_fp16_t` values by a given float multiplier.
- **Inputs**:
    - `n`: The number of elements in the vector to be scaled.
    - `y`: A pointer to the array of `ggml_fp16_t` values that will be scaled.
    - `v`: The float value by which to scale each element of the vector.
- **Control Flow**:
    - The function checks if SIMD (Single Instruction, Multiple Data) is enabled.
    - If SIMD is enabled, it calculates the number of elements that can be processed in parallel and initializes a vector with the scaling factor.
    - It then processes the vector in chunks, multiplying each element by the scaling factor using SIMD operations.
    - After processing the main chunk, it handles any remaining elements individually.
    - If SIMD is not enabled, it simply scales each element in a loop.
- **Output**: The function modifies the input array `y` in place, scaling each element by the factor `v`.


---
### ggml\_vec\_norm\_f32<!-- {{#callable:ggml_vec_norm_f32}} -->
Calculates the Euclidean norm of a vector.
- **Inputs**:
    - `n`: The number of elements in the vector.
    - `s`: A pointer to a float where the result (norm) will be stored.
    - `x`: A pointer to the input vector of floats.
- **Control Flow**:
    - Calls the [`ggml_vec_dot_f32`](vec.cpp.driver.md#ggml_vec_dot_f32) function to compute the dot product of the vector with itself, storing the result in `*s`.
    - The dot product result is then passed to the `sqrtf` function to compute the square root, which gives the Euclidean norm.
- **Output**: The function outputs the Euclidean norm of the vector, stored in the location pointed to by `s`.
- **Functions called**:
    - [`ggml_vec_dot_f32`](vec.cpp.driver.md#ggml_vec_dot_f32)


---
### ggml\_vec\_sqr\_f32<!-- {{#callable:ggml_vec_sqr_f32}} -->
Computes the element-wise square of a vector of floats.
- **Inputs**:
    - `n`: The number of elements in the input vector.
    - `y`: Pointer to the output vector where the squared results will be stored.
    - `x`: Pointer to the input vector containing the values to be squared.
- **Control Flow**:
    - A for loop iterates from 0 to n-1.
    - During each iteration, the square of the element at index i of the input vector x is calculated and stored in the output vector y at the same index.
- **Output**: The function does not return a value; instead, it modifies the output vector y in place with the squared values of the input vector x.


---
### ggml\_vec\_sqr\_f16<!-- {{#callable:ggml_vec_sqr_f16}} -->
Calculates the element-wise square of a vector of half-precision floating-point numbers.
- **Inputs**:
    - `n`: The number of elements in the input vector.
    - `y`: Pointer to the output vector where the squared results will be stored.
    - `x`: Pointer to the input vector containing half-precision floating-point numbers.
- **Control Flow**:
    - Iterates over each element of the input vector `x` from index 0 to `n-1`.
    - Converts each half-precision floating-point number in `x` to single-precision using `GGML_FP16_TO_FP32`.
    - Calculates the square of the converted single-precision value.
    - Converts the squared value back to half-precision using `GGML_FP32_TO_FP16` and stores it in the output vector `y`.
- **Output**: The function does not return a value; instead, it populates the output vector `y` with the squared values of the input vector `x`.


---
### ggml\_vec\_sqrt\_f32<!-- {{#callable:ggml_vec_sqrt_f32}} -->
Calculates the square root of each element in a float array.
- **Inputs**:
    - `n`: The number of elements in the input array.
    - `y`: Pointer to the output array where the square roots will be stored.
    - `x`: Pointer to the input array containing the values to be square rooted.
- **Control Flow**:
    - A for loop iterates from 0 to n-1.
    - In each iteration, the square root of the element at index i of the input array x is computed using the `sqrtf` function.
    - The computed square root is then stored in the output array y at the same index i.
- **Output**: The function does not return a value; instead, it populates the output array y with the square roots of the elements from the input array x.


---
### ggml\_vec\_sqrt\_f16<!-- {{#callable:ggml_vec_sqrt_f16}} -->
Calculates the square root of each element in a vector of half-precision floating-point numbers.
- **Inputs**:
    - `n`: The number of elements in the input vector.
    - `y`: Pointer to the output vector where the results (square roots) will be stored.
    - `x`: Pointer to the input vector containing half-precision floating-point numbers.
- **Control Flow**:
    - Iterates over each element of the input vector `x` from index 0 to `n-1`.
    - For each element, converts the half-precision value to single-precision, computes the square root, and converts the result back to half-precision.
    - Stores the computed square root in the corresponding index of the output vector `y`.
- **Output**: The function does not return a value; instead, it populates the output vector `y` with the square roots of the elements from the input vector `x`.


---
### ggml\_vec\_log\_f32<!-- {{#callable:ggml_vec_log_f32}} -->
Computes the natural logarithm of each element in a float array.
- **Inputs**:
    - `n`: The number of elements in the input array.
    - `y`: Pointer to the output array where the logarithm results will be stored.
    - `x`: Pointer to the input array containing the values for which the logarithm will be computed.
- **Control Flow**:
    - Iterates over each element in the input array `x` from index 0 to `n-1`.
    - For each element, computes the natural logarithm using `logf` and stores the result in the corresponding index of the output array `y`.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the logarithm of each element from the input array `x`.


---
### ggml\_vec\_log\_f16<!-- {{#callable:ggml_vec_log_f16}} -->
Computes the element-wise natural logarithm of an array of half-precision floating-point numbers.
- **Inputs**:
    - `n`: The number of elements in the input array.
    - `y`: Pointer to the output array where the results will be stored.
    - `x`: Pointer to the input array of half-precision floating-point numbers.
- **Control Flow**:
    - Iterates over each element in the input array `x` from index 0 to `n-1`.
    - For each element, it converts the half-precision float to a single-precision float using `GGML_FP16_TO_FP32`, computes the natural logarithm using `logf`, and then converts the result back to half-precision using `GGML_FP32_TO_FP16` before storing it in the output array `y`.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the logarithm of each element from the input array `x`.


---
### ggml\_vec\_sin\_f32<!-- {{#callable:ggml_vec_sin_f32}} -->
Computes the sine of each element in a float array.
- **Inputs**:
    - `n`: The number of elements in the input array.
    - `y`: Pointer to the output array where the sine results will be stored.
    - `x`: Pointer to the input array containing the values for which the sine will be computed.
- **Control Flow**:
    - A for loop iterates from 0 to n-1.
    - In each iteration, the sine of the current element in the input array `x` is computed and stored in the corresponding index of the output array `y`.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the sine values of the input array `x`.


---
### ggml\_vec\_sin\_f16<!-- {{#callable:ggml_vec_sin_f16}} -->
Computes the sine of each element in a vector of half-precision floating-point numbers.
- **Inputs**:
    - `n`: The number of elements in the input vector.
    - `y`: Pointer to the output vector where the sine results will be stored.
    - `x`: Pointer to the input vector containing half-precision floating-point numbers.
- **Control Flow**:
    - Iterates over each element of the input vector `x` from index 0 to `n-1`.
    - Converts each half-precision floating-point number in `x` to single-precision using `GGML_FP16_TO_FP32`.
    - Calculates the sine of the converted single-precision value using the `sinf` function.
    - Converts the result back to half-precision using `GGML_FP32_TO_FP16` and stores it in the output vector `y`.
- **Output**: The function does not return a value; instead, it populates the output vector `y` with the sine of each corresponding element from the input vector `x`.


---
### ggml\_vec\_cos\_f32<!-- {{#callable:ggml_vec_cos_f32}} -->
The `ggml_vec_cos_f32` function computes the cosine of each element in a given float array.
- **Inputs**:
    - `n`: An integer representing the number of elements in the input array.
    - `y`: A pointer to an array where the computed cosine values will be stored.
    - `x`: A pointer to an array containing the input float values for which the cosine will be calculated.
- **Control Flow**:
    - The function uses a for loop that iterates from 0 to n-1.
    - In each iteration, it calculates the cosine of the value at the corresponding index of the input array `x` using the `cosf` function.
    - The computed cosine value is then stored in the output array `y` at the same index.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the cosine values of the input array `x`.


---
### ggml\_vec\_cos\_f16<!-- {{#callable:ggml_vec_cos_f16}} -->
Computes the cosine of each element in a vector of half-precision floating-point numbers and stores the result in another vector.
- **Inputs**:
    - `n`: The number of elements in the input vector.
    - `y`: A pointer to the output vector where the cosine results will be stored.
    - `x`: A pointer to the input vector containing half-precision floating-point numbers.
- **Control Flow**:
    - Iterates over each element of the input vector `x` from index 0 to `n-1`.
    - Converts each half-precision floating-point number in `x` to single-precision using `GGML_FP16_TO_FP32`.
    - Calculates the cosine of the converted value using `cosf`.
    - Converts the result back to half-precision using `GGML_FP32_TO_FP16` and stores it in the output vector `y`.
- **Output**: The function does not return a value; instead, it populates the output vector `y` with the cosine values of the input vector `x`.


---
### ggml\_vec\_abs\_f32<!-- {{#callable:ggml_vec_abs_f32}} -->
Computes the absolute values of elements in a float array.
- **Inputs**:
    - `n`: The number of elements in the input array.
    - `y`: Pointer to the output array where the absolute values will be stored.
    - `x`: Pointer to the input array containing the values to be converted to absolute.
- **Control Flow**:
    - A for loop iterates from 0 to n, processing each element of the input array.
    - For each element, the absolute value is computed using `fabsf` and stored in the output array.
- **Output**: The function does not return a value; instead, it modifies the output array `y` in place to contain the absolute values of the input array `x`.


---
### ggml\_vec\_abs\_f16<!-- {{#callable:ggml_vec_abs_f16}} -->
Computes the absolute values of elements in a vector of half-precision floating-point numbers.
- **Inputs**:
    - `n`: The number of elements in the input vector.
    - `y`: Pointer to the output vector where the absolute values will be stored.
    - `x`: Pointer to the input vector containing half-precision floating-point numbers.
- **Control Flow**:
    - Iterates over each element of the input vector `x` from index 0 to `n-1`.
    - For each element, it converts the half-precision float to a single-precision float, computes its absolute value using `fabsf`, and then converts it back to half-precision before storing it in the output vector `y`.
- **Output**: The function does not return a value; instead, it populates the output vector `y` with the absolute values of the corresponding elements from the input vector `x`.


---
### ggml\_vec\_sgn\_f32<!-- {{#callable:ggml_vec_sgn_f32}} -->
The `ggml_vec_sgn_f32` function computes the sign of each element in a float array, returning 1 for positive values, -1 for negative values, and 0 for zeros.
- **Inputs**:
    - `n`: An integer representing the number of elements in the input array.
    - `y`: A pointer to an output array where the sign results will be stored.
    - `x`: A pointer to the input array of float values for which the sign will be computed.
- **Control Flow**:
    - The function iterates over each element of the input array `x` from index 0 to `n-1`.
    - For each element `x[i]`, it checks if the value is greater than 0, less than 0, or equal to 0.
    - It assigns 1 to `y[i]` if `x[i]` is greater than 0, -1 if `x[i]` is less than 0, and 0 if `x[i]` is equal to 0.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the computed sign values corresponding to each element in the input array `x`.


---
### ggml\_vec\_sgn\_f16<!-- {{#callable:ggml_vec_sgn_f16}} -->
The `ggml_vec_sgn_f16` function computes the sign of each element in a vector of half-precision floating-point numbers.
- **Inputs**:
    - `n`: An integer representing the number of elements in the input vector.
    - `y`: A pointer to an array of `ggml_fp16_t` where the output sign values will be stored.
    - `x`: A pointer to an array of `ggml_fp16_t` containing the input values for which the sign will be computed.
- **Control Flow**:
    - A for loop iterates over each element of the input vector from index 0 to n-1.
    - For each element, the function converts the half-precision float to a single-precision float using `GGML_FP16_TO_FP32`.
    - It then checks the value: if it is greater than 0, it assigns 1.0 to the output; if less than 0, it assigns -1.0; if equal to 0, it assigns 0.0.
    - The computed sign is then converted back to half-precision using `GGML_FP32_TO_FP16` and stored in the output array.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the sign of each corresponding element from the input array `x`.


---
### ggml\_vec\_step\_f32<!-- {{#callable:ggml_vec_step_f32}} -->
The `ggml_vec_step_f32` function computes the step function for each element in the input vector.
- **Inputs**:
    - `n`: An integer representing the number of elements in the input vector.
    - `y`: A pointer to an array where the output values (0 or 1) will be stored.
    - `x`: A pointer to an input array of float values that will be evaluated.
- **Control Flow**:
    - The function iterates over each element of the input array `x` from index 0 to `n-1`.
    - For each element `x[i]`, it checks if the value is greater than 0.
    - If `x[i]` is greater than 0, `y[i]` is set to 1.0; otherwise, it is set to 0.0.
- **Output**: The function does not return a value; instead, it modifies the output array `y` in place, filling it with 1s and 0s based on the input array `x`.


---
### ggml\_vec\_step\_f16<!-- {{#callable:ggml_vec_step_f16}} -->
The `ggml_vec_step_f16` function applies a step function to each element of a vector of half-precision floating-point numbers.
- **Inputs**:
    - `n`: An integer representing the number of elements in the input vector.
    - `y`: A pointer to an array of `ggml_fp16_t` where the results of the step function will be stored.
    - `x`: A pointer to an array of `ggml_fp16_t` containing the input values to which the step function will be applied.
- **Control Flow**:
    - A for loop iterates from 0 to n, processing each element of the input vector.
    - For each element, the function converts the half-precision floating-point value from `x` to single-precision, checks if it is greater than zero, and assigns 1.0 or 0.0 to the corresponding position in `y` based on this condition.
    - The result is then converted back to half-precision and stored in the output array `y`.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the results of the step function applied to each element of the input array `x`.


---
### ggml\_vec\_tanh\_f32<!-- {{#callable:ggml_vec_tanh_f32}} -->
Computes the hyperbolic tangent of each element in a float array.
- **Inputs**:
    - `n`: The number of elements in the input array.
    - `y`: Pointer to the output array where the results will be stored.
    - `x`: Pointer to the input array containing the values to be transformed.
- **Control Flow**:
    - A for loop iterates from 0 to n, processing each element of the input array.
    - For each index i, the hyperbolic tangent of the input value x[i] is computed using the `tanhf` function.
    - The computed value is then stored in the output array y[i].
- **Output**: The function does not return a value; instead, it populates the output array y with the hyperbolic tangent of each corresponding element from the input array x.


---
### ggml\_vec\_tanh\_f16<!-- {{#callable:ggml_vec_tanh_f16}} -->
Computes the hyperbolic tangent of each element in a vector of half-precision floating-point numbers.
- **Inputs**:
    - `n`: The number of elements in the input vector.
    - `y`: Pointer to the output vector where the results will be stored.
    - `x`: Pointer to the input vector containing half-precision floating-point numbers.
- **Control Flow**:
    - Iterates over each element of the input vector `x` from index 0 to `n-1`.
    - Converts each half-precision floating-point number in `x` to single-precision using `GGML_FP16_TO_FP32`.
    - Calculates the hyperbolic tangent of the converted value using `tanhf`.
    - Converts the result back to half-precision using `GGML_FP32_TO_FP16` and stores it in the output vector `y`.
- **Output**: The function does not return a value; instead, it populates the output vector `y` with the hyperbolic tangent of each corresponding element from the input vector `x`.


---
### ggml\_vec\_elu\_f32<!-- {{#callable:ggml_vec_elu_f32}} -->
Applies the Exponential Linear Unit (ELU) activation function to each element of the input vector.
- **Inputs**:
    - `n`: The number of elements in the input vector.
    - `y`: Pointer to the output vector where the results of the ELU function will be stored.
    - `x`: Pointer to the input vector containing the values to which the ELU function will be applied.
- **Control Flow**:
    - Iterates over each element of the input vector from index 0 to n-1.
    - For each element, checks if the value is greater than 0.
    - If the value is greater than 0, it assigns the value directly to the output vector.
    - If the value is less than or equal to 0, it computes the ELU value using the expm1f function and assigns it to the output vector.
- **Output**: The output vector `y` contains the transformed values after applying the ELU function to each corresponding element in the input vector `x`.


---
### ggml\_vec\_elu\_f16<!-- {{#callable:ggml_vec_elu_f16}} -->
The `ggml_vec_elu_f16` function applies the Exponential Linear Unit (ELU) activation function to each element of a vector of half-precision floating-point numbers.
- **Inputs**:
    - `n`: An integer representing the number of elements in the input vector.
    - `y`: A pointer to an array of `ggml_fp16_t` where the results of the ELU function will be stored.
    - `x`: A pointer to an array of `ggml_fp16_t` containing the input values to which the ELU function will be applied.
- **Control Flow**:
    - The function iterates over each element of the input vector `x` from index 0 to `n-1`.
    - For each element `x[i]`, it converts the half-precision float to a single-precision float using `GGML_FP16_TO_FP32`.
    - It then computes the ELU function using `expm1f`, which calculates exp(x) - 1, and converts the result back to half-precision using `GGML_FP32_TO_FP16`.
    - The computed value is stored in the corresponding index `y[i]` of the output array.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the results of the ELU activation applied to each element of the input array `x`.


---
### ggml\_vec\_relu\_f32<!-- {{#callable:ggml_vec_relu_f32}} -->
Applies the ReLU (Rectified Linear Unit) activation function element-wise to a vector.
- **Inputs**:
    - `n`: The number of elements in the input vector.
    - `y`: Pointer to the output vector where the results will be stored.
    - `x`: Pointer to the input vector containing the values to be processed.
- **Control Flow**:
    - Iterates over each element of the input vector `x` from index 0 to `n-1`.
    - For each element, checks if it is greater than 0.
    - If the element is greater than 0, it is copied to the corresponding position in the output vector `y`.
    - If the element is less than or equal to 0, 0 is assigned to the corresponding position in the output vector `y`.
- **Output**: The function does not return a value; instead, it modifies the output vector `y` in place, containing the ReLU applied results.


---
### ggml\_vec\_relu\_f16<!-- {{#callable:ggml_vec_relu_f16}} -->
The `ggml_vec_relu_f16` function applies the ReLU activation function to each element of a vector of half-precision floating-point numbers.
- **Inputs**:
    - `n`: An integer representing the number of elements in the input vector.
    - `y`: A pointer to an array of `ggml_fp16_t` where the output values will be stored.
    - `x`: A pointer to an array of `ggml_fp16_t` containing the input values to which the ReLU function will be applied.
- **Control Flow**:
    - A for loop iterates over each element of the input vector from 0 to n-1.
    - Within the loop, each element of the input vector `x` is converted from half-precision to single-precision using `GGML_FP16_TO_FP32`.
    - The ReLU function is applied: if the converted value is greater than 0, it is retained; otherwise, it is set to 0.
    - The result is then converted back to half-precision using `GGML_FP32_TO_FP16` and stored in the output array `y`.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the results of the ReLU activation applied to each element of the input array `x`.


---
### ggml\_vec\_leaky\_relu\_f32<!-- {{#callable:ggml_vec_leaky_relu_f32}} -->
Applies the leaky ReLU activation function to each element of the input vector.
- **Inputs**:
    - `n`: The number of elements in the input vector.
    - `y`: Pointer to the output vector where the results will be stored.
    - `x`: Pointer to the input vector containing the values to be transformed.
    - `ns`: The slope for negative input values, determining how much negative values are scaled.
- **Control Flow**:
    - Iterates over each element of the input vector `x` from index 0 to `n-1`.
    - For each element, checks if it is greater than 0; if so, it retains its value.
    - If the element is less than or equal to 0, it is scaled by the negative slope `ns` and added to the output.
- **Output**: The function modifies the output vector `y` in place, containing the transformed values after applying the leaky ReLU function.


---
### ggml\_vec\_leaky\_relu\_f16<!-- {{#callable:ggml_vec_leaky_relu_f16}} -->
Applies the leaky ReLU activation function to a vector of half-precision floating-point numbers.
- **Inputs**:
    - `n`: The number of elements in the input vector.
    - `y`: Pointer to the output vector where the results will be stored.
    - `x`: Pointer to the input vector of half-precision floating-point numbers.
    - `ns`: The slope for negative input values.
- **Control Flow**:
    - Iterates over each element of the input vector `x` from index 0 to `n-1`.
    - Converts each half-precision floating-point value from `x` to single-precision using `GGML_FP16_TO_FP32`.
    - Applies the leaky ReLU function: if the value is greater than 0, it remains unchanged; if less than or equal to 0, it is multiplied by `ns`.
    - Converts the result back to half-precision using `GGML_FP32_TO_FP16` and stores it in the output vector `y`.
- **Output**: The function does not return a value; instead, it populates the output vector `y` with the results of the leaky ReLU activation applied to the input vector `x`.


---
### ggml\_vec\_sigmoid\_f32<!-- {{#callable:ggml_vec_sigmoid_f32}} -->
Computes the sigmoid function for each element in a vector.
- **Inputs**:
    - `n`: The number of elements in the input vector.
    - `y`: Pointer to the output vector where the results of the sigmoid function will be stored.
    - `x`: Pointer to the input vector containing the values to be transformed by the sigmoid function.
- **Control Flow**:
    - Iterates over each element in the input vector `x` from index 0 to `n-1`.
    - For each element `x[i]`, computes the sigmoid value using the formula 1 / (1 + exp(-x[i])).
    - Stores the computed sigmoid value in the corresponding index `y[i]` of the output vector.
- **Output**: The function does not return a value; instead, it populates the output vector `y` with the computed sigmoid values for each element in the input vector `x`.


---
### ggml\_vec\_sigmoid\_f16<!-- {{#callable:ggml_vec_sigmoid_f16}} -->
Computes the sigmoid function for each element in a vector of half-precision floating-point numbers.
- **Inputs**:
    - `n`: The number of elements in the input vector.
    - `y`: Pointer to the output vector where the results of the sigmoid function will be stored.
    - `x`: Pointer to the input vector containing half-precision floating-point numbers.
- **Control Flow**:
    - Iterates over each element of the input vector `x` from index 0 to `n-1`.
    - For each element, it converts the half-precision float to a single-precision float using `GGML_FP16_TO_FP32`.
    - Calculates the sigmoid value using the formula 1 / (1 + exp(-x[i])).
    - Converts the result back to half-precision using `GGML_FP32_TO_FP16` and stores it in the output vector `y`.
- **Output**: The output vector `y` contains the sigmoid values of the input vector `x`, with each value represented as a half-precision floating-point number.


---
### ggml\_vec\_hardswish\_f32<!-- {{#callable:ggml_vec_hardswish_f32}} -->
Applies the hard swish activation function element-wise to a vector.
- **Inputs**:
    - `n`: The number of elements in the input vector.
    - `y`: Pointer to the output vector where the results will be stored.
    - `x`: Pointer to the input vector on which the hard swish function will be applied.
- **Control Flow**:
    - Iterates over each element of the input vector `x` from index 0 to `n-1`.
    - For each element `x[i]`, computes the hard swish value using the formula: `x[i] * fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f))`.
    - Stores the computed value in the corresponding index `y[i]` of the output vector.
- **Output**: The function does not return a value; instead, it populates the output vector `y` with the results of the hard swish activation applied to each element of the input vector `x`.


---
### ggml\_vec\_hardswish\_f16<!-- {{#callable:ggml_vec_hardswish_f16}} -->
Applies the hard-swish activation function to each element of a vector of half-precision floating-point numbers.
- **Inputs**:
    - `n`: The number of elements in the input vector.
    - `y`: Pointer to the output vector where the results will be stored.
    - `x`: Pointer to the input vector containing half-precision floating-point numbers.
- **Control Flow**:
    - Iterates over each element of the input vector `x` from index 0 to `n-1`.
    - Converts each half-precision floating-point number in `x` to single-precision using `GGML_FP16_TO_FP32`.
    - Calculates the hard-swish value using the formula: v * fminf(1.0f, fmaxf(0.0f, (v + 3.0f) / 6.0f)).
    - Converts the result back to half-precision using `GGML_FP32_TO_FP16` and stores it in the output vector `y`.
- **Output**: The function does not return a value; instead, it populates the output vector `y` with the computed hard-swish values.


---
### ggml\_vec\_hardsigmoid\_f32<!-- {{#callable:ggml_vec_hardsigmoid_f32}} -->
Applies the hard sigmoid activation function to each element of the input vector.
- **Inputs**:
    - `n`: The number of elements in the input vector.
    - `y`: Pointer to the output vector where the results will be stored.
    - `x`: Pointer to the input vector containing the values to be transformed.
- **Control Flow**:
    - Iterates over each element of the input vector `x` from index 0 to `n-1`.
    - For each element, computes the hard sigmoid value using the formula: f(x) = min(1, max(0, (x[i] + 3) / 6)).
    - Stores the computed value in the corresponding index of the output vector `y`.
- **Output**: The function does not return a value; instead, it modifies the output vector `y` in place with the computed hard sigmoid values.


---
### ggml\_vec\_hardsigmoid\_f16<!-- {{#callable:ggml_vec_hardsigmoid_f16}} -->
Applies the hard sigmoid activation function to each element of the input vector.
- **Inputs**:
    - `n`: The number of elements in the input vector.
    - `y`: Pointer to the output vector where the results will be stored.
    - `x`: Pointer to the input vector containing the values to be transformed.
- **Control Flow**:
    - Iterates over each element of the input vector `x` from index 0 to `n-1`.
    - For each element, it computes the hard sigmoid value using the formula: f(x) = min(1, max(0, (x + 3) / 6)).
    - The computed value is then converted from float to half-precision float and stored in the output vector `y`.
- **Output**: The function does not return a value; instead, it populates the output vector `y` with the transformed values.


---
### ggml\_vec\_exp\_f32<!-- {{#callable:ggml_vec_exp_f32}} -->
Computes the element-wise exponential of an array of single-precision floating-point numbers.
- **Inputs**:
    - `n`: The number of elements in the input array.
    - `y`: Pointer to the output array where the results will be stored.
    - `x`: Pointer to the input array containing the values to be exponentiated.
- **Control Flow**:
    - A for loop iterates from 0 to n-1.
    - In each iteration, the exponential of the i-th element of the input array x is computed using the `expf` function.
    - The computed value is then stored in the i-th position of the output array y.
- **Output**: The function does not return a value; instead, it populates the output array y with the exponentials of the corresponding elements in the input array x.


---
### ggml\_vec\_exp\_f16<!-- {{#callable:ggml_vec_exp_f16}} -->
The `ggml_vec_exp_f16` function computes the element-wise exponential of an array of half-precision floating-point numbers.
- **Inputs**:
    - `n`: An integer representing the number of elements in the input array.
    - `y`: A pointer to an array of `ggml_fp16_t` where the results (exponentials) will be stored.
    - `x`: A pointer to an array of `ggml_fp16_t` containing the input values for which the exponential will be computed.
- **Control Flow**:
    - A for loop iterates from 0 to n, processing each element of the input array.
    - For each element, the function converts the half-precision floating-point value to single precision, computes its exponential using `expf`, and then converts the result back to half-precision before storing it in the output array.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the computed exponential values of the input array `x`.


---
### ggml\_gelu\_f32<!-- {{#callable:ggml_gelu_f32}} -->
Computes the Gaussian Error Linear Unit (GELU) activation function for a given float input.
- **Inputs**:
    - `x`: A float value representing the input to the GELU function.
- **Control Flow**:
    - The function calculates the output using the formula: 0.5 * x * (1.0 + tanh(SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x))).
    - It utilizes the hyperbolic tangent function `tanhf` to compute the non-linear transformation of the input.
- **Output**: Returns a float value that represents the output of the GELU activation function applied to the input x.


---
### ggml\_vec\_gelu\_f16<!-- {{#callable:ggml_vec_gelu_f16}} -->
The `ggml_vec_gelu_f16` function applies the Gaussian Error Linear Unit (GELU) activation function to a vector of half-precision floating-point numbers.
- **Inputs**:
    - `n`: An integer representing the number of elements in the input vector.
    - `y`: A pointer to an array of `ggml_fp16_t` where the output results will be stored.
    - `x`: A pointer to an array of `ggml_fp16_t` containing the input values to which the GELU function will be applied.
- **Control Flow**:
    - The function casts the input pointer `x` to a pointer of type `uint16_t` to access the underlying binary representation of the half-precision floats.
    - A loop iterates over each element from 0 to n-1, applying the precomputed GELU values from `ggml_table_gelu_f16` to the corresponding input values.
    - The results are stored in the output array `y`.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the GELU activation results for each input element.


---
### ggml\_vec\_gelu\_erf\_f16<!-- {{#callable:ggml_vec_gelu_erf_f16}} -->
Computes the Gaussian Error Linear Unit (GELU) activation function for a vector of half-precision floating-point numbers.
- **Inputs**:
    - `n`: The number of elements in the input vector.
    - `y`: A pointer to the output vector where the results will be stored.
    - `x`: A pointer to the input vector containing half-precision floating-point numbers.
- **Control Flow**:
    - Iterates over each element in the input vector `x` from index 0 to `n-1`.
    - Converts each half-precision floating-point number in `x` to single-precision using `GGML_FP16_TO_FP32`.
    - Calculates the GELU activation using the formula: 0.5 * xi * (1.0 + erff(xi * SQRT_2_INV).
    - Converts the result back to half-precision using `GGML_FP32_TO_FP16` and stores it in the output vector `y`.
- **Output**: The function does not return a value; instead, it populates the output vector `y` with the computed GELU values for each corresponding input in `x`.


---
### ggml\_vec\_gelu\_f32<!-- {{#callable:ggml_vec_gelu_f32}} -->
The `ggml_vec_gelu_f32` function applies the Gaussian Error Linear Unit (GELU) activation function to each element of an input vector.
- **Inputs**:
    - `n`: An integer representing the number of elements in the input vector.
    - `y`: A pointer to an array where the results of the GELU activation will be stored.
    - `x`: A pointer to the input array containing the values to which the GELU function will be applied.
- **Control Flow**:
    - The function iterates over each element of the input array `x` from index 0 to `n-1`.
    - For each element `x[i]`, it checks if the value is less than or equal to -10.0 or greater than or equal to 10.0.
    - If `x[i]` is less than or equal to -10.0, it sets `y[i]` to 0.0.
    - If `x[i]` is greater than or equal to 10.0, it sets `y[i]` to `x[i]`.
    - For values of `x[i]` within the range (-10.0, 10.0), it converts `x[i]` to half-precision float, retrieves the corresponding GELU value from a precomputed table, and stores it in `y[i]`.
- **Output**: The function outputs the results of the GELU activation function applied to each element of the input vector `x`, stored in the output vector `y`.
- **Functions called**:
    - [`ggml_gelu_f32`](#ggml_gelu_f32)


---
### ggml\_vec\_gelu\_erf\_f32<!-- {{#callable:ggml_vec_gelu_erf_f32}} -->
Computes the Gaussian Error Linear Unit (GELU) activation function using the error function for a vector of float values.
- **Inputs**:
    - `n`: The number of elements in the input vector.
    - `y`: Pointer to the output vector where the results of the GELU function will be stored.
    - `x`: Pointer to the input vector containing the values to be transformed by the GELU function.
- **Control Flow**:
    - Iterates over each element in the input vector `x` from index 0 to `n-1`.
    - For each element `xi` in `x`, computes the GELU value using the formula: 0.5 * xi * (1.0 + erff(xi * SQRT_2_INV).
    - Stores the computed value in the corresponding index of the output vector `y`.
- **Output**: The function does not return a value; instead, it populates the output vector `y` with the computed GELU values for each input element.


---
### ggml\_gelu\_quick\_f32<!-- {{#callable:ggml_gelu_quick_f32}} -->
Computes the quick Gaussian Error Linear Unit (GELU) activation function for a given float input.
- **Inputs**:
    - `x`: A float value representing the input to the GELU activation function.
- **Control Flow**:
    - The function calculates the output using the formula: x * (1.0 / (1.0 + expf(GELU_QUICK_COEF * x))).
    - It applies the exponential function to the product of a constant coefficient and the input value.
    - The result of the exponential function is used to compute the final output by dividing 1.0 by the sum of 1.0 and the exponential result, and then multiplying by the input x.
- **Output**: Returns a float value that represents the output of the quick GELU activation function.


---
### ggml\_vec\_gelu\_quick\_f32<!-- {{#callable:ggml_vec_gelu_quick_f32}} -->
The `ggml_vec_gelu_quick_f32` function applies the quick Gaussian Error Linear Unit (GELU) activation function to each element of an input vector.
- **Inputs**:
    - `n`: An integer representing the number of elements in the input vector.
    - `y`: A pointer to an array where the results of the GELU activation will be stored.
    - `x`: A pointer to an array containing the input values to which the GELU activation will be applied.
- **Control Flow**:
    - The function iterates over each element of the input array `x` from index 0 to `n-1`.
    - For each element `x[i]`, it computes the GELU activation using the [`ggml_gelu_quick_f32`](#ggml_gelu_quick_f32) function.
    - The result is stored in the corresponding index `y[i]` of the output array.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the results of the GELU activation applied to each element of the input array `x`.
- **Functions called**:
    - [`ggml_gelu_quick_f32`](#ggml_gelu_quick_f32)


---
### ggml\_vec\_gelu\_quick\_f16<!-- {{#callable:ggml_vec_gelu_quick_f16}} -->
The `ggml_vec_gelu_quick_f16` function applies the quick Gaussian Error Linear Unit (GELU) activation function to each element of a vector in half-precision floating point format.
- **Inputs**:
    - `n`: An integer representing the number of elements in the input vector.
    - `y`: A pointer to an array of `ggml_fp16_t` where the results of the GELU activation will be stored.
    - `x`: A pointer to an array of `ggml_fp16_t` containing the input values to which the GELU activation will be applied.
- **Control Flow**:
    - A for loop iterates over each element of the input vector from 0 to n-1.
    - Within the loop, each element of the input vector `x` is converted from half-precision to single-precision using `GGML_FP16_TO_FP32`.
    - The GELU activation function is computed using the formula: v * (1.0f / (1.0f + expf(GELU_QUICK_COEF * v))).
    - The result is then converted back to half-precision using `GGML_FP32_TO_FP16` and stored in the output array `y`.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the results of the GELU activation applied to each element of the input array `x`.


---
### ggml\_silu\_f32<!-- {{#callable:ggml_silu_f32}} -->
Computes the Sigmoid Linear Unit (SiLU) activation function for a given float input.
- **Inputs**:
    - `x`: A float value representing the input to the SiLU function.
- **Control Flow**:
    - The function calculates the exponential of the negative input using `expf(-x)`.
    - It then computes the output by dividing the input `x` by the sum of 1 and the computed exponential.
- **Output**: Returns a float value representing the result of the SiLU function, calculated as x / (1 + exp(-x)).


---
### ggml\_silu\_f16<!-- {{#callable:ggml_silu_f16}} -->
Computes the Sigmoid Linear Unit (SiLU) activation function for a half-precision floating point input.
- **Inputs**:
    - `x`: A half-precision floating point value (`ggml_fp16_t`) representing the input to the SiLU function.
- **Control Flow**:
    - Converts the input `x` from half-precision to single-precision floating point using `GGML_FP16_TO_FP32`.
    - Calculates the SiLU value using the formula: v / (1.0 + exp(-v)), where v is the converted single-precision value.
    - Converts the result back to half-precision using `GGML_FP32_TO_FP16` and returns it.
- **Output**: Returns the computed SiLU value as a half-precision floating point number (`ggml_fp16_t`).


---
### exp\_ps\_sve<!-- {{#callable:exp_ps_sve}} -->
Computes the exponential of a vector of single-precision floating-point numbers using SIMD operations.
- **Inputs**:
    - `pg`: A predicate mask of type `svbool_t` that determines which elements of the input vector are active for the computation.
    - `src`: A vector of type `svfloat32_t` containing the input values for which the exponential function will be computed.
- **Control Flow**:
    - The function begins by defining several constants used in the computation, including logarithmic constants and masks.
    - It computes the product of the input vector `src` and the constant `log2_e` to prepare for the exponential calculation.
    - The result is rounded to the nearest integer to determine the exponent `n` for the base 2.
    - The fractional part of the exponent is calculated and adjusted to create a new value `b`.
    - The integer part is then right-shifted to obtain `v`, which is used to compute the exponential of `v`.
    - The result is scaled by `2^n` to account for the integer exponent.
    - The function computes a correction term based on the fractional part and combines it with the scaled exponential result to produce the final output.
- **Output**: Returns a vector of type `svfloat32_t` containing the computed exponential values for the input elements.


---
### ggml\_v\_expf<!-- {{#callable:ggml_v_expf}} -->
Computes the exponential of each element in a vector using SIMD instructions.
- **Inputs**:
    - `x`: A `__m128` vector containing the input values for which the exponential function will be computed.
- **Control Flow**:
    - Initialize a constant vector `r` with a specific floating-point value.
    - Calculate `z` as a combination of `x` and `r` using a multiply-add operation.
    - Compute `n` as the difference between `z` and `r`.
    - Calculate `b` using a series of multiply and subtract operations involving `n` and `x`.
    - Extract the exponent part from `z` and create a vector `k` for scaling the result.
    - Check if `n` exceeds certain thresholds to determine the output calculation path.
    - If `n` is within bounds, compute the final result using a polynomial approximation.
    - If `n` is out of bounds, handle special cases for very large or very small values.
- **Output**: Returns a `__m128` vector containing the computed exponential values for each element in the input vector.


---
### ggml\_v\_silu<!-- {{#callable:ggml_v_silu}} -->
Computes the Sigmoid Linear Unit (SiLU) activation function for a vector of single-precision floating-point values.
- **Inputs**:
    - `x`: A vector of type `__m128` containing single-precision floating-point values to which the SiLU function will be applied.
- **Control Flow**:
    - Initialize a constant vector `one` with all elements set to 1.0.
    - Initialize a constant vector `zero` with all elements set to 0.0.
    - Negate the input vector `x` to create `neg_x`.
    - Compute the exponential of `neg_x` using the [`ggml_v_expf`](#ggml_v_expf) function, resulting in `exp_neg_x`.
    - Calculate `one_plus_exp_neg_x` by adding `one` to `exp_neg_x`.
    - Return the result of dividing `x` by `one_plus_exp_neg_x`.
- **Output**: Returns a vector of type `__m128` containing the computed SiLU values for each element in the input vector.
- **Functions called**:
    - [`ggml_v_expf`](#ggml_v_expf)


---
### ggml\_vec\_silu\_f16<!-- {{#callable:ggml_vec_silu_f16}} -->
The `ggml_vec_silu_f16` function applies the Sigmoid Linear Unit (SiLU) activation function to each element of a vector of half-precision floating-point numbers.
- **Inputs**:
    - `n`: An integer representing the number of elements in the input vector.
    - `y`: A pointer to an array of `ggml_fp16_t` where the results of the SiLU function will be stored.
    - `x`: A pointer to an array of `ggml_fp16_t` containing the input values to which the SiLU function will be applied.
- **Control Flow**:
    - The function iterates over each element of the input vector `x` from index 0 to `n-1`.
    - For each element, it computes the SiLU value using the [`ggml_silu_f16`](#ggml_silu_f16) function and stores the result in the corresponding index of the output vector `y`.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the SiLU results for each corresponding input in `x`.
- **Functions called**:
    - [`ggml_silu_f16`](#ggml_silu_f16)


---
### ggml\_silu\_backward\_f32<!-- {{#callable:ggml_silu_backward_f32}} -->
Calculates the gradient of the Sigmoid Linear Unit (SiLU) function with respect to its input.
- **Inputs**:
    - `x`: The input value for which the SiLU gradient is calculated.
    - `dy`: The gradient of the loss with respect to the output of the SiLU function.
- **Control Flow**:
    - Calculates the sigmoid of the input `x` using the formula s = 1.0f / (1.0f + expf(-x)).
    - Returns the product of `dy`, the sigmoid value `s`, and the expression (1.0f + x * (1.0f - s)).
- **Output**: Returns the computed gradient of the SiLU function with respect to the input `x`.


---
### ggml\_silu\_backward\_f16<!-- {{#callable:ggml_silu_backward_f16}} -->
Calculates the backward pass of the Sigmoid Linear Unit (SiLU) activation function for half-precision floating point inputs.
- **Inputs**:
    - `x`: Input value in half-precision floating point format (ggml_fp16_t) representing the input to the SiLU function.
    - `dy`: Input value in half-precision floating point format (ggml_fp16_t) representing the gradient of the loss with respect to the output of the SiLU function.
- **Control Flow**:
    - Converts the input `x` from half-precision to single-precision floating point using `GGML_FP16_TO_FP32`.
    - Calculates the sigmoid of the input `v` using the formula `s = 1.0f/(1.0f + expf(-v))`.
    - Computes the gradient of the SiLU function using the formula `dy * s * (1.0f + v * (1.0f - s))`.
    - Converts the result back to half-precision floating point using `GGML_FP32_TO_FP16` and returns it.
- **Output**: Returns the computed gradient in half-precision floating point format (ggml_fp16_t).


---
### ggml\_vec\_silu\_backward\_f32<!-- {{#callable:ggml_vec_silu_backward_f32}} -->
Computes the backward pass of the Sigmoid Linear Unit (SiLU) function for a vector of inputs.
- **Inputs**:
    - `n`: The number of elements in the input vectors.
    - `dx`: Pointer to the output vector where the gradients will be stored.
    - `x`: Pointer to the input vector containing the original values.
    - `dy`: Pointer to the input vector containing the gradients from the next layer.
- **Control Flow**:
    - Iterates over each element from 0 to n-1.
    - For each index i, computes the gradient using the [`ggml_silu_backward_f32`](#ggml_silu_backward_f32) function with the corresponding elements from x and dy.
    - Stores the computed gradient in the output vector dx.
- **Output**: The function does not return a value; instead, it populates the output vector dx with the computed gradients.
- **Functions called**:
    - [`ggml_silu_backward_f32`](#ggml_silu_backward_f32)


---
### ggml\_vec\_silu\_backward\_f16<!-- {{#callable:ggml_vec_silu_backward_f16}} -->
Computes the backward pass of the Sigmoid Linear Unit (SiLU) activation function for a vector of half-precision floating-point values.
- **Inputs**:
    - `n`: The number of elements in the input vectors.
    - `dx`: Pointer to the output vector where the gradients will be stored.
    - `x`: Pointer to the input vector containing the original values.
    - `dy`: Pointer to the input vector containing the gradients from the next layer.
- **Control Flow**:
    - Iterates over each element from 0 to n-1.
    - For each index i, computes the gradient using the [`ggml_silu_backward_f16`](#ggml_silu_backward_f16) function with the corresponding elements from x and dy.
    - Stores the computed gradient in the output vector dx.
- **Output**: The function does not return a value; instead, it populates the output vector dx with the computed gradients.
- **Functions called**:
    - [`ggml_silu_backward_f16`](#ggml_silu_backward_f16)


---
### ggml\_vec\_sum\_f32<!-- {{#callable:ggml_vec_sum_f32}} -->
Calculates the sum of a vector of floats.
- **Inputs**:
    - `n`: The number of elements in the input vector.
    - `s`: A pointer to a float where the result (sum) will be stored.
    - `x`: A pointer to the input vector of floats.
- **Control Flow**:
    - Checks if the `GGML_USE_ACCELERATE` macro is defined.
    - If not defined, initializes a sum variable to 0.0 and iterates over the input vector `x`, adding each element to the sum.
    - Stores the final sum in the location pointed to by `s`.
    - If `GGML_USE_ACCELERATE` is defined, uses the Accelerate framework's `vDSP_sve` function to compute the sum.
- **Output**: The function outputs the sum of the elements in the input vector `x` into the variable pointed to by `s`.


---
### ggml\_vec\_sum\_f32\_ggf<!-- {{#callable:ggml_vec_sum_f32_ggf}} -->
Calculates the sum of a vector of `float` values and stores the result in a specified location.
- **Inputs**:
    - `n`: The number of elements in the input vector.
    - `s`: A pointer to a `ggml_float` where the sum will be stored.
    - `x`: A pointer to an array of `float` values that will be summed.
- **Control Flow**:
    - Initializes a variable `sum` to 0.0 to hold the cumulative sum.
    - Iterates over each element in the input array `x` from index 0 to `n-1`.
    - In each iteration, adds the current element `x[i]` (cast to `ggml_float`) to `sum`.
    - After the loop, assigns the final value of `sum` to the location pointed to by `s`.
- **Output**: The function does not return a value; instead, it outputs the sum of the elements in the input array `x` through the pointer `s`.


---
### ggml\_vec\_sum\_f16\_ggf<!-- {{#callable:ggml_vec_sum_f16_ggf}} -->
Calculates the sum of a vector of half-precision floating-point numbers and stores the result in a single-precision floating-point variable.
- **Inputs**:
    - `n`: The number of elements in the input vector.
    - `s`: A pointer to a float where the sum will be stored.
    - `x`: A pointer to an array of `ggml_fp16_t` (half-precision floating-point) values to be summed.
- **Control Flow**:
    - Initialize a float variable `sum` to 0.0f.
    - Iterate over the range from 0 to `n`.
    - In each iteration, convert the `ggml_fp16_t` value at index `i` to a float using `GGML_FP16_TO_FP32` and add it to `sum`.
    - After the loop, store the final sum in the location pointed to by `s`.
- **Output**: The function does not return a value; instead, it outputs the computed sum through the pointer `s`.


---
### ggml\_vec\_sum\_bf16\_ggf<!-- {{#callable:ggml_vec_sum_bf16_ggf}} -->
Calculates the sum of a vector of `ggml_bf16_t` values and stores the result in a float pointer.
- **Inputs**:
    - `n`: An integer representing the number of elements in the input vector.
    - `s`: A pointer to a float where the resulting sum will be stored.
    - `x`: A pointer to an array of `ggml_bf16_t` values to be summed.
- **Control Flow**:
    - Initializes a float variable `sum` to 0.0f.
    - Iterates over the range from 0 to `n`.
    - In each iteration, converts the `ggml_bf16_t` value at index `i` to a float using `GGML_BF16_TO_FP32` and adds it to `sum`.
    - After the loop, assigns the total `sum` to the location pointed to by `s`.
- **Output**: The function outputs the total sum of the input vector as a float, stored at the address pointed to by `s`.


---
### ggml\_vec\_max\_f32<!-- {{#callable:ggml_vec_max_f32}} -->
Finds the maximum value in a vector of floats.
- **Inputs**:
    - `n`: The number of elements in the input vector.
    - `s`: A pointer to a float where the maximum value will be stored.
    - `x`: A pointer to the input vector of floats.
- **Control Flow**:
    - If `GGML_USE_ACCELERATE` is not defined, initialize `max` to negative infinity.
    - Iterate over each element in the input vector `x` up to `n`.
    - Update `max` with the maximum value found between the current `max` and the current element `x[i]`.
    - Store the final maximum value in the location pointed to by `s`.
    - If `GGML_USE_ACCELERATE` is defined, use the Accelerate framework's `vDSP_maxv` function to compute the maximum value.
- **Output**: The maximum value found in the input vector is stored in the location pointed to by `s`.


---
### ggml\_vec\_norm\_inv\_f32<!-- {{#callable:ggml_vec_norm_inv_f32}} -->
Calculates the inverse of the norm of a vector.
- **Inputs**:
    - `n`: The number of elements in the vector.
    - `s`: A pointer to a float where the result (inverse of the norm) will be stored.
    - `x`: A pointer to the input vector of floats.
- **Control Flow**:
    - Calls the [`ggml_vec_norm_f32`](#ggml_vec_norm_f32) function to compute the norm of the vector `x` and store it in `s`.
    - Calculates the inverse of the computed norm by taking `1.f / (*s)` and updates the value pointed to by `s`.
- **Output**: The function does not return a value; instead, it updates the value pointed to by `s` with the inverse of the norm of the vector `x`.
- **Functions called**:
    - [`ggml_vec_norm_f32`](#ggml_vec_norm_f32)


---
### ggml\_vec\_argmax\_f32<!-- {{#callable:ggml_vec_argmax_f32}} -->
Finds the index of the maximum value in a float array.
- **Inputs**:
    - `n`: The number of elements in the array.
    - `s`: A pointer to an integer where the index of the maximum value will be stored.
    - `x`: A pointer to the array of float values to be searched.
- **Control Flow**:
    - Initialize `max` to negative infinity and `idx` to 0.
    - Iterate over each element in the array `x` from index 0 to `n-1`.
    - For each element, update `max` if the current element is greater than `max`.
    - If the current element equals `max`, update `idx` to the current index.
    - After the loop, store the index of the maximum value in the location pointed to by `s`.
- **Output**: The function does not return a value; instead, it updates the integer pointed to by `s` with the index of the maximum value found in the array.


# Function Declarations (Public API)

---
### ggml\_vec\_dot\_f32<!-- {{#callable_declaration:ggml_vec_dot_f32}} -->
Computes the dot product of two float vectors.
- **Description**: This function calculates the dot product of two float arrays, `x` and `y`, each of length `n`, and stores the result in the location pointed to by `s`. It is designed to be used in scenarios where vectorized operations are beneficial for performance. The function requires that `nrc` is set to 1, and it is expected that the input arrays are properly aligned and non-null. The function does not handle invalid input values, so the caller must ensure that all preconditions are met before calling this function.
- **Inputs**:
    - `n`: The number of elements in the vectors `x` and `y`. Must be a positive integer.
    - `s`: A pointer to a float where the result of the dot product will be stored. Must not be null.
    - `bs`: Byte stride for the result pointer `s`. This parameter is unused in the function.
    - `x`: A pointer to the first float vector. Must not be null and should have at least `n` elements.
    - `bx`: Byte stride for the vector `x`. This parameter is unused in the function.
    - `y`: A pointer to the second float vector. Must not be null and should have at least `n` elements.
    - `by`: Byte stride for the vector `y`. This parameter is unused in the function.
    - `nrc`: Must be set to 1. The function asserts this value and does not operate correctly if it is different.
- **Output**: The result of the dot product is stored in the location pointed to by `s`. The function does not return a value.
- **See also**: [`ggml_vec_dot_f32`](vec.cpp.driver.md#ggml_vec_dot_f32)  (Implementation)


---
### ggml\_vec\_dot\_bf16<!-- {{#callable_declaration:ggml_vec_dot_bf16}} -->
Computes the dot product of two vectors with bfloat16 elements.
- **Description**: This function calculates the dot product of two vectors, `x` and `y`, each containing `n` elements of type `ggml_bf16_t`, and stores the result in the location pointed to by `s`. It is designed for use in environments where bfloat16 precision is sufficient and can leverage SIMD instructions for performance optimization. The function requires that `nrc` is set to 1, and it is expected that the caller ensures this precondition. The function does not modify the input vectors `x` and `y`, and the result is accumulated in a floating-point format to minimize precision loss.
- **Inputs**:
    - `n`: The number of elements in the vectors `x` and `y`. Must be a positive integer.
    - `s`: A pointer to a float where the result of the dot product will be stored. Must not be null.
    - `bs`: Byte stride for the result pointer `s`. This parameter is unused in the function.
    - `x`: A pointer to the first vector of `ggml_bf16_t` elements. Must not be null.
    - `bx`: Byte stride for the vector `x`. This parameter is unused in the function.
    - `y`: A pointer to the second vector of `ggml_bf16_t` elements. Must not be null.
    - `by`: Byte stride for the vector `y`. This parameter is unused in the function.
    - `nrc`: Must be set to 1. The function asserts this condition and will not operate correctly if it is not met.
- **Output**: The result of the dot product is stored in the location pointed to by `s`.
- **See also**: [`ggml_vec_dot_bf16`](vec.cpp.driver.md#ggml_vec_dot_bf16)  (Implementation)


---
### ggml\_vec\_dot\_f16<!-- {{#callable_declaration:ggml_vec_dot_f16}} -->
Computes the dot product of two half-precision floating-point vectors.
- **Description**: This function calculates the dot product of two vectors, `x` and `y`, each containing `n` elements of half-precision floating-point numbers (fp16). The result is stored in the location pointed to by `s`, which is a single-precision floating-point number (float). This function is useful in scenarios where vector operations are needed, such as in machine learning or signal processing applications. It is important to ensure that the `nrc` parameter is set to 1, as this is a precondition for the function to operate correctly. The function does not handle invalid input values for `nrc` and assumes that the input vectors are valid and properly aligned.
- **Inputs**:
    - `n`: The number of elements in the vectors `x` and `y`. Must be a positive integer.
    - `s`: A pointer to a float where the result of the dot product will be stored. Must not be null. Caller retains ownership.
    - `bs`: Byte stride for the result pointer `s`. This parameter is unused in the function.
    - `x`: A pointer to the first vector of half-precision floating-point numbers. Must not be null. Caller retains ownership.
    - `bx`: Byte stride for the vector `x`. This parameter is unused in the function.
    - `y`: A pointer to the second vector of half-precision floating-point numbers. Must not be null. Caller retains ownership.
    - `by`: Byte stride for the vector `y`. This parameter is unused in the function.
    - `nrc`: Must be set to 1. This is a precondition for the function to operate correctly.
- **Output**: The result of the dot product is stored in the location pointed to by `s`.
- **See also**: [`ggml_vec_dot_f16`](vec.cpp.driver.md#ggml_vec_dot_f16)  (Implementation)


---
### ggml\_vec\_silu\_f32<!-- {{#callable_declaration:ggml_vec_silu_f32}} -->
Applies the SiLU activation function to each element of a float array.
- **Description**: This function processes an array of single-precision floating-point numbers, applying the Sigmoid Linear Unit (SiLU) activation function to each element. It is typically used in neural network computations where the SiLU function is desired for its smooth, non-linear properties. The function requires the input and output arrays to be distinct and of the same length, specified by the parameter `n`. The input array `x` is read-only, while the output array `y` is modified in place to store the results. Ensure that `y` has sufficient space to hold `n` elements.
- **Inputs**:
    - `n`: The number of elements in the input and output arrays. Must be a non-negative integer.
    - `y`: A pointer to an array of floats where the results will be stored. Must not be null and must have space for at least `n` elements. The caller is responsible for managing the memory of this array.
    - `x`: A pointer to a read-only array of floats containing the input data. Must not be null and must have at least `n` elements.
- **Output**: None
- **See also**: [`ggml_vec_silu_f32`](vec.cpp.driver.md#ggml_vec_silu_f32)  (Implementation)


---
### ggml\_vec\_soft\_max\_f32<!-- {{#callable_declaration:ggml_vec_soft_max_f32}} -->
Computes the softmax function on a vector of floats.
- **Description**: This function calculates the softmax of a vector of floats, storing the result in the output array and returning the sum of the exponentials. It is typically used in machine learning to convert a vector of raw scores into probabilities. The function requires the input vector to be preprocessed to subtract the maximum value for numerical stability. The output array must be preallocated and have the same size as the input vector. The function assumes that the input and output arrays are valid and non-null.
- **Inputs**:
    - `n`: The number of elements in the input and output arrays. Must be a positive integer.
    - `y`: A pointer to a preallocated array of floats where the softmax results will be stored. Must not be null.
    - `x`: A pointer to an array of floats representing the input vector. Must not be null.
    - `max`: A float representing the maximum value in the input vector, used for numerical stability.
- **Output**: Returns the sum of the exponentials of the input vector elements after subtracting the maximum value.
- **See also**: [`ggml_vec_soft_max_f32`](vec.cpp.driver.md#ggml_vec_soft_max_f32)  (Implementation)


---
### ggml\_vec\_log\_soft\_max\_f32<!-- {{#callable_declaration:ggml_vec_log_soft_max_f32}} -->
Computes the logarithm of the softmax function for a vector of floats.
- **Description**: This function calculates the logarithm of the softmax function for a given vector of floats, storing the intermediate results in the output array. It is typically used in machine learning applications where the log of the softmax is required for numerical stability or further computations. The function requires the caller to provide the maximum value of the input vector to ensure numerical stability. The input and output arrays must not be null, and the function assumes that the input array contains at least 'n' elements.
- **Inputs**:
    - `n`: The number of elements in the input and output arrays. Must be a positive integer.
    - `y`: A pointer to an array of floats where the intermediate results will be stored. Must not be null and must have space for at least 'n' elements. The caller retains ownership.
    - `x`: A pointer to an array of floats representing the input vector. Must not be null and must contain at least 'n' elements. The caller retains ownership.
    - `max`: A float representing the maximum value in the input array 'x'. This is used to improve numerical stability during computation.
- **Output**: Returns the logarithm of the sum of exponentials of the input vector elements, adjusted by the provided maximum value.
- **See also**: [`ggml_vec_log_soft_max_f32`](vec.cpp.driver.md#ggml_vec_log_soft_max_f32)  (Implementation)


