# Purpose
The provided C++ code is a comprehensive library of functions designed to perform a wide range of tensor operations essential for machine learning and neural network computations. It includes operations such as duplication, addition, scaling, convolution, pooling, normalization, and more complex mechanisms like flash attention and RWKV, all implemented in a modular fashion to ensure flexibility and reusability across different neural network architectures. The code is structured to handle various data types and memory layouts, supporting both floating-point and quantized types, and is optimized for high-performance computing with multi-threading capabilities. While it does not serve as a standalone executable, it defines a set of public APIs intended for integration into larger machine learning frameworks, ensuring efficient, scalable, and robust execution. This library is a critical component for building and training neural networks, particularly in image processing and sequence modeling, offering custom operations and error handling to enhance its versatility and reliability.
# Imports and Dependencies

---
- `ops.h`
- `ggml-cpu.h`
- `ggml-impl.h`
- `binary-ops.h`
- `unary-ops.h`
- `vec.h`
- `float.h`


# Data Structures

---
### ggml\_conv\_2d\_dw\_params<!-- {{#data_structure:ggml_conv_2d_dw_params}} -->
- **Type**: `struct`
- **Members**:
    - `channels`: Represents the number of channels in the input data.
    - `batch`: Indicates the batch size for the convolution operation.
    - `src_w`: Specifies the width of the source/input data.
    - `src_h`: Specifies the height of the source/input data.
    - `dst_w`: Specifies the width of the destination/output data.
    - `dst_h`: Specifies the height of the destination/output data.
    - `knl_w`: Defines the width of the convolution kernel.
    - `knl_h`: Defines the height of the convolution kernel.
    - `stride_x`: Indicates the stride along the x-axis for the convolution.
    - `stride_y`: Indicates the stride along the y-axis for the convolution.
    - `pad_x`: Specifies the padding along the x-axis.
    - `pad_y`: Specifies the padding along the y-axis.
    - `dilation_x`: Defines the dilation rate along the x-axis for the convolution.
    - `dilation_y`: Defines the dilation rate along the y-axis for the convolution.
- **Description**: The `ggml_conv_2d_dw_params` struct is designed to encapsulate parameters for a 2D depthwise convolution operation. It includes fields for specifying the dimensions of the input and output data, the size of the convolution kernel, and the configuration of the convolution operation such as stride, padding, and dilation. This struct is essential for setting up and executing depthwise convolution operations in a neural network or image processing context, where each input channel is convolved with a separate kernel.


# Functions

---
### ggml\_compute\_forward\_dup\_same\_cont<!-- {{#callable:ggml_compute_forward_dup_same_cont}} -->
The `ggml_compute_forward_dup_same_cont` function duplicates data from a source tensor to a destination tensor in a parallelized manner, ensuring both tensors are contiguous and of the same type.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for computation, including the thread index and the total number of threads.
    - `dst`: A pointer to a `ggml_tensor` structure representing the destination tensor where data will be copied.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It asserts that the number of elements in `dst` matches that of `src0`, both tensors are contiguous, and they are of the same type.
    - The size of each element in the source tensor is determined using [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size).
    - The function calculates the number of elements in blocks for parallel processing and determines the range of elements to be processed by the current thread.
    - If the calculated range is valid (i.e., `k0 < k1`), it performs a memory copy from the source tensor to the destination tensor for the specified range.
- **Output**: The function does not return a value; it directly modifies the destination tensor `dst` by copying data from the source tensor `src0`.
- **Functions called**:
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_blck_size`](../ggml.c.driver.md#ggml_blck_size)


---
### ggml\_compute\_forward\_dup\_f16<!-- {{#callable:ggml_compute_forward_dup_f16}} -->
The `ggml_compute_forward_dup_f16` function performs a forward duplication operation on a tensor, copying data from a source tensor to a destination tensor with support for different data types and parallel processing.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for computation, including thread index and total number of threads.
    - `dst`: A pointer to a `ggml_tensor` structure representing the destination tensor where data will be copied.
- **Control Flow**:
    - The function begins by asserting that the number of elements in the destination tensor matches that of the source tensor.
    - It initializes local variables for thread management and calculates the range of rows to process for the current thread.
    - If the source and destination tensors have the same type and compatible sizes, it performs a direct memory copy of the data for the specified row range.
    - If the tensors are contiguous, it handles specific data types (e.g., `F16`, `F32`) and performs the necessary conversions or copies based on the tensor type.
    - If the tensors are not contiguous, it iterates through the tensor dimensions and performs element-wise copying, adjusting indices as necessary.
- **Output**: The function does not return a value; it modifies the destination tensor in place by copying data from the source tensor according to the specified parameters and conditions.
- **Functions called**:
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_get_type_traits_cpu`](ggml-cpu.c.driver.md#ggml_get_type_traits_cpu)
    - [`ggml_blck_size`](../ggml.c.driver.md#ggml_blck_size)


---
### ggml\_compute\_forward\_dup\_bf16<!-- {{#callable:ggml_compute_forward_dup_bf16}} -->
The `ggml_compute_forward_dup_bf16` function performs a forward duplication operation on a tensor, handling various data types and parallelizing the operation across multiple threads.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for computation, including thread index and total number of threads.
    - `dst`: A pointer to a `ggml_tensor` structure representing the destination tensor where the duplicated data will be stored.
- **Control Flow**:
    - The function begins by asserting that the number of elements in the destination tensor matches that of the source tensor.
    - It calculates the number of rows to process per thread and determines the range of rows for the current thread.
    - If the source and destination tensors have the same type and compatible sizes, it performs a direct memory copy of the data.
    - If the destination tensor is contiguous, it handles specific data types (BF16, F16, F32) and performs the necessary conversions or copies.
    - If the destination tensor is not contiguous, it iterates through the tensor dimensions and performs element-wise copying or conversion based on the destination type.
    - The function includes error handling for unsupported types and conditions.
- **Output**: The function does not return a value; it modifies the destination tensor in place with the duplicated data from the source tensor.
- **Functions called**:
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_get_type_traits_cpu`](ggml-cpu.c.driver.md#ggml_get_type_traits_cpu)
    - [`ggml_blck_size`](../ggml.c.driver.md#ggml_blck_size)


---
### ggml\_compute\_forward\_dup\_f32<!-- {{#callable:ggml_compute_forward_dup_f32}} -->
The `ggml_compute_forward_dup_f32` function duplicates the data from a source tensor to a destination tensor, handling various data types and optimizing for performance through parallel processing.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for computation, including thread index and total number of threads.
    - `dst`: A pointer to a `ggml_tensor` structure representing the destination tensor where data will be duplicated.
- **Control Flow**:
    - The function begins by asserting that the number of elements in the destination tensor matches that of the source tensor.
    - It initializes local variables for managing parallel processing based on the number of threads.
    - If the source and destination tensors are of the same type and size, it performs a direct memory copy of the data in a row-wise manner.
    - If the destination tensor is contiguous, it checks the data type and performs the appropriate copying or quantization based on the type.
    - If the destination tensor is not contiguous, it calculates the appropriate indices and copies data element by element, handling different data types (float, half-float, bfloat) accordingly.
    - The function includes error handling for unsupported types and ensures that data is copied correctly across different tensor dimensions.
- **Output**: The function does not return a value; it modifies the destination tensor in place by duplicating the data from the source tensor.
- **Functions called**:
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_get_type_traits_cpu`](ggml-cpu.c.driver.md#ggml_get_type_traits_cpu)
    - [`ggml_blck_size`](../ggml.c.driver.md#ggml_blck_size)


---
### ggml\_compute\_forward\_dup\_bytes<!-- {{#callable:ggml_compute_forward_dup_bytes}} -->
This function duplicates the contents of a source tensor into a destination tensor, handling various memory layouts and parallelization.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation, including threading information.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the data will be duplicated.
- **Control Flow**:
    - The function begins by asserting that the number of elements in the destination tensor matches that of the source tensor and that both tensors are of the same type.
    - If both the source and destination tensors are contiguous in memory, it calls [`ggml_compute_forward_dup_same_cont`](#ggml_compute_forward_dup_same_cont) to handle the duplication efficiently.
    - The function calculates the number of rows to process per thread and determines the range of rows for the current thread.
    - If the source and destination tensors have the same shape and type, it performs a row-wise copy using `memcpy`.
    - If the destination tensor is contiguous but the source is not, it handles the copying by iterating through the dimensions and copying data accordingly.
    - If neither tensor is contiguous, it uses a more complex nested loop structure to copy data while managing the indices for both source and destination tensors.
- **Output**: The function does not return a value; it modifies the destination tensor in place by duplicating the data from the source tensor.
- **Functions called**:
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_compute_forward_dup_same_cont`](#ggml_compute_forward_dup_same_cont)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_row_size`](../ggml.c.driver.md#ggml_row_size)
    - [`ggml_blck_size`](../ggml.c.driver.md#ggml_blck_size)


---
### ggml\_compute\_forward\_dup\_q<!-- {{#callable:ggml_compute_forward_dup_q}} -->
The `ggml_compute_forward_dup_q` function performs a forward computation for duplicating quantized data from a source tensor to a destination tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for the computation, including thread indices.
    - `dst`: A pointer to a `ggml_tensor` structure representing the destination tensor where the computed results will be stored.
- **Control Flow**:
    - The function retrieves the source tensors `src0` and `src1` from the destination tensor `dst`.
    - It asserts that the destination tensor is contiguous in the first dimension and that it has the appropriate size.
    - The function calculates the number of rows `nr` in the source tensor and determines the range of rows to process for the current thread based on the parameters.
    - A loop iterates over the assigned range of rows, calculating offsets for both the source and destination tensors.
    - For each row, it calls the `dequantize_row_q` function to convert the quantized data from `src0` to floating-point format in `dst`.
- **Output**: The function does not return a value; it modifies the destination tensor `dst` in place with the dequantized data.
- **Functions called**:
    - [`ggml_get_type_traits`](../ggml.c.driver.md#ggml_get_type_traits)
    - [`ggml_blck_size`](../ggml.c.driver.md#ggml_blck_size)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)


---
### ggml\_compute\_forward\_dup<!-- {{#callable:ggml_compute_forward_dup}} -->
Computes the forward duplication of a tensor based on its type and parameters.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the duplication will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks if the type of `src0` matches the type of `dst` and calls [`ggml_compute_forward_dup_bytes`](#ggml_compute_forward_dup_bytes) if they match.
    - If the types do not match, it enters a switch statement to handle different tensor types: `GGML_TYPE_F16`, `GGML_TYPE_BF16`, and `GGML_TYPE_F32`, calling the respective duplication functions.
    - If `src0` is quantized and `dst` is of type `GGML_TYPE_F32`, it calls [`ggml_compute_forward_dup_q`](#ggml_compute_forward_dup_q).
    - If none of the conditions are met, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place based on the duplication logic determined by the input tensor types.
- **Functions called**:
    - [`ggml_compute_forward_dup_bytes`](#ggml_compute_forward_dup_bytes)
    - [`ggml_compute_forward_dup_f16`](#ggml_compute_forward_dup_f16)
    - [`ggml_compute_forward_dup_bf16`](#ggml_compute_forward_dup_bf16)
    - [`ggml_compute_forward_dup_f32`](#ggml_compute_forward_dup_f32)
    - [`ggml_is_quantized`](../ggml.c.driver.md#ggml_is_quantized)
    - [`ggml_compute_forward_dup_q`](#ggml_compute_forward_dup_q)


---
### ggml\_compute\_forward\_add\_q\_f32<!-- {{#callable:ggml_compute_forward_add_q_f32}} -->
Computes the element-wise addition of two quantized tensors and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation, including thread information and workspace data.
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the addition will be stored.
- **Control Flow**:
    - The function begins by retrieving the source tensors `src0` and `src1` from the destination tensor `dst`.
    - It asserts that the shapes of `src0`, `src1`, and `dst` are the same.
    - The number of rows in `src0` is determined and stored in `nr`.
    - Local variables for binary operations are initialized, including thread indices and tensor types.
    - Assertions are made to ensure the tensor types and sizes are compatible for the operation.
    - The number of rows processed per thread is calculated, and the range of rows for the current thread is determined.
    - A loop iterates over the assigned row range, performing the following for each row:
    -   - Calculates the indices for accessing the data in `src0`, `src1`, and `dst`.
    -   - Retrieves the corresponding rows from `src0`, `src1`, and `dst`.
    -   - Dequantizes the row from `src0` into a temporary buffer.
    -   - Adds the values from `src1` to the temporary buffer.
    -   - Quantizes the result back into the destination tensor `dst`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the result of the addition of the two source tensors.
- **Functions called**:
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_get_type_traits`](../ggml.c.driver.md#ggml_get_type_traits)
    - [`ggml_get_type_traits_cpu`](ggml-cpu.c.driver.md#ggml_get_type_traits_cpu)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_is_quantized`](../ggml.c.driver.md#ggml_is_quantized)
    - [`ggml_vec_acc_f32`](vec.h.driver.md#ggml_vec_acc_f32)


---
### ggml\_compute\_forward\_add<!-- {{#callable:ggml_compute_forward_add}} -->
The `ggml_compute_forward_add` function computes the forward addition operation for a given tensor based on its data type.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result of the addition will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of `src0` using a switch statement to determine the appropriate computation method.
    - If `src0` is of a non-quantized type (F32, F16, BF16), it calls [`ggml_compute_forward_add_non_quantized`](binary-ops.cpp.driver.md#ggml_compute_forward_add_non_quantized) to perform the addition.
    - If `src0` is of a quantized type (various Q types), it calls [`ggml_compute_forward_add_q_f32`](#ggml_compute_forward_add_q_f32) to handle the addition for quantized tensors.
    - If `src0` has an unsupported type, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to store the result of the addition operation.
- **Functions called**:
    - [`ggml_compute_forward_add_non_quantized`](binary-ops.cpp.driver.md#ggml_compute_forward_add_non_quantized)
    - [`ggml_compute_forward_add_q_f32`](#ggml_compute_forward_add_q_f32)


---
### ggml\_compute\_forward\_add1\_f32<!-- {{#callable:ggml_compute_forward_add1_f32}} -->
Computes the element-wise addition of a scalar tensor to each element of a source tensor and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for the computation, including the thread index and total number of threads.
    - `dst`: A pointer to a `ggml_tensor` that serves as the destination tensor where the result of the addition will be stored.
- **Control Flow**:
    - The function retrieves the source tensors `src0` and `src1` from the destination tensor `dst`.
    - It asserts that `src0` and `dst` have the same shape and that `src1` is a scalar tensor.
    - The number of rows in `src0` is determined, and the number of rows to be processed by each thread is calculated.
    - The function calculates the range of rows (`ir0` to `ir1`) that the current thread will process.
    - A loop iterates over the assigned row range, calculating the appropriate indices for accessing elements in the tensors.
    - Depending on whether the `GGML_USE_ACCELERATE` macro is defined, it either uses the Accelerate framework's vector addition function or a custom vector addition function to perform the addition of the scalar to each element of `src0` and store the result in `dst`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the result of the addition operation.
- **Functions called**:
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_is_scalar`](../ggml.c.driver.md#ggml_is_scalar)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_vec_add1_f32`](vec.h.driver.md#ggml_vec_add1_f32)


---
### ggml\_compute\_forward\_add1\_f16\_f32<!-- {{#callable:ggml_compute_forward_add1_f16_f32}} -->
Computes the element-wise addition of a scalar value to a tensor of type `ggml_fp16_t` and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation, including thread index and total number of threads.
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the addition will be stored.
- **Control Flow**:
    - The function begins by retrieving the source tensors `src0` and `src1` from the destination tensor `dst`.
    - It asserts that `src0` and `dst` have the same shape and that `src1` is a scalar.
    - The scalar value to be added is extracted from `src1`.
    - The function calculates the number of rows in `src0` and determines the range of rows to be processed by the current thread based on the total number of threads.
    - A loop iterates over the assigned range of rows, performing the addition of the scalar value to each element of `src0` and storing the result in `dst`.
    - Inside the loop, the function calculates the appropriate indices for accessing the elements of the tensors.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place, storing the result of the addition operation.
- **Functions called**:
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_is_scalar`](../ggml.c.driver.md#ggml_is_scalar)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)


---
### ggml\_compute\_forward\_add1\_f16\_f16<!-- {{#callable:ggml_compute_forward_add1_f16_f16}} -->
Computes the element-wise addition of a scalar to a tensor in half-precision floating point format.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation, including thread index and total number of threads.
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the addition will be stored.
- **Control Flow**:
    - The function begins by retrieving the source tensors `src0` and `src1` from the destination tensor `dst`.
    - It asserts that `src0` and `dst` have the same shape and that `src1` is a scalar.
    - The scalar value to be added is extracted from `src1` and converted from half-precision to single-precision float.
    - The function calculates the number of rows in `src0` and determines the range of rows to be processed by the current thread based on the total number of threads.
    - It iterates over the assigned range of rows, calculating the appropriate indices for accessing elements in the tensors.
    - For each element in the specified row, it performs the addition of the scalar to the corresponding element in `src0` and stores the result in `dst`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the result of the addition.
- **Functions called**:
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_is_scalar`](../ggml.c.driver.md#ggml_is_scalar)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)


---
### ggml\_compute\_forward\_add1\_q\_f32<!-- {{#callable:ggml_compute_forward_add1_q_f32}} -->
Computes the forward addition of a scalar value to each element of a quantized tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for computation, including thread index and number of threads.
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the addition will be stored.
- **Control Flow**:
    - Retrieve the source tensors `src0` and `src1` from the destination tensor `dst`.
    - Assert that `src0` and `dst` have the same shape and that `src1` is a scalar.
    - Extract the scalar value from `src1` for addition.
    - Calculate the number of rows in `src0` and determine the range of rows to process for the current thread.
    - Loop through the assigned rows, unquantizing each row from `src0`, adding the scalar value, and then quantizing the result back to `dst`.
- **Output**: The function modifies the `dst` tensor in place, storing the result of adding the scalar value from `src1` to each element of the quantized tensor `src0`.
- **Functions called**:
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_is_scalar`](../ggml.c.driver.md#ggml_is_scalar)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_get_type_traits`](../ggml.c.driver.md#ggml_get_type_traits)
    - [`ggml_get_type_traits_cpu`](ggml-cpu.c.driver.md#ggml_get_type_traits_cpu)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_is_quantized`](../ggml.c.driver.md#ggml_is_quantized)
    - [`ggml_vec_acc1_f32`](vec.h.driver.md#ggml_vec_acc1_f32)


---
### ggml\_compute\_forward\_add1\_bf16\_f32<!-- {{#callable:ggml_compute_forward_add1_bf16_f32}} -->
Computes the element-wise addition of a scalar value to a tensor in bf16 format.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation, including thread indices.
    - `dst`: A pointer to the destination `ggml_tensor` where the result will be stored.
- **Control Flow**:
    - The function begins by retrieving the source tensors `src0` and `src1` from the destination tensor `dst`.
    - It asserts that `src0` and `dst` have the same shape and that `src1` is a scalar.
    - The scalar value to be added is extracted from `src1`.
    - The function calculates the number of rows in `src0` and determines the range of rows to be processed by the current thread.
    - It enters a loop to iterate over the assigned rows, calculating the appropriate indices for accessing elements in the tensors.
    - For each element in the row, it converts the bf16 value to float, adds the scalar, and converts it back to bf16 before storing it in `dst`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the result of the addition.
- **Functions called**:
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_is_scalar`](../ggml.c.driver.md#ggml_is_scalar)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)


---
### ggml\_compute\_forward\_add1\_bf16\_bf16<!-- {{#callable:ggml_compute_forward_add1_bf16_bf16}} -->
Computes the element-wise addition of a scalar to a tensor in bf16 format.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation, including thread index and total number of threads.
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the addition will be stored.
- **Control Flow**:
    - The function begins by retrieving the source tensors `src0` and `src1` from the destination tensor `dst`.
    - It asserts that `src0` and `dst` have the same shape and that `src1` is a scalar.
    - The scalar value to be added is extracted from `src1` and converted from bf16 to float.
    - The number of rows in `src0` is determined, and the number of rows to process per thread is calculated.
    - The function calculates the range of rows to be processed by the current thread based on its index.
    - A loop iterates over the assigned rows, performing the addition of the scalar to each element of `src0` and storing the result in `dst`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the results of the addition.
- **Functions called**:
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_is_scalar`](../ggml.c.driver.md#ggml_is_scalar)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)


---
### ggml\_compute\_forward\_add1<!-- {{#callable:ggml_compute_forward_add1}} -->
The `ggml_compute_forward_add1` function computes the forward addition operation for tensors based on their data types.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the addition will be stored.
- **Control Flow**:
    - The function retrieves the source tensors `src0` and `src1` from the destination tensor `dst`.
    - It uses a switch statement to determine the data type of `src0`.
    - For each data type (e.g., `GGML_TYPE_F32`, `GGML_TYPE_F16`, `GGML_TYPE_BF16`, and various quantized types), it calls the appropriate addition function based on the types of `src0` and `src1`.
    - If an unsupported type is encountered, the function calls `GGML_ABORT` to indicate a fatal error.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the result of the addition operation.
- **Functions called**:
    - [`ggml_compute_forward_add1_f32`](#ggml_compute_forward_add1_f32)
    - [`ggml_compute_forward_add1_f16_f16`](#ggml_compute_forward_add1_f16_f16)
    - [`ggml_compute_forward_add1_f16_f32`](#ggml_compute_forward_add1_f16_f32)
    - [`ggml_compute_forward_add1_bf16_bf16`](#ggml_compute_forward_add1_bf16_bf16)
    - [`ggml_compute_forward_add1_bf16_f32`](#ggml_compute_forward_add1_bf16_f32)
    - [`ggml_compute_forward_add1_q_f32`](#ggml_compute_forward_add1_q_f32)


---
### ggml\_compute\_forward\_acc\_f32<!-- {{#callable:ggml_compute_forward_acc_f32}} -->
The `ggml_compute_forward_acc_f32` function performs an element-wise addition of two source tensors and accumulates the result into a destination tensor, with options for in-place operation and multi-threading support.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for computation, including thread index and total number of threads.
    - `dst`: A pointer to a `ggml_tensor` structure that serves as the destination tensor for the accumulated results.
- **Control Flow**:
    - The function begins by retrieving the source tensors `src0` and `src1` from the destination tensor `dst`.
    - It asserts that the shapes of `src0` and `dst` are the same and that both are contiguous in memory.
    - The function extracts parameters from `dst->op_params` to determine the strides and offset for the operation.
    - If the operation is not in-place and the current thread is the first one, it copies data from `src0` to `dst` to initialize the accumulation.
    - A barrier is used to synchronize threads after the initialization copy.
    - The function calculates the number of rows and columns in `src1` and determines the range of rows to be processed by the current thread.
    - It iterates over the assigned row range, performing element-wise addition of the corresponding elements from `src0` and `src1`, storing the result in `dst`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place, accumulating the results of the addition from `src0` and `src1`.
- **Functions called**:
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)
    - [`ggml_barrier`](ggml-cpu.c.driver.md#ggml_barrier)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_element_size`](../ggml.c.driver.md#ggml_element_size)
    - [`ggml_vec_add_f32`](vec.h.driver.md#ggml_vec_add_f32)


---
### ggml\_compute\_forward\_acc<!-- {{#callable:ggml_compute_forward_acc}} -->
The `ggml_compute_forward_acc` function computes the forward accumulation for a tensor based on its data type.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of the source tensor `src0` using a switch statement.
    - If the type is `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_acc_f32`](#ggml_compute_forward_acc_f32) function to perform the computation.
    - For all other types, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; it modifies the destination tensor `dst` in place based on the computation performed.
- **Functions called**:
    - [`ggml_compute_forward_acc_f32`](#ggml_compute_forward_acc_f32)


---
### ggml\_compute\_forward\_sum\_f32<!-- {{#callable:ggml_compute_forward_sum_f32}} -->
Computes the sum of elements in a tensor and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing computation parameters, including the index of the current operation.
    - `dst`: A pointer to a `ggml_tensor` structure where the computed sum will be stored.
- **Control Flow**:
    - The function first checks if the current operation index (`params->ith`) is not zero; if so, it returns early.
    - It asserts that the destination tensor (`dst`) is a scalar and that the source tensor (`src0`) has a data type of float.
    - Local variables for the number of elements and byte sizes of the source tensor are initialized.
    - A nested loop iterates over the dimensions of the source tensor, calculating the sum of its elements using [`ggml_vec_sum_f32_ggf`](vec.h.driver.md#ggml_vec_sum_f32_ggf) for each row.
    - The computed row sums are accumulated into a total sum, which is then stored in the destination tensor.
- **Output**: The function outputs the total sum of the elements from the source tensor (`src0`) into the first element of the destination tensor (`dst`).
- **Functions called**:
    - [`ggml_is_scalar`](../ggml.c.driver.md#ggml_is_scalar)
    - [`ggml_vec_sum_f32_ggf`](vec.h.driver.md#ggml_vec_sum_f32_ggf)


---
### ggml\_compute\_forward\_sum\_f16<!-- {{#callable:ggml_compute_forward_sum_f16}} -->
Computes the sum of elements from a source tensor of half-precision floats and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing computation parameters, including the index of the current operation.
    - `dst`: A pointer to a `ggml_tensor` structure where the computed sum will be stored.
- **Control Flow**:
    - Checks if the current operation index (`params->ith`) is not zero; if so, the function returns early without performing any computation.
    - Asserts that the destination tensor (`dst`) is a scalar and that the source tensor (`src0`) has the correct data type size for half-precision floats.
    - Initializes local variables for the number of elements and byte sizes of the source tensor.
    - Iterates over the dimensions of the source tensor, summing the elements using the [`ggml_vec_sum_f16_ggf`](vec.h.driver.md#ggml_vec_sum_f16_ggf) function for each row and accumulating the total sum.
    - Stores the final computed sum in the destination tensor after converting it from float to half-precision float.
- **Output**: The function does not return a value but updates the destination tensor (`dst`) with the computed sum, stored as a half-precision float.
- **Functions called**:
    - [`ggml_is_scalar`](../ggml.c.driver.md#ggml_is_scalar)
    - [`ggml_vec_sum_f16_ggf`](vec.h.driver.md#ggml_vec_sum_f16_ggf)


---
### ggml\_compute\_forward\_sum\_bf16<!-- {{#callable:ggml_compute_forward_sum_bf16}} -->
Computes the sum of elements from a source tensor and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing computation parameters, including the index of the current operation.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the result of the sum operation.
- **Control Flow**:
    - The function first checks if the current operation index (`params->ith`) is not zero; if so, it returns early.
    - It asserts that the destination tensor (`dst`) is a scalar and that the source tensor (`src0`) has the correct byte size for its elements.
    - Local variables for the number of elements and byte sizes of the source tensor are initialized.
    - A nested loop iterates over the dimensions of the source tensor, calculating the sum of its elements using [`ggml_vec_sum_bf16_ggf`](vec.h.driver.md#ggml_vec_sum_bf16_ggf) for each row.
    - The computed row sums are accumulated into a total sum.
    - Finally, the total sum is converted to `ggml_bf16_t` format and stored in the destination tensor.
- **Output**: The function outputs the total sum of the elements from the source tensor as a `ggml_bf16_t` value stored in the destination tensor.
- **Functions called**:
    - [`ggml_is_scalar`](../ggml.c.driver.md#ggml_is_scalar)
    - [`ggml_vec_sum_bf16_ggf`](vec.h.driver.md#ggml_vec_sum_bf16_ggf)


---
### ggml\_compute\_forward\_sum<!-- {{#callable:ggml_compute_forward_sum}} -->
Computes the forward sum of a tensor based on its data type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` structure that will hold the result of the sum operation.
- **Control Flow**:
    - Retrieves the source tensor `src0` from the destination tensor `dst`.
    - Checks the data type of `src0` using a switch statement.
    - Calls the appropriate sum computation function based on the data type: [`ggml_compute_forward_sum_f32`](#ggml_compute_forward_sum_f32) for float32, [`ggml_compute_forward_sum_f16`](#ggml_compute_forward_sum_f16) for float16, and [`ggml_compute_forward_sum_bf16`](#ggml_compute_forward_sum_bf16) for bfloat16.
    - If the data type does not match any of the expected types, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor to contain the result of the sum operation.
- **Functions called**:
    - [`ggml_compute_forward_sum_f32`](#ggml_compute_forward_sum_f32)
    - [`ggml_compute_forward_sum_f16`](#ggml_compute_forward_sum_f16)
    - [`ggml_compute_forward_sum_bf16`](#ggml_compute_forward_sum_bf16)


---
### ggml\_compute\_forward\_sum\_rows\_f32<!-- {{#callable:ggml_compute_forward_sum_rows_f32}} -->
Computes the sum of each row in a source tensor and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing computation parameters, including the index of the current operation.
    - `dst`: A pointer to a `ggml_tensor` structure where the results of the row sums will be stored.
- **Control Flow**:
    - The function first checks if the current operation index (`params->ith`) is not zero; if so, it returns immediately.
    - It asserts that the data types of the source and destination tensors are both `float`.
    - It verifies the dimensions of the tensors to ensure they are compatible for the operation.
    - It iterates over the dimensions of the tensor, summing each row of the source tensor using [`ggml_vec_sum_f32`](vec.h.driver.md#ggml_vec_sum_f32) and storing the result in the corresponding position in the destination tensor.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the sums of the rows from the `src0` tensor.
- **Functions called**:
    - [`ggml_vec_sum_f32`](vec.h.driver.md#ggml_vec_sum_f32)


---
### ggml\_compute\_forward\_sum\_rows<!-- {{#callable:ggml_compute_forward_sum_rows}} -->
The `ggml_compute_forward_sum_rows` function computes the sum of rows for a given tensor based on its type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the result of the row summation.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of `src0` using a switch statement.
    - If the type is `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_sum_rows_f32`](#ggml_compute_forward_sum_rows_f32) function to perform the summation.
    - For any other type, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor to contain the result of the row summation.
- **Functions called**:
    - [`ggml_compute_forward_sum_rows_f32`](#ggml_compute_forward_sum_rows_f32)


---
### ggml\_compute\_forward\_mean\_f32<!-- {{#callable:ggml_compute_forward_mean_f32}} -->
Computes the mean of a tensor's elements and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing computation parameters, including the index of the current operation.
    - `dst`: A pointer to a `ggml_tensor` structure where the computed mean will be stored.
- **Control Flow**:
    - Checks if the current operation index (`params->ith`) is not zero; if so, the function returns early.
    - Asserts that the data type of the source tensor (`src0`) is a float.
    - Validates the dimensions of the tensor to ensure they match expected values.
    - Iterates over the dimensions of the tensor to compute the sum of elements using [`ggml_vec_sum_f32`](vec.h.driver.md#ggml_vec_sum_f32).
    - Divides the computed sum by the number of elements to obtain the mean and stores it in the destination tensor.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the mean of the elements from the source tensor.
- **Functions called**:
    - [`ggml_vec_sum_f32`](vec.h.driver.md#ggml_vec_sum_f32)


---
### ggml\_compute\_forward\_mean<!-- {{#callable:ggml_compute_forward_mean}} -->
Computes the mean of a tensor based on its type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for computation.
    - `dst`: A pointer to `ggml_tensor` where the result of the mean computation will be stored.
- **Control Flow**:
    - Retrieves the source tensor `src0` from the destination tensor `dst`.
    - Checks the type of `src0` to determine the appropriate computation method.
    - If `src0` is of type `GGML_TYPE_F32`, it calls the function [`ggml_compute_forward_mean_f32`](#ggml_compute_forward_mean_f32) to perform the mean computation.
    - If `src0` is of an unsupported type, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor to contain the computed mean.
- **Functions called**:
    - [`ggml_compute_forward_mean_f32`](#ggml_compute_forward_mean_f32)


---
### ggml\_compute\_forward\_argmax\_f32<!-- {{#callable:ggml_compute_forward_argmax_f32}} -->
Computes the indices of the maximum values along the first dimension of a source tensor and stores them in a destination tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing computation parameters, including the index of the current computation.
    - `dst`: A pointer to a `ggml_tensor` structure where the result (indices of maximum values) will be stored.
- **Control Flow**:
    - Checks if the current computation index (`params->ith`) is not zero; if so, the function returns early without performing any computation.
    - Validates that the data types of the source and destination tensors are both `float` by asserting their byte sizes.
    - Retrieves the dimensions and byte sizes of the source tensor (`src0`) and destination tensor (`dst`).
    - Iterates over the second dimension of the source tensor (`ne01`), extracting each slice of data and computing the index of the maximum value using [`ggml_vec_argmax_f32`](vec.h.driver.md#ggml_vec_argmax_f32).
    - Stores the computed index in the destination tensor at the corresponding position.
- **Output**: The function does not return a value; instead, it populates the `dst` tensor with the indices of the maximum values found in the first dimension of the `src0` tensor.
- **Functions called**:
    - [`ggml_vec_argmax_f32`](vec.h.driver.md#ggml_vec_argmax_f32)


---
### ggml\_compute\_forward\_argmax<!-- {{#callable:ggml_compute_forward_argmax}} -->
The `ggml_compute_forward_argmax` function computes the forward argmax operation for a given tensor based on its type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` that represents the destination tensor where the result will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of the source tensor `src0` using a switch statement.
    - If the type is `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_argmax_f32`](#ggml_compute_forward_argmax_f32) function to perform the computation.
    - If the type is not recognized, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the destination tensor `dst` in place with the computed argmax results.
- **Functions called**:
    - [`ggml_compute_forward_argmax_f32`](#ggml_compute_forward_argmax_f32)


---
### ggml\_compute\_forward\_count\_equal\_i32<!-- {{#callable:ggml_compute_forward_count_equal_i32}} -->
Computes the count of equal elements between two `int32_t` tensors and stores the result in a destination tensor of type `int64_t`.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for computation, including thread information and workspace data.
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the count will be stored.
- **Control Flow**:
    - The function begins by retrieving the source tensors `src0` and `src1` from the destination tensor `dst`.
    - It asserts that both source tensors are of type `GGML_TYPE_I32` and that they have the same shape, while also checking that the destination tensor is a scalar of type `GGML_TYPE_I64`.
    - The number of rows in the first source tensor is determined, and the thread's specific range of rows to process is calculated based on the total number of threads.
    - A loop iterates over the assigned rows, extracting data from both source tensors and counting the number of equal elements.
    - If the current thread is not the first one, it stores its local count in the `sums` array.
    - A barrier is used to synchronize threads, ensuring all threads have completed their calculations before proceeding.
    - The first thread aggregates the counts from all threads and stores the final result in the destination tensor.
- **Output**: The output is stored in the `dst` tensor, which contains the total count of equal elements between the two input tensors.
- **Functions called**:
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_is_scalar`](../ggml.c.driver.md#ggml_is_scalar)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_barrier`](ggml-cpu.c.driver.md#ggml_barrier)


---
### ggml\_compute\_forward\_count\_equal<!-- {{#callable:ggml_compute_forward_count_equal}} -->
The `ggml_compute_forward_count_equal` function computes the count of equal elements in a tensor based on its type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` that serves as the destination tensor for the computation.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of `src0` using a switch statement.
    - If `src0` is of type `GGML_TYPE_I32`, it calls the helper function [`ggml_compute_forward_count_equal_i32`](#ggml_compute_forward_count_equal_i32) to perform the computation.
    - If `src0` is of any other type, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor based on the computation of equal elements from `src0`.
- **Functions called**:
    - [`ggml_compute_forward_count_equal_i32`](#ggml_compute_forward_count_equal_i32)


---
### ggml\_compute\_forward\_repeat\_f32<!-- {{#callable:ggml_compute_forward_repeat_f32}} -->
The `ggml_compute_forward_repeat_f32` function performs a repeat operation on a source tensor and writes the result to a destination tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation, including the index of the current operation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result will be stored.
- **Control Flow**:
    - The function first checks if the current operation index (`params->ith`) is not zero; if so, it returns immediately, indicating that this operation should not be performed.
    - It asserts that the source tensor can be repeated into the destination tensor using [`ggml_can_repeat`](../ggml.c.driver.md#ggml_can_repeat).
    - Local variables are initialized to determine the number of repetitions for each dimension of the tensor based on the source and destination tensor shapes.
    - The function contains nested loops that iterate over the dimensions of the destination tensor, copying data from the source tensor to the destination tensor using [`ggml_vec_cpy_f32`](vec.h.driver.md#ggml_vec_cpy_f32).
- **Output**: The function does not return a value; instead, it modifies the destination tensor `dst` in place by filling it with repeated values from the source tensor `src0`.
- **Functions called**:
    - [`ggml_can_repeat`](../ggml.c.driver.md#ggml_can_repeat)
    - [`ggml_vec_cpy_f32`](vec.h.driver.md#ggml_vec_cpy_f32)


---
### ggml\_compute\_forward\_repeat\_f16<!-- {{#callable:ggml_compute_forward_repeat_f16}} -->
The `ggml_compute_forward_repeat_f16` function performs a repeat operation on a source tensor and writes the result to a destination tensor, effectively replicating the source tensor's data across multiple dimensions.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation, including the current index of the operation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result will be stored.
- **Control Flow**:
    - The function first checks if the current index (`params->ith`) is not zero; if so, it returns early, as the operation is only valid for the first index.
    - It asserts that the source tensor can be repeated using the [`ggml_can_repeat`](../ggml.c.driver.md#ggml_can_repeat) function.
    - Local variables are initialized to calculate the number of repetitions for each dimension based on the tensor's shape.
    - The function contains nested loops that iterate over the dimensions of the destination tensor, copying data from the source tensor to the destination tensor for each repetition.
    - Within the innermost loop, the function copies data from the source tensor to the destination tensor using a simple element-wise copy.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place by filling it with repeated data from the `src0` tensor.
- **Functions called**:
    - [`ggml_can_repeat`](../ggml.c.driver.md#ggml_can_repeat)


---
### ggml\_compute\_forward\_repeat<!-- {{#callable:ggml_compute_forward_repeat}} -->
The `ggml_compute_forward_repeat` function computes the forward repeat operation for a tensor based on its data type.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the data type of `src0` using a switch statement.
    - If the data type is `GGML_TYPE_F16`, `GGML_TYPE_BF16`, or `GGML_TYPE_I16`, it calls [`ggml_compute_forward_repeat_f16`](#ggml_compute_forward_repeat_f16) to perform the computation.
    - If the data type is `GGML_TYPE_F32` or `GGML_TYPE_I32`, it calls [`ggml_compute_forward_repeat_f32`](#ggml_compute_forward_repeat_f32) instead.
    - If the data type does not match any of the expected types, it triggers an abort with a fatal error message.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place based on the computation performed.
- **Functions called**:
    - [`ggml_compute_forward_repeat_f16`](#ggml_compute_forward_repeat_f16)
    - [`ggml_compute_forward_repeat_f32`](#ggml_compute_forward_repeat_f32)


---
### ggml\_compute\_forward\_repeat\_back\_f32<!-- {{#callable:ggml_compute_forward_repeat_back_f32}} -->
Computes the forward repeat operation for a tensor by accumulating values from a source tensor into a destination tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing computation parameters, including the index of the current operation.
    - `dst`: A pointer to a `ggml_tensor` structure representing the destination tensor where the results will be stored.
- **Control Flow**:
    - The function first checks if the current operation index (`params->ith`) is not zero; if so, it returns early.
    - It asserts that the destination tensor can repeat the source tensor using [`ggml_can_repeat`](../ggml.c.driver.md#ggml_can_repeat).
    - Local variables for tensor dimensions and sizes are initialized.
    - If the destination tensor is contiguous, it sets all values to zero using [`ggml_vec_set_f32`](vec.h.driver.md#ggml_vec_set_f32).
    - If not contiguous, it iterates through the tensor dimensions to set each slice to zero.
    - The function then performs nested loops to accumulate values from the source tensor into the destination tensor based on the repeat counts calculated earlier.
- **Output**: The function does not return a value; instead, it modifies the destination tensor `dst` in place by accumulating values from the source tensor `src0`.
- **Functions called**:
    - [`ggml_can_repeat`](../ggml.c.driver.md#ggml_can_repeat)
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_vec_set_f32`](vec.h.driver.md#ggml_vec_set_f32)
    - [`ggml_vec_acc_f32`](vec.h.driver.md#ggml_vec_acc_f32)


---
### ggml\_compute\_forward\_repeat\_back<!-- {{#callable:ggml_compute_forward_repeat_back}} -->
The `ggml_compute_forward_repeat_back` function computes a forward operation for a tensor based on its type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` that serves as the destination tensor for the computation.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of `src0` using a switch statement.
    - If `src0` is of type `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_repeat_back_f32`](#ggml_compute_forward_repeat_back_f32) function to perform the computation.
    - If `src0` is of any other type, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; it modifies the destination tensor `dst` based on the computation performed.
- **Functions called**:
    - [`ggml_compute_forward_repeat_back_f32`](#ggml_compute_forward_repeat_back_f32)


---
### ggml\_compute\_forward\_concat\_any<!-- {{#callable:ggml_compute_forward_concat_any}} -->
The `ggml_compute_forward_concat_any` function concatenates two source tensors along a specified dimension and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for the computation, including the current index (`ith`) and the total number of threads (`nth`).
    - `dst`: A pointer to a `ggml_tensor` structure that serves as the destination tensor where the concatenated result will be stored.
- **Control Flow**:
    - The function retrieves the first and second source tensors from the destination tensor's source array.
    - It calculates the size of the data type of the first source tensor.
    - It asserts that the specified dimension for concatenation is valid (between 0 and 3).
    - It initializes an output array to keep track of the size of the first source tensor along the specified dimension.
    - The function iterates over the dimensions of the tensors, using nested loops to access each element.
    - For each element, it checks if the current indices are within the bounds of the first source tensor; if so, it retrieves data from the first source tensor, otherwise, it retrieves data from the second source tensor.
    - The retrieved data is then copied into the appropriate position in the destination tensor.
- **Output**: The function does not return a value; instead, it modifies the destination tensor in place to contain the concatenated data from the two source tensors.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_get_op_params_i32`](../ggml-impl.h.driver.md#ggml_get_op_params_i32)


---
### ggml\_compute\_forward\_concat\_i8<!-- {{#callable:ggml_compute_forward_concat_i8}} -->
The `ggml_compute_forward_concat_i8` function concatenates two input tensors of type `int8_t` along a specified dimension and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for the computation, including the current index (`ith`) and the total number of threads (`nth`).
    - `dst`: A pointer to a `ggml_tensor` structure that serves as the destination tensor where the concatenated result will be stored.
- **Control Flow**:
    - The function begins by retrieving the source tensors `src0` and `src1` from the destination tensor `dst`.
    - It asserts that the data type of the source tensors is `int8_t`.
    - The function retrieves the dimension along which to concatenate from the destination tensor's parameters.
    - It initializes an output offset array `o` to track the size of the first source tensor along the specified dimension.
    - A nested loop structure iterates over the dimensions of the tensors, checking bounds to determine whether to read from `src0` or `src1`.
    - The appropriate data is copied from the source tensor to the destination tensor based on the current indices.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the concatenated data from `src0` and `src1`.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_get_op_params_i32`](../ggml-impl.h.driver.md#ggml_get_op_params_i32)


---
### ggml\_compute\_forward\_concat\_f16<!-- {{#callable:ggml_compute_forward_concat_f16}} -->
This function concatenates two tensors of type `ggml_fp16_t` along a specified dimension.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation, including the thread index and total number of threads.
    - `dst`: A pointer to the destination `ggml_tensor` where the concatenated result will be stored.
- **Control Flow**:
    - The function retrieves the source tensors `src0` and `src1` from the destination tensor `dst`.
    - It asserts that the size of the data type of `src0` is equal to the size of `ggml_fp16_t`.
    - It retrieves the dimension along which to concatenate from the operation parameters.
    - It initializes an output index array `o` to track the size of the first source tensor along the specified dimension.
    - The function iterates over the dimensions of the tensors using nested loops, checking bounds to determine whether to read from `src0` or `src1`.
    - For each valid index, it calculates the appropriate memory address for both source tensors and writes the value to the destination tensor.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the concatenated data from `src0` and `src1`.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_get_op_params_i32`](../ggml-impl.h.driver.md#ggml_get_op_params_i32)


---
### ggml\_compute\_forward\_concat\_f32<!-- {{#callable:ggml_compute_forward_concat_f32}} -->
The `ggml_compute_forward_concat_f32` function concatenates two source tensors along a specified dimension and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation, including the current thread index (`ith`) and the total number of threads (`nth`).
    - `dst`: A pointer to the destination `ggml_tensor` where the concatenated result will be stored.
- **Control Flow**:
    - The function begins by retrieving the source tensors `src0` and `src1` from the destination tensor `dst`.
    - It asserts that the data type size of `src0` is equal to the size of a float.
    - The dimension along which to concatenate is obtained from the destination tensor's operation parameters.
    - It initializes an output offset array `o` to track the size of the first source tensor along the specified dimension.
    - A nested loop structure iterates over the dimensions of the tensors, checking bounds to determine whether to read from `src0` or `src1`.
    - The appropriate data is copied from the source tensor to the destination tensor based on the current indices.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the concatenated data from `src0` and `src1`.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_get_op_params_i32`](../ggml-impl.h.driver.md#ggml_get_op_params_i32)


---
### ggml\_compute\_forward\_concat<!-- {{#callable:ggml_compute_forward_concat}} -->
The `ggml_compute_forward_concat` function computes the forward concatenation of tensors based on their data type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to the destination `ggml_tensor` where the concatenated result will be stored.
- **Control Flow**:
    - The function retrieves the source tensor type from the first source tensor of the destination tensor.
    - A switch statement is used to determine the data type of the source tensor.
    - Depending on the data type, the function calls the appropriate helper function to perform the concatenation: [`ggml_compute_forward_concat_f16`](#ggml_compute_forward_concat_f16), [`ggml_compute_forward_concat_i8`](#ggml_compute_forward_concat_i8), [`ggml_compute_forward_concat_f32`](#ggml_compute_forward_concat_f32), or [`ggml_compute_forward_concat_any`](#ggml_compute_forward_concat_any).
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to hold the concatenated result.
- **Functions called**:
    - [`ggml_compute_forward_concat_f16`](#ggml_compute_forward_concat_f16)
    - [`ggml_compute_forward_concat_i8`](#ggml_compute_forward_concat_i8)
    - [`ggml_compute_forward_concat_f32`](#ggml_compute_forward_concat_f32)
    - [`ggml_compute_forward_concat_any`](#ggml_compute_forward_concat_any)


---
### ggml\_compute\_forward\_gelu\_f32<!-- {{#callable:ggml_compute_forward_gelu_f32}} -->
Computes the GELU (Gaussian Error Linear Unit) activation function for a given tensor in a parallelized manner.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for computation, including the thread index (`ith`) and total number of threads (`nth`).
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the output of the GELU computation.
- **Control Flow**:
    - The function begins by retrieving the source tensor (`src0`) from the destination tensor (`dst`).
    - It asserts that both the source and destination tensors are contiguous and have the same shape.
    - The number of channels (`nc`) and rows (`nr`) in the source tensor are determined.
    - The number of rows processed per thread is calculated, and the range of rows for the current thread is established.
    - A loop iterates over the assigned row range, applying the [`ggml_vec_gelu_f32`](vec.h.driver.md#ggml_vec_gelu_f32) function to compute the GELU activation for each row.
    - In debug mode, it checks that the output values are neither NaN nor infinite after computation.
- **Output**: The function does not return a value; instead, it populates the `dst` tensor with the computed GELU values based on the input tensor `src0`.
- **Functions called**:
    - [`ggml_is_contiguous_1`](../ggml.c.driver.md#ggml_is_contiguous_1)
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_vec_gelu_f32`](vec.h.driver.md#ggml_vec_gelu_f32)


---
### ggml\_compute\_forward\_gelu\_f16<!-- {{#callable:ggml_compute_forward_gelu_f16}} -->
Computes the GELU activation function for a tensor in half-precision floating point format.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for computation, including the thread index and total number of threads.
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the GELU computation will be stored.
- **Control Flow**:
    - The function begins by retrieving the source tensor from the destination tensor's source array.
    - It asserts that both the source and destination tensors are contiguous and have the same shape.
    - It calculates the number of channels and rows in the source tensor.
    - The number of rows processed per thread is determined, and the specific range of rows for the current thread is calculated.
    - A loop iterates over the assigned row range, applying the GELU function to each row using [`ggml_vec_gelu_f16`](vec.h.driver.md#ggml_vec_gelu_f16).
    - In debug mode, it checks that the output values are neither NaN nor infinite after computation.
- **Output**: The function does not return a value; instead, it populates the `dst` tensor with the computed GELU values.
- **Functions called**:
    - [`ggml_is_contiguous_1`](../ggml.c.driver.md#ggml_is_contiguous_1)
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_vec_gelu_f16`](vec.h.driver.md#ggml_vec_gelu_f16)


---
### ggml\_compute\_forward\_gelu<!-- {{#callable:ggml_compute_forward_gelu}} -->
Computes the forward pass of the Gaussian Error Linear Unit (GELU) activation function based on the input tensor type.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure where the result of the GELU computation will be stored.
- **Control Flow**:
    - Retrieves the source tensor `src0` from the destination tensor `dst`.
    - Checks the type of the source tensor `src0` using a switch statement.
    - If the type is `GGML_TYPE_F32`, calls the function [`ggml_compute_forward_gelu_f32`](#ggml_compute_forward_gelu_f32) to perform the computation.
    - If the type is `GGML_TYPE_F16`, calls the function [`ggml_compute_forward_gelu_f16`](#ggml_compute_forward_gelu_f16) to perform the computation.
    - If the type is neither `GGML_TYPE_F32` nor `GGML_TYPE_F16`, triggers an abort with a fatal error message.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the results of the GELU activation function based on the input tensor type.
- **Functions called**:
    - [`ggml_compute_forward_gelu_f32`](#ggml_compute_forward_gelu_f32)
    - [`ggml_compute_forward_gelu_f16`](#ggml_compute_forward_gelu_f16)


---
### ggml\_compute\_forward\_gelu\_erf\_f32<!-- {{#callable:ggml_compute_forward_gelu_erf_f32}} -->
Computes the GELU activation function using the error function for a given tensor in a parallelized manner.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for computation, including the thread index (`ith`) and total number of threads (`nth`).
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the GELU computation will be stored.
- **Control Flow**:
    - The function begins by retrieving the source tensor `src0` from the destination tensor `dst`.
    - Assertions are made to ensure that both `src0` and `dst` are contiguous and have the same shape.
    - The number of channels (`nc`) and rows (`nr`) in the source tensor are determined.
    - The number of rows to process per thread is calculated, and the specific range of rows for the current thread is determined.
    - A loop iterates over the assigned row range, applying the [`ggml_vec_gelu_erf_f32`](vec.h.driver.md#ggml_vec_gelu_erf_f32) function to compute the GELU activation for each row.
    - In debug mode, additional assertions check that the computed values are neither NaN nor infinite.
- **Output**: The function does not return a value; instead, it populates the `dst` tensor with the results of the GELU activation function applied to the input tensor.
- **Functions called**:
    - [`ggml_is_contiguous_1`](../ggml.c.driver.md#ggml_is_contiguous_1)
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_vec_gelu_erf_f32`](vec.h.driver.md#ggml_vec_gelu_erf_f32)


---
### ggml\_compute\_forward\_gelu\_erf\_f16<!-- {{#callable:ggml_compute_forward_gelu_erf_f16}} -->
Computes the GELU activation function using the error function for half-precision floating-point tensors.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for computation, including the thread index and total number of threads.
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the GELU computation will be stored.
- **Control Flow**:
    - The function begins by retrieving the source tensor `src0` from the destination tensor `dst`.
    - It asserts that both the source and destination tensors are contiguous and have the same shape.
    - The function calculates the number of channels (`nc`) and the number of rows (`nr`) in the source tensor.
    - It determines how many rows each thread will process and calculates the range of rows (`ir0` to `ir1`) for the current thread based on its index.
    - A loop iterates over the assigned rows, applying the [`ggml_vec_gelu_erf_f16`](vec.h.driver.md#ggml_vec_gelu_erf_f16) function to compute the GELU activation for each row.
    - In debug mode, it checks that the computed values are neither NaN nor infinite.
- **Output**: The function does not return a value; instead, it populates the `dst` tensor with the computed GELU values based on the input tensor.
- **Functions called**:
    - [`ggml_is_contiguous_1`](../ggml.c.driver.md#ggml_is_contiguous_1)
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_vec_gelu_erf_f16`](vec.h.driver.md#ggml_vec_gelu_erf_f16)


---
### ggml\_compute\_forward\_gelu\_erf<!-- {{#callable:ggml_compute_forward_gelu_erf}} -->
The `ggml_compute_forward_gelu_erf` function computes the GELU activation using the error function for different tensor types.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` that will hold the result of the GELU computation.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of the source tensor `src0` using a switch statement.
    - If the type is `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_gelu_erf_f32`](#ggml_compute_forward_gelu_erf_f32) function to perform the computation.
    - If the type is `GGML_TYPE_F16`, it calls the [`ggml_compute_forward_gelu_erf_f16`](#ggml_compute_forward_gelu_erf_f16) function instead.
    - If the type is neither `GGML_TYPE_F32` nor `GGML_TYPE_F16`, it triggers an abort with a fatal error message.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the computed GELU values.
- **Functions called**:
    - [`ggml_compute_forward_gelu_erf_f32`](#ggml_compute_forward_gelu_erf_f32)
    - [`ggml_compute_forward_gelu_erf_f16`](#ggml_compute_forward_gelu_erf_f16)


---
### ggml\_compute\_forward\_gelu\_quick\_f32<!-- {{#callable:ggml_compute_forward_gelu_quick_f32}} -->
Computes the GELU activation function in a parallelized manner for a given tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for computation, including the thread index (`ith`) and total number of threads (`nth`).
    - `dst`: A pointer to a `ggml_tensor` structure where the result of the GELU computation will be stored.
- **Control Flow**:
    - The function begins by retrieving the source tensor (`src0`) from the destination tensor (`dst`).
    - It asserts that both the source and destination tensors are contiguous and have the same shape.
    - The number of channels (`nc`) and rows (`nr`) in the source tensor are determined.
    - The number of rows to process per thread is calculated, and the specific range of rows for the current thread is determined.
    - A loop iterates over the assigned range of rows, applying the [`ggml_vec_gelu_quick_f32`](vec.h.driver.md#ggml_vec_gelu_quick_f32) function to compute the GELU activation for each row.
    - In debug mode, additional assertions check that the output values are neither NaN nor infinite.
- **Output**: The function does not return a value; instead, it populates the `dst` tensor with the results of the GELU activation applied to the input tensor.
- **Functions called**:
    - [`ggml_is_contiguous_1`](../ggml.c.driver.md#ggml_is_contiguous_1)
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_vec_gelu_quick_f32`](vec.h.driver.md#ggml_vec_gelu_quick_f32)


---
### ggml\_compute\_forward\_gelu\_quick\_f16<!-- {{#callable:ggml_compute_forward_gelu_quick_f16}} -->
Computes the GELU activation function in a quick manner for half-precision floating-point tensors.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for the computation, including the thread index (`ith`) and the total number of threads (`nth`).
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the output of the GELU computation.
- **Control Flow**:
    - The function begins by retrieving the source tensor (`src0`) from the destination tensor (`dst`).
    - It asserts that both the source and destination tensors are contiguous and have the same shape.
    - The number of channels (`nc`) and rows (`nr`) in the source tensor are determined.
    - The number of rows processed per thread is calculated, and the range of rows for the current thread is established.
    - A loop iterates over the assigned row range, applying the [`ggml_vec_gelu_quick_f16`](vec.h.driver.md#ggml_vec_gelu_quick_f16) function to compute the GELU activation for each row.
    - In debug mode, additional assertions check that the output values are neither NaN nor infinite.
- **Output**: The function does not return a value; instead, it populates the `dst` tensor with the results of the GELU activation applied to the input tensor.
- **Functions called**:
    - [`ggml_is_contiguous_1`](../ggml.c.driver.md#ggml_is_contiguous_1)
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_vec_gelu_quick_f16`](vec.h.driver.md#ggml_vec_gelu_quick_f16)


---
### ggml\_compute\_forward\_gelu\_quick<!-- {{#callable:ggml_compute_forward_gelu_quick}} -->
Computes the forward pass of the GELU activation function quickly based on the input tensor type.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure where the result of the computation will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of the source tensor `src0` using a switch statement.
    - If the type is `GGML_TYPE_F32`, it calls [`ggml_compute_forward_gelu_quick_f32`](#ggml_compute_forward_gelu_quick_f32) to perform the computation.
    - If the type is `GGML_TYPE_F16`, it calls [`ggml_compute_forward_gelu_quick_f16`](#ggml_compute_forward_gelu_quick_f16) for the computation.
    - If the type is neither `GGML_TYPE_F32` nor `GGML_TYPE_F16`, it triggers an abort with a fatal error message.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the computed GELU activation results.
- **Functions called**:
    - [`ggml_compute_forward_gelu_quick_f32`](#ggml_compute_forward_gelu_quick_f32)
    - [`ggml_compute_forward_gelu_quick_f16`](#ggml_compute_forward_gelu_quick_f16)


---
### ggml\_compute\_forward\_silu\_f32<!-- {{#callable:ggml_compute_forward_silu_f32}} -->
Computes the forward pass of the Sigmoid Linear Unit (SiLU) activation function for a given tensor in a parallelized manner.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for computation, including the thread index (`ith`) and total number of threads (`nth`).
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the output of the SiLU activation function.
- **Control Flow**:
    - The function begins by retrieving the source tensor (`src0`) from the destination tensor (`dst`).
    - It asserts that both the source and destination tensors are contiguous and have the same shape.
    - It calculates the number of channels (`nc`) and the number of rows (`nr`) in the source tensor.
    - The number of rows processed per thread is computed, and the specific range of rows for the current thread is determined.
    - A loop iterates over the assigned range of rows, applying the SiLU function to each row using [`ggml_vec_silu_f32`](vec.cpp.driver.md#ggml_vec_silu_f32).
    - In debug mode, it checks that the output values are neither NaN nor infinite after computation.
- **Output**: The function does not return a value; instead, it populates the `dst` tensor with the results of the SiLU activation applied to the input tensor.
- **Functions called**:
    - [`ggml_is_contiguous_1`](../ggml.c.driver.md#ggml_is_contiguous_1)
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_vec_silu_f32`](vec.cpp.driver.md#ggml_vec_silu_f32)


---
### ggml\_compute\_forward\_silu\_f16<!-- {{#callable:ggml_compute_forward_silu_f16}} -->
Computes the forward pass of the Sigmoid Linear Unit (SiLU) activation function for half-precision floating-point tensors.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for the computation, including the thread index and total number of threads.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the output of the SiLU computation.
- **Control Flow**:
    - The function begins by retrieving the source tensor `src0` from the destination tensor `dst`.
    - It asserts that both `src0` and `dst` are contiguous and have the same shape.
    - It calculates the number of channels `nc` and the number of rows `nr` in the source tensor.
    - The number of rows processed per thread is computed, and the range of rows for the current thread is determined.
    - A loop iterates over the assigned rows, applying the [`ggml_vec_silu_f16`](vec.h.driver.md#ggml_vec_silu_f16) function to compute the SiLU activation for each row.
    - In debug mode, it checks that the output values are neither NaN nor infinite after computation.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the results of the SiLU activation applied to the input tensor.
- **Functions called**:
    - [`ggml_is_contiguous_1`](../ggml.c.driver.md#ggml_is_contiguous_1)
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_vec_silu_f16`](vec.h.driver.md#ggml_vec_silu_f16)


---
### ggml\_compute\_forward\_silu<!-- {{#callable:ggml_compute_forward_silu}} -->
Computes the forward pass of the SiLU activation function for different tensor types.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` that will hold the result of the SiLU activation function.
- **Control Flow**:
    - Retrieves the source tensor `src0` from the destination tensor `dst`.
    - Checks the type of the source tensor `src0` using a switch statement.
    - If the type is `GGML_TYPE_F32`, calls the function [`ggml_compute_forward_silu_f32`](#ggml_compute_forward_silu_f32) to perform the computation.
    - If the type is `GGML_TYPE_F16`, calls the function [`ggml_compute_forward_silu_f16`](#ggml_compute_forward_silu_f16) to perform the computation.
    - If the type is neither, triggers an abort with a fatal error message.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the computed SiLU activation results.
- **Functions called**:
    - [`ggml_compute_forward_silu_f32`](#ggml_compute_forward_silu_f32)
    - [`ggml_compute_forward_silu_f16`](#ggml_compute_forward_silu_f16)


---
### ggml\_compute\_forward\_leaky\_relu\_f32<!-- {{#callable:ggml_compute_forward_leaky_relu_f32}} -->
Computes the forward pass of the leaky ReLU activation function for a given tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing computation parameters.
    - `dst`: A pointer to a `ggml_tensor` structure where the result of the leaky ReLU computation will be stored.
- **Control Flow**:
    - Checks if the current computation index is zero; if not, the function returns immediately.
    - Validates that the source tensor (`src0`) and destination tensor (`dst`) are contiguous and have the same shape.
    - Retrieves the number of rows (`n`) and the number of columns (`nc`) from the source tensor.
    - Copies the negative slope parameter from the destination tensor's operation parameters.
    - Iterates over each row of the source tensor, applying the leaky ReLU function to each corresponding row and storing the result in the destination tensor.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the results of the leaky ReLU activation applied to the `src0` tensor.
- **Functions called**:
    - [`ggml_is_contiguous_1`](../ggml.c.driver.md#ggml_is_contiguous_1)
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_vec_leaky_relu_f32`](vec.h.driver.md#ggml_vec_leaky_relu_f32)


---
### ggml\_compute\_forward\_leaky\_relu\_f16<!-- {{#callable:ggml_compute_forward_leaky_relu_f16}} -->
Computes the forward pass of the leaky ReLU activation function for half-precision floating point tensors.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for the computation, including the index of the current operation.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the output of the leaky ReLU operation.
- **Control Flow**:
    - Checks if the current operation index is not zero; if so, the function returns early.
    - Asserts that the source tensor (`src0`) and destination tensor (`dst`) are contiguous and have the same shape.
    - Retrieves the number of rows (`n`) and the number of columns (`nc`) from the source tensor.
    - Copies the negative slope parameter from the destination tensor's operation parameters.
    - Asserts that the byte sizes of the destination and source tensors are correct for half-precision floating point values.
    - Iterates over each row of the source tensor, applying the leaky ReLU function to each corresponding row in the destination tensor.
- **Output**: The function modifies the `dst` tensor in place, storing the result of the leaky ReLU activation applied to the `src0` tensor.
- **Functions called**:
    - [`ggml_is_contiguous_1`](../ggml.c.driver.md#ggml_is_contiguous_1)
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_vec_leaky_relu_f16`](vec.h.driver.md#ggml_vec_leaky_relu_f16)


---
### ggml\_compute\_forward\_leaky\_relu<!-- {{#callable:ggml_compute_forward_leaky_relu}} -->
The `ggml_compute_forward_leaky_relu` function computes the forward pass of the leaky ReLU activation function for a given tensor based on its data type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` that will hold the result of the leaky ReLU computation.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the data type of `src0` using a switch statement.
    - If the type is `GGML_TYPE_F32`, it calls [`ggml_compute_forward_leaky_relu_f32`](#ggml_compute_forward_leaky_relu_f32) to perform the computation.
    - If the type is `GGML_TYPE_F16`, it calls [`ggml_compute_forward_leaky_relu_f16`](#ggml_compute_forward_leaky_relu_f16) instead.
    - If the type does not match any expected types, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the results of the leaky ReLU activation.
- **Functions called**:
    - [`ggml_compute_forward_leaky_relu_f32`](#ggml_compute_forward_leaky_relu_f32)
    - [`ggml_compute_forward_leaky_relu_f16`](#ggml_compute_forward_leaky_relu_f16)


---
### ggml\_compute\_forward\_silu\_back\_f32<!-- {{#callable:ggml_compute_forward_silu_back_f32}} -->
Computes the backward pass of the SiLU activation function for a given gradient tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for the computation, including the thread index and total number of threads.
    - `dst`: A pointer to a `ggml_tensor` that will hold the output of the backward computation, which is derived from the source tensors.
- **Control Flow**:
    - The function begins by retrieving the source tensors `grad` and `src1` from the `dst` tensor.
    - It asserts that all involved tensors are contiguous and have the same shape to ensure valid operations.
    - The number of channels (`nc`) and rows (`nr`) for the source tensor `src1` are determined.
    - The function calculates the number of rows each thread will process and determines the specific row range for the current thread based on its index.
    - A loop iterates over the assigned rows, calling [`ggml_vec_silu_backward_f32`](vec.h.driver.md#ggml_vec_silu_backward_f32) to perform the SiLU backward operation for each row.
    - In debug mode, it checks that the computed values in the `dst` tensor are neither NaN nor infinite.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to store the results of the backward computation.
- **Functions called**:
    - [`ggml_is_contiguous_1`](../ggml.c.driver.md#ggml_is_contiguous_1)
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_vec_silu_backward_f32`](vec.h.driver.md#ggml_vec_silu_backward_f32)


---
### ggml\_compute\_forward\_silu\_back\_f16<!-- {{#callable:ggml_compute_forward_silu_back_f16}} -->
Computes the backward pass of the Sigmoid Linear Unit (SiLU) activation function for half-precision floating-point tensors.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for the computation, including the thread index (`ith`) and the total number of threads (`nth`).
    - `dst`: A pointer to a `ggml_tensor` that will hold the output gradients, which is derived from the source tensors.
- **Control Flow**:
    - The function begins by retrieving the source tensors (`grad` and `src1`) from the `dst` tensor.
    - It asserts that the tensors are contiguous and have the same shape to ensure valid operations.
    - The number of channels (`nc`) and rows (`nr`) are determined from `src1`.
    - The number of rows processed per thread is calculated, and the specific range of rows for the current thread is determined.
    - A loop iterates over the assigned row range, calling [`ggml_vec_silu_backward_f16`](vec.h.driver.md#ggml_vec_silu_backward_f16) to compute the backward gradients for each row.
    - In debug mode, additional assertions check that the computed values are not NaN or infinite.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to store the computed gradients from the backward pass.
- **Functions called**:
    - [`ggml_is_contiguous_1`](../ggml.c.driver.md#ggml_is_contiguous_1)
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_vec_silu_backward_f16`](vec.h.driver.md#ggml_vec_silu_backward_f16)


---
### ggml\_compute\_forward\_silu\_back<!-- {{#callable:ggml_compute_forward_silu_back}} -->
The `ggml_compute_forward_silu_back` function computes the backward pass of the Swish activation function for different tensor types.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of the source tensor `src0` using a switch statement.
    - If the type is `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_silu_back_f32`](#ggml_compute_forward_silu_back_f32) function to perform the computation for 32-bit floating point tensors.
    - If the type is `GGML_TYPE_F16`, it calls the [`ggml_compute_forward_silu_back_f16`](#ggml_compute_forward_silu_back_f16) function for 16-bit floating point tensors.
    - If the type does not match any known types, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place based on the computations performed for the specified tensor type.
- **Functions called**:
    - [`ggml_compute_forward_silu_back_f32`](#ggml_compute_forward_silu_back_f32)
    - [`ggml_compute_forward_silu_back_f16`](#ggml_compute_forward_silu_back_f16)


---
### ggml\_compute\_forward\_norm\_f32<!-- {{#callable:ggml_compute_forward_norm_f32}} -->
Computes the forward normalization of a tensor by centering and scaling its values.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for computation, including the current thread index (`ith`) and the total number of threads (`nth`).
    - `dst`: A pointer to the destination `ggml_tensor` where the normalized output will be stored.
- **Control Flow**:
    - The function begins by asserting that the source tensor (`src0`) and destination tensor (`dst`) have the same shape.
    - It checks that the data type of the source tensor is `float`.
    - The function retrieves the thread index and total number of threads from `params` to facilitate parallel processing.
    - It reads a small constant `eps` from the destination tensor's operation parameters to prevent division by zero.
    - The main computation is performed in a nested loop structure iterating over the dimensions of the tensor, processing slices of the tensor based on the current thread index.
    - For each slice, it calculates the mean of the values, centers the values by subtracting the mean, and computes the variance.
    - It scales the centered values using the computed variance and the `eps` value to ensure numerical stability.
    - Finally, the scaled values are written to the destination tensor.
- **Output**: The function does not return a value but modifies the `dst` tensor in place, storing the normalized values.
- **Functions called**:
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_vec_scale_f32`](vec.h.driver.md#ggml_vec_scale_f32)


---
### ggml\_compute\_forward\_norm<!-- {{#callable:ggml_compute_forward_norm}} -->
The `ggml_compute_forward_norm` function computes the forward normalization of a tensor based on its type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` that will hold the result of the normalization.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of `src0` using a switch statement.
    - If `src0` is of type `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_norm_f32`](#ggml_compute_forward_norm_f32) function to perform the normalization.
    - If `src0` is of any other type, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the computed normalization result.
- **Functions called**:
    - [`ggml_compute_forward_norm_f32`](#ggml_compute_forward_norm_f32)


---
### ggml\_compute\_forward\_rms\_norm\_f32<!-- {{#callable:ggml_compute_forward_rms_norm_f32}} -->
Computes the RMS normalization of a tensor and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing computation parameters, including the current thread index (`ith`) and the total number of threads (`nth`).
    - `dst`: A pointer to the destination `ggml_tensor` where the normalized results will be stored.
- **Control Flow**:
    - The function begins by retrieving the source tensor (`src0`) from the destination tensor's source array.
    - It asserts that the shapes of `src0` and `dst` are the same and that the data type of `src0` is a float.
    - The function extracts the epsilon value from the destination tensor's operation parameters and asserts that it is non-negative.
    - A nested loop iterates over the dimensions of the tensor, processing slices based on the current thread index and total number of threads.
    - For each slice, it calculates the sum of squares of the elements, computes the mean, and then scales the original tensor values by the inverse of the square root of the mean plus epsilon.
    - The scaled values are then copied to the destination tensor.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the RMS normalized values of the input tensor.
- **Functions called**:
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_vec_scale_f32`](vec.h.driver.md#ggml_vec_scale_f32)


---
### ggml\_compute\_forward\_rms\_norm<!-- {{#callable:ggml_compute_forward_rms_norm}} -->
Computes the forward RMS normalization for a tensor based on its type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` that will hold the result of the RMS normalization.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of `src0` using a switch statement.
    - If `src0` is of type `GGML_TYPE_F32`, it calls the helper function [`ggml_compute_forward_rms_norm_f32`](#ggml_compute_forward_rms_norm_f32) to perform the computation.
    - If `src0` is of any other type, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor to contain the result of the RMS normalization.
- **Functions called**:
    - [`ggml_compute_forward_rms_norm_f32`](#ggml_compute_forward_rms_norm_f32)


---
### ggml\_compute\_forward\_rms\_norm\_back\_f32<!-- {{#callable:ggml_compute_forward_rms_norm_back_f32}} -->
Computes the backward pass of the RMS normalization operation for a neural network layer.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for the computation, including the current thread index and total number of threads.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the output gradients from the backward pass.
- **Control Flow**:
    - The function begins by retrieving the source tensors `src0` and `src1` from the `dst` tensor, which represent the gradients from the forward pass and the input to the RMS normalization, respectively.
    - It asserts that the shapes of `src0`, `src1`, and `dst` are the same and that the data types of the tensors are valid (float).
    - The function then initializes local variables and retrieves the epsilon value from the operation parameters.
    - A nested loop iterates over the dimensions of the tensors, processing each slice of the tensors based on the current thread index and total number of threads.
    - Within the innermost loop, it computes the sum of squares and the sum of products of the input tensor `src1` and the gradient tensor `src0`.
    - It calculates the adjusted RMS value and prepares to compute the gradients for the backward pass.
    - Finally, it updates the output tensor `dst` with the computed gradients using vector operations.
- **Output**: The function modifies the `dst` tensor in place, storing the computed gradients resulting from the backward pass of the RMS normalization operation.
- **Functions called**:
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_vec_cpy_f32`](vec.h.driver.md#ggml_vec_cpy_f32)
    - [`ggml_vec_scale_f32`](vec.h.driver.md#ggml_vec_scale_f32)
    - [`ggml_vec_acc_f32`](vec.h.driver.md#ggml_vec_acc_f32)


---
### ggml\_compute\_forward\_rms\_norm\_back<!-- {{#callable:ggml_compute_forward_rms_norm_back}} -->
The `ggml_compute_forward_rms_norm_back` function computes the backward pass of RMS normalization for a given tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` that represents the destination tensor where the result will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of the source tensor `src0` using a switch statement.
    - If the type is `GGML_TYPE_F32`, it calls the function [`ggml_compute_forward_rms_norm_back_f32`](#ggml_compute_forward_rms_norm_back_f32) to perform the computation.
    - If the type is not recognized, it triggers an abort with a fatal error message.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place based on the computed results from the backward RMS normalization.
- **Functions called**:
    - [`ggml_compute_forward_rms_norm_back_f32`](#ggml_compute_forward_rms_norm_back_f32)


---
### ggml\_compute\_forward\_group\_norm\_f32<!-- {{#callable:ggml_compute_forward_group_norm_f32}} -->
Computes the forward group normalization for a tensor in floating-point format.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing computation parameters, including the index of the current thread and the total number of threads.
    - `dst`: A pointer to the destination `ggml_tensor` where the normalized output will be stored.
- **Control Flow**:
    - The function begins by asserting that the source tensor (`src0`) has the same shape as the destination tensor (`dst`).
    - It retrieves the number of channels and groups from the tensor's parameters.
    - A loop iterates over the groups, processing each group in parallel based on the thread index.
    - For each group, it calculates the mean of the values across the specified dimensions.
    - It then computes the variance and scales the values based on the calculated mean and variance.
    - Finally, the normalized values are stored in the destination tensor.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the group-normalized values.
- **Functions called**:
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_vec_scale_f32`](vec.h.driver.md#ggml_vec_scale_f32)


---
### ggml\_compute\_forward\_group\_norm<!-- {{#callable:ggml_compute_forward_group_norm}} -->
The `ggml_compute_forward_group_norm` function computes the forward group normalization for a given tensor based on its type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the result of the group normalization.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of `src0` using a switch statement.
    - If the type is `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_group_norm_f32`](#ggml_compute_forward_group_norm_f32) function to perform the computation.
    - If the type is not recognized, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the computed group normalization results.
- **Functions called**:
    - [`ggml_compute_forward_group_norm_f32`](#ggml_compute_forward_group_norm_f32)


---
### ggml\_compute\_forward\_l2\_norm\_f32<!-- {{#callable:ggml_compute_forward_l2_norm_f32}} -->
Computes the L2 norm of a tensor and scales its values accordingly.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing computation parameters, including the current thread index (`ith`) and the total number of threads (`nth`).
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the output of the L2 norm computation.
- **Control Flow**:
    - The function begins by retrieving the source tensor (`src0`) from the destination tensor (`dst`).
    - It asserts that the shapes of `src0` and `dst` are the same and that the data type of `src0` is a float.
    - The function extracts the thread index (`ith`) and the total number of threads (`nth`) from the `params` structure.
    - It initializes a variable `eps` by copying the first float from the operation parameters of `dst` and asserts that `eps` is non-negative.
    - A nested loop iterates over the dimensions of the tensor, where the outer loops iterate over the last three dimensions (`i03`, `i02`), and the inner loop iterates over the first dimension (`i01`) with a stride of `nth`.
    - Within the innermost loop, it calculates the sum of squares of the elements in the tensor slice, scales the values by the computed L2 norm, and stores the result back in the destination tensor.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place, storing the scaled values based on the computed L2 norm.
- **Functions called**:
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_vec_scale_f32`](vec.h.driver.md#ggml_vec_scale_f32)


---
### ggml\_compute\_forward\_l2\_norm<!-- {{#callable:ggml_compute_forward_l2_norm}} -->
Computes the L2 norm of a tensor based on its type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` that will hold the result of the L2 norm computation.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of `src0` using a switch statement.
    - If `src0` is of type `GGML_TYPE_F32`, it calls the helper function [`ggml_compute_forward_l2_norm_f32`](#ggml_compute_forward_l2_norm_f32) to perform the computation.
    - If `src0` is of any other type, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor to contain the computed L2 norm.
- **Functions called**:
    - [`ggml_compute_forward_l2_norm_f32`](#ggml_compute_forward_l2_norm_f32)


---
### ggml\_compute\_forward\_out\_prod\_f32<!-- {{#callable:ggml_compute_forward_out_prod_f32}} -->
Computes the outer product of two tensors and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for computation, including thread information.
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the outer product will be stored.
- **Control Flow**:
    - The function begins by asserting that the types of the destination and source tensors are `GGML_TYPE_F32`.
    - It initializes local variables for tensor dimensions and checks for compatibility between the source tensors.
    - If the current thread index is zero, it initializes the destination tensor to zero.
    - The function calculates the total number of rows in the destination tensor and determines the range of rows to be processed by the current thread.
    - It uses nested loops to iterate over the dimensions of the tensors, performing the outer product computation in blocks for efficiency.
    - The inner loop performs the actual multiplication and accumulation of values from the source tensors into the destination tensor.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the result of the outer product of the two source tensors.
- **Functions called**:
    - [`ggml_vec_set_f32`](vec.h.driver.md#ggml_vec_set_f32)
    - [`ggml_barrier`](ggml-cpu.c.driver.md#ggml_barrier)
    - [`ggml_vec_mad_f32_unroll`](vec.h.driver.md#ggml_vec_mad_f32_unroll)
    - [`ggml_vec_mad_f32`](vec.h.driver.md#ggml_vec_mad_f32)


---
### ggml\_compute\_forward\_out\_prod\_q\_f32<!-- {{#callable:ggml_compute_forward_out_prod_q_f32}} -->
Computes the outer product of two tensors and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for computation, including thread information.
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the outer product will be stored.
- **Control Flow**:
    - Extracts source tensors `src0` and `src1` from the destination tensor `dst`.
    - Validates tensor dimensions and types using assertions to ensure compatibility.
    - Initializes the destination tensor to zero if the current thread index is zero.
    - Calculates the total number of rows in the destination tensor and determines the range of rows to process for the current thread.
    - Iterates over the assigned row range, calculating indices for the destination and source tensors.
    - For each row, it dequantizes the corresponding row from `src0` and performs the outer product with `src1`, accumulating the result in `dst`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the result of the outer product.
- **Functions called**:
    - [`ggml_get_type_traits`](../ggml.c.driver.md#ggml_get_type_traits)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_vec_set_f32`](vec.h.driver.md#ggml_vec_set_f32)
    - [`ggml_barrier`](ggml-cpu.c.driver.md#ggml_barrier)
    - [`ggml_vec_mad_f32`](vec.h.driver.md#ggml_vec_mad_f32)


---
### ggml\_compute\_forward\_out\_prod<!-- {{#callable:ggml_compute_forward_out_prod}} -->
Computes the forward output product for a given tensor based on its type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` structure representing the destination tensor where the output will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of `src0` using a switch statement to determine the appropriate computation method.
    - For specific tensor types (e.g., `GGML_TYPE_Q4_0`, `GGML_TYPE_F32`), it calls the corresponding computation function.
    - If the tensor type is `GGML_TYPE_F16`, it triggers a fatal error indicating that this case is not yet implemented.
    - For any unsupported tensor types, it also triggers a fatal error.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place based on the computation performed.
- **Functions called**:
    - [`ggml_compute_forward_out_prod_q_f32`](#ggml_compute_forward_out_prod_q_f32)
    - [`ggml_compute_forward_out_prod_f32`](#ggml_compute_forward_out_prod_f32)


---
### ggml\_compute\_forward\_scale\_f32<!-- {{#callable:ggml_compute_forward_scale_f32}} -->
The `ggml_compute_forward_scale_f32` function scales the elements of a source tensor by a specified scale factor and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for the computation, including the thread index and total number of threads.
    - `dst`: A pointer to a `ggml_tensor` structure that serves as the destination tensor where the scaled results will be stored.
- **Control Flow**:
    - The function begins by retrieving the source tensor `src0` from the destination tensor `dst`.
    - It asserts that both `src0` and `dst` are contiguous and have the same shape.
    - The scale factor is extracted from the operation parameters of the destination tensor.
    - The function calculates the number of rows per thread and determines the range of rows that the current thread will process.
    - A loop iterates over the assigned row range, copying data from `src0` to `dst` if they are not the same, and then applies the scaling operation to each row using [`ggml_vec_scale_f32`](vec.h.driver.md#ggml_vec_scale_f32).
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place by scaling its elements based on the specified scale factor.
- **Functions called**:
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_vec_scale_f32`](vec.h.driver.md#ggml_vec_scale_f32)


---
### ggml\_compute\_forward\_scale<!-- {{#callable:ggml_compute_forward_scale}} -->
The `ggml_compute_forward_scale` function computes the forward scaling operation for a tensor based on its type.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of the source tensor `src0` using a switch statement.
    - If the type is `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_scale_f32`](#ggml_compute_forward_scale_f32) function to perform the scaling operation.
    - If the type is not recognized, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place based on the scaling operation performed.
- **Functions called**:
    - [`ggml_compute_forward_scale_f32`](#ggml_compute_forward_scale_f32)


---
### ggml\_compute\_forward\_set\_f32<!-- {{#callable:ggml_compute_forward_set_f32}} -->
The `ggml_compute_forward_set_f32` function copies data from a source tensor to a destination tensor with optional in-place modification and handles multi-threading synchronization.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for computation, including thread index and total number of threads.
    - `dst`: A pointer to a `ggml_tensor` structure representing the destination tensor where data will be set.
- **Control Flow**:
    - The function begins by retrieving source tensors `src0` and `src1` from the destination tensor `dst`.
    - It asserts that `src0` and `dst` have the same shape and that both are contiguous in memory.
    - The function extracts parameters from `dst->op_params` to determine the sizes and offsets for the data copy operation.
    - If the operation is not in-place and the thread index is zero, it performs a memory copy from `src0` to `dst` to initialize the destination tensor.
    - A barrier is used to synchronize threads after the initialization copy to prevent race conditions.
    - The function calculates the number of rows and columns in `src1` and determines the range of rows each thread will process.
    - It iterates over the assigned row range, calculating the appropriate indices for accessing elements in `src0` and `dst` based on the specified offsets and strides.
    - For each row, it calls [`ggml_vec_cpy_f32`](vec.h.driver.md#ggml_vec_cpy_f32) to copy the data from `src1` to the calculated position in `dst`.
- **Output**: The function does not return a value; it modifies the destination tensor `dst` in place by copying data from the source tensor `src1` based on the specified parameters.
- **Functions called**:
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)
    - [`ggml_barrier`](ggml-cpu.c.driver.md#ggml_barrier)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_element_size`](../ggml.c.driver.md#ggml_element_size)
    - [`ggml_vec_cpy_f32`](vec.h.driver.md#ggml_vec_cpy_f32)


---
### ggml\_compute\_forward\_set\_i32<!-- {{#callable:ggml_compute_forward_set_i32}} -->
The `ggml_compute_forward_set_i32` function sets the values of a destination tensor (`dst`) based on the values from a source tensor (`src1`), optionally using a source tensor (`src0`) for initialization.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for the computation, including thread information.
    - `dst`: A pointer to a `ggml_tensor` structure representing the destination tensor where values will be set.
- **Control Flow**:
    - The function begins by retrieving the source tensors `src0` and `src1` from the destination tensor `dst`.
    - It asserts that `src0` and `dst` have the same shape and that both are contiguous in memory.
    - The function extracts parameters from `dst->op_params`, including sizes and an offset for setting values.
    - If the operation is not in-place and it is the first thread, it copies data from `src0` to `dst` to initialize it.
    - A barrier is used to synchronize threads after the initialization copy.
    - The function calculates the number of rows and columns in `src1` and determines the range of rows to process for the current thread.
    - It iterates over the assigned row range, calculating the appropriate indices for accessing elements in `src0`, `src1`, and `dst` based on the specified offset.
    - For each row, it copies values from `src1` to the calculated position in `dst` using the [`ggml_vec_cpy_i32`](vec.h.driver.md#ggml_vec_cpy_i32) function.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place by setting its values based on the specified source tensor `src1`.
- **Functions called**:
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)
    - [`ggml_barrier`](ggml-cpu.c.driver.md#ggml_barrier)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_element_size`](../ggml.c.driver.md#ggml_element_size)
    - [`ggml_vec_cpy_i32`](vec.h.driver.md#ggml_vec_cpy_i32)


---
### ggml\_compute\_forward\_set<!-- {{#callable:ggml_compute_forward_set}} -->
The `ggml_compute_forward_set` function computes the forward set operation for a given tensor based on its data type.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of the source tensor `src0` using a switch statement.
    - If the type is `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_set_f32`](#ggml_compute_forward_set_f32) function to handle the computation.
    - If the type is `GGML_TYPE_I32`, it calls the [`ggml_compute_forward_set_i32`](#ggml_compute_forward_set_i32) function for the computation.
    - For all other types, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; it modifies the destination tensor `dst` in place based on the computation performed.
- **Functions called**:
    - [`ggml_compute_forward_set_f32`](#ggml_compute_forward_set_f32)
    - [`ggml_compute_forward_set_i32`](#ggml_compute_forward_set_i32)


---
### ggml\_compute\_forward\_cpy<!-- {{#callable:ggml_compute_forward_cpy}} -->
Copies the data from a source tensor to a destination tensor using the specified computation parameters.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where data will be copied.
- **Control Flow**:
    - The function calls [`ggml_compute_forward_dup`](#ggml_compute_forward_dup) with the provided parameters and destination tensor.
    - No additional logic or control flow is present; the function directly delegates the operation to another function.
- **Output**: The function does not return a value; it performs an operation that modifies the destination tensor in place.
- **Functions called**:
    - [`ggml_compute_forward_dup`](#ggml_compute_forward_dup)


---
### ggml\_compute\_forward\_cont<!-- {{#callable:ggml_compute_forward_cont}} -->
This function computes the forward operation for a continuation tensor by duplicating the input tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` that will store the result of the forward computation.
- **Control Flow**:
    - The function calls [`ggml_compute_forward_dup`](#ggml_compute_forward_dup) with the provided parameters and destination tensor.
    - No conditional logic or loops are present; the function directly delegates the computation to another function.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place based on the computation performed by [`ggml_compute_forward_dup`](#ggml_compute_forward_dup).
- **Functions called**:
    - [`ggml_compute_forward_dup`](#ggml_compute_forward_dup)


---
### ggml\_compute\_forward\_reshape<!-- {{#callable:ggml_compute_forward_reshape}} -->
This function is a placeholder that does not perform any operations.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for computation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor for the reshape operation.
- **Control Flow**:
    - This function does not contain any operational logic and simply marks the inputs as unused.
    - It effectively acts as a no-operation (NOP) function.
- **Output**: The function does not produce any output or return value.


---
### ggml\_compute\_forward\_view<!-- {{#callable:ggml_compute_forward_view}} -->
This function is a no-operation (NOP) placeholder that takes in compute parameters and a destination tensor but does not perform any operations.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for computation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where results would be stored.
- **Control Flow**:
    - The function begins by marking the `params` and `dst` inputs as unused to avoid compiler warnings.
    - No further operations or computations are performed within the function body.
- **Output**: The function does not produce any output or return value, as it is designed to be a placeholder.


---
### ggml\_compute\_forward\_permute<!-- {{#callable:ggml_compute_forward_permute}} -->
This function is a no-operation (NOP) placeholder that does not perform any computation.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for computation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor for the computation.
- **Control Flow**:
    - The function begins by marking the `params` and `dst` inputs as unused to avoid compiler warnings.
    - No further operations or computations are performed within the function.
- **Output**: The function does not produce any output or return value, as it is designed to do nothing.


---
### ggml\_compute\_forward\_transpose<!-- {{#callable:ggml_compute_forward_transpose}} -->
This function is a placeholder that does not perform any operations.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for computation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor for the computation.
- **Control Flow**:
    - The function begins by marking the `params` and `dst` inputs as unused to avoid compiler warnings.
    - No further operations or computations are performed within the function.
- **Output**: The function does not produce any output as it is a no-operation (NOP) function.


---
### ggml\_compute\_forward\_get\_rows\_q<!-- {{#callable:ggml_compute_forward_get_rows_q}} -->
The `ggml_compute_forward_get_rows_q` function processes rows of a quantized tensor and dequantizes them into a destination tensor using multi-threading.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for computation, including the current thread index (`ith`) and the total number of threads (`nth`).
    - `dst`: A pointer to a `ggml_tensor` structure that serves as the destination tensor where the dequantized rows will be stored.
- **Control Flow**:
    - The function retrieves source tensors `src0` and `src1` from the destination tensor `dst`.
    - It initializes local variables for the number of channels (`nc`), number of rows (`nr`), and the type of the source tensor.
    - Assertions are made to ensure the integrity of tensor dimensions and types.
    - The function calculates the number of rows each thread will process and determines the range of rows for the current thread.
    - A loop iterates over the assigned row range, calculating indices to access the quantized data in `src1` and dequantizing it into `dst` using the appropriate function.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place by filling it with dequantized data from the source tensors.
- **Functions called**:
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_get_type_traits`](../ggml.c.driver.md#ggml_get_type_traits)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)


---
### ggml\_compute\_forward\_get\_rows\_f16<!-- {{#callable:ggml_compute_forward_get_rows_f16}} -->
The `ggml_compute_forward_get_rows_f16` function computes and retrieves specific rows from a source tensor and stores them in a destination tensor, utilizing multi-threading for efficiency.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for computation, including the current thread index (`ith`) and the total number of threads (`nth`).
    - `dst`: A pointer to a `ggml_tensor` structure that serves as the destination tensor where the computed rows will be stored.
- **Control Flow**:
    - The function retrieves source tensors `src0` and `src1` from the destination tensor `dst`.
    - It asserts that the dimensions and sizes of the tensors are consistent with expected values.
    - The number of rows (`nr`) is calculated from `src1`, and the number of rows per thread (`dr`) is determined based on the total number of threads.
    - The function calculates the range of rows (`ir0` to `ir1`) that the current thread will process.
    - A loop iterates over the assigned row range, calculating indices to access elements in the source tensors.
    - For each row, it retrieves an index from `src1`, asserts its validity, and converts the corresponding half-precision floating-point values from `src0` to single-precision floating-point values, storing them in `dst`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place by populating it with the computed rows from `src0` based on the indices retrieved from `src1`.
- **Functions called**:
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_cpu_fp16_to_fp32`](ggml-cpu.c.driver.md#ggml_cpu_fp16_to_fp32)


---
### ggml\_compute\_forward\_get\_rows\_bf16<!-- {{#callable:ggml_compute_forward_get_rows_bf16}} -->
The `ggml_compute_forward_get_rows_bf16` function computes and retrieves rows from a source tensor in bf16 format and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for computation, including thread indices.
    - `dst`: A pointer to a `ggml_tensor` structure that serves as the destination tensor where the computed results will be stored.
- **Control Flow**:
    - The function retrieves source tensors `src0` and `src1` from the destination tensor `dst`.
    - It asserts that the dimensions and sizes of the tensors are consistent with expected values.
    - The number of rows per thread is calculated based on the total number of rows and the number of threads.
    - The function determines the range of rows to process for the current thread using the calculated row range.
    - A loop iterates over the assigned row range, converting bf16 data from `src0` to float and storing it in `dst`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the computed float values derived from the bf16 source tensor.
- **Functions called**:
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_cpu_bf16_to_fp32`](ggml-cpu.c.driver.md#ggml_cpu_bf16_to_fp32)


---
### ggml\_compute\_forward\_get\_rows\_f32<!-- {{#callable:ggml_compute_forward_get_rows_f32}} -->
The `ggml_compute_forward_get_rows_f32` function retrieves specific rows from a source tensor and copies them into a destination tensor based on provided parameters.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for computation, including the thread index and total number of threads.
    - `dst`: A pointer to a `ggml_tensor` structure that serves as the destination tensor where the rows will be copied.
- **Control Flow**:
    - The function begins by extracting source tensors `src0` and `src1` from the destination tensor `dst`.
    - It initializes local variables for tensor dimensions and asserts to ensure the integrity of tensor sizes and data types.
    - The number of rows to process per thread is calculated, and the specific range of rows for the current thread is determined.
    - A loop iterates over the assigned row range, calculating indices to access the source tensor's data and copying the corresponding rows to the destination tensor using [`ggml_vec_cpy_f32`](vec.h.driver.md#ggml_vec_cpy_f32).
- **Output**: The function does not return a value; instead, it modifies the destination tensor `dst` in place by copying the specified rows from the source tensor.
- **Functions called**:
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_vec_cpy_f32`](vec.h.driver.md#ggml_vec_cpy_f32)


---
### ggml\_compute\_forward\_get\_rows<!-- {{#callable:ggml_compute_forward_get_rows}} -->
The `ggml_compute_forward_get_rows` function computes rows from a source tensor based on its data type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure where the computed rows will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It uses a switch statement to determine the type of the source tensor `src0`.
    - Depending on the type, it calls the appropriate function to compute the rows: [`ggml_compute_forward_get_rows_q`](#ggml_compute_forward_get_rows_q), [`ggml_compute_forward_get_rows_f16`](#ggml_compute_forward_get_rows_f16), [`ggml_compute_forward_get_rows_bf16`](#ggml_compute_forward_get_rows_bf16), or [`ggml_compute_forward_get_rows_f32`](#ggml_compute_forward_get_rows_f32).
    - If the type does not match any expected types, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the computed rows based on the source tensor's type.
- **Functions called**:
    - [`ggml_compute_forward_get_rows_q`](#ggml_compute_forward_get_rows_q)
    - [`ggml_compute_forward_get_rows_f16`](#ggml_compute_forward_get_rows_f16)
    - [`ggml_compute_forward_get_rows_bf16`](#ggml_compute_forward_get_rows_bf16)
    - [`ggml_compute_forward_get_rows_f32`](#ggml_compute_forward_get_rows_f32)


---
### ggml\_compute\_forward\_get\_rows\_back\_f32\_f16<!-- {{#callable:ggml_compute_forward_get_rows_back_f32_f16}} -->
Computes the forward operation to retrieve rows from a source tensor and accumulate their values into a destination tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing computation parameters, including the index of the current operation.
    - `dst`: A pointer to a `ggml_tensor` structure that serves as the destination tensor where the results will be stored.
- **Control Flow**:
    - The function first retrieves the source tensors `src0` and `src1` from the destination tensor `dst`.
    - It checks if the current operation index (`params->ith`) is not zero; if true, it exits early.
    - It asserts that the destination tensor is contiguous in memory.
    - The function initializes the destination tensor's data to zero using `memset`.
    - It retrieves the number of columns (`nc`) from `src0` and the number of rows (`nr`) from `src1`.
    - It asserts that the number of elements in the destination tensor matches the number of columns in `src0` and that the data type of `src0` is `ggml_fp16_t`.
    - A nested loop iterates over each row index `i` of `src1`, retrieves the corresponding row index `r`, and then iterates over each column index `j` to accumulate the values from `src0` into `dst` after converting them from `ggml_fp16_t` to `float`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place by accumulating the converted values from the specified rows of `src0` based on the indices provided in `src1`.
- **Functions called**:
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_compute\_forward\_get\_rows\_back\_f32<!-- {{#callable:ggml_compute_forward_get_rows_back_f32}} -->
Computes the forward operation for retrieving rows from a source tensor and adding them to a destination tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing computation parameters, including the index of the current operation.
    - `dst`: A pointer to a `ggml_tensor` structure that serves as the destination tensor where the results will be stored.
- **Control Flow**:
    - The function first retrieves the source tensors `src0` and `src1` from the destination tensor `dst`.
    - It checks if the current operation index (`params->ith`) is not zero; if true, it exits early.
    - It asserts that the destination tensor is contiguous in memory.
    - The function initializes the destination tensor's data to zero using `memset`.
    - It retrieves the number of columns (`nc`) from `src0` and the number of rows (`nr`) from `src1`.
    - It asserts that the number of elements in the destination tensor matches the number of columns in `src0` and that the data type of `src0` is `float`.
    - A loop iterates over each row index in `src1`, retrieves the corresponding row index `r`, and adds the data from `src0` to the appropriate location in `dst` using [`ggml_vec_add_f32`](vec.h.driver.md#ggml_vec_add_f32).
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place by adding the specified rows from `src0` based on the indices provided in `src1`.
- **Functions called**:
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_vec_add_f32`](vec.h.driver.md#ggml_vec_add_f32)


---
### ggml\_compute\_forward\_get\_rows\_back<!-- {{#callable:ggml_compute_forward_get_rows_back}} -->
The `ggml_compute_forward_get_rows_back` function computes the forward operation for retrieving rows from a tensor based on its data type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure that serves as the destination tensor for the computation.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the data type of `src0` using a switch statement.
    - If the type is `GGML_TYPE_F16`, it calls [`ggml_compute_forward_get_rows_back_f32_f16`](#ggml_compute_forward_get_rows_back_f32_f16) to perform the computation.
    - If the type is `GGML_TYPE_F32`, it calls [`ggml_compute_forward_get_rows_back_f32`](#ggml_compute_forward_get_rows_back_f32) for the computation.
    - If the type is neither, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place based on the computation performed.
- **Functions called**:
    - [`ggml_compute_forward_get_rows_back_f32_f16`](#ggml_compute_forward_get_rows_back_f32_f16)
    - [`ggml_compute_forward_get_rows_back_f32`](#ggml_compute_forward_get_rows_back_f32)


---
### ggml\_compute\_forward\_diag\_f32<!-- {{#callable:ggml_compute_forward_diag_f32}} -->
Computes a diagonal matrix from the source tensor, setting non-diagonal elements to zero.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing computation parameters, including the index of the current operation.
    - `dst`: A pointer to the destination `ggml_tensor` where the diagonal matrix will be stored.
- **Control Flow**:
    - Checks if the current operation index (`params->ith`) is not zero; if so, the function returns early.
    - Retrieves the source tensor (`src0`) from the destination tensor's source array.
    - Performs assertions to ensure the dimensions and sizes of the tensors are consistent.
    - Iterates over the dimensions of the tensor, specifically for each index in the last three dimensions.
    - For each position in the diagonal, sets the corresponding element in the destination tensor to the source tensor's value and sets all other elements in that row to zero.
- **Output**: The function modifies the `dst` tensor in place to contain a diagonal matrix derived from the `src0` tensor, with all non-diagonal elements set to zero.


---
### ggml\_compute\_forward\_diag<!-- {{#callable:ggml_compute_forward_diag}} -->
The `ggml_compute_forward_diag` function computes the forward diagonal of a tensor based on its type.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of the source tensor `src0` using a switch statement.
    - If the type is `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_diag_f32`](#ggml_compute_forward_diag_f32) function to perform the computation.
    - If the type is not recognized, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the computed forward diagonal values.
- **Functions called**:
    - [`ggml_compute_forward_diag_f32`](#ggml_compute_forward_diag_f32)


---
### ggml\_compute\_forward\_diag\_mask\_f32<!-- {{#callable:ggml_compute_forward_diag_mask_f32}} -->
Computes a forward diagonal mask in a tensor by setting specific elements to a given value based on the input parameters.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing computation parameters such as thread index and total threads.
    - `dst`: A pointer to the destination `ggml_tensor` where the computed mask will be stored.
    - `value`: A float value that will be assigned to specific elements in the destination tensor.
- **Control Flow**:
    - The function begins by retrieving the source tensor from the destination tensor's source array.
    - It extracts the current thread index and total number of threads from the `params` structure.
    - The number of past elements is obtained from the destination tensor's operation parameters.
    - A check is performed to determine if the operation is in-place, which affects how data is copied.
    - If not in-place and the current thread is the first, it copies data from the source tensor to the destination tensor.
    - A barrier is used to synchronize threads after the copy operation.
    - The function calculates the dimensions of the source tensor and iterates over the tensor to set specific elements to the provided value based on the diagonal masking logic.
- **Output**: The function modifies the `dst` tensor in place, setting certain elements to the specified `value` based on the forward diagonal masking criteria.
- **Functions called**:
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)
    - [`ggml_barrier`](ggml-cpu.c.driver.md#ggml_barrier)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)


---
### ggml\_compute\_forward\_diag\_mask\_inf<!-- {{#callable:ggml_compute_forward_diag_mask_inf}} -->
The `ggml_compute_forward_diag_mask_inf` function computes a diagonal mask for a tensor, specifically handling the case for 32-bit floating point tensors.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` that will hold the result of the computation.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of `src0` using a switch statement.
    - If `src0` is of type `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_diag_mask_f32`](#ggml_compute_forward_diag_mask_f32) function with the provided parameters and a negative infinity value.
    - If `src0` is of any other type, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the computed diagonal mask.
- **Functions called**:
    - [`ggml_compute_forward_diag_mask_f32`](#ggml_compute_forward_diag_mask_f32)


---
### ggml\_compute\_forward\_diag\_mask\_zero<!-- {{#callable:ggml_compute_forward_diag_mask_zero}} -->
Computes a diagonal mask for a tensor, specifically handling the `GGML_TYPE_F32` type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for computation.
    - `dst`: A pointer to `ggml_tensor` that will hold the result of the computation.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of `src0` using a switch statement.
    - If `src0` is of type `GGML_TYPE_F32`, it calls the function [`ggml_compute_forward_diag_mask_f32`](#ggml_compute_forward_diag_mask_f32) with the provided parameters and destination tensor.
    - If `src0` is of any other type, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place based on the diagonal mask computation.
- **Functions called**:
    - [`ggml_compute_forward_diag_mask_f32`](#ggml_compute_forward_diag_mask_f32)


---
### ggml\_compute\_forward\_soft\_max\_f32<!-- {{#callable:ggml_compute_forward_soft_max_f32}} -->
Computes the forward softmax operation for a tensor using specified parameters and handles potential scaling and bias adjustments.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing computation parameters, including thread index and total number of threads.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the output of the softmax computation.
- **Control Flow**:
    - The function begins by asserting that the destination tensor is contiguous and that the source tensor shapes match.
    - It retrieves scaling and bias parameters from the destination tensor's operation parameters.
    - The function calculates the number of heads and their logarithmic representation for further processing.
    - It determines the range of rows to process for the current thread based on the total number of threads.
    - For each row in the determined range, it computes a slope based on the head index and bias, and prepares the data for softmax computation.
    - The function copies and scales the input data, applying any additional mask from a second source tensor if provided.
    - It computes the maximum value in the processed data to facilitate the softmax calculation.
    - The softmax values are computed and normalized, ensuring that the output values are valid and finite.
- **Output**: The function modifies the `dst` tensor in place to contain the softmax values computed from the input tensor, ensuring that the output is properly scaled and normalized.
- **Functions called**:
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_vec_cpy_f32`](vec.h.driver.md#ggml_vec_cpy_f32)
    - [`ggml_vec_scale_f32`](vec.h.driver.md#ggml_vec_scale_f32)
    - [`ggml_vec_max_f32`](vec.h.driver.md#ggml_vec_max_f32)
    - [`ggml_vec_soft_max_f32`](vec.cpp.driver.md#ggml_vec_soft_max_f32)


---
### ggml\_compute\_forward\_soft\_max<!-- {{#callable:ggml_compute_forward_soft_max}} -->
Computes the forward softmax operation for a given tensor based on its type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` where the result of the softmax operation will be stored.
- **Control Flow**:
    - Retrieves the source tensor `src0` from the destination tensor `dst`.
    - Checks the type of the source tensor `src0` using a switch statement.
    - If the type is `GGML_TYPE_F32`, it calls the function [`ggml_compute_forward_soft_max_f32`](#ggml_compute_forward_soft_max_f32) to perform the softmax computation.
    - If the type is not recognized, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor to contain the result of the softmax operation.
- **Functions called**:
    - [`ggml_compute_forward_soft_max_f32`](#ggml_compute_forward_soft_max_f32)


---
### ggml\_compute\_forward\_soft\_max\_ext\_back\_f32<!-- {{#callable:ggml_compute_forward_soft_max_ext_back_f32}} -->
Computes the backward pass of the softmax function for a given tensor, applying gradients and scaling.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure containing parameters for computation, including thread indices.
    - `dst`: Pointer to a `ggml_tensor` that will store the result of the backward softmax computation.
- **Control Flow**:
    - Asserts that the source tensors (`src0`, `src1`) and destination tensor (`dst`) are contiguous and have the same shape.
    - Extracts the `scale` and `max_bias` parameters from the destination tensor's operation parameters.
    - Calculates the number of rows per thread and determines the range of rows to process for the current thread.
    - Iterates over the assigned rows, performing the backward softmax computation using vector operations.
    - Computes the dot product of the output and gradient tensors, copies the gradient to the destination, and applies the necessary transformations.
- **Output**: The function modifies the `dst` tensor in place, storing the computed gradients for the backward pass of the softmax operation.
- **Functions called**:
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_vec_dot_f32`](vec.cpp.driver.md#ggml_vec_dot_f32)
    - [`ggml_vec_cpy_f32`](vec.h.driver.md#ggml_vec_cpy_f32)
    - [`ggml_vec_acc1_f32`](vec.h.driver.md#ggml_vec_acc1_f32)
    - [`ggml_vec_mul_f32`](vec.h.driver.md#ggml_vec_mul_f32)
    - [`ggml_vec_scale_f32`](vec.h.driver.md#ggml_vec_scale_f32)


---
### ggml\_compute\_forward\_soft\_max\_ext\_back<!-- {{#callable:ggml_compute_forward_soft_max_ext_back}} -->
Computes the backward pass of the softmax operation for a tensor based on its type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` that represents the destination tensor where the result will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of `src0` using a switch statement.
    - If `src0` is of type `GGML_TYPE_F32`, it calls the function [`ggml_compute_forward_soft_max_ext_back_f32`](#ggml_compute_forward_soft_max_ext_back_f32) to perform the computation.
    - If `src0` is of any other type, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place based on the computed softmax values.
- **Functions called**:
    - [`ggml_compute_forward_soft_max_ext_back_f32`](#ggml_compute_forward_soft_max_ext_back_f32)


---
### ggml\_compute\_forward\_clamp\_f32<!-- {{#callable:ggml_compute_forward_clamp_f32}} -->
The `ggml_compute_forward_clamp_f32` function clamps the values of a source tensor to a specified minimum and maximum range and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for the computation, including the current thread index (`ith`) and the total number of threads (`nth`).
    - `dst`: A pointer to a `ggml_tensor` structure that serves as the destination tensor where the clamped values will be stored.
- **Control Flow**:
    - The function retrieves the source tensor from the destination tensor's source array.
    - It extracts the minimum and maximum clamp values from the `op_params` of the destination tensor.
    - The function calculates the number of rows (`n`) and the number of columns (`nc`) in the source tensor.
    - It asserts that the size of the destination tensor's data type is correct.
    - A loop iterates over the rows of the source tensor, processing only the rows assigned to the current thread based on `ith` and `nth`.
    - Within the row loop, another loop iterates over each column, clamping the value of each element to the specified range using the `MAX` and `MIN` functions.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place, storing the clamped values derived from the `src0` tensor.
- **Functions called**:
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)


---
### ggml\_compute\_forward\_clamp\_f16<!-- {{#callable:ggml_compute_forward_clamp_f16}} -->
Computes the clamped values of a source tensor and stores them in a destination tensor using half-precision floating-point format.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing computation parameters, including the index of the current thread and the total number of threads.
    - `dst`: A pointer to the destination `ggml_tensor` where the clamped results will be stored.
- **Control Flow**:
    - The function retrieves the source tensor from the destination tensor's source array.
    - It extracts the minimum and maximum clamp values from the operation parameters.
    - The function calculates the number of rows (`n`) and channels (`nc`) in the source tensor.
    - It asserts that the sizes of the destination and source tensor data types are correct.
    - A loop iterates over the rows of the source tensor, processing them in parallel based on the thread index and total threads.
    - Within the row loop, another loop iterates over the channels, converting each value from half-precision to single-precision, clamping it, and converting it back to half-precision before storing it in the destination tensor.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with clamped values based on the specified minimum and maximum.
- **Functions called**:
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)


---
### ggml\_compute\_forward\_clamp<!-- {{#callable:ggml_compute_forward_clamp}} -->
The `ggml_compute_forward_clamp` function processes a tensor by applying a clamping operation based on the tensor's data type.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result of the clamping operation will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It then checks the data type of `src0` using a switch statement.
    - If the data type is `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_clamp_f32`](#ggml_compute_forward_clamp_f32) function to perform the clamping operation.
    - If the data type is `GGML_TYPE_F16`, it calls the [`ggml_compute_forward_clamp_f16`](#ggml_compute_forward_clamp_f16) function for the clamping operation.
    - For all other data types, the function triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place based on the clamping operation applied to the `src0` tensor.
- **Functions called**:
    - [`ggml_compute_forward_clamp_f32`](#ggml_compute_forward_clamp_f32)
    - [`ggml_compute_forward_clamp_f16`](#ggml_compute_forward_clamp_f16)


---
### rope\_yarn\_ramp<!-- {{#callable:rope_yarn_ramp}} -->
Calculates a ramp value based on input parameters to normalize a value between a specified range.
- **Inputs**:
    - `low`: The lower bound of the range used for normalization.
    - `high`: The upper bound of the range used for normalization.
    - `i0`: An integer input that is used to calculate the ramp value.
- **Control Flow**:
    - Calculates the normalized value `y` by subtracting `low` from half of `i0` and dividing by the maximum of 0.001 and the difference between `high` and `low`.
    - Clamps the value of `y` to ensure it is between 0 and 1 using the `MIN` and `MAX` functions.
    - Returns the final ramp value as 1 minus the clamped value of `y`.
- **Output**: Returns a float representing the ramp value, which is clamped between 0 and 1.


---
### rope\_yarn<!-- {{#callable:rope_yarn}} -->
Calculates the cosine and sine of an angle adjusted for extrapolation and scaling.
- **Inputs**:
    - `theta_extrap`: The extrapolated angle in radians.
    - `freq_scale`: A scaling factor for frequency.
    - `corr_dims`: An array containing two correction dimensions.
    - `i0`: An index used for ramp calculation.
    - `ext_factor`: A factor that influences the extent of extrapolation.
    - `mscale`: A magnitude scaling factor.
    - `cos_theta`: Pointer to store the calculated cosine of the adjusted angle.
    - `sin_theta`: Pointer to store the calculated sine of the adjusted angle.
- **Control Flow**:
    - Calculates an interpolated angle `theta_interp` based on `theta_extrap` and `freq_scale`.
    - If `ext_factor` is not zero, it computes a mixing factor `ramp_mix` using the [`rope_yarn_ramp`](#rope_yarn_ramp) function and adjusts `theta` accordingly.
    - Updates the `mscale` based on the logarithm of the inverse of `freq_scale`.
    - Calculates the cosine and sine of the final `theta`, scaled by `mscale`, and stores the results in the provided pointers.
- **Output**: The function outputs the cosine and sine of the adjusted angle through the provided pointers, scaled by the magnitude factor.
- **Functions called**:
    - [`rope_yarn_ramp`](#rope_yarn_ramp)


---
### ggml\_rope\_cache\_init<!-- {{#callable:ggml_rope_cache_init}} -->
Initializes a cache for rotary embeddings using specified parameters and frequency factors.
- **Inputs**:
    - `theta_base`: The base angle used for the rotary embedding calculations.
    - `freq_scale`: A scaling factor for the frequency.
    - `freq_factors`: An array of frequency factors that may modify the theta value.
    - `corr_dims`: An array containing two dimensions for correlation.
    - `ne0`: The number of elements to process, expected to be even.
    - `ext_factor`: An external factor that influences the embedding calculations.
    - `mscale`: A scaling factor for the cache values.
    - `cache`: A pointer to the cache array where results will be stored.
    - `sin_sign`: A sign multiplier for the sine component of the cache.
    - `theta_scale`: A scaling factor for updating theta in each iteration.
- **Control Flow**:
    - Initializes the variable `theta` with the value of `theta_base`.
    - Iterates over a range from 0 to `ne0`, incrementing by 2 in each iteration.
    - Calculates the frequency factor `ff` based on the `freq_factors` array or defaults to 1.0 if not provided.
    - Calls the [`rope_yarn`](#rope_yarn) function to compute values for the cache using the current `theta`, `ff`, and other parameters.
    - Multiplies the second cache element by `sin_sign` to adjust its value.
    - Updates `theta` by multiplying it with `theta_scale` for the next iteration.
- **Output**: The function does not return a value; instead, it modifies the `cache` array in place with computed rotary embedding values.
- **Functions called**:
    - [`rope_yarn`](#rope_yarn)


---
### ggml\_mrope\_cache\_init<!-- {{#callable:ggml_mrope_cache_init}} -->
Initializes a cache for rotary embeddings based on various parameters and frequency factors.
- **Inputs**:
    - `theta_base_t`: Base theta value for the time dimension.
    - `theta_base_h`: Base theta value for the height dimension.
    - `theta_base_w`: Base theta value for the width dimension.
    - `theta_base_e`: Base theta value for the extra position id, used in vision encoders.
    - `sections`: An array of integers representing the number of sections for each dimension.
    - `indep_sects`: A boolean indicating whether to compute theta independently for each section.
    - `freq_scale`: A scaling factor for frequency.
    - `freq_factors`: An array of frequency factors, or NULL if not used.
    - `corr_dims`: An array of two floats representing correction dimensions.
    - `ne0`: An integer representing the total number of elements to process.
    - `ext_factor`: An external scaling factor.
    - `mscale`: A scaling factor for the cache.
    - `cache`: A pointer to the cache array where results will be stored.
    - `sin_sign`: A float that modifies the sine value in the cache.
    - `theta_scale`: A scaling factor for theta values.
- **Control Flow**:
    - Initializes theta values from the base parameters.
    - Calculates the total number of sections and their dimensions.
    - Asserts that the total dimensions do not exceed the specified limit (ne0).
    - Iterates over a range of indices (i0) in steps of 2, processing each pair of cache entries.
    - Determines the frequency factor to apply based on the input array.
    - Calculates the current sector based on the index and updates theta values if independent sections are enabled.
    - Calls the [`rope_yarn`](#rope_yarn) function to compute values for the cache based on the current theta and other parameters.
    - Applies a sine modification to the second cache entry.
    - Scales the theta values for the next iteration.
- **Output**: The function does not return a value but populates the provided cache array with computed values based on the input parameters.
- **Functions called**:
    - [`rope_yarn`](#rope_yarn)


---
### ggml\_compute\_forward\_rope\_f32<!-- {{#callable:ggml_compute_forward_rope_f32}} -->
Computes the forward rotary position embedding for tensors using specified parameters.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing computation parameters such as thread index and number of threads.
    - `dst`: A pointer to the destination `ggml_tensor` where the computed results will be stored.
    - `forward`: A boolean indicating the direction of the rotation; true for forward rotation and false for backward.
- **Control Flow**:
    - Extracts source tensors from the destination tensor.
    - Initializes various parameters from the operation parameters of the destination tensor.
    - Validates the dimensions and types of the tensors involved.
    - Calculates the number of rows to process per thread and determines the range of rows for the current thread.
    - Computes the frequency factors and initializes the cache based on the specified mode (e.g., NEOX, MROPE, VISION).
    - Iterates over the batch and sequence length dimensions to apply the rotary position embedding transformation.
    - Handles different modes of operation to apply the appropriate transformation logic based on the tensor types and dimensions.
- **Output**: The function modifies the `dst` tensor in place, storing the computed rotary position embeddings based on the input tensors and parameters.
- **Functions called**:
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_rope_cache_init`](#ggml_rope_cache_init)
    - [`ggml_mrope_cache_init`](#ggml_mrope_cache_init)


---
### ggml\_compute\_forward\_rope\_f16<!-- {{#callable:ggml_compute_forward_rope_f16}} -->
The `ggml_compute_forward_rope_f16` function computes the forward rotation of tensor data using rotary positional encoding for a specified number of dimensions and context.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for computation, including thread index and total number of threads.
    - `dst`: A pointer to the destination `ggml_tensor` where the computed results will be stored.
    - `forward`: A boolean indicating the direction of the rotation; true for forward rotation and false for backward rotation.
- **Control Flow**:
    - The function begins by extracting source tensors and parameters from the destination tensor's operation parameters.
    - It initializes various constants and checks for validity of dimensions and tensor types.
    - The function calculates the number of rows to process per thread and determines the range of rows for the current thread.
    - It computes frequency scaling and initializes cache based on the specified mode (e.g., NEOX, MROPE, or VISION).
    - The main computation loop iterates over the tensor dimensions, applying the appropriate rotation transformations based on the mode and the forward flag.
    - Results are stored in the destination tensor, with special handling for different modes affecting how data is processed.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the computed forward rotated values based on the input tensors and parameters.
- **Functions called**:
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_rope_cache_init`](#ggml_rope_cache_init)
    - [`ggml_mrope_cache_init`](#ggml_mrope_cache_init)


---
### ggml\_compute\_forward\_rope<!-- {{#callable:ggml_compute_forward_rope}} -->
The `ggml_compute_forward_rope` function computes the forward pass of a rope operation based on the type of the source tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` that will hold the result of the computation.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of `src0` using a switch statement.
    - If `src0` is of type `GGML_TYPE_F16`, it calls [`ggml_compute_forward_rope_f16`](#ggml_compute_forward_rope_f16) with the provided parameters and destination tensor.
    - If `src0` is of type `GGML_TYPE_F32`, it calls [`ggml_compute_forward_rope_f32`](#ggml_compute_forward_rope_f32) similarly.
    - If `src0` is of an unsupported type, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place based on the computations performed.
- **Functions called**:
    - [`ggml_compute_forward_rope_f16`](#ggml_compute_forward_rope_f16)
    - [`ggml_compute_forward_rope_f32`](#ggml_compute_forward_rope_f32)


---
### ggml\_compute\_forward\_rope\_back<!-- {{#callable:ggml_compute_forward_rope_back}} -->
The `ggml_compute_forward_rope_back` function computes the forward pass of a tensor operation based on the type of the source tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` that represents the destination tensor where the result will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of the source tensor `src0` using a switch statement.
    - If the type is `GGML_TYPE_F16`, it calls [`ggml_compute_forward_rope_f16`](#ggml_compute_forward_rope_f16) with the provided parameters and destination tensor.
    - If the type is `GGML_TYPE_F32`, it calls [`ggml_compute_forward_rope_f32`](#ggml_compute_forward_rope_f32) with the same arguments.
    - If the type does not match either case, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the destination tensor `dst` based on the computation performed.
- **Functions called**:
    - [`ggml_compute_forward_rope_f16`](#ggml_compute_forward_rope_f16)
    - [`ggml_compute_forward_rope_f32`](#ggml_compute_forward_rope_f32)


---
### ggml\_compute\_forward\_conv\_transpose\_1d\_f16\_f32<!-- {{#callable:ggml_compute_forward_conv_transpose_1d_f16_f32}} -->
Performs a 1D transposed convolution operation using half-precision and single-precision floating-point tensors.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for the computation, including thread information and workspace data.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the result of the transposed convolution operation.
- **Control Flow**:
    - The function begins by asserting the types of the input tensors to ensure they are correct.
    - If the current thread index is zero, it initializes the workspace data and permutes the kernel and source data to the required format.
    - It then sets up a barrier to synchronize threads before proceeding with the main computation.
    - The function calculates the range of rows to process for the current thread and iterates over these rows to perform the convolution operation.
    - For each output position, it computes the dot product of the kernel and the corresponding input data, accumulating the results in the destination tensor.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the results of the transposed convolution operation.
- **Functions called**:
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)
    - [`ggml_barrier`](ggml-cpu.c.driver.md#ggml_barrier)
    - [`ggml_vec_dot_f16`](vec.cpp.driver.md#ggml_vec_dot_f16)


---
### ggml\_compute\_forward\_conv\_transpose\_1d\_f32<!-- {{#callable:ggml_compute_forward_conv_transpose_1d_f32}} -->
Computes the forward transposed convolution for 1D tensors using float32 data.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for computation, including thread information and workspace data.
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the convolution will be stored.
- **Control Flow**:
    - The function begins by asserting that the types of the source tensors (`src0` and `src1`) and the destination tensor (`dst`) are all `GGML_TYPE_F32`.
    - If the current thread index (`ith`) is 0, it initializes the workspace data (`params->wdata`) to zero and prepares the kernel data from `src0` and source data from `src1`.
    - The kernel data is rearranged from a shape of (K x Cout x Cin) to (Cin x K x Cout) and stored in the workspace.
    - The source data from `src1` is also rearranged and stored in the workspace.
    - The destination tensor's data is zeroed out to prepare for accumulation.
    - A barrier is used to synchronize threads before proceeding with the main computation.
    - The total number of rows in the destination tensor is calculated, and the rows are divided among the available threads.
    - Each thread processes its assigned rows, performing a dot product operation for each element in the destination tensor using the prepared kernel and source data.
- **Output**: The function does not return a value but modifies the `dst` tensor in place to contain the result of the transposed convolution operation.
- **Functions called**:
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)
    - [`ggml_barrier`](ggml-cpu.c.driver.md#ggml_barrier)
    - [`ggml_vec_dot_f32`](vec.cpp.driver.md#ggml_vec_dot_f32)


---
### ggml\_compute\_forward\_conv\_transpose\_1d<!-- {{#callable:ggml_compute_forward_conv_transpose_1d}} -->
The `ggml_compute_forward_conv_transpose_1d` function performs a 1D transposed convolution operation on a tensor based on its data type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the transposed convolution will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the data type of `src0` using a switch statement.
    - If the type is `GGML_TYPE_F16`, it calls [`ggml_compute_forward_conv_transpose_1d_f16_f32`](#ggml_compute_forward_conv_transpose_1d_f16_f32) to perform the computation.
    - If the type is `GGML_TYPE_F32`, it calls [`ggml_compute_forward_conv_transpose_1d_f32`](#ggml_compute_forward_conv_transpose_1d_f32) for the computation.
    - If the type is neither `GGML_TYPE_F16` nor `GGML_TYPE_F32`, it triggers an abort with a fatal error message.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the results of the transposed convolution operation.
- **Functions called**:
    - [`ggml_compute_forward_conv_transpose_1d_f16_f32`](#ggml_compute_forward_conv_transpose_1d_f16_f32)
    - [`ggml_compute_forward_conv_transpose_1d_f32`](#ggml_compute_forward_conv_transpose_1d_f32)


---
### ggml\_compute\_forward\_im2col\_f32<!-- {{#callable:ggml_compute_forward_im2col_f32}} -->
Computes the im2col transformation for a 2D or 1D tensor, rearranging input data into a column format suitable for convolution operations.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for computation, including the thread index (`ith`) and the total number of threads (`nth`).
    - `dst`: A pointer to `ggml_tensor` that serves as the destination tensor where the im2col output will be stored.
- **Control Flow**:
    - The function begins by asserting that the source tensors are of type `GGML_TYPE_F32`.
    - It retrieves operation parameters from `dst->op_params`, including strides, paddings, and dimensions for the im2col transformation.
    - The function determines whether the operation is 2D or 1D based on the parameters.
    - It initializes the output tensor dimensions based on the input tensor dimensions and the specified parameters.
    - A nested loop structure iterates over the input tensor dimensions, processing each input element and mapping it to the output tensor format.
    - For each position in the output tensor, it calculates the corresponding input indices, applying padding and stride adjustments.
    - If the calculated input indices are out of bounds, it assigns a value of 0; otherwise, it copies the corresponding input value to the output tensor.
- **Output**: The function does not return a value; instead, it populates the `dst` tensor with the rearranged data in im2col format.


---
### ggml\_compute\_forward\_im2col\_f16<!-- {{#callable:ggml_compute_forward_im2col_f16}} -->
Computes the im2col transformation for a 2D or 1D input tensor, converting it into a column format suitable for convolution operations.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for computation, including the thread index and total number of threads.
    - `dst`: A pointer to the destination `ggml_tensor` where the output of the im2col operation will be stored.
- **Control Flow**:
    - The function begins by asserting the types of the source tensors to ensure they are correct (F16 for `src0` and F32 for `src1`).
    - It retrieves operation parameters from the `dst` tensor, including stride, padding, and dimensions for the im2col transformation.
    - The function determines whether the operation is 2D or 1D based on the parameters and sets the corresponding dimensions.
    - It initializes a nested loop structure to iterate over the input batches, output height, output width, and input channels, processing each element accordingly.
    - Within the innermost loop, it calculates the source indices and checks for boundary conditions, assigning values to the destination tensor or setting them to zero if out of bounds.
- **Output**: The function does not return a value; instead, it populates the `dst` tensor with the transformed data in im2col format.


---
### ggml\_compute\_forward\_im2col<!-- {{#callable:ggml_compute_forward_im2col}} -->
The `ggml_compute_forward_im2col` function computes the im2col transformation for a given tensor based on its data type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor to which the im2col transformation will be applied.
- **Control Flow**:
    - The function begins by checking the type of the `dst` tensor using a switch statement.
    - If the type is `GGML_TYPE_F16`, it calls the [`ggml_compute_forward_im2col_f16`](#ggml_compute_forward_im2col_f16) function to perform the transformation.
    - If the type is `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_im2col_f32`](#ggml_compute_forward_im2col_f32) function instead.
    - If the type does not match either case, it triggers an abort with a fatal error message.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place based on the im2col transformation.
- **Functions called**:
    - [`ggml_compute_forward_im2col_f16`](#ggml_compute_forward_im2col_f16)
    - [`ggml_compute_forward_im2col_f32`](#ggml_compute_forward_im2col_f32)


---
### ggml\_compute\_forward\_im2col\_back\_f32<!-- {{#callable:ggml_compute_forward_im2col_back_f32}} -->
Computes the backward pass of the im2col operation for a convolution layer, aggregating gradients from the forward pass.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for computation, including thread indices.
    - `dst`: A pointer to `ggml_tensor` structure where the computed gradients will be stored.
- **Control Flow**:
    - Assert that the types of the input tensors (`src0`, `src1`) and the destination tensor (`dst`) are all `GGML_TYPE_F32`.
    - Extract parameters from `dst->op_params` which include stride, padding, and dimensions for the convolution operation.
    - Determine the number of input channels, height, and width based on whether the operation is 2D or not.
    - Iterate over each input sample, input channel, height, and width to compute the gradients.
    - For each position in the output, check if the corresponding input position is valid based on the stride and padding.
    - Accumulate gradients from the forward pass into the destination tensor.
- **Output**: The function outputs the computed gradients in the `dst` tensor, which corresponds to the backward pass of the im2col operation.


---
### ggml\_compute\_forward\_conv\_transpose\_2d<!-- {{#callable:ggml_compute_forward_conv_transpose_2d}} -->
Computes the forward 2D transposed convolution operation.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure containing computation parameters including thread information and workspace.
    - `dst`: Pointer to a `ggml_tensor` structure that will hold the output of the convolution operation.
- **Control Flow**:
    - The function begins by asserting the types of the input tensors to ensure they are correct.
    - If the current thread index (`ith`) is 0, it initializes the workspace and permutes the kernel and source data to the required format.
    - It calculates the number of patches in the output tensor and determines the range of patches to process for the current thread.
    - The main computation loop iterates over the patches assigned to the current thread, performing the convolution operation using the permuted data.
- **Output**: The output is stored in the `dst` tensor, which contains the result of the transposed convolution operation.
- **Functions called**:
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)
    - [`ggml_barrier`](ggml-cpu.c.driver.md#ggml_barrier)
    - [`ggml_get_op_params_i32`](../ggml-impl.h.driver.md#ggml_get_op_params_i32)
    - [`ggml_vec_dot_f16`](vec.cpp.driver.md#ggml_vec_dot_f16)


---
### ggml\_compute\_forward\_conv\_2d\_dw\_cwhn<!-- {{#callable:ggml_compute_forward_conv_2d_dw_cwhn}} -->
Computes the forward pass of a 2D depthwise convolution operation.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure containing computation parameters such as the number of threads.
    - `src`: Pointer to a `ggml_tensor` structure representing the input tensor.
    - `kernel`: Pointer to a `ggml_tensor` structure representing the convolution kernel.
    - `dst`: Pointer to a `ggml_tensor` structure where the output will be stored.
    - `p`: Reference to a `ggml_conv_2d_dw_params` structure containing parameters specific to the 2D convolution operation.
- **Control Flow**:
    - Calculates the total number of output rows and determines the range of rows to process for the current thread.
    - Iterates over the output rows assigned to the current thread.
    - For each output position, calculates the corresponding source position based on the convolution parameters.
    - If SIMD is enabled, processes the convolution using vectorized operations for the channels in packages.
    - If SIMD is not enabled, processes the convolution using scalar operations for the remaining channels.
- **Output**: The function populates the `dst` tensor with the results of the 2D depthwise convolution operation.


---
### ggml\_compute\_forward\_conv\_2d\_dw\_whcn<!-- {{#callable:ggml_compute_forward_conv_2d_dw_whcn}} -->
Computes the forward pass of a 2D depthwise convolution operation.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure containing parameters for computation, including thread information.
    - `src`: Pointer to a `ggml_tensor` structure representing the input tensor for the convolution.
    - `kernel`: Pointer to a `ggml_tensor` structure representing the convolution kernel (filter) tensor.
    - `dst`: Pointer to a `ggml_tensor` structure where the output of the convolution will be stored.
    - `p`: Reference to a `ggml_conv_2d_dw_params` structure containing parameters specific to the 2D depthwise convolution, such as dimensions and strides.
- **Control Flow**:
    - Calculates the total number of channels and batches to process based on the parameters provided.
    - Determines the range of indices for the current thread to process using the `ith` and `nth` values from `params`.
    - Iterates over each channel and batch index assigned to the current thread.
    - For each output pixel in the destination tensor, initializes a sum accumulator for the convolution result.
    - Nested loops iterate over the kernel dimensions, calculating the corresponding source pixel indices while applying padding and stride.
    - Checks for valid source pixel indices to avoid out-of-bounds access, accumulating the convolution result into the sum.
    - Stores the computed sum into the appropriate position in the destination tensor.
- **Output**: The function does not return a value; instead, it populates the `dst` tensor with the results of the 2D depthwise convolution operation.


---
### ggml\_compute\_forward\_conv\_2d\_dw<!-- {{#callable:ggml_compute_forward_conv_2d_dw}} -->
Computes the forward pass of a 2D depthwise convolution operation.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for computation.
    - `dst`: A pointer to `ggml_tensor` that will hold the output of the convolution operation.
- **Control Flow**:
    - Extracts the `kernel` and `src` tensors from the `dst` tensor's source array.
    - Initializes a `ggml_conv_2d_dw_params` structure with dimensions and parameters derived from the `src` and `kernel` tensors.
    - Validates that the number of channels in the `kernel` matches the channels in the `src` tensor and that the batch size in `dst` matches the batch size in `src`.
    - Checks if the `src` tensor is contiguous in memory; if so, calls [`ggml_compute_forward_conv_2d_dw_whcn`](#ggml_compute_forward_conv_2d_dw_whcn) to perform the convolution.
    - If the `src` tensor is contiguous in channels, it asserts the memory layout of the `kernel` and calls [`ggml_compute_forward_conv_2d_dw_cwhn`](#ggml_compute_forward_conv_2d_dw_cwhn).
    - If neither condition is met, it aborts the operation with an error message.
- **Output**: The function does not return a value but populates the `dst` tensor with the result of the convolution operation.
- **Functions called**:
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_compute_forward_conv_2d_dw_whcn`](#ggml_compute_forward_conv_2d_dw_whcn)
    - [`ggml_is_contiguous_channels`](../ggml.c.driver.md#ggml_is_contiguous_channels)
    - [`ggml_compute_forward_conv_2d_dw_cwhn`](#ggml_compute_forward_conv_2d_dw_cwhn)


---
### ggml\_compute\_forward\_pool\_1d\_sk\_p0<!-- {{#callable:ggml_compute_forward_pool_1d_sk_p0}} -->
Computes the forward pooling operation (average or max) for a 1D tensor based on the specified parameters.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing computation parameters, including the index of the current operation.
    - `op`: An enumeration value of type `ggml_op_pool` that specifies the pooling operation to perform (average, max, or count).
    - `k`: An integer representing the size of the pooling window.
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the pooling operation will be stored.
- **Control Flow**:
    - The function first checks if the current operation index (`params->ith`) is not zero; if so, it returns immediately.
    - It retrieves the source tensor from the destination tensor and asserts that its type is either `GGML_TYPE_F32` or `GGML_TYPE_F16`.
    - The function initializes pointers for the source data and destination data, and calculates the number of rows in the destination tensor.
    - It enters a loop that continues until all source data has been processed, iterating over each row of the source tensor.
    - For each row, it initializes the destination value based on the pooling operation type (average or max).
    - It then enters a nested loop to process each element in the pooling window, updating the destination value according to the specified pooling operation.
    - After processing all elements in the pooling window, it finalizes the destination value for the current row (dividing by `k` for average).
    - Finally, it advances the source and destination pointers to the next row.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the results of the pooling operation.
- **Functions called**:
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)


---
### ggml\_compute\_forward\_pool\_1d<!-- {{#callable:ggml_compute_forward_pool_1d}} -->
Computes the forward pass of a 1D pooling operation on a tensor.
- **Inputs**:
    - `params`: Pointer to a structure containing computation parameters for the operation.
    - `dst`: Pointer to the destination `ggml_tensor` where the result of the pooling operation will be stored.
- **Control Flow**:
    - Extracts operation parameters from the `dst` tensor's `op_params` array.
    - Validates that padding is not supported by asserting that `p0` equals 0.
    - Validates that the stride equals the kernel size by asserting that `k0` equals `s0`.
    - Calls the helper function [`ggml_compute_forward_pool_1d_sk_p0`](#ggml_compute_forward_pool_1d_sk_p0) to perform the actual pooling computation.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the results of the pooling operation.
- **Functions called**:
    - [`ggml_compute_forward_pool_1d_sk_p0`](#ggml_compute_forward_pool_1d_sk_p0)


---
### ggml\_compute\_forward\_pool\_2d<!-- {{#callable:ggml_compute_forward_pool_2d}} -->
Computes the forward 2D pooling operation (average or max) on a source tensor and stores the result in a destination tensor.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure containing computation parameters.
    - `dst`: Pointer to a `ggml_tensor` structure where the result of the pooling operation will be stored.
- **Control Flow**:
    - Checks if the source tensor type is either `GGML_TYPE_F32` or `GGML_TYPE_F16` and asserts this condition.
    - Returns early if the computation index (`params->ith`) is not zero.
    - Extracts pooling operation type and parameters (kernel size, stride, padding) from the destination tensor's operation parameters.
    - Iterates over the source tensor data in chunks, processing each 2D slice based on the specified pooling operation (average or max).
    - For each output position, initializes the output value based on the pooling operation type.
    - Calculates the corresponding input indices and accumulates values from the source tensor based on the pooling operation.
    - Finalizes the output value by averaging for average pooling or leaving it as is for max pooling.
- **Output**: The function does not return a value but modifies the `dst` tensor in place to contain the results of the pooling operation.
- **Functions called**:
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)


---
### ggml\_compute\_forward\_pool\_2d\_back<!-- {{#callable:ggml_compute_forward_pool_2d_back}} -->
Computes the backward pass of a 2D pooling operation, updating gradients based on the specified pooling type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing computation parameters, including the index of the current operation.
    - `dst`: A pointer to a `ggml_tensor` structure representing the destination tensor where gradients will be accumulated.
- **Control Flow**:
    - The function first checks if the current operation index is zero; if not, it returns immediately.
    - It retrieves pooling operation parameters from the `dst` tensor's operation parameters.
    - The function initializes the destination tensor's data to zero.
    - It iterates over the destination tensor's data in chunks, processing each pixel based on the pooling operation type (max or average).
    - For max pooling, it finds the maximum value and its position in the specified kernel area and updates the gradient accordingly.
    - For average pooling, it distributes the gradient evenly across the kernel area.
    - The function continues processing until all data in the destination tensor is handled.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to reflect the computed gradients from the backward pooling operation.
- **Functions called**:
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)


---
### ggml\_compute\_forward\_upscale\_f32<!-- {{#callable:ggml_compute_forward_upscale_f32}} -->
The `ggml_compute_forward_upscale_f32` function performs upscaling of a tensor using either nearest neighbor or bilinear interpolation based on the specified scaling mode.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for the computation, including the current thread index (`ith`) and the total number of threads (`nth`).
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the output of the upscaling operation.
- **Control Flow**:
    - The function begins by asserting that the source tensor (`src0`) is of type `GGML_TYPE_F32`.
    - It retrieves the scaling factors for each dimension based on the source tensor's shape.
    - The scaling mode is determined from the operation parameters.
    - If the scaling mode is `GGML_SCALE_MODE_NEAREST`, it iterates through the dimensions of the destination tensor, calculating the corresponding indices in the source tensor and copying values directly.
    - If the scaling mode is `GGML_SCALE_MODE_BILINEAR`, it calculates the interpolated values by fetching surrounding pixel values and applying bilinear interpolation.
    - If the scaling mode is unsupported, the function aborts with an error message.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the upscaled values based on the specified scaling mode.
- **Functions called**:
    - [`ggml_get_op_params_i32`](../ggml-impl.h.driver.md#ggml_get_op_params_i32)


---
### ggml\_compute\_forward\_upscale<!-- {{#callable:ggml_compute_forward_upscale}} -->
The `ggml_compute_forward_upscale` function computes the upscale operation for a tensor based on its type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the upscale operation will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of the source tensor `src0` using a switch statement.
    - If the type is `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_upscale_f32`](#ggml_compute_forward_upscale_f32) function to perform the upscale operation.
    - If the type is not recognized, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the result of the upscale operation.
- **Functions called**:
    - [`ggml_compute_forward_upscale_f32`](#ggml_compute_forward_upscale_f32)


---
### ggml\_compute\_forward\_pad\_f32<!-- {{#callable:ggml_compute_forward_pad_f32}} -->
The `ggml_compute_forward_pad_f32` function performs a padded copy of data from a source tensor to a destination tensor, filling in zeros for out-of-bounds indices.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for the computation, including the current thread index (`ith`) and the total number of threads (`nth`).
    - `dst`: A pointer to a `ggml_tensor` structure representing the destination tensor where the padded data will be stored.
- **Control Flow**:
    - The function begins by asserting that the data types of the source and destination tensors are both `float`.
    - It retrieves the source tensor from the destination tensor's source array.
    - The function initializes local variables for the destination data pointer and tensor dimensions.
    - It then enters a nested loop structure to iterate over the dimensions of the tensors, with the outermost loop iterating over the second dimension (`i2`), the middle loop iterating over the first dimension (`i1`), and the innermost loops iterating over the remaining dimensions (`i0` and `i3`).
    - For each index combination, it calculates the destination index and retrieves the corresponding source pointer.
    - If the current indices are within the bounds of the source tensor dimensions, it copies the value from the source to the destination; otherwise, it sets the destination value to zero.
- **Output**: The function does not return a value; instead, it modifies the destination tensor in place by filling it with either the copied values from the source tensor or zeros for out-of-bounds indices.


---
### ggml\_compute\_forward\_pad<!-- {{#callable:ggml_compute_forward_pad}} -->
The `ggml_compute_forward_pad` function computes the forward padding operation for a tensor based on its type.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of the source tensor `src0` using a switch statement.
    - If the type is `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_pad_f32`](#ggml_compute_forward_pad_f32) function to perform the padding operation.
    - If the type is not recognized, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the destination tensor `dst` in place based on the padding operation performed.
- **Functions called**:
    - [`ggml_compute_forward_pad_f32`](#ggml_compute_forward_pad_f32)


---
### ggml\_compute\_forward\_pad\_reflect\_1d<!-- {{#callable:ggml_compute_forward_pad_reflect_1d}} -->
Computes the forward padding of a 1D tensor using reflection based on specified padding parameters.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing computation parameters such as the current thread index (`ith`) and the total number of threads (`nth`).
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the padding operation will be stored.
- **Control Flow**:
    - The function begins by asserting that the source tensor (`src0`) and the destination tensor (`dst`) are of type `GGML_TYPE_F32`.
    - It retrieves the padding parameters `p0` and `p1` from the destination tensor's operation parameters.
    - The function then iterates over the dimensions of the tensor, specifically the last three dimensions (`i3`, `i2`, and `i1`), processing the data in parallel based on the thread index.
    - For each position in the destination tensor, it calculates the left and right padding positions based on the specified padding parameters.
    - It copies the data from the source tensor to the left position in the destination tensor.
    - Finally, it fills the left and right padding areas by reflecting the values from the edges of the tensor.
- **Output**: The function modifies the `dst` tensor in place, filling it with the reflected values based on the specified padding parameters, effectively extending the tensor with mirrored data.
- **Functions called**:
    - [`ggml_vec_cpy_f32`](vec.h.driver.md#ggml_vec_cpy_f32)


---
### ggml\_compute\_forward\_arange\_f32<!-- {{#callable:ggml_compute_forward_arange_f32}} -->
Computes a range of float values from a start to a stop value with a specified step and stores them in a destination tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing computation parameters, including the current thread index (`ith`) and the total number of threads (`nth`).
    - `dst`: A pointer to a `ggml_tensor` structure where the computed range of float values will be stored.
- **Control Flow**:
    - Asserts that the size of the destination tensor's first dimension is equal to the size of a float.
    - Retrieves the start, stop, and step values from the operation parameters of the destination tensor.
    - Calculates the total number of steps required to generate the range of values based on the start, stop, and step values.
    - Asserts that the number of elements in the destination tensor matches the calculated number of steps.
    - Iterates over the range of steps, incrementing by the number of threads, and computes the value for each step, storing it in the destination tensor.
- **Output**: The function does not return a value; instead, it populates the `dst` tensor with computed float values representing the range from start to stop, incremented by step.
- **Functions called**:
    - [`ggml_get_op_params_f32`](../ggml-impl.h.driver.md#ggml_get_op_params_f32)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)


---
### ggml\_compute\_forward\_arange<!-- {{#callable:ggml_compute_forward_arange}} -->
The `ggml_compute_forward_arange` function computes the forward arange operation for a tensor based on its type.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result will be stored.
- **Control Flow**:
    - The function begins by checking the type of the `dst` tensor.
    - If the type is `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_arange_f32`](#ggml_compute_forward_arange_f32) function to perform the computation.
    - If the type is not recognized, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place based on the computation performed.
- **Functions called**:
    - [`ggml_compute_forward_arange_f32`](#ggml_compute_forward_arange_f32)


---
### ggml\_compute\_forward\_timestep\_embedding\_f32<!-- {{#callable:ggml_compute_forward_timestep_embedding_f32}} -->
Computes the forward timestep embedding for a tensor using cosine and sine functions.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing computation parameters such as the current index (`ith`) and the total number of threads (`nth`).
    - `dst`: A pointer to a `ggml_tensor` structure where the computed timestep embeddings will be stored.
- **Control Flow**:
    - The function begins by asserting that the first source tensor (`src0`) has a data type of float.
    - It retrieves the current index (`ith`) and the total number of threads (`nth`) from the `params` structure.
    - It initializes local variables for tensor operations and retrieves the embedding dimension (`dim`) and maximum period (`max_period`) from the destination tensor's parameters.
    - The function calculates half of the dimension (`half`) to separate the cosine and sine components.
    - A loop iterates over the number of elements in the destination tensor, processing each element based on the current thread index.
    - Within the loop, another loop computes the cosine and sine values for each timestep using the frequency derived from the maximum period and assigns them to the appropriate positions in the destination tensor.
    - If the dimension is odd and the current thread is the first one, it sets the last element of the embedding to zero.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to store the computed timestep embeddings.
- **Functions called**:
    - [`ggml_get_op_params_i32`](../ggml-impl.h.driver.md#ggml_get_op_params_i32)


---
### ggml\_compute\_forward\_timestep\_embedding<!-- {{#callable:ggml_compute_forward_timestep_embedding}} -->
Computes the forward timestep embedding for a given tensor based on its type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` where the result of the computation will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of `src0` using a switch statement.
    - If `src0` is of type `GGML_TYPE_F32`, it calls the function [`ggml_compute_forward_timestep_embedding_f32`](#ggml_compute_forward_timestep_embedding_f32) to perform the computation.
    - If `src0` is of any other type, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the computed timestep embedding.
- **Functions called**:
    - [`ggml_compute_forward_timestep_embedding_f32`](#ggml_compute_forward_timestep_embedding_f32)


---
### ggml\_compute\_forward\_argsort\_f32<!-- {{#callable:ggml_compute_forward_argsort_f32}} -->
Computes the sorted order of elements in a tensor based on specified sorting order.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for computation, including the current thread index (`ith`) and the total number of threads (`nth`).
    - `dst`: A pointer to a `ggml_tensor` structure where the sorted indices will be stored.
- **Control Flow**:
    - The function retrieves the source tensor from the destination tensor's source array.
    - It asserts that the size of the data type is correct (float).
    - It retrieves the number of rows in the source tensor and the sorting order from the operation parameters.
    - A loop iterates over the rows of the source tensor, processing each row based on the thread index and total number of threads.
    - For each row, it initializes an array of indices corresponding to the elements in that row.
    - A nested loop implements a bubble sort algorithm to sort the indices based on the specified order (ascending or descending) using the values from the source tensor.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the sorted indices of the elements from the source tensor.
- **Functions called**:
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_get_op_params_i32`](../ggml-impl.h.driver.md#ggml_get_op_params_i32)


---
### ggml\_compute\_forward\_argsort<!-- {{#callable:ggml_compute_forward_argsort}} -->
The `ggml_compute_forward_argsort` function computes the forward argsort operation for a tensor based on its data type.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the result of the argsort operation.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the data type of `src0` using a switch statement.
    - If the data type is `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_argsort_f32`](#ggml_compute_forward_argsort_f32) function to perform the argsort operation.
    - If the data type is not recognized, it triggers an abort with a fatal error message.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor to contain the results of the argsort operation if the input tensor is of type `GGML_TYPE_F32`.
- **Functions called**:
    - [`ggml_compute_forward_argsort_f32`](#ggml_compute_forward_argsort_f32)


---
### ggml\_compute\_forward\_flash\_attn\_ext\_f16<!-- {{#callable:ggml_compute_forward_flash_attn_ext_f16}} -->
Computes the forward pass of a flash attention mechanism using half-precision floating point tensors.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for computation, including thread indices.
    - `q`: A pointer to the query tensor (`ggml_tensor`) used in the attention mechanism.
    - `k`: A pointer to the key tensor (`ggml_tensor`) used in the attention mechanism.
    - `v`: A pointer to the value tensor (`ggml_tensor`) used in the attention mechanism.
    - `mask`: A pointer to an optional mask tensor (`ggml_tensor`) that can be applied to the attention scores.
    - `dst`: A pointer to the destination tensor (`ggml_tensor`) where the output of the attention computation will be stored.
- **Control Flow**:
    - Initializes local variables for tensor dimensions and sizes using `GGML_TENSOR_LOCALS` macros.
    - Validates tensor properties and dimensions using assertions to ensure compatibility.
    - Calculates broadcasting factors for the tensors based on their dimensions.
    - Determines the number of rows in the query tensor and calculates the range of rows to process for the current thread.
    - Extracts parameters such as scale, max bias, and logit softcap from the destination tensor's operation parameters.
    - Loops over the rows of the query tensor, performing vector dot products with the key tensor and applying softmax attention logic.
    - Handles both half-precision and single-precision floating point operations based on the type of the value tensor.
    - Accumulates results and normalizes the output before storing it in the destination tensor.
- **Output**: The function outputs the computed attention values into the destination tensor (`dst`), which contains the results of the attention mechanism applied to the input tensors.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_get_type_traits_cpu`](ggml-cpu.c.driver.md#ggml_get_type_traits_cpu)
    - [`ggml_get_type_traits`](../ggml.c.driver.md#ggml_get_type_traits)
    - [`ggml_vec_scale_f16`](vec.h.driver.md#ggml_vec_scale_f16)
    - [`ggml_vec_mad_f16`](vec.h.driver.md#ggml_vec_mad_f16)
    - [`ggml_vec_scale_f32`](vec.h.driver.md#ggml_vec_scale_f32)
    - [`ggml_vec_mad_f32`](vec.h.driver.md#ggml_vec_mad_f32)


---
### ggml\_compute\_forward\_flash\_attn\_ext<!-- {{#callable:ggml_compute_forward_flash_attn_ext}} -->
Computes the forward pass of the flash attention mechanism with support for different precision types.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure containing parameters for computation.
    - `q`: Pointer to a `ggml_tensor` representing the query tensor.
    - `k`: Pointer to a `ggml_tensor` representing the key tensor.
    - `v`: Pointer to a `ggml_tensor` representing the value tensor.
    - `mask`: Pointer to a `ggml_tensor` representing the attention mask.
    - `dst`: Pointer to a `ggml_tensor` where the result of the computation will be stored.
- **Control Flow**:
    - The function begins by checking the precision type specified in the `dst` tensor's operation parameters.
    - If the precision is either `GGML_PREC_DEFAULT` or `GGML_PREC_F32`, it calls the [`ggml_compute_forward_flash_attn_ext_f16`](#ggml_compute_forward_flash_attn_ext_f16) function to perform the computation using F32 accumulators.
    - If the precision type is not recognized, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the computed attention results.
- **Functions called**:
    - [`ggml_compute_forward_flash_attn_ext_f16`](#ggml_compute_forward_flash_attn_ext_f16)


---
### ggml\_compute\_forward\_flash\_attn\_back\_f32<!-- {{#callable:ggml_compute_forward_flash_attn_back_f32}} -->
Computes the forward pass of the flash attention mechanism with backpropagation for gradient computation.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for computation, including thread information.
    - `masked`: A boolean indicating whether the attention mechanism should be masked.
    - `dst`: A pointer to `ggml_tensor` where the output gradients will be stored.
- **Control Flow**:
    - Extracts source tensors `q`, `k`, `v`, and `d` from the destination tensor `dst`.
    - Initializes local variables for tensor dimensions and sizes.
    - Performs assertions to validate tensor dimensions and sizes.
    - If the current thread index is 0, initializes the destination tensor's data to zero.
    - Calculates the number of elements for each tensor and prepares offsets for accessing tensor data.
    - Determines the number of rows in tensor `k` and calculates the range of rows to process for the current thread.
    - Iterates over the rows of `k`, performing dot products with `q` and applying scaling and softmax operations.
    - Computes gradients for `q`, `k`, and `v` based on the computed attention scores and the input gradients.
- **Output**: The function modifies the `dst` tensor in place to store the computed gradients for the attention mechanism.
- **Functions called**:
    - [`ggml_up`](../ggml-impl.h.driver.md#ggml_up)
    - [`ggml_barrier`](ggml-cpu.c.driver.md#ggml_barrier)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_blck_size`](../ggml.c.driver.md#ggml_blck_size)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_vec_dot_f32`](vec.cpp.driver.md#ggml_vec_dot_f32)
    - [`ggml_vec_scale_f32`](vec.h.driver.md#ggml_vec_scale_f32)
    - [`ggml_vec_max_f32`](vec.h.driver.md#ggml_vec_max_f32)
    - [`ggml_vec_sum_f32`](vec.h.driver.md#ggml_vec_sum_f32)
    - [`ggml_vec_soft_max_f32`](vec.cpp.driver.md#ggml_vec_soft_max_f32)
    - [`ggml_vec_set_f32`](vec.h.driver.md#ggml_vec_set_f32)
    - [`ggml_vec_mad_f32`](vec.h.driver.md#ggml_vec_mad_f32)
    - [`ggml_vec_acc1_f32`](vec.h.driver.md#ggml_vec_acc1_f32)
    - [`ggml_vec_mul_f32`](vec.h.driver.md#ggml_vec_mul_f32)


---
### ggml\_compute\_forward\_flash\_attn\_back<!-- {{#callable:ggml_compute_forward_flash_attn_back}} -->
The `ggml_compute_forward_flash_attn_back` function computes the forward pass of the flash attention mechanism based on the input tensor type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `masked`: A boolean indicating whether the attention mechanism should be masked.
    - `dst`: A pointer to a `ggml_tensor` where the result of the computation will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `q` from the `dst` tensor's source array.
    - It checks the type of the tensor `q` using a switch statement.
    - If the type is `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_flash_attn_back_f32`](#ggml_compute_forward_flash_attn_back_f32) function to perform the computation.
    - If the type is not recognized, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the computed results.
- **Functions called**:
    - [`ggml_compute_forward_flash_attn_back_f32`](#ggml_compute_forward_flash_attn_back_f32)


---
### ggml\_compute\_forward\_ssm\_conv\_f32<!-- {{#callable:ggml_compute_forward_ssm_conv_f32}} -->
Computes the forward pass of a 1D convolution operation using floating-point precision.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure containing parameters for computation, including thread indices.
    - `dst`: Pointer to a `ggml_tensor` structure that will hold the output of the convolution operation.
- **Control Flow**:
    - Extracts source tensors `src0` and `src1` from the destination tensor `dst`.
    - Calculates the number of channels, sequence lengths, and other dimensions from the source tensors.
    - Validates the dimensions and data types of the tensors using assertions.
    - Determines the number of rows to process per thread and calculates the range of rows for the current thread.
    - Iterates over the sequences and tokens to perform the convolution operation.
    - For each row in the determined range, computes the dot product of the sliding window from `src0` and the weights from `src1`, storing the result in the output tensor `dst`.
- **Output**: The function does not return a value but populates the `dst` tensor with the results of the convolution operation.


---
### ggml\_compute\_forward\_ssm\_conv<!-- {{#callable:ggml_compute_forward_ssm_conv}} -->
The `ggml_compute_forward_ssm_conv` function computes the forward pass of a specific convolution operation based on the type of the source tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result will be stored.
- **Control Flow**:
    - The function begins by checking the type of the first source tensor in the `dst` tensor's `src` array.
    - If the type is `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_ssm_conv_f32`](#ggml_compute_forward_ssm_conv_f32) function to perform the computation.
    - If the type does not match any expected type, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place based on the computation performed.
- **Functions called**:
    - [`ggml_compute_forward_ssm_conv_f32`](#ggml_compute_forward_ssm_conv_f32)


---
### ggml\_compute\_forward\_ssm\_scan\_f32<!-- {{#callable:ggml_compute_forward_ssm_scan_f32}} -->
Computes the forward pass of a state-space model using a scan operation for floating-point tensors.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure containing parameters for computation, including the thread index (`ith`) and total number of threads (`nth`).
    - `dst`: Pointer to a `ggml_tensor` structure where the output of the computation will be stored.
- **Control Flow**:
    - Extracts source tensors from the `dst` tensor, which include state (`src0`), input (`src1`), time derivatives (`src2`), and matrices (`src3`, `src4`, `src5`).
    - Validates the dimensions and data types of the input tensors to ensure they are compatible for the operations.
    - Calculates the number of rows to process per thread and determines the range of rows for the current thread.
    - Iterates over sequences and tokens, performing computations based on the selected architecture (ARM SVE or non-SVE).
    - For each token, computes the softplus activation of the time derivative and updates the state using matrix multiplications and additions.
    - Stores the computed state and outputs the result for each inner dimension.
- **Output**: The function does not return a value but populates the `dst` tensor with the computed results of the forward pass, which includes updated states and outputs for each sequence and token.
- **Functions called**:
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`exp_ps_sve`](vec.h.driver.md#exp_ps_sve)


---
### ggml\_compute\_forward\_ssm\_scan<!-- {{#callable:ggml_compute_forward_ssm_scan}} -->
The `ggml_compute_forward_ssm_scan` function computes a forward scan operation on a tensor based on its type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` which is the destination tensor where the result will be stored.
- **Control Flow**:
    - The function begins by checking the type of the first source tensor in the `dst` tensor's source array.
    - If the type is `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_ssm_scan_f32`](#ggml_compute_forward_ssm_scan_f32) function to perform the computation.
    - If the type does not match, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place based on the computation performed.
- **Functions called**:
    - [`ggml_compute_forward_ssm_scan_f32`](#ggml_compute_forward_ssm_scan_f32)


---
### ggml\_compute\_forward\_win\_part\_f32<!-- {{#callable:ggml_compute_forward_win_part_f32}} -->
The `ggml_compute_forward_win_part_f32` function computes a forward pass for a windowed operation on a tensor, populating the destination tensor based on the source tensor and specified parameters.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params`, which contains parameters for the computation (unused in this function).
    - `dst`: A pointer to `ggml_tensor`, which is the destination tensor where the results of the computation will be stored.
- **Control Flow**:
    - The function retrieves the source tensor from the destination tensor's source array.
    - It extracts the necessary parameters from the destination tensor's operation parameters, including the dimensions for the windowed operation.
    - Assertions are made to ensure the dimensions of the source and destination tensors are consistent.
    - A nested loop structure iterates over the dimensions of the output tensor, calculating indices for both the source and destination tensors.
    - Within the innermost loop, it checks if the calculated indices exceed the bounds of the source tensor; if so, it sets the corresponding destination value to 0.0f, otherwise it copies the value from the source tensor.
- **Output**: The function does not return a value but modifies the `dst` tensor in place, filling it with computed values based on the `src0` tensor and the specified windowing parameters.


---
### ggml\_compute\_forward\_win\_part<!-- {{#callable:ggml_compute_forward_win_part}} -->
The `ggml_compute_forward_win_part` function computes a forward pass for a specific tensor operation based on the type of the source tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of the source tensor `src0` using a switch statement.
    - If the type is `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_win_part_f32`](#ggml_compute_forward_win_part_f32) function to perform the computation.
    - If the type does not match any expected case, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the destination tensor `dst` in place based on the computation performed.
- **Functions called**:
    - [`ggml_compute_forward_win_part_f32`](#ggml_compute_forward_win_part_f32)


---
### ggml\_compute\_forward\_win\_unpart\_f32<!-- {{#callable:ggml_compute_forward_win_unpart_f32}} -->
The `ggml_compute_forward_win_unpart_f32` function computes a forward pass for a tensor operation with windowing, copying data from a source tensor to a destination tensor while applying padding.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params`, which contains parameters for the computation (unused in this function).
    - `dst`: A pointer to `ggml_tensor`, which represents the destination tensor where the computed results will be stored.
- **Control Flow**:
    - The function retrieves the source tensor from the destination tensor's source array.
    - It calculates the necessary padding based on the window size and the dimensions of the source tensor.
    - It asserts that the number of elements in the source tensor matches the expected value.
    - A nested loop iterates over the dimensions of the destination tensor, calculating the appropriate indices for both the source and destination tensors.
    - The computed value from the source tensor is assigned to the corresponding position in the destination tensor.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the computed data based on the source tensor.


---
### ggml\_compute\_forward\_win\_unpart<!-- {{#callable:ggml_compute_forward_win_unpart}} -->
The `ggml_compute_forward_win_unpart` function computes a forward operation on a tensor based on its type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` that serves as the destination tensor for the computation.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of `src0` using a switch statement.
    - If `src0` is of type `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_win_unpart_f32`](#ggml_compute_forward_win_unpart_f32) function to perform the computation.
    - If `src0` is of any other type, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the destination tensor `dst` based on the computation performed.
- **Functions called**:
    - [`ggml_compute_forward_win_unpart_f32`](#ggml_compute_forward_win_unpart_f32)


---
### ggml\_compute\_forward\_unary<!-- {{#callable:ggml_compute_forward_unary}} -->
The `ggml_compute_forward_unary` function computes the result of a unary operation on a tensor based on the specified operation type.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` that represents the destination tensor where the result of the unary operation will be stored.
- **Control Flow**:
    - The function retrieves the unary operation type from the destination tensor `dst` using [`ggml_get_unary_op`](../ggml.c.driver.md#ggml_get_unary_op).
    - A switch statement is used to determine which unary operation to perform based on the retrieved operation type.
    - For each case in the switch statement, the corresponding unary operation function is called with `params` and `dst` as arguments.
    - If the operation type does not match any predefined unary operations, the function calls `GGML_ABORT` to indicate a fatal error.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the result of the unary operation.
- **Functions called**:
    - [`ggml_get_unary_op`](../ggml.c.driver.md#ggml_get_unary_op)
    - [`ggml_compute_forward_abs`](unary-ops.cpp.driver.md#ggml_compute_forward_abs)
    - [`ggml_compute_forward_sgn`](unary-ops.cpp.driver.md#ggml_compute_forward_sgn)
    - [`ggml_compute_forward_neg`](unary-ops.cpp.driver.md#ggml_compute_forward_neg)
    - [`ggml_compute_forward_step`](unary-ops.cpp.driver.md#ggml_compute_forward_step)
    - [`ggml_compute_forward_tanh`](unary-ops.cpp.driver.md#ggml_compute_forward_tanh)
    - [`ggml_compute_forward_elu`](unary-ops.cpp.driver.md#ggml_compute_forward_elu)
    - [`ggml_compute_forward_relu`](unary-ops.cpp.driver.md#ggml_compute_forward_relu)
    - [`ggml_compute_forward_sigmoid`](unary-ops.cpp.driver.md#ggml_compute_forward_sigmoid)
    - [`ggml_compute_forward_gelu`](#ggml_compute_forward_gelu)
    - [`ggml_compute_forward_gelu_erf`](#ggml_compute_forward_gelu_erf)
    - [`ggml_compute_forward_gelu_quick`](#ggml_compute_forward_gelu_quick)
    - [`ggml_compute_forward_silu`](#ggml_compute_forward_silu)
    - [`ggml_compute_forward_hardswish`](unary-ops.cpp.driver.md#ggml_compute_forward_hardswish)
    - [`ggml_compute_forward_hardsigmoid`](unary-ops.cpp.driver.md#ggml_compute_forward_hardsigmoid)
    - [`ggml_compute_forward_exp`](unary-ops.cpp.driver.md#ggml_compute_forward_exp)


---
### ggml\_compute\_forward\_get\_rel\_pos\_f16<!-- {{#callable:ggml_compute_forward_get_rel_pos_f16}} -->
This function computes a forward pass to get relative positional data in a tensor format.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params`, which contains parameters for the computation but is unused in this function.
    - `dst`: A pointer to `ggml_tensor` that serves as the destination tensor where the computed relative positional data will be stored.
- **Control Flow**:
    - The function begins by retrieving the source tensor `src0` from the destination tensor `dst`.
    - It initializes local variables for tensor operations and retrieves the width `w` from the tensor dimensions.
    - The function then casts the data pointers of the source and destination tensors to `ggml_fp16_t` type for half-precision floating-point operations.
    - A nested loop structure iterates over the dimensions of the tensor, where the outer loop iterates over the second dimension `ne2`, the middle loop iterates over the first dimension `ne1`, and the innermost loop iterates over the zeroth dimension `ne0`.
    - Within the innermost loop, the function calculates the position `pos` based on the current indices and the width, and assigns the corresponding value from the source tensor to the destination tensor.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the computed relative positional data based on the source tensor.


---
### ggml\_compute\_forward\_get\_rel\_pos<!-- {{#callable:ggml_compute_forward_get_rel_pos}} -->
The `ggml_compute_forward_get_rel_pos` function computes relative positions based on the type of the source tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` that will hold the result of the computation.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of `src0` using a switch statement.
    - If `src0` is of type `GGML_TYPE_F16` or `GGML_TYPE_BF16`, it calls the function [`ggml_compute_forward_get_rel_pos_f16`](#ggml_compute_forward_get_rel_pos_f16) with the provided parameters and destination tensor.
    - If `src0` is of any other type, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor based on the computation performed by [`ggml_compute_forward_get_rel_pos_f16`](#ggml_compute_forward_get_rel_pos_f16).
- **Functions called**:
    - [`ggml_compute_forward_get_rel_pos_f16`](#ggml_compute_forward_get_rel_pos_f16)


---
### ggml\_compute\_forward\_add\_rel\_pos\_f32<!-- {{#callable:ggml_compute_forward_add_rel_pos_f32}} -->
Computes the forward addition of relative positional encodings to a destination tensor using two source tensors.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for computation, including thread indices.
    - `dst`: A pointer to the destination `ggml_tensor` where the result will be stored.
- **Control Flow**:
    - Extracts source tensors `src0`, `src1`, and `src2` from the destination tensor `dst`.
    - Checks if the operation is in-place; if not and if it's the first thread, it copies data from `src0` to `dst`.
    - Synchronizes threads using [`ggml_barrier`](ggml-cpu.c.driver.md#ggml_barrier).
    - Calculates the number of patches and determines the range of patches to process for the current thread.
    - Iterates over the determined range of patches and dimensions, performing element-wise addition of values from `src1` and `src2` to `dst`.
- **Output**: The function modifies the `dst` tensor in place, adding the values from `src1` and `src2` based on their relative positions.
- **Functions called**:
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)
    - [`ggml_barrier`](ggml-cpu.c.driver.md#ggml_barrier)


---
### ggml\_compute\_forward\_add\_rel\_pos<!-- {{#callable:ggml_compute_forward_add_rel_pos}} -->
The `ggml_compute_forward_add_rel_pos` function computes the forward addition of relative positions for a given tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` that serves as the destination tensor for the computation.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of `src0` using a switch statement.
    - If `src0` is of type `GGML_TYPE_F32`, it calls the function [`ggml_compute_forward_add_rel_pos_f32`](#ggml_compute_forward_add_rel_pos_f32) to perform the computation.
    - If `src0` is of any other type, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the destination tensor `dst` based on the computation performed.
- **Functions called**:
    - [`ggml_compute_forward_add_rel_pos_f32`](#ggml_compute_forward_add_rel_pos_f32)


---
### ggml\_compute\_forward\_rwkv\_wkv6\_f32<!-- {{#callable:ggml_compute_forward_rwkv_wkv6_f32}} -->
Computes the forward pass of the RWKV v6 model using the provided tensors and parameters.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing computation parameters such as thread index and total threads.
    - `dst`: A pointer to `ggml_tensor` structure where the computed results will be stored.
- **Control Flow**:
    - The function begins by extracting dimensions and sizes from the `dst` tensor and its source tensors.
    - It checks if the current thread index exceeds the number of heads, returning early if so.
    - It initializes the destination data and state, and sets up strides for accessing tensor data.
    - If the current thread index is zero, it initializes the destination data to zero.
    - The function then enters a loop over time steps, processing each head in parallel based on the thread index.
    - For each head, it computes the necessary offsets and retrieves values from the source tensors.
    - It performs vectorized operations using SIMD instructions if available, or falls back to scalar operations otherwise.
    - The core computation involves updating the destination tensor and the state based on the RWKV model's equations.
    - Finally, it handles any remaining elements that were not processed in the vectorized loop.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the computed results of the RWKV forward pass.
- **Functions called**:
    - [`ggml_barrier`](ggml-cpu.c.driver.md#ggml_barrier)


---
### ggml\_compute\_forward\_rwkv\_wkv6<!-- {{#callable:ggml_compute_forward_rwkv_wkv6}} -->
The `ggml_compute_forward_rwkv_wkv6` function computes a forward pass for a specific tensor operation based on the type of the source tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the result of the computation.
- **Control Flow**:
    - The function retrieves the first source tensor from the destination tensor's source array.
    - It checks the type of the source tensor (`src0`).
    - If the type is `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_rwkv_wkv6_f32`](#ggml_compute_forward_rwkv_wkv6_f32) function to perform the computation.
    - If the type is not recognized, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place based on the computation performed.
- **Functions called**:
    - [`ggml_compute_forward_rwkv_wkv6_f32`](#ggml_compute_forward_rwkv_wkv6_f32)


---
### ggml\_compute\_forward\_gla\_f32<!-- {{#callable:ggml_compute_forward_gla_f32}} -->
Computes the forward pass of a generalized linear attention mechanism for a tensor in a multi-head attention context.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for computation, including thread indices.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the output of the computation.
- **Control Flow**:
    - The function begins by extracting dimensions and parameters from the `dst` tensor and the `params` structure.
    - It checks if the current thread index exceeds the number of heads, returning early if so.
    - The function initializes the output tensor to zero for the first head and synchronizes threads using a barrier.
    - It defines vectorization parameters based on the available hardware capabilities (AVX, ARM, etc.).
    - The main computation is performed in nested loops iterating over time steps, heads, and dimensions, applying the attention mechanism.
    - For each head, it computes the key, query, and value tensors, and updates the output tensor and state based on the attention formula.
    - The function handles both vectorized and non-vectorized cases, ensuring efficient computation.
- **Output**: The function modifies the `dst` tensor in place, storing the results of the forward pass of the generalized linear attention mechanism.
- **Functions called**:
    - [`ggml_get_op_params_f32`](../ggml-impl.h.driver.md#ggml_get_op_params_f32)
    - [`ggml_barrier`](ggml-cpu.c.driver.md#ggml_barrier)


---
### ggml\_compute\_forward\_gla<!-- {{#callable:ggml_compute_forward_gla}} -->
The `ggml_compute_forward_gla` function computes the forward pass of a generalized linear activation for a given tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of the source tensor `src0` using a switch statement.
    - If the type is `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_gla_f32`](#ggml_compute_forward_gla_f32) function to perform the computation.
    - If the type is not recognized, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place based on the computation performed.
- **Functions called**:
    - [`ggml_compute_forward_gla_f32`](#ggml_compute_forward_gla_f32)


---
### ggml\_compute\_forward\_rwkv\_wkv7\_f32<!-- {{#callable:ggml_compute_forward_rwkv_wkv7_f32}} -->
Computes the forward pass of a RWKV model layer using floating-point precision.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing computation parameters, including the current thread index (`ith`) and total number of threads (`nth`).
    - `dst`: A pointer to `ggml_tensor` structure where the computed results will be stored.
- **Control Flow**:
    - The function begins by extracting dimensions and parameters from the `dst` tensor and the `params` structure.
    - It checks if the current thread index (`ith`) exceeds the number of heads (`HEADS`), returning early if so.
    - The function calculates the start and end indices for the current head based on the total number of heads and threads.
    - It retrieves data pointers for the input tensors (r, w, k, v, a, b) and initializes necessary stride variables.
    - The main computation is performed in a nested loop structure iterating over time steps (`T`), heads, and the head size, performing matrix operations and accumulating results.
    - Depending on whether SIMD is enabled, it uses either scalar or vectorized operations to compute the results efficiently.
    - The computed results are stored in the `dst_data` array.
- **Output**: The function does not return a value but populates the `dst` tensor with the computed results of the RWKV layer.


---
### ggml\_compute\_forward\_rwkv\_wkv7<!-- {{#callable:ggml_compute_forward_rwkv_wkv7}} -->
The `ggml_compute_forward_rwkv_wkv7` function computes the forward pass for a specific tensor operation based on the type of the source tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result will be stored.
- **Control Flow**:
    - The function retrieves the first source tensor from the destination tensor's source array.
    - It checks the type of the source tensor (`src0`).
    - If the type is `GGML_TYPE_F32`, it calls the [`ggml_compute_forward_rwkv_wkv7_f32`](#ggml_compute_forward_rwkv_wkv7_f32) function to perform the computation.
    - If the type is not recognized, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the destination tensor `dst` in place based on the computation performed.
- **Functions called**:
    - [`ggml_compute_forward_rwkv_wkv7_f32`](#ggml_compute_forward_rwkv_wkv7_f32)


---
### ggml\_compute\_forward\_map\_custom1<!-- {{#callable:ggml_compute_forward_map_custom1}} -->
The `ggml_compute_forward_map_custom1` function computes a forward mapping operation using a custom function defined in the tensor's operation parameters.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing computation parameters such as the current index and total number of operations.
    - `dst`: A pointer to a `ggml_tensor` structure that serves as the destination tensor for the computation and contains the source tensor and operation parameters.
- **Control Flow**:
    - The function retrieves the source tensor `a` from the destination tensor `dst`.
    - It initializes a `ggml_map_custom1_op_params` structure `p` by copying the operation parameters from `dst`.
    - The custom mapping function `p.fun` is then called with the destination tensor `dst`, the source tensor `a`, and additional parameters from `params` and `p`.
- **Output**: The function does not return a value; instead, it modifies the destination tensor `dst` in place based on the results of the custom mapping function.


---
### ggml\_compute\_forward\_map\_custom2<!-- {{#callable:ggml_compute_forward_map_custom2}} -->
Executes a custom mapping operation on two source tensors and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing computation parameters such as the current index and total number of operations.
    - `dst`: A pointer to a `ggml_tensor` structure that serves as the destination tensor for the result of the mapping operation.
- **Control Flow**:
    - Extracts the first and second source tensors from the `dst` tensor's source array.
    - Copies the operation parameters from the `dst` tensor into a local structure `p`.
    - Calls the mapping function specified in `p.fun`, passing the destination tensor, the two source tensors, and additional parameters from `params` and `p`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the result of the custom mapping operation.


---
### ggml\_compute\_forward\_map\_custom3<!-- {{#callable:ggml_compute_forward_map_custom3}} -->
Computes a custom mapping operation on three source tensors and stores the result in the destination tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing computation parameters such as indices for the operation.
    - `dst`: A pointer to a `ggml_tensor` structure that serves as the destination tensor for the result of the mapping operation.
- **Control Flow**:
    - Extracts the source tensors `a`, `b`, and `c` from the `dst` tensor's source array.
    - Copies the operation parameters from `dst->op_params` into a local structure `p` of type `ggml_map_custom3_op_params`.
    - Calls the function `p.fun` with the destination tensor, the three source tensors, and additional parameters from `params` and `p`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the result of the custom mapping operation.


---
### ggml\_compute\_forward\_custom<!-- {{#callable:ggml_compute_forward_custom}} -->
Executes a custom operation on a tensor using specified computation parameters.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing computation parameters such as indices.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor on which the custom operation will be performed.
- **Control Flow**:
    - Copies the operation parameters from the destination tensor `dst` into a local variable `p` of type `ggml_custom_op_params`.
    - Calls the function pointer `fun` from the `p` structure, passing the destination tensor `dst`, the ith and nth parameters from `params`, and any user data stored in `p`.
- **Output**: The function does not return a value; it modifies the destination tensor `dst` in place based on the custom operation defined by `p.fun`.


---
### ggml\_compute\_forward\_cross\_entropy\_loss\_f32<!-- {{#callable:ggml_compute_forward_cross_entropy_loss_f32}} -->
Computes the forward cross-entropy loss for a batch of predictions and targets.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for computation, including thread information and workspace.
    - `dst`: A pointer to a `ggml_tensor` where the computed cross-entropy loss will be stored.
- **Control Flow**:
    - The function begins by asserting the types and shapes of the input tensors `src0` and `src1` to ensure they are valid for cross-entropy loss computation.
    - It calculates the number of classes (`nc`) and the number of rows (`nr`) in the input tensors.
    - The function determines the range of rows to process for the current thread based on the total number of threads and the current thread index.
    - For each row in the determined range, it retrieves the corresponding predictions and targets, computes the softmax values, and accumulates the loss for that thread.
    - After processing all rows, the thread's computed loss is stored in a shared array, and a barrier is used to synchronize threads.
    - Finally, the main thread (ith == 0) sums the losses from all threads and normalizes the final loss value before storing it in the destination tensor.
- **Output**: The function outputs the computed cross-entropy loss in the `dst` tensor, normalized by the number of rows.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_is_scalar`](../ggml.c.driver.md#ggml_is_scalar)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_vec_max_f32`](vec.h.driver.md#ggml_vec_max_f32)
    - [`ggml_vec_log_soft_max_f32`](vec.cpp.driver.md#ggml_vec_log_soft_max_f32)
    - [`ggml_vec_add1_f32`](vec.h.driver.md#ggml_vec_add1_f32)
    - [`ggml_vec_mul_f32`](vec.h.driver.md#ggml_vec_mul_f32)
    - [`ggml_vec_sum_f32`](vec.h.driver.md#ggml_vec_sum_f32)
    - [`ggml_barrier`](ggml-cpu.c.driver.md#ggml_barrier)


---
### ggml\_compute\_forward\_cross\_entropy\_loss<!-- {{#callable:ggml_compute_forward_cross_entropy_loss}} -->
Computes the forward cross-entropy loss for a given tensor based on its type.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` that serves as the destination tensor for the computed loss.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of `src0` using a switch statement.
    - If `src0` is of type `GGML_TYPE_F32`, it calls the function [`ggml_compute_forward_cross_entropy_loss_f32`](#ggml_compute_forward_cross_entropy_loss_f32) to perform the computation.
    - If `src0` is of any other type, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor to hold the computed cross-entropy loss.
- **Functions called**:
    - [`ggml_compute_forward_cross_entropy_loss_f32`](#ggml_compute_forward_cross_entropy_loss_f32)


---
### ggml\_compute\_forward\_cross\_entropy\_loss\_back\_f32<!-- {{#callable:ggml_compute_forward_cross_entropy_loss_back_f32}} -->
Computes the backward pass of the cross-entropy loss function for a neural network using gradient information.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation, including the thread index (`ith`) and the total number of threads (`nth`).
    - `dst`: A pointer to a `ggml_tensor` that will store the computed gradients for the backward pass.
- **Control Flow**:
    - The function begins by extracting the source tensors (`grad`, `src0f`, and `src1f`) from the `dst` tensor.
    - It asserts that all tensors are contiguous and have the same shape.
    - The number of rows (`nr`) and columns (`nc`) are determined from `src0f`.
    - The function calculates the range of rows to process for the current thread based on the total number of threads.
    - For each row in the assigned range, it performs the following steps:
    - 1. Retrieves the data pointers for the current row of `dst`, `src0f`, and `src1f`.
    - 2. Computes the softmax of `src0f` and scales it.
    - 3. Subtracts `src1f` from the softmax result and scales the result by the gradient divided by the number of rows.
    - Assertions are made to check for NaN and infinity values during debugging.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to store the computed gradients for the backward pass of the cross-entropy loss.
- **Functions called**:
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_vec_max_f32`](vec.h.driver.md#ggml_vec_max_f32)
    - [`ggml_vec_soft_max_f32`](vec.cpp.driver.md#ggml_vec_soft_max_f32)
    - [`ggml_vec_scale_f32`](vec.h.driver.md#ggml_vec_scale_f32)
    - [`ggml_vec_sub_f32`](vec.h.driver.md#ggml_vec_sub_f32)


---
### ggml\_compute\_forward\_cross\_entropy\_loss\_back<!-- {{#callable:ggml_compute_forward_cross_entropy_loss_back}} -->
Computes the backward pass of the cross-entropy loss for a given tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` that represents the destination tensor where the result will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of `src0` to determine the appropriate computation method.
    - If `src0` is of type `GGML_TYPE_F32`, it calls the function [`ggml_compute_forward_cross_entropy_loss_back_f32`](#ggml_compute_forward_cross_entropy_loss_back_f32) to perform the computation.
    - If `src0` is of any other type, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place with the computed backward cross-entropy loss.
- **Functions called**:
    - [`ggml_compute_forward_cross_entropy_loss_back_f32`](#ggml_compute_forward_cross_entropy_loss_back_f32)


---
### ggml\_compute\_forward\_opt\_step\_adamw\_f32<!-- {{#callable:ggml_compute_forward_opt_step_adamw_f32}} -->
This function performs a single optimization step using the AdamW algorithm for updating model weights based on gradients.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for the computation, including the thread index and total number of threads.
    - `dst`: A pointer to a `ggml_tensor` structure that serves as the destination tensor for the computed weights after the optimization step.
- **Control Flow**:
    - The function begins by extracting source tensors from the `dst` tensor, which include the current weights, gradients, and AdamW parameters.
    - It asserts that the shapes of the source tensors are compatible and that the AdamW parameters tensor contains exactly 7 elements.
    - The function calculates the number of rows to process per thread and determines the range of rows for the current thread based on its index.
    - It retrieves the AdamW parameters (learning rate, beta values, epsilon, weight decay, and adjusted beta values) from the parameters tensor.
    - A loop iterates over the assigned range of rows, calculating the necessary indices and offsets for accessing the data in the tensors.
    - Within the row loop, another loop processes each element in the current row, updating the moving averages `m` and `v`, and then applying the AdamW update rule to the weights `w`.
- **Output**: The function does not return a value; instead, it updates the weights in the `dst` tensor in place based on the AdamW optimization algorithm.
- **Functions called**:
    - [`ggml_are_same_shape`](../ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_get_data_f32`](../ggml.c.driver.md#ggml_get_data_f32)


---
### ggml\_compute\_forward\_opt\_step\_adamw<!-- {{#callable:ggml_compute_forward_opt_step_adamw}} -->
The `ggml_compute_forward_opt_step_adamw` function computes an optimization step using the AdamW algorithm for a given tensor.
- **Inputs**:
    - `params`: A pointer to `ggml_compute_params` structure containing parameters for the computation.
    - `dst`: A pointer to `ggml_tensor` that represents the destination tensor where the result will be stored.
- **Control Flow**:
    - The function retrieves the source tensor `src0` from the destination tensor `dst`.
    - It checks the type of the source tensor `src0` using a switch statement.
    - If the type is `GGML_TYPE_F32`, it calls the helper function [`ggml_compute_forward_opt_step_adamw_f32`](#ggml_compute_forward_opt_step_adamw_f32) to perform the computation.
    - If the type is not recognized, it triggers a fatal error using `GGML_ABORT`.
- **Output**: The function does not return a value; instead, it modifies the destination tensor `dst` in place based on the optimization step computed.
- **Functions called**:
    - [`ggml_compute_forward_opt_step_adamw_f32`](#ggml_compute_forward_opt_step_adamw_f32)


