# Purpose
This C header file defines a comprehensive set of function prototypes for performing various tensor operations, primarily intended for use in machine learning or numerical computing contexts. The file includes a series of `ggml_compute_forward_*` functions, each designed to execute a specific operation on tensors, such as duplication, addition, normalization, pooling, convolution, and more. These functions are part of a broader library, likely related to the "ggml" library, which is suggested by the inclusion of the "ggml.h" header. The functions are designed to be used in a forward computation context, indicating their role in the forward pass of neural network computations or similar processes.

The file also includes a mechanism to determine the cache line size, which is crucial for optimizing memory access patterns in high-performance computing applications. The use of `#pragma once` ensures that the header is included only once per compilation unit, preventing duplicate definitions. The file is structured to be compatible with both C and C++ environments, as indicated by the `extern "C"` block, which facilitates linkage in C++ projects. Overall, this header file provides a broad range of tensor manipulation functionalities, serving as a public API for developers to integrate and utilize these operations in their applications.
# Imports and Dependencies

---
- `ggml.h`


# Global Variables

---
### CACHE\_LINE\_SIZE\_F32
- **Type**: `size_t`
- **Description**: `CACHE_LINE_SIZE_F32` is a constant variable that represents the size of a cache line in terms of the number of `float` elements it can hold. It is calculated by dividing the `CACHE_LINE_SIZE` by the size of a `float` data type.
- **Use**: This variable is used to determine the number of `float` elements that fit into a cache line, which can be useful for optimizing memory access patterns in numerical computations.


# Function Declarations (Public API)

---
### ggml\_compute\_forward\_dup<!-- {{#callable_declaration:ggml_compute_forward_dup}} -->
Duplicates the data from the source tensor to the destination tensor.
- **Description**: This function is used to duplicate the contents of a source tensor into a destination tensor. It should be called when the destination tensor is ready to receive data, and the source tensor must be specified as the first source tensor in the destination's source array. The function handles different tensor types and performs the duplication accordingly. If the source tensor type is not compatible with the destination tensor type, the function will attempt to convert the data if possible. If the types are incompatible and cannot be converted, an error will be raised.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor. Must not be null and must have a valid source tensor set.
- **Output**: Returns nothing. The destination tensor is populated with the duplicated data from the source tensor.
- **See also**: [`ggml_compute_forward_dup`](ops.cpp.driver.md#ggml_compute_forward_dup)  (Implementation)


---
### ggml\_compute\_forward\_add<!-- {{#callable_declaration:ggml_compute_forward_add}} -->
Computes the element-wise addition of tensors.
- **Description**: This function is used to perform element-wise addition of tensors, where the result is stored in the destination tensor. It should be called when the destination tensor has been properly initialized and its source tensor is set. The function handles different tensor types, including both quantized and non-quantized formats. If the source tensor type is unsupported, the function will abort execution, indicating a fatal error.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result will be stored. Must not be null and must have its source tensor set.
- **Output**: None
- **See also**: [`ggml_compute_forward_add`](ops.cpp.driver.md#ggml_compute_forward_add)  (Implementation)


---
### ggml\_compute\_forward\_add1<!-- {{#callable_declaration:ggml_compute_forward_add1}} -->
Computes the forward addition operation.
- **Description**: This function is used to perform a forward addition operation on the specified destination tensor, which is expected to have two source tensors. It should be called after the necessary initialization of the `ggml_compute_params` and the destination tensor. The function handles various tensor data types, including floating-point and quantized formats, and will execute the appropriate addition operation based on the types of the source tensors. If the source tensor types are incompatible or unsupported, the function will abort execution.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that represents the destination tensor where the result will be stored. Must not be null and must have two source tensors set.
- **Output**: Returns nothing. The result of the addition operation is stored in the destination tensor.
- **See also**: [`ggml_compute_forward_add1`](ops.cpp.driver.md#ggml_compute_forward_add1)  (Implementation)


---
### ggml\_compute\_forward\_acc<!-- {{#callable_declaration:ggml_compute_forward_acc}} -->
Computes the forward accumulation of a tensor.
- **Description**: This function is used to perform forward accumulation on a tensor, which is typically part of a larger computation graph. It should be called with valid parameters after the necessary initialization of the computation environment. The function expects the destination tensor to have a source tensor of a compatible type, and it will abort if the source tensor type is unsupported. It is important to ensure that the `params` and `dst` pointers are valid and properly initialized before calling this function.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that represents the destination tensor for the accumulation. Must not be null and should have a valid source tensor.
- **Output**: None
- **See also**: [`ggml_compute_forward_acc`](ops.cpp.driver.md#ggml_compute_forward_acc)  (Implementation)


---
### ggml\_compute\_forward\_sum<!-- {{#callable_declaration:ggml_compute_forward_sum}} -->
Computes the sum of the input tensor.
- **Description**: This function is used to compute the sum of elements in a tensor and store the result in a destination tensor. It should be called when the input tensor is properly initialized and its type is supported. The function handles different tensor types, including `GGML_TYPE_F32`, `GGML_TYPE_F16`, and `GGML_TYPE_BF16`. If the input tensor type is unsupported, the function will abort execution. Ensure that the destination tensor is allocated and can hold the result of the sum.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the result of the sum. Must not be null and should be properly allocated to store the output.
- **Output**: Returns nothing. The result of the sum is stored in the `dst` tensor.
- **See also**: [`ggml_compute_forward_sum`](ops.cpp.driver.md#ggml_compute_forward_sum)  (Implementation)


---
### ggml\_compute\_forward\_sum\_rows<!-- {{#callable_declaration:ggml_compute_forward_sum_rows}} -->
Computes the sum of rows in a tensor.
- **Description**: This function is used to compute the sum of rows for a given tensor and store the result in a destination tensor. It should be called when the source tensor is properly initialized and of type `GGML_TYPE_F32`. If the source tensor type is not supported, the function will abort execution. Ensure that the destination tensor is appropriately allocated to hold the result.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` where the result will be stored. Must not be null and should be properly allocated to hold the sum of rows.
- **Output**: None
- **See also**: [`ggml_compute_forward_sum_rows`](ops.cpp.driver.md#ggml_compute_forward_sum_rows)  (Implementation)


---
### ggml\_compute\_forward\_mean<!-- {{#callable_declaration:ggml_compute_forward_mean}} -->
Computes the mean of the input tensor.
- **Description**: This function is used to calculate the mean of the values in a source tensor and store the result in a destination tensor. It should be called when the destination tensor is properly initialized and its source tensor is set. The function expects the source tensor to be of type `GGML_TYPE_F32`. If the source tensor type is not supported, the function will abort execution, indicating a fatal error.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that will hold the result of the mean computation. Must not be null and must have its source tensor set.
- **Output**: None
- **See also**: [`ggml_compute_forward_mean`](ops.cpp.driver.md#ggml_compute_forward_mean)  (Implementation)


---
### ggml\_compute\_forward\_argmax<!-- {{#callable_declaration:ggml_compute_forward_argmax}} -->
Computes the argmax of a tensor.
- **Description**: This function is used to compute the argmax of a source tensor and store the result in a destination tensor. It should be called when the destination tensor is properly initialized and its source tensor is set. The function expects the source tensor to be of type `GGML_TYPE_F32`. If the source tensor type is unsupported, the function will abort execution, indicating a fatal error. Therefore, it is crucial to ensure that the source tensor type is valid before calling this function.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the result of the argmax computation. The source tensor must be set in `dst->src[0]` and must not be null.
- **Output**: Returns nothing. The result of the argmax computation is stored in the destination tensor.
- **See also**: [`ggml_compute_forward_argmax`](ops.cpp.driver.md#ggml_compute_forward_argmax)  (Implementation)


---
### ggml\_compute\_forward\_count\_equal<!-- {{#callable_declaration:ggml_compute_forward_count_equal}} -->
Counts the number of equal elements in a tensor.
- **Description**: This function is used to compute the count of elements in a source tensor that are equal to a specified value. It should be called with a valid `ggml_tensor` that has been properly initialized and is of type `GGML_TYPE_I32`. The function expects the destination tensor to be allocated and ready to receive the result. If the source tensor type is not supported, the function will abort execution, indicating a fatal error.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` where the result will be stored. Must not be null and must be of type `GGML_TYPE_I32`.
- **Output**: The function does not return a value. Instead, it populates the `dst` tensor with the count of equal elements.
- **See also**: [`ggml_compute_forward_count_equal`](ops.cpp.driver.md#ggml_compute_forward_count_equal)  (Implementation)


---
### ggml\_compute\_forward\_repeat<!-- {{#callable_declaration:ggml_compute_forward_repeat}} -->
Computes the forward repeat operation.
- **Description**: This function is used to perform a repeat operation on a tensor, which is useful in various tensor manipulations. It should be called when the destination tensor has been properly initialized and is ready to receive the result of the operation. The function expects the source tensor type to be compatible with the operation, and it will handle specific types accordingly. If an unsupported tensor type is provided, the function will abort execution, indicating a fatal error.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result will be stored. Must not be null and should be properly initialized.
- **Output**: This function does not return a value and does not mutate any inputs directly; it modifies the `dst` tensor to contain the result of the repeat operation.
- **See also**: [`ggml_compute_forward_repeat`](ops.cpp.driver.md#ggml_compute_forward_repeat)  (Implementation)


---
### ggml\_compute\_forward\_repeat\_back<!-- {{#callable_declaration:ggml_compute_forward_repeat_back}} -->
Computes the forward repeat back operation.
- **Description**: This function is intended to perform a forward repeat back operation on a tensor, which is typically used in neural network computations. It should be called with a valid `ggml_compute_params` structure and a destination tensor that has been properly initialized. The source tensor type must be `GGML_TYPE_F32`; otherwise, the function will abort execution. It is important to ensure that the destination tensor's source is set correctly before invoking this function.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` that will hold the result of the computation. Must not be null and must have its source tensor set correctly.
- **Output**: None
- **See also**: [`ggml_compute_forward_repeat_back`](ops.cpp.driver.md#ggml_compute_forward_repeat_back)  (Implementation)


---
### ggml\_compute\_forward\_concat<!-- {{#callable_declaration:ggml_compute_forward_concat}} -->
Concatenates multiple tensors into a single destination tensor.
- **Description**: This function is used to concatenate multiple source tensors into a single destination tensor. It should be called when the destination tensor is properly initialized and its source tensors are set. The function handles different tensor data types, ensuring that the concatenation is performed correctly based on the type of the first source tensor. If the source tensor type is unsupported, the function will handle it gracefully without causing a crash.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the result of the concatenation. Must not be null and should have its source tensors set appropriately.
- **Output**: The function does not return a value and does not mutate any inputs.
- **See also**: [`ggml_compute_forward_concat`](ops.cpp.driver.md#ggml_compute_forward_concat)  (Implementation)


---
### ggml\_compute\_forward\_silu\_back<!-- {{#callable_declaration:ggml_compute_forward_silu_back}} -->
Computes the backward pass of the SiLU activation function.
- **Description**: This function is intended to be called during the backward pass of a neural network training process, specifically for the SiLU (Sigmoid Linear Unit) activation function. It requires a valid `ggml_compute_params` structure and a destination tensor `dst` that must have a source tensor set. The source tensor's type must be either `GGML_TYPE_F32` or `GGML_TYPE_F16`. If the source tensor type is unsupported, the function will abort execution. Ensure that the tensors are properly initialized and allocated before calling this function.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` that will hold the result of the computation. This tensor must have its source tensor set and must not be null.
- **Output**: None
- **See also**: [`ggml_compute_forward_silu_back`](ops.cpp.driver.md#ggml_compute_forward_silu_back)  (Implementation)


---
### ggml\_compute\_forward\_norm<!-- {{#callable_declaration:ggml_compute_forward_norm}} -->
Computes the forward normalization of a tensor.
- **Description**: This function is used to compute the forward normalization of a tensor, which is typically part of a neural network's forward pass. It should be called when the destination tensor is ready to receive the normalized output. The function expects the source tensor to be of type `GGML_TYPE_F32`. If the source tensor type is not supported, the function will abort execution, indicating a fatal error.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that will hold the result of the normalization. Must not be null and must have a valid source tensor of type `GGML_TYPE_F32`.
- **Output**: None
- **See also**: [`ggml_compute_forward_norm`](ops.cpp.driver.md#ggml_compute_forward_norm)  (Implementation)


---
### ggml\_compute\_forward\_rms\_norm<!-- {{#callable_declaration:ggml_compute_forward_rms_norm}} -->
Computes the RMS normalization of a tensor.
- **Description**: This function is used to perform RMS normalization on a tensor, which is a common operation in machine learning and data processing. It should be called when you have a destination tensor that is expected to hold the result of the normalization. The source tensor for the operation is expected to be the first source tensor of the destination tensor. It is important to ensure that the source tensor is of type `GGML_TYPE_F32`, as the function does not handle other types and will abort if an unsupported type is encountered.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the result of the RMS normalization. The tensor must have at least one source tensor, and the first source tensor must be of type `GGML_TYPE_F32`. If the tensor does not meet these criteria, the function will abort.
- **Output**: None
- **See also**: [`ggml_compute_forward_rms_norm`](ops.cpp.driver.md#ggml_compute_forward_rms_norm)  (Implementation)


---
### ggml\_compute\_forward\_rms\_norm\_back<!-- {{#callable_declaration:ggml_compute_forward_rms_norm_back}} -->
Computes the backward pass of RMS normalization.
- **Description**: This function is intended to be called during the backward pass of a neural network training process, specifically for the RMS normalization operation. It should be invoked after the forward pass of RMS normalization has been executed. The function expects the destination tensor to have a source tensor of type `GGML_TYPE_F32`. If the source tensor type is not supported, the function will abort execution, indicating a fatal error.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that will hold the result of the backward computation. Must not be null and must have a valid source tensor of type `GGML_TYPE_F32`.
- **Output**: None
- **See also**: [`ggml_compute_forward_rms_norm_back`](ops.cpp.driver.md#ggml_compute_forward_rms_norm_back)  (Implementation)


---
### ggml\_compute\_forward\_group\_norm<!-- {{#callable_declaration:ggml_compute_forward_group_norm}} -->
Computes the forward group normalization.
- **Description**: This function is intended to perform group normalization on a tensor, which is a common operation in neural network training and inference. It should be called when the destination tensor has been properly initialized and is ready to receive the results of the normalization. The function expects the source tensor to be of type `GGML_TYPE_F32`, and it will abort if the type is not supported. Ensure that the `params` and `dst` pointers are valid and not null before calling this function.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that will hold the result of the group normalization. Must not be null and should be properly initialized.
- **Output**: None
- **See also**: [`ggml_compute_forward_group_norm`](ops.cpp.driver.md#ggml_compute_forward_group_norm)  (Implementation)


---
### ggml\_compute\_forward\_l2\_norm<!-- {{#callable_declaration:ggml_compute_forward_l2_norm}} -->
Computes the L2 norm of a tensor.
- **Description**: This function is used to compute the L2 norm of a tensor, which is a common operation in various numerical and machine learning applications. It should be called when the destination tensor is properly initialized and has a source tensor of type `GGML_TYPE_F32`. If the source tensor type is not supported, the function will abort execution, indicating a fatal error. Ensure that the input parameters are valid before calling this function to avoid unexpected behavior.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that will hold the result of the L2 norm computation. Must not be null and must have a valid source tensor of type `GGML_TYPE_F32`.
- **Output**: None
- **See also**: [`ggml_compute_forward_l2_norm`](ops.cpp.driver.md#ggml_compute_forward_l2_norm)  (Implementation)


---
### ggml\_compute\_forward\_out\_prod<!-- {{#callable_declaration:ggml_compute_forward_out_prod}} -->
Computes the outer product of tensors.
- **Description**: This function is used to compute the outer product of two tensors and store the result in the destination tensor. It should be called with valid parameters after ensuring that the source tensor is properly initialized. The function handles various tensor types, and if an unsupported type is provided, it will trigger an error. It is important to ensure that the destination tensor is appropriately sized to hold the result of the operation.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` structure where the result will be stored. Must not be null and should be properly initialized to hold the output of the operation.
- **Output**: Returns nothing. The result of the outer product is stored in the `dst` tensor.
- **See also**: [`ggml_compute_forward_out_prod`](ops.cpp.driver.md#ggml_compute_forward_out_prod)  (Implementation)


---
### ggml\_compute\_forward\_scale<!-- {{#callable_declaration:ggml_compute_forward_scale}} -->
Computes the scaled output tensor.
- **Description**: This function is used to compute the scaled output of a tensor based on the provided parameters. It should be called when you need to apply a scaling operation to a tensor, specifically when the source tensor is of type `GGML_TYPE_F32`. The function expects that the destination tensor (`dst`) has been properly initialized and that its source tensor (`src[0]`) is valid. If the source tensor type is not supported, the function will abort execution, indicating a fatal error.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the result of the scaling operation. Must not be null and should be properly initialized.
- **Output**: Returns nothing and directly modifies the `dst` tensor to contain the scaled output.
- **See also**: [`ggml_compute_forward_scale`](ops.cpp.driver.md#ggml_compute_forward_scale)  (Implementation)


---
### ggml\_compute\_forward\_set<!-- {{#callable_declaration:ggml_compute_forward_set}} -->
Sets the destination tensor based on the source tensor.
- **Description**: This function is used to set the values of a destination tensor based on the values of its source tensor. It should be called when the destination tensor is properly initialized and its source tensor is set. The function handles different tensor types, and if the source tensor type is unsupported, it will trigger an error. Ensure that the destination tensor is valid and that its source tensor is correctly assigned before calling this function.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure representing the destination tensor. Must not be null and should have a valid source tensor assigned.
- **Output**: None
- **See also**: [`ggml_compute_forward_set`](ops.cpp.driver.md#ggml_compute_forward_set)  (Implementation)


---
### ggml\_compute\_forward\_cpy<!-- {{#callable_declaration:ggml_compute_forward_cpy}} -->
Copies data from one tensor to another.
- **Description**: This function is used to copy the contents of a source tensor to a destination tensor. It should be called when you need to duplicate tensor data, ensuring that the destination tensor is properly allocated and has the same dimensions as the source tensor. It is important to ensure that the `params` and `dst` parameters are valid and initialized before calling this function to avoid undefined behavior.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that represents the destination tensor. Must not be null and should be properly allocated to hold the copied data.
- **Output**: None
- **See also**: [`ggml_compute_forward_cpy`](ops.cpp.driver.md#ggml_compute_forward_cpy)  (Implementation)


---
### ggml\_compute\_forward\_cont<!-- {{#callable_declaration:ggml_compute_forward_cont}} -->
Computes the forward operation for a tensor.
- **Description**: This function is intended to be called when performing a forward computation on a tensor, specifically to handle the continuation of a computation sequence. It should be invoked after the necessary initialization of the `ggml_compute_params` and `ggml_tensor` structures. The function does not return a value, but it modifies the `dst` tensor based on the computation defined by the parameters. Ensure that the `params` and `dst` are valid and properly initialized before calling this function to avoid undefined behavior.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null and should be properly initialized before use.
    - `dst`: Pointer to a `ggml_tensor` structure where the result of the computation will be stored. Must not be null and should be allocated and initialized before calling this function.
- **Output**: None
- **See also**: [`ggml_compute_forward_cont`](ops.cpp.driver.md#ggml_compute_forward_cont)  (Implementation)


---
### ggml\_compute\_forward\_reshape<!-- {{#callable_declaration:ggml_compute_forward_reshape}} -->
Reshapes a tensor.
- **Description**: This function is intended to reshape a tensor as specified by the parameters. It should be called when a tensor needs to be modified in shape without altering its data. The function does not perform any operations and is a no-operation (NOP), meaning it does not change the state of the provided tensor or parameters. It is important to ensure that the `params` and `dst` pointers are valid before calling this function, as passing null pointers may lead to undefined behavior.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that represents the destination tensor to be reshaped. Must not be null.
- **Output**: None
- **See also**: [`ggml_compute_forward_reshape`](ops.cpp.driver.md#ggml_compute_forward_reshape)  (Implementation)


---
### ggml\_compute\_forward\_view<!-- {{#callable_declaration:ggml_compute_forward_view}} -->
Performs a no-operation for the given parameters.
- **Description**: This function is intended to be called when no computation is required for the specified parameters. It serves as a placeholder and does not modify any state or produce any output. It is useful in scenarios where a function call is syntactically required but no action is needed. There are no specific preconditions for calling this function, but it should be noted that it does not perform any operations or checks.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains the parameters for computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that represents the destination tensor. Must not be null.
- **Output**: None
- **See also**: [`ggml_compute_forward_view`](ops.cpp.driver.md#ggml_compute_forward_view)  (Implementation)


---
### ggml\_compute\_forward\_permute<!-- {{#callable_declaration:ggml_compute_forward_permute}} -->
Permutes the elements of a tensor.
- **Description**: This function is intended to be called when you need to permute the elements of a tensor, typically as part of a larger computation involving tensor manipulations. It is important to ensure that the `params` and `dst` parameters are properly initialized before calling this function. The function does not perform any operations on the inputs, and thus does not modify the state of the provided parameters or the destination tensor.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that represents the destination tensor for the operation. Must not be null.
- **Output**: None
- **See also**: [`ggml_compute_forward_permute`](ops.cpp.driver.md#ggml_compute_forward_permute)  (Implementation)


---
### ggml\_compute\_forward\_transpose<!-- {{#callable_declaration:ggml_compute_forward_transpose}} -->
Computes the forward transpose of a tensor.
- **Description**: This function is intended to be called as part of a series of tensor operations. It is designed to handle the forward computation of a transpose operation on a tensor, which is a common operation in various mathematical and machine learning contexts. It is important to ensure that the `params` and `dst` parameters are properly initialized before calling this function. The function does not modify the input parameters, and it is safe to call even if the parameters are not valid, as it does not perform any operations.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that will hold the result of the transpose operation. Must not be null.
- **Output**: None
- **See also**: [`ggml_compute_forward_transpose`](ops.cpp.driver.md#ggml_compute_forward_transpose)  (Implementation)


---
### ggml\_compute\_forward\_get\_rows<!-- {{#callable_declaration:ggml_compute_forward_get_rows}} -->
Retrieves rows from a tensor.
- **Description**: This function is used to extract specific rows from a tensor and store the result in a destination tensor. It should be called with valid parameters after the necessary initialization of the tensors involved. The source tensor must be properly configured and of a supported type, as the function handles various tensor types differently. If the source tensor type is unsupported, the function will abort execution, indicating a fatal error.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure where the result will be stored. Must not be null and should be properly initialized to hold the output.
- **Output**: The function does not return a value. The output is written directly to the `dst` tensor, which will contain the extracted rows from the source tensor.
- **See also**: [`ggml_compute_forward_get_rows`](ops.cpp.driver.md#ggml_compute_forward_get_rows)  (Implementation)


---
### ggml\_compute\_forward\_get\_rows\_back<!-- {{#callable_declaration:ggml_compute_forward_get_rows_back}} -->
Computes the backward operation for retrieving rows from a tensor.
- **Description**: This function is intended to be called when performing backward computations in a neural network or similar context, specifically to retrieve rows from a tensor based on the provided parameters. It is essential to ensure that the `dst` tensor has been properly initialized and that its source tensor is of a supported type (either `GGML_TYPE_F16` or `GGML_TYPE_F32`). If the source tensor type is unsupported, the function will abort execution. This function should be used in conjunction with other forward and backward operations to maintain the integrity of the computation graph.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that will hold the result of the operation. Must not be null and should be properly initialized with a valid source tensor.
- **Output**: None
- **See also**: [`ggml_compute_forward_get_rows_back`](ops.cpp.driver.md#ggml_compute_forward_get_rows_back)  (Implementation)


---
### ggml\_compute\_forward\_diag<!-- {{#callable_declaration:ggml_compute_forward_diag}} -->
Computes the diagonal of a tensor.
- **Description**: This function is used to compute the diagonal of a tensor and store the result in the destination tensor. It should be called when the source tensor is of type `GGML_TYPE_F32`. The function expects the `params` to be properly initialized and the `dst` tensor to have a valid source tensor. If the source tensor type is not supported, the function will abort execution.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains the parameters for computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure where the result will be stored. Must not be null and must have a valid source tensor.
- **Output**: None
- **See also**: [`ggml_compute_forward_diag`](ops.cpp.driver.md#ggml_compute_forward_diag)  (Implementation)


---
### ggml\_compute\_forward\_diag\_mask\_inf<!-- {{#callable_declaration:ggml_compute_forward_diag_mask_inf}} -->
Applies a diagonal mask with negative infinity values.
- **Description**: This function is intended to be used for applying a diagonal mask to a tensor, where the masked values are set to negative infinity. It should be called when the destination tensor is prepared and has a valid source tensor. The function expects the source tensor to be of type `GGML_TYPE_F32`. If the source tensor type is not supported, the function will abort execution, indicating a fatal error.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that will receive the result of the computation. Must not be null and must have a valid source tensor.
- **Output**: None
- **See also**: [`ggml_compute_forward_diag_mask_inf`](ops.cpp.driver.md#ggml_compute_forward_diag_mask_inf)  (Implementation)


---
### ggml\_compute\_forward\_diag\_mask\_zero<!-- {{#callable_declaration:ggml_compute_forward_diag_mask_zero}} -->
Computes a diagonal mask with zero values.
- **Description**: This function is intended to be called when a diagonal mask with zero values is needed for a tensor operation. It should be invoked with a valid `ggml_tensor` that is properly initialized and has a source tensor of type `GGML_TYPE_F32`. If the source tensor type is not supported, the function will abort execution. Ensure that the `params` argument is also valid and points to a properly initialized `ggml_compute_params` structure.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` where the result will be stored. Must not be null and should be properly initialized.
- **Output**: None
- **See also**: [`ggml_compute_forward_diag_mask_zero`](ops.cpp.driver.md#ggml_compute_forward_diag_mask_zero)  (Implementation)


---
### ggml\_compute\_forward\_soft\_max<!-- {{#callable_declaration:ggml_compute_forward_soft_max}} -->
Computes the softmax of the input tensor.
- **Description**: This function is used to compute the softmax of a tensor, which is a common operation in machine learning for converting logits to probabilities. It should be called when the destination tensor is prepared to receive the output of the softmax operation. The input tensor must be of type `GGML_TYPE_F32`, and if it is of a different type, the function will abort execution. Ensure that the `params` structure is properly initialized before calling this function.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` where the result will be stored. Must not be null and must be of type `GGML_TYPE_F32`.
- **Output**: The function does not return a value. It mutates the `dst` tensor to contain the softmax results.
- **See also**: [`ggml_compute_forward_soft_max`](ops.cpp.driver.md#ggml_compute_forward_soft_max)  (Implementation)


---
### ggml\_compute\_forward\_soft\_max\_ext\_back<!-- {{#callable_declaration:ggml_compute_forward_soft_max_ext_back}} -->
Computes the backward pass of the softmax operation.
- **Description**: This function is intended to be called during the backward pass of a neural network training process, specifically after a softmax operation has been performed. It requires a valid `ggml_compute_params` structure and a destination tensor where the results will be stored. The function expects the source tensor to be of type `GGML_TYPE_F32`. If the source tensor type is not supported, the function will abort execution, indicating a fatal error. Ensure that the input tensors are properly initialized and of the correct type before calling this function.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` where the results of the computation will be stored. Must not be null and should be properly initialized.
- **Output**: None
- **See also**: [`ggml_compute_forward_soft_max_ext_back`](ops.cpp.driver.md#ggml_compute_forward_soft_max_ext_back)  (Implementation)


---
### ggml\_compute\_forward\_rope<!-- {{#callable_declaration:ggml_compute_forward_rope}} -->
Computes the forward pass of the rope operation.
- **Description**: This function is used to perform the forward computation of the rope operation on a tensor. It should be called with valid parameters after the necessary initialization of the computation environment. The function expects the destination tensor to have a source tensor of a compatible type, specifically either `GGML_TYPE_F16` or `GGML_TYPE_F32`. If the source tensor type is unsupported, the function will abort execution, indicating a fatal error. It is important to ensure that the destination tensor is properly allocated and that the source tensor is set before calling this function.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that will hold the result of the computation. Must not be null and should have a valid source tensor set.
- **Output**: None
- **See also**: [`ggml_compute_forward_rope`](ops.cpp.driver.md#ggml_compute_forward_rope)  (Implementation)


---
### ggml\_compute\_forward\_rope\_back<!-- {{#callable_declaration:ggml_compute_forward_rope_back}} -->
Computes the backward pass for the rope operation.
- **Description**: This function is intended to be called during the backward pass of a neural network computation, specifically for the rope operation. It requires that the `dst` tensor has been properly initialized and that its source tensor is of a supported type (either `GGML_TYPE_F16` or `GGML_TYPE_F32`). If the source tensor type is unsupported, the function will abort execution. It is important to ensure that the `params` structure is valid and properly configured before calling this function.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` that will hold the result of the computation. Must not be null and should be properly initialized.
- **Output**: None
- **See also**: [`ggml_compute_forward_rope_back`](ops.cpp.driver.md#ggml_compute_forward_rope_back)  (Implementation)


---
### ggml\_compute\_forward\_clamp<!-- {{#callable_declaration:ggml_compute_forward_clamp}} -->
Clamps the values of a tensor.
- **Description**: This function is used to apply a clamping operation on a tensor, which restricts its values to a specified range. It should be called when you need to ensure that the values in the destination tensor do not exceed certain limits, typically after the tensor has been initialized and populated with data. The function expects the source tensor type to be compatible with the clamping operation; if the source tensor type is unsupported, the function will abort execution. It is important to ensure that the `params` and `dst` pointers are valid and properly initialized before calling this function.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that will hold the result of the clamping operation. Must not be null and should be properly initialized.
- **Output**: None
- **See also**: [`ggml_compute_forward_clamp`](ops.cpp.driver.md#ggml_compute_forward_clamp)  (Implementation)


---
### ggml\_compute\_forward\_conv\_transpose\_1d<!-- {{#callable_declaration:ggml_compute_forward_conv_transpose_1d}} -->
Computes the forward transpose convolution for 1D tensors.
- **Description**: This function is used to perform a forward transpose convolution operation on a 1D tensor, which is typically part of a neural network layer. It should be called when the destination tensor has been properly initialized and is ready to receive the results of the convolution operation. The function expects the source tensor type to be either `GGML_TYPE_F16` or `GGML_TYPE_F32`, and it will handle the computation accordingly. If the source tensor type is unsupported, the function will abort execution, indicating a fatal error.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the result of the convolution operation. Must not be null and should be properly initialized before calling this function.
- **Output**: This function does not return a value and does not mutate any inputs.
- **See also**: [`ggml_compute_forward_conv_transpose_1d`](ops.cpp.driver.md#ggml_compute_forward_conv_transpose_1d)  (Implementation)


---
### ggml\_compute\_forward\_im2col<!-- {{#callable_declaration:ggml_compute_forward_im2col}} -->
Computes the im2col transformation for a tensor.
- **Description**: This function is used to perform the im2col operation on a tensor, which is commonly used in convolutional neural networks to rearrange image data into a format suitable for matrix multiplication. It should be called with a valid `ggml_compute_params` structure and a destination tensor that is properly initialized. The function will handle tensors of type `GGML_TYPE_F16` and `GGML_TYPE_F32`, and will abort if the tensor type is unsupported.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` that will hold the result of the im2col operation. Must not be null and must be of type `GGML_TYPE_F16` or `GGML_TYPE_F32`.
- **Output**: None
- **See also**: [`ggml_compute_forward_im2col`](ops.cpp.driver.md#ggml_compute_forward_im2col)  (Implementation)


---
### ggml\_compute\_forward\_im2col\_back\_f32<!-- {{#callable_declaration:ggml_compute_forward_im2col_back_f32}} -->
Computes the backward pass for im2col operation.
- **Description**: This function is intended to be called during the backward pass of a neural network training process, specifically for the im2col operation used in convolutional layers. It takes gradients from the forward pass output and computes the corresponding gradients for the input tensor. The function expects that the input tensors are of type `GGML_TYPE_F32` and that the destination tensor is properly allocated to hold the computed gradients. It is crucial to ensure that the parameters provided in the `dst` tensor's operation parameters are correctly set, as they define the stride, padding, and dimensions of the convolution operation. Invalid parameters or tensor types will result in assertions failing.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains execution parameters. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that will hold the computed gradients. Must not be null and must be of type `GGML_TYPE_F32`.
- **Output**: None
- **See also**: [`ggml_compute_forward_im2col_back_f32`](ops.cpp.driver.md#ggml_compute_forward_im2col_back_f32)  (Implementation)


---
### ggml\_compute\_forward\_conv\_transpose\_2d<!-- {{#callable_declaration:ggml_compute_forward_conv_transpose_2d}} -->
Computes the forward pass of a 2D transposed convolution.
- **Description**: This function is used to perform a forward pass of a 2D transposed convolution operation, which is typically used in neural networks for upsampling. It must be called with valid input tensors that are properly initialized and of the correct types: the first source tensor must be of type `GGML_TYPE_F16`, the second source tensor must be of type `GGML_TYPE_F32`, and the destination tensor must be of type `GGML_TYPE_F32`. The function handles the computation in a multi-threaded manner, and it is important to ensure that the `params` structure is correctly set up before calling this function. If the input tensors do not meet the type requirements, the function will assert and terminate.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: A pointer to the destination `ggml_tensor` where the result will be stored. Must not be null and must be of type `GGML_TYPE_F32`.
- **Output**: The function does not return a value and mutates the `dst` tensor to contain the result of the transposed convolution operation.
- **See also**: [`ggml_compute_forward_conv_transpose_2d`](ops.cpp.driver.md#ggml_compute_forward_conv_transpose_2d)  (Implementation)


---
### ggml\_compute\_forward\_conv\_2d\_dw<!-- {{#callable_declaration:ggml_compute_forward_conv_2d_dw}} -->
Computes the forward pass of a 2D depthwise convolution.
- **Description**: This function is used to perform a forward pass of a 2D depthwise convolution operation on the input tensor. It should be called after ensuring that the input tensors are properly initialized and that the destination tensor is allocated. The function expects the source tensor to have a contiguous memory layout or a specific channel layout. If the memory layout is not supported, the function will abort. The parameters for the convolution, such as stride, padding, and dilation, are expected to be set in the destination tensor's operation parameters.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that will hold the result of the convolution. Must not be null and should be properly allocated with the correct dimensions and operation parameters.
- **Output**: None
- **See also**: [`ggml_compute_forward_conv_2d_dw`](ops.cpp.driver.md#ggml_compute_forward_conv_2d_dw)  (Implementation)


---
### ggml\_compute\_forward\_pool\_1d<!-- {{#callable_declaration:ggml_compute_forward_pool_1d}} -->
Computes the forward pass of a 1D pooling operation.
- **Description**: This function is used to perform a forward pooling operation on a 1D tensor. It must be called with valid parameters that specify the pooling operation type, kernel size, and stride, which are expected to be set in the `dst` tensor's operation parameters. The function does not support padding, and it requires that the kernel size and stride are equal. If the input parameters do not meet these requirements, the function will assert and terminate.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that will hold the result of the pooling operation. Must not be null and should have valid operation parameters set.
- **Output**: None
- **See also**: [`ggml_compute_forward_pool_1d`](ops.cpp.driver.md#ggml_compute_forward_pool_1d)  (Implementation)


---
### ggml\_compute\_forward\_pool\_2d<!-- {{#callable_declaration:ggml_compute_forward_pool_2d}} -->
Computes the forward pass of a 2D pooling operation.
- **Description**: This function is used to perform a 2D pooling operation on a source tensor, producing a destination tensor. It must be called with valid parameters, specifically after initializing the `ggml_compute_params` structure and ensuring that the destination tensor has a valid source tensor. The function supports different pooling operations, such as average and max pooling, and handles edge cases where the source tensor dimensions may not align perfectly with the pooling parameters. It is important to note that the function will not execute if the `ith` parameter in `params` is not zero.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that will hold the result of the pooling operation. Must not be null and must have a valid source tensor set.
- **Output**: None
- **See also**: [`ggml_compute_forward_pool_2d`](ops.cpp.driver.md#ggml_compute_forward_pool_2d)  (Implementation)


---
### ggml\_compute\_forward\_pool\_2d\_back<!-- {{#callable_declaration:ggml_compute_forward_pool_2d_back}} -->
Computes the backward pass of a 2D pooling operation.
- **Description**: This function is intended to be called during the backward pass of a neural network training process, specifically after a 2D pooling operation has been performed. It updates the gradients in the destination tensor based on the gradients from the source tensor and the pooling operation type (max or average). It is crucial to ensure that this function is called with the correct parameters, particularly that the `params` structure is properly initialized and that the `dst` tensor has valid source tensors. The function assumes that the first index of `params` is zero, and it will return early if this condition is not met. Additionally, the destination tensor must be of type `GGML_TYPE_F32` or `GGML_TYPE_F16`.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that represents the destination tensor where the gradients will be accumulated. Must not be null and must have valid source tensors.
- **Output**: None
- **See also**: [`ggml_compute_forward_pool_2d_back`](ops.cpp.driver.md#ggml_compute_forward_pool_2d_back)  (Implementation)


---
### ggml\_compute\_forward\_upscale<!-- {{#callable_declaration:ggml_compute_forward_upscale}} -->
Computes the upscale operation for a tensor.
- **Description**: This function is used to perform an upscale operation on a destination tensor, which is specified by the `dst` parameter. It must be called with a valid `ggml_compute_params` structure and a destination tensor that has a source tensor of type `GGML_TYPE_F32`. If the source tensor type is not supported, the function will abort execution. This function is typically used in scenarios where tensor resizing is required, such as in image processing or neural network operations.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the upscale operation will be stored. Must not be null and must have a valid source tensor of type `GGML_TYPE_F32`.
- **Output**: None
- **See also**: [`ggml_compute_forward_upscale`](ops.cpp.driver.md#ggml_compute_forward_upscale)  (Implementation)


---
### ggml\_compute\_forward\_pad<!-- {{#callable_declaration:ggml_compute_forward_pad}} -->
Computes the forward padding operation.
- **Description**: This function is used to perform a forward padding operation on a tensor, which is typically required in various neural network architectures. It should be called when the destination tensor is prepared to receive the padded output. The function expects the source tensor type to be `GGML_TYPE_F32`, and if the type is unsupported, it will trigger an abort. Ensure that the `params` and `dst` pointers are valid and properly initialized before calling this function.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that will hold the result of the padding operation. Must not be null and should be properly initialized.
- **Output**: None
- **See also**: [`ggml_compute_forward_pad`](ops.cpp.driver.md#ggml_compute_forward_pad)  (Implementation)


---
### ggml\_compute\_forward\_pad\_reflect\_1d<!-- {{#callable_declaration:ggml_compute_forward_pad_reflect_1d}} -->
Computes the forward padding of a tensor using reflection.
- **Description**: This function is used to apply reflective padding to a 1D tensor, which is useful in various signal processing and neural network applications. It should be called after the necessary initialization of the `ggml_tensor` structures, specifically ensuring that the source tensor is of type `GGML_TYPE_F32`. The function modifies the destination tensor by reflecting its edges based on the specified padding parameters. It is important to ensure that the padding values are valid and that the destination tensor has been allocated with sufficient space to accommodate the reflected values.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: A pointer to the destination `ggml_tensor` where the result will be stored. Must not be null and must be of type `GGML_TYPE_F32`.
- **Output**: Returns nothing and directly modifies the `dst` tensor to contain the padded values.
- **See also**: [`ggml_compute_forward_pad_reflect_1d`](ops.cpp.driver.md#ggml_compute_forward_pad_reflect_1d)  (Implementation)


---
### ggml\_compute\_forward\_arange<!-- {{#callable_declaration:ggml_compute_forward_arange}} -->
Computes a tensor containing a sequence of evenly spaced values.
- **Description**: This function is used to generate a tensor that contains a sequence of values spaced evenly within a specified range. It should be called when the destination tensor is properly initialized and of the correct type. The function currently supports only tensors of type `GGML_TYPE_F32`. If the destination tensor is of an unsupported type, the function will abort execution, indicating a fatal error.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` where the result will be stored. Must not be null and must be of type `GGML_TYPE_F32`.
- **Output**: None
- **See also**: [`ggml_compute_forward_arange`](ops.cpp.driver.md#ggml_compute_forward_arange)  (Implementation)


---
### ggml\_compute\_forward\_timestep\_embedding<!-- {{#callable_declaration:ggml_compute_forward_timestep_embedding}} -->
Computes the forward timestep embedding.
- **Description**: This function is used to compute the forward timestep embedding for a given tensor. It should be called when the destination tensor is prepared to receive the computed values. The function expects the source tensor to be of type `GGML_TYPE_F32`. If the source tensor type is not supported, the function will abort execution, indicating a fatal error.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` where the result will be stored. Must not be null and must be properly initialized.
- **Output**: None
- **See also**: [`ggml_compute_forward_timestep_embedding`](ops.cpp.driver.md#ggml_compute_forward_timestep_embedding)  (Implementation)


---
### ggml\_compute\_forward\_argsort<!-- {{#callable_declaration:ggml_compute_forward_argsort}} -->
Computes the argsort of a tensor.
- **Description**: This function is used to compute the argsort of a tensor, which is a common operation in data processing and machine learning tasks. It should be called when the destination tensor is prepared to receive the sorted indices. The input tensor must be of type `GGML_TYPE_F32`, and the function will abort if the type is not supported. Ensure that the `params` and `dst` pointers are valid and properly initialized before calling this function.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` where the result will be stored. Must not be null and must be of type `GGML_TYPE_F32`.
- **Output**: None
- **See also**: [`ggml_compute_forward_argsort`](ops.cpp.driver.md#ggml_compute_forward_argsort)  (Implementation)


---
### ggml\_compute\_forward\_leaky\_relu<!-- {{#callable_declaration:ggml_compute_forward_leaky_relu}} -->
Computes the forward pass of the leaky ReLU activation function.
- **Description**: This function is used to apply the leaky ReLU activation function to a tensor, transforming its values based on the leaky ReLU formula. It should be called when you need to apply this activation function to the output of a neural network layer. The function expects the destination tensor to have a source tensor of compatible type, which can be either `GGML_TYPE_F32` or `GGML_TYPE_F16`. If the source tensor type is unsupported, the function will abort execution, so it is important to ensure that the source tensor is of the correct type before calling this function.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the result of the leaky ReLU computation. Must not be null and must have a valid source tensor.
- **Output**: The function does not return a value and directly modifies the `dst` tensor to contain the results of the leaky ReLU activation.
- **See also**: [`ggml_compute_forward_leaky_relu`](ops.cpp.driver.md#ggml_compute_forward_leaky_relu)  (Implementation)


---
### ggml\_compute\_forward\_flash\_attn\_ext<!-- {{#callable_declaration:ggml_compute_forward_flash_attn_ext}} -->
Computes the forward pass of the flash attention mechanism.
- **Description**: This function is used to perform the forward computation of the flash attention mechanism, which is typically called during the forward pass of a neural network. It requires the input tensors for queries, keys, values, and an optional mask, and produces an output tensor that holds the result of the attention computation. It is important to ensure that the `dst` tensor is properly allocated and has the correct parameters set before calling this function. The function expects the precision of the output tensor to be either default or `F32`, and will abort if an unsupported precision is provided.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `q`: A pointer to the query tensor (`ggml_tensor`). Must not be null and should have compatible dimensions for the attention computation.
    - `k`: A pointer to the key tensor (`ggml_tensor`). Must not be null and should have compatible dimensions for the attention computation.
    - `v`: A pointer to the value tensor (`ggml_tensor`). Must not be null and should have compatible dimensions for the attention computation.
    - `mask`: A pointer to an optional mask tensor (`ggml_tensor`). Can be null if no masking is required, but if provided, it must have compatible dimensions.
    - `dst`: A pointer to the destination tensor (`ggml_tensor`) where the result will be stored. Must not be null and should be properly allocated with the expected output dimensions.
- **Output**: Returns nothing. The result of the attention computation is stored in the `dst` tensor.
- **See also**: [`ggml_compute_forward_flash_attn_ext`](ops.cpp.driver.md#ggml_compute_forward_flash_attn_ext)  (Implementation)


---
### ggml\_compute\_forward\_flash\_attn\_back<!-- {{#callable_declaration:ggml_compute_forward_flash_attn_back}} -->
Computes the backward pass of the flash attention mechanism.
- **Description**: This function is intended to be called during the backward pass of a neural network that utilizes flash attention. It requires a valid `ggml_compute_params` structure and a destination tensor where the results will be stored. The `masked` parameter indicates whether the attention should be masked. It is crucial that the destination tensor's source tensor is of type `GGML_TYPE_F32`, as the function does not handle other types and will abort if an unsupported type is encountered.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `masked`: A boolean value indicating whether the attention should be masked. Valid values are true or false.
    - `dst`: A pointer to a `ggml_tensor` where the results of the computation will be stored. The source tensor of `dst` must be of type `GGML_TYPE_F32`. Must not be null.
- **Output**: None
- **See also**: [`ggml_compute_forward_flash_attn_back`](ops.cpp.driver.md#ggml_compute_forward_flash_attn_back)  (Implementation)


---
### ggml\_compute\_forward\_ssm\_conv<!-- {{#callable_declaration:ggml_compute_forward_ssm_conv}} -->
Computes the forward pass of a state space model convolution.
- **Description**: This function is intended to be called when performing a forward computation for a state space model convolution operation. It requires a valid `ggml_compute_params` structure and a destination tensor `dst` where the result will be stored. The function expects that the first source tensor in `dst` is of type `GGML_TYPE_F32`. If the tensor type is unsupported, the function will abort execution. It is important to ensure that the parameters are properly initialized before calling this function.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` that serves as the destination for the computation result. The first source tensor within `dst` must be of type `GGML_TYPE_F32`. Must not be null.
- **Output**: None
- **See also**: [`ggml_compute_forward_ssm_conv`](ops.cpp.driver.md#ggml_compute_forward_ssm_conv)  (Implementation)


---
### ggml\_compute\_forward\_ssm\_scan<!-- {{#callable_declaration:ggml_compute_forward_ssm_scan}} -->
Computes the forward scan of a state space model.
- **Description**: This function is intended to be called when performing computations related to state space models, specifically for the forward scan operation. It requires a valid `ggml_compute_params` structure and a destination tensor (`dst`) where the results will be stored. The function expects that the first source tensor in `dst` is of type `GGML_TYPE_F32`. If the type is not supported, the function will abort execution, indicating a fatal error. It is important to ensure that the parameters are properly initialized before calling this function.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` that serves as the destination for the computation results. The first source tensor within `dst` must be of type `GGML_TYPE_F32`. Must not be null.
- **Output**: None
- **See also**: [`ggml_compute_forward_ssm_scan`](ops.cpp.driver.md#ggml_compute_forward_ssm_scan)  (Implementation)


---
### ggml\_compute\_forward\_win\_part<!-- {{#callable_declaration:ggml_compute_forward_win_part}} -->
Computes the forward window partition.
- **Description**: This function is intended to be called when performing computations that require partitioning a tensor into windows. It should be invoked with valid parameters after ensuring that the destination tensor is properly initialized and has a source tensor of type `GGML_TYPE_F32`. If the source tensor type is not supported, the function will abort execution, indicating a fatal error.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that will hold the result of the computation. Must not be null and must have a valid source tensor of type `GGML_TYPE_F32`.
- **Output**: None
- **See also**: [`ggml_compute_forward_win_part`](ops.cpp.driver.md#ggml_compute_forward_win_part)  (Implementation)


---
### ggml\_compute\_forward\_win\_unpart<!-- {{#callable_declaration:ggml_compute_forward_win_unpart}} -->
Computes the forward operation for a window without partitioning.
- **Description**: This function is intended to be called when performing a forward computation on a tensor that represents a window operation without partitioning. It requires that the `dst` tensor has a source tensor of type `GGML_TYPE_F32`. If the source tensor type is not supported, the function will abort execution. Ensure that the `params` and `dst` pointers are valid and properly initialized before calling this function.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that will hold the result of the computation. Must not be null and must have a valid source tensor of type `GGML_TYPE_F32`.
- **Output**: None
- **See also**: [`ggml_compute_forward_win_unpart`](ops.cpp.driver.md#ggml_compute_forward_win_unpart)  (Implementation)


---
### ggml\_compute\_forward\_unary<!-- {{#callable_declaration:ggml_compute_forward_unary}} -->
Computes the result of a unary operation on a tensor.
- **Description**: This function is used to perform a unary operation on a specified tensor, which is determined by the operation type associated with the tensor. It should be called after ensuring that the tensor is properly initialized and that the operation type is valid. The function will handle various unary operations such as absolute value, negation, and activation functions like ReLU and sigmoid. If an invalid operation type is detected, the function will abort execution.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the result will be stored. Must not be null and must have a valid unary operation type.
- **Output**: None
- **See also**: [`ggml_compute_forward_unary`](ops.cpp.driver.md#ggml_compute_forward_unary)  (Implementation)


---
### ggml\_compute\_forward\_get\_rel\_pos<!-- {{#callable_declaration:ggml_compute_forward_get_rel_pos}} -->
Computes the relative position for a tensor.
- **Description**: This function is intended to be called when you need to compute the relative position of elements in a tensor. It requires that the `dst` tensor has a source tensor of type `GGML_TYPE_F16` or `GGML_TYPE_BF16`. If the source tensor type is not supported, the function will abort execution. Ensure that the `params` and `dst` pointers are valid and properly initialized before calling this function.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure where the result will be stored. Must not be null and must have a valid source tensor of type `GGML_TYPE_F16` or `GGML_TYPE_BF16`.
- **Output**: None
- **See also**: [`ggml_compute_forward_get_rel_pos`](ops.cpp.driver.md#ggml_compute_forward_get_rel_pos)  (Implementation)


---
### ggml\_compute\_forward\_add\_rel\_pos<!-- {{#callable_declaration:ggml_compute_forward_add_rel_pos}} -->
Computes the forward addition of relative positions.
- **Description**: This function is intended to be called when you need to compute the addition of relative positions in a tensor. It requires that the `dst` tensor has a source tensor of type `GGML_TYPE_F32`. If the source tensor type is not supported, the function will abort execution. Ensure that the `params` and `dst` pointers are valid and properly initialized before calling this function.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that will hold the result of the computation. Must not be null and must have a valid source tensor of type `GGML_TYPE_F32`.
- **Output**: None
- **See also**: [`ggml_compute_forward_add_rel_pos`](ops.cpp.driver.md#ggml_compute_forward_add_rel_pos)  (Implementation)


---
### ggml\_compute\_forward\_rwkv\_wkv6<!-- {{#callable_declaration:ggml_compute_forward_rwkv_wkv6}} -->
Computes the forward RWKV WKV6 operation.
- **Description**: This function is intended to perform the forward RWKV WKV6 computation on a given destination tensor. It should be called with valid parameters after ensuring that the source tensor type is compatible, specifically of type `GGML_TYPE_F32`. If the source tensor type is not supported, the function will abort execution. It is crucial to ensure that the `params` and `dst` pointers are valid and properly initialized before calling this function.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that will hold the result of the computation. Must not be null and should have a valid source tensor of type `GGML_TYPE_F32`.
- **Output**: This function does not return a value and does not mutate any inputs directly. The result of the computation is stored in the `dst` tensor.
- **See also**: [`ggml_compute_forward_rwkv_wkv6`](ops.cpp.driver.md#ggml_compute_forward_rwkv_wkv6)  (Implementation)


---
### ggml\_compute\_forward\_rwkv\_wkv7<!-- {{#callable_declaration:ggml_compute_forward_rwkv_wkv7}} -->
Computes the forward pass for RWKV WKV7.
- **Description**: This function is intended to be called during the forward computation phase of a neural network model that utilizes the RWKV architecture. It requires a valid `ggml_compute_params` structure to specify computation parameters and a destination tensor (`dst`) where the results will be stored. The function expects the source tensor of `dst` to be of type `GGML_TYPE_F32`. If the source tensor type is unsupported, the function will abort execution. It is important to ensure that the `dst` tensor is properly initialized and that its source tensor is set before calling this function.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` where the results of the computation will be stored. Must not be null and must have a valid source tensor of type `GGML_TYPE_F32`.
- **Output**: Returns nothing. The output is written to the `dst` tensor, which will contain the results of the forward computation.
- **See also**: [`ggml_compute_forward_rwkv_wkv7`](ops.cpp.driver.md#ggml_compute_forward_rwkv_wkv7)  (Implementation)


---
### ggml\_compute\_forward\_gla<!-- {{#callable_declaration:ggml_compute_forward_gla}} -->
Computes the forward pass for the GLA operation.
- **Description**: This function is intended to be called when performing a forward computation for the GLA operation on a tensor. It requires that the `dst` tensor has a source tensor of type `GGML_TYPE_F32`. If the source tensor type is not supported, the function will abort execution. Ensure that the `params` and `dst` pointers are valid and properly initialized before calling this function.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that will hold the result of the computation. Must not be null and must have a valid source tensor of type `GGML_TYPE_F32`.
- **Output**: None
- **See also**: [`ggml_compute_forward_gla`](ops.cpp.driver.md#ggml_compute_forward_gla)  (Implementation)


---
### ggml\_compute\_forward\_map\_custom1<!-- {{#callable_declaration:ggml_compute_forward_map_custom1}} -->
Computes a custom mapping operation.
- **Description**: This function is intended to perform a custom mapping operation on a tensor, utilizing parameters provided in the `ggml_compute_params` structure. It should be called when you need to apply a specific mapping function defined in the tensor's operation parameters. Ensure that the `dst` tensor has been properly initialized and that its source tensor is valid. The function will invoke the mapping function specified in the operation parameters, which may have side effects depending on the implementation of that function.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains computation parameters. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` structure that will receive the result of the mapping operation. Must not be null and should be properly initialized with a valid source tensor.
- **Output**: Returns nothing and directly modifies the `dst` tensor based on the custom mapping operation.
- **See also**: [`ggml_compute_forward_map_custom1`](ops.cpp.driver.md#ggml_compute_forward_map_custom1)  (Implementation)


---
### ggml\_compute\_forward\_map\_custom2<!-- {{#callable_declaration:ggml_compute_forward_map_custom2}} -->
Computes a custom mapping operation.
- **Description**: This function is intended to perform a custom mapping operation on the provided destination tensor using two source tensors. It should be called when the destination tensor has been properly initialized and is ready to receive the results of the mapping operation. The function expects that the source tensors are valid and properly set up in the destination tensor's `src` array. It is important to ensure that the `params` structure is correctly populated before calling this function, as it contains necessary parameters for the computation.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the result of the mapping operation. Must not be null and should have its `src` array populated with valid source tensors.
- **Output**: None
- **See also**: [`ggml_compute_forward_map_custom2`](ops.cpp.driver.md#ggml_compute_forward_map_custom2)  (Implementation)


---
### ggml\_compute\_forward\_map\_custom3<!-- {{#callable_declaration:ggml_compute_forward_map_custom3}} -->
Computes a custom mapping operation.
- **Description**: This function is intended to perform a custom mapping operation on tensors, utilizing the provided computation parameters. It should be called when you need to apply a specific operation defined by the user, which is indicated by the parameters in the `ggml_tensor` structure. Ensure that the `dst` tensor has at least three source tensors set, as the function expects to access them for computation. The behavior of the function is dependent on the custom operation defined in the `dst` tensor's operation parameters.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains computation parameters. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` structure that will hold the result of the computation. Must not be null and must have at least three source tensors set.
- **Output**: Returns nothing and mutates the `dst` tensor to hold the result of the custom mapping operation.
- **See also**: [`ggml_compute_forward_map_custom3`](ops.cpp.driver.md#ggml_compute_forward_map_custom3)  (Implementation)


---
### ggml\_compute\_forward\_custom<!-- {{#callable_declaration:ggml_compute_forward_custom}} -->
Computes a custom operation on a tensor.
- **Description**: This function is intended to perform a custom operation on a specified tensor, utilizing parameters provided in a `ggml_compute_params` structure. It should be called when you need to apply a user-defined operation to a tensor, ensuring that the `params` and `dst` are properly initialized beforehand. The function expects that the `dst` tensor has valid operation parameters set, and it will invoke the custom function defined in those parameters. If the input parameters are invalid or the operation cannot be performed, the behavior is undefined.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null and should be properly initialized before calling this function.
    - `dst`: A pointer to a `ggml_tensor` structure that represents the destination tensor where the operation will be applied. Must not be null and should have valid operation parameters set.
- **Output**: None
- **See also**: [`ggml_compute_forward_custom`](ops.cpp.driver.md#ggml_compute_forward_custom)  (Implementation)


---
### ggml\_compute\_forward\_cross\_entropy\_loss<!-- {{#callable_declaration:ggml_compute_forward_cross_entropy_loss}} -->
Computes the forward cross-entropy loss.
- **Description**: This function is used to calculate the forward cross-entropy loss between predicted and actual values, which is commonly used in classification tasks. It should be called with a valid `ggml_tensor` that contains the source data for the loss computation. The function expects the destination tensor to be properly initialized and to have a source tensor of type `GGML_TYPE_F32`. If the source tensor type is not supported, the function will abort execution, indicating a fatal error.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure that contains parameters for the computation. Must not be null.
    - `dst`: A pointer to a `ggml_tensor` that will hold the result of the loss computation. Must not be null and should be properly initialized with a source tensor of type `GGML_TYPE_F32`.
- **Output**: Returns nothing. The result of the computation is stored in the `dst` tensor.
- **See also**: [`ggml_compute_forward_cross_entropy_loss`](ops.cpp.driver.md#ggml_compute_forward_cross_entropy_loss)  (Implementation)


---
### ggml\_compute\_forward\_cross\_entropy\_loss\_back<!-- {{#callable_declaration:ggml_compute_forward_cross_entropy_loss_back}} -->
Computes the backward pass of the cross-entropy loss.
- **Description**: This function is intended to be called during the backward pass of a neural network training process, specifically to compute gradients for the cross-entropy loss. It should be invoked after the forward pass has been completed and the destination tensor has been properly set up. The function expects the source tensor to be of type `GGML_TYPE_F32`, and it will abort if the tensor type is unsupported. Ensure that the `params` and `dst` pointers are valid and properly initialized before calling this function.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` that will receive the computed gradients. Must not be null and should be properly initialized to hold the results.
- **Output**: None
- **See also**: [`ggml_compute_forward_cross_entropy_loss_back`](ops.cpp.driver.md#ggml_compute_forward_cross_entropy_loss_back)  (Implementation)


---
### ggml\_compute\_forward\_opt\_step\_adamw<!-- {{#callable_declaration:ggml_compute_forward_opt_step_adamw}} -->
Computes the forward step of the AdamW optimization algorithm.
- **Description**: This function is intended to be called during the optimization process to update the destination tensor using the AdamW algorithm. It should be invoked after the necessary initialization of the `ggml_compute_params` and the source tensor. The function expects the source tensor to be of type `GGML_TYPE_F32`, and it will abort if the tensor type is unsupported. Ensure that the destination tensor is properly allocated and that the parameters are valid before calling this function.
- **Inputs**:
    - `params`: Pointer to a `ggml_compute_params` structure that contains the parameters for the computation. Must not be null.
    - `dst`: Pointer to a `ggml_tensor` that will receive the result of the computation. Must not be null and should be properly initialized.
- **Output**: None
- **See also**: [`ggml_compute_forward_opt_step_adamw`](ops.cpp.driver.md#ggml_compute_forward_opt_step_adamw)  (Implementation)


