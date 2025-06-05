# Purpose
This code is a C header file that declares functions related to tensor operations, likely for a machine learning or numerical computation library. The functions are designed to interact with a backend, possibly utilizing AMX (Advanced Matrix Extensions) for optimized matrix operations. The header includes declarations for functions that determine desired workspace size, calculate allocation size for tensors, convert tensor weights, and perform matrix multiplication. The inclusion of `#pragma once` ensures the file is only included once during compilation, preventing duplicate definitions.
# Imports and Dependencies

---
- `common.h`


# Function Declarations (Public API)

---
### ggml\_backend\_amx\_desired\_wsize<!-- {{#callable_declaration:ggml_backend_amx_desired_wsize}} -->
Calculate the desired workspace size for AMX backend operations on a tensor.
- **Description**: This function computes the desired workspace size needed for AMX backend operations based on the provided destination tensor. It is used to determine the memory requirements for processing the tensor with AMX instructions. The function should be called when preparing to perform operations that require AMX-specific optimizations. It assumes that the destination tensor is properly initialized and that its source tensor is set. The function returns zero if the tensor type is floating-point (specifically GGML_TYPE_F16), indicating no additional workspace is needed for such types.
- **Inputs**:
    - `dst`: A pointer to a ggml_tensor structure representing the destination tensor. The tensor must be initialized and have a valid source tensor. The function does not modify the tensor, and the caller retains ownership. If the tensor type is GGML_TYPE_F16, the function returns zero.
- **Output**: Returns the size in bytes of the desired workspace for the given tensor. If the tensor type is GGML_TYPE_F16, the function returns zero.
- **See also**: [`ggml_backend_amx_desired_wsize`](mmq.cpp.driver.md#ggml_backend_amx_desired_wsize)  (Implementation)


---
### ggml\_backend\_amx\_get\_alloc\_size<!-- {{#callable_declaration:ggml_backend_amx_get_alloc_size}} -->
Calculates the required allocation size for a tensor in the AMX backend.
- **Description**: Use this function to determine the memory allocation size needed for a given tensor when using the AMX backend. This is particularly useful for optimizing memory usage based on the tensor's type and dimensions. The function should be called with a valid tensor structure, and it will return the size in bytes. It handles different tensor types, including those with AMX kernel support, and adjusts the size calculation accordingly.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor for which the allocation size is to be calculated. The tensor must be properly initialized and must not be null. The function expects the tensor to have valid type and dimension information.
- **Output**: Returns the size in bytes required for the tensor's allocation in the AMX backend. The size calculation depends on the tensor's type and dimensions.
- **See also**: [`ggml_backend_amx_get_alloc_size`](mmq.cpp.driver.md#ggml_backend_amx_get_alloc_size)  (Implementation)


---
### ggml\_backend\_amx\_convert\_weight<!-- {{#callable_declaration:ggml_backend_amx_convert_weight}} -->
Convert tensor weights to a packed format for AMX backend.
- **Description**: This function is used to convert the weights of a tensor into a packed format suitable for the AMX backend. It must be called with the intention of converting the entire tensor, as partial conversions are not supported. The function requires that the offset is zero and the size matches the total number of bytes in the tensor. This ensures that the entire tensor is processed in one operation.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor whose weights are to be converted. The tensor must be fully initialized and the conversion will affect its data.
    - `data`: A pointer to the source data that will be used for the conversion. This data should be in a format compatible with the tensor's type.
    - `offset`: A size_t value representing the offset in bytes from the start of the tensor's data. Must be zero, as only full tensor conversion is supported.
    - `size`: A size_t value representing the size in bytes of the data to be converted. Must equal the total number of bytes in the tensor, as determined by `ggml_nbytes(tensor)`, to ensure full conversion.
- **Output**: None
- **See also**: [`ggml_backend_amx_convert_weight`](mmq.cpp.driver.md#ggml_backend_amx_convert_weight)  (Implementation)


---
### ggml\_backend\_amx\_mul\_mat<!-- {{#callable_declaration:ggml_backend_amx_mul_mat}} -->
Performs matrix multiplication using AMX backend.
- **Description**: This function is used to perform matrix multiplication on tensors using the AMX backend. It requires a compute parameters structure and a destination tensor, which also contains the source tensors for the operation. The function supports floating-point types and quantized types, handling them differently based on the type. It is designed to be called in a parallelized context, utilizing multiple threads for computation. The function assumes that the destination tensor is properly initialized and that the source tensors are set as its dependencies. It is important to ensure that the workspace size specified in the compute parameters is sufficient for the operation, as insufficient workspace will result in an error. This function is typically used in high-performance computing scenarios where matrix operations are a bottleneck.
- **Inputs**:
    - `params`: A pointer to a ggml_compute_params structure that contains parameters for the computation, including workspace data and thread information. Must not be null.
    - `dst`: A pointer to a ggml_tensor structure that serves as the destination for the matrix multiplication result. It must have its source tensors set, which are used as the operands for the multiplication. The caller retains ownership.
- **Output**: None
- **See also**: [`ggml_backend_amx_mul_mat`](mmq.cpp.driver.md#ggml_backend_amx_mul_mat)  (Implementation)


