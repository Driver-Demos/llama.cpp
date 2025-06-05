# Purpose
The provided code is a sophisticated implementation of a Metal-based backend for a machine learning library, designed to accelerate tensor operations on Apple devices using the Metal API. It is part of a larger system, likely involving machine learning or high-performance computing tasks, and is structured to handle a variety of operations common in neural network computations, such as matrix multiplication, element-wise operations, sorting, activation functions, and attention mechanisms. The code includes components for device and memory management, kernel definitions, and execution logic, all unified under the theme of GPU-accelerated computation using Metal. It is not a standalone executable but a library intended for integration into larger systems, providing backend functionality for executing tensor operations on Metal-supported devices. The code is highly specialized for optimizing operations on Apple's hardware, making it suitable for applications requiring high-performance processing, such as deep learning frameworks, and includes error handling and logging to ensure robustness and traceability.
# Global Variables

---
### MTLGPUFamilyMetal3\_GGML
- **Type**: `NSInteger`
- **Description**: `MTLGPUFamilyMetal3_GGML` is a static constant of type `NSInteger` with a value of 5001. It serves as an overload for the `MTLGPUFamilyMetal3` enumeration, which may not be available in some environments.
- **Use**: This variable is used to check if a Metal device supports the `MTLGPUFamilyMetal3_GGML` family, which is a custom extension of the Metal GPU family enumeration.


---
### g\_ggml\_backend\_metal\_reg
- **Type**: `struct ggml_backend_reg`
- **Description**: The `g_ggml_backend_metal_reg` is a global variable of type `struct ggml_backend_reg`. It is used to store registration information for the Metal backend in the GGML library. This structure is initialized in the `ggml_backend_metal_reg` function, which likely sets up the necessary configurations and parameters for the Metal backend to function properly.
- **Use**: This variable is used to hold registration details for the Metal backend, facilitating its initialization and configuration within the GGML library.


---
### g\_ggml\_backend\_metal\_device
- **Type**: `struct ggml_backend_device`
- **Description**: The `g_ggml_backend_metal_device` is a global variable of type `struct ggml_backend_device`. It is used to store information about the Metal device being used in the application. This structure is initialized in the `ggml_backend_metal_reg` function and is crucial for managing the Metal backend operations.
- **Use**: This variable is used to manage and store information about the Metal device for backend operations.


---
### g\_ggml\_ctx\_dev\_main
- **Type**: `struct ggml_backend_metal_device_context`
- **Description**: The `g_ggml_ctx_dev_main` variable is a global instance of the `ggml_backend_metal_device_context` structure, which holds information about a Metal device. This structure includes details such as the Metal device itself, reference count, library, and various capabilities like SIMD group reduction, matrix multiplication, residency sets, and bfloat support. It also contains a character array for the device name.
- **Use**: This variable is used to store and manage the context and capabilities of the default Metal GPU device for the application.


# Data Structures

---
### ggml\_backend\_metal\_device\_context
- **Type**: `struct`
- **Members**:
    - `mtl_device`: A reference to the Metal device.
    - `mtl_device_ref_count`: Reference count for the Metal device.
    - `mtl_library`: A reference to the Metal library.
    - `has_simdgroup_reduction`: Indicates if the device supports SIMD group reduction.
    - `has_simdgroup_mm`: Indicates if the device supports SIMD group matrix multiplication.
    - `has_residency_sets`: Indicates if the device supports residency sets.
    - `has_bfloat`: Indicates if the device supports bfloat16 data type.
    - `use_bfloat`: Indicates if bfloat16 is used.
    - `name`: Name of the Metal device.
- **Description**: The `ggml_backend_metal_device_context` structure holds information about a Metal device, specifically for use with the GGML backend. It includes references to the Metal device and library, as well as flags indicating support for various features such as SIMD group operations and bfloat16 data type. The structure also maintains a reference count for the device and stores the device's name.


---
### ggml\_metal\_kernel
- **Type**: `struct`
- **Members**:
    - `pipeline`: Holds the Metal compute pipeline state for the kernel.
- **Description**: The `ggml_metal_kernel` structure is designed to encapsulate a Metal compute pipeline state object, which is used to execute a specific compute kernel on a Metal device. This structure is part of a larger system that leverages Metal for GPU-accelerated computations, particularly in the context of machine learning or other high-performance computing tasks. The `pipeline` member is a reference to the Metal compute pipeline state, which is essential for executing the associated kernel on the GPU.


---
### ggml\_metal\_kernel\_type
- **Type**: `enum`
- **Members**:
    - `GGML_METAL_KERNEL_TYPE_ADD`: Represents the addition kernel type.
    - `GGML_METAL_KERNEL_TYPE_ADD_ROW`: Represents the addition of a row kernel type.
    - `GGML_METAL_KERNEL_TYPE_SUB`: Represents the subtraction kernel type.
    - `GGML_METAL_KERNEL_TYPE_SUB_ROW`: Represents the subtraction of a row kernel type.
    - `GGML_METAL_KERNEL_TYPE_MUL`: Represents the multiplication kernel type.
    - `GGML_METAL_KERNEL_TYPE_MUL_ROW`: Represents the multiplication of a row kernel type.
    - `GGML_METAL_KERNEL_TYPE_DIV`: Represents the division kernel type.
    - `GGML_METAL_KERNEL_TYPE_DIV_ROW`: Represents the division of a row kernel type.
    - `GGML_METAL_KERNEL_TYPE_REPEAT_F32`: Represents the repeat kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_REPEAT_F16`: Represents the repeat kernel type for 16-bit floats.
    - `GGML_METAL_KERNEL_TYPE_REPEAT_I32`: Represents the repeat kernel type for 32-bit integers.
    - `GGML_METAL_KERNEL_TYPE_REPEAT_I16`: Represents the repeat kernel type for 16-bit integers.
    - `GGML_METAL_KERNEL_TYPE_SCALE`: Represents the scaling kernel type.
    - `GGML_METAL_KERNEL_TYPE_SCALE_4`: Represents the scaling kernel type with a factor of 4.
    - `GGML_METAL_KERNEL_TYPE_CLAMP`: Represents the clamping kernel type.
    - `GGML_METAL_KERNEL_TYPE_TANH`: Represents the hyperbolic tangent kernel type.
    - `GGML_METAL_KERNEL_TYPE_RELU`: Represents the ReLU (Rectified Linear Unit) kernel type.
    - `GGML_METAL_KERNEL_TYPE_SIGMOID`: Represents the sigmoid function kernel type.
    - `GGML_METAL_KERNEL_TYPE_GELU`: Represents the Gaussian Error Linear Unit kernel type.
    - `GGML_METAL_KERNEL_TYPE_GELU_4`: Represents the Gaussian Error Linear Unit kernel type with a factor of 4.
    - `GGML_METAL_KERNEL_TYPE_GELU_ERF`: Represents the Gaussian Error Linear Unit with error function kernel type.
    - `GGML_METAL_KERNEL_TYPE_GELU_ERF_4`: Represents the Gaussian Error Linear Unit with error function kernel type with a factor of 4.
    - `GGML_METAL_KERNEL_TYPE_GELU_QUICK`: Represents the quick Gaussian Error Linear Unit kernel type.
    - `GGML_METAL_KERNEL_TYPE_GELU_QUICK_4`: Represents the quick Gaussian Error Linear Unit kernel type with a factor of 4.
    - `GGML_METAL_KERNEL_TYPE_SILU`: Represents the Sigmoid Linear Unit kernel type.
    - `GGML_METAL_KERNEL_TYPE_SILU_4`: Represents the Sigmoid Linear Unit kernel type with a factor of 4.
    - `GGML_METAL_KERNEL_TYPE_ELU`: Represents the Exponential Linear Unit kernel type.
    - `GGML_METAL_KERNEL_TYPE_SOFT_MAX_F16`: Represents the softmax kernel type for 16-bit floats.
    - `GGML_METAL_KERNEL_TYPE_SOFT_MAX_F16_4`: Represents the softmax kernel type for 16-bit floats with a factor of 4.
    - `GGML_METAL_KERNEL_TYPE_SOFT_MAX_F32`: Represents the softmax kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_SOFT_MAX_F32_4`: Represents the softmax kernel type for 32-bit floats with a factor of 4.
    - `GGML_METAL_KERNEL_TYPE_DIAG_MASK_INF`: Represents the diagonal mask with infinity kernel type.
    - `GGML_METAL_KERNEL_TYPE_DIAG_MASK_INF_8`: Represents the diagonal mask with infinity kernel type with a factor of 8.
    - `GGML_METAL_KERNEL_TYPE_GET_ROWS_F32`: Represents the get rows kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_GET_ROWS_F16`: Represents the get rows kernel type for 16-bit floats.
    - `GGML_METAL_KERNEL_TYPE_GET_ROWS_BF16`: Represents the get rows kernel type for bfloat16.
    - `GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_0`: Represents the get rows kernel type for Q4_0 quantization.
    - `GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_1`: Represents the get rows kernel type for Q4_1 quantization.
    - `GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_0`: Represents the get rows kernel type for Q5_0 quantization.
    - `GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_1`: Represents the get rows kernel type for Q5_1 quantization.
    - `GGML_METAL_KERNEL_TYPE_GET_ROWS_Q8_0`: Represents the get rows kernel type for Q8_0 quantization.
    - `GGML_METAL_KERNEL_TYPE_GET_ROWS_Q2_K`: Represents the get rows kernel type for Q2_K quantization.
    - `GGML_METAL_KERNEL_TYPE_GET_ROWS_Q3_K`: Represents the get rows kernel type for Q3_K quantization.
    - `GGML_METAL_KERNEL_TYPE_GET_ROWS_Q4_K`: Represents the get rows kernel type for Q4_K quantization.
    - `GGML_METAL_KERNEL_TYPE_GET_ROWS_Q5_K`: Represents the get rows kernel type for Q5_K quantization.
    - `GGML_METAL_KERNEL_TYPE_GET_ROWS_Q6_K`: Represents the get rows kernel type for Q6_K quantization.
    - `GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_XXS`: Represents the get rows kernel type for IQ2_XXS quantization.
    - `GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_XS`: Represents the get rows kernel type for IQ2_XS quantization.
    - `GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ3_XXS`: Represents the get rows kernel type for IQ3_XXS quantization.
    - `GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ3_S`: Represents the get rows kernel type for IQ3_S quantization.
    - `GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ2_S`: Represents the get rows kernel type for IQ2_S quantization.
    - `GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ1_S`: Represents the get rows kernel type for IQ1_S quantization.
    - `GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ1_M`: Represents the get rows kernel type for IQ1_M quantization.
    - `GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ4_NL`: Represents the get rows kernel type for IQ4_NL quantization.
    - `GGML_METAL_KERNEL_TYPE_GET_ROWS_IQ4_XS`: Represents the get rows kernel type for IQ4_XS quantization.
    - `GGML_METAL_KERNEL_TYPE_GET_ROWS_I32`: Represents the get rows kernel type for 32-bit integers.
    - `GGML_METAL_KERNEL_TYPE_RMS_NORM`: Represents the RMS normalization kernel type.
    - `GGML_METAL_KERNEL_TYPE_L2_NORM`: Represents the L2 normalization kernel type.
    - `GGML_METAL_KERNEL_TYPE_GROUP_NORM`: Represents the group normalization kernel type.
    - `GGML_METAL_KERNEL_TYPE_NORM`: Represents the normalization kernel type.
    - `GGML_METAL_KERNEL_TYPE_SSM_CONV_F32`: Represents the SSM convolution kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_SSM_SCAN_F32`: Represents the SSM scan kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_RWKV_WKV6_F32`: Represents the RWKV WKV6 kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_RWKV_WKV7_F32`: Represents the RWKV WKV7 kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_F32_F32`: Represents the matrix-vector multiplication kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32`: Represents the matrix-vector multiplication kernel type for 16-bit floats and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32_1ROW`: Represents the matrix-vector multiplication kernel type for 16-bit floats and 32-bit floats with one row.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F32_L4`: Represents the matrix-vector multiplication kernel type for 16-bit floats and 32-bit floats with a factor of 4.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_F16_F16`: Represents the matrix-vector multiplication kernel type for 16-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_BF16_F32`: Represents the matrix-vector multiplication kernel type for bfloat16 and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_BF16_F32_1ROW`: Represents the matrix-vector multiplication kernel type for bfloat16 and 32-bit floats with one row.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_BF16_F32_L4`: Represents the matrix-vector multiplication kernel type for bfloat16 and 32-bit floats with a factor of 4.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_BF16_BF16`: Represents the matrix-vector multiplication kernel type for bfloat16.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_0_F32`: Represents the matrix-vector multiplication kernel type for Q4_0 quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_1_F32`: Represents the matrix-vector multiplication kernel type for Q4_1 quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_0_F32`: Represents the matrix-vector multiplication kernel type for Q5_0 quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_1_F32`: Represents the matrix-vector multiplication kernel type for Q5_1 quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_Q8_0_F32`: Represents the matrix-vector multiplication kernel type for Q8_0 quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_F16_F32_R1_2`: Represents the extended matrix-vector multiplication kernel type for 16-bit floats and 32-bit floats with R1_2.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_F16_F32_R1_3`: Represents the extended matrix-vector multiplication kernel type for 16-bit floats and 32-bit floats with R1_3.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_F16_F32_R1_4`: Represents the extended matrix-vector multiplication kernel type for 16-bit floats and 32-bit floats with R1_4.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_F16_F32_R1_5`: Represents the extended matrix-vector multiplication kernel type for 16-bit floats and 32-bit floats with R1_5.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_0_F32_R1_2`: Represents the extended matrix-vector multiplication kernel type for Q4_0 quantization and 32-bit floats with R1_2.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_0_F32_R1_3`: Represents the extended matrix-vector multiplication kernel type for Q4_0 quantization and 32-bit floats with R1_3.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_0_F32_R1_4`: Represents the extended matrix-vector multiplication kernel type for Q4_0 quantization and 32-bit floats with R1_4.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_0_F32_R1_5`: Represents the extended matrix-vector multiplication kernel type for Q4_0 quantization and 32-bit floats with R1_5.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_1_F32_R1_2`: Represents the extended matrix-vector multiplication kernel type for Q4_1 quantization and 32-bit floats with R1_2.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_1_F32_R1_3`: Represents the extended matrix-vector multiplication kernel type for Q4_1 quantization and 32-bit floats with R1_3.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_1_F32_R1_4`: Represents the extended matrix-vector multiplication kernel type for Q4_1 quantization and 32-bit floats with R1_4.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_1_F32_R1_5`: Represents the extended matrix-vector multiplication kernel type for Q4_1 quantization and 32-bit floats with R1_5.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_0_F32_R1_2`: Represents the extended matrix-vector multiplication kernel type for Q5_0 quantization and 32-bit floats with R1_2.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_0_F32_R1_3`: Represents the extended matrix-vector multiplication kernel type for Q5_0 quantization and 32-bit floats with R1_3.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_0_F32_R1_4`: Represents the extended matrix-vector multiplication kernel type for Q5_0 quantization and 32-bit floats with R1_4.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_0_F32_R1_5`: Represents the extended matrix-vector multiplication kernel type for Q5_0 quantization and 32-bit floats with R1_5.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_1_F32_R1_2`: Represents the extended matrix-vector multiplication kernel type for Q5_1 quantization and 32-bit floats with R1_2.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_1_F32_R1_3`: Represents the extended matrix-vector multiplication kernel type for Q5_1 quantization and 32-bit floats with R1_3.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_1_F32_R1_4`: Represents the extended matrix-vector multiplication kernel type for Q5_1 quantization and 32-bit floats with R1_4.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_1_F32_R1_5`: Represents the extended matrix-vector multiplication kernel type for Q5_1 quantization and 32-bit floats with R1_5.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q8_0_F32_R1_2`: Represents the extended matrix-vector multiplication kernel type for Q8_0 quantization and 32-bit floats with R1_2.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q8_0_F32_R1_3`: Represents the extended matrix-vector multiplication kernel type for Q8_0 quantization and 32-bit floats with R1_3.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q8_0_F32_R1_4`: Represents the extended matrix-vector multiplication kernel type for Q8_0 quantization and 32-bit floats with R1_4.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q8_0_F32_R1_5`: Represents the extended matrix-vector multiplication kernel type for Q8_0 quantization and 32-bit floats with R1_5.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_K_F32_R1_2`: Represents the extended matrix-vector multiplication kernel type for Q4_K quantization and 32-bit floats with R1_2.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_K_F32_R1_3`: Represents the extended matrix-vector multiplication kernel type for Q4_K quantization and 32-bit floats with R1_3.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_K_F32_R1_4`: Represents the extended matrix-vector multiplication kernel type for Q4_K quantization and 32-bit floats with R1_4.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q4_K_F32_R1_5`: Represents the extended matrix-vector multiplication kernel type for Q4_K quantization and 32-bit floats with R1_5.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_K_F32_R1_2`: Represents the extended matrix-vector multiplication kernel type for Q5_K quantization and 32-bit floats with R1_2.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_K_F32_R1_3`: Represents the extended matrix-vector multiplication kernel type for Q5_K quantization and 32-bit floats with R1_3.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_K_F32_R1_4`: Represents the extended matrix-vector multiplication kernel type for Q5_K quantization and 32-bit floats with R1_4.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q5_K_F32_R1_5`: Represents the extended matrix-vector multiplication kernel type for Q5_K quantization and 32-bit floats with R1_5.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q6_K_F32_R1_2`: Represents the extended matrix-vector multiplication kernel type for Q6_K quantization and 32-bit floats with R1_2.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q6_K_F32_R1_3`: Represents the extended matrix-vector multiplication kernel type for Q6_K quantization and 32-bit floats with R1_3.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q6_K_F32_R1_4`: Represents the extended matrix-vector multiplication kernel type for Q6_K quantization and 32-bit floats with R1_4.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_Q6_K_F32_R1_5`: Represents the extended matrix-vector multiplication kernel type for Q6_K quantization and 32-bit floats with R1_5.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_2`: Represents the extended matrix-vector multiplication kernel type for IQ4_NL quantization and 32-bit floats with R1_2.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_3`: Represents the extended matrix-vector multiplication kernel type for IQ4_NL quantization and 32-bit floats with R1_3.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_4`: Represents the extended matrix-vector multiplication kernel type for IQ4_NL quantization and 32-bit floats with R1_4.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_EXT_IQ4_NL_F32_R1_5`: Represents the extended matrix-vector multiplication kernel type for IQ4_NL quantization and 32-bit floats with R1_5.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_Q2_K_F32`: Represents the matrix-vector multiplication kernel type for Q2_K quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_Q3_K_F32`: Represents the matrix-vector multiplication kernel type for Q3_K quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_Q4_K_F32`: Represents the matrix-vector multiplication kernel type for Q4_K quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_Q5_K_F32`: Represents the matrix-vector multiplication kernel type for Q5_K quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_Q6_K_F32`: Represents the matrix-vector multiplication kernel type for Q6_K quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_XXS_F32`: Represents the matrix-vector multiplication kernel type for IQ2_XXS quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_XS_F32`: Represents the matrix-vector multiplication kernel type for IQ2_XS quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_IQ3_XXS_F32`: Represents the matrix-vector multiplication kernel type for IQ3_XXS quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_IQ3_S_F32`: Represents the matrix-vector multiplication kernel type for IQ3_S quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_IQ2_S_F32`: Represents the matrix-vector multiplication kernel type for IQ2_S quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_IQ1_S_F32`: Represents the matrix-vector multiplication kernel type for IQ1_S quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_IQ1_M_F32`: Represents the matrix-vector multiplication kernel type for IQ1_M quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_IQ4_NL_F32`: Represents the matrix-vector multiplication kernel type for IQ4_NL quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_IQ4_XS_F32`: Represents the matrix-vector multiplication kernel type for IQ4_XS quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F32_F32`: Represents the matrix-vector multiplication kernel type for 32-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_ID_F16_F32`: Represents the matrix-vector multiplication kernel type for 16-bit floats and 32-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_ID_BF16_F32`: Represents the matrix-vector multiplication kernel type for bfloat16 and 32-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_0_F32`: Represents the matrix-vector multiplication kernel type for Q4_0 quantization and 32-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_1_F32`: Represents the matrix-vector multiplication kernel type for Q4_1 quantization and 32-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_0_F32`: Represents the matrix-vector multiplication kernel type for Q5_0 quantization and 32-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_1_F32`: Represents the matrix-vector multiplication kernel type for Q5_1 quantization and 32-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q8_0_F32`: Represents the matrix-vector multiplication kernel type for Q8_0 quantization and 32-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q2_K_F32`: Represents the matrix-vector multiplication kernel type for Q2_K quantization and 32-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q3_K_F32`: Represents the matrix-vector multiplication kernel type for Q3_K quantization and 32-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q4_K_F32`: Represents the matrix-vector multiplication kernel type for Q4_K quantization and 32-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q5_K_F32`: Represents the matrix-vector multiplication kernel type for Q5_K quantization and 32-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_ID_Q6_K_F32`: Represents the matrix-vector multiplication kernel type for Q6_K quantization and 32-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_XXS_F32`: Represents the matrix-vector multiplication kernel type for IQ2_XXS quantization and 32-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_XS_F32`: Represents the matrix-vector multiplication kernel type for IQ2_XS quantization and 32-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ3_XXS_F32`: Represents the matrix-vector multiplication kernel type for IQ3_XXS quantization and 32-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ3_S_F32`: Represents the matrix-vector multiplication kernel type for IQ3_S quantization and 32-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ2_S_F32`: Represents the matrix-vector multiplication kernel type for IQ2_S quantization and 32-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ1_S_F32`: Represents the matrix-vector multiplication kernel type for IQ1_S quantization and 32-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ1_M_F32`: Represents the matrix-vector multiplication kernel type for IQ1_M quantization and 32-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ4_NL_F32`: Represents the matrix-vector multiplication kernel type for IQ4_NL quantization and 32-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MV_ID_IQ4_XS_F32`: Represents the matrix-vector multiplication kernel type for IQ4_XS quantization and 32-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_F32_F32`: Represents the matrix-matrix multiplication kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_F16_F32`: Represents the matrix-matrix multiplication kernel type for 16-bit floats and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_BF16_F32`: Represents the matrix-matrix multiplication kernel type for bfloat16 and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_0_F32`: Represents the matrix-matrix multiplication kernel type for Q4_0 quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_1_F32`: Represents the matrix-matrix multiplication kernel type for Q4_1 quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_0_F32`: Represents the matrix-matrix multiplication kernel type for Q5_0 quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_1_F32`: Represents the matrix-matrix multiplication kernel type for Q5_1 quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_Q8_0_F32`: Represents the matrix-matrix multiplication kernel type for Q8_0 quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_Q2_K_F32`: Represents the matrix-matrix multiplication kernel type for Q2_K quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_Q3_K_F32`: Represents the matrix-matrix multiplication kernel type for Q3_K quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_Q4_K_F32`: Represents the matrix-matrix multiplication kernel type for Q4_K quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_Q5_K_F32`: Represents the matrix-matrix multiplication kernel type for Q5_K quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_Q6_K_F32`: Represents the matrix-matrix multiplication kernel type for Q6_K quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_XXS_F32`: Represents the matrix-matrix multiplication kernel type for IQ2_XXS quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_XS_F32`: Represents the matrix-matrix multiplication kernel type for IQ2_XS quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_IQ3_XXS_F32`: Represents the matrix-matrix multiplication kernel type for IQ3_XXS quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_IQ3_S_F32`: Represents the matrix-matrix multiplication kernel type for IQ3_S quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_IQ2_S_F32`: Represents the matrix-matrix multiplication kernel type for IQ2_S quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_IQ1_S_F32`: Represents the matrix-matrix multiplication kernel type for IQ1_S quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_IQ1_M_F32`: Represents the matrix-matrix multiplication kernel type for IQ1_M quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_IQ4_NL_F32`: Represents the matrix-matrix multiplication kernel type for IQ4_NL quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_IQ4_XS_F32`: Represents the matrix-matrix multiplication kernel type for IQ4_XS quantization and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_MAP0_F16`: Represents the matrix-matrix multiplication kernel type for 16-bit floats with ID map 0.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_MAP1_F32`: Represents the matrix-matrix multiplication kernel type for 32-bit floats with ID map 1.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_F32_F16`: Represents the matrix-matrix multiplication kernel type for 32-bit floats and 16-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_F16_F16`: Represents the matrix-matrix multiplication kernel type for 16-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_BF16_F16`: Represents the matrix-matrix multiplication kernel type for bfloat16 and 16-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_0_F16`: Represents the matrix-matrix multiplication kernel type for Q4_0 quantization and 16-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_1_F16`: Represents the matrix-matrix multiplication kernel type for Q4_1 quantization and 16-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_0_F16`: Represents the matrix-matrix multiplication kernel type for Q5_0 quantization and 16-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_1_F16`: Represents the matrix-matrix multiplication kernel type for Q5_1 quantization and 16-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q8_0_F16`: Represents the matrix-matrix multiplication kernel type for Q8_0 quantization and 16-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q2_K_F16`: Represents the matrix-matrix multiplication kernel type for Q2_K quantization and 16-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q3_K_F16`: Represents the matrix-matrix multiplication kernel type for Q3_K quantization and 16-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q4_K_F16`: Represents the matrix-matrix multiplication kernel type for Q4_K quantization and 16-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q5_K_F16`: Represents the matrix-matrix multiplication kernel type for Q5_K quantization and 16-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_Q6_K_F16`: Represents the matrix-matrix multiplication kernel type for Q6_K quantization and 16-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_XXS_F16`: Represents the matrix-matrix multiplication kernel type for IQ2_XXS quantization and 16-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_XS_F16`: Represents the matrix-matrix multiplication kernel type for IQ2_XS quantization and 16-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ3_XXS_F16`: Represents the matrix-matrix multiplication kernel type for IQ3_XXS quantization and 16-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ3_S_F16`: Represents the matrix-matrix multiplication kernel type for IQ3_S quantization and 16-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ2_S_F16`: Represents the matrix-matrix multiplication kernel type for IQ2_S quantization and 16-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ1_S_F16`: Represents the matrix-matrix multiplication kernel type for IQ1_S quantization and 16-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ1_M_F16`: Represents the matrix-matrix multiplication kernel type for IQ1_M quantization and 16-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ4_NL_F16`: Represents the matrix-matrix multiplication kernel type for IQ4_NL quantization and 16-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_MUL_MM_ID_IQ4_XS_F16`: Represents the matrix-matrix multiplication kernel type for IQ4_XS quantization and 16-bit floats with identity.
    - `GGML_METAL_KERNEL_TYPE_ROPE_NORM_F32`: Represents the ROPE normalization kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_ROPE_NORM_F16`: Represents the ROPE normalization kernel type for 16-bit floats.
    - `GGML_METAL_KERNEL_TYPE_ROPE_MULTI_F32`: Represents the ROPE multi kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_ROPE_MULTI_F16`: Represents the ROPE multi kernel type for 16-bit floats.
    - `GGML_METAL_KERNEL_TYPE_ROPE_VISION_F32`: Represents the ROPE vision kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_ROPE_VISION_F16`: Represents the ROPE vision kernel type for 16-bit floats.
    - `GGML_METAL_KERNEL_TYPE_ROPE_NEOX_F32`: Represents the ROPE NEOX kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_ROPE_NEOX_F16`: Represents the ROPE NEOX kernel type for 16-bit floats.
    - `GGML_METAL_KERNEL_TYPE_IM2COL_F16`: Represents the im2col kernel type for 16-bit floats.
    - `GGML_METAL_KERNEL_TYPE_IM2COL_F32`: Represents the im2col kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_IM2COL_EXT_F16`: Represents the extended im2col kernel type for 16-bit floats.
    - `GGML_METAL_KERNEL_TYPE_IM2COL_EXT_F32`: Represents the extended im2col kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_CONV_TRANSPOSE_1D_F32_F32`: Represents the 1D convolution transpose kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_CONV_TRANSPOSE_1D_F16_F32`: Represents the 1D convolution transpose kernel type for 16-bit floats and 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_UPSCALE_F32`: Represents the upscale kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_PAD_F32`: Represents the padding kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_PAD_REFLECT_1D_F32`: Represents the 1D reflective padding kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_ARANGE_F32`: Represents the arange kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_TIMESTEP_EMBEDDING_F32`: Represents the timestep embedding kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_ARGSORT_F32_I32_ASC`: Represents the ascending argsort kernel type for 32-bit floats and 32-bit integers.
    - `GGML_METAL_KERNEL_TYPE_ARGSORT_F32_I32_DESC`: Represents the descending argsort kernel type for 32-bit floats and 32-bit integers.
    - `GGML_METAL_KERNEL_TYPE_LEAKY_RELU_F32`: Represents the leaky ReLU kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H64`: Represents the extended flash attention kernel type for 16-bit floats with H64.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H80`: Represents the extended flash attention kernel type for 16-bit floats with H80.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H96`: Represents the extended flash attention kernel type for 16-bit floats with H96.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H112`: Represents the extended flash attention kernel type for 16-bit floats with H112.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H128`: Represents the extended flash attention kernel type for 16-bit floats with H128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H192`: Represents the extended flash attention kernel type for 16-bit floats with H192.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_HK192_HV128`: Represents the extended flash attention kernel type for 16-bit floats with HK192 and HV128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H256`: Represents the extended flash attention kernel type for 16-bit floats with H256.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_HK576_HV512`: Represents the extended flash attention kernel type for 16-bit floats with HK576 and HV512.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H64`: Represents the extended flash attention kernel type for bfloat16 with H64.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H80`: Represents the extended flash attention kernel type for bfloat16 with H80.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H96`: Represents the extended flash attention kernel type for bfloat16 with H96.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H112`: Represents the extended flash attention kernel type for bfloat16 with H112.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H128`: Represents the extended flash attention kernel type for bfloat16 with H128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H192`: Represents the extended flash attention kernel type for bfloat16 with H192.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_HK192_HV128`: Represents the extended flash attention kernel type for bfloat16 with HK192 and HV128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_H256`: Represents the extended flash attention kernel type for bfloat16 with H256.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_BF16_HK576_HV512`: Represents the extended flash attention kernel type for bfloat16 with HK576 and HV512.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H64`: Represents the extended flash attention kernel type for Q4_0 quantization with H64.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H80`: Represents the extended flash attention kernel type for Q4_0 quantization with H80.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H96`: Represents the extended flash attention kernel type for Q4_0 quantization with H96.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H112`: Represents the extended flash attention kernel type for Q4_0 quantization with H112.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H128`: Represents the extended flash attention kernel type for Q4_0 quantization with H128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H192`: Represents the extended flash attention kernel type for Q4_0 quantization with H192.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_HK192_HV128`: Represents the extended flash attention kernel type for Q4_0 quantization with HK192 and HV128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_H256`: Represents the extended flash attention kernel type for Q4_0 quantization with H256.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_0_HK576_HV512`: Represents the extended flash attention kernel type for Q4_0 quantization with HK576 and HV512.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H64`: Represents the extended flash attention kernel type for Q4_1 quantization with H64.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H80`: Represents the extended flash attention kernel type for Q4_1 quantization with H80.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H96`: Represents the extended flash attention kernel type for Q4_1 quantization with H96.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H112`: Represents the extended flash attention kernel type for Q4_1 quantization with H112.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H128`: Represents the extended flash attention kernel type for Q4_1 quantization with H128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H192`: Represents the extended flash attention kernel type for Q4_1 quantization with H192.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_HK192_HV128`: Represents the extended flash attention kernel type for Q4_1 quantization with HK192 and HV128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_H256`: Represents the extended flash attention kernel type for Q4_1 quantization with H256.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q4_1_HK576_HV512`: Represents the extended flash attention kernel type for Q4_1 quantization with HK576 and HV512.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H64`: Represents the extended flash attention kernel type for Q5_0 quantization with H64.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H80`: Represents the extended flash attention kernel type for Q5_0 quantization with H80.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H96`: Represents the extended flash attention kernel type for Q5_0 quantization with H96.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H112`: Represents the extended flash attention kernel type for Q5_0 quantization with H112.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H128`: Represents the extended flash attention kernel type for Q5_0 quantization with H128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H192`: Represents the extended flash attention kernel type for Q5_0 quantization with H192.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_HK192_HV128`: Represents the extended flash attention kernel type for Q5_0 quantization with HK192 and HV128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_H256`: Represents the extended flash attention kernel type for Q5_0 quantization with H256.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_0_HK576_HV512`: Represents the extended flash attention kernel type for Q5_0 quantization with HK576 and HV512.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H64`: Represents the extended flash attention kernel type for Q5_1 quantization with H64.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H80`: Represents the extended flash attention kernel type for Q5_1 quantization with H80.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H96`: Represents the extended flash attention kernel type for Q5_1 quantization with H96.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H112`: Represents the extended flash attention kernel type for Q5_1 quantization with H112.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H128`: Represents the extended flash attention kernel type for Q5_1 quantization with H128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H192`: Represents the extended flash attention kernel type for Q5_1 quantization with H192.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_HK192_HV128`: Represents the extended flash attention kernel type for Q5_1 quantization with HK192 and HV128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_H256`: Represents the extended flash attention kernel type for Q5_1 quantization with H256.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q5_1_HK576_HV512`: Represents the extended flash attention kernel type for Q5_1 quantization with HK576 and HV512.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H64`: Represents the extended flash attention kernel type for Q8_0 quantization with H64.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H80`: Represents the extended flash attention kernel type for Q8_0 quantization with H80.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H96`: Represents the extended flash attention kernel type for Q8_0 quantization with H96.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H112`: Represents the extended flash attention kernel type for Q8_0 quantization with H112.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H128`: Represents the extended flash attention kernel type for Q8_0 quantization with H128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H192`: Represents the extended flash attention kernel type for Q8_0 quantization with H192.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_HK192_HV128`: Represents the extended flash attention kernel type for Q8_0 quantization with HK192 and HV128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H256`: Represents the extended flash attention kernel type for Q8_0 quantization with H256.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_HK576_HV512`: Represents the extended flash attention kernel type for Q8_0 quantization with HK576 and HV512.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_F16_H64`: Represents the extended vector flash attention kernel type for 16-bit floats with H64.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_BF16_H64`: Represents the extended vector flash attention kernel type for bfloat16 with H64.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_0_H64`: Represents the extended vector flash attention kernel type for Q4_0 quantization with H64.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_1_H64`: Represents the extended vector flash attention kernel type for Q4_1 quantization with H64.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_0_H64`: Represents the extended vector flash attention kernel type for Q5_0 quantization with H64.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_1_H64`: Represents the extended vector flash attention kernel type for Q5_1 quantization with H64.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q8_0_H64`: Represents the extended vector flash attention kernel type for Q8_0 quantization with H64.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_F16_H96`: Represents the extended vector flash attention kernel type for 16-bit floats with H96.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_BF16_H96`: Represents the extended vector flash attention kernel type for bfloat16 with H96.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_0_H96`: Represents the extended vector flash attention kernel type for Q4_0 quantization with H96.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_1_H96`: Represents the extended vector flash attention kernel type for Q4_1 quantization with H96.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_0_H96`: Represents the extended vector flash attention kernel type for Q5_0 quantization with H96.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_1_H96`: Represents the extended vector flash attention kernel type for Q5_1 quantization with H96.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q8_0_H96`: Represents the extended vector flash attention kernel type for Q8_0 quantization with H96.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_F16_H128`: Represents the extended vector flash attention kernel type for 16-bit floats with H128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_BF16_H128`: Represents the extended vector flash attention kernel type for bfloat16 with H128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_0_H128`: Represents the extended vector flash attention kernel type for Q4_0 quantization with H128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_1_H128`: Represents the extended vector flash attention kernel type for Q4_1 quantization with H128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_0_H128`: Represents the extended vector flash attention kernel type for Q5_0 quantization with H128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_1_H128`: Represents the extended vector flash attention kernel type for Q5_1 quantization with H128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q8_0_H128`: Represents the extended vector flash attention kernel type for Q8_0 quantization with H128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_F16_H192`: Represents the extended vector flash attention kernel type for 16-bit floats with H192.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_BF16_H192`: Represents the extended vector flash attention kernel type for bfloat16 with H192.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_0_H192`: Represents the extended vector flash attention kernel type for Q4_0 quantization with H192.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_1_H192`: Represents the extended vector flash attention kernel type for Q4_1 quantization with H192.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_0_H192`: Represents the extended vector flash attention kernel type for Q5_0 quantization with H192.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_1_H192`: Represents the extended vector flash attention kernel type for Q5_1 quantization with H192.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q8_0_H192`: Represents the extended vector flash attention kernel type for Q8_0 quantization with H192.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_F16_HK192_HV128`: Represents the extended vector flash attention kernel type for 16-bit floats with HK192 and HV128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_BF16_HK192_HV128`: Represents the extended vector flash attention kernel type for bfloat16 with HK192 and HV128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_0_HK192_HV128`: Represents the extended vector flash attention kernel type for Q4_0 quantization with HK192 and HV128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_1_HK192_HV128`: Represents the extended vector flash attention kernel type for Q4_1 quantization with HK192 and HV128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_0_HK192_HV128`: Represents the extended vector flash attention kernel type for Q5_0 quantization with HK192 and HV128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_1_HK192_HV128`: Represents the extended vector flash attention kernel type for Q5_1 quantization with HK192 and HV128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q8_0_HK192_HV128`: Represents the extended vector flash attention kernel type for Q8_0 quantization with HK192 and HV128.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_F16_H256`: Represents the extended vector flash attention kernel type for 16-bit floats with H256.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_BF16_H256`: Represents the extended vector flash attention kernel type for bfloat16 with H256.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_0_H256`: Represents the extended vector flash attention kernel type for Q4_0 quantization with H256.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_1_H256`: Represents the extended vector flash attention kernel type for Q4_1 quantization with H256.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_0_H256`: Represents the extended vector flash attention kernel type for Q5_0 quantization with H256.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_1_H256`: Represents the extended vector flash attention kernel type for Q5_1 quantization with H256.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q8_0_H256`: Represents the extended vector flash attention kernel type for Q8_0 quantization with H256.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_F16_HK576_HV512`: Represents the extended vector flash attention kernel type for 16-bit floats with HK576 and HV512.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_BF16_HK576_HV512`: Represents the extended vector flash attention kernel type for bfloat16 with HK576 and HV512.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_0_HK576_HV512`: Represents the extended vector flash attention kernel type for Q4_0 quantization with HK576 and HV512.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q4_1_HK576_HV512`: Represents the extended vector flash attention kernel type for Q4_1 quantization with HK576 and HV512.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_0_HK576_HV512`: Represents the extended vector flash attention kernel type for Q5_0 quantization with HK576 and HV512.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q5_1_HK576_HV512`: Represents the extended vector flash attention kernel type for Q5_1 quantization with HK576 and HV512.
    - `GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_VEC_Q8_0_HK576_HV512`: Represents the extended vector flash attention kernel type for Q8_0 quantization with HK576 and HV512.
    - `GGML_METAL_KERNEL_TYPE_SET_I32`: Represents the set kernel type for 32-bit integers.
    - `GGML_METAL_KERNEL_TYPE_SET_F32`: Represents the set kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_CPY_F32_F32`: Represents the copy kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_CPY_F32_F16`: Represents the copy kernel type for 32-bit floats to 16-bit floats.
    - `GGML_METAL_KERNEL_TYPE_CPY_F32_BF16`: Represents the copy kernel type for 32-bit floats to bfloat16.
    - `GGML_METAL_KERNEL_TYPE_CPY_F16_F16`: Represents the copy kernel type for 16-bit floats.
    - `GGML_METAL_KERNEL_TYPE_CPY_F16_F32`: Represents the copy kernel type for 16-bit floats to 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_CPY_BF16_F32`: Represents the copy kernel type for bfloat16 to 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_CPY_BF16_BF16`: Represents the copy kernel type for bfloat16.
    - `GGML_METAL_KERNEL_TYPE_CPY_F32_Q8_0`: Represents the copy kernel type for 32-bit floats to Q8_0 quantization.
    - `GGML_METAL_KERNEL_TYPE_CPY_F32_Q4_0`: Represents the copy kernel type for 32-bit floats to Q4_0 quantization.
    - `GGML_METAL_KERNEL_TYPE_CPY_F32_Q4_1`: Represents the copy kernel type for 32-bit floats to Q4_1 quantization.
    - `GGML_METAL_KERNEL_TYPE_CPY_F32_Q5_0`: Represents the copy kernel type for 32-bit floats to Q5_0 quantization.
    - `GGML_METAL_KERNEL_TYPE_CPY_F32_Q5_1`: Represents the copy kernel type for 32-bit floats to Q5_1 quantization.
    - `GGML_METAL_KERNEL_TYPE_CPY_F32_IQ4_NL`: Represents the copy kernel type for 32-bit floats to IQ4_NL quantization.
    - `GGML_METAL_KERNEL_TYPE_CPY_Q4_0_F32`: Represents the copy kernel type for Q4_0 quantization to 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_CPY_Q4_0_F16`: Represents the copy kernel type for Q4_0 quantization to 16-bit floats.
    - `GGML_METAL_KERNEL_TYPE_CPY_Q4_1_F32`: Represents the copy kernel type for Q4_1 quantization to 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_CPY_Q4_1_F16`: Represents the copy kernel type for Q4_1 quantization to 16-bit floats.
    - `GGML_METAL_KERNEL_TYPE_CPY_Q5_0_F32`: Represents the copy kernel type for Q5_0 quantization to 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_CPY_Q5_0_F16`: Represents the copy kernel type for Q5_0 quantization to 16-bit floats.
    - `GGML_METAL_KERNEL_TYPE_CPY_Q5_1_F32`: Represents the copy kernel type for Q5_1 quantization to 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_CPY_Q5_1_F16`: Represents the copy kernel type for Q5_1 quantization to 16-bit floats.
    - `GGML_METAL_KERNEL_TYPE_CPY_Q8_0_F32`: Represents the copy kernel type for Q8_0 quantization to 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_CPY_Q8_0_F16`: Represents the copy kernel type for Q8_0 quantization to 16-bit floats.
    - `GGML_METAL_KERNEL_TYPE_CONCAT`: Represents the concatenation kernel type.
    - `GGML_METAL_KERNEL_TYPE_SQR`: Represents the square kernel type.
    - `GGML_METAL_KERNEL_TYPE_SQRT`: Represents the square root kernel type.
    - `GGML_METAL_KERNEL_TYPE_SIN`: Represents the sine function kernel type.
    - `GGML_METAL_KERNEL_TYPE_COS`: Represents the cosine function kernel type.
    - `GGML_METAL_KERNEL_TYPE_NEG`: Represents the negation kernel type.
    - `GGML_METAL_KERNEL_TYPE_SUM_ROWS`: Represents the sum of rows kernel type.
    - `GGML_METAL_KERNEL_TYPE_POOL_2D_AVG_F32`: Represents the 2D average pooling kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_POOL_2D_MAX_F32`: Represents the 2D max pooling kernel type for 32-bit floats.
    - `GGML_METAL_KERNEL_TYPE_ARGMAX`: Represents the argmax kernel type.
    - `GGML_METAL_KERNEL_TYPE_COUNT`: Represents the count of kernel types.
- **Description**: The `ggml_metal_kernel_type` is an enumeration that defines various types of Metal kernels used in the GGML library for different mathematical operations and transformations. Each enumerator represents a specific kernel type, such as addition, multiplication, or specialized operations like softmax and normalization, tailored for different data types and configurations. This enumeration is crucial for selecting the appropriate kernel for a given operation in a Metal-based computation context.


---
### ggml\_metal\_heap
- **Type**: `struct`
- **Members**:
    - `n_unused`: Tracks the number of times the heap was unused.
    - `n_alloc`: Counts the total number of buffer allocations in the heap.
    - `offs`: Represents the current offset in the heap, reset after each node to reuse memory.
    - `obj`: Holds the currently allocated MTLHeap objects in the heap.
    - `bufs`: An NSMutableArray containing the buffers allocated in the heap.
- **Description**: The `ggml_metal_heap` structure is designed to manage memory allocation for Metal buffers in a GPU context. It keeps track of memory usage and allocations, allowing for efficient reuse of memory by resetting offsets after each node operation. This structure is crucial for optimizing memory management in GPU computations, ensuring that memory is allocated and deallocated efficiently to prevent memory leaks and optimize performance.


---
### ggml\_metal\_heap\_ptr
- **Type**: `class`
- **Members**:
    - `data`: A pointer to a `ggml_metal_heap` structure.
- **Description**: The `ggml_metal_heap_ptr` is an Objective-C class that acts as a wrapper for a pointer to a `ggml_metal_heap` structure. This class is used to manage and reference the `ggml_metal_heap` within the context of an Objective-C environment, particularly when dealing with Metal API resources. The `data` property holds the actual pointer to the `ggml_metal_heap`, allowing for operations and memory management to be performed on the heap.


---
### ggml\_metal\_mem\_pool
- **Type**: `struct`
- **Members**:
    - `device`: Represents the Metal device associated with the memory pool.
    - `n_heaps`: Tracks the total number of heaps ever created, including those removed.
    - `heaps`: An NSMutableArray holding the current heaps in the memory pool.
    - `heaps_to_remove`: An NSMutableArray holding indices of heaps to be removed.
- **Description**: The `ggml_metal_mem_pool` structure is designed to manage memory allocation for Metal-based computations. It maintains a collection of memory heaps, represented by `heaps`, which are used to allocate temporary buffers during compute operations. The structure also tracks the total number of heaps created over its lifetime with `n_heaps`, and manages a list of heaps that are scheduled for removal with `heaps_to_remove`. This memory pool is crucial for efficient memory management in Metal, allowing for dynamic allocation and deallocation of memory resources as needed by the compute tasks.


---
### ggml\_metal\_command\_buffer
- **Type**: `struct`
- **Members**:
    - `obj`: A Metal command buffer object used for encoding and executing commands on the GPU.
    - `mem_pool`: A memory pool associated with the command buffer for allocating temporary buffers during computation.
- **Description**: The `ggml_metal_command_buffer` structure is designed to manage a Metal command buffer and its associated memory pool. The command buffer (`obj`) is used to encode and execute commands on the GPU, while the memory pool (`mem_pool`) provides a mechanism for allocating temporary buffers needed during the execution of these commands. This structure is part of a larger system that interfaces with Apple's Metal framework to perform computations on the GPU, optimizing performance by managing resources efficiently.


---
### ggml\_backend\_metal\_context
- **Type**: `struct`
- **Members**:
    - `device`: A Metal device object representing the GPU device.
    - `queue`: A Metal command queue for submitting command buffers to the GPU.
    - `d_queue`: A dispatch queue for concurrent operations.
    - `kernels`: An array of Metal compute pipeline states for various kernel operations.
    - `capture_next_compute`: A boolean indicating if the next compute operation should be captured.
    - `capture_started`: A boolean indicating if a capture has started.
    - `capture_scope`: A Metal capture scope object for debugging.
    - `n_cb`: The number of extra threads used to submit command buffers.
    - `n_nodes_0`: The number of nodes submitted by the main thread.
    - `n_nodes_1`: The remaining number of nodes submitted by additional threads.
    - `n_nodes_per_cb`: The number of nodes per command buffer.
    - `gf`: A pointer to a graph structure for computation.
    - `encode_async`: A block for asynchronous encoding operations.
    - `cmd_bufs`: An array of command buffers used for GPU operations.
    - `abort_callback`: A callback function to abort graph computation.
    - `abort_callback_data`: Data associated with the abort callback.
- **Description**: The `ggml_backend_metal_context` structure is designed to manage and execute GPU-based computations using Apple's Metal framework. It encapsulates a Metal device and command queue, along with a set of compute pipeline states for executing various kernel operations. The structure supports concurrent execution through a dispatch queue and manages command buffers for efficient GPU resource utilization. It also includes mechanisms for capturing and debugging GPU operations, as well as handling asynchronous encoding tasks. The context is integral to executing graph-based computations on the GPU, with support for aborting operations via a callback mechanism.


---
### ggml\_backend\_metal\_buffer
- **Type**: `struct`
- **Members**:
    - `data`: Pointer to the data stored in the buffer.
    - `size`: Size of the data in the buffer.
    - `metal`: Metal buffer object for GPU operations.
- **Description**: The `ggml_backend_metal_buffer` structure is designed to manage data buffers for Metal-based GPU operations. It contains a pointer to the data, the size of the data, and a Metal buffer object that facilitates GPU processing. This structure is essential for handling data that needs to be processed on a Metal-supported GPU, ensuring efficient memory management and data transfer between the CPU and GPU.


---
### ggml\_backend\_metal\_buffer\_context
- **Type**: `struct`
- **Members**:
    - `all_data`: Pointer to the entire data block.
    - `all_size`: Size of the entire data block.
    - `owned`: Indicates if the context owns the data.
    - `n_buffers`: Number of buffers used to avoid maximum buffer size limitation.
    - `buffers`: Array of Metal buffers used for device memory mapping.
    - `rset`: Optional Metal residency set for managing memory residency.
- **Description**: The `ggml_backend_metal_buffer_context` structure is designed to manage memory buffers for Metal devices, particularly in the context of the GGML backend. It holds a pointer to the entire data block and its size, and it can manage multiple buffers to circumvent the maximum buffer size limitation when using memory mapping. The structure also includes an optional Metal residency set to handle memory residency efficiently, and a flag to indicate ownership of the data.


# Functions

---
### ggml\_backend\_metal\_device\_acq
The `ggml_backend_metal_device_acq` function acquires a Metal device for a given context, initializing it if necessary, and updates the context with device capabilities and reference count.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_metal_device_context` structure that holds information about the Metal device, including its reference count and capabilities.
- **Control Flow**:
    - Assert that the context (`ctx`) is not NULL.
    - Check if the Metal device (`mtl_device`) in the context is nil (uninitialized).
    - If the device is nil, initialize it using `MTLCreateSystemDefaultDevice()`.
    - If the device is successfully initialized, update the context with the device's capabilities, such as SIMD group reduction, SIMD group matrix multiplication, residency sets, and bfloat support.
    - Increment the device reference count (`mtl_device_ref_count`) in the context.
    - Return the Metal device (`mtl_device`).
- **Output**: Returns an `id<MTLDevice>`, which is the acquired Metal device for the given context.


---
### ggml\_backend\_metal\_device\_rel
The `ggml_backend_metal_device_rel` function releases a Metal device and its associated resources when the reference count reaches zero.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_metal_device_context` structure, which contains information about the Metal device and its resources.
- **Control Flow**:
    - Assert that the `ctx` is not NULL and that the `mtl_device_ref_count` is greater than zero.
    - Decrement the `mtl_device_ref_count`.
    - If `mtl_device_ref_count` reaches zero, release the `mtl_library` if it exists and set it to nil.
    - Release the `mtl_device` if it exists and set it to nil.
- **Output**: The function does not return any value; it performs cleanup operations on the Metal device context.


---
### ggml\_metal\_heap\_init
The `ggml_metal_heap_init` function initializes a Metal heap for memory management on a Metal device.
- **Inputs**:
    - `device`: An `id<MTLDevice>` representing the Metal device on which the heap is to be created.
    - `size`: A `size_t` value representing the size of the heap to be created.
- **Control Flow**:
    - Allocate memory for a `ggml_metal_heap` structure using `calloc`.
    - Create a `MTLHeapDescriptor` and set its properties: `storageMode` to `MTLStorageModePrivate`, `cpuCacheMode` to `MTLCPUCacheModeDefaultCache`, `type` to `MTLHeapTypePlacement`, and `size` to the provided size.
    - Initialize the `n_unused` and `n_alloc` fields of the heap to 0.
    - Create a new Metal heap using the device and descriptor, and assign it to the `obj` field of the heap.
    - Check if the heap creation was successful; if not, log an error, free the allocated memory, and return `false`.
    - Release the descriptor to free its resources.
    - Initialize the `bufs` field of the heap as a new `NSMutableArray`.
    - Return the initialized `ggml_metal_heap` structure.
- **Output**: Returns a pointer to the initialized `ggml_metal_heap` structure, or `NULL` if the heap creation fails.


---
### ggml\_metal\_heap\_reset
The `ggml_metal_heap_reset` function resets the state of a Metal heap, releasing all associated buffers and marking the heap's memory as volatile for potential reuse by the operating system.
- **Inputs**:
    - `heap`: A pointer to a `ggml_metal_heap` structure representing the Metal heap to be reset.
- **Control Flow**:
    - Set the heap's offset (`offs`) to 0, indicating that the heap is ready for reuse.
    - Check if the heap's buffer list (`bufs`) is empty; if not, reset the `n_unused` counter to 0, otherwise increment it.
    - Iterate over each buffer in the heap's buffer list, releasing each buffer to free its resources.
    - Clear the buffer list to remove all buffer references.
    - Set the heap's purgeable state to `MTLPurgeableStateVolatile`, allowing the operating system to reclaim the memory if needed.
- **Output**: The function does not return a value; it modifies the state of the provided `ggml_metal_heap` structure in place.


---
### ggml\_metal\_heap\_free
The `ggml_metal_heap_free` function releases all resources associated with a given Metal heap and deallocates its memory.
- **Inputs**:
    - `heap`: A pointer to a `ggml_metal_heap` structure representing the Metal heap to be freed.
- **Control Flow**:
    - Check if the `heap` is `nil`; if so, return immediately as there is nothing to free.
    - Call `ggml_metal_heap_reset` to reset the heap, releasing all associated buffers and marking the memory as volatile.
    - Release the Metal heap object (`heap->obj`) and the buffer array (`heap->bufs`).
    - Deallocate the memory for the `ggml_metal_heap` structure using `free`.
- **Output**: The function does not return any value; it performs cleanup operations on the provided heap.


---
### ggml\_metal\_mem\_pool\_init
The `ggml_metal_mem_pool_init` function initializes a memory pool for Metal device memory management.
- **Inputs**: None
- **Control Flow**:
    - Allocate memory for a `ggml_metal_mem_pool` structure using `calloc`.
    - Initialize the `n_heaps` field to 0, indicating no heaps have been created yet.
    - Create and initialize two `NSMutableArray` objects for `heaps` and `heaps_to_remove`.
    - Return the initialized `ggml_metal_mem_pool` structure.
- **Output**: A pointer to an initialized `ggml_metal_mem_pool` structure.


---
### ggml\_metal\_mem\_pool\_free
The `ggml_metal_mem_pool_free` function releases all memory resources associated with a given Metal memory pool.
- **Inputs**:
    - `mem_pool`: A pointer to a `ggml_metal_mem_pool` structure representing the memory pool to be freed.
- **Control Flow**:
    - Log the start of the memory pool freeing process, including the number of heaps and total heaps created.
    - Initialize variables `size_all` and `size_cur` to zero to track memory sizes.
    - Iterate over each heap in the `mem_pool->heaps` array.
    - For each heap, log its details including address, number of allocations, unused count, size, and buffer count.
    - If the heap has buffers, add its size to `size_cur`.
    - Add the heap's size to `size_all`.
    - Free the heap using `ggml_metal_heap_free` and release the heap pointer.
    - Release the `heaps` and `heaps_to_remove` arrays.
    - If `size_all` is greater than zero, log the total and current sizes of all heaps.
    - Free the `mem_pool` structure.
- **Output**: The function does not return any value; it performs cleanup operations on the provided memory pool.


---
### ggml\_metal\_mem\_pool\_reset
The `ggml_metal_mem_pool_reset` function resets the memory pool by clearing all heaps and removing unused ones.
- **Inputs**:
    - `mem_pool`: A pointer to a `ggml_metal_mem_pool` structure representing the memory pool to be reset.
- **Control Flow**:
    - Iterate over each heap in the memory pool's `heaps` array.
    - For each heap, call `ggml_metal_heap_reset` to reset its offset and release its buffers.
    - Check if a heap has been unused for a certain threshold (128 times) and mark it for removal if so.
    - If there are heaps marked for removal, iterate over them in reverse order to remove them from the `heaps` array and free their resources.
    - Clear the `heaps_to_remove` array after removing the marked heaps.
- **Output**: The function does not return a value; it modifies the memory pool in place.


---
### ggml\_metal\_mem\_pool\_clear
The `ggml_metal_mem_pool_clear` function resets the offset of all heaps in a Metal memory pool to zero.
- **Inputs**:
    - `mem_pool`: A pointer to a `ggml_metal_mem_pool` structure, which represents the Metal memory pool containing heaps to be cleared.
- **Control Flow**:
    - Iterate over each `ggml_metal_heap_ptr` in the `heaps` array of the `mem_pool`.
    - For each heap, set the `offs` attribute to zero, effectively resetting the heap's offset.
- **Output**: The function does not return any value; it modifies the `mem_pool` in place.


---
### ggml\_metal\_mem\_pool\_alloc
The `ggml_metal_mem_pool_alloc` function allocates a Metal buffer from a memory pool, creating a new heap if necessary.
- **Inputs**:
    - `mem_pool`: A pointer to a `ggml_metal_mem_pool` structure, which manages heaps of Metal buffers.
    - `size`: The size of the buffer to allocate, in bytes.
- **Control Flow**:
    - Align the requested size to a 256-byte boundary.
    - Iterate over existing heaps in the memory pool to find one with enough space for the aligned size.
    - If a suitable heap is found, allocate a new Metal buffer from it, update the heap's offset, and return the buffer.
    - If no suitable heap is found, create a new heap with the aligned size, allocate a buffer from it, and add the heap to the memory pool.
    - Return the newly allocated Metal buffer.
- **Output**: Returns an `id<MTLBuffer>` representing the allocated Metal buffer, or `nil` if allocation fails.


---
### ggml\_metal\_host\_malloc
The `ggml_metal_host_malloc` function allocates memory on the host system, using either `vm_allocate` on macOS or `posix_memalign` on other systems, and returns a pointer to the allocated memory.
- **Inputs**:
    - `n`: The size of the memory to allocate, in bytes.
- **Control Flow**:
    - Check if the target operating system is macOS.
    - If on macOS, use `vm_allocate` to allocate memory and check for errors.
    - If not on macOS, use `posix_memalign` to allocate memory and check for errors.
    - Return the pointer to the allocated memory.
- **Output**: A pointer to the allocated memory, or NULL if the allocation fails.


---
### ggml\_metal\_load\_library
The `ggml_metal_load_library` function loads a Metal library for a given Metal device, either from an embedded source, a precompiled library, or by compiling from source, with optional support for bfloat16.
- **Inputs**:
    - `device`: An `id<MTLDevice>` representing the Metal device for which the library is to be loaded.
    - `use_bfloat`: A boolean indicating whether to use bfloat16 support in the library.
- **Control Flow**:
    - Check if the library is embedded using the `GGML_METAL_EMBED_LIBRARY` macro.
    - If embedded, initialize the library source from the embedded data.
    - If not embedded, attempt to locate a precompiled library in the bundle or the binary's directory.
    - If a precompiled library is found, load it using `newLibraryWithURL:error:`.
    - If no precompiled library is found, attempt to load the source from a file and compile it using `newLibraryWithSource:options:error:`.
    - If any errors occur during loading or compilation, log the error and return `NULL`.
- **Output**: Returns an `id<MTLLibrary>` representing the loaded Metal library, or `NULL` if loading fails.


---
### ggml\_metal\_init
The `ggml_metal_init` function initializes a Metal backend context for GPU computations, setting up the necessary device, command queue, and loading Metal kernels for various operations.
- **Inputs**:
    - `dev`: A pointer to a `ggml_backend_dev_t` structure representing the backend device context.
- **Control Flow**:
    - Log the start of the allocation process.
    - If on macOS and not in debug mode, list all available Metal devices.
    - Allocate memory for a `ggml_backend_metal_context` structure.
    - Acquire a Metal device using `ggml_backend_metal_device_acq`.
    - Log the selected default device name.
    - Create a new command queue for the device.
    - If command queue creation fails, log an error and return NULL.
    - Create a concurrent dispatch queue for Metal operations.
    - Load the Metal library if not already loaded in the device context.
    - Log the GPU name and supported GPU families.
    - Log various device capabilities such as SIMD group support and bfloat support.
    - Initialize command buffers and memory pools for each command buffer.
    - Log the recommended maximum working set size if available.
    - Load and initialize Metal kernels for various operations.
    - Return the initialized Metal context.
- **Output**: A pointer to a `ggml_backend_metal_context` structure, or NULL if initialization fails.


---
### ggml\_metal\_free
The `ggml_metal_free` function deallocates resources associated with a Metal backend context in the GGML library.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_metal_context` structure representing the Metal backend context to be deallocated.
- **Control Flow**:
    - Log the start of the deallocation process.
    - Iterate over all kernel types and release their associated pipeline resources.
    - Release the asynchronous encoding block if it exists.
    - Release the Metal command queue associated with the context.
    - Iterate over the command buffers, releasing their memory pools.
    - Release the dispatch queue used for concurrent operations.
    - Free the memory allocated for the context structure.
- **Output**: The function does not return any value; it performs cleanup operations to free resources.


---
### ggml\_backend\_metal\_buffer\_rset\_init
The `ggml_backend_metal_buffer_rset_init` function initializes a Metal residency set for a given buffer context if the device supports residency sets.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_metal_buffer_context` structure, which contains information about the buffer context to be initialized.
    - `ctx_dev`: A pointer to a `ggml_backend_metal_device_context` structure, which contains information about the Metal device context, including whether it supports residency sets.
    - `device`: An `id<MTLDevice>` representing the Metal device on which the residency set is to be initialized.
- **Control Flow**:
    - Check if the device supports residency sets using the `has_residency_sets` flag in `ctx_dev`.
    - If residency sets are supported, create a `MTLResidencySetDescriptor` and set its label and initial capacity based on the number of buffers in `ctx`.
    - Attempt to create a new residency set using the descriptor and assign it to `ctx->rset`.
    - If an error occurs during the creation of the residency set, log the error and return `false`.
    - If successful, add each buffer's Metal allocation to the residency set.
    - Commit the residency set and request residency for it.
    - Return `true` if the residency set is successfully initialized, otherwise return `false`.
- **Output**: A boolean value indicating whether the residency set was successfully initialized (`true`) or not (`false`).


---
### ggml\_backend\_metal\_buffer\_rset\_free
The `ggml_backend_metal_buffer_rset_free` function releases the residency set associated with a Metal buffer context, if it exists.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_metal_buffer_context` structure, which contains information about the Metal buffer and its residency set.
- **Control Flow**:
    - Check if the residency set (`rset`) exists in the context.
    - If the residency set exists, call `endResidency` to end its residency.
    - Remove all allocations from the residency set using `removeAllAllocations`.
    - Release the residency set by calling `release`.
- **Output**: The function does not return any value; it performs cleanup operations on the residency set.


---
### ggml\_metal\_get\_buffer
The `ggml_metal_get_buffer` function retrieves the Metal buffer associated with a given tensor and calculates the offset within that buffer.
- **Inputs**:
    - `t`: A pointer to a `ggml_tensor` structure representing the tensor for which the Metal buffer is to be retrieved.
    - `offs`: A pointer to a `size_t` variable where the offset within the Metal buffer will be stored.
- **Control Flow**:
    - Retrieve the total size of the tensor using `ggml_nbytes` function.
    - Determine the buffer context from the tensor's buffer or its source view buffer.
    - Iterate over the buffers in the buffer context to find the one that fully contains the tensor data.
    - Calculate the offset within the found buffer and store it in the provided `offs` variable.
    - Return the Metal buffer associated with the found buffer.
- **Output**: Returns an `id<MTLBuffer>` representing the Metal buffer that contains the tensor data.


---
### ggml\_metal\_supports\_op
The `ggml_metal_supports_op` function checks if a given operation on a tensor is supported by the Metal backend, considering the device's capabilities and the operation's requirements.
- **Inputs**:
    - `ctx_dev`: A pointer to a `ggml_backend_metal_device_context` structure that contains information about the Metal device's capabilities.
    - `op`: A pointer to a `ggml_tensor` structure representing the operation to be checked for support.
- **Control Flow**:
    - Check if the device supports bfloat16 operations and if the operation uses bfloat16; if not supported, return false.
    - Switch on the operation type (`op->op`) to determine if the operation is supported.
    - For unary operations, check if the operation is one of the supported types (e.g., TANH, RELU) and if the source tensor is contiguous and of type F32.
    - For binary operations (e.g., ADD, SUB), check if the source tensor types are F32.
    - For operations like SUM_ROWS, SOFT_MAX, and GROUP_NORM, check if the device supports SIMD group reduction and if the source tensor is contiguous.
    - For operations like RMS_NORM and L2_NORM, check if the device supports SIMD group reduction and if the source tensor's first dimension is a multiple of 4 and contiguous.
    - For operations like GET_ROWS, check if the operation's fourth dimension is 1.
    - Return true if all checks pass, indicating the operation is supported.
- **Output**: A boolean value indicating whether the operation is supported by the Metal backend.


---
### ggml\_metal\_encode\_node
The `ggml_metal_encode_node` function encodes a computational graph node for execution on a Metal device using a compute command encoder.
- **Inputs**:
    - `backend`: A `ggml_backend_t` structure representing the backend context for Metal operations.
    - `idx`: An integer representing the index of the node in the computational graph.
    - `encoder`: An `id<MTLComputeCommandEncoder>` object used to encode commands for the Metal device.
    - `mem_pool`: A pointer to a `ggml_metal_mem_pool` structure used for memory management during encoding.
- **Control Flow**:
    - Retrieve the context and device context from the backend.
    - Get the node from the computational graph using the provided index.
    - Check if the destination tensor is empty and return true if it is.
    - Determine the operation type of the node and select the appropriate Metal kernel pipeline.
    - Set up the kernel arguments based on the operation type and tensor properties.
    - Configure the compute command encoder with the selected pipeline and arguments.
    - Dispatch the compute command with appropriate threadgroup sizes based on the operation.
    - Return true if the node is successfully encoded.
- **Output**: A boolean value indicating whether the node was successfully encoded for execution on the Metal device.


