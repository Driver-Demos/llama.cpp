# Purpose
This C header file is an internal component of the GGML library, focused on providing functions for quantization and dequantization of data. The file defines a series of functions that convert floating-point data into various quantized formats and vice versa. These functions are essential for optimizing data storage and processing, particularly in scenarios where memory and computational efficiency are critical, such as in machine learning and signal processing applications. The quantization functions are categorized by different quantization levels and types, such as q4, q5, q8, and others, each representing a specific method of reducing the precision of the data to save space or improve processing speed.

The file also includes functions for "Activation aWare Quantization," which utilize an importance matrix to perform quantization with consideration of the data's significance, potentially improving the accuracy of the quantized data. The use of `GGML_API` indicates that these functions are part of the public API, intended for use by the CPU backend of the GGML library. The file is structured to be compatible with both C and C++ environments, as indicated by the `extern "C"` block, ensuring that the functions can be used in a wide range of applications. Overall, this header file is a specialized component of the GGML library, providing essential functionality for data quantization and dequantization.
# Imports and Dependencies

---
- `ggml-common.h`
- `ggml.h`


# Function Declarations (Public API)

---
### quantize\_row\_q4\_0\_ref<!-- {{#callable_declaration:quantize_row_q4_0_ref}} -->
Quantizes a row of floating-point values into a specific format.
- **Description**: This function is used to quantize a row of floating-point values into a compressed format suitable for storage or transmission. It should be called when you have a block of data that needs to be quantized, and the length of the data must be a multiple of the quantization block size. The function computes the maximum absolute value in each block and uses it to scale the values appropriately. It is important to ensure that the input pointer is not null and that the length parameter is valid; otherwise, the behavior is undefined.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values to be quantized. Must not be null and should point to a valid memory region containing at least 'k' elements.
    - `y`: A pointer to a destination array where the quantized data will be stored. Must not be null and should point to a valid memory region that can hold the quantized output.
    - `k`: An integer representing the number of elements to process. Must be a positive integer and a multiple of the quantization block size.
- **Output**: The function does not return a value and mutates the output buffer pointed to by 'y' with the quantized data.
- **See also**: [`quantize_row_q4_0_ref`](ggml-quants.c.driver.md#quantize_row_q4_0_ref)  (Implementation)


---
### quantize\_row\_q4\_1\_ref<!-- {{#callable_declaration:quantize_row_q4_1_ref}} -->
Quantizes a row of floating-point values into a specific format.
- **Description**: This function is used to quantize a row of floating-point values into a compressed format suitable for storage or transmission. It should be called when you have a block of data that needs to be quantized, and the length of the data must be a multiple of the quantization block size. The function computes the minimum and maximum values of each block, scales the values accordingly, and stores the quantized results in the provided output structure. It is important to ensure that the input pointer is not null and that the output structure is properly allocated before calling this function.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values to be quantized. Must not be null and should point to a valid memory region containing at least 'k' elements.
    - `y`: A pointer to a pre-allocated output structure where the quantized data will be stored. Must not be null and should be large enough to hold 'k / QK4_1' blocks.
    - `k`: An integer representing the total number of floating-point values to quantize. Must be a positive integer and a multiple of QK4_1.
- **Output**: The function does not return a value. It populates the output structure with the quantized data based on the input values.
- **See also**: [`quantize_row_q4_1_ref`](ggml-quants.c.driver.md#quantize_row_q4_1_ref)  (Implementation)


---
### quantize\_row\_q5\_0\_ref<!-- {{#callable_declaration:quantize_row_q5_0_ref}} -->
Quantizes a row of floating-point values into a specific format.
- **Description**: This function is used to quantize a row of floating-point values into a compressed format suitable for storage or transmission. It should be called when you have a block of data that needs to be quantized, and the length of the data must be a multiple of the quantization block size. The function modifies the output structure to store the quantized values and their associated metadata. It is important to ensure that the input pointer is valid and that the output structure has been properly allocated before calling this function.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values to be quantized. Must not be null and should point to a valid memory region containing at least 'k' elements.
    - `y`: A pointer to a pre-allocated output structure where the quantized data will be stored. Must not be null and should be large enough to hold the quantized results.
    - `k`: An integer representing the total number of floating-point values to quantize. Must be a positive integer and a multiple of the quantization block size.
- **Output**: The function does not return a value. It populates the output structure with the quantized data.
- **See also**: [`quantize_row_q5_0_ref`](ggml-quants.c.driver.md#quantize_row_q5_0_ref)  (Implementation)


---
### quantize\_row\_q5\_1\_ref<!-- {{#callable_declaration:quantize_row_q5_1_ref}} -->
Quantizes a row of floating-point values into a compressed format.
- **Description**: This function is used to quantize a row of floating-point values into a specific compressed format, which is useful for reducing memory usage and improving performance in machine learning applications. It must be called with a valid input array of floats and a corresponding output structure to hold the quantized data. The input length, specified by the parameter `k`, must be a multiple of the quantization block size, which is defined internally. If `k` is not a valid multiple, the function will assert and terminate. The function computes the minimum and maximum values of each block of data, normalizes the values, and stores the quantized results in the output structure.
- **Inputs**:
    - `x`: A pointer to an array of `float` values that represent the input data to be quantized. This array must not be null and should contain at least `k` elements.
    - `y`: A pointer to a `block_q5_1` structure where the quantized output will be stored. This structure must not be null and should be properly allocated before calling the function.
    - `k`: An integer representing the total number of input elements to be quantized. This value must be a positive integer and a multiple of the quantization block size.
- **Output**: The function does not return a value. It populates the output structure `y` with the quantized data derived from the input array.
- **See also**: [`quantize_row_q5_1_ref`](ggml-quants.c.driver.md#quantize_row_q5_1_ref)  (Implementation)


---
### quantize\_row\_q8\_0\_ref<!-- {{#callable_declaration:quantize_row_q8_0_ref}} -->
Quantizes a row of floating-point values into a specific format.
- **Description**: This function is used to quantize a row of floating-point values into a compressed format suitable for efficient storage or processing. It should be called when you have a block of floating-point data that needs to be quantized, and the length of this data must be a multiple of `QK8_0`. The function computes the maximum absolute value of the input data to determine the scaling factor for quantization. It is important to ensure that the input pointer is valid and that the output structure is properly allocated before calling this function.
- **Inputs**:
    - `x`: A pointer to an array of `float` values representing the input data to be quantized. This array must not be null and should contain at least `k` elements, where `k` is a multiple of `QK8_0`.
    - `y`: A pointer to a `block_q8_0` structure where the quantized output will be stored. This structure must be allocated by the caller and should be large enough to hold the quantized data for `k / QK8_0` blocks.
    - `k`: An integer representing the number of elements in the input array `x`. This value must be a positive integer and a multiple of `QK8_0`.
- **Output**: The function does not return a value. It populates the `y` structure with the quantized data derived from the input array.
- **See also**: [`quantize_row_q8_0_ref`](ggml-quants.c.driver.md#quantize_row_q8_0_ref)  (Implementation)


---
### quantize\_row\_q8\_1\_ref<!-- {{#callable_declaration:quantize_row_q8_1_ref}} -->
Quantizes a row of floating-point values into a specific format.
- **Description**: This function is used to quantize a row of floating-point values into a compressed format suitable for efficient storage or processing. It should be called when you have a block of floating-point data that needs to be quantized into the `block_q8_1` format. The input array must be properly allocated and contain at least `k` elements, where `k` must be a multiple of 32. The function computes the maximum absolute value of the input data, scales the values accordingly, and stores the quantized results in the output structure. It is important to ensure that the input pointer is not null and that the output structure is properly allocated before calling this function.
- **Inputs**:
    - `x`: A pointer to an array of `float` values to be quantized. Must not be null and must contain at least `k` elements.
    - `y`: A pointer to a `block_q8_1` structure where the quantized results will be stored. Must not be null and must be properly allocated.
    - `k`: An integer representing the number of elements to process. Must be a positive multiple of 32.
- **Output**: The function does not return a value. It populates the `block_q8_1` structure pointed to by `y` with the quantized data.
- **See also**: [`quantize_row_q8_1_ref`](ggml-quants.c.driver.md#quantize_row_q8_1_ref)  (Implementation)


---
### quantize\_row\_q2\_K\_ref<!-- {{#callable_declaration:quantize_row_q2_K_ref}} -->
Quantizes a row of floating-point values into a specific format.
- **Description**: This function is used to quantize a row of floating-point values into a compressed format suitable for efficient storage or processing. It should be called when the input data is ready for quantization, and the output buffer must be allocated and provided by the caller. The function expects the length of the input data (`k`) to be a multiple of `QK_K`, which is a predefined constant. If the input data is invalid or does not meet the required conditions, the function may produce undefined behavior or incorrect results.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values to be quantized. Must not be null and should point to a valid memory region containing at least `k` elements.
    - `y`: A pointer to a pre-allocated output structure where the quantized data will be stored. Must not be null and should be large enough to hold the quantized results.
    - `k`: An integer representing the number of elements in the input array. Must be a positive integer and a multiple of `QK_K`.
- **Output**: The function does not return a value. It populates the output structure pointed to by `y` with the quantized data.
- **See also**: [`quantize_row_q2_K_ref`](ggml-quants.c.driver.md#quantize_row_q2_K_ref)  (Implementation)


---
### quantize\_row\_q3\_K\_ref<!-- {{#callable_declaration:quantize_row_q3_K_ref}} -->
Quantizes a row of floating-point values into a specific format.
- **Description**: This function is used to quantize a row of floating-point values into a compressed format suitable for storage or transmission. It should be called when the input data is ready for quantization, and the output buffer must be properly allocated to hold the quantized data. The function expects the length of the input data, `k`, to be a multiple of `QK_K`, which is a predefined constant. If `k` is not a multiple of `QK_K`, the behavior is undefined. The function modifies the output structure to store the quantized values and associated metadata.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values to be quantized. Must not be null and should point to at least `k` elements.
    - `y`: A pointer to a `block_q3_K` structure where the quantized output will be stored. Must not be null and should be properly allocated.
    - `k`: An integer representing the number of elements in the input array `x`. Must be a positive integer and a multiple of `QK_K`.
- **Output**: The function does not return a value. It populates the `y` structure with the quantized data derived from the input array.
- **See also**: [`quantize_row_q3_K_ref`](ggml-quants.c.driver.md#quantize_row_q3_K_ref)  (Implementation)


---
### quantize\_row\_q4\_K\_ref<!-- {{#callable_declaration:quantize_row_q4_K_ref}} -->
Quantizes a row of floating-point values into a specific format.
- **Description**: This function is used to quantize a row of floating-point values into a compressed format suitable for efficient storage or processing. It should be called with a valid input array of floats and a corresponding output structure to hold the quantized data. The parameter `k` must be a multiple of `QK_K`, which defines the expected size of the input data. The function processes the input in blocks, calculating scales and minimum values for quantization, and it will handle the quantization of values accordingly. If the input data is invalid or does not meet the specified conditions, the behavior is undefined.
- **Inputs**:
    - `x`: A pointer to an array of `float` values representing the input data to be quantized. This array must not be null and should contain at least `k` elements.
    - `y`: A pointer to a `block_q4_K` structure where the quantized output will be stored. This structure must not be null and should be properly allocated before calling the function.
    - `k`: An integer representing the number of elements to process from the input array. This value must be a positive integer and a multiple of `QK_K`.
- **Output**: The function does not return a value. It populates the output structure `y` with the quantized data derived from the input array.
- **See also**: [`quantize_row_q4_K_ref`](ggml-quants.c.driver.md#quantize_row_q4_K_ref)  (Implementation)


---
### quantize\_row\_q5\_K\_ref<!-- {{#callable_declaration:quantize_row_q5_K_ref}} -->
Quantizes a row of floating-point data into a specific format.
- **Description**: This function is used to quantize a row of floating-point values into a compressed format suitable for efficient storage and processing. It should be called with a valid input array of floats and a corresponding output structure to hold the quantized data. The parameter `k` must be a multiple of `QK_K`, which defines the size of the data being processed. The function handles the quantization process, including scaling and determining minimum values, and it will assert if `k` is not valid. The output structure will be populated with the quantized data, and the function does not return a value.
- **Inputs**:
    - `x`: A pointer to an array of floats representing the input data to be quantized. This array must not be null and should contain at least `k` elements.
    - `y`: A pointer to a `block_q5_K` structure where the quantized output will be stored. This structure must not be null and should be properly allocated before calling the function.
    - `k`: An integer representing the number of elements to process. It must be a positive integer and a multiple of `QK_K`.
- **Output**: None
- **See also**: [`quantize_row_q5_K_ref`](ggml-quants.c.driver.md#quantize_row_q5_K_ref)  (Implementation)


---
### quantize\_row\_q6\_K\_ref<!-- {{#callable_declaration:quantize_row_q6_K_ref}} -->
Quantizes a row of floating-point values into a specific format.
- **Description**: This function is used to quantize a row of floating-point values into a compressed format suitable for storage or transmission. It should be called with a valid input array of floating-point numbers and a pre-allocated output structure. The parameter `k` must be a multiple of `QK_K`, which defines the number of elements to process. If the maximum absolute scale of the input values is below a defined threshold, the output will be set to zero. The function modifies the output structure to contain the quantized values and scales, and it is important to ensure that the input and output pointers are not null.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values to be quantized. Must not be null and should contain at least `k` elements.
    - `y`: A pointer to a pre-allocated `block_q6_K` structure where the quantized output will be stored. Must not be null.
    - `k`: An integer representing the number of elements to process. Must be a multiple of `QK_K`.
- **Output**: The function does not return a value. It populates the `y` structure with the quantized data based on the input values.
- **See also**: [`quantize_row_q6_K_ref`](ggml-quants.c.driver.md#quantize_row_q6_K_ref)  (Implementation)


---
### quantize\_row\_q8\_K\_ref<!-- {{#callable_declaration:quantize_row_q8_K_ref}} -->
Quantizes a row of floating-point values into a specified format.
- **Description**: This function is used to quantize a row of floating-point values into a compressed format suitable for efficient storage or processing. It should be called when the input data is ready for quantization, and the length of the input must be a multiple of `QK_K`. The function processes the input in blocks, determining the maximum absolute value for each block to scale the quantization appropriately. If the maximum absolute value is zero, the corresponding output block will be set to zero. The function modifies the output structure to store the quantized values and their associated metadata.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values to be quantized. Must not be null and should point to a valid memory region containing at least `k` elements.
    - `y`: A pointer to a `block_q8_K` structure where the quantized output will be stored. Must not be null and the caller retains ownership of this memory.
    - `k`: An integer representing the number of elements in the input array `x`. Must be a positive integer and a multiple of `QK_K`.
- **Output**: The function does not return a value. It populates the `y` structure with quantized data based on the input provided.
- **See also**: [`quantize_row_q8_K_ref`](ggml-quants.c.driver.md#quantize_row_q8_K_ref)  (Implementation)


---
### quantize\_row\_tq1\_0\_ref<!-- {{#callable_declaration:quantize_row_tq1_0_ref}} -->
Quantizes a row of floating-point values into a specific format.
- **Description**: This function is used to quantize a row of floating-point values into a compressed format suitable for storage or transmission. It must be called with a valid pointer to an array of floats and a properly allocated output structure. The parameter `k` must be a multiple of `QK_K`, which defines the number of elements processed in each quantization block. The function will compute the maximum absolute value of the input data and use it for normalization. If the input data is invalid or if `k` is not a multiple of `QK_K`, the behavior is undefined.
- **Inputs**:
    - `x`: A pointer to an array of floats representing the input data to be quantized. Must not be null and should contain at least `k` elements.
    - `y`: A pointer to a `block_tq1_0` structure where the quantized output will be stored. Must not be null and should be properly allocated before calling the function.
    - `k`: An integer representing the number of elements to process. Must be a positive integer and a multiple of `QK_K`.
- **Output**: The function does not return a value. It populates the `y` structure with the quantized data based on the input provided.
- **See also**: [`quantize_row_tq1_0_ref`](ggml-quants.c.driver.md#quantize_row_tq1_0_ref)  (Implementation)


---
### quantize\_row\_tq2\_0\_ref<!-- {{#callable_declaration:quantize_row_tq2_0_ref}} -->
Quantizes a row of floating-point values into a specified format.
- **Description**: This function is used to quantize a row of floating-point values into a compressed format suitable for storage or transmission. It must be called with a valid input array and a properly allocated output structure. The parameter `k` must be a multiple of `QK_K`, which defines the number of elements processed in each quantization block. The function computes the absolute maximum value of the input data to normalize the quantization process, and it will handle cases where the maximum value is zero by setting the corresponding quantized values accordingly.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values to be quantized. Must not be null and should contain at least 'k' elements.
    - `y`: A pointer to a pre-allocated output structure where the quantized data will be stored. Must not be null.
    - `k`: An integer representing the number of elements to process. Must be a positive integer and a multiple of QK_K.
- **Output**: The function does not return a value. It populates the output structure `y` with the quantized data based on the input array `x`.
- **See also**: [`quantize_row_tq2_0_ref`](ggml-quants.c.driver.md#quantize_row_tq2_0_ref)  (Implementation)


---
### quantize\_row\_iq3\_xxs\_ref<!-- {{#callable_declaration:quantize_row_iq3_xxs_ref}} -->
Quantizes a row of floating-point data into a specific format.
- **Description**: This function is used to quantize a row of floating-point data into a compressed format suitable for efficient storage or processing. It should be called when you have a valid input array and a destination block prepared to receive the quantized data. The parameter `k` must be a multiple of `QK_K`, which is a precondition for the function to execute correctly. If `k` does not meet this requirement, the function will assert and terminate. The quantization process modifies the destination block to store the quantized representation of the input data.
- **Inputs**:
    - `x`: A pointer to the input array of floating-point values to be quantized. Must not be null and should point to a valid memory region containing at least `k` elements.
    - `y`: A pointer to the destination block where the quantized data will be stored. Must not be null and should point to a valid memory region that can hold the quantized representation.
    - `k`: An integer representing the number of elements to quantize. Must be a non-negative integer that is a multiple of `QK_K`.
- **Output**: This function does not return a value. It modifies the destination block `y` to contain the quantized data.
- **See also**: [`quantize_row_iq3_xxs_ref`](ggml-quants.c.driver.md#quantize_row_iq3_xxs_ref)  (Implementation)


---
### quantize\_row\_iq4\_nl\_ref<!-- {{#callable_declaration:quantize_row_iq4_nl_ref}} -->
Quantizes a row of floating-point data into a specific format.
- **Description**: This function is used to quantize a row of floating-point data into a specified format, which is useful in scenarios where reduced precision is required for storage or computation efficiency. It must be called with a valid input array and a properly allocated output structure. The parameter `k` must be a multiple of `QK4_NL`, as it determines the number of elements to process. If `k` is not a valid multiple, the function will assert and terminate. The output structure will be populated with the quantized data, and the function does not return a value.
- **Inputs**:
    - `x`: A pointer to the input array of floating-point values to be quantized. Must not be null and should point to an array with at least `k` elements.
    - `y`: A pointer to a pre-allocated `block_iq4_nl` structure where the quantized output will be stored. Must not be null.
    - `k`: An integer representing the number of elements to process. Must be a positive integer and a multiple of `QK4_NL`.
- **Output**: None
- **See also**: [`quantize_row_iq4_nl_ref`](ggml-quants.c.driver.md#quantize_row_iq4_nl_ref)  (Implementation)


---
### quantize\_row\_iq4\_xs\_ref<!-- {{#callable_declaration:quantize_row_iq4_xs_ref}} -->
Quantizes a row of floating-point values into a specific format.
- **Description**: This function is used to quantize a row of floating-point values into a specified format, which is useful in scenarios where reduced precision is acceptable to save memory or improve performance. It must be called with a valid pointer to the input data and a properly allocated output structure. The quantization process is dependent on the parameter `k`, which must be a multiple of a predefined constant. If `k` does not meet this requirement, the function will trigger an assertion failure.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values to be quantized. Must not be null and should point to a valid memory region containing at least `k` elements.
    - `y`: A pointer to a `block_iq4_xs` structure where the quantized output will be stored. Must not be null and should point to a valid, allocated memory region.
    - `k`: An integer representing the number of elements to quantize. Must be a multiple of a predefined constant (QK_K). If `k` is not valid, the function will assert and terminate.
- **Output**: The function does not return a value and directly modifies the output structure pointed to by `y`.
- **See also**: [`quantize_row_iq4_xs_ref`](ggml-quants.c.driver.md#quantize_row_iq4_xs_ref)  (Implementation)


---
### quantize\_row\_iq3\_s\_ref<!-- {{#callable_declaration:quantize_row_iq3_s_ref}} -->
Quantizes a row of floating-point data into a specific format.
- **Description**: This function is intended to be used for quantizing a row of floating-point data into a specified format, which is useful in scenarios such as model compression or efficient data storage. It must be called with a valid pointer to the source data and a destination block that can hold the quantized data. The parameter `k` must be a multiple of `QK_K`, which is a precondition for the function to execute correctly. If `k` does not meet this requirement, the function will assert and terminate. The function does not return a value, but it modifies the destination block to contain the quantized representation of the input data.
- **Inputs**:
    - `x`: A pointer to the source array of floating-point values to be quantized. Must not be null and should point to a valid memory region containing at least `k` elements.
    - `y`: A pointer to the destination block where the quantized data will be stored. Must not be null and should point to a valid memory region that can accommodate the quantized output.
    - `k`: An integer representing the number of elements to quantize. Must be a non-negative integer and a multiple of `QK_K`.
- **Output**: None
- **See also**: [`quantize_row_iq3_s_ref`](ggml-quants.c.driver.md#quantize_row_iq3_s_ref)  (Implementation)


---
### quantize\_row\_iq2\_s\_ref<!-- {{#callable_declaration:quantize_row_iq2_s_ref}} -->
Quantizes a row of floating-point data into a specific format.
- **Description**: This function is intended to be used for quantizing a row of floating-point data into a specified format, which is useful in scenarios where reduced precision is required for storage or processing efficiency. It must be called with a valid pointer to the source data and a destination structure that can hold the quantized data. The parameter `k` must be a multiple of `QK_K`, which is a precondition for the function to execute correctly. If `k` does not meet this requirement, the function will not proceed, ensuring that the quantization process is only performed with valid configurations.
- **Inputs**:
    - `x`: A pointer to the source array of floating-point values to be quantized. Must not be null and should point to a valid memory region containing at least `k` elements.
    - `y`: A pointer to the destination structure where the quantized data will be stored. Must not be null and should be properly allocated to hold the quantized representation.
    - `k`: An integer representing the number of elements to quantize. Must be a multiple of `QK_K` to ensure valid quantization.
- **Output**: This function does not return a value and does not mutate any of the input parameters directly.
- **See also**: [`quantize_row_iq2_s_ref`](ggml-quants.c.driver.md#quantize_row_iq2_s_ref)  (Implementation)


---
### dequantize\_row\_q4\_0<!-- {{#callable_declaration:dequantize_row_q4_0}} -->
Dequantizes a row of quantized data.
- **Description**: This function is used to convert quantized data back into floating-point representation. It should be called when you have a block of quantized data that needs to be transformed into a usable format for further processing. The function expects that the input size `k` is a multiple of the constant `QK4_0`, which defines the quantization scheme. If this precondition is not met, the function will assert and terminate. The output is written directly to the provided output buffer, which must be large enough to hold the resulting floating-point values.
- **Inputs**:
    - `x`: A pointer to a block of quantized data of type `block_q4_0`. Must not be null and must point to valid memory containing the quantized values.
    - `y`: A pointer to a float array where the dequantized output will be stored. Must not be null and must point to a buffer large enough to hold at least `k` floating-point values.
    - `k`: An integer representing the total number of quantized elements to process. Must be a positive integer and a multiple of `QK4_0`.
- **Output**: The function does not return a value. Instead, it populates the output buffer `y` with the dequantized floating-point values derived from the input `x`.
- **See also**: [`dequantize_row_q4_0`](ggml-quants.c.driver.md#dequantize_row_q4_0)  (Implementation)


---
### dequantize\_row\_q4\_1<!-- {{#callable_declaration:dequantize_row_q4_1}} -->
Dequantizes a row of quantized data.
- **Description**: This function is used to convert quantized data back into floating-point representation. It should be called when you have a block of quantized data and you want to retrieve the original floating-point values. The function expects that the input data is properly aligned and that the length of the data (k) is a multiple of the defined quantization size. If the input data does not meet these conditions, the behavior is undefined.
- **Inputs**:
    - `x`: A pointer to a `block_q4_1` structure containing the quantized data. Must not be null.
    - `y`: A pointer to a float array where the dequantized output will be stored. Must not be null and should have enough space to hold the output data.
    - `k`: An integer representing the total number of elements to dequantize. Must be a positive integer and a multiple of the quantization size.
- **Output**: The function does not return a value. It populates the output array `y` with the dequantized floating-point values based on the input data.
- **See also**: [`dequantize_row_q4_1`](ggml-quants.c.driver.md#dequantize_row_q4_1)  (Implementation)


---
### dequantize\_row\_q5\_0<!-- {{#callable_declaration:dequantize_row_q5_0}} -->
Dequantizes a row of quantized data.
- **Description**: This function is used to convert quantized data back into floating-point representation. It should be called when you have a block of quantized data represented by `block_q5_0` and you want to retrieve the original floating-point values. The parameter `k` must be a multiple of the constant `QK5_0`, which defines the expected size of the quantized data. If `k` is not a valid multiple, the function will assert and terminate. The output is written to the provided array `y`, which must be large enough to hold the resulting floating-point values.
- **Inputs**:
    - `x`: A pointer to a `block_q5_0` structure containing the quantized data. Must not be null.
    - `y`: A pointer to a float array where the dequantized values will be stored. Must not be null and must be large enough to hold at least `k` floating-point values.
    - `k`: An integer representing the number of elements to dequantize. Must be a positive integer and a multiple of `QK5_0`.
- **Output**: The function does not return a value. Instead, it populates the array pointed to by `y` with the dequantized floating-point values.
- **See also**: [`dequantize_row_q5_0`](ggml-quants.c.driver.md#dequantize_row_q5_0)  (Implementation)


---
### dequantize\_row\_q5\_1<!-- {{#callable_declaration:dequantize_row_q5_1}} -->
Dequantizes a row of quantized data.
- **Description**: This function is used to convert quantized data back into floating-point representation. It should be called when you have a block of quantized data represented by `block_q5_1` and you want to retrieve the original floating-point values. The parameter `k` must be a multiple of the constant `QK5_1`, which defines the expected size of the quantized data blocks. If `k` is not a valid multiple, the function will assert and terminate. The output is written to the provided array `y`, which must be large enough to hold the resulting floating-point values.
- **Inputs**:
    - `x`: A pointer to a `block_q5_1` structure containing the quantized data. Must not be null.
    - `y`: A pointer to a float array where the dequantized output will be stored. Must not be null and must have sufficient space to hold the output.
    - `k`: An integer representing the total number of quantized elements to process. Must be a positive integer and a multiple of `QK5_1`.
- **Output**: The function does not return a value. Instead, it populates the array pointed to by `y` with the dequantized floating-point values.
- **See also**: [`dequantize_row_q5_1`](ggml-quants.c.driver.md#dequantize_row_q5_1)  (Implementation)


---
### dequantize\_row\_q8\_0<!-- {{#callable_declaration:dequantize_row_q8_0}} -->
Dequantizes a row of quantized data.
- **Description**: This function is used to convert quantized data back into floating-point representation. It should be called when you have a valid `block_q8_0` structure containing quantized values and you want to retrieve the corresponding floating-point values into the provided output array. The parameter `k` must be a multiple of `QK8_0`, which defines the expected size of the quantized data. If `k` is not a valid multiple, the function will assert and terminate. The output array must be large enough to hold the resulting floating-point values.
- **Inputs**:
    - `x`: A pointer to a `block_q8_0` structure containing the quantized data. Must not be null.
    - `y`: A pointer to a float array where the dequantized values will be stored. Must not be null and must have enough space to hold at least `k` floating-point values.
    - `k`: An integer representing the number of quantized elements to process. Must be a positive integer and a multiple of `QK8_0`.
- **Output**: The function does not return a value. It populates the output array `y` with the dequantized floating-point values corresponding to the input quantized data.
- **See also**: [`dequantize_row_q8_0`](ggml-quants.c.driver.md#dequantize_row_q8_0)  (Implementation)


---
### dequantize\_row\_q2\_K<!-- {{#callable_declaration:dequantize_row_q2_K}} -->
Dequantizes a row of quantized data.
- **Description**: This function is used to convert quantized data back into floating-point representation. It should be called with a valid `block_q2_K` structure that contains the quantized data and associated parameters. The parameter `k` must be a multiple of `QK_K`, as it determines the number of quantized elements to process. The output is written to the provided float array `y`, which must have sufficient space allocated to hold the resulting dequantized values. If `k` is not a multiple of `QK_K`, the behavior is undefined.
- **Inputs**:
    - `x`: A pointer to a `block_q2_K` structure containing the quantized data. Must not be null.
    - `y`: A pointer to a float array where the dequantized values will be stored. Must not be null and must have enough space to hold the output.
    - `k`: An integer representing the number of quantized elements to dequantize. Must be a multiple of `QK_K`.
- **Output**: The function does not return a value; instead, it populates the float array pointed to by `y` with the dequantized results.
- **See also**: [`dequantize_row_q2_K`](ggml-quants.c.driver.md#dequantize_row_q2_K)  (Implementation)


---
### dequantize\_row\_q3\_K<!-- {{#callable_declaration:dequantize_row_q3_K}} -->
Dequantizes a row of quantized data.
- **Description**: This function is used to convert quantized data back into floating-point representation. It should be called with a valid `block_q3_K` structure that contains the quantized data and associated metadata. The parameter `k` must be a multiple of `QK_K`, which defines the expected size of the input data. The output is written to the provided float pointer `y`, which must point to a valid memory location capable of holding the resulting dequantized values. If `k` is not a multiple of `QK_K`, the behavior is undefined.
- **Inputs**:
    - `x`: A pointer to a `block_q3_K` structure containing the quantized data. Must not be null.
    - `y`: A pointer to a float array where the dequantized values will be stored. Must point to a valid memory location.
    - `k`: An integer representing the number of elements to dequantize. Must be a multiple of `QK_K`.
- **Output**: The function does not return a value. It writes the dequantized floating-point values directly to the memory location pointed to by `y`.
- **See also**: [`dequantize_row_q3_K`](ggml-quants.c.driver.md#dequantize_row_q3_K)  (Implementation)


---
### dequantize\_row\_q4\_K<!-- {{#callable_declaration:dequantize_row_q4_K}} -->
Dequantizes a row of quantized data.
- **Description**: This function is used to convert quantized data back into floating-point representation. It should be called with a valid `block_q4_K` structure that contains the quantized data and associated parameters. The function expects the length of the data, specified by `k`, to be a multiple of `QK_K`. If `k` is not a multiple of `QK_K`, the behavior is undefined. The output is written to the provided float pointer `y`, which must be allocated with sufficient space to hold the resulting dequantized values.
- **Inputs**:
    - `x`: A pointer to a `block_q4_K` structure containing the quantized data. Must not be null.
    - `y`: A pointer to a float array where the dequantized values will be stored. Must not be null and must have enough allocated space to hold the output.
    - `k`: An integer representing the number of elements to dequantize. Must be a positive integer and a multiple of `QK_K`.
- **Output**: The function does not return a value. It writes the dequantized floating-point values directly to the memory location pointed to by `y`.
- **See also**: [`dequantize_row_q4_K`](ggml-quants.c.driver.md#dequantize_row_q4_K)  (Implementation)


---
### dequantize\_row\_q5\_K<!-- {{#callable_declaration:dequantize_row_q5_K}} -->
Dequantizes a row of quantized data.
- **Description**: This function is used to convert quantized data back into floating-point representation. It should be called with a valid `block_q5_K` structure that contains the quantized data and associated scaling factors. The parameter `k` must be a multiple of `QK_K`, which defines the number of elements processed in each block. The function will write the resulting floating-point values into the provided output buffer `y`. It is important to ensure that the output buffer has sufficient space to hold the dequantized values, as the function does not perform bounds checking on the output buffer.
- **Inputs**:
    - `x`: A pointer to a `block_q5_K` structure containing the quantized data. Must not be null.
    - `y`: A pointer to a float array where the dequantized values will be stored. Must not be null and must have enough space to hold the output.
    - `k`: An integer representing the number of elements to dequantize. Must be a multiple of `QK_K`.
- **Output**: The function does not return a value. It populates the output buffer `y` with the dequantized floating-point values.
- **See also**: [`dequantize_row_q5_K`](ggml-quants.c.driver.md#dequantize_row_q5_K)  (Implementation)


---
### dequantize\_row\_q6\_K<!-- {{#callable_declaration:dequantize_row_q6_K}} -->
Dequantizes a row of quantized data.
- **Description**: This function is used to convert quantized data back into floating-point representation. It should be called when you have a valid `block_q6_K` structure containing quantized values and you want to retrieve the corresponding floating-point values into the provided output buffer. The parameter `k` must be a multiple of `QK_K`, which defines the expected size of the quantized data. If `k` is not a valid multiple, the behavior is undefined. The output buffer must be large enough to hold the resulting floating-point values.
- **Inputs**:
    - `x`: A pointer to a `block_q6_K` structure containing the quantized data. Must not be null.
    - `y`: A pointer to a float array where the dequantized values will be stored. Must not be null and must have sufficient space to hold the output.
    - `k`: An integer representing the number of elements to dequantize. Must be a positive integer and a multiple of `QK_K`.
- **Output**: The function does not return a value. It populates the output buffer `y` with the dequantized floating-point values.
- **See also**: [`dequantize_row_q6_K`](ggml-quants.c.driver.md#dequantize_row_q6_K)  (Implementation)


---
### dequantize\_row\_q8\_K<!-- {{#callable_declaration:dequantize_row_q8_K}} -->
Dequantizes a row of quantized data.
- **Description**: This function is used to convert quantized data back into floating-point representation. It should be called when you have a valid `block_q8_K` structure containing quantized values and you want to retrieve the corresponding floating-point values into the provided output buffer. The parameter `k` must be a multiple of `QK_K`, which defines the expected size of the quantized data. If `k` is not a valid multiple, the behavior is undefined. The output buffer must be large enough to hold the resulting floating-point values.
- **Inputs**:
    - `x`: A pointer to a `block_q8_K` structure containing the quantized data. Must not be null.
    - `y`: A pointer to a float array where the dequantized values will be stored. Must not be null and must have sufficient space to hold the output data.
    - `k`: An integer representing the number of elements to dequantize. Must be a non-negative integer and a multiple of `QK_K`.
- **Output**: The function does not return a value. It populates the output buffer `y` with the dequantized floating-point values derived from the input `x`.
- **See also**: [`dequantize_row_q8_K`](ggml-quants.c.driver.md#dequantize_row_q8_K)  (Implementation)


---
### dequantize\_row\_tq1\_0<!-- {{#callable_declaration:dequantize_row_tq1_0}} -->
Dequantizes a row of data from a specified block.
- **Description**: This function is used to convert quantized data back into its floating-point representation. It should be called with a valid `block_tq1_0` structure that contains the quantized data and a destination buffer for the resulting floating-point values. The parameter `k` must be a multiple of `QK_K`, as it determines the number of elements to process. The function will write the dequantized values into the provided output buffer, which must be large enough to hold the results. If the input parameters do not meet the specified requirements, the behavior is undefined.
- **Inputs**:
    - `x`: A pointer to a `block_tq1_0` structure containing the quantized data. Must not be null.
    - `y`: A pointer to a float array where the dequantized values will be stored. Must not be null and must have enough space to hold the output.
    - `k`: An integer representing the number of elements to dequantize. Must be a multiple of `QK_K`.
- **Output**: The function does not return a value. It populates the output buffer pointed to by `y` with the dequantized floating-point values.
- **See also**: [`dequantize_row_tq1_0`](ggml-quants.c.driver.md#dequantize_row_tq1_0)  (Implementation)


---
### dequantize\_row\_tq2\_0<!-- {{#callable_declaration:dequantize_row_tq2_0}} -->
Dequantizes a row of data from a specified block.
- **Description**: This function is used to convert quantized data back into its floating-point representation. It should be called when you have a block of quantized data and you want to retrieve the original floating-point values. The parameter `k` must be a multiple of `QK_K`, as it determines the number of quantized blocks to process. The function will write the dequantized values into the provided output buffer, which must be large enough to hold the resulting data.
- **Inputs**:
    - `x`: A pointer to a `block_tq2_0` structure containing the quantized data. Must not be null.
    - `y`: A pointer to a float array where the dequantized values will be stored. Must not be null and must have sufficient space to hold the output.
    - `k`: An integer representing the number of quantized elements to process. Must be a non-negative integer and a multiple of `QK_K`.
- **Output**: The function does not return a value. It populates the output buffer pointed to by `y` with the dequantized floating-point values.
- **See also**: [`dequantize_row_tq2_0`](ggml-quants.c.driver.md#dequantize_row_tq2_0)  (Implementation)


---
### dequantize\_row\_iq2\_xxs<!-- {{#callable_declaration:dequantize_row_iq2_xxs}} -->
Dequantizes a row of data from a specific format.
- **Description**: This function is used to convert quantized data back into its floating-point representation. It should be called with a valid pointer to a `block_iq2_xxs` structure containing the quantized data, and a pointer to a float array where the dequantized values will be stored. The parameter `k` must be a multiple of `QK_K`, as it determines the number of elements to process. If `k` is not a valid multiple, the behavior is undefined. The function will write the dequantized values into the provided output array, which must have sufficient space allocated to hold the results.
- **Inputs**:
    - `x`: A pointer to a `block_iq2_xxs` structure containing the quantized data. Must not be null.
    - `y`: A pointer to a float array where the dequantized values will be stored. Must not be null and must have enough space allocated for the output.
    - `k`: An integer representing the number of elements to dequantize. Must be a multiple of `QK_K`.
- **Output**: The function does not return a value. It populates the output array `y` with the dequantized floating-point values.
- **See also**: [`dequantize_row_iq2_xxs`](ggml-quants.c.driver.md#dequantize_row_iq2_xxs)  (Implementation)


---
### dequantize\_row\_iq2\_xs<!-- {{#callable_declaration:dequantize_row_iq2_xs}} -->
Dequantizes a row of data from a specific block format.
- **Description**: This function is used to convert quantized data back into its floating-point representation. It should be called when you have a valid `block_iq2_xs` structure containing quantized data and you want to retrieve the corresponding floating-point values. The parameter `k` must be a multiple of `QK_K`, which defines the expected size of the input data. The function will write the dequantized values into the provided output buffer, which must be large enough to hold the resulting data.
- **Inputs**:
    - `x`: A pointer to a `block_iq2_xs` structure containing the quantized data. Must not be null.
    - `y`: A pointer to a float array where the dequantized values will be stored. The caller retains ownership of this buffer, which must be large enough to hold the output data.
    - `k`: An integer representing the number of elements to dequantize. Must be a non-negative integer that is a multiple of `QK_K`.
- **Output**: The function does not return a value. Instead, it populates the output buffer `y` with the dequantized floating-point values.
- **See also**: [`dequantize_row_iq2_xs`](ggml-quants.c.driver.md#dequantize_row_iq2_xs)  (Implementation)


---
### dequantize\_row\_iq2\_s<!-- {{#callable_declaration:dequantize_row_iq2_s}} -->
Dequantizes a row of data from a specific format.
- **Description**: This function is used to convert data from a quantized format back to its floating-point representation. It should be called when you have a block of quantized data that needs to be processed or analyzed in its original form. The function expects the quantized data to be structured correctly and the length `k` to be a multiple of `QK_K`. If `k` is not a valid multiple, the behavior is undefined.
- **Inputs**:
    - `x`: A pointer to a constant array of `block_iq2_s` structures containing the quantized data. Must not be null.
    - `y`: A pointer to a float array where the dequantized output will be stored. Must not be null and should have sufficient space to hold the output.
    - `k`: An integer representing the number of elements to dequantize. Must be a positive integer and a multiple of `QK_K`.
- **Output**: The function does not return a value. It populates the output array `y` with the dequantized floating-point values.
- **See also**: [`dequantize_row_iq2_s`](ggml-quants.c.driver.md#dequantize_row_iq2_s)  (Implementation)


---
### dequantize\_row\_iq3\_xxs<!-- {{#callable_declaration:dequantize_row_iq3_xxs}} -->
Dequantizes a row of data from a specific format.
- **Description**: This function is used to convert quantized data back into its floating-point representation. It should be called when you have a block of quantized data represented by `x` and you want to retrieve the corresponding floating-point values into the array `y`. The parameter `k` must be a multiple of `QK_K`, which defines the expected size of the quantized data. If `k` is not a valid multiple, the function will assert and terminate. The output array `y` will be populated with the dequantized values, and the function does not return a value.
- **Inputs**:
    - `x`: A pointer to a `block_iq3_xxs` structure containing the quantized data. Must not be null.
    - `y`: A pointer to a float array where the dequantized values will be stored. Must not be null and should have enough space to hold the output.
    - `k`: An integer representing the number of elements to dequantize. Must be a multiple of `QK_K`.
- **Output**: None
- **See also**: [`dequantize_row_iq3_xxs`](ggml-quants.c.driver.md#dequantize_row_iq3_xxs)  (Implementation)


---
### dequantize\_row\_iq1\_s<!-- {{#callable_declaration:dequantize_row_iq1_s}} -->
Dequantizes a row of data from a specified block.
- **Description**: This function is used to convert quantized data back into its floating-point representation. It should be called when you have a block of quantized data and need to retrieve the original values. The parameter `k` must be a multiple of `QK_K`, which defines the number of elements processed in each block. The function will write the dequantized values into the provided output buffer. Ensure that the output buffer is large enough to hold the resulting data.
- **Inputs**:
    - `x`: A pointer to a constant `block_iq1_s` structure containing the quantized data. Must not be null.
    - `y`: A pointer to a float array where the dequantized values will be stored. The caller retains ownership of this buffer, which must be large enough to hold the output.
    - `k`: An integer representing the number of elements to dequantize. Must be a non-negative integer and a multiple of `QK_K`.
- **Output**: The function does not return a value. It populates the output buffer `y` with the dequantized floating-point values.
- **See also**: [`dequantize_row_iq1_s`](ggml-quants.c.driver.md#dequantize_row_iq1_s)  (Implementation)


---
### dequantize\_row\_iq1\_m<!-- {{#callable_declaration:dequantize_row_iq1_m}} -->
Dequantizes a row of data from a specified block.
- **Description**: This function is used to convert quantized data back into its floating-point representation. It should be called when you have a valid `block_iq1_m` structure containing quantized data and you want to retrieve the corresponding floating-point values. The parameter `k` must be a multiple of `QK_K`, which defines the number of elements processed in each block. The function will write the dequantized values into the provided output buffer, which must be large enough to hold the results.
- **Inputs**:
    - `x`: A pointer to a `block_iq1_m` structure containing the quantized data. Must not be null.
    - `y`: A pointer to a float array where the dequantized values will be stored. Must not be null and must have sufficient space to hold the output.
    - `k`: An integer representing the number of elements to dequantize. Must be a non-negative integer and a multiple of `QK_K`.
- **Output**: The function does not return a value. It populates the output buffer `y` with the dequantized floating-point values.
- **See also**: [`dequantize_row_iq1_m`](ggml-quants.c.driver.md#dequantize_row_iq1_m)  (Implementation)


---
### dequantize\_row\_iq4\_nl<!-- {{#callable_declaration:dequantize_row_iq4_nl}} -->
Dequantizes a row of data from a block structure.
- **Description**: This function is used to convert quantized data back into its floating-point representation. It should be called with a valid `block_iq4_nl` structure that contains the quantized data and a pre-allocated output buffer for the resulting floating-point values. The parameter `k` must be a multiple of `QK4_NL`, which defines the number of elements processed in each block. If `k` is not a valid multiple, the behavior is undefined. The function will populate the output buffer with the dequantized values, which will be written in place, and the output buffer must be large enough to hold the results.
- **Inputs**:
    - `x`: A pointer to a `block_iq4_nl` structure containing the quantized data. Must not be null.
    - `y`: A pointer to a pre-allocated float array where the dequantized values will be stored. Must not be null and must have enough space to hold the output.
    - `k`: An integer representing the number of elements to dequantize. Must be a multiple of `QK4_NL`.
- **Output**: The function does not return a value. It populates the output buffer `y` with the dequantized floating-point values.
- **See also**: [`dequantize_row_iq4_nl`](ggml-quants.c.driver.md#dequantize_row_iq4_nl)  (Implementation)


---
### dequantize\_row\_iq4\_xs<!-- {{#callable_declaration:dequantize_row_iq4_xs}} -->
Dequantizes a row of data from a block.
- **Description**: This function is used to convert quantized data back into its floating-point representation. It should be called when you have a block of quantized data and you want to retrieve the original floating-point values. The parameter `k` must be a multiple of `QK_K`, which defines the number of elements processed in each block. The function will write the dequantized values into the provided output buffer, which must be large enough to hold the results. It is important to ensure that the input block is valid and properly initialized before calling this function.
- **Inputs**:
    - `x`: A pointer to a `block_iq4_xs` structure containing the quantized data. Must not be null and should point to a valid block.
    - `y`: A pointer to a float array where the dequantized values will be stored. Must not be null and should have enough space to hold the output data.
    - `k`: An integer representing the number of elements to dequantize. Must be a non-negative integer and a multiple of `QK_K`.
- **Output**: The function does not return a value. It populates the output buffer `y` with the dequantized floating-point values based on the input block `x`.
- **See also**: [`dequantize_row_iq4_xs`](ggml-quants.c.driver.md#dequantize_row_iq4_xs)  (Implementation)


---
### dequantize\_row\_iq3\_s<!-- {{#callable_declaration:dequantize_row_iq3_s}} -->
Dequantizes a row of data from a specified block.
- **Description**: This function is used to convert quantized data back into its floating-point representation. It should be called with a valid `block_iq3_s` structure that contains the quantized data and associated metadata. The parameter `k` must be a multiple of `QK_K`, which defines the number of elements to process. The function will write the dequantized values into the provided output buffer `y`, which must be large enough to hold the results. If `x` is null or `y` is null, the behavior is undefined.
- **Inputs**:
    - `x`: A pointer to a `block_iq3_s` structure containing the quantized data. Must not be null.
    - `y`: A pointer to a float array where the dequantized values will be stored. Must not be null and must have sufficient space to hold the output.
    - `k`: An integer representing the number of elements to dequantize. Must be a multiple of `QK_K`.
- **Output**: The function does not return a value. It populates the output buffer `y` with the dequantized floating-point values.
- **See also**: [`dequantize_row_iq3_s`](ggml-quants.c.driver.md#dequantize_row_iq3_s)  (Implementation)


---
### quantize\_iq2\_xxs<!-- {{#callable_declaration:quantize_iq2_xxs}} -->
Quantizes input data using an importance matrix.
- **Description**: This function is used to quantize a set of input data represented as floating-point values into a more compact format, utilizing an importance matrix to guide the quantization process. It should be called when there is a need to reduce the memory footprint of the data while preserving its essential characteristics. The function expects that the number of elements per row is a multiple of `QK_K`, and it will process the specified number of rows, writing the quantized output to the provided destination buffer. It is important to ensure that the destination buffer is large enough to hold the quantized data, as the function does not perform bounds checking on the output buffer size. The function will return the total size of the quantized data produced.
- **Inputs**:
    - `src`: Pointer to the source array of floating-point values to be quantized. Must not be null.
    - `dst`: Pointer to the destination buffer where the quantized data will be stored. Must not be null and must be large enough to hold the output data.
    - `nrow`: The number of rows to process. Must be a positive integer.
    - `n_per_row`: The number of elements per row. Must be a positive integer and a multiple of `QK_K`.
    - `quant_weights`: Pointer to the array of quantization weights used to guide the quantization process. Must not be null.
- **Output**: Returns the total size of the quantized data produced in bytes.
- **See also**: [`quantize_iq2_xxs`](ggml-quants.c.driver.md#quantize_iq2_xxs)  (Implementation)


---
### quantize\_iq2\_xs<!-- {{#callable_declaration:quantize_iq2_xs}} -->
Quantizes input data into a specified format.
- **Description**: This function is used to quantize a source array of floating-point values into a destination buffer, which is expected to be pre-allocated. It is essential to ensure that the number of elements per row (`n_per_row`) is a multiple of `QK_K`, as this is a requirement for the quantization process. The function should be called after initializing the necessary data structures and before using the quantized data. The quantization process will iterate over the specified number of rows (`nrow`), applying the quantization weights provided in `quant_weights` to each row of the source data.
- **Inputs**:
    - `src`: Pointer to the source array of floating-point values to be quantized. Must not be null.
    - `dst`: Pointer to the destination buffer where the quantized data will be stored. Must not be null and should be large enough to hold the quantized output.
    - `nrow`: The number of rows to process. Must be a non-negative integer.
    - `n_per_row`: The number of elements per row. Must be a positive integer and a multiple of QK_K.
    - `quant_weights`: Pointer to an array of quantization weights. Must not be null.
- **Output**: Returns the total size in bytes of the quantized data written to the destination buffer.
- **See also**: [`quantize_iq2_xs`](ggml-quants.c.driver.md#quantize_iq2_xs)  (Implementation)


---
### quantize\_iq2\_s<!-- {{#callable_declaration:quantize_iq2_s}} -->
Quantizes input data into a compressed format.
- **Description**: This function is used to quantize a set of floating-point data into a more compact representation, which is useful for reducing memory usage and improving performance in machine learning applications. It should be called when you have a source array of floating-point values that you want to quantize, and you have allocated sufficient memory for the destination buffer. The function expects that the number of elements per row is a multiple of `QK_K`, and it will process the input data row by row, applying quantization weights to each row. If the input parameters do not meet the expected conditions, the function may not behave as intended.
- **Inputs**:
    - `src`: Pointer to the source array of floating-point values to be quantized. Must not be null.
    - `dst`: Pointer to the destination buffer where the quantized data will be stored. Must not be null and should have enough allocated space to hold the output.
    - `nrow`: The number of rows in the source data. Must be a positive integer.
    - `n_per_row`: The number of elements per row in the source data. Must be a positive integer and a multiple of `QK_K`.
    - `quant_weights`: Pointer to an array of quantization weights. Must not be null.
- **Output**: Returns the total size in bytes of the quantized data written to the destination buffer.
- **See also**: [`quantize_iq2_s`](ggml-quants.c.driver.md#quantize_iq2_s)  (Implementation)


---
### quantize\_iq3\_xxs<!-- {{#callable_declaration:quantize_iq3_xxs}} -->
Quantizes input data into a compressed format.
- **Description**: This function is used to quantize a set of floating-point data into a more compact representation, which is useful for reducing memory usage and improving performance in machine learning applications. It must be called with a valid source array and a destination buffer that is large enough to hold the quantized data. The function expects the number of elements per row to be a multiple of a predefined constant, and it will process the input data row by row, applying quantization weights to each row. If the input parameters do not meet the expected conditions, the function may not behave as intended.
- **Inputs**:
    - `src`: Pointer to the source array of floating-point values to be quantized. Must not be null.
    - `dst`: Pointer to the destination buffer where the quantized data will be stored. Must not be null and must be large enough to hold the output.
    - `nrow`: The number of rows to process. Must be a positive integer.
    - `n_per_row`: The number of elements in each row. Must be a positive integer and a multiple of QK_K.
    - `quant_weights`: Pointer to an array of quantization weights. Must not be null.
- **Output**: Returns the total size in bytes of the quantized data produced in the destination buffer.
- **See also**: [`quantize_iq3_xxs`](ggml-quants.c.driver.md#quantize_iq3_xxs)  (Implementation)


---
### quantize\_iq1\_s<!-- {{#callable_declaration:quantize_iq1_s}} -->
Quantizes input data into a specific format.
- **Description**: This function is used to quantize a set of floating-point data into a more compact representation, which is useful for reducing memory usage and improving performance in machine learning applications. It must be called with a valid source array and a destination buffer that is large enough to hold the quantized data. The function expects the number of elements per row to be a multiple of a predefined constant, and it will assert if this condition is not met. The quantization process is performed for a specified number of rows, and the function will return the total size of the quantized data produced.
- **Inputs**:
    - `src`: Pointer to the source array of floating-point values to be quantized. Must not be null.
    - `dst`: Pointer to the destination buffer where the quantized data will be stored. Must not be null and should be large enough to hold the output.
    - `nrow`: The number of rows to process. Must be a positive integer.
    - `n_per_row`: The number of elements per row. Must be a positive integer and a multiple of QK_K.
    - `quant_weights`: Pointer to an array of quantization weights. Must not be null.
- **Output**: Returns the total size in bytes of the quantized data produced.
- **See also**: [`quantize_iq1_s`](ggml-quants.c.driver.md#quantize_iq1_s)  (Implementation)


---
### quantize\_iq1\_m<!-- {{#callable_declaration:quantize_iq1_m}} -->
Quantizes input data into a specified format.
- **Description**: This function is used to quantize a matrix of floating-point values into a more compact representation, which is useful for reducing memory usage and improving performance in machine learning applications. It must be called with a valid source matrix and a destination buffer that is large enough to hold the quantized data. The number of elements per row must be a multiple of a predefined constant, and the function will assert this condition. The quantization process is performed for each row of the input matrix, and the function will return the total size of the quantized data produced.
- **Inputs**:
    - `src`: Pointer to the source array of floating-point values to be quantized. Must not be null.
    - `dst`: Pointer to the destination buffer where the quantized data will be stored. Must not be null and should be large enough to hold the output.
    - `nrow`: The number of rows in the source matrix. Must be a non-negative integer.
    - `n_per_row`: The number of elements in each row of the source matrix. Must be a positive integer and a multiple of a predefined constant.
    - `quant_weights`: Pointer to an array of quantization weights used during the quantization process. Must not be null.
- **Output**: Returns the total size in bytes of the quantized data produced, calculated as the product of the number of rows and the number of blocks.
- **See also**: [`quantize_iq1_m`](ggml-quants.c.driver.md#quantize_iq1_m)  (Implementation)


---
### quantize\_iq4\_nl<!-- {{#callable_declaration:quantize_iq4_nl}} -->
Quantizes a source array into a destination buffer.
- **Description**: This function is used to quantize a floating-point source array into a specified format for efficient storage or processing. It should be called when you have a source array of floats that you want to convert into a quantized representation, typically for machine learning or data compression purposes. The function requires that the number of elements per row is a multiple of a predefined constant, ensuring proper alignment for quantization. If the input parameters are invalid, such as a non-multiple of the required size, the function will assert and terminate.
- **Inputs**:
    - `src`: Pointer to the source array of floats to be quantized. Must not be null.
    - `dst`: Pointer to the destination buffer where the quantized data will be stored. Must not be null and must have sufficient allocated space to hold the output.
    - `nrow`: The number of rows in the source array. Must be a positive integer.
    - `n_per_row`: The number of elements per row in the source array. Must be a positive integer and a multiple of QK4_NL.
    - `quant_weights`: Pointer to an optional array of quantization weights. Can be null if no weights are used.
- **Output**: Returns the total size in bytes of the quantized data written to the destination buffer.
- **See also**: [`quantize_iq4_nl`](ggml-quants.c.driver.md#quantize_iq4_nl)  (Implementation)


---
### quantize\_iq4\_xs<!-- {{#callable_declaration:quantize_iq4_xs}} -->
Quantizes input data into a specific format.
- **Description**: This function is used to quantize a set of floating-point data into a compressed format suitable for efficient storage or processing. It should be called when you have a source array of floats that you want to quantize into a destination buffer, with the number of rows and the number of elements per row specified. The function expects that the number of elements per row is a multiple of a predefined constant, and it will assert if this condition is not met. The quantization process may utilize optional quantization weights to adjust the quantization behavior. The output will be written to the destination buffer, which must be large enough to hold the quantized data.
- **Inputs**:
    - `src`: Pointer to the source array of floats to be quantized. Must not be null and should point to a valid memory region containing at least nrow * n_per_row elements.
    - `dst`: Pointer to the destination buffer where the quantized data will be stored. Must not be null and should be allocated with sufficient size to hold the output data.
    - `nrow`: The number of rows of data to quantize. Must be a positive integer.
    - `n_per_row`: The number of elements per row. Must be a positive integer and a multiple of QK_K.
    - `quant_weights`: Pointer to an optional array of quantization weights. Can be null, in which case default weights will be used. If provided, it should point to a valid memory region with at least nblock * QK_K elements.
- **Output**: Returns the total size in bytes of the quantized data written to the destination buffer.
- **See also**: [`quantize_iq4_xs`](ggml-quants.c.driver.md#quantize_iq4_xs)  (Implementation)


---
### quantize\_iq3\_s<!-- {{#callable_declaration:quantize_iq3_s}} -->
Quantizes input data into a specific format.
- **Description**: This function is used to quantize a set of floating-point data into a more compact representation, which is useful for reducing memory usage and improving performance in machine learning applications. It must be called with a valid source array and a destination buffer that is large enough to hold the quantized data. The function expects the number of elements per row to be a multiple of a predefined constant, and it will process the input data row by row. If the input parameters do not meet the expected conditions, the function may not behave as intended.
- **Inputs**:
    - `src`: Pointer to the source array of floating-point values to be quantized. Must not be null.
    - `dst`: Pointer to the destination buffer where the quantized data will be stored. Must not be null and should be large enough to hold the output data.
    - `nrow`: The number of rows to process. Must be a non-negative integer.
    - `n_per_row`: The number of elements per row. Must be a positive integer and a multiple of a predefined constant.
    - `quant_weights`: Pointer to an array of quantization weights. Must not be null and should match the expected size for the quantization process.
- **Output**: Returns the total size in bytes of the quantized data produced in the destination buffer.
- **See also**: [`quantize_iq3_s`](ggml-quants.c.driver.md#quantize_iq3_s)  (Implementation)


---
### quantize\_tq1\_0<!-- {{#callable_declaration:quantize_tq1_0}} -->
Quantizes a source array into a destination buffer.
- **Description**: This function is used to quantize a source array of floating-point values into a specified destination buffer, which is typically used for efficient storage or processing. It should be called when you need to convert a continuous range of floating-point values into a quantized format, particularly when working with large datasets. The function expects the source pointer to be valid and non-null, and the destination buffer must be large enough to hold the quantized data. The parameters `nrow` and `n_per_row` define the dimensions of the data being quantized, and it is important that these values are positive. If invalid values are provided, such as negative dimensions, the behavior is undefined.
- **Inputs**:
    - `src`: Pointer to the source array of floating-point values to be quantized. Must not be null.
    - `dst`: Pointer to the destination buffer where the quantized data will be stored. Must not be null and must be large enough to hold the quantized output.
    - `nrow`: The number of rows in the source data. Must be a positive integer.
    - `n_per_row`: The number of elements per row in the source data. Must be a positive integer.
    - `quant_weights`: Pointer to an array of quantization weights. This parameter is not used in the current implementation.
- **Output**: Returns the total size of the quantized data in bytes, calculated as the product of the number of rows and the size of each row in the quantized format.
- **See also**: [`quantize_tq1_0`](ggml-quants.c.driver.md#quantize_tq1_0)  (Implementation)


---
### quantize\_tq2\_0<!-- {{#callable_declaration:quantize_tq2_0}} -->
Quantizes a source array into a destination buffer.
- **Description**: This function is used to quantize a source array of floating-point values into a specified destination buffer, which is typically used for efficient storage or processing. It should be called when you need to convert a floating-point representation into a quantized format, and it requires that the destination buffer has been allocated with sufficient size to hold the quantized data. The function does not utilize the `quant_weights` parameter, so it can be passed as `NULL`. It is important to ensure that `nrow` and `n_per_row` are positive integers, as negative or zero values may lead to undefined behavior.
- **Inputs**:
    - `src`: Pointer to the source array of floating-point values. Must not be null.
    - `dst`: Pointer to the destination buffer where the quantized data will be stored. Must not be null.
    - `nrow`: The number of rows to be quantized. Must be a positive integer.
    - `n_per_row`: The number of elements per row. Must be a positive integer.
    - `quant_weights`: Pointer to quantization weights, which is not used in this function. Can be passed as null.
- **Output**: Returns the total size in bytes of the quantized data produced in the destination buffer.
- **See also**: [`quantize_tq2_0`](ggml-quants.c.driver.md#quantize_tq2_0)  (Implementation)


---
### quantize\_q2\_K<!-- {{#callable_declaration:quantize_q2_K}} -->
Quantizes a source array into a destination buffer.
- **Description**: This function is used to quantize a floating-point source array into a specified format, storing the result in a destination buffer. It should be called when you need to convert a dense representation of data into a more compact form, which is useful for reducing memory usage and improving performance in certain applications. The function requires the number of rows and the number of elements per row to be specified. If the `quant_weights` parameter is null, a reference quantization method is used; otherwise, a specific implementation is applied. It is important to ensure that the destination buffer is large enough to hold the quantized data, as the function does not perform bounds checking on the destination.
- **Inputs**:
    - `src`: Pointer to the source array of floats to be quantized. Must not be null.
    - `dst`: Pointer to the destination buffer where the quantized data will be stored. Must not be null and must have sufficient space to hold the output.
    - `nrow`: The number of rows in the source data. Must be a positive integer.
    - `n_per_row`: The number of elements per row in the source data. Must be a positive integer.
    - `quant_weights`: Pointer to an array of quantization weights. Can be null, in which case a reference quantization method is used.
- **Output**: Returns the total size of the quantized data in bytes, calculated as the number of rows multiplied by the size of each row.
- **See also**: [`quantize_q2_K`](ggml-quants.c.driver.md#quantize_q2_K)  (Implementation)


---
### quantize\_q3\_K<!-- {{#callable_declaration:quantize_q3_K}} -->
Quantizes a source array into a destination buffer.
- **Description**: This function is used to quantize a floating-point source array into a specified format, storing the result in a destination buffer. It should be called when you need to convert a dense representation of data into a more compact form, which is useful for reducing memory usage and improving performance in certain applications. The function requires the number of rows and the number of elements per row to be specified. If the `quant_weights` parameter is provided, it will be used during the quantization process; otherwise, a reference quantization method will be applied. It is important to ensure that the destination buffer is large enough to hold the quantized data, as the function does not perform bounds checking on the destination buffer.
- **Inputs**:
    - `src`: Pointer to the source array of floats to be quantized. Must not be null.
    - `dst`: Pointer to the destination buffer where the quantized data will be stored. Must not be null and must have sufficient space to hold the quantized output.
    - `nrow`: The number of rows in the source array. Must be a positive integer.
    - `n_per_row`: The number of elements per row in the source array. Must be a positive integer.
    - `quant_weights`: Pointer to an array of quantization weights. Can be null, in which case a reference quantization method is used.
- **Output**: Returns the total size of the quantized data in bytes, calculated as the number of rows multiplied by the size of each quantized row.
- **See also**: [`quantize_q3_K`](ggml-quants.c.driver.md#quantize_q3_K)  (Implementation)


---
### quantize\_q4\_K<!-- {{#callable_declaration:quantize_q4_K}} -->
Quantizes a source array into a destination buffer.
- **Description**: This function is used to quantize a floating-point source array into a specified format, storing the result in a destination buffer. It should be called when you need to convert a dense representation of data into a more compact form, which is useful for reducing memory usage and improving performance in certain applications. The function requires the number of rows and the number of elements per row to be specified. If the `quant_weights` parameter is provided, it will be used during the quantization process; otherwise, a reference quantization method will be applied. It is important to ensure that the destination buffer is large enough to hold the quantized data, as the function does not perform bounds checking on the destination buffer.
- **Inputs**:
    - `src`: Pointer to the source array of floats to be quantized. Must not be null.
    - `dst`: Pointer to the destination buffer where the quantized data will be stored. Must not be null and must have sufficient space to hold the quantized output.
    - `nrow`: The number of rows in the source array. Must be a positive integer.
    - `n_per_row`: The number of elements per row in the source array. Must be a positive integer.
    - `quant_weights`: Pointer to an array of quantization weights. Can be null, in which case a reference quantization method is used.
- **Output**: Returns the total size of the quantized data in bytes, calculated as the number of rows multiplied by the size of each row.
- **See also**: [`quantize_q4_K`](ggml-quants.c.driver.md#quantize_q4_K)  (Implementation)


---
### quantize\_q5\_K<!-- {{#callable_declaration:quantize_q5_K}} -->
Quantizes a source array into a destination buffer.
- **Description**: This function is used to quantize a floating-point array into a specified format, which can be useful for reducing memory usage and improving performance in machine learning applications. It should be called with a valid source array and a destination buffer that has been allocated to hold the quantized data. The function can handle both cases where quantization weights are provided or not; if weights are not provided, a reference quantization method is used. It is important to ensure that the dimensions specified by `nrow` and `n_per_row` are correct, as incorrect values may lead to undefined behavior.
- **Inputs**:
    - `src`: Pointer to the source array of floats to be quantized. Must not be null.
    - `dst`: Pointer to the destination buffer where the quantized data will be stored. Must not be null.
    - `nrow`: The number of rows to be quantized. Must be a positive integer.
    - `n_per_row`: The number of elements per row in the source array. Must be a positive integer.
    - `quant_weights`: Pointer to an array of quantization weights. Can be null, in which case a reference quantization method is used.
- **Output**: Returns the total size of the quantized data in bytes, calculated as the product of the number of rows and the size of each row in the quantized format.
- **See also**: [`quantize_q5_K`](ggml-quants.c.driver.md#quantize_q5_K)  (Implementation)


---
### quantize\_q6\_K<!-- {{#callable_declaration:quantize_q6_K}} -->
Quantizes a source array into a destination buffer.
- **Description**: This function is used to quantize a floating-point source array into a specified format, storing the result in a destination buffer. It should be called when you need to convert a dense representation of data into a more compact form, which is useful for reducing memory usage and improving performance in certain applications. The function requires the number of rows and the number of elements per row to be specified. If the `quant_weights` parameter is provided, it will be used during the quantization process; otherwise, a reference quantization method will be applied. It is important to ensure that the destination buffer is large enough to hold the quantized data, as the function does not perform bounds checking on the destination.
- **Inputs**:
    - `src`: Pointer to the source array of floats to be quantized. Must not be null.
    - `dst`: Pointer to the destination buffer where the quantized data will be stored. Must not be null.
    - `nrow`: The number of rows in the source array. Must be a positive integer.
    - `n_per_row`: The number of elements per row in the source array. Must be a positive integer.
    - `quant_weights`: Pointer to an array of quantization weights. Can be null, in which case a reference quantization method is used.
- **Output**: Returns the total size of the quantized data in bytes, calculated as the number of rows multiplied by the size of each row.
- **See also**: [`quantize_q6_K`](ggml-quants.c.driver.md#quantize_q6_K)  (Implementation)


---
### quantize\_q4\_0<!-- {{#callable_declaration:quantize_q4_0}} -->
Quantizes a source array into a destination buffer.
- **Description**: This function is used to quantize a floating-point source array into a specified format, storing the result in a destination buffer. It should be called when you need to convert a set of floating-point values into a quantized representation, which is often required for efficient storage or processing in machine learning applications. The function expects the source and destination buffers to be properly allocated and sized according to the number of rows and the number of elements per row. If the `quant_weights` parameter is null, a reference quantization method is used; otherwise, a specific quantization method utilizing the provided weights is applied. It is important to ensure that the `nrow` and `n_per_row` parameters are positive, as negative or zero values may lead to undefined behavior.
- **Inputs**:
    - `src`: Pointer to the source array of floating-point values. Must not be null and should point to a valid memory region containing at least `nrow * n_per_row` elements.
    - `dst`: Pointer to the destination buffer where the quantized data will be stored. Must not be null and should be allocated with sufficient size to hold the quantized output.
    - `nrow`: The number of rows to process. Must be a positive integer.
    - `n_per_row`: The number of elements per row. Must be a positive integer.
    - `quant_weights`: Pointer to an array of quantization weights. Can be null, in which case a reference quantization method is used. If not null, it must point to a valid memory region.
- **Output**: Returns the total size of the quantized data in bytes, calculated as `nrow` multiplied by the size of each quantized row.
- **See also**: [`quantize_q4_0`](ggml-quants.c.driver.md#quantize_q4_0)  (Implementation)


---
### quantize\_q4\_1<!-- {{#callable_declaration:quantize_q4_1}} -->
Quantizes a source array into a destination buffer.
- **Description**: This function is used to quantize a source array of floating-point values into a specified format, storing the result in a destination buffer. It is essential to call this function with valid parameters, particularly ensuring that the `src` and `dst` pointers are not null. The function can handle quantization with or without an importance matrix, indicated by the `quant_weights` parameter. If `quant_weights` is null, a reference quantization method is used. The function processes the data in rows, and the number of rows and the number of elements per row must be specified. The output size is determined by the number of rows and the size of each quantized row.
- **Inputs**:
    - `src`: Pointer to the source array of floating-point values to be quantized. Must not be null.
    - `dst`: Pointer to the destination buffer where the quantized data will be stored. Must not be null.
    - `nrow`: The number of rows to process. Must be a positive integer.
    - `n_per_row`: The number of elements per row. Must be a positive integer.
    - `quant_weights`: Pointer to an array of quantization weights. Can be null, in which case a reference quantization method is used.
- **Output**: Returns the total size of the quantized data in bytes, calculated as the number of rows multiplied by the size of each quantized row.
- **See also**: [`quantize_q4_1`](ggml-quants.c.driver.md#quantize_q4_1)  (Implementation)


---
### quantize\_q5\_0<!-- {{#callable_declaration:quantize_q5_0}} -->
Quantizes a source array into a destination buffer.
- **Description**: This function is used to quantize a floating-point source array into a specified format, storing the result in a destination buffer. It should be called when you need to convert floating-point data into a quantized representation, which is often necessary for reducing memory usage or improving performance in machine learning applications. The function requires the number of rows and the number of elements per row to be specified. If the `quant_weights` parameter is null, a reference quantization method is used. The function handles the quantization process row by row, and it is important to ensure that the destination buffer is large enough to hold the quantized data.
- **Inputs**:
    - `src`: Pointer to the source array of floats to be quantized. Must not be null.
    - `dst`: Pointer to the destination buffer where the quantized data will be stored. Must not be null.
    - `nrow`: The number of rows in the source array. Must be a positive integer.
    - `n_per_row`: The number of elements per row in the source array. Must be a positive integer.
    - `quant_weights`: Pointer to an array of quantization weights. Can be null, in which case a reference quantization method is used.
- **Output**: Returns the total size of the quantized data in bytes, calculated as the number of rows multiplied by the size of each quantized row.
- **See also**: [`quantize_q5_0`](ggml-quants.c.driver.md#quantize_q5_0)  (Implementation)


---
### quantize\_q5\_1<!-- {{#callable_declaration:quantize_q5_1}} -->
Quantizes a source array into a destination buffer.
- **Description**: This function is used to quantize a floating-point source array into a specified format, storing the result in a destination buffer. It should be called when you need to convert floating-point data into a quantized representation, which is often required for efficient storage or processing in machine learning applications. The function expects the source and destination buffers to be properly allocated and sized according to the number of rows and elements per row. If the `quant_weights` parameter is null, a reference quantization method is used. Otherwise, a specific quantization method utilizing the provided weights is applied. It is important to ensure that the dimensions specified by `nrow` and `n_per_row` are positive integers.
- **Inputs**:
    - `src`: Pointer to the source array of floating-point values. Must not be null. The data should be organized in a contiguous block, with `nrow` rows and `n_per_row` elements per row.
    - `dst`: Pointer to the destination buffer where the quantized data will be stored. Must not be null. The buffer should be large enough to hold the quantized representation of the data.
    - `nrow`: The number of rows to process. Must be a positive integer.
    - `n_per_row`: The number of elements per row. Must be a positive integer.
    - `quant_weights`: Pointer to an array of quantization weights. Can be null. If not null, it should point to a valid array of weights corresponding to the quantization process.
- **Output**: Returns the total size of the quantized data in bytes, calculated as the number of rows multiplied by the size of each quantized row.
- **See also**: [`quantize_q5_1`](ggml-quants.c.driver.md#quantize_q5_1)  (Implementation)


---
### quantize\_q8\_0<!-- {{#callable_declaration:quantize_q8_0}} -->
Quantizes a source array of floats into a destination buffer.
- **Description**: This function is used to quantize a given source array of floating-point values into a specified destination buffer, which is typically used for efficient storage or processing. It should be called when you need to convert a floating-point representation into a quantized format, specifically for `Q8_0` quantization. The function expects that the source and destination buffers are properly allocated and that the dimensions specified by `nrow` and `n_per_row` are valid. It is important to ensure that the destination buffer is large enough to hold the quantized data, as insufficient space may lead to undefined behavior.
- **Inputs**:
    - `src`: A pointer to the source array of floats to be quantized. Must not be null and should point to a valid memory region containing at least `nrow * n_per_row` elements.
    - `dst`: A pointer to the destination buffer where the quantized data will be stored. Must not be null and should be allocated with sufficient size to hold the quantized output.
    - `nrow`: An integer representing the number of rows in the source data. Must be a positive value.
    - `n_per_row`: An integer representing the number of elements per row in the source data. Must be a positive value.
    - `quant_weights`: A pointer to an array of quantization weights. This parameter is not used in the current implementation and can be passed as null.
- **Output**: Returns the total size in bytes of the quantized data written to the destination buffer, calculated as `nrow * row_size`, where `row_size` is determined by the quantization format.
- **See also**: [`quantize_q8_0`](ggml-quants.c.driver.md#quantize_q8_0)  (Implementation)


---
### iq2xs\_init\_impl<!-- {{#callable_declaration:iq2xs_init_impl}} -->
Initializes the IQ2XS quantization grid.
- **Description**: This function must be called to initialize the quantization grid for the specified type before any quantization operations can be performed. It sets up the necessary data structures and memory allocations based on the provided `type`. If the grid for the specified type has already been initialized, the function will return early without making any changes. It is important to ensure that this function is called only once per type to avoid memory leaks or undefined behavior.
- **Inputs**:
    - `type`: Specifies the type of quantization to initialize. Valid values are defined in the `ggml_type` enumeration. The function will handle the initialization based on the provided type. Must not be null.
- **Output**: None
- **See also**: [`iq2xs_init_impl`](ggml-quants.c.driver.md#iq2xs_init_impl)  (Implementation)


---
### iq2xs\_free\_impl<!-- {{#callable_declaration:iq2xs_free_impl}} -->
Frees allocated resources for a specified quantization type.
- **Description**: This function should be called to release memory associated with a specific quantization type after it is no longer needed. It is essential to ensure that the type provided is valid, as it must match one of the predefined quantization types. If the function is called with an invalid type, it will assert and terminate the program. The function safely frees memory for the grid, map, and neighbours associated with the specified type, setting their pointers to NULL to prevent dangling references.
- **Inputs**:
    - `type`: Specifies the quantization type for which resources should be freed. Valid values are limited to `GGML_TYPE_IQ2_XXS`, `GGML_TYPE_IQ2_XS`, `GGML_TYPE_IQ1_S`, `GGML_TYPE_IQ1_M`, and `GGML_TYPE_IQ2_S`. The caller must ensure that the type is valid before calling this function; otherwise, the program will terminate due to an assertion failure.
- **Output**: None
- **See also**: [`iq2xs_free_impl`](ggml-quants.c.driver.md#iq2xs_free_impl)  (Implementation)


---
### iq3xs\_init\_impl<!-- {{#callable_declaration:iq3xs_init_impl}} -->
Initializes the IQ3XS data structure.
- **Description**: This function is used to initialize the IQ3XS data structure with a specified grid size. It must be called before any operations that depend on the IQ3XS structure. The grid size must be either 256 or 512; passing any other value may lead to undefined behavior. The function allocates memory for the grid and its associated mappings, and it will not reinitialize an already initialized grid for the same size. It is important to ensure that the corresponding free function is called to release the allocated resources when they are no longer needed.
- **Inputs**:
    - `grid_size`: Specifies the size of the grid to initialize. Valid values are 256 or 512. Must not be null. If an invalid value is provided, the behavior is undefined.
- **Output**: None
- **See also**: [`iq3xs_init_impl`](ggml-quants.c.driver.md#iq3xs_init_impl)  (Implementation)


---
### iq3xs\_free\_impl<!-- {{#callable_declaration:iq3xs_free_impl}} -->
Frees allocated resources for a specified grid size.
- **Description**: This function should be called to release memory associated with the specified grid size, which must be either 256 or 512. It is important to ensure that this function is called only after the corresponding resources have been allocated. If the grid size is invalid, the function will assert and terminate. After successful execution, all associated memory for the specified grid size will be freed, and the pointers will be set to NULL to prevent dangling references.
- **Inputs**:
    - `grid_size`: Specifies the size of the grid to free resources for. Valid values are 256 or 512. The function will assert if an invalid value is provided. The caller does not retain ownership of any resources associated with this grid size after this function is called.
- **Output**: None
- **See also**: [`iq3xs_free_impl`](ggml-quants.c.driver.md#iq3xs_free_impl)  (Implementation)


