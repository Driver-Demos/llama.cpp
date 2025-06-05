# Purpose
The provided C code is a comprehensive implementation of various quantization and dequantization functions, designed to efficiently compress and optimize the storage and processing of floating-point data within a larger system, likely a machine learning or data processing library. It includes multiple quantization schemes, such as 2-bit to 8-bit quantization, as well as specialized methods like ternary and "true" 2-bit quantization, each tailored to specific data types and precision requirements. The code is structured as a library module, featuring both public APIs for external integration and static helper functions for internal use, ensuring flexibility and adaptability to various use cases. It employs performance-focused techniques, such as SIMD operations, to handle large data sets efficiently, and includes validation functions to maintain data integrity. Overall, the code serves as a specialized library component, offering efficient quantization solutions for data-intensive applications, with configurable parameters and a modular approach to implementation.
# Imports and Dependencies

---
- `ggml-common.h`
- `ggml-quants.h`
- `ggml-impl.h`
- `ggml-cpu/ggml-cpu-impl.h`
- `ggml-cpu.h`
- `math.h`
- `string.h`
- `assert.h`
- `float.h`
- `stdlib.h`
- `stdio.h`


# Data Structures

---
### iq2\_entry\_t
- **Type**: `struct`
- **Members**:
    - `grid`: A pointer to a 64-bit unsigned integer array representing the grid data.
    - `map`: A pointer to an integer array used for mapping purposes.
    - `neighbours`: A pointer to a 16-bit unsigned integer array representing neighboring elements.
- **Description**: The `iq2_entry_t` structure is designed to manage and store data related to a grid system, including its mapping and neighboring elements. It uses pointers to dynamically allocated arrays for flexibility in handling varying sizes of grid data, mapping information, and neighbor relationships. This structure is likely used in applications where spatial relationships and grid-based data management are crucial, such as in simulations or game development.


---
### iq3\_entry\_t
- **Type**: `struct`
- **Members**:
    - `grid`: A pointer to a 32-bit unsigned integer array representing the grid.
    - `map`: A pointer to an integer array representing the map.
    - `neighbours`: A pointer to a 16-bit unsigned integer array representing the neighbours.
- **Description**: The `iq3_entry_t` structure is designed to encapsulate pointers to arrays that represent a grid, a map, and neighbours, likely used in a context where these elements are dynamically allocated and manipulated. The use of pointers allows for flexible memory management and efficient data handling, which is essential in applications that require dynamic data structures or large datasets.


# Functions

---
### quantize\_row\_q4\_0\_ref<!-- {{#callable:quantize_row_q4_0_ref}} -->
The `quantize_row_q4_0_ref` function quantizes a row of floating-point values into a specific compressed format.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that are to be quantized.
    - `y`: A pointer to a `block_q4_0` structure where the quantized results will be stored.
    - `k`: An integer representing the total number of elements in the input array, which must be a multiple of `QK4_0`.
- **Control Flow**:
    - The function asserts that `k` is a multiple of `QK4_0` to ensure valid input size.
    - It calculates the number of blocks `nb` by dividing `k` by `QK4_0`.
    - For each block, it initializes `amax` and `max` to find the maximum absolute value in the block.
    - It computes a scaling factor `d` based on the maximum value and derives an inverse `id` for normalization.
    - The maximum value is converted to a half-precision floating-point and stored in the output structure.
    - The function then quantizes pairs of values from the input array into 4-bit representations and stores them in the output structure.
- **Output**: The function outputs a `block_q4_0` structure containing the quantized values, where each block consists of a scaled value and a compressed representation of the original values.


---
### quantize\_row\_q4\_1\_ref<!-- {{#callable:quantize_row_q4_1_ref}} -->
The `quantize_row_q4_1_ref` function quantizes a row of floating-point values into a compressed format using a specific quantization scheme.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that are to be quantized.
    - `y`: A pointer to a `block_q4_1` structure where the quantized results will be stored.
    - `k`: An integer representing the total number of elements in the input array, which must be a multiple of `QK4_1`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK4_1`.
    - It calculates the number of blocks `nb` by dividing `k` by `QK4_1`.
    - For each block, it initializes `min` and `max` values to find the minimum and maximum of the corresponding segment of the input array.
    - It computes the quantization step size `d` and its inverse `id` based on the found `min` and `max` values.
    - The minimum value and the quantization step are stored in the output structure as half-precision floats.
    - For each pair of values in the block, it normalizes them, quantizes them to 4 bits, and stores them in the output structure.
- **Output**: The function outputs a `block_q4_1` structure filled with quantized data, including the minimum value, quantization step, and the quantized values for each block.


---
### quantize\_row\_q5\_0\_ref<!-- {{#callable:quantize_row_q5_0_ref}} -->
The `quantize_row_q5_0_ref` function quantizes a row of floating-point values into a specific compressed format.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that are to be quantized.
    - `y`: A pointer to a `block_q5_0` structure where the quantized results will be stored.
    - `k`: An integer representing the total number of elements in `x`, which must be a multiple of `QK5_0`.
- **Control Flow**:
    - The function asserts that `k` is a multiple of `QK5_0` to ensure valid input.
    - It calculates the number of blocks `nb` by dividing `k` by `QK5_0`.
    - For each block, it initializes `amax` and `max` to find the maximum absolute value in the block.
    - It computes a scaling factor `d` based on the maximum value and calculates its inverse `id`.
    - The scaled value `d` is converted to a half-precision float and stored in the output structure.
    - The function then quantizes pairs of values from the input array into a compressed format, storing them in the output structure.
    - Finally, it combines the 5th bits of the quantized values into a single variable `qh` and stores it in the output structure.
- **Output**: The function does not return a value but populates the `y` structure with quantized data, including the compressed values and the scaling factor.


---
### quantize\_row\_q5\_1\_ref<!-- {{#callable:quantize_row_q5_1_ref}} -->
The `quantize_row_q5_1_ref` function quantizes a row of floating-point values into a compressed format suitable for efficient storage.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that are to be quantized.
    - `y`: A pointer to a `block_q5_1` structure where the quantized results will be stored.
    - `k`: An integer representing the total number of elements in the input array `x`, which must be a multiple of `QK5_1`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK5_1`.
    - It calculates the number of blocks `nb` by dividing `k` by `QK5_1`.
    - For each block, it initializes `min` and `max` values to find the range of the block's values.
    - It iterates through the elements of the block to determine the minimum and maximum values.
    - The difference `d` between `max` and `min` is calculated, and an inverse `id` is computed for normalization.
    - The normalized values are quantized into 8-bit format and stored in the `qs` array of the `y` structure.
    - The 5th bits of the quantized values are extracted and stored in the `qh` variable.
    - Finally, the `qh` value is copied into the corresponding field of the `y` structure.
- **Output**: The function does not return a value; instead, it populates the `y` structure with quantized data, including the minimum value, the difference, and the quantized representation of the input values.


---
### quantize\_row\_q8\_0\_ref<!-- {{#callable:quantize_row_q8_0_ref}} -->
The `quantize_row_q8_0_ref` function quantizes a row of floating-point values into a block of quantized integers.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that are to be quantized.
    - `y`: A pointer to a `block_q8_0` structure where the quantized results will be stored.
    - `k`: An integer representing the total number of elements in the input array, which must be a multiple of `QK8_0`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK8_0`.
    - It calculates the number of blocks `nb` by dividing `k` by `QK8_0`.
    - For each block, it initializes `amax` to zero and iterates through the elements to find the absolute maximum value.
    - It computes the scaling factor `d` based on the maximum value and calculates its inverse `id`.
    - The maximum value `d` is converted to a half-precision floating-point and stored in the output structure.
    - Finally, it quantizes each element in the block by scaling and rounding, storing the results in the output structure.
- **Output**: The function does not return a value; instead, it populates the `y` structure with the quantized data, where each block contains the maximum value and the quantized integers.


---
### quantize\_row\_q8\_1\_ref<!-- {{#callable:quantize_row_q8_1_ref}} -->
Quantizes a row of floating-point values into a block of quantized 8-bit integers and computes a scale factor.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values to be quantized.
    - `y`: A pointer to a `block_q8_1` structure where the quantized values and scale will be stored.
    - `k`: An integer representing the total number of floating-point values, which must be a multiple of `QK8_1`.
- **Control Flow**:
    - The function begins by asserting that `QK8_1` is equal to 32 and that `k` is a multiple of `QK8_1`.
    - It calculates the number of blocks `nb` by dividing `k` by `QK8_1`.
    - For each block, it initializes `amax` to find the absolute maximum value in the block.
    - It iterates through the elements of the block to compute `amax` and then calculates the scale factor `d`.
    - If `d` is non-zero, it computes the inverse `id` for normalization.
    - The scale factor `d` is stored in the `y` structure as a half-precision float.
    - It then quantizes the values in pairs, rounding them and storing them in the `qs` array of the `y` structure.
    - The sum of the quantized values is computed and stored in the `s` field of the `y` structure as a half-precision float.
- **Output**: The function does not return a value; instead, it modifies the `y` structure to contain the quantized values and the computed scale.


---
### dequantize\_row\_q4\_0<!-- {{#callable:dequantize_row_q4_0}} -->
The `dequantize_row_q4_0` function converts quantized data from a `block_q4_0` structure into floating-point values.
- **Inputs**:
    - `x`: A pointer to an array of `block_q4_0` structures containing quantized data.
    - `y`: A pointer to an array of floats where the dequantized output will be stored.
    - `k`: An integer representing the total number of quantized elements, which must be a multiple of `QK4_0`.
- **Control Flow**:
    - The function asserts that `k` is a multiple of `QK4_0` to ensure valid input.
    - It calculates the number of blocks `nb` by dividing `k` by `QK4_0`.
    - A loop iterates over each block, extracting the quantized values and converting them to floating-point.
    - Within the block loop, another loop processes half of the quantized values, extracting two values per iteration, adjusting them, and storing the results in the output array.
- **Output**: The function outputs an array of floats `y`, which contains the dequantized values corresponding to the input quantized data.


---
### dequantize\_row\_q4\_1<!-- {{#callable:dequantize_row_q4_1}} -->
The `dequantize_row_q4_1` function converts quantized data from a `block_q4_1` structure into floating-point values.
- **Inputs**:
    - `x`: A pointer to an array of `block_q4_1` structures containing quantized data.
    - `y`: A pointer to an array of floats where the dequantized output will be stored.
    - `k`: An integer representing the total number of quantized elements, which must be a multiple of `QK4_1`.
- **Control Flow**:
    - The function asserts that `k` is a multiple of `QK4_1` to ensure valid input.
    - It calculates the number of blocks `nb` by dividing `k` by `QK4_1`.
    - A loop iterates over each block, extracting the dequantization parameters `d` and `m` from the `block_q4_1` structure.
    - Within the block loop, another loop processes half of the quantized values, extracting two quantized values `x0` and `x1` from each entry.
    - The dequantized values are computed using the formula `y[i*qk + j + 0] = x0*d + m` and `y[i*qk + j + qk/2] = x1*d + m`, storing them in the output array.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the dequantized floating-point values derived from the input quantized data.


---
### dequantize\_row\_q5\_0<!-- {{#callable:dequantize_row_q5_0}} -->
The `dequantize_row_q5_0` function converts quantized data from a `block_q5_0` structure into floating-point values.
- **Inputs**:
    - `x`: A pointer to a constant `block_q5_0` structure containing quantized data.
    - `y`: A pointer to a float array where the dequantized output will be stored.
    - `k`: An integer representing the total number of quantized elements, which must be a multiple of `QK5_0`.
- **Control Flow**:
    - The function asserts that `k` is a multiple of `QK5_0` to ensure valid input.
    - It calculates the number of blocks `nb` by dividing `k` by `QK5_0`.
    - A loop iterates over each block, extracting the floating-point scale factor `d` from the `d` field of the `block_q5_0` structure.
    - The quantized header `qh` is copied from the `qh` field of the current block.
    - A nested loop processes half of the quantized values, extracting and transforming them into floating-point values using bit manipulation and scaling with `d`.
    - The transformed values are stored in the output array `y` at the appropriate indices.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the dequantized floating-point values derived from the input `x`.


---
### dequantize\_row\_q5\_1<!-- {{#callable:dequantize_row_q5_1}} -->
The `dequantize_row_q5_1` function converts quantized data from a `block_q5_1` structure into floating-point values.
- **Inputs**:
    - `x`: A pointer to an array of `block_q5_1` structures containing quantized data.
    - `y`: A pointer to a float array where the dequantized output will be stored.
    - `k`: An integer representing the total number of quantized elements, which must be a multiple of `QK5_1`.
- **Control Flow**:
    - The function asserts that `k` is a multiple of `QK5_1` to ensure valid input.
    - It calculates the number of blocks `nb` by dividing `k` by `QK5_1`.
    - A loop iterates over each block, extracting the dequantization parameters `d` and `m` from the `block_q5_1` structure.
    - The quantized values are extracted from the `qh` field using bit manipulation to reconstruct the original values.
    - Another loop processes half of the quantized values, applying the dequantization formula to populate the output array `y`.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the dequantized floating-point values derived from the input quantized data.


---
### dequantize\_row\_q8\_0<!-- {{#callable:dequantize_row_q8_0}} -->
The `dequantize_row_q8_0` function converts quantized data from a `block_q8_0` structure into floating-point values.
- **Inputs**:
    - `x`: A pointer to an array of `block_q8_0` structures containing quantized data.
    - `y`: A pointer to an array of floats where the dequantized output will be stored.
    - `k`: An integer representing the total number of quantized elements, which must be a multiple of `QK8_0`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK8_0` to ensure valid input.
    - It calculates `nb`, the number of blocks, by dividing `k` by `QK8_0`.
    - A loop iterates over each block, converting the quantized data to floating-point values.
    - Within the block loop, it retrieves the floating-point representation of the quantized data using `GGML_FP16_TO_FP32`.
    - Another nested loop iterates over the quantized values in the block, scaling them by the floating-point value and storing the result in the output array.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the dequantized floating-point values.


---
### nearest\_int<!-- {{#callable:nearest_int}} -->
Converts a floating-point number to the nearest integer representation using bit manipulation.
- **Inputs**:
    - `fval`: A floating-point number that is to be converted to the nearest integer.
- **Control Flow**:
    - Assert that the absolute value of `fval` is less than or equal to 4194303.
    - Add 12582912 to `fval` and store the result in `val`.
    - Copy the bit representation of `val` into an integer variable `i` using `memcpy`.
    - Return the result of the bitwise operation on `i` to obtain the nearest integer.
- **Output**: An integer that represents the nearest integer value of the input floating-point number.


---
### make\_q3\_quants<!-- {{#callable:make_q3_quants}} -->
The `make_q3_quants` function quantizes an array of floating-point values and optionally adjusts the quantization based on root mean square error.
- **Inputs**:
    - `n`: The number of elements in the input array `x`.
    - `nmax`: The maximum allowable quantization level.
    - `x`: A pointer to an array of floating-point values to be quantized.
    - `L`: A pointer to an array where the quantized levels will be stored.
    - `do_rmse`: A boolean flag indicating whether to perform root mean square error adjustment.
- **Control Flow**:
    - Initialize `max` and `amax` to zero.
    - Iterate through the input array `x` to find the maximum absolute value and its corresponding original value.
    - If the maximum absolute value is less than a defined threshold (`GROUP_MAX_EPS`), set all elements in `L` to zero and return 0.
    - Calculate the inverse scale factor `iscale` based on the maximum value and `nmax`.
    - If `do_rmse` is true, perform additional calculations to adjust the quantization levels based on the weighted sum of the original values and their quantized levels.
    - Iterate up to 5 times to refine the quantization levels if changes are made during the adjustment phase.
    - After adjustments, add `nmax` to each quantized level in `L` and return the weighted average of the original values.
    - If `do_rmse` is false, directly quantize the values and store them in `L`, adding `nmax` to each level before returning the inverse scale factor.
- **Output**: The function returns a float value which is either the weighted average of the original values adjusted for quantization or the inverse scale factor used for quantization.
- **Functions called**:
    - [`nearest_int`](#nearest_int)


---
### make\_qkx1\_quants<!-- {{#callable:make_qkx1_quants}} -->
The `make_qkx1_quants` function quantizes an array of floating-point values based on specified parameters and returns a scaling factor.
- **Inputs**:
    - `n`: The number of elements in the input array `x`.
    - `nmax`: The maximum value for quantization.
    - `x`: A pointer to an array of floating-point values to be quantized.
    - `L`: A pointer to an array where the quantized values will be stored.
    - `the_min`: A pointer to a float where the minimum value will be stored.
    - `ntry`: The number of iterations to attempt for refining the quantization.
    - `alpha`: A smoothing factor used to update the minimum value.
- **Control Flow**:
    - Initialize `min` and `max` with the first element of `x` and iterate through the array to find the overall minimum and maximum values.
    - If `max` equals `min`, set all elements in `L` to 0, assign 0 to `the_min`, and return 0.
    - Adjust `min` to 0 if it is greater than 0, then calculate the initial `iscale` and `scale` based on `nmax` and the range of values.
    - Enter a loop that runs for `ntry` iterations to refine the quantization, where for each element in `x`, the nearest integer quantization level is calculated and stored in `L`.
    - Calculate `sumlx` and `suml2` to derive a new `scale` based on the quantized values.
    - Update `min` using a weighted average of the previous `min` and the average of the differences between `x` and the scaled quantized values.
    - If no changes were made to `L`, exit the loop early.
    - Store the negative of the final `min` in `the_min` and return the final `scale`.
- **Output**: Returns a float representing the scaling factor used for quantization.
- **Functions called**:
    - [`nearest_int`](#nearest_int)


---
### get\_scale\_min\_k4<!-- {{#callable:get_scale_min_k4}} -->
The `get_scale_min_k4` function computes two values based on the input index and a source array, storing the results in provided output pointers.
- **Inputs**:
    - `j`: An integer index used to determine which elements of the input array `q` to access.
    - `q`: A pointer to a constant array of `uint8_t` values from which data is read.
    - `d`: A pointer to a `uint8_t` variable where the first computed value will be stored.
    - `m`: A pointer to a `uint8_t` variable where the second computed value will be stored.
- **Control Flow**:
    - The function checks if the index `j` is less than 4.
    - If `j` is less than 4, it computes `*d` and `*m` using the first four elements of `q`.
    - If `j` is 4 or greater, it computes `*d` and `*m` using elements from `q` with bitwise operations to combine values.
- **Output**: The function does not return a value; instead, it modifies the values pointed to by `d` and `m` based on the calculations performed.


---
### quantize\_row\_q2\_K\_ref<!-- {{#callable:quantize_row_q2_K_ref}} -->
The `quantize_row_q2_K_ref` function quantizes a row of floating-point values into a compressed format suitable for efficient storage and processing.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that represent the input data to be quantized.
    - `y`: A pointer to a `block_q2_K` structure where the quantized output will be stored.
    - `k`: An integer representing the number of elements in the input array, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K` and calculates the number of blocks `nb` based on `k`.
    - It initializes several arrays for storing intermediate values such as scales, minimums, and quantized values.
    - For each block, it computes the scales and minimum values for the input data, updating the maximum scale and minimum found.
    - If a valid maximum scale is found, it calculates the inverse scale and updates the `scales` in the output structure `y`.
    - If a valid maximum minimum is found, it similarly updates the `dmin` field in `y`.
    - The function then quantizes the input data based on the computed scales and minimums, storing the results in the `L` array.
    - Finally, it packs the quantized values into the output structure `y` in a compressed format.
- **Output**: The function outputs a quantized representation of the input data in the `y` structure, which includes scales, minimum values, and the quantized data itself.
- **Functions called**:
    - [`nearest_int`](#nearest_int)


---
### dequantize\_row\_q2\_K<!-- {{#callable:dequantize_row_q2_K}} -->
The `dequantize_row_q2_K` function converts quantized data from a `block_q2_K` structure into floating-point values.
- **Inputs**:
    - `x`: A pointer to a `block_q2_K` structure containing quantized data and scaling factors.
    - `y`: A pointer to a float array where the dequantized output will be stored.
    - `k`: An integer representing the total number of quantized elements, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K`.
    - It calculates the number of blocks `nb` by dividing `k` by `QK_K`.
    - A loop iterates over each block, extracting the dequantization parameters `d` and `min` from the `block_q2_K` structure.
    - Within the block loop, another loop processes the quantized data in chunks of 128 elements, using nested loops to handle scaling and dequantization.
    - For each scale factor, it computes the dequantized values and stores them in the output array `y`.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the dequantized floating-point values derived from the input quantized data.


---
### make\_qp\_quants<!-- {{#callable:make_qp_quants}} -->
The `make_qp_quants` function quantizes an array of floating-point values based on specified weights and a maximum quantization level.
- **Inputs**:
    - `n`: The number of elements in the input array `x`.
    - `nmax`: The maximum allowable quantized value.
    - `x`: A pointer to an array of floating-point values to be quantized.
    - `L`: A pointer to an array where the quantized values will be stored.
    - `quant_weights`: A pointer to an array of weights corresponding to each element in `x`.
- **Control Flow**:
    - Initialize `max` to zero and find the maximum value in the input array `x`.
    - If `max` is zero, set all elements in `L` to zero and return 0.
    - Calculate the initial scale factor `iscale` based on `max` and `nmax`.
    - Quantize the input values in `x` and store them in `L` using the initial scale.
    - Calculate the mean squared error (MSE) for the initial quantization.
    - Iterate over a range of adjustments to `iscale` to minimize the MSE.
    - Recalculate the quantized values in `L` and update them based on the best MSE found.
    - Perform up to 5 iterations to refine the quantized values in `L` based on the weights.
    - Return the final average of the quantized values weighted by `quant_weights`.
- **Output**: The function returns the weighted average of the quantized values, which is computed as the sum of the products of the original values and their quantized counterparts divided by the sum of the squares of the quantized values.
- **Functions called**:
    - [`nearest_int`](#nearest_int)


---
### quantize\_row\_q2\_K\_impl<!-- {{#callable:quantize_row_q2_K_impl}} -->
The `quantize_row_q2_K_impl` function quantizes a row of floating-point values into a compressed format using specified quantization weights.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values to be quantized.
    - `y`: A pointer to a `block_q2_K` structure where the quantized output will be stored.
    - `k`: An integer representing the number of elements in the input array, which must be a multiple of `QK_K`.
    - `quant_weights`: A pointer to an array of quantization weights used in the quantization process.
- **Control Flow**:
    - The function begins by asserting that `quant_weights` is not null and that `k` is a multiple of `QK_K`.
    - It calculates the number of blocks `nb` by dividing `k` by `QK_K`.
    - For each block, it initializes necessary arrays and computes the sum of squares of the input values.
    - It calculates the variance `sigma2` and uses it to compute weights for quantization.
    - The function then calls `make_qkx3_quants` to generate quantization scales and minimum values for each segment of the input.
    - It converts the computed values to half-precision and stores them in the output structure.
    - If `requantize` is true, it further refines the quantization based on the computed scales and minimums.
    - Finally, it packs the quantized values into the output structure in a specific format.
- **Output**: The function outputs a quantized representation of the input data in the `y` structure, including quantized values, scales, and minimums.
- **Functions called**:
    - [`make_qp_quants`](#make_qp_quants)
    - [`nearest_int`](#nearest_int)


---
### quantize\_q2\_K<!-- {{#callable:quantize_q2_K}} -->
The `quantize_q2_K` function quantizes a source array of floats into a destination buffer using specified quantization weights or a reference method.
- **Inputs**:
    - `src`: A pointer to the source array of floats that will be quantized.
    - `dst`: A pointer to the destination buffer where the quantized data will be stored.
    - `nrow`: The number of rows to process in the source array.
    - `n_per_row`: The number of elements per row in the source array.
    - `quant_weights`: A pointer to an array of quantization weights; if NULL, a reference quantization method is used.
- **Control Flow**:
    - The function calculates the size of each row in the quantized output using [`ggml_row_size`](ggml.c.driver.md#ggml_row_size).
    - If `quant_weights` is NULL, it calls [`quantize_row_q2_K_ref`](#quantize_row_q2_K_ref) to quantize the entire source array using a reference method.
    - If `quant_weights` is provided, it enters a loop to quantize each row individually using [`quantize_row_q2_K_impl`](#quantize_row_q2_K_impl), updating the source and destination pointers accordingly.
    - The loop iterates over the number of rows specified by `nrow`, processing `n_per_row` elements for each row.
- **Output**: The function returns the total size of the quantized data in bytes, calculated as the product of the number of rows and the size of each row.
- **Functions called**:
    - [`ggml_row_size`](ggml.c.driver.md#ggml_row_size)
    - [`quantize_row_q2_K_ref`](#quantize_row_q2_K_ref)
    - [`quantize_row_q2_K_impl`](#quantize_row_q2_K_impl)


---
### quantize\_row\_q3\_K\_ref<!-- {{#callable:quantize_row_q3_K_ref}} -->
The `quantize_row_q3_K_ref` function quantizes a row of floating-point values into a compressed format suitable for efficient storage and processing.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that are to be quantized.
    - `y`: A pointer to a `block_q3_K` structure where the quantized data will be stored.
    - `k`: An integer representing the number of elements in the input array `x`, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K` and calculates the number of blocks `nb` based on `k`.
    - It initializes arrays for storing quantization levels and scales.
    - For each block, it computes the quantization scales and finds the maximum scale value.
    - If a maximum scale is found, it calculates the inverse scale and quantizes the input values into the `scales` array.
    - The function then constructs a high mask to indicate which quantized values exceed a certain threshold.
    - Finally, it packs the quantized values into the `qs` array of the `block_q3_K` structure.
- **Output**: The function does not return a value; instead, it modifies the `block_q3_K` structure pointed to by `y` to store the quantized representation of the input data.
- **Functions called**:
    - [`make_q3_quants`](#make_q3_quants)
    - [`nearest_int`](#nearest_int)


---
### dequantize\_row\_q3\_K<!-- {{#callable:dequantize_row_q3_K}} -->
The `dequantize_row_q3_K` function performs dequantization of a quantized row of data, converting it into floating-point values based on provided scales and masks.
- **Inputs**:
    - `x`: A pointer to an array of `block_q3_K` structures containing quantized data, scales, and masks.
    - `y`: A pointer to a float array where the dequantized output will be stored.
    - `k`: An integer representing the total number of quantized elements, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K` and calculates `nb`, the number of blocks.
    - It initializes masks for processing the quantized data and prepares an auxiliary array for scale values.
    - A loop iterates over each block of quantized data, extracting the floating-point representation and the quantized values.
    - Within a nested loop, it processes the quantized values in chunks, applying the scales and masks to generate the dequantized output.
    - The inner loop handles the bit manipulation of quantized values and applies the scaling factor to produce the final floating-point values.
- **Output**: The function outputs a series of dequantized floating-point values stored in the array pointed to by `y`.


---
### quantize\_row\_q3\_K\_impl<!-- {{#callable:quantize_row_q3_K_impl}} -->
The `quantize_row_q3_K_impl` function quantizes a row of floating-point values into a compressed format using specified quantization weights.
- **Inputs**:
    - `x`: A pointer to an array of `float` values representing the input data to be quantized.
    - `y`: A pointer to a `block_q3_K` structure where the quantized output will be stored.
    - `n_per_row`: An integer representing the number of elements in each row, which must be a multiple of `QK_K`.
    - `quant_weights`: A pointer to an array of `float` values used as quantization weights; can be `NULL`.
- **Control Flow**:
    - The function begins by asserting that `n_per_row` is a multiple of `QK_K`.
    - It calculates the number of blocks `nb` by dividing `n_per_row` by `QK_K`.
    - For each block, it computes the sum of squares of the input values to derive `sigma2`.
    - It then iterates over segments of the input data, calculating weights based on the quantization weights or the input values themselves.
    - The function computes scales and quantization levels for each segment and stores them in the output structure.
    - It constructs a high mask to indicate which quantized values exceed a certain threshold.
    - Finally, it packs the quantized values into the output structure and moves to the next block.
- **Output**: The function does not return a value but populates the `y` structure with quantized data, including scales, quantization levels, and a high mask.
- **Functions called**:
    - [`nearest_int`](#nearest_int)


---
### quantize\_q3\_K<!-- {{#callable:quantize_q3_K}} -->
The `quantize_q3_K` function quantizes a source array of floats into a destination buffer using either a reference or an implementation-specific quantization method based on the presence of quantization weights.
- **Inputs**:
    - `src`: A pointer to the source array of floats that will be quantized.
    - `dst`: A pointer to the destination buffer where the quantized data will be stored.
    - `nrow`: The number of rows in the source data to be quantized.
    - `n_per_row`: The number of elements per row in the source data.
    - `quant_weights`: A pointer to an array of quantization weights; if NULL, a reference quantization method is used.
- **Control Flow**:
    - The function calculates the size of each row in the quantized output using [`ggml_row_size`](ggml.c.driver.md#ggml_row_size).
    - If `quant_weights` is NULL, it calls [`quantize_row_q3_K_ref`](#quantize_row_q3_K_ref) to quantize the entire source array using a reference method.
    - If `quant_weights` is not NULL, it enters a loop that iterates over each row, calling [`quantize_row_q3_K_impl`](#quantize_row_q3_K_impl) to quantize each row with the provided weights.
    - The source pointer is incremented by `n_per_row` after processing each row, and the destination pointer is incremented by the calculated `row_size`.
- **Output**: The function returns the total size of the quantized data in bytes, calculated as the product of the number of rows and the size of each row.
- **Functions called**:
    - [`ggml_row_size`](ggml.c.driver.md#ggml_row_size)
    - [`quantize_row_q3_K_ref`](#quantize_row_q3_K_ref)
    - [`quantize_row_q3_K_impl`](#quantize_row_q3_K_impl)


---
### quantize\_row\_q4\_K\_ref<!-- {{#callable:quantize_row_q4_K_ref}} -->
Quantizes a row of floating-point values into a compressed format using specified scaling and minimum values.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values to be quantized.
    - `y`: A pointer to a `block_q4_K` structure where the quantized output will be stored.
    - `k`: An integer representing the number of elements in `x`, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K`.
    - It calculates the number of blocks `nb` based on `k`.
    - For each block, it initializes variables for maximum scale and minimum values.
    - It iterates over segments of the input array `x`, calculating scales and minimums using the `make_qkx2_quants` function.
    - It determines the inverse scale and minimum values for quantization.
    - The quantized scales and minimums are stored in the `y` structure.
    - For each segment, it computes the quantized values based on the calculated scales and minimums.
    - Finally, it packs the quantized values into the output array `y`.
- **Output**: The function outputs a quantized representation of the input array `x` stored in the `y` structure, which includes quantized scales, minimums, and the quantized values.
- **Functions called**:
    - [`nearest_int`](#nearest_int)
    - [`get_scale_min_k4`](#get_scale_min_k4)


---
### dequantize\_row\_q4\_K<!-- {{#callable:dequantize_row_q4_K}} -->
The `dequantize_row_q4_K` function converts quantized data from a `block_q4_K` structure into floating-point values and stores them in an output array.
- **Inputs**:
    - `x`: A pointer to an array of `block_q4_K` structures containing quantized data and scaling factors.
    - `y`: A pointer to a float array where the dequantized output values will be stored.
    - `k`: An integer representing the total number of quantized elements, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function asserts that `k` is a multiple of `QK_K` to ensure valid input.
    - It calculates the number of blocks `nb` by dividing `k` by `QK_K`.
    - For each block, it retrieves the quantized data and scaling factors.
    - It converts the fixed-point values `d` and `min` from half-precision to single-precision floats.
    - It iterates over the quantized data in chunks of 64, retrieving scale and minimum values for each chunk.
    - For each chunk, it computes the dequantized values using the scaling factors and stores them in the output array.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the dequantized floating-point values derived from the input quantized data.
- **Functions called**:
    - [`get_scale_min_k4`](#get_scale_min_k4)


---
### quantize\_row\_q4\_K\_impl<!-- {{#callable:quantize_row_q4_K_impl}} -->
The `quantize_row_q4_K_impl` function quantizes a row of floating-point values into a compressed format using specified quantization weights.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values to be quantized.
    - `y`: A pointer to a `block_q4_K` structure where the quantized output will be stored.
    - `n_per_row`: An integer representing the number of elements in the row, which must be a multiple of `QK_K`.
    - `quant_weights`: A pointer to an array of quantization weights, or NULL if default weights should be used.
- **Control Flow**:
    - The function begins by asserting that `n_per_row` is a multiple of `QK_K`.
    - It calculates the number of blocks `nb` based on `n_per_row`.
    - For each block, it computes the sum of squares of the input values to derive `sigma2` and `av_x`.
    - It then calculates weights for quantization based on the provided `quant_weights` or defaults to using `av_x`.
    - The function computes scales and minimum values for quantization using helper functions `make_qkx3_quants` and [`make_qp_quants`](#make_qp_quants).
    - It populates the `y` structure with quantized scales and values.
    - Finally, it quantizes the input values into a compressed format and stores them in the `qs` field of the `y` structure.
- **Output**: The function does not return a value but populates the `y` structure with quantized data, including scales, minimum values, and the quantized representation of the input data.
- **Functions called**:
    - [`make_qp_quants`](#make_qp_quants)
    - [`get_scale_min_k4`](#get_scale_min_k4)
    - [`nearest_int`](#nearest_int)


---
### quantize\_q4\_K<!-- {{#callable:quantize_q4_K}} -->
The `quantize_q4_K` function quantizes a source array of floats into a destination buffer using specified quantization weights or a reference method.
- **Inputs**:
    - `src`: A pointer to the source array of floats that will be quantized.
    - `dst`: A pointer to the destination buffer where the quantized data will be stored.
    - `nrow`: The number of rows to process in the source array.
    - `n_per_row`: The number of elements per row in the source array.
    - `quant_weights`: A pointer to an array of quantization weights; if NULL, a reference quantization method is used.
- **Control Flow**:
    - The function calculates the size of each row in the quantized format using [`ggml_row_size`](ggml.c.driver.md#ggml_row_size).
    - If `quant_weights` is NULL, it calls [`quantize_row_q4_K_ref`](#quantize_row_q4_K_ref) to quantize the entire source array using a reference method.
    - If `quant_weights` is provided, it enters a loop to quantize each row individually using [`quantize_row_q4_K_impl`](#quantize_row_q4_K_impl), updating the source and destination pointers accordingly.
- **Output**: The function returns the total size of the quantized data in bytes, calculated as the number of rows multiplied by the size of each row.
- **Functions called**:
    - [`ggml_row_size`](ggml.c.driver.md#ggml_row_size)
    - [`quantize_row_q4_K_ref`](#quantize_row_q4_K_ref)
    - [`quantize_row_q4_K_impl`](#quantize_row_q4_K_impl)


---
### quantize\_row\_q5\_K\_ref<!-- {{#callable:quantize_row_q5_K_ref}} -->
Quantizes a row of floating-point values into a compressed format using a specific quantization scheme.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that are to be quantized.
    - `y`: A pointer to a `block_q5_K` structure where the quantized output will be stored.
    - `k`: An integer representing the number of elements in `x`, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K` and calculates the number of blocks `nb`.
    - It initializes arrays for storing quantization parameters such as scales and minimum values.
    - For each block, it computes the maximum scale and minimum value from the input data.
    - It then calculates the inverse scale and minimum, and quantizes the scales and minimums into the output structure `y`.
    - The function further processes the quantized values to generate the final quantized representation in `L`.
    - Finally, it packs the quantized values into the output arrays `qh` and `ql`.
- **Output**: The function does not return a value; instead, it populates the `y` structure with quantized scales and values derived from the input array `x`.
- **Functions called**:
    - [`nearest_int`](#nearest_int)
    - [`get_scale_min_k4`](#get_scale_min_k4)


---
### dequantize\_row\_q5\_K<!-- {{#callable:dequantize_row_q5_K}} -->
The `dequantize_row_q5_K` function converts quantized data from a `block_q5_K` structure into floating-point values.
- **Inputs**:
    - `x`: A pointer to an array of `block_q5_K` structures containing quantized data.
    - `y`: A pointer to a float array where the dequantized output will be stored.
    - `k`: An integer representing the total number of quantized elements, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K`.
    - It calculates the number of blocks `nb` by dividing `k` by `QK_K`.
    - A loop iterates over each block in the input array `x`.
    - Within the block loop, it retrieves the quantized low (`ql`) and high (`qh`) values and the scaling factors (`d` and `min`).
    - Another loop processes the quantized data in chunks of 64, retrieving scale and minimum values for each chunk.
    - For each chunk, it computes the dequantized values using the quantized data and scaling factors, storing the results in the output array `y`.
- **Output**: The function outputs a series of floating-point values in the array pointed to by `y`, which are the dequantized representations of the input quantized data.
- **Functions called**:
    - [`get_scale_min_k4`](#get_scale_min_k4)


---
### quantize\_row\_q5\_K\_impl<!-- {{#callable:quantize_row_q5_K_impl}} -->
The `quantize_row_q5_K_impl` function quantizes a row of floating-point values into a compressed format using specified quantization weights.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values to be quantized.
    - `y`: A pointer to a `block_q5_K` structure where the quantized output will be stored.
    - `n_per_row`: An integer representing the number of elements in the row, which must be a multiple of `QK_K`.
    - `quant_weights`: A pointer to an array of quantization weights, which can be NULL.
- **Control Flow**:
    - The function begins by asserting that `n_per_row` is a multiple of `QK_K`.
    - It calculates the number of blocks `nb` by dividing `n_per_row` by `QK_K`.
    - For each block, it computes the sum of squares of the input values to derive `sigma2` and `av_x`.
    - It then calculates weights based on the provided `quant_weights` or uses an alternative method if they are NULL.
    - The function computes scales and minimum values for quantization using helper functions `make_qkx3_quants` and [`make_qp_quants`](#make_qp_quants).
    - It updates the `y` structure with quantized scales and derived values.
    - The quantized values are computed and stored in the `qh` and `qs` arrays of the `y` structure.
    - Finally, it increments the input pointer `x` to process the next block.
- **Output**: The function does not return a value but populates the `y` structure with quantized data, including scales, minimum values, and quantized representations of the input data.
- **Functions called**:
    - [`make_qp_quants`](#make_qp_quants)
    - [`get_scale_min_k4`](#get_scale_min_k4)
    - [`nearest_int`](#nearest_int)


---
### quantize\_q5\_K<!-- {{#callable:quantize_q5_K}} -->
The `quantize_q5_K` function quantizes a source array of floats into a destination buffer using specified quantization weights or a reference method.
- **Inputs**:
    - `src`: A pointer to the source array of floats that will be quantized.
    - `dst`: A pointer to the destination buffer where the quantized data will be stored.
    - `nrow`: The number of rows to process in the source array.
    - `n_per_row`: The number of elements per row in the source array.
    - `quant_weights`: A pointer to an array of quantization weights; if NULL, a reference quantization method is used.
- **Control Flow**:
    - The function calculates the size of each row in the quantized output using [`ggml_row_size`](ggml.c.driver.md#ggml_row_size).
    - If `quant_weights` is NULL, it calls [`quantize_row_q5_K_ref`](#quantize_row_q5_K_ref) to quantize the entire source array using a reference method.
    - If `quant_weights` is provided, it enters a loop to quantize each row individually using [`quantize_row_q5_K_impl`](#quantize_row_q5_K_impl), updating the source and destination pointers accordingly.
- **Output**: The function returns the total size of the quantized data in bytes, calculated as the number of rows multiplied by the size of each row.
- **Functions called**:
    - [`ggml_row_size`](ggml.c.driver.md#ggml_row_size)
    - [`quantize_row_q5_K_ref`](#quantize_row_q5_K_ref)
    - [`quantize_row_q5_K_impl`](#quantize_row_q5_K_impl)


---
### quantize\_row\_q6\_K\_ref<!-- {{#callable:quantize_row_q6_K_ref}} -->
The `quantize_row_q6_K_ref` function quantizes a row of floating-point values into a compressed format suitable for efficient storage.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that are to be quantized.
    - `y`: A pointer to a `block_q6_K` structure where the quantized output will be stored.
    - `k`: An integer representing the number of elements in `x`, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K` and calculates the number of blocks `nb` to process.
    - It initializes arrays for storing quantization levels and scales.
    - For each block, it computes the quantization scale and identifies the maximum absolute scale.
    - If the maximum absolute scale is below a threshold, it sets the corresponding output block to zero.
    - Otherwise, it calculates the inverse scale and populates the output structure with quantized values.
    - The quantized values are derived from the input values adjusted by the computed scale.
    - Finally, the function packs the quantized values into the output structure and moves to the next block.
- **Output**: The function outputs a quantized representation of the input data in the `y` structure, which includes quantized levels and scales.
- **Functions called**:
    - [`nearest_int`](#nearest_int)


---
### dequantize\_row\_q6\_K<!-- {{#callable:dequantize_row_q6_K}} -->
The `dequantize_row_q6_K` function performs dequantization of a row of quantized data into floating-point values.
- **Inputs**:
    - `x`: A pointer to a `block_q6_K` structure containing quantized data and associated scales.
    - `y`: A pointer to a float array where the dequantized output will be stored.
    - `k`: An integer representing the total number of quantized elements, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function asserts that `k` is a multiple of `QK_K` to ensure valid input.
    - It calculates the number of blocks `nb` by dividing `k` by `QK_K`.
    - A loop iterates over each block, extracting the floating-point representation of the data from `x[i].d`.
    - Within each block, another loop processes the quantized values in chunks of 128, iterating over 32 elements at a time.
    - For each element, it retrieves quantized values from `ql` and `qh`, applies bit manipulation to decode them, and scales them using the corresponding values from `sc`.
    - The results are stored in the output array `y`, which is updated after processing each chunk.
- **Output**: The function outputs a dequantized array of floating-point values in the `y` array, corresponding to the quantized input data.


---
### quantize\_row\_q6\_K\_impl<!-- {{#callable:quantize_row_q6_K_impl}} -->
The `quantize_row_q6_K_impl` function quantizes a row of floating-point values into a specific block format using quantization weights.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values to be quantized.
    - `y`: A pointer to a `block_q6_K` structure where the quantized output will be stored.
    - `n_per_row`: An integer representing the number of elements in the row, which must be a multiple of `QK_K`.
    - `quant_weights`: A pointer to an array of quantization weights, which can be NULL.
- **Control Flow**:
    - The function begins by asserting that `n_per_row` is a multiple of `QK_K`.
    - It calculates the number of blocks `nb` by dividing `n_per_row` by `QK_K`.
    - For each block, it initializes variables to track the maximum scale and absolute scale.
    - It iterates over `QK_K/16` to compute the scale for each segment of the input array, using quantization weights if provided.
    - If the maximum absolute scale is below a threshold, it sets the corresponding output block to zero and continues to the next block.
    - Otherwise, it calculates the inverse scale and populates the output block with quantized values based on the input data.
    - Finally, it packs the quantized values into the output structure `y`.
- **Output**: The function does not return a value but modifies the `y` structure in place to contain the quantized representation of the input data.
- **Functions called**:
    - [`nearest_int`](#nearest_int)


---
### quantize\_q6\_K<!-- {{#callable:quantize_q6_K}} -->
The `quantize_q6_K` function quantizes a source array of floats into a destination buffer using specified quantization weights or a reference method.
- **Inputs**:
    - `src`: A pointer to the source array of floats that will be quantized.
    - `dst`: A pointer to the destination buffer where the quantized data will be stored.
    - `nrow`: The number of rows to process in the source array.
    - `n_per_row`: The number of elements per row in the source array.
    - `quant_weights`: A pointer to an array of quantization weights; if NULL, a reference quantization method is used.
- **Control Flow**:
    - The function calculates the size of each row in the quantized format using [`ggml_row_size`](ggml.c.driver.md#ggml_row_size).
    - If `quant_weights` is NULL, it calls [`quantize_row_q6_K_ref`](#quantize_row_q6_K_ref) to quantize the entire source array using a reference method.
    - If `quant_weights` is provided, it enters a loop to quantize each row individually using [`quantize_row_q6_K_impl`](#quantize_row_q6_K_impl), updating the source and destination pointers accordingly.
- **Output**: The function returns the total size of the quantized data in bytes, calculated as the number of rows multiplied by the size of each row.
- **Functions called**:
    - [`ggml_row_size`](ggml.c.driver.md#ggml_row_size)
    - [`quantize_row_q6_K_ref`](#quantize_row_q6_K_ref)
    - [`quantize_row_q6_K_impl`](#quantize_row_q6_K_impl)


---
### quantize\_row\_q4\_0\_impl<!-- {{#callable:quantize_row_q4_0_impl}} -->
The `quantize_row_q4_0_impl` function quantizes a row of floating-point values into a specific format using optional quantization weights.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values to be quantized.
    - `y`: A pointer to a `block_q4_0` structure where the quantized results will be stored.
    - `n_per_row`: An integer representing the number of elements in the row to be quantized.
    - `quant_weights`: A pointer to an array of quantization weights, which can be NULL.
- **Control Flow**:
    - The function starts by checking if `quant_weights` is NULL; if so, it calls [`quantize_row_q4_0_ref`](#quantize_row_q4_0_ref) to perform a reference quantization.
    - If `quant_weights` is provided, it initializes local arrays for weights and quantization levels.
    - It calculates the sum of squares of the input array `x` to derive `sigma2`, which is used for scaling.
    - The function then processes the input array in blocks of size `QK4_0`, calculating weights for each block based on the quantization weights and the derived `sigma2`.
    - For each block, it calls `make_qx_quants` to perform the quantization and stores the results in the output structure `y`.
- **Output**: The function does not return a value but populates the `y` structure with quantized data, including a floating-point representation and quantization levels.
- **Functions called**:
    - [`quantize_row_q4_0_ref`](#quantize_row_q4_0_ref)


---
### quantize\_q4\_0<!-- {{#callable:quantize_q4_0}} -->
The `quantize_q4_0` function quantizes a source array of floats into a destination buffer using specified quantization weights.
- **Inputs**:
    - `src`: A pointer to the source array of floats that will be quantized.
    - `dst`: A pointer to the destination buffer where the quantized data will be stored.
    - `nrow`: The number of rows to process in the source array.
    - `n_per_row`: The number of elements per row in the source array.
    - `quant_weights`: A pointer to an array of quantization weights; if NULL, a reference quantization method is used.
- **Control Flow**:
    - The function first checks if `quant_weights` is NULL.
    - If `quant_weights` is NULL, it calls [`quantize_row_q4_0_ref`](#quantize_row_q4_0_ref) to perform quantization without weights and returns the total size of the quantized data.
    - If `quant_weights` is not NULL, it calculates the size of each row in the quantized format.
    - It then enters a loop that iterates over each row, calling [`quantize_row_q4_0_impl`](#quantize_row_q4_0_impl) to quantize the current row using the provided weights.
    - After processing each row, it updates the source and destination pointers to point to the next row.
- **Output**: The function returns the total size of the quantized data in bytes, calculated as the number of rows multiplied by the size of each row.
- **Functions called**:
    - [`quantize_row_q4_0_ref`](#quantize_row_q4_0_ref)
    - [`ggml_row_size`](ggml.c.driver.md#ggml_row_size)
    - [`quantize_row_q4_0_impl`](#quantize_row_q4_0_impl)


---
### quantize\_row\_q4\_1\_impl<!-- {{#callable:quantize_row_q4_1_impl}} -->
The `quantize_row_q4_1_impl` function quantizes a row of floating-point values into a specific format using optional quantization weights.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values representing the input data to be quantized.
    - `y`: A pointer to a `block_q4_1` structure where the quantized output will be stored.
    - `n_per_row`: An integer representing the number of elements in each row of the input data.
    - `quant_weights`: A pointer to an array of quantization weights, which can be NULL; if NULL, a reference quantization method is used.
- **Control Flow**:
    - The function begins by checking if `quant_weights` is NULL; if so, it calls [`quantize_row_q4_1_ref`](#quantize_row_q4_1_ref) to perform a reference quantization.
    - If `quant_weights` is provided, it initializes local arrays for weights and quantization levels.
    - It calculates the variance of the input data by summing the squares of the elements and dividing by `n_per_row`.
    - The function then processes the input data in blocks of size `QK4_1`, computing weights based on the quantization weights and the variance.
    - For each block, it calls `make_qkx3_quants` to perform the quantization, storing the results in the output structure `y`.
- **Output**: The function does not return a value but populates the `y` structure with quantized data, including quantized values and metadata.
- **Functions called**:
    - [`quantize_row_q4_1_ref`](#quantize_row_q4_1_ref)


---
### quantize\_q4\_1<!-- {{#callable:quantize_q4_1}} -->
The `quantize_q4_1` function quantizes a source array of floats into a destination buffer using specified quantization weights or a reference implementation if no weights are provided.
- **Inputs**:
    - `src`: A pointer to the source array of floats that will be quantized.
    - `dst`: A pointer to the destination buffer where the quantized data will be stored.
    - `nrow`: The number of rows to be processed in the quantization.
    - `n_per_row`: The number of elements per row in the source array.
    - `quant_weights`: A pointer to an array of quantization weights; if NULL, a reference quantization method is used.
- **Control Flow**:
    - The function first checks if `quant_weights` is NULL.
    - If `quant_weights` is NULL, it calls [`quantize_row_q4_1_ref`](#quantize_row_q4_1_ref) to perform quantization using a reference method.
    - If `quant_weights` is not NULL, it calculates the size of each row in the quantized output.
    - It then iterates over each row, calling [`quantize_row_q4_1_impl`](#quantize_row_q4_1_impl) to perform the quantization for each row using the provided weights.
    - The source pointer is incremented by `n_per_row` after processing each row, and the destination pointer is incremented by the calculated row size.
- **Output**: The function returns the total size of the quantized data in bytes, calculated as the number of rows multiplied by the size of each row.
- **Functions called**:
    - [`quantize_row_q4_1_ref`](#quantize_row_q4_1_ref)
    - [`ggml_row_size`](ggml.c.driver.md#ggml_row_size)
    - [`quantize_row_q4_1_impl`](#quantize_row_q4_1_impl)


---
### quantize\_row\_q5\_0\_impl<!-- {{#callable:quantize_row_q5_0_impl}} -->
The `quantize_row_q5_0_impl` function quantizes a row of floating-point values into a specific format using optional quantization weights.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values to be quantized.
    - `y`: A pointer to a `block_q5_0` structure where the quantized results will be stored.
    - `n_per_row`: An integer representing the number of elements in the row to be quantized.
    - `quant_weights`: A pointer to an array of quantization weights, or NULL to use a reference quantization method.
- **Control Flow**:
    - The function starts by checking if `quant_weights` is NULL; if so, it calls [`quantize_row_q5_0_ref`](#quantize_row_q5_0_ref) to perform reference quantization.
    - It calculates the sum of squares of the input array `x` to derive `sigma2`, which is used for scaling.
    - The function then processes the input array in blocks of size `QK5_0`, applying the quantization weights to each block.
    - For each block, it computes the quantized values and stores them in the `y` structure, including both the quantized data and a header with additional information.
- **Output**: The function does not return a value but populates the `y` structure with quantized data and metadata derived from the input array `x`.
- **Functions called**:
    - [`quantize_row_q5_0_ref`](#quantize_row_q5_0_ref)


---
### quantize\_q5\_0<!-- {{#callable:quantize_q5_0}} -->
The `quantize_q5_0` function quantizes a source array of floats into a destination buffer using specified quantization weights or a reference method if no weights are provided.
- **Inputs**:
    - `src`: A pointer to the source array of floats that will be quantized.
    - `dst`: A pointer to the destination buffer where the quantized data will be stored.
    - `nrow`: The number of rows to be processed in the quantization.
    - `n_per_row`: The number of elements per row in the source array.
    - `quant_weights`: A pointer to an array of quantization weights; if NULL, a reference quantization method is used.
- **Control Flow**:
    - The function first checks if `quant_weights` is NULL.
    - If `quant_weights` is NULL, it calls [`quantize_row_q5_0_ref`](#quantize_row_q5_0_ref) to perform quantization using a reference method and returns the total size of the quantized data.
    - If `quant_weights` is not NULL, it calculates the size of each row in the quantized format.
    - It then iterates over each row, calling [`quantize_row_q5_0_impl`](#quantize_row_q5_0_impl) to perform the quantization for each row using the provided weights.
    - The source pointer is incremented by `n_per_row` after processing each row, and the destination pointer is incremented by the calculated `row_size`.
- **Output**: The function returns the total size of the quantized data in bytes, calculated as the number of rows multiplied by the size of each row.
- **Functions called**:
    - [`quantize_row_q5_0_ref`](#quantize_row_q5_0_ref)
    - [`ggml_row_size`](ggml.c.driver.md#ggml_row_size)
    - [`quantize_row_q5_0_impl`](#quantize_row_q5_0_impl)


---
### quantize\_row\_q5\_1\_impl<!-- {{#callable:quantize_row_q5_1_impl}} -->
The `quantize_row_q5_1_impl` function quantizes a row of floating-point values into a compressed format using specified quantization weights.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values representing the input data to be quantized.
    - `y`: A pointer to a `block_q5_1` structure where the quantized output will be stored.
    - `n_per_row`: An integer representing the number of elements in the input array `x` to be processed.
    - `quant_weights`: A pointer to an array of quantization weights used to scale the input values; if null, a reference quantization method is used.
- **Control Flow**:
    - The function begins by checking if `quant_weights` is null; if so, it calls [`quantize_row_q5_1_ref`](#quantize_row_q5_1_ref) to perform quantization without weights.
    - It calculates the variance of the input data by summing the squares of the elements in `x` and dividing by `n_per_row`.
    - The function then processes the input data in blocks of size `QK5_1`, calculating weights for each element based on the quantization weights and the computed variance.
    - For each block, it calls `make_qkx3_quants` to perform the quantization, storing the results in the `y` structure.
    - Finally, it constructs a compressed representation of the quantized values and stores it in the `qh` field of the `y` structure.
- **Output**: The function does not return a value but populates the `y` structure with quantized data, including the quantized values and metadata for each processed block.
- **Functions called**:
    - [`quantize_row_q5_1_ref`](#quantize_row_q5_1_ref)


---
### quantize\_q5\_1<!-- {{#callable:quantize_q5_1}} -->
The `quantize_q5_1` function quantizes a source array of floats into a destination buffer using specified quantization weights or a reference method if weights are not provided.
- **Inputs**:
    - `src`: A pointer to the source array of floats that will be quantized.
    - `dst`: A pointer to the destination buffer where the quantized data will be stored.
    - `nrow`: The number of rows to be processed in the quantization.
    - `n_per_row`: The number of elements per row in the source array.
    - `quant_weights`: A pointer to an array of quantization weights; if NULL, a reference quantization method is used.
- **Control Flow**:
    - The function first checks if `quant_weights` is NULL.
    - If `quant_weights` is NULL, it calls [`quantize_row_q5_1_ref`](#quantize_row_q5_1_ref) to perform quantization using a reference method and returns the total size of the quantized data.
    - If `quant_weights` is not NULL, it calculates the size of each row in the quantized format.
    - It then iterates over each row, calling [`quantize_row_q5_1_impl`](#quantize_row_q5_1_impl) to perform the quantization for each row using the provided weights.
    - The source pointer is incremented by `n_per_row` after processing each row, and the destination pointer is incremented by the calculated row size.
- **Output**: The function returns the total size of the quantized data in bytes, calculated as the number of rows multiplied by the size of each row.
- **Functions called**:
    - [`quantize_row_q5_1_ref`](#quantize_row_q5_1_ref)
    - [`ggml_row_size`](ggml.c.driver.md#ggml_row_size)
    - [`quantize_row_q5_1_impl`](#quantize_row_q5_1_impl)


---
### quantize\_q8\_0<!-- {{#callable:quantize_q8_0}} -->
This function quantizes a source array of floats into a destination buffer using a specific quantization method.
- **Inputs**:
    - `src`: A pointer to the source array of floats that will be quantized.
    - `dst`: A pointer to the destination buffer where the quantized data will be stored.
    - `nrow`: The number of rows in the source array.
    - `n_per_row`: The number of elements per row in the source array.
    - `quant_weights`: A pointer to quantization weights, which is not used in this function.
- **Control Flow**:
    - The function begins by defining the size of each row in the quantized format using [`ggml_row_size`](ggml.c.driver.md#ggml_row_size) with the type `GGML_TYPE_Q8_0` and the number of elements per row.
    - It then calls the [`quantize_row_q8_0_ref`](#quantize_row_q8_0_ref) function to perform the actual quantization of the source data into the destination buffer, processing a total of `nrow * n_per_row` elements.
    - Finally, the function returns the total size of the quantized data in bytes, calculated as `nrow * row_size`.
- **Output**: The function outputs the total size of the quantized data in bytes.
- **Functions called**:
    - [`ggml_row_size`](ggml.c.driver.md#ggml_row_size)
    - [`quantize_row_q8_0_ref`](#quantize_row_q8_0_ref)


---
### quantize\_row\_tq1\_0\_ref<!-- {{#callable:quantize_row_tq1_0_ref}} -->
The `quantize_row_tq1_0_ref` function quantizes a row of floating-point values into a specific format for efficient storage.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that are to be quantized.
    - `y`: A pointer to a `block_tq1_0` structure where the quantized results will be stored.
    - `k`: An integer representing the number of elements in `x`, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K`.
    - It calculates the number of blocks `nb` by dividing `k` by `QK_K`.
    - For each block, it initializes `amax` to find the absolute maximum value in the current segment of `x`.
    - It computes the inverse of the maximum value `id` to normalize the quantization.
    - The maximum value is stored in the `d` field of the `y` structure after converting it to half-precision.
    - The function then quantizes the data in `x` into the `qs` array of `y` in chunks of 32 bytes and 16 bytes, using a nested loop to process groups of 5 elements.
    - Finally, it quantizes the remaining elements into the `qh` array of `y` in groups of 4 elements.
- **Output**: The function does not return a value; instead, it populates the `y` structure with quantized data derived from the input array `x`.


---
### quantize\_row\_tq2\_0\_ref<!-- {{#callable:quantize_row_tq2_0_ref}} -->
The `quantize_row_tq2_0_ref` function quantizes a row of floating-point values into a specific format for efficient storage.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that are to be quantized.
    - `y`: A pointer to a `block_tq2_0` structure where the quantized results will be stored.
    - `k`: An integer representing the total number of elements in `x`, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K`.
    - It calculates the number of blocks `nb` by dividing `k` by `QK_K`.
    - For each block, it initializes `amax` to zero and iterates over `QK_K` elements to find the absolute maximum value in the current block.
    - It computes the inverse of the maximum value `id` to normalize the quantization process.
    - The maximum value is converted to half-precision and stored in `y[i].d`.
    - The function then quantizes the values in chunks of 32 bytes, processing 4 elements at a time and encoding them into a single byte.
    - The quantized values are stored in the `qs` array of the `y` structure.
- **Output**: The function does not return a value; instead, it modifies the `y` structure in place to store the quantized representation of the input data.


---
### quantize\_tq1\_0<!-- {{#callable:quantize_tq1_0}} -->
The `quantize_tq1_0` function quantizes a source array of floats into a destination buffer using a specified row size.
- **Inputs**:
    - `src`: A pointer to the source array of floats that will be quantized.
    - `dst`: A pointer to the destination buffer where the quantized data will be stored.
    - `nrow`: The number of rows in the source data to be quantized.
    - `n_per_row`: The number of elements per row in the source data.
    - `quant_weights`: A pointer to an array of quantization weights, which is not used in this function.
- **Control Flow**:
    - The function begins by ignoring the `quant_weights` parameter as it is not utilized.
    - It calculates the size of each row in the quantized format using the [`ggml_row_size`](ggml.c.driver.md#ggml_row_size) function with the type `GGML_TYPE_TQ1_0` and the number of elements per row.
    - The function then calls [`quantize_row_tq1_0_ref`](#quantize_row_tq1_0_ref), passing the source pointer, destination pointer, and the total number of elements to quantize (calculated as `nrow * n_per_row`).
    - Finally, it returns the total size of the quantized data in bytes, calculated as `nrow * row_size`.
- **Output**: The function returns the total size in bytes of the quantized data stored in the destination buffer.
- **Functions called**:
    - [`ggml_row_size`](ggml.c.driver.md#ggml_row_size)
    - [`quantize_row_tq1_0_ref`](#quantize_row_tq1_0_ref)


---
### quantize\_tq2\_0<!-- {{#callable:quantize_tq2_0}} -->
The `quantize_tq2_0` function quantizes a source array of floats into a destination buffer using a specified row size.
- **Inputs**:
    - `src`: A pointer to the source array of floats that will be quantized.
    - `dst`: A pointer to the destination buffer where the quantized data will be stored.
    - `nrow`: The number of rows in the source data.
    - `n_per_row`: The number of elements per row in the source data.
    - `quant_weights`: A pointer to quantization weights, which is not used in this function.
- **Control Flow**:
    - The function begins by defining the size of each row based on the type `GGML_TYPE_TQ2_0` and the number of elements per row using the [`ggml_row_size`](ggml.c.driver.md#ggml_row_size) function.
    - The [`quantize_row_tq2_0_ref`](#quantize_row_tq2_0_ref) function is called to perform the actual quantization of the source data into the destination buffer, processing a total of `nrow * n_per_row` elements.
    - Finally, the function returns the total size of the quantized data in bytes, calculated as `nrow * row_size`.
- **Output**: The function outputs the total size of the quantized data in bytes.
- **Functions called**:
    - [`ggml_row_size`](ggml.c.driver.md#ggml_row_size)
    - [`quantize_row_tq2_0_ref`](#quantize_row_tq2_0_ref)


---
### dequantize\_row\_tq1\_0<!-- {{#callable:dequantize_row_tq1_0}} -->
The `dequantize_row_tq1_0` function converts quantized data from a `block_tq1_0` structure into floating-point values and stores them in an output array.
- **Inputs**:
    - `x`: A pointer to a `block_tq1_0` structure containing quantized data.
    - `y`: A pointer to a float array where the dequantized output values will be stored.
    - `k`: An integer representing the total number of quantized elements, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K` and calculates `nb`, the number of blocks.
    - It initializes a lookup array `pow3` containing powers of 3 for scaling the quantized values.
    - A loop iterates over each block, converting the quantized data to floating-point values.
    - Within the first nested loop, it processes the first part of the quantized data in chunks of 32, applying the scaling and storing the results.
    - The second nested loop processes the remaining quantized data in chunks of 16, similarly scaling and storing the results.
    - Finally, a loop processes the `qh` array of quantized values, applying the same scaling and storing the results.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the dequantized floating-point values derived from the input quantized data.


---
### dequantize\_row\_tq2\_0<!-- {{#callable:dequantize_row_tq2_0}} -->
The `dequantize_row_tq2_0` function converts quantized data from a `block_tq2_0` structure into floating-point values.
- **Inputs**:
    - `x`: A pointer to a `block_tq2_0` structure containing quantized data.
    - `y`: A pointer to a float array where the dequantized values will be stored.
    - `k`: An integer representing the total number of quantized elements, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K` to ensure valid input.
    - It calculates `nb`, the number of blocks to process, by dividing `k` by `QK_K`.
    - A loop iterates over each block, extracting the dequantization factor `d` from the `d` field of the current block.
    - Within the block loop, another loop iterates over the quantized values in chunks of 32 bytes.
    - For each chunk, it further loops through 4 segments, extracting 2 bits at a time to retrieve the quantized value `q`.
    - The dequantized value is computed by scaling `q` with `d` and stored in the output array `y`.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the dequantized floating-point values derived from the input quantized data.


---
### dequantize\_row\_iq2\_xxs<!-- {{#callable:dequantize_row_iq2_xxs}} -->
The `dequantize_row_iq2_xxs` function converts quantized data from a `block_iq2_xxs` structure into floating-point values and stores them in an output array.
- **Inputs**:
    - `x`: A pointer to a constant array of `block_iq2_xxs` structures containing quantized data.
    - `y`: A pointer to a float array where the dequantized output will be stored.
    - `k`: An integer representing the total number of quantized elements, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K`.
    - It calculates the number of blocks `nb` by dividing `k` by `QK_K`.
    - A loop iterates over each block in the input array `x`.
    - Within the block loop, it converts the quantized float value `d` from half-precision to single-precision.
    - Another loop iterates over the quantized data in chunks of 32 bits, extracting two `uint32_t` values into `aux32`.
    - It calculates a base dequantized value `db` using the extracted data.
    - A nested loop processes each of the four segments of the quantized data, retrieving a grid of values and determining the sign for each output element.
    - Finally, it populates the output array `y` with the dequantized values, adjusting for sign based on the extracted data.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the dequantized floating-point values derived from the input quantized data.


---
### dequantize\_row\_iq2\_xs<!-- {{#callable:dequantize_row_iq2_xs}} -->
The `dequantize_row_iq2_xs` function performs dequantization of a row of data from a quantized format into floating-point representation.
- **Inputs**:
    - `x`: A pointer to a constant `block_iq2_xs` structure that contains the quantized data to be dequantized.
    - `y`: A pointer to a float array where the dequantized output will be stored.
    - `k`: An integer representing the total number of quantized elements, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K` to ensure valid input.
    - It calculates `nb`, the number of blocks to process, by dividing `k` by `QK_K`.
    - A loop iterates over each block, extracting the dequantized value `d` from the input data.
    - Within this loop, another loop iterates over `QK_K/32` to process each scale factor.
    - For each scale factor, it computes two derived values stored in `db` based on the quantized data and scale.
    - A nested loop processes 4 quantized values, where for each value, it retrieves a grid and sign information.
    - An innermost loop iterates over 8 elements, applying the dequantization formula and storing the results in the output array `y`.
- **Output**: The function outputs a dequantized float array `y`, populated with the floating-point representation of the quantized input data.


---
### dequantize\_row\_iq2\_s<!-- {{#callable:dequantize_row_iq2_s}} -->
The `dequantize_row_iq2_s` function performs dequantization of a row of quantized data into floating-point values.
- **Inputs**:
    - `x`: A pointer to a constant array of `block_iq2_s` structures containing quantized data.
    - `y`: A pointer to a float array where the dequantized output will be stored.
    - `k`: An integer representing the number of quantized blocks, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K`.
    - It calculates the number of blocks `nb` by dividing `k` by `QK_K`.
    - A loop iterates over each block, extracting the dequantization factor `d` and pointers to quantization scales and signs.
    - Within a nested loop, it processes each block in chunks of 32 bits, calculating two dequantization values `db[0]` and `db[1]` based on the scales.
    - Another loop iterates over 4 segments, where it retrieves the appropriate grid values and applies the dequantization logic to fill the output array `y` with the computed floating-point values, adjusting for signs.
- **Output**: The function outputs a dequantized array of floating-point values stored in the array pointed to by `y`.


---
### dequantize\_row\_iq3\_xxs<!-- {{#callable:dequantize_row_iq3_xxs}} -->
The `dequantize_row_iq3_xxs` function performs dequantization of a row of data from a quantized format into floating-point values.
- **Inputs**:
    - `x`: A pointer to a constant array of `block_iq3_xxs` structures containing quantized data.
    - `y`: A pointer to a float array where the dequantized output will be stored.
    - `k`: An integer representing the total number of quantized elements, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K` to ensure valid input.
    - It calculates the number of blocks `nb` by dividing `k` by `QK_K`.
    - A loop iterates over each block, extracting the floating-point representation of the data and the quantization scales.
    - Within this loop, another loop processes each block in chunks of 32 bits, extracting the scale and sign information.
    - For each of the four segments in the current chunk, it retrieves the corresponding grid values and applies the dequantization formula to compute the final floating-point values.
    - The results are stored in the output array `y`, which is updated in each iteration.
- **Output**: The function outputs an array of floating-point values in `y`, which are the dequantized results corresponding to the input quantized data in `x`.


---
### dequantize\_row\_iq3\_s<!-- {{#callable:dequantize_row_iq3_s}} -->
The `dequantize_row_iq3_s` function performs dequantization of a row of quantized data into floating-point values.
- **Inputs**:
    - `x`: A pointer to a constant `block_iq3_s` structure containing quantized data.
    - `y`: A pointer to a float array where the dequantized values will be stored.
    - `k`: An integer representing the total number of quantized elements, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K` and calculates `nb`, the number of blocks to process.
    - A loop iterates over each block, extracting the dequantization factor `d` and pointers to quantization scales and signs.
    - Within the block loop, another loop processes pairs of 32-bit segments, calculating two dequantization factors `db1` and `db2` based on the scales.
    - For each of the two segments, a nested loop iterates over 4 elements, retrieving grid values and applying the dequantization formula, storing results in the output array `y`.
    - The inner loops handle sign adjustments based on the `signs` array, ensuring correct handling of negative values.
    - After processing the first segment, the function updates pointers for `qs`, `signs`, and `qh` to prepare for the next segment.
- **Output**: The function outputs a dequantized array of floating-point values in the `y` array, corresponding to the quantized input data.


---
### dequantize\_row\_iq1\_s<!-- {{#callable:dequantize_row_iq1_s}} -->
The `dequantize_row_iq1_s` function performs dequantization of a row of quantized data into floating-point values.
- **Inputs**:
    - `x`: A pointer to a constant array of `block_iq1_s` structures containing quantized data.
    - `y`: A pointer to a float array where the dequantized output will be stored.
    - `k`: An integer representing the total number of quantized blocks, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K`.
    - It calculates the number of blocks `nb` by dividing `k` by `QK_K`.
    - A loop iterates over each block, extracting the floating-point representation `d`, and pointers to quantization scales `qs` and quantization headers `qh`.
    - Within this loop, another loop iterates over `QK_K/32` to process each quantization header.
    - For each header, it calculates a scaling factor `dl` and a delta value based on the header's sign bit.
    - A nested loop processes 4 segments of the quantization scale, where it retrieves a grid of quantized values and applies the scaling and delta to produce 8 dequantized float values, which are stored in the output array `y`.
- **Output**: The function outputs a dequantized array of floating-point values in the `y` array, corresponding to the quantized input data.


---
### dequantize\_row\_iq1\_m<!-- {{#callable:dequantize_row_iq1_m}} -->
The `dequantize_row_iq1_m` function performs dequantization of a row of quantized data into floating-point values.
- **Inputs**:
    - `x`: A pointer to a `block_iq1_m` structure containing quantized data and scaling information.
    - `y`: A pointer to a float array where the dequantized output will be stored.
    - `k`: An integer representing the total number of quantized elements, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K` and calculates the number of blocks `nb` as `k / QK_K`.
    - It initializes arrays for `delta` and `idx`, and a variable `scale` to hold the scaling factor.
    - A loop iterates over each block, extracting scale values and calculating the floating-point scale `d`.
    - Within the block loop, another loop processes quantized values, calculating two different deltas and indices for accessing the quantized grid.
    - For each index, it retrieves the corresponding grid values and applies the scaling and delta adjustments to populate the output array `y`.
    - The inner loops handle the processing of 8 values at a time, updating the output pointer `y` accordingly.
- **Output**: The function outputs a dequantized array of floating-point values in the `y` array, corresponding to the quantized input data from `x`.


---
### dequantize\_row\_iq4\_nl<!-- {{#callable:dequantize_row_iq4_nl}} -->
The `dequantize_row_iq4_nl` function performs dequantization of a block of quantized data into floating-point values.
- **Inputs**:
    - `x`: A pointer to an array of `block_iq4_nl` structures containing quantized data.
    - `y`: A pointer to an array of floats where the dequantized output will be stored.
    - `k`: An integer representing the total number of quantized elements, which must be a multiple of `QK4_NL`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK4_NL` to ensure valid input.
    - It calculates the number of blocks `nb` by dividing `k` by `QK4_NL`.
    - A loop iterates over each block, extracting the quantized values and the scaling factor.
    - Within the block loop, another loop processes half of the quantized values, applying the dequantization formula and storing the results in the output array.
    - The output pointer `y` is incremented by `QK4_NL` after processing each block, and the quantized data pointer `qs` is adjusted accordingly.
- **Output**: The function outputs an array of floating-point values in `y`, which are the dequantized representations of the input quantized data from `x`.


---
### dequantize\_row\_iq4\_xs<!-- {{#callable:dequantize_row_iq4_xs}} -->
The `dequantize_row_iq4_xs` function performs dequantization of a row of quantized data into floating-point values.
- **Inputs**:
    - `x`: A pointer to an array of `block_iq4_xs` structures containing quantized data.
    - `y`: A pointer to an array of floats where the dequantized output will be stored.
    - `k`: An integer representing the total number of quantized elements, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K`.
    - It calculates the number of blocks `nb` by dividing `k` by `QK_K`.
    - A loop iterates over each block, extracting the quantized values and scale factors.
    - Within the block loop, another loop processes the quantized data in chunks, calculating the dequantized values based on the scale and quantized values.
    - The dequantized values are stored in the output array `y`, and pointers are updated accordingly.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the dequantized floating-point values derived from the input quantized data.


---
### quantize\_row\_q8\_K\_ref<!-- {{#callable:quantize_row_q8_K_ref}} -->
The `quantize_row_q8_K_ref` function quantizes a row of floating-point values into a block of quantized integers while computing the necessary scaling factors and sums.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that are to be quantized.
    - `y`: A pointer to a `block_q8_K` structure where the quantized results and additional data will be stored.
    - `k`: An integer representing the total number of elements in `x`, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K`.
    - It calculates the number of blocks `nb` by dividing `k` by `QK_K`.
    - A loop iterates over each block, initializing `max` and `amax` to zero.
    - Within the block loop, another loop iterates over `QK_K` elements to find the maximum absolute value and its corresponding original value.
    - If the maximum absolute value (`amax`) is zero, it sets the quantized data to zero and continues to the next block.
    - If `amax` is non-zero, it calculates the inverse scale factor `iscale` based on the maximum value.
    - Another loop quantizes each element in the block using the [`nearest_int`](#nearest_int) function and stores the results in `y[i].qs`.
    - A final loop computes the sum of quantized values in groups of 16 and stores these sums in `y[i].bsums`.
    - The inverse scale factor is stored in `y[i].d` before moving to the next block.
- **Output**: The function does not return a value; instead, it modifies the `y` structure in place, storing the quantized values, their sums, and the inverse scale factor for each block.
- **Functions called**:
    - [`nearest_int`](#nearest_int)


---
### dequantize\_row\_q8\_K<!-- {{#callable:dequantize_row_q8_K}} -->
The `dequantize_row_q8_K` function converts quantized data from a `block_q8_K` structure into floating-point values.
- **Inputs**:
    - `x`: A pointer to an array of `block_q8_K` structures containing quantized data.
    - `y`: A pointer to a float array where the dequantized values will be stored.
    - `k`: An integer representing the total number of quantized elements, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function asserts that `k` is a multiple of `QK_K` to ensure valid input.
    - It calculates the number of blocks `nb` by dividing `k` by `QK_K`.
    - It iterates over each block and each quantized value within the block, multiplying the block's scalar `d` by each quantized value `qs[j]`.
    - The result of the multiplication is stored in the output array `y`.
- **Output**: The function outputs a sequence of floating-point values in the array pointed to by `y`, which are the dequantized results derived from the input quantized data.


---
### iq2\_data\_index<!-- {{#callable:iq2_data_index}} -->
Determines the index corresponding to a given `ggml_type` enumeration.
- **Inputs**:
    - `type`: An enumeration of type `ggml_type` that specifies the type of data for which the index is to be determined.
- **Control Flow**:
    - First, the function asserts that the provided `type` is one of the valid `ggml_type` values using `GGML_ASSERT`.
    - Then, it evaluates the `type` against several conditions to return an appropriate index based on the specific type.
- **Output**: Returns an integer index corresponding to the input `type`, where each type maps to a specific index value.


---
### iq2\_grid\_size<!-- {{#callable:iq2_grid_size}} -->
Calculates the grid size based on the specified `ggml_type`.
- **Inputs**:
    - `type`: An enumeration value of type `ggml_type` that specifies the type of grid.
- **Control Flow**:
    - First, the function asserts that the provided `type` is one of the valid `ggml_type` values using `GGML_ASSERT`.
    - Then, it evaluates the `type` using a series of conditional expressions to determine the corresponding grid size.
    - The function returns 256 for `GGML_TYPE_IQ2_XXS`, 512 for `GGML_TYPE_IQ2_XS`, a predefined constant `NGRID_IQ1S` for `GGML_TYPE_IQ1_S` and `GGML_TYPE_IQ1_M`, and 1024 for all other types.
- **Output**: Returns an integer representing the grid size corresponding to the input `type`.


---
### iq2\_compare\_func<!-- {{#callable:iq2_compare_func}} -->
Compares two integer arrays based on their first and second elements.
- **Inputs**:
    - `left`: A pointer to the first integer array to compare.
    - `right`: A pointer to the second integer array to compare.
- **Control Flow**:
    - Cast the `left` and `right` pointers to integer pointers `l` and `r` respectively.
    - Compare the first elements of the arrays pointed to by `l` and `r`.
    - If the first element of `l` is less than that of `r`, return -1.
    - If the first element of `l` is greater than that of `r`, return 1.
    - If the first elements are equal, compare the second elements.
    - If the second element of `l` is less than that of `r`, return -1.
    - If the second element of `l` is greater than that of `r`, return 1.
    - If both elements are equal, return 0.
- **Output**: Returns an integer indicating the comparison result: -1 if `left` is less than `right`, 1 if greater, and 0 if they are equal.


---
### iq2xs\_init\_impl<!-- {{#callable:iq2xs_init_impl}} -->
Initializes the IQ2XS grid and its associated data structures based on the specified type.
- **Inputs**:
    - `type`: An enumeration value of type `ggml_type` that specifies the type of grid to initialize.
- **Control Flow**:
    - Calculates the index and grid size based on the input `type` using [`iq2_data_index`](#iq2_data_index) and [`iq2_grid_size`](#iq2_grid_size).
    - Checks if the grid for the specified index is already initialized; if so, the function returns early.
    - Defines static arrays for different grid configurations based on the type.
    - Allocates memory for the grid and initializes it based on the selected grid configuration.
    - Populates the `kmap_q2xs` array with initial values and sets up a mapping from grid positions to indices.
    - Calculates distances for neighbors and populates the `kneighbors_q2xs` array with the nearest neighbors for each position.
    - Frees any dynamically allocated memory before the function exits.
- **Output**: The function does not return a value but initializes global structures that can be accessed later, specifically setting up the grid and neighbor mappings in `iq2_data`.
- **Functions called**:
    - [`iq2_data_index`](#iq2_data_index)
    - [`iq2_grid_size`](#iq2_grid_size)


---
### iq2xs\_free\_impl<!-- {{#callable:iq2xs_free_impl}} -->
Frees allocated memory for grid, map, and neighbours associated with a specific `ggml_type`.
- **Inputs**:
    - `type`: An enumeration value of type `ggml_type` that specifies which data structure to free.
- **Control Flow**:
    - The function begins by asserting that the provided `type` is one of the valid `ggml_type` values.
    - It retrieves the index corresponding to the `type` using the [`iq2_data_index`](#iq2_data_index) function.
    - If the `grid` member of the `iq2_data` structure at the retrieved index is not NULL, it proceeds to free the memory allocated for `grid`, `map`, and `neighbours`.
    - After freeing each of these pointers, it sets them to NULL to avoid dangling pointers.
- **Output**: The function does not return a value; it performs memory deallocation and ensures that pointers are set to NULL.
- **Functions called**:
    - [`iq2_data_index`](#iq2_data_index)


---
### iq2\_find\_best\_neighbour<!-- {{#callable:iq2_find_best_neighbour}} -->
Finds the best neighbor from a list based on a weighted distance metric.
- **Inputs**:
    - `neighbours`: A pointer to an array of `uint16_t` values where the first element indicates the number of neighbors, and subsequent elements are indices of the neighbors in the `grid`.
    - `grid`: A pointer to an array of `uint64_t` values representing the grid data from which neighbors are selected.
    - `xval`: A pointer to an array of `float` values representing the reference values to compare against.
    - `weight`: A pointer to an array of `float` values representing the weights for each dimension in the distance calculation.
    - `scale`: A `float` value used to scale the grid values during the distance calculation.
    - `L`: A pointer to an array of `int8_t` values where the result will be stored after finding the best neighbor.
- **Control Flow**:
    - The function starts by retrieving the number of neighbors from the `neighbours` array and asserts that it is greater than zero.
    - It initializes `best_d2` to the maximum float value and `grid_index` to -1.
    - A loop iterates over each neighbor, calculating the weighted squared distance (`d2`) between the scaled grid values and the reference values (`xval`).
    - If the calculated distance `d2` is less than the current `best_d2`, it updates `best_d2` and records the current neighbor's index in `grid_index`.
    - After evaluating all neighbors, it asserts that a valid `grid_index` was found.
    - It retrieves the best neighbor's grid values and populates the output array `L` with adjusted values based on the best neighbor.
    - Finally, it returns the index of the best neighbor.
- **Output**: Returns the index of the best neighbor found in the `neighbours` array.


---
### quantize\_row\_iq2\_xxs\_impl<!-- {{#callable:quantize_row_iq2_xxs_impl}} -->
The `quantize_row_iq2_xxs_impl` function quantizes a row of floating-point data using specified quantization weights and stores the result in a structured format.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values representing the input data to be quantized.
    - `vy`: A pointer to a structure where the quantized output will be stored.
    - `n`: An integer representing the number of elements in the input data, which must be a multiple of `QK_K`.
    - `quant_weights`: A pointer to an array of floating-point values representing the quantization weights used in the quantization process.
- **Control Flow**:
    - The function begins by asserting the validity of input parameters and initializing necessary variables.
    - It calculates the number of blocks based on the input size and iterates over each block.
    - For each block, it computes the sum of squares of the input values to derive a variance estimate.
    - It then calculates weights based on the quantization weights and the variance, and processes the input values to determine their signs and magnitudes.
    - The function uses a nested loop to find the best quantization levels and scales for the input values, adjusting for signs and ensuring the quantization is optimal.
    - Finally, it stores the quantized values and scales in the output structure.
- **Output**: The function outputs a quantized representation of the input data in the `vy` structure, including quantized values and their corresponding scales.
- **Functions called**:
    - [`iq2_data_index`](#iq2_data_index)
    - [`make_qp_quants`](#make_qp_quants)
    - [`nearest_int`](#nearest_int)
    - [`iq2_find_best_neighbour`](#iq2_find_best_neighbour)


---
### quantize\_row\_iq2\_xs\_impl<!-- {{#callable:quantize_row_iq2_xs_impl}} -->
The `quantize_row_iq2_xs_impl` function quantizes a row of floating-point data using specified quantization weights and stores the results in a structured format.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that represent the data to be quantized.
    - `vy`: A pointer to a structure where the quantized results will be stored.
    - `n`: An integer representing the number of elements in the input array, which must be a multiple of `QK_K`.
    - `quant_weights`: A pointer to an array of floating-point values that represent the quantization weights used in the quantization process.
- **Control Flow**:
    - The function begins by asserting the validity of input parameters and initializing necessary variables.
    - It calculates the number of blocks based on the input size and iterates over each block.
    - For each block, it computes the sum of squares and the variance of the input data.
    - It then calculates weights based on the quantization weights and the variance.
    - The function processes the input data in groups, determining the quantization levels and scales for each group.
    - It checks if the quantized values are on a predefined grid and adjusts them if necessary.
    - Finally, it stores the quantized values and scales in the output structure.
- **Output**: The function outputs quantized data stored in the structure pointed to by `vy`, including quantized values and their corresponding scales.
- **Functions called**:
    - [`iq2_data_index`](#iq2_data_index)
    - [`nearest_int`](#nearest_int)
    - [`iq2_find_best_neighbour`](#iq2_find_best_neighbour)


---
### quantize\_iq2\_xxs<!-- {{#callable:quantize_iq2_xxs}} -->
The `quantize_iq2_xxs` function quantizes input data from a source array into a destination array using specified quantization weights.
- **Inputs**:
    - `src`: A pointer to the source array of float values that will be quantized.
    - `dst`: A pointer to the destination memory where the quantized data will be stored.
    - `nrow`: The number of rows to process in the source array.
    - `n_per_row`: The number of elements in each row of the source array, which must be a multiple of `QK_K`.
    - `quant_weights`: A pointer to an array of quantization weights used during the quantization process.
- **Control Flow**:
    - The function asserts that `n_per_row` is a multiple of `QK_K` to ensure valid quantization.
    - It calculates the number of blocks (`nblock`) by dividing `n_per_row` by `QK_K`.
    - A loop iterates over each row of the source array, calling [`quantize_row_iq2_xxs_impl`](#quantize_row_iq2_xxs_impl) to perform the quantization for each row.
    - After processing each row, the source pointer is incremented by `n_per_row`, and the destination pointer is incremented by the size of the block.
- **Output**: The function returns the total size in bytes of the quantized data stored in the destination array, calculated as the product of `nrow`, `nblock`, and the size of `block_iq2_xxs`.
- **Functions called**:
    - [`quantize_row_iq2_xxs_impl`](#quantize_row_iq2_xxs_impl)


---
### quantize\_iq2\_xs<!-- {{#callable:quantize_iq2_xs}} -->
The `quantize_iq2_xs` function quantizes input data from a source array into a destination array using specified quantization weights.
- **Inputs**:
    - `src`: A pointer to the source array of float values that will be quantized.
    - `dst`: A pointer to the destination memory where the quantized data will be stored.
    - `nrow`: The number of rows to process in the source array.
    - `n_per_row`: The number of elements in each row of the source array, which must be a multiple of `QK_K`.
    - `quant_weights`: A pointer to an array of float values used as weights for the quantization process.
- **Control Flow**:
    - The function asserts that `n_per_row` is a multiple of `QK_K` to ensure valid quantization.
    - It calculates the number of blocks (`nblock`) by dividing `n_per_row` by `QK_K`.
    - A loop iterates over each row from 0 to `nrow`, calling [`quantize_row_iq2_xs_impl`](#quantize_row_iq2_xs_impl) to perform the quantization for each row.
    - After processing each row, the source pointer is incremented by `n_per_row`, and the destination pointer is incremented by the size of the block.
- **Output**: The function returns the total size in bytes of the quantized data stored in the destination array, calculated as `nrow * nblock * sizeof(block_iq2_xs)`.
- **Functions called**:
    - [`quantize_row_iq2_xs_impl`](#quantize_row_iq2_xs_impl)


---
### iq3\_data\_index<!-- {{#callable:iq3_data_index}} -->
Determines the index corresponding to a given `grid_size` of either 256 or 512.
- **Inputs**:
    - `grid_size`: An integer representing the size of the grid, which must be either 256 or 512.
- **Control Flow**:
    - The function asserts that `grid_size` is either 256 or 512 using `GGML_ASSERT`.
    - If `grid_size` is 256, the function returns 0; if it is 512, it returns 1.
- **Output**: Returns an integer index: 0 for a `grid_size` of 256 and 1 for a `grid_size` of 512.


---
### iq3\_compare\_func<!-- {{#callable:iq3_compare_func}} -->
Compares two integer arrays based on their first two elements.
- **Inputs**:
    - `left`: A pointer to the first integer array to compare.
    - `right`: A pointer to the second integer array to compare.
- **Control Flow**:
    - The function casts the input pointers to integer pointers.
    - It compares the first elements of the two arrays; if the first element of `left` is less than that of `right`, it returns -1.
    - If the first element of `left` is greater than that of `right`, it returns 1.
    - If the first elements are equal, it then compares the second elements; if the second element of `left` is less than that of `right`, it returns -1.
    - If the second element of `left` is greater than that of `right`, it returns 1.
    - If both elements are equal, it returns 0.
- **Output**: Returns an integer indicating the comparison result: -1 if `left` is less than `right`, 1 if greater, and 0 if they are equal.


---
### iq3xs\_init\_impl<!-- {{#callable:iq3xs_init_impl}} -->
Initializes the IQ3XS grid and its associated data structures based on the specified grid size.
- **Inputs**:
    - `grid_size`: An integer representing the size of the grid to be initialized, which can be either 256 or 512.
- **Control Flow**:
    - Calculates the index in the `iq3_data` array using the [`iq3_data_index`](#iq3_data_index) function based on the provided `grid_size`.
    - Checks if the grid at the calculated index is already initialized; if so, the function returns early.
    - Defines two static arrays, `kgrid_256` and `kgrid_512`, which contain predefined grid values for the respective sizes.
    - Allocates memory for the grid and populates it based on the selected grid size.
    - Initializes a mapping array `kmap_q3xs` to track the indices of the grid points.
    - Calculates the squared distances from each grid point to all possible positions and sorts them to find neighbors.
    - Populates the `kneighbors_q3xs` array with the nearest neighbors for each grid point, while also keeping track of the number of neighbors.
- **Output**: The function does not return a value but initializes the `iq3_data` structure with the grid, mapping, and neighbor information for further processing.
- **Functions called**:
    - [`iq3_data_index`](#iq3_data_index)


---
### iq3xs\_free\_impl<!-- {{#callable:iq3xs_free_impl}} -->
Frees allocated memory for grid-related data structures based on the specified grid size.
- **Inputs**:
    - `grid_size`: An integer representing the size of the grid, which must be either 256 or 512.
- **Control Flow**:
    - The function asserts that `grid_size` is either 256 or 512 using `GGML_ASSERT`.
    - It calculates the index in the `iq3_data` array corresponding to the provided `grid_size` by calling `iq3_data_index(grid_size)`.
    - If the `grid` pointer at the calculated index is not NULL, it proceeds to free the memory allocated for `grid`, `map`, and `neighbours`.
    - After freeing each pointer, it sets them to NULL to avoid dangling pointers.
- **Output**: The function does not return a value; it performs memory deallocation for the specified grid's data structures.
- **Functions called**:
    - [`iq3_data_index`](#iq3_data_index)


---
### iq3\_find\_best\_neighbour<!-- {{#callable:iq3_find_best_neighbour}} -->
Finds the best neighbor in a grid based on a weighted distance metric.
- **Inputs**:
    - `neighbours`: An array of indices representing neighboring grid points, where the first element indicates the number of neighbors.
    - `grid`: A pointer to an array representing the grid data, where each grid point contains 4 values.
    - `xval`: An array of 4 float values representing the target values to compare against.
    - `weight`: An array of 4 float values representing the weights for each dimension in the distance calculation.
    - `scale`: A float value used to scale the grid values during the distance calculation.
    - `L`: A pointer to an array of 4 int8_t values where the result will be stored.
- **Control Flow**:
    - The function starts by retrieving the number of neighbors from the `neighbours` array and asserts that it is greater than zero.
    - It initializes `best_d2` to the maximum float value and `grid_index` to -1.
    - A loop iterates over each neighbor, calculating the weighted squared distance (`d2`) between the scaled grid values and the target values (`xval`).
    - If the calculated distance `d2` is less than the current `best_d2`, it updates `best_d2` and records the current neighbor's index as `grid_index`.
    - After evaluating all neighbors, it asserts that a valid `grid_index` was found.
    - It retrieves the grid values corresponding to the best neighbor and populates the output array `L` with transformed values.
- **Output**: Returns the index of the best neighbor found in the grid.


---
### quantize\_row\_iq3\_xxs\_impl<!-- {{#callable:quantize_row_iq3_xxs_impl}} -->
The `quantize_row_iq3_xxs_impl` function performs quantization of input data into a specified format based on provided weights and grid parameters.
- **Inputs**:
    - `grid_size`: An integer representing the size of the grid used for quantization.
    - `x`: A pointer to an array of floats representing the input data to be quantized.
    - `vy`: A pointer to a memory location where the quantized output will be stored.
    - `n`: An integer representing the number of elements in the input data.
    - `quant_weights`: A pointer to an array of floats representing the quantization weights, which can be NULL.
- **Control Flow**:
    - The function begins by determining the index for the quantization data based on the `grid_size`.
    - It retrieves the quantization grid, mapping, and neighbor data from a global structure.
    - Assertions are made to ensure that necessary data structures are initialized and that the input size is valid.
    - The function initializes various local variables and arrays for processing the input data.
    - A loop iterates over blocks of the input data, performing calculations to determine quantization scales and indices.
    - Within the loop, further nested loops process segments of the input data, calculating weights and determining quantization levels.
    - The function checks if the calculated scales are valid and adjusts them if necessary, ensuring they are non-negative.
    - Finally, the quantized data is stored in the output pointer, and the function continues to the next block until all input data is processed.
- **Output**: The function does not return a value but writes the quantized data to the memory location pointed to by `vy`, which is structured based on the `grid_size`.
- **Functions called**:
    - [`iq3_data_index`](#iq3_data_index)
    - [`nearest_int`](#nearest_int)
    - [`iq3_find_best_neighbour`](#iq3_find_best_neighbour)


---
### quantize\_iq3\_xxs<!-- {{#callable:quantize_iq3_xxs}} -->
The `quantize_iq3_xxs` function quantizes a source array of floats into a destination buffer using specified quantization weights.
- **Inputs**:
    - `src`: A pointer to the source array of floats that will be quantized.
    - `dst`: A pointer to the destination buffer where the quantized data will be stored.
    - `nrow`: The number of rows to process in the source array.
    - `n_per_row`: The number of elements per row in the source array, which must be a multiple of `QK_K`.
    - `quant_weights`: A pointer to an array of quantization weights used during the quantization process.
- **Control Flow**:
    - The function asserts that `n_per_row` is a multiple of `QK_K` using `GGML_ASSERT`.
    - It calculates the number of blocks (`nblock`) by dividing `n_per_row` by `QK_K`.
    - A loop iterates over each row from 0 to `nrow`, calling [`quantize_row_iq3_xxs_impl`](#quantize_row_iq3_xxs_impl) to perform the quantization for each row.
    - After processing each row, the source pointer `src` is incremented by `n_per_row`, and the destination pointer `qrow` is incremented by the size of the block.
- **Output**: The function returns the total size of the quantized data in bytes, calculated as the product of `nrow`, `nblock`, and the size of `block_iq3_xxs`.
- **Functions called**:
    - [`quantize_row_iq3_xxs_impl`](#quantize_row_iq3_xxs_impl)


---
### quantize\_row\_iq3\_xxs\_ref<!-- {{#callable:quantize_row_iq3_xxs_ref}} -->
Quantizes a row of floating-point values into a specified format.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that need to be quantized.
    - `y`: A pointer to a `block_iq3_xxs` structure where the quantized output will be stored.
    - `k`: An integer representing the number of elements to quantize, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K` to ensure valid input for quantization.
    - Then, it calls the [`quantize_row_iq3_xxs_impl`](#quantize_row_iq3_xxs_impl) function with parameters including a fixed value of 256, the input array `x`, the output structure `y`, and the count `k`.
- **Output**: The function does not return a value; instead, it modifies the `block_iq3_xxs` structure pointed to by `y` to contain the quantized representation of the input data.
- **Functions called**:
    - [`quantize_row_iq3_xxs_impl`](#quantize_row_iq3_xxs_impl)


---
### quantize\_row\_iq3\_s\_impl<!-- {{#callable:quantize_row_iq3_s_impl}} -->
The `quantize_row_iq3_s_impl` function quantizes a row of floating-point data into a compressed format using specified quantization weights and scales.
- **Inputs**:
    - `block_size`: An integer representing the size of the blocks to be processed.
    - `x`: A pointer to an array of floats representing the input data to be quantized.
    - `vy`: A pointer to a memory location where the quantized output will be stored.
    - `n`: An integer representing the total number of elements in the input data.
    - `quant_weights`: A pointer to an array of floats representing the quantization weights, or NULL if not used.
    - `scales`: A pointer to an array of floats where the computed scales for quantization will be stored.
    - `weight`: A pointer to an array of floats used to store intermediate weight calculations.
    - `xval`: A pointer to an array of floats used to store the absolute values of the input data.
    - `L`: A pointer to an array of int8_t used to store quantization levels.
    - `Laux`: A pointer to an auxiliary array of int8_t used for temporary storage of quantization levels.
    - `waux`: A pointer to an array of floats used for temporary storage of weights.
    - `is_on_grid`: A pointer to an array of booleans indicating whether each block is on the quantization grid.
    - `is_on_grid_aux`: A pointer to an auxiliary array of booleans for temporary storage of grid status.
    - `block_signs`: A pointer to an array of uint8_t used to store the signs of the quantized values.
- **Control Flow**:
    - The function begins by asserting the validity of input parameters and initializing necessary variables.
    - It calculates the number of blocks based on the input size and iterates over each block.
    - For each block, it computes the sum of squares and the variance of the input data.
    - It then calculates weights based on the quantization weights or directly from the input data.
    - The function quantizes the input data by determining the best scale and quantization levels for each block.
    - It checks if the quantized values are on the grid and adjusts them if necessary.
    - Finally, it stores the quantized values and scales in the output structure.
- **Output**: The function outputs quantized data in the specified format, including the quantization levels and scales, stored in the provided output pointer.
- **Functions called**:
    - [`iq3_data_index`](#iq3_data_index)
    - [`nearest_int`](#nearest_int)
    - [`iq3_find_best_neighbour`](#iq3_find_best_neighbour)


---
### quantize\_iq3\_s<!-- {{#callable:quantize_iq3_s}} -->
The `quantize_iq3_s` function quantizes input floating-point data into a compressed format using specified quantization weights.
- **Inputs**:
    - `src`: A pointer to the source array of floating-point values to be quantized.
    - `dst`: A pointer to the destination where the quantized data will be stored.
    - `nrow`: The number of rows of data to process.
    - `n_per_row`: The number of elements in each row, which must be a multiple of `QK_K`.
    - `quant_weights`: A pointer to an array of quantization weights used during the quantization process.
- **Control Flow**:
    - The function begins by asserting that `n_per_row` is a multiple of `QK_K` to ensure valid processing.
    - It calculates the number of blocks (`nblock`) by dividing `n_per_row` by `QK_K`.
    - Arrays for scales, weights, values, and auxiliary data are initialized to hold intermediate results.
    - A loop iterates over each row of the input data, calling [`quantize_row_iq3_s_impl`](#quantize_row_iq3_s_impl) to perform the quantization for that row.
    - After processing each row, the source pointer is incremented to point to the next row, and the destination pointer is updated accordingly.
- **Output**: The function returns the total size in bytes of the quantized data produced, calculated as the product of the number of rows, the number of blocks, and the size of each block.
- **Functions called**:
    - [`quantize_row_iq3_s_impl`](#quantize_row_iq3_s_impl)


---
### quantize\_row\_iq3\_s\_ref<!-- {{#callable:quantize_row_iq3_s_ref}} -->
The `quantize_row_iq3_s_ref` function quantizes a row of floating-point values into a specified format using a helper function.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that represent the data to be quantized.
    - `y`: A pointer to a `block_iq3_s` structure where the quantized output will be stored.
    - `k`: An integer that specifies the number of elements to quantize, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that `k` is a multiple of `QK_K` to ensure valid input.
    - If the assertion passes, it calls the [`quantize_iq3_s`](#quantize_iq3_s) function with the input array `x`, output structure `y`, a fixed value of 1, and the integer `k`.
- **Output**: The function does not return a value; instead, it modifies the `block_iq3_s` structure pointed to by `y` to contain the quantized representation of the input data.
- **Functions called**:
    - [`quantize_iq3_s`](#quantize_iq3_s)


---
### iq1\_find\_best\_neighbour<!-- {{#callable:iq1_find_best_neighbour}} -->
Finds the best neighbor grid point based on weighted scores calculated from neighbor values.
- **Inputs**:
    - `neighbours`: An array of indices representing neighboring grid points, where the first element indicates the number of neighbors.
    - `grid`: A pointer to the grid data, which contains values for each grid point.
    - `xval`: An array of float values used in the score calculation.
    - `weight`: An array of float weights corresponding to the `xval` elements.
    - `scale`: A pointer to a float where the calculated scale factor will be stored.
    - `L`: A pointer to an array of int8_t where the resulting best neighbor values will be stored.
    - `ngrid`: The total number of grid points available.
- **Control Flow**:
    - The function starts by asserting that the number of neighbors is greater than zero.
    - It initializes variables to track the best score and the corresponding grid index.
    - It iterates over the neighbors to calculate a weighted score based on their values and the provided weights.
    - If a better score is found, it updates the best score and the scale factor.
    - If no suitable neighbor is found, it iterates over all grid points to find the best score again.
    - If still no grid point is found, it prints diagnostic information about the neighbors.
    - Finally, it applies a fudge factor to the scale and stores the best neighbor's values in the output array.
- **Output**: Returns the index of the best neighbor grid point found, ensuring that a valid index is always returned.


---
### iq1\_find\_best\_neighbour2<!-- {{#callable:iq1_find_best_neighbour2}} -->
Finds the best neighbor grid index based on a weighted distance metric.
- **Inputs**:
    - `neighbours`: An array of indices representing neighboring grid points.
    - `grid`: A pointer to the grid data containing the values for each grid point.
    - `xval`: An array of target values to compare against.
    - `weight`: An array of weights used in the distance calculation.
    - `scale`: A scaling factor applied to the grid values.
    - `xg`: An array of grid values used for comparison.
    - `L`: An output array where the best neighbor indices will be stored.
    - `ngrid`: The total number of grid points available.
- **Control Flow**:
    - The function starts by asserting that the number of neighbors is greater than zero.
    - It initializes the best score to a maximum float value and sets the grid index to -1.
    - It iterates over the neighbors to calculate a weighted squared distance (d2) for each neighbor.
    - If a neighbor's distance is less than the current best score, it updates the best score and the grid index.
    - If no suitable neighbor is found, it iterates over all grid points to find the best match using the same distance calculation.
    - If still no match is found, it prints diagnostic information about the neighbors and their computed values.
    - Finally, it asserts that a valid grid index was found and populates the output array L with the best neighbor's indices.
- **Output**: Returns the index of the best neighbor grid point.


---
### iq1\_sort\_helper<!-- {{#callable:iq1_sort_helper}} -->
Compares two floating-point numbers for sorting.
- **Inputs**:
    - `left`: A pointer to the first floating-point number to compare.
    - `right`: A pointer to the second floating-point number to compare.
- **Control Flow**:
    - Cast the `left` and `right` pointers to `float` pointers.
    - Compare the values pointed to by `l` and `r` using conditional operators.
    - Return -1 if the left value is less than the right value, 1 if greater, or 0 if they are equal.
- **Output**: Returns an integer indicating the relative order of the two floating-point numbers: -1 if the first is less, 1 if greater, and 0 if they are equal.


---
### quantize\_row\_iq1\_s\_impl<!-- {{#callable:quantize_row_iq1_s_impl}} -->
The `quantize_row_iq1_s_impl` function performs quantization of a row of floating-point data using specified quantization weights and outputs the quantized representation.
- **Inputs**:
    - `x`: A pointer to an array of floats representing the input data to be quantized.
    - `vy`: A pointer to a block structure where the quantized output will be stored.
    - `n`: An integer representing the number of elements in the input array.
    - `quant_weights`: A pointer to an array of floats representing the quantization weights.
    - `scales`: A pointer to an array where the computed scales for quantization will be stored.
    - `weight`: A pointer to an array used to store intermediate weight calculations.
    - `sumx`: A pointer to an array used to accumulate sums of input values.
    - `sumw`: A pointer to an array used to accumulate sums of weights.
    - `pairs`: A pointer to an array used for sorting and storing pairs of values during quantization.
    - `L`: A pointer to an array of int8_t used to store quantization levels.
    - `index`: A pointer to an array of uint16_t used to store indices of quantized values.
    - `shifts`: A pointer to an array of int8_t used to store shift values for quantization.
- **Control Flow**:
    - The function begins by asserting the validity of input parameters and initializing necessary variables.
    - It calculates the number of blocks based on the input size and iterates over each block.
    - For each block, it computes the sum of squares and determines the maximum scale.
    - It then processes each block by calculating weights and performing an exhaustive search for optimal quantization boundaries.
    - The function sorts the weights and computes cumulative sums to facilitate the search for the best quantization scale.
    - It checks if the quantized values are on the grid and adjusts them if necessary, storing results in the output structure.
    - Finally, it computes the final scale and shift values for each block and stores them in the output.
- **Output**: The function outputs a quantized representation of the input data in the structure pointed to by `vy`, along with scales and shifts for each block.
- **Functions called**:
    - [`iq2_data_index`](#iq2_data_index)
    - [`iq1_find_best_neighbour2`](#iq1_find_best_neighbour2)
    - [`nearest_int`](#nearest_int)


---
### quantize\_iq1\_s<!-- {{#callable:quantize_iq1_s}} -->
The `quantize_iq1_s` function quantizes input data into a specified format using given weights and stores the result in a destination buffer.
- **Inputs**:
    - `src`: A pointer to the source array of floats that contains the data to be quantized.
    - `dst`: A pointer to the destination buffer where the quantized data will be stored.
    - `nrow`: The number of rows of data to process.
    - `n_per_row`: The number of elements in each row, which must be a multiple of `QK_K`.
    - `quant_weights`: A pointer to an array of floats representing the quantization weights.
- **Control Flow**:
    - The function begins by asserting that `n_per_row` is a multiple of `QK_K` to ensure valid processing.
    - It initializes several arrays for scales, weights, and other intermediate calculations.
    - The number of blocks is calculated by dividing `n_per_row` by `QK_K`.
    - A loop iterates over each row of the input data, calling the [`quantize_row_iq1_s_impl`](#quantize_row_iq1_s_impl) function to perform the quantization for each row.
    - After processing each row, the source pointer is incremented by `n_per_row`, and the destination pointer is incremented by the size of the block.
- **Output**: The function returns the total size in bytes of the quantized data stored in the destination buffer, calculated as the product of the number of rows and the number of blocks multiplied by the size of each block.
- **Functions called**:
    - [`quantize_row_iq1_s_impl`](#quantize_row_iq1_s_impl)


---
### quantize\_row\_iq1\_m\_impl<!-- {{#callable:quantize_row_iq1_m_impl}} -->
The `quantize_row_iq1_m_impl` function performs quantization of a row of floating-point data using specified weights and scales, producing a compressed representation.
- **Inputs**:
    - `x`: A pointer to an array of floats representing the input data to be quantized.
    - `vy`: A pointer to a block of memory where the quantized output will be stored.
    - `n`: An integer representing the number of elements in the input array `x`.
    - `quant_weights`: A pointer to an array of floats representing the quantization weights, which can be NULL.
    - `scales`: A pointer to an array of floats where the computed scales for quantization will be stored.
    - `weight`: A pointer to an array of floats used to store computed weights for quantization.
    - `pairs`: A pointer to an array of floats used for temporary storage of pairs during quantization.
    - `L`: A pointer to an array of int8_t used to store quantization levels.
    - `index`: A pointer to an array of uint16_t used to store indices of quantized values.
    - `shifts`: A pointer to an array of int8_t used to store shift values for quantization.
- **Control Flow**:
    - The function begins by asserting the validity of input parameters and initializing necessary variables.
    - It calculates the number of blocks based on the input size and iterates over each block.
    - For each block, it computes the sum of squares and determines the weights based on the provided quantization weights.
    - The function then performs an exhaustive search to find the optimal quantization boundaries and scales for the block.
    - It checks if the quantized values are on the grid and adjusts them if necessary, storing the results in the output structure.
    - Finally, it computes the final scales and shifts for the quantized data and stores them in the output.
- **Output**: The function outputs a quantized representation of the input data in the specified output structure, along with scales and shifts used for quantization.
- **Functions called**:
    - [`iq2_data_index`](#iq2_data_index)
    - [`iq1_find_best_neighbour2`](#iq1_find_best_neighbour2)
    - [`nearest_int`](#nearest_int)


---
### quantize\_iq1\_m<!-- {{#callable:quantize_iq1_m}} -->
The `quantize_iq1_m` function quantizes a source array of floats into a destination buffer using specified quantization weights and parameters.
- **Inputs**:
    - `src`: A pointer to the source array of floats that will be quantized.
    - `dst`: A pointer to the destination buffer where the quantized data will be stored.
    - `nrow`: The number of rows to process in the source array.
    - `n_per_row`: The number of elements per row in the source array, which must be a multiple of `QK_K`.
    - `quant_weights`: A pointer to an array of quantization weights used during the quantization process.
- **Control Flow**:
    - The function begins by asserting that `n_per_row` is a multiple of `QK_K` to ensure valid processing.
    - It initializes several arrays for scales, weights, and other quantization parameters.
    - The number of blocks is calculated by dividing `n_per_row` by `QK_K`.
    - A loop iterates over each row of the source array, calling the [`quantize_row_iq1_m_impl`](#quantize_row_iq1_m_impl) function to perform the actual quantization for each row.
    - After processing each row, the source pointer is incremented by `n_per_row`, and the destination pointer is incremented by the size of the block.
- **Output**: The function returns the total size in bytes of the quantized data stored in the destination buffer, calculated as the product of the number of rows, the number of blocks, and the size of each block.
- **Functions called**:
    - [`quantize_row_iq1_m_impl`](#quantize_row_iq1_m_impl)


---
### best\_index\_int8<!-- {{#callable:best_index_int8}} -->
Finds the index of the closest value to `x` in a sorted array of `int8_t` values.
- **Inputs**:
    - `n`: The number of elements in the array `val`.
    - `val`: A pointer to a sorted array of `int8_t` values.
    - `x`: A floating-point value to compare against the elements of `val`.
- **Control Flow**:
    - Checks if `x` is less than or equal to the first element of `val`, returning index 0 if true.
    - Checks if `x` is greater than or equal to the last element of `val`, returning index n-1 if true.
    - Initializes two indices, `ml` (lower bound) and `mu` (upper bound), for binary search.
    - Enters a while loop that continues until the difference between `mu` and `ml` is greater than 1.
    - Calculates the midpoint index `mav` and compares `x` with the value at `val[mav]` to adjust the bounds.
    - After exiting the loop, compares the distances from `x` to the values at `val[mu-1]` and `val[mu]` to determine the closest index.
- **Output**: Returns the index of the closest value in the array `val` to the input value `x`.


---
### quantize\_row\_iq4\_nl\_impl<!-- {{#callable:quantize_row_iq4_nl_impl}} -->
The `quantize_row_iq4_nl_impl` function quantizes a row of floating-point values into a compressed format using a specified quantization scheme.
- **Inputs**:
    - `super_block_size`: An integer representing the size of the super block to be processed.
    - `block_size`: An integer representing the size of each block within the super block.
    - `x`: A pointer to an array of floats representing the input data to be quantized.
    - `dh`: A pointer to an array of `ggml_fp16_t` where the computed scale factors will be stored.
    - `q4`: A pointer to an array of uint8_t where the quantized output will be stored.
    - `scales_h`: A pointer to an array of uint16_t for storing high bits of scale factors.
    - `scales_l`: A pointer to an array of uint8_t for storing low bits of scale factors.
    - `scales`: A pointer to an array of floats for storing the computed scale factors.
    - `weight`: A pointer to an array of floats for storing weights used in quantization.
    - `L`: A pointer to an array of uint8_t for storing the quantized indices.
    - `values`: A pointer to an array of int8_t representing the quantization values.
    - `quant_weights`: A pointer to an array of floats representing the quantization weights, or NULL if not used.
    - `ntry`: An integer indicating the number of attempts to refine the quantization.
- **Control Flow**:
    - The function begins by calculating the variance of the input data `x` and initializes the output arrays.
    - It iterates over blocks of the input data, calculating weights based on the presence of `quant_weights`.
    - For each block, it finds the maximum absolute value and computes a scaling factor based on the quantization values.
    - The function attempts to refine the scaling factor through multiple iterations if `ntry` is greater than zero.
    - After processing all blocks, it handles the case where multiple blocks exist by packing the scales into high and low bits.
    - Finally, it constructs the quantized output by combining the quantized indices into the `q4` array.
- **Output**: The function does not return a value but modifies the output arrays `dh`, `q4`, `scales_h`, `scales_l`, and `L` in place to store the quantized representation and associated scale factors.
- **Functions called**:
    - [`best_index_int8`](#best_index_int8)
    - [`nearest_int`](#nearest_int)


---
### quantize\_iq4\_nl<!-- {{#callable:quantize_iq4_nl}} -->
The `quantize_iq4_nl` function quantizes input floating-point data into a compressed format using specified quantization weights.
- **Inputs**:
    - `src`: A pointer to the source array of floating-point values to be quantized.
    - `dst`: A pointer to the destination memory where the quantized data will be stored.
    - `nrow`: The number of rows of data to process.
    - `n_per_row`: The number of elements in each row, which must be a multiple of `QK4_NL`.
    - `quant_weights`: A pointer to an array of quantization weights, or NULL if no weights are used.
- **Control Flow**:
    - The function asserts that `n_per_row` is a multiple of `QK4_NL` to ensure valid processing.
    - It calculates the number of blocks (`nblock`) by dividing `n_per_row` by `QK4_NL`.
    - A loop iterates over each row of the input data, processing `nrow` times.
    - Within the row loop, another loop iterates over each block, calling [`quantize_row_iq4_nl_impl`](#quantize_row_iq4_nl_impl) to perform the quantization for each block.
    - The source pointer `src` is incremented by `n_per_row` after processing each row, and the destination pointer `qrow` is incremented by the size of the quantized block.
- **Output**: The function returns the total size in bytes of the quantized data, calculated as `nrow * nblock * sizeof(block_iq4_nl)`.
- **Functions called**:
    - [`quantize_row_iq4_nl_impl`](#quantize_row_iq4_nl_impl)


---
### quantize\_row\_iq4\_nl\_ref<!-- {{#callable:quantize_row_iq4_nl_ref}} -->
The `quantize_row_iq4_nl_ref` function quantizes a row of floating-point values into a specified block format.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values that are to be quantized.
    - `y`: A pointer to a `block_iq4_nl` structure where the quantized output will be stored.
    - `k`: An integer representing the total number of elements in the input array, which must be a multiple of `QK4_NL`.
- **Control Flow**:
    - The function asserts that `k` is a multiple of `QK4_NL` to ensure valid input.
    - It calculates the number of blocks (`nblock`) by dividing `k` by `QK4_NL`.
    - An array `L` and a `weight` array are initialized to hold quantization parameters.
    - A loop iterates over each block, calling the [`quantize_row_iq4_nl_impl`](#quantize_row_iq4_nl_impl) function to perform the actual quantization for each segment of the input array.
- **Output**: The function does not return a value; instead, it modifies the `block_iq4_nl` structure pointed to by `y` to store the quantized data.
- **Functions called**:
    - [`quantize_row_iq4_nl_impl`](#quantize_row_iq4_nl_impl)


---
### quantize\_iq4\_xs<!-- {{#callable:quantize_iq4_xs}} -->
The `quantize_iq4_xs` function quantizes input floating-point data into a compressed format suitable for efficient storage.
- **Inputs**:
    - `src`: A pointer to the source array of floating-point values to be quantized.
    - `dst`: A pointer to the destination buffer where the quantized data will be stored.
    - `nrow`: The number of rows of data to be processed.
    - `n_per_row`: The number of elements in each row, which must be a multiple of `QK_K`.
    - `quant_weights`: A pointer to an array of quantization weights, or NULL if not used.
- **Control Flow**:
    - The function asserts that `n_per_row` is a multiple of `QK_K` to ensure valid processing.
    - It calculates the number of blocks (`nblock`) by dividing `n_per_row` by `QK_K`.
    - A loop iterates over each row of the input data, processing `nrow` times.
    - Within the row loop, another loop iterates over each block, calling [`quantize_row_iq4_nl_impl`](#quantize_row_iq4_nl_impl) to perform the actual quantization for each block.
    - The source pointer is incremented by `n_per_row` after processing each row, and the destination pointer is incremented by the size of the quantized block.
- **Output**: The function returns the total size in bytes of the quantized data produced, calculated as `nrow * nblock * sizeof(block_iq4_xs)`.
- **Functions called**:
    - [`quantize_row_iq4_nl_impl`](#quantize_row_iq4_nl_impl)


---
### quantize\_row\_iq4\_xs\_ref<!-- {{#callable:quantize_row_iq4_xs_ref}} -->
The `quantize_row_iq4_xs_ref` function quantizes a row of floating-point values into a specified format.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values to be quantized.
    - `y`: A pointer to a `block_iq4_xs` structure where the quantized output will be stored.
    - `k`: An integer representing the size of the input data, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that the value of `k` is a multiple of `QK_K` to ensure valid input size.
    - If the assertion passes, it calls the [`quantize_iq4_xs`](#quantize_iq4_xs) function, passing the input array `x`, the output structure `y`, a constant value of 1, the size `k`, and a NULL pointer for additional parameters.
- **Output**: The function does not return a value; instead, it modifies the `block_iq4_xs` structure pointed to by `y` to contain the quantized representation of the input data.
- **Functions called**:
    - [`quantize_iq4_xs`](#quantize_iq4_xs)


---
### quantize\_row\_iq2\_s\_impl<!-- {{#callable:quantize_row_iq2_s_impl}} -->
The `quantize_row_iq2_s_impl` function quantizes a row of floating-point values into a specific format using predefined quantization weights and scales.
- **Inputs**:
    - `x`: A pointer to an array of float values representing the input data to be quantized.
    - `vy`: A pointer to a block structure where the quantized output will be stored.
    - `n`: An integer representing the number of elements in the input array, which must be a multiple of QK_K.
    - `quant_weights`: A pointer to an array of float values representing the quantization weights, or NULL if default weights should be used.
- **Control Flow**:
    - The function begins by asserting the validity of input parameters and initializing necessary variables.
    - It calculates the number of blocks based on the input size and iterates over each block.
    - For each block, it computes the sum of squares of the input values to derive a variance estimate.
    - It then calculates weights based on the quantization weights or defaults to a fixed formula if none are provided.
    - The function checks if the maximum value in the block is below a threshold, skipping quantization if so.
    - It attempts to find the best quantization scale by iterating over potential scales and checking if they yield valid grid indices.
    - If any values are not on the grid, it attempts to find the best neighboring quantization points.
    - Finally, it stores the quantized values and scales in the output structure.
- **Output**: The function outputs a quantized representation of the input data in the `vy` structure, including quantized values and scales, while ensuring that the quantization adheres to specified constraints.
- **Functions called**:
    - [`iq2_data_index`](#iq2_data_index)
    - [`nearest_int`](#nearest_int)
    - [`iq2_find_best_neighbour`](#iq2_find_best_neighbour)


---
### quantize\_iq2\_s<!-- {{#callable:quantize_iq2_s}} -->
The `quantize_iq2_s` function quantizes input data from a source array into a destination buffer using specified quantization weights.
- **Inputs**:
    - `src`: A pointer to the source array of float values that will be quantized.
    - `dst`: A pointer to the destination buffer where the quantized data will be stored.
    - `nrow`: The number of rows to process in the source array.
    - `n_per_row`: The number of elements in each row of the source array, which must be a multiple of `QK_K`.
    - `quant_weights`: A pointer to an array of float values used as weights for the quantization process.
- **Control Flow**:
    - The function asserts that `n_per_row` is a multiple of `QK_K` to ensure valid quantization.
    - It calculates the number of blocks (`nblock`) by dividing `n_per_row` by `QK_K`.
    - A loop iterates over each row of the source array, calling [`quantize_row_iq2_s_impl`](#quantize_row_iq2_s_impl) to perform the quantization for each row.
    - After processing each row, the source pointer is incremented by `n_per_row`, and the destination pointer is incremented by the size of the block.
- **Output**: The function returns the total size in bytes of the quantized data stored in the destination buffer, calculated as the product of the number of rows, the number of blocks, and the size of each block.
- **Functions called**:
    - [`quantize_row_iq2_s_impl`](#quantize_row_iq2_s_impl)


---
### quantize\_row\_iq2\_s\_ref<!-- {{#callable:quantize_row_iq2_s_ref}} -->
The `quantize_row_iq2_s_ref` function quantizes a row of floating-point values into a specified format.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values to be quantized.
    - `y`: A pointer to a `block_iq2_s` structure where the quantized output will be stored.
    - `k`: An integer representing the number of elements to quantize, which must be a multiple of `QK_K`.
- **Control Flow**:
    - The function begins by asserting that the value of `k` is a multiple of `QK_K` to ensure valid input.
    - If the assertion passes, it calls the [`quantize_iq2_s`](#quantize_iq2_s) function, passing the input array `x`, the output structure `y`, a constant value of 1, the integer `k`, and a NULL pointer for additional parameters.
- **Output**: The function does not return a value; instead, it modifies the `block_iq2_s` structure pointed to by `y` to contain the quantized representation of the input data.
- **Functions called**:
    - [`quantize_iq2_s`](#quantize_iq2_s)


---
### validate\_float<!-- {{#callable:validate_float}} -->
Validates whether a given float is neither infinite nor NaN.
- **Inputs**:
    - `f`: The float value to be validated.
    - `i`: The index of the block where the float is located, used for error reporting.
- **Control Flow**:
    - Checks if the float `f` is infinite using `isinf(f)`; if true, logs an error message and returns false.
    - Checks if the float `f` is NaN using `isnan(f)`; if true, logs an error message and returns false.
    - If neither condition is met, returns true, indicating the float is valid.
- **Output**: Returns a boolean value: true if the float is valid, false if it is infinite or NaN.


---
### isinf\_fp16<!-- {{#callable:isinf_fp16}} -->
Determines if a given half-precision floating-point number is infinite.
- **Inputs**:
    - `f`: A half-precision floating-point number represented as `ggml_fp16_t`.
- **Control Flow**:
    - The function checks if the exponent bits of the half-precision number are all set to 1 (indicating infinity).
    - It also checks if the fraction bits are all zero, confirming that it is indeed an infinite value.
- **Output**: Returns a boolean value: true if the input represents positive or negative infinity, false otherwise.


---
### isnan\_fp16<!-- {{#callable:isnan_fp16}} -->
Determines if a half-precision floating-point number is NaN (Not a Number).
- **Inputs**:
    - `f`: A half-precision floating-point number represented as `ggml_fp16_t`.
- **Control Flow**:
    - The function checks if the exponent bits of the half-precision float `f` are all set to 1 (indicating a special value).
    - It then checks if the fraction bits are not all zero, which confirms that the value is NaN.
- **Output**: Returns `true` if `f` is NaN, otherwise returns `false`.


---
### validate\_fp16<!-- {{#callable:validate_fp16}} -->
The `validate_fp16` function checks if a given half-precision floating-point number is neither infinite nor NaN.
- **Inputs**:
    - `f`: A half-precision floating-point number of type `ggml_fp16_t` to be validated.
    - `i`: A size_t index representing the block number where the value is located.
- **Control Flow**:
    - The function first checks if the input `f` is infinite using the [`isinf_fp16`](#isinf_fp16) function.
    - If `f` is infinite, it logs an error message to `stderr` indicating the block index and returns false.
    - Next, it checks if `f` is NaN using the [`isnan_fp16`](#isnan_fp16) function.
    - If `f` is NaN, it logs a different error message to `stderr` and returns false.
    - If neither condition is met, the function returns true, indicating that the value is valid.
- **Output**: The function returns a boolean value: true if the input is valid (not infinite or NaN), and false otherwise.
- **Functions called**:
    - [`isinf_fp16`](#isinf_fp16)
    - [`isnan_fp16`](#isnan_fp16)


---
### ggml\_validate\_row\_data<!-- {{#callable:ggml_validate_row_data}} -->
Validates the integrity of row data based on its type and size.
- **Inputs**:
    - `type`: An enumeration value representing the data type to validate.
    - `data`: A pointer to the data to be validated.
    - `nbytes`: The size in bytes of the data to validate.
- **Control Flow**:
    - Checks if the `type` is within valid bounds; if not, logs an error and returns false.
    - Validates that `nbytes` is a multiple of the size of the specified `type`; if not, logs an error and returns false.
    - Calculates the number of elements (`nb`) based on `nbytes` and the size of the type.
    - Uses a switch statement to handle validation for different data types, performing specific checks for each type.
    - For types like `GGML_TYPE_BF16`, `GGML_TYPE_F16`, and `GGML_TYPE_F32`, it checks for NaN and infinity values using SIMD instructions if available.
    - For other types, it calls specific validation macros or functions to perform the necessary checks.
    - If all checks pass, the function returns true.
- **Output**: Returns true if the data is valid according to the specified type and size; otherwise, returns false.
- **Functions called**:
    - [`ggml_type_size`](ggml.c.driver.md#ggml_type_size)
    - [`ggml_type_name`](ggml.c.driver.md#ggml_type_name)
    - [`validate_fp16`](#validate_fp16)
    - [`validate_float`](#validate_float)


