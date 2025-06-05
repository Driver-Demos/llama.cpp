# Purpose
The provided code is a comprehensive suite of Metal Shading Language (MSL) kernels crafted for high-performance computing on Apple devices using the Metal API, with a focus on machine learning and neural network computations. Its primary purpose is to execute a variety of mathematical and data manipulation operations, particularly emphasizing tensor operations, quantization, and dequantization, which are crucial for optimizing neural network layers. The code is organized into specialized kernel functions that handle operations like matrix-matrix and matrix-vector multiplication, pooling, and data type conversions, all optimized for GPU execution. It supports various quantized data types, such as q4, q5, and q8, to minimize memory usage and computational demands. Designed as a library with public APIs, these kernels can be integrated into larger machine learning frameworks, providing efficient, parallelized processing capabilities for neural network computations on Apple's Metal framework.
# Global Variables

---
### kvalues\_iq4nl\_f
- **Type**: `float[16]`
- **Description**: `kvalues_iq4nl_f` is a constant array of 16 floating-point values defined in the global scope. The array contains a sequence of pre-defined float values ranging from -127.0 to 113.0.
- **Use**: This array is used in the dequantization process for `iq4_nl` blocks, where each element in the array represents a specific quantization level.


# Data Structures

---
### bfloat4x4
- **Type**: `typedef`
- **Members**:
    - `bfloat`: Represents a 16-bit floating-point number in the bfloat16 format.
    - `4`: Specifies the number of rows in the matrix.
    - `4`: Specifies the number of columns in the matrix.
- **Description**: The `bfloat4x4` is a matrix data structure defined using the Metal shading language, specifically for use in GPU programming. It is a 4x4 matrix where each element is of type `bfloat`, a 16-bit floating-point number in the bfloat16 format. This matrix is used in scenarios where reduced precision is acceptable, such as in certain graphics and machine learning computations, to save memory and improve performance on compatible hardware.


---
### ggml\_sort\_order
- **Type**: `enum`
- **Members**:
    - `GGML_SORT_ORDER_ASC`: Represents ascending sort order.
    - `GGML_SORT_ORDER_DESC`: Represents descending sort order.
- **Description**: The `ggml_sort_order` is an enumeration that defines two possible sorting orders: ascending and descending. It is used to specify the order in which elements should be sorted, typically in sorting algorithms or functions that require a sort order parameter. The enumeration provides a clear and concise way to indicate the desired sorting direction.


---
### block\_q5\_0
- **Type**: `struct`
- **Members**:
    - `d`: Represents a scaling factor for the quantized values.
    - `qs`: An array of quantized values, each represented as a 4-bit integer.
    - `qh`: An array of high bits for the quantized values, used for additional precision.
- **Description**: The `block_q5_0` structure is designed to store quantized data in a compact form, using a combination of 4-bit quantized values and additional high bits for increased precision. The structure includes a scaling factor `d` to convert the quantized values back to their original floating-point representation. This data structure is typically used in scenarios where memory efficiency is crucial, such as in machine learning models deployed on resource-constrained devices.


---
### block\_q5\_1
- **Type**: `struct`
- **Members**:
    - `d`: Represents the quantization delta value for the block.
    - `m`: Stores the minimum value in the block for quantization.
    - `qs`: An array of quantized values for the block.
    - `qh`: An array of high bits for quantization, used to store additional precision.
- **Description**: The `block_q5_1` structure is designed to store quantized data in a compact form, specifically for a quantization scheme that uses 5 bits per value. It includes fields for the quantization delta (`d`) and the minimum value (`m`) to allow for dequantization. The `qs` array holds the quantized values, while the `qh` array stores additional high bits to increase precision. This structure is used in scenarios where memory efficiency is crucial, such as in GPU kernels for machine learning tasks.


---
### block\_iq4\_nl
- **Type**: `struct`
- **Members**:
    - `d`: Represents a scaling factor for the quantized values.
    - `qs`: An array of quantized values, each represented as a 4-bit integer.
- **Description**: The `block_iq4_nl` structure is designed to store quantized data in a compact form, using 4-bit quantization for each value. It includes a scaling factor `d` to adjust the quantized values back to their original range. This structure is used in the context of neural network operations, where efficient storage and computation of quantized data are crucial for performance.


---
### block\_q4\_0
- **Type**: `struct`
- **Members**:
    - `d`: Represents a scaling factor for the quantized values.
    - `qs`: An array of quantized values, each represented as a 4-bit integer.
    - `qh`: An array of high bits for the quantized values, used for additional precision.
- **Description**: The `block_q5_0` structure is designed to store quantized data in a compact form, using 4-bit integers for each quantized value. It includes a scaling factor `d` to convert these quantized values back to their original floating-point representation. The `qs` array holds the main quantized values, while the `qh` array stores additional high bits to enhance precision. This structure is typically used in scenarios where memory efficiency is crucial, such as in neural network computations on GPUs.


---
### block\_q4\_1
- **Type**: `struct`
- **Members**:
    - `d`: Represents a scaling factor for the quantized values.
    - `qs`: An array of quantized values, each represented as a 4-bit integer.
    - `qh`: An array of high bits for the quantized values, used for additional precision.
- **Description**: The `block_q4_1` structure is designed to store quantized data in a compact form, using 4-bit quantization for each value. It includes a scaling factor `d` to convert the quantized values back to their original floating-point representation. The `qs` array holds the quantized values, while the `qh` array provides additional bits for higher precision, allowing for more accurate reconstruction of the original data. This structure is typically used in scenarios where memory efficiency is crucial, such as in neural network inference on resource-constrained devices.


---
### block\_q8\_0
- **Type**: `struct`
- **Members**:
    - `d`: Represents a scaling factor for the quantized values.
    - `qs`: An array of quantized values, each represented as a 4-bit integer.
    - `qh`: An array of high bits for the quantized values, used for additional precision.
- **Description**: The `block_q8_0` structure is designed to store quantized data in a compact form, using 4-bit integers for each quantized value. It includes a scaling factor `d` to convert these quantized values back to their original floating-point representation. The `qs` array holds the main quantized values, while the `qh` array provides additional high bits to enhance precision. This structure is typically used in scenarios where memory efficiency is crucial, such as in neural network computations on devices with limited resources.


---
### block\_q2\_K
- **Type**: `struct`
- **Members**:
    - `scales`: An array of uint8_t representing the scales for quantization.
    - `qs`: An array of uint16_t representing the quantized values.
    - `d`: A half-precision floating-point value representing the scale factor.
- **Description**: The `block_q2_K` structure is designed to store quantized data for efficient computation, particularly in the context of matrix-vector or matrix-matrix operations. It contains quantized values (`qs`) and associated scales (`scales`) to allow for dequantization during computation. The `d` member is a scale factor used to adjust the quantized values back to their original range. This structure is typically used in scenarios where memory efficiency and computational speed are critical, such as in neural network inference on specialized hardware.


---
### block\_q3\_K
- **Type**: `struct`
- **Members**:
    - `d`: Represents a scaling factor for the quantized values.
    - `qs`: An array of quantized values, each represented as a 4-bit integer.
    - `qh`: An array of high bits used for additional precision in quantization.
- **Description**: The `block_q3_K` structure is designed to store quantized data in a compact form, using a combination of 4-bit quantized values and additional high bits for precision. The `d` member is a scaling factor that is used to dequantize the stored values back to their original floating-point representation. The `qs` array holds the main quantized values, while the `qh` array provides extra bits to enhance the precision of these quantized values. This structure is typically used in scenarios where memory efficiency is crucial, such as in neural network inference on resource-constrained devices.


---
### block\_q4\_K
- **Type**: `struct`
- **Members**:
    - `d`: Represents a scaling factor for the quantized values.
    - `qs`: An array of quantized values, each represented as a 4-bit integer.
- **Description**: The `block_q4_K` structure is designed to store quantized data in a compact form, using 4-bit integers to represent each quantized value. This structure includes a scaling factor `d` to adjust the quantized values back to their original range when needed. The `qs` array holds the quantized values, allowing efficient storage and processing of data in applications where memory usage is a concern.


---
### block\_q5\_K
- **Type**: `struct`
- **Members**:
    - `d`: Represents a scaling factor for the quantized values.
    - `qs`: An array of quantized values, each represented as a 4-bit integer.
    - `qh`: An array of high bits for the quantized values, used for additional precision.
- **Description**: The `block_q5_K` structure is designed to store quantized data in a compact form, using a combination of 4-bit quantized values and additional high bits for precision. It includes a scaling factor `d` to convert the quantized values back to their original floating-point representation. This structure is typically used in scenarios where memory efficiency is crucial, such as in neural network computations on resource-constrained devices.


---
### block\_q6\_K
- **Type**: `struct`
- **Members**:
    - `d`: Represents a scaling factor for the quantized values.
    - `qs`: An array of quantized values, each represented as a 4-bit integer.
    - `qh`: An array of high bits used for additional precision in quantization.
- **Description**: The `block_q6_K` structure is designed to store quantized data in a compact form, using a combination of 4-bit quantized values and additional high bits for increased precision. The structure includes a scaling factor `d` to convert the quantized values back to their original floating-point representation. This data structure is typically used in scenarios where memory efficiency is crucial, such as in neural network models where large amounts of data need to be processed efficiently.


---
### block\_iq2\_xxs
- **Type**: `struct`
- **Members**:
    - `qs`: An array of quantized values.
    - `d`: A float representing a scaling factor.
- **Description**: The `block_iq2_xxs` structure is designed to store quantized data with a scaling factor. It contains an array `qs` for storing quantized values and a float `d` that acts as a scaling factor to adjust the quantized values to their original range. This structure is typically used in scenarios where data needs to be compressed and efficiently stored, such as in machine learning models or graphics processing.


---
### block\_iq2\_xs
- **Type**: `struct`
- **Members**:
    - `qs`: An array of uint16_t values representing quantized data.
    - `scales`: An array of uint8_t values representing scaling factors.
    - `d`: A half-precision floating-point value representing a scaling factor.
- **Description**: The `block_iq2_xs` structure is designed to store quantized data along with scaling factors for efficient computation. It contains an array of quantized values (`qs`), an array of scaling factors (`scales`), and a half-precision floating-point value (`d`) that serves as an additional scaling factor. This structure is used in operations that involve quantized data processing, allowing for compact storage and efficient computation by leveraging the scaling factors to adjust the quantized values during processing.


---
### block\_iq3\_xxs
- **Type**: `struct`
- **Members**:
    - `qs`: An array of 8-bit unsigned integers representing quantized values.
    - `d`: A half-precision floating-point value representing a scaling factor.
- **Description**: The `block_iq3_xxs` structure is designed to store quantized data in a compact form, utilizing an array of 8-bit unsigned integers (`qs`) to hold the quantized values and a half-precision floating-point number (`d`) to represent a scaling factor. This structure is used in the context of quantized matrix operations, where it facilitates efficient storage and computation by reducing the precision of the data while maintaining a scaling factor to approximate the original values.


---
### block\_iq3\_s
- **Type**: `struct`
- **Members**:
    - `d`: Represents a scaling factor for the quantized values.
    - `qs`: An array of quantized values.
- **Description**: The `block_iq3_s` structure is designed to store quantized data along with a scaling factor. The `d` member is a float that acts as a scaling factor for the quantized values stored in the `qs` array. This structure is used in the context of quantization and dequantization processes, where the quantized values are stored in a compact form and can be scaled back to their original range using the scaling factor.


---
### block\_iq2\_s
- **Type**: `struct`
- **Members**:
    - `d`: Represents a scaling factor for the block.
    - `qs`: An array of quantized values.
- **Description**: The `block_iq2_s` structure is designed to store quantized data with a scaling factor. It contains a floating-point member `d` that acts as a scaling factor, and an array `qs` that holds quantized values. This structure is typically used in scenarios where data needs to be stored in a compressed format, with the scaling factor allowing for the reconstruction of the original data values.


---
### block\_iq1\_s
- **Type**: `struct`
- **Members**:
    - `d`: Represents a scaling factor for the quantized values.
    - `qs`: An array of quantized values, each represented as a 4-bit integer.
    - `qh`: An array of high bits for the quantized values, used for additional precision.
- **Description**: The `block_iq1_s` structure is designed to store quantized data in a compact form, using a combination of 4-bit quantized values and additional high bits for precision. It includes a scaling factor `d` to convert the quantized values back to their original floating-point representation. This structure is typically used in scenarios where memory efficiency is crucial, such as in GPU kernels for machine learning applications.


---
### block\_iq1\_m
- **Type**: `struct`
- **Members**:
    - `qs`: An array of quantized values.
    - `qh`: An array of high bits for quantized values.
    - `scales`: An array of scales for quantized values.
- **Description**: The `block_iq1_m` structure is used to store quantized data with associated high bits and scales. It is designed to efficiently represent and process quantized values in a compact form, which is useful in scenarios where memory and computational efficiency are critical, such as in machine learning models or signal processing applications.


---
### block\_iq4\_xs
- **Type**: `struct`
- **Members**:
    - `qs`: An array of quantized values for the data block.
    - `d`: A float representing the scaling factor for the quantized values.
- **Description**: The `block_iq4_xs` structure is designed to store quantized data in a compact form, using an array of quantized values (`qs`) and a scaling factor (`d`) to represent the original data. This structure is typically used in scenarios where memory efficiency is crucial, such as in machine learning models or data compression algorithms, where the quantized values can be used to approximate the original data with reduced precision.


# Functions

---
### kernel\_add\_row
The `kernel_add_row` function performs element-wise addition of a source row tensor to a destination tensor, effectively broadcasting the source row across the destination tensor.
- **Inputs**:
    - `args`: A constant structure containing metadata and parameters for the operation, including tensor dimensions.
    - `src0`: A pointer to the first input tensor (the destination tensor) of type `device const float4 *`.
    - `src1`: A pointer to the second input tensor (the source row tensor) of type `device const float4 *`.
    - `dst`: A pointer to the output tensor where the result of the addition will be stored, of type `device float4 *`.
    - `tpig`: A thread position index in the grid, used to identify the specific thread executing the kernel.
- **Control Flow**:
    - The function calculates the number of elements in the source tensor by dividing the total number of elements by 4 (since each float4 contains 4 floats).
    - It then performs the addition operation by iterating over the elements of the destination tensor, adding the corresponding elements from the source row tensor based on the thread index.
    - The addition is performed in a loop, where each thread processes a specific index in the destination tensor, effectively broadcasting the source row across the destination.
- **Output**: The output is stored in the destination tensor, which contains the result of the element-wise addition of the source row tensor to the destination tensor.


---
### dequantize\_q8\_0
The `dequantize_q8_0` function converts quantized 8-bit integer data into floating-point representation.
- **Inputs**:
    - `xb`: A pointer to a `block_q8_0` structure containing the quantized data and scaling factor.
    - `il`: An integer indicating the index of the quantization block to process.
    - `reg`: A reference to a thread-local variable where the dequantized floating-point data will be stored.
- **Control Flow**:
    - The function retrieves the quantized data from the `block_q8_0` structure.
    - It calculates the scaling factor from the `d` field of the structure.
    - A loop iterates over 16 elements, multiplying each quantized value by the scaling factor and storing the result in a 4x4 float matrix.
    - Finally, the resulting matrix is assigned to the output reference.
- **Output**: The function does not return a value; instead, it populates the `reg` variable with the dequantized floating-point data.


---
### dequantize\_q5\_K
The `dequantize_q5_K` function converts quantized data from a `block_q5_K` structure into floating-point representation.
- **Inputs**:
    - `xb`: A pointer to a `block_q5_K` structure containing the quantized data and scaling factors.
    - `il`: An integer indicating the index of the quantization level to be processed.
    - `reg`: A reference to a thread-local variable where the dequantized floating-point values will be stored.
- **Control Flow**:
    - The function begins by extracting the quantized values and scaling factors from the `block_q5_K` structure.
    - It calculates the scaling factors based on the input index `il` and prepares masks for processing the quantized values.
    - A loop iterates over the quantized values, applying the appropriate scaling and masking to convert them into floating-point values.
    - The results are accumulated into a temporary floating-point array before being stored in the output register.
- **Output**: The function does not return a value directly; instead, it populates the `reg` variable with the dequantized floating-point values.


---
### kernel\_div\_row
The `kernel_div_row` function performs element-wise division of two input tensors, broadcasting the second tensor across the first.
- **Inputs**:
    - `args`: A constant structure containing the kernel arguments, including tensor dimensions and offsets.
    - `src0`: A device pointer to the first input tensor (numerator) containing float4 values.
    - `src1`: A device pointer to the second input tensor (denominator) containing float4 values.
    - `dst`: A device pointer to the output tensor where the result of the division will be stored.
    - `tpig`: A uint3 variable representing the thread position in the grid.
- **Control Flow**:
    - The function calculates the number of elements in the row to be processed based on the input tensor dimensions.
    - It retrieves the appropriate pointers for the input tensors and the output tensor based on the current thread's position.
    - A loop iterates over the elements of the first tensor, performing division by the corresponding elements of the second tensor, while handling broadcasting.
    - The results are stored in the output tensor.
- **Output**: The output is a tensor containing the results of the element-wise division of `src0` by `src1`, with broadcasting applied as necessary.


---
### dequantize\_q5\_0
The `dequantize_q5_0` function converts quantized data from a `block_q5_0` structure into floating-point representation.
- **Inputs**:
    - `xb`: A pointer to a `block_q5_0` structure containing the quantized data.
    - `il`: An integer indicating the index of the quantized data to be processed.
    - `reg`: A reference to a thread-local variable where the dequantized floating-point data will be stored.
- **Control Flow**:
    - The function retrieves the quantized values from the `block_q5_0` structure.
    - It calculates the scaling factors based on the `il` parameter and the data stored in the structure.
    - A loop iterates over the quantized values, extracting bits and applying the scaling factors to convert them into floating-point values.
    - The results are stored in a temporary 4x4 floating-point matrix.
    - Finally, the matrix is assigned to the output reference variable.
- **Output**: The function does not return a value; instead, it populates the `reg` variable with the dequantized floating-point data.


---
### kernel\_ssm\_conv\_f32
The `kernel_ssm_conv_f32` function performs a single-step convolution operation on input tensors using a specified kernel.
- **Inputs**:
    - `src0`: A pointer to the source tensor containing the input data for the convolution operation.
    - `src1`: A pointer to the kernel tensor that will be used for the convolution operation.
    - `dst`: A pointer to the destination tensor where the result of the convolution will be stored.
    - `args`: A structure containing various parameters for the convolution operation, such as tensor dimensions and strides.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tpitg`: A 3D vector indicating the position of the thread within the thread group.
    - `ntg`: A 3D vector indicating the number of threads per thread group.
- **Control Flow**:
    - The function begins by extracting the indices for the input tensors based on the thread group and thread positions.
    - It calculates the offsets for accessing the source tensors based on the provided arguments.
    - The function then retrieves the input data and kernel data from the source tensors.
    - A loop iterates over the number of channels, performing the convolution operation by multiplying the input data with the kernel and accumulating the results.
    - Finally, the result of the convolution is stored in the destination tensor.
- **Output**: The output is a tensor containing the result of the convolution operation, stored in the memory location pointed to by the `dst` parameter.


---
### kernel\_gelu\_erf
The `kernel_gelu_erf` function applies the Gaussian Error Linear Unit (GELU) activation function using the error function approximation.
- **Inputs**:
    - `src0`: A pointer to the input tensor containing the values to which the GELU activation will be applied.
    - `dst`: A pointer to the output tensor where the results of the GELU activation will be stored.
    - `tpig`: A thread position index used to identify the specific thread's position in the grid.
- **Control Flow**:
    - The function retrieves the input value from `src0` at the position indicated by `tpig`.
    - It computes the GELU activation using the formula: `0.5 * x * (1.0 + erf(x / sqrt(2)))`, where `erf` is the error function.
    - The result is stored in the output tensor `dst` at the same position.
- **Output**: The output is the result of applying the GELU activation function to the input values, stored in the `dst` tensor.


---
### dequantize\_iq4\_nl
The `dequantize_iq4_nl` function dequantizes a block of quantized data from a specific format into floating-point values.
- **Inputs**:
    - `xb`: A pointer to a `block_iq4_nl` structure that contains the quantized data and scaling factor.
    - `il`: An integer indicating the index of the quantization level to be processed.
    - `reg`: A reference to a thread-local variable where the dequantized floating-point values will be stored.
- **Control Flow**:
    - The function retrieves the quantized data from the `xb` structure.
    - It calculates the scaling factor `d` from the `d` field of the `xb` structure.
    - A loop iterates over the four rows of the output, extracting and dequantizing the corresponding values from the quantized data.
    - The dequantized values are computed using a lookup table `kvalues_iq4nl_f` and stored in the `reg` variable.
- **Output**: The function outputs the dequantized floating-point values into the `reg` variable, which is used for further processing.


---
### kernel\_sigmoid
The `kernel_sigmoid` function computes the sigmoid activation function for each element in the input array.
- **Inputs**:
    - `src0`: A pointer to the input array of floats, where each element will be processed by the sigmoid function.
    - `dst`: A pointer to the output array of floats, where the results of the sigmoid function will be stored.
    - `tpig`: An index representing the position of the current thread in the grid, used to access the correct element in the input array.
- **Control Flow**:
    - The function iterates over each element in the input array using the thread index `tpig`.
    - For each element, it calculates the sigmoid value using the formula: 1 / (1 + exp(-src0[tpig])).
    - The computed sigmoid value is then stored in the corresponding position in the output array.
- **Output**: The output is an array of floats containing the sigmoid values of the input elements.


---
### dequantize\_iq2\_xxs
The `dequantize_iq2_xxs` function performs dequantization of input data from a specific quantization format into floating-point representation.
- **Inputs**:
    - `xb`: A pointer to a `block_iq2_xxs` structure that contains the quantized data and scaling factors.
    - `il`: An integer indicating the index of the quantization block being processed, which can range from 0 to 15.
    - `reg`: A reference to a thread-local variable where the dequantized floating-point values will be stored.
- **Control Flow**:
    - The function calculates the index of the quantization block based on the input index `il`.
    - It retrieves the quantized data from the `qs` array in the `block_iq2_xxs` structure.
    - The function computes the scaling factor based on the quantization parameters.
    - It uses a grid of pre-defined values to map the quantized values to their corresponding floating-point representations.
    - The dequantized values are stored in the `reg` variable, which is structured as a 4x4 matrix.
- **Output**: The output is a 4x4 matrix of floating-point values that represent the dequantized data from the input quantization block.


---
### dequantize\_iq2\_xs
The `dequantize_iq2_xs` function dequantizes input data from a specific quantization format into floating-point representation.
- **Inputs**:
    - `xb`: A pointer to a `block_iq2_xs` structure containing the quantized data and scaling factors.
    - `il`: An integer indicating the index of the quantization block being processed, which determines which part of the data to dequantize.
    - `reg`: A reference to a thread-local variable where the dequantized floating-point values will be stored.
- **Control Flow**:
    - The function first calculates the index of the quantization block based on the input index `il`.
    - It retrieves the quantized data from the `qs` array in the `block_iq2_xs` structure.
    - The function then computes the scaling factor based on the `scales` array and the current block index.
    - It iterates over the quantized values, applying the scaling factor and sign adjustments to produce the final dequantized values.
    - The results are stored in the `reg` variable, which is structured to hold the dequantized floating-point values.
- **Output**: The function does not return a value; instead, it populates the `reg` variable with the dequantized floating-point values derived from the quantized input data.


---
### mul\_vec\_q\_n\_f32\_impl
`mul_vec_q_n_f32_impl` performs matrix-vector multiplication for quantized vectors using various quantization schemes.
- **Inputs**:
    - `args`: A structure containing parameters for the multiplication operation, including dimensions and offsets.
    - `src0`: A pointer to the source tensor containing quantized data.
    - `src1`: A pointer to the source tensor containing the vector to be multiplied.
    - `dst`: A pointer to the destination tensor where the result will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the position of the SIMD group within the thread group.
- **Control Flow**:
    - The function begins by calculating the number of blocks based on the input dimensions.
    - It retrieves the current row and column indices from the thread group indices.
    - It calculates the offsets for accessing the source tensors based on the current indices.
    - The function then initializes an array to hold the results of the multiplication.
    - It iterates over the quantized blocks, performing the multiplication and accumulation of results.
    - Finally, it stores the computed results in the destination tensor.
- **Output**: The function outputs the result of the matrix-vector multiplication in the destination tensor.


---
### kernel\_gelu\_quick
The `kernel_gelu_quick` function applies the Gaussian Error Linear Unit (GELU) activation function quickly to input values.
- **Inputs**:
    - `src0`: A pointer to the input tensor containing the values to be transformed by the GELU activation function.
    - `dst`: A pointer to the output tensor where the transformed values will be stored.
    - `tpig`: A thread position index used to identify the specific thread's position in the grid.
- **Control Flow**:
    - The function retrieves the input value for the current thread using the `tpig` index.
    - It computes the GELU activation using the formula: x * (1.0 / (1.0 + exp(GELU_QUICK_COEF * x))).
    - The result is stored in the output tensor at the corresponding position.
- **Output**: The output is a tensor containing the transformed values after applying the GELU activation function.


---
### kernel\_mul\_mv\_ext\_q4x4\_f32\_disp
The `kernel_mul_mv_ext_q4x4_f32_disp` function performs matrix-vector multiplication for quantized 4x4 blocks in a Metal kernel.
- **Inputs**:
    - `args`: A constant structure containing the arguments for the matrix-vector multiplication, including dimensions and offsets.
    - `src0`: A pointer to the source matrix data, which is quantized.
    - `src1`: A pointer to the source vector data, which is used in the multiplication.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the position of the SIMD group within the thread group.
- **Control Flow**:
    - The function begins by determining the number of chunks and the current thread's position within the grid.
    - It calculates offsets for accessing the source and destination data based on the current thread's indices.
    - The function then loads the quantized data from the source matrix and the vector data from the source vector.
    - It performs the matrix-vector multiplication in chunks, accumulating results in shared memory.
    - Finally, it writes the accumulated results to the destination memory.
- **Output**: The output is stored in the destination pointer, which contains the result of the matrix-vector multiplication.


---
### kernel\_diag\_mask\_inf
The `kernel_diag_mask_inf` function applies a diagonal mask to a tensor, setting elements to negative infinity based on a specified condition.
- **Inputs**:
    - `src0`: A device pointer to the input tensor from which values are copied.
    - `dst`: A device pointer to the output tensor where the masked values will be stored.
    - `args`: A constant structure containing parameters such as n_past and dimensions of the tensor.
    - `tpig`: A 3D index representing the position of the thread group in the grid.
- **Control Flow**:
    - The function calculates the indices for the 3D tensor based on the thread group index.
    - It checks if the current index exceeds the sum of n_past and the second index; if so, it sets the corresponding output element to negative infinity.
    - Otherwise, it copies the value from the input tensor to the output tensor.
- **Output**: The output tensor with elements set to negative infinity where the condition is met, and original values from the input tensor otherwise.


---
### kernel\_sum\_rows
The `kernel_sum_rows` function computes the sum of each row in a 2D tensor and stores the result in a 1D tensor.
- **Inputs**:
    - `src0`: A pointer to the input tensor from which the rows will be summed.
    - `dst`: A pointer to the output tensor where the sum of each row will be stored.
    - `args`: A structure containing the necessary parameters for the summation, including the dimensions of the tensor.
    - `tpig`: A 3D vector indicating the position of the thread in the grid.
- **Control Flow**:
    - The function first retrieves the indices of the current thread in the grid.
    - It checks if the current indices are within the bounds of the tensor dimensions.
    - It calculates the pointer to the current row of the input tensor based on the indices.
    - It initializes a variable to hold the sum of the row.
    - It iterates over each element in the row, accumulating the sum.
    - Finally, it stores the computed sum in the corresponding position of the output tensor.
- **Output**: The output is a 1D tensor containing the sum of each row from the input tensor.


---
### kernel\_tanh
The `kernel_tanh` function computes the hyperbolic tangent of each element in the input array and stores the results in the output array.
- **Inputs**:
    - `src0`: A pointer to the input array containing the values for which the hyperbolic tangent will be computed.
    - `dst`: A pointer to the output array where the computed hyperbolic tangent values will be stored.
    - `tpig`: An index representing the position of the thread in the grid, used to access the correct element in the input and output arrays.
- **Control Flow**:
    - The function retrieves the input value at the index specified by `tpig`.
    - It computes the hyperbolic tangent of the input value using the `precise::tanh` function.
    - The computed value is then stored in the output array at the same index.
- **Output**: The output is an array of the same size as the input, containing the hyperbolic tangent of each input value.


---
### kernel\_argsort\_f32\_i32
The `kernel_argsort_f32_i32` function performs a bitonic sort on a row of floating-point values and returns the sorted indices.
- **Inputs**:
    - `x`: A pointer to the input array of floating-point values to be sorted.
    - `dst`: A pointer to the output array where the sorted indices will be stored.
    - `args`: A structure containing parameters for the sorting operation, including the number of columns and padding.
    - `shared_values`: A threadgroup shared memory array used to hold intermediate values during the sorting process.
    - `tgpig`: A 3D index indicating the position of the threadgroup in the grid.
    - `tpitg`: A 3D index indicating the position of the thread within the threadgroup.
- **Control Flow**:
    - The function first calculates the column and row indices based on the threadgroup and thread indices.
    - It initializes the shared memory with the index of the current column.
    - A barrier is used to synchronize all threads in the threadgroup before proceeding with the sorting.
    - The bitonic sort algorithm is applied in a series of steps, where pairs of indices are compared and swapped based on the specified sort order.
    - After sorting, the final sorted indices are written back to the output array, excluding any padding.
- **Output**: The output is an array of sorted indices corresponding to the input floating-point values.


---
### kernel\_upscale\_f32
The `kernel_upscale_f32` function performs an upscale operation on a tensor by copying and scaling its values based on specified scaling factors.
- **Inputs**:
    - `src0`: A pointer to the source tensor data that needs to be upscaled.
    - `dst`: A pointer to the destination tensor where the upscaled data will be stored.
    - `args`: A structure containing the scaling factors for each dimension of the tensor.
- **Control Flow**:
    - The function calculates the indices for the source tensor based on the upscale factors provided in `args`.
    - It iterates over the output tensor dimensions, scaling the indices to access the corresponding values in the source tensor.
    - For each index in the output tensor, it copies the value from the source tensor to the destination tensor, applying the scaling factor.
- **Output**: The function does not return a value; instead, it writes the upscaled tensor data directly to the memory location pointed to by `dst`.


---
### kernel\_mul\_mv\_q5\_1\_f32
The `kernel_mul_mv_q5_1_f32` function performs matrix-vector multiplication for quantized data using a specific quantization scheme.
- **Inputs**:
    - `args`: A constant structure containing parameters for the matrix-vector multiplication operation, including dimensions and offsets.
    - `src0`: A pointer to the source matrix data, which is quantized using the Q5_1 scheme.
    - `src1`: A pointer to the source vector data that will be multiplied with the matrix.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the position of the SIMD group within the thread group.
- **Control Flow**:
    - The function begins by calling a template implementation `mul_vec_q_n_f32_impl` with specific parameters for the Q5_1 quantization scheme.
    - It calculates the number of blocks and the offsets for the source and destination data based on the input parameters.
    - The function then iterates over the blocks of the source matrix and performs the multiplication with the source vector.
    - Results are accumulated and stored in the destination pointer.
- **Output**: The output is stored in the destination pointer, which contains the result of the matrix-vector multiplication.


---
### kernel\_rwkv\_wkv6\_f32
The `kernel_rwkv_wkv6_f32` function implements a kernel for processing RWKV (Recurrent Weighted Key-Value) operations in a neural network context.
- **Inputs**:
    - `k`: A device pointer to the key tensor used in the RWKV operation.
    - `v`: A device pointer to the value tensor used in the RWKV operation.
    - `r`: A device pointer to the recurrent tensor used in the RWKV operation.
    - `tf`: A device pointer to the tensor containing transformation factors.
    - `td`: A device pointer to the tensor containing decay factors.
    - `state_in`: A device pointer to the input state tensor for the RWKV operation.
    - `dst`: A device pointer to the output tensor where results will be stored.
    - `B`: A constant reference to the batch size.
    - `T`: A constant reference to the total number of sequence tokens.
    - `C`: A constant reference to the number of channels.
    - `H`: A constant reference to the number of heads.
    - `tgpig`: A 3D vector representing the thread group position in the grid.
    - `tpitg`: A 3D vector representing the thread position in the thread group.
    - `ntg`: A 3D vector representing the number of threads per thread group.
- **Control Flow**:
    - The function begins by calculating the head size and determining the batch and head IDs based on the thread group position.
    - It checks if the batch ID and head ID are within valid bounds; if not, it returns early.
    - The function initializes threadgroup arrays for keys, recurrent values, transformation factors, decay factors, and the state.
    - It populates the state array from the input state tensor.
    - The function enters a loop over sequence tokens, where it processes each token in chunks.
    - Within the loop, it loads the key, recurrent, transformation, and decay values into threadgroup arrays.
    - It computes the output value by performing operations on the loaded values and the current state.
    - Finally, it updates the output tensor with the computed values and the updated state.
- **Output**: The function outputs the computed values for each token in the destination tensor and updates the state tensor with the new state.


---
### kernel\_mul\_row
The `kernel_mul_row` function performs element-wise multiplication of two input float4 arrays, broadcasting the second array across the first.
- **Inputs**:
    - `args`: A constant structure containing metadata for the operation, including dimensions and offsets.
    - `src0`: A device pointer to the first input array of type `float4`.
    - `src1`: A device pointer to the second input array of type `float4`.
    - `dst`: A device pointer to the output array where the result of the multiplication will be stored.
    - `tpig`: A thread position index in the grid, used to identify the specific thread executing the kernel.
- **Control Flow**:
    - The function calculates the number of elements in the row based on the input arguments.
    - It retrieves the corresponding elements from the input arrays `src0` and `src1`.
    - The multiplication is performed element-wise, with `src1` being broadcasted across the rows of `src0`.
    - The result is stored in the output array `dst`.
- **Output**: The output is a device pointer to an array of type `float4`, containing the results of the element-wise multiplication.


---
### dequantize\_q6\_K
The `dequantize_q6_K` function performs dequantization of a quantized tensor block using specific scaling and masking techniques.
- **Inputs**:
    - `xb`: A pointer to a `block_q6_K` structure that contains the quantized data and associated parameters for dequantization.
    - `il`: An integer indicating the index of the quantization level to be processed.
    - `reg`: A reference to a thread-local variable where the dequantized results will be stored.
- **Control Flow**:
    - The function begins by calculating the offsets for the low and high quantized values based on the input index `il`.
    - It retrieves the quantized values from the `block_q6_K` structure and initializes scaling factors based on the quantization parameters.
    - The function then iterates over the quantized values, applying the appropriate scaling and masking to compute the dequantized values.
    - Finally, the computed values are stored in the `reg` variable for further processing.
- **Output**: The output is the dequantized tensor values stored in the `reg` variable, which can be used in subsequent computations.


---
### kernel\_diag\_mask\_inf\_8
The `kernel_diag_mask_inf_8` function applies a diagonal mask to a 4-component float vector, setting elements to negative infinity based on a specified condition.
- **Inputs**:
    - `src0`: A pointer to the source float4 vector from which values are read.
    - `dst`: A pointer to the destination float4 vector where the masked values are written.
    - `args`: A structure containing parameters including 'n_past', which determines the masking condition.
    - `tpig`: A 3D index representing the position of the thread in the grid.
- **Control Flow**:
    - The function calculates the linear index based on the 3D thread position.
    - It checks if the current index exceeds the sum of 'n_past' and the second index.
    - If the condition is met, the corresponding element in the destination vector is set to negative infinity.
    - Otherwise, the value from the source vector is copied to the destination.
- **Output**: The function modifies the destination vector in place, setting certain elements to negative infinity based on the masking condition.


---
### kernel\_scale\_4
The `kernel_scale_4` function scales a 4-component vector by a specified scalar value.
- **Inputs**:
    - `src0`: A pointer to the source vector of type `float4` that contains the values to be scaled.
    - `dst`: A pointer to the destination vector of type `float4` where the scaled result will be stored.
    - `scale`: A constant reference to a float value that represents the scaling factor.
    - `tpig`: A thread position index in the grid, used to identify the specific thread executing the kernel.
- **Control Flow**:
    - The function retrieves the current thread's index using `tpig`.
    - It accesses the source vector `src0` at the current thread index and scales each component by the `scale` factor.
    - The scaled values are then stored in the destination vector `dst` at the same index.
- **Output**: The output is a scaled vector stored in the destination pointer `dst`, where each component of the original vector is multiplied by the `scale` factor.


---
### kernel\_norm
The `kernel_norm` function normalizes the input tensor by subtracting the mean and scaling by the standard deviation.
- **Inputs**:
    - `args`: A structure containing normalization parameters such as epsilon and tensor dimensions.
    - `src0`: A pointer to the input tensor data that needs to be normalized.
    - `dst`: A pointer to the output tensor where the normalized data will be stored.
    - `shmem_f32`: Shared memory for intermediate float values used during normalization.
    - `tgpig`: Thread group position in the grid, indicating the current thread group being processed.
    - `tpitg`: Thread position within the thread group, indicating the current thread's index.
    - `sgitg`: SIMD group index within the thread group, used for SIMD operations.
    - `tiisg`: Thread index within the SIMD group, used for accessing shared memory.
    - `ntg`: Number of threads per thread group, used for parallel processing.
- **Control Flow**:
    - The function begins by initializing shared memory for storing intermediate results.
    - It calculates the sum of the input tensor elements in parallel, accumulating results in shared memory.
    - After the sum is computed, it calculates the mean by dividing the total sum by the number of elements.
    - The function then computes the variance by calculating the squared differences from the mean.
    - Finally, it normalizes the input tensor by subtracting the mean and scaling by the standard deviation, storing the result in the output tensor.
- **Output**: The output is a normalized tensor where each element has been adjusted based on the mean and standard deviation of the input tensor.


---
### kernel\_sub
The `kernel_sub` function performs element-wise subtraction of two tensors in a Metal kernel.
- **Inputs**:
    - `args`: A constant structure containing the kernel arguments, including tensor dimensions and offsets.
    - `src0`: A device pointer to the first source tensor from which values will be subtracted.
    - `src1`: A device pointer to the second source tensor from which values will be subtracted.
    - `dst`: A device pointer to the destination tensor where the result of the subtraction will be stored.
    - `tgpig`: A uint3 variable representing the position of the threadgroup in the grid.
    - `tpitg`: A ushort3 variable representing the position of the thread within the threadgroup.
    - `ntg`: A ushort3 variable representing the number of threads per threadgroup.
- **Control Flow**:
    - The function begins by extracting the indices for the current threadgroup and thread position.
    - It calculates the offsets for the source tensors based on the provided arguments.
    - A loop iterates over the elements of the tensors, performing the subtraction operation for each element.
    - The result of the subtraction is stored in the destination tensor.
- **Output**: The output is a tensor stored in the `dst` pointer, which contains the result of the element-wise subtraction of `src1` from `src0`.


---
### kernel\_mul\_mv\_ext\_q4\_f32\_disp
The `kernel_mul_mv_ext_q4_f32_disp` function is a Metal kernel that performs matrix-vector multiplication for quantized 4-bit representations, allowing for efficient computation on GPU.
- **Inputs**:
    - `args`: A constant structure containing parameters for the matrix-vector multiplication, including dimensions and offsets.
    - `src0`: A pointer to the source matrix in quantized format (4-bit) that will be multiplied.
    - `src1`: A pointer to the source vector (float) that will be multiplied with the matrix.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the position of the SIMD group within the thread group.
- **Control Flow**:
    - The function begins by determining the number of chunks and the current thread's position within the grid.
    - It calculates offsets for accessing the source matrix and vector based on the current thread's indices.
    - The function then loads the quantized matrix and the vector into local memory for processing.
    - It performs the matrix-vector multiplication in chunks, accumulating results in shared memory.
    - Finally, the results are written back to the destination memory.
- **Output**: The output is stored in the destination pointer `dst`, which contains the result of the matrix-vector multiplication.


---
### dequantize\_bf16\_t4
The `dequantize_bf16_t4` function dequantizes a tensor of `bfloat` values from a source pointer into a target register, scaling the values based on the provided parameters.
- **Inputs**:
    - `src`: A pointer to the source tensor of `bfloat` values to be dequantized.
    - `il`: An integer indicating the index of the quantization level to be used for dequantization.
    - `reg`: A reference to the target register where the dequantized values will be stored.
- **Control Flow**:
    - The function retrieves the `bfloat` value from the source pointer based on the provided index.
    - The value is then directly assigned to the target register without additional scaling or transformation.
- **Output**: The output is the dequantized value stored in the target register, which is of type `type4`.


---
### kernel\_sin
The `kernel_sin` function computes the sine of each element in the input array and stores the results in the output array.
- **Inputs**:
    - `src0`: A device pointer to an array of float values representing the input angles in radians.
    - `dst`: A device pointer to an array where the computed sine values will be stored.
    - `tpig`: A uint3 variable representing the thread position in the grid.
- **Control Flow**:
    - The function iterates over each element in the input array using the thread index.
    - For each element, it computes the sine using the `sin` function.
    - The computed sine value is then stored in the corresponding position in the output array.
- **Output**: The output is an array of float values containing the sine of the input angles.


---
### kernel\_cpy\_f32\_q8\_0
The `kernel_cpy_f32_q8_0` function copies data from a source tensor to a destination tensor while quantizing the values into a specific format.
- **Inputs**:
    - `args`: A structure containing metadata for the copy operation, including dimensions and offsets.
    - `src0`: A pointer to the source tensor data in float format.
    - `dst`: A pointer to the destination tensor where the quantized data will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tpitg`: A 3D vector indicating the position of the thread within the thread group.
    - `ntg`: A 3D vector indicating the number of threads per thread group.
- **Control Flow**:
    - Calculate the linear index 'n' based on the thread group position and the dimensions of the tensor.
    - Determine the indices for the source and destination tensors based on the calculated index 'n'.
    - Iterate over the quantization blocks, copying data from the source tensor to the destination tensor.
    - For each block, calculate the maximum absolute value to determine the scaling factor for quantization.
    - Quantize the values from the source tensor and store them in the destination tensor.
- **Output**: The function does not return a value; instead, it writes the quantized data directly to the destination tensor.


---
### kernel\_ssm\_scan\_f32
The `kernel_ssm_scan_f32` function implements a scanning operation for stateful sequence models, processing input tensors through a series of transformations and updates.
- **Inputs**:
    - `src0`: A pointer to the input tensor representing the current state.
    - `src1`: A pointer to the input tensor representing the current input sequence.
    - `src2`: A pointer to the input tensor representing the previous state.
    - `src3`: A pointer to the input tensor representing the transformation matrix A.
    - `src4`: A pointer to the input tensor representing the transformation matrix B.
    - `src5`: A pointer to the input tensor representing the transformation matrix C.
    - `dst`: A pointer to the output tensor where the results will be stored.
    - `args`: A structure containing the parameters for the scanning operation, including dimensions and offsets.
    - `tgpig`: A 3D vector representing the position of the thread group in the grid.
    - `tpitg`: A 3D vector representing the position of the thread within the thread group.
    - `ntg`: A 3D vector representing the number of threads per thread group.
- **Control Flow**:
    - The function begins by extracting the dimensions and offsets from the `args` structure.
    - It iterates over the sequence tokens, processing each token in the input tensors.
    - For each token, it retrieves the current state and input, applies transformations using matrices A, B, and C, and updates the state.
    - The results are accumulated and stored in the output tensor.
- **Output**: The output is a tensor containing the updated states after processing the input sequences through the scanning operation.


---
### kernel\_mul\_mv\_1row
The `kernel_mul_mv_1row` function performs a matrix-vector multiplication where the vector is treated as a single row, effectively broadcasting the vector across the rows of the matrix.
- **Inputs**:
    - `args`: A constant structure containing the parameters for the matrix-vector multiplication, including dimensions and offsets.
    - `src0`: A pointer to the source matrix data in device memory.
    - `src1`: A pointer to the source vector data in device memory.
    - `dst`: A pointer to the destination memory where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
- **Control Flow**:
    - The function calculates the offsets for the source matrix and vector based on the input indices.
    - It retrieves the corresponding row from the matrix and the vector.
    - The function then performs the multiplication of the vector with the matrix row and accumulates the result.
    - Finally, it stores the result in the destination memory.
- **Output**: The output is stored in the destination memory, which contains the result of the matrix-vector multiplication for the specified row.


---
### kernel\_mul\_mv\_q4\_1\_f32
The `kernel_mul_mv_q4_1_f32` function performs matrix-vector multiplication for quantized 4-bit data using a specific kernel implementation.
- **Inputs**:
    - `args`: A constant structure containing the parameters for the matrix-vector multiplication operation, including dimensions and offsets.
    - `src0`: A pointer to the source matrix data, which is quantized in 4-bit format.
    - `src1`: A pointer to the source vector data, which is used in the multiplication.
    - `dst`: A pointer to the destination memory where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the position of the SIMD group within the thread group.
- **Control Flow**:
    - The function begins by calling a template function `mul_vec_q_n_f32_impl` with specific parameters for the quantized block type `block_q4_1`.
    - The template function handles the actual multiplication logic, iterating over the quantized data and performing the necessary calculations.
    - It computes the inner product between the quantized matrix rows and the vector, accumulating the results into a destination array.
- **Output**: The output is stored in the destination pointer `dst`, which contains the results of the matrix-vector multiplication.


---
### dequantize\_iq4\_xs
The `dequantize_iq4_xs` function dequantizes a block of quantized data from a specific format into floating-point values.
- **Inputs**:
    - `xb`: A pointer to a `block_iq4_xs` structure containing the quantized data and scaling information.
    - `il`: An integer indicating the index of the block being processed, which determines how the quantized values are accessed.
    - `reg`: A reference to a thread-local variable where the dequantized floating-point values will be stored.
- **Control Flow**:
    - The function begins by calculating the index of the quantized data block based on the input index `il`.
    - It retrieves the scaling factor from the `block_iq4_xs` structure.
    - A loop iterates over the four rows of the quantized data, extracting and dequantizing the values based on the quantization scheme.
    - The dequantized values are computed using a lookup table and stored in the `reg` variable.
- **Output**: The output is a set of four floating-point values stored in the `reg` variable, representing the dequantized data from the input block.


---
### kernel\_pad\_f32
The `kernel_pad_f32` function pads a source tensor with zeros or retains its values based on the specified padding dimensions.
- **Inputs**:
    - `src0`: A pointer to the source tensor data that needs to be padded.
    - `dst`: A pointer to the destination tensor where the padded result will be stored.
    - `args`: A structure containing padding parameters such as the number of elements in each dimension.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tpitg`: A 3D vector indicating the position of the thread within the thread group.
    - `ntg`: A 3D vector indicating the number of threads per thread group.
- **Control Flow**:
    - The function calculates the indices for the source and destination tensors based on the thread group and thread indices.
    - It checks if the current indices are within the bounds of the source tensor dimensions.
    - If the indices are valid, it copies the corresponding values from the source tensor to the destination tensor.
    - If the indices exceed the source tensor dimensions, it fills the destination tensor with zeros.
- **Output**: The output is a padded tensor stored in the destination pointer, with values copied from the source tensor and zeros added where necessary.


---
### erf\_approx
The `erf_approx` function computes an approximation of the error function for a given input value.
- **Inputs**:
    - `x`: A value of type T for which the error function approximation is to be computed.
- **Control Flow**:
    - The function first determines the sign of the input value `x` and takes its absolute value.
    - It then computes a temporary variable `t` based on the input value `x`.
    - Using a polynomial approximation, it calculates the value of `y` which approximates the error function.
    - Finally, it returns the product of the sign of `x` and the computed value `y`.
- **Output**: The function returns a value of type T that represents the approximation of the error function for the input value.


---
### kernel\_div
The `kernel_div` function performs element-wise division of two tensors on a GPU using Metal shading language.
- **Inputs**:
    - `args`: A constant structure containing the kernel arguments, including tensor dimensions and offsets.
    - `src0`: A pointer to the first input tensor (numerator) stored in device memory.
    - `src1`: A pointer to the second input tensor (denominator) stored in device memory.
    - `dst`: A pointer to the output tensor where the result of the division will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tpitg`: A 3D vector indicating the position of the thread within the thread group.
    - `ntg`: A 3D vector indicating the number of threads per thread group.
- **Control Flow**:
    - The function calculates the indices for the current thread group and thread position.
    - It computes the offsets for accessing the input tensors based on the calculated indices.
    - A loop iterates over the elements of the output tensor, performing division for each element from the two input tensors.
    - The result of the division is stored in the output tensor.
- **Output**: The output is a tensor containing the results of the element-wise division of the two input tensors.


---
### rope\_yarn\_corr\_factor
Calculates the correction factor for the rope and yarn embedding based on the number of dimensions, context size, number of rotations, and base frequency.
- **Inputs**:
    - `n_dims`: The number of dimensions for the embedding.
    - `n_ctx_orig`: The original context size.
    - `n_rot`: The number of rotations.
    - `base`: The base frequency used in the calculation.
- **Control Flow**:
    - Calculates the correction factor using the formula derived from the relationship between the number of rotations and the dimensions.
    - Returns the calculated correction factor.
- **Output**: A float value representing the correction factor for the rope and yarn embedding.


---
### kernel\_conv\_transpose\_1d
The `kernel_conv_transpose_1d` function performs a 1D transposed convolution operation on input tensors.
- **Inputs**:
    - `src0`: A pointer to the input tensor containing the source data for the convolution operation.
    - `src1`: A pointer to the tensor containing the convolution kernel weights.
    - `dst`: A pointer to the output tensor where the result of the transposed convolution will be stored.
    - `args`: A structure containing parameters for the convolution operation, such as input and output dimensions.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tgpg`: A 3D vector indicating the number of thread groups in the grid.
- **Control Flow**:
    - The function iterates over the input channels and applies the transposed convolution for each channel.
    - For each input channel, it calculates the output value by summing the products of the kernel weights and the corresponding input values.
    - The output value is then stored in the destination tensor at the appropriate index.
- **Output**: The output is a tensor containing the result of the transposed convolution operation, which is stored in the memory location pointed to by 'dst'.


---
### kernel\_leaky\_relu\_f32
The `kernel_leaky_relu_f32` function applies the leaky ReLU activation function to each element of the input tensor.
- **Inputs**:
    - `src0`: A pointer to the input tensor containing the values to which the leaky ReLU function will be applied.
    - `dst`: A pointer to the output tensor where the results of the leaky ReLU function will be stored.
    - `args`: A structure containing the parameters for the leaky ReLU operation, including the slope for negative values.
    - `tpig`: The thread position in the grid, used to determine which element of the tensor to process.
- **Control Flow**:
    - The function iterates over each element of the input tensor using the thread position provided by `tpig`.
    - For each element, it checks if the value is greater than zero.
    - If the value is greater than zero, it is directly assigned to the output tensor.
    - If the value is less than or equal to zero, it is multiplied by the slope defined in `args` before being assigned to the output tensor.
- **Output**: The output tensor contains the results of applying the leaky ReLU activation function to the input tensor, with negative values scaled by the specified slope.


---
### dequantize\_iq3\_xxs
The `dequantize_iq3_xxs` function performs dequantization of input data from a specific quantization format into floating-point representation.
- **Inputs**:
    - `xb`: A pointer to a `block_iq3_xxs` structure containing the quantized data and associated parameters.
    - `il`: An integer index indicating the specific block of quantized data to process, ranging from 0 to 15.
    - `reg`: A reference to a thread-local variable where the dequantized floating-point values will be stored.
- **Control Flow**:
    - The function begins by calculating the block index based on the input index `il`.
    - It retrieves the quantized data from the `xb` structure and initializes the dequantization parameters.
    - The function then iterates over the quantized values, applying the dequantization formula to convert them into floating-point values.
    - Finally, the dequantized values are stored in the `reg` variable for further processing.
- **Output**: The output is a set of dequantized floating-point values stored in the `reg` variable, which can be used in subsequent computations.


---
### dequantize\_q8\_0\_t4
The `dequantize_q8_0_t4` function dequantizes a quantized tensor of type `block_q8_0` into a float array using a specified index.
- **Inputs**:
    - `xb`: A pointer to a `block_q8_0` structure that contains the quantized data and scaling factor.
    - `il`: An integer index used to determine which part of the quantized data to process.
    - `reg`: A thread-local float array where the dequantized values will be stored.
- **Control Flow**:
    - The function retrieves the quantized data from the `block_q8_0` structure using the provided index.
    - It calculates the dequantized values by multiplying the quantized values by the scaling factor.
    - The results are stored in the provided thread-local float array.
- **Output**: The function does not return a value; instead, it populates the `reg` array with the dequantized float values.


---
### kernel\_set
The `kernel_set` function copies data from a source tensor to a destination tensor based on specified indices and offsets.
- **Inputs**:
    - `args`: A constant structure containing metadata for the operation, including tensor dimensions and offsets.
    - `src0`: A pointer to the source tensor from which data will be copied.
    - `src1`: A pointer to the source tensor that provides the data to be copied.
    - `dst`: A pointer to the destination tensor where the data will be copied.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tpitg`: A 3D vector indicating the position of the thread within the thread group.
    - `ntg`: A 3D vector indicating the number of threads per thread group.
- **Control Flow**:
    - Calculate the linear index 'n' based on the thread group indices and tensor dimensions.
    - Determine the corresponding indices (i3, i2, i1) for the destination tensor based on 'n'.
    - Calculate the pointer to the destination data using the computed indices and offsets.
    - Iterate over the range of the first dimension of the tensor, copying data from the source tensor to the destination tensor.
- **Output**: The function does not return a value; instead, it modifies the destination tensor in place by copying data from the source tensor.


---
### kernel\_rope\_vision
The `kernel_rope_vision` function applies a rotary positional encoding transformation to input data based on specified parameters.
- **Inputs**:
    - `args`: A constant structure containing parameters for the rotary encoding, including dimensions and frequency scaling.
    - `src0`: A device pointer to the input tensor that will be transformed.
    - `src1`: A device pointer to the tensor containing positional indices.
    - `src2`: A device pointer to an optional tensor used for frequency scaling.
    - `dst`: A device pointer to the output tensor where the transformed data will be stored.
    - `tiitg`: The thread index within the thread group.
    - `tptg`: The thread position in the thread group.
    - `tgpig`: The thread group position in the grid.
- **Control Flow**:
    - Calculate correction dimensions based on the input parameters.
    - Iterate over the input tensor, applying the rotary encoding transformation based on the positional indices.
    - For each dimension, compute the cosine and sine values based on the calculated angles.
    - Store the transformed values in the output tensor.
- **Output**: The output tensor containing the transformed data after applying the rotary positional encoding.


---
### kernel\_flash\_attn\_ext
The `kernel_flash_attn_ext` function implements an efficient attention mechanism using Metal shaders for GPU acceleration.
- **Inputs**:
    - `args`: A constant structure containing parameters for the attention mechanism, including dimensions and scaling factors.
    - `q`: A device pointer to the query tensor.
    - `k`: A device pointer to the key tensor.
    - `v`: A device pointer to the value tensor.
    - `mask`: A device pointer to the mask tensor, used to prevent attention to certain positions.
    - `dst`: A device pointer to the output tensor where the results of the attention computation will be stored.
    - `shmem_f16`: A threadgroup shared memory buffer for intermediate calculations.
    - `tgpig`: A uint3 variable representing the position of the threadgroup in the grid.
    - `ntg`: A ushort3 variable representing the number of threads per threadgroup.
    - `tiisg`: A ushort variable representing the index of the thread within the SIMD group.
    - `sgitg`: A ushort variable representing the index of the SIMD group.
- **Control Flow**:
    - The function begins by initializing shared memory and loading query data into it.
    - It then zeroes out the output and shared memory buffers.
    - The function proceeds to compute the attention scores by iterating over the key and value tensors.
    - It applies a softmax operation to the computed scores to normalize them.
    - Finally, it accumulates the results into the output tensor.
- **Output**: The output is a tensor containing the results of the attention mechanism, which is stored in the `dst` pointer.


---
### kernel\_sqr
The `kernel_sqr` function computes the square of each element in the input array and stores the result in the output array.
- **Inputs**:
    - `src0`: A device pointer to the input array of floats, where each element will be squared.
    - `dst`: A device pointer to the output array of floats, where the squared results will be stored.
    - `tpig`: A thread position index in the grid, used to identify the specific thread's position for processing.
- **Control Flow**:
    - The function iterates over each element of the input array using the thread index provided by `tpig`.
    - For each element, it computes the square by multiplying the element by itself.
    - The result is then stored in the corresponding position in the output array.
- **Output**: The output is an array of floats where each element is the square of the corresponding element from the input array.


---
### kernel\_mul\_mv\_ext\_q4x4\_f32\_impl
The `kernel_mul_mv_ext_q4x4_f32_impl` function performs matrix-vector multiplication for quantized 4x4 blocks in a Metal kernel.
- **Inputs**:
    - `args`: A constant structure containing the arguments for the matrix-vector multiplication, including dimensions and offsets.
    - `src0`: A pointer to the source matrix data, which is quantized.
    - `src1`: A pointer to the source vector data that will be multiplied with the matrix.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the position of the SIMD group within the thread group.
- **Control Flow**:
    - The function begins by calculating the number of chunks and the current thread's indices.
    - It retrieves the offsets for the source and destination data based on the current thread's indices.
    - The function then loads the quantized matrix and vector data into local variables.
    - It performs the matrix-vector multiplication in chunks, accumulating results into a local sum.
    - Finally, the results are stored back into the destination memory.
- **Output**: The output is stored in the destination pointer, which contains the result of the matrix-vector multiplication.


---
### kernel\_soft\_max
The `kernel_soft_max` function computes the softmax of input tensors in a parallelized manner using Metal shading language.
- **Inputs**:
    - `src0`: A pointer to the input tensor containing the values for which the softmax is to be computed.
    - `src1`: A pointer to an optional mask tensor that can modify the input values before softmax computation.
    - `dst`: A pointer to the output tensor where the computed softmax values will be stored.
    - `args`: A structure containing parameters for the softmax operation, including dimensions and scaling factors.
    - `buf`: A threadgroup buffer used for intermediate calculations, specifically for storing maximum values and sums.
    - `tgpig`: A uint3 variable representing the position of the threadgroup in the grid.
    - `tpitg`: A uint variable representing the position of the thread within the threadgroup.
    - `sgitg`: A uint variable representing the index of the SIMD group within the threadgroup.
    - `tiisg`: A uint variable representing the index of the thread within the SIMD group.
    - `ntg`: A uint variable representing the number of threads per threadgroup.
- **Control Flow**:
    - The function begins by calculating the indices for the input tensor based on the threadgroup and thread indices.
    - It retrieves the input values from `src0` and applies any modifications from `src1` if a mask is provided.
    - The maximum value among the input values is computed in parallel to prevent overflow during exponentiation.
    - The softmax values are computed by exponentiating the adjusted input values and normalizing them by the sum of the exponentials.
    - Finally, the computed softmax values are stored in the output tensor `dst`.
- **Output**: The function outputs a tensor containing the softmax values computed from the input tensor.


---
### dequantize\_q4\_0\_t4
The `dequantize_q4_0_t4` function dequantizes a quantized tensor block of type `block_q4_0` into a thread-local float array.
- **Inputs**:
    - `xb`: A pointer to a `block_q4_0` structure that contains the quantized data and scaling factors.
    - `il`: An integer indicating the index of the quantization level to be processed.
    - `reg`: A thread-local float array where the dequantized values will be stored.
- **Control Flow**:
    - The function retrieves the quantized values from the `block_q4_0` structure based on the provided index `il`.
    - It calculates two scaling factors, `d1` and `d2`, based on the quantization level.
    - A loop iterates over the quantized values, applying the scaling factors and storing the results in the `reg` array.
- **Output**: The function outputs a dequantized float array stored in the `reg` variable, which contains the processed values based on the quantized input.


---
### dequantize\_f16\_t4
The `dequantize_f16_t4` function converts a 4-element half-precision floating-point vector from a quantized format to a floating-point format.
- **Inputs**:
    - `src`: A pointer to a device array of `half` values representing the quantized input data.
    - `il`: An integer indicating the index of the quantized block to be processed.
    - `reg`: A reference to a thread-local variable of type `type4` where the dequantized output will be stored.
- **Control Flow**:
    - The function retrieves the quantized data from the `src` pointer.
    - It then directly assigns the retrieved data to the `reg` variable, effectively converting the quantized format to a floating-point format.
- **Output**: The function does not return a value; instead, it modifies the `reg` variable to hold the dequantized floating-point representation of the input data.


---
### dequantize\_q5\_0\_t4
The `dequantize_q5_0_t4` function dequantizes a quantized tensor block of type `block_q5_0` into a float array.
- **Inputs**:
    - `xb`: A pointer to a `block_q5_0` structure that contains the quantized data and scaling factors.
    - `il`: An integer indicating the index of the quantization level to be processed.
    - `reg`: A reference to a thread-local float array where the dequantized values will be stored.
- **Control Flow**:
    - The function retrieves the quantized values from the `block_q5_0` structure.
    - It calculates the scaling factors based on the input level `il`.
    - A loop iterates over the quantized values, extracting bits and applying the scaling factors to compute the dequantized values.
    - The results are stored in the provided `reg` array.
- **Output**: The function does not return a value; instead, it populates the `reg` array with the dequantized float values.


---
### kernel\_mul\_mv\_ext\_q4\_f32\_impl
The `kernel_mul_mv_ext_q4_f32_impl` function performs matrix-vector multiplication for quantized 4-bit representations using Metal shading language.
- **Inputs**:
    - `args`: A constant structure containing parameters for the matrix-vector multiplication operation, including dimensions and offsets.
    - `src0`: A pointer to the source matrix in quantized format (4-bit) that will be multiplied.
    - `src1`: A pointer to the source vector (float) that will be multiplied with the matrix.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the position of the SIMD group within the thread group.
- **Control Flow**:
    - The function begins by defining the number of chunks per thread and calculating the number of rows and columns based on the input dimensions.
    - It calculates the offsets for accessing the source matrix and vector based on the current thread's position.
    - The function then loads the quantized matrix and the vector into local variables.
    - It performs the matrix-vector multiplication in chunks, accumulating results in a local array.
    - Finally, it writes the accumulated results to the destination array, ensuring proper synchronization between threads.
- **Output**: The output is a matrix-vector multiplication result stored in the destination pointer, represented in floating-point format.


---
### kernel\_group\_norm
The `kernel_group_norm` function performs group normalization on a tensor, adjusting the values based on the mean and variance calculated over specified groups.
- **Inputs**:
    - `src0`: A pointer to the input tensor data that needs to be normalized.
    - `dst`: A pointer to the output tensor where the normalized values will be stored.
    - `args`: A structure containing parameters for normalization, including the number of groups and dimensions of the tensor.
    - `buf`: A threadgroup buffer used for intermediate calculations, specifically for storing partial sums.
    - `tgpig`: A 3D index indicating the position of the threadgroup in the grid.
    - `tpitg`: A 1D index indicating the position of the thread within the threadgroup.
    - `sgitg`: An index indicating the position of the thread within the SIMD group.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `ntg`: A 1D array indicating the number of threads per threadgroup.
- **Control Flow**:
    - Calculate the total number of elements in the tensor and the group size based on the number of groups.
    - Determine the start and end indices for the current threadgroup based on its position.
    - Initialize a temporary variable to hold the sum of the elements for normalization.
    - Iterate over the elements assigned to the current threadgroup, summing their values.
    - Synchronize the threadgroup to ensure all threads have completed their calculations.
    - Calculate the mean of the summed values.
    - Iterate again over the elements to compute the variance based on the mean.
    - Calculate the scale factor for normalization using the variance.
    - Apply the normalization to each element and store the result in the output tensor.
- **Output**: The function outputs a tensor with normalized values, where each group of elements has been adjusted based on the calculated mean and variance.


---
### kernel\_gelu
The `kernel_gelu` function applies the Gaussian Error Linear Unit (GELU) activation function to each element of the input tensor.
- **Inputs**:
    - `src0`: A pointer to the input tensor containing the values to which the GELU function will be applied.
    - `dst`: A pointer to the output tensor where the results of the GELU function will be stored.
    - `tpig`: An index representing the position of the current thread in the grid.
- **Control Flow**:
    - The function retrieves the input value at the current thread's position.
    - It computes the GELU activation using the formula: 0.5 * x * (1 + tanh(SQRT_2_OVER_PI * x * (1 + GELU_COEF_A * x * x))).
    - The result is stored in the output tensor at the same position.
- **Output**: The output tensor contains the results of applying the GELU activation function to each element of the input tensor.


---
### dequantize\_q2\_K
The `dequantize_q2_K` function converts quantized data from a `block_q2_K` structure into floating-point values.
- **Inputs**:
    - `xb`: A pointer to a `block_q2_K` structure containing the quantized data and scaling information.
    - `il`: An integer indicating the index of the quantization level to be processed.
    - `reg`: A reference to a thread-local variable where the dequantized floating-point values will be stored.
- **Control Flow**:
    - The function retrieves the scaling factor and minimum value from the `block_q2_K` structure.
    - It calculates the appropriate quantization parameters based on the index `il`.
    - A loop iterates over the quantized values, applying the scaling and minimum adjustments to convert them to floating-point values.
    - The results are stored in the `reg` variable.
- **Output**: The function does not return a value; instead, it populates the `reg` variable with the dequantized floating-point values.


---
### dequantize\_q4\_1
The `dequantize_q4_1` function converts quantized data from a `block_q4_1` structure into floating-point representation.
- **Inputs**:
    - `xb`: A pointer to a `block_q4_1` structure containing the quantized data to be dequantized.
    - `il`: An integer indicating the index of the quantized data block to process.
    - `reg`: A reference to a thread-local variable where the resulting floating-point data will be stored.
- **Control Flow**:
    - The function retrieves the quantized values from the `block_q4_1` structure.
    - It calculates scaling factors based on the input index `il` and the parameters stored in `xb`.
    - A loop iterates over the quantized values, applying the scaling factors and offsets to convert them to floating-point values.
    - The results are stored in a 4x4 floating-point matrix, which is then assigned to the output reference `reg`.
- **Output**: The function outputs a 4x4 matrix of floating-point values representing the dequantized data.


---
### kernel\_argmax
The `kernel_argmax` function computes the indices of the maximum values across rows of a given input tensor.
- **Inputs**:
    - `x`: A device pointer to the input tensor containing the values from which to find the maximum.
    - `dst`: A device pointer to the output tensor where the indices of the maximum values will be stored.
    - `ncols`: A constant reference to the number of columns in the input tensor.
    - `nb01`: A constant reference to the byte offset for the second dimension of the input tensor.
    - `shared_maxval`: A threadgroup shared memory array to hold the maximum values found by each thread.
    - `shared_argmax`: A threadgroup shared memory array to hold the indices of the maximum values found by each thread.
    - `tgpig`: A uint3 variable representing the position of the threadgroup in the grid.
    - `tpitg`: A uint variable representing the position of the thread within the threadgroup.
    - `sgitg`: A uint variable representing the index of the SIMD group within the threadgroup.
    - `tiisg`: A uint variable representing the index of the thread within the SIMD group.
    - `ntg`: A uint variable representing the number of threads per threadgroup.
- **Control Flow**:
    - The function begins by calculating the row pointer for the current threadgroup based on its index.
    - It initializes local variables to hold the maximum value and its corresponding index.
    - Each thread iterates over the columns of the input tensor, updating the local maximum and index if a larger value is found.
    - After processing, the maximum value found by each thread is reduced using SIMD operations to find the overall maximum.
    - The final maximum value and its index are stored in the output tensor.
- **Output**: The function outputs the index of the maximum value found in each row of the input tensor, stored in the `dst` tensor.


---
### kernel\_clamp
The `kernel_clamp` function clamps the values of a source array to a specified minimum and maximum range.
- **Inputs**:
    - `src0`: A pointer to the input array of floats that will be clamped.
    - `dst`: A pointer to the output array where the clamped values will be stored.
    - `min`: A constant float value representing the minimum threshold for clamping.
    - `max`: A constant float value representing the maximum threshold for clamping.
    - `tpig`: A thread position index in the grid, used to identify the specific thread's position.
- **Control Flow**:
    - The function iterates over the elements of the input array using the thread index.
    - For each element, it checks if the value is less than the minimum; if so, it sets the value to the minimum.
    - If the value is greater than the maximum, it sets the value to the maximum.
    - Otherwise, it retains the original value.
- **Output**: The output is an array of floats where each element has been clamped to the specified minimum and maximum values.


---
### kernel\_mul\_mv\_q8\_0\_f32
The `kernel_mul_mv_q8_0_f32` function performs matrix-vector multiplication for quantized 8-bit data.
- **Inputs**:
    - `args`: A constant structure containing parameters for the matrix-vector multiplication operation.
    - `src0`: A pointer to the source matrix data, which is quantized in the `block_q8_0` format.
    - `src1`: A pointer to the source vector data, which is expected to be in float format.
    - `dst`: A pointer to the destination buffer where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the position of the SIMD group within the thread group.
- **Control Flow**:
    - The function begins by calculating the number of blocks in the source matrix based on the input dimensions.
    - It then iterates over the rows of the matrix, processing each row in chunks defined by the SIMD width.
    - For each chunk, it retrieves the corresponding quantized values from the source matrix and the float values from the source vector.
    - The function computes the dot product of the quantized values and the float vector, accumulating the results.
    - Finally, it stores the computed results in the destination buffer.
- **Output**: The output is stored in the destination buffer, which contains the results of the matrix-vector multiplication.


---
### rope\_yarn
`rope_yarn` computes the rotary embeddings for a given position and frequency scaling.
- **Inputs**:
    - `theta_extrap`: The extrapolated angle used for calculating the rotary embeddings.
    - `freq_scale`: The frequency scaling factor applied to the angle.
    - `corr_dims`: An array containing correction dimensions for the embeddings.
    - `i0`: The current index used for ramping the correction.
    - `ext_factor`: The external factor used for adjusting the embeddings.
    - `mscale`: The magnitude scaling factor applied to the cosine and sine values.
    - `cos_theta`: A pointer to store the computed cosine value.
    - `sin_theta`: A pointer to store the computed sine value.
- **Control Flow**:
    - The function first calculates the interpolated angle based on the frequency scale and extrapolated angle.
    - If the external factor is not zero, it computes a ramp mix based on the correction dimensions and adjusts the angle accordingly.
    - The magnitude scaling is adjusted based on the frequency scale.
    - Finally, the cosine and sine of the adjusted angle are computed and scaled by the magnitude scaling factor.
- **Output**: The function outputs the computed cosine and sine values through the provided pointers.


---
### kernel\_cpy
The `kernel_cpy` function copies data from a source tensor to a destination tensor, potentially applying a transformation based on the specified quantization format.
- **Inputs**:
    - `src0`: A pointer to the source tensor data that is to be copied.
    - `dst`: A pointer to the destination tensor where the data will be copied.
    - `args`: A structure containing metadata about the tensor dimensions and offsets for the copy operation.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tpitg`: A 3D vector indicating the position of the thread within the thread group.
    - `ntg`: A 3D vector indicating the number of threads per thread group.
- **Control Flow**:
    - The function calculates the linear index `n` based on the thread group and thread indices.
    - It computes the corresponding multi-dimensional indices (i3, i2, i1, i0) based on the linear index.
    - It retrieves the source data pointer based on the calculated indices.
    - The function iterates over the destination tensor, copying data from the source tensor to the destination tensor.
- **Output**: The function does not return a value; instead, it modifies the destination tensor in place with the copied data.


---
### dequantize\_iq1\_m
The `dequantize_iq1_m` function performs dequantization of input data from a specific quantization format to floating-point representation.
- **Inputs**:
    - `xb`: A pointer to a `block_iq1_m` structure that contains the quantized data and scaling information.
    - `il`: An integer indicating the index of the quantization block being processed.
    - `reg`: A reference to a thread-local variable where the dequantized floating-point values will be stored.
- **Control Flow**:
    - The function begins by calculating the index of the quantization block based on the input index `il`.
    - It retrieves the scaling factors from the `sc` array in the `xb` structure.
    - The function then computes the dequantization parameters, including the scaling factor and the minimum value.
    - It iterates over the quantized values, applying the dequantization formula to convert them to floating-point values.
    - Finally, the computed values are stored in the `reg` variable.
- **Output**: The output is a set of dequantized floating-point values stored in the `reg` variable, which can be used for further processing.


---
### kernel\_silu\_4
The `kernel_silu_4` function applies the Sigmoid-Weighted Linear Unit (SiLU) activation function to a 4-component vector.
- **Inputs**:
    - `src0`: A pointer to the input tensor of type `float4`, which contains the values to be transformed by the SiLU function.
    - `dst`: A pointer to the output tensor of type `float4`, where the results of the SiLU transformation will be stored.
    - `tpig`: A thread position index in the grid, used to identify the specific thread executing the kernel.
- **Control Flow**:
    - The function retrieves the input vector from `src0` at the position indicated by `tpig`.
    - It computes the SiLU activation for each component of the input vector using the formula: `x / (1.0 + exp(-x))`.
    - The results are stored in the output vector `dst` at the same position.
- **Output**: The output is a `float4` vector containing the transformed values after applying the SiLU activation function.


---
### dequantize\_q4\_1\_t4
The `dequantize_q4_1_t4` function dequantizes a quantized tensor block of type `block_q4_1` into a thread-local register of type `type4`.
- **Inputs**:
    - `xb`: A pointer to a `block_q4_1` structure that contains the quantized data and scaling factors.
    - `il`: An integer indicating the index of the quantization level to be processed.
    - `reg`: A reference to a thread-local variable of type `type4` where the dequantized values will be stored.
- **Control Flow**:
    - The function retrieves the quantized values from the `block_q4_1` structure based on the provided index `il`.
    - It calculates the scaling factors `d1`, `d2`, and `m` based on the values in the `block_q4_1` structure.
    - It determines the appropriate masks for the quantized values based on the index `il`.
    - The function then iterates over the quantized values, applying the scaling factors and masks to compute the dequantized values.
    - Finally, the computed values are stored in the provided thread-local register `reg`.
- **Output**: The function does not return a value; instead, it populates the `reg` variable with the dequantized values.


---
### kernel\_cos
The `kernel_cos` function computes the cosine of each element in the input array and stores the results in the output array.
- **Inputs**:
    - `src0`: A device pointer to the input array containing the values for which the cosine will be computed.
    - `dst`: A device pointer to the output array where the computed cosine values will be stored.
    - `tpig`: A thread position index in the grid, used to identify the specific thread's position for computation.
- **Control Flow**:
    - The function iterates over each element in the input array using the thread index.
    - For each element, it computes the cosine using the `cos` function.
    - The computed cosine value is then stored in the corresponding position in the output array.
- **Output**: The output is an array of the same size as the input, containing the cosine of each input value.


---
### kernel\_flash\_attn\_ext\_vec
The `kernel_flash_attn_ext_vec` function implements a vectorized flash attention mechanism for processing queries, keys, and values in a neural network context.
- **Inputs**:
    - `args`: A constant structure containing parameters for the flash attention operation, including dimensions and scaling factors.
    - `q`: A device pointer to the query tensor data.
    - `k`: A device pointer to the key tensor data.
    - `v`: A device pointer to the value tensor data.
    - `mask`: A device pointer to the mask tensor data, used to prevent attention to certain positions.
    - `dst`: A device pointer to the output tensor where the results of the attention operation will be stored.
    - `shmem_f16`: A threadgroup shared memory buffer for storing intermediate results.
    - `tgpig`: A 3D vector representing the position of the threadgroup in the grid.
    - `ntg`: A 3D vector representing the number of threads per threadgroup.
    - `tiisg`: The index of the thread within the SIMD group.
    - `sgitg`: The index of the SIMD group.
- **Control Flow**:
    - The function begins by extracting the indices for the current threadgroup and initializing shared memory for queries and results.
    - It loads the query data into shared memory and initializes the output tensor.
    - The function then enters a loop to process the key and value tensors, performing the attention calculations.
    - For each query, it computes the attention scores using the keys and applies the mask if provided.
    - The results are accumulated in shared memory and then reduced to obtain the final attention output.
    - Finally, the results are written to the output tensor.
- **Output**: The output is a tensor containing the results of the attention mechanism, which combines the input queries, keys, and values based on the computed attention scores.


---
### kernel\_mul\_mv\_q4\_0\_f32
The `kernel_mul_mv_q4_0_f32` function performs matrix-vector multiplication for quantized 4-bit data using a specific kernel implementation.
- **Inputs**:
    - `args`: A constant structure containing the arguments for the matrix-vector multiplication, including dimensions and offsets.
    - `src0`: A pointer to the source matrix data, which is quantized in the `block_q4_0` format.
    - `src1`: A pointer to the source vector data, which is expected to be in a float format.
    - `dst`: A pointer to the destination buffer where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the position of the SIMD group within the thread group.
- **Control Flow**:
    - The function begins by calling a helper function `mul_vec_q_n_f32_impl` with the appropriate parameters for processing the matrix-vector multiplication.
    - Inside `mul_vec_q_n_f32_impl`, the function calculates offsets for accessing the source data based on the provided indices.
    - It then iterates over the quantized blocks of the source matrix, performing the multiplication with the source vector and accumulating the results.
    - The results are stored in the destination buffer, ensuring that the correct scaling and quantization are applied.
- **Output**: The output is stored in the destination buffer pointed to by `dst`, which contains the results of the matrix-vector multiplication in a float format.


---
### kernel\_scale
The `kernel_scale` function scales the input tensor by a specified constant factor.
- **Inputs**:
    - `src0`: A pointer to the input tensor that will be scaled.
    - `dst`: A pointer to the output tensor where the scaled values will be stored.
    - `scale`: A constant float reference that specifies the scaling factor.
    - `tpig`: A uint value representing the thread position in the grid.
- **Control Flow**:
    - The function retrieves the value at the position indicated by `tpig` from the input tensor `src0`.
    - It multiplies this value by the `scale` factor.
    - The result is stored in the output tensor `dst` at the same position indicated by `tpig`.
- **Output**: The output is the scaled tensor stored in `dst`, where each element is the corresponding element from `src0` multiplied by `scale`.


---
### kernel\_add
The `kernel_add` function performs element-wise addition of two tensors, supporting broadcasting and non-contiguous memory layouts.
- **Inputs**:
    - `args`: A constant reference to a structure containing metadata about the tensor dimensions and offsets.
    - `src0`: A pointer to the first input tensor in device memory.
    - `src1`: A pointer to the second input tensor in device memory.
    - `dst`: A pointer to the output tensor in device memory where the result will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tpitg`: A 3D vector indicating the position of the thread within the thread group.
    - `ntg`: A 3D vector indicating the number of threads per thread group.
- **Control Flow**:
    - The function calculates the indices for accessing the input tensors based on the thread group and thread positions.
    - It computes the pointers for the source tensors (`src0` and `src1`) and the destination tensor (`dst`) using the calculated indices and offsets.
    - A loop iterates over the elements of the tensors, performing the addition operation for each element and storing the result in the destination tensor.
- **Output**: The output is stored in the `dst` tensor, which contains the result of the element-wise addition of `src0` and `src1`.


---
### dequantize\_iq2\_s
The `dequantize_iq2_s` function dequantizes input data from a specific quantization format into a floating-point representation.
- **Inputs**:
    - `xb`: A pointer to a `block_iq2_s` structure containing the quantized data and associated parameters.
    - `il`: An integer indicating the index of the quantization block being processed.
    - `reg`: A reference to a thread-local variable where the dequantized output will be stored.
- **Control Flow**:
    - The function calculates the scaling factor `dl` based on the quantization parameters.
    - It retrieves the quantized values from the input structure `xb` and processes them in pairs.
    - For each pair of quantized values, it applies the scaling factor and stores the result in the output register `reg`.
- **Output**: The function outputs the dequantized floating-point values into the `reg` variable, which can be used for further processing.


---
### kernel\_mul\_mv\_q5\_0\_f32
The `kernel_mul_mv_q5_0_f32` function performs matrix-vector multiplication for quantized data using a specific quantization scheme (Q5_0).
- **Inputs**:
    - `args`: A constant structure containing parameters for the matrix-vector multiplication operation, including dimensions and offsets.
    - `src0`: A pointer to the source matrix data, which is quantized using the Q5_0 scheme.
    - `src1`: A pointer to the source vector data that will be multiplied with the matrix.
    - `dst`: A pointer to the destination memory where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the position of the SIMD group within the thread group.
- **Control Flow**:
    - The function begins by calling the `mul_vec_q_n_f32_impl` template function, which handles the actual multiplication logic.
    - It calculates the number of blocks and the current row based on the thread group indices.
    - It retrieves the source vector and matrix data from the provided pointers.
    - The function then iterates over the quantized matrix blocks, performing the multiplication with the vector and accumulating the results.
    - Finally, it stores the computed results in the destination memory.
- **Output**: The output is stored in the destination pointer `dst`, which contains the result of the matrix-vector multiplication.


---
### kernel\_rope\_neox
The `kernel_rope_neox` function applies a rotary positional encoding transformation to input tensors based on specified parameters.
- **Inputs**:
    - `args`: A constant structure containing parameters for the rotary encoding, including dimensions and frequency scaling.
    - `src0`: A device pointer to the input tensor that will be transformed.
    - `src1`: A device pointer to the tensor containing positional indices.
    - `src2`: A device pointer to an optional tensor for frequency scaling.
    - `dst`: A device pointer to the output tensor where the transformed values will be stored.
    - `tiitg`: The thread index within the thread group.
    - `tptg`: The thread position in the thread group.
    - `tgpig`: The thread group position in the grid.
- **Control Flow**:
    - The function begins by calculating correction dimensions based on the input parameters.
    - It retrieves the base angle from the positional indices and computes the inverse of the number of dimensions.
    - For each thread, it iterates over the dimensions, applying the rotary encoding transformation based on the calculated angles.
    - The cosine and sine values of the angles are computed and used to transform the input tensor values.
    - Finally, the transformed values are stored in the output tensor.
- **Output**: The output is a tensor containing the transformed values after applying the rotary positional encoding.


---
### kernel\_repeat
The `kernel_repeat` function replicates elements from a source tensor across a specified dimension.
- **Inputs**:
    - `args`: A constant structure containing metadata for the repeat operation, including the dimensions and offsets.
    - `src0`: A pointer to the source tensor from which elements are to be repeated.
    - `dst`: A pointer to the destination tensor where the repeated elements will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tpitg`: A 3D vector indicating the position of the thread within the thread group.
    - `ntg`: A 3D vector indicating the number of threads per thread group.
- **Control Flow**:
    - The function calculates the indices for the source and destination tensors based on the thread group and thread indices.
    - It uses modulo operations to determine the appropriate source element to replicate based on the repeat dimensions.
    - A loop iterates over the number of elements to be repeated, copying the appropriate source element to the destination tensor.
- **Output**: The function does not return a value; instead, it modifies the destination tensor in place with the repeated elements from the source tensor.


---
### kernel\_sqrt
The `kernel_sqrt` function computes the square root of each element in the input array and stores the results in the output array.
- **Inputs**:
    - `src0`: A device pointer to the input array containing the values for which the square root is to be calculated.
    - `dst`: A device pointer to the output array where the results of the square root calculations will be stored.
    - `tpig`: A thread position index in the grid, used to identify the specific thread executing the kernel.
- **Control Flow**:
    - The function iterates over each element in the input array using the thread index to determine which element to process.
    - For each element, it calculates the square root using the `sqrt` function.
    - The result is then stored in the corresponding position in the output array.
- **Output**: The output is an array containing the square roots of the input values, with each element corresponding to the input element at the same index.


---
### dequantize\_f32
The `dequantize_f32` function converts quantized float values from a source pointer to a specified register type.
- **Inputs**:
    - `src`: A pointer to the source data of type `float4x4` that contains the quantized float values.
    - `il`: An integer indicating the index or location within the quantized data to be processed.
    - `reg`: A reference to a thread-local variable of type `type4x4` where the dequantized values will be stored.
- **Control Flow**:
    - The function begins by dereferencing the `src` pointer to access the quantized float values.
    - The dereferenced values are then cast to the specified register type and assigned to `reg`.
- **Output**: The function does not return a value; instead, it modifies the `reg` variable to hold the dequantized float values.


---
### kernel\_cpy\_f32\_q4\_1
The `kernel_cpy_f32_q4_1` function copies data from a source tensor to a destination tensor while quantizing the values into a specific format.
- **Inputs**:
    - `args`: A structure containing the necessary parameters for the copy operation, including dimensions and offsets.
    - `src0`: A pointer to the source tensor data that will be copied and quantized.
    - `dst`: A pointer to the destination tensor where the quantized data will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tpitg`: A 3D vector indicating the position of the thread within the thread group.
    - `ntg`: A 3D vector indicating the number of threads per thread group.
- **Control Flow**:
    - The function calculates the linear index `n` based on the thread group and tensor dimensions.
    - It computes the indices for the source and destination tensors based on the calculated linear index.
    - The function iterates over the quantization blocks, copying data from the source tensor to the destination tensor.
    - For each block, it calculates the maximum value to determine the quantization scale.
    - The values are quantized and stored in the destination tensor along with the scale factor.
- **Output**: The function does not return a value; instead, it modifies the destination tensor in place with the quantized data.


---
### dequantize\_iq1\_s
The `dequantize_iq1_s` function performs dequantization of input data from a specific quantization format to floating-point representation.
- **Inputs**:
    - `xb`: A pointer to a `block_iq1_s` structure that contains the quantized data and scaling information.
    - `il`: An integer indicating the index of the block being processed, which determines how the quantized data is accessed.
    - `reg`: A reference to a thread-local variable where the dequantized floating-point values will be stored.
- **Control Flow**:
    - The function calculates the index of the block and retrieves the quantized data from the `xb` structure.
    - It computes the scaling factor based on the quantization parameters.
    - The function then iterates over the quantized values, applying the scaling and offset to convert them to floating-point values.
    - Finally, the dequantized values are stored in the `reg` variable.
- **Output**: The output is a set of dequantized floating-point values stored in the `reg` variable, which can be used for further processing.


---
### kernel\_gelu\_4
The `kernel_gelu_4` function applies the Gaussian Error Linear Unit (GELU) activation function to a 4-component vector.
- **Inputs**:
    - `src0`: A pointer to a device array of `float4` values representing the input vector to which the GELU function will be applied.
    - `dst`: A pointer to a device array of `float4` values where the result of the GELU function will be stored.
    - `tpig`: A thread position index in the grid, used to identify the specific thread's position for processing.
- **Control Flow**:
    - The function retrieves the input vector from `src0` using the thread index `tpig`.
    - It computes the GELU activation using the formula: `0.5 * x * (1 + tanh(SQRT_2_OVER_PI * x * (1 + GELU_COEF_A * x * x)))`.
    - The result is stored in the `dst` array at the same index as the input.
- **Output**: The output is a `float4` vector containing the result of applying the GELU activation function to the input vector.


---
### block\_q\_n\_dot\_y
Calculates the inner product between a quantized block and a vector.
- **Inputs**:
    - `qb_curr`: A pointer to the current quantized block structure.
    - `sumy`: The sum of the vector elements.
    - `yl`: A thread-local array of floats representing the vector.
    - `il`: An integer indicating the starting index for quantization.
- **Control Flow**:
    - The function retrieves the scaling factor 'd' from the quantized block.
    - It initializes an accumulator array 'acc' to zero.
    - It retrieves the quantized values from the block based on the input index 'il'.
    - For each pair of quantized values, it computes the weighted sum with the corresponding elements from 'yl'.
    - Finally, it returns the computed inner product adjusted by the scaling factor.
- **Output**: Returns the computed inner product as a float.


---
### kernel\_rope\_norm
The `kernel_rope_norm` function applies a rotary positional encoding transformation to input tensors based on specified parameters.
- **Inputs**:
    - `args`: A constant structure containing parameters for the rotary encoding, including dimensions and scaling factors.
    - `src0`: A device pointer to the input tensor that will be transformed.
    - `src1`: A device pointer to the tensor containing positional indices.
    - `src2`: A device pointer to an optional tensor used for frequency scaling.
    - `dst`: A device pointer to the output tensor where the transformed values will be stored.
    - `tiitg`: An index for the thread within the thread group.
    - `tptg`: A 3D vector representing the thread position in the thread group.
    - `tgpig`: A 3D vector representing the thread group position in the grid.
- **Control Flow**:
    - The function begins by calculating correction dimensions based on the input parameters.
    - It retrieves the base angle from the positional indices and computes the inverse of the number of dimensions.
    - For each thread, it calculates the angle for the rotary encoding based on the positional index and frequency scaling.
    - The cosine and sine values of the calculated angle are computed and stored.
    - The function then applies the rotary transformation to the input tensor, modifying the output tensor accordingly.
- **Output**: The output is a tensor containing the transformed values after applying the rotary positional encoding.


---
### kernel\_timestep\_embedding\_f32
The `kernel_timestep_embedding_f32` function computes a timestep embedding for a given input tensor.
- **Inputs**:
    - `src0`: A device pointer to the input tensor containing the timestep values.
    - `dst`: A device pointer to the output tensor where the computed embeddings will be stored.
    - `args`: A constant structure containing parameters for the embedding calculation, including the maximum period and dimension.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tpitg`: A 3D vector indicating the position of the thread within the thread group.
    - `ntg`: A 3D vector indicating the number of threads per thread group.
- **Control Flow**:
    - The function starts by determining the index of the current timestep from the input tensor.
    - It calculates the half dimension of the embedding size.
    - For each thread, it computes the cosine and sine of the product of the timestep and frequency, storing the results in the output tensor.
    - If the embedding dimension is odd, the last element is set to zero.
- **Output**: The output is a tensor containing the computed timestep embeddings, with each embedding consisting of cosine and sine values based on the input timestep.


---
### dequantize\_iq4\_nl\_t4
The `dequantize_iq4_nl_t4` function dequantizes a block of quantized data from a specific format into a floating-point representation.
- **Inputs**:
    - `xb`: A pointer to a `block_iq4_nl` structure that contains the quantized data and scaling information.
    - `il`: An integer indicating the index of the quantization level to be processed.
    - `reg`: A thread-local reference to an array of floats where the dequantized values will be stored.
- **Control Flow**:
    - The function begins by extracting the quantized data from the `xb` structure.
    - It calculates the scaling factor `d` from the `d` field of the `xb` structure.
    - A loop iterates over the four rows of the quantized data, extracting and dequantizing each value based on the quantization scheme.
    - The dequantized values are computed using a lookup table `kvalues_iq4nl_f` and stored in the `reg` array.
- **Output**: The function outputs the dequantized floating-point values into the `reg` array, which can then be used for further processing.


---
### rope\_yarn\_ramp
`rope_yarn_ramp` computes a ramp function based on input parameters to facilitate interpolation in a rotary embedding context.
- **Inputs**:
    - `low`: The lower bound for the ramp function.
    - `high`: The upper bound for the ramp function.
    - `i0`: An integer index used to determine the position within the ramp.
- **Control Flow**:
    - Calculates the normalized value `y` based on the input index `i0` and the provided bounds `low` and `high`.
    - Applies clamping to ensure `y` is within the range [0, 1].
    - Returns the computed ramp value as `1.0 - min(1.0, max(0.0, y))`.
- **Output**: Returns a float representing the ramp value, which is used for scaling in rotary embeddings.


---
### dequantize\_q5\_1\_t4
The `dequantize_q5_1` function converts quantized data from a `block_q5_1` structure into floating-point format.
- **Inputs**:
    - `xb`: A pointer to a `block_q5_1` structure containing the quantized data.
    - `il`: An integer indicating the index of the quantized data to be processed.
    - `reg`: A reference to a thread-local variable where the dequantized floating-point data will be stored.
- **Control Flow**:
    - The function retrieves the quantized data from the `block_q5_1` structure.
    - It calculates the scaling factors based on the input index `il`.
    - The function iterates over the quantized data, applying the scaling factors and storing the results in the `reg` variable.
- **Output**: The function does not return a value; instead, it populates the `reg` variable with the dequantized floating-point data.


---
### dequantize\_q4\_K
The `dequantize_q4_K` function performs dequantization of a quantized tensor block using specific scaling and masking techniques.
- **Inputs**:
    - `xb`: A pointer to a `block_q4_K` structure that contains the quantized data and scaling factors.
    - `il`: An integer indicating the index of the quantization level to be processed.
    - `reg`: A reference to a thread-local variable where the dequantized results will be stored.
- **Control Flow**:
    - The function begins by calculating the starting address of the quantized data based on the input block pointer `xb`.
    - It then determines the scaling factors for the dequantization process based on the input index `il`.
    - A loop iterates over the quantized data, applying the appropriate scaling and masking to each element.
    - The results are accumulated into a temporary register before being stored in the output variable `reg`.
- **Output**: The function outputs the dequantized tensor data into the `reg` variable, which is used for further processing in the neural network.


---
### dequantize\_bf16
The `dequantize_bf16` function converts a block of bfloat16 data into a specified type, typically for further processing in a neural network context.
- **Inputs**:
    - `src`: A pointer to a device memory block containing bfloat16 data to be dequantized.
    - `il`: An index indicating the specific location or layer of data being processed.
    - `reg`: A reference to a thread-local variable where the dequantized result will be stored.
- **Control Flow**:
    - The function begins by dereferencing the `src` pointer to access the bfloat16 data.
    - It then assigns the data from `src` directly to `reg`, effectively converting the bfloat16 representation to the specified type.
    - The function does not perform any additional calculations or transformations on the data.
- **Output**: The output is the dequantized data stored in the `reg` variable, which can be used for further computations.


---
### kernel\_rope\_multi
The `kernel_rope_multi` function applies a rotary positional encoding to input tensors based on specified parameters.
- **Inputs**:
    - `args`: A constant structure containing parameters for the rotary encoding, including dimensions and frequency scaling.
    - `src0`: A device pointer to the input tensor that will be transformed with rotary encoding.
    - `src1`: A device pointer to the tensor containing positional indices.
    - `src2`: A device pointer to an optional tensor used for frequency scaling.
    - `dst`: A device pointer to the output tensor where the results of the rotary encoding will be stored.
    - `tiitg`: An index for the thread within the thread group.
    - `tptg`: A 3D vector representing the position of the thread within the thread group.
    - `tgpig`: A 3D vector representing the position of the thread group within the grid.
- **Control Flow**:
    - The function begins by calculating correction dimensions based on the input parameters.
    - It retrieves the base angle from the positional indices and computes the inverse of the number of dimensions.
    - For each thread, it calculates the angle for the rotary encoding based on the input tensor and the frequency scaling.
    - The cosine and sine values of the calculated angle are computed and stored in the output tensor.
    - If the current index exceeds the number of dimensions, the function simply copies the input value to the output tensor.
- **Output**: The output tensor contains the transformed values after applying the rotary positional encoding, with the same shape as the input tensor.


---
### kernel\_gelu\_quick\_4
The `kernel_gelu_quick_4` function applies the GELU activation function to a vector of `float4` values using a quick approximation.
- **Inputs**:
    - `src0`: A pointer to the input vector of `float4` values that will be processed by the GELU activation function.
    - `dst`: A pointer to the output vector of `float4` values where the results of the GELU activation will be stored.
    - `tpig`: An index representing the position of the thread in the grid, used to access the appropriate element in the input and output vectors.
- **Control Flow**:
    - The function retrieves the `float4` value from the input vector `src0` at the position specified by `tpig`.
    - It computes the GELU activation using the formula: `dst[tpig] = x * (1.0f / (1.0f + exp(GELU_QUICK_COEF * x)));` where `x` is the input value.
    - The result is stored in the output vector `dst` at the same position `tpig`.
- **Output**: The output is a vector of `float4` values where each element has been transformed by the GELU activation function.


---
### kernel\_mul\_mv\_q8\_0\_f32\_impl
The `kernel_mul_mv_q8_0_f32_impl` function performs matrix-vector multiplication for quantized 8-bit data.
- **Inputs**:
    - `args`: A constant structure containing the parameters for the matrix-vector multiplication, including dimensions and offsets.
    - `src0`: A pointer to the source matrix data, which is quantized in 8-bit format.
    - `src1`: A pointer to the source vector data, which is in floating-point format.
    - `dst`: A pointer to the destination buffer where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the position of the SIMD group within the thread group.
- **Control Flow**:
    - The function begins by calculating the number of blocks and the current row and column indices based on the thread group indices.
    - It then calculates the offsets for accessing the source matrix and vector data.
    - The function initializes an array to hold the results of the multiplication.
    - It iterates over the quantized blocks of the matrix, performing the multiplication with the vector and accumulating the results.
    - Finally, it stores the computed results in the destination buffer.
- **Output**: The output is a matrix-vector product stored in the destination buffer, where each element corresponds to the result of multiplying a row of the quantized matrix with the input vector.


---
### kernel\_neg
The `kernel_neg` function negates each element of the input tensor and stores the result in the output tensor.
- **Inputs**:
    - `src0`: A device pointer to the input tensor containing the values to be negated.
    - `dst`: A device pointer to the output tensor where the negated values will be stored.
    - `tpig`: A thread position index in the grid, used to identify the specific thread's position for processing.
- **Control Flow**:
    - The function iterates over each element of the input tensor using the thread position index.
    - For each element, it negates the value and stores it in the corresponding position in the output tensor.
- **Output**: The output is a tensor where each element is the negation of the corresponding element in the input tensor.


---
### dequantize\_q3\_K
The `dequantize_q3_K` function performs dequantization of a quantized tensor block using specific scaling and masking techniques.
- **Inputs**:
    - `xb`: A pointer to a `block_q3_K` structure that contains the quantized data and associated parameters for dequantization.
    - `il`: An integer index indicating the specific quantization level or block to be processed.
    - `reg`: A reference to a thread-local variable where the dequantized results will be stored.
- **Control Flow**:
    - The function begins by extracting the scaling factors and quantized data from the `xb` structure.
    - It calculates the appropriate scaling and masking values based on the input index `il`.
    - A loop iterates over the quantized data, applying the scaling and masking to produce the dequantized values.
    - The results are stored in the `reg` variable, which is structured to hold the dequantized tensor.
- **Output**: The function outputs the dequantized tensor values into the `reg` variable, which can be used for further processing in the neural network.


---
### get\_scale\_min\_k4\_just2
The `get_scale_min_k4_just2` function retrieves scale and minimum values from a quantized data structure.
- **Inputs**:
    - `j`: An integer index used to determine the specific quantized values to retrieve.
    - `k`: An integer offset used to access the quantized data.
    - `q`: A pointer to the quantized data array.
- **Control Flow**:
    - The function checks if the index `j` is less than 4 to determine the retrieval method.
    - If `j` is less than 4, it retrieves values directly from the quantized data using bitwise operations.
    - If `j` is 4 or greater, it combines values from two different parts of the quantized data using bitwise shifts and masks.
- **Output**: Returns a `uchar2` structure containing the scale and minimum values extracted from the quantized data.


---
### kernel\_mul\_mv\_l4
The `kernel_mul_mv_l4` function performs matrix-vector multiplication for quantized data types in a Metal kernel.
- **Inputs**:
    - `args`: A constant structure containing the parameters for the matrix-vector multiplication, including dimensions and offsets.
    - `src0`: A pointer to the source matrix data, which is expected to be in a quantized format.
    - `src1`: A pointer to the source vector data, which is expected to be in a float format.
    - `dst`: A pointer to the destination memory where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
- **Control Flow**:
    - The function calculates the row index and the corresponding offsets for accessing the source data based on the input parameters.
    - It retrieves the quantized matrix from `src0` and the vector from `src1`.
    - For each row in the matrix, it computes the dot product with the vector and accumulates the results.
    - The results are stored in the destination memory `dst`.
- **Output**: The output is a matrix-vector multiplication result stored in the destination pointer `dst`, where each row of the matrix is multiplied by the vector.


---
### dequantize\_iq3\_s
The `dequantize_iq3_s` function performs dequantization of input data from a specific quantization format (IQ3) into a floating-point representation.
- **Inputs**:
    - `xb`: A pointer to a `block_iq3_s` structure that contains the quantized data and associated parameters for dequantization.
    - `il`: An integer index indicating the specific quantization block to process.
    - `reg`: A reference to a thread-local variable where the dequantized floating-point values will be stored.
- **Control Flow**:
    - The function begins by extracting the quantized data and parameters from the `block_iq3_s` structure.
    - It calculates the scaling factors based on the input parameters.
    - A loop iterates over the quantized values, applying the dequantization formula to convert the quantized values into floating-point values.
    - The results are stored in the `reg` variable, which is used for further processing.
- **Output**: The function does not return a value directly; instead, it populates the `reg` variable with the dequantized floating-point values.


---
### kernel\_l2\_norm
Calculates the L2 normalization of input tensors.
- **Inputs**:
    - `args`: A structure containing parameters for the L2 normalization operation, including dimensions and offsets.
    - `src0`: A pointer to the input tensor data that needs to be normalized.
    - `dst`: A pointer to the output tensor where the normalized data will be stored.
    - `shmem_f32`: Shared memory for intermediate calculations, specifically for storing partial sums.
    - `tgpig`: Thread group position in the grid, indicating the current thread group's coordinates.
    - `tpitg`: Thread position within the thread group, indicating the current thread's coordinates.
    - `sgitg`: SIMD group index within the thread group, used for SIMD operations.
    - `tiisg`: Thread index within the SIMD group, used for accessing shared memory.
    - `ntg`: Number of threads per thread group, used for parallel processing.
- **Control Flow**:
    - Initialize shared memory for partial sums.
    - Calculate the L2 norm by summing the squares of the input tensor elements.
    - Use a barrier to synchronize threads before reducing the sum in shared memory.
    - Calculate the mean and variance from the summed values.
    - Scale the input tensor by the calculated normalization factor and store the result in the output tensor.
- **Output**: The output tensor containing the L2 normalized values of the input tensor.


---
### rope\_yarn\_corr\_dims
The `rope_yarn_corr_dims` function calculates the correction dimensions for the YaRN algorithm based on the number of dimensions, original context size, frequency base, and two beta parameters.
- **Inputs**:
    - `n_dims`: An integer representing the number of dimensions for the correction.
    - `n_ctx_orig`: An integer representing the original context size.
    - `freq_base`: A float representing the base frequency used in calculations.
    - `beta_fast`: A float representing the fast beta parameter for correction.
    - `beta_slow`: A float representing the slow beta parameter for correction.
    - `dims`: An array of floats where the calculated correction dimensions will be stored.
- **Control Flow**:
    - The function first calculates the correction factor for both fast and slow beta parameters using the `rope_yarn_corr_factor` function.
    - It then assigns the calculated values to the first and second elements of the `dims` array, ensuring they are within the valid range of dimensions.
- **Output**: The function does not return a value but modifies the `dims` array to contain the calculated correction dimensions.


---
### kernel\_im2col\_ext
The `kernel_im2col_ext` function transforms a 4D input tensor into a 2D output tensor by applying a specified padding and stride, effectively preparing the data for convolution operations.
- **Inputs**:
    - `x`: A pointer to the input tensor data, which is a 4D tensor representing the input feature maps.
    - `dst`: A pointer to the output tensor data, where the transformed 2D data will be stored.
    - `args`: A structure containing the parameters for the operation, including dimensions, padding, stride, and other necessary configurations.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tgpg`: A 3D vector indicating the number of thread groups in the grid.
    - `tpitg`: A 3D vector indicating the position of the thread within the thread group.
    - `ntg`: A 3D vector indicating the number of threads per thread group.
- **Control Flow**:
    - The function begins by calculating the output dimensions based on the input dimensions, padding, and stride.
    - It then iterates over the input tensor, checking if the current indices are within the valid range after applying padding and stride.
    - If the indices are valid, it copies the corresponding input value to the output tensor at the calculated position.
    - If the indices are out of bounds, it assigns a default value (usually zero) to the output tensor.
- **Output**: The output is a 2D tensor that contains the transformed data from the input tensor, arranged according to the specified padding and stride.


---
### kernel\_gelu\_erf\_4
The `kernel_gelu_erf_4` function applies the Gaussian Error Linear Unit (GELU) activation function using the error function (erf) on a 4-component vector input.
- **Inputs**:
    - `src0`: A pointer to the input tensor of type `float4`, which contains the values to be transformed by the GELU activation function.
    - `dst`: A pointer to the output tensor of type `float4`, where the transformed values will be stored.
    - `tpig`: An index representing the position of the thread in the grid, used to determine which element of the input tensor to process.
- **Control Flow**:
    - The function retrieves the input value from `src0` at the position specified by `tpig`.
    - It computes the GELU activation using the formula: `0.5 * x * (1.0 + erf(x / sqrt(2)))`.
    - The result is stored in the output tensor `dst` at the same position.
- **Output**: The output is a transformed tensor of type `float4`, containing the results of applying the GELU activation function to the input values.


---
### dequantize\_q5\_1
The `dequantize_q5_1` function converts quantized data from a `block_q5_1` structure into floating-point representation.
- **Inputs**:
    - `xb`: A pointer to a `block_q5_1` structure containing the quantized data.
    - `il`: An integer indicating the index of the quantized data to be processed.
    - `reg`: A reference to a thread-local variable where the dequantized floating-point data will be stored.
- **Control Flow**:
    - The function retrieves the quantized data from the `block_q5_1` structure.
    - It calculates the scaling factors based on the quantization parameters.
    - It iterates over the quantized data, applying the scaling factors and storing the results in the `reg` variable.
- **Output**: The function outputs the dequantized floating-point values into the `reg` variable.


---
### kernel\_sub\_row
The `kernel_sub_row` function performs element-wise subtraction of a row vector from a matrix, broadcasting the row vector across the matrix.
- **Inputs**:
    - `args`: A constant structure containing the kernel arguments, including dimensions and offsets for the source and destination buffers.
    - `src0`: A device pointer to the first source matrix (float4) from which values will be subtracted.
    - `src1`: A device pointer to the second source row vector (float4) that will be subtracted from each row of the first source matrix.
    - `dst`: A device pointer to the destination matrix (float4) where the result of the subtraction will be stored.
    - `tpig`: A thread position index in the grid, indicating the specific thread's position in the grid.
- **Control Flow**:
    - The function calculates the number of elements in the row vector by dividing the total number of elements by 4 (since float4 contains 4 floats).
    - It then performs the subtraction operation for each element in the row vector from the corresponding elements in the matrix, using the thread index to determine which row to operate on.
    - The result of the subtraction is stored in the destination matrix at the appropriate index.
- **Output**: The output is a device pointer to the destination matrix (float4) containing the results of the element-wise subtraction.


---
### dequantize\_f16
The `dequantize_f16` function converts half-precision floating-point values from a source buffer into a specified register format.
- **Inputs**:
    - `src`: A pointer to a device buffer containing half-precision floating-point values to be dequantized.
    - `il`: An index indicating the location of the data to be processed within the source buffer.
    - `reg`: A reference to a thread-local variable where the dequantized values will be stored.
- **Control Flow**:
    - The function reads the half-precision values from the `src` buffer.
    - It then converts these values into the specified format and stores them in the `reg` variable.
- **Output**: The function does not return a value; instead, it modifies the `reg` variable to hold the dequantized values.


---
### kernel\_elu
The `kernel_elu` function applies the Exponential Linear Unit (ELU) activation function to each element of the input tensor.
- **Inputs**:
    - `src0`: A device pointer to the input tensor containing the values to which the ELU function will be applied.
    - `dst`: A device pointer to the output tensor where the results of the ELU function will be stored.
    - `tpig`: A thread position index in the grid, used to identify the specific thread's position for processing.
- **Control Flow**:
    - The function retrieves the input value from `src0` at the position indicated by `tpig`.
    - It checks if the input value is greater than zero; if so, it directly assigns it to the output.
    - If the input value is less than or equal to zero, it computes the ELU value using the formula: exp(x) - 1.
    - The computed value is then stored in the output tensor at the corresponding position.
- **Output**: The output tensor `dst` contains the results of applying the ELU activation function to each element of the input tensor `src0`.


---
### kernel\_arange\_f32
The `kernel_arange_f32` function generates a sequence of floating-point numbers in a specified range and stores them in a device memory buffer.
- **Inputs**:
    - `dst`: A pointer to the device memory where the generated sequence will be stored.
    - `args`: A structure containing parameters for the arange operation, including the start value, step size, and the number of elements to generate.
- **Control Flow**:
    - The function iterates over a range defined by the number of elements to generate (args.ne0).
    - For each index, it calculates the value based on the start value and step size, and stores it in the destination buffer.
- **Output**: The function does not return a value; instead, it populates the destination buffer with the generated sequence of floating-point numbers.


---
### kernel\_rms\_norm
The `kernel_rms_norm` function applies RMS normalization to a tensor, scaling its values based on the root mean square of the tensor's elements.
- **Inputs**:
    - `args`: A constant structure containing parameters for the RMS normalization, including the number of elements and offsets.
    - `src0`: A pointer to the input tensor data that will be normalized.
    - `dst`: A pointer to the output tensor where the normalized values will be stored.
    - `shmem_f32`: A threadgroup shared memory buffer for intermediate floating-point calculations.
    - `tgpig`: A 3D vector representing the position of the threadgroup in the grid.
    - `tpitg`: A 2D vector representing the position of the thread within the threadgroup.
    - `sgitg`: An index representing the position of the thread within the SIMD group.
    - `tiisg`: An index representing the position of the thread within the SIMD group.
    - `ntg`: A 3D vector representing the number of threads per threadgroup.
- **Control Flow**:
    - The function initializes shared memory for storing intermediate results.
    - It calculates the sum of squares of the elements in the input tensor using parallel reduction.
    - The mean of the squared values is computed and used to calculate the normalization scale.
    - The input tensor is then scaled by the calculated normalization factor and stored in the output tensor.
- **Output**: The output is a tensor with the same shape as the input tensor, where each element is normalized by the RMS value.


---
### kernel\_im2col
The `kernel_im2col` function transforms a 2D input tensor into a 2D column matrix suitable for convolution operations.
- **Inputs**:
    - `x`: A pointer to the input tensor data, which is a 2D array of floats.
    - `dst`: A pointer to the output destination where the transformed data will be stored.
    - `args`: A structure containing parameters for the im2col operation, including input dimensions and padding.
    - `tgpig`: A 3D vector indicating the position of the threadgroup in the grid.
    - `tgpg`: A 3D vector indicating the number of threadgroups in the grid.
    - `tpitg`: A 3D vector indicating the position of the thread within the threadgroup.
    - `ntg`: A 3D vector indicating the number of threads per threadgroup.
- **Control Flow**:
    - The function calculates the output position based on the input dimensions and the current thread's position.
    - It checks if the calculated input indices are within the valid range of the input tensor.
    - If the indices are valid, it retrieves the corresponding value from the input tensor and stores it in the output destination.
    - If the indices are out of bounds, it sets the output value to zero.
- **Output**: The output is a transformed 2D column matrix stored in the destination pointer, where each column corresponds to a sliding window of the input tensor.


---
### kernel\_rwkv\_wkv7\_f32
The `kernel_rwkv_wkv7_f32` function implements a kernel for processing RWKV (Recurrent Weighted Key-Value) operations in a neural network using Metal shading language.
- **Inputs**:
    - `r`: A device pointer to the input tensor containing the recurrent weights.
    - `w`: A device pointer to the input tensor containing the weights.
    - `k`: A device pointer to the input tensor containing the keys.
    - `v`: A device pointer to the input tensor containing the values.
    - `a`: A device pointer to the input tensor containing additional parameters for the operation.
    - `b`: A device pointer to the input tensor containing additional parameters for the operation.
    - `state_in`: A device pointer to the input tensor containing the initial state.
    - `dst`: A device pointer to the output tensor where results will be stored.
    - `B`: A constant reference to the batch size.
    - `T`: A constant reference to the total number of sequence tokens.
    - `C`: A constant reference to the number of channels.
    - `H`: A constant reference to the number of heads.
    - `tgpig`: A 3D vector representing the thread group position in the grid.
    - `tpitg`: A 3D vector representing the thread position in the thread group.
    - `ntg`: A 3D vector representing the number of threads per thread group.
- **Control Flow**:
    - The function begins by defining the head size and calculating the batch and head IDs based on the thread group position.
    - It checks if the batch ID and head ID are within valid bounds.
    - The function initializes thread group arrays for storing intermediate values.
    - It retrieves the initial state for the current head and stores it in a local state array.
    - The function enters a loop to process each sequence token, loading the relevant key, value, and additional parameters into local arrays.
    - For each token, it computes the output by performing operations involving the keys, values, and the current state.
    - The results are accumulated and stored in the output tensor, along with the updated state.
- **Output**: The function outputs a tensor containing the results of the RWKV operation, along with updated state information for each head.


---
### kernel\_silu
The `kernel_silu` function applies the Sigmoid-Weighted Linear Unit (SiLU) activation function to each element of the input tensor.
- **Inputs**:
    - `src0`: A device pointer to the input tensor containing the values to which the SiLU function will be applied.
    - `dst`: A device pointer to the output tensor where the results of the SiLU function will be stored.
    - `tpig`: A thread position index that indicates the position of the thread in the grid.
- **Control Flow**:
    - The function retrieves the input value from `src0` at the position indicated by `tpig`.
    - It computes the SiLU activation using the formula: output = x / (1 + exp(-x)).
    - The result is then stored in the output tensor `dst` at the same position.
- **Output**: The output is a tensor containing the results of applying the SiLU activation function to each element of the input tensor.


---
### kernel\_soft\_max\_4
The `kernel_soft_max_4` function computes the softmax of a 4-dimensional tensor using SIMD operations for efficiency.
- **Inputs**:
    - `src0`: A pointer to the input tensor from which the softmax will be computed.
    - `src1`: A pointer to an optional mask tensor that can modify the input values.
    - `dst`: A pointer to the output tensor where the softmax results will be stored.
    - `args`: A structure containing parameters for the softmax operation, including dimensions and scaling factors.
    - `buf`: A threadgroup buffer used for intermediate calculations.
    - `tgpig`: A 3D index indicating the position of the threadgroup in the grid.
    - `tpitg`: A 1D index indicating the position of the thread within the threadgroup.
    - `sgitg`: An index indicating the position of the SIMD group within the threadgroup.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `ntg`: A 3D vector indicating the number of threads per threadgroup.
- **Control Flow**:
    - The function begins by calculating the indices for the input tensor based on the threadgroup and thread indices.
    - It retrieves the input tensor and mask values, initializing variables for the softmax computation.
    - The function computes the maximum value in the input tensor to stabilize the softmax calculation.
    - It then computes the exponential of the adjusted input values and sums them up.
    - Finally, it normalizes the exponentials by dividing by the sum to produce the softmax output.
- **Output**: The output is a tensor containing the softmax values computed from the input tensor, stored in the location pointed to by 'dst'.


---
### kernel\_pad\_reflect\_1d\_f32
The `kernel_pad_reflect_1d_f32` function applies a 1D reflective padding to a source tensor.
- **Inputs**:
    - `src0`: A pointer to the source tensor data that will be padded.
    - `dst`: A pointer to the destination tensor where the padded result will be stored.
    - `args`: A structure containing padding parameters such as the number of elements to pad on each side.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tgpg`: A 3D vector indicating the number of thread groups in the grid.
    - `tpitg`: A 3D vector indicating the position of the thread within the thread group.
    - `ntg`: A 3D vector indicating the number of threads per thread group.
- **Control Flow**:
    - The function first calculates the indices for the source tensor based on the thread group and thread positions.
    - It checks if the current thread's indices are within the bounds of the source tensor.
    - If the indices are valid, it applies reflective padding based on the specified padding parameters.
    - If the indices are out of bounds, it sets the corresponding elements in the destination tensor to zero.
- **Output**: The function does not return a value; instead, it writes the padded tensor directly to the destination pointer.


---
### dequantize\_q4\_0
The `dequantize_q4_0` function converts quantized data from a `block_q4_0` structure into floating-point representation.
- **Inputs**:
    - `xb`: A pointer to a `block_q4_0` structure containing the quantized data.
    - `il`: An integer indicating the index for the quantization level.
    - `reg`: A reference to a thread-local variable where the dequantized floating-point data will be stored.
- **Control Flow**:
    - The function begins by extracting the quantized values from the `block_q4_0` structure.
    - It calculates scaling factors based on the `il` parameter to adjust the dequantization process.
    - A loop iterates over the quantized values, applying the scaling factors and storing the results in a temporary floating-point matrix.
    - Finally, the results are assigned to the output reference variable.
- **Output**: The function does not return a value; instead, it populates the `reg` variable with the dequantized floating-point data.


---
### kernel\_relu
The `kernel_relu` function applies the ReLU activation function to each element of the input tensor.
- **Inputs**:
    - `src0`: A device pointer to the input tensor containing float values.
    - `dst`: A device pointer to the output tensor where the results will be stored.
    - `tpig`: A thread position index in the grid, used to identify the specific thread's position.
- **Control Flow**:
    - The function iterates over each element of the input tensor using the thread position index.
    - For each element, it applies the ReLU function, which sets negative values to zero and keeps positive values unchanged.
    - The results are stored in the output tensor.
- **Output**: The output tensor contains the result of applying the ReLU activation function to each element of the input tensor.


---
### kernel\_mul\_mv
The `kernel_mul_mv` function performs matrix-vector multiplication for quantized tensors using Metal shading language.
- **Inputs**:
    - `args`: A constant structure containing the parameters for the matrix-vector multiplication, including dimensions and offsets.
    - `src0`: A pointer to the source tensor (matrix) in device memory.
    - `src1`: A pointer to the source vector in device memory.
    - `dst`: A pointer to the destination tensor (result) in device memory.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
- **Control Flow**:
    - The function calculates the row and column indices based on the thread group and thread indices.
    - It computes the offset for the source tensor and the destination tensor based on the calculated indices.
    - The function iterates over the elements of the source tensor and performs the multiplication with the corresponding elements of the vector.
    - The results are accumulated and stored in the destination tensor.
- **Output**: The output is a tensor that contains the result of the matrix-vector multiplication.


---
### kernel\_cpy\_f32\_q4\_0
The `kernel_cpy_f32_q4_0` function copies and quantizes floating-point data from a source buffer to a destination buffer in a specific quantization format (Q4_0).
- **Inputs**:
    - `args`: A structure containing metadata for the copy operation, including dimensions and offsets.
    - `src0`: A pointer to the source buffer containing floating-point data to be copied.
    - `dst`: A pointer to the destination buffer where the quantized data will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tpitg`: A 3D vector indicating the position of the thread within the thread group.
    - `ntg`: A 3D vector indicating the number of threads per thread group.
- **Control Flow**:
    - Calculate the linear index 'n' based on the thread group and input dimensions.
    - Determine the indices for the source and destination buffers based on 'n'.
    - Iterate over the quantization blocks, processing each block of data.
    - For each block, find the maximum absolute value to determine the scale factor.
    - Quantize the data based on the calculated scale and store it in the destination buffer.
- **Output**: The function does not return a value; instead, it writes the quantized data directly to the destination buffer.


---
### kernel\_mul
The `kernel_mul` function performs element-wise multiplication of two tensors in a Metal kernel.
- **Inputs**:
    - `args`: A constant structure containing metadata for the operation, including tensor dimensions and offsets.
    - `src0`: A pointer to the first input tensor data in device memory.
    - `src1`: A pointer to the second input tensor data in device memory.
    - `dst`: A pointer to the output tensor data in device memory where the result will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tpitg`: A 3D vector indicating the position of the thread within the thread group.
    - `ntg`: A 3D vector indicating the number of threads per thread group.
- **Control Flow**:
    - The function calculates the indices for the input tensors based on the thread group and thread positions.
    - It computes the pointers to the specific elements of the input tensors based on the calculated indices.
    - A loop iterates over the elements of the output tensor, performing multiplication of corresponding elements from the two input tensors.
    - The results are stored in the output tensor at the appropriate indices.
- **Output**: The function does not return a value; instead, it writes the results of the element-wise multiplication directly to the output tensor specified by the `dst` pointer.


---
### kernel\_cpy\_f32\_q5\_0
The `kernel_cpy_f32_q5_0` function performs a quantization operation on a source tensor of floats and copies the result to a destination tensor in a specific quantized format.
- **Inputs**:
    - `args`: A constant structure containing parameters for the copy operation, including dimensions and offsets.
    - `src0`: A pointer to the source tensor in device memory, containing float values to be quantized.
    - `dst`: A pointer to the destination tensor in device memory, where the quantized values will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tpitg`: A 3D vector indicating the position of the thread within the thread group.
    - `ntg`: A 3D vector indicating the number of threads per thread group.
- **Control Flow**:
    - The function begins by calculating the indices for the current thread group based on the input parameters.
    - It computes the linear index `n` for the source tensor based on the thread group indices and the dimensions provided in `args`.
    - The function then calculates the indices `i3`, `i2`, `i1`, and `i0` to determine the specific location in the source tensor.
    - A pointer to the destination data structure is created based on the calculated indices.
    - A loop iterates over the quantization blocks, processing `QK5_0` elements at a time.
    - For each block, the maximum absolute value is determined, and the quantization scale is calculated.
    - The quantized values are computed and stored in the destination tensor, along with the quantization parameters.
- **Output**: The function does not return a value but writes the quantized data directly to the destination tensor in device memory.


---
### kernel\_cpy\_f32\_q5\_1
The `kernel_cpy_f32_q5_1` function performs quantization of floating-point data into a specific format for efficient storage and processing.
- **Inputs**:
    - `args`: A constant structure containing parameters for the copy operation, including dimensions and offsets.
    - `src0`: A pointer to the source data in device memory, which contains the floating-point values to be quantized.
    - `dst`: A pointer to the destination data in device memory, where the quantized values will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tpitg`: A 3D vector indicating the position of the thread within the thread group.
    - `ntg`: A 3D vector indicating the number of threads per thread group.
- **Control Flow**:
    - The function begins by calculating the indices for the current thread group based on the input parameters.
    - It computes the total number of elements to process and determines the indices for accessing the source data.
    - A loop iterates over the quantization process, where it reads floating-point values from the source, computes the minimum and maximum values, and calculates the scale factor for quantization.
    - The quantized values are then stored in the destination array, along with the computed scale and minimum values.
- **Output**: The function does not return a value directly; instead, it writes the quantized data to the destination pointer in device memory.


---
### best\_index\_int8
`best_index_int8` performs a binary search to find the index of the closest value to `x` in a sorted array.
- **Inputs**:
    - `n`: The number of elements in the sorted array `val`.
    - `val`: A pointer to a constant array of floats that is sorted in ascending order.
    - `x`: The float value for which the closest index is to be found.
- **Control Flow**:
    - If `x` is less than or equal to the first element of `val`, return 0.
    - If `x` is greater than or equal to the last element of `val`, return n-1.
    - Initialize two indices, `ml` (lower) and `mu` (upper), to 0 and n-1 respectively.
    - While the difference between `mu` and `ml` is greater than 1, calculate the midpoint `mav`.
    - If `x` is less than `val[mav]`, set `mu` to `mav`; otherwise, set `ml` to `mav`.
    - After exiting the loop, return the index of the closest value between `val[mu-1]` and `val[mu]`.
- **Output**: Returns the index of the closest value to `x` in the sorted array `val`.


---
### kernel\_cpy\_f32\_iq4\_nl
`kernel_cpy_f32_iq4_nl` is a Metal kernel function that copies and quantizes floating-point data from a source buffer to a destination buffer using a specific quantization scheme.
- **Inputs**:
    - `args`: A constant structure containing parameters for the copy operation, including dimensions and offsets.
    - `src0`: A device pointer to the source data buffer containing floating-point values.
    - `dst`: A device pointer to the destination data buffer where the quantized values will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tpitg`: A 3D vector indicating the position of the thread within the thread group.
    - `ntg`: A 3D vector indicating the number of threads per thread group.
- **Control Flow**:
    - Calculate the linear index `n` based on the thread group position and the dimensions provided in `args`.
    - Determine the indices `i3`, `i2`, `i1`, and `i0` for accessing the source and destination buffers.
    - Allocate a pointer `dst_data` to the destination buffer based on the calculated indices.
    - Iterate over the data in chunks defined by `QK4_NL`, processing each chunk in a loop.
    - For each chunk, find the maximum absolute value and the maximum value in the source data.
    - Calculate the scaling factor `d` based on the maximum value and a predefined constant.
    - Quantize the source values into 8-bit integers using a lookup function and store them in the destination buffer.
    - Store the computed scaling factor in the destination buffer.
- **Output**: The function does not return a value but writes the quantized data and scaling factors directly to the destination buffer.


---
### kernel\_cpy\_q\_f32
`kernel_cpy_q_f32` is a templated kernel function that copies and dequantizes data from a source buffer to a destination buffer using specified quantization parameters.
- **Inputs**:
    - `args`: A constant reference to a structure containing kernel arguments, including dimensions and offsets for the source and destination buffers.
    - `src0`: A pointer to the source buffer containing quantized data.
    - `dst`: A pointer to the destination buffer where the dequantized data will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tpitg`: A 3D vector indicating the position of the thread within the thread group.
    - `ntg`: A 3D vector indicating the number of threads per thread group.
- **Control Flow**:
    - The function calculates the indices for accessing the source and destination buffers based on the thread group and thread positions.
    - It iterates over the quantized data in blocks defined by the template parameters, performing dequantization for each block.
    - The dequantization is performed using a specified function passed as a template parameter, which converts the quantized data into a floating-point format.
    - The resulting floating-point data is then stored in the destination buffer.
- **Output**: The function does not return a value; instead, it writes the dequantized data directly to the destination buffer.


---
### kernel\_concat
`kernel_concat` concatenates two input tensors along a specified dimension.
- **Inputs**:
    - `args`: A constant structure containing parameters for the concatenation operation, including the dimension along which to concatenate and the sizes of the input tensors.
    - `src0`: A device pointer to the first input tensor.
    - `src1`: A device pointer to the second input tensor.
    - `dst`: A device pointer to the output tensor where the concatenated result will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tpitg`: A 3D vector indicating the position of the thread within the thread group.
    - `ntg`: A 3D vector indicating the number of threads per thread group.
- **Control Flow**:
    - The function begins by extracting the indices for the thread group and initializing an output array `o` to hold the size of the concatenated dimension.
    - It then iterates over the first dimension of the output tensor, checking if the current index is within the bounds of the first input tensor.
    - If the index is valid, it calculates the corresponding index in `src0` and assigns the value to the output tensor.
    - If the index exceeds the bounds of `src0`, it calculates the corresponding index in `src1` and assigns that value to the output tensor.
- **Output**: The function outputs a tensor that is the result of concatenating `src0` and `src1` along the specified dimension.


---
### kernel\_mul\_mv\_q2\_K\_f32\_impl
The `kernel_mul_mv_q2_K_f32_impl` function performs matrix-vector multiplication using quantized 2-bit weights and outputs the result in floating-point format.
- **Inputs**:
    - `args`: A structure containing the parameters for the multiplication operation, including dimensions and offsets.
    - `src0`: A pointer to the source data containing the quantized weights.
    - `src1`: A pointer to the source data containing the input vector.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the position of the SIMD group within the thread group.
- **Control Flow**:
    - Calculate the number of blocks and the first row index based on the thread group and SIMD group indices.
    - Compute the offsets for accessing the quantized weights and input vector based on the calculated indices.
    - Load the quantized weights and input vector from the device memory.
    - Perform the matrix-vector multiplication using the quantized weights and accumulate the results.
    - Store the final result in the destination memory.
- **Output**: The function outputs the result of the matrix-vector multiplication as a floating-point vector in the specified destination memory.


---
### kernel\_mul\_mv\_q2\_K\_f32
`kernel_mul_mv_q2_K_f32` performs matrix-vector multiplication using quantized 2-bit weights.
- **Inputs**:
    - `args`: A structure containing parameters for the multiplication, including dimensions and offsets.
    - `src0`: A pointer to the source data containing the quantized matrix.
    - `src1`: A pointer to the source data containing the vector to be multiplied.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the SIMD group index within the thread group.
- **Control Flow**:
    - Calculate the number of blocks and the first row index based on the thread group and SIMD group indices.
    - Compute offsets for accessing the source data based on the input arguments.
    - Load the quantized matrix and vector from the source pointers.
    - Perform the matrix-vector multiplication in a loop, accumulating results into a temporary array.
    - Store the final results into the destination pointer.
- **Output**: The function outputs the result of the matrix-vector multiplication into the `dst` pointer.


---
### kernel\_mul\_mv\_q3\_K\_f32\_impl
`kernel_mul_mv_q3_K_f32_impl` performs matrix-vector multiplication using quantized weights and a specific kernel configuration.
- **Inputs**:
    - `args`: A structure containing the parameters for the multiplication, including dimensions and offsets.
    - `src0`: A pointer to the source data containing the quantized weights.
    - `src1`: A pointer to the source data containing the input vector.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the position of the SIMD group within the thread group.
- **Control Flow**:
    - Calculate the number of blocks and the row and column indices based on the thread group position.
    - Determine the offsets for accessing the source data based on the calculated indices.
    - Load the quantized weights from `src0` and the input vector from `src1`.
    - Perform the matrix-vector multiplication in a loop, accumulating results into a temporary sum.
    - Store the final result in the destination pointer `dst`.
- **Output**: The function outputs the result of the matrix-vector multiplication into the `dst` pointer.


---
### kernel\_mul\_mv\_q3\_K\_f32
`kernel_mul_mv_q3_K_f32` performs matrix-vector multiplication using quantized data.
- **Inputs**:
    - `args`: A structure containing parameters for the multiplication, including dimensions and offsets.
    - `src0`: A pointer to the source data representing the quantized matrix.
    - `src1`: A pointer to the source data representing the vector to be multiplied.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the SIMD group within the thread group.
- **Control Flow**:
    - The function begins by calculating the number of blocks and the row and column indices based on the thread group position.
    - It computes offsets for accessing the source data based on the current row and column indices.
    - The function then enters a loop to process the matrix in blocks, performing the multiplication for each block.
    - Within the loop, it loads the quantized data from `src0` and the vector from `src1`, performing the multiplication and accumulation.
    - Finally, the results are stored in the destination buffer `dst`.
- **Output**: The output is a pointer to the destination buffer containing the result of the matrix-vector multiplication.


---
### kernel\_mul\_mv\_q4\_K\_f32\_impl
`kernel_mul_mv_q4_K_f32_impl` performs matrix-vector multiplication using quantized 4-bit weights.
- **Inputs**:
    - `args`: A structure containing the parameters for the multiplication, including dimensions and offsets.
    - `src0`: A pointer to the source data containing the quantized weights.
    - `src1`: A pointer to the source data containing the input vector.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread in the SIMD group.
    - `sgitg`: An index indicating the position of the SIMD group in the thread group.
- **Control Flow**:
    - Calculate the number of blocks and the first row index based on the thread group and SIMD group indices.
    - Determine the offsets for accessing the source data based on the input parameters.
    - Load the quantized weights and input vector into local memory.
    - Perform the matrix-vector multiplication in a loop, accumulating results into a temporary storage.
    - Store the final results back to the destination memory.
- **Output**: The output is a vector resulting from the multiplication of the quantized weights with the input vector, stored in the specified destination.


---
### kernel\_mul\_mv\_q4\_K\_f32
The `kernel_mul_mv_q4_K_f32` function performs matrix-vector multiplication using quantized 4-bit weights and floating-point values.
- **Inputs**:
    - `args`: A structure containing the parameters for the multiplication operation, including dimensions and offsets.
    - `src0`: A pointer to the source data containing the quantized weights.
    - `src1`: A pointer to the source data containing the floating-point vector to be multiplied.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the position of the SIMD group within the thread group.
- **Control Flow**:
    - The function begins by defining constants for masks used in the quantization process.
    - It calculates the number of blocks and the row and column indices based on the thread group position.
    - The function then computes the offsets for accessing the source data based on the input parameters.
    - A loop iterates over the quantized weights, performing the multiplication with the floating-point vector.
    - The results are accumulated and stored in the destination buffer.
- **Output**: The output is a floating-point vector resulting from the matrix-vector multiplication, stored in the memory location pointed to by `dst`.


---
### kernel\_mul\_mv\_q5\_K\_f32\_impl
`kernel_mul_mv_q5_K_f32_impl` performs matrix-vector multiplication using quantized weights.
- **Inputs**:
    - `args`: A structure containing the parameters for the multiplication, including dimensions and offsets.
    - `src0`: A pointer to the source data containing the quantized weights.
    - `src1`: A pointer to the source data containing the input vector.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the position of the SIMD group within the thread group.
- **Control Flow**:
    - Calculate the number of blocks and the row and column indices based on the thread group position.
    - Determine the offsets for accessing the source data based on the calculated indices.
    - Load the quantized weights and input vector from the respective source pointers.
    - Perform the matrix-vector multiplication in a loop, accumulating results into a temporary sum.
    - Store the final result into the destination pointer.
- **Output**: The function outputs the result of the matrix-vector multiplication into the `dst` pointer.


---
### kernel\_mul\_mv\_q5\_K\_f32
`kernel_mul_mv_q5_K_f32` performs matrix-vector multiplication using quantized data with a specific kernel configuration.
- **Inputs**:
    - `args`: A structure containing the parameters for the multiplication, including dimensions and offsets.
    - `src0`: A pointer to the source data for the matrix in a quantized format.
    - `src1`: A pointer to the source data for the vector in a float format.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the position of the SIMD group within the thread group.
- **Control Flow**:
    - Calculate the number of blocks and the row and column indices based on the thread group position.
    - Determine the offsets for accessing the source data based on the current row and column.
    - Load the quantized matrix data from `src0` and the float vector data from `src1`.
    - Perform the multiplication in a loop over the blocks, accumulating results in a temporary storage.
    - Store the final results in the destination buffer `dst`.
- **Output**: The function outputs the result of the matrix-vector multiplication in the specified destination buffer, formatted as floating-point values.


---
### kernel\_mul\_mv\_q6\_K\_f32\_impl
Implements a matrix-vector multiplication operation for quantized data.
- **Inputs**:
    - `args`: A structure containing various parameters for the multiplication operation, including dimensions and offsets.
    - `src0`: A pointer to the source data representing the quantized matrix.
    - `src1`: A pointer to the source data representing the vector to be multiplied.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
- **Control Flow**:
    - The function begins by calculating the offsets for accessing the source data based on the input arguments.
    - It then retrieves the quantized matrix and vector data from the device memory.
    - A loop iterates over the blocks of the matrix, performing the multiplication with the vector.
    - The results are accumulated and stored in the destination memory.
- **Output**: The output is a vector resulting from the multiplication of the quantized matrix and the input vector, stored in the specified destination.


---
### kernel\_mul\_mv\_q6\_K\_f32
`kernel_mul_mv_q6_K_f32` performs matrix-vector multiplication using quantized 6-bit values.
- **Inputs**:
    - `args`: A structure containing parameters for the multiplication, including dimensions and offsets.
    - `src0`: A pointer to the source matrix in a quantized format.
    - `src1`: A pointer to the source vector.
    - `dst`: A pointer to the destination where the result will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the position of the SIMD group within the thread group.
- **Control Flow**:
    - Calculate the number of blocks and the first row index based on the thread group and SIMD group indices.
    - Determine the offsets for accessing the source matrices based on the input parameters.
    - Load the quantized matrix and vector data from the device memory.
    - Perform the matrix-vector multiplication using the loaded data, applying the quantization scale.
    - Store the result back to the destination memory.
- **Output**: The function outputs the result of the matrix-vector multiplication in the specified destination.


---
### kernel\_mul\_mv\_iq2\_xxs\_f32\_impl
`kernel_mul_mv_iq2_xxs_f32_impl` performs matrix-vector multiplication using a specific quantization scheme for 2-bit integers.
- **Inputs**:
    - `args`: A structure containing parameters for the multiplication operation, including dimensions and offsets.
    - `src0`: A pointer to the source matrix data, which is quantized.
    - `src1`: A pointer to the source vector data, which is in floating-point format.
    - `dst`: A pointer to the destination buffer where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
- **Control Flow**:
    - Calculate the number of blocks and the current row and column indices based on the thread group position.
    - Determine the offsets for accessing the source data based on the current indices.
    - Load the quantized data from `src0` and the floating-point data from `src1` into local variables.
    - Perform the multiplication by iterating over the quantized data and accumulating the results into a sum.
    - Store the final computed result into the destination buffer `dst`.
- **Output**: The function outputs the result of the matrix-vector multiplication in the destination buffer, stored as floating-point values.


---
### kernel\_mul\_mv\_iq2\_xxs\_f32
`kernel_mul_mv_iq2_xxs_f32` performs matrix-vector multiplication using a specific quantization scheme.
- **Inputs**:
    - `args`: A structure containing parameters for the multiplication operation, including dimensions and offsets.
    - `src0`: A pointer to the source matrix data, which is quantized.
    - `src1`: A pointer to the source vector data, which is in floating-point format.
    - `dst`: A pointer to the destination buffer where the result will be stored.
    - `shmem`: A pointer to shared memory used for intermediate calculations.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
- **Control Flow**:
    - Calculate the number of blocks and the first row index based on the thread indices.
    - Determine the offsets for accessing the source matrices based on the input parameters.
    - Load the quantized matrix data from `src0` and the floating-point vector data from `src1`.
    - Perform the multiplication for each row of the output matrix, accumulating results into a sum.
    - Store the final computed results into the destination buffer `dst`.
- **Output**: The function outputs the result of the matrix-vector multiplication into the `dst` buffer, with the results being in floating-point format.


---
### kernel\_mul\_mv\_iq2\_xs\_f32\_impl
Implements a kernel for multiplying a matrix with a vector using quantized data.
- **Inputs**:
    - `args`: A structure containing various parameters for the multiplication operation, including dimensions and offsets.
    - `src0`: A pointer to the source matrix data, which is quantized.
    - `src1`: A pointer to the source vector data, which is in floating-point format.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the position of the SIMD group within the thread group.
- **Control Flow**:
    - Calculate the number of blocks and the first row based on the thread indices.
    - Determine the offsets for accessing the source data based on the input parameters.
    - Load the quantized matrix data and the floating-point vector data into local variables.
    - Perform the multiplication operation in a loop, iterating over the necessary dimensions.
    - Store the results back into the destination pointer after the computation is complete.
- **Output**: The function outputs the result of the matrix-vector multiplication into the specified destination pointer.


---
### kernel\_mul\_mv\_iq2\_xs\_f32
`kernel_mul_mv_iq4_xs_f32` performs matrix-vector multiplication using quantized 4-bit integer values.
- **Inputs**:
    - `args`: A structure containing parameters for the multiplication operation, including dimensions and offsets.
    - `src0`: A pointer to the source data containing the quantized matrix.
    - `src1`: A pointer to the source data containing the vector to be multiplied.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `shmem`: Shared memory for thread group communication.
    - `tgpig`: Thread group position in the grid.
    - `tiisg`: Thread index in the SIMD group.
- **Control Flow**:
    - Calculate the number of blocks and the first row index based on the thread group and SIMD indices.
    - Load the quantized matrix and vector data from the device memory into local variables.
    - Perform the multiplication in a loop, iterating over the number of columns in the matrix.
    - Accumulate the results into a temporary variable.
    - Store the final result back to the destination memory.
- **Output**: The function outputs the result of the matrix-vector multiplication as a float array in the specified destination.


---
### kernel\_mul\_mv\_iq3\_xxs\_f32\_impl
`kernel_mul_mv_iq3_xxs_f32_impl` performs matrix-vector multiplication using a specific quantization scheme for 3-bit integers.
- **Inputs**:
    - `args`: A structure containing parameters for the multiplication operation, including dimensions and offsets.
    - `src0`: A pointer to the source data representing the quantized matrix.
    - `src1`: A pointer to the source data representing the vector to be multiplied.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `shmem`: A pointer to shared memory used for intermediate calculations.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
- **Control Flow**:
    - Calculate the number of blocks and the first row index based on the thread group and SIMD group indices.
    - Determine the offsets for accessing the source data based on the input parameters.
    - Load the quantized matrix and vector data from the device memory.
    - Perform the multiplication operation in a loop, processing blocks of data.
    - Accumulate the results into a temporary storage for final output.
    - Store the final results back to the destination memory.
- **Output**: The function outputs the result of the matrix-vector multiplication stored in the `dst` pointer.


---
### kernel\_mul\_mv\_iq3\_xxs\_f32
`kernel_mul_mv_iq3_xxs_f32` performs matrix-vector multiplication using 3-bit quantized input data.
- **Inputs**:
    - `args`: A structure containing parameters for the multiplication operation, including dimensions and offsets.
    - `src0`: A pointer to the source data containing the quantized matrix.
    - `src1`: A pointer to the source data containing the vector to be multiplied.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `shmem`: Shared memory for threadgroup operations.
    - `tgpig`: Threadgroup position in the grid.
    - `tiisg`: Thread index in the SIMD group.
    - `sgitg`: SIMD group index in the threadgroup.
- **Control Flow**:
    - Calculate the number of blocks and the first row index based on the threadgroup position.
    - Load the quantized matrix and vector from the respective source pointers.
    - Perform the multiplication in a loop over the blocks, accumulating results in shared memory.
    - Store the final results back to the destination pointer.
- **Output**: The function outputs the result of the matrix-vector multiplication into the `dst` pointer.


---
### kernel\_mul\_mv\_iq3\_s\_f32\_impl
`kernel_mul_mv_iq3_s_f32_impl` performs matrix-vector multiplication for quantized 3-bit signed integers.
- **Inputs**:
    - `args`: A structure containing the parameters for the multiplication, including dimensions and offsets.
    - `src0`: A pointer to the source data representing the quantized matrix.
    - `src1`: A pointer to the source data representing the vector to be multiplied.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `shmem`: A pointer to shared memory used for intermediate calculations.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
- **Control Flow**:
    - Calculate the number of blocks and the first row index based on the thread group and SIMD group indices.
    - Determine the offsets for accessing the source matrices based on the input parameters.
    - Load the quantized matrix and vector data from the device memory into local variables.
    - Perform the multiplication by iterating over the rows and columns of the matrix and vector, accumulating the results.
    - Store the final result back into the destination memory.
- **Output**: The function outputs the result of the matrix-vector multiplication as a float array in the destination pointer.


---
### kernel\_mul\_mv\_iq3\_s\_f32
`kernel_mul_mv_iq3_s_f32` performs a matrix-vector multiplication using quantized 3-bit signed integers.
- **Inputs**:
    - `args`: A structure containing parameters for the multiplication, including dimensions and offsets.
    - `src0`: A pointer to the source matrix data, which is quantized.
    - `src1`: A pointer to the source vector data, which is in floating-point format.
    - `dst`: A pointer to the destination buffer where the result will be stored.
    - `shmem`: Shared memory for thread group communication.
    - `tgpig`: Thread group position in the grid.
    - `tiisg`: Thread index in the SIMD group.
- **Control Flow**:
    - Calculate the number of blocks and the first row index based on the thread group position.
    - Determine the offsets for accessing the source data based on the input dimensions.
    - Load the quantized matrix from `src0` and the floating-point vector from `src1`.
    - Perform the multiplication for each row of the matrix with the vector, accumulating results.
    - Store the final results in the destination buffer.
- **Output**: The function outputs the result of the matrix-vector multiplication in the destination buffer, which is in floating-point format.


---
### kernel\_mul\_mv\_iq2\_s\_f32\_impl
`kernel_mul_mv_iq2_s_f32_impl` performs matrix-vector multiplication for quantized 2-bit integer inputs with scaling.
- **Inputs**:
    - `args`: A structure containing the parameters for the multiplication, including dimensions and offsets.
    - `src0`: A pointer to the source data containing the quantized matrix.
    - `src1`: A pointer to the source data containing the vector to be multiplied.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `shmem`: Shared memory for thread group communication.
    - `tgpig`: Thread group position in the grid.
    - `tiisg`: Thread index in the SIMD group.
    - `sgitg`: SIMD group index in the thread group.
- **Control Flow**:
    - Calculate the number of blocks and the first row index based on the thread group and SIMD group indices.
    - Determine the offsets for accessing the source data based on the input dimensions.
    - Load the quantized matrix and vector from the device memory.
    - Perform the multiplication in a loop over the number of blocks, accumulating results in shared memory.
    - Store the final results back to the destination memory after processing all blocks.
- **Output**: The function outputs the result of the matrix-vector multiplication as a float array in the destination pointer.


---
### kernel\_mul\_mv\_iq2\_s\_f32
`kernel_mul_mv_iq2_s_f32` performs matrix-vector multiplication using 2-bit quantized input data.
- **Inputs**:
    - `args`: A structure containing parameters for the multiplication operation, including dimensions and offsets.
    - `src0`: A pointer to the source data containing the quantized matrix.
    - `src1`: A pointer to the source data containing the vector to be multiplied.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `shmem`: Shared memory for thread group communication.
    - `tgpig`: Thread group position in the grid.
    - `tiisg`: Thread index in the SIMD group.
    - `sgitg`: SIMD group index in the thread group.
- **Control Flow**:
    - Calculate the number of blocks and the first row index based on the thread group and SIMD group indices.
    - Determine the offsets for accessing the source data based on the input arguments.
    - Load the quantized matrix and vector data from the device memory.
    - Perform the multiplication operation for each row of the matrix with the vector, accumulating the results.
    - Store the final results in the destination memory.
- **Output**: The function outputs the result of the matrix-vector multiplication as a float array in the specified destination.


---
### kernel\_mul\_mv\_iq1\_s\_f32\_impl
Implements a kernel for multiplying a matrix with a vector using quantized integer representation.
- **Inputs**:
    - `args`: A structure containing parameters for the multiplication operation, including dimensions and offsets.
    - `src0`: A pointer to the source matrix data in quantized format.
    - `src1`: A pointer to the source vector data in floating-point format.
    - `dst`: A pointer to the destination buffer where the result will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
- **Control Flow**:
    - Calculate the number of blocks and the first row index based on the thread group and SIMD group indices.
    - Determine the offsets for accessing the source matrix and vector data.
    - Iterate over the blocks of the matrix, performing the multiplication with the vector.
    - Load the quantized matrix data and the vector data into local memory.
    - Perform the multiplication and accumulate the results.
    - Store the final result in the destination buffer.
- **Output**: The function outputs the result of the matrix-vector multiplication into the specified destination buffer.


---
### kernel\_mul\_mv\_iq1\_s\_f32
`kernel_mul_mv_iq1_s_f32` performs matrix-vector multiplication with quantized input data.
- **Inputs**:
    - `args`: A structure containing parameters for the multiplication operation, including dimensions and offsets.
    - `src0`: A pointer to the source data containing quantized matrix values.
    - `src1`: A pointer to the source data containing the vector to be multiplied.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
- **Control Flow**:
    - Calculate the number of blocks and the first row index based on the thread indices.
    - Determine the offsets for accessing the source data based on the input arguments.
    - Iterate over the blocks of the input matrix, performing the multiplication for each block.
    - Load the quantized values from the source matrix and vector into local variables.
    - Perform the multiplication and accumulate the results into a temporary sum.
    - Store the final result into the destination buffer.
- **Output**: The function outputs the result of the matrix-vector multiplication into the `dst` pointer, which contains floating-point values.


---
### kernel\_mul\_mv\_iq1\_m\_f32\_impl
The `kernel_mul_mv_iq1_m_f32_impl` function performs matrix-vector multiplication for quantized input data using a specific quantization scheme.
- **Inputs**:
    - `args`: A structure containing parameters for the multiplication operation, including dimensions and offsets for the input and output data.
    - `src0`: A pointer to the source data (input matrix) in quantized format.
    - `src1`: A pointer to the source data (input vector) in float format.
    - `dst`: A pointer to the destination data (output vector) where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
- **Control Flow**:
    - Calculate the number of blocks and the first row index based on the thread group and SIMD group indices.
    - Determine the offsets for accessing the input data based on the calculated indices.
    - Load the quantized input data and the float input vector into local variables.
    - Perform the matrix-vector multiplication by iterating over the rows of the input matrix and accumulating the results into the output vector.
    - Store the computed results back into the destination pointer.
- **Output**: The function outputs a vector in the destination pointer `dst`, which contains the result of the matrix-vector multiplication.


---
### kernel\_mul\_mv\_iq1\_m\_f32
`kernel_mul_mv_iq1_m_f32` performs matrix-vector multiplication with quantized input data.
- **Inputs**:
    - `args`: A structure containing the parameters for the multiplication, including dimensions and offsets.
    - `src0`: A pointer to the source data representing the quantized matrix.
    - `src1`: A pointer to the source data representing the vector to be multiplied.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
- **Control Flow**:
    - Calculate the number of blocks and the first row index based on the thread group and SIMD group indices.
    - Determine the offsets for accessing the source data based on the input arguments.
    - Load the quantized matrix and vector data from the device memory.
    - Perform the multiplication for each row of the matrix with the vector, accumulating the results.
    - Store the final results in the destination memory.
- **Output**: The function outputs the result of the matrix-vector multiplication into the specified destination pointer.


---
### kernel\_mul\_mv\_iq4\_nl\_f32\_impl
`kernel_mul_mv_iq4_nl_f32` performs matrix-vector multiplication with quantized 4-bit integer inputs and outputs floating-point results.
- **Inputs**:
    - `args`: A structure containing parameters for the multiplication operation, including dimensions and offsets.
    - `src0`: A pointer to the source data containing the quantized input matrix.
    - `src1`: A pointer to the source data containing the input vector.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `shmem`: Shared memory for thread group communication.
    - `tgpig`: Thread group position in the grid.
    - `tiisg`: Thread index in the SIMD group.
    - `sgitg`: SIMD group index in the thread group.
- **Control Flow**:
    - Calculate the number of blocks and the first row index based on the thread group and SIMD group indices.
    - Load the quantized input matrix and the input vector from the device memory.
    - Perform the multiplication in a loop over the blocks of the input matrix, accumulating results in shared memory.
    - Store the final results back to the destination memory after processing all blocks.
- **Output**: The output is a floating-point vector resulting from the multiplication of the quantized input matrix and the input vector.


---
### kernel\_mul\_mv\_iq4\_nl\_f32
`kernel_mul_mv_iq4_nl_f32` performs matrix-vector multiplication with quantized input using a non-linear dequantization method.
- **Inputs**:
    - `args`: A structure containing parameters for the multiplication operation, including dimensions and offsets.
    - `src0`: A pointer to the source data containing the quantized matrix.
    - `src1`: A pointer to the source data containing the vector to be multiplied.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `shmem`: A pointer to shared memory used for intermediate calculations.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the position of the SIMD group within the thread group.
- **Control Flow**:
    - Calculate the number of blocks and the first row index based on the thread group and SIMD group indices.
    - Determine the offsets for accessing the source data based on the input parameters.
    - Load the quantized matrix and vector data from the device memory.
    - Perform the multiplication operation using a loop over the quantized data, applying the non-linear dequantization.
    - Accumulate the results into a temporary sum variable.
    - Store the final result in the destination memory.
- **Output**: The function outputs the result of the matrix-vector multiplication in the destination pointer `dst`.


---
### kernel\_mul\_mv\_iq4\_xs\_f32\_impl
`kernel_mul_mv_iq4_xs_f32_impl` performs matrix-vector multiplication using quantized 4-bit integers with a specific implementation for the IQ4 format.
- **Inputs**:
    - `args`: A structure containing the parameters for the multiplication, including dimensions and offsets.
    - `src0`: A pointer to the source data representing the quantized matrix.
    - `src1`: A pointer to the source data representing the vector to be multiplied.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `shmem`: A pointer to shared memory used for intermediate calculations.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the position of the SIMD group within the thread group.
- **Control Flow**:
    - Calculate the number of blocks and the first row index based on the thread indices.
    - Determine the offsets for accessing the source data based on the input parameters.
    - Load the quantized matrix and vector data from the device memory.
    - Perform the multiplication in a loop, accumulating results into shared memory.
    - Store the final results back to the destination memory after processing all blocks.
- **Output**: The function outputs the result of the matrix-vector multiplication stored in the destination pointer `dst`.


---
### kernel\_mul\_mv\_iq4\_xs\_f32
The `kernel_mul_mv_iq4_xs_f32` function performs matrix-vector multiplication using quantized 4-bit integers with a specific scaling and dequantization process.
- **Inputs**:
    - `args`: A structure containing parameters for the multiplication operation, including dimensions and offsets for the source and destination matrices.
    - `src0`: A pointer to the source matrix data, which is expected to be in a quantized format.
    - `src1`: A pointer to the vector data that will be multiplied with the quantized matrix.
    - `dst`: A pointer to the destination memory where the result of the multiplication will be stored.
    - `shmem`: Shared memory used for intermediate calculations during the multiplication process.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the position of the SIMD group within the thread group.
- **Control Flow**:
    - The function begins by calculating the offsets for the source and destination matrices based on the input parameters.
    - It then retrieves the quantized matrix and vector data from the specified memory locations.
    - The function enters a loop to process the matrix in blocks, performing dequantization of the matrix data.
    - For each block, it loads the corresponding vector data and performs the multiplication, accumulating results in shared memory.
    - After processing all blocks, the results are written back to the destination memory.
- **Output**: The output is stored in the destination pointer `dst`, which contains the result of the matrix-vector multiplication in floating-point format.


---
### kernel\_get\_rows\_q
`kernel_get_rows_q` retrieves rows from a quantized source tensor based on indices provided in another tensor.
- **Inputs**:
    - `src0`: Pointer to the source tensor containing quantized data.
    - `src1`: Pointer to the tensor containing row indices.
    - `dst`: Pointer to the destination tensor where the retrieved rows will be stored.
    - `args`: Structure containing metadata for the operation, including dimensions and offsets.
    - `tgpig`: Thread group position in the grid.
    - `tiitg`: Thread index in the thread group.
    - `tptg`: Number of threads per thread group.
- **Control Flow**:
    - Calculate the row index `r` from `src1` using the provided thread group indices.
    - Iterate over the range of `ne00` divided by 16, using `tiitg` to determine the current index.
    - For each index, retrieve the corresponding quantized data from `src0` using the calculated row index and the current index.
    - Dequantize the retrieved data into a temporary variable `temp` using the provided `dequantize_func`.
    - Store the dequantized data into the destination tensor `dst` at the appropriate location.
- **Output**: The function does not return a value; instead, it writes the retrieved and dequantized rows directly to the `dst` tensor.


---
### kernel\_get\_rows\_f
`kernel_get_rows_f` retrieves rows from a source tensor based on indices specified in another tensor.
- **Inputs**:
    - `src0`: Pointer to the source tensor from which rows are to be retrieved.
    - `src1`: Pointer to the tensor containing the indices of the rows to retrieve.
    - `dst`: Pointer to the destination tensor where the retrieved rows will be stored.
    - `args`: Structure containing parameters such as dimensions and offsets for the operation.
    - `tgpig`: Thread group position in the grid, indicating the current thread's position.
    - `tiitg`: Thread index within the thread group.
    - `tptg`: Number of threads per thread group.
- **Control Flow**:
    - The function calculates the row index `r` from `src1` using the provided thread group and index.
    - It iterates over the range of rows to be retrieved based on the thread index `tiitg`.
    - For each index, it retrieves the corresponding row from `src0` and stores it in `dst`.
- **Output**: The function does not return a value; instead, it writes the retrieved rows directly to the destination tensor `dst`.


---
### kernel\_get\_rows\_i32
`kernel_get_rows_i32` retrieves rows of 32-bit integers from a source buffer based on indices specified in another buffer.
- **Inputs**:
    - `src0`: Pointer to the source buffer containing the data to be retrieved.
    - `src1`: Pointer to the buffer containing the indices of the rows to retrieve.
    - `dst`: Pointer to the destination buffer where the retrieved rows will be stored.
    - `args`: Structure containing metadata for the operation, including dimensions and offsets.
    - `tgpig`: Thread group position in the grid, used for parallel processing.
    - `tiitg`: Thread index in the thread group.
    - `tptg`: Thread position in the thread group.
- **Control Flow**:
    - Calculate the indices for the current thread based on its position in the grid.
    - Retrieve the row index from `src1` using the calculated indices.
    - Iterate over the range of rows to be retrieved based on the thread index.
    - For each index, copy the corresponding row from `src0` to `dst`.
- **Output**: The function does not return a value; instead, it writes the retrieved rows directly to the `dst` buffer.


---
### kernel\_mul\_mm
`kernel_mul_mm` performs matrix multiplication on quantized matrices using SIMD operations.
- **Inputs**:
    - `args`: A structure containing parameters for the matrix multiplication, including dimensions and offsets.
    - `src0`: Pointer to the first input matrix (quantized format).
    - `src1`: Pointer to the second input matrix (quantized format).
    - `dst`: Pointer to the output matrix where the result will be stored.
    - `shmem`: Shared memory for storing intermediate results.
    - `tgpig`: Thread group position in the grid.
    - `tiitg`: Thread index in the thread group.
    - `sgitg`: SIMD group index in the thread group.
- **Control Flow**:
    - Calculate the number of rows and columns for the current block based on the input dimensions.
    - Load quantized data from the input matrices into shared memory.
    - Perform matrix multiplication using SIMD operations on the loaded data.
    - Store the results back to the output matrix, ensuring to handle cases where the block size is smaller than expected.
- **Output**: The output is a matrix resulting from the multiplication of the two input matrices, stored in the memory pointed to by `dst`.


---
### kernel\_mul\_mm\_id\_map0
The `kernel_mul_mm_id_map0` function performs a mapping operation for matrix multiplication, specifically for a given expert ID, and stores the results in a specified destination buffer.
- **Inputs**:
    - `args`: A constant structure containing parameters for the kernel operation, including dimensions and offsets for the input and output data.
    - `src1`: A device pointer to the first source matrix used in the multiplication operation.
    - `src2`: A device pointer to the second source matrix, which contains expert IDs for mapping.
    - `hsrc1`: A device pointer to the destination buffer where the mapped results will be stored.
    - `htpe`: A device pointer to a temporary buffer used for storing expert counts.
    - `hids`: A device pointer to an array that holds the IDs of the experts.
    - `tgpig`: A 3D vector representing the position of the thread group in the grid.
    - `tpitg`: A 3D vector representing the position of the thread within the thread group.
    - `ntg`: A 3D vector representing the number of threads per thread group.
- **Control Flow**:
    - The function begins by extracting the expert ID from the thread group position.
    - It initializes a counter for the number of mappings performed.
    - It iterates over the number of tokens, checking each expert ID in the second source matrix.
    - For each matching expert ID, it copies the corresponding data from the first source matrix to the destination buffer.
    - It updates the IDs array with the new mapping for each expert.
    - Finally, it updates the temporary buffer with the total number of mappings for the expert.
- **Output**: The function does not return a value but updates the destination buffer with the mapped results and the temporary buffer with the count of mappings.


---
### kernel\_mul\_mm\_id\_map1
`kernel_mul_mm_id_map1` is a kernel function that maps input data from a source to a destination based on an index mapping.
- **Inputs**:
    - `args`: A constant structure containing parameters for the kernel, including dimensions and offsets.
    - `hdst`: A device pointer to the source data that will be mapped.
    - `hids`: A device pointer to the index mapping data.
    - `dst`: A device pointer to the destination where the mapped data will be written.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tpitg`: A 3D vector indicating the position of the thread within the thread group.
    - `ntg`: A 3D vector indicating the number of threads per thread group.
- **Control Flow**:
    - The function retrieves the expert ID and token index from the thread group position.
    - It calculates the total number of elements to be processed based on the input parameters.
    - For each token, it checks if the current expert ID matches the expected ID.
    - If it matches, it copies the corresponding data from the source to the destination based on the index mapping.
    - Finally, it updates the total count of mapped elements for the current expert.
- **Output**: The function writes the mapped data to the destination pointer `dst`, based on the index mapping provided in `hids`.


---
### kernel\_mul\_mm\_id
The `kernel_mul_mm_id` function performs matrix multiplication with an indirect mapping of input data based on specified indices.
- **Inputs**:
    - `args`: A constant structure containing parameters for the matrix multiplication operation, including dimensions and offsets.
    - `src0`: A pointer to the first input matrix data in device memory.
    - `src1`: A pointer to the second input matrix data in device memory.
    - `tpe`: A pointer to an array of indices that determine which rows of the first matrix to use.
    - `dst`: A pointer to the output matrix data in device memory where the result will be stored.
    - `shmem`: A pointer to shared memory used for intermediate calculations.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiitg`: An index indicating the thread index within the thread group.
    - `tiisg`: An index indicating the thread index within the SIMD group.
    - `sgitg`: An index indicating the SIMD group index within the thread group.
- **Control Flow**:
    - The function begins by determining the expert ID and the corresponding index for the input data.
    - It calculates the offsets for the input matrices based on the provided indices.
    - The function then checks if the current block is within the valid range of the output matrix.
    - It retrieves the appropriate rows from the first input matrix based on the indices specified in the second input matrix.
    - The function performs the matrix multiplication using the specified rows and stores the result in the output matrix.
- **Output**: The output is a matrix stored in device memory, which contains the results of the matrix multiplication based on the specified input matrices and indices.


---
### kernel\_mul\_mv\_id
The `kernel_mul_mv_id` function performs matrix-vector multiplication with quantized input data.
- **Inputs**:
    - `args`: A structure containing parameters for the multiplication operation, including dimensions and offsets.
    - `src0`: A pointer to the source matrix data, which is quantized.
    - `src1`: A pointer to the source vector data.
    - `dst`: A pointer to the destination where the result of the multiplication will be stored.
    - `ids`: A pointer to an array of indices used to access specific rows in the source matrix.
    - `shmem`: A pointer to shared memory used for intermediate calculations.
    - `tgpig`: A 3D vector indicating the position of the thread group in the grid.
    - `tiitg`: An index indicating the position of the thread within the thread group.
    - `tiisg`: An index indicating the position of the thread within the SIMD group.
    - `sgitg`: An index indicating the position of the thread group in the SIMD group.
- **Control Flow**:
    - The function begins by determining the expert ID and the index for the current token.
    - It calculates the offsets for accessing the source matrix and vector based on the provided indices.
    - The function then retrieves the quantized matrix and vector data from the respective source pointers.
    - It performs the matrix-vector multiplication using the specified implementation function.
    - Finally, the results are stored in the destination pointer.
- **Output**: The output is stored in the destination pointer, which contains the result of the matrix-vector multiplication.


---
### kernel\_pool\_2d\_max\_f32
`kernel_pool_2d_max_f32` computes the maximum value in a 2D pooling operation over a specified input tensor.
- **Inputs**:
    - `src0`: A pointer to the input tensor containing the data to be pooled.
    - `dst`: A pointer to the output tensor where the pooled results will be stored.
    - `args`: A structure containing parameters for the pooling operation, including input and output dimensions, kernel size, stride, and padding.
    - `gid`: The global thread ID used to identify which element of the output tensor to compute.
- **Control Flow**:
    - The function first checks if the global thread ID `gid` exceeds the number of parallel elements; if so, it returns early.
    - It calculates the index of the current output element based on `gid`, and determines the corresponding input tensor slice based on the pooling parameters.
    - It initializes a variable `res` to negative infinity to track the maximum value found in the pooling window.
    - The function iterates over the input tensor slice defined by the kernel size and stride, updating `res` with the maximum value found.
    - Finally, it stores the computed maximum value in the output tensor at the appropriate index.
- **Output**: The function outputs the maximum values computed from the input tensor into the destination tensor.


---
### kernel\_pool\_2d\_avg\_f32
`kernel_pool_2d_avg_f32` computes the average pooling of a 2D input tensor.
- **Inputs**:
    - `src0`: A pointer to the input tensor containing the data to be pooled.
    - `dst`: A pointer to the output tensor where the pooled results will be stored.
    - `args`: A structure containing parameters for the pooling operation, including input and output dimensions, kernel size, stride, and padding.
    - `gid`: The global thread ID used to identify which element of the output tensor to compute.
- **Control Flow**:
    - The function first checks if the global thread ID exceeds the number of parallel elements; if so, it returns early.
    - It calculates the indices for the input and output tensors based on the global thread ID.
    - It determines the start and end indices for the height and width of the pooling operation based on the kernel size, stride, and padding.
    - It initializes a variable to hold the accumulated sum of the values within the pooling window.
    - It iterates over the input tensor within the defined pooling window, accumulating the sum of the values.
    - Finally, it computes the average by dividing the accumulated sum by the area of the pooling window and stores the result in the output tensor.
- **Output**: The function outputs the average pooled values into the destination tensor.


