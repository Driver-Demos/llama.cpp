# Purpose
This code is a GLSL (OpenGL Shading Language) compute shader designed to perform 2D depthwise convolution operations on input data. The shader is structured to handle two different data layouts, WHCN (Width-Height-Channel-Number) and CWHN (Channel-Width-Height-Number), which are common in neural network computations. The shader uses push constants to define convolution parameters such as the number of elements, batch size, channel count, and dimensions of the source, destination, and kernel data. These parameters are crucial for configuring the convolution operation to match the specific requirements of the input data and desired output.

The shader includes two main functions, `conv_2d_dw_whcn` and `conv_2d_dw_cwhn`, each tailored to handle a specific data layout. These functions compute the convolution by iterating over the kernel dimensions and applying the specified stride, padding, and dilation parameters. The shader uses buffer objects to read input data (`src_data` and `knl_data`) and write the output (`dst_data`). The use of `fma` (fused multiply-add) operations within the convolution functions optimizes the computation by reducing rounding errors and improving performance.

The `main` function orchestrates the execution of the shader by calculating a global index for each invocation and determining which convolution function to execute based on a preprocessor directive (`WHCN`). This design allows the shader to be flexible and adaptable to different data layouts, making it suitable for integration into larger graphics or compute pipelines where efficient convolution operations are required. The shader's use of layout qualifiers and buffer bindings ensures that data is correctly aligned and accessible during execution, which is critical for achieving high performance on GPU hardware.
# Functions

---
### conv\_2d\_dw\_whcn
The `conv_2d_dw_whcn` function performs a 2D depthwise convolution on input data using a specified kernel and returns the computed result for a given index.
- **Inputs**:
    - `idx`: An unsigned integer representing the index of the output element to compute.
- **Control Flow**:
    - Calculate the output coordinates (dst_x, dst_y) and the batch and channel indices (n, c) from the input index.
    - Determine the starting indices for the source data and kernel data based on the batch and channel indices.
    - Initialize a sum variable to accumulate the convolution result.
    - Iterate over the kernel height (knl_h) and width (knl_w) to compute the convolution.
    - For each kernel position, calculate the corresponding source data position, applying stride, dilation, and padding adjustments.
    - Check if the calculated source position is within bounds; if not, skip the current kernel position.
    - Convert the source and kernel data to FLOAT_TYPE and perform a fused multiply-add operation to accumulate the result.
    - Return the accumulated sum as the convolution result for the given index.
- **Output**: A FLOAT_TYPE value representing the result of the 2D depthwise convolution for the specified index.


---
### conv\_2d\_dw\_cwhn
The `conv_2d_dw_cwhn` function performs a 2D depthwise convolution on input data using a specified kernel, considering the CWHN (Channels, Width, Height, Number of batches) data layout.
- **Inputs**:
    - `idx`: An unsigned integer representing the index of the output element to compute.
- **Control Flow**:
    - Calculate the channel index `c`, destination x-coordinate `dst_x`, batch index `n`, and destination y-coordinate `dst_y` from the input index `idx`.
    - Initialize `src_i` to point to the start of the source data for the current batch and `src_row` and `knl_row` to represent the width of a row in the source and kernel data, respectively.
    - Initialize `sum` to accumulate the convolution result.
    - Iterate over each kernel row `knl_y`, compute the corresponding source y-coordinate `src_y`, and skip the iteration if `src_y` is out of bounds.
    - Within each kernel row, iterate over each kernel column `knl_x`, compute the corresponding source x-coordinate `src_x`, and skip the iteration if `src_x` is out of bounds.
    - For each valid source position, retrieve the source value `v` and kernel value `k`, and update `sum` using the fused multiply-add operation `fma(v, k, sum)`.
    - Return the accumulated `sum` as the convolution result for the given index.
- **Output**: The function returns a `FLOAT_TYPE` value representing the result of the convolution operation for the specified index.


---
### main
The `main` function performs a 2D depthwise convolution on input data using either WHCN or CWHN data layout and writes the result to an output buffer.
- **Inputs**:
    - `gl_GlobalInvocationID`: A built-in variable that provides the global invocation ID for the current work item, used to calculate the index for processing.
- **Control Flow**:
    - Calculate the index `idx` using the global invocation IDs and a fixed size of 262144 and 512 for the y and x dimensions respectively.
    - Check if `idx` is greater than or equal to `p.ne`, and if so, return early to avoid processing out-of-bounds data.
    - Depending on the preprocessor directive `WHCN`, call either `conv_2d_dw_whcn(idx)` or `conv_2d_dw_cwhn(idx)` to perform the convolution operation.
    - Store the result of the convolution in the `dst_data` buffer at the position `idx`.
- **Output**: The function does not return a value but writes the convolution result to the `dst_data` buffer at the calculated index.


