# Purpose
This source code is a GLSL (OpenGL Shading Language) compute shader designed to perform matrix-vector multiplication operations on the GPU. The shader is written for OpenGL version 4.5 and utilizes the `GL_EXT_shader_explicit_arithmetic_types_int32` extension to handle explicit 32-bit integer arithmetic. The code is structured to handle computations in parallel using the GPU's compute capabilities, which is evident from the use of `layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;` to define the workgroup size and the use of `gl_WorkGroupID` and `gl_LocalInvocationID` to manage thread execution.

The shader includes a function `calc_superblock` that performs the core computation of multiplying a matrix with a vector. This function processes data in blocks, leveraging the GPU's parallel processing power to handle multiple elements simultaneously. It uses various bit manipulation techniques to unpack and scale data, which suggests that the input data is packed in a compact format to optimize memory usage and bandwidth. The function `compute_outputs` orchestrates the computation by determining the offsets and iterating over the data blocks, calling `calc_superblock` for each block, and finally reducing the results.

The `main` function serves as the entry point for the shader, determining the range of rows to process based on the workgroup's ID and the total number of workgroups. It ensures that the computation is performed in chunks of `NUM_ROWS`, adjusting for any remaining rows that do not fit into a complete chunk. This shader is a specialized piece of code focused on efficient matrix-vector multiplication, likely used in applications requiring high-performance linear algebra computations, such as graphics rendering, scientific simulations, or machine learning inference on the GPU.
# Global Variables

---
### temp
- **Type**: `FLOAT_TYPE[][]`
- **Description**: The `temp` variable is a two-dimensional array of type `FLOAT_TYPE` with dimensions `NUM_COLS` by `NUM_ROWS`. It is used to store intermediate results during the computation of matrix operations in the shader program.
- **Use**: `temp` is used to accumulate and store results of matrix-vector multiplications and other arithmetic operations within the `calc_superblock` and `compute_outputs` functions.


# Functions

---
### calc\_superblock
The `calc_superblock` function computes a superblock of matrix data using various offsets and scales, performing complex arithmetic operations on packed data.
- **Inputs**:
    - `a_offset`: The offset for accessing data in matrix A.
    - `b_offset`: The offset for accessing data in matrix B.
    - `v_im`: An index used to determine which part of the data to compute, either 0 or 1.
    - `l0`: An index used for accessing quantized data.
    - `q_offset`: The offset for accessing quantized scales.
    - `y_offset`: The offset for accessing data in matrix B.
    - `i`: The current block index being processed.
    - `num_blocks_per_row`: The number of blocks per row in the matrix.
    - `first_row`: The index of the first row to process.
    - `num_rows`: The number of rows to process.
- **Control Flow**:
    - Calculate indices y1_idx and y2_idx based on input i and y_offset.
    - Iterate over each row (n) from 0 to num_rows.
    - For each row, calculate the base index ib0 for accessing data in matrix A.
    - Retrieve and unpack scale values from packed data structures.
    - Unpack quantized values and apply offsets to them.
    - Iterate over each column (j) from 0 to NUM_COLS.
    - For each column, retrieve and process data from matrix B using offsets and indices.
    - Perform fused multiply-add (fma) operations to compute intermediate results sx, sy, sz, sw, and smin.
    - Update the temp array with the computed values using fma operations.
- **Output**: The function does not return a value; it updates the global temp array with computed results for each superblock.


---
### compute\_outputs
The `compute_outputs` function calculates and reduces the results of matrix-vector multiplications for a specified range of rows in a parallelized manner using GPU threads.
- **Inputs**:
    - `first_row`: The starting row index for the computation.
    - `num_rows`: The number of rows to process starting from the first_row.
- **Control Flow**:
    - Initialize offsets for matrices A, B, and D using `get_offsets` function.
    - Calculate the number of blocks per row based on the number of columns and a constant `QUANT_K`.
    - Determine the size of iterations (`it_size`) and thread identifiers (`tid`, `itid`, `ix`) based on the local workgroup size.
    - Calculate indices `il`, `ir`, `v_im`, `v_in`, `l0`, `q_offset`, and `y_offset` for accessing data within the superblock.
    - Initialize a temporary storage array `temp` to zero for storing intermediate results.
    - Iterate over blocks of rows (`i`) and call `calc_superblock` to perform matrix-vector multiplication and store results in `temp`.
    - Call `reduce_result` to finalize and store the computed results from `temp` into the output matrix.
- **Output**: The function does not return a value; it modifies global state by writing the computed results to a specified output location.


---
### main
The `main` function orchestrates the computation of matrix-vector products in a parallelized manner using GPU shaders, processing rows in blocks and handling edge cases for remaining rows.
- **Inputs**: None
- **Control Flow**:
    - Calculate the starting row index `first_row` based on the workgroup and grid dimensions.
    - Check if the number of rows to process (`first_row + NUM_ROWS`) is within the bounds of `p.stride_d`.
    - If within bounds, call `compute_outputs` to process `NUM_ROWS` starting from `first_row`.
    - If not within bounds, check if `first_row` is beyond `p.stride_d` and return if true.
    - Otherwise, call `compute_outputs` to process the remaining rows from `first_row` to `p.stride_d`.
- **Output**: The function does not return a value; it performs computations and writes results to a global memory buffer.


