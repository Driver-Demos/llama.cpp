# Purpose
This code is a GLSL compute shader designed for performing matrix-vector multiplication with specific optimizations for different data types and quantization formats. The shader is written in GLSL version 450 and utilizes the `GL_EXT_shader_explicit_arithmetic_types_int32` extension to handle explicit arithmetic operations on 32-bit integers. The shader is structured to work with a variety of data formats, including floating-point and quantized formats, and it adapts its processing strategy based on the data type being used, as indicated by the preprocessor directives that define `K_PER_ITER`.

The shader's primary functionality is encapsulated in the `iter` and `compute_outputs` functions. The `iter` function performs the core matrix-vector multiplication operations, leveraging vectorized operations and dequantization to efficiently compute the results. It uses loop unrolling to optimize the performance of these operations, particularly when handling quantized data. The `compute_outputs` function orchestrates the overall computation, determining the number of iterations required and managing the loop unrolling strategy to ensure efficient execution. It also handles edge cases where the number of columns (`p.ncols`) does not align perfectly with the block size, ensuring that out-of-bounds accesses are avoided.

The shader is designed to be executed in a parallel computing environment, as indicated by the use of `gl_LocalInvocationID` and `gl_WorkGroupID` to manage work distribution across different threads. This allows the shader to efficiently handle large-scale matrix-vector multiplication tasks by distributing the workload across multiple compute units. The inclusion of the `main` function, which serves as the entry point for the shader, further emphasizes its role as an executable component within a larger graphics or compute pipeline. The shader is likely part of a broader system that requires high-performance linear algebra computations, such as a graphics rendering engine or a machine learning inference engine.
# Global Variables

---
### a\_offset
- **Type**: `uint`
- **Description**: The `a_offset` variable is a global unsigned integer that represents an offset value used in matrix operations within the shader program. It is used to adjust the starting index for accessing elements in a data structure, specifically in the context of dequantization and matrix multiplication operations.
- **Use**: `a_offset` is used to calculate the starting index for accessing matrix data during dequantization and matrix multiplication operations.


---
### b\_offset
- **Type**: `uint`
- **Description**: The `b_offset` variable is a global unsigned integer that represents an offset value used in accessing elements of a data structure, specifically `data_b` or `data_b_v4`, within the shader program. It is used to calculate the starting index for accessing elements in a buffer, which is crucial for operations like matrix multiplication and data fetching in the shader.
- **Use**: `b_offset` is used to determine the starting index for accessing elements in the `data_b` or `data_b_v4` buffers during shader execution.


---
### d\_offset
- **Type**: `uint`
- **Description**: The `d_offset` is a global variable of type `uint` that represents an offset value used in the computation of outputs in a shader program. It is one of several offset variables, including `a_offset` and `b_offset`, which are used to adjust indices when accessing data arrays. The `d_offset` is specifically used in the `reduce_result` function to determine where to store the computed results.
- **Use**: `d_offset` is used to adjust the index for storing computed results in the `reduce_result` function.


---
### y\_offset
- **Type**: `uint`
- **Description**: The `y_offset` variable is a global unsigned integer that is used to determine an offset value for accessing elements in a data structure, specifically within the context of matrix operations in a shader program. It is calculated based on the quantization factor `QUANT_R` and the constant `QUANT_K`, which are related to the data format and processing requirements.
- **Use**: `y_offset` is used to adjust the index when accessing elements in the data structure `data_b` during matrix multiplication operations.


# Functions

---
### iter
The `iter` function performs a matrix-vector multiplication for a specified block of rows and columns, updating a temporary result matrix with the computed values.
- **Inputs**:
    - `temp`: A 2D array of type FLOAT_TYPE with dimensions [NUM_COLS][NUM_ROWS] that stores intermediate results of the matrix-vector multiplication.
    - `first_row`: An unsigned integer representing the index of the first row to process in the matrix.
    - `num_rows`: An unsigned integer indicating the number of rows to process starting from 'first_row'.
    - `tid`: An unsigned integer representing the thread ID used for parallel processing.
    - `i`: An unsigned integer representing the current iteration index, used to calculate the column index.
    - `lastiter`: A boolean flag indicating whether the current iteration is the last one, used to handle out-of-bounds conditions.
- **Control Flow**:
    - The function iterates over each column index 'j' from 0 to NUM_COLS.
    - For each column, it calculates the column index 'col' based on the iteration index 'i', block size, and thread ID.
    - It computes quantization indices 'iqs' and 'iybs' for accessing data blocks.
    - Depending on the value of K_PER_ITER, it either fetches a pair of vectors 'bv0' and 'bv1' or single elements 'b0' and 'b1' from the data source 'data_b'.
    - For each row index 'n' from 0 to num_rows, it calculates the block index 'ib' and updates 'ibi'.
    - If K_PER_ITER is 8, it dequantizes two vectors 'v' and 'v2', applies scaling and offset if necessary, and performs matrix multiplication using dot products, updating 'temp[j][n]'.
    - If K_PER_ITER is not 8, it dequantizes a single vector 'v', performs fused multiply-add operations with 'b0' and 'b1', and updates 'temp[j][n]'.
- **Output**: The function updates the 'temp' array with the results of the matrix-vector multiplication for the specified rows and columns.


---
### compute\_outputs
The `compute_outputs` function performs matrix multiplication and accumulation for a specified range of rows in a parallelized manner using GPU shaders.
- **Inputs**:
    - `first_row`: The index of the first row to process in the matrix.
    - `num_rows`: The number of rows to process starting from the first_row.
- **Control Flow**:
    - Retrieve the local thread ID using `gl_LocalInvocationID.x`.
    - Calculate offsets for matrices A, B, and D using `get_offsets` and adjust `a_offset` and `y_offset` based on quantization parameters.
    - Initialize a temporary matrix `temp` with dimensions `NUM_COLS` by `NUM_ROWS` to zero.
    - Determine the number of iterations required to process the columns based on `p.ncols`, `K_PER_ITER`, and `BLOCK_SIZE`.
    - Adjust the number of iterations to ensure all columns are processed, considering edge cases for odd dimensions.
    - Use a loop to iterate over the columns, partially unrolling the loop for performance optimization.
    - Call the `iter` function within the loop to perform matrix multiplication and accumulation for each block of columns.
    - After processing all iterations, call `reduce_result` to finalize the computation and store the results.
- **Output**: The function does not return a value; it modifies the global state by writing the computed results to a specified output buffer.


---
### main
The `main` function orchestrates the computation of matrix-vector multiplication using a GPU shader, handling different data types and quantization formats.
- **Inputs**:
    - `None`: The function does not take any direct input parameters, but it uses global variables and constants defined in the shader environment.
- **Control Flow**:
    - Calculate the starting row index for the current workgroup using `gl_WorkGroupID` and `gl_NumWorkGroups`.
    - Optionally initialize shared memory for quantization if `NEEDS_INIT_IQ_SHMEM` is defined.
    - Check if the current workgroup can process a full block of `NUM_ROWS`; if so, call `compute_outputs` with `NUM_ROWS`.
    - If not enough rows remain, adjust the number of rows to process and call `compute_outputs` with the remaining rows.
    - Exit the function if the starting row index is beyond the data stride `p.stride_d`.
- **Output**: The function does not return a value; it performs computations and writes results to global memory.


