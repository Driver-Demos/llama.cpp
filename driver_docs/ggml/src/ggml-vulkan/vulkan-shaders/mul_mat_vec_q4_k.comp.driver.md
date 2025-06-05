# Purpose
This source code is a GLSL (OpenGL Shading Language) compute shader designed to perform matrix-vector multiplication operations on the GPU. The shader is written for version 450 of GLSL and utilizes the `GL_EXT_shader_explicit_arithmetic_types_int32` extension, which allows for explicit handling of 32-bit integer arithmetic types. The shader is structured to handle computations in parallel, leveraging the GPU's architecture to efficiently process large datasets.

The primary functionality of this shader is encapsulated in the `calc_superblock` and `compute_outputs` functions. The `calc_superblock` function is responsible for processing a "superblock" of data, which involves unpacking and scaling integer data, performing fused multiply-add (FMA) operations, and storing intermediate results in a temporary buffer. The `compute_outputs` function orchestrates the overall computation by determining offsets, initializing temporary storage, and invoking `calc_superblock` for each block of data. It also calls `reduce_result` to finalize the computation and store the results.

The shader is designed to be executed in a parallel fashion, with each workgroup handling a portion of the data. The `main` function sets up the execution by determining the starting row for each workgroup and invoking `compute_outputs` to process the data. The shader is highly optimized for performance, using techniques such as loop unrolling and vectorized operations to maximize throughput on the GPU. This code is part of a larger system that likely involves other shaders and host-side code to manage data transfer and execution control.
# Global Variables

---
### temp
- **Type**: `FLOAT_TYPE[][]`
- **Description**: The `temp` variable is a two-dimensional array of type `FLOAT_TYPE` with dimensions `NUM_COLS` by `NUM_ROWS`. It is used to store intermediate results during the computation of matrix operations in the shader program.
- **Use**: `temp` is used to accumulate and store intermediate results of matrix computations, which are later reduced and used in further calculations.


# Functions

---
### calc\_superblock
The `calc_superblock` function performs matrix-vector multiplication and scaling operations on a subset of data blocks, storing the results in a temporary buffer.
- **Inputs**:
    - `a_offset`: The offset in the data array for matrix A.
    - `b_offset`: The offset in the data array for matrix B.
    - `v_im`: An index used to determine which part of the data to process, either 0 or 1.
    - `q_offset`: The offset for quantized data in the data array.
    - `y_offset`: The offset for the y-index in the data array.
    - `i`: The current block index being processed.
    - `num_blocks_per_row`: The number of blocks per row in the matrix.
    - `first_row`: The index of the first row to process.
    - `num_rows`: The number of rows to process.
- **Control Flow**:
    - Calculate indices y1_idx and y2_idx based on input i and y_offset.
    - Iterate over each row (n) from 0 to num_rows.
    - Calculate the block index ib0 for the current row and block.
    - Retrieve and convert data from data_a and data_a_packed16 arrays to floating-point scales and quantized values.
    - Unpack and convert quantized scales and values to floating-point vectors.
    - Iterate over each column (j) from 0 to NUM_COLS.
    - Retrieve and convert data from data_b_v4 array to floating-point vectors by10, by132, by20, and by232.
    - Perform fused multiply-add (fma) operations to compute sx, sy, sz, sw, and smin using the unpacked scales and quantized values.
    - Update the temporary buffer temp[j][n] with the computed values using fma operations.
- **Output**: The function does not return a value; it updates the global temporary buffer `temp` with computed results for each block and row.


---
### compute\_outputs
The `compute_outputs` function calculates and reduces the results of matrix-vector multiplications for a specified range of rows in a parallelized manner using GPU threads.
- **Inputs**:
    - `first_row`: The starting row index from which the computation begins.
    - `num_rows`: The number of rows to process starting from the first_row.
- **Control Flow**:
    - Initialize offsets for matrices A, B, and D using `get_offsets` function.
    - Calculate the number of blocks per row based on the number of columns and a constant `QUANT_K`.
    - Determine the size of iterations and thread identifiers based on the local workgroup size and local invocation ID.
    - Calculate various indices and offsets for processing data blocks and vectors.
    - Initialize a temporary storage array `temp` to zero for accumulating results.
    - Iterate over blocks of data, calling `calc_superblock` to perform matrix-vector multiplications and accumulate results in `temp`.
    - Call `reduce_result` to finalize and store the computed results from `temp` into the output matrix.
- **Output**: The function does not return a value but updates the output matrix with computed results from the matrix-vector multiplications.


---
### main
The `main` function orchestrates the computation of matrix-vector multiplication in a parallelized manner using GPU shader capabilities.
- **Inputs**: None
- **Control Flow**:
    - Calculate the starting row index `first_row` based on the work group and grid dimensions.
    - Check if the number of rows to process (`NUM_ROWS`) fits within the remaining rows (`p.stride_d`).
    - If there are enough rows, call `compute_outputs` with `NUM_ROWS`; otherwise, call it with the remaining rows.
    - If `first_row` exceeds `p.stride_d`, exit the function early.
- **Output**: The function does not return a value; it performs computations and writes results to a global buffer.


