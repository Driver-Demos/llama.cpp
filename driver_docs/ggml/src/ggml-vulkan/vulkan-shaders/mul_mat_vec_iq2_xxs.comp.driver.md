# Purpose
This source code is a GLSL compute shader designed for parallel processing on a GPU. It is specifically tailored for matrix-vector multiplication, as indicated by the inclusion of "mul_mat_vec_base.comp" and the operations performed within the `calc_superblock` and `compute_outputs` functions. The shader utilizes explicit arithmetic types and is structured to handle large-scale computations by dividing the workload into smaller blocks, which are processed concurrently by multiple threads within a workgroup. The use of `layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;` suggests that the shader is configured to operate with a flexible number of threads in the x-dimension, allowing for dynamic adjustment based on the input data size.

The core functionality of the shader is encapsulated in the `calc_superblock` function, which performs the matrix-vector multiplication for a specific block of data. This function leverages unrolling and fused multiply-add (FMA) operations to optimize performance, taking advantage of the GPU's parallel processing capabilities. The shader processes data in a packed format, using bit manipulation and vector operations to efficiently handle the arithmetic operations required for the multiplication. The `compute_outputs` function orchestrates the overall computation, initializing temporary storage, invoking `calc_superblock` for each block, and finally reducing the results to produce the final output.

The `main` function serves as the entry point for the shader, determining the range of rows to process based on the workgroup and grid dimensions. It initializes shared memory and manages the execution flow to ensure that all necessary computations are completed before the results are reduced and stored. This shader is a specialized component of a larger graphics or compute pipeline, designed to be executed on a GPU to leverage its parallel processing power for efficient matrix-vector multiplication.
# Global Variables

---
### temp
- **Type**: `FLOAT_TYPE[][]`
- **Description**: The `temp` variable is a two-dimensional array of type `FLOAT_TYPE`, with dimensions defined by `NUM_COLS` and `NUM_ROWS`. It is used to store intermediate results during the computation of matrix operations in the shader program.
- **Use**: `temp` is used to accumulate and store intermediate floating-point results for each column and row during the execution of the `calc_superblock` and `compute_outputs` functions.


# Functions

---
### calc\_superblock
The `calc_superblock` function computes a matrix-vector multiplication for a specific block of data using packed and unpacked data formats, updating a temporary result matrix.
- **Inputs**:
    - `a_offset`: The offset in the data array 'a' to start reading from.
    - `b_offset`: The offset in the data array 'b' to start reading from.
    - `itid`: The thread index within a workgroup, ranging from 0 to 15.
    - `i`: The current block index being processed.
    - `num_blocks_per_row`: The number of blocks per row in the matrix.
    - `first_row`: The index of the first row to process.
    - `num_rows`: The number of rows to process.
- **Control Flow**:
    - Calculate the y-index for accessing data in array 'b' using the block index 'i' and thread index 'itid'.
    - Determine the block index 'ibi' in array 'a' using the offsets and block index 'i'.
    - Iterate over each row 'n' up to 'num_rows'.
    - For each row, compute a scaling factor 'db' using data from 'a' and a packed signscale value.
    - Iterate over two sub-blocks 'l' within each row, extracting quantized values and sign bits.
    - For each sub-block, unpack grid values and iterate over each column 'j'.
    - For each column, load vector data from 'b', compute a weighted sum using fused multiply-add operations, and update the temporary result matrix 'temp'.
    - Increment the block index 'ibi' by the number of blocks per row for the next iteration.
- **Output**: The function updates the global temporary matrix 'temp' with computed values for each column and row processed.


---
### compute\_outputs
The `compute_outputs` function calculates and reduces matrix block results for a given range of rows using parallel processing in a compute shader.
- **Inputs**:
    - `first_row`: The starting row index for the computation.
    - `num_rows`: The number of rows to process starting from the first_row.
- **Control Flow**:
    - Initialize offsets for matrices A, B, and D using `get_offsets` function.
    - Calculate the number of blocks per row based on the number of columns and a constant `QUANT_K`.
    - Determine the number of blocks each workgroup will process and the thread's local ID within the workgroup.
    - Initialize a temporary storage array `temp` to zero for accumulating results.
    - Iterate over blocks assigned to the current workgroup and call `calc_superblock` to compute partial results for each block.
    - After processing all blocks, call `reduce_result` to finalize and store the results.
- **Output**: The function does not return a value but updates the global memory with computed results for the specified rows.


---
### main
The `main` function initializes shared memory and coordinates the computation of matrix-vector products across workgroups, handling edge cases where the number of rows to process is less than the defined block size.
- **Inputs**:
    - `None`: The function does not take any direct input parameters, but it relies on global variables and constants such as `gl_WorkGroupID`, `gl_NumWorkGroups`, and `p.stride_d`.
- **Control Flow**:
    - Calculate `first_row` based on the workgroup's ID and the number of workgroups.
    - Initialize shared memory using `init_iq_shmem` with the workgroup size.
    - Check if the number of rows to process (`first_row + NUM_ROWS`) is within the bounds of `p.stride_d`.
    - If within bounds, call `compute_outputs` with `first_row` and `NUM_ROWS`.
    - If not within bounds, check if `first_row` is beyond `p.stride_d` and return if true.
    - Otherwise, call `compute_outputs` with `first_row` and the remaining rows (`p.stride_d - first_row`).
- **Output**: The function does not return a value; it performs computations and writes results to global memory.


