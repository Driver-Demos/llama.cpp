# Purpose
This source code is a GLSL (OpenGL Shading Language) compute shader designed to perform matrix-vector multiplication using a parallel processing approach. The shader is written for the OpenGL 4.5 version and utilizes the `GL_EXT_shader_explicit_arithmetic_types_int32` extension to handle explicit 32-bit integer arithmetic types. The code is structured to execute on the GPU, leveraging the parallel nature of graphics hardware to efficiently compute the results of matrix-vector operations, which are common in graphics and scientific computing applications.

The shader defines two main functions: `calc_superblock` and `compute_outputs`. The `calc_superblock` function is responsible for processing a "superblock" of data, which involves iterating over rows and columns of the input matrices and performing arithmetic operations to compute partial results. It uses a combination of bit manipulation and floating-point arithmetic to handle packed data formats, which are optimized for GPU processing. The `compute_outputs` function orchestrates the overall computation by initializing temporary storage, invoking `calc_superblock` for each block of data, and then reducing the results to produce the final output.

The `main` function serves as the entry point for the shader, setting up the initial conditions and determining the range of data to process based on the workgroup and grid dimensions. It ensures that the computation is performed in chunks of `NUM_ROWS`, handling any remaining rows if the total number of rows is not a multiple of `NUM_ROWS`. This shader is a specialized piece of code that provides narrow functionality focused on matrix-vector multiplication, optimized for execution on a GPU to take advantage of its parallel processing capabilities.
# Global Variables

---
### temp
- **Type**: `FLOAT_TYPE[][]`
- **Description**: The variable `temp` is a two-dimensional array of type `FLOAT_TYPE` with dimensions `NUM_COLS` by `NUM_ROWS`. It is used to store intermediate results of matrix-vector multiplication operations within the compute shader.
- **Use**: This variable is used to accumulate and store the results of floating-point operations performed in the `calc_superblock` function, which are later reduced and outputted in the `compute_outputs` function.


# Functions

---
### calc\_superblock
The `calc_superblock` function computes a matrix multiplication operation on a subset of data blocks, applying specific transformations and storing intermediate results in a temporary buffer.
- **Inputs**:
    - `a_offset`: The offset in the data array 'a' from which to start processing.
    - `b_offset`: The offset in the data array 'b' from which to start processing.
    - `itid`: The thread index within a workgroup, ranging from 0 to 15.
    - `i`: The current block index being processed.
    - `num_blocks_per_row`: The number of blocks per row in the data matrix.
    - `first_row`: The index of the first row to process.
    - `num_rows`: The number of rows to process.
- **Control Flow**:
    - Calculate the y-index based on the current block index and thread index.
    - Determine the block index in the data array 'a' using the provided offsets and indices.
    - Iterate over the number of rows specified, processing each row in the loop.
    - For each row, calculate a scaling factor 'db' using data from 'data_a' and 'data_a_packed16'.
    - Iterate twice to process two sets of data per row, extracting and transforming data using bit manipulation and vector operations.
    - For each column, compute a sum using fused multiply-add operations with transformed data and store the result in the temporary buffer 'temp'.
    - Increment the block index 'ibi' for the next row.
- **Output**: The function does not return a value but updates the global temporary buffer 'temp' with computed results for each column and row processed.


---
### compute\_outputs
The `compute_outputs` function calculates and stores the results of matrix-vector multiplications for a specified range of rows using parallel processing.
- **Inputs**:
    - `first_row`: The starting row index from which the computation begins.
    - `num_rows`: The number of rows to process starting from the first_row.
- **Control Flow**:
    - Initialize offsets for matrices A and B using `get_offsets` function.
    - Calculate the number of blocks per row based on the number of columns and a constant `QUANT_K`.
    - Determine the number of blocks each workgroup will process and the thread's local ID within the workgroup.
    - Initialize a temporary storage array `temp` to zero for accumulating results.
    - Iterate over blocks assigned to the current workgroup and call `calc_superblock` to compute partial results for each block.
    - After processing all blocks, call `reduce_result` to finalize and store the computed results.
- **Output**: The function does not return a value but updates a global memory buffer with the computed results of the matrix-vector multiplication.


---
### main
The `main` function initializes shared memory and computes matrix-vector multiplication outputs for a specified range of rows, handling edge cases where the number of rows is less than expected.
- **Inputs**: None
- **Control Flow**:
    - Calculate the starting row index `first_row` based on the work group ID and number of work groups.
    - Initialize shared memory using `init_iq_shmem` with the work group size.
    - Check if the number of rows to process (`first_row + NUM_ROWS`) is within the bounds of `p.stride_d`.
    - If within bounds, call `compute_outputs` with `first_row` and `NUM_ROWS`.
    - If not within bounds, check if `first_row` is beyond `p.stride_d` and return if true.
    - Otherwise, call `compute_outputs` with `first_row` and the remaining rows (`p.stride_d - first_row`).
- **Output**: The function does not return a value; it performs computations and writes results to a shared memory or buffer.


