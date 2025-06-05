# Purpose
This source code is a GLSL compute shader designed to perform matrix-vector multiplication using a specialized approach that involves processing data in blocks or "superblocks." The shader is written for the OpenGL Shading Language version 450 and utilizes the `GL_EXT_shader_explicit_arithmetic_types_int32` extension to handle explicit arithmetic operations with 32-bit integers. The code is structured to be executed on a GPU, leveraging parallel processing capabilities to efficiently handle large-scale computations, which is typical in graphics and scientific computing applications.

The shader defines two main functions, `calc_superblock` and `compute_outputs`, which are responsible for the core computation tasks. The `calc_superblock` function processes a block of data, performing arithmetic operations on matrix and vector elements, and accumulates the results in a temporary buffer. It uses a combination of bit manipulation and arithmetic operations to handle data scaling and sign adjustments, which suggests that the data might be quantized or compressed in some form. The `compute_outputs` function orchestrates the execution of `calc_superblock` across multiple blocks, initializing temporary storage and managing the distribution of work across the GPU's workgroups and threads.

The `main` function serves as the entry point for the shader, setting up the initial conditions and determining the range of data to be processed based on the workgroup and grid dimensions. It ensures that the computation is performed in chunks of `NUM_ROWS`, handling any remaining rows that do not fit into a complete chunk. This shader is part of a larger system, likely involving other shaders or host-side code, that manages data input and output, as well as the overall execution flow. The inclusion of the `mul_mat_vec_base.comp` file suggests that this shader builds upon a base implementation, possibly providing additional functionality or optimizations specific to the application's requirements.
# Global Variables

---
### temp
- **Type**: `FLOAT_TYPE[][]`
- **Description**: The variable `temp` is a two-dimensional array of type `FLOAT_TYPE` with dimensions `NUM_COLS` by `NUM_ROWS`. It is used to store intermediate results during the computation of matrix operations in the shader program.
- **Use**: `temp` is used to accumulate and store the results of matrix-vector multiplications and is updated within the `calc_superblock` and `compute_outputs` functions.


# Functions

---
### calc\_superblock
The `calc_superblock` function computes a matrix-vector multiplication for a specific block of data using quantized and packed data formats.
- **Inputs**:
    - `a_offset`: The offset in the data array for matrix A.
    - `b_offset`: The offset in the data array for matrix B.
    - `ib32`: An index used for accessing specific parts of the data, particularly for scaling and quantization.
    - `i`: The current block index being processed.
    - `num_blocks_per_row`: The number of blocks per row in the matrix.
    - `first_row`: The index of the first row to process.
    - `num_rows`: The number of rows to process.
- **Control Flow**:
    - Calculate the y-index using the block index and ib32.
    - Initialize the ibi index for accessing data in matrix A.
    - Loop over each row to be processed, calculating the scaled data value and quantization scale.
    - Initialize a sum array for each column to zero.
    - Loop over a fixed number of iterations to unpack and process quantized data, updating the sum for each column using fused multiply-add operations.
    - Update the temporary storage with the scaled sum for each column.
    - Increment the ibi index to move to the next block in the row.
- **Output**: The function updates the global `temp` array with computed values for each column and row processed.


---
### compute\_outputs
The `compute_outputs` function calculates and reduces matrix multiplication results for a specified range of rows using parallel processing in a compute shader.
- **Inputs**:
    - `first_row`: The starting row index for the computation.
    - `num_rows`: The number of rows to process starting from the first_row.
- **Control Flow**:
    - Initialize offsets for matrices A, B, and D using `get_offsets` function.
    - Calculate the number of blocks per row based on the number of columns and a constant `QUANT_K`.
    - Determine the number of blocks each workgroup will process and the thread's local ID within the workgroup.
    - Initialize a temporary storage array `temp` to zero for accumulating results.
    - Iterate over blocks assigned to the current thread, calling `calc_superblock` to compute partial results for each block.
    - Call `reduce_result` to finalize and store the computed results from `temp` into the output matrix.
- **Output**: The function does not return a value; it writes the computed results to a global output matrix, reducing the results from the temporary storage.


---
### main
The `main` function orchestrates the computation of matrix-vector multiplication using a parallelized approach in a shader program.
- **Inputs**: None
- **Control Flow**:
    - Calculate the `first_row` based on the work group ID and number of work groups.
    - Initialize shared memory for intermediate calculations using `init_iq_shmem`.
    - Check if the number of rows to process (`first_row + NUM_ROWS`) is within the bounds of `p.stride_d`.
    - If within bounds, call `compute_outputs` with `NUM_ROWS`; otherwise, check if `first_row` is beyond `p.stride_d` and return if true.
    - If `first_row` is not beyond `p.stride_d`, call `compute_outputs` with the remaining rows (`p.stride_d - first_row`).
- **Output**: The function does not return a value; it performs computations and updates global memory as part of a shader program.


