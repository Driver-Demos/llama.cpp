# Purpose
This source code is a GLSL (OpenGL Shading Language) compute shader designed to perform matrix-vector multiplication operations. The shader is written for the OpenGL 4.5 version and utilizes the `GL_EXT_shader_explicit_arithmetic_types_int32` extension to handle explicit 32-bit integer arithmetic types. The primary function of this shader is to compute the product of a matrix and a vector, leveraging parallel processing capabilities of the GPU to efficiently handle large-scale computations. The shader is structured to work with blocks of data, using shared memory to cache intermediate results and optimize performance.

The code is organized around two main functions: `calc_superblock` and `compute_outputs`. The `calc_superblock` function is responsible for processing a "superblock" of data, which involves unpacking and manipulating data from input buffers, performing arithmetic operations, and storing results in a temporary buffer. It uses bitwise operations and vectorized arithmetic to efficiently handle data, taking advantage of the GPU's parallel processing capabilities. The `compute_outputs` function orchestrates the overall computation by determining the offsets and dimensions of the data blocks to be processed, and it calls `calc_superblock` iteratively to process each block. It also handles the reduction of results into the final output.

The shader is designed to be executed in a parallel fashion, with each workgroup handling a portion of the data. The `main` function sets up the execution by determining the starting row for each workgroup and invoking `compute_outputs` to perform the matrix-vector multiplication. The use of shared memory, barriers for synchronization, and unrolling of loops are key technical components that enhance the performance of the shader. This code is a specialized component within a larger graphics or compute pipeline, providing a focused functionality for high-performance matrix computations on the GPU.
# Global Variables

---
### sccache
- **Type**: `FLOAT_TYPE[2][BLOCK_SIZE/16][2][8]`
- **Description**: The `sccache` variable is a shared memory cache used within the compute shader to store intermediate results of matrix-vector multiplications. It is a 4-dimensional array where the dimensions are likely used to represent different aspects of the computation, such as different blocks, threads, or vector components. The use of `FLOAT_TYPE` suggests that it can be configured to use different floating-point precisions depending on the shader's requirements.
- **Use**: `sccache` is used to temporarily store and access intermediate computation results across different threads within a workgroup, facilitating efficient parallel processing.


---
### temp
- **Type**: `FLOAT_TYPE[][]`
- **Description**: The `temp` variable is a two-dimensional array of type `FLOAT_TYPE` with dimensions `[NUM_COLS][NUM_ROWS]`. It is used to store intermediate results during the computation of matrix operations in the shader program.
- **Use**: `temp` is used to accumulate and store the results of matrix-vector multiplications for each column and row during the execution of the `calc_superblock` and `compute_outputs` functions.


---
### csel
- **Type**: `uint`
- **Description**: The variable `csel` is a global unsigned integer used within the shader program. It is initialized to 0 and is used to toggle between two states, likely for indexing purposes in a shared cache or buffer.
- **Use**: `csel` is used to alternate between two indices in the `sccache` shared memory array during the execution of the `calc_superblock` function.


# Functions

---
### calc\_superblock
The `calc_superblock` function performs matrix block calculations using shared memory and thread synchronization in a compute shader.
- **Inputs**:
    - `a_offset`: The offset for accessing data in matrix A.
    - `b_offset`: The offset for accessing data in matrix B.
    - `ix`: The index of the current thread group in the x dimension.
    - `itid8`: The thread index within a group, modulo 8.
    - `v_im`: A value indicating which half of the data is being processed (0 or 1).
    - `v_im4`: A value derived from `v_im` used for bit shifting operations.
    - `v_in`: The thread index within a group, adjusted for `v_im`.
    - `hm_m`: An array of 4 uint32_t values used for bitmask operations.
    - `q_offset`: The offset for accessing quantized data.
    - `y_offset`: The offset for accessing data in matrix B, adjusted for the current row.
    - `s_shift`: A shift value used for accessing scale data.
    - `i`: The current block index being processed.
    - `num_blocks_per_row`: The number of blocks per row in the matrix.
    - `first_row`: The index of the first row being processed.
    - `num_rows`: The number of rows to process.
    - `all_threads`: A boolean indicating whether all threads are used for processing.
- **Control Flow**:
    - Calculate the index `y_idx` based on the current block and row offsets.
    - Iterate over each row to be processed, updating the block index `ib0` and toggling the `csel` variable.
    - If not all threads are used, check if the current block index `i` is within the number of blocks per row and update the shared cache `sccache` accordingly.
    - Calculate the bitmask `hmk` and unpack it into four vectors `hmk_0` to `hmk_3`.
    - Retrieve and unpack quantized data `qs_u32` into four vectors `qs_u32_0` to `qs_u32_6`.
    - If all threads are used, update the shared cache `sccache` with scale data.
    - Retrieve the scale factor `d` for the current block.
    - Iterate over each column, retrieve data from matrix B, and compute the sum using fused multiply-add operations with the shared cache and unpacked data.
    - Update the temporary storage `temp` with the computed sum for each column.
- **Output**: The function does not return a value; it updates the shared cache `sccache` and temporary storage `temp` with computed results for matrix block operations.


---
### compute\_outputs
The `compute_outputs` function calculates and reduces matrix block outputs for a given range of rows using parallel processing in a compute shader.
- **Inputs**:
    - `first_row`: The starting row index for the computation.
    - `num_rows`: The number of rows to process starting from the first_row.
- **Control Flow**:
    - Initialize offsets for matrices A, B, and D using `get_offsets` function.
    - Calculate the number of blocks per row based on the number of columns and a constant `QUANT_K`.
    - Determine thread-specific indices and offsets for processing blocks using local invocation IDs and workgroup size.
    - Initialize a temporary storage array `temp` to zero for accumulating results.
    - Calculate shift and mask values for processing data blocks in parallel.
    - Iterate over blocks that can be processed by all threads and call `calc_superblock` with `all_threads` set to true.
    - Process any remaining blocks that require partial thread usage by calling `calc_superblock` with `all_threads` set to false.
    - Reduce the results stored in `temp` and store them in the output matrix using `reduce_result`.
- **Output**: The function does not return a value but writes the computed results to a shared output matrix.


---
### main
The `main` function orchestrates the computation of matrix-vector multiplication in a parallelized manner using GPU shader programming.
- **Inputs**: None
- **Control Flow**:
    - Calculate the starting row index `first_row` based on the work group ID and number of work groups.
    - Check if the number of rows to process (`NUM_ROWS`) fits within the remaining rows (`p.stride_d`).
    - If there are enough rows, call `compute_outputs` with `NUM_ROWS`; otherwise, call it with the remaining rows.
    - If `first_row` exceeds `p.stride_d`, exit the function early.
- **Output**: The function does not return a value; it performs computations and writes results to a shared memory or buffer.


