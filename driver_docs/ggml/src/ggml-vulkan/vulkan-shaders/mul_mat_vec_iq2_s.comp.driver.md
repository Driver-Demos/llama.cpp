# Purpose
This source code is a GLSL (OpenGL Shading Language) compute shader designed to perform matrix-vector multiplication using a specialized approach that involves quantization and parallel processing. The shader is written for the OpenGL 4.5 version and utilizes the `GL_EXT_shader_explicit_arithmetic_types_int32` extension to handle explicit arithmetic types. The code is structured to be executed on the GPU, leveraging the parallel processing capabilities of modern graphics hardware to efficiently compute the results of matrix-vector operations.

The primary functionality of this shader is encapsulated in two main functions: `calc_superblock` and `compute_outputs`. The `calc_superblock` function is responsible for processing a "superblock" of data, which involves reading quantized matrix data, applying scaling factors, and performing arithmetic operations to compute intermediate results. This function uses a combination of bit manipulation and vectorized operations to handle the quantized data efficiently. The `compute_outputs` function orchestrates the overall computation by initializing temporary storage, iterating over blocks of data, and invoking `calc_superblock` to perform the necessary calculations. It also calls `reduce_result` to finalize the computation and store the results.

The shader is designed to be executed in a parallel fashion, with each workgroup handling a portion of the data. The use of local and global workgroup identifiers allows the shader to distribute the workload across multiple threads, maximizing the utilization of the GPU's processing power. The `main` function serves as the entry point, setting up the initial conditions and determining the range of data to be processed based on the workgroup's position. This shader is a specialized component within a larger graphics or compute pipeline, likely part of a system that requires high-performance matrix operations, such as machine learning inference or real-time graphics rendering.
# Global Variables

---
### temp
- **Type**: `FLOAT_TYPE[][]`
- **Description**: The `temp` variable is a two-dimensional array of type `FLOAT_TYPE`, with dimensions defined by `NUM_COLS` and `NUM_ROWS`. It is used to store intermediate results during the computation of matrix operations in the shader program.
- **Use**: This variable is used to accumulate and store the results of floating-point arithmetic operations performed in the `calc_superblock` function, which are later reduced and outputted in the `compute_outputs` function.


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
    - Calculate the y-index and nibble shift based on input parameters.
    - Determine the initial block index 'ibi' using the offsets and block indices.
    - Iterate over each row specified by 'num_rows'.
    - For each row, compute a scaling factor and a transformed data value 'db'.
    - Unpack and process quantized data from 'data_a' and 'data_a_packed16'.
    - Iterate over two sub-elements of the quantized data, applying sign and grid transformations.
    - For each column in 'NUM_COLS', retrieve and process data from 'data_b_v4'.
    - Compute a sum using fused multiply-add operations with the transformed data and store it in the temporary buffer 'temp'.
    - Increment the block index 'ibi' for the next row.
- **Output**: The function does not return a value but updates the global 'temp' array with computed results for each column and row processed.


---
### compute\_outputs
The `compute_outputs` function calculates and stores the results of matrix-vector multiplications for a specified range of rows using parallel processing.
- **Inputs**:
    - `first_row`: The starting row index for the computation.
    - `num_rows`: The number of rows to process starting from the first_row.
- **Control Flow**:
    - Initialize offsets for matrices A, B, and D using `get_offsets` function.
    - Calculate the number of blocks per row based on the number of columns and a constant `QUANT_K`.
    - Determine the number of blocks each workgroup will process and the thread's local ID and index within the workgroup.
    - Initialize a temporary storage array `temp` to zero for accumulating results.
    - Iterate over blocks assigned to the current workgroup and call `calc_superblock` to compute partial results for each block.
    - Call `reduce_result` to finalize and store the computed results from `temp` into the output matrix.
- **Output**: The function does not return a value but updates the global output matrix with computed results for the specified rows.


---
### main
The `main` function initializes shared memory and coordinates the computation of matrix-vector products over a range of rows, handling edge cases where the number of rows is less than expected.
- **Inputs**: None
- **Control Flow**:
    - Calculate the starting row index `first_row` based on the workgroup ID and number of workgroups.
    - Initialize shared memory using `init_iq_shmem` with the workgroup size.
    - Check if the number of rows to process (`first_row + NUM_ROWS`) is within the bounds of `p.stride_d`.
    - If within bounds, call `compute_outputs` to process `NUM_ROWS` rows starting from `first_row`.
    - If not within bounds, check if `first_row` is beyond `p.stride_d` and return if true.
    - Otherwise, call `compute_outputs` to process the remaining rows from `first_row` to `p.stride_d`.
- **Output**: The function does not return a value; it performs computations and writes results to a shared memory or global memory space.


