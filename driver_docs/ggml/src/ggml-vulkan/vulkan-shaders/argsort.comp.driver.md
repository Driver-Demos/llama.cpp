# Purpose
This code is a compute shader written in GLSL (OpenGL Shading Language) version 450, designed to perform a parallel bitonic sort on a dataset. The shader is intended to be executed on a GPU, leveraging its parallel processing capabilities to efficiently sort data. The primary functionality of this shader is to sort elements in a buffer using the bitonic sort algorithm, which is well-suited for parallel execution due to its predictable data access patterns and ability to be broken down into smaller sorting networks.

The shader defines several key components, including a local workgroup size specified by `BLOCK_SIZE`, which determines the number of threads that will execute in parallel. It uses two buffer objects: a read-only buffer `A` that contains the data to be sorted, and a writable buffer `D` where the sorted indices are stored. The shader also utilizes a push constant block to pass parameters such as the number of columns (`ncols`), padded columns (`ncols_pad`), and the sorting order (`order`), which can be ascending or descending.

The main function of the shader implements the bitonic sort algorithm. It initializes indices in a shared memory array `dst_row`, and then iteratively performs sorting operations using a series of nested loops. The `swap` function is used to exchange elements based on comparisons, and synchronization is achieved using barriers to ensure correct data dependencies across threads. The final sorted indices are written back to the buffer `D`, making the results available for further processing or retrieval by the host application. This shader is a specialized component within a larger graphics or compute pipeline, providing efficient sorting capabilities for applications such as graphics rendering, data analysis, or scientific computing.
# Functions

---
### swap
The `swap` function exchanges the values at two specified indices in a shared integer array.
- **Inputs**:
    - `idx0`: The first index in the shared array `dst_row` whose value is to be swapped.
    - `idx1`: The second index in the shared array `dst_row` whose value is to be swapped.
- **Control Flow**:
    - Retrieve the value at index `idx0` from the shared array `dst_row` and store it in a temporary variable `tmp`.
    - Assign the value at index `idx1` in `dst_row` to the position at index `idx0`.
    - Assign the value stored in `tmp` to the position at index `idx1` in `dst_row`.
- **Output**: The function does not return a value; it modifies the shared array `dst_row` in place.


---
### main
The `main` function performs a parallel bitonic sort on a shared buffer using GPU compute shaders.
- **Inputs**:
    - `gl_LocalInvocationID.x`: The local invocation index in the x-dimension, representing the column index for the current thread.
    - `gl_WorkGroupID.y`: The workgroup index in the y-dimension, representing the row index for the current thread.
    - `p.ncols`: The number of columns in the data to be sorted.
    - `p.ncols_pad`: The padded number of columns, used for alignment in the sorting process.
    - `p.order`: The sorting order, where `ASC` indicates ascending order.
    - `data_a`: The input buffer containing the data to be sorted.
    - `data_d`: The output buffer where sorted indices will be stored.
- **Control Flow**:
    - Initialize the column index `col` and row index `row` based on the local and workgroup IDs.
    - Calculate the row offset using the row index and the number of columns.
    - Initialize the `dst_row` shared buffer with column indices if within the padded column limit.
    - Use a barrier to synchronize threads after initialization.
    - Perform a bitonic sort using nested loops: the outer loop iterates over the sequence length `k`, doubling each time, and the inner loop iterates over the half-sequence length `j`, halving each time.
    - Calculate `ixj` as the bitwise XOR of `col` and `j` to determine the partner index for comparison and potential swapping.
    - Check conditions based on the sorting order and column indices to decide whether to swap elements in `dst_row`.
    - Use a barrier to synchronize threads after each potential swap.
    - After sorting, store the sorted indices from `dst_row` into the output buffer `data_d` if within the column limit.
- **Output**: The function outputs sorted indices into the buffer `data_d`, representing the order of elements in `data_a` according to the specified sorting order.


