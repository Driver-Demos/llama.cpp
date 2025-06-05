# Purpose
This source code file contains two OpenCL kernel functions, `kernel_sub` and `kernel_sub_row`, which are designed to perform element-wise subtraction operations on data arrays. The file is intended to be used in a parallel computing environment, leveraging the capabilities of OpenCL to execute operations across multiple processing units. The code is structured to handle data in a highly parallelized manner, making it suitable for applications that require high-performance computations, such as graphics processing or scientific simulations.

The `kernel_sub` function is a more generalized kernel that performs element-wise subtraction between two source arrays, `src0` and `src1`, and stores the result in a destination array, `dst`. It uses a combination of global and local IDs to determine the specific elements to operate on, allowing it to efficiently handle multi-dimensional data. The function is designed to work with data that is organized in a complex, multi-dimensional structure, as indicated by the numerous offset and size parameters. This kernel is capable of broadcasting elements from `src1` across `src0`, which is a common operation in matrix and tensor computations.

The `kernel_sub_row` function is a specialized kernel that assumes `src1` is a row vector and performs a broadcast subtraction across `src0`. It uses a more straightforward indexing method to achieve this, which is optimized for performance by avoiding the modulus operation. This kernel is particularly useful for operations where a single row of data needs to be subtracted from multiple rows in a larger dataset, a common requirement in linear algebra and machine learning tasks. Both kernels are designed to be executed on a GPU or other parallel processing hardware, taking advantage of OpenCL's ability to distribute computations across many cores.
# Functions

---
### kernel\_sub
The `kernel_sub` function performs element-wise subtraction between two global memory buffers, `src0` and `src1`, and stores the result in a third buffer, `dst`, using OpenCL parallel processing.
- **Inputs**:
    - `src0`: A global memory buffer of type `char*` representing the first source array.
    - `offset0`: An unsigned long integer representing the offset to be added to `src0`.
    - `src1`: A global memory buffer of type `char*` representing the second source array.
    - `offset1`: An unsigned long integer representing the offset to be added to `src1`.
    - `dst`: A global memory buffer of type `char*` where the result of the subtraction will be stored.
    - `offsetd`: An unsigned long integer representing the offset to be added to `dst`.
    - `nb00`: An unsigned long integer representing the byte stride for the first dimension of `src0`.
    - `nb01`: An unsigned long integer representing the byte stride for the second dimension of `src0`.
    - `nb02`: An unsigned long integer representing the byte stride for the third dimension of `src0`.
    - `nb03`: An unsigned long integer representing the byte stride for the fourth dimension of `src0`.
    - `ne10`: An integer representing the extent of the first dimension of `src1`.
    - `ne11`: An integer representing the extent of the second dimension of `src1`.
    - `ne12`: An integer representing the extent of the third dimension of `src1`.
    - `ne13`: An integer representing the extent of the fourth dimension of `src1`.
    - `nb10`: An unsigned long integer representing the byte stride for the first dimension of `src1`.
    - `nb11`: An unsigned long integer representing the byte stride for the second dimension of `src1`.
    - `nb12`: An unsigned long integer representing the byte stride for the third dimension of `src1`.
    - `nb13`: An unsigned long integer representing the byte stride for the fourth dimension of `src1`.
    - `ne0`: An integer representing the extent of the first dimension of the operation.
    - `nb0`: An unsigned long integer representing the byte stride for the first dimension of `dst`.
    - `nb1`: An unsigned long integer representing the byte stride for the second dimension of `dst`.
    - `nb2`: An unsigned long integer representing the byte stride for the third dimension of `dst`.
    - `nb3`: An unsigned long integer representing the byte stride for the fourth dimension of `dst`.
- **Control Flow**:
    - Adjust the pointers `src0`, `src1`, and `dst` by their respective offsets.
    - Retrieve the group IDs for the third, second, and first dimensions using `get_group_id`.
    - Calculate indices `i13`, `i12`, and `i11` using modulo operations with `ne13`, `ne12`, and `ne11` respectively.
    - Compute pointers `src0_ptr`, `src1_ptr`, and `dst_ptr` using the calculated indices and byte strides.
    - Iterate over the local work items using a for loop, incrementing by the local size.
    - Within the loop, calculate `i10` using modulo operation with `ne10`.
    - Perform element-wise subtraction between the elements pointed by `src0_ptr` and `src1_ptr`, and store the result in `dst_ptr`.
- **Output**: The function does not return a value; it writes the result of the subtraction directly into the `dst` buffer.


---
### kernel\_sub\_row
The `kernel_sub_row` function performs element-wise subtraction of a broadcasted row from a source array and stores the result in a destination array using OpenCL.
- **Inputs**:
    - `src0`: A global pointer to the source array of type `float4` from which elements will be subtracted.
    - `offset0`: An unsigned long integer representing the byte offset to be applied to `src0`.
    - `src1`: A global pointer to the source row array of type `float4` that will be broadcasted and subtracted from `src0`.
    - `offset1`: An unsigned long integer representing the byte offset to be applied to `src1`.
    - `dst`: A global pointer to the destination array of type `float4` where the result of the subtraction will be stored.
    - `offsetd`: An unsigned long integer representing the byte offset to be applied to `dst`.
    - `ne`: An integer representing the number of elements in the row `src1` that will be broadcasted.
- **Control Flow**:
    - Adjust the pointers `src0`, `src1`, and `dst` by their respective offsets to point to the correct starting positions in memory.
    - Calculate the global ID `gid` for the current work item using `get_global_id(0)`.
    - Compute the index `idx1` for the `src1` array by calculating `gid - (gid/ne)*ne`, which is equivalent to `gid % ne`.
    - Perform element-wise subtraction of `src1[idx1]` from `src0[gid]` and store the result in `dst[gid]`.
- **Output**: The function does not return a value; it writes the result of the subtraction directly into the `dst` array.


