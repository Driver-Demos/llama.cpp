# Purpose
This source code file contains OpenCL kernels designed for performing addition operations on tensors. The primary functionality provided by this file is the addition of two tensors, with support for non-contiguous tensors and broadcasting across specific dimensions. The file includes two distinct kernels: `kernel_add` and `kernel_add_row`. The `kernel_add` function is a general-purpose kernel that facilitates the addition of two tensors, accommodating non-contiguous data and supporting broadcasting across dimensions 1, 2, and 3. However, it is noted for its lack of efficiency. The `kernel_add_row` function is optimized for scenarios where the second source tensor (`src1`) is a row, and it broadcasts this row across the first source tensor (`src0`).

The technical components of the file include the use of OpenCL's parallel computing capabilities, such as work-group and work-item identifiers, to efficiently perform operations on large datasets. The kernels utilize global memory pointers and offsets to access and manipulate data, and they employ broadcasting techniques to handle tensor operations across different dimensions. The file also enables the `cl_khr_fp16` extension, which suggests support for half-precision floating-point operations, although the kernels themselves operate on `float` and `float4` data types.

Overall, this file is a specialized library intended for use in environments that require tensor operations, particularly in applications involving machine learning or scientific computing where tensor manipulations are common. The kernels defined here are not standalone executables but are designed to be invoked within a larger OpenCL context, providing essential functionality for tensor addition with specific optimizations for broadcasting scenarios.
# Functions

---
### kernel\_add
The `kernel_add` function is an OpenCL kernel that performs element-wise addition of two tensors, supporting non-contiguous tensors and broadcasting across specific dimensions.
- **Inputs**:
    - `src0`: A global pointer to the first source tensor, represented as a character array.
    - `offset0`: An offset in bytes to be added to the `src0` pointer.
    - `src1`: A global pointer to the second source tensor, represented as a character array.
    - `offset1`: An offset in bytes to be added to the `src1` pointer.
    - `dst`: A global pointer to the destination tensor, represented as a character array.
    - `offsetd`: An offset in bytes to be added to the `dst` pointer.
    - `ne00, ne01, ne02, ne03`: Dimensions of the first source tensor `src0`.
    - `nb00, nb01, nb02, nb03`: Byte strides for each dimension of the first source tensor `src0`.
    - `ne10, ne11, ne12, ne13`: Dimensions of the second source tensor `src1`.
    - `nb10, nb11, nb12, nb13`: Byte strides for each dimension of the second source tensor `src1`.
    - `ne0, ne1, ne2, ne3`: Dimensions of the destination tensor `dst`.
    - `nb0, nb1, nb2, nb3`: Byte strides for each dimension of the destination tensor `dst`.
- **Control Flow**:
    - Adjust the pointers `src0`, `src1`, and `dst` by their respective offsets.
    - Retrieve the group IDs for the third, second, and first dimensions to determine the current work group.
    - Calculate indices `i13`, `i12`, and `i11` for the second source tensor `src1` using modulo operations with its dimensions.
    - Compute pointers `src0_ptr`, `src1_ptr`, and `dst_ptr` for the current work group based on the calculated indices and byte strides.
    - Iterate over the local work items using a loop, adjusting the index `i0` by the local size.
    - Within the loop, calculate the index `i10` for `src1` using modulo operation with its first dimension.
    - Perform element-wise addition of the corresponding elements from `src0` and `src1`, storing the result in `dst`.
- **Output**: The function does not return a value; it writes the result of the addition directly to the `dst` tensor in global memory.


---
### kernel\_add\_row
The `kernel_add_row` function performs element-wise addition of a row tensor `src1` to a larger tensor `src0`, storing the result in `dst`, with broadcasting support for the row tensor.
- **Inputs**:
    - `src0`: A global pointer to the first input tensor, represented as a float4 array, which will be added to the broadcasted row tensor.
    - `offset0`: An offset in bytes to be applied to the `src0` pointer to reach the starting point of the data.
    - `src1`: A global pointer to the second input tensor, represented as a float4 array, which is a row tensor to be broadcasted and added to `src0`.
    - `offset1`: An offset in bytes to be applied to the `src1` pointer to reach the starting point of the data.
    - `dst`: A global pointer to the output tensor, represented as a float4 array, where the result of the addition will be stored.
    - `offsetd`: An offset in bytes to be applied to the `dst` pointer to reach the starting point of the data.
    - `ne`: An integer representing the number of elements in the row tensor `src1` to be broadcasted.
- **Control Flow**:
    - Adjust the pointers `src0`, `src1`, and `dst` by their respective offsets to point to the correct starting positions in memory.
    - Calculate the global ID `gid` for the current work item using `get_global_id(0)`.
    - Compute the index `idx1` for the row tensor `src1` using `gid - (gid/ne)*ne`, which is equivalent to `gid % ne` but optimized for performance.
    - Perform element-wise addition of `src0[gid]` and `src1[idx1]`, storing the result in `dst[gid]`.
- **Output**: The function outputs the result of the element-wise addition of `src0` and the broadcasted `src1` into the `dst` tensor, with each element being a `float4`.


