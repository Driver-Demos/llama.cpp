# Purpose
This source code file contains two OpenCL kernel functions, `kernel_mul` and `kernel_mul_row`, which are designed to perform element-wise multiplication operations on data arrays. The file is intended to be used in a parallel computing environment, leveraging the capabilities of OpenCL to execute operations across multiple processing units. The code is structured to handle data in a highly parallelized manner, making it suitable for applications that require high-performance computations, such as graphics processing or scientific simulations.

The `kernel_mul` function is a general-purpose kernel that multiplies elements from two source arrays (`src0` and `src1`) and stores the result in a destination array (`dst`). It uses a series of offsets and strides to navigate through the multi-dimensional data, allowing for flexible data manipulation. The function utilizes OpenCL's work-group and work-item concepts to distribute the workload across available compute units, ensuring efficient parallel execution. The use of global memory pointers and offsets indicates that this kernel is designed to handle large datasets, typical in high-performance computing scenarios.

The `kernel_mul_row` function is a specialized kernel that assumes the second source array (`src1`) is a single row, which is broadcast across the first source array (`src0`). This function is optimized for scenarios where a row-wise multiplication is required, and it avoids the use of the modulus operator for performance reasons. Instead, it calculates the index using arithmetic operations, which can be more efficient on certain hardware architectures. Both kernels are designed to be integrated into larger OpenCL programs, providing essential functionality for matrix and vector operations.
# Functions

---
### kernel\_mul
The `kernel_mul` function performs element-wise multiplication of two multi-dimensional arrays, broadcasting one array across the other, and stores the result in a destination array.
- **Inputs**:
    - `src0`: A global pointer to the first source array, which is a multi-dimensional array of characters.
    - `offset0`: An offset in bytes to be added to the base address of src0.
    - `src1`: A global pointer to the second source array, which is a multi-dimensional array of characters.
    - `offset1`: An offset in bytes to be added to the base address of src1.
    - `dst`: A global pointer to the destination array where the result of the multiplication will be stored.
    - `offsetd`: An offset in bytes to be added to the base address of dst.
    - `ne00, ne01, ne02, ne03`: Dimensions of the first source array, src0.
    - `nb00, nb01, nb02, nb03`: Byte strides for each dimension of the first source array, src0.
    - `ne10, ne11, ne12, ne13`: Dimensions of the second source array, src1.
    - `nb10, nb11, nb12, nb13`: Byte strides for each dimension of the second source array, src1.
    - `ne0, ne1, ne2, ne3`: Dimensions of the destination array, dst.
    - `nb0, nb1, nb2, nb3`: Byte strides for each dimension of the destination array, dst.
- **Control Flow**:
    - Adjust the base pointers of src0, src1, and dst by their respective offsets.
    - Retrieve the group IDs for the third, second, and first dimensions to determine the current work group.
    - Calculate indices for src1 using modulo operations to handle broadcasting.
    - Compute pointers to the current positions in src0, src1, and dst based on the group IDs and byte strides.
    - Iterate over the first dimension of the destination array using the local ID and local size to parallelize the operation.
    - For each element in the iteration, perform element-wise multiplication of the corresponding elements from src0 and src1, and store the result in dst.
- **Output**: The function does not return a value; it writes the result of the element-wise multiplication into the destination array, dst.


---
### kernel\_mul\_row
The `kernel_mul_row` function performs element-wise multiplication of a source array with a broadcasted row vector and stores the result in a destination array.
- **Inputs**:
    - `src0`: A global pointer to the source array of type `float4` that will be multiplied by the broadcasted row vector.
    - `offset0`: An offset in bytes to be added to the `src0` pointer to get the actual starting point of the source array.
    - `src1`: A global pointer to the row vector of type `float4` that will be broadcasted and multiplied with `src0`.
    - `offset1`: An offset in bytes to be added to the `src1` pointer to get the actual starting point of the row vector.
    - `dst`: A global pointer to the destination array of type `float4` where the result of the multiplication will be stored.
    - `offsetd`: An offset in bytes to be added to the `dst` pointer to get the actual starting point of the destination array.
    - `ne`: An integer representing the number of elements in the row vector `src1`.
- **Control Flow**:
    - Adjust the pointers `src0`, `src1`, and `dst` by adding their respective offsets to point to the correct starting positions in memory.
    - Retrieve the global ID of the current work item using `get_global_id(0)`.
    - Calculate the index `idx1` for the row vector `src1` by using the formula `gid - (gid/ne)*ne`, which is equivalent to `gid % ne`.
    - Perform element-wise multiplication of the `gid`-th element of `src0` with the `idx1`-th element of `src1` and store the result in the `gid`-th position of `dst`.
- **Output**: The function does not return a value; it writes the result of the element-wise multiplication directly into the `dst` array.


