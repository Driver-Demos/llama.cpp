# Purpose
The provided code consists of two OpenCL kernel functions, `kernel_concat_f32_contiguous` and `kernel_concat_f32_non_contiguous`, which are designed to perform concatenation operations on floating-point data arrays. These kernels are intended to be executed on a parallel computing device, such as a GPU, to efficiently handle large-scale data processing tasks. The primary purpose of these kernels is to concatenate two source arrays (`src0` and `src1`) into a destination array (`dst`) along a specified dimension, which can be either contiguous or non-contiguous in memory.

The `kernel_concat_f32_contiguous` function handles the concatenation of arrays that are stored contiguously in memory. It uses global IDs to determine the indices for accessing elements in the destination array and checks the specified dimension (`dim`) to decide how to split the data between the two source arrays. The function calculates the appropriate indices for each element based on the provided dimensions and offsets, ensuring that data from `src0` and `src1` are correctly placed into `dst`.

The `kernel_concat_f32_non_contiguous` function, on the other hand, deals with non-contiguous memory layouts, where the data may have different strides. This kernel uses a similar approach to determine which source array to use for each element, but it also accounts for the strides of the arrays, allowing it to handle more complex memory layouts. The function iterates over the elements of the destination array, calculating the correct source indices based on the specified dimension and strides, and performs the concatenation accordingly. Both kernels are designed to be flexible and efficient, supporting a wide range of data shapes and memory layouts for concatenation operations.
# Functions

---
### kernel\_concat\_f32\_contiguous
The `kernel_concat_f32_contiguous` function concatenates two 3D float arrays along a specified dimension into a destination array, assuming contiguous memory layout.
- **Inputs**:
    - `p_src0`: Pointer to the first source array in global memory.
    - `off_src0`: Offset in bytes to the start of the first source array.
    - `p_src1`: Pointer to the second source array in global memory.
    - `off_src1`: Offset in bytes to the start of the second source array.
    - `p_dst`: Pointer to the destination array in global memory.
    - `off_dst`: Offset in bytes to the start of the destination array.
    - `d_ne00, d_ne01, d_ne02`: Dimensions of the first source array.
    - `d_ne10, d_ne11, d_ne12`: Dimensions of the second source array.
    - `d_ne0, d_ne1, d_ne2`: Dimensions of the destination array.
    - `dim`: The dimension along which the concatenation is performed.
- **Control Flow**:
    - Initialize pointers to the source and destination arrays using the provided offsets.
    - Retrieve the global indices for the current thread along each dimension of the destination array.
    - Check if the current indices are within the bounds of the destination array dimensions; if not, exit the function.
    - Calculate the linear index for the destination array based on the current indices.
    - Determine which source array to use based on the specified concatenation dimension and the current index along that dimension.
    - Calculate the linear index for the selected source array and copy the value to the destination array at the calculated index.
- **Output**: The function does not return a value; it writes the concatenated result directly into the destination array in global memory.


---
### kernel\_concat\_f32\_non\_contiguous
The function `kernel_concat_f32_non_contiguous` concatenates two non-contiguous float arrays along a specified dimension into a destination array.
- **Inputs**:
    - `p_src0`: Pointer to the first source array (as a char pointer).
    - `off_src0`: Offset in bytes for the first source array.
    - `p_src1`: Pointer to the second source array (as a char pointer).
    - `off_src1`: Offset in bytes for the second source array.
    - `p_dst`: Pointer to the destination array (as a char pointer).
    - `off_dst`: Offset in bytes for the destination array.
    - `ne00, ne01, ne02, ne03`: Dimensions of the first source array.
    - `nb00, nb01, nb02, nb03`: Strides for the first source array.
    - `nb10, nb11, nb12, nb13`: Strides for the second source array.
    - `d_ne0, d_ne1, d_ne2, d_ne3`: Dimensions of the destination array.
    - `d_nb0, d_nb1, d_nb2, d_nb3`: Strides for the destination array.
    - `dim`: The dimension along which to concatenate the arrays.
- **Control Flow**:
    - Initialize base pointers for source and destination arrays using provided offsets.
    - Retrieve global indices for the destination array dimensions.
    - Check if the current indices are within the bounds of the destination array dimensions; if not, exit the function.
    - Iterate over the first dimension of the destination array.
    - Determine whether to use the first or second source array based on the current index and the specified dimension for concatenation.
    - Calculate the pointer to the current element in the source array based on the determined source and strides.
    - Calculate the pointer to the current element in the destination array based on the current indices and destination strides.
    - Copy the value from the source array to the destination array.
- **Output**: The function does not return a value; it modifies the destination array in place by concatenating the source arrays along the specified dimension.


