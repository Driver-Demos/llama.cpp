# Purpose
This source code file is an OpenCL kernel implementation designed to perform the hyperbolic tangent (tanh) operation on multi-dimensional data arrays, specifically for both 32-bit floating-point (f32) and 16-bit floating-point (f16) data types. The file includes two kernel functions, `kernel_tanh_f32_nd` and `kernel_tanh_f16_nd`, which are responsible for applying the tanh function to each element of the input tensor and storing the result in the corresponding position of the output tensor. The kernels are designed to work with multi-dimensional data, as indicated by the parameters that define the dimensions and strides of the input and output tensors.

The code also includes preprocessor directives to enable specific OpenCL extensions and define attributes for subgroup sizes, which are crucial for optimizing the execution on different GPU architectures. The extensions and attributes are conditionally enabled based on the presence of certain macros, such as `cl_intel_required_subgroup_size` and `cl_qcom_reqd_sub_group_size`, which suggest that the code is optimized for Intel and Qualcomm GPUs, respectively. This allows the kernels to leverage hardware-specific features for improved performance.

Overall, this file provides a specialized and efficient implementation of the tanh operation for use in parallel computing environments, particularly those involving GPU acceleration. It is a part of a broader computational framework that likely involves other mathematical operations and is intended to be integrated into larger applications that require high-performance numerical computations, such as machine learning or scientific simulations.
# Functions

---
### kernel\_tanh\_f32\_nd
The `kernel_tanh_f32_nd` function applies the hyperbolic tangent function to each element of a multi-dimensional float tensor and stores the result in a destination tensor.
- **Inputs**:
    - `p_src0_base`: A pointer to the base address of the source tensor in global memory.
    - `off_src0_abs`: An offset in bytes from the base address to the start of the source tensor data.
    - `p_dst_base`: A pointer to the base address of the destination tensor in global memory.
    - `off_dst_abs`: An offset in bytes from the base address to the start of the destination tensor data.
    - `ne00, ne01, ne02, ne03`: Dimensions of the source tensor.
    - `nb00, nb01, nb02, nb03`: Byte strides for each dimension of the source tensor.
    - `ne10, ne11, ne12, ne13`: Dimensions of the destination tensor.
    - `nb10, nb11, nb12, nb13`: Byte strides for each dimension of the destination tensor.
- **Control Flow**:
    - Retrieve the global IDs for the first three dimensions (i0, i1, i2) using `get_global_id` function.
    - Check if the current indices (i0, i1, i2) are within the bounds of the destination tensor dimensions (ne10, ne11, ne12).
    - Iterate over the fourth dimension (i3) of the destination tensor.
    - Calculate the source offset in bytes using the current indices and the source tensor strides.
    - Calculate the destination offset in bytes using the current indices and the destination tensor strides.
    - Retrieve the source value from the calculated source offset and apply the hyperbolic tangent function.
    - Store the result at the calculated destination offset.
- **Output**: The function does not return a value; it writes the results directly to the destination tensor in global memory.


---
### kernel\_tanh\_f16\_nd
The `kernel_tanh_f16_nd` function applies the hyperbolic tangent function to each element of a multi-dimensional tensor of half-precision floating-point numbers and stores the result in a destination tensor.
- **Inputs**:
    - `p_src0_base`: A pointer to the base of the source tensor in global memory.
    - `off_src0_abs`: An offset in bytes to the start of the source tensor from `p_src0_base`.
    - `p_dst_base`: A pointer to the base of the destination tensor in global memory.
    - `off_dst_abs`: An offset in bytes to the start of the destination tensor from `p_dst_base`.
    - `ne00, ne01, ne02, ne03`: Dimensions of the source tensor.
    - `nb00, nb01, nb02, nb03`: Byte strides for each dimension of the source tensor.
    - `ne10, ne11, ne12, ne13`: Dimensions of the destination tensor.
    - `nb10, nb11, nb12, nb13`: Byte strides for each dimension of the destination tensor.
- **Control Flow**:
    - Retrieve the global IDs for the first three dimensions using `get_global_id` for parallel execution.
    - Check if the current indices `i0`, `i1`, and `i2` are within the bounds of the destination tensor dimensions `ne10`, `ne11`, and `ne12`.
    - Iterate over the fourth dimension `i3` of the tensor.
    - Calculate the source offset in the tensor using the provided strides and indices.
    - Calculate the destination offset in the tensor using the provided strides and indices.
    - Retrieve the source value from the calculated source offset and apply the `tanh` function to it.
    - Store the result of the `tanh` function at the calculated destination offset.
- **Output**: The function does not return a value; it writes the results directly to the destination tensor in global memory.


