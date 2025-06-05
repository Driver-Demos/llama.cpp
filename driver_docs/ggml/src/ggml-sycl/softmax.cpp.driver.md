# Purpose
This C++ source code file implements a specialized version of the softmax function using SYCL, a parallel programming model for heterogeneous computing. The code is designed to perform efficient softmax calculations on floating-point data, potentially with a mask, across multiple rows and columns. The primary function, [`soft_max_f32`](#soft_max_f32), is a templated function that computes the softmax of input data, applying scaling and bias adjustments, and is optimized for execution on SYCL-enabled devices. The function handles various configurations of input dimensions and block sizes, making it adaptable to different hardware capabilities and data sizes. The code also includes a submitter function, [`soft_max_f32_submitter`](#soft_max_f32_submitter), which sets up the SYCL kernel execution environment, and a higher-level function, [`soft_max_f32_sycl`](#soft_max_f32_sycl), which determines the appropriate kernel configuration based on the input data and device capabilities.

The file is part of a larger system, likely a machine learning or data processing library, as indicated by the use of softmax—a common function in neural networks. It provides a narrow but critical functionality focused on efficient computation of the softmax operation on SYCL-compatible devices. The code is structured to handle different data types for the mask (either half-precision or single-precision floats) and includes logic to manage device-specific constraints such as local memory size and workgroup dimensions. The [`ggml_sycl_op_soft_max`](#ggml_sycl_op_soft_max) function serves as the public API, interfacing with the rest of the system to perform the softmax operation on specified tensors, ensuring compatibility with the GGML framework's data structures and execution context.
# Imports and Dependencies

---
- `softmax.hpp`


# Functions

---
### soft\_max\_f32<!-- {{#callable:soft_max_f32}} -->
The `soft_max_f32` function computes the softmax of a matrix with optional masking and scaling, optimized for parallel execution using SYCL.
- **Inputs**:
    - `x`: Pointer to the input matrix of floats.
    - `mask`: Pointer to the mask matrix, which can be of any type T, used for optional biasing.
    - `dst`: Pointer to the destination matrix where the softmax results will be stored.
    - `ncols_par`: Number of columns in the input matrix, used if `ncols_template` is zero.
    - `nrows_y`: Number of rows in the mask matrix, used for broadcasting.
    - `scale`: Scaling factor applied to the input matrix.
    - `max_bias`: Maximum bias value used in the ALiBi (Attention Linear Bias) calculation.
    - `m0`: Base value for ALiBi calculation when head index is less than `n_head_log2`.
    - `m1`: Base value for ALiBi calculation when head index is greater than or equal to `n_head_log2`.
    - `n_head_log2`: Log base 2 of the number of heads, used in ALiBi calculation.
    - `item_ct1`: SYCL item object representing the current work item in the parallel execution.
    - `buf`: Pointer to a buffer used for intermediate calculations and reductions.
- **Control Flow**:
    - Determine the number of columns to process based on `ncols_template` and `ncols_par`.
    - Calculate thread and warp identifiers for parallel execution.
    - Compute the ALiBi slope if `max_bias` is greater than zero, using `m0`, `m1`, and `n_head_log2`.
    - Initialize a buffer for storing intermediate values, either in shared memory or directly in the destination matrix.
    - Iterate over columns in blocks, applying scaling and optional masking to compute initial values and find the maximum value in each block.
    - Perform a warp-level reduction to find the maximum value across all threads in the block.
    - Compute exponentials of the adjusted values and accumulate their sum.
    - Perform a warp-level reduction to find the sum of exponentials across all threads in the block.
    - Normalize the values by dividing each by the sum of exponentials to compute the final softmax values.
    - Store the computed softmax values in the destination matrix.
- **Output**: The function outputs the softmax-transformed values of the input matrix into the destination matrix `dst`.
- **Functions called**:
    - [`warp_reduce_max`](common.hpp.driver.md#warp_reduce_max)
    - [`warp_reduce_sum`](common.hpp.driver.md#warp_reduce_sum)


---
### soft\_max\_f32\_submitter<!-- {{#callable:soft_max_f32_submitter}} -->
The `soft_max_f32_submitter` function submits a SYCL kernel to compute the softmax operation on a given input matrix with optional masking and scaling, using specified block and thread configurations.
- **Inputs**:
    - `x`: A pointer to the input matrix of type `float`.
    - `mask`: A pointer to the mask matrix of type `T`, which is optional and can be used to modify the input values.
    - `dst`: A pointer to the output matrix where the softmax results will be stored.
    - `ncols_par`: The number of columns in the input matrix, used as a parameter if `ncols_template` is zero.
    - `nrows_y`: The number of rows in the mask matrix, used for broadcasting.
    - `scale`: A scaling factor applied to the input values.
    - `max_bias`: A bias value used in the ALiBi (Attention Linear Bias) calculation.
    - `m0`: A parameter used in the ALiBi calculation for determining the slope.
    - `m1`: Another parameter used in the ALiBi calculation for determining the slope.
    - `n_head_log2`: The logarithm base 2 of the number of heads, used in the ALiBi calculation.
    - `block_nums`: A `sycl::range<3>` object specifying the number of blocks in each dimension for the kernel execution.
    - `block_dims`: A `sycl::range<3>` object specifying the dimensions of each block for the kernel execution.
    - `n_local_scratch`: The size of the local scratch memory required for the kernel execution.
    - `stream`: A pointer to the SYCL queue where the kernel will be submitted.
- **Control Flow**:
    - The function begins by submitting a task to the provided SYCL queue `stream`.
    - A local accessor `local_buf_acc` is created for local memory usage within the kernel.
    - The function launches a parallel SYCL kernel using `cgh.parallel_for` with the specified block and thread dimensions.
    - Within the kernel, the `soft_max_f32` function is called to perform the actual softmax computation, utilizing the local buffer for temporary storage.
- **Output**: The function does not return a value; it writes the computed softmax results to the `dst` output matrix.
- **Functions called**:
    - [`get_pointer`](common.hpp.driver.md#get_pointer)


---
### soft\_max\_f32\_sycl<!-- {{#callable:soft_max_f32_sycl}} -->
The `soft_max_f32_sycl` function computes the softmax of a matrix using SYCL for parallel execution, with optional masking and scaling.
- **Inputs**:
    - `x`: A pointer to the input matrix of type float.
    - `mask`: A pointer to the mask matrix of type T, which is optional.
    - `dst`: A pointer to the destination matrix where the result will be stored.
    - `ncols_x`: The number of columns in the input matrix x.
    - `nrows_x`: The number of rows in the input matrix x.
    - `nrows_y`: The number of rows in the mask matrix.
    - `scale`: A scaling factor applied to the input values.
    - `max_bias`: A bias value used in the ALiBi (Attention Linear Bias) computation.
    - `stream`: A pointer to the SYCL queue for executing the kernel.
    - `device`: The device ID for which the kernel is being executed.
- **Control Flow**:
    - Initialize the number of threads per block (nth) to WARP_SIZE and adjust it based on the number of columns and maximum block size.
    - Calculate block dimensions and number of blocks based on the input matrix dimensions.
    - Compute the number of local scratch memory needed and check if it fits within the device's local memory size.
    - Calculate ALiBi parameters m0 and m1 based on max_bias and n_head_log2.
    - If local scratch memory is sufficient, choose a specific kernel configuration based on the number of columns (ncols_x) and call `soft_max_f32_submitter` with appropriate template parameters.
    - If local scratch memory is insufficient, call `soft_max_f32_submitter` with a default configuration.
- **Output**: The function does not return a value but writes the computed softmax values to the destination matrix `dst`.
- **Functions called**:
    - [`ggml_sycl_info`](ggml-sycl.cpp.driver.md#ggml_sycl_info)


---
### ggml\_sycl\_op\_soft\_max<!-- {{#callable:ggml_sycl_op_soft_max}} -->
The `ggml_sycl_op_soft_max` function computes the softmax operation on a tensor using SYCL for parallel processing, optionally applying a mask.
- **Inputs**:
    - `ctx`: A reference to a `ggml_backend_sycl_context` object, which provides the SYCL device and stream context for the operation.
    - `dst`: A pointer to a `ggml_tensor` object that serves as both the destination for the softmax result and the source of input data and parameters.
- **Control Flow**:
    - The function begins by asserting that the first source tensor and the destination tensor are of type `GGML_TYPE_F32` and that the optional second source tensor (mask) is either `GGML_TYPE_F16` or `GGML_TYPE_F32`.
    - It retrieves the dimensions of the input tensor and initializes `scale` and `max_bias` from the operation parameters stored in `dst`.
    - The function sets the SYCL device and retrieves the main stream from the context.
    - Depending on the type of the optional mask tensor (`src[1]`), it calls `soft_max_f32_sycl` with the appropriate template type (`sycl::half` or `float`) to perform the softmax operation, passing the input data, mask, and other parameters.
    - If no mask is provided, it calls `soft_max_f32_sycl` with a `nullptr` for the mask.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place to contain the result of the softmax operation.
- **Functions called**:
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_sycl_set_device`](common.hpp.driver.md#ggml_sycl_set_device)


