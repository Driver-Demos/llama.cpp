# Purpose
This C++ source file is designed to implement SYCL-based parallel computing kernels for processing sequences in a neural network context, specifically for the RWKV (Recurrent Weighted Key-Value) model. The file defines two main kernel functions, [`rwkv_wkv6_f32_kernel`](#rwkv_wkv6_f32_kernel) and [`rwkv_wkv7_f32_kernel`](#rwkv_wkv7_f32_kernel), which are responsible for performing computations on sequences of data using SYCL, a parallel computing framework that allows for execution on heterogeneous platforms. These kernels are designed to handle operations involving key-value pairs and time-mixing parameters, leveraging shared memory and vectorized operations to optimize performance. The kernels are templated on a block size, which is set to a constant value of 64, aligning with CUDA block size conventions.

The file also includes two functions, [`ggml_sycl_op_rwkv_wkv6`](#ggml_sycl_op_rwkv_wkv6) and [`ggml_sycl_op_rwkv_wkv7`](#ggml_sycl_op_rwkv_wkv7), which serve as interfaces to launch the respective kernels. These functions configure the execution environment, including memory allocation and kernel launch parameters, and ensure that the data is correctly prepared and passed to the kernels. The functions utilize SYCL's parallel execution model to distribute the workload across available compute units, making use of local memory for efficient data access. The code is structured to be part of a larger library or application, likely related to machine learning or neural network processing, and is intended to be integrated with other components that manage data and execution contexts.
# Imports and Dependencies

---
- `sycl/sycl.hpp`
- `wkv.hpp`


# Global Variables

---
### WKV\_BLOCK\_SIZE
- **Type**: ``int``
- **Description**: `WKV_BLOCK_SIZE` is a global constant integer set to 64, which is used to define the block size for CUDA operations. It is specifically matched to `CUDA_WKV_BLOCK_SIZE`, ensuring consistency in block size across different parts of the code.
- **Use**: This variable is used to define the block size for SYCL kernels, ensuring that the operations are performed with the correct block size for optimal performance.


# Functions

---
### rwkv\_wkv6\_f32\_kernel<!-- {{#callable:rwkv_wkv6_f32_kernel}} -->
The `rwkv_wkv6_f32_kernel` function processes sequences of data using a parallelized approach with SYCL, performing operations on key, value, and other parameters to compute a weighted sum and update state information.
- **Inputs**:
    - `B`: The number of batches.
    - `T`: The total number of tokens in the sequence.
    - `C`: The number of channels.
    - `H`: The number of heads.
    - `k`: Pointer to the key data array.
    - `v`: Pointer to the value data array.
    - `r`: Pointer to the r parameter data array.
    - `tf`: Pointer to the time-mixing forward parameter data array.
    - `td`: Pointer to the time-mixing decay parameter data array.
    - `s`: Pointer to the initial state data array.
    - `dst`: Pointer to the destination array where results are stored.
    - `item_ct1`: SYCL item object providing information about the current work item.
    - `shared_mem`: Pointer to shared memory used for intermediate calculations.
- **Control Flow**:
    - Initialize thread and block identifiers using SYCL's `nd_item` object.
    - Calculate head size, batch index, head index, state size, and number of sequence tokens per batch.
    - Set up pointers for shared memory to store intermediate data for keys, r, tf, and td.
    - Load the initial state from the input state array into a local state array.
    - Synchronize threads to ensure shared memory is ready for use.
    - Load time-mixing forward parameters into shared memory and synchronize threads again.
    - Iterate over the sequence tokens, processing each token in parallel across threads.
    - Load current timestep data (k, r, td) into shared memory and synchronize threads.
    - For each token, compute the key-value product and accumulate a weighted sum using vectorized operations.
    - Update the local state with the computed values and store the results in the destination array.
    - After processing all tokens, save the final state back to the destination array.
- **Output**: The function outputs the computed weighted sum for each token in the sequence and updates the state information in the destination array.


---
### rwkv\_wkv7\_f32\_kernel<!-- {{#callable:rwkv_wkv7_f32_kernel}} -->
The `rwkv_wkv7_f32_kernel` function is a SYCL kernel that processes sequences of data using a block-based approach to perform computations involving multiple input arrays and updates a destination array with the results.
- **Inputs**:
    - `B`: The number of batches.
    - `T`: The total number of tokens.
    - `C`: The number of channels.
    - `H`: The number of heads.
    - `r`: Pointer to the array of r values.
    - `w`: Pointer to the array of w values.
    - `k`: Pointer to the array of k values.
    - `v`: Pointer to the array of v values.
    - `a`: Pointer to the array of a values.
    - `b`: Pointer to the array of b values.
    - `s`: Pointer to the array of initial state values.
    - `dst`: Pointer to the destination array where results are stored.
    - `item_ct1`: SYCL item object providing information about the current work item.
    - `shared_mem`: Pointer to shared memory used for intermediate calculations.
- **Control Flow**:
    - Initialize thread and block indices using SYCL item object.
    - Calculate head size, batch index, head index, state size, and number of sequence tokens.
    - Set up shared memory pointers for r, w, k, a, and b arrays.
    - Initialize local state array from the input state array s.
    - Iterate over sequence tokens for the current batch and head, updating shared memory with current timestep data.
    - Synchronize threads using barriers before and after shared memory operations.
    - Compute intermediate values using vectorized operations with sycl::float4 for better performance.
    - Update the local state array and accumulate results into the destination array dst.
    - Store the final state back into the destination array.
- **Output**: The function updates the destination array `dst` with computed results and stores the final state for each head and batch.


---
### ggml\_sycl\_op\_rwkv\_wkv6<!-- {{#callable:ggml_sycl_op_rwkv_wkv6}} -->
The function `ggml_sycl_op_rwkv_wkv6` executes a SYCL kernel for processing RWKV6 model operations on a tensor using parallel computation.
- **Inputs**:
    - `ctx`: A reference to a `ggml_backend_sycl_context` object, which provides the SYCL stream for kernel execution.
    - `dst`: A pointer to a `ggml_tensor` object, which contains the destination tensor and its source data tensors for the operation.
- **Control Flow**:
    - Initialize a debug print scope for the function with the destination tensor and number of source tensors.
    - Extract data pointers for the six source tensors from the destination tensor's source array.
    - Retrieve dimensions B, T, C, and H from the source tensors and the destination tensor.
    - Assert that the type of the sixth source tensor is `GGML_TYPE_F32` and that C is divisible by H, with C/H being either `WKV_BLOCK_SIZE` or `WKV_BLOCK_SIZE * 2`.
    - Obtain the SYCL stream from the context.
    - Calculate the shared memory size needed for the kernel execution based on C and H.
    - Define the block and grid dimensions for the SYCL kernel execution.
    - Submit the SYCL kernel for execution using the calculated dimensions and shared memory size, choosing between two kernel configurations based on the value of C/H.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place by executing the SYCL kernel to perform the RWKV6 operation.


---
### ggml\_sycl\_op\_rwkv\_wkv7<!-- {{#callable:ggml_sycl_op_rwkv_wkv7}} -->
The function `ggml_sycl_op_rwkv_wkv7` executes a SYCL kernel to perform a parallel computation on tensor data using the RWKV model with specific parameters.
- **Inputs**:
    - `ctx`: A reference to a `ggml_backend_sycl_context` object, which provides the SYCL stream for kernel execution.
    - `dst`: A pointer to a `ggml_tensor` object, which contains the destination tensor and its source tensors for the operation.
- **Control Flow**:
    - Initialize a debug print scope for the operation with the function name and destination tensor.
    - Extract data pointers for the source tensors from the destination tensor's source array.
    - Retrieve dimensions B, T, C, and H from the tensor's shape information.
    - Assert that the source tensor type is `GGML_TYPE_F32` and that C is divisible by H, and check if C/H matches the expected block size.
    - Obtain the SYCL stream from the context for kernel submission.
    - Calculate the shared memory size required for the kernel execution based on the block size and number of elements.
    - Define the block and grid dimensions for the SYCL kernel execution.
    - Submit the SYCL kernel for execution using the calculated dimensions and shared memory, choosing between two kernel variants based on the block size.
- **Output**: The function does not return a value; it modifies the `dst` tensor in place by executing the SYCL kernel.


