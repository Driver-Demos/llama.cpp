# Purpose
This source code file is a CUDA-based implementation for performing operations related to Flash Attention, a technique used in neural networks to efficiently compute attention mechanisms. The file provides a collection of CUDA device functions and kernel launches that are designed to handle various quantization and dequantization operations, vector dot products, and the combination of results for attention computations. The code is structured to support different data types and quantization levels, such as Q4, Q5, Q8, and F16, which are used to optimize memory usage and computational efficiency on GPUs.

The file defines several key components, including function pointers for different kernel types (`fattn_kernel_t`, `vec_dot_KQ_f16_t`, `vec_dot_KQ_f32_t`), and templates for vector dot product operations (`vec_dot_fattn_vec_KQ_q4_0`, `vec_dot_fattn_vec_KQ_q4_1`, etc.). These templates are specialized for different quantization schemes and data types, allowing the code to handle a variety of input formats. The file also includes utility functions for quantizing and dequantizing data, which are crucial for converting between different numerical representations used in neural network computations.

The main purpose of this file is to provide a highly optimized and flexible framework for performing attention-related computations on NVIDIA GPUs. It leverages CUDA's parallel processing capabilities to efficiently handle large-scale data operations, making it suitable for use in deep learning models that require fast and memory-efficient attention mechanisms. The code is designed to be integrated into larger systems, where it can be called upon to perform specific tasks related to attention computation, such as calculating dot products between query and key vectors, applying softmax functions, and combining partial results to produce final attention scores.
# Imports and Dependencies

---
- `common.cuh`
- `convert.cuh`
- `vecdotq.cuh`
- `cstdint`


# Functions

---
### vec\_dot\_fattn\_vec\_KQ\_q4\_0
The function `vec_dot_fattn_vec_KQ_q4_0` computes the dot product between a quantized key vector and a query vector, applying specific transformations and scaling based on the quantization format q4_0.
- **Inputs**:
    - `K_c`: A pointer to the quantized key vector data, expected to be in q4_0 format.
    - `Q_v`: A pointer to the query vector data, though it is not used in this function.
    - `Q_q8`: A pointer to an array of integers representing the quantized query vector.
    - `Q_ds_v`: A pointer to the scaling factors for the query vector, used for dequantization.
- **Control Flow**:
    - Initialize a sum variable to accumulate the result.
    - Iterate over the key vector in chunks determined by the warp size.
    - For each chunk, calculate the index and shift values for accessing the quantized data.
    - Extract and shift the quantized values from the key vector.
    - Retrieve the corresponding quantized query value.
    - Compute the dot product of the quantized values using `ggml_cuda_dp4a`.
    - Check if FP16 is available and if the template type T is half, then perform operations using half precision.
    - If not using half precision, perform operations using float precision.
    - Accumulate the result into the sum variable.
- **Output**: Returns the computed dot product as a value of type T, which can be either half or float depending on the template parameter.


---
### vec\_dot\_fattn\_vec\_KQ\_q4\_1
The `vec_dot_fattn_vec_KQ_q4_1` function computes the dot product of quantized vectors K and Q using a specific quantization format (q4_1) and applies scaling and dequantization to produce a floating-point result.
- **Inputs**:
    - `K_c`: A pointer to the quantized vector K, represented as a character array.
    - `Q_v`: A pointer to the quantized vector Q, represented as a void pointer, but not used in this function.
    - `Q_q8`: A pointer to an integer array representing the quantized Q vector in an 8-bit format.
    - `Q_ds_v`: A pointer to a dequantization scale vector, represented as a void pointer, which is used to scale the dot product results.
- **Control Flow**:
    - Initialize a sum variable to accumulate the result.
    - Iterate over the quantized data in chunks determined by the warp size and data type size.
    - For each chunk, calculate indices and shifts to extract quantized values from K and Q.
    - Compute the dot product of the extracted values using the `ggml_cuda_dp4a` function.
    - Check if FP16 is available and if the template type T is half, then perform operations using half precision.
    - If FP16 is not available or T is not half, perform operations using float precision.
    - Accumulate the scaled and dequantized results into the sum variable.
- **Output**: The function returns a value of type T, which is the accumulated and scaled dot product result of the input vectors K and Q.


---
### vec\_dot\_fattn\_vec\_KQ\_q5\_0
The function `vec_dot_fattn_vec_KQ_q5_0` computes the dot product between a quantized key vector and a query vector, applying specific transformations and scaling based on the quantization format q5_0.
- **Inputs**:
    - `K_c`: A pointer to the quantized key vector data, expected to be in the q5_0 format.
    - `Q_v`: A pointer to the query vector data, which is not used in this function.
    - `Q_q8`: A pointer to an array of integers representing the quantized query vector.
    - `Q_ds_v`: A pointer to the scaling factors for the query vector, used to adjust the dot product result.
- **Control Flow**:
    - The function begins by casting the key vector data `K_c` to a `block_q5_0` type pointer `K_q5_0`.
    - A loop iterates over the key vector data in chunks determined by the dimension `D` and the `warp_size`.
    - Within the loop, the index `k_KQ` is calculated for each thread, and the corresponding block and shift values are determined.
    - The quantized value `v` is extracted from the key vector using bit manipulation, and additional bits are added from a higher precision part `qh` to form a complete quantized value.
    - The quantized query value `u` is retrieved from `Q_q8`.
    - The dot product of `v` and `u` is computed using the `ggml_cuda_dp4a` function, resulting in `sumi`.
    - If FP16 is available and the template type `T` is `half`, the function uses `half2` operations to compute the final sum, applying scaling factors from `Q_ds_v`.
    - If FP16 is not available or `T` is not `half`, the function uses `float2` operations to compute the final sum, applying scaling factors from `Q_ds_v`.
    - The loop unrolls to optimize performance, and the final sum is returned as the result of the function.
- **Output**: The function returns a value of type `T`, which is the computed dot product of the key and query vectors, adjusted by scaling factors.


---
### vec\_dot\_fattn\_vec\_KQ\_q5\_1
The `vec_dot_fattn_vec_KQ_q5_1` function computes the dot product of quantized vectors K and Q using a specific quantization format (q5_1) and returns the result as a type T.
- **Inputs**:
    - `K_c`: A pointer to the quantized vector K, represented as a character array.
    - `Q_v`: A pointer to the vector Q, represented as a void pointer, but not used in this function.
    - `Q_q8`: A pointer to an integer array representing the quantized vector Q.
    - `Q_ds_v`: A pointer to the scaling factors for Q, represented as a void pointer.
- **Control Flow**:
    - Initialize a sum variable of type T to accumulate the result.
    - Iterate over the elements of the vectors in chunks determined by the warp size.
    - For each chunk, calculate the index and shift values to access the appropriate elements of K and Q.
    - Extract and combine the quantized values from K using bit manipulation to form the integer v.
    - Retrieve the corresponding quantized value from Q, denoted as u.
    - Compute the dot product of v and u using the CUDA intrinsic function `ggml_cuda_dp4a`.
    - Check if FP16 is available and if T is half, then perform operations using half precision.
    - If T is not half, perform operations using float precision.
    - Accumulate the result into the sum variable.
    - Return the accumulated sum as the result of the function.
- **Output**: The function returns the dot product result as a value of type T, which can be either half or float depending on the template parameter.


---
### vec\_dot\_fattn\_vec\_KQ\_q8\_0
The `vec_dot_fattn_vec_KQ_q8_0` function computes the dot product between quantized vectors K and Q, specifically for the q8_0 quantization format, using CUDA for parallel processing.
- **Inputs**:
    - `K_c`: A pointer to the quantized K vector data, expected to be in the q8_0 format.
    - `Q_v`: A pointer to the Q vector data, which is not used in this function.
    - `Q_q8`: A pointer to the quantized Q vector data, used in the dot product computation.
    - `Q_ds_v`: A pointer to the dequantization scale data for Q, used to adjust the dot product result.
- **Control Flow**:
    - The function begins by casting the input K vector data to the `block_q8_0` type.
    - A loop iterates over the data in chunks determined by the warp size, processing each chunk in parallel using CUDA threads.
    - Within the loop, the function calculates indices and shifts to extract the relevant quantized values from K and Q.
    - The function retrieves the quantized value from K using the `get_int_b2` function and the current index.
    - The dequantization scale for Q is retrieved based on the template type T, which can be either `half` or `float`.
    - The function calls `vec_dot_q8_0_q8_1_impl` to compute the dot product for the current chunk, accumulating the result in `sum`.
    - The loop continues until all chunks are processed, and the final accumulated sum is returned.
- **Output**: The function returns the computed dot product as a value of type T, which can be either `half` or `float`, depending on the template parameter.


---
### vec\_dot\_fattn\_vec\_KQ\_f16
The `vec_dot_fattn_vec_KQ_f16` function computes the dot product of two vectors, K and Q, using half-precision floating-point arithmetic, optimized for CUDA execution.
- **Inputs**:
    - `K_c`: A pointer to the character array representing the K vector in half-precision floating-point format.
    - `Q_v`: A pointer to the character array representing the Q vector in half-precision floating-point format.
    - `Q_q8`: A pointer to an integer array, which is unused in this function.
    - `Q_ds_v`: A pointer to a void array, which is unused in this function.
- **Control Flow**:
    - Check if the type T is half (half-precision floating-point).
    - If T is half, cast Q_v to a half2 pointer and initialize a half2 sum variable.
    - Iterate over the elements of K and Q in steps of warp_size, computing the element-wise product and accumulating the result in sum2.
    - Return the sum of the low and high halves of sum2 if T is half.
    - If T is not half, cast Q_v to a float2 pointer and initialize a float sum variable.
    - Iterate over the elements of K and Q in steps of warp_size, computing the element-wise product and accumulating the result in sum.
    - Return the accumulated sum.
- **Output**: The function returns the dot product of the K and Q vectors as a value of type T, which can be either half or float.


---
### quantize\_q8\_1\_to\_shared
The `quantize_q8_1_to_shared` function quantizes a block of floating-point values into an 8-bit integer representation and stores the quantized values and associated scaling factors in shared memory.
- **Inputs**:
    - `x`: A pointer to an array of floating-point values to be quantized.
    - `scale`: A scaling factor applied to the input values before quantization.
    - `yq32`: A pointer to an array where the quantized 8-bit integer values will be stored.
    - `yds`: A pointer to an array where the scaling factors and sums will be stored, either as half2 or float2 depending on the template type Tds.
- **Control Flow**:
    - Initialize an array `vals` to store scaled input values.
    - Scale each input value by the provided `scale` and store in `vals`.
    - Compute the maximum absolute value `amax` and the sum of `vals`.
    - Use warp-level shuffling to find the maximum `amax` and sum across threads.
    - Calculate the quantization divisor `d` as `amax / 127`.
    - If `d` is not zero, quantize each value in `vals` by dividing by `d` and rounding, storing the result in `q8`.
    - Store the quantized 8-bit integer values in `yq32`.
    - If the thread index is a multiple of `QI8_1`, store the scaling factor `d` and sum in `yds` as either half2 or float2.
- **Output**: The function does not return a value but modifies the `yq32` and `yds` arrays to store quantized values and scaling factors, respectively.


---
### dequantize\_1\_q4\_0
The `dequantize_1_q4_0` function dequantizes a quantized value from a Q4_0 format to a floating-point value.
- **Inputs**:
    - `vx`: A pointer to the quantized data block of type `block_q4_0`.
    - `i`: An index of type `int64_t` indicating the position of the value to dequantize within the quantized data block.
- **Control Flow**:
    - Cast the input pointer `vx` to a pointer of type `block_q4_0`.
    - Calculate the block index `ib` by dividing `i` by `QK4_0`.
    - Calculate the index `iqs` within the block by taking the modulus of `i` with `QK4_0/2`.
    - Determine the shift amount by dividing the modulus of `i` with `QK4_0` by `QK4_0/2`.
    - Retrieve the dequantization factor `d` from the block at index `ib`.
    - Extract the quantized value `q0` from the block's `qs` array at index `iqs`.
    - Calculate the quantized value `q` by shifting `q0` right by `4*shift`, masking with `0x0F`, and subtracting 8.
    - If FP16 is available and the template type `T` is `half`, return the product of `d` and `q` as `half`.
    - Otherwise, return the product of `d` and `q` as `float`.
- **Output**: A dequantized value of type `T`, which can be either `half` or `float`, depending on the template parameter.


---
### dequantize\_1\_q4\_1
The `dequantize_1_q4_1` function dequantizes a quantized value from a Q4_1 format to a floating-point value.
- **Inputs**:
    - `vx`: A pointer to the quantized data block of type `block_q4_1`.
    - `i`: An index of type `int64_t` indicating the position of the value to dequantize within the quantized data block.
- **Control Flow**:
    - Cast the input pointer `vx` to a pointer of type `block_q4_1` and assign it to `x`.
    - Calculate the block index `ib` by dividing `i` by `QK4_1`.
    - Calculate the index `iqs` within the block by taking the modulus of `i` with `QK4_1/2`.
    - Determine the shift value by dividing the modulus of `i` with `QK4_1` by `QK4_1/2`.
    - Retrieve the `dm` value from the block at index `ib`.
    - Extract the quantized value `q0` from the block's `qs` array at index `iqs`.
    - Shift and mask `q0` to obtain the quantized integer `q`.
    - Check if FP16 is available and if the template type `T` is `half`, then perform dequantization using half precision.
    - If not using half precision, perform dequantization using float precision.
- **Output**: The function returns a dequantized value of type `T`, which can be either `half` or `float`, depending on the template parameter.


---
### dequantize\_1\_q5\_0
The `dequantize_1_q5_0` function dequantizes a specific element from a quantized block of type `block_q5_0` using a given index.
- **Inputs**:
    - `vx`: A pointer to the quantized data block of type `block_q5_0`.
    - `i`: An integer index specifying which element to dequantize within the block.
- **Control Flow**:
    - Cast the input pointer `vx` to a pointer of type `block_q5_0`.
    - Calculate the block index `ib` by dividing `i` by `QK5_0`.
    - Calculate the index `idq` and `iqs` within the block using modulo operations with `QK5_0` and `QK5_0/2`, respectively.
    - Determine the shift value based on the position within the block.
    - Extract the quantized low and high bits `ql0` and `qh0` from the block using the calculated indices and shift values.
    - Combine the low and high bits to form the full quantized value `q`.
    - Adjust the quantized value `q` by subtracting 16 to center it around zero.
    - If FP16 is available and the template type `T` is `half`, return the product of the dequantization factor `d` and the quantized value `q` as a `half` type.
    - Otherwise, return the product as a `float` type.
- **Output**: The function returns the dequantized value of type `T`, which can be either `half` or `float`, depending on the template parameter and availability of FP16 support.


---
### dequantize\_1\_q5\_1
The `dequantize_1_q5_1` function dequantizes a value from a quantized 5-bit format to a floating-point format using specific scaling and offset parameters.
- **Inputs**:
    - `vx`: A pointer to the quantized data block of type `block_q5_1`.
    - `i`: An index of type `int64_t` indicating the position of the value to be dequantized within the quantized data block.
- **Control Flow**:
    - The function casts the input pointer `vx` to a pointer of type `block_q5_1`.
    - It calculates the block index `ib`, the index within the quantized block `idq`, and the shift amount based on the input index `i`.
    - It retrieves the quantization scale and offset from the `dm` field of the block.
    - It extracts the low and high parts of the quantized value using bit manipulation on the `qs` and `qh` fields of the block.
    - It combines the low and high parts to form the full quantized value `q`.
    - The function checks if the target type `T` is `half` and performs dequantization using half-precision arithmetic if available.
    - If `half` is not available, it performs dequantization using single-precision floating-point arithmetic.
- **Output**: The function returns the dequantized value as a type `T`, which can be either `half` or `float` depending on the template parameter.


---
### dequantize\_1\_q8\_0
The `dequantize_1_q8_0` function dequantizes a quantized value from a block of type `block_q8_0` using a specified index.
- **Inputs**:
    - `vx`: A pointer to the block of type `block_q8_0` containing quantized values.
    - `i`: An integer index specifying which quantized value to dequantize within the block.
- **Control Flow**:
    - Cast the input pointer `vx` to a pointer of type `block_q8_0`.
    - Calculate the block index `ib` by dividing the index `i` by `QK8_0`.
    - Calculate the quantized value index `iqs` by taking the modulus of `i` with `QK8_0`.
    - Retrieve the dequantization factor `d` from the block at index `ib`.
    - Retrieve the quantized value `q` from the block at index `ib` and position `iqs`.
    - If the type `T` is `half`, return the product of `d` and `q` as a `half` type value.
    - Otherwise, return the product of `d` and `q` as a `float` type value.
- **Output**: The function returns the dequantized value as either a `half` or `float` type, depending on the template parameter `T`.


---
### dequantize\_1\_f16
The `dequantize_1_f16` function retrieves a half-precision floating-point value from a given array at a specified index.
- **Inputs**:
    - `vx`: A pointer to a memory location containing half-precision floating-point values.
    - `i`: An integer index specifying the position of the value to retrieve from the array.
- **Control Flow**:
    - The function casts the input pointer `vx` to a pointer of type `half`.
    - It accesses the `i`-th element of the array pointed to by the casted pointer.
- **Output**: Returns the half-precision floating-point value located at the specified index `i` in the array.


---
### flash\_attn\_stream\_k\_fixup
The `flash_attn_stream_k_fixup` function is a CUDA kernel that adjusts partial results of a matrix operation to ensure correct final results by iterating over previous blocks and combining them with current results.
- **Inputs**:
    - `dst`: A pointer to the destination array where the final results will be stored.
    - `dst_fixup`: A pointer to the array containing partial results that need to be fixed up.
    - `ne01`: An integer representing the first dimension size of the input matrix.
    - `ne02`: An integer representing the second dimension size of the input matrix.
    - `ne11`: An integer representing the first dimension size of the matrix used for iteration.
- **Control Flow**:
    - Calculate the number of columns as the product of ncols1 and ncols2.
    - Determine the block and thread indices for CUDA execution.
    - Calculate the starting and stopping indices for the current block's data processing.
    - Check conditions to determine if the current block has data to process or if it should return early.
    - Load the partial result that needs a fixup from the destination and fixup arrays.
    - Iterate over previous blocks to compute combined results by adjusting the current and new value accumulators based on maximum values.
    - Scale the current and new values using exponential functions to avoid NaNs and ensure numerical stability.
    - Write the final adjusted result back to the destination array.
- **Output**: The function does not return a value but writes the adjusted results back to the `dst` array.


---
### flash\_attn\_combine\_results
The `flash_attn_combine_results` function combines partial results from multiple parallel blocks to compute the final attention output in a CUDA kernel.
- **Inputs**:
    - `VKQ_parts`: A pointer to the array of partial results for the attention computation, with each element representing a part of the VKQ (Value-Key-Query) matrix.
    - `VKQ_meta`: A pointer to the array of metadata associated with each partial result, containing information like maximum values for scaling.
    - `dst`: A pointer to the destination array where the final combined results will be stored.
    - `parallel_blocks`: An integer representing the number of parallel blocks used in the computation, which determines how many partial results are combined.
- **Control Flow**:
    - The function begins by adjusting the pointers for `VKQ_parts`, `VKQ_meta`, and `dst` based on the block index and grid dimensions.
    - It initializes a shared memory array `meta` to store metadata for each parallel block.
    - The function calculates the maximum value `kqmax` from the metadata to use for scaling the results.
    - It iterates over each parallel block to compute the scaled sum of the VKQ parts and their corresponding metadata values.
    - The function applies a mask to the scaling factor to flush small values to zero, avoiding NaNs.
    - Finally, it writes the combined result to the `dst` array by dividing the accumulated numerator by the denominator.
- **Output**: The function outputs the final combined attention results in the `dst` array, with each element representing the normalized attention score for a specific query-key pair.


---
### on\_no\_fattn\_vec\_case
The `on_no_fattn_vec_case` function handles unsupported key-value (KV) type combinations for specific head sizes in a CUDA-based attention mechanism, and terminates the program with an error message.
- **Inputs**:
    - `D`: An integer representing the head size for which the KV type combination is being checked.
- **Control Flow**:
    - The function checks if the head size `D` is 64.
    - If `D` is 64, it prints an error message indicating unsupported KV type combinations for head size 64 and suggests compiling with a specific flag for V cache quantization support.
    - The function then calls `GGML_ABORT` to terminate the program with a fatal error.
    - If `D` is 128, it prints an error message indicating unsupported KV type combinations for head size 128 and lists the supported combinations.
    - It suggests compiling with a specific flag for all combinations of quantization types and then calls `GGML_ABORT` to terminate the program with a fatal error.
    - For any other value of `D`, it prints an error message indicating unsupported KV type combinations for the given head size and states that only f16 is supported.
    - The function then calls `GGML_ABORT` to terminate the program with a fatal error.
- **Output**: This function does not return any value as it is marked with `[[noreturn]]` and terminates the program using `GGML_ABORT`.


---
### launch\_fattn
The `launch_fattn` function configures and launches a CUDA kernel for performing flash attention operations on input tensors, handling various data types and configurations.
- **Inputs**:
    - `ctx`: A reference to a `ggml_backend_cuda_context` object, which provides the CUDA context and resources needed for execution.
    - `dst`: A pointer to a `ggml_tensor` object that represents the destination tensor where the result of the attention operation will be stored.
    - `fattn_kernel`: A function pointer to the CUDA kernel that performs the flash attention operation.
    - `nwarps`: An integer specifying the number of warps to be used in the CUDA kernel execution.
    - `nbytes_shared`: A size_t value indicating the number of bytes of shared memory to be used by the CUDA kernel.
    - `KQ_row_granularity`: An integer specifying the granularity of the KQ rows, which affects how the kernel processes the input data.
    - `need_f16_K`: A boolean indicating whether the K tensor needs to be converted to half-precision floating point (FP16).
    - `need_f16_V`: A boolean indicating whether the V tensor needs to be converted to half-precision floating point (FP16).
    - `stream_k`: A boolean indicating whether the kernel should use a streaming approach for processing the K tensor.
    - `warp_size`: An integer specifying the size of a warp, with a default value of `WARP_SIZE`.
- **Control Flow**:
    - The function begins by asserting various conditions on the input tensors and their properties, ensuring they meet the requirements for the operation.
    - It allocates memory for temporary storage of the K and V tensors in FP16 format if needed, using the CUDA context's memory pool.
    - The function calculates the number of tiles and blocks needed for the CUDA kernel execution based on the input tensor dimensions and the number of warps.
    - It configures the CUDA kernel launch parameters, including the number of blocks, threads per block, and shared memory size.
    - The function launches the specified `fattn_kernel` CUDA kernel with the configured parameters to perform the flash attention operation.
    - If the `stream_k` option is enabled, it may perform a fixup operation to adjust the results for fractional tiles.
    - If multiple parallel blocks are used, it combines the results from these blocks into the final output tensor.
    - The function checks for CUDA errors after kernel execution to ensure successful completion.
- **Output**: The function does not return a value; it writes the result of the flash attention operation to the `dst` tensor.


