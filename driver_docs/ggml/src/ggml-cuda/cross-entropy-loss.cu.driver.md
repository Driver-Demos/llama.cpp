# Purpose
This source code file is designed to implement the forward and backward computation of the cross-entropy loss function using CUDA for parallel processing on NVIDIA GPUs. The file includes two main CUDA kernel functions: `cross_entropy_loss_f32` and `cross_entropy_loss_back_f32`. These functions are templated to optionally use shared memory for performance optimization. The forward function computes the cross-entropy loss by first calculating the softmax of the logits and then computing the loss based on the provided labels. The backward function computes the gradient of the loss with respect to the logits, which is essential for backpropagation in neural network training.

The file also defines two public functions, `ggml_cuda_cross_entropy_loss` and `ggml_cuda_cross_entropy_loss_back`, which serve as interfaces to launch the CUDA kernels. These functions handle the setup of CUDA streams, memory allocation, and kernel execution. They ensure that the input tensors are of the correct type and shape, and they manage the shared memory configuration based on the device's capabilities. The functions use assertions to validate the input data and ensure that the tensors are contiguous in memory, which is crucial for efficient GPU computation.

Overall, this file provides a specialized implementation of cross-entropy loss computation for use in GPU-accelerated machine learning frameworks. It leverages CUDA's parallel processing capabilities to efficiently compute the loss and its gradient, which are critical operations in training deep learning models. The use of templates and conditional shared memory usage allows for flexibility and optimization based on the specific hardware configuration.
# Imports and Dependencies

---
- `common.cuh`
- `cross-entropy-loss.cuh`
- `sum.cuh`
- `cmath`
- `cstdint`


# Functions

---
### cross\_entropy\_loss\_f32
The `cross_entropy_loss_f32` function computes the cross-entropy loss for a batch of logits and labels using CUDA for parallel processing.
- **Inputs**:
    - `logits`: A pointer to an array of float values representing the predicted logits for each class.
    - `labels`: A pointer to an array of float values representing the true labels for each class.
    - `dst`: A pointer to an array where the computed loss for each batch will be stored.
    - `nclasses`: An integer representing the number of classes.
    - `k`: An integer representing the number of samples in the batch.
- **Control Flow**:
    - The function begins by setting up shared memory if `use_shared` is true, and adjusts the pointers for `logits` and `labels` based on the block index.
    - It initializes `max_logit` to negative infinity and iterates over the classes to find the maximum logit value, storing values in shared memory if `use_shared` is true.
    - The maximum logit value is reduced across the warp using `warp_reduce_max`.
    - The function calculates the sum of exponentials of the adjusted logits (logits minus max_logit) and reduces this sum across the warp using `warp_reduce_sum`.
    - It computes the log of the sum of exponentials to use in the softmax calculation.
    - The function calculates the cross-entropy loss by iterating over the classes, adjusting logits, and accumulating the weighted sum of these values by the labels.
    - The accumulated loss is reduced across the warp, negated, and divided by `k` to get the average loss per sample.
    - If the thread index is zero, it writes the computed loss to the `dst` array at the position corresponding to the block index.
- **Output**: The function outputs the computed cross-entropy loss for each batch, stored in the `dst` array.


---
### cross\_entropy\_loss\_back\_f32
The `cross_entropy_loss_back_f32` function computes the gradient of the cross-entropy loss with respect to the logits for a batch of data using CUDA parallelization.
- **Inputs**:
    - `grad`: A pointer to the gradient of the loss with respect to the output, represented as a float.
    - `logits`: A pointer to the logits (predicted values before softmax) for each class, represented as a float array.
    - `labels`: A pointer to the true labels for each class, represented as a float array.
    - `dst`: A pointer to the destination array where the computed gradients with respect to the logits will be stored.
    - `nclasses`: An integer representing the number of classes.
- **Control Flow**:
    - Initialize shared memory if `use_shared` is true.
    - Adjust pointers for `logits`, `labels`, and `dst` based on the block index and number of classes.
    - Find the maximum logit value across the classes using parallel reduction to stabilize the softmax computation.
    - Compute the exponentiated logits adjusted by the maximum logit value and accumulate their sum.
    - Calculate the scale for the softmax by taking the reciprocal of the sum of exponentiated logits.
    - Compute the gradient of the loss with respect to each logit by adjusting the softmax-scaled logits with the labels and scaling by the gradient divided by the number of rows.
- **Output**: The function outputs the gradient of the cross-entropy loss with respect to the logits, stored in the `dst` array.


---
### ggml\_cuda\_cross\_entropy\_loss
The `ggml_cuda_cross_entropy_loss` function computes the cross-entropy loss for given logits and labels using CUDA for parallel processing.
- **Inputs**:
    - `ctx`: A reference to the `ggml_backend_cuda_context` which provides the CUDA context and resources.
    - `dst`: A pointer to a `ggml_tensor` where the computed cross-entropy loss will be stored.
- **Control Flow**:
    - Retrieve the source tensors `src0` (logits) and `src1` (labels) from the `dst` tensor's source array.
    - Assert that the data types of `src0`, `src1`, and `dst` are all `GGML_TYPE_F32` and that they are contiguous in memory.
    - Determine the number of classes (`ne00`) and the number of rows (`nrows`) from the dimensions of `src0`.
    - Obtain pointers to the data arrays of `src0`, `src1`, and `dst`.
    - Retrieve the CUDA stream and memory pool from the context `ctx`.
    - Calculate the dimensions for CUDA kernel execution (`blocks_dim` and `blocks_num`) and the size of shared memory required (`nbytes_shared`).
    - Check if the shared memory required is within the device's limit (`smpbo`).
    - If within limit, set the CUDA function attribute for maximum dynamic shared memory size and launch the `cross_entropy_loss_f32` kernel with shared memory enabled.
    - If not within limit, launch the `cross_entropy_loss_f32` kernel without shared memory.
    - Check for any CUDA errors after kernel execution.
    - Use the `sum_f32_cuda` function to combine results from individual blocks into the final output stored in `dst`.
- **Output**: The function outputs the computed cross-entropy loss into the `dst` tensor, with each element corresponding to the loss for a row of logits and labels.


---
### ggml\_cuda\_cross\_entropy\_loss\_back
The `ggml_cuda_cross_entropy_loss_back` function computes the gradient of the cross-entropy loss with respect to the logits using CUDA for parallel processing.
- **Inputs**:
    - `ctx`: A `ggml_backend_cuda_context` object that provides the CUDA context and resources for execution.
    - `dst`: A `ggml_tensor` object where the computed gradient of the cross-entropy loss will be stored.
- **Control Flow**:
    - Retrieve the gradient tensor `grad` and the logits and labels tensors `src0f` and `src1f` from the `dst` tensor's source attributes.
    - Assert that the data types of `src0f`, `src1f`, `grad`, and `dst` are all `GGML_TYPE_F32` and that they are contiguous in memory.
    - Assert that `grad` is a scalar and that `src0f`, `src1f`, and `dst` have the same shape.
    - Calculate the number of classes `ne00` and the number of rows `nrows` from `src0f`.
    - Obtain pointers to the data of `grad`, `src0f`, `src1f`, and `dst`.
    - Determine the CUDA stream from the context `ctx`.
    - Set up CUDA grid and block dimensions for kernel execution.
    - Check if the shared memory required is within the device's limit and adjust CUDA function attributes if necessary.
    - Launch the `cross_entropy_loss_back_f32` kernel with appropriate template parameters based on shared memory availability.
    - Handle CUDA errors after kernel execution.
- **Output**: The function does not return a value but writes the computed gradient of the cross-entropy loss into the `dst` tensor.


