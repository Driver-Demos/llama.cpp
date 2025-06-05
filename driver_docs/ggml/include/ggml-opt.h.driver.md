# Purpose
This C source code file provides a high-level interface for training machine learning models using the GGML (General Graph Machine Learning) library. It extends the basic functionality of GGML by offering a more user-friendly API for common tasks such as dataset management and model optimization. The file defines several key components, including structures for datasets, optimization contexts, and optimization results, as well as enumerations for loss types and build types. It also includes functions for initializing and managing datasets, configuring optimization parameters, and executing training and evaluation processes. The code is designed to be integrated into user applications, allowing developers to leverage its high-level functions for model training without delving into the lower-level details of GGML.

The file is structured to facilitate the creation and manipulation of datasets and optimization contexts, providing functions to initialize, shuffle, and batch datasets, as well as to configure and execute optimization processes. It defines a public API with functions prefixed by `GGML_API`, indicating their availability for external use. The code also includes callback mechanisms for customizing optimizer parameters and monitoring training progress. The high-level functions at the end of the file are particularly noteworthy, as they are designed to be easily adapted for user code, providing a clear pathway for integrating GGML-based model training into broader applications. Overall, this file serves as a comprehensive toolkit for model training within the GGML framework, offering both flexibility and ease of use.
# Imports and Dependencies

---
- `ggml.h`
- `ggml-backend.h`
- `stdint.h`


# Data Structures

---
### ggml\_opt\_loss\_type
- **Type**: `enum`
- **Members**:
    - `GGML_OPT_LOSS_TYPE_MEAN`: Represents a loss type where the mean of the loss values is calculated.
    - `GGML_OPT_LOSS_TYPE_SUM`: Represents a loss type where the sum of the loss values is calculated.
    - `GGML_OPT_LOSS_TYPE_CROSS_ENTROPY`: Represents a loss type using cross-entropy, typically used for classification tasks.
    - `GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR`: Represents a loss type using mean squared error, often used for regression tasks.
- **Description**: The `ggml_opt_loss_type` enumeration defines various built-in loss types that can be minimized by the optimizer in the GGML framework. These loss types include mean, sum, cross-entropy, and mean squared error, each serving different purposes in model training, such as classification or regression. This enumeration allows users to select the appropriate loss function for their specific machine learning problem.


---
### ggml\_opt\_build\_type
- **Type**: `enum`
- **Members**:
    - `GGML_OPT_BUILD_TYPE_FORWARD`: Represents the forward pass build type with a value of 10.
    - `GGML_OPT_BUILD_TYPE_GRAD`: Represents the gradient computation build type with a value of 20.
    - `GGML_OPT_BUILD_TYPE_OPT`: Represents the optimization build type with a value of 30.
- **Description**: The `ggml_opt_build_type` enumeration defines different types of build processes used in the GGML optimization context. Each enumerator represents a specific phase in the optimization process: forward pass, gradient computation, and optimization. These build types are used to control the construction of compute graphs for different stages of model training and evaluation.


---
### ggml\_opt\_optimizer\_params
- **Type**: `struct`
- **Members**:
    - `adamw`: A nested structure containing parameters specific to the AdamW optimizer, including learning rate, beta coefficients, epsilon for numerical stability, and weight decay.
- **Description**: The `ggml_opt_optimizer_params` structure is designed to encapsulate parameters for the AdamW optimizer, which is a variant of the Adam optimizer that includes weight decay. This structure is part of a larger framework for training models using GGML, providing a high-level interface for optimization tasks. The parameters within the `adamw` sub-structure are crucial for controlling the behavior of the optimizer, affecting how it updates model weights during training.


---
### ggml\_opt\_params
- **Type**: `struct`
- **Members**:
    - `backend_sched`: Defines which backends are used to construct the compute graphs.
    - `ctx_compute`: Pointer to a ggml_context used for static allocation of compute graphs.
    - `inputs`: Pointer to a ggml_tensor representing the input data for the optimization.
    - `outputs`: Pointer to a ggml_tensor representing the output data for the optimization.
    - `loss_type`: Specifies the type of loss function to be used during optimization.
    - `build_type`: Specifies the type of build process for the optimization context.
    - `opt_period`: Determines the number of gradient accumulation steps before an optimizer step is performed.
    - `get_opt_pars`: Callback function for calculating optimizer parameters.
    - `get_opt_pars_ud`: User data for the optimizer parameters calculation callback.
- **Description**: The `ggml_opt_params` structure is designed to encapsulate parameters necessary for setting up and executing optimization processes in the GGML framework. It includes configuration for backend scheduling, static graph allocation, input and output tensor management, and loss and build type specifications. Additionally, it provides mechanisms for defining the frequency of optimizer steps and includes a callback system for dynamic optimizer parameter calculation, allowing for flexible and efficient optimization workflows.


# Function Declarations (Public API)

---
### ggml\_opt\_dataset\_free<!-- {{#callable_declaration:ggml_opt_dataset_free}} -->
Frees the resources associated with a dataset.
- **Description**: Use this function to release all resources allocated for a dataset when it is no longer needed. This is essential to prevent memory leaks in applications that manage datasets dynamically. Ensure that the dataset is not used after calling this function, as it will be invalidated.
- **Inputs**:
    - `dataset`: A handle to the dataset to be freed. Must be a valid dataset handle obtained from ggml_opt_dataset_init. Passing an invalid or already freed handle results in undefined behavior.
- **Output**: None
- **See also**: [`ggml_opt_dataset_free`](../src/ggml-opt.cpp.driver.md#ggml_opt_dataset_free)  (Implementation)


---
### ggml\_opt\_dataset\_ndata<!-- {{#callable_declaration:ggml_opt_dataset_ndata}} -->
Retrieve the total number of datapoints in a dataset.
- **Description**: Use this function to obtain the total number of datapoints contained within a given dataset. This is useful for understanding the size of the dataset and for iterating over the data. The function should be called with a valid dataset object that has been properly initialized. It is important to ensure that the dataset is not null before calling this function to avoid undefined behavior.
- **Inputs**:
    - `dataset`: A handle to a ggml_opt_dataset_t object representing the dataset. Must be a valid, non-null dataset that has been initialized using ggml_opt_dataset_init.
- **Output**: Returns the total number of datapoints in the dataset as an int64_t value.
- **See also**: [`ggml_opt_dataset_ndata`](../src/ggml-opt.cpp.driver.md#ggml_opt_dataset_ndata)  (Implementation)


---
### ggml\_opt\_dataset\_data<!-- {{#callable_declaration:ggml_opt_dataset_data}} -->
Retrieve the data tensor from a dataset.
- **Description**: This function is used to access the underlying data tensor of a given dataset. It is typically called when you need to directly manipulate or inspect the data stored within a dataset object. The function assumes that the dataset has been properly initialized and is valid. It is important to ensure that the dataset is not null before calling this function to avoid undefined behavior.
- **Inputs**:
    - `dataset`: A handle to a dataset object from which the data tensor is to be retrieved. Must not be null, and should be a valid, initialized dataset.
- **Output**: Returns a pointer to a ggml_tensor structure representing the data tensor of the dataset. The shape of this tensor is [ne_datapoint, ndata].
- **See also**: [`ggml_opt_dataset_data`](../src/ggml-opt.cpp.driver.md#ggml_opt_dataset_data)  (Implementation)


---
### ggml\_opt\_dataset\_labels<!-- {{#callable_declaration:ggml_opt_dataset_labels}} -->
Retrieve the labels tensor from a dataset.
- **Description**: Use this function to access the labels tensor associated with a given dataset. This is typically used when you need to work with or analyze the labels of the dataset, such as during model training or evaluation. Ensure that the dataset has been properly initialized before calling this function. The function returns a pointer to the tensor storing the labels, which allows for direct manipulation or inspection of the label data.
- **Inputs**:
    - `dataset`: A handle to a dataset from which the labels tensor is to be retrieved. It must be a valid, initialized dataset object. If the dataset is not properly initialized, the behavior is undefined.
- **Output**: A pointer to the ggml_tensor structure that contains the labels of the dataset.
- **See also**: [`ggml_opt_dataset_labels`](../src/ggml-opt.cpp.driver.md#ggml_opt_dataset_labels)  (Implementation)


---
### ggml\_opt\_dataset\_shuffle<!-- {{#callable_declaration:ggml_opt_dataset_shuffle}} -->
Shuffles a dataset using a specified random number generator.
- **Description**: Use this function to shuffle the order of datapoints in a dataset, either entirely or up to a specified index, using the random number generator from the provided optimization context. This is useful for preparing data for training models to ensure randomness and prevent overfitting. The function must be called with a valid dataset and optimization context. If the `idata` parameter is negative, the entire dataset is shuffled. Otherwise, the function shuffles datapoints up to the specified index, which must be a multiple of the dataset's shard size.
- **Inputs**:
    - `opt_ctx`: A valid optimization context that provides the random number generator used for shuffling. Must not be null.
    - `dataset`: A dataset to be shuffled. Must not be null and should be properly initialized.
    - `idata`: An integer specifying the number of datapoints to shuffle. If negative, the entire dataset is shuffled. If non-negative, it must be less than or equal to the total number of datapoints and a multiple of the dataset's shard size.
- **Output**: None
- **See also**: [`ggml_opt_dataset_shuffle`](../src/ggml-opt.cpp.driver.md#ggml_opt_dataset_shuffle)  (Implementation)


---
### ggml\_opt\_dataset\_get\_batch<!-- {{#callable_declaration:ggml_opt_dataset_get_batch}} -->
Retrieve a batch of data and labels from the dataset.
- **Description**: This function is used to extract a specific batch of data and labels from a given dataset, which is useful for training or evaluating machine learning models. The function requires that the data and labels tensors are contiguous and match the types of the dataset's internal tensors. The batch index must be within the valid range, and the function assumes that the dataset has been properly initialized and possibly shuffled beforehand. It is important to ensure that the data and labels tensors are correctly sized to accommodate the batch being retrieved.
- **Inputs**:
    - `dataset`: A handle to the dataset from which the batch is to be retrieved. It must be a valid, initialized dataset object.
    - `data_batch`: A pointer to a ggml_tensor where the data for the batch will be stored. It must be contiguous and match the data type of the dataset's data tensor. Must not be null.
    - `labels_batch`: A pointer to a ggml_tensor where the labels for the batch will be stored. It must be contiguous and match the data type of the dataset's labels tensor. Can be null if the dataset does not include labels.
    - `ibatch`: The index of the batch to retrieve. It must be within the range of available batches in the dataset.
- **Output**: None
- **See also**: [`ggml_opt_dataset_get_batch`](../src/ggml-opt.cpp.driver.md#ggml_opt_dataset_get_batch)  (Implementation)


---
### ggml\_opt\_dataset\_get\_batch\_host<!-- {{#callable_declaration:ggml_opt_dataset_get_batch_host}} -->
Retrieve a batch of data and labels from a dataset into host memory.
- **Description**: This function is used to extract a specific batch of data and optionally labels from a given dataset and copy them into provided host memory buffers. It is typically called during the training or evaluation of machine learning models to access data in manageable chunks. The function requires that the number of data elements in the batch is a multiple of the dataset's shard size. The labels buffer can be null if the dataset does not include labels. The function ensures that the requested batch index is within the valid range of the dataset's permutation.
- **Inputs**:
    - `dataset`: A handle to the dataset from which data and labels are to be retrieved. It must be a valid, initialized dataset object.
    - `data_batch`: A pointer to a memory buffer where the data for the batch will be copied. The buffer must be large enough to hold 'nb_data_batch' elements.
    - `nb_data_batch`: The number of data elements to be included in the batch. It must be a multiple of the dataset's shard size.
    - `labels_batch`: A pointer to a memory buffer where the labels for the batch will be copied, or null if labels are not needed. If not null, the buffer must be large enough to hold the corresponding number of label elements.
    - `ibatch`: The index of the batch to retrieve. It must be within the valid range of batch indices for the dataset.
- **Output**: None
- **See also**: [`ggml_opt_dataset_get_batch_host`](../src/ggml-opt.cpp.driver.md#ggml_opt_dataset_get_batch_host)  (Implementation)


---
### ggml\_opt\_get\_default\_optimizer\_params<!-- {{#callable_declaration:ggml_opt_get_default_optimizer_params}} -->
Returns the default optimizer parameters for the AdamW optimizer.
- **Description**: Use this function to obtain a set of default parameters for the AdamW optimizer, which are hard-coded and constant. This is useful when you want to initialize an optimizer with standard settings without needing to specify each parameter manually. The function does not utilize the `userdata` parameter, so it can be safely passed as `NULL` or any other value.
- **Inputs**:
    - `userdata`: Arbitrary user data passed to the function, which is not used in this implementation. Can be `NULL` or any other value.
- **Output**: Returns a `ggml_opt_optimizer_params` structure containing default values for the AdamW optimizer parameters, including learning rate, beta values, epsilon, and weight decay.
- **See also**: [`ggml_opt_get_default_optimizer_params`](../src/ggml-opt.cpp.driver.md#ggml_opt_get_default_optimizer_params)  (Implementation)


---
### ggml\_opt\_get\_constant\_optimizer\_params<!-- {{#callable_declaration:ggml_opt_get_constant_optimizer_params}} -->
Returns optimizer parameters from user-provided data.
- **Description**: Use this function to retrieve optimizer parameters that are stored in a user-provided data structure. It is useful when you have a constant set of optimizer parameters that you want to apply without modification. The function expects a pointer to a data structure containing the optimizer parameters and will return these parameters directly. Ensure that the `userdata` pointer is valid and points to a properly initialized `ggml_opt_optimizer_params` structure.
- **Inputs**:
    - `userdata`: A pointer to a `ggml_opt_optimizer_params` structure. The pointer must not be null and should point to a valid and properly initialized structure. The function will directly cast and return the data pointed to by this parameter.
- **Output**: Returns a `ggml_opt_optimizer_params` structure containing the optimizer parameters extracted from the provided `userdata`.
- **See also**: [`ggml_opt_get_constant_optimizer_params`](../src/ggml-opt.cpp.driver.md#ggml_opt_get_constant_optimizer_params)  (Implementation)


---
### ggml\_opt\_default\_params<!-- {{#callable_declaration:ggml_opt_default_params}} -->
Initialize optimization parameters with default values.
- **Description**: Use this function to obtain a set of default optimization parameters for initializing a new optimization context. It requires specifying the backend scheduler and the type of loss function to be used. This function is typically called when setting up an optimization process, ensuring that all parameters are initialized with sensible defaults, except for those explicitly provided as arguments. It is important to call this function before starting any optimization to ensure that the context is properly configured.
- **Inputs**:
    - `backend_sched`: Specifies the backend scheduler to be used for constructing compute graphs. The caller must provide a valid `ggml_backend_sched_t` value.
    - `loss_type`: Defines the type of loss function to be minimized by the optimizer. Must be one of the values from the `ggml_opt_loss_type` enumeration, such as `GGML_OPT_LOSS_TYPE_MEAN` or `GGML_OPT_LOSS_TYPE_SUM`.
- **Output**: Returns a `ggml_opt_params` structure with default values set for most fields, except for those specified by the input parameters.
- **See also**: [`ggml_opt_default_params`](../src/ggml-opt.cpp.driver.md#ggml_opt_default_params)  (Implementation)


---
### ggml\_opt\_free<!-- {{#callable_declaration:ggml_opt_free}} -->
Frees resources associated with an optimization context.
- **Description**: Use this function to release all resources and memory associated with a given optimization context when it is no longer needed. This function should be called to prevent memory leaks after the optimization process is complete. It is safe to call this function with a null context, in which case it will have no effect.
- **Inputs**:
    - `opt_ctx`: A handle to the optimization context to be freed. Must be a valid context created by ggml_opt_init, or null. The caller retains ownership and responsibility for ensuring it is valid before calling.
- **Output**: None
- **See also**: [`ggml_opt_free`](../src/ggml-opt.cpp.driver.md#ggml_opt_free)  (Implementation)


---
### ggml\_opt\_reset<!-- {{#callable_declaration:ggml_opt_reset}} -->
Resets the optimization context and optionally the optimizer state.
- **Description**: Use this function to reset the optimization context, which includes setting gradients to zero and initializing the loss. If the `optimizer` parameter is true, the optimizer state is also reset, and the iteration counter is set to 1. This function is typically called at the beginning of a new training session or when reinitializing the optimization process. Ensure that the `opt_ctx` is a valid and properly initialized optimization context before calling this function.
- **Inputs**:
    - `opt_ctx`: A valid optimization context of type `ggml_opt_context_t`. The caller must ensure this context is properly initialized and not null.
    - `optimizer`: A boolean value indicating whether to reset the optimizer state. If true, the optimizer state is reset and the iteration counter is set to 1; if false, only the gradients are reset.
- **Output**: None
- **See also**: [`ggml_opt_reset`](../src/ggml-opt.cpp.driver.md#ggml_opt_reset)  (Implementation)


---
### ggml\_opt\_static\_graphs<!-- {{#callable_declaration:ggml_opt_static_graphs}} -->
Checks if the optimization context uses static graphs.
- **Description**: This function determines whether the optimization context is configured to use static graphs, which can affect how the computation graphs are allocated and managed. It is useful for understanding the memory allocation strategy of the optimization process. This function should be called when you need to verify the graph allocation mode of a given optimization context. Ensure that the `opt_ctx` is properly initialized before calling this function.
- **Inputs**:
    - `opt_ctx`: A valid `ggml_opt_context_t` representing the optimization context. It must be properly initialized and not null. The function will return false if the context is invalid or not configured to use static graphs.
- **Output**: Returns a boolean value: `true` if the optimization context uses static graphs, `false` otherwise.
- **See also**: [`ggml_opt_static_graphs`](../src/ggml-opt.cpp.driver.md#ggml_opt_static_graphs)  (Implementation)


---
### ggml\_opt\_inputs<!-- {{#callable_declaration:ggml_opt_inputs}} -->
Retrieve the input tensor from the optimization context.
- **Description**: Use this function to access the input tensor associated with a given optimization context. This is typically used when you need to inspect or manipulate the input data for a forward graph in a machine learning model. Ensure that the optimization context is properly initialized before calling this function. Note that if static graphs are not used, the returned tensor pointer may become invalid after subsequent calls to functions that modify the graph.
- **Inputs**:
    - `opt_ctx`: A valid optimization context from which the input tensor is to be retrieved. Must not be null, and should be properly initialized before use.
- **Output**: Returns a pointer to the input tensor associated with the provided optimization context.
- **See also**: [`ggml_opt_inputs`](../src/ggml-opt.cpp.driver.md#ggml_opt_inputs)  (Implementation)


---
### ggml\_opt\_outputs<!-- {{#callable_declaration:ggml_opt_outputs}} -->
Retrieve the output tensor from an optimization context.
- **Description**: Use this function to access the output tensor associated with a given optimization context. This is typically used in the context of model training or evaluation, where the output tensor represents the results of a forward pass through the model. Ensure that the optimization context is properly initialized before calling this function. The returned tensor is part of the optimization context and should not be freed or modified directly by the caller.
- **Inputs**:
    - `opt_ctx`: A valid optimization context from which to retrieve the output tensor. Must not be null. The caller retains ownership and responsibility for managing the lifecycle of this context.
- **Output**: Returns a pointer to the ggml_tensor structure representing the output tensor of the optimization context.
- **See also**: [`ggml_opt_outputs`](../src/ggml-opt.cpp.driver.md#ggml_opt_outputs)  (Implementation)


---
### ggml\_opt\_labels<!-- {{#callable_declaration:ggml_opt_labels}} -->
Retrieve the labels tensor from the optimization context.
- **Description**: Use this function to access the labels tensor associated with a given optimization context. This is typically used in scenarios where you need to compare model outputs against known labels during training or evaluation. Ensure that the optimization context is properly initialized and contains valid label data before calling this function.
- **Inputs**:
    - `opt_ctx`: A valid optimization context from which the labels tensor is to be retrieved. Must not be null and should be properly initialized.
- **Output**: Returns a pointer to the ggml_tensor structure representing the labels within the given optimization context.
- **See also**: [`ggml_opt_labels`](../src/ggml-opt.cpp.driver.md#ggml_opt_labels)  (Implementation)


---
### ggml\_opt\_loss<!-- {{#callable_declaration:ggml_opt_loss}} -->
Retrieve the loss tensor from the optimization context.
- **Description**: Use this function to access the scalar tensor that represents the loss value within a given optimization context. This function is typically called after an optimization step or evaluation to obtain the current loss value, which is crucial for monitoring the training process. Ensure that the optimization context is properly initialized and valid before calling this function to avoid undefined behavior.
- **Inputs**:
    - `opt_ctx`: A valid optimization context from which the loss tensor is to be retrieved. The context must be properly initialized and not null.
- **Output**: Returns a pointer to a ggml_tensor structure representing the loss value in the optimization context.
- **See also**: [`ggml_opt_loss`](../src/ggml-opt.cpp.driver.md#ggml_opt_loss)  (Implementation)


---
### ggml\_opt\_pred<!-- {{#callable_declaration:ggml_opt_pred}} -->
Retrieve the prediction tensor from the optimization context.
- **Description**: Use this function to access the tensor that contains the predictions made by the model outputs within a given optimization context. This function is typically called after the model has been evaluated to obtain the predictions for further analysis or comparison with actual labels. Ensure that the optimization context is properly initialized and used in a valid state before calling this function.
- **Inputs**:
    - `opt_ctx`: A valid optimization context from which the prediction tensor is to be retrieved. Must not be null and should be properly initialized before use.
- **Output**: Returns a pointer to the ggml_tensor structure representing the predictions made by the model outputs.
- **See also**: [`ggml_opt_pred`](../src/ggml-opt.cpp.driver.md#ggml_opt_pred)  (Implementation)


---
### ggml\_opt\_ncorrect<!-- {{#callable_declaration:ggml_opt_ncorrect}} -->
Retrieve the tensor representing the number of correct predictions.
- **Description**: Use this function to access the tensor that stores the count of correct predictions made by the model during optimization. This function is useful for evaluating the model's performance by comparing the predicted outputs against the actual labels. It is expected that the optimization context has been properly initialized and is in a valid state before calling this function.
- **Inputs**:
    - `opt_ctx`: A valid optimization context from which the number of correct predictions tensor is retrieved. Must not be null and should be properly initialized.
- **Output**: Returns a pointer to a ggml_tensor structure representing the number of correct predictions.
- **See also**: [`ggml_opt_ncorrect`](../src/ggml-opt.cpp.driver.md#ggml_opt_ncorrect)  (Implementation)


---
### ggml\_opt\_grad\_acc<!-- {{#callable_declaration:ggml_opt_grad_acc}} -->
Retrieve the gradient accumulator tensor for a specified node in the optimization context.
- **Description**: This function is used to obtain the gradient accumulator tensor associated with a specific node within a given optimization context. It is typically called during the training process to access the accumulated gradients for a node, which are used in optimization algorithms to update model parameters. The function requires a valid optimization context and a node tensor for which the gradient accumulator is needed. It is important to ensure that the optimization context and node are properly initialized and valid before calling this function.
- **Inputs**:
    - `opt_ctx`: A valid optimization context of type `ggml_opt_context_t`. It must be properly initialized and not null.
    - `node`: A pointer to a `ggml_tensor` representing the node for which the gradient accumulator is requested. It must be a valid tensor and not null.
- **Output**: Returns a pointer to a `ggml_tensor` that represents the gradient accumulator for the specified node.
- **See also**: [`ggml_opt_grad_acc`](../src/ggml-opt.cpp.driver.md#ggml_opt_grad_acc)  (Implementation)


---
### ggml\_opt\_result\_free<!-- {{#callable_declaration:ggml_opt_result_free}} -->
Frees resources associated with an optimization result.
- **Description**: Use this function to release any resources or memory associated with a `ggml_opt_result_t` object when it is no longer needed. This is essential to prevent memory leaks in applications that utilize optimization results. Ensure that the `result` is a valid, initialized `ggml_opt_result_t` object before calling this function. After calling, the `result` should not be used unless it is re-initialized.
- **Inputs**:
    - `result`: A `ggml_opt_result_t` object representing the optimization result to be freed. It must be a valid, initialized object. Passing an invalid or uninitialized object may lead to undefined behavior.
- **Output**: None
- **See also**: [`ggml_opt_result_free`](../src/ggml-opt.cpp.driver.md#ggml_opt_result_free)  (Implementation)


---
### ggml\_opt\_result\_reset<!-- {{#callable_declaration:ggml_opt_result_reset}} -->
Resets the optimization result structure to its initial state.
- **Description**: Use this function to clear and reset the state of an optimization result structure before reusing it for a new optimization process. This function sets the number of data points to zero, clears any stored loss and prediction data, and resets the count of correct predictions. It is essential to call this function before starting a new optimization if the same result structure is to be reused, ensuring that no residual data from previous optimizations affects the new results.
- **Inputs**:
    - `result`: A pointer to a ggml_opt_result_t structure that must not be null. The caller retains ownership of this structure, and it is expected to be a valid, initialized optimization result object. Invalid or null pointers will lead to undefined behavior.
- **Output**: None
- **See also**: [`ggml_opt_result_reset`](../src/ggml-opt.cpp.driver.md#ggml_opt_result_reset)  (Implementation)


---
### ggml\_opt\_result\_ndata<!-- {{#callable_declaration:ggml_opt_result_ndata}} -->
Retrieve the number of datapoints from an optimization result.
- **Description**: Use this function to obtain the total number of datapoints associated with a given optimization result. This is useful when you need to know the size of the dataset that was processed during optimization. Ensure that the `result` parameter is a valid, initialized optimization result object before calling this function. The function writes the number of datapoints to the location pointed to by `ndata`.
- **Inputs**:
    - `result`: A valid `ggml_opt_result_t` object representing the optimization result. It must be properly initialized before use.
    - `ndata`: A pointer to an `int64_t` where the function will store the number of datapoints. Must not be null.
- **Output**: None
- **See also**: [`ggml_opt_result_ndata`](../src/ggml-opt.cpp.driver.md#ggml_opt_result_ndata)  (Implementation)


---
### ggml\_opt\_result\_loss<!-- {{#callable_declaration:ggml_opt_result_loss}} -->
Calculates the loss and uncertainty from an optimization result.
- **Description**: Use this function to compute the average loss and its uncertainty from a given optimization result. It is typically called after an optimization process to evaluate the performance of the model. The function requires a valid optimization result and pointers to store the calculated loss and uncertainty. If the number of batches in the result is zero, the loss is set to 0.0 and the uncertainty to NaN. If the uncertainty pointer is null, the uncertainty calculation is skipped. The function handles cases with fewer than two batches by setting the uncertainty to NaN.
- **Inputs**:
    - `result`: A pointer to a ggml_opt_result_t structure containing the optimization results. Must not be null.
    - `loss`: A pointer to a double where the calculated loss will be stored. Must not be null.
    - `unc`: A pointer to a double where the calculated uncertainty will be stored. Can be null if uncertainty is not needed.
- **Output**: The function writes the calculated loss to the location pointed to by 'loss' and the uncertainty to 'unc' if it is not null. Returns None.
- **See also**: [`ggml_opt_result_loss`](../src/ggml-opt.cpp.driver.md#ggml_opt_result_loss)  (Implementation)


---
### ggml\_opt\_result\_pred<!-- {{#callable_declaration:ggml_opt_result_pred}} -->
Copies prediction data from an optimization result to a provided array.
- **Description**: Use this function to extract prediction data from a given optimization result and store it in a provided integer array. This function is typically called after an optimization process to retrieve the predictions made by the model. Ensure that the provided array is large enough to hold all prediction values from the result. The function assumes that the result contains valid prediction data and that the provided array is not null.
- **Inputs**:
    - `result`: A handle to an optimization result object from which prediction data will be extracted. It must be a valid, non-null pointer to a ggml_opt_result_t.
    - `pred`: A pointer to an integer array where the prediction data will be stored. The array must be pre-allocated and large enough to hold all predictions from the result. It must not be null.
- **Output**: None
- **See also**: [`ggml_opt_result_pred`](../src/ggml-opt.cpp.driver.md#ggml_opt_result_pred)  (Implementation)


---
### ggml\_opt\_result\_accuracy<!-- {{#callable_declaration:ggml_opt_result_accuracy}} -->
Calculates the accuracy and its uncertainty from an optimization result.
- **Description**: Use this function to compute the accuracy of predictions and optionally the uncertainty of this accuracy from a given optimization result. The function requires a valid optimization result object and a pointer to store the calculated accuracy. If uncertainty is also needed, provide a non-null pointer for it; otherwise, pass NULL to ignore uncertainty calculation. The function handles cases where the number of correct predictions is negative by setting the accuracy to NaN, and it requires at least two data points to compute a valid uncertainty.
- **Inputs**:
    - `result`: A pointer to a ggml_opt_result_t object containing the optimization result data. It must be a valid, non-null pointer.
    - `accuracy`: A pointer to a double where the calculated accuracy will be stored. Must be a valid, non-null pointer.
    - `unc`: A pointer to a double where the uncertainty of the accuracy will be stored. Can be NULL if uncertainty is not needed.
- **Output**: The function writes the calculated accuracy to the location pointed to by accuracy, and if unc is not NULL, writes the uncertainty to the location pointed to by unc.
- **See also**: [`ggml_opt_result_accuracy`](../src/ggml-opt.cpp.driver.md#ggml_opt_result_accuracy)  (Implementation)


---
### ggml\_opt\_prepare\_alloc<!-- {{#callable_declaration:ggml_opt_prepare_alloc}} -->
Prepare the optimization context for graph allocation.
- **Description**: This function sets up the optimization context by associating it with a compute context, a computation graph, and input and output tensors. It must be called before allocating graphs with `ggml_opt_alloc` when not using static graphs. This preparation step is crucial for ensuring that the optimization process has the necessary resources and configurations to proceed correctly.
- **Inputs**:
    - `opt_ctx`: A handle to the optimization context that will be prepared. It must not have static graphs already allocated.
    - `ctx_compute`: A pointer to a `ggml_context` structure that represents the compute context. This must be a valid, non-null pointer.
    - `gf`: A pointer to a `ggml_cgraph` structure representing the computation graph. This must be a valid, non-null pointer.
    - `inputs`: A pointer to a `ggml_tensor` structure representing the input tensor. This must be a valid, non-null pointer.
    - `outputs`: A pointer to a `ggml_tensor` structure representing the output tensor. This must be a valid, non-null pointer.
- **Output**: None
- **See also**: [`ggml_opt_prepare_alloc`](../src/ggml-opt.cpp.driver.md#ggml_opt_prepare_alloc)  (Implementation)


---
### ggml\_opt\_alloc<!-- {{#callable_declaration:ggml_opt_alloc}} -->
Allocate the next graph for evaluation in the optimization context.
- **Description**: This function prepares the optimization context for the next evaluation by allocating the appropriate graph, either for a forward pass or for both forward and backward passes, depending on the `backward` parameter. It must be called exactly once before invoking `ggml_opt_eval`. The function ensures that the context is ready for evaluation by setting up the necessary graph and resetting any previous allocations if needed. It is crucial to call this function after preparing the allocation with `ggml_opt_prepare_alloc` if static graphs are not used.
- **Inputs**:
    - `opt_ctx`: A handle to the optimization context, which must be properly initialized and not null. The context should not be in an evaluation-ready state when this function is called.
    - `backward`: A boolean flag indicating whether to prepare the graph for both forward and backward passes (true) or only for the forward pass (false).
- **Output**: None
- **See also**: [`ggml_opt_alloc`](../src/ggml-opt.cpp.driver.md#ggml_opt_alloc)  (Implementation)


---
### ggml\_opt\_eval<!-- {{#callable_declaration:ggml_opt_eval}} -->
Evaluates the optimization context and updates the result.
- **Description**: This function is used to perform an evaluation step on the given optimization context, updating the optimization result if provided. It should be called after the optimization context has been properly initialized and prepared for evaluation. The function handles the computation of the optimization graph and updates the iteration count and optimization period index. If the result parameter is provided, it updates the result with the current loss, predictions, and number of correct predictions. The function also manages the state of the optimization context, ensuring it is ready for subsequent operations.
- **Inputs**:
    - `opt_ctx`: A valid optimization context that has been initialized and prepared for evaluation. It must not be null and should be in a state ready for evaluation.
    - `result`: An optional optimization result structure to be updated with the evaluation results. If null, the function will perform the evaluation without updating any result data.
- **Output**: None
- **See also**: [`ggml_opt_eval`](../src/ggml-opt.cpp.driver.md#ggml_opt_eval)  (Implementation)


---
### ggml\_opt\_epoch<!-- {{#callable_declaration:ggml_opt_epoch}} -->
Performs a training and evaluation epoch on a dataset using a specified optimization context.
- **Description**: This function is used to conduct a single epoch of training and evaluation on a given dataset using the provided optimization context. It splits the dataset into training and evaluation sections based on the specified data index split point. The function requires that the optimization context is set up with static graphs. During the epoch, it processes batches of data, performing training on the initial segment and evaluation on the remaining segment. Optional callbacks can be provided to execute custom logic after each batch is processed during training and evaluation. This function is typically used when more control over the training and evaluation process is needed compared to higher-level functions.
- **Inputs**:
    - `opt_ctx`: The optimization context to use for the epoch. Must be initialized and configured with static graphs. The caller retains ownership.
    - `dataset`: The dataset to be used for training and evaluation. Must be initialized and contain the necessary data and labels. The caller retains ownership.
    - `result_train`: The result object to accumulate training results. Can be NULL if training results are not needed. The caller retains ownership.
    - `result_eval`: The result object to accumulate evaluation results. Can be NULL if evaluation results are not needed. The caller retains ownership.
    - `idata_split`: The data index at which to split the dataset into training and evaluation sections. If negative, the entire dataset is used for training. Must be a multiple of the batch size.
    - `callback_train`: An optional callback function to be called after each training batch is processed. Can be NULL if no callback is needed.
    - `callback_eval`: An optional callback function to be called after each evaluation batch is processed. Can be NULL if no callback is needed.
- **Output**: None
- **See also**: [`ggml_opt_epoch`](../src/ggml-opt.cpp.driver.md#ggml_opt_epoch)  (Implementation)


---
### ggml\_opt\_epoch\_callback\_progress\_bar<!-- {{#callable_declaration:ggml_opt_epoch_callback_progress_bar}} -->
Displays a progress bar for the optimization process on stderr.
- **Description**: This function is used to display a progress bar on the standard error output during the evaluation of an optimization context. It provides a visual representation of the progress of training or validation by showing the current batch number out of the total number of batches. The function also displays additional information such as data processed, loss, accuracy, elapsed time, and estimated time of arrival (ETA). It is typically used as a callback during the optimization process to give users real-time feedback on the progress of their model training or evaluation.
- **Inputs**:
    - `train`: A boolean indicating whether the current evaluation is for training (true) or validation (false).
    - `opt_ctx`: A ggml_opt_context_t representing the optimization context. The caller retains ownership and it must be a valid context.
    - `dataset`: A ggml_opt_dataset_t representing the dataset being used. The caller retains ownership and it must be a valid dataset.
    - `result`: A ggml_opt_result_t representing the result of the optimization process. The caller retains ownership and it must be a valid result object.
    - `ibatch`: An int64_t representing the current batch number that has been evaluated so far. It must be a non-negative integer.
    - `ibatch_max`: An int64_t representing the total number of batches in the dataset subsection. It must be a positive integer.
    - `t_start_us`: An int64_t representing the start time of the evaluation in microseconds. It is used to calculate elapsed time and ETA.
- **Output**: None
- **See also**: [`ggml_opt_epoch_callback_progress_bar`](../src/ggml-opt.cpp.driver.md#ggml_opt_epoch_callback_progress_bar)  (Implementation)


---
### ggml\_opt\_fit<!-- {{#callable_declaration:ggml_opt_fit}} -->
Fits a model to a dataset using specified optimization parameters.
- **Description**: This function is used to train a model by fitting it to a given dataset using specified optimization parameters and loss type. It iterates over the dataset for a specified number of epochs, performing optimization steps on batches of data. The function requires a backend scheduler, a compute context, input and output tensors, a dataset, a loss type, and a callback for optimizer parameters. It also allows specifying the number of epochs, the logical batch size, the validation split ratio, and whether to suppress output messages. The function should be called after setting up the model and dataset according to the intended usage guidelines.
- **Inputs**:
    - `backend_sched`: Defines the backend scheduler used for constructing compute graphs. Must be a valid ggml_backend_sched_t value.
    - `ctx_compute`: A pointer to a ggml_context used for temporary tensor allocations during output calculations. Must not be null.
    - `inputs`: A pointer to a ggml_tensor representing the input data with shape [ne_datapoint, ndata_batch]. Must not be null.
    - `outputs`: A pointer to a ggml_tensor representing the output data. If labels are used, it must have shape [ne_label, ndata_batch]. Must not be null.
    - `dataset`: A ggml_opt_dataset_t representing the dataset containing data and optionally labels. Must be properly initialized.
    - `loss_type`: Specifies the loss type to minimize. Must be a valid ggml_opt_loss_type value.
    - `get_opt_pars`: A callback function to retrieve optimizer parameters. The userdata is a pointer to the current epoch (int64_t). Must be a valid function pointer.
    - `nepoch`: The number of times the dataset should be iterated over. Must be a positive integer.
    - `nbatch_logical`: The number of datapoints per optimizer step. Must be a multiple of the number of datapoints in the input/output tensors.
    - `val_split`: The fraction of the dataset to use for validation. Must be a float in the range [0.0f, 1.0f).
    - `silent`: A boolean indicating whether to suppress informational prints to stderr. Set to true to suppress output.
- **Output**: None
- **See also**: [`ggml_opt_fit`](../src/ggml-opt.cpp.driver.md#ggml_opt_fit)  (Implementation)


