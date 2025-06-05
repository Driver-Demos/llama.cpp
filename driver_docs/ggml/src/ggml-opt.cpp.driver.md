# Purpose
This C++ source code file is part of a machine learning framework, specifically designed to handle optimization tasks. It provides a comprehensive set of functionalities for managing datasets, defining optimization contexts, and executing optimization processes. The code is structured around several key components: `ggml_opt_dataset`, `ggml_opt_context`, and `ggml_opt_result`, each encapsulating different aspects of the optimization workflow. The `ggml_opt_dataset` structure is responsible for initializing, managing, and shuffling datasets, as well as retrieving data batches for training. The `ggml_opt_context` structure defines the optimization context, including the setup of computational graphs, loss functions, and optimizer parameters. It supports various loss types such as mean, sum, cross-entropy, and mean squared error, and implements the AdamW optimization algorithm.

The file also includes high-level functions for executing optimization epochs and fitting models to datasets. The [`ggml_opt_epoch`](#ggml_opt_epoch) function orchestrates the training and evaluation process over multiple batches, while [`ggml_opt_fit`](#ggml_opt_fit) provides a complete training loop with support for validation splits and progress tracking. The code is designed to be flexible and efficient, utilizing backend scheduling and memory management to optimize performance. It defines public APIs for initializing and freeing datasets and optimization contexts, as well as for retrieving results such as loss and accuracy. Overall, this file serves as a core component of a machine learning library, providing essential tools for training and evaluating models.
# Imports and Dependencies

---
- `ggml-opt.h`
- `ggml.h`
- `ggml-alloc.h`
- `ggml-backend.h`
- `ggml-impl.h`
- `algorithm`
- `cmath`
- `cstdint`
- `cinttypes`
- `map`
- `random`
- `vector`


# Data Structures

---
### ggml\_opt\_dataset<!-- {{#data_structure:ggml_opt_dataset}} -->
- **Type**: `struct`
- **Members**:
    - `ctx`: A pointer to a ggml_context structure, initialized to nullptr.
    - `buf`: A ggml_backend_buffer_t type, initialized to nullptr.
    - `data`: A pointer to a ggml_tensor structure for data, initialized to nullptr.
    - `labels`: A pointer to a ggml_tensor structure for labels, initialized to nullptr.
    - `ndata`: An int64_t representing the number of data points, initialized to -1.
    - `ndata_shard`: An int64_t representing the number of data shards, initialized to -1.
    - `nbs_data`: A size_t representing the number of bytes per data shard, initialized to -1.
    - `nbs_labels`: A size_t representing the number of bytes per label shard, initialized to -1.
    - `permutation`: A vector of int64_t used to store the permutation of data shards.
- **Description**: The `ggml_opt_dataset` struct is designed to manage datasets for optimization tasks, encapsulating data and label tensors, context, and backend buffer information. It includes metadata such as the number of data points and shards, as well as byte sizes for data and label shards. Additionally, it maintains a permutation vector to facilitate data shuffling, which is crucial for training models with stochastic gradient descent or similar optimization algorithms.


---
### ggml\_opt\_context<!-- {{#data_structure:ggml_opt_context}} -->
- **Type**: `struct`
- **Members**:
    - `backend_sched`: A scheduling backend for the optimization process.
    - `allocated_graph`: Pointer to the allocated computation graph.
    - `allocated_graph_copy`: Pointer to a copy of the allocated computation graph.
    - `ctx_static`: Pointer to a static context for managing memory.
    - `ctx_cpu`: Pointer to a CPU context for managing memory.
    - `ctx_compute`: Pointer to a compute context for managing memory.
    - `ctx_copy`: Pointer to a context for copying operations.
    - `buf_static`: Static buffer for backend operations.
    - `buf_cpu`: CPU buffer for backend operations.
    - `rng`: Random number generator for shuffling and other stochastic processes.
    - `loss_type`: Type of loss function used in optimization.
    - `build_type`: Type of build process for the optimization graph.
    - `build_type_alloc`: Type of build process for allocation purposes.
    - `inputs`: Pointer to the input tensor for the optimization process.
    - `outputs`: Pointer to the output tensor for the optimization process.
    - `labels`: Pointer to the labels tensor for supervised learning.
    - `loss`: Pointer to the tensor representing the loss value.
    - `pred`: Pointer to the tensor representing predictions.
    - `ncorrect`: Pointer to the tensor representing the number of correct predictions.
    - `gf`: Pointer to the forward computation graph.
    - `gb_grad`: Pointer to the backward gradient computation graph.
    - `gb_opt`: Pointer to the backward optimization computation graph.
    - `static_graphs`: Boolean indicating if static graphs are used.
    - `eval_ready`: Boolean indicating if the evaluation is ready to be performed.
    - `grad_accs`: Vector of pointers to tensors for gradient accumulation.
    - `grad_m`: Vector of pointers to tensors for first moment estimates in optimization.
    - `grad_v`: Vector of pointers to tensors for second moment estimates in optimization.
    - `iter`: Current iteration number in the optimization process.
    - `opt_period`: Period of optimization steps before updating parameters.
    - `opt_i`: Current index in the optimization period.
    - `loss_per_datapoint`: Boolean indicating if loss is calculated per datapoint.
    - `get_opt_pars`: Function pointer to get optimizer parameters.
    - `get_opt_pars_ud`: User data for the optimizer parameters function.
    - `adamw_params`: Pointer to the tensor containing AdamW optimizer parameters.
- **Description**: The `ggml_opt_context` struct is a comprehensive data structure designed to manage the context and state of an optimization process in a machine learning framework. It includes various pointers to contexts and computation graphs, buffers for backend operations, and parameters for optimization algorithms such as AdamW. The struct also maintains state information such as the current iteration, optimization period, and whether static graphs are used. Additionally, it holds vectors for gradient accumulation and moment estimates, which are crucial for optimization algorithms. This struct is central to managing the lifecycle and execution of optimization tasks, including setting up computation graphs, handling memory contexts, and executing optimization steps.


---
### ggml\_opt\_result<!-- {{#data_structure:ggml_opt_result}} -->
- **Type**: `struct`
- **Members**:
    - `ndata`: Stores the number of data points processed, initialized to 0.
    - `loss`: A vector of floats representing the loss values for each batch.
    - `pred`: A vector of integers representing the predicted values for each data point.
    - `ncorrect`: Stores the number of correct predictions, initialized to 0.
    - `opt_period`: Indicates the optimization period, initialized to -1.
    - `loss_per_datapoint`: A boolean indicating if the loss is calculated per data point, initialized to false.
- **Description**: The `ggml_opt_result` struct is designed to store the results of an optimization process, including the number of data points processed (`ndata`), the loss values for each batch (`loss`), the predicted values (`pred`), and the number of correct predictions (`ncorrect`). It also includes configuration parameters such as the optimization period (`opt_period`) and a flag to indicate if the loss is calculated per data point (`loss_per_datapoint`). This struct is essential for tracking the performance and outcomes of optimization algorithms in machine learning contexts.


# Functions

---
### ggml\_opt\_dataset\_init<!-- {{#callable:ggml_opt_dataset_init}} -->
The `ggml_opt_dataset_init` function initializes a dataset structure for optimization, setting up data and label tensors, context, and shard permutations.
- **Inputs**:
    - `type_data`: The data type for the data tensor, specified as an enum `ggml_type`.
    - `type_label`: The data type for the label tensor, specified as an enum `ggml_type`.
    - `ne_datapoint`: The number of elements in each data point, must be greater than 0.
    - `ne_label`: The number of elements in each label, must be non-negative.
    - `ndata`: The total number of data points, must be greater than 0.
    - `ndata_shard`: The number of data points per shard, must be greater than 0.
- **Control Flow**:
    - Assert that `ne_datapoint` is greater than 0, `ne_label` is non-negative, `ndata` is greater than 0, and `ndata_shard` is greater than 0.
    - Allocate a new `ggml_opt_dataset` structure and set `ndata` and `ndata_shard` fields.
    - Initialize a `ggml_context` with specific parameters and assign it to the dataset's context.
    - Create a 2D tensor for data using the context, data type, number of elements per data point, and total data points, then calculate and store the number of bytes per shard for data.
    - If `ne_label` is greater than 0, create a 2D tensor for labels similarly and calculate the number of bytes per shard for labels; otherwise, set labels to `nullptr` and bytes per shard to 0.
    - Allocate backend buffer for context tensors and assign it to the dataset's buffer.
    - Calculate the number of shards and initialize a permutation vector with indices from 0 to the number of shards.
    - Return the initialized dataset structure.
- **Output**: Returns a pointer to the initialized `ggml_opt_dataset_t` structure containing the dataset configuration and tensors.
- **Functions called**:
    - [`ggml_tensor_overhead`](ggml.c.driver.md#ggml_tensor_overhead)
    - [`ggml_init`](ggml.c.driver.md#ggml_init)
    - [`ggml_new_tensor_2d`](ggml.c.driver.md#ggml_new_tensor_2d)
    - [`ggml_nbytes`](ggml.c.driver.md#ggml_nbytes)
    - [`ggml_backend_alloc_ctx_tensors_from_buft`](ggml-alloc.c.driver.md#ggml_backend_alloc_ctx_tensors_from_buft)


---
### ggml\_opt\_dataset\_free<!-- {{#callable:ggml_opt_dataset_free}} -->
The `ggml_opt_dataset_free` function deallocates memory associated with a `ggml_opt_dataset_t` object.
- **Inputs**:
    - `dataset`: A pointer to a `ggml_opt_dataset_t` object that needs to be freed.
- **Control Flow**:
    - Call [`ggml_backend_buffer_free`](ggml-backend.cpp.driver.md#ggml_backend_buffer_free) to free the backend buffer associated with the dataset.
    - Call [`ggml_free`](ggml.c.driver.md#ggml_free) to free the context associated with the dataset.
    - Delete the dataset object itself to free its memory.
- **Output**: This function does not return any value.
- **Functions called**:
    - [`ggml_backend_buffer_free`](ggml-backend.cpp.driver.md#ggml_backend_buffer_free)
    - [`ggml_free`](ggml.c.driver.md#ggml_free)


---
### ggml\_opt\_dataset\_ndata<!-- {{#callable:ggml_opt_dataset_ndata}} -->
The function `ggml_opt_dataset_ndata` retrieves the number of data points in a given optimization dataset.
- **Inputs**:
    - `dataset`: A `ggml_opt_dataset_t` object representing the optimization dataset from which the number of data points is to be retrieved.
- **Control Flow**:
    - The function accesses the `ndata` member of the `ggml_opt_dataset` structure pointed to by the `dataset` argument.
    - It returns the value of `ndata`, which represents the number of data points in the dataset.
- **Output**: An `int64_t` value representing the number of data points in the dataset.


---
### ggml\_opt\_dataset\_data<!-- {{#callable:ggml_opt_dataset_data}} -->
The function `ggml_opt_dataset_data` retrieves the data tensor from a given optimization dataset.
- **Inputs**:
    - `dataset`: A `ggml_opt_dataset_t` object representing the optimization dataset from which the data tensor is to be retrieved.
- **Control Flow**:
    - The function takes a single argument, `dataset`, which is a pointer to a `ggml_opt_dataset` structure.
    - It directly accesses the `data` member of the `dataset` structure and returns it.
- **Output**: Returns a pointer to a `ggml_tensor` structure, which represents the data tensor stored in the given dataset.


---
### ggml\_opt\_dataset\_labels<!-- {{#callable:ggml_opt_dataset_labels}} -->
The function `ggml_opt_dataset_labels` retrieves the labels tensor from a given optimization dataset.
- **Inputs**:
    - `dataset`: A `ggml_opt_dataset_t` object representing the optimization dataset from which the labels tensor is to be retrieved.
- **Control Flow**:
    - The function accesses the `labels` member of the `dataset` structure and returns it.
- **Output**: A pointer to a `ggml_tensor` structure representing the labels tensor of the dataset.


---
### ggml\_opt\_dataset\_shuffle<!-- {{#callable:ggml_opt_dataset_shuffle}} -->
The `ggml_opt_dataset_shuffle` function shuffles the permutation vector of a dataset either entirely or up to a specified shard index using a random number generator from the optimization context.
- **Inputs**:
    - `opt_ctx`: A `ggml_opt_context_t` object that contains the random number generator used for shuffling.
    - `dataset`: A `ggml_opt_dataset_t` object representing the dataset whose permutation vector is to be shuffled.
    - `idata`: An `int64_t` value indicating the index up to which the permutation should be shuffled; if negative, the entire permutation is shuffled.
- **Control Flow**:
    - Assert that `idata` is less than or equal to the total number of data points in the dataset.
    - If `idata` is negative, shuffle the entire permutation vector of the dataset using the random number generator from `opt_ctx`.
    - If `idata` is non-negative, assert that it is a multiple of `ndata_shard` from the dataset.
    - Calculate `ishard_max` as the quotient of `idata` divided by `ndata_shard`.
    - Shuffle the permutation vector from the beginning up to `ishard_max` using the random number generator from `opt_ctx`.
- **Output**: This function does not return any value; it modifies the permutation vector of the dataset in place.


---
### ggml\_opt\_dataset\_get\_batch<!-- {{#callable:ggml_opt_dataset_get_batch}} -->
The `ggml_opt_dataset_get_batch` function retrieves a specific batch of data and labels from a dataset based on a given batch index.
- **Inputs**:
    - `dataset`: A `ggml_opt_dataset_t` object representing the dataset from which to retrieve the batch.
    - `data_batch`: A pointer to a `ggml_tensor` where the data batch will be stored; must be contiguous and of the same type as the dataset's data.
    - `labels_batch`: A pointer to a `ggml_tensor` where the labels batch will be stored; must be contiguous and of the same type as the dataset's labels, or can be `nullptr` if the dataset has no labels.
    - `ibatch`: An `int64_t` representing the index of the batch to retrieve.
- **Control Flow**:
    - Assert that `data_batch` is not null and is contiguous, and if `labels_batch` is provided, it must also be contiguous.
    - Assert that the presence of `labels_batch` matches the presence of labels in the dataset.
    - Assert that the data types of `data_batch` and `labels_batch` (if present) match those of the dataset's data and labels, respectively.
    - Calculate the number of data shards per batch based on the size of `data_batch` and the dataset's data shard size.
    - If `labels_batch` is provided, assert that its size matches the expected size based on the number of shards per batch and the dataset's label shard size.
    - Assert that the total number of shards for the requested batch does not exceed the dataset's permutation size.
    - Iterate over each shard in the batch, using the dataset's permutation to determine the shard index.
    - For each shard, set the data in `data_batch` using the backend tensor set function.
    - If `labels_batch` is provided, set the labels in `labels_batch` using the backend tensor set function.
- **Output**: The function does not return a value; it modifies `data_batch` and `labels_batch` in place to contain the requested batch of data and labels.
- **Functions called**:
    - [`ggml_is_contiguous`](ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_nbytes`](ggml.c.driver.md#ggml_nbytes)
    - [`ggml_backend_tensor_set`](ggml-backend.cpp.driver.md#ggml_backend_tensor_set)


---
### ggml\_opt\_dataset\_get\_batch\_host<!-- {{#callable:ggml_opt_dataset_get_batch_host}} -->
The `ggml_opt_dataset_get_batch_host` function retrieves a batch of data and optionally labels from a dataset, copying them into provided memory buffers for processing on the host.
- **Inputs**:
    - `dataset`: A `ggml_opt_dataset_t` object representing the dataset from which to retrieve the batch.
    - `data_batch`: A pointer to a memory buffer where the data batch will be copied.
    - `nb_data_batch`: The size of the data batch in bytes.
    - `labels_batch`: A pointer to a memory buffer where the labels batch will be copied, or `nullptr` if labels are not needed.
    - `ibatch`: The index of the batch to retrieve.
- **Control Flow**:
    - Assert that the presence of `labels_batch` matches the presence of labels in the dataset.
    - Assert that `nb_data_batch` is a multiple of `dataset->nbs_data`.
    - Calculate the number of shards per batch as `nb_data_batch / dataset->nbs_data`.
    - Assert that the batch index and shards per batch do not exceed the dataset's permutation size.
    - Iterate over each shard in the batch, using the dataset's permutation to determine the shard index.
    - Copy data from the dataset's data buffer to the `data_batch` buffer for each shard.
    - If `labels_batch` is not `nullptr`, copy labels from the dataset's labels buffer to the `labels_batch` buffer for each shard.
- **Output**: The function does not return a value; it modifies the `data_batch` and `labels_batch` buffers in place.


---
### ggml\_opt\_get\_default\_optimizer\_params<!-- {{#callable:ggml_opt_get_default_optimizer_params}} -->
The function `ggml_opt_get_default_optimizer_params` initializes and returns a set of default parameters for the AdamW optimizer.
- **Inputs**:
    - `userdata`: A pointer to user data, which is not used in this function.
- **Control Flow**:
    - The function begins by marking the `userdata` parameter as unused with `GGML_UNUSED(userdata);`.
    - A `ggml_opt_optimizer_params` structure named `result` is declared.
    - The function sets default values for the AdamW optimizer parameters within the `result` structure: `alpha` to 0.001, `beta1` to 0.9, `beta2` to 0.999, `eps` to 1e-8, and `wd` to 0.0.
    - The function returns the `result` structure containing the initialized optimizer parameters.
- **Output**: A `ggml_opt_optimizer_params` structure containing default AdamW optimizer parameters.


---
### ggml\_opt\_get\_constant\_optimizer\_params<!-- {{#callable:ggml_opt_get_constant_optimizer_params}} -->
The function `ggml_opt_get_constant_optimizer_params` retrieves constant optimizer parameters from a given user data pointer.
- **Inputs**:
    - `userdata`: A pointer to user data that is expected to point to a `ggml_opt_optimizer_params` structure.
- **Control Flow**:
    - The function casts the `userdata` pointer to a `ggml_opt_optimizer_params` pointer.
    - It then dereferences this pointer to return the `ggml_opt_optimizer_params` structure.
- **Output**: Returns a `ggml_opt_optimizer_params` structure containing optimizer parameters.


---
### ggml\_opt\_default\_params<!-- {{#callable:ggml_opt_default_params}} -->
The `ggml_opt_default_params` function initializes and returns a `ggml_opt_params` structure with default optimization parameters based on the provided backend scheduler and loss type.
- **Inputs**:
    - `backend_sched`: A `ggml_backend_sched_t` type representing the backend scheduler to be used for optimization.
    - `loss_type`: An `enum ggml_opt_loss_type` indicating the type of loss function to be used in the optimization process.
- **Control Flow**:
    - The function takes two parameters: `backend_sched` and `loss_type`.
    - It returns a `ggml_opt_params` structure initialized with default values, where `backend_sched` and `loss_type` are set to the provided arguments.
    - Other fields in the structure are set to default values, such as `ctx_compute` and `inputs` being set to `nullptr`, `build_type` set to `GGML_OPT_BUILD_TYPE_OPT`, `opt_period` set to `1`, and `get_opt_pars` set to `ggml_opt_get_default_optimizer_params`.
- **Output**: A `ggml_opt_params` structure initialized with default values for optimization parameters.


---
### map\_tensor<!-- {{#callable:map_tensor}} -->
The `map_tensor` function duplicates a given tensor and its dependencies, storing the mapping in a provided map to avoid redundant duplications.
- **Inputs**:
    - `tensor_map`: A reference to a map that associates original tensors with their duplicated counterparts.
    - `ctx`: A pointer to a ggml_context, which is used for memory management during tensor duplication.
    - `tensor`: A pointer to the ggml_tensor that needs to be duplicated.
- **Control Flow**:
    - Check if the input tensor is null and return nullptr if true.
    - Check if the tensor already exists in the map; if so, return the mapped tensor.
    - Duplicate the tensor using [`ggml_dup_tensor`](ggml.c.driver.md#ggml_dup_tensor) and store the new tensor in the map.
    - Copy various properties from the original tensor to the new tensor, including operation, dimensions, flags, parameters, name, data, buffer, extra, view offsets, and view source.
    - Recursively duplicate the view source tensor and source tensors, updating the map accordingly.
    - Return the newly created tensor.
- **Output**: A pointer to the newly duplicated ggml_tensor.
- **Functions called**:
    - [`ggml_dup_tensor`](ggml.c.driver.md#ggml_dup_tensor)


---
### dup\_graph<!-- {{#callable:dup_graph}} -->
The `dup_graph` function duplicates a computational graph, including its nodes, leafs, and gradients, from a source graph to a destination graph within a given context.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure, which provides the context for memory allocation and graph operations.
    - `src`: A pointer to a `ggml_cgraph` structure representing the source computational graph to be duplicated.
- **Control Flow**:
    - Initialize a map to track the mapping between source and destination tensors.
    - Create a new destination graph `dst` using [`ggml_new_graph_custom`](ggml.c.driver.md#ggml_new_graph_custom) with the same size as the source graph and enable gradients.
    - Iterate over each leaf tensor in the source graph, map it to a new tensor in the destination graph using [`map_tensor`](#map_tensor), and expand the destination graph with this tensor using [`ggml_build_forward_expand`](ggml.c.driver.md#ggml_build_forward_expand).
    - Assert that the number of leafs in the destination graph matches the source graph.
    - Iterate over each node tensor in the source graph, map it to a new tensor in the destination graph using [`map_tensor`](#map_tensor), and expand the destination graph with this tensor using [`ggml_build_forward_expand`](ggml.c.driver.md#ggml_build_forward_expand).
    - Assert that the number of nodes in the destination graph matches the source graph.
    - For each node in the source graph, find the corresponding gradient indices in both the source and destination graphs using [`ggml_hash_find`](ggml-impl.h.driver.md#ggml_hash_find).
    - Assert that the gradient indices are valid and used in both graphs.
    - Copy the gradients and gradient accumulators from the source graph to the destination graph using the mapped gradient indices.
    - Return the duplicated destination graph `dst`.
- **Output**: A pointer to the newly created `ggml_cgraph` structure, which is a duplicate of the source graph with all nodes, leafs, and gradients copied over.
- **Functions called**:
    - [`ggml_new_graph_custom`](ggml.c.driver.md#ggml_new_graph_custom)
    - [`ggml_build_forward_expand`](ggml.c.driver.md#ggml_build_forward_expand)
    - [`map_tensor`](#map_tensor)
    - [`ggml_hash_find`](ggml-impl.h.driver.md#ggml_hash_find)
    - [`ggml_bitset_get`](ggml-impl.h.driver.md#ggml_bitset_get)


---
### ggml\_opt\_build<!-- {{#callable:ggml_opt_build}} -->
The `ggml_opt_build` function initializes and configures the optimization context for a given computation graph, setting up necessary resources and loss calculations based on the specified optimization and loss types.
- **Inputs**:
    - `opt_ctx`: A `ggml_opt_context_t` structure that contains the context and configuration for the optimization process, including computation graphs, input/output tensors, and various optimization parameters.
- **Control Flow**:
    - Assert that the compute context is set and inputs are allocated if using static graphs.
    - Determine if gradient accumulation is needed based on the build type and static graph settings.
    - Set the input and output tensors for the optimization context.
    - Count the number of parameter nodes in the computation graph and assert no extra loss terms are present.
    - Initialize the static context if not already set, allocating memory for gradients, optimizer momenta, labels, loss, predictions, and correctness metrics as needed.
    - Initialize the CPU context for optimizer parameters, freeing any existing context and buffer.
    - Select the appropriate context for results based on whether static graphs are used.
    - Calculate the loss based on the specified loss type (mean, sum, cross-entropy, or mean squared error), scaling as necessary, and set the loss as the output.
    - If using cross-entropy loss, calculate predictions and correctness metrics, setting them as outputs.
    - Handle buffer allocation for static graphs and return early if only forward pass is needed.
    - Initialize gradient accumulators and optimizer momenta if not already set, based on the build type.
    - Duplicate the computation graph for backward gradient calculation and build the backward pass.
    - Allocate buffers for static graphs if needed and reset the backward gradient graph if only gradient calculation is needed.
    - Duplicate the computation graph for optimization, setting up AdamW optimizer steps for parameter nodes.
    - Allocate buffers for static graphs if needed and reset the optimization graph.
    - Allocate CPU buffers for the context.
- **Output**: The function does not return a value; it modifies the `opt_ctx` structure in place, setting up the necessary contexts, buffers, and computation graphs for optimization.
- **Functions called**:
    - [`ggml_set_input`](ggml.c.driver.md#ggml_set_input)
    - [`ggml_set_output`](ggml.c.driver.md#ggml_set_output)
    - [`ggml_tensor_overhead`](ggml.c.driver.md#ggml_tensor_overhead)
    - [`ggml_init`](ggml.c.driver.md#ggml_init)
    - [`ggml_free`](ggml.c.driver.md#ggml_free)
    - [`ggml_backend_buffer_free`](ggml-backend.cpp.driver.md#ggml_backend_buffer_free)
    - [`ggml_sum`](ggml.c.driver.md#ggml_sum)
    - [`ggml_set_name`](ggml.c.driver.md#ggml_set_name)
    - [`ggml_nelements`](ggml.c.driver.md#ggml_nelements)
    - [`ggml_scale`](ggml.c.driver.md#ggml_scale)
    - [`ggml_dup_tensor`](ggml.c.driver.md#ggml_dup_tensor)
    - [`ggml_cross_entropy_loss`](ggml.c.driver.md#ggml_cross_entropy_loss)
    - [`ggml_sub`](ggml.c.driver.md#ggml_sub)
    - [`ggml_sqr`](ggml.c.driver.md#ggml_sqr)
    - [`ggml_set_loss`](ggml.c.driver.md#ggml_set_loss)
    - [`ggml_build_forward_expand`](ggml.c.driver.md#ggml_build_forward_expand)
    - [`ggml_argmax`](ggml.c.driver.md#ggml_argmax)
    - [`ggml_count_equal`](ggml.c.driver.md#ggml_count_equal)
    - [`ggml_backend_alloc_ctx_tensors`](ggml-alloc.c.driver.md#ggml_backend_alloc_ctx_tensors)
    - [`ggml_new_tensor`](ggml.c.driver.md#ggml_new_tensor)
    - [`ggml_graph_dup`](ggml.c.driver.md#ggml_graph_dup)
    - [`ggml_build_backward_expand`](ggml.c.driver.md#ggml_build_backward_expand)
    - [`ggml_graph_reset`](ggml.c.driver.md#ggml_graph_reset)
    - [`ggml_new_tensor_1d`](ggml.c.driver.md#ggml_new_tensor_1d)
    - [`ggml_graph_get_grad`](ggml.c.driver.md#ggml_graph_get_grad)
    - [`ggml_opt_step_adamw`](ggml.c.driver.md#ggml_opt_step_adamw)
    - [`ggml_backend_alloc_ctx_tensors_from_buft`](ggml-alloc.c.driver.md#ggml_backend_alloc_ctx_tensors_from_buft)


---
### ggml\_opt\_init<!-- {{#callable:ggml_opt_init}} -->
The `ggml_opt_init` function initializes and returns a new optimization context based on the provided optimization parameters.
- **Inputs**:
    - `params`: A `ggml_opt_params` structure containing various parameters for optimization, such as backend scheduling, compute context, loss type, build type, inputs, outputs, optimization period, and optimizer parameter retrieval function.
- **Control Flow**:
    - Allocate memory for a new `ggml_opt_context` structure and assign it to `result`.
    - Copy the values from `params` to the corresponding fields in `result`.
    - Assert that the optimization period (`opt_period`) is at least 1.
    - Set `static_graphs` to the value of `ctx_compute` to determine if static graphs are used.
    - If `static_graphs` is false, assert that `inputs` and `outputs` are null and return `result`.
    - If `static_graphs` is true, assert that `inputs` and `outputs` are not null.
    - Create a new forward computation graph (`gf`) using `ctx_compute` and expand it with `outputs`.
    - Call [`ggml_opt_build`](#ggml_opt_build) to further configure the optimization context.
    - Return the initialized `ggml_opt_context` structure.
- **Output**: A `ggml_opt_context_t` structure, which is a pointer to the newly initialized optimization context.
- **Functions called**:
    - [`ggml_new_graph_custom`](ggml.c.driver.md#ggml_new_graph_custom)
    - [`ggml_build_forward_expand`](ggml.c.driver.md#ggml_build_forward_expand)
    - [`ggml_opt_build`](#ggml_opt_build)


---
### ggml\_opt\_free<!-- {{#callable:ggml_opt_free}} -->
The `ggml_opt_free` function deallocates and cleans up resources associated with a given optimization context.
- **Inputs**:
    - `opt_ctx`: A pointer to a `ggml_opt_context_t` structure representing the optimization context to be freed.
- **Control Flow**:
    - Check if `opt_ctx` is `nullptr`; if so, return immediately.
    - Call [`ggml_backend_buffer_free`](ggml-backend.cpp.driver.md#ggml_backend_buffer_free) to free the static buffer associated with `opt_ctx`.
    - Call [`ggml_backend_buffer_free`](ggml-backend.cpp.driver.md#ggml_backend_buffer_free) to free the CPU buffer associated with `opt_ctx`.
    - Call [`ggml_free`](ggml.c.driver.md#ggml_free) to free the static context associated with `opt_ctx`.
    - Call [`ggml_free`](ggml.c.driver.md#ggml_free) to free the CPU context associated with `opt_ctx`.
    - Delete the `opt_ctx` pointer to free the memory allocated for the optimization context.
- **Output**: This function does not return any value; it performs cleanup operations on the provided optimization context.
- **Functions called**:
    - [`ggml_backend_buffer_free`](ggml-backend.cpp.driver.md#ggml_backend_buffer_free)
    - [`ggml_free`](ggml.c.driver.md#ggml_free)


---
### ggml\_opt\_reset<!-- {{#callable:ggml_opt_reset}} -->
The `ggml_opt_reset` function resets the optimization context by resetting the appropriate computation graph and optionally setting the iteration counter to 1.
- **Inputs**:
    - `opt_ctx`: A `ggml_opt_context_t` object representing the optimization context, which contains various fields related to the optimization process.
    - `optimizer`: A boolean flag indicating whether to reset the optimizer graph (`true`) or the gradient graph (`false`).
- **Control Flow**:
    - Check if the `optimizer` flag is true.
    - If true, reset the optimizer graph (`gb_opt`) in the optimization context and set the iteration counter (`iter`) to 1.
    - If false, reset the gradient graph (`gb_grad`) in the optimization context.
- **Output**: This function does not return any value; it modifies the state of the `opt_ctx` object.
- **Functions called**:
    - [`ggml_graph_reset`](ggml.c.driver.md#ggml_graph_reset)


---
### ggml\_opt\_static\_graphs<!-- {{#callable:ggml_opt_static_graphs}} -->
The function `ggml_opt_static_graphs` checks if the optimization context is using static graphs.
- **Inputs**:
    - `opt_ctx`: A `ggml_opt_context_t` object representing the optimization context, which contains various settings and states for the optimization process.
- **Control Flow**:
    - The function accesses the `static_graphs` member of the `opt_ctx` structure.
    - It returns the value of `static_graphs`, which is a boolean indicating whether static graphs are being used.
- **Output**: A boolean value indicating whether the optimization context is using static graphs.


---
### ggml\_opt\_inputs<!-- {{#callable:ggml_opt_inputs}} -->
The `ggml_opt_inputs` function retrieves the input tensor from a given optimization context.
- **Inputs**:
    - `opt_ctx`: A `ggml_opt_context_t` object representing the optimization context from which the input tensor is to be retrieved.
- **Control Flow**:
    - The function accesses the `inputs` member of the `opt_ctx` structure.
    - It returns the `inputs` tensor associated with the given optimization context.
- **Output**: A pointer to a `ggml_tensor` representing the inputs of the optimization context.


---
### ggml\_opt\_outputs<!-- {{#callable:ggml_opt_outputs}} -->
The function `ggml_opt_outputs` retrieves the output tensor from a given optimization context.
- **Inputs**:
    - `opt_ctx`: A `ggml_opt_context_t` object representing the optimization context from which the output tensor is to be retrieved.
- **Control Flow**:
    - The function accesses the `outputs` member of the `opt_ctx` structure and returns it.
- **Output**: A pointer to a `ggml_tensor` structure representing the outputs of the optimization context.


---
### ggml\_opt\_labels<!-- {{#callable:ggml_opt_labels}} -->
The function `ggml_opt_labels` retrieves the labels tensor from a given optimization context.
- **Inputs**:
    - `opt_ctx`: A `ggml_opt_context_t` object representing the optimization context from which the labels tensor is to be retrieved.
- **Control Flow**:
    - The function accesses the `labels` member of the `opt_ctx` structure and returns it.
- **Output**: A pointer to a `ggml_tensor` structure representing the labels tensor associated with the given optimization context.


---
### ggml\_opt\_loss<!-- {{#callable:ggml_opt_loss}} -->
The `ggml_opt_loss` function retrieves the loss tensor from a given optimization context.
- **Inputs**:
    - `opt_ctx`: A `ggml_opt_context_t` object representing the optimization context from which the loss tensor is to be retrieved.
- **Control Flow**:
    - The function accesses the `loss` member of the `opt_ctx` structure and returns it.
- **Output**: A pointer to a `ggml_tensor` representing the loss tensor from the optimization context.


---
### ggml\_opt\_pred<!-- {{#callable:ggml_opt_pred}} -->
The function `ggml_opt_pred` retrieves the prediction tensor from a given optimization context.
- **Inputs**:
    - `opt_ctx`: A `ggml_opt_context_t` object representing the optimization context from which the prediction tensor is to be retrieved.
- **Control Flow**:
    - The function accesses the `pred` member of the `opt_ctx` structure and returns it.
- **Output**: A pointer to a `ggml_tensor` structure representing the prediction tensor associated with the given optimization context.


---
### ggml\_opt\_ncorrect<!-- {{#callable:ggml_opt_ncorrect}} -->
The function `ggml_opt_ncorrect` retrieves the `ncorrect` tensor from a given optimization context.
- **Inputs**:
    - `opt_ctx`: A `ggml_opt_context_t` object representing the optimization context from which the `ncorrect` tensor is to be retrieved.
- **Control Flow**:
    - The function takes a single argument, `opt_ctx`, which is a pointer to a `ggml_opt_context` structure.
    - It directly accesses the `ncorrect` member of the `opt_ctx` structure and returns it.
- **Output**: A pointer to a `ggml_tensor` representing the `ncorrect` tensor from the optimization context.


---
### ggml\_opt\_grad\_acc<!-- {{#callable:ggml_opt_grad_acc}} -->
The `ggml_opt_grad_acc` function retrieves the gradient accumulation tensor for a given node in the optimization context.
- **Inputs**:
    - `opt_ctx`: A `ggml_opt_context_t` object representing the optimization context, which contains various settings and states for the optimization process.
    - `node`: A pointer to a `ggml_tensor` structure representing the node for which the gradient accumulation tensor is to be retrieved.
- **Control Flow**:
    - The function calls [`ggml_graph_get_grad_acc`](ggml.c.driver.md#ggml_graph_get_grad_acc) with the optimization graph (`gb_opt`) from the context and the specified node.
    - It returns the result of this call, which is the gradient accumulation tensor for the node.
- **Output**: A pointer to a `ggml_tensor` structure representing the gradient accumulation tensor for the specified node.
- **Functions called**:
    - [`ggml_graph_get_grad_acc`](ggml.c.driver.md#ggml_graph_get_grad_acc)


---
### ggml\_opt\_result\_init<!-- {{#callable:ggml_opt_result_init}} -->
The `ggml_opt_result_init` function initializes and returns a new instance of the `ggml_opt_result` structure.
- **Inputs**: None
- **Control Flow**:
    - The function creates a new instance of the `ggml_opt_result` structure using the `new` keyword.
    - It returns the newly created instance.
- **Output**: A pointer to a newly allocated `ggml_opt_result` structure is returned.


---
### ggml\_opt\_result\_free<!-- {{#callable:ggml_opt_result_free}} -->
The `ggml_opt_result_free` function deallocates memory associated with a `ggml_opt_result_t` object.
- **Inputs**:
    - `result`: A `ggml_opt_result_t` object that needs to be deallocated.
- **Control Flow**:
    - The function takes a `ggml_opt_result_t` object as an argument.
    - It uses the `delete` operator to deallocate the memory associated with the `result` object.
- **Output**: The function does not return any value.


---
### ggml\_opt\_result\_reset<!-- {{#callable:ggml_opt_result_reset}} -->
The `ggml_opt_result_reset` function resets the fields of a `ggml_opt_result_t` structure to their initial states.
- **Inputs**:
    - `result`: A pointer to a `ggml_opt_result_t` structure that holds optimization results, which will be reset by this function.
- **Control Flow**:
    - Set the `ndata` field of the `result` structure to 0.
    - Clear the `loss` vector of the `result` structure.
    - Clear the `pred` vector of the `result` structure.
    - Set the `ncorrect` field of the `result` structure to 0.
- **Output**: This function does not return any value; it modifies the `result` structure in place.


---
### ggml\_opt\_result\_ndata<!-- {{#callable:ggml_opt_result_ndata}} -->
The function `ggml_opt_result_ndata` assigns the `ndata` value from a `ggml_opt_result_t` structure to a given pointer.
- **Inputs**:
    - `result`: A `ggml_opt_result_t` structure from which the `ndata` value is retrieved.
    - `ndata`: A pointer to an `int64_t` where the `ndata` value from the `result` will be stored.
- **Control Flow**:
    - The function takes a `ggml_opt_result_t` structure and an `int64_t` pointer as arguments.
    - It assigns the `ndata` value from the `result` structure to the location pointed to by `ndata`.
- **Output**: The function does not return a value; it modifies the value pointed to by `ndata`.


---
### ggml\_opt\_result\_loss<!-- {{#callable:ggml_opt_result_loss}} -->
The `ggml_opt_result_loss` function calculates the mean loss and its uncertainty from a given optimization result.
- **Inputs**:
    - `result`: A `ggml_opt_result_t` object containing the loss data and configuration for the optimization result.
    - `loss`: A pointer to a double where the calculated mean loss will be stored.
    - `unc`: A pointer to a double where the calculated uncertainty of the loss will be stored, or `nullptr` if uncertainty is not needed.
- **Control Flow**:
    - Retrieve the number of batches from the `result` object.
    - If there are no batches, set `loss` to 0.0 and `unc` to `NAN`, then return.
    - Initialize `sum` and `sum_squared` to 0.0 for accumulating loss values.
    - Iterate over each loss value in `result->loss`, scale it if `loss_per_datapoint` is true, and update `sum` and `sum_squared`.
    - Calculate the mean loss and store it in `loss`, adjusting for `loss_per_datapoint` if necessary.
    - If `unc` is `nullptr`, return without calculating uncertainty.
    - If there is only one batch, set `unc` to `NAN` and return.
    - Calculate the variance of the loss and store the standard deviation in `unc`, adjusting for `loss_per_datapoint` if necessary.
- **Output**: The function outputs the mean loss in the `loss` pointer and the uncertainty of the loss in the `unc` pointer, if provided.


---
### ggml\_opt\_result\_pred<!-- {{#callable:ggml_opt_result_pred}} -->
The `ggml_opt_result_pred` function copies prediction results from a `ggml_opt_result_t` structure to an integer array.
- **Inputs**:
    - `result`: A `ggml_opt_result_t` structure containing prediction results in its `pred` vector.
    - `pred`: A pointer to an integer array where the prediction results will be copied.
- **Control Flow**:
    - Iterate over the `pred` vector in the `result` structure.
    - Copy each element from the `result->pred` vector to the corresponding index in the `pred` array.
- **Output**: The function does not return a value; it modifies the `pred` array in place.


---
### ggml\_opt\_result\_accuracy<!-- {{#callable:ggml_opt_result_accuracy}} -->
The `ggml_opt_result_accuracy` function calculates the accuracy and its uncertainty from a given optimization result.
- **Inputs**:
    - `result`: A `ggml_opt_result_t` structure containing the optimization result data, including the number of correct predictions and total data points.
    - `accuracy`: A pointer to a double where the calculated accuracy will be stored.
    - `unc`: A pointer to a double where the calculated uncertainty of the accuracy will be stored, or `nullptr` if uncertainty is not needed.
- **Control Flow**:
    - Calculate the accuracy as the ratio of correct predictions to total data points if the number of correct predictions is non-negative; otherwise, set accuracy to NaN.
    - If the `unc` pointer is not null, calculate the uncertainty using the standard error formula if the number of correct predictions is non-negative and there are at least two data points; otherwise, set uncertainty to NaN.
    - Return from the function if `unc` is null after calculating accuracy.
- **Output**: The function does not return a value but modifies the values pointed to by `accuracy` and `unc`.


---
### ggml\_opt\_prepare\_alloc<!-- {{#callable:ggml_opt_prepare_alloc}} -->
The `ggml_opt_prepare_alloc` function initializes an optimization context with the provided compute context, computation graph, input, and output tensors.
- **Inputs**:
    - `opt_ctx`: A `ggml_opt_context_t` object representing the optimization context to be prepared.
    - `ctx_compute`: A pointer to a `ggml_context` structure representing the compute context to be used for the optimization.
    - `gf`: A pointer to a `ggml_cgraph` structure representing the computation graph for the optimization.
    - `inputs`: A pointer to a `ggml_tensor` structure representing the input tensors for the optimization.
    - `outputs`: A pointer to a `ggml_tensor` structure representing the output tensors for the optimization.
- **Control Flow**:
    - Assert that the optimization context does not use static graphs with `GGML_ASSERT(!opt_ctx->static_graphs);`.
    - Assign the `ctx_compute` parameter to the `ctx_compute` field of the `opt_ctx`.
    - Assign the `gf` parameter to the `gf` field of the `opt_ctx`.
    - Assign the `inputs` parameter to the `inputs` field of the `opt_ctx`.
    - Assign the `outputs` parameter to the `outputs` field of the `opt_ctx`.
- **Output**: This function does not return any value; it modifies the `opt_ctx` in place.


---
### ggml\_opt\_alloc<!-- {{#callable:ggml_opt_alloc}} -->
The `ggml_opt_alloc` function allocates and prepares the necessary computational graph for optimization based on the current optimization context and whether a backward pass is required.
- **Inputs**:
    - `opt_ctx`: A pointer to a `ggml_opt_context_t` structure that holds the current state and configuration of the optimization process.
    - `backward`: A boolean flag indicating whether a backward pass is required for gradient computation.
- **Control Flow**:
    - Assert that the optimization context is not ready for evaluation using `GGML_ASSERT(!opt_ctx->eval_ready);`.
    - If the build type is `GGML_OPT_BUILD_TYPE_OPT`, the optimization period is greater than 1, and the current optimization index is 0, reset the gradient graph using `ggml_graph_reset(opt_ctx->gb_grad);`.
    - Determine the next build type based on the `backward` flag and the optimization index, setting it to either `GGML_OPT_BUILD_TYPE_OPT`, `GGML_OPT_BUILD_TYPE_GRAD`, or `GGML_OPT_BUILD_TYPE_FORWARD`.
    - If static graphs are not used, call `ggml_opt_build(opt_ctx);` to build the necessary computational graph.
    - Select the appropriate graph (`gf`, `gb_grad`, or `gb_opt`) based on the current build type.
    - Assert that the selected graph is not null using `GGML_ASSERT(graph);`.
    - If the selected graph is already allocated, set `eval_ready` to true and return.
    - Reset the backend scheduler using `ggml_backend_sched_reset(opt_ctx->backend_sched);`.
    - If static graphs are used, initialize a new context with the required memory size and duplicate the graph; otherwise, set `allocated_graph_copy` to the current graph.
    - Allocate the graph in the backend scheduler using `ggml_backend_sched_alloc_graph(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);`.
    - Set `allocated_graph` to the current graph and mark the context as ready for evaluation by setting `eval_ready` to true.
- **Output**: The function does not return a value; it modifies the `opt_ctx` to prepare it for evaluation.
- **Functions called**:
    - [`ggml_graph_reset`](ggml.c.driver.md#ggml_graph_reset)
    - [`ggml_opt_build`](#ggml_opt_build)
    - [`ggml_backend_sched_reset`](ggml-backend.cpp.driver.md#ggml_backend_sched_reset)
    - [`ggml_tensor_overhead`](ggml.c.driver.md#ggml_tensor_overhead)
    - [`ggml_graph_overhead_custom`](ggml.c.driver.md#ggml_graph_overhead_custom)
    - [`ggml_free`](ggml.c.driver.md#ggml_free)
    - [`ggml_init`](ggml.c.driver.md#ggml_init)
    - [`dup_graph`](#dup_graph)
    - [`ggml_backend_sched_alloc_graph`](ggml-backend.cpp.driver.md#ggml_backend_sched_alloc_graph)


---
### ggml\_opt\_eval<!-- {{#callable:ggml_opt_eval}} -->
The `ggml_opt_eval` function evaluates an optimization context by computing the current graph, updating iteration counters, and storing results in the provided result structure.
- **Inputs**:
    - `opt_ctx`: A `ggml_opt_context_t` structure representing the optimization context, containing information about the current state of the optimization process.
    - `result`: A `ggml_opt_result_t` structure where the results of the evaluation, such as loss and predictions, will be stored.
- **Control Flow**:
    - Assert that the optimization context is ready for evaluation using `GGML_ASSERT(opt_ctx->eval_ready)`.
    - Check if the allocated graph is the optimization graph (`opt_ctx->allocated_graph == opt_ctx->gb_opt`).
    - Retrieve optimizer parameters using `opt_ctx->get_opt_pars` and assert their validity.
    - Calculate `beta1h` and `beta2h` for the AdamW optimizer using the current iteration count.
    - Store optimizer parameters and calculated values in `adamw_par_data`.
    - Compute the graph using [`ggml_backend_sched_graph_compute`](ggml-backend.cpp.driver.md#ggml_backend_sched_graph_compute) with the allocated graph copy.
    - Update the iteration counter and optimization index based on the current graph.
    - If the context does not use static graphs, reset various graph-related pointers to `nullptr`.
    - Set `opt_ctx->eval_ready` to `false` to indicate the evaluation is complete.
    - If `result` is not `nullptr`, update its fields with the current loss, predictions, and number of correct predictions.
- **Output**: The function does not return a value but updates the `result` structure with the current loss, predictions, and number of correct predictions if it is provided.
- **Functions called**:
    - [`ggml_get_data_f32`](ggml.c.driver.md#ggml_get_data_f32)
    - [`ggml_backend_sched_graph_compute`](ggml-backend.cpp.driver.md#ggml_backend_sched_graph_compute)
    - [`ggml_is_scalar`](ggml.c.driver.md#ggml_is_scalar)
    - [`ggml_backend_tensor_get`](ggml-backend.cpp.driver.md#ggml_backend_tensor_get)
    - [`ggml_nbytes`](ggml.c.driver.md#ggml_nbytes)


---
### ggml\_opt\_epoch<!-- {{#callable:ggml_opt_epoch}} -->
The `ggml_opt_epoch` function performs a single epoch of training and evaluation on a dataset using a given optimization context, splitting the data into training and evaluation batches and invoking callbacks for each batch.
- **Inputs**:
    - `opt_ctx`: The optimization context containing the configuration and state for the optimization process.
    - `dataset`: The dataset to be used for training and evaluation, containing input data and labels.
    - `result_train`: The result object to store training results, including loss and predictions.
    - `result_eval`: The result object to store evaluation results, including loss and predictions.
    - `idata_split`: The index at which to split the dataset into training and evaluation parts; if negative, the entire dataset is used for training.
    - `callback_train`: A callback function to be called after each training batch is processed.
    - `callback_eval`: A callback function to be called after each evaluation batch is processed.
- **Control Flow**:
    - Assert that static graphs are required for the function to proceed.
    - Retrieve input and label tensors from the optimization context and data tensor from the dataset.
    - Calculate the number of data points and batches based on the input tensor dimensions.
    - Determine the split index for training and evaluation batches, adjusting if necessary.
    - Iterate over the training batches, allocating resources, fetching batches, evaluating, and invoking the training callback if provided.
    - Iterate over the evaluation batches, allocating resources, fetching batches, evaluating, and invoking the evaluation callback if provided.
- **Output**: The function does not return a value; it modifies the `result_train` and `result_eval` objects with the results of the training and evaluation processes.
- **Functions called**:
    - [`ggml_opt_static_graphs`](#ggml_opt_static_graphs)
    - [`ggml_opt_inputs`](#ggml_opt_inputs)
    - [`ggml_opt_labels`](#ggml_opt_labels)
    - [`ggml_opt_dataset_data`](#ggml_opt_dataset_data)
    - [`ggml_opt_alloc`](#ggml_opt_alloc)
    - [`ggml_opt_dataset_get_batch`](#ggml_opt_dataset_get_batch)
    - [`ggml_opt_eval`](#ggml_opt_eval)


---
### ggml\_opt\_epoch\_callback\_progress\_bar<!-- {{#callable:ggml_opt_epoch_callback_progress_bar}} -->
The `ggml_opt_epoch_callback_progress_bar` function displays a progress bar and performance metrics for a training or validation epoch in a machine learning optimization process.
- **Inputs**:
    - `train`: A boolean indicating whether the current epoch is for training (`true`) or validation (`false`).
    - `opt_ctx`: The optimization context containing information about the current optimization process.
    - `dataset`: The dataset being used for the optimization process.
    - `result`: The result object that stores the outcomes of the optimization, such as loss and accuracy.
    - `ibatch`: The current batch index within the epoch.
    - `ibatch_max`: The maximum number of batches in the epoch.
    - `t_start_us`: The start time of the epoch in microseconds.
- **Control Flow**:
    - Prints the training or validation label based on the `train` flag.
    - Calculates and prints a progress bar using Unicode characters to represent the fill level of each segment.
    - Retrieves the batch size from the optimization context and calculates the current and maximum data indices.
    - Retrieves and prints the loss and accuracy values along with their uncertainties from the result object.
    - Calculates the elapsed time since the start of the epoch and formats it into hours, minutes, and seconds.
    - Estimates the remaining time (ETA) for the epoch based on the elapsed time and remaining batches.
    - Prints the formatted progress bar, data indices, loss, accuracy, elapsed time, and ETA to the standard error stream.
    - If the current batch is the last one, prints a newline character to complete the progress bar display.
    - Flushes the standard error stream to ensure the output is displayed immediately.
- **Output**: The function does not return a value; it outputs the progress bar and metrics to the standard error stream.
- **Functions called**:
    - [`ggml_opt_inputs`](#ggml_opt_inputs)
    - [`ggml_opt_result_loss`](#ggml_opt_result_loss)
    - [`ggml_opt_result_accuracy`](#ggml_opt_result_accuracy)


---
### ggml\_opt\_fit<!-- {{#callable:ggml_opt_fit}} -->
The `ggml_opt_fit` function performs training of a model using a specified optimization algorithm over a given dataset, with options for validation split and silent mode.
- **Inputs**:
    - `backend_sched`: A scheduling object for backend operations, determining how computations are scheduled.
    - `ctx_compute`: A pointer to the computation context used for executing operations.
    - `inputs`: A tensor representing the input data for the model.
    - `outputs`: A tensor representing the output data for the model.
    - `dataset`: The dataset object containing the data and labels for training and validation.
    - `loss_type`: An enumeration specifying the type of loss function to use during optimization.
    - `get_opt_pars`: A function pointer to retrieve optimizer parameters, which may vary per epoch.
    - `nepoch`: The number of epochs to train the model.
    - `nbatch_logical`: The logical batch size for training, determining how data is split into batches.
    - `val_split`: A float value between 0 and 1 indicating the proportion of data to use for validation.
    - `silent`: A boolean flag indicating whether to suppress output during training.
- **Control Flow**:
    - Initialize timing and start time for the training process.
    - Calculate the number of data points and physical batch size from the dataset and input tensor.
    - Assert that the data can be evenly divided into logical and physical batches.
    - Calculate the optimization period and number of logical batches.
    - Determine the index for splitting data into training and validation sets based on the validation split ratio.
    - Initialize optimization parameters and context using the provided backend scheduler and loss type.
    - Shuffle the dataset if the logical batch size is smaller than the total data size.
    - Initialize result objects for training and validation results.
    - Set up an epoch callback function if not in silent mode.
    - Iterate over the number of epochs, shuffling the dataset before each epoch if necessary.
    - Reset the training and validation results at the start of each epoch.
    - Print epoch progress if not in silent mode.
    - Perform training and validation for each epoch using the [`ggml_opt_epoch`](#ggml_opt_epoch) function.
    - Print the total training time if not in silent mode.
    - Free the optimization context and result objects after training.
- **Output**: The function does not return a value; it performs training and updates the model parameters in place.
- **Functions called**:
    - [`ggml_opt_dataset_data`](#ggml_opt_dataset_data)
    - [`ggml_opt_default_params`](#ggml_opt_default_params)
    - [`ggml_opt_init`](#ggml_opt_init)
    - [`ggml_opt_dataset_shuffle`](#ggml_opt_dataset_shuffle)
    - [`ggml_opt_result_init`](#ggml_opt_result_init)
    - [`ggml_opt_result_reset`](#ggml_opt_result_reset)
    - [`ggml_opt_epoch`](#ggml_opt_epoch)
    - [`ggml_opt_free`](#ggml_opt_free)
    - [`ggml_opt_result_free`](#ggml_opt_result_free)


