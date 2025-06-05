# Purpose
This C++ source code file is designed to perform Principal Component Analysis (PCA) using a power iteration method, leveraging the GGML library for tensor operations and potentially utilizing GPU acceleration through CUDA or Metal backends. The code is structured around a namespace `PCA`, which encapsulates the functionality related to PCA computations. It defines several key structures, including `pca_params` for input parameters, `pca_result` for storing results of each iteration, and [`pca_model`](#pca_modelpca_model) for managing the computational model and resources. The [`pca_model`](#pca_modelpca_model) class initializes the computational context, sets up tensors for input data, and manages backend resources, supporting both CPU and GPU execution environments.

The file includes functions to build computational graphs ([`build_graph_piter`](#PCAbuild_graph_piter)), execute power iterations ([`power_iteration`](#PCApower_iteration)), and run the overall PCA process ([`run_pca`](#PCArun_pca)). The code is designed to handle multiple layers of input data, iterating over each to compute the principal components. It uses a power iteration method to iteratively refine the eigenvectors, with the ability to adjust parameters such as the number of iterations, batch size, and tolerance for convergence. The code is modular, allowing for easy integration into larger systems, and it provides a clear API for performing PCA on tensor data, making it suitable for applications in machine learning and data analysis where dimensionality reduction is required.
# Imports and Dependencies

---
- `common.h`
- `llama.h`
- `ggml.h`
- `ggml-cuda.h`
- `ggml-metal.h`
- `cstdio`
- `ctime`
- `random`
- `string`
- `vector`


# Data Structures

---
### pca\_params<!-- {{#data_structure:PCA::pca_params}} -->
- **Type**: `struct`
- **Members**:
    - `n_threads`: Specifies the number of threads to use, defaulting to 1.
    - `n_batch`: Indicates the number of iterations to perform in one batch, with a default value of 20.
    - `n_iterations`: Defines the total number of iterations to execute, defaulting to 1000.
    - `tolerance`: Sets the tolerance level for convergence, with a default value of 1e-7.
    - `i_layer`: Used for debugging, indicating the current layer index, defaulting to 0.
    - `n_layers`: Used for debugging, representing the total number of layers, defaulting to 0.
- **Description**: The `pca_params` struct is designed to encapsulate the parameters required for performing Principal Component Analysis (PCA) computations. It includes settings for the number of threads (`n_threads`), batch size (`n_batch`), total iterations (`n_iterations`), and convergence tolerance (`tolerance`). Additionally, it contains debugging fields `i_layer` and `n_layers` to track the current layer and total layers during the PCA process. This struct provides a configurable interface for controlling the execution of PCA, allowing for adjustments in computational resources and precision.


---
### pca\_result<!-- {{#data_structure:PCA::pca_result}} -->
- **Type**: `struct`
- **Members**:
    - `calculated_square`: A pointer to a ggml_tensor representing the calculated square matrix, initialized to NULL.
    - `eigenvectors`: A vector of pointers to ggml_tensor objects representing the eigenvectors.
    - `distances`: A vector of floats representing the distances calculated during PCA iterations.
- **Description**: The `pca_result` struct is designed to store the results of a Principal Component Analysis (PCA) computation. It includes a pointer to a tensor for the calculated square matrix, a vector of tensor pointers for the eigenvectors, and a vector of floats for the distances between successive eigenvectors during the iterative PCA process. This structure is used to capture and organize the output of PCA computations, facilitating further analysis or processing.


---
### pca\_model<!-- {{#data_structure:PCA::pca_model}} -->
- **Type**: `struct`
- **Members**:
    - `backend`: Represents the backend used for computation, initialized to NULL.
    - `buffer`: Holds the buffer associated with the backend.
    - `ctx`: Context for computing the graph on the target device.
    - `ctx_host`: Host context for storing results.
    - `dev_input`: Tensor on the target device representing the input data.
    - `dev_square`: Tensor on the target device representing the square of the input data.
    - `dev_eigenvector`: Tensor on the target device representing the eigenvector.
- **Description**: The `pca_model` struct is designed to facilitate Principal Component Analysis (PCA) computations on various hardware backends, such as CUDA or CPU. It initializes the necessary contexts and tensors for performing PCA, including input data, squared data, and eigenvectors. The struct manages memory allocation and initialization of these tensors, and it supports backend-specific operations to optimize performance on the target device. The constructor sets up the backend and initializes the tensors, while the destructor ensures proper cleanup of resources.
- **Member Functions**:
    - [`PCA::pca_model::pca_model`](#pca_modelpca_model)
    - [`PCA::pca_model::~pca_model`](#pca_modelpca_model)

**Methods**

---
#### pca\_model::pca\_model<!-- {{#callable:PCA::pca_model::pca_model}} -->
The `pca_model` constructor initializes a PCA model by setting up the computational backend, creating necessary tensors, and initializing an eigenvector with a random normalized vector.
- **Inputs**:
    - `t_input`: A pointer to a `ggml_tensor` structure representing the input data for the PCA model, with dimensions [n_samples, n_embd].
- **Control Flow**:
    - Check if CUDA is enabled and attempt to initialize the CUDA backend; if unsuccessful, print an error message.
    - If CUDA is not enabled or initialization fails, attempt to initialize the CPU backend.
    - Initialize a GGML context with parameters for memory size, buffer, and allocation settings.
    - Extract the number of samples and embedding dimensions from the input tensor.
    - Create three tensors on the target device: `dev_input`, `dev_square`, and `dev_eigenvector`, with appropriate dimensions and types.
    - Set names for the created tensors for identification purposes.
    - Allocate memory for the context tensors using the backend.
    - Copy the input tensor data to the `dev_input` tensor on the backend.
    - Generate a random vector, normalize it, and set it as the initial eigenvector in the `dev_eigenvector` tensor.
- **Output**: The function does not return a value; it initializes the PCA model's internal state and tensors.
- **Functions called**:
    - [`ggml_tensor_overhead`](../../ggml/src/ggml.c.driver.md#ggml_tensor_overhead)
    - [`ggml_init`](../../ggml/src/ggml.c.driver.md#ggml_init)
    - [`ggml_new_tensor_2d`](../../ggml/src/ggml.c.driver.md#ggml_new_tensor_2d)
    - [`ggml_new_tensor_1d`](../../ggml/src/ggml.c.driver.md#ggml_new_tensor_1d)
    - [`ggml_set_name`](../../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_backend_alloc_ctx_tensors`](../../ggml/src/ggml-alloc.c.driver.md#ggml_backend_alloc_ctx_tensors)
    - [`ggml_backend_tensor_set`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set)
    - [`ggml_nbytes`](../../ggml/src/ggml.c.driver.md#ggml_nbytes)
    - [`ggml_nelements`](../../ggml/src/ggml.c.driver.md#ggml_nelements)
- **See also**: [`PCA::pca_model`](#PCApca_model)  (Data Structure)


---
#### pca\_model::\~pca\_model<!-- {{#callable:PCA::pca_model::~pca_model}} -->
The destructor `~pca_model` releases resources by freeing the context, buffer, and backend associated with a PCA model.
- **Inputs**: None
- **Control Flow**:
    - Call `ggml_free(ctx)` to release the context used for computing the graph on the target device.
    - Call `ggml_backend_buffer_free(buffer)` to free the buffer associated with the backend.
    - Call `ggml_backend_free(backend)` to release the backend resources.
- **Output**: This function does not return any value; it is a destructor that cleans up resources.
- **Functions called**:
    - [`ggml_free`](../../ggml/src/ggml.c.driver.md#ggml_free)
    - [`ggml_backend_buffer_free`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_free)
    - [`ggml_backend_free`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_free)
- **See also**: [`PCA::pca_model`](#PCApca_model)  (Data Structure)



# Functions

---
### print\_debug\_tensor<!-- {{#callable:print_debug_tensor}} -->
The `print_debug_tensor` function prints the name, type, and dimensions of a given tensor, and optionally prints a portion of its data.
- **Inputs**:
    - `t`: A pointer to a `ggml_tensor` structure representing the tensor to be printed.
    - `with_data`: A boolean flag indicating whether to print the tensor's data (default is true).
- **Control Flow**:
    - Prints the function name, tensor name, tensor type, and its dimensions using `printf`.
    - Checks if `with_data` is false, and if so, returns immediately without printing data.
    - If `with_data` is true, prints the first few elements of the tensor's data up to `DEBUG_POS` using a loop and [`ggml_get_f32_nd`](../../ggml/src/ggml-cpu/ggml-cpu.c.driver.md#ggml_get_f32_nd).
    - Ends the data printout with an ellipsis to indicate continuation.
- **Output**: The function does not return any value; it outputs information to the standard output (console).
- **Functions called**:
    - [`ggml_type_name`](../../ggml/src/ggml.c.driver.md#ggml_type_name)
    - [`ggml_get_f32_nd`](../../ggml/src/ggml-cpu/ggml-cpu.c.driver.md#ggml_get_f32_nd)


---
### build\_graph\_piter<!-- {{#callable:PCA::build_graph_piter}} -->
The `build_graph_piter` function constructs a computational graph for performing power iteration on a PCA model, optionally calculating a square matrix from the input data.
- **Inputs**:
    - `params`: A `pca_params` structure containing parameters for PCA computations, including the number of batches (`n_batch`).
    - `model`: A `pca_model` object containing the PCA model data, including input tensors and eigenvectors.
    - `calc_square`: A boolean flag indicating whether to calculate the square matrix from the input data (`model.dev_input`).
- **Control Flow**:
    - Assert that the number of batches (`params.n_batch`) is greater than zero.
    - Initialize a buffer and context for building the graph with a fixed buffer size.
    - Create a new graph (`gf`) within the initialized context (`ctx0`).
    - If `calc_square` is true, compute the square matrix of the input data and assign it to `tmp_square`.
    - Iterate over the number of batches (`params.n_batch`):
    -   - Compute `b_tensor` as the product of the input square matrix and the old eigenvector.
    -   - Normalize `b_tensor` by dividing it by its row-wise square root sum.
    -   - Calculate the distance between the new and old eigenvectors using a custom subtraction method.
    -   - Update `old_eigen` to the current `b_tensor`.
    -   - Expand the graph with the distance operation node.
    - Free the temporary context used for building the graph.
    - Return the constructed graph (`gf`).
- **Output**: A pointer to a `ggml_cgraph` structure representing the constructed computational graph for the PCA power iteration.
- **Functions called**:
    - [`ggml_tensor_overhead`](../../ggml/src/ggml.c.driver.md#ggml_tensor_overhead)
    - [`ggml_graph_overhead`](../../ggml/src/ggml.c.driver.md#ggml_graph_overhead)
    - [`ggml_init`](../../ggml/src/ggml.c.driver.md#ggml_init)
    - [`ggml_new_graph`](../../ggml/src/ggml.c.driver.md#ggml_new_graph)
    - [`ggml_mul_mat`](../../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_set_name`](../../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_div_inplace`](../../ggml/src/ggml.c.driver.md#ggml_div_inplace)
    - [`ggml_sqrt_inplace`](../../ggml/src/ggml.c.driver.md#ggml_sqrt_inplace)
    - [`ggml_sum_rows`](../../ggml/src/ggml.c.driver.md#ggml_sum_rows)
    - [`ggml_sqr`](../../ggml/src/ggml.c.driver.md#ggml_sqr)
    - [`ggml_format_name`](../../ggml/src/ggml.c.driver.md#ggml_format_name)
    - [`ggml_add`](../../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_scale`](../../ggml/src/ggml.c.driver.md#ggml_scale)
    - [`ggml_sqr_inplace`](../../ggml/src/ggml.c.driver.md#ggml_sqr_inplace)
    - [`ggml_build_forward_expand`](../../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
    - [`ggml_free`](../../ggml/src/ggml.c.driver.md#ggml_free)


---
### compute\_piter<!-- {{#callable:PCA::compute_piter}} -->
The `compute_piter` function executes a power iteration process on a PCA model graph to compute eigenvectors and distances, storing the results in a `pca_result` structure.
- **Inputs**:
    - `params`: A `pca_params` structure containing parameters for PCA computations, such as the number of threads, batch size, and tolerance.
    - `model`: A `pca_model` object representing the PCA model, including backend information and device tensors.
    - `gf`: A pointer to a `ggml_cgraph` structure representing the computation graph for the PCA process.
    - `allocr`: A `ggml_gallocr_t` allocator used for allocating tensors within the graph.
    - `result`: A `pca_result` structure where the computed eigenvectors, distances, and optionally the calculated square matrix will be stored.
- **Control Flow**:
    - Allocate tensors for the computation graph using the provided allocator.
    - Check if the backend is CPU and set the number of threads accordingly.
    - Compute the graph using the backend and store the result status.
    - If the computation is successful, initialize the result structure by clearing and resizing the eigenvectors and distances vectors.
    - Iterate over the nodes in the computation graph to extract eigenvectors and distances based on node names, storing them in the result structure.
    - Identify and store the 'tmp_square' node if it exists.
- **Output**: Returns a `ggml_status` indicating the success or failure of the graph computation.
- **Functions called**:
    - [`ggml_gallocr_alloc_graph`](../../ggml/src/ggml-alloc.c.driver.md#ggml_gallocr_alloc_graph)
    - [`ggml_backend_is_cpu`](../../ggml/src/ggml-cpu/ggml-cpu.cpp.driver.md#ggml_backend_is_cpu)
    - [`ggml_backend_cpu_set_n_threads`](../../ggml/src/ggml-cpu/ggml-cpu.cpp.driver.md#ggml_backend_cpu_set_n_threads)
    - [`ggml_backend_graph_compute`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_graph_compute)
    - [`ggml_graph_n_nodes`](../../ggml/src/ggml.c.driver.md#ggml_graph_n_nodes)
    - [`ggml_graph_node`](../../ggml/src/ggml.c.driver.md#ggml_graph_node)
    - [`ggml_backend_tensor_get`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_get)


---
### power\_iteration<!-- {{#callable:PCA::power_iteration}} -->
The `power_iteration` function performs the power iteration method to compute the principal eigenvector of a matrix represented by the input tensor.
- **Inputs**:
    - `params`: A `pca_params` structure containing parameters for PCA computations, including number of threads, batch size, number of iterations, and tolerance.
    - `input`: A `ggml_tensor` representing the input matrix with shape [n_samples, n_embd].
    - `output`: A `ggml_tensor` where the resulting principal eigenvector will be stored.
- **Control Flow**:
    - Initialize a PCA model using the input tensor.
    - Create a new allocator for graph computations based on the backend type.
    - Calculate the number of iterations by dividing total iterations by batch size.
    - Iterate over the number of iterations, building and computing the graph for each batch.
    - In the first iteration, calculate the square of the input matrix if needed.
    - For each batch, compute the power iteration and check if the distance is below the tolerance to break early.
    - Copy the last eigenvector to be used as input for the next iteration.
    - After all iterations, copy the last eigenvector to the output tensor.
    - Free the allocated resources.
- **Output**: The function outputs the principal eigenvector of the input matrix into the provided output tensor.
- **Functions called**:
    - [`PCA::build_graph_piter`](#PCAbuild_graph_piter)
    - [`PCA::compute_piter`](#PCAcompute_piter)
    - [`ggml_backend_tensor_copy`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_copy)
    - [`ggml_backend_tensor_get`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_get)
    - [`ggml_nbytes`](../../ggml/src/ggml.c.driver.md#ggml_nbytes)
    - [`ggml_gallocr_free`](../../ggml/src/ggml-alloc.c.driver.md#ggml_gallocr_free)


---
### run\_pca<!-- {{#callable:PCA::run_pca}} -->
The `run_pca` function performs Principal Component Analysis (PCA) on a set of input tensors using power iteration to compute the principal components and stores the results in output tensors.
- **Inputs**:
    - `params`: A reference to a `pca_params` structure containing parameters for PCA computation such as number of threads, batch size, number of iterations, and tolerance.
    - `v_input`: A constant reference to a vector of pointers to `ggml_tensor` structures, where each tensor represents input data with shape [n_samples, n_embd].
    - `v_output`: A constant reference to a vector of pointers to `ggml_tensor` structures, where each tensor is used to store the output principal components.
- **Control Flow**:
    - Prints a message indicating the start of PCA computation.
    - Iterates over each input tensor in `v_input`.
    - For each input tensor, prepares the corresponding output tensor in `v_output` by formatting its name.
    - Sets the current layer index and total number of layers in the `params` structure.
    - Calls the [`power_iteration`](#PCApower_iteration) function to compute the principal component for the current input tensor and store it in the output tensor.
    - Prints a message indicating the completion of PCA computation for the current layer.
- **Output**: The function does not return a value but modifies the `v_output` tensors to store the computed principal components for each corresponding input tensor.
- **Functions called**:
    - [`ggml_format_name`](../../ggml/src/ggml.c.driver.md#ggml_format_name)
    - [`PCA::power_iteration`](#PCApower_iteration)


