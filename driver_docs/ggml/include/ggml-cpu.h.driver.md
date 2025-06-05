# Purpose
This C header file provides a comprehensive interface for managing and executing computational graphs using the GGML library, with a focus on optimizing performance on NUMA (Non-Uniform Memory Access) systems. It defines a variety of functions and structures that facilitate the creation, manipulation, and execution of computational graphs, which are essential for tasks such as machine learning and data processing. The file includes definitions for handling NUMA strategies, thread pool management, and tensor operations, which are crucial for parallel computation and efficient resource utilization. The `ggml_cplan` structure is central to preparing compute plans for graph execution, while the `ggml_numa_strategy` enumeration and associated functions enable performance tuning on systems with multiple NUMA nodes.

Additionally, the file provides a set of APIs for creating and manipulating tensors, including functions for setting and retrieving integer and floating-point values in various dimensions. It also includes functions for initializing and managing thread pools, which are vital for parallel processing. The file further offers system information functions to detect CPU capabilities, such as support for specific instruction sets, which can be leveraged to optimize computations. The backend functions, particularly those related to CPU initialization and configuration, underscore the file's role in setting up and managing the computational environment for executing GGML-based tasks. Overall, this header file serves as a critical component of the GGML library, providing the necessary interfaces and utilities for efficient graph computation and resource management.
# Imports and Dependencies

---
- `ggml.h`
- `ggml-backend.h`


# Data Structures

---
### ggml\_cplan
- **Type**: `struct`
- **Members**:
    - `work_size`: Size of the work buffer, calculated by `ggml_graph_plan()`.
    - `work_data`: Pointer to the work buffer, which must be allocated by the caller before calling `ggml_graph_compute()`.
    - `n_threads`: Number of threads to be used for computation.
    - `threadpool`: Pointer to a `ggml_threadpool` structure for managing threads.
    - `abort_callback`: Callback function to abort `ggml_graph_compute` if it returns true.
    - `abort_callback_data`: Pointer to data passed to the abort callback function.
- **Description**: The `ggml_cplan` structure is designed to manage the computation plan for the `ggml_graph_compute()` function. It includes a work buffer whose size is determined by `ggml_graph_plan()`, and which must be allocated by the caller. The structure also manages threading through a specified number of threads and an optional thread pool. Additionally, it supports an abort mechanism via a callback function, allowing the computation to be halted based on custom conditions.


---
### ggml\_numa\_strategy
- **Type**: `enum`
- **Members**:
    - `GGML_NUMA_STRATEGY_DISABLED`: Represents a strategy where NUMA is disabled.
    - `GGML_NUMA_STRATEGY_DISTRIBUTE`: Represents a strategy where resources are distributed across NUMA nodes.
    - `GGML_NUMA_STRATEGY_ISOLATE`: Represents a strategy where resources are isolated to specific NUMA nodes.
    - `GGML_NUMA_STRATEGY_NUMACTL`: Represents a strategy that uses the numactl tool for NUMA management.
    - `GGML_NUMA_STRATEGY_MIRROR`: Represents a strategy where resources are mirrored across NUMA nodes.
    - `GGML_NUMA_STRATEGY_COUNT`: Represents the count of NUMA strategies available.
- **Description**: The `ggml_numa_strategy` enum defines various strategies for managing Non-Uniform Memory Access (NUMA) configurations in a system. It provides options to disable NUMA, distribute resources across nodes, isolate resources to specific nodes, use the numactl tool for management, or mirror resources across nodes. This enum is used to optimize performance on systems with multiple NUMA nodes by selecting an appropriate strategy.


---
### ggml\_type\_traits\_cpu
- **Type**: `struct`
- **Members**:
    - `from_float`: A function pointer for converting from a float type.
    - `vec_dot`: A function pointer for performing vector dot product operations.
    - `vec_dot_type`: An enumeration indicating the type of vector dot product.
    - `nrows`: An integer representing the number of rows to process simultaneously.
- **Description**: The `ggml_type_traits_cpu` structure is designed to encapsulate CPU-specific traits and operations for handling different data types in the GGML library. It includes function pointers for converting from float types and performing vector dot product operations, as well as an enumeration to specify the type of vector dot product. Additionally, it holds an integer to define the number of rows that can be processed simultaneously, which is crucial for optimizing performance in parallel computing environments.


# Function Declarations (Public API)

---
### ggml\_numa\_init<!-- {{#callable_declaration:ggml_numa_init}} -->
Initializes NUMA support for the system.
- **Description**: This function should be called once to set up NUMA (Non-Uniform Memory Access) support, which can enhance performance on systems with multiple NUMA nodes. It configures the NUMA strategy based on the provided flag and detects the number of NUMA nodes and CPUs available on the system. It is important to call this function before any operations that depend on NUMA awareness. If NUMA is already initialized, the function will log a warning and return without making any changes. The function may also log a warning if NUMA balancing is enabled, as this can negatively impact performance.
- **Inputs**:
    - `numa_flag`: Specifies the NUMA strategy to use. Valid values are defined in the `ggml_numa_strategy` enum, which includes options like `GGML_NUMA_STRATEGY_DISABLED`, `GGML_NUMA_STRATEGY_DISTRIBUTE`, and others. The caller retains ownership of this parameter, and it must not be null. If an invalid value is provided, the function will not initialize NUMA and will return without making changes.
- **Output**: The function does not return a value and does not mutate any inputs.
- **See also**: [`ggml_numa_init`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_numa_init)  (Implementation)


---
### ggml\_is\_numa<!-- {{#callable_declaration:ggml_is_numa}} -->
Checks if the system has more than one NUMA node.
- **Description**: This function is used to determine if the system is configured with multiple Non-Uniform Memory Access (NUMA) nodes, which can affect performance and memory allocation strategies. It should be called after initializing the NUMA settings with `ggml_numa_init()`. The function returns a boolean value indicating whether the system has more than one NUMA node, which can be useful for optimizing memory access patterns in applications that are sensitive to memory locality.
- **Inputs**:
    - `none`: This function does not take any parameters.
- **Output**: Returns true if the system has more than one NUMA node; otherwise, it returns false.
- **See also**: [`ggml_is_numa`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_is_numa)  (Implementation)


---
### ggml\_new\_i32<!-- {{#callable_declaration:ggml_new_i32}} -->
Creates a new tensor of type int32.
- **Description**: This function is used to create a new tensor that holds a single 32-bit integer value. It must be called with a valid context that has been initialized, as the function relies on the context for memory allocation. The created tensor will contain the specified integer value. If the context is not properly initialized or if memory allocation fails, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `struct ggml_context` that must be valid and initialized. This context is used for memory management. Must not be null.
    - `value`: The 32-bit integer value to be stored in the new tensor. This can be any valid int32 value.
- **Output**: Returns a pointer to the newly created `struct ggml_tensor` containing the specified integer value. If the tensor creation fails, the return value is undefined.
- **See also**: [`ggml_new_i32`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_new_i32)  (Implementation)


---
### ggml\_new\_f32<!-- {{#callable_declaration:ggml_new_f32}} -->
Creates a new tensor of type float with a specified value.
- **Description**: This function is used to create a new tensor of type float (`GGML_TYPE_F32`) initialized with a specific value. It must be called with a valid context that has been properly initialized, as the function asserts that the context is not set to disallow memory allocation. The created tensor will have a single dimension with one element, and the specified float value will be set in that tensor. If the context is invalid or memory allocation fails, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that must be valid and initialized. This context is required for memory allocation and management. It must not be null.
    - `value`: A float value that will be assigned to the newly created tensor. This value can be any valid float.
- **Output**: Returns a pointer to the newly created `ggml_tensor` initialized with the specified float value. If the function fails to create the tensor, the return value is undefined.
- **See also**: [`ggml_new_f32`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_new_f32)  (Implementation)


---
### ggml\_set\_i32<!-- {{#callable_declaration:ggml_set_i32}} -->
Sets the value of a tensor to a specified 32-bit integer.
- **Description**: This function is used to assign a specific 32-bit integer value to all elements of a given tensor. It is essential to ensure that the tensor has been properly initialized and that its type is compatible with the operation. The function will handle various tensor types, including 8-bit, 16-bit, and 32-bit integers, as well as floating-point types, converting the integer value as necessary. If the tensor type is unsupported, the function will abort execution. It is important to note that the tensor must not be null when passed to this function.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure that represents the tensor to be modified. This parameter must not be null and must point to a valid tensor that has been initialized. The tensor's type must be compatible with the operation; otherwise, the function will abort.
    - `value`: An integer value of type `int32_t` that will be set for each element of the tensor. This value can be any valid 32-bit integer.
- **Output**: Returns a pointer to the modified `ggml_tensor` structure, allowing for method chaining or further operations on the tensor.
- **See also**: [`ggml_set_i32`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_set_i32)  (Implementation)


---
### ggml\_set\_f32<!-- {{#callable_declaration:ggml_set_f32}} -->
Sets all elements of a tensor to a specified float value.
- **Description**: This function is used to assign a specific floating-point value to all elements of a given tensor. It is essential to ensure that the tensor has been properly initialized and is of a compatible type before calling this function. The function will handle various tensor types, including integer and floating-point representations, and will convert the float value as necessary. If the tensor type is unsupported, the function will abort execution. It is important to note that this function modifies the tensor in place.
- **Inputs**:
    - `tensor`: A pointer to the `ggml_tensor` structure that represents the tensor to be modified. This pointer must not be null and must point to a valid tensor that has been initialized. The tensor's type must be compatible with the float value being set; otherwise, the function will abort.
    - `value`: The float value to set for all elements of the tensor. This value can be any valid floating-point number. There are no specific ownership expectations for this parameter.
- **Output**: Returns a pointer to the modified `ggml_tensor` structure, allowing for method chaining or further operations on the tensor.
- **See also**: [`ggml_set_f32`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_set_f32)  (Implementation)


---
### ggml\_get\_i32\_1d<!-- {{#callable_declaration:ggml_get_i32_1d}} -->
Retrieves a 32-bit integer from a one-dimensional tensor.
- **Description**: This function is used to access the value at a specified index in a one-dimensional tensor. It is important to ensure that the tensor is properly initialized and that the index is within the valid range of the tensor's dimensions. If the tensor is not contiguous, the function will unravel the index to access the appropriate element. The function supports various tensor data types, including 8-bit, 16-bit, and 32-bit integers, as well as floating-point types, converting them to 32-bit integers as necessary. Calling this function with an invalid index may lead to undefined behavior.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor from which to retrieve the integer. Must not be null.
    - `i`: An integer index specifying the position of the element to retrieve. Must be within the bounds of the tensor's size; otherwise, behavior is undefined.
- **Output**: Returns the 32-bit integer value located at the specified index in the tensor.
- **See also**: [`ggml_get_i32_1d`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_get_i32_1d)  (Implementation)


---
### ggml\_set\_i32\_1d<!-- {{#callable_declaration:ggml_set_i32_1d}} -->
Sets a 1D integer value in a tensor.
- **Description**: This function is used to assign a specific integer value to a specified index of a 1D tensor. It is essential to ensure that the tensor is properly initialized and that the index provided is within the valid range of the tensor's dimensions. If the tensor is not contiguous, the function will unravel the index to set the value correctly. The function will assert the tensor's data type and size to ensure that the value being set is compatible with the tensor's format, and it will abort if an unsupported type is encountered.
- **Inputs**:
    - `tensor`: A pointer to the `ggml_tensor` structure representing the tensor in which the value will be set. Must not be null and should be properly initialized.
    - `i`: An integer representing the index in the tensor where the value will be set. This index must be within the bounds of the tensor's dimensions.
    - `value`: The integer value to be set at the specified index. This value will be stored in the tensor's data.
- **Output**: None
- **See also**: [`ggml_set_i32_1d`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_set_i32_1d)  (Implementation)


---
### ggml\_get\_i32\_nd<!-- {{#callable_declaration:ggml_get_i32_nd}} -->
Retrieves a 32-bit integer from a multi-dimensional tensor.
- **Description**: This function is used to access a specific element from a multi-dimensional tensor represented by the `ggml_tensor` structure. It requires the tensor to be properly initialized and allocated before calling. The indices provided must be within the bounds of the tensor's dimensions; otherwise, the behavior is undefined. The function supports various tensor data types, and it will return the corresponding integer value based on the tensor's type. It is important to ensure that the tensor is not null before calling this function.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor from which to retrieve the value. Must not be null.
    - `i0`: The first index for accessing the tensor element. Must be within the bounds of the tensor's first dimension.
    - `i1`: The second index for accessing the tensor element. Must be within the bounds of the tensor's second dimension.
    - `i2`: The third index for accessing the tensor element. Must be within the bounds of the tensor's third dimension.
    - `i3`: The fourth index for accessing the tensor element. Must be within the bounds of the tensor's fourth dimension.
- **Output**: Returns the 32-bit integer value located at the specified indices in the tensor. If the tensor type is not compatible, the function will abort.
- **See also**: [`ggml_get_i32_nd`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_get_i32_nd)  (Implementation)


---
### ggml\_set\_i32\_nd<!-- {{#callable_declaration:ggml_set_i32_nd}} -->
Sets a value in a multi-dimensional tensor.
- **Description**: This function is used to assign a specified integer value to a specific location within a multi-dimensional tensor. It is essential to ensure that the `tensor` has been properly initialized and allocated before calling this function. The indices `i0`, `i1`, `i2`, and `i3` must be within the bounds of the tensor's dimensions; otherwise, the behavior is undefined. The function will handle different tensor data types, converting the integer value as necessary, but it is crucial to ensure that the tensor type is compatible with the value being set.
- **Inputs**:
    - `tensor`: A pointer to the `ggml_tensor` structure representing the tensor to be modified. Must not be null and must point to a valid tensor that has been initialized.
    - `i0`: The first index for the multi-dimensional tensor. Must be within the bounds of the tensor's first dimension.
    - `i1`: The second index for the multi-dimensional tensor. Must be within the bounds of the tensor's second dimension.
    - `i2`: The third index for the multi-dimensional tensor. Must be within the bounds of the tensor's third dimension.
    - `i3`: The fourth index for the multi-dimensional tensor. Must be within the bounds of the tensor's fourth dimension.
    - `value`: The integer value to set at the specified tensor location. This value will be converted to the appropriate type based on the tensor's data type.
- **Output**: None
- **See also**: [`ggml_set_i32_nd`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_set_i32_nd)  (Implementation)


---
### ggml\_get\_f32\_1d<!-- {{#callable_declaration:ggml_get_f32_1d}} -->
Retrieves a 32-bit floating point value from a 1D tensor.
- **Description**: This function is used to access a specific element from a 1D tensor, identified by its index. It is essential to ensure that the tensor is properly initialized and that the index provided is within the valid range of the tensor's dimensions. If the tensor is not contiguous, the function will handle the index unraveling internally. The function may return different types of values based on the tensor's data type, including conversions from other formats to 32-bit floating point. Calling this function with an invalid index may lead to undefined behavior.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor from which to retrieve the value. Must not be null.
    - `i`: An integer representing the index of the element to retrieve. It must be within the bounds of the tensor's size; otherwise, the behavior is undefined.
- **Output**: Returns the 32-bit floating point value located at the specified index in the tensor.
- **See also**: [`ggml_get_f32_1d`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_get_f32_1d)  (Implementation)


---
### ggml\_set\_f32\_1d<!-- {{#callable_declaration:ggml_set_f32_1d}} -->
Sets a float value in a 1D tensor.
- **Description**: This function is used to assign a specific float value to an element at a given index in a 1D tensor. It is important to ensure that the tensor is properly initialized and that the index is within the valid range of the tensor's dimensions. If the tensor is not contiguous, the function will convert the 1D index to multi-dimensional indices before setting the value. The function handles different tensor data types, including integer and floating-point types, and will abort if an unsupported type is encountered.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor in which the value will be set. Must not be null and should be properly initialized.
    - `i`: An integer representing the index of the element in the tensor to be set. Must be within the valid range of the tensor's dimensions; otherwise, behavior is undefined.
    - `value`: A float value to be assigned to the specified index in the tensor. This value will be stored in the tensor's data.
- **Output**: None
- **See also**: [`ggml_set_f32_1d`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_set_f32_1d)  (Implementation)


---
### ggml\_get\_f32\_nd<!-- {{#callable_declaration:ggml_get_f32_nd}} -->
Retrieves a 32-bit floating-point value from a multi-dimensional tensor.
- **Description**: This function is used to access a specific element in a multi-dimensional tensor represented by the `ggml_tensor` structure. It requires the caller to provide the tensor and the indices corresponding to the desired element's position in the tensor's dimensions. The function should be called only after the tensor has been properly initialized and populated with data. If the provided indices are out of bounds or if the tensor type is unsupported, the behavior is undefined, and the program may abort.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure that contains the data to be accessed. Must not be null and must be properly initialized.
    - `i0`: The index for the first dimension of the tensor. Must be within the bounds of the tensor's first dimension.
    - `i1`: The index for the second dimension of the tensor. Must be within the bounds of the tensor's second dimension.
    - `i2`: The index for the third dimension of the tensor. Must be within the bounds of the tensor's third dimension.
    - `i3`: The index for the fourth dimension of the tensor. Must be within the bounds of the tensor's fourth dimension.
- **Output**: Returns the 32-bit floating-point value located at the specified indices in the tensor. If the tensor type is not supported, the program may abort.
- **See also**: [`ggml_get_f32_nd`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_get_f32_nd)  (Implementation)


---
### ggml\_set\_f32\_nd<!-- {{#callable_declaration:ggml_set_f32_nd}} -->
Sets a value in a multi-dimensional tensor.
- **Description**: This function is used to assign a floating-point value to a specific element in a multi-dimensional tensor. It should be called after the tensor has been properly initialized and allocated. The indices provided must be within the bounds of the tensor's dimensions; otherwise, the behavior is undefined. The function supports various tensor data types, and the value will be converted accordingly if necessary. It is important to ensure that the tensor is not null before calling this function.
- **Inputs**:
    - `tensor`: A pointer to the `ggml_tensor` structure representing the tensor. Must not be null.
    - `i0`: The first index for the multi-dimensional tensor. Must be within the bounds of the tensor's first dimension.
    - `i1`: The second index for the multi-dimensional tensor. Must be within the bounds of the tensor's second dimension.
    - `i2`: The third index for the multi-dimensional tensor. Must be within the bounds of the tensor's third dimension.
    - `i3`: The fourth index for the multi-dimensional tensor. Must be within the bounds of the tensor's fourth dimension.
    - `value`: The floating-point value to set in the tensor. This value will be stored in the appropriate format based on the tensor's type.
- **Output**: None
- **See also**: [`ggml_set_f32_nd`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_set_f32_nd)  (Implementation)


---
### ggml\_threadpool\_new<!-- {{#callable_declaration:ggml_threadpool_new}} -->
Creates a new thread pool.
- **Description**: This function is used to create a new thread pool for managing concurrent tasks. It should be called when you need to perform parallel computations and want to manage threads efficiently. The caller must provide a valid `ggml_threadpool_params` structure, which contains the necessary parameters for configuring the thread pool. If the provided parameters are invalid or if memory allocation fails, the function may return a null pointer, indicating that the thread pool could not be created.
- **Inputs**:
    - `tpp`: A pointer to a `ggml_threadpool_params` structure that contains the configuration parameters for the thread pool. This parameter must not be null and should be properly initialized before calling the function. If the parameter is invalid or null, the function will not create a thread pool.
- **Output**: Returns a pointer to a newly created `ggml_threadpool` structure. If the creation fails, it returns null.
- **See also**: [`ggml_threadpool_new`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_threadpool_new)  (Implementation)


---
### ggml\_threadpool\_free<!-- {{#callable_declaration:ggml_threadpool_free}} -->
Frees the resources associated with a thread pool.
- **Description**: This function should be called to properly release all resources allocated for a `ggml_threadpool` instance. It is essential to ensure that the thread pool is no longer in use before calling this function, as it will stop all worker threads and wait for them to finish. If the provided `threadpool` pointer is null, the function will simply return without performing any action. After calling this function, the `threadpool` pointer should not be used, as it will have been freed.
- **Inputs**:
    - `threadpool`: A pointer to a `ggml_threadpool` structure that needs to be freed. Must not be null; if null, the function does nothing.
- **Output**: None
- **See also**: [`ggml_threadpool_free`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_threadpool_free)  (Implementation)


---
### ggml\_threadpool\_pause<!-- {{#callable_declaration:ggml_threadpool_pause}} -->
Pauses the execution of a thread pool.
- **Description**: This function is used to pause the execution of a thread pool, which can be useful when you need to temporarily halt processing without destroying the thread pool. It should be called when the thread pool is active and has been properly initialized. If the thread pool is already paused, calling this function will have no effect. It is important to ensure that the thread pool is not in use by any threads when pausing to avoid potential race conditions.
- **Inputs**:
    - `threadpool`: A pointer to a `struct ggml_threadpool` that represents the thread pool to be paused. This pointer must not be null and should point to a valid, initialized thread pool. If an invalid pointer is provided, the behavior is undefined.
- **Output**: None
- **See also**: [`ggml_threadpool_pause`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_threadpool_pause)  (Implementation)


---
### ggml\_threadpool\_resume<!-- {{#callable_declaration:ggml_threadpool_resume}} -->
Resumes a paused thread pool.
- **Description**: This function is intended to be called when a thread pool has been paused, allowing it to resume processing tasks. It should be invoked after the thread pool has been created and initialized, specifically after a call to `ggml_threadpool_pause` has been made. The function ensures that the thread pool is resumed safely, handling any necessary locking to maintain thread safety. If the thread pool is not paused, the function has no effect.
- **Inputs**:
    - `threadpool`: A pointer to a `struct ggml_threadpool` that represents the thread pool to be resumed. This pointer must not be null and should point to a valid thread pool that has been previously created and initialized. If the thread pool is not in a paused state, the function will simply return without making any changes.
- **Output**: Returns nothing and does not mutate any inputs.
- **See also**: [`ggml_threadpool_resume`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_threadpool_resume)  (Implementation)


---
### ggml\_graph\_plan<!-- {{#callable_declaration:ggml_graph_plan}} -->
Creates a computation plan for a graph.
- **Description**: This function prepares a computation plan for executing a graph defined by `cgraph`. It must be called before invoking `ggml_graph_compute()`. The number of threads can be specified, and if not provided or invalid, a default value will be used. The caller is responsible for allocating memory for the `work_data` buffer if the computed `work_size` is greater than zero. It is important to ensure that the `cgraph` is valid and properly initialized before calling this function.
- **Inputs**:
    - `cgraph`: A pointer to a `ggml_cgraph` structure representing the computation graph. Must not be null.
    - `n_threads`: An integer specifying the number of threads to use for computation. Must be greater than zero; otherwise, a default value will be used.
    - `threadpool`: A pointer to a `ggml_threadpool` structure for managing threads. Can be null, in which case a disposable threadpool will be created.
- **Output**: Returns a `ggml_cplan` structure containing the computed work size, thread count, and threadpool information. The `work_data` field will be null and must be allocated by the caller if `work_size` is greater than zero.
- **See also**: [`ggml_graph_plan`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_graph_plan)  (Implementation)


---
### ggml\_graph\_compute<!-- {{#callable_declaration:ggml_graph_compute}} -->
Computes the graph defined in the compute plan.
- **Description**: This function is used to execute a computation graph that has been prepared using a compute plan. It must be called after the `ggml_graph_plan()` function, which sets up the necessary parameters and allocates memory for the work buffer if required. The caller must ensure that the `cplan` parameter is valid and that the number of threads specified is greater than zero. If the `work_size` in the plan is greater than zero, the caller must allocate memory for `work_data` before invoking this function. The function may create a disposable thread pool if none is provided, and it will clean up this thread pool after execution. It is important to handle any potential errors indicated by the return status.
- **Inputs**:
    - `cgraph`: A pointer to the computation graph structure that defines the operations to be performed. Must not be null.
    - `cplan`: A pointer to the compute plan structure that contains the configuration for the computation. Must not be null and must have a positive number of threads. If `work_size` is greater than zero, `work_data` must be allocated by the caller.
- **Output**: Returns a status code indicating the success or failure of the computation. A successful execution will return `GGML_STATUS_SUCCESS`, while other values indicate different error conditions.
- **See also**: [`ggml_graph_compute`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_graph_compute)  (Implementation)


---
### ggml\_graph\_compute\_with\_ctx<!-- {{#callable_declaration:ggml_graph_compute_with_ctx}} -->
Computes a graph using a specified context.
- **Description**: This function is intended for executing a computation graph within a specified context. It should be called after preparing the computation graph with `ggml_graph_plan()`, which calculates the necessary work size. The caller must ensure that the context has sufficient memory allocated for the work data, as this function will utilize that memory for computation. It is important to note that if the context does not have enough memory, the function may fail. Additionally, the number of threads to be used for computation can be specified, allowing for parallel execution.
- **Inputs**:
    - `ctx`: A pointer to a `struct ggml_context` that represents the context in which the computation will occur. This must not be null and must have sufficient memory allocated for the work data.
    - `cgraph`: A pointer to a `struct ggml_cgraph` that represents the computation graph to be executed. This must not be null.
    - `n_threads`: An integer specifying the number of threads to use for computation. This should be a positive integer, and if set to zero, the default number of threads will be used.
- **Output**: Returns a status code of type `enum ggml_status` indicating the success or failure of the computation. A successful execution will return a status indicating success, while failure will return an appropriate error code.
- **See also**: [`ggml_graph_compute_with_ctx`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_graph_compute_with_ctx)  (Implementation)


---
### ggml\_cpu\_has\_sse3<!-- {{#callable_declaration:ggml_cpu_has_sse3}} -->
Checks if the CPU supports SSE3.
- **Description**: This function is used to determine whether the CPU architecture supports the SSE3 instruction set. It can be called at any point in the program to check for SSE3 support, which may be relevant for optimizing performance in applications that can leverage these instructions. There are no specific preconditions for calling this function, and it does not modify any state or data.
- **Inputs**:
    - `none`: This function takes no parameters.
- **Output**: Returns 1 if the CPU supports SSE3, and 0 otherwise.
- **See also**: [`ggml_cpu_has_sse3`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_sse3)  (Implementation)


---
### ggml\_cpu\_has\_ssse3<!-- {{#callable_declaration:ggml_cpu_has_ssse3}} -->
Checks if the CPU supports SSSE3.
- **Description**: This function is used to determine whether the CPU on which the program is running supports the SSSE3 instruction set. It can be called at any point after the program starts, and is particularly useful for optimizing performance by enabling or disabling features based on CPU capabilities. The function will return a non-zero value if SSSE3 is supported, and zero otherwise.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns 1 if the CPU supports SSSE3, and 0 if it does not.
- **See also**: [`ggml_cpu_has_ssse3`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_ssse3)  (Implementation)


---
### ggml\_cpu\_has\_avx<!-- {{#callable_declaration:ggml_cpu_has_avx}} -->
Checks if the CPU supports AVX instructions.
- **Description**: This function is used to determine whether the CPU on which the program is running supports Advanced Vector Extensions (AVX). It is particularly useful for optimizing performance in applications that can leverage AVX for vectorized operations. The function can be called at any point after the program starts, and it will return a value indicating the presence of AVX support. There are no side effects associated with calling this function.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns 1 if the CPU supports AVX instructions, and 0 otherwise.
- **See also**: [`ggml_cpu_has_avx`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_avx)  (Implementation)


---
### ggml\_cpu\_has\_avx\_vnni<!-- {{#callable_declaration:ggml_cpu_has_avx_vnni}} -->
Checks if the CPU supports AVX VNNI instructions.
- **Description**: This function is used to determine whether the CPU on which the program is running supports AVX VNNI (Vector Neural Network Instructions) extensions. It can be called at any point after the program starts, and it is particularly useful for optimizing performance in applications that can leverage these instructions for enhanced computational efficiency. The function will return a non-zero value if AVX VNNI is supported, and zero otherwise.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns 1 if the CPU supports AVX VNNI instructions, and 0 if it does not.
- **See also**: [`ggml_cpu_has_avx_vnni`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_avx_vnni)  (Implementation)


---
### ggml\_cpu\_has\_avx2<!-- {{#callable_declaration:ggml_cpu_has_avx2}} -->
Checks if the CPU supports AVX2 instructions.
- **Description**: This function is used to determine whether the CPU on which the program is running supports AVX2 (Advanced Vector Extensions 2) instructions. It is typically called during initialization or configuration stages of an application to enable or optimize features that rely on AVX2. The function does not require any parameters and can be called at any time after the program starts. It is important to note that the return value will be 1 if AVX2 is supported and 0 otherwise.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns 1 if the CPU supports AVX2 instructions, otherwise returns 0.
- **See also**: [`ggml_cpu_has_avx2`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_avx2)  (Implementation)


---
### ggml\_cpu\_has\_bmi2<!-- {{#callable_declaration:ggml_cpu_has_bmi2}} -->
Checks if the CPU supports BMI2 instructions.
- **Description**: This function is used to determine if the CPU architecture supports BMI2 (Bit Manipulation Instruction Set 2) instructions, which can enhance performance for certain operations. It is typically called during initialization or setup phases of an application to enable or optimize features that rely on BMI2. The function does not require any parameters and can be called at any time, but it is advisable to check this capability before executing operations that may benefit from BMI2 instructions.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns 1 if the CPU supports BMI2 instructions, and 0 otherwise.
- **See also**: [`ggml_cpu_has_bmi2`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_bmi2)  (Implementation)


---
### ggml\_cpu\_has\_f16c<!-- {{#callable_declaration:ggml_cpu_has_f16c}} -->
Checks if the CPU supports the F16C instruction set.
- **Description**: This function is used to determine whether the CPU on which the program is running supports the F16C instruction set, which is used for half-precision floating-point operations. It is typically called during initialization or setup phases of an application to enable or optimize features that rely on this instruction set. The function does not require any parameters and can be called at any time after the program starts. It is safe to call multiple times, and it will consistently return the same result for the same CPU.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns 1 if the CPU supports the F16C instruction set, and 0 otherwise.
- **See also**: [`ggml_cpu_has_f16c`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_f16c)  (Implementation)


---
### ggml\_cpu\_has\_fma<!-- {{#callable_declaration:ggml_cpu_has_fma}} -->
Checks if the CPU supports FMA instructions.
- **Description**: This function is used to determine whether the CPU has support for Fused Multiply-Add (FMA) instructions, which can enhance performance for certain mathematical operations. It is typically called during initialization or setup phases of an application to optimize performance based on the capabilities of the underlying hardware. The function does not require any parameters and can be called at any time, but it is advisable to check for FMA support before performing operations that could benefit from it.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns 1 if the CPU supports FMA instructions, and 0 otherwise.
- **See also**: [`ggml_cpu_has_fma`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_fma)  (Implementation)


---
### ggml\_cpu\_has\_avx512<!-- {{#callable_declaration:ggml_cpu_has_avx512}} -->
Checks if the CPU supports AVX512 instructions.
- **Description**: This function is used to determine whether the CPU on which the program is running supports AVX512 instruction set extensions. It is particularly useful for optimizing performance in applications that can leverage these advanced vector instructions. The function can be called at any point after the program starts, and it will return a value indicating support for AVX512. There are no side effects associated with this function.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns 1 if the CPU supports AVX512 instructions, and 0 otherwise.
- **See also**: [`ggml_cpu_has_avx512`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_avx512)  (Implementation)


---
### ggml\_cpu\_has\_avx512\_vbmi<!-- {{#callable_declaration:ggml_cpu_has_avx512_vbmi}} -->
Checks if the CPU supports AVX512-VBMI.
- **Description**: This function is used to determine if the CPU on which the program is running supports the AVX512-VBMI instruction set. It can be called at any point after the program has started, and is particularly useful for optimizing performance in applications that can leverage AVX512-VBMI capabilities. The function will return a non-zero value if the feature is supported, and zero otherwise. There are no side effects associated with calling this function.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns 1 if the CPU supports AVX512-VBMI, and 0 if it does not.
- **See also**: [`ggml_cpu_has_avx512_vbmi`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_avx512_vbmi)  (Implementation)


---
### ggml\_cpu\_has\_avx512\_vnni<!-- {{#callable_declaration:ggml_cpu_has_avx512_vnni}} -->
Checks if the CPU supports AVX512 VNNI.
- **Description**: This function is used to determine whether the CPU architecture supports the AVX512 VNNI instruction set, which can enhance performance for certain workloads. It is typically called during initialization or configuration stages of an application to enable or optimize features that rely on this instruction set. The function does not require any parameters and can be called at any time, but it is advisable to check this capability before attempting to use AVX512 VNNI-specific features in your application.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns 1 if the CPU supports AVX512 VNNI, and 0 otherwise.
- **See also**: [`ggml_cpu_has_avx512_vnni`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_avx512_vnni)  (Implementation)


---
### ggml\_cpu\_has\_avx512\_bf16<!-- {{#callable_declaration:ggml_cpu_has_avx512_bf16}} -->
Checks if the CPU supports AVX512 BF16 instructions.
- **Description**: This function is used to determine whether the CPU architecture supports AVX512 BF16 (Bfloat16) instructions, which can enhance performance for certain computations. It is particularly useful for applications that require optimized numerical processing. The function can be called at any time, but it is typically used during initialization or configuration phases of an application to enable or disable features based on hardware capabilities.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns 1 if the CPU supports AVX512 BF16 instructions, and 0 otherwise.
- **See also**: [`ggml_cpu_has_avx512_bf16`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_avx512_bf16)  (Implementation)


---
### ggml\_cpu\_has\_amx\_int8<!-- {{#callable_declaration:ggml_cpu_has_amx_int8}} -->
Checks if the CPU supports AMX INT8 instructions.
- **Description**: This function is used to determine whether the CPU architecture supports AMX INT8 instructions, which can enhance performance for certain workloads. It is particularly useful for applications that may benefit from these instructions, allowing developers to conditionally enable features or optimizations based on the CPU capabilities. The function can be called at any time, and it does not require any prior initialization. It is safe to call in any context, including multi-threaded environments.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns 1 if the CPU supports AMX INT8 instructions, and 0 otherwise.
- **See also**: [`ggml_cpu_has_amx_int8`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_amx_int8)  (Implementation)


---
### ggml\_cpu\_has\_neon<!-- {{#callable_declaration:ggml_cpu_has_neon}} -->
Checks if the CPU supports NEON instructions.
- **Description**: This function is used to determine whether the CPU architecture supports NEON SIMD (Single Instruction, Multiple Data) instructions, which can enhance performance for certain operations. It should be called when you need to optimize your application for ARM architectures that can leverage NEON capabilities. The function returns a non-zero value if NEON is supported, and zero otherwise. It is important to note that this function does not require any parameters and can be called at any time after the application has started.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns a non-zero integer if NEON instructions are supported by the CPU, and zero if they are not.
- **See also**: [`ggml_cpu_has_neon`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_neon)  (Implementation)


---
### ggml\_cpu\_has\_arm\_fma<!-- {{#callable_declaration:ggml_cpu_has_arm_fma}} -->
Checks if the CPU supports ARM FMA instructions.
- **Description**: This function is used to determine if the CPU architecture supports the ARM Fused Multiply-Add (FMA) feature, which can enhance performance for certain mathematical operations. It is particularly useful for applications that require optimized numerical computations. The function can be called at any time, and it will return a value indicating the presence of the FMA feature. There are no specific preconditions for calling this function, and it does not modify any state or data.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns 1 if the CPU supports ARM FMA instructions, and 0 otherwise.
- **See also**: [`ggml_cpu_has_arm_fma`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_arm_fma)  (Implementation)


---
### ggml\_cpu\_has\_fp16\_va<!-- {{#callable_declaration:ggml_cpu_has_fp16_va}} -->
Checks if the CPU supports FP16 vector arithmetic.
- **Description**: This function is used to determine whether the CPU architecture supports FP16 vector arithmetic, which can be beneficial for performance in certain computations. It is particularly useful for applications that can leverage FP16 operations for efficiency. The function can be called at any time, but it is typically used during initialization or configuration stages of an application to optimize performance based on the capabilities of the underlying hardware.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns 1 if the CPU supports FP16 vector arithmetic, and 0 otherwise.
- **See also**: [`ggml_cpu_has_fp16_va`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_fp16_va)  (Implementation)


---
### ggml\_cpu\_has\_dotprod<!-- {{#callable_declaration:ggml_cpu_has_dotprod}} -->
Checks if the CPU supports dot product operations.
- **Description**: This function is used to determine if the current CPU architecture supports dot product operations, which can enhance performance for certain mathematical computations. It should be called when you need to optimize your application based on the capabilities of the underlying hardware. The function returns a non-zero value if dot product support is available, and zero otherwise. It is important to note that this function does not require any parameters and can be called at any time after the application has started.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns a non-zero integer if the CPU supports dot product operations; otherwise, it returns zero.
- **See also**: [`ggml_cpu_has_dotprod`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_dotprod)  (Implementation)


---
### ggml\_cpu\_has\_matmul\_int8<!-- {{#callable_declaration:ggml_cpu_has_matmul_int8}} -->
Checks if the CPU supports INT8 matrix multiplication.
- **Description**: This function is used to determine whether the current CPU architecture supports INT8 matrix multiplication operations. It is particularly useful for optimizing performance in applications that can leverage this feature. The function should be called when you need to check for hardware capabilities before executing operations that may benefit from INT8 matrix multiplication. It is safe to call this function at any time, and it will return a non-zero value if the feature is supported, or zero if it is not.
- **Inputs**:
    - `none`: This function does not take any parameters.
- **Output**: Returns a non-zero integer if the CPU supports INT8 matrix multiplication; otherwise, it returns zero.
- **See also**: [`ggml_cpu_has_matmul_int8`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_matmul_int8)  (Implementation)


---
### ggml\_cpu\_has\_sve<!-- {{#callable_declaration:ggml_cpu_has_sve}} -->
Checks if the CPU supports SVE.
- **Description**: This function is used to determine whether the CPU architecture supports Scalable Vector Extension (SVE) features. It should be called when there is a need to optimize performance based on the capabilities of the underlying hardware. The function returns a non-zero value if SVE is supported, and zero otherwise. It is important to note that this function does not require any parameters and can be called at any time after the program has started.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns a non-zero integer if SVE is supported by the CPU, and zero if it is not.
- **See also**: [`ggml_cpu_has_sve`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_sve)  (Implementation)


---
### ggml\_cpu\_get\_sve\_cnt<!-- {{#callable_declaration:ggml_cpu_get_sve_cnt}} -->
Returns the SVE vector length in bytes.
- **Description**: This function is used to determine the size of the Scalable Vector Extension (SVE) vector length in bytes, which is relevant for optimizing performance on ARM architectures that support SVE. It should be called when the application needs to adapt its processing based on the capabilities of the underlying hardware. If the architecture does not support SVE, the function will return 0, indicating that SVE is not available.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns an integer representing the SVE vector length in bytes, or 0 if SVE is not supported on the current architecture.
- **See also**: [`ggml_cpu_get_sve_cnt`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_get_sve_cnt)  (Implementation)


---
### ggml\_cpu\_has\_sme<!-- {{#callable_declaration:ggml_cpu_has_sme}} -->
Checks if the CPU supports SME.
- **Description**: This function is used to determine if the CPU architecture supports Scalable Matrix Extension (SME). It should be called when there is a need to optimize performance based on CPU capabilities, particularly in applications that can leverage SME for enhanced computational efficiency. The function returns a non-zero value if SME is supported, and zero otherwise. It is important to note that this function is only relevant on ARM architectures that define the appropriate feature macros.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns a non-zero integer if the CPU supports SME; otherwise, it returns zero.
- **See also**: [`ggml_cpu_has_sme`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_sme)  (Implementation)


---
### ggml\_cpu\_has\_riscv\_v<!-- {{#callable_declaration:ggml_cpu_has_riscv_v}} -->
Checks if the CPU supports RISC-V vector extensions.
- **Description**: This function is used to determine whether the CPU architecture supports RISC-V vector extensions, which can be beneficial for optimizing performance in applications that utilize vectorized operations. It is typically called during initialization or setup phases of an application to conditionally enable features or optimizations that rely on these extensions. The function does not require any parameters and can be called at any time, but it is advisable to check for support before attempting to use RISC-V vector-specific functionalities.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns 1 if the CPU supports RISC-V vector extensions, and 0 otherwise.
- **See also**: [`ggml_cpu_has_riscv_v`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_riscv_v)  (Implementation)


---
### ggml\_cpu\_has\_vsx<!-- {{#callable_declaration:ggml_cpu_has_vsx}} -->
Checks if the CPU supports VSX.
- **Description**: This function is used to determine if the CPU architecture supports the VSX (Vector-Scalar Extension) instruction set. It is particularly useful for optimizing performance in applications that can leverage these specific CPU capabilities. The function can be called at any point after the program starts, and it will return a value indicating support for VSX. There are no special preconditions for calling this function, and it does not modify any state or data.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns 1 if the CPU supports VSX, and 0 otherwise.
- **See also**: [`ggml_cpu_has_vsx`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_vsx)  (Implementation)


---
### ggml\_cpu\_has\_vxe<!-- {{#callable_declaration:ggml_cpu_has_vxe}} -->
Checks if the CPU supports VXE instructions.
- **Description**: This function is used to determine if the CPU architecture supports VXE instructions, which may be relevant for optimizing performance in certain applications. It can be called at any point in the program, but it is typically used during initialization or configuration stages to enable or disable features based on the CPU capabilities. The function returns a non-zero value if VXE support is present, and zero otherwise.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns 1 if the CPU supports VXE instructions, and 0 if it does not.
- **See also**: [`ggml_cpu_has_vxe`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_vxe)  (Implementation)


---
### ggml\_cpu\_has\_wasm\_simd<!-- {{#callable_declaration:ggml_cpu_has_wasm_simd}} -->
Checks if the CPU supports WebAssembly SIMD.
- **Description**: This function is used to determine whether the current CPU architecture supports WebAssembly SIMD (Single Instruction, Multiple Data) instructions. It can be called at any time to check for SIMD support, which may be relevant for optimizing performance in applications that utilize parallel processing capabilities. The function does not require any parameters and can be safely called without prior initialization.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns 1 if WebAssembly SIMD is supported, and 0 otherwise.
- **See also**: [`ggml_cpu_has_wasm_simd`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_wasm_simd)  (Implementation)


---
### ggml\_cpu\_has\_llamafile<!-- {{#callable_declaration:ggml_cpu_has_llamafile}} -->
Checks if the system supports the LLAMAFILE feature.
- **Description**: This function is used to determine whether the LLAMAFILE feature is available on the current system. It can be called at any time to check for support, and it is particularly useful for conditional compilation or feature checks in applications that may utilize LLAMAFILE. The function does not require any parameters and can be called without prior initialization. It is safe to call this function in any context where feature detection is necessary.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: Returns 1 if the LLAMAFILE feature is supported, and 0 otherwise.
- **See also**: [`ggml_cpu_has_llamafile`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_has_llamafile)  (Implementation)


---
### ggml\_get\_type\_traits\_cpu<!-- {{#callable_declaration:ggml_get_type_traits_cpu}} -->
Retrieves type traits for a specified data type.
- **Description**: This function is used to obtain the type traits associated with a specific data type, which can be useful for operations that depend on the characteristics of that type. It should be called with a valid `ggml_type` enumeration value. If an invalid type is provided, the behavior is undefined, so it is important to ensure that the type is within the valid range defined by the `ggml_type` enum.
- **Inputs**:
    - `type`: An enumeration value of type `ggml_type` that specifies the data type for which traits are requested. Valid values are defined in the `ggml_type` enum. Must not be null or invalid; otherwise, the behavior is undefined.
- **Output**: Returns a pointer to a `ggml_type_traits_cpu` structure that contains the traits for the specified type. If the type is invalid, the returned pointer may not be valid.
- **See also**: [`ggml_get_type_traits_cpu`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_get_type_traits_cpu)  (Implementation)


---
### ggml\_cpu\_init<!-- {{#callable_declaration:ggml_cpu_init}} -->
Initializes the CPU backend for the library.
- **Description**: This function must be called before using any other functions in the library that rely on CPU computations. It sets up necessary internal structures and initializes various mathematical tables used for computations. It is safe to call this function multiple times, but it will only perform the initialization on the first call. Ensure that this function is called in a single-threaded context to avoid potential race conditions.
- **Inputs**:
    - `None`: This function does not take any parameters.
- **Output**: This function does not return a value and does not mutate any inputs.
- **See also**: [`ggml_cpu_init`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_init)  (Implementation)


---
### ggml\_backend\_is\_cpu<!-- {{#callable_declaration:ggml_backend_is_cpu}} -->
Determines if the specified backend is a CPU backend.
- **Description**: This function is used to check whether a given backend is specifically a CPU backend. It should be called with a valid `ggml_backend_t` object that has been properly initialized. If the backend is `NULL`, or if it does not match the CPU backend identifier, the function will return false. This is useful for ensuring that operations intended for CPU backends are not mistakenly applied to other types of backends.
- **Inputs**:
    - `backend`: A pointer to a `ggml_backend_t` structure representing the backend to check. Must not be null; if it is null, the function will return false.
- **Output**: Returns true if the specified backend is a CPU backend; otherwise, returns false.
- **See also**: [`ggml_backend_is_cpu`](../src/ggml-cpu/ggml-cpu.cpp.driver.md#ggml_backend_is_cpu)  (Implementation)


---
### ggml\_backend\_cpu\_set\_n\_threads<!-- {{#callable_declaration:ggml_backend_cpu_set_n_threads}} -->
Sets the number of threads for CPU backend operations.
- **Description**: This function is used to configure the number of threads that the CPU backend will utilize for its operations. It should be called after initializing the CPU backend and before performing any computations. The value for the number of threads must be a positive integer, and setting it to an invalid value may lead to undefined behavior. It is important to ensure that the backend is indeed a CPU backend before calling this function.
- **Inputs**:
    - `backend_cpu`: A pointer to a `ggml_backend_t` structure representing the CPU backend. Must not be null and must be a valid CPU backend.
    - `n_threads`: An integer representing the number of threads to be used. Must be a positive integer. If a non-positive value is provided, the behavior is undefined.
- **Output**: None
- **See also**: [`ggml_backend_cpu_set_n_threads`](../src/ggml-cpu/ggml-cpu.cpp.driver.md#ggml_backend_cpu_set_n_threads)  (Implementation)


---
### ggml\_backend\_cpu\_set\_threadpool<!-- {{#callable_declaration:ggml_backend_cpu_set_threadpool}} -->
Sets the thread pool for the CPU backend.
- **Description**: This function is used to assign a specific thread pool to the CPU backend, which can optimize parallel processing tasks. It should be called after initializing the CPU backend and can be used to switch thread pools if needed. If a different thread pool is already assigned, it will be paused before the new one is set. Ensure that the provided thread pool is valid and properly initialized.
- **Inputs**:
    - `backend_cpu`: The CPU backend to which the thread pool will be assigned. Must not be null and must be a valid CPU backend.
    - `threadpool`: The thread pool to be set for the CPU backend. This can be null if no thread pool is desired. If a non-null thread pool is provided, it must be properly initialized.
- **Output**: None
- **See also**: [`ggml_backend_cpu_set_threadpool`](../src/ggml-cpu/ggml-cpu.cpp.driver.md#ggml_backend_cpu_set_threadpool)  (Implementation)


---
### ggml\_backend\_cpu\_set\_abort\_callback<!-- {{#callable_declaration:ggml_backend_cpu_set_abort_callback}} -->
Sets an abort callback for the CPU backend.
- **Description**: This function is used to register a callback that will be invoked to abort ongoing computations in the CPU backend. It should be called after initializing the CPU backend and before starting any computations. The provided callback will be triggered when an abort condition is met, allowing the user to handle the abort gracefully. The `abort_callback_data` parameter can be used to pass additional data to the callback function. If the `abort_callback` is set to `NULL`, no abort action will be taken.
- **Inputs**:
    - `backend_cpu`: A pointer to the CPU backend structure. Must not be null and must be a valid CPU backend.
    - `abort_callback`: A function pointer to the callback that will be invoked to handle abort requests. Can be null if no callback is desired.
    - `abort_callback_data`: A pointer to user-defined data that will be passed to the abort callback. Can be null if no data is needed.
- **Output**: None
- **See also**: [`ggml_backend_cpu_set_abort_callback`](../src/ggml-cpu/ggml-cpu.cpp.driver.md#ggml_backend_cpu_set_abort_callback)  (Implementation)


---
### ggml\_cpu\_fp32\_to\_fp16<!-- {{#callable_declaration:ggml_cpu_fp32_to_fp16}} -->
Converts an array of 32-bit floating-point numbers to 16-bit floating-point format.
- **Description**: This function is used to convert an array of `float` values into an array of `ggml_fp16_t` values, which represent 16-bit floating-point numbers. It is essential to ensure that the input array is valid and that the output array has sufficient space allocated to hold the converted values. The function processes the input array in chunks for efficiency, and it should be called when the conversion from 32-bit to 16-bit precision is required, such as in machine learning applications where memory usage is a concern. The caller must ensure that the input pointer is not null and that the output pointer is valid and has enough allocated memory for `n` elements.
- **Inputs**:
    - `x`: Pointer to the input array of `float` values. Must not be null. The array should contain at least `n` elements.
    - `y`: Pointer to the output array of `ggml_fp16_t` values. Must not be null. The caller is responsible for allocating enough memory to hold `n` elements.
    - `n`: The number of elements to convert. Must be a non-negative integer. If `n` is zero, the function performs no operations.
- **Output**: The function does not return a value and does not mutate any inputs.
- **See also**: [`ggml_cpu_fp32_to_fp16`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_fp32_to_fp16)  (Implementation)


---
### ggml\_cpu\_fp16\_to\_fp32<!-- {{#callable_declaration:ggml_cpu_fp16_to_fp32}} -->
Converts an array of half-precision floating-point values to single-precision.
- **Description**: This function is used to convert an array of half-precision floating-point values (`ggml_fp16_t`) to single-precision floating-point values (`float`). It is essential to call this function when you need to process or manipulate half-precision data in a format that requires single-precision. The input array must not be null, and the output array must be allocated with sufficient space to hold the converted values. The function handles the conversion for all elements specified by the count parameter, and it is safe to call with a count of zero, which will result in no operation.
- **Inputs**:
    - `x`: A pointer to the input array of half-precision floating-point values. Must not be null.
    - `y`: A pointer to the output array where the converted single-precision values will be stored. Must be allocated with at least 'n' elements.
    - `n`: The number of elements to convert. Must be non-negative.
- **Output**: The function does not return a value and directly modifies the output array.
- **See also**: [`ggml_cpu_fp16_to_fp32`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_fp16_to_fp32)  (Implementation)


---
### ggml\_cpu\_fp32\_to\_bf16<!-- {{#callable_declaration:ggml_cpu_fp32_to_bf16}} -->
Converts an array of `float` values to `ggml_bf16_t` format.
- **Description**: This function is used to convert a specified number of `float` values into the `ggml_bf16_t` format, which is a lower precision representation. It should be called when there is a need to reduce the memory footprint of floating-point data, particularly in machine learning applications. The input array must not be null, and the output array must be allocated with sufficient space to hold the converted values. The function does not perform any checks on the validity of the input values, but it assumes that the input size `n` is non-negative.
- **Inputs**:
    - `x`: A pointer to an array of `float` values to be converted. Must not be null.
    - `y`: A pointer to an array of `ggml_bf16_t` where the converted values will be stored. Must not be null and must have at least `n` elements allocated.
    - `n`: The number of elements to convert. Must be a non-negative integer.
- **Output**: This function returns nothing and directly populates the output array `y` with the converted values.
- **See also**: [`ggml_cpu_fp32_to_bf16`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_fp32_to_bf16)  (Implementation)


---
### ggml\_cpu\_bf16\_to\_fp32<!-- {{#callable_declaration:ggml_cpu_bf16_to_fp32}} -->
Converts an array of `ggml_bf16_t` values to an array of `float` values.
- **Description**: This function is used to convert a specified number of elements from an array of `ggml_bf16_t` (bfloat16) format to an array of `float` (32-bit floating point) format. It is essential to ensure that the output array has sufficient space allocated to hold the converted values, as the function does not perform any internal memory allocation. The conversion is performed for `n` elements, starting from the pointers provided. If `n` is less than or equal to zero, the function will not perform any conversion, and the output array will remain unchanged.
- **Inputs**:
    - `x`: A pointer to the input array of `ggml_bf16_t` values. Must not be null. The function expects the caller to ensure that the array has at least `n` elements.
    - `y`: A pointer to the output array where the converted `float` values will be stored. Must not be null. The caller must ensure that this array has enough space allocated to hold at least `n` float values.
    - `n`: An integer representing the number of elements to convert. Must be non-negative. If `n` is zero or negative, no conversion occurs.
- **Output**: The function does not return a value and does not mutate any inputs.
- **See also**: [`ggml_cpu_bf16_to_fp32`](../src/ggml-cpu/ggml-cpu.c.driver.md#ggml_cpu_bf16_to_fp32)  (Implementation)


