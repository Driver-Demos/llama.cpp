# Purpose
This C++ source file is a header file that defines a collection of functions and utilities for performing various tensor operations using the CANN (Compute Architecture for Neural Networks) backend. The file includes a wide range of tensor manipulation functions such as element-wise operations, matrix multiplications, normalization techniques, and activation functions, all optimized for execution on the CANN platform. The operations are designed to be executed asynchronously, leveraging the CANN backend's capabilities for high-performance computation, particularly in the context of neural network operations.

The file defines a series of functions that serve as public APIs for tensor operations, such as `ggml_cann_repeat`, `ggml_cann_leaky_relu`, and `ggml_cann_concat`, among others. These functions are designed to be used in conjunction with the CANN backend, providing a standardized interface for executing complex tensor operations. Additionally, the file includes utility classes and macros for managing ACL (Ascend Computing Language) resources, ensuring proper memory management and task execution. The use of templates and macros, such as `GGML_CANN_CALL_UNARY_OP`, facilitates the implementation of these operations by abstracting common patterns and reducing boilerplate code. Overall, this header file is a comprehensive toolkit for developers working with tensor operations in a CANN-optimized environment.
# Imports and Dependencies

---
- `functional`
- `aclnnop/aclnn_abs.h`
- `aclnnop/aclnn_neg.h`
- `aclnnop/aclnn_exp.h`
- `aclnnop/aclnn_arange.h`
- `aclnnop/aclnn_argsort.h`
- `aclnnop/aclnn_cat.h`
- `aclnnop/aclnn_clamp.h`
- `aclnnop/aclnn_gelu.h`
- `aclnnop/aclnn_gelu_v2.h`
- `aclnnop/aclnn_sigmoid.h`
- `aclnnop/aclnn_hardsigmoid.h`
- `aclnnop/aclnn_hardswish.h`
- `aclnnop/aclnn_leaky_relu.h`
- `aclnnop/aclnn_relu.h`
- `aclnnop/aclnn_silu.h`
- `aclnnop/aclnn_tanh.h`
- `aclnnop/aclnn_sqrt.h`
- `aclnnop/aclnn_sin.h`
- `aclnnop/aclnn_cos.h`
- `aclnnop/aclnn_log.h`
- `aclnnop/aclnn_sign.h`
- `acl_tensor.h`
- `common.h`


# Data Structures

---
### aclnn\_task<!-- {{#data_structure:aclnn_task}} -->
- **Type**: `class`
- **Members**:
    - `aclnn_func_`: A function pointer to the ACLNN function to be executed.
    - `workspace_addr_`: A pointer to the workspace memory address used during task execution.
    - `workspace_size_`: The size of the workspace memory in bytes.
    - `executor_`: A pointer to the ACL operation executor responsible for executing the task.
    - `stream_`: The ACL runtime stream on which the task is executed.
- **Description**: The `aclnn_task` class is a specialized task class that inherits from `cann_task` and is designed to execute a specific ACLNN function asynchronously. It encapsulates the necessary resources and parameters required for the execution of an ACLNN operation, including a function pointer to the ACLNN function, workspace memory address and size, an executor for the operation, and the runtime stream. The class provides a `run_task` method that overrides the base class method to execute the ACLNN function using the provided resources, ensuring that the operation is performed on the specified stream.
- **Member Functions**:
    - [`aclnn_task::aclnn_task`](#aclnn_taskaclnn_task)
    - [`aclnn_task::run_task`](#aclnn_taskrun_task)
- **Inherits From**:
    - `cann_task`

**Methods**

---
#### aclnn\_task::aclnn\_task<!-- {{#callable:aclnn_task::aclnn_task}} -->
The `aclnn_task` constructor initializes a task object to execute a specified ACLNN function asynchronously with given resources and stream.
- **Inputs**:
    - `aclnn_func`: A function pointer of type `aclnn_func_t` representing the ACLNN function to be executed.
    - `workspace_addr`: A pointer to the memory address allocated for workspace needed by the ACLNN function.
    - `workspace_size`: A 64-bit unsigned integer specifying the size of the workspace memory.
    - `executor`: A pointer to an `aclOpExecutor` object that manages the execution of the operation.
    - `stream`: An `aclrtStream` object representing the stream on which the task will be executed.
- **Control Flow**:
    - The constructor initializes the member variables `aclnn_func_`, `workspace_addr_`, `workspace_size_`, `executor_`, and `stream_` with the provided arguments.
- **Output**: An instance of the `aclnn_task` class, ready to execute the specified ACLNN function when the `run_task` method is called.
- **See also**: [`aclnn_task`](#aclnn_task)  (Data Structure)


---
#### aclnn\_task::run\_task<!-- {{#callable:aclnn_task::run_task}} -->
The `run_task` method executes a specified ACLNN function using provided workspace, executor, and stream parameters.
- **Inputs**:
    - `None`: This method does not take any direct input parameters as it operates on member variables of the `aclnn_task` class.
- **Control Flow**:
    - The method calls the ACLNN function stored in `aclnn_func_` with the parameters `workspace_addr_`, `workspace_size_`, `executor_`, and `stream_`.
    - The `ACL_CHECK` macro is used to ensure that the function call is successful, likely throwing an error or handling it if the function fails.
- **Output**: The method does not return any value; it performs an operation using the ACLNN function and checks for success.
- **See also**: [`aclnn_task`](#aclnn_task)  (Data Structure)



---
### release\_resource\_task<!-- {{#data_structure:release_resource_task}} -->
- **Type**: `class`
- **Members**:
    - `resource_`: A private member variable that holds a vector of ACL resources to be released.
- **Description**: The `release_resource_task` class is a specialized task class derived from `cann_task` designed to manage the release of ACL resources. It encapsulates a vector of `any_acl_resource` objects, which are unique pointers with custom deleters for managing the lifecycle of ACL resources. The primary function of this class is to clear the resources when the `run_task` method is invoked, ensuring that all resources are properly released and memory is managed efficiently.
- **Member Functions**:
    - [`release_resource_task::release_resource_task`](#release_resource_taskrelease_resource_task)
    - [`release_resource_task::run_task`](#release_resource_taskrun_task)
- **Inherits From**:
    - `cann_task`

**Methods**

---
#### release\_resource\_task::release\_resource\_task<!-- {{#callable:release_resource_task::release_resource_task}} -->
The `release_resource_task` constructor initializes a task to manage and release a collection of ACL resources.
- **Inputs**:
    - `resources`: A rvalue reference to a vector of `any_acl_resource` objects, representing the ACL resources to be managed and released.
- **Control Flow**:
    - The constructor takes a vector of `any_acl_resource` objects as an rvalue reference.
    - It uses `std::move` to transfer ownership of the resources to the `resource_` member variable.
- **Output**: There is no return value as this is a constructor for initializing the `release_resource_task` object.
- **See also**: [`release_resource_task`](#release_resource_task)  (Data Structure)


---
#### release\_resource\_task::run\_task<!-- {{#callable:release_resource_task::run_task}} -->
The `run_task` method in the `release_resource_task` class clears the vector of ACL resources to release them.
- **Inputs**: None
- **Control Flow**:
    - The method accesses the `resource_` member variable, which is a vector of `any_acl_resource`.
    - It calls the `clear` method on the `resource_` vector, which removes all elements from the vector, effectively releasing the resources.
- **Output**: The method does not return any value; it performs an action of clearing the resources.
- **See also**: [`release_resource_task`](#release_resource_task)  (Data Structure)



---
### async\_memcpy\_task<!-- {{#data_structure:async_memcpy_task}} -->
- **Type**: `class`
- **Members**:
    - `dst_`: Pointer to the destination memory location for the copy operation.
    - `src_`: Pointer to the source memory location for the copy operation.
    - `size_`: Size of the memory to be copied in bytes.
    - `kind_`: Type of memory copy operation (e.g., host-to-device, device-to-host).
    - `stream_`: Stream on which the asynchronous memory copy operation is executed.
- **Description**: The `async_memcpy_task` class is a specialized task class derived from `cann_task` designed to perform asynchronous memory copy operations using the ACL (Ascend Computing Language) runtime. It encapsulates the details of the memory copy, including the source and destination pointers, the size of the data to be copied, the type of memory copy operation, and the stream on which the operation is to be executed. This class is part of a larger framework for managing asynchronous tasks and is used to offload memory copy operations to the device, allowing for efficient data transfer in parallel with other computations.
- **Member Functions**:
    - [`async_memcpy_task::async_memcpy_task`](#async_memcpy_taskasync_memcpy_task)
    - [`async_memcpy_task::run_task`](#async_memcpy_taskrun_task)
- **Inherits From**:
    - `cann_task`

**Methods**

---
#### async\_memcpy\_task::async\_memcpy\_task<!-- {{#callable:async_memcpy_task::async_memcpy_task}} -->
The `async_memcpy_task` class is designed to perform asynchronous memory copy operations using the ACL runtime.
- **Inputs**:
    - `dst`: A pointer to the destination memory location where data will be copied to.
    - `src`: A pointer to the source memory location from where data will be copied.
    - `size`: The size of the data to be copied, in bytes.
    - `kind`: The type of memory copy operation, specified by `aclrtMemcpyKind` (e.g., host-to-device, device-to-host).
    - `stream`: The ACL runtime stream on which the asynchronous copy operation will be executed.
- **Control Flow**:
    - The constructor initializes the task with the destination and source pointers, size of data, memory copy kind, and the stream.
    - The `run_task` method is overridden to execute the asynchronous memory copy using `aclrtMemcpyAsync`.
- **Output**: The function does not return a value; it performs an asynchronous memory copy operation.
- **See also**: [`async_memcpy_task`](#async_memcpy_task)  (Data Structure)


---
#### async\_memcpy\_task::run\_task<!-- {{#callable:async_memcpy_task::run_task}} -->
The `run_task` method in the `async_memcpy_task` class performs an asynchronous memory copy operation using the ACL runtime.
- **Inputs**:
    - `dst_`: A pointer to the destination memory location where data will be copied to.
    - `src_`: A pointer to the source memory location from where data will be copied.
    - `size_`: The size of the data to be copied, in bytes.
    - `kind_`: The type of memory copy operation, specified by `aclrtMemcpyKind` (e.g., host-to-device, device-to-host).
    - `stream_`: The ACL stream on which the asynchronous memory copy operation will be executed.
- **Control Flow**:
    - The method calls `aclrtMemcpyAsync` with the destination, source, size, kind, and stream parameters.
    - The `ACL_CHECK` macro is used to ensure that the `aclrtMemcpyAsync` call is successful, handling any errors that may occur.
- **Output**: The method does not return a value; it performs an asynchronous operation to copy memory from the source to the destination.
- **See also**: [`async_memcpy_task`](#async_memcpy_task)  (Data Structure)



---
### async\_memset\_task<!-- {{#data_structure:async_memset_task}} -->
- **Type**: `class`
- **Members**:
    - `buffer_`: A pointer to the memory buffer to be set.
    - `size_`: The size of the memory buffer in bytes.
    - `value_`: The integer value to set in the memory buffer.
    - `stream_`: The ACL runtime stream used for asynchronous operations.
- **Description**: The `async_memset_task` class is a specialized task class derived from `cann_task` designed to perform asynchronous memory set operations using the ACL runtime. It encapsulates the details required to set a specified memory buffer to a given integer value asynchronously on a specified stream. This class is particularly useful in scenarios where non-blocking memory operations are needed to improve performance by allowing other tasks to execute concurrently while the memory set operation is in progress.
- **Member Functions**:
    - [`async_memset_task::async_memset_task`](#async_memset_taskasync_memset_task)
    - [`async_memset_task::run_task`](#async_memset_taskrun_task)
- **Inherits From**:
    - `cann_task`

**Methods**

---
#### async\_memset\_task::async\_memset\_task<!-- {{#callable:async_memset_task::async_memset_task}} -->
The `async_memset_task` class is designed to perform asynchronous memory set operations on a specified buffer using a given value and stream.
- **Inputs**:
    - `buffer`: A pointer to the memory buffer that will be set.
    - `size`: The size of the memory buffer in bytes.
    - `value`: The integer value to set in the buffer.
    - `stream`: The ACL runtime stream used for the asynchronous operation.
- **Control Flow**:
    - The constructor initializes the task with the provided buffer, size, value, and stream.
    - The `run_task` method is overridden to execute the `aclrtMemsetAsync` function, which sets the memory asynchronously using the specified parameters.
- **Output**: There is no direct output from the function, but the specified buffer is asynchronously set to the given value.
- **See also**: [`async_memset_task`](#async_memset_task)  (Data Structure)


---
#### async\_memset\_task::run\_task<!-- {{#callable:async_memset_task::run_task}} -->
The `run_task` method in the `async_memset_task` class executes an asynchronous memory set operation on a specified buffer using a given stream.
- **Inputs**:
    - `buffer_`: A pointer to the memory buffer that will be set.
    - `size_`: The size of the memory buffer in bytes.
    - `value_`: The integer value to set in the buffer.
    - `stream_`: The ACL stream used for the asynchronous operation.
- **Control Flow**:
    - The method calls `aclrtMemsetAsync` with the buffer, size, value, and stream as arguments.
    - The `ACL_CHECK` macro is used to ensure the operation completes successfully.
- **Output**: The method does not return a value; it performs an asynchronous operation to set the memory buffer.
- **See also**: [`async_memset_task`](#async_memset_task)  (Data Structure)



# Functions

---
### destroy<!-- {{#callable:destroy}} -->
The `destroy` function is a static method that safely destroys an `aclTensorList` resource by calling `aclDestroyTensorList` and checking for errors using `ACL_CHECK`.
- **Inputs**:
    - `p`: A void pointer to the `aclTensorList` resource that needs to be destroyed.
- **Control Flow**:
    - The function casts the void pointer `p` to an `aclTensorList` pointer using `static_cast`.
    - It calls `aclDestroyTensorList` with the casted pointer to destroy the resource.
    - The `ACL_CHECK` macro is used to ensure that the destruction operation is successful and to handle any errors.
- **Output**: The function does not return any value; it performs a cleanup operation on the resource pointed to by `p`.


---
### make\_acl\_resource<!-- {{#callable:make_acl_resource}} -->
The `make_acl_resource` function creates a smart pointer for an ACL resource with a custom deleter to ensure proper resource management.
- **Inputs**:
    - `ptr`: A raw pointer to an ACL resource of type `T` that needs to be managed.
- **Control Flow**:
    - The function takes a raw pointer `ptr` of type `T` as input.
    - It creates an `any_acl_resource` smart pointer, which is a `std::unique_ptr` with a custom deleter.
    - The custom deleter is a lambda function that calls `acl_resource_traits<T>::destroy` to properly destroy the resource when the smart pointer goes out of scope.
- **Output**: Returns an `any_acl_resource` smart pointer that manages the lifecycle of the ACL resource pointed to by `ptr`.


---
### register\_acl\_resources<!-- {{#callable:register_acl_resources}} -->
The `register_acl_resources` function registers multiple ACL resources into a vector for lifetime management by converting raw pointers to smart pointers with custom deleters.
- **Inputs**:
    - `vec`: A reference to a vector of `any_acl_resource` where the ACL resources will be stored.
    - `args`: Variadic template arguments representing raw pointers to ACL resources that need to be registered.
- **Control Flow**:
    - The function uses a fold expression to iterate over the variadic arguments `args`.
    - For each argument in `args`, it calls [`make_acl_resource`](#make_acl_resource) to create a smart pointer with a custom deleter for the ACL resource.
    - The resulting smart pointer is then added to the vector `vec` using `emplace_back`.
- **Output**: The function does not return any value; it modifies the input vector `vec` by adding smart pointers to the ACL resources.
- **Functions called**:
    - [`make_acl_resource`](#make_acl_resource)


---
### ggml\_cann\_release\_resources<!-- {{#callable:ggml_cann_release_resources}} -->
The `ggml_cann_release_resources` function registers and releases multiple ACL resources, optionally deferring the release using a task if asynchronous mode is enabled.
- **Inputs**:
    - `ctx`: A reference to a `ggml_backend_cann_context` object that manages task submission and async mode.
    - `args`: A variadic list of ACL resources to be registered and released.
- **Control Flow**:
    - Initialize a vector `resources` to hold ACL resources.
    - Call [`register_acl_resources`](#register_acl_resources) to populate `resources` with the provided `args`.
    - Check if `ctx.async_mode` is true.
    - If true, create a `release_resource_task` with the `resources` and submit it to `ctx.task_queue`.
- **Output**: The function does not return a value; it manages the lifecycle of ACL resources, potentially deferring their release.
- **Functions called**:
    - [`register_acl_resources`](#register_acl_resources)


---
### ggml\_cann\_async\_memcpy<!-- {{#callable:ggml_cann_async_memcpy}} -->
The `ggml_cann_async_memcpy` function performs an asynchronous memory copy operation, either by submitting a task to a queue if in async mode or directly using `aclrtMemcpyAsync` otherwise.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_backend_cann_context` object that contains the context for the CANN backend operations, including stream and async mode configuration.
    - `dst`: A pointer to the destination memory address where the data will be copied to.
    - `src`: A pointer to the source memory address from where the data will be copied.
    - `len`: The size of the memory to be copied, in bytes.
    - `kind`: An `aclrtMemcpyKind` value that specifies the type of memory copy (e.g., host-to-device, device-to-host, etc.).
- **Control Flow**:
    - Check if the context `ctx` is in async mode.
    - If in async mode, create a new `async_memcpy_task` with the provided parameters and submit it to the task queue.
    - If not in async mode, perform the memory copy directly using `aclrtMemcpyAsync` with the provided parameters.
- **Output**: The function does not return a value; it performs the memory copy operation asynchronously or directly based on the context's async mode.


---
### ggml\_cann\_async\_memset<!-- {{#callable:ggml_cann_async_memset}} -->
The `ggml_cann_async_memset` function performs an asynchronous memory set operation on a buffer, either by submitting a task to a task queue if in async mode or directly calling the ACL API otherwise.
- **Inputs**:
    - `ctx`: A reference to a `ggml_backend_cann_context` object, which contains the context for backend operations, including async mode and stream information.
    - `buffer`: A pointer to the memory buffer that will be set.
    - `size`: The size of the memory buffer in bytes.
    - `value`: The integer value to set in the buffer.
- **Control Flow**:
    - Check if the context (`ctx`) is in async mode.
    - If in async mode, create a unique pointer to an `async_memset_task` with the buffer, size, value, and stream from the context, and submit this task to the context's task queue.
    - If not in async mode, directly call `aclrtMemsetAsync` with the buffer, size, value, and stream from the context.
- **Output**: The function does not return a value; it performs the memory set operation asynchronously.


---
### ggml\_cann\_binary\_op<!-- {{#callable:ggml_cann_binary_op}} -->
The `ggml_cann_binary_op` function applies a specified binary operation to two input tensors, handling broadcasting if necessary, and stores the result in a destination tensor using the CANN backend.
- **Inputs**:
    - `ctx`: A reference to the `ggml_backend_cann_context` object, which manages execution and resources for the CANN backend.
    - `dst`: A pointer to the `ggml_tensor` object that serves as the destination tensor for the operation, containing the result of the binary operation.
- **Control Flow**:
    - Retrieve the source tensors `src0` and `src1` from the `dst` tensor's source array.
    - Declare pointers for ACL tensors: `acl_src0`, `acl_src1`, and `acl_dst`.
    - Call [`bcast_shape`](aclnn_ops.cpp.driver.md#bcast_shape) to prepare broadcast-compatible ACL tensors for `src0`, `src1`, and `dst`, assigning them to `acl_src0`, `acl_src1`, and `acl_dst` respectively.
    - Invoke the `binary_op` function with the context and the prepared ACL tensors to perform the binary operation.
    - Release the resources associated with the ACL tensors using [`ggml_cann_release_resources`](#ggml_cann_release_resources).
- **Output**: The function does not return a value; it modifies the `dst` tensor in place to store the result of the binary operation.
- **Functions called**:
    - [`bcast_shape`](aclnn_ops.cpp.driver.md#bcast_shape)
    - [`ggml_cann_release_resources`](#ggml_cann_release_resources)


---
### ggml\_cann\_unary\_op<!-- {{#callable:ggml_cann_unary_op}} -->
The `ggml_cann_unary_op` function applies a specified unary operation to a source tensor and stores the result in a destination tensor using the CANN backend.
- **Inputs**:
    - `ctx`: A reference to the `ggml_backend_cann_context` object, which manages the execution context and resources for CANN operations.
    - `dst`: A pointer to the `ggml_tensor` object representing the destination tensor where the result of the unary operation will be stored; its source tensor is accessed via `dst->src[0]`.
- **Control Flow**:
    - Retrieve the source tensor from the destination tensor's source array (`dst->src[0]`).
    - Create ACL tensor representations for both the source and destination tensors using [`ggml_cann_create_tensor`](acl_tensor.h.driver.md#ggml_cann_create_tensor).
    - Invoke the provided unary operation function, passing the context and the ACL tensor representations of the source and destination tensors.
    - Release the resources associated with the ACL tensors using [`ggml_cann_release_resources`](#ggml_cann_release_resources).
- **Output**: The function does not return a value; it modifies the destination tensor in place by applying the unary operation.
- **Functions called**:
    - [`ggml_cann_create_tensor`](acl_tensor.h.driver.md#ggml_cann_create_tensor)
    - [`ggml_cann_release_resources`](#ggml_cann_release_resources)


