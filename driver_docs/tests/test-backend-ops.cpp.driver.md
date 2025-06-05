# Purpose
This C++ source code file is designed to define and execute a series of tests for various operations (ops) and backends within the GGML (General Graph Machine Learning) framework. The primary purpose of this file is to ensure that the results of different backends computing the same GGML operations are consistent, both in the forward and backward passes. The file is structured into three main sections: general setup, definition of GGML operations to be tested, and the specification of which tests to run.

The code includes a variety of test cases for different GGML operations, such as unary operations, matrix multiplications, and more complex operations like softmax and attention mechanisms. Each test case is encapsulated in a struct that inherits from a base `test_case` struct, which provides a common interface for building computation graphs, initializing tensors, and evaluating the results. The tests are designed to cover a wide range of scenarios, including different data types, tensor shapes, and operation parameters. The file also includes functionality to compare the performance of different backends and to verify the correctness of gradient computations using finite differences. The tests can be filtered and executed based on specific operation names, backend types, and parameter configurations, making the file a comprehensive tool for validating and benchmarking GGML operations across different computational environments.
# Imports and Dependencies

---
- `ggml.h`
- `ggml-alloc.h`
- `ggml-backend.h`
- `ggml-cpp.h`
- `algorithm`
- `array`
- `cfloat`
- `cinttypes`
- `cstdint`
- `cstdio`
- `cstdlib`
- `cstring`
- `future`
- `memory`
- `random`
- `regex`
- `string`
- `thread`
- `vector`


# Global Variables

---
### all\_types
- **Type**: ``ggml_type[]``
- **Description**: The `all_types` variable is a static constant array of `ggml_type` enumerations, representing various data types supported by the GGML library. It includes floating-point types like `GGML_TYPE_F32`, `GGML_TYPE_F16`, and `GGML_TYPE_BF16`, as well as several quantized types such as `GGML_TYPE_Q4_0`, `GGML_TYPE_Q5_0`, and others. Additionally, it contains integer quantized types like `GGML_TYPE_IQ2_XXS` and `GGML_TYPE_IQ4_NL`. This array is used to define the range of data types that can be tested or utilized within the GGML framework.
- **Use**: This variable is used to enumerate and test various data types supported by the GGML library in different operations and backends.


---
### base\_types
- **Type**: ``ggml_type[]``
- **Description**: The `base_types` variable is a static constant array of `ggml_type` elements, which are enumerations representing different data types used in the GGML library. This array includes types such as `GGML_TYPE_F32`, `GGML_TYPE_F16`, `GGML_TYPE_Q8_0`, `GGML_TYPE_Q4_0`, `GGML_TYPE_Q4_1`, `GGML_TYPE_Q4_K`, and `GGML_TYPE_IQ2_XXS`. These types are used for various operations and tests within the GGML framework.
- **Use**: This variable is used to define a set of base data types for testing and operations within the GGML library.


---
### other\_types
- **Type**: ``ggml_type[]``
- **Description**: The `other_types` variable is a static constant array of `ggml_type` enumerations. It contains a list of various quantized and floating-point types used in the GGML library, such as `GGML_TYPE_Q4_1`, `GGML_TYPE_Q5_0`, `GGML_TYPE_Q8_0`, and `GGML_TYPE_BF16`. These types represent different data formats and precisions for tensors in the library.
- **Use**: This variable is used to define a set of data types that can be utilized in GGML operations and tests.


# Data Structures

---
### test\_mode<!-- {{#data_structure:test_mode}} -->
- **Type**: `enum`
- **Members**:
    - `MODE_TEST`: Represents the test mode for running tests.
    - `MODE_PERF`: Represents the performance mode for running tests.
    - `MODE_GRAD`: Represents the gradient checking mode for running tests.
- **Description**: The `test_mode` enum defines three modes for running tests: `MODE_TEST` for standard testing, `MODE_PERF` for performance evaluation, and `MODE_GRAD` for gradient checking.


---
### test\_case<!-- {{#data_structure:test_case}} -->
- **Type**: `struct`
- **Members**:
    - `gf`: Pointer to a `ggml_cgraph` structure representing the forward graph.
    - `gb`: Pointer to a `ggml_cgraph` structure representing the backward graph.
    - `mode`: Current mode of the test case, represented by the `test_mode` enum.
    - `sentinels`: Vector of pointers to `ggml_tensor` structures used for overflow checks.
- **Description**: The `test_case` struct serves as a base class for defining test cases for various GGML operations, providing a framework for building computation graphs, evaluating performance, and checking gradients. It includes virtual methods for operation description, variable representation, and graph construction, along with several member variables for managing the state of the test case.
- **Member Functions**:
    - [`test_case::~test_case`](#test_casetest_case)
    - [`test_case::op_desc`](#test_caseop_desc)
    - [`test_case::vars`](#test_casevars)
    - [`test_case::max_nmse_err`](#test_casemax_nmse_err)
    - [`test_case::max_maa_err`](#test_casemax_maa_err)
    - [`test_case::grad_eps`](#test_casegrad_eps)
    - [`test_case::grad_precise`](#test_casegrad_precise)
    - [`test_case::grad_nmax`](#test_casegrad_nmax)
    - [`test_case::grad_expect`](#test_casegrad_expect)
    - [`test_case::initialize_tensors`](#test_caseinitialize_tensors)
    - [`test_case::op_size`](#test_caseop_size)
    - [`test_case::op_flops`](#test_caseop_flops)
    - [`test_case::add_sentinel`](#test_caseadd_sentinel)
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`test_case::ggml_new_tensor_1d`](#test_caseggml_new_tensor_1d)
    - [`test_case::ggml_new_tensor_2d`](#test_caseggml_new_tensor_2d)
    - [`test_case::ggml_new_tensor_3d`](#test_caseggml_new_tensor_3d)
    - [`test_case::ggml_new_tensor_4d`](#test_caseggml_new_tensor_4d)
    - [`test_case::eval`](#test_caseeval)
    - [`test_case::eval_perf`](#test_caseeval_perf)
    - [`test_case::eval_grad`](#test_caseeval_grad)

**Methods**

---
#### test\_case::\~test\_case<!-- {{#callable:test_case::~test_case}} -->
The `~test_case` function is a virtual destructor for the `test_case` class, ensuring proper cleanup of derived class instances.
- **Inputs**: None
- **Control Flow**:
    - The function does not contain any control flow statements as it is a destructor.
    - It is called when an object of a derived class is destroyed, allowing for proper resource deallocation.
- **Output**: The function does not return any value; it ensures that resources are released when an instance of `test_case` or its derived classes is destroyed.
- **See also**: [`test_case`](#test_case)  (Data Structure)


---
#### test\_case::op\_desc<!-- {{#callable:test_case::op_desc}} -->
The `op_desc` method returns the operation description of a given `ggml_tensor`.
- **Inputs**:
    - `t`: A pointer to a `ggml_tensor` whose operation description is to be retrieved.
- **Control Flow**:
    - Calls the [`ggml_op_desc`](../ggml/src/ggml.c.driver.md#ggml_op_desc) function with the input tensor `t`.
    - Returns the result of the [`ggml_op_desc`](../ggml/src/ggml.c.driver.md#ggml_op_desc) function.
- **Output**: A string representing the operation description of the input tensor.
- **Functions called**:
    - [`ggml_op_desc`](../ggml/src/ggml.c.driver.md#ggml_op_desc)
- **See also**: [`test_case`](#test_case)  (Data Structure)


---
#### test\_case::vars<!-- {{#callable:test_case::vars}} -->
The `vars` method returns an empty string, serving as a placeholder for derived classes to provide variable descriptions.
- **Inputs**: None
- **Control Flow**:
    - The method directly returns an empty string without any conditions or loops.
- **Output**: The output is an empty string, indicating that there are no variables to describe for the current instance of the class.
- **See also**: [`test_case`](#test_case)  (Data Structure)


---
#### test\_case::max\_nmse\_err<!-- {{#callable:test_case::max_nmse_err}} -->
The `max_nmse_err` function returns a constant value representing the maximum normalized mean squared error.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a constant value without any conditional statements or loops.
- **Output**: The output is a double value of 1e-7, which signifies the maximum allowable normalized mean squared error.
- **See also**: [`test_case`](#test_case)  (Data Structure)


---
#### test\_case::max\_maa\_err<!-- {{#callable:test_case::max_maa_err}} -->
The `max_maa_err` function returns a constant maximum absolute asymmetry error value of 1e-4.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a constant value without any conditional logic or loops.
- **Output**: The output is a double value representing the maximum absolute asymmetry error, which is fixed at 1e-4.
- **See also**: [`test_case`](#test_case)  (Data Structure)


---
#### test\_case::grad\_eps<!-- {{#callable:test_case::grad_eps}} -->
The `grad_eps` function returns a constant gradient epsilon value of 0.1.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a constant value without any conditional logic or loops.
- **Output**: The output is a float value of 0.1, which is used as a small perturbation for gradient calculations.
- **See also**: [`test_case`](#test_case)  (Data Structure)


---
#### test\_case::grad\_precise<!-- {{#callable:test_case::grad_precise}} -->
The `grad_precise` function is a virtual method that indicates whether to use a precise gradient estimation method.
- **Inputs**: None
- **Control Flow**:
    - The function always returns false, indicating that the default gradient estimation method will be used.
- **Output**: The output is a boolean value, specifically false, indicating that the gradient estimation will not be precise.
- **See also**: [`test_case`](#test_case)  (Data Structure)


---
#### test\_case::grad\_nmax<!-- {{#callable:test_case::grad_nmax}} -->
The `grad_nmax` function returns the maximum number of gradients to be checked, defaulting to 10000.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a constant value of 10000 without any conditional logic or iterations.
- **Output**: The output is an integer value of 10000, representing the maximum number of gradients to be checked.
- **See also**: [`test_case`](#test_case)  (Data Structure)


---
#### test\_case::grad\_expect<!-- {{#callable:test_case::grad_expect}} -->
The `grad_expect` function returns an empty vector, indicating no expected gradient values.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns an empty vector without any computation or condition checks.
- **Output**: The output is an empty `std::vector<float>`, which signifies that there are no expected gradient values to check against.
- **See also**: [`test_case`](#test_case)  (Data Structure)


---
#### test\_case::initialize\_tensors<!-- {{#callable:test_case::initialize_tensors}} -->
The `initialize_tensors` method initializes all tensors in a given `ggml_context` by applying a uniform random initialization.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that contains the state and data for tensor operations.
- **Control Flow**:
    - The method iterates over all tensors in the context using [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor) and [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor).
    - For each tensor, it calls the [`init_tensor_uniform`](#init_tensor_uniform) function to initialize the tensor with random values.
- **Output**: This method does not return a value; it modifies the tensors in place within the provided `ggml_context`.
- **Functions called**:
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`init_tensor_uniform`](#init_tensor_uniform)
- **See also**: [`test_case`](#test_case)  (Data Structure)


---
#### test\_case::op\_size<!-- {{#callable:test_case::op_size}} -->
The `op_size` function calculates the total size in bytes of a `ggml_tensor`, including the size of its source tensors.
- **Inputs**:
    - `t`: A pointer to a `ggml_tensor` whose size is to be calculated.
- **Control Flow**:
    - The function initializes a variable `size` with the size of the tensor `t` obtained from the [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes) function.
    - It then iterates over the source tensors of `t` (up to `GGML_MAX_SRC`), checking if each source tensor is not NULL.
    - For each non-NULL source tensor, it adds its size (obtained from [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)) to the total `size`.
    - Finally, it returns the total calculated size.
- **Output**: Returns the total size in bytes of the `ggml_tensor` and its source tensors.
- **Functions called**:
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)
- **See also**: [`test_case`](#test_case)  (Data Structure)


---
#### test\_case::op\_flops<!-- {{#callable:test_case::op_flops}} -->
The `op_flops` function returns zero, indicating no floating point operations for the given tensor.
- **Inputs**:
    - `t`: A pointer to a `ggml_tensor` structure representing the tensor for which the floating point operations are to be calculated.
- **Control Flow**:
    - The function uses the `GGML_UNUSED` macro to indicate that the input parameter `t` is not used within the function body.
    - The function directly returns a constant value of 0.
- **Output**: The function outputs a `uint64_t` value of 0, representing the number of floating point operations for the specified tensor.
- **See also**: [`test_case`](#test_case)  (Data Structure)


---
#### test\_case::add\_sentinel<!-- {{#callable:test_case::add_sentinel}} -->
The `add_sentinel` function adds a sentinel tensor to the `sentinels` vector in the `test_case` class if the current mode is not performance or gradient mode.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations.
- **Control Flow**:
    - The function first checks if the `mode` is either `MODE_PERF` or `MODE_GRAD`.
    - If the mode matches either of these, the function returns early without adding a sentinel.
    - If the mode does not match, it creates a new 1D tensor of type `GGML_TYPE_F32` with a size defined by `sentinel_size`.
    - The newly created tensor is named using the current size of the `sentinels` vector.
    - Finally, the new tensor is pushed back into the `sentinels` vector.
- **Output**: The function does not return a value; it modifies the `sentinels` vector by adding a new tensor if the current mode is not performance or gradient mode.
- **Functions called**:
    - [`ggml_format_name`](../ggml/src/ggml.c.driver.md#ggml_format_name)
- **See also**: [`test_case`](#test_case)  (Data Structure)


---
#### test\_case::ggml\_new\_tensor<!-- {{#callable:test_case::ggml_new_tensor}} -->
Creates a new `ggml_tensor` and adds a sentinel tensor for overflow checking.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for tensor operations.
    - `type`: The type of the tensor being created, specified by the `ggml_type` enumeration.
    - `n_dims`: The number of dimensions for the tensor.
    - `ne`: An array of integers representing the size of each dimension of the tensor.
- **Control Flow**:
    - Calls the `::ggml_new_tensor` function to create a new tensor of the specified type and dimensions.
    - Invokes the [`add_sentinel`](#test_caseadd_sentinel) function to add a sentinel tensor for overflow checking.
    - Returns the newly created tensor.
- **Output**: Returns a pointer to the newly created `ggml_tensor`.
- **Functions called**:
    - [`test_case::add_sentinel`](#test_caseadd_sentinel)
- **See also**: [`test_case`](#test_case)  (Data Structure)


---
#### test\_case::ggml\_new\_tensor\_1d<!-- {{#callable:test_case::ggml_new_tensor_1d}} -->
Creates a new 1D tensor and adds a sentinel for overflow checking.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and state for tensor operations.
    - `type`: The `ggml_type` indicating the data type of the tensor (e.g., float, int, etc.).
    - `ne0`: An integer representing the number of elements in the 1D tensor.
- **Control Flow**:
    - Calls the `::ggml_new_tensor_1d` function to create a new tensor of the specified type and size.
    - Invokes the [`add_sentinel`](#test_caseadd_sentinel) function to add a sentinel tensor for overflow checking.
    - Returns the created tensor.
- **Output**: Returns a pointer to the newly created `ggml_tensor`.
- **Functions called**:
    - [`test_case::add_sentinel`](#test_caseadd_sentinel)
- **See also**: [`test_case`](#test_case)  (Data Structure)


---
#### test\_case::ggml\_new\_tensor\_2d<!-- {{#callable:test_case::ggml_new_tensor_2d}} -->
Creates a new 2D tensor in the specified context with a given type and dimensions.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` in which the tensor will be created.
    - `type`: The data type of the tensor, specified by the `ggml_type` enumeration.
    - `ne0`: The size of the first dimension of the tensor.
    - `ne1`: The size of the second dimension of the tensor.
- **Control Flow**:
    - Calls the `::ggml_new_tensor_2d` function to create a new tensor with the specified context, type, and dimensions.
    - Invokes the [`add_sentinel`](#test_caseadd_sentinel) function to add a sentinel tensor for overflow checking.
    - Returns the newly created tensor.
- **Output**: Returns a pointer to the newly created `ggml_tensor`.
- **Functions called**:
    - [`test_case::add_sentinel`](#test_caseadd_sentinel)
- **See also**: [`test_case`](#test_case)  (Data Structure)


---
#### test\_case::ggml\_new\_tensor\_3d<!-- {{#callable:test_case::ggml_new_tensor_3d}} -->
Creates a new 3D tensor in the specified context with a given type and dimensions.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` in which the tensor will be created.
    - `type`: The type of the tensor, specified by the `ggml_type` enumeration.
    - `ne0`: The size of the first dimension of the tensor.
    - `ne1`: The size of the second dimension of the tensor.
    - `ne2`: The size of the third dimension of the tensor.
- **Control Flow**:
    - Calls the `ggml_new_tensor_3d` function to create a new tensor with the specified dimensions and type.
    - Invokes the [`add_sentinel`](#test_caseadd_sentinel) function to add a sentinel tensor for overflow checking.
    - Returns the newly created tensor.
- **Output**: Returns a pointer to the newly created `ggml_tensor`.
- **Functions called**:
    - [`test_case::add_sentinel`](#test_caseadd_sentinel)
- **See also**: [`test_case`](#test_case)  (Data Structure)


---
#### test\_case::ggml\_new\_tensor\_4d<!-- {{#callable:test_case::ggml_new_tensor_4d}} -->
Creates a new 4-dimensional tensor in the specified context with a sentinel for overflow checking.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` in which the tensor will be created.
    - `type`: The data type of the tensor, specified by the `ggml_type` enumeration.
    - `ne0`: The size of the first dimension of the tensor.
    - `ne1`: The size of the second dimension of the tensor.
    - `ne2`: The size of the third dimension of the tensor.
    - `ne3`: The size of the fourth dimension of the tensor.
- **Control Flow**:
    - Calls the `ggml_new_tensor_4d` function to create a new tensor with the specified dimensions and type.
    - Invokes the [`add_sentinel`](#test_caseadd_sentinel) function to add a sentinel tensor for overflow checking.
    - Returns the newly created tensor.
- **Output**: Returns a pointer to the newly created `ggml_tensor`.
- **Functions called**:
    - [`test_case::add_sentinel`](#test_caseadd_sentinel)
- **See also**: [`test_case`](#test_case)  (Data Structure)


---
#### test\_case::eval<!-- {{#callable:test_case::eval}} -->
The `eval` function evaluates the performance and correctness of operations across two specified backends.
- **Inputs**:
    - `backend1`: The first backend to be evaluated, of type `ggml_backend_t`.
    - `backend2`: The second backend to be evaluated, of type `ggml_backend_t`.
    - `op_name`: A string representing the name of the operation to be evaluated; if it does not match the operation description, the evaluation is skipped.
- **Control Flow**:
    - Sets the mode to `MODE_TEST` to indicate that the function is in testing mode.
    - Initializes the `ggml_context` with parameters for memory allocation.
    - Builds a computation graph using the [`build_graph`](#test_examplebuild_graph) method.
    - If `op_name` is provided and does not match the operation description, the function exits early.
    - Checks if both backends support the operations defined in the graph.
    - If any backend does not support the operations, the function exits early.
    - Allocates memory for the backend tensors.
    - Builds the forward computation graph.
    - Initializes the tensors with random values.
    - Defines a callback function to compare the outputs of the two backends.
    - Compares the outputs of the two backends using the defined callback.
    - Frees the allocated memory and context.
    - Prints the result of the evaluation, indicating success or failure.
- **Output**: Returns a boolean indicating whether the evaluation was successful (true) or not (false).
- **Functions called**:
    - [`ggml_tensor_overhead`](../ggml/src/ggml.c.driver.md#ggml_tensor_overhead)
    - [`ggml_graph_overhead`](../ggml/src/ggml.c.driver.md#ggml_graph_overhead)
    - [`ggml_init`](../ggml/src/ggml.c.driver.md#ggml_init)
    - [`ggml_new_graph`](../ggml/src/ggml.c.driver.md#ggml_new_graph)
    - [`test_case::add_sentinel`](#test_caseadd_sentinel)
    - [`test_example::build_graph`](#test_examplebuild_graph)
    - [`test_case::op_desc`](#test_caseop_desc)
    - [`ggml_free`](../ggml/src/ggml.c.driver.md#ggml_free)
    - [`test_case::vars`](#test_casevars)
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`ggml_backend_supports_op`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_supports_op)
    - [`ggml_backend_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_name)
    - [`ggml_backend_alloc_ctx_tensors`](../ggml/src/ggml-alloc.c.driver.md#ggml_backend_alloc_ctx_tensors)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
    - [`ggml_graph_add_node`](../ggml/src/ggml.c.driver.md#ggml_graph_add_node)
    - [`test_case::initialize_tensors`](#test_caseinitialize_tensors)
    - [`test_case::max_nmse_err`](#test_casemax_nmse_err)
    - [`ggml_backend_tensor_get`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_get)
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)
    - [`tensor_to_float`](#tensor_to_float)
    - [`ggml_op_desc`](../ggml/src/ggml.c.driver.md#ggml_op_desc)
    - [`isinf_or_max`](#isinf_or_max)
    - [`nmse`](#nmse)
    - [`ggml_backend_compare_graph_backend`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_compare_graph_backend)
    - [`ggml_backend_buffer_free`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_free)
- **See also**: [`test_case`](#test_case)  (Data Structure)


---
#### test\_case::eval\_perf<!-- {{#callable:test_case::eval_perf}} -->
Evaluates the performance of a specified operation on a given backend by measuring execution time and memory usage.
- **Inputs**:
    - `backend`: The backend on which the operation will be evaluated, represented as a `ggml_backend_t` type.
    - `op_name`: A string representing the name of the operation to evaluate; if it does not match the operation being built, the function will skip the evaluation.
- **Control Flow**:
    - Sets the mode to performance evaluation.
    - Initializes parameters for the graph and context.
    - Builds the computation graph for the operation.
    - Checks if the operation name matches the provided `op_name` and skips if it does not.
    - Prints the operation description and checks if the backend supports the operation.
    - Allocates memory for tensors on the specified backend.
    - Randomizes the tensors for the operation.
    - Builds the computation graph and performs a warmup run to initialize the backend.
    - Determines the number of runs based on the operation's floating point operations (FLOPs) or memory size.
    - Duplicates the operation in the graph for the determined number of runs.
    - Calculates the total memory required for the operation.
    - Runs the operation for at least one second while accumulating execution time and memory usage.
    - Prints the performance metrics including runs per second and memory bandwidth.
- **Output**: Returns true if the evaluation was successful, otherwise returns false.
- **Functions called**:
    - [`ggml_tensor_overhead`](../ggml/src/ggml.c.driver.md#ggml_tensor_overhead)
    - [`ggml_graph_overhead_custom`](../ggml/src/ggml.c.driver.md#ggml_graph_overhead_custom)
    - [`test_example::build_graph`](#test_examplebuild_graph)
    - [`test_case::op_desc`](#test_caseop_desc)
    - [`test_case::vars`](#test_casevars)
    - [`ggml_backend_supports_op`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_supports_op)
    - [`ggml_backend_alloc_ctx_tensors`](../ggml/src/ggml-alloc.c.driver.md#ggml_backend_alloc_ctx_tensors)
    - [`test_case::initialize_tensors`](#test_caseinitialize_tensors)
    - [`ggml_new_graph_custom`](../ggml/src/ggml.c.driver.md#ggml_new_graph_custom)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
    - [`ggml_backend_graph_compute`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_graph_compute)
    - [`ggml_status_to_string`](../ggml/src/ggml.c.driver.md#ggml_status_to_string)
    - [`ggml_backend_dev_type`](../ggml/include/ggml-backend.h.driver.md#ggml_backend_dev_type)
    - [`test_case::op_flops`](#test_caseop_flops)
    - [`ggml_graph_size`](../ggml/src/ggml.c.driver.md#ggml_graph_size)
    - [`ggml_graph_n_nodes`](../ggml/src/ggml.c.driver.md#ggml_graph_n_nodes)
    - [`test_case::op_size`](#test_caseop_size)
    - [`ggml_graph_add_node`](../ggml/src/ggml.c.driver.md#ggml_graph_add_node)
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)
    - [`ggml_is_view_op`](#ggml_is_view_op)
    - [`ggml_graph_node`](../ggml/src/ggml.c.driver.md#ggml_graph_node)
- **See also**: [`test_case`](#test_case)  (Data Structure)


---
#### test\_case::eval\_grad<!-- {{#callable:test_case::eval_grad}} -->
Evaluates the gradients of a computational graph for a given operation using a specified backend.
- **Inputs**:
    - `backend`: The backend to be used for computation, specified as a `ggml_backend_t` type.
    - `op_name`: A string representing the name of the operation to evaluate; if it is nullptr, all operations are considered.
- **Control Flow**:
    - Sets the mode to gradient evaluation.
    - Initializes parameters for the `ggml_context` and creates two computational graphs.
    - Builds the computational graph using the [`build_graph`](#test_examplebuild_graph) method.
    - Checks if the operation name matches the expected operation or if it is an optimization step, returning early if it does not.
    - Validates the output tensor type and checks if the backend supports the operations for all tensors in the graph.
    - Counts the number of gradients to be evaluated and skips the evaluation if it exceeds a predefined maximum.
    - Calculates the loss and sets up the forward and backward passes for the graph.
    - Computes the gradients using the backend and checks for non-finite values in the computed gradients.
    - Compares the computed gradients with numerically estimated gradients and checks for errors.
- **Output**: Returns true if the gradient evaluation is successful and the computed gradients match the expected values; otherwise, returns false.
- **Functions called**:
    - [`test_case::grad_expect`](#test_casegrad_expect)
    - [`ggml_tensor_overhead`](../ggml/src/ggml.c.driver.md#ggml_tensor_overhead)
    - [`ggml_graph_overhead_custom`](../ggml/src/ggml.c.driver.md#ggml_graph_overhead_custom)
    - [`ggml_new_graph_custom`](../ggml/src/ggml.c.driver.md#ggml_new_graph_custom)
    - [`test_example::build_graph`](#test_examplebuild_graph)
    - [`test_case::op_desc`](#test_caseop_desc)
    - [`test_case::vars`](#test_casevars)
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`ggml_backend_supports_op`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_supports_op)
    - [`ggml_backend_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_name)
    - [`ggml_nelements`](../ggml/src/ggml.c.driver.md#ggml_nelements)
    - [`test_case::grad_nmax`](#test_casegrad_nmax)
    - [`ggml_is_scalar`](../ggml/src/ggml.c.driver.md#ggml_is_scalar)
    - [`ggml_sum`](../ggml/src/ggml.c.driver.md#ggml_sum)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_set_loss`](../ggml/src/ggml.c.driver.md#ggml_set_loss)
    - [`ggml_build_forward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_forward_expand)
    - [`ggml_graph_cpy`](../ggml/src/ggml.c.driver.md#ggml_graph_cpy)
    - [`ggml_build_backward_expand`](../ggml/src/ggml.c.driver.md#ggml_build_backward_expand)
    - [`ggml_graph_n_nodes`](../ggml/src/ggml.c.driver.md#ggml_graph_n_nodes)
    - [`ggml_graph_get_grad`](../ggml/src/ggml.c.driver.md#ggml_graph_get_grad)
    - [`ggml_backend_alloc_ctx_tensors`](../ggml/src/ggml-alloc.c.driver.md#ggml_backend_alloc_ctx_tensors)
    - [`test_case::initialize_tensors`](#test_caseinitialize_tensors)
    - [`ggml_graph_reset`](../ggml/src/ggml.c.driver.md#ggml_graph_reset)
    - [`ggml_backend_graph_compute`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_graph_compute)
    - [`ggml_status_to_string`](../ggml/src/ggml.c.driver.md#ggml_status_to_string)
    - [`tensor_to_float`](#tensor_to_float)
    - [`ggml_op_desc`](../ggml/src/ggml.c.driver.md#ggml_op_desc)
    - [`test_case::grad_eps`](#test_casegrad_eps)
    - [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set)
    - [`ggml_backend_tensor_get`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_get)
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)
    - [`mean_abs_asymm`](#mean_abs_asymm)
    - [`test_case::max_maa_err`](#test_casemax_maa_err)
- **See also**: [`test_case`](#test_case)  (Data Structure)



---
### callback\_userdata<!-- {{#data_structure:test_case::eval::callback_userdata}} -->
- **Type**: `struct`
- **Members**:
    - `ok`: Indicates whether the callback operation was successful.
    - `max_err`: Specifies the maximum allowable error for the callback.
    - `backend1`: Represents the first backend used in the callback.
    - `backend2`: Represents the second backend used in the callback.
- **Description**: The `callback_userdata` struct is designed to hold user data for callback functions, including a success flag, maximum error threshold, and references to two backend types for comparison during operations.


---
### test\_example<!-- {{#data_structure:test_example}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The type of the input tensors.
    - `ne`: The shape of the input tensors.
- **Description**: The `test_example` struct is a derived class from `test_case` that encapsulates the properties and methods necessary to define a test case for a specific GGML operation, including the type and shape of input tensors, and methods to build a compute graph for the operation.
- **Member Functions**:
    - [`test_example::vars`](#test_examplevars)
    - [`test_example::test_example`](#test_exampletest_example)
    - [`test_example::build_graph`](#test_examplebuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_example::vars<!-- {{#callable:test_example::vars}} -->
The `vars` function returns a string representation of the `test_example` class's parameters.
- **Inputs**: None
- **Control Flow**:
    - The function calls the macro `VARS_TO_STR2` with `type` and `ne` as arguments.
    - The `VARS_TO_STR2` macro formats the parameters into a string.
- **Output**: The output is a string that represents the `type` and the shape of the input tensors defined by `ne`.
- **See also**: [`test_example`](#test_example)  (Data Structure)


---
#### test\_example::test\_example<!-- {{#callable:test_example::test_example}} -->
The `test_example` constructor initializes a test case with specified tensor type and shape.
- **Inputs**:
    - `type`: The type of the input tensors, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array representing the shape of the input tensors, defaulting to {10, 5, 4, 3}.
- **Control Flow**:
    - The constructor initializes the member variables `type` and `ne` using an initializer list.
    - If no arguments are provided, default values are used for both `type` and `ne`.
- **Output**: The constructor does not return a value but initializes the `test_example` object with the specified tensor type and shape.
- **See also**: [`test_example`](#test_example)  (Data Structure)


---
#### test\_example::build\_graph<!-- {{#callable:test_example::build_graph}} -->
The `build_graph` function constructs a simple compute graph for testing tensor addition.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and state for tensor operations.
- **Control Flow**:
    - Create two input tensors `a` and `b` using [`ggml_new_tensor`](#test_caseggml_new_tensor), specifying the context, type, dimensions, and shape.
    - Set names for the tensors `a` and `b` for easier debugging.
    - Perform the addition operation on tensors `a` and `b` using [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add), storing the result in `out`.
    - Set the name for the output tensor `out`.
    - Return the output tensor `out`.
- **Output**: Returns a pointer to the output tensor resulting from the addition of tensors `a` and `b`.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
- **See also**: [`test_example`](#test_example)  (Data Structure)



---
### test\_unary<!-- {{#data_structure:test_unary}} -->
- **Type**: `struct`
- **Members**:
    - `op`: The unary operation to be performed.
    - `type`: The data type of the tensor.
    - `ne_a`: An array representing the dimensions of the input tensor.
    - `v`: An integer indicating the view type (1 for non-contiguous).
- **Description**: The `test_unary` struct is a derived class from `test_case` that encapsulates the parameters and methods necessary to test unary operations on tensors, including the operation type, tensor data type, dimensions of the input tensor, and a view flag to indicate whether the tensor is contiguous or not.
- **Member Functions**:
    - [`test_unary::vars`](#test_unaryvars)
    - [`test_unary::test_unary`](#test_unarytest_unary)
    - [`test_unary::build_graph`](#test_unarybuild_graph)
    - [`test_unary::initialize_tensors`](#test_unaryinitialize_tensors)
    - [`test_unary::grad_eps`](#test_unarygrad_eps)
    - [`test_unary::grad_expect`](#test_unarygrad_expect)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_unary::vars<!-- {{#callable:test_unary::vars}} -->
The `vars` method returns a string representation of the object's variables.
- **Inputs**: None
- **Control Flow**:
    - The method directly calls the `VARS_TO_STR3` macro with the member variables `type`, `ne_a`, and `v`.
- **Output**: The output is a string that concatenates the string representations of the `type`, `ne_a`, and `v` variables.
- **See also**: [`test_unary`](#test_unary)  (Data Structure)


---
#### test\_unary::test\_unary<!-- {{#callable:test_unary::test_unary}} -->
The `test_unary` constructor initializes a test case for unary operations with specified parameters.
- **Inputs**:
    - `op`: The unary operation to be tested, represented by the `ggml_unary_op` enum.
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne_a`: An array representing the dimensions of the input tensor, defaulting to {128, 2, 2, 2}.
    - `v`: An integer indicating the view type, defaulting to 0.
- **Control Flow**:
    - The constructor initializes member variables with the provided parameters.
    - The `vars` method is overridden to return a string representation of the test case variables.
    - The `build_graph` method constructs the computation graph for the unary operation, handling both contiguous and non-contiguous tensor views based on the value of `v`.
- **Output**: The constructor does not return a value but initializes the `test_unary` object for further use in testing unary operations.
- **See also**: [`test_unary`](#test_unary)  (Data Structure)


---
#### test\_unary::build\_graph<!-- {{#callable:test_unary::build_graph}} -->
The `build_graph` function constructs a computation graph for a unary operation on a tensor.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for tensor operations.
- **Control Flow**:
    - Check if the unary operation supports gradient computation based on the operation type.
    - If the view flag `v` is set, create a new tensor `a` with modified dimensions and set its name.
    - If the view flag is not set, create a new tensor `a` with the original dimensions.
    - Call the [`ggml_unary`](../ggml/src/ggml.c.driver.md#ggml_unary) function to apply the specified unary operation on tensor `a`.
    - Set the name of the output tensor and return it.
- **Output**: Returns a pointer to the output tensor resulting from applying the unary operation.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_view_4d`](../ggml/src/ggml.c.driver.md#ggml_view_4d)
    - [`ggml_unary`](../ggml/src/ggml.c.driver.md#ggml_unary)
- **See also**: [`test_unary`](#test_unary)  (Data Structure)


---
#### test\_unary::initialize\_tensors<!-- {{#callable:test_unary::initialize_tensors}} -->
The `initialize_tensors` method initializes all tensors in the given `ggml_context` by setting their values uniformly within a specified range.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that contains the tensors to be initialized.
- **Control Flow**:
    - The method starts a loop that retrieves the first tensor in the context using [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor).
    - It continues to loop through all tensors in the context until there are no more tensors (i.e., `t` is NULL).
    - For each tensor, it calls the [`init_tensor_uniform`](#init_tensor_uniform) function to initialize the tensor with values uniformly distributed between -150 and 150.
- **Output**: The method does not return a value; it modifies the tensors in place by initializing their values.
- **Functions called**:
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`init_tensor_uniform`](#init_tensor_uniform)
- **See also**: [`test_unary`](#test_unary)  (Data Structure)


---
#### test\_unary::grad\_eps<!-- {{#callable:test_unary::grad_eps}} -->
The `grad_eps` function returns a constant float value of 15.0f.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a constant value without any conditional statements or loops.
- **Output**: The output is a float value of 15.0f.
- **See also**: [`test_unary`](#test_unary)  (Data Structure)


---
#### test\_unary::grad\_expect<!-- {{#callable:test_unary::grad_expect}} -->
The `grad_expect` function computes the expected gradients for various unary operations.
- **Inputs**:
    - `op`: A constant of type `ggml_unary_op` that specifies the unary operation for which the gradient is being calculated.
- **Control Flow**:
    - The function checks the value of `op` to determine which unary operation is being used.
    - If `op` is `GGML_UNARY_OP_ABS`, it returns a vector containing -1.0f and 1.0f.
    - If `op` is `GGML_UNARY_OP_SGN` or `GGML_UNARY_OP_STEP`, it returns a vector containing 0.0f.
    - If `op` is `GGML_UNARY_OP_RELU`, it returns a vector containing 0.0f and 1.0f.
    - If `op` does not match any of the specified cases, it returns an empty vector.
- **Output**: The function returns a vector of floats representing the expected gradients for the specified unary operation.
- **See also**: [`test_unary`](#test_unary)  (Data Structure)



---
### test\_get\_rows<!-- {{#data_structure:test_get_rows}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `n`: The number of columns.
    - `m`: The number of rows.
    - `r`: The number of rows to retrieve.
    - `b`: The batch size.
    - `v`: Indicates if the source is a non-contiguous view.
- **Description**: The `test_get_rows` struct is designed to facilitate the testing of operations that retrieve specific rows from a tensor, encapsulating parameters such as the tensor's data type, dimensions, and batch size, along with a flag indicating whether the source tensor is a non-contiguous view.
- **Member Functions**:
    - [`test_get_rows::vars`](#test_get_rowsvars)
    - [`test_get_rows::test_get_rows`](#test_get_rowstest_get_rows)
    - [`test_get_rows::build_graph`](#test_get_rowsbuild_graph)
    - [`test_get_rows::initialize_tensors`](#test_get_rowsinitialize_tensors)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_get\_rows::vars<!-- {{#callable:test_get_rows::vars}} -->
The `vars` method returns a string representation of the object's member variables.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR6` with the member variables `type`, `n`, `m`, `r`, `b`, and `v`.
    - The `VARS_TO_STR6` macro constructs a formatted string that represents the values of these member variables.
- **Output**: The method returns a string that contains the formatted representation of the member variables.
- **See also**: [`test_get_rows`](#test_get_rows)  (Data Structure)


---
#### test\_get\_rows::test\_get\_rows<!-- {{#callable:test_get_rows::test_get_rows}} -->
The `test_get_rows` constructor initializes a test case for retrieving specific rows from a tensor.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `n`: The number of columns in the input tensor, defaulting to 10.
    - `m`: The number of rows in the input tensor, defaulting to 5.
    - `r`: The number of rows to retrieve from the input tensor, defaulting to 3.
    - `b`: The batch size, defaulting to 1.
    - `v`: A boolean indicating whether to create a view of the rows, defaulting to false.
- **Control Flow**:
    - The constructor initializes member variables with the provided parameters.
    - It sets the number of columns, rows, rows to get, batch size, and view flag for the test case.
- **Output**: The constructor does not return a value but initializes the `test_get_rows` object with the specified parameters.
- **See also**: [`test_get_rows`](#test_get_rows)  (Data Structure)


---
#### test\_get\_rows::build\_graph<!-- {{#callable:test_get_rows::build_graph}} -->
The `build_graph` function constructs a computational graph for a tensor operation involving input tensors and their configurations.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure that manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a 3D tensor `in` with specified dimensions and type, and names it 'in'.
    - Creates a 2D tensor `rows` for indexing rows, names it 'rows', and optionally creates a view of it if `v` is true.
    - Checks if gradient computation is supported based on the types of `in` and `rows`.
    - If gradients are supported, sets `in` as a parameter tensor.
    - Calls [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows) to extract rows from `in` based on the indices in `rows` and names the output tensor 'out'.
    - Returns the output tensor `out`.
- **Output**: Returns a pointer to the output tensor `out`, which contains the selected rows from the input tensor `in` based on the indices specified in `rows`.
- **Functions called**:
    - [`test_case::ggml_new_tensor_3d`](#test_caseggml_new_tensor_3d)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`test_case::ggml_new_tensor_2d`](#test_caseggml_new_tensor_2d)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_is_matrix`](../ggml/src/ggml.c.driver.md#ggml_is_matrix)
    - [`ggml_is_vector`](../ggml/src/ggml.c.driver.md#ggml_is_vector)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_get_rows`](../ggml/src/ggml.c.driver.md#ggml_get_rows)
- **See also**: [`test_get_rows`](#test_get_rows)  (Data Structure)


---
#### test\_get\_rows::initialize\_tensors<!-- {{#callable:test_get_rows::initialize_tensors}} -->
The `initialize_tensors` function initializes the tensors in the given `ggml_context` by populating them with random data or uniform values based on their type.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that holds the state and data for tensor operations.
- **Control Flow**:
    - Iterates through all tensors in the context using [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor) and [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor).
    - Checks if the tensor type is `GGML_TYPE_I32`.
    - If the tensor is a view operation, it skips initialization for that tensor.
    - Creates a vector of integers with size `r * b` and fills it with random values between 0 and `m`.
    - Sets the tensor data using [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set) with the generated random data.
    - For tensors of other types, it calls [`init_tensor_uniform`](#init_tensor_uniform) to initialize them with uniform values.
- **Output**: The function does not return a value; it modifies the tensors in the provided `ggml_context` directly.
- **Functions called**:
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`ggml_is_view_op`](#ggml_is_view_op)
    - [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set)
    - [`init_tensor_uniform`](#init_tensor_uniform)
- **See also**: [`test_get_rows`](#test_get_rows)  (Data Structure)



---
### test\_get\_rows\_back<!-- {{#data_structure:test_get_rows_back}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `n`: The number of columns.
    - `m`: The number of rows.
    - `r`: The number of rows to retrieve.
    - `b`: The batch size.
    - `v`: Indicates if the source is a non-contiguous view.
- **Description**: The `test_get_rows_back` struct is a derived class from `test_case` that is designed to test the functionality of retrieving rows from a tensor in a backward operation context. It contains parameters that define the dimensions and characteristics of the tensor, including its type, number of rows and columns, the specific rows to retrieve, and whether the source tensor is a view. This struct is used to set up and execute tests that validate the correctness of the row retrieval operation in the context of a neural network's backward pass.
- **Member Functions**:
    - [`test_get_rows_back::vars`](#test_get_rows_backvars)
    - [`test_get_rows_back::test_get_rows_back`](#test_get_rows_backtest_get_rows_back)
    - [`test_get_rows_back::build_graph`](#test_get_rows_backbuild_graph)
    - [`test_get_rows_back::initialize_tensors`](#test_get_rows_backinitialize_tensors)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_get\_rows\_back::vars<!-- {{#callable:test_get_rows_back::vars}} -->
The `vars` method returns a string representation of the class's member variables.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR6` with the member variables `type`, `n`, `m`, `r`, `b`, and `v`.
    - The `VARS_TO_STR6` macro formats these variables into a string representation.
- **Output**: The method returns a string that represents the values of the member variables in a formatted manner.
- **See also**: [`test_get_rows_back`](#test_get_rows_back)  (Data Structure)


---
#### test\_get\_rows\_back::test\_get\_rows\_back<!-- {{#callable:test_get_rows_back::test_get_rows_back}} -->
The `test_get_rows_back` constructor initializes a test case for retrieving rows from a tensor with specified parameters.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `n`: The number of columns in the input tensor, defaulting to 10.
    - `m`: The number of rows in the input tensor, defaulting to 5.
    - `r`: The number of rows to retrieve, defaulting to 3.
    - `b`: The batch size, defaulting to 1.
    - `v`: A boolean indicating whether to use a view for the rows, defaulting to false.
- **Control Flow**:
    - The constructor initializes member variables with the provided parameters.
    - It sets the `type`, `n`, `m`, `r`, `b`, and `v` attributes of the `test_get_rows_back` instance.
- **Output**: This constructor does not return a value but initializes an instance of the `test_get_rows_back` class with the specified parameters.
- **See also**: [`test_get_rows_back`](#test_get_rows_back)  (Data Structure)


---
#### test\_get\_rows\_back::build\_graph<!-- {{#callable:test_get_rows_back::build_graph}} -->
The `build_graph` function constructs a computational graph for a neural network operation.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a 3D tensor `in_forward` with dimensions defined by `n`, `m`, and `b`.
    - Sets the name of `in_forward` to 'in_forward'.
    - Creates a 2D tensor `rows` of type `GGML_TYPE_I32` with dimensions `r` and `b`.
    - Sets the name of `rows` to 'rows'.
    - If the boolean `v` is true, modifies `rows` to be a view of half its original size.
    - Creates a 3D tensor `grad` with dimensions defined by `n`, `r`, and `b`.
    - Sets the name of `grad` to 'grad'.
    - Calls [`ggml_get_rows_back`](../ggml/src/ggml.c.driver.md#ggml_get_rows_back) to compute the output tensor `out` using `grad`, `rows`, and `in_forward`.
    - Sets the name of `out` to 'out'.
- **Output**: Returns the output tensor `out`, which is the result of the backward operation based on the input tensors.
- **Functions called**:
    - [`test_case::ggml_new_tensor_3d`](#test_caseggml_new_tensor_3d)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`test_case::ggml_new_tensor_2d`](#test_caseggml_new_tensor_2d)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_get_rows_back`](../ggml/src/ggml.c.driver.md#ggml_get_rows_back)
- **See also**: [`test_get_rows_back`](#test_get_rows_back)  (Data Structure)


---
#### test\_get\_rows\_back::initialize\_tensors<!-- {{#callable:test_get_rows_back::initialize_tensors}} -->
Initializes tensors in a given `ggml_context` by populating them with random data or uniform values based on their type.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that holds the state and data for tensor operations.
- **Control Flow**:
    - Iterates through all tensors in the context using [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor) and [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor).
    - Checks if the tensor type is `GGML_TYPE_I32`.
    - If the tensor is a view operation, it skips initialization for that tensor.
    - Creates a vector of integers with size `r * b` and fills it with random values between 0 and `m`.
    - Sets the tensor data using [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set) with the generated random data.
    - For tensors of other types, it calls [`init_tensor_uniform`](#init_tensor_uniform) to initialize them with uniform values.
- **Output**: The function does not return a value; it modifies the tensors in the provided `ggml_context` directly.
- **Functions called**:
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`ggml_is_view_op`](#ggml_is_view_op)
    - [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set)
    - [`init_tensor_uniform`](#init_tensor_uniform)
- **See also**: [`test_get_rows_back`](#test_get_rows_back)  (Data Structure)



---
### test\_argmax<!-- {{#data_structure:test_argmax}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
- **Description**: The `test_argmax` struct is a derived class from `test_case` that is designed to test the argmax operation on tensors. It contains a data type and a tensor shape, and it provides methods to build a computation graph for the argmax operation and to initialize tensor values for testing.
- **Member Functions**:
    - [`test_argmax::vars`](#test_argmaxvars)
    - [`test_argmax::test_argmax`](#test_argmaxtest_argmax)
    - [`test_argmax::build_graph`](#test_argmaxbuild_graph)
    - [`test_argmax::initialize_tensors`](#test_argmaxinitialize_tensors)
    - [`test_argmax::max_nmse_err`](#test_argmaxmax_nmse_err)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_argmax::vars<!-- {{#callable:test_argmax::vars}} -->
The `vars` method returns a string representation of the `test_argmax` class's parameters.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR2` with the class's member variables `type` and `ne`.
    - The `VARS_TO_STR2` macro formats these variables into a string representation.
- **Output**: The method outputs a string that represents the `type` and `ne` member variables of the `test_argmax` class.
- **See also**: [`test_argmax`](#test_argmax)  (Data Structure)


---
#### test\_argmax::test\_argmax<!-- {{#callable:test_argmax::test_argmax}} -->
The `test_argmax` constructor initializes a test case for the argmax operation with specified tensor type and dimensions.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array of four integers representing the dimensions of the tensor, defaulting to {10, 100, 1, 1}.
- **Control Flow**:
    - The constructor initializes the member variables `type` and `ne` with the provided arguments.
    - If no arguments are provided, default values are used.
- **Output**: The constructor does not return a value but initializes the `test_argmax` object with the specified tensor type and dimensions.
- **See also**: [`test_argmax`](#test_argmax)  (Data Structure)


---
#### test\_argmax::build\_graph<!-- {{#callable:test_argmax::build_graph}} -->
The `build_graph` method constructs a computation graph for a tensor operation that computes the argmax of a newly created tensor.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for tensor operations.
- **Control Flow**:
    - A new tensor `a` is created using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions.
    - The name of tensor `a` is set to 'a' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - The [`ggml_argmax`](../ggml/src/ggml.c.driver.md#ggml_argmax) function is called with the context and tensor `a` to compute the argmax, resulting in a new tensor `out`.
    - The name of tensor `out` is set to 'out' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - The method returns the tensor `out`.
- **Output**: Returns a pointer to the resulting tensor `out`, which contains the indices of the maximum values from tensor `a`.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_argmax`](../ggml/src/ggml.c.driver.md#ggml_argmax)
- **See also**: [`test_argmax`](#test_argmax)  (Data Structure)


---
#### test\_argmax::initialize\_tensors<!-- {{#callable:test_argmax::initialize_tensors}} -->
Initializes all tensors in a given `ggml_context` with either unique values or uniform random values.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that contains the tensors to be initialized.
- **Control Flow**:
    - A random number generator is initialized using `std::random_device`.
    - The function iterates over all tensors in the context using [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor) and [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor).
    - For each tensor, if its type is `GGML_TYPE_F32`, it initializes the tensor with unique values to avoid ties.
    - For each row in the tensor, a vector of floats is created, filled with sequential indices, shuffled, and then set in the tensor.
    - If the tensor type is not `GGML_TYPE_F32`, it calls [`init_tensor_uniform`](#init_tensor_uniform) to initialize the tensor with uniform random values.
- **Output**: The function does not return a value; it modifies the tensors in the provided `ggml_context` directly.
- **Functions called**:
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`ggml_nrows`](../ggml/src/ggml.c.driver.md#ggml_nrows)
    - [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set)
    - [`init_tensor_uniform`](#init_tensor_uniform)
- **See also**: [`test_argmax`](#test_argmax)  (Data Structure)


---
#### test\_argmax::max\_nmse\_err<!-- {{#callable:test_argmax::max_nmse_err}} -->
The `max_nmse_err` function returns a constant value of 0.0.
- **Inputs**: None
- **Control Flow**:
    - The function does not contain any control flow statements such as loops or conditionals.
    - It directly returns a constant value.
- **Output**: The output is a double value of 0.0, representing the maximum normalized mean squared error.
- **See also**: [`test_argmax`](#test_argmax)  (Data Structure)



---
### test\_count\_equal<!-- {{#data_structure:test_count_equal}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
- **Description**: The `test_count_equal` struct is a derived class from `test_case` that is designed to test the functionality of counting equal elements in tensors. It contains a tensor type and its dimensions, and it overrides methods to build a computation graph for the operation and to represent its variables as a string.
- **Member Functions**:
    - [`test_count_equal::vars`](#test_count_equalvars)
    - [`test_count_equal::test_count_equal`](#test_count_equaltest_count_equal)
    - [`test_count_equal::build_graph`](#test_count_equalbuild_graph)
    - [`test_count_equal::max_nmse_err`](#test_count_equalmax_nmse_err)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_count\_equal::vars<!-- {{#callable:test_count_equal::vars}} -->
The `vars` method returns a string representation of the `test_count_equal` class's member variables.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR2` with `type` and `ne` as arguments.
    - The `VARS_TO_STR2` macro formats the member variables into a string representation.
- **Output**: The method returns a string that represents the `type` and `ne` member variables of the `test_count_equal` class.
- **See also**: [`test_count_equal`](#test_count_equal)  (Data Structure)


---
#### test\_count\_equal::test\_count\_equal<!-- {{#callable:test_count_equal::test_count_equal}} -->
The `test_count_equal` constructor initializes a test case for counting equal elements between two tensors.
- **Inputs**:
    - `type`: The data type of the tensors, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array representing the dimensions of the tensors, defaulting to {4, 500, 1, 1}.
- **Control Flow**:
    - The constructor initializes the member variables `type` and `ne` with the provided arguments.
    - If no arguments are provided, default values are used.
- **Output**: The constructor does not return a value but initializes the `test_count_equal` object with specified tensor type and dimensions.
- **See also**: [`test_count_equal`](#test_count_equal)  (Data Structure)


---
#### test\_count\_equal::build\_graph<!-- {{#callable:test_count_equal::build_graph}} -->
The `build_graph` function constructs a computation graph for a specific test case involving tensor operations.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and resources for tensor operations.
- **Control Flow**:
    - Create a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions, and set its name to 'a'.
    - Compute the argmax of tensor `a` using [`ggml_argmax`](../ggml/src/ggml.c.driver.md#ggml_argmax), storing the result in `a_argmax`, and set its name to 'a_argmax'.
    - Create another tensor `b` similar to `a`, and set its name to 'b'.
    - Compute the argmax of tensor `b` using [`ggml_argmax`](../ggml/src/ggml.c.driver.md#ggml_argmax), storing the result in `b_argmax`, and set its name to 'b_argmax'.
    - Count the number of equal elements between `a_argmax` and `b_argmax` using [`ggml_count_equal`](../ggml/src/ggml.c.driver.md#ggml_count_equal), storing the result in `out`, and set its name to 'out'.
    - Return the output tensor `out`.
- **Output**: Returns a pointer to a `ggml_tensor` that contains the result of counting equal elements between the argmax results of tensors `a` and `b`.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_argmax`](../ggml/src/ggml.c.driver.md#ggml_argmax)
    - [`ggml_count_equal`](../ggml/src/ggml.c.driver.md#ggml_count_equal)
- **See also**: [`test_count_equal`](#test_count_equal)  (Data Structure)


---
#### test\_count\_equal::max\_nmse\_err<!-- {{#callable:test_count_equal::max_nmse_err}} -->
The `max_nmse_err` function returns a constant value of 0.0.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a constant value without any conditions or loops.
- **Output**: The output is a double value of 0.0.
- **See also**: [`test_count_equal`](#test_count_equal)  (Data Structure)



---
### test\_repeat<!-- {{#data_structure:test_repeat}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
    - `nr`: An array representing the repeat counts for each dimension.
- **Description**: The `test_repeat` struct is a derived class from `test_case` that is used to define a test case for the repeat operation in a tensor computation framework. It contains fields for the tensor type, its dimensions, and the repeat counts for each dimension, allowing for the construction of a tensor graph that tests the repeat functionality.
- **Member Functions**:
    - [`test_repeat::vars`](#test_repeatvars)
    - [`test_repeat::op_size`](#test_repeatop_size)
    - [`test_repeat::test_repeat`](#test_repeattest_repeat)
    - [`test_repeat::build_graph`](#test_repeatbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_repeat::vars<!-- {{#callable:test_repeat::vars}} -->
The `vars` method returns a string representation of the object's variables.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the `VARS_TO_STR3` macro with the member variables `type`, `ne`, and `nr`.
    - The `VARS_TO_STR3` macro formats these variables into a string representation.
- **Output**: The method returns a string that represents the values of the `type`, `ne`, and `nr` member variables.
- **See also**: [`test_repeat`](#test_repeat)  (Data Structure)


---
#### test\_repeat::op\_size<!-- {{#callable:test_repeat::op_size}} -->
The `op_size` method calculates the size of a `ggml_tensor` object, doubling its byte size.
- **Inputs**:
    - `t`: A pointer to a `ggml_tensor` whose size in bytes is to be calculated.
- **Control Flow**:
    - The function calls `ggml_nbytes(t)` to get the size in bytes of the tensor `t`.
    - The result is then multiplied by 2 to account for the operation size, and this value is returned.
- **Output**: Returns a `size_t` value representing the size of the tensor in bytes, multiplied by 2.
- **Functions called**:
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)
- **See also**: [`test_repeat`](#test_repeat)  (Data Structure)


---
#### test\_repeat::test\_repeat<!-- {{#callable:test_repeat::test_repeat}} -->
The `test_repeat` constructor initializes a test case for the GGML repeat operation with specified tensor types and dimensions.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array of four integers representing the dimensions of the tensor.
    - `nr`: An array of four integers representing the repeat factors for each dimension.
- **Control Flow**:
    - The constructor initializes the member variables `type`, `ne`, and `nr` with the provided arguments.
    - If no arguments are provided, default values are used for `type`, `ne`, and `nr`.
- **Output**: The constructor does not return a value but initializes an instance of the `test_repeat` class with the specified parameters.
- **See also**: [`test_repeat`](#test_repeat)  (Data Structure)


---
#### test\_repeat::build\_graph<!-- {{#callable:test_repeat::build_graph}} -->
The `build_graph` function constructs a computational graph for a tensor operation in the context of a neural network.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and state for tensor operations.
- **Control Flow**:
    - Creates a 4D tensor `target` using [`ggml_new_tensor_4d`](#test_caseggml_new_tensor_4d) with dimensions based on the product of `ne` and `nr` arrays.
    - Sets the name of the `target` tensor to 'target'.
    - Creates a new tensor `src` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with dimensions defined by `ne`.
    - Marks `src` as a parameter tensor using [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param).
    - Sets the name of the `src` tensor to 'src'.
    - Calls [`ggml_repeat`](../ggml/src/ggml.c.driver.md#ggml_repeat) to create an output tensor `out` by repeating `src` into `target`.
    - Sets the name of the `out` tensor to 'out'.
    - Returns the `out` tensor.
- **Output**: Returns a pointer to the output tensor `out`, which is the result of repeating the `src` tensor into the `target` tensor.
- **Functions called**:
    - [`test_case::ggml_new_tensor_4d`](#test_caseggml_new_tensor_4d)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_repeat`](../ggml/src/ggml.c.driver.md#ggml_repeat)
- **See also**: [`test_repeat`](#test_repeat)  (Data Structure)



---
### test\_repeat\_back<!-- {{#data_structure:test_repeat_back}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
    - `nr`: An array representing the repeat counts for each dimension.
    - `v`: A boolean indicating if the source tensor is a non-contiguous view.
- **Description**: The `test_repeat_back` struct is a derived class from `test_case` that encapsulates the parameters and behavior for testing the repeat-back operation in a tensor computation context. It includes fields for the tensor type, dimensions, repeat counts, and a flag indicating whether the source tensor is a non-contiguous view, facilitating the construction of a computation graph for testing purposes.
- **Member Functions**:
    - [`test_repeat_back::vars`](#test_repeat_backvars)
    - [`test_repeat_back::op_size`](#test_repeat_backop_size)
    - [`test_repeat_back::test_repeat_back`](#test_repeat_backtest_repeat_back)
    - [`test_repeat_back::build_graph`](#test_repeat_backbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_repeat\_back::vars<!-- {{#callable:test_repeat_back::vars}} -->
The `vars` method returns a string representation of the object's variable state.
- **Inputs**: None
- **Control Flow**:
    - The method calls the `VARS_TO_STR4` macro with the member variables `type`, `ne`, `nr`, and `v`.
    - The `VARS_TO_STR4` macro formats these variables into a string representation.
- **Output**: The output is a string that represents the values of the member variables in a formatted manner.
- **See also**: [`test_repeat_back`](#test_repeat_back)  (Data Structure)


---
#### test\_repeat\_back::op\_size<!-- {{#callable:test_repeat_back::op_size}} -->
The `op_size` function calculates the size of a given `ggml_tensor` object, doubling the number of bytes it occupies.
- **Inputs**:
    - `t`: A pointer to a `ggml_tensor` object whose size in bytes is to be calculated.
- **Control Flow**:
    - The function calls `ggml_nbytes(t)` to get the size in bytes of the tensor `t`.
    - The result from `ggml_nbytes(t)` is then multiplied by 2 to account for the operation size.
- **Output**: Returns a `size_t` value representing the total size of the tensor in bytes, multiplied by 2.
- **Functions called**:
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)
- **See also**: [`test_repeat_back`](#test_repeat_back)  (Data Structure)


---
#### test\_repeat\_back::test\_repeat\_back<!-- {{#callable:test_repeat_back::test_repeat_back}} -->
The `test_repeat_back` constructor initializes a test case for the backward operation of a tensor repeat operation.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array of four integers representing the dimensions of the tensor, defaulting to {8, 6, 4, 2}.
    - `nr`: An array of four integers representing the repeat factors for each dimension, defaulting to {2, 2, 2, 2}.
    - `v`: A boolean indicating whether the source tensor is a non-contiguous view, defaulting to false.
- **Control Flow**:
    - The constructor initializes member variables `type`, `ne`, `nr`, and `v` with the provided arguments.
    - The `ne` and `nr` arrays are used to define the shape and repeat factors of the tensor involved in the test.
    - The constructor is part of a larger structure that inherits from `test_case`, which likely includes methods for building and evaluating the test.
- **Output**: The constructor does not return a value but initializes the state of the `test_repeat_back` object for further operations.
- **See also**: [`test_repeat_back`](#test_repeat_back)  (Data Structure)


---
#### test\_repeat\_back::build\_graph<!-- {{#callable:test_repeat_back::build_graph}} -->
The `build_graph` function constructs a computational graph for a tensor operation involving source and target tensors.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a 4D tensor `src` using [`ggml_new_tensor_4d`](#test_caseggml_new_tensor_4d) with dimensions based on the input parameters.
    - Sets the name of the `src` tensor to 'src'.
    - If the boolean flag `v` is true, it performs several assertions to ensure the dimensions of `ne` and `nr` are valid.
    - Calculates new dimensions for `src` based on the values in `nr` and creates a view of `src` if necessary.
    - Creates a new tensor `target` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with dimensions specified by `ne`.
    - Sets the name of the `target` tensor to 'target'.
    - Calls [`ggml_repeat_back`](../ggml/src/ggml.c.driver.md#ggml_repeat_back) to create an output tensor `out` by repeating the `src` tensor to match the `target` tensor.
    - Sets the name of the `out` tensor to 'out'.
    - Returns the `out` tensor.
- **Output**: Returns a pointer to the output tensor `out`, which is the result of the tensor operation.
- **Functions called**:
    - [`test_case::ggml_new_tensor_4d`](#test_caseggml_new_tensor_4d)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_view_4d`](../ggml/src/ggml.c.driver.md#ggml_view_4d)
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_repeat_back`](../ggml/src/ggml.c.driver.md#ggml_repeat_back)
- **See also**: [`test_repeat_back`](#test_repeat_back)  (Data Structure)



---
### test\_dup<!-- {{#data_structure:test_dup}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
    - `permute`: An array indicating the permutation of dimensions.
    - `_use_permute`: A boolean flag indicating whether to use permutation.
- **Description**: The `test_dup` struct is a derived class from `test_case` that is designed to test the duplication of tensors in a computational graph, allowing for optional permutation of dimensions. It contains fields for the tensor type, dimensions, permutation settings, and a flag to indicate if permutation should be applied during the duplication process.
- **Member Functions**:
    - [`test_dup::vars`](#test_dupvars)
    - [`test_dup::test_dup`](#test_duptest_dup)
    - [`test_dup::build_graph`](#test_dupbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_dup::vars<!-- {{#callable:test_dup::vars}} -->
The `vars` method generates a string representation of the object's variables, including type, shape, and permutation information.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - Calls the `VARS_TO_STR2` macro to convert the `type` and `ne` member variables into a string representation.
    - Checks if `_use_permute` is true; if so, appends the permutation information to the string.
    - Returns the constructed string.
- **Output**: Returns a string that represents the object's variables, formatted for easy readability.
- **See also**: [`test_dup`](#test_dup)  (Data Structure)


---
#### test\_dup::test\_dup<!-- {{#callable:test_dup::test_dup}} -->
The `test_dup` constructor initializes a test case for duplicating tensors with optional permutation.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array representing the dimensions of the tensor, defaulting to {10, 10, 20, 1}.
    - `permute`: An array specifying the permutation of dimensions, defaulting to {0, 0, 0, 0}.
- **Control Flow**:
    - The constructor initializes member variables `type`, `ne`, and `permute` with the provided arguments.
    - It calculates `_use_permute` to determine if any permutation is applied based on the sum of the `permute` array.
- **Output**: The constructor does not return a value but initializes the `test_dup` object with the specified parameters.
- **See also**: [`test_dup`](#test_dup)  (Data Structure)


---
#### test\_dup::build\_graph<!-- {{#callable:test_dup::build_graph}} -->
The `build_graph` function constructs a computation graph for a tensor operation, optionally permuting the tensor based on specified dimensions.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure that manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a new tensor `src` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions.
    - Sets parameters for the `src` tensor using [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param).
    - If `_use_permute` is true, the `src` tensor is permuted using [`ggml_permute`](../ggml/src/ggml.c.driver.md#ggml_permute) with the specified permutation indices.
    - Duplicates the (possibly permuted) `src` tensor into `out` using [`ggml_dup`](../ggml/src/ggml.c.driver.md#ggml_dup).
    - Returns the `out` tensor.
- **Output**: Returns a pointer to the output tensor `out`, which is a duplicate of the input tensor `src`, potentially permuted.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_permute`](../ggml/src/ggml.c.driver.md#ggml_permute)
    - [`ggml_dup`](../ggml/src/ggml.c.driver.md#ggml_dup)
- **See also**: [`test_dup`](#test_dup)  (Data Structure)



---
### test\_set<!-- {{#data_structure:test_set}} -->
- **Type**: `struct`
- **Members**:
    - `type_src`: The source tensor type.
    - `type_dst`: The destination tensor type.
    - `ne`: An array representing the dimensions of the tensor.
    - `dim`: An integer representing the dimension to be processed.
- **Description**: The `test_set` struct is a derived class from `test_case` that encapsulates the parameters and methods necessary for testing tensor operations, specifically focusing on the source and destination tensor types, their dimensions, and the processing dimension.
- **Member Functions**:
    - [`test_set::vars`](#test_setvars)
    - [`test_set::op_size`](#test_setop_size)
    - [`test_set::test_set`](#test_settest_set)
    - [`test_set::build_graph`](#test_setbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_set::vars<!-- {{#callable:test_set::vars}} -->
The `vars` method returns a string representation of the variable states in the `test_set` class.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the `VARS_TO_STR4` macro with the member variables `type_src`, `type_dst`, `ne`, and `dim`.
    - The `VARS_TO_STR4` macro formats these variables into a string representation.
- **Output**: The method returns a formatted string that represents the current state of the variables in the `test_set` class.
- **See also**: [`test_set`](#test_set)  (Data Structure)


---
#### test\_set::op\_size<!-- {{#callable:test_set::op_size}} -->
The `op_size` function calculates the total size in bytes of a `ggml_tensor` and its source tensor.
- **Inputs**:
    - `t`: A pointer to a `ggml_tensor` structure whose size is to be calculated.
- **Control Flow**:
    - The function calls `ggml_nbytes(t)` to get the size of the tensor `t`.
    - It then accesses the first source tensor `t->src[0]` and calls [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes) on it to get its size.
    - The total size is computed by adding the sizes of `t` and `t->src[0]`.
- **Output**: Returns the total size in bytes as a `size_t` value.
- **Functions called**:
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)
- **See also**: [`test_set`](#test_set)  (Data Structure)


---
#### test\_set::test\_set<!-- {{#callable:test_set::test_set}} -->
The `test_set` constructor initializes a test case for a tensor operation with specified source and destination types, tensor dimensions, and a dimension for the operation.
- **Inputs**:
    - `type_src`: The source tensor type, defaulting to `GGML_TYPE_F32`.
    - `type_dst`: The destination tensor type, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array representing the dimensions of the tensor, defaulting to {6, 5, 4, 3}.
    - `dim`: An integer representing the dimension along which the operation will be performed, defaulting to 1.
- **Control Flow**:
    - The constructor initializes member variables `type_src`, `type_dst`, `ne`, and `dim` with the provided arguments.
    - If no arguments are provided, default values are used for each member variable.
- **Output**: The constructor does not return a value but initializes an instance of the `test_set` class with the specified parameters.
- **See also**: [`test_set`](#test_set)  (Data Structure)


---
#### test\_set::build\_graph<!-- {{#callable:test_set::build_graph}} -->
The `build_graph` function constructs a computational graph for a tensor operation involving source and destination tensors.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a source tensor `src` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with dimensions defined by `ne`.
    - Sets parameters and names for the `src` tensor.
    - Doubles the dimensions of `ne` to create a destination tensor `dst`.
    - Calculates the offset for the backward pass based on the dimensions of `src` and `dst`.
    - Creates an output tensor `out` by setting the `dst` tensor with the `src` tensor and the calculated offset.
    - Returns the output tensor `out`.
- **Output**: Returns a pointer to the output tensor `out`, which represents the result of the tensor operation.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_set`](../ggml/src/ggml.c.driver.md#ggml_set)
- **See also**: [`test_set`](#test_set)  (Data Structure)



---
### test\_cpy<!-- {{#data_structure:test_cpy}} -->
- **Type**: `struct`
- **Members**:
    - `type_src`: The source tensor type.
    - `type_dst`: The destination tensor type.
    - `ne`: An array representing the dimensions of the tensor.
    - `permute_src`: An array defining the permutation for the source tensor.
    - `permute_dst`: An array defining the permutation for the destination tensor.
    - `_src_use_permute`: A boolean indicating if source permutation is used.
    - `_dst_use_permute`: A boolean indicating if destination permutation is used.
- **Description**: The `test_cpy` struct is designed to facilitate the testing of tensor copy operations in a machine learning context. It inherits from `test_case` and contains fields that specify the source and destination tensor types, their dimensions, and any permutations that should be applied to the tensors during the copy operation. The struct also includes boolean flags to indicate whether permutations are utilized for the source and destination tensors.
- **Member Functions**:
    - [`test_cpy::vars`](#test_cpyvars)
    - [`test_cpy::max_nmse_err`](#test_cpymax_nmse_err)
    - [`test_cpy::op_size`](#test_cpyop_size)
    - [`test_cpy::test_cpy`](#test_cpytest_cpy)
    - [`test_cpy::build_graph`](#test_cpybuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_cpy::vars<!-- {{#callable:test_cpy::vars}} -->
The `vars` method returns a string representation of the class's member variables.
- **Inputs**:
    - `none`: This method does not take any input parameters.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR5` with the member variables `type_src`, `type_dst`, `ne`, `permute_src`, and `permute_dst`.
    - The `VARS_TO_STR5` macro formats these variables into a string representation.
- **Output**: The method returns a string that represents the values of the member variables in a formatted manner.
- **See also**: [`test_cpy`](#test_cpy)  (Data Structure)


---
#### test\_cpy::max\_nmse\_err<!-- {{#callable:test_cpy::max_nmse_err}} -->
The `max_nmse_err` function returns a constant value representing the maximum normalized mean squared error allowed for the test case.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a constant value without any conditional statements or loops.
- **Output**: The output is a double value of 1e-6, which indicates the maximum normalized mean squared error for the test case.
- **See also**: [`test_cpy`](#test_cpy)  (Data Structure)


---
#### test\_cpy::op\_size<!-- {{#callable:test_cpy::op_size}} -->
The `op_size` function calculates the total size in bytes of a `ggml_tensor` and its source tensor.
- **Inputs**:
    - `t`: A pointer to a `ggml_tensor` whose size is to be calculated.
- **Control Flow**:
    - Calls `ggml_nbytes(t)` to get the size of the tensor `t`.
    - Calls `ggml_nbytes(t->src[0])` to get the size of the first source tensor of `t`.
    - Returns the sum of the sizes obtained from the two previous calls.
- **Output**: Returns the total size in bytes of the tensor `t` and its first source tensor.
- **Functions called**:
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)
- **See also**: [`test_cpy`](#test_cpy)  (Data Structure)


---
#### test\_cpy::test\_cpy<!-- {{#callable:test_cpy::test_cpy}} -->
The `test_cpy` constructor initializes a test case for copying tensors with specified source and destination types, dimensions, and permutation options.
- **Inputs**:
    - `type_src`: The data type of the source tensor, defaulting to `GGML_TYPE_F32`.
    - `type_dst`: The data type of the destination tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array representing the dimensions of the tensor, defaulting to {10, 10, 10, 1}.
    - `permute_src`: An array specifying the permutation of the source tensor dimensions, defaulting to {0, 0, 0, 0}.
    - `permute_dst`: An array specifying the permutation of the destination tensor dimensions, defaulting to {0, 0, 0, 0}.
- **Control Flow**:
    - The constructor initializes member variables with the provided parameters or default values.
    - It calculates whether permutation is used for the source and destination tensors based on the permutation arrays.
- **Output**: The constructor does not return a value but initializes the `test_cpy` object with the specified parameters.
- **See also**: [`test_cpy`](#test_cpy)  (Data Structure)


---
#### test\_cpy::build\_graph<!-- {{#callable:test_cpy::build_graph}} -->
The `build_graph` function constructs a computational graph for a tensor operation involving source and destination tensors, with optional permutation.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a new tensor `src` of type `type_src` with dimensions specified by `ne`.
    - Sets parameters for the `src` tensor and names it 'src'.
    - If `_src_use_permute` is true, permutes the `src` tensor and renames it to 'src_permuted'.
    - Creates a new tensor `dst` of type `type_dst` with dimensions matching `src`.
    - Names the `dst` tensor 'dst'.
    - If `_dst_use_permute` is true, permutes the `dst` tensor and renames it to 'dst_permuted'.
    - Copies the contents of `src` to `dst` and names the output tensor 'out'.
    - Returns the output tensor.
- **Output**: Returns a pointer to the output tensor `out`, which contains the copied data from `src` to `dst`.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_permute`](../ggml/src/ggml.c.driver.md#ggml_permute)
    - [`ggml_cpy`](../ggml/src/ggml.c.driver.md#ggml_cpy)
- **See also**: [`test_cpy`](#test_cpy)  (Data Structure)



---
### test\_cont<!-- {{#data_structure:test_cont}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
- **Description**: The `test_cont` structure is a derived class from `test_case` that encapsulates the properties and methods necessary to define a test case for a tensor operation, specifically focusing on the construction of a tensor with a specified type and dimensions, and includes functionality for building a computation graph for the operation.
- **Member Functions**:
    - [`test_cont::vars`](#test_contvars)
    - [`test_cont::test_cont`](#test_conttest_cont)
    - [`test_cont::build_graph`](#test_contbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_cont::vars<!-- {{#callable:test_cont::vars}} -->
The `vars` method returns a string representation of the `test_cont` class's member variables.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR2` with the class's member variables `type` and `ne`.
    - The `VARS_TO_STR2` macro formats these variables into a string representation.
- **Output**: The method outputs a string that represents the `type` and `ne` member variables of the `test_cont` class.
- **See also**: [`test_cont`](#test_cont)  (Data Structure)


---
#### test\_cont::test\_cont<!-- {{#callable:test_cont::test_cont}} -->
The `test_cont` constructor initializes a `test_cont` object with a specified tensor type and dimensions.
- **Inputs**:
    - `type`: An optional parameter of type `ggml_type` that specifies the data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An optional parameter of type `std::array<int64_t, 4>` that specifies the dimensions of the tensor, defaulting to {10, 10, 10, 1}.
- **Control Flow**:
    - The constructor initializes the member variables `type` and `ne` with the provided arguments.
    - If no arguments are provided, default values are used for both `type` and `ne`.
- **Output**: The constructor does not return a value but initializes the `test_cont` object with the specified tensor type and dimensions.
- **See also**: [`test_cont`](#test_cont)  (Data Structure)


---
#### test\_cont::build\_graph<!-- {{#callable:test_cont::build_graph}} -->
The `build_graph` function constructs a computation graph for a tensor operation involving transposition and continuity.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and state for tensor operations.
- **Control Flow**:
    - Creates a new tensor `src` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions.
    - Sets parameters for the `src` tensor using [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param).
    - Names the `src` tensor as 'src' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Transposes the `src` tensor and assigns it back to `src`, renaming it to 'src_transposed'.
    - Creates a new tensor `out` that is a continuous version of `src` using [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont).
    - Names the `out` tensor as 'out' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Returns the `out` tensor.
- **Output**: Returns a pointer to the output tensor `out`, which is a continuous version of the transposed input tensor.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_transpose`](../ggml/src/ggml.c.driver.md#ggml_transpose)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
- **See also**: [`test_cont`](#test_cont)  (Data Structure)



---
### test\_bin\_bcast<!-- {{#data_structure:test_bin_bcast}} -->
- **Type**: `struct`
- **Members**:
    - `op`: A function pointer type for binary operations on tensors.
    - `type`: The data type of the tensors involved in the operation.
    - `ne`: An array representing the dimensions of the tensors.
    - `nr`: An array representing the repeat counts for each dimension.
- **Description**: The `test_bin_bcast` struct is designed to facilitate testing of binary operations on tensors, allowing for broadcasting of dimensions during operations. It includes a function pointer for the operation, the data type of the tensors, and arrays to specify the dimensions and repeat counts for the tensors involved.
- **Member Functions**:
    - [`test_bin_bcast::vars`](#test_bin_bcastvars)
    - [`test_bin_bcast::op_size`](#test_bin_bcastop_size)
    - [`test_bin_bcast::test_bin_bcast`](#test_bin_bcasttest_bin_bcast)
    - [`test_bin_bcast::build_graph`](#test_bin_bcastbuild_graph)
    - [`test_bin_bcast::initialize_tensors`](#test_bin_bcastinitialize_tensors)
    - [`test_bin_bcast::grad_eps`](#test_bin_bcastgrad_eps)
    - [`test_bin_bcast::grad_precise`](#test_bin_bcastgrad_precise)
    - [`test_bin_bcast::max_maa_err`](#test_bin_bcastmax_maa_err)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_bin\_bcast::vars<!-- {{#callable:test_bin_bcast::vars}} -->
The `vars` method returns a string representation of the class's member variables.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR3` with the class's member variables: `type`, `ne`, and `nr`.
    - The `VARS_TO_STR3` macro formats these variables into a string representation.
- **Output**: The method returns a string that represents the values of the member variables in a formatted manner.
- **See also**: [`test_bin_bcast`](#test_bin_bcast)  (Data Structure)


---
#### test\_bin\_bcast::op\_size<!-- {{#callable:test_bin_bcast::op_size}} -->
The `op_size` function calculates the size of a given `ggml_tensor` multiplied by three.
- **Inputs**:
    - `t`: A pointer to a `ggml_tensor` whose size in bytes is to be calculated.
- **Control Flow**:
    - The function calls `ggml_nbytes(t)` to get the size in bytes of the tensor `t`.
    - The result is then multiplied by 3 to compute the final size.
- **Output**: Returns the total size in bytes of the tensor `t` multiplied by three.
- **Functions called**:
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)
- **See also**: [`test_bin_bcast`](#test_bin_bcast)  (Data Structure)


---
#### test\_bin\_bcast::test\_bin\_bcast<!-- {{#callable:test_bin_bcast::test_bin_bcast}} -->
The `test_bin_bcast` constructor initializes a test case for binary broadcasting operations in a neural network context.
- **Inputs**:
    - `op`: A function pointer to the binary operation to be tested, which takes three `ggml_tensor` pointers as arguments.
    - `type`: The data type of the tensors, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array of four integers representing the shape of the input tensors.
    - `nr`: An array of four integers representing the repeat factors for each dimension of the input tensors.
- **Control Flow**:
    - The constructor initializes member variables with the provided parameters.
    - The `op` parameter is used to determine the binary operation to be performed during the test.
    - The `type`, `ne`, and `nr` parameters define the characteristics of the tensors involved in the operation.
- **Output**: The constructor does not return a value but initializes an instance of the `test_bin_bcast` class, which can be used to run tests on binary broadcasting operations.
- **See also**: [`test_bin_bcast`](#test_bin_bcast)  (Data Structure)


---
#### test\_bin\_bcast::build\_graph<!-- {{#callable:test_bin_bcast::build_graph}} -->
The `build_graph` function constructs a computational graph for a binary operation involving two tensors.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a 4D tensor `a` using [`ggml_new_tensor_4d`](#test_caseggml_new_tensor_4d) with dimensions based on the product of `ne` and `nr` arrays.
    - Sets the name of tensor `a` to 'a'.
    - Creates a tensor `b` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with dimensions specified by `ne`.
    - Sets the name of tensor `b` to 'b'.
    - Checks if the operation supports gradients based on the operation type and the shapes of tensors `a` and `b`.
    - If gradients are supported, sets parameters for tensors `a` and `b` using [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param).
    - Performs the specified operation (e.g., addition) on tensors `a` and `b` and stores the result in `out`.
    - Sets the name of the output tensor `out` to 'out'.
    - Returns the output tensor `out`.
- **Output**: Returns a pointer to the resulting tensor `out` after performing the specified operation on tensors `a` and `b`.
- **Functions called**:
    - [`test_case::ggml_new_tensor_4d`](#test_caseggml_new_tensor_4d)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_are_same_shape`](../ggml/src/ggml.c.driver.md#ggml_are_same_shape)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
- **See also**: [`test_bin_bcast`](#test_bin_bcast)  (Data Structure)


---
#### test\_bin\_bcast::initialize\_tensors<!-- {{#callable:test_bin_bcast::initialize_tensors}} -->
The `initialize_tensors` method initializes all tensors in the given `ggml_context` based on the operation type.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that holds the state and data for tensor operations.
- **Control Flow**:
    - Iterates through all tensors in the context using [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor) and [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor).
    - Checks if the operation type (`op`) is either `ggml_mul` or `ggml_div`.
    - If the operation is multiplication or division, it initializes the tensor with a uniform distribution between 0.9 and 1.1 to avoid numerical issues around zero.
    - For other operations, it initializes the tensor with a default uniform distribution.
- **Output**: This function does not return a value; it modifies the tensors in the context directly.
- **Functions called**:
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`init_tensor_uniform`](#init_tensor_uniform)
- **See also**: [`test_bin_bcast`](#test_bin_bcast)  (Data Structure)


---
#### test\_bin\_bcast::grad\_eps<!-- {{#callable:test_bin_bcast::grad_eps}} -->
Calculates the gradient epsilon value based on the operation type and tensor dimensions.
- **Inputs**:
    - `op`: An operation type that determines the calculation method (e.g., multiplication).
    - `ne`: An array of integers representing the dimensions of the tensors involved in the operation.
- **Control Flow**:
    - The function checks if the operation type is `ggml_mul`.
    - If the operation is multiplication, it calculates the product of the dimensions in `ne` and multiplies it by 0.1.
    - If the operation is not multiplication, it simply returns 1.
- **Output**: Returns a float value representing the gradient epsilon, which is either 0.1 times the product of the dimensions or 1.
- **See also**: [`test_bin_bcast`](#test_bin_bcast)  (Data Structure)


---
#### test\_bin\_bcast::grad\_precise<!-- {{#callable:test_bin_bcast::grad_precise}} -->
The `grad_precise` function checks if the operation is division (`ggml_div`) to determine if gradient calculations should be precise.
- **Inputs**:
    - `None`: This function does not take any input arguments.
- **Control Flow**:
    - The function evaluates the member variable `op` to see if it is equal to `ggml_div`.
    - If `op` is equal to `ggml_div`, the function returns `true`, indicating that precise gradient calculations are required.
    - If `op` is not equal to `ggml_div`, the function returns `false`, indicating that precise gradient calculations are not necessary.
- **Output**: The function returns a boolean value: `true` if the operation is division, otherwise `false`.
- **See also**: [`test_bin_bcast`](#test_bin_bcast)  (Data Structure)


---
#### test\_bin\_bcast::max\_maa\_err<!-- {{#callable:test_bin_bcast::max_maa_err}} -->
The `max_maa_err` function returns the maximum allowable absolute asymmetry error based on the operation type.
- **Inputs**: None
- **Control Flow**:
    - The function checks the value of the `op` member variable.
    - If `op` is equal to `ggml_add`, it returns 1e-4.
    - Otherwise, it returns 1e-3.
- **Output**: The output is a double representing the maximum allowable absolute asymmetry error, which is either 1e-4 or 1e-3 depending on the operation type.
- **See also**: [`test_bin_bcast`](#test_bin_bcast)  (Data Structure)



---
### test\_add1<!-- {{#data_structure:test_add1}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
- **Description**: The `test_add1` struct is a derived class from `test_case` that defines a test case for a specific tensor operation in the GGML library, encapsulating the tensor type and its dimensions, and providing functionality to build a computation graph for the addition operation.
- **Member Functions**:
    - [`test_add1::vars`](#test_add1vars)
    - [`test_add1::test_add1`](#test_add1test_add1)
    - [`test_add1::build_graph`](#test_add1build_graph)
    - [`test_add1::grad_eps`](#test_add1grad_eps)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_add1::vars<!-- {{#callable:test_add1::vars}} -->
The `vars` method returns a string representation of the `test_add1` class's parameters.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR2` with the class's member variables `type` and `ne`.
    - The `VARS_TO_STR2` macro formats these variables into a string representation.
- **Output**: The method outputs a string that represents the `type` and `ne` member variables of the `test_add1` class.
- **See also**: [`test_add1`](#test_add1)  (Data Structure)


---
#### test\_add1::test\_add1<!-- {{#callable:test_add1::test_add1}} -->
The `test_add1` constructor initializes a test case for the addition operation with specified tensor type and dimensions.
- **Inputs**:
    - `type`: The data type of the tensors, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array of four integers representing the dimensions of the tensor, defaulting to {10, 5, 4, 3}.
- **Control Flow**:
    - The constructor initializes the member variables `type` and `ne` with the provided arguments or default values.
    - The `vars` method is overridden to return a string representation of the `type` and `ne` for debugging purposes.
- **Output**: The constructor does not return a value but initializes the `test_add1` object with the specified tensor type and dimensions.
- **See also**: [`test_add1`](#test_add1)  (Data Structure)


---
#### test\_add1::build\_graph<!-- {{#callable:test_add1::build_graph}} -->
The `build_graph` function constructs a computational graph for a specific tensor operation.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which holds the context for tensor operations.
- **Control Flow**:
    - Creates a new tensor `a` with 4 dimensions using [`ggml_new_tensor`](#test_caseggml_new_tensor) and sets its parameters and name.
    - Creates a new 1D tensor `b` using [`ggml_new_tensor_1d`](#test_caseggml_new_tensor_1d), sets its name, but does not set its parameters.
    - Computes the output tensor `out` by adding tensor `a` and tensor `b` using [`ggml_add1`](../ggml/src/ggml.c.driver.md#ggml_add1).
    - Sets the name of the output tensor `out`.
- **Output**: Returns the output tensor `out`, which is the result of adding tensor `a` and tensor `b`.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`test_case::ggml_new_tensor_1d`](#test_caseggml_new_tensor_1d)
    - [`ggml_add1`](../ggml/src/ggml.c.driver.md#ggml_add1)
- **See also**: [`test_add1`](#test_add1)  (Data Structure)


---
#### test\_add1::grad\_eps<!-- {{#callable:test_add1::grad_eps}} -->
Calculates the gradient epsilon value based on the dimensions of the tensor.
- **Inputs**: None
- **Control Flow**:
    - The function directly computes the product of the first four elements of the `ne` array.
    - The result is multiplied by 0.1f.
- **Output**: Returns a float value representing the computed gradient epsilon.
- **See also**: [`test_add1`](#test_add1)  (Data Structure)



---
### test\_scale<!-- {{#data_structure:test_scale}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
    - `scale`: A scaling factor applied to the tensor.
- **Description**: The `test_scale` struct is a derived class from `test_case` that encapsulates the properties and behavior necessary to test scaling operations on tensors, including the tensor's data type, its dimensions, and a scaling factor.
- **Member Functions**:
    - [`test_scale::vars`](#test_scalevars)
    - [`test_scale::test_scale`](#test_scaletest_scale)
    - [`test_scale::build_graph`](#test_scalebuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_scale::vars<!-- {{#callable:test_scale::vars}} -->
The `vars` method returns a string representation of the object's variables.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR3` with the member variables `type`, `ne`, and `scale`.
    - The result of the macro call is returned as a string.
- **Output**: The output is a string that represents the values of the member variables `type`, `ne`, and `scale`.
- **See also**: [`test_scale`](#test_scale)  (Data Structure)


---
#### test\_scale::test\_scale<!-- {{#callable:test_scale::test_scale}} -->
The `test_scale` constructor initializes a test case for scaling operations with specified tensor type, dimensions, and scale factor.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array of four integers representing the dimensions of the tensor, defaulting to {10, 10, 10, 10}.
    - `scale`: A float representing the scaling factor, defaulting to 2.0f.
- **Control Flow**:
    - The constructor initializes the member variables `type`, `ne`, and `scale` using an initializer list.
    - The default values for `type`, `ne`, and `scale` are provided, allowing for flexibility in creating instances of `test_scale`.
- **Output**: The constructor does not return a value but initializes an instance of the `test_scale` class with the specified parameters.
- **See also**: [`test_scale`](#test_scale)  (Data Structure)


---
#### test\_scale::build\_graph<!-- {{#callable:test_scale::build_graph}} -->
`build_graph` constructs a computation graph for a scaling operation on a tensor.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for tensor operations.
- **Control Flow**:
    - Create a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type, dimensions, and data.
    - Set the tensor `a` as a parameter using [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param).
    - Assign a name 'a' to the tensor `a` using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Scale the tensor `a` by a specified factor using [`ggml_scale`](../ggml/src/ggml.c.driver.md#ggml_scale), resulting in a new tensor `out`.
    - Assign a name 'out' to the tensor `out` using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Return the tensor `out`.
- **Output**: Returns a pointer to the output tensor `out`, which contains the scaled values of the input tensor `a`.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_scale`](../ggml/src/ggml.c.driver.md#ggml_scale)
- **See also**: [`test_scale`](#test_scale)  (Data Structure)



---
### test\_silu\_back<!-- {{#data_structure:test_silu_back}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
    - `eps`: A small value used for numerical stability.
- **Description**: The `test_silu_back` struct is a derived class from `test_case` designed to test the backward pass of the SiLU (Sigmoid Linear Unit) activation function in a neural network. It contains a tensor type, dimensions for the tensor, and a small epsilon value for numerical stability during gradient calculations.
- **Member Functions**:
    - [`test_silu_back::vars`](#test_silu_backvars)
    - [`test_silu_back::test_silu_back`](#test_silu_backtest_silu_back)
    - [`test_silu_back::build_graph`](#test_silu_backbuild_graph)
    - [`test_silu_back::grad_precise`](#test_silu_backgrad_precise)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_silu\_back::vars<!-- {{#callable:test_silu_back::vars}} -->
The `vars` method returns a string representation of the object's member variables.
- **Inputs**:
    - `none`: This method does not take any input parameters.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR3` with the member variables `type`, `ne`, and `eps`.
    - The `VARS_TO_STR3` macro formats these variables into a string representation.
- **Output**: The method returns a string that represents the values of the member variables `type`, `ne`, and `eps`.
- **See also**: [`test_silu_back`](#test_silu_back)  (Data Structure)


---
#### test\_silu\_back::test\_silu\_back<!-- {{#callable:test_silu_back::test_silu_back}} -->
The `test_silu_back` constructor initializes a test case for the backward pass of the SiLU activation function.
- **Inputs**:
    - `type`: The data type of the tensors, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array representing the dimensions of the tensor, defaulting to {64, 5, 4, 3}.
    - `eps`: A small float value used for numerical stability, defaulting to 1e-6f.
- **Control Flow**:
    - The constructor initializes the member variables `type`, `ne`, and `eps` with the provided arguments or default values.
    - The `test_silu_back` class inherits from `test_case`, which likely provides additional functionality for running tests.
- **Output**: The constructor does not return a value but initializes the state of the `test_silu_back` object.
- **See also**: [`test_silu_back`](#test_silu_back)  (Data Structure)


---
#### test\_silu\_back::build\_graph<!-- {{#callable:test_silu_back::build_graph}} -->
The `build_graph` method constructs a computational graph for a specific neural network operation.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a new tensor `a` of specified type and dimensions using [`ggml_new_tensor`](#test_caseggml_new_tensor).
    - Sets the name of tensor `a` to 'a' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Creates another tensor `grad` of the same type and dimensions as `a`.
    - Sets the name of tensor `grad` to 'grad'.
    - Calls [`ggml_silu_back`](../ggml/src/ggml.c.driver.md#ggml_silu_back) to compute the output tensor `out` using `a` and `grad`.
    - Sets the name of tensor `out` to 'out'.
    - Returns the output tensor `out`.
- **Output**: Returns a pointer to the output tensor `out`, which contains the result of the `silu` operation applied to `a` with respect to `grad`.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_silu_back`](../ggml/src/ggml.c.driver.md#ggml_silu_back)
- **See also**: [`test_silu_back`](#test_silu_back)  (Data Structure)


---
#### test\_silu\_back::grad\_precise<!-- {{#callable:test_silu_back::grad_precise}} -->
The `grad_precise` method in the `test_silu_back` class always returns true, indicating that a precise gradient estimation is desired.
- **Inputs**: None
- **Control Flow**:
    - The method does not contain any control flow statements such as conditionals or loops.
    - It directly returns a boolean value.
- **Output**: The output is a boolean value, specifically 'true'.
- **See also**: [`test_silu_back`](#test_silu_back)  (Data Structure)



---
### test\_norm<!-- {{#data_structure:test_norm}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
    - `v`: A boolean indicating if the tensor is a non-contiguous view.
    - `eps`: A small value used for numerical stability in calculations.
- **Description**: The `test_norm` struct is a derived class from `test_case` that encapsulates the parameters and methods necessary to perform normalization tests on tensors, including their type, dimensions, view status, and a small epsilon value for stability.
- **Member Functions**:
    - [`test_norm::vars`](#test_normvars)
    - [`test_norm::test_norm`](#test_normtest_norm)
    - [`test_norm::build_graph`](#test_normbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_norm::vars<!-- {{#callable:test_norm::vars}} -->
The `vars` method returns a string representation of the object's member variables.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR4` with the member variables `type`, `ne`, `v`, and `eps`.
    - The `VARS_TO_STR4` macro formats these variables into a string representation.
- **Output**: The method outputs a string that represents the values of the member variables in a formatted manner.
- **See also**: [`test_norm`](#test_norm)  (Data Structure)


---
#### test\_norm::test\_norm<!-- {{#callable:test_norm::test_norm}} -->
The `test_norm` constructor initializes a test case for normalizing tensors with specified parameters.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array of four integers representing the dimensions of the tensor to be normalized, defaulting to {64, 5, 4, 3}.
    - `v`: A boolean indicating whether the tensor is a non-contiguous view, defaulting to false.
    - `eps`: A small float value used to prevent division by zero during normalization, defaulting to 1e-6f.
- **Control Flow**:
    - The constructor initializes the member variables `type`, `ne`, `v`, and `eps` with the provided arguments.
    - The member variables are used to configure the normalization behavior in the `build_graph` method.
- **Output**: The constructor does not return a value but initializes an instance of the `test_norm` class with the specified parameters.
- **See also**: [`test_norm`](#test_norm)  (Data Structure)


---
#### test\_norm::build\_graph<!-- {{#callable:test_norm::build_graph}} -->
The `build_graph` function constructs a computational graph for a tensor operation involving normalization.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages memory and resources for tensor operations.
- **Control Flow**:
    - A new tensor `a` is created using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions.
    - The tensor `a` is named 'a'.
    - If the boolean flag `v` is true, a view of tensor `a` is created with half the dimensions and named 'view of a'.
    - The tensor `out` is computed by normalizing tensor `a` using [`ggml_norm`](../ggml/src/ggml.c.driver.md#ggml_norm) with a specified epsilon value.
    - The tensor `out` is named 'out'.
- **Output**: Returns a pointer to the output tensor `out`, which contains the normalized values.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_view_4d`](../ggml/src/ggml.c.driver.md#ggml_view_4d)
    - [`ggml_norm`](../ggml/src/ggml.c.driver.md#ggml_norm)
- **See also**: [`test_norm`](#test_norm)  (Data Structure)



---
### test\_rms\_norm<!-- {{#data_structure:test_rms_norm}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
    - `v`: A boolean indicating whether the tensor is a non-contiguous view.
    - `eps`: A small constant used for numerical stability.
- **Description**: The `test_rms_norm` struct is a derived class from `test_case` that encapsulates the parameters and behavior for testing the RMS normalization operation in a tensor computation context. It includes attributes for the tensor type, dimensions, view status, and a small epsilon value for numerical stability, along with methods for building the computation graph and initializing tensors.
- **Member Functions**:
    - [`test_rms_norm::vars`](#test_rms_normvars)
    - [`test_rms_norm::test_rms_norm`](#test_rms_normtest_rms_norm)
    - [`test_rms_norm::build_graph`](#test_rms_normbuild_graph)
    - [`test_rms_norm::initialize_tensors`](#test_rms_norminitialize_tensors)
    - [`test_rms_norm::grad_eps`](#test_rms_normgrad_eps)
    - [`test_rms_norm::grad_precise`](#test_rms_normgrad_precise)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_rms\_norm::vars<!-- {{#callable:test_rms_norm::vars}} -->
The `vars` method returns a string representation of the object's member variables.
- **Inputs**:
    - `none`: This method does not take any input parameters.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR4` with the member variables `type`, `ne`, `v`, and `eps`.
    - The `VARS_TO_STR4` macro formats these variables into a string representation.
- **Output**: The method returns a string that represents the values of the member variables in a formatted manner.
- **See also**: [`test_rms_norm`](#test_rms_norm)  (Data Structure)


---
#### test\_rms\_norm::test\_rms\_norm<!-- {{#callable:test_rms_norm::test_rms_norm}} -->
The `test_rms_norm` constructor initializes a test case for RMS normalization with specified tensor type, dimensions, view flag, and epsilon value.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array of four integers representing the dimensions of the tensor, defaulting to {64, 5, 4, 3}.
    - `v`: A boolean flag indicating whether the tensor is a non-contiguous view, defaulting to false.
    - `eps`: A small float value used for numerical stability in the RMS normalization, defaulting to 1e-6f.
- **Control Flow**:
    - The constructor initializes the member variables `type`, `ne`, `v`, and `eps` with the provided arguments.
    - If no arguments are provided, default values are used for each member variable.
- **Output**: The constructor does not return a value but initializes an instance of the `test_rms_norm` class with the specified parameters.
- **See also**: [`test_rms_norm`](#test_rms_norm)  (Data Structure)


---
#### test\_rms\_norm::build\_graph<!-- {{#callable:test_rms_norm::build_graph}} -->
The `build_graph` function constructs a computational graph for a tensor operation involving normalization.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with specified type and dimensions.
    - Sets the tensor `a` as a parameter using [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param).
    - If the boolean flag `v` is true, creates a view of tensor `a` with half the dimensions and sets its name.
    - Calls [`ggml_rms_norm`](../ggml/src/ggml.c.driver.md#ggml_rms_norm) to apply RMS normalization on tensor `a` and stores the result in `out`.
    - Sets the name of the output tensor `out` and returns it.
- **Output**: Returns a pointer to the output tensor `out`, which contains the result of the RMS normalization operation.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_view_4d`](../ggml/src/ggml.c.driver.md#ggml_view_4d)
    - [`ggml_rms_norm`](../ggml/src/ggml.c.driver.md#ggml_rms_norm)
- **See also**: [`test_rms_norm`](#test_rms_norm)  (Data Structure)


---
#### test\_rms\_norm::initialize\_tensors<!-- {{#callable:test_rms_norm::initialize_tensors}} -->
The `initialize_tensors` function initializes all tensors in a given `ggml_context` with uniform random values between -10 and 10.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that contains the state and information about the tensors to be initialized.
- **Control Flow**:
    - The function starts a loop that retrieves the first tensor from the context using [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor).
    - It continues to loop through all tensors in the context using [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor) until there are no more tensors (i.e., until `t` is NULL).
    - For each tensor `t`, it calls the [`init_tensor_uniform`](#init_tensor_uniform) function to initialize the tensor with random values in the range of -10 to 10.
- **Output**: The function does not return a value; it modifies the tensors in place by initializing them with random values.
- **Functions called**:
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`init_tensor_uniform`](#init_tensor_uniform)
- **See also**: [`test_rms_norm`](#test_rms_norm)  (Data Structure)


---
#### test\_rms\_norm::grad\_eps<!-- {{#callable:test_rms_norm::grad_eps}} -->
The `grad_eps` function returns a constant gradient epsilon value of 1.0f.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a constant value without any conditional statements or loops.
- **Output**: The output is a float value of 1.0f, representing the gradient epsilon.
- **See also**: [`test_rms_norm`](#test_rms_norm)  (Data Structure)


---
#### test\_rms\_norm::grad\_precise<!-- {{#callable:test_rms_norm::grad_precise}} -->
The `grad_precise` function returns a boolean value indicating whether to use a precise gradient estimation method.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the boolean value `true` without any conditions or loops.
- **Output**: The output is a boolean value, specifically `true`.
- **See also**: [`test_rms_norm`](#test_rms_norm)  (Data Structure)



---
### test\_rms\_norm\_back<!-- {{#data_structure:test_rms_norm_back}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
    - `eps`: A small constant used for numerical stability.
- **Description**: The `test_rms_norm_back` struct is designed to represent a test case for the backward pass of the RMS normalization operation in a neural network. It inherits from the `test_case` base class and contains members that define the tensor type, its dimensions, and a small epsilon value for numerical stability. This struct is used to build a computational graph for testing the correctness of the RMS normalization backward operation.
- **Member Functions**:
    - [`test_rms_norm_back::vars`](#test_rms_norm_backvars)
    - [`test_rms_norm_back::test_rms_norm_back`](#test_rms_norm_backtest_rms_norm_back)
    - [`test_rms_norm_back::build_graph`](#test_rms_norm_backbuild_graph)
    - [`test_rms_norm_back::initialize_tensors`](#test_rms_norm_backinitialize_tensors)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_rms\_norm\_back::vars<!-- {{#callable:test_rms_norm_back::vars}} -->
The `vars` method returns a string representation of the class's member variables.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR3` with the member variables `type`, `ne`, and `eps`.
    - The `VARS_TO_STR3` macro formats these variables into a string representation.
- **Output**: The method returns a formatted string that represents the values of the member variables `type`, `ne`, and `eps`.
- **See also**: [`test_rms_norm_back`](#test_rms_norm_back)  (Data Structure)


---
#### test\_rms\_norm\_back::test\_rms\_norm\_back<!-- {{#callable:test_rms_norm_back::test_rms_norm_back}} -->
The `test_rms_norm_back` constructor initializes a test case for the backward pass of the RMS normalization operation.
- **Inputs**:
    - `type`: The data type of the tensors, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array representing the dimensions of the tensors, defaulting to {64, 5, 4, 3}.
    - `eps`: A small float value used for numerical stability, defaulting to 1e-6f.
- **Control Flow**:
    - The constructor initializes the member variables `type`, `ne`, and `eps` with the provided arguments or default values.
    - The `vars` method is overridden to return a string representation of the test case parameters.
- **Output**: The constructor does not return a value but initializes the `test_rms_norm_back` object with the specified parameters.
- **See also**: [`test_rms_norm_back`](#test_rms_norm_back)  (Data Structure)


---
#### test\_rms\_norm\_back::build\_graph<!-- {{#callable:test_rms_norm_back::build_graph}} -->
The `build_graph` function constructs a computational graph for the RMS normalization operation.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for the graph.
- **Control Flow**:
    - Creates a new tensor `a` of type `type` with dimensions specified by `ne` and assigns it the name 'a'.
    - Creates another tensor `b` similarly to `a` and assigns it the name 'b'.
    - Calls the [`ggml_rms_norm_back`](../ggml/src/ggml.c.driver.md#ggml_rms_norm_back) function with `ctx`, `a`, `b`, and `eps` to compute the RMS normalization and stores the result in `out`.
    - Sets the name of the output tensor `out` to 'out'.
    - Returns the output tensor `out`.
- **Output**: Returns a pointer to the output tensor resulting from the RMS normalization operation.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_rms_norm_back`](../ggml/src/ggml.c.driver.md#ggml_rms_norm_back)
- **See also**: [`test_rms_norm_back`](#test_rms_norm_back)  (Data Structure)


---
#### test\_rms\_norm\_back::initialize\_tensors<!-- {{#callable:test_rms_norm_back::initialize_tensors}} -->
The `initialize_tensors` method initializes all tensors in a given `ggml_context` with uniform random values between -10 and 10.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that holds the state of the tensor context.
- **Control Flow**:
    - The method iterates over all tensors in the context using a loop that retrieves the first tensor and continues until there are no more tensors.
    - For each tensor retrieved, it calls the [`init_tensor_uniform`](#init_tensor_uniform) function to initialize the tensor with random values in the range of -10 to 10.
- **Output**: This method does not return a value; it modifies the tensors in place within the provided `ggml_context`.
- **Functions called**:
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`init_tensor_uniform`](#init_tensor_uniform)
- **See also**: [`test_rms_norm_back`](#test_rms_norm_back)  (Data Structure)



---
### test\_ssm\_conv<!-- {{#data_structure:test_ssm_conv}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type used for the convolution operation.
    - `ne_a`: An array representing the dimensions of the first input tensor.
    - `ne_b`: An array representing the dimensions of the second input tensor.
- **Description**: The `test_ssm_conv` struct is a derived class from `test_case` that encapsulates the parameters and methods necessary to perform and test a specific type of convolution operation (SSM convolution) using two input tensors. It includes the data type for the operation and the dimensions of the input tensors, along with a method to build the computation graph for the convolution.
- **Member Functions**:
    - [`test_ssm_conv::vars`](#test_ssm_convvars)
    - [`test_ssm_conv::test_ssm_conv`](#test_ssm_convtest_ssm_conv)
    - [`test_ssm_conv::build_graph`](#test_ssm_convbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_ssm\_conv::vars<!-- {{#callable:test_ssm_conv::vars}} -->
The `vars` method returns a string representation of the class's member variables.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR3` with the class's member variables: `type`, `ne_a`, and `ne_b`.
    - The `VARS_TO_STR3` macro formats these variables into a string representation.
- **Output**: The method returns a string that represents the values of the member variables in a formatted manner.
- **See also**: [`test_ssm_conv`](#test_ssm_conv)  (Data Structure)


---
#### test\_ssm\_conv::test\_ssm\_conv<!-- {{#callable:test_ssm_conv::test_ssm_conv}} -->
The `test_ssm_conv` constructor initializes a test case for a specific convolution operation with specified tensor types and shapes.
- **Inputs**:
    - `type`: The data type of the tensors used in the convolution operation, defaulting to `GGML_TYPE_F32`.
    - `ne_a`: An array representing the shape of the first tensor, defaulting to {10, 10, 10, 1}.
    - `ne_b`: An array representing the shape of the second tensor, defaulting to {3, 3, 1, 1}.
- **Control Flow**:
    - The constructor initializes the member variables `type`, `ne_a`, and `ne_b` with the provided arguments or default values.
    - These member variables are used to define the characteristics of the convolution operation that will be tested.
- **Output**: The constructor does not return a value but sets up the state of the `test_ssm_conv` instance for further testing of the convolution operation.
- **See also**: [`test_ssm_conv`](#test_ssm_conv)  (Data Structure)


---
#### test\_ssm\_conv::build\_graph<!-- {{#callable:test_ssm_conv::build_graph}} -->
The `build_graph` function constructs a computation graph for a specific operation involving two tensors.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure that manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified `ctx`, `type`, and dimensions defined by `ne_a`.
    - Creates another tensor `b` similarly using [`ggml_new_tensor`](#test_caseggml_new_tensor) with dimensions defined by `ne_b`.
    - Calls [`ggml_ssm_conv`](../ggml/src/ggml.c.driver.md#ggml_ssm_conv) to perform a specific convolution operation on tensors `a` and `b`, storing the result in `out`.
    - Returns the resulting tensor `out`.
- **Output**: Returns a pointer to the resulting tensor `out` which contains the output of the convolution operation.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_ssm_conv`](../ggml/src/ggml.c.driver.md#ggml_ssm_conv)
- **See also**: [`test_ssm_conv`](#test_ssm_conv)  (Data Structure)



---
### test\_ssm\_scan<!-- {{#data_structure:test_ssm_scan}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type used for the tensors.
    - `d_state`: The dimension of the state.
    - `d_inner`: The dimension of the inner layer.
    - `n_seq_tokens`: The number of sequence tokens.
    - `n_seqs`: The number of sequences.
- **Description**: The `test_ssm_scan` struct is a derived type from `test_case` that encapsulates parameters for testing a state-space model scan operation, including dimensions for state, inner layers, and sequence tokens, along with the data type used for the tensors.
- **Member Functions**:
    - [`test_ssm_scan::vars`](#test_ssm_scanvars)
    - [`test_ssm_scan::test_ssm_scan`](#test_ssm_scantest_ssm_scan)
    - [`test_ssm_scan::build_graph`](#test_ssm_scanbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_ssm\_scan::vars<!-- {{#callable:test_ssm_scan::vars}} -->
The `vars` method returns a string representation of the member variables of the `test_ssm_scan` class.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR5` with the member variables `type`, `d_state`, `d_inner`, `n_seq_tokens`, and `n_seqs`.
    - The `VARS_TO_STR5` macro constructs a string representation of these variables.
- **Output**: The method returns a string that represents the values of the member variables of the `test_ssm_scan` class.
- **See also**: [`test_ssm_scan`](#test_ssm_scan)  (Data Structure)


---
#### test\_ssm\_scan::test\_ssm\_scan<!-- {{#callable:test_ssm_scan::test_ssm_scan}} -->
The `test_ssm_scan` constructor initializes a test case for a state-space model scan with specified parameters.
- **Inputs**:
    - `type`: The data type of the tensors used in the test, defaulting to `GGML_TYPE_F32`.
    - `d_state`: The dimensionality of the state representation, defaulting to 32.
    - `d_inner`: The dimensionality of the inner representation, defaulting to 32.
    - `n_seq_tokens`: The number of sequence tokens to process, defaulting to 32.
    - `n_seqs`: The number of sequences to process, defaulting to 32.
- **Control Flow**:
    - The constructor initializes member variables with the provided parameters or defaults.
    - It sets up the test case for a state-space model scan operation.
- **Output**: The constructor does not return a value but initializes the `test_ssm_scan` object with the specified parameters.
- **See also**: [`test_ssm_scan`](#test_ssm_scan)  (Data Structure)


---
#### test\_ssm\_scan::build\_graph<!-- {{#callable:test_ssm_scan::build_graph}} -->
The `build_graph` function constructs a computational graph for a specific model by creating and initializing multiple `ggml_tensor` objects.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates several `ggml_tensor` objects with specified dimensions and types using [`ggml_new_tensor`](#test_caseggml_new_tensor).
    - Calls [`ggml_ssm_scan`](../ggml/src/ggml.c.driver.md#ggml_ssm_scan) to perform a specific operation on the created tensors, which likely represents a scan operation in a state-space model.
    - Returns the output tensor from the [`ggml_ssm_scan`](../ggml/src/ggml.c.driver.md#ggml_ssm_scan) function.
- **Output**: Returns a pointer to a `ggml_tensor` that represents the output of the scan operation performed on the input tensors.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_ssm_scan`](../ggml/src/ggml.c.driver.md#ggml_ssm_scan)
- **See also**: [`test_ssm_scan`](#test_ssm_scan)  (Data Structure)



---
### test\_rwkv\_wkv6<!-- {{#data_structure:test_rwkv_wkv6}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `head_count`: The number of attention heads.
    - `head_size`: The size of each attention head.
    - `n_seq_tokens`: The number of sequence tokens.
    - `n_seqs`: The number of sequences.
- **Description**: The `test_rwkv_wkv6` struct is a derived class from `test_case` that encapsulates parameters for testing a specific RWKV model configuration, including the tensor type, number of heads, head size, and sequence dimensions, facilitating the construction of a computation graph for model evaluation.
- **Member Functions**:
    - [`test_rwkv_wkv6::vars`](#test_rwkv_wkv6vars)
    - [`test_rwkv_wkv6::test_rwkv_wkv6`](#test_rwkv_wkv6test_rwkv_wkv6)
    - [`test_rwkv_wkv6::build_graph`](#test_rwkv_wkv6build_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_rwkv\_wkv6::vars<!-- {{#callable:test_rwkv_wkv6::vars}} -->
The `vars` method returns a string representation of the object's member variables.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the `VARS_TO_STR5` macro with the member variables `type`, `head_count`, `head_size`, `n_seq_tokens`, and `n_seqs`.
    - The result of the macro call is returned as a string.
- **Output**: The output is a string that represents the values of the member variables formatted according to the `VARS_TO_STR5` macro.
- **See also**: [`test_rwkv_wkv6`](#test_rwkv_wkv6)  (Data Structure)


---
#### test\_rwkv\_wkv6::test\_rwkv\_wkv6<!-- {{#callable:test_rwkv_wkv6::test_rwkv_wkv6}} -->
The `test_rwkv_wkv6` constructor initializes a test case for the RWKV model with specified parameters.
- **Inputs**:
    - `type`: The data type for the tensors, defaulting to `GGML_TYPE_F32`.
    - `head_count`: The number of attention heads, defaulting to 32.
    - `head_size`: The size of each attention head, defaulting to 64.
    - `n_seq_tokens`: The number of sequence tokens, defaulting to 32.
    - `n_seqs`: The number of sequences, defaulting to 32.
- **Control Flow**:
    - The constructor initializes member variables with the provided parameters.
    - It uses an initializer list to set the values of `type`, `head_count`, `head_size`, `n_seq_tokens`, and `n_seqs`.
- **Output**: The constructor does not return a value but initializes an instance of the `test_rwkv_wkv6` class with the specified parameters.
- **See also**: [`test_rwkv_wkv6`](#test_rwkv_wkv6)  (Data Structure)


---
#### test\_rwkv\_wkv6::build\_graph<!-- {{#callable:test_rwkv_wkv6::build_graph}} -->
The `build_graph` function constructs a computational graph for a neural network model by creating and initializing various tensors.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and resources for tensor operations.
- **Control Flow**:
    - Calculate the total number of tokens as the product of `n_seq_tokens` and `n_seqs`.
    - Create several tensors (`r`, `k`, `v`, `tf`, `td`, `s`) using [`ggml_new_tensor`](#test_caseggml_new_tensor) with specified dimensions and types.
    - Call the [`ggml_rwkv_wkv6`](../ggml/src/ggml.c.driver.md#ggml_rwkv_wkv6) function with the created tensors to compute the output tensor.
    - Return the output tensor.
- **Output**: Returns a pointer to a `ggml_tensor` that represents the output of the neural network model after processing the input tensors.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_rwkv_wkv6`](../ggml/src/ggml.c.driver.md#ggml_rwkv_wkv6)
- **See also**: [`test_rwkv_wkv6`](#test_rwkv_wkv6)  (Data Structure)



---
### test\_gla<!-- {{#data_structure:test_gla}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type used for the tensors.
    - `head_count`: The number of attention heads.
    - `head_size`: The size of each attention head.
    - `n_seq_tokens`: The number of sequence tokens.
    - `n_seqs`: The number of sequences.
- **Description**: The `test_gla` struct is a derived class from `test_case` that encapsulates parameters and methods for testing gated linear attention mechanisms in neural networks, specifically designed to handle multiple sequences and attention heads.
- **Member Functions**:
    - [`test_gla::vars`](#test_glavars)
    - [`test_gla::test_gla`](#test_glatest_gla)
    - [`test_gla::build_graph`](#test_glabuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_gla::vars<!-- {{#callable:test_gla::vars}} -->
The `vars` method returns a string representation of the member variables of the `test_gla` class.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR5` with the member variables `type`, `head_count`, `head_size`, `n_seq_tokens`, and `n_seqs`.
    - The `VARS_TO_STR5` macro constructs a string representation of these variables.
- **Output**: The method returns a string that represents the values of the member variables of the `test_gla` class.
- **See also**: [`test_gla`](#test_gla)  (Data Structure)


---
#### test\_gla::test\_gla<!-- {{#callable:test_gla::test_gla}} -->
The `test_gla` constructor initializes a `test_gla` object with specified parameters for a gated linear attention test.
- **Inputs**:
    - `type`: The data type for the tensors, defaulting to `GGML_TYPE_F32`.
    - `head_count`: The number of attention heads, defaulting to 32.
    - `head_size`: The size of each attention head, defaulting to 64.
    - `n_seq_tokens`: The number of sequence tokens, defaulting to 32.
    - `n_seqs`: The number of sequences, defaulting to 32.
- **Control Flow**:
    - The constructor initializes member variables with the provided arguments or default values.
    - The member variables include `type`, `head_count`, `head_size`, `n_seq_tokens`, and `n_seqs`.
- **Output**: The constructor does not return a value but initializes the `test_gla` object for further operations.
- **See also**: [`test_gla`](#test_gla)  (Data Structure)


---
#### test\_gla::build\_graph<!-- {{#callable:test_gla::build_graph}} -->
The `build_graph` function constructs a computational graph for a neural network layer by creating and initializing various tensors.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages memory and resources for tensor operations.
- **Control Flow**:
    - Calculate the total number of tokens by multiplying `n_seq_tokens` and `n_seqs`.
    - Create four 3D tensors (`q`, `k`, `v`, `g`) for query, key, value, and gated tensors respectively, using the [`ggml_new_tensor`](#test_caseggml_new_tensor) function.
    - Create a 2D tensor (`s`) for storing intermediate results.
    - Call the [`ggml_gated_linear_attn`](../ggml/src/ggml.c.driver.md#ggml_gated_linear_attn) function with the created tensors and return the output tensor.
- **Output**: Returns a pointer to the output tensor resulting from the gated linear attention operation.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_gated_linear_attn`](../ggml/src/ggml.c.driver.md#ggml_gated_linear_attn)
- **See also**: [`test_gla`](#test_gla)  (Data Structure)



---
### test\_rwkv\_wkv7<!-- {{#data_structure:test_rwkv_wkv7}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type used for the model.
    - `head_count`: The number of attention heads.
    - `head_size`: The size of each attention head.
    - `n_seq_tokens`: The number of sequence tokens.
    - `n_seqs`: The number of sequences.
- **Description**: The `test_rwkv_wkv7` struct is a derived class from `test_case` that encapsulates parameters and methods for testing the RWKV model architecture, specifically focusing on the configuration of attention heads, sequence tokens, and their respective data types.
- **Member Functions**:
    - [`test_rwkv_wkv7::vars`](#test_rwkv_wkv7vars)
    - [`test_rwkv_wkv7::test_rwkv_wkv7`](#test_rwkv_wkv7test_rwkv_wkv7)
    - [`test_rwkv_wkv7::build_graph`](#test_rwkv_wkv7build_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_rwkv\_wkv7::vars<!-- {{#callable:test_rwkv_wkv7::vars}} -->
The `vars` method returns a string representation of the member variables of the `test_rwkv_wkv7` class.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR5` with the member variables `type`, `head_count`, `head_size`, `n_seq_tokens`, and `n_seqs`.
    - The `VARS_TO_STR5` macro formats these variables into a string representation.
- **Output**: The method returns a string that represents the values of the member variables of the `test_rwkv_wkv7` class.
- **See also**: [`test_rwkv_wkv7`](#test_rwkv_wkv7)  (Data Structure)


---
#### test\_rwkv\_wkv7::test\_rwkv\_wkv7<!-- {{#callable:test_rwkv_wkv7::test_rwkv_wkv7}} -->
The `test_rwkv_wkv7` constructor initializes a test case for the RWKV WKV7 model with specified parameters.
- **Inputs**:
    - `type`: The data type for the tensors, defaulting to `GGML_TYPE_F32`.
    - `head_count`: The number of attention heads, defaulting to 32.
    - `head_size`: The size of each attention head, defaulting to 64.
    - `n_seq_tokens`: The number of sequence tokens, defaulting to 32.
    - `n_seqs`: The number of sequences, defaulting to 32.
- **Control Flow**:
    - The constructor initializes member variables with the provided parameters or defaults.
    - It sets the `type`, `head_count`, `head_size`, `n_seq_tokens`, and `n_seqs` for the test case.
- **Output**: The constructor does not return a value but initializes an instance of the `test_rwkv_wkv7` class.
- **See also**: [`test_rwkv_wkv7`](#test_rwkv_wkv7)  (Data Structure)


---
#### test\_rwkv\_wkv7::build\_graph<!-- {{#callable:test_rwkv_wkv7::build_graph}} -->
The `build_graph` function constructs a computational graph for a neural network model by creating and initializing various tensors.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure that manages memory and resources for tensor operations.
- **Control Flow**:
    - Calculate the total number of tokens as the product of `n_seq_tokens` and `n_seqs`.
    - Create six new tensors (`r`, `w`, `k`, `v`, `a`, `b`) of type `ggml_type` with dimensions based on `head_size`, `head_count`, and `n_tokens`.
    - Normalize tensors `a` and `b` using L2 normalization to prevent NaN outputs during computations.
    - Create a tensor `s` for storing intermediate results with dimensions based on `head_size`, `head_count`, and `n_seqs`.
    - Call the [`ggml_rwkv_wkv7`](../ggml/src/ggml.c.driver.md#ggml_rwkv_wkv7) function with the created tensors to compute the final output tensor.
    - Return the output tensor.
- **Output**: Returns a pointer to the output tensor resulting from the computations performed in the graph.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_l2_norm`](../ggml/src/ggml.c.driver.md#ggml_l2_norm)
    - [`ggml_rwkv_wkv7`](../ggml/src/ggml.c.driver.md#ggml_rwkv_wkv7)
- **See also**: [`test_rwkv_wkv7`](#test_rwkv_wkv7)  (Data Structure)



---
### test\_mul\_mat<!-- {{#data_structure:test_mul_mat}} -->
- **Type**: `struct`
- **Members**:
    - `type_a`: The data type of the first matrix.
    - `type_b`: The data type of the second matrix.
    - `m`: The number of rows in the output matrix.
    - `n`: The number of columns in the output matrix.
    - `k`: The number of columns in the first matrix and rows in the second matrix.
    - `bs`: An array representing the batch sizes for dimensions 3 and 4.
    - `nr`: An array representing the repeat counts for dimensions 3 and 4.
    - `per`: An array representing the permutation of dimensions.
    - `v`: A boolean indicating whether the matrices are non-contiguous views.
- **Description**: The `test_mul_mat` struct is designed to represent a test case for matrix multiplication operations, inheriting from `test_case`. It contains various parameters such as the types of the matrices, their dimensions, batch sizes, repeat counts, and permutation information, which are essential for configuring and executing the matrix multiplication tests.
- **Member Functions**:
    - [`test_mul_mat::vars`](#test_mul_matvars)
    - [`test_mul_mat::max_nmse_err`](#test_mul_matmax_nmse_err)
    - [`test_mul_mat::grad_nmax`](#test_mul_matgrad_nmax)
    - [`test_mul_mat::op_flops`](#test_mul_matop_flops)
    - [`test_mul_mat::test_mul_mat`](#test_mul_mattest_mul_mat)
    - [`test_mul_mat::build_graph`](#test_mul_matbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_mul\_mat::vars<!-- {{#callable:test_mul_mat::vars}} -->
The `vars` method returns a string representation of the member variables of the `test_mul_mat` class.
- **Inputs**:
    - `type_a`: The data type of the first matrix.
    - `type_b`: The data type of the second matrix.
    - `m`: The number of rows in the first matrix.
    - `n`: The number of columns in the second matrix.
    - `k`: The number of columns in the first matrix and rows in the second matrix.
    - `bs`: An array representing the batch sizes for dimensions 3 and 4.
    - `nr`: An array representing the repeat counts for dimensions 3 and 4.
    - `per`: An array representing the permutation of dimensions.
    - `v`: A boolean indicating whether the matrices are non-contiguous views.
- **Control Flow**:
    - The method starts by calling the `VARS_TO_STR9` macro with the member variables to create a string representation.
    - It returns the generated string representation.
- **Output**: The output is a string that contains the values of the member variables formatted as key-value pairs.
- **See also**: [`test_mul_mat`](#test_mul_mat)  (Data Structure)


---
#### test\_mul\_mat::max\_nmse\_err<!-- {{#callable:test_mul_mat::max_nmse_err}} -->
The `max_nmse_err` function returns a constant maximum normalized mean squared error value.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a constant value without any conditional logic or loops.
- **Output**: The output is a double value representing the maximum normalized mean squared error, specifically 5e-4.
- **See also**: [`test_mul_mat`](#test_mul_mat)  (Data Structure)


---
#### test\_mul\_mat::grad\_nmax<!-- {{#callable:test_mul_mat::grad_nmax}} -->
The `grad_nmax` function returns a constant value of 20000, which likely represents the maximum number of gradients to be computed.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a constant value without any conditional logic or loops.
- **Output**: The output is an integer value of 20000.
- **See also**: [`test_mul_mat`](#test_mul_mat)  (Data Structure)


---
#### test\_mul\_mat::op\_flops<!-- {{#callable:test_mul_mat::op_flops}} -->
Calculates the number of floating point operations (FLOPs) for a matrix multiplication operation.
- **Inputs**:
    - `t`: A pointer to a `ggml_tensor` structure, which is unused in this function.
- **Control Flow**:
    - The function begins by marking the input tensor `t` as unused using `GGML_UNUSED(t)`.
    - It then calculates the total number of FLOPs using the formula: 2 * m * n * k * bs[0] * nr[0] * bs[1] * nr[1].
- **Output**: Returns a `uint64_t` value representing the total number of floating point operations required for the matrix multiplication.
- **See also**: [`test_mul_mat`](#test_mul_mat)  (Data Structure)


---
#### test\_mul\_mat::test\_mul\_mat<!-- {{#callable:test_mul_mat::test_mul_mat}} -->
The `test_mul_mat` function is a test case for matrix multiplication operations in the GGML framework.
- **Inputs**:
    - `type_a`: The data type of the first matrix (default is GGML_TYPE_F32).
    - `type_b`: The data type of the second matrix (default is GGML_TYPE_F32).
    - `m`: The number of rows in the output matrix.
    - `n`: The number of columns in the output matrix.
    - `k`: The number of columns in the first matrix and rows in the second matrix.
    - `bs`: A 2-element array representing the batch sizes for dimensions 3 and 4.
    - `nr`: A 2-element array representing the repeat counts for dimensions 3 and 4.
    - `per`: A 4-element array representing the permutation of dimensions.
    - `v`: A boolean indicating whether the matrices are non-contiguous views.
- **Control Flow**:
    - The function initializes the member variables with the provided parameters.
    - It checks if the dimensions need to be permuted based on the `per` array.
    - If permutation is required, it creates tensors with permuted dimensions and sets their names.
    - If no permutation is needed, it creates the tensors directly based on the specified dimensions.
    - The function then calls `ggml_mul_mat` to perform the matrix multiplication and returns the output tensor.
- **Output**: The function returns a pointer to the resulting tensor from the matrix multiplication operation.
- **See also**: [`test_mul_mat`](#test_mul_mat)  (Data Structure)


---
#### test\_mul\_mat::build\_graph<!-- {{#callable:test_mul_mat::build_graph}} -->
The `build_graph` function constructs a computation graph for matrix multiplication of two tensors.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for tensor operations.
- **Control Flow**:
    - The function begins by declaring two tensor pointers, `a` and `b`.
    - It calculates the number of dimensions that need to be permuted based on the `per` array.
    - If any dimensions need to be permuted, it asserts that exactly two dimensions are permuted and that the tensors are not quantized.
    - It creates new tensors `a` and `b` with permuted dimensions and sets their parameters if they are not quantized.
    - If no permutation is needed, it checks if the tensors are views and creates them accordingly.
    - Finally, it performs matrix multiplication using [`ggml_mul_mat`](../ggml/src/ggml.c.driver.md#ggml_mul_mat) and returns the resulting tensor.
- **Output**: Returns a pointer to the resulting tensor from the matrix multiplication of `a` and `b`.
- **Functions called**:
    - [`ggml_is_quantized`](../ggml/src/ggml.c.driver.md#ggml_is_quantized)
    - [`test_case::ggml_new_tensor_4d`](#test_caseggml_new_tensor_4d)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_permute`](../ggml/src/ggml.c.driver.md#ggml_permute)
    - [`ggml_view_4d`](../ggml/src/ggml.c.driver.md#ggml_view_4d)
    - [`ggml_mul_mat`](../ggml/src/ggml.c.driver.md#ggml_mul_mat)
- **See also**: [`test_mul_mat`](#test_mul_mat)  (Data Structure)



---
### test\_mul\_mat\_id<!-- {{#data_structure:test_mul_mat_id}} -->
- **Type**: `struct`
- **Members**:
    - `type_a`: The data type of the first matrix.
    - `type_b`: The data type of the second matrix.
    - `n_mats`: The number of matrices to be multiplied.
    - `n_used`: The number of matrices that are actually used in the operation.
    - `b`: A boolean indicating whether to broadcast the second matrix.
    - `m`: The number of rows in the first matrix.
    - `n`: The number of columns in the second matrix.
    - `k`: The number of columns in the first matrix and rows in the second matrix.
- **Description**: The `test_mul_mat_id` struct is designed to facilitate the testing of matrix multiplication operations, specifically for cases where multiple matrices are involved. It inherits from `test_case` and contains several member variables that define the types and dimensions of the matrices involved in the multiplication. The struct also includes methods for building the computation graph and initializing the tensors used in the operation.
- **Member Functions**:
    - [`test_mul_mat_id::vars`](#test_mul_mat_idvars)
    - [`test_mul_mat_id::max_nmse_err`](#test_mul_mat_idmax_nmse_err)
    - [`test_mul_mat_id::op_flops`](#test_mul_mat_idop_flops)
    - [`test_mul_mat_id::test_mul_mat_id`](#test_mul_mat_idtest_mul_mat_id)
    - [`test_mul_mat_id::build_graph`](#test_mul_mat_idbuild_graph)
    - [`test_mul_mat_id::initialize_tensors`](#test_mul_mat_idinitialize_tensors)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_mul\_mat\_id::vars<!-- {{#callable:test_mul_mat_id::vars}} -->
The `vars` method returns a string representation of the internal state variables of the `test_mul_mat_id` class.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR8` with the class's member variables as arguments.
    - The `VARS_TO_STR8` macro formats these variables into a string representation.
- **Output**: The output is a string that contains the formatted values of the member variables of the `test_mul_mat_id` class.
- **See also**: [`test_mul_mat_id`](#test_mul_mat_id)  (Data Structure)


---
#### test\_mul\_mat\_id::max\_nmse\_err<!-- {{#callable:test_mul_mat_id::max_nmse_err}} -->
The `max_nmse_err` function returns a constant maximum normalized mean squared error value.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a predefined constant value without any conditional logic or loops.
- **Output**: The output is a double precision floating-point number representing the maximum normalized mean squared error, specifically 5e-4.
- **See also**: [`test_mul_mat_id`](#test_mul_mat_id)  (Data Structure)


---
#### test\_mul\_mat\_id::op\_flops<!-- {{#callable:test_mul_mat_id::op_flops}} -->
The `op_flops` function calculates the number of floating-point operations (FLOPs) required for a matrix multiplication operation.
- **Inputs**:
    - `t`: A pointer to a `ggml_tensor` structure, which is unused in this function.
- **Control Flow**:
    - The function does not contain any conditional statements or loops.
    - It directly computes the FLOPs using the formula based on the member variables of the class.
- **Output**: Returns a `uint64_t` value representing the total number of floating-point operations calculated as 2 * m * k * n * n_used.
- **See also**: [`test_mul_mat_id`](#test_mul_mat_id)  (Data Structure)


---
#### test\_mul\_mat\_id::test\_mul\_mat\_id<!-- {{#callable:test_mul_mat_id::test_mul_mat_id}} -->
The `test_mul_mat_id` constructor initializes a test case for matrix multiplication with identity behavior.
- **Inputs**:
    - `type_a`: The data type for the first matrix, defaulting to `GGML_TYPE_F32`.
    - `type_b`: The data type for the second matrix, defaulting to `GGML_TYPE_F32`.
    - `n_mats`: The total number of matrices to be used in the operation, defaulting to 8.
    - `n_used`: The number of matrices that will actually be used, must be less than or equal to `n_mats`, defaulting to 2.
    - `b`: A boolean indicating whether to broadcast the second matrix, defaulting to false.
    - `m`: The number of rows in the first matrix, defaulting to 32.
    - `n`: The number of columns in the second matrix, defaulting to 32.
    - `k`: The number of columns in the first matrix and rows in the second matrix, defaulting to 32.
- **Control Flow**:
    - The constructor initializes member variables with the provided arguments.
    - It asserts that the number of used matrices (`n_used`) does not exceed the total number of matrices (`n_mats`).
- **Output**: The constructor does not return a value but initializes the state of the `test_mul_mat_id` object.
- **See also**: [`test_mul_mat_id`](#test_mul_mat_id)  (Data Structure)


---
#### test\_mul\_mat\_id::build\_graph<!-- {{#callable:test_mul_mat_id::build_graph}} -->
The `build_graph` function constructs a computation graph for matrix multiplication and tensor operations.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and tensor operations.
- **Control Flow**:
    - Creates a 3D tensor `as` of type `type_a` with dimensions (k, m, n_mats) and names it 'as'.
    - Creates a 2D tensor `ids` of type `GGML_TYPE_I32` with dimensions (n_mats, n) and names it 'ids'.
    - If `n_used` is not equal to `n_mats`, it creates a view of `ids` with dimensions (n_used, n) and names it 'view_of_ids'.
    - Creates a 3D tensor `b` of type `type_b` with dimensions (k, 1 or n_used, n) based on the value of `b` and names it 'b'.
    - Calls the [`ggml_mul_mat_id`](../ggml/src/ggml.c.driver.md#ggml_mul_mat_id) function to perform matrix multiplication on `as`, `b`, and `ids`, storing the result in `out`.
    - Names the output tensor 'out' and returns it.
- **Output**: Returns a pointer to the resulting tensor `out` after performing the matrix multiplication and tensor operations.
- **Functions called**:
    - [`test_case::ggml_new_tensor_3d`](#test_caseggml_new_tensor_3d)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`test_case::ggml_new_tensor_2d`](#test_caseggml_new_tensor_2d)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_mul_mat_id`](../ggml/src/ggml.c.driver.md#ggml_mul_mat_id)
- **See also**: [`test_mul_mat_id`](#test_mul_mat_id)  (Data Structure)


---
#### test\_mul\_mat\_id::initialize\_tensors<!-- {{#callable:test_mul_mat_id::initialize_tensors}} -->
Initializes tensors in a `ggml_context` by populating them with either shuffled integer IDs or uniform random values.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that contains the tensors to be initialized.
- **Control Flow**:
    - A random number generator is initialized using `std::random_device`.
    - A loop iterates over all tensors in the context using [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor) and [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor).
    - For each tensor, if its type is `GGML_TYPE_I32`, it checks if it is a view operation; if not, it initializes the tensor with shuffled IDs.
    - For each row in the tensor, a vector of IDs is created, filled with values based on the number of matrices (`n_mats`), shuffled, and set in the tensor.
    - If the tensor type is not `GGML_TYPE_I32`, it calls [`init_tensor_uniform`](#init_tensor_uniform) to initialize the tensor with uniform random values.
- **Output**: This function does not return a value; it modifies the tensors in the provided `ggml_context` directly.
- **Functions called**:
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`ggml_is_view_op`](#ggml_is_view_op)
    - [`ggml_nrows`](../ggml/src/ggml.c.driver.md#ggml_nrows)
    - [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set)
    - [`init_tensor_uniform`](#init_tensor_uniform)
- **See also**: [`test_mul_mat_id`](#test_mul_mat_id)  (Data Structure)



---
### test\_out\_prod<!-- {{#data_structure:test_out_prod}} -->
- **Type**: `struct`
- **Members**:
    - `type_a`: The data type of the first tensor.
    - `type_b`: The data type of the second tensor.
    - `m`: The number of rows in the first tensor.
    - `n`: The number of columns in the second tensor.
    - `k`: The number of columns in the first tensor and rows in the second tensor.
    - `bs`: An array representing the batch sizes for dimensions 3 and 4.
    - `nr`: An array representing the repeat counts for dimensions 3 and 4.
    - `trans_b`: A boolean indicating whether the second tensor should be transposed.
- **Description**: The `test_out_prod` struct is designed to represent a test case for an outer product operation between two tensors, encapsulating their types, dimensions, and other relevant parameters necessary for constructing the computation graph in a machine learning context.
- **Member Functions**:
    - [`test_out_prod::vars`](#test_out_prodvars)
    - [`test_out_prod::max_nmse_err`](#test_out_prodmax_nmse_err)
    - [`test_out_prod::test_out_prod`](#test_out_prodtest_out_prod)
    - [`test_out_prod::build_graph`](#test_out_prodbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_out\_prod::vars<!-- {{#callable:test_out_prod::vars}} -->
The `vars` method returns a string representation of the member variables of the `test_out_prod` class.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR8` with the member variables of the class.
    - The macro constructs a string representation of the variables.
- **Output**: The method returns a string that represents the values of the member variables of the `test_out_prod` class.
- **See also**: [`test_out_prod`](#test_out_prod)  (Data Structure)


---
#### test\_out\_prod::max\_nmse\_err<!-- {{#callable:test_out_prod::max_nmse_err}} -->
The `max_nmse_err` function returns a constant maximum normalized mean squared error value.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a predefined constant value without any conditional logic or loops.
- **Output**: The output is a double value representing the maximum normalized mean squared error, specifically 5e-4.
- **See also**: [`test_out_prod`](#test_out_prod)  (Data Structure)


---
#### test\_out\_prod::test\_out\_prod<!-- {{#callable:test_out_prod::test_out_prod}} -->
The `test_out_prod` constructor initializes a test case for output production using two tensor types and various dimensions.
- **Inputs**:
    - `type_a`: The data type of the first tensor, defaulting to `GGML_TYPE_F32`.
    - `type_b`: The data type of the second tensor, defaulting to `GGML_TYPE_F32`.
    - `m`: The number of rows in the first tensor, defaulting to 32.
    - `n`: The number of columns in the second tensor, defaulting to 32.
    - `k`: The number of columns in the first tensor and rows in the second tensor, defaulting to 32.
    - `bs`: A 2-element array representing the batch size dimensions, defaulting to {10, 10}.
    - `nr`: A 2-element array representing the repeat dimensions, defaulting to {2, 2}.
    - `trans_b`: A boolean indicating whether to transpose the second tensor, defaulting to false.
- **Control Flow**:
    - The constructor initializes member variables with the provided parameters or default values.
    - It sets up the dimensions and types for the tensors involved in the output production test.
- **Output**: The constructor does not return a value but initializes the `test_out_prod` object with the specified parameters.
- **See also**: [`test_out_prod`](#test_out_prod)  (Data Structure)


---
#### test\_out\_prod::build\_graph<!-- {{#callable:test_out_prod::build_graph}} -->
The `build_graph` function constructs a computation graph for matrix multiplication and tensor operations.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a 4D tensor `a` of type `type_a` with dimensions (m, k, bs[0], bs[1]) and names it 'a'.
    - Checks if `trans_b` is true; if so, creates a 4D tensor `b` of type `type_b` with dimensions (k, n, bs[0]*nr[0], bs[1]*nr[1]) and transposes it.
    - If `trans_b` is false, creates a 4D tensor `b` of type `type_b` with dimensions (n, k, bs[0]*nr[0], bs[1]*nr[1]).
    - Names the tensor `b` as 'b'.
    - Calls [`ggml_out_prod`](../ggml/src/ggml.c.driver.md#ggml_out_prod) with tensors `a` and `b` to compute the outer product and stores the result in `out`.
    - Names the output tensor `out` as 'out'.
    - Returns the output tensor `out`.
- **Output**: Returns a pointer to the resulting tensor `out`, which contains the result of the outer product of tensors `a` and `b`.
- **Functions called**:
    - [`test_case::ggml_new_tensor_4d`](#test_caseggml_new_tensor_4d)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_transpose`](../ggml/src/ggml.c.driver.md#ggml_transpose)
    - [`ggml_out_prod`](../ggml/src/ggml.c.driver.md#ggml_out_prod)
- **See also**: [`test_out_prod`](#test_out_prod)  (Data Structure)



---
### test\_sqr<!-- {{#data_structure:test_sqr}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
- **Description**: The `test_sqr` structure is a derived class from `test_case` that is designed to test the square operation on tensors. It contains a tensor type and its dimensions, and it overrides methods to provide specific behavior for the test case, including variable representation and graph building for the square operation.
- **Member Functions**:
    - [`test_sqr::vars`](#test_sqrvars)
    - [`test_sqr::test_sqr`](#test_sqrtest_sqr)
    - [`test_sqr::build_graph`](#test_sqrbuild_graph)
    - [`test_sqr::grad_eps`](#test_sqrgrad_eps)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_sqr::vars<!-- {{#callable:test_sqr::vars}} -->
The `vars` method returns a string representation of the `test_sqr` class's parameters.
- **Inputs**:
    - `none`: This method does not take any input parameters.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR2` with the class's member variables `type` and `ne`.
    - The `VARS_TO_STR2` macro formats these variables into a string representation.
- **Output**: The method returns a string that represents the `type` and `ne` member variables of the `test_sqr` class.
- **See also**: [`test_sqr`](#test_sqr)  (Data Structure)


---
#### test\_sqr::test\_sqr<!-- {{#callable:test_sqr::test_sqr}} -->
The `test_sqr` constructor initializes a test case for the square operation with specified tensor type and dimensions.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array representing the dimensions of the tensor, defaulting to {10, 5, 4, 3}.
- **Control Flow**:
    - The constructor initializes the member variables `type` and `ne` with the provided arguments.
    - If no arguments are provided, it uses the default values for `type` and `ne`.
- **Output**: The constructor does not return a value but initializes the `test_sqr` object with the specified tensor type and dimensions.
- **See also**: [`test_sqr`](#test_sqr)  (Data Structure)


---
#### test\_sqr::build\_graph<!-- {{#callable:test_sqr::build_graph}} -->
The `build_graph` function constructs a computation graph for a square operation on a tensor.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions.
    - Sets the tensor `a` as a parameter using [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param).
    - Names the tensor `a` as 'a' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Calculates the square of tensor `a` using [`ggml_sqr`](../ggml/src/ggml.c.driver.md#ggml_sqr) and stores the result in `out`.
    - Names the output tensor `out` as 'out' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Returns the output tensor `out`.
- **Output**: Returns a pointer to the output tensor `out`, which contains the squared values of the input tensor `a`.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_sqr`](../ggml/src/ggml.c.driver.md#ggml_sqr)
- **See also**: [`test_sqr`](#test_sqr)  (Data Structure)


---
#### test\_sqr::grad\_eps<!-- {{#callable:test_sqr::grad_eps}} -->
Calculates a gradient epsilon value based on the dimensions of a tensor.
- **Inputs**:
    - `ne`: An array of integers representing the dimensions of a tensor, where `ne[0]`, `ne[1]`, `ne[2]`, and `ne[3]` are used in the calculation.
- **Control Flow**:
    - The function multiplies the first four elements of the `ne` array together.
    - The result is then multiplied by 0.025 (which is 0.1 * 0.25) to compute the final gradient epsilon value.
- **Output**: Returns a float value representing the computed gradient epsilon.
- **See also**: [`test_sqr`](#test_sqr)  (Data Structure)



---
### test\_sqrt<!-- {{#data_structure:test_sqrt}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
- **Description**: The `test_sqrt` struct is a derived class from `test_case` that is designed to test the square root operation on tensors. It contains a `ggml_type` member to specify the data type of the tensor and an array of integers to define the tensor's dimensions. The struct overrides methods to provide specific behavior for variable representation and graph building for the square root operation.
- **Member Functions**:
    - [`test_sqrt::vars`](#test_sqrtvars)
    - [`test_sqrt::test_sqrt`](#test_sqrttest_sqrt)
    - [`test_sqrt::build_graph`](#test_sqrtbuild_graph)
    - [`test_sqrt::initialize_tensors`](#test_sqrtinitialize_tensors)
    - [`test_sqrt::grad_eps`](#test_sqrtgrad_eps)
    - [`test_sqrt::grad_precise`](#test_sqrtgrad_precise)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_sqrt::vars<!-- {{#callable:test_sqrt::vars}} -->
The `vars` method returns a string representation of the `test_sqrt` class's parameters.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR2` with the class's member variables `type` and `ne`.
    - The `VARS_TO_STR2` macro formats these variables into a string representation.
- **Output**: The method outputs a string that represents the `type` and `ne` member variables of the `test_sqrt` class.
- **See also**: [`test_sqrt`](#test_sqrt)  (Data Structure)


---
#### test\_sqrt::test\_sqrt<!-- {{#callable:test_sqrt::test_sqrt}} -->
The `test_sqrt` constructor initializes a test case for the square root operation with specified tensor type and dimensions.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array representing the dimensions of the tensor, defaulting to {10, 3, 3, 2}.
- **Control Flow**:
    - The constructor initializes the member variables `type` and `ne` with the provided arguments.
    - If no arguments are provided, it uses the default values for `type` and `ne`.
- **Output**: The constructor does not return a value but initializes the `test_sqrt` object with the specified tensor type and dimensions.
- **See also**: [`test_sqrt`](#test_sqrt)  (Data Structure)


---
#### test\_sqrt::build\_graph<!-- {{#callable:test_sqrt::build_graph}} -->
The `build_graph` function constructs a computational graph for a tensor operation involving the square root.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions.
    - Sets the tensor `a` as a parameter using [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param).
    - Names the tensor `a` as 'a' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Calculates the square root of tensor `a` and stores the result in tensor `out` using [`ggml_sqrt`](../ggml/src/ggml.c.driver.md#ggml_sqrt).
    - Names the output tensor `out` as 'out' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Returns the output tensor `out`.
- **Output**: Returns a pointer to a `ggml_tensor` that contains the square root of the input tensor `a`.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_sqrt`](../ggml/src/ggml.c.driver.md#ggml_sqrt)
- **See also**: [`test_sqrt`](#test_sqrt)  (Data Structure)


---
#### test\_sqrt::initialize\_tensors<!-- {{#callable:test_sqrt::initialize_tensors}} -->
The `initialize_tensors` method initializes all tensors in the given `ggml_context` with uniformly distributed positive values.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that holds the state of the tensor context.
- **Control Flow**:
    - The method iterates over all tensors in the context using a loop that retrieves the first tensor and continues until there are no more tensors.
    - For each tensor, it calls the [`init_tensor_uniform`](#init_tensor_uniform) function to initialize the tensor with values uniformly distributed between 50.0 and 100.0.
- **Output**: This method does not return a value; it modifies the tensors in place by initializing them with positive values.
- **Functions called**:
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`init_tensor_uniform`](#init_tensor_uniform)
- **See also**: [`test_sqrt`](#test_sqrt)  (Data Structure)


---
#### test\_sqrt::grad\_eps<!-- {{#callable:test_sqrt::grad_eps}} -->
The `grad_eps` function returns a constant gradient epsilon value of 20.0f.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a constant value without any conditional statements or loops.
- **Output**: The output is a float value of 20.0f, representing the gradient epsilon.
- **See also**: [`test_sqrt`](#test_sqrt)  (Data Structure)


---
#### test\_sqrt::grad\_precise<!-- {{#callable:test_sqrt::grad_precise}} -->
The `grad_precise` function is an overridden method that always returns true, indicating that gradient estimation should be performed with high precision.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a boolean value without any conditional statements or loops.
- **Output**: The output is a boolean value, specifically 'true'.
- **See also**: [`test_sqrt`](#test_sqrt)  (Data Structure)



---
### test\_log<!-- {{#data_structure:test_log}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
- **Description**: The `test_log` structure is a derived class from `test_case` that encapsulates the properties and behavior of a logarithmic operation test case, including the tensor type and its dimensions, and provides methods for building computation graphs and initializing tensors.
- **Member Functions**:
    - [`test_log::vars`](#test_logvars)
    - [`test_log::test_log`](#test_logtest_log)
    - [`test_log::build_graph`](#test_logbuild_graph)
    - [`test_log::initialize_tensors`](#test_loginitialize_tensors)
    - [`test_log::grad_precise`](#test_loggrad_precise)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_log::vars<!-- {{#callable:test_log::vars}} -->
The `vars` method returns a string representation of the `test_log` class's member variables.
- **Inputs**: None
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR2` with `type` and `ne` as arguments.
    - The `VARS_TO_STR2` macro formats the input arguments into a string representation.
- **Output**: The output is a string that represents the `type` and `ne` member variables of the `test_log` class.
- **See also**: [`test_log`](#test_log)  (Data Structure)


---
#### test\_log::test\_log<!-- {{#callable:test_log::test_log}} -->
The `test_log` constructor initializes a test case for the logarithm operation with specified tensor type and dimensions.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array of four integers representing the dimensions of the tensor, defaulting to {10, 5, 4, 3}.
- **Control Flow**:
    - The constructor initializes the member variables `type` and `ne` with the provided arguments.
    - If no arguments are provided, default values are used.
- **Output**: The constructor does not return a value but initializes the `test_log` object with the specified tensor type and dimensions.
- **See also**: [`test_log`](#test_log)  (Data Structure)


---
#### test\_log::build\_graph<!-- {{#callable:test_log::build_graph}} -->
The `build_graph` function constructs a computation graph for a tensor operation involving logarithmic transformation.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions.
    - Sets the tensor `a` as a parameter using [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param).
    - Names the tensor `a` as 'a' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Applies the logarithmic operation on tensor `a` to create tensor `out` using [`ggml_log`](../ggml/src/ggml.c.driver.md#ggml_log).
    - Names the output tensor `out` as 'out' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Returns the output tensor `out`.
- **Output**: Returns a pointer to the output tensor `out`, which contains the logarithmic values of the input tensor `a`.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_log`](../ggml/src/ggml.c.driver.md#ggml_log)
- **See also**: [`test_log`](#test_log)  (Data Structure)


---
#### test\_log::initialize\_tensors<!-- {{#callable:test_log::initialize_tensors}} -->
The `initialize_tensors` method initializes all tensors in the given `ggml_context` with uniform random values between 0.9 and 1.1.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that holds the state and data for tensor operations.
- **Control Flow**:
    - The method starts a loop that retrieves the first tensor from the context using [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor).
    - It continues to loop through all tensors in the context using [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor) until there are no more tensors (i.e., `t` is NULL).
    - For each tensor `t`, it calls the [`init_tensor_uniform`](#init_tensor_uniform) function to initialize the tensor with values clustered around 1.0, specifically between 0.9 and 1.1.
- **Output**: The function does not return a value; it modifies the tensors in place within the provided `ggml_context`.
- **Functions called**:
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`init_tensor_uniform`](#init_tensor_uniform)
- **See also**: [`test_log`](#test_log)  (Data Structure)


---
#### test\_log::grad\_precise<!-- {{#callable:test_log::grad_precise}} -->
The `grad_precise` method in the `test_log` class always returns true, indicating that gradient estimation should be performed with high precision.
- **Inputs**: None
- **Control Flow**:
    - The method does not contain any control flow statements such as conditionals or loops.
    - It directly returns a boolean value.
- **Output**: The output is a boolean value, specifically 'true'.
- **See also**: [`test_log`](#test_log)  (Data Structure)



---
### test\_sin<!-- {{#data_structure:test_sin}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
- **Description**: The `test_sin` structure is a derived class from `test_case` that encapsulates the properties and methods necessary to test the sine operation on tensors, including the tensor's data type and its dimensions.
- **Member Functions**:
    - [`test_sin::vars`](#test_sinvars)
    - [`test_sin::test_sin`](#test_sintest_sin)
    - [`test_sin::build_graph`](#test_sinbuild_graph)
    - [`test_sin::initialize_tensors`](#test_sininitialize_tensors)
    - [`test_sin::max_maa_err`](#test_sinmax_maa_err)
    - [`test_sin::grad_eps`](#test_singrad_eps)
    - [`test_sin::grad_precise`](#test_singrad_precise)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_sin::vars<!-- {{#callable:test_sin::vars}} -->
The `vars` method returns a string representation of the `test_sin` class's parameters.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR2` with the class's member variables `type` and `ne` as arguments.
    - The `VARS_TO_STR2` macro formats these variables into a string representation.
- **Output**: The method outputs a string that represents the `type` and `ne` member variables of the `test_sin` class.
- **See also**: [`test_sin`](#test_sin)  (Data Structure)


---
#### test\_sin::test\_sin<!-- {{#callable:test_sin::test_sin}} -->
The `test_sin` constructor initializes a test case for the sine operation with specified tensor type and dimensions.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array representing the dimensions of the tensor, defaulting to {10, 2, 2, 2}.
- **Control Flow**:
    - The constructor initializes the member variables `type` and `ne` with the provided arguments.
    - If no arguments are provided, default values are used.
- **Output**: The constructor does not return a value but initializes the `test_sin` object with the specified tensor type and dimensions.
- **See also**: [`test_sin`](#test_sin)  (Data Structure)


---
#### test\_sin::build\_graph<!-- {{#callable:test_sin::build_graph}} -->
The `build_graph` method constructs a computation graph for a sine operation on a tensor.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions.
    - Sets the tensor `a` as a parameter using [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param).
    - Names the tensor `a` as 'a' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Computes the sine of tensor `a` and stores the result in tensor `out` using [`ggml_sin`](../ggml/src/ggml.c.driver.md#ggml_sin).
    - Names the output tensor `out` as 'out' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Returns the output tensor `out`.
- **Output**: Returns a pointer to the output tensor `out`, which contains the sine values of the input tensor `a`.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_sin`](../ggml/src/ggml.c.driver.md#ggml_sin)
- **See also**: [`test_sin`](#test_sin)  (Data Structure)


---
#### test\_sin::initialize\_tensors<!-- {{#callable:test_sin::initialize_tensors}} -->
Initializes all tensors in the given `ggml_context` with uniform random values within the range [-6.5, 6.5].
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that contains the tensors to be initialized.
- **Control Flow**:
    - Iterates through all tensors in the context using [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor) and [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor).
    - For each tensor, calls [`init_tensor_uniform`](#init_tensor_uniform) to initialize it with random values in the specified range.
- **Output**: This function does not return a value; it modifies the tensors in the provided context directly.
- **Functions called**:
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`init_tensor_uniform`](#init_tensor_uniform)
- **See also**: [`test_sin`](#test_sin)  (Data Structure)


---
#### test\_sin::max\_maa\_err<!-- {{#callable:test_sin::max_maa_err}} -->
The `max_maa_err` function returns a constant maximum absolute asymmetry error value.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a constant value without any conditional statements or loops.
- **Output**: The function outputs a double value of 1e-3, representing the maximum allowable absolute asymmetry error.
- **See also**: [`test_sin`](#test_sin)  (Data Structure)


---
#### test\_sin::grad\_eps<!-- {{#callable:test_sin::grad_eps}} -->
The `grad_eps` function returns a constant gradient epsilon value of 0.2.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a constant value without any conditional logic or loops.
- **Output**: The output is a float value of 0.2, representing the gradient epsilon used in gradient calculations.
- **See also**: [`test_sin`](#test_sin)  (Data Structure)


---
#### test\_sin::grad\_precise<!-- {{#callable:test_sin::grad_precise}} -->
The `grad_precise` function returns a boolean value indicating whether to use a precise gradient estimation method.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the boolean value 'true'.
- **Output**: The output is a boolean value, specifically 'true'.
- **See also**: [`test_sin`](#test_sin)  (Data Structure)



---
### test\_cos<!-- {{#data_structure:test_cos}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
- **Description**: The `test_cos` structure is a derived class from `test_case` that is used to define a test case for the cosine operation in a tensor computation framework. It contains a tensor type and its dimensions, and it overrides methods to build a computation graph and initialize tensor values for testing.
- **Member Functions**:
    - [`test_cos::vars`](#test_cosvars)
    - [`test_cos::test_cos`](#test_costest_cos)
    - [`test_cos::build_graph`](#test_cosbuild_graph)
    - [`test_cos::initialize_tensors`](#test_cosinitialize_tensors)
    - [`test_cos::max_maa_err`](#test_cosmax_maa_err)
    - [`test_cos::grad_eps`](#test_cosgrad_eps)
    - [`test_cos::grad_precise`](#test_cosgrad_precise)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_cos::vars<!-- {{#callable:test_cos::vars}} -->
The `vars` method returns a string representation of the `test_cos` class's member variables.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method directly calls the `VARS_TO_STR2` macro with the `type` and `ne` member variables.
    - The `VARS_TO_STR2` macro formats these variables into a string representation.
- **Output**: The output is a string that represents the `type` and `ne` member variables of the `test_cos` class.
- **See also**: [`test_cos`](#test_cos)  (Data Structure)


---
#### test\_cos::test\_cos<!-- {{#callable:test_cos::test_cos}} -->
The `test_cos` constructor initializes a test case for the cosine operation with specified tensor type and dimensions.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array of four integers representing the dimensions of the tensor, defaulting to {10, 2, 2, 2}.
- **Control Flow**:
    - The constructor initializes the member variables `type` and `ne` with the provided arguments.
    - If no arguments are provided, default values are used.
- **Output**: The constructor does not return a value but initializes the `test_cos` object with the specified tensor type and dimensions.
- **See also**: [`test_cos`](#test_cos)  (Data Structure)


---
#### test\_cos::build\_graph<!-- {{#callable:test_cos::build_graph}} -->
The `build_graph` method constructs a computation graph for a cosine operation on a tensor.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for tensor operations.
- **Control Flow**:
    - A new tensor `a` is created using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions.
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param) is called on tensor `a` to mark it as a parameter tensor.
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name) is used to assign the name 'a' to the tensor `a` for identification.
    - The cosine of tensor `a` is computed using [`ggml_cos`](../ggml/src/ggml.c.driver.md#ggml_cos), resulting in a new tensor `out`.
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name) is called again to name the output tensor 'out'.
    - The output tensor `out` is returned from the function.
- **Output**: Returns a pointer to the output tensor `out`, which contains the cosine values of the input tensor `a`.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_cos`](../ggml/src/ggml.c.driver.md#ggml_cos)
- **See also**: [`test_cos`](#test_cos)  (Data Structure)


---
#### test\_cos::initialize\_tensors<!-- {{#callable:test_cos::initialize_tensors}} -->
The `initialize_tensors` method initializes all tensors in the given `ggml_context` with uniform random values within the range [-6.5, 6.5].
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that holds the state and data for tensor operations.
- **Control Flow**:
    - The method iterates over all tensors in the context using a loop that retrieves the first tensor and continues until there are no more tensors.
    - For each tensor, it calls the [`init_tensor_uniform`](#init_tensor_uniform) function to initialize the tensor with random values in the specified range.
- **Output**: This method does not return a value; it modifies the tensors in place within the provided `ggml_context`.
- **Functions called**:
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`init_tensor_uniform`](#init_tensor_uniform)
- **See also**: [`test_cos`](#test_cos)  (Data Structure)


---
#### test\_cos::max\_maa\_err<!-- {{#callable:test_cos::max_maa_err}} -->
The `max_maa_err` function returns a constant maximum absolute asymmetry error value.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a constant value without any conditional logic or loops.
- **Output**: The function outputs a double value of 1e-3, representing the maximum allowable absolute asymmetry error.
- **See also**: [`test_cos`](#test_cos)  (Data Structure)


---
#### test\_cos::grad\_eps<!-- {{#callable:test_cos::grad_eps}} -->
The `grad_eps` function returns a constant gradient epsilon value of 0.2.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a constant value without any conditional logic or loops.
- **Output**: The output is a float value of 0.2, representing the gradient epsilon.
- **See also**: [`test_cos`](#test_cos)  (Data Structure)


---
#### test\_cos::grad\_precise<!-- {{#callable:test_cos::grad_precise}} -->
The `grad_precise` function returns a boolean value indicating whether to use a precise gradient estimation method.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the boolean value `true` without any conditions or loops.
- **Output**: The output is a boolean value, specifically `true`.
- **See also**: [`test_cos`](#test_cos)  (Data Structure)



---
### test\_clamp<!-- {{#data_structure:test_clamp}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
    - `min`: The minimum value for clamping.
    - `max`: The maximum value for clamping.
- **Description**: The `test_clamp` struct is a derived class from `test_case` that encapsulates the parameters and behavior for testing a clamping operation on tensors, including the data type, dimensions, and the minimum and maximum values for clamping.
- **Member Functions**:
    - [`test_clamp::vars`](#test_clampvars)
    - [`test_clamp::test_clamp`](#test_clamptest_clamp)
    - [`test_clamp::build_graph`](#test_clampbuild_graph)
    - [`test_clamp::grad_eps`](#test_clampgrad_eps)
    - [`test_clamp::grad_expect`](#test_clampgrad_expect)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_clamp::vars<!-- {{#callable:test_clamp::vars}} -->
The `vars` method returns a string representation of the member variables of the `test_clamp` class.
- **Inputs**:
    - `none`: This method does not take any input parameters.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR4` with the member variables `type`, `ne`, `min`, and `max`.
    - The `VARS_TO_STR4` macro constructs a formatted string representation of these variables.
- **Output**: The method returns a string that represents the values of the member variables of the `test_clamp` class.
- **See also**: [`test_clamp`](#test_clamp)  (Data Structure)


---
#### test\_clamp::test\_clamp<!-- {{#callable:test_clamp::test_clamp}} -->
The `test_clamp` constructor initializes a test case for clamping tensor values within specified minimum and maximum bounds.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array representing the dimensions of the tensor, defaulting to {10, 5, 4, 3}.
    - `min`: The minimum value for clamping, defaulting to -0.5f.
    - `max`: The maximum value for clamping, defaulting to 0.5f.
- **Control Flow**:
    - The constructor initializes the member variables `type`, `ne`, `min`, and `max` with the provided arguments or default values.
    - The `vars` method is overridden to return a string representation of the test case variables for logging purposes.
- **Output**: The constructor does not return a value but initializes an instance of the `test_clamp` class with the specified parameters.
- **See also**: [`test_clamp`](#test_clamp)  (Data Structure)


---
#### test\_clamp::build\_graph<!-- {{#callable:test_clamp::build_graph}} -->
The `build_graph` function constructs a computational graph for a tensor operation involving clamping.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions.
    - Sets the name of tensor `a` to 'a' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Applies the `ggml_clamp` operation to tensor `a`, resulting in a new tensor `out` that is clamped between specified minimum and maximum values.
    - Sets the name of tensor `out` to 'out' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Returns the tensor `out`.
- **Output**: Returns a pointer to a `ggml_tensor` that represents the output of the clamping operation.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
- **See also**: [`test_clamp`](#test_clamp)  (Data Structure)


---
#### test\_clamp::grad\_eps<!-- {{#callable:test_clamp::grad_eps}} -->
Returns a small constant value of 0.01 as the gradient epsilon.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a constant value without any conditions or loops.
- **Output**: A float value of 0.01, which is used as a small perturbation for gradient calculations.
- **See also**: [`test_clamp`](#test_clamp)  (Data Structure)


---
#### test\_clamp::grad\_expect<!-- {{#callable:test_clamp::grad_expect}} -->
The `grad_expect` function returns a vector containing the expected gradient values for a specific test case.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a vector containing two float values, 0.0f and 1.0f.
- **Output**: The output is a `std::vector<float>` containing the values {0.0f, 1.0f}.
- **See also**: [`test_clamp`](#test_clamp)  (Data Structure)



---
### test\_diag\_mask\_inf<!-- {{#data_structure:test_diag_mask_inf}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
    - `n_past`: An integer representing the number of past tokens.
- **Description**: The `test_diag_mask_inf` struct is a derived class from `test_case` that encapsulates the parameters and behavior for testing the diagonal masking operation in a tensor, specifically designed for handling tensors of a specified type and shape, along with a parameter for the number of past tokens to consider.
- **Member Functions**:
    - [`test_diag_mask_inf::vars`](#test_diag_mask_infvars)
    - [`test_diag_mask_inf::test_diag_mask_inf`](#test_diag_mask_inftest_diag_mask_inf)
    - [`test_diag_mask_inf::build_graph`](#test_diag_mask_infbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_diag\_mask\_inf::vars<!-- {{#callable:test_diag_mask_inf::vars}} -->
The `vars` method returns a string representation of the object's member variables.
- **Inputs**: None
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR3` with the member variables `type`, `ne`, and `n_past`.
    - The `VARS_TO_STR3` macro formats these variables into a string.
- **Output**: The output is a string that represents the values of the member variables in a formatted manner.
- **See also**: [`test_diag_mask_inf`](#test_diag_mask_inf)  (Data Structure)


---
#### test\_diag\_mask\_inf::test\_diag\_mask\_inf<!-- {{#callable:test_diag_mask_inf::test_diag_mask_inf}} -->
The `test_diag_mask_inf` constructor initializes a test case for a diagonal masking operation with specified tensor type, shape, and past sequence length.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array representing the shape of the tensor, defaulting to {10, 10, 3, 2}.
    - `n_past`: An integer representing the number of past tokens to consider, defaulting to 5.
- **Control Flow**:
    - The constructor initializes the member variables `type`, `ne`, and `n_past` with the provided arguments or default values.
    - The `vars` method is overridden to return a string representation of the test case variables for debugging purposes.
- **Output**: The constructor does not return a value but initializes an instance of the `test_diag_mask_inf` class with the specified parameters.
- **See also**: [`test_diag_mask_inf`](#test_diag_mask_inf)  (Data Structure)


---
#### test\_diag\_mask\_inf::build\_graph<!-- {{#callable:test_diag_mask_inf::build_graph}} -->
The `build_graph` function constructs a computational graph for a tensor operation in the context of a neural network.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with specified type and dimensions.
    - Sets the tensor `a` as a parameter using [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param).
    - Names the tensor `a` as 'a' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Calls [`ggml_diag_mask_inf`](../ggml/src/ggml.c.driver.md#ggml_diag_mask_inf) to create a diagonal mask tensor `out` based on `a` and `n_past`.
    - Names the output tensor `out` as 'out' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Returns the output tensor `out`.
- **Output**: Returns a pointer to a `ggml_tensor` that represents the output of the diagonal masking operation.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_diag_mask_inf`](../ggml/src/ggml.c.driver.md#ggml_diag_mask_inf)
- **See also**: [`test_diag_mask_inf`](#test_diag_mask_inf)  (Data Structure)



---
### test\_soft\_max<!-- {{#data_structure:test_soft_max}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type used for the softmax operation.
    - `ne`: An array representing the dimensions of the input tensor.
    - `mask`: A boolean indicating whether to apply a mask.
    - `m_prec`: The precision type for the mask tensor.
    - `scale`: A scaling factor applied during the softmax operation.
    - `max_bias`: A bias value that can affect the softmax output.
- **Description**: The `test_soft_max` struct is designed to test the softmax operation in a neural network context, inheriting from `test_case`. It contains parameters that define the operation's behavior, including the data type, input dimensions, masking options, precision settings, scaling factors, and bias values. This struct facilitates the construction of a computational graph for the softmax operation, allowing for performance and correctness testing.
- **Member Functions**:
    - [`test_soft_max::vars`](#test_soft_maxvars)
    - [`test_soft_max::max_nmse_err`](#test_soft_maxmax_nmse_err)
    - [`test_soft_max::test_soft_max`](#test_soft_maxtest_soft_max)
    - [`test_soft_max::build_graph`](#test_soft_maxbuild_graph)
    - [`test_soft_max::grad_precise`](#test_soft_maxgrad_precise)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_soft\_max::vars<!-- {{#callable:test_soft_max::vars}} -->
The `vars` method returns a string representation of the member variables of the `test_soft_max` class.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR6` with the member variables `type`, `ne`, `mask`, `m_prec`, `scale`, and `max_bias`.
    - The `VARS_TO_STR6` macro constructs a string representation of these variables.
- **Output**: The method returns a string that represents the values of the member variables in a formatted manner.
- **See also**: [`test_soft_max`](#test_soft_max)  (Data Structure)


---
#### test\_soft\_max::max\_nmse\_err<!-- {{#callable:test_soft_max::max_nmse_err}} -->
The `max_nmse_err` function returns a constant value representing the maximum normalized mean squared error allowed for the test case.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a constant value without any conditional logic or loops.
- **Output**: The output is a double value of 1e-6, which indicates the maximum allowable normalized mean squared error for the test case.
- **See also**: [`test_soft_max`](#test_soft_max)  (Data Structure)


---
#### test\_soft\_max::test\_soft\_max<!-- {{#callable:test_soft_max::test_soft_max}} -->
The `test_soft_max` constructor initializes a test case for the softmax operation with specified parameters.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array representing the dimensions of the tensor, defaulting to {10, 5, 4, 3}.
    - `mask`: A boolean indicating whether to use a mask, defaulting to false.
    - `m_prec`: The precision type for the mask tensor, defaulting to `GGML_TYPE_F32`.
    - `scale`: A scaling factor for the softmax operation, defaulting to 1.0.
    - `max_bias`: A bias value to be added to the softmax output, defaulting to 0.0.
- **Control Flow**:
    - The constructor initializes member variables with the provided parameters.
    - If the `mask` parameter is true, a mask tensor is created with the specified precision and dimensions.
    - The `build_graph` method is called to create the computation graph for the softmax operation.
- **Output**: The constructor does not return a value but initializes the `test_soft_max` object with the specified parameters.
- **See also**: [`test_soft_max`](#test_soft_max)  (Data Structure)


---
#### test\_soft\_max::build\_graph<!-- {{#callable:test_soft_max::build_graph}} -->
The `build_graph` function constructs a computation graph for a softmax operation with optional masking.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages the memory and state for tensor operations.
- **Control Flow**:
    - Creates a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions, and sets it as a parameter.
    - If the `mask` member variable is true, a new 2D tensor `mask` is created and named accordingly.
    - The function then computes the softmax of tensor `a` (and `mask` if applicable) using [`ggml_soft_max_ext`](../ggml/src/ggml.c.driver.md#ggml_soft_max_ext).
    - The output tensor is named 'out' and returned.
- **Output**: Returns a pointer to the output tensor resulting from the softmax operation.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`test_case::ggml_new_tensor_2d`](#test_caseggml_new_tensor_2d)
    - [`ggml_soft_max_ext`](../ggml/src/ggml.c.driver.md#ggml_soft_max_ext)
- **See also**: [`test_soft_max`](#test_soft_max)  (Data Structure)


---
#### test\_soft\_max::grad\_precise<!-- {{#callable:test_soft_max::grad_precise}} -->
The `grad_precise` function returns a boolean value indicating whether to use a precise gradient estimation method.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the boolean value 'true' without any conditions or loops.
- **Output**: The output is a boolean value 'true', indicating that the gradient estimation should be precise.
- **See also**: [`test_soft_max`](#test_soft_max)  (Data Structure)



---
### test\_soft\_max\_back<!-- {{#data_structure:test_soft_max_back}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
    - `scale`: A scaling factor for the softmax operation.
    - `max_bias`: A maximum bias value to be applied during the softmax operation.
- **Description**: The `test_soft_max_back` struct is a derived class from `test_case` that encapsulates the parameters and behavior necessary to test the backward pass of the softmax operation in a neural network context. It includes fields for the tensor type, dimensions, scaling factor, and maximum bias, which are essential for constructing the computational graph and evaluating the gradients during backpropagation.
- **Member Functions**:
    - [`test_soft_max_back::vars`](#test_soft_max_backvars)
    - [`test_soft_max_back::test_soft_max_back`](#test_soft_max_backtest_soft_max_back)
    - [`test_soft_max_back::build_graph`](#test_soft_max_backbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_soft\_max\_back::vars<!-- {{#callable:test_soft_max_back::vars}} -->
The `vars` method returns a string representation of the member variables of the `test_soft_max_back` class.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR4` with the member variables `type`, `ne`, `scale`, and `max_bias`.
    - The result of the macro call is returned as a string.
- **Output**: The output is a string that represents the values of the member variables formatted in a specific way.
- **See also**: [`test_soft_max_back`](#test_soft_max_back)  (Data Structure)


---
#### test\_soft\_max\_back::test\_soft\_max\_back<!-- {{#callable:test_soft_max_back::test_soft_max_back}} -->
The `test_soft_max_back` constructor initializes a test case for the backward pass of the softmax operation.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array representing the dimensions of the tensor, defaulting to {10, 5, 4, 3}.
    - `scale`: A scaling factor for the softmax operation, defaulting to 1.0f.
    - `max_bias`: A bias added to the softmax operation, defaulting to 0.0f.
- **Control Flow**:
    - The constructor initializes member variables with the provided arguments or default values.
    - The `type`, `ne`, `scale`, and `max_bias` are stored as constants for the test case.
- **Output**: The constructor does not return a value but initializes the `test_soft_max_back` object with the specified parameters.
- **See also**: [`test_soft_max_back`](#test_soft_max_back)  (Data Structure)


---
#### test\_soft\_max\_back::build\_graph<!-- {{#callable:test_soft_max_back::build_graph}} -->
The `build_graph` function constructs a computational graph for a softmax operation with two input tensors.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure that manages memory and resources for tensor operations.
- **Control Flow**:
    - Create a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions, and set its name to 'a'.
    - Create another tensor `b` similarly, and set its name to 'b'.
    - Call [`ggml_soft_max_ext_back`](../ggml/src/ggml.c.driver.md#ggml_soft_max_ext_back) with tensors `a`, `b`, and additional parameters `scale` and `max_bias` to compute the output tensor.
    - Set the name of the output tensor to 'out'.
    - Return the output tensor.
- **Output**: Returns a pointer to the output tensor resulting from the softmax operation applied to the input tensors.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_soft_max_ext_back`](../ggml/src/ggml.c.driver.md#ggml_soft_max_ext_back)
- **See also**: [`test_soft_max_back`](#test_soft_max_back)  (Data Structure)



---
### test\_rope<!-- {{#data_structure:test_rope}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The type of the tensor.
    - `ne_a`: An array representing the dimensions of the tensor.
    - `n_dims`: The number of dimensions of the tensor.
    - `mode`: An integer representing the mode of operation.
    - `n_ctx`: The context size used for generating positions.
    - `fs`: Frequency scale factor.
    - `ef`: External factor for scaling.
    - `af`: Attention factor.
    - `ff`: A boolean flag.
    - `v`: An integer representing the view.
    - `forward`: A boolean indicating the forward pass.
- **Description**: The `test_rope` struct is designed to represent a test case for a specific tensor operation, encapsulating various parameters such as tensor type, dimensions, and operational modes. It includes fields for managing the tensor's properties, such as frequency scaling and attention factors, and is intended for use in testing the functionality of tensor operations in a machine learning context.
- **Member Functions**:
    - [`test_rope::vars`](#test_ropevars)
    - [`test_rope::test_rope`](#test_ropetest_rope)
    - [`test_rope::build_graph`](#test_ropebuild_graph)
    - [`test_rope::initialize_tensors`](#test_ropeinitialize_tensors)
    - [`test_rope::max_maa_err`](#test_ropemax_maa_err)
    - [`test_rope::grad_precise`](#test_ropegrad_precise)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_rope::vars<!-- {{#callable:test_rope::vars}} -->
The `vars` method returns a string representation of the internal state of the `test_rope` class.
- **Inputs**:
    - `type`: The data type of the tensor.
    - `ne_a`: An array representing the dimensions of the tensor.
    - `n_dims`: The number of dimensions.
    - `mode`: An integer representing the mode of operation.
    - `n_ctx`: The context size used for generating positions.
    - `fs`: Frequency scale factor.
    - `ef`: External factor.
    - `af`: Attention factor.
    - `ff`: A boolean flag indicating a specific condition.
    - `v`: An integer indicating the view type.
    - `forward`: A boolean indicating the direction of operation.
- **Control Flow**:
    - The method constructs a string representation of the internal state using the `VARS_TO_STR10` macro.
    - The macro takes various member variables of the class as arguments to create a formatted string.
- **Output**: Returns a string that represents the current state of the `test_rope` instance, formatted according to the specified member variables.
- **See also**: [`test_rope`](#test_rope)  (Data Structure)


---
#### test\_rope::test\_rope<!-- {{#callable:test_rope::test_rope}} -->
The `test_rope` constructor initializes a `test_rope` object with specified parameters for testing the RoPE (Rotary Positional Encoding) functionality.
- **Inputs**:
    - `type`: The data type for the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne_a`: An array of four integers representing the dimensions of the tensor.
    - `n_dims`: An integer representing the number of dimensions, defaulting to 10.
    - `mode`: An integer representing the mode of operation, defaulting to 0.
    - `n_ctx`: An integer representing the context size, defaulting to 512.
    - `fs`: A float representing the frequency scale, defaulting to 1.0.
    - `ef`: A float representing the external factor, defaulting to 0.0.
    - `af`: A float representing the attention factor, defaulting to 0.0.
    - `ff`: A boolean indicating whether to use frequency factors, defaulting to false.
    - `v`: An integer representing the view, defaulting to 0.
    - `forward`: A boolean indicating the direction of operation, defaulting to true.
- **Control Flow**:
    - The constructor initializes member variables with the provided parameters.
    - If the view flag is set, it modifies the dimensions of the tensor accordingly.
    - It checks the mode to determine the type of positional encoding to apply.
    - It creates the necessary tensors for the operation based on the specified parameters.
    - The output tensor is created based on the mode and whether the operation is forward or backward.
- **Output**: The constructor does not return a value but initializes the `test_rope` object with the specified parameters for further testing.
- **See also**: [`test_rope`](#test_rope)  (Data Structure)


---
#### test\_rope::build\_graph<!-- {{#callable:test_rope::build_graph}} -->
The `build_graph` function constructs a computation graph for a tensor operation based on various parameters.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for tensor operations.
- **Control Flow**:
    - Check if the view flag `v` is set; if so, modify the dimensions of the tensor `a` accordingly.
    - Create a new tensor `a` with specified dimensions and type, and set its name.
    - Determine if the mode is for multi-rope or vision-based operations.
    - Create a position tensor `pos` based on the determined mode.
    - Optionally create a frequency tensor `freq` if the `ff` flag is set.
    - Depending on the mode, either call [`ggml_rope_multi`](../ggml/src/ggml.c.driver.md#ggml_rope_multi) or [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext) to compute the output tensor.
    - Set the name of the output tensor and return it.
- **Output**: Returns a pointer to the output tensor resulting from the computation graph.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_view_4d`](../ggml/src/ggml.c.driver.md#ggml_view_4d)
    - [`test_case::ggml_new_tensor_1d`](#test_caseggml_new_tensor_1d)
    - [`ggml_rope_multi`](../ggml/src/ggml.c.driver.md#ggml_rope_multi)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
- **See also**: [`test_rope`](#test_rope)  (Data Structure)


---
#### test\_rope::initialize\_tensors<!-- {{#callable:test_rope::initialize_tensors}} -->
The `initialize_tensors` function initializes the tensors in a given `ggml_context` based on their types and dimensions.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that contains the state and data for tensor operations.
- **Control Flow**:
    - Iterates through all tensors in the context using [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor) and [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor).
    - Checks if the tensor type is `GGML_TYPE_I32` to initialize position IDs with random values.
    - If the tensor type is not `GGML_TYPE_I32`, it checks the first dimension of the tensor to determine the initialization range for frequency factors.
    - Calls [`init_tensor_uniform`](#init_tensor_uniform) to initialize the tensor with uniform values based on the conditions.
- **Output**: The function does not return a value; it modifies the tensors in the provided `ggml_context` directly.
- **Functions called**:
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set)
    - [`init_tensor_uniform`](#init_tensor_uniform)
- **See also**: [`test_rope`](#test_rope)  (Data Structure)


---
#### test\_rope::max\_maa\_err<!-- {{#callable:test_rope::max_maa_err}} -->
The `max_maa_err` function returns a constant maximum absolute asymmetry error value.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a constant value without any conditional statements or loops.
- **Output**: The output is a double value of 1e-3, representing the maximum allowable absolute asymmetry error.
- **See also**: [`test_rope`](#test_rope)  (Data Structure)


---
#### test\_rope::grad\_precise<!-- {{#callable:test_rope::grad_precise}} -->
The `grad_precise` function always returns true, indicating that the gradient estimation method is precise.
- **Inputs**: None
- **Control Flow**:
    - The function contains no control flow statements as it directly returns a boolean value.
- **Output**: The function outputs a boolean value, specifically 'true'.
- **See also**: [`test_rope`](#test_rope)  (Data Structure)



---
### test\_pool2d<!-- {{#data_structure:test_pool2d}} -->
- **Type**: `struct`
- **Members**:
    - `pool_type`: Specifies the type of pooling operation to be performed.
    - `type_input`: Defines the data type of the input tensor.
    - `ne_input`: An array representing the dimensions of the input tensor.
    - `k0`: The height of the pooling kernel.
    - `k1`: The width of the pooling kernel.
    - `s0`: The vertical stride of the pooling operation.
    - `s1`: The horizontal stride of the pooling operation.
    - `p0`: The vertical padding applied to the input tensor.
    - `p1`: The horizontal padding applied to the input tensor.
- **Description**: The `test_pool2d` struct is designed to facilitate the testing of 2D pooling operations in a neural network context. It inherits from `test_case` and contains parameters that define the pooling operation, including the type of pooling, input tensor type, input dimensions, kernel size, stride, and padding. This struct is essential for setting up and executing tests that validate the behavior and performance of 2D pooling layers.
- **Member Functions**:
    - [`test_pool2d::vars`](#test_pool2dvars)
    - [`test_pool2d::test_pool2d`](#test_pool2dtest_pool2d)
    - [`test_pool2d::build_graph`](#test_pool2dbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_pool2d::vars<!-- {{#callable:test_pool2d::vars}} -->
The `vars` method returns a string representation of the internal state of the `test_pool2d` class.
- **Inputs**:
    - `pool_type`: An enumeration value representing the type of pooling operation (e.g., average or max pooling).
    - `type_input`: The data type of the input tensor.
    - `ne_input`: An array representing the dimensions of the input tensor.
    - `k0`: The height of the pooling kernel.
    - `k1`: The width of the pooling kernel.
    - `s0`: The vertical stride of the pooling operation.
    - `s1`: The horizontal stride of the pooling operation.
    - `p0`: The vertical padding applied to the input tensor.
    - `p1`: The horizontal padding applied to the input tensor.
- **Control Flow**:
    - The method begins by calling `ggml_new_tensor` to create a new tensor for the input based on the specified `type_input` and `ne_input` dimensions.
    - It sets the parameter for the input tensor using `ggml_set_param`.
    - The method then calls `ggml_pool_2d` to perform the pooling operation on the input tensor with the specified parameters.
    - Finally, it sets the name of the output tensor to 'out' and returns it.
- **Output**: The output is a tensor resulting from the pooling operation, which is named 'out'.
- **See also**: [`test_pool2d`](#test_pool2d)  (Data Structure)


---
#### test\_pool2d::test\_pool2d<!-- {{#callable:test_pool2d::test_pool2d}} -->
The `test_pool2d` constructor initializes a test case for 2D pooling operations with specified parameters.
- **Inputs**:
    - `pool_type`: Specifies the type of pooling operation (e.g., average or max pooling).
    - `type_input`: The data type of the input tensor.
    - `ne_input`: An array representing the dimensions of the input tensor, specifically [input_width, input_height, input_channels, batch_size].
    - `k0`: The height of the pooling kernel.
    - `k1`: The width of the pooling kernel.
    - `s0`: The vertical stride of the pooling operation.
    - `s1`: The horizontal stride of the pooling operation.
    - `p0`: The vertical padding applied to the input tensor.
    - `p1`: The horizontal padding applied to the input tensor.
- **Control Flow**:
    - The constructor initializes member variables with the provided parameters.
    - The `build_graph` method is called to create the computation graph for the pooling operation.
    - A new tensor for the input is created with the specified dimensions and type.
    - The pooling operation is performed using the specified parameters, and the output tensor is generated.
- **Output**: The output is a tensor resulting from the 2D pooling operation applied to the input tensor.
- **See also**: [`test_pool2d`](#test_pool2d)  (Data Structure)


---
#### test\_pool2d::build\_graph<!-- {{#callable:test_pool2d::build_graph}} -->
The `build_graph` function constructs a computational graph for a 2D pooling operation.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for the graph.
- **Control Flow**:
    - Creates a new tensor `input` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with specified dimensions and type.
    - Sets the parameter flag for the `input` tensor using [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param).
    - Names the `input` tensor as 'input' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Performs a 2D pooling operation on the `input` tensor using [`ggml_pool_2d`](../ggml/src/ggml.c.driver.md#ggml_pool_2d), storing the result in `out`.
    - Names the output tensor `out` as 'out' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Returns the `out` tensor.
- **Output**: Returns a pointer to the output tensor resulting from the 2D pooling operation.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_pool_2d`](../ggml/src/ggml.c.driver.md#ggml_pool_2d)
- **See also**: [`test_pool2d`](#test_pool2d)  (Data Structure)



---
### test\_conv\_transpose\_1d<!-- {{#data_structure:test_conv_transpose_1d}} -->
- **Type**: `struct`
- **Members**:
    - `ne_input`: An array representing the dimensions of the input tensor.
    - `ne_kernel`: An array representing the dimensions of the kernel tensor.
    - `s0`: An integer representing the stride for the convolution operation.
    - `p0`: An integer representing the padding for the convolution operation.
    - `d0`: An integer representing the dilation for the convolution operation.
- **Description**: The `test_conv_transpose_1d` struct is designed to test the 1D transposed convolution operation, inheriting from `test_case`. It contains parameters for the input tensor dimensions, kernel dimensions, stride, padding, and dilation, which are essential for configuring the convolution operation. The struct also includes a method to generate a string representation of its variables for debugging purposes.
- **Member Functions**:
    - [`test_conv_transpose_1d::vars`](#test_conv_transpose_1dvars)
    - [`test_conv_transpose_1d::test_conv_transpose_1d`](#test_conv_transpose_1dtest_conv_transpose_1d)
    - [`test_conv_transpose_1d::build_graph`](#test_conv_transpose_1dbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_conv\_transpose\_1d::vars<!-- {{#callable:test_conv_transpose_1d::vars}} -->
The `vars` method returns a string representation of the internal state variables of the `test_conv_transpose_1d` class.
- **Inputs**:
    - `none`: This method does not take any input parameters.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR5` with the member variables `ne_input`, `ne_kernel`, `s0`, `p0`, and `d0`.
    - The `VARS_TO_STR5` macro formats these variables into a string representation.
- **Output**: The method returns a string that represents the values of the member variables of the class, formatted for easy readability.
- **See also**: [`test_conv_transpose_1d`](#test_conv_transpose_1d)  (Data Structure)


---
#### test\_conv\_transpose\_1d::test\_conv\_transpose\_1d<!-- {{#callable:test_conv_transpose_1d::test_conv_transpose_1d}} -->
The `test_conv_transpose_1d` constructor initializes a test case for a 1D transposed convolution operation with specified input and kernel dimensions, stride, padding, and dilation.
- **Inputs**:
    - `ne_input`: An array of four integers representing the dimensions of the input tensor, specifically [input_width, input_channels, 1 (assert in cpu kernel), 1 (should be batch)].
    - `ne_kernel`: An array of four integers representing the dimensions of the kernel tensor, specifically [kernel_width, output_channels, input_channels, 1 (should be batch)].
    - `s0`: An integer representing the stride for the convolution operation.
    - `p0`: An integer representing the padding for the convolution operation.
    - `d0`: An integer representing the dilation for the convolution operation.
- **Control Flow**:
    - The constructor initializes member variables with the provided input parameters.
    - The `vars` method is overridden to return a string representation of the input and kernel dimensions, stride, padding, and dilation.
    - The `build_graph` method creates input and kernel tensors, applies the transposed convolution operation, and returns the output tensor.
- **Output**: The constructor does not return a value but initializes the test case with the specified parameters for later use in testing the transposed convolution operation.
- **See also**: [`test_conv_transpose_1d`](#test_conv_transpose_1d)  (Data Structure)


---
#### test\_conv\_transpose\_1d::build\_graph<!-- {{#callable:test_conv_transpose_1d::build_graph}} -->
The `build_graph` function constructs a computational graph for a transposed convolution operation.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and resources for the graph.
- **Control Flow**:
    - Creates a new tensor `input` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with type `GGML_TYPE_F32` and dimensions specified by `ne_input`.
    - Sets the name of the `input` tensor to 'input' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Creates a new tensor `kernel` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with type `GGML_TYPE_F32` and dimensions specified by `ne_kernel`.
    - Sets the name of the `kernel` tensor to 'kernel' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Calls [`ggml_conv_transpose_1d`](../ggml/src/ggml.c.driver.md#ggml_conv_transpose_1d) to perform the transposed convolution operation using `kernel` and `input`, along with stride, padding, and dilation parameters.
    - Sets the name of the output tensor `out` to 'out' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Returns the output tensor `out`.
- **Output**: Returns a pointer to the output tensor resulting from the transposed convolution operation.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_conv_transpose_1d`](../ggml/src/ggml.c.driver.md#ggml_conv_transpose_1d)
- **See also**: [`test_conv_transpose_1d`](#test_conv_transpose_1d)  (Data Structure)



---
### test\_im2col<!-- {{#data_structure:test_im2col}} -->
- **Type**: `struct`
- **Members**:
    - `type_input`: The data type of the input tensor.
    - `type_kernel`: The data type of the kernel tensor.
    - `dst_type`: The data type of the output tensor.
    - `ne_input`: The dimensions of the input tensor.
    - `ne_kernel`: The dimensions of the kernel tensor.
    - `s0`: The stride in the first dimension.
    - `s1`: The stride in the second dimension.
    - `p0`: The padding in the first dimension.
    - `p1`: The padding in the second dimension.
    - `d0`: The dilation in the first dimension.
    - `d1`: The dilation in the second dimension.
    - `is_2D`: A boolean indicating if the operation is 2D.
- **Description**: The `test_im2col` struct is designed to facilitate testing of the im2col operation, which transforms image data into a column format suitable for convolution operations. It contains parameters defining the input and kernel tensor types, their dimensions, as well as parameters for stride, padding, and dilation, allowing for flexible configuration of the convolution operation.
- **Member Functions**:
    - [`test_im2col::vars`](#test_im2colvars)
    - [`test_im2col::test_im2col`](#test_im2coltest_im2col)
    - [`test_im2col::build_graph`](#test_im2colbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_im2col::vars<!-- {{#callable:test_im2col::vars}} -->
The `vars` method returns a string representation of the internal state variables of the `test_im2col` class.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the `VARS_TO_STR12` macro with the class's member variables to format them into a string.
    - The formatted string includes types and dimensions of input and kernel tensors, stride, padding, dilation, and whether the operation is 2D.
- **Output**: The method returns a `std::string` that represents the formatted state of the `test_im2col` instance.
- **See also**: [`test_im2col`](#test_im2col)  (Data Structure)


---
#### test\_im2col::test\_im2col<!-- {{#callable:test_im2col::test_im2col}} -->
The `test_im2col` function is a constructor for the `test_im2col` class that initializes parameters for testing the im2col operation in a neural network context.
- **Inputs**:
    - `type_input`: The data type of the input tensor, defaulting to `GGML_TYPE_F32`.
    - `type_kernel`: The data type of the kernel tensor, defaulting to `GGML_TYPE_F16`.
    - `dst_type`: The data type of the output tensor, defaulting to `GGML_TYPE_F32`.
    - `ne_input`: An array representing the dimensions of the input tensor, defaulting to {10, 10, 3, 1}.
    - `ne_kernel`: An array representing the dimensions of the kernel tensor, defaulting to {3, 3, 3, 1}.
    - `s0`: The stride in the first dimension, defaulting to 1.
    - `s1`: The stride in the second dimension, defaulting to 1.
    - `p0`: The padding in the first dimension, defaulting to 1.
    - `p1`: The padding in the second dimension, defaulting to 1.
    - `d0`: The dilation in the first dimension, defaulting to 1.
    - `d1`: The dilation in the second dimension, defaulting to 1.
    - `is_2D`: A boolean indicating whether the operation is 2D, defaulting to true.
- **Control Flow**:
    - The constructor initializes member variables with the provided parameters or defaults.
    - It sets the types for input, kernel, and output tensors, as well as their respective dimensions.
    - It also configures the stride, padding, dilation, and whether the operation is 2D.
- **Output**: The constructor does not return a value but initializes an instance of the `test_im2col` class with the specified parameters.
- **See also**: [`test_im2col`](#test_im2col)  (Data Structure)


---
#### test\_im2col::build\_graph<!-- {{#callable:test_im2col::build_graph}} -->
The `build_graph` function constructs a computational graph for a neural network operation involving input and kernel tensors.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a new tensor `input` of type `type_input` with dimensions specified by `ne_input`.
    - Sets the parameter flag for the `input` tensor.
    - Names the `input` tensor as 'input'.
    - Creates a new tensor `kernel` of type `type_kernel` with dimensions specified by `ne_kernel`.
    - Names the `kernel` tensor as 'kernel'.
    - Calls the [`ggml_im2col`](../ggml/src/ggml.c.driver.md#ggml_im2col) function to perform the im2col operation, passing the `kernel`, `input`, and various parameters for stride, padding, and dilation.
    - Names the output tensor from [`ggml_im2col`](../ggml/src/ggml.c.driver.md#ggml_im2col) as 'out'.
    - Returns the output tensor `out`.
- **Output**: Returns a pointer to the output tensor resulting from the im2col operation, which is named 'out'.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_im2col`](../ggml/src/ggml.c.driver.md#ggml_im2col)
- **See also**: [`test_im2col`](#test_im2col)  (Data Structure)



---
### test\_conv\_2d\_dw<!-- {{#data_structure:test_conv_2d_dw}} -->
- **Type**: `struct`
- **Members**:
    - `ne_input`: An array representing the dimensions of the input tensor.
    - `ne_kernel`: An array representing the dimensions of the kernel tensor.
    - `stride`: An integer representing the stride for the convolution operation.
    - `padding`: An integer representing the padding applied to the input tensor.
    - `dilation`: An integer representing the dilation rate for the convolution.
    - `cwhn`: A boolean indicating whether the input is in channel-most-contiguous format.
- **Description**: The `test_conv_2d_dw` struct is designed to represent a test case for a 2D depthwise convolution operation, inheriting from `test_case`. It contains parameters for the input tensor dimensions, kernel dimensions, stride, padding, dilation, and a flag for the memory layout of the input tensor. This struct is used to build a computational graph for testing the convolution operation in a neural network context.
- **Member Functions**:
    - [`test_conv_2d_dw::vars`](#test_conv_2d_dwvars)
    - [`test_conv_2d_dw::test_conv_2d_dw`](#test_conv_2d_dwtest_conv_2d_dw)
    - [`test_conv_2d_dw::build_graph`](#test_conv_2d_dwbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_conv\_2d\_dw::vars<!-- {{#callable:test_conv_2d_dw::vars}} -->
The `vars` method returns a string representation of the internal parameters of the `test_conv_2d_dw` class.
- **Inputs**:
    - `ne_input`: An array of four integers representing the dimensions of the input tensor.
    - `ne_kernel`: An array of four integers representing the dimensions of the kernel tensor.
    - `stride`: An integer representing the stride for the convolution operation.
    - `padding`: An integer representing the padding applied to the input tensor.
    - `dilation`: An integer representing the dilation rate for the convolution operation.
    - `cwhn`: A boolean indicating whether the input and kernel tensors are in channel-most-contiguous format.
- **Control Flow**:
    - The method constructs a string by calling the macro `VARS_TO_STR6` with the parameters `ne_input`, `ne_kernel`, `stride`, `padding`, `dilation`, and `cwhn`.
    - The `VARS_TO_STR6` macro formats these parameters into a string representation.
- **Output**: Returns a string that represents the values of the parameters of the `test_conv_2d_dw` instance.
- **See also**: [`test_conv_2d_dw`](#test_conv_2d_dw)  (Data Structure)


---
#### test\_conv\_2d\_dw::test\_conv\_2d\_dw<!-- {{#callable:test_conv_2d_dw::test_conv_2d_dw}} -->
The `test_conv_2d_dw` constructor initializes a 2D depthwise convolution test case with specified input and kernel dimensions, stride, padding, dilation, and memory layout.
- **Inputs**:
    - `ne_input`: An array of four integers representing the dimensions of the input tensor, typically in the format {height, width, channels, batch_size}.
    - `ne_kernel`: An array of four integers representing the dimensions of the kernel tensor, typically in the format {kernel_height, kernel_width, input_channels, output_channels}.
    - `stride`: An integer representing the stride of the convolution operation.
    - `padding`: An integer representing the amount of zero-padding added to both sides of the input.
    - `dilation`: An integer representing the dilation rate of the convolution.
    - `cwhn`: A boolean indicating whether the input and kernel tensors are in channel-most-contiguous format (CWHN) or not.
- **Control Flow**:
    - The constructor initializes member variables with the provided parameters.
    - If `cwhn` is true, it permutes the input and kernel tensors to change their memory layout to channel-most-contiguous format.
    - The `build_graph` method is called to create the computation graph for the convolution operation.
- **Output**: The constructor does not return a value but initializes an instance of the `test_conv_2d_dw` class, which can be used to build a computation graph for a 2D depthwise convolution operation.
- **See also**: [`test_conv_2d_dw`](#test_conv_2d_dw)  (Data Structure)


---
#### test\_conv\_2d\_dw::build\_graph<!-- {{#callable:test_conv_2d_dw::build_graph}} -->
The `build_graph` function constructs a computational graph for a 2D depthwise convolution operation.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for the graph.
- **Control Flow**:
    - Creates a new tensor `input` from `ne_input` data and names it 'input'.
    - Creates a new tensor `kernel` from `ne_kernel` data and names it 'kernel'.
    - If `cwhn` is true, it permutes the memory layout of `input` and `kernel` to channel-most-contiguous format.
    - Calls [`ggml_conv_2d_dw_direct`](../ggml/src/ggml.c.driver.md#ggml_conv_2d_dw_direct) to perform the depthwise convolution using the `kernel` and `input` tensors with specified stride, padding, and dilation.
    - Names the output tensor 'out' and returns it.
- **Output**: Returns a pointer to the output tensor resulting from the depthwise convolution operation.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_permute`](../ggml/src/ggml.c.driver.md#ggml_permute)
    - [`ggml_conv_2d_dw_direct`](../ggml/src/ggml.c.driver.md#ggml_conv_2d_dw_direct)
- **See also**: [`test_conv_2d_dw`](#test_conv_2d_dw)  (Data Structure)



---
### test\_concat<!-- {{#data_structure:test_concat}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensors.
    - `ne_a`: An array representing the dimensions of the first tensor.
    - `ne_b_d`: The dimension size for the second tensor along the specified dimension.
    - `dim`: The dimension along which the concatenation occurs.
    - `v`: An integer representing view flags for the tensors.
- **Description**: The `test_concat` struct is designed to facilitate the testing of tensor concatenation operations in a neural network context. It inherits from `test_case` and contains fields that define the types and dimensions of the tensors involved in the concatenation operation. The struct includes parameters for the tensor types, their dimensions, and flags for handling non-contiguous views, allowing for flexible testing scenarios.
- **Member Functions**:
    - [`test_concat::vars`](#test_concatvars)
    - [`test_concat::test_concat`](#test_concattest_concat)
    - [`test_concat::build_graph`](#test_concatbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_concat::vars<!-- {{#callable:test_concat::vars}} -->
The `vars` method returns a string representation of the member variables of the `test_concat` class.
- **Inputs**:
    - `type`: The data type of the tensors used in the concatenation.
    - `ne_a`: An array representing the dimensions of the first tensor to be concatenated.
    - `ne_b_d`: The size of the dimension along which the second tensor will be concatenated.
    - `dim`: The dimension along which the concatenation will occur.
    - `v`: An integer flag indicating whether the tensors are views (non-contiguous).
- **Control Flow**:
    - The method constructs a string representation of the member variables using the `VARS_TO_STR5` macro.
    - It includes the type, dimensions of the first tensor, the size of the second tensor's dimension, and the view flag.
- **Output**: The output is a string that represents the values of the member variables of the `test_concat` class, formatted for easy readability.
- **See also**: [`test_concat`](#test_concat)  (Data Structure)


---
#### test\_concat::test\_concat<!-- {{#callable:test_concat::test_concat}} -->
The `test_concat` constructor initializes a test case for concatenating tensors along a specified dimension.
- **Inputs**:
    - `type`: The data type of the tensors to be concatenated, defaulting to `GGML_TYPE_F32`.
    - `ne_a`: An array representing the dimensions of the first tensor to be concatenated, defaulting to {10, 5, 5, 5}.
    - `ne_b_d`: The size of the dimension along which the second tensor will be concatenated.
    - `dim`: The dimension along which the concatenation will occur, defaulting to 2.
    - `v`: An integer flag indicating whether the tensors are views (non-contiguous), defaulting to 0.
- **Control Flow**:
    - The constructor initializes the member variables with the provided parameters.
    - If the view flag `v` indicates that the first tensor is non-contiguous, it modifies the dimensions of the first tensor accordingly.
    - It creates a second tensor with dimensions based on `ne_a` and `ne_b_d`.
    - The method `ggml_concat` is called to concatenate the two tensors along the specified dimension.
- **Output**: The output is a tensor resulting from the concatenation of the two input tensors along the specified dimension.
- **See also**: [`test_concat`](#test_concat)  (Data Structure)


---
#### test\_concat::build\_graph<!-- {{#callable:test_concat::build_graph}} -->
The `build_graph` function constructs a computational graph for tensor operations based on specified parameters.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for tensor operations.
- **Control Flow**:
    - The function initializes a new tensor `a` based on the value of `v` and the dimensions specified in `ne_a`.
    - If `v` has the least significant bit set, it modifies the dimensions of `ne_a` and creates a view of the tensor.
    - A second tensor `b` is created similarly based on `ne_b` derived from `ne_a`.
    - The function concatenates tensors `a` and `b` along the specified dimension `dim` and returns the resulting tensor.
- **Output**: Returns a pointer to the resulting concatenated tensor after performing the specified operations.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_view_4d`](../ggml/src/ggml.c.driver.md#ggml_view_4d)
    - [`ggml_concat`](../ggml/src/ggml.c.driver.md#ggml_concat)
- **See also**: [`test_concat`](#test_concat)  (Data Structure)



---
### test\_argsort<!-- {{#data_structure:test_argsort}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
    - `order`: The sorting order for the argsort operation.
- **Description**: The `test_argsort` struct is a derived class from `test_case` that encapsulates the parameters and behavior for testing the argsort operation on tensors, including the data type, dimensions, and sorting order.
- **Member Functions**:
    - [`test_argsort::vars`](#test_argsortvars)
    - [`test_argsort::test_argsort`](#test_argsorttest_argsort)
    - [`test_argsort::build_graph`](#test_argsortbuild_graph)
    - [`test_argsort::initialize_tensors`](#test_argsortinitialize_tensors)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_argsort::vars<!-- {{#callable:test_argsort::vars}} -->
The `vars` method returns a string representation of the internal state of the `test_argsort` class.
- **Inputs**: None
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR3` with the class's member variables `type`, `ne`, and `order`.
    - The `VARS_TO_STR3` macro formats these member variables into a string representation.
- **Output**: The output is a string that represents the values of `type`, `ne`, and `order` in a formatted manner.
- **See also**: [`test_argsort`](#test_argsort)  (Data Structure)


---
#### test\_argsort::test\_argsort<!-- {{#callable:test_argsort::test_argsort}} -->
The `test_argsort` constructor initializes a test case for sorting tensors based on specified parameters.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array representing the dimensions of the tensor, defaulting to {16, 10, 10, 10}.
    - `order`: The sorting order, defaulting to `GGML_SORT_ORDER_ASC`.
- **Control Flow**:
    - The constructor initializes the member variables `type`, `ne`, and `order` with the provided arguments or defaults.
    - The `vars` method is overridden to return a string representation of the parameters for logging or debugging purposes.
- **Output**: The constructor does not return a value but initializes an instance of the `test_argsort` class with the specified parameters.
- **See also**: [`test_argsort`](#test_argsort)  (Data Structure)


---
#### test\_argsort::build\_graph<!-- {{#callable:test_argsort::build_graph}} -->
The `build_graph` function constructs a computation graph for sorting a tensor.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for the tensor operations.
- **Control Flow**:
    - Creates a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions.
    - Sets the name of tensor `a` to 'a' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Calls [`ggml_argsort`](../ggml/src/ggml.c.driver.md#ggml_argsort) to perform sorting on tensor `a` based on the specified order, storing the result in tensor `out`.
    - Sets the name of tensor `out` to 'out'.
    - Returns the sorted tensor `out`.
- **Output**: Returns a pointer to the sorted tensor `out`.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_argsort`](../ggml/src/ggml.c.driver.md#ggml_argsort)
- **See also**: [`test_argsort`](#test_argsort)  (Data Structure)


---
#### test\_argsort::initialize\_tensors<!-- {{#callable:test_argsort::initialize_tensors}} -->
Initializes tensors in a `ggml_context` with random values based on their type.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that contains the tensors to be initialized.
- **Control Flow**:
    - A random number generator is initialized using `std::random_device` and `std::default_random_engine`.
    - A loop iterates over all tensors in the context using [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor) and [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor).
    - For each tensor, the type is checked: if it is `GGML_TYPE_I32`, it initializes the tensor with random integers; if it is `GGML_TYPE_F32`, it initializes it with unique float values.
    - The data for each tensor is shuffled to ensure randomness before being set in the backend using [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set).
    - If the tensor type is neither `GGML_TYPE_I32` nor `GGML_TYPE_F32`, the function aborts with a fatal error.
- **Output**: The function does not return a value; it modifies the tensors in the provided `ggml_context` directly.
- **Functions called**:
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`ggml_nelements`](../ggml/src/ggml.c.driver.md#ggml_nelements)
    - [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set)
    - [`ggml_nrows`](../ggml/src/ggml.c.driver.md#ggml_nrows)
- **See also**: [`test_argsort`](#test_argsort)  (Data Structure)



---
### test\_sum<!-- {{#data_structure:test_sum}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
- **Description**: The `test_sum` struct is a derived class from `test_case` that is designed to test the summation operation on tensors of a specified type and shape. It contains a member `type` that indicates the data type of the tensor and an array `ne` that defines the dimensions of the tensor. The struct overrides the `vars` method to return a string representation of its parameters and the `build_graph` method to create a computation graph for the summation operation.
- **Member Functions**:
    - [`test_sum::vars`](#test_sumvars)
    - [`test_sum::test_sum`](#test_sumtest_sum)
    - [`test_sum::build_graph`](#test_sumbuild_graph)
    - [`test_sum::grad_eps`](#test_sumgrad_eps)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_sum::vars<!-- {{#callable:test_sum::vars}} -->
The `vars` method returns a string representation of the `test_sum` class's parameters.
- **Inputs**: None
- **Control Flow**:
    - The method calls the `VARS_TO_STR2` macro with `type` and `ne` as arguments.
    - The result of the macro call is returned as a string.
- **Output**: The output is a string that represents the `type` and `ne` attributes of the `test_sum` instance.
- **See also**: [`test_sum`](#test_sum)  (Data Structure)


---
#### test\_sum::test\_sum<!-- {{#callable:test_sum::test_sum}} -->
The `test_sum` constructor initializes a test case for summing tensors with specified data type and dimensions.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array of four integers representing the dimensions of the tensor, defaulting to {10, 5, 4, 3}.
- **Control Flow**:
    - The constructor initializes the member variables `type` and `ne` with the provided arguments.
    - If no arguments are provided, it uses the default values for `type` and `ne`.
- **Output**: The constructor does not return a value but initializes the `test_sum` object with the specified tensor type and dimensions.
- **See also**: [`test_sum`](#test_sum)  (Data Structure)


---
#### test\_sum::build\_graph<!-- {{#callable:test_sum::build_graph}} -->
The `build_graph` function constructs a computational graph for a tensor operation.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations.
- **Control Flow**:
    - Creates a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions.
    - Sets the parameter flag for tensor `a` using [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param).
    - Names the tensor `a` as 'a' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Calculates the sum of tensor `a` using [`ggml_sum`](../ggml/src/ggml.c.driver.md#ggml_sum), storing the result in tensor `out`.
    - Names the output tensor `out` as 'out' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Returns the output tensor `out`.
- **Output**: Returns a pointer to the output tensor `out`, which contains the sum of the input tensor `a`.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_sum`](../ggml/src/ggml.c.driver.md#ggml_sum)
- **See also**: [`test_sum`](#test_sum)  (Data Structure)


---
#### test\_sum::grad\_eps<!-- {{#callable:test_sum::grad_eps}} -->
Calculates the gradient epsilon value based on the dimensions of the tensor.
- **Inputs**: None
- **Control Flow**:
    - The function computes the product of the four dimensions of the `ne` array.
    - It then takes the square root of this product.
    - Finally, it multiplies the result by 0.1 and returns the value.
- **Output**: Returns a float value representing the gradient epsilon, calculated as 0.1 times the square root of the product of the four dimensions in the `ne` array.
- **See also**: [`test_sum`](#test_sum)  (Data Structure)



---
### test\_sum\_rows<!-- {{#data_structure:test_sum_rows}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
- **Description**: The `test_sum_rows` struct is a derived class from `test_case` that is designed to test the summation of rows in a tensor. It contains a `ggml_type` member to specify the data type of the tensor and an array of four integers to define the tensor's dimensions. The struct overrides the `vars` method to return a string representation of its parameters and provides a constructor to initialize its members.
- **Member Functions**:
    - [`test_sum_rows::vars`](#test_sum_rowsvars)
    - [`test_sum_rows::test_sum_rows`](#test_sum_rowstest_sum_rows)
    - [`test_sum_rows::build_graph`](#test_sum_rowsbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_sum\_rows::vars<!-- {{#callable:test_sum_rows::vars}} -->
The `vars` method returns a string representation of the `test_sum_rows` class's parameters.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method directly calls the `VARS_TO_STR2` macro with the class's member variables `type` and `ne` as arguments.
    - The `VARS_TO_STR2` macro formats these variables into a string representation.
- **Output**: The output is a string that represents the `type` and `ne` member variables of the `test_sum_rows` class.
- **See also**: [`test_sum_rows`](#test_sum_rows)  (Data Structure)


---
#### test\_sum\_rows::test\_sum\_rows<!-- {{#callable:test_sum_rows::test_sum_rows}} -->
The `test_sum_rows` constructor initializes a test case for summing rows of a tensor.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array representing the dimensions of the tensor, defaulting to {10, 5, 4, 3}.
- **Control Flow**:
    - The constructor initializes the member variables `type` and `ne` with the provided arguments.
    - If no arguments are provided, default values are used.
- **Output**: The constructor does not return a value but initializes an instance of the `test_sum_rows` class.
- **See also**: [`test_sum_rows`](#test_sum_rows)  (Data Structure)


---
#### test\_sum\_rows::build\_graph<!-- {{#callable:test_sum_rows::build_graph}} -->
The `build_graph` method constructs a computational graph for a tensor operation.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which holds the context for tensor operations.
- **Control Flow**:
    - Creates a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions.
    - Sets the parameter flag for tensor `a` using [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param).
    - Names the tensor `a` as 'a' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Calculates the sum of the rows of tensor `a` using [`ggml_sum_rows`](../ggml/src/ggml.c.driver.md#ggml_sum_rows), storing the result in tensor `out`.
    - Names the output tensor `out` as 'out' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Returns the output tensor `out`.
- **Output**: Returns a pointer to the output tensor `out`, which contains the sum of the rows of tensor `a`.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_sum_rows`](../ggml/src/ggml.c.driver.md#ggml_sum_rows)
- **See also**: [`test_sum_rows`](#test_sum_rows)  (Data Structure)



---
### test\_mean<!-- {{#data_structure:test_mean}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
- **Description**: The `test_mean` struct is a derived class from `test_case` that encapsulates the properties and behavior necessary to test the mean operation on tensors, including the tensor's data type and its dimensions.
- **Member Functions**:
    - [`test_mean::vars`](#test_meanvars)
    - [`test_mean::test_mean`](#test_meantest_mean)
    - [`test_mean::build_graph`](#test_meanbuild_graph)
    - [`test_mean::grad_eps`](#test_meangrad_eps)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_mean::vars<!-- {{#callable:test_mean::vars}} -->
The `vars` method returns a string representation of the `test_mean` class's parameters.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR2` with `type` and `ne` as arguments.
    - The `VARS_TO_STR2` macro formats the parameters into a string representation.
- **Output**: The output is a string that represents the `type` and `ne` attributes of the `test_mean` instance.
- **See also**: [`test_mean`](#test_mean)  (Data Structure)


---
#### test\_mean::test\_mean<!-- {{#callable:test_mean::test_mean}} -->
The `test_mean` constructor initializes a test case for computing the mean of a tensor with specified data type and dimensions.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array of four integers representing the dimensions of the tensor, defaulting to {10, 5, 4, 3}.
- **Control Flow**:
    - The constructor initializes the member variables `type` and `ne` with the provided arguments.
    - If no arguments are provided, default values are used.
- **Output**: The constructor does not return a value but initializes an instance of the `test_mean` class with the specified tensor type and dimensions.
- **See also**: [`test_mean`](#test_mean)  (Data Structure)


---
#### test\_mean::build\_graph<!-- {{#callable:test_mean::build_graph}} -->
The `build_graph` method constructs a computational graph for a mean operation on a tensor.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and state for tensor operations.
- **Control Flow**:
    - Creates a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions.
    - Sets the tensor `a` as a parameter using [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param).
    - Names the tensor `a` as 'a' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Calculates the mean of tensor `a` using [`ggml_mean`](../ggml/src/ggml.c.driver.md#ggml_mean), storing the result in tensor `out`.
    - Names the output tensor `out` as 'out' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Returns the output tensor `out`.
- **Output**: Returns a pointer to the output tensor `out`, which contains the mean of the input tensor `a`.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_mean`](../ggml/src/ggml.c.driver.md#ggml_mean)
- **See also**: [`test_mean`](#test_mean)  (Data Structure)


---
#### test\_mean::grad\_eps<!-- {{#callable:test_mean::grad_eps}} -->
Calculates the gradient epsilon value based on the dimensions of the tensor.
- **Inputs**: None
- **Control Flow**:
    - The function directly computes the product of the first four elements of the `ne` array.
    - The result is multiplied by 0.1f.
- **Output**: Returns a float value representing the computed gradient epsilon.
- **See also**: [`test_mean`](#test_mean)  (Data Structure)



---
### test\_upscale<!-- {{#data_structure:test_upscale}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
    - `scale_factor`: The factor by which to upscale the tensor.
    - `transpose`: A boolean indicating whether to transpose the tensor.
    - `mode`: The scaling mode to be used during the upscaling process.
- **Description**: The `test_upscale` struct is designed to define a test case for upscaling operations on tensors, inheriting from `test_case`. It contains parameters such as the tensor's data type, its dimensions, a scale factor for upscaling, a transpose flag, and a scaling mode, which collectively determine how the tensor will be processed during the test.
- **Member Functions**:
    - [`test_upscale::vars`](#test_upscalevars)
    - [`test_upscale::test_upscale`](#test_upscaletest_upscale)
    - [`test_upscale::build_graph`](#test_upscalebuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_upscale::vars<!-- {{#callable:test_upscale::vars}} -->
The `vars` method returns a string representation of the member variables of the `test_upscale` class.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR5` with the member variables `type`, `ne`, `scale_factor`, `mode`, and `transpose`.
    - The `VARS_TO_STR5` macro constructs a string representation of these variables.
- **Output**: The method returns a string that represents the values of the member variables of the `test_upscale` class.
- **See also**: [`test_upscale`](#test_upscale)  (Data Structure)


---
#### test\_upscale::test\_upscale<!-- {{#callable:test_upscale::test_upscale}} -->
The `test_upscale` constructor initializes a test case for the upscale operation with specified parameters.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array representing the dimensions of the tensor, defaulting to {512, 512, 3, 1}.
    - `scale_factor`: An integer representing the factor by which to upscale the tensor, defaulting to 2.
    - `mode`: The scaling mode to use, defaulting to `GGML_SCALE_MODE_NEAREST`.
    - `transpose`: A boolean indicating whether to transpose the tensor before upscaling, defaulting to false.
- **Control Flow**:
    - The constructor initializes member variables with the provided parameters.
    - The `vars` method is overridden to return a string representation of the parameters for debugging purposes.
    - The `build_graph` method creates a tensor and applies the upscale operation based on the specified parameters.
- **Output**: The constructor does not return a value but initializes the `test_upscale` object with the specified parameters.
- **See also**: [`test_upscale`](#test_upscale)  (Data Structure)


---
#### test\_upscale::build\_graph<!-- {{#callable:test_upscale::build_graph}} -->
The `build_graph` function constructs a computational graph for a tensor operation, optionally transposing the tensor and applying an upscale operation.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure that manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with specified type and dimensions.
    - Sets the name of tensor `a` to 'a'.
    - If the `transpose` flag is true, the tensor `a` is transposed and its name is updated to 'a_transposed'.
    - An upscale operation is performed on tensor `a` using [`ggml_upscale`](../ggml/src/ggml.c.driver.md#ggml_upscale), producing the output tensor `out`.
    - The name of the output tensor `out` is set to 'out'.
    - The function returns the output tensor `out`.
- **Output**: Returns a pointer to the output tensor `out`, which is the result of the upscale operation applied to tensor `a`.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_transpose`](../ggml/src/ggml.c.driver.md#ggml_transpose)
    - [`ggml_upscale`](../ggml/src/ggml.c.driver.md#ggml_upscale)
- **See also**: [`test_upscale`](#test_upscale)  (Data Structure)



---
### test\_upscale\_ext<!-- {{#data_structure:test_upscale_ext}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the input tensor.
    - `ne_tgt`: An array representing the target dimensions for the upscale operation.
    - `mode`: The scaling mode used for the upscale operation.
- **Description**: The `test_upscale_ext` struct is designed to define a test case for an upscale operation in a tensor processing context, inheriting from `test_case`. It contains fields for the tensor's data type, its dimensions, the target dimensions for the upscale operation, and the scaling mode to be used. This struct facilitates the construction of a computational graph for testing the upscale functionality.
- **Member Functions**:
    - [`test_upscale_ext::vars`](#test_upscale_extvars)
    - [`test_upscale_ext::test_upscale_ext`](#test_upscale_exttest_upscale_ext)
    - [`test_upscale_ext::build_graph`](#test_upscale_extbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_upscale\_ext::vars<!-- {{#callable:test_upscale_ext::vars}} -->
The `vars` method returns a string representation of the object's member variables formatted for debugging.
- **Inputs**:
    - `none`: This method does not take any input parameters.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR4` with the member variables `type`, `ne`, `ne_tgt`, and `mode`.
    - The `VARS_TO_STR4` macro formats these variables into a string representation.
- **Output**: The method returns a formatted string that represents the values of the member variables of the class.
- **See also**: [`test_upscale_ext`](#test_upscale_ext)  (Data Structure)


---
#### test\_upscale\_ext::test\_upscale\_ext<!-- {{#callable:test_upscale_ext::test_upscale_ext}} -->
The `test_upscale_ext` constructor initializes a test case for the upscale operation with specified tensor types, source dimensions, target dimensions, and scaling mode.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array of four integers representing the dimensions of the source tensor.
    - `ne_tgt`: An array of four integers representing the target dimensions after upscaling.
    - `mode`: The scaling mode used for the upscale operation, defaulting to `GGML_SCALE_MODE_NEAREST`.
- **Control Flow**:
    - The constructor initializes member variables with the provided parameters.
    - The `build_graph` method creates a new tensor of the specified type and dimensions.
    - It then calls the `ggml_upscale_ext` function to perform the upscale operation on the created tensor.
- **Output**: The output is a tensor resulting from the upscale operation, which is created and named 'out'.
- **See also**: [`test_upscale_ext`](#test_upscale_ext)  (Data Structure)


---
#### test\_upscale\_ext::build\_graph<!-- {{#callable:test_upscale_ext::build_graph}} -->
The `build_graph` function constructs a computation graph for a tensor upscale operation.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions.
    - Sets the name of tensor `a` to 'a'.
    - Calls [`ggml_upscale_ext`](../ggml/src/ggml.c.driver.md#ggml_upscale_ext) to upscale tensor `a` to the target dimensions specified in `ne_tgt` using the defined scaling mode.
    - Sets the name of the output tensor to 'out'.
    - Returns the output tensor.
- **Output**: Returns a pointer to the output tensor after performing the upscale operation.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_upscale_ext`](../ggml/src/ggml.c.driver.md#ggml_upscale_ext)
- **See also**: [`test_upscale_ext`](#test_upscale_ext)  (Data Structure)



---
### test\_group\_norm<!-- {{#data_structure:test_group_norm}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type used for the tensor.
    - `ne`: An array representing the dimensions of the tensor.
    - `num_groups`: The number of groups for normalization.
    - `eps`: A small value to prevent division by zero during normalization.
- **Description**: The `test_group_norm` struct is a derived class from `test_case` that encapsulates the parameters and behavior necessary to perform group normalization on tensors, including the data type, tensor dimensions, number of groups, and a small epsilon value to ensure numerical stability.
- **Member Functions**:
    - [`test_group_norm::vars`](#test_group_normvars)
    - [`test_group_norm::test_group_norm`](#test_group_normtest_group_norm)
    - [`test_group_norm::build_graph`](#test_group_normbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_group\_norm::vars<!-- {{#callable:test_group_norm::vars}} -->
The `vars` method returns a string representation of the class's member variables.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR4` with the class's member variables: `type`, `ne`, `num_groups`, and `eps`.
    - The `VARS_TO_STR4` macro formats these variables into a string representation.
- **Output**: The method returns a formatted string that includes the values of the member variables.
- **See also**: [`test_group_norm`](#test_group_norm)  (Data Structure)


---
#### test\_group\_norm::test\_group\_norm<!-- {{#callable:test_group_norm::test_group_norm}} -->
The `test_group_norm` constructor initializes a test case for group normalization with specified parameters.
- **Inputs**:
    - `type`: The data type for the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array representing the shape of the tensor, defaulting to {64, 64, 320, 1}.
    - `num_groups`: The number of groups for normalization, defaulting to 32.
    - `eps`: A small value to prevent division by zero, defaulting to 1e-6.
- **Control Flow**:
    - The constructor initializes the member variables `type`, `ne`, `num_groups`, and `eps` with the provided arguments or default values.
    - These member variables are used to configure the group normalization operation in the test case.
- **Output**: The constructor does not return a value but initializes the `test_group_norm` object with the specified parameters.
- **See also**: [`test_group_norm`](#test_group_norm)  (Data Structure)


---
#### test\_group\_norm::build\_graph<!-- {{#callable:test_group_norm::build_graph}} -->
The `build_graph` function constructs a computation graph for group normalization.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which holds the context for tensor operations.
- **Control Flow**:
    - Creates a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions.
    - Sets the name of tensor `a` to 'a' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Calls [`ggml_group_norm`](../ggml/src/ggml.c.driver.md#ggml_group_norm) to perform group normalization on tensor `a`, producing the output tensor `out`.
    - Sets the name of tensor `out` to 'out' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Returns the output tensor `out`.
- **Output**: Returns a pointer to the output tensor resulting from the group normalization operation.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_group_norm`](../ggml/src/ggml.c.driver.md#ggml_group_norm)
- **See also**: [`test_group_norm`](#test_group_norm)  (Data Structure)



---
### test\_l2\_norm<!-- {{#data_structure:test_l2_norm}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
    - `eps`: A small value to prevent division by zero in calculations.
- **Description**: The `test_l2_norm` struct is a derived class from `test_case` that encapsulates the parameters and methods necessary to perform L2 normalization on tensors, including the tensor's data type, its dimensions, and a small epsilon value to ensure numerical stability during calculations.
- **Member Functions**:
    - [`test_l2_norm::vars`](#test_l2_normvars)
    - [`test_l2_norm::test_l2_norm`](#test_l2_normtest_l2_norm)
    - [`test_l2_norm::build_graph`](#test_l2_normbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_l2\_norm::vars<!-- {{#callable:test_l2_norm::vars}} -->
The `vars` method returns a string representation of the `test_l2_norm` class's parameters, specifically the `type` and `ne` attributes.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method directly calls the `VARS_TO_STR2` macro with the `type` and `ne` attributes.
    - The result of the macro call is returned as a string.
- **Output**: The output is a string that represents the `type` and `ne` attributes of the `test_l2_norm` instance, formatted according to the `VARS_TO_STR2` macro.
- **See also**: [`test_l2_norm`](#test_l2_norm)  (Data Structure)


---
#### test\_l2\_norm::test\_l2\_norm<!-- {{#callable:test_l2_norm::test_l2_norm}} -->
The `test_l2_norm` constructor initializes a test case for the L2 normalization operation with specified tensor type, shape, and epsilon value.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array representing the shape of the tensor, defaulting to {64, 64, 320, 1}.
    - `eps`: A small float value used to prevent division by zero, defaulting to 1e-12.
- **Control Flow**:
    - The constructor initializes the member variables `type`, `ne`, and `eps` with the provided arguments.
    - If no arguments are provided, default values are used for each member variable.
- **Output**: The constructor does not return a value but initializes an instance of the `test_l2_norm` class with the specified parameters.
- **See also**: [`test_l2_norm`](#test_l2_norm)  (Data Structure)


---
#### test\_l2\_norm::build\_graph<!-- {{#callable:test_l2_norm::build_graph}} -->
The `build_graph` function constructs a computation graph for a tensor operation involving L2 normalization.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions.
    - Sets the name of tensor `a` to 'a' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Calculates the L2 normalization of tensor `a` using [`ggml_l2_norm`](../ggml/src/ggml.c.driver.md#ggml_l2_norm), storing the result in tensor `out`.
    - Sets the name of tensor `out` to 'out' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Returns the tensor `out`.
- **Output**: Returns a pointer to the output tensor `out`, which contains the result of the L2 normalization operation.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_l2_norm`](../ggml/src/ggml.c.driver.md#ggml_l2_norm)
- **See also**: [`test_l2_norm`](#test_l2_norm)  (Data Structure)



---
### test\_acc<!-- {{#data_structure:test_acc}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensors.
    - `ne_a`: The shape of the first tensor.
    - `ne_b`: The shape of the second tensor.
- **Description**: The `test_acc` structure is a derived class from `test_case` that encapsulates the parameters and methods necessary to perform tests on tensor operations, specifically focusing on the accumulation of two tensors of specified shapes and types.
- **Member Functions**:
    - [`test_acc::vars`](#test_accvars)
    - [`test_acc::test_acc`](#test_acctest_acc)
    - [`test_acc::build_graph`](#test_accbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_acc::vars<!-- {{#callable:test_acc::vars}} -->
The `vars` method returns a string representation of the object's variables, formatted using the `VARS_TO_STR3` macro.
- **Inputs**:
    - `type`: The data type of the tensor, represented by `ggml_type`.
    - `ne_a`: An array of integers representing the dimensions of the first tensor.
    - `ne_b`: An array of integers representing the dimensions of the second tensor.
- **Control Flow**:
    - The method calls the `VARS_TO_STR3` macro with `type`, `ne_a`, and `ne_b` as arguments to generate a formatted string.
    - The formatted string is returned as the output of the method.
- **Output**: A string that represents the variables of the `test_acc` object, formatted according to the specified types and dimensions.
- **See also**: [`test_acc`](#test_acc)  (Data Structure)


---
#### test\_acc::test\_acc<!-- {{#callable:test_acc::test_acc}} -->
The `test_acc` constructor initializes a test case for the GGML operation with specified tensor types and shapes.
- **Inputs**:
    - `type`: The type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne_a`: An array representing the dimensions of the first tensor, defaulting to {256, 17, 1, 1}.
    - `ne_b`: An array representing the dimensions of the second tensor, defaulting to {256, 16, 1, 1}.
- **Control Flow**:
    - The constructor initializes the member variables `type`, `ne_a`, and `ne_b` with the provided arguments or default values.
    - The constructor is part of the `test_acc` struct which inherits from `test_case`.
- **Output**: The constructor does not return a value but initializes the `test_acc` object with the specified parameters.
- **See also**: [`test_acc`](#test_acc)  (Data Structure)


---
#### test\_acc::build\_graph<!-- {{#callable:test_acc::build_graph}} -->
The `build_graph` method constructs a computational graph for a specific operation involving two tensors.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and resources for tensor operations.
- **Control Flow**:
    - Create a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with dimensions specified by `ne_a` and type `type`.
    - Set `a` as a parameter tensor using [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param).
    - Assign the name 'a' to tensor `a` using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Create another tensor `b` similarly to `a`, but using dimensions specified by `ne_b`.
    - Set `b` as a parameter tensor and name it 'b'.
    - Compute the output tensor `out` by calling [`ggml_acc`](../ggml/src/ggml.c.driver.md#ggml_acc), which performs an accumulation operation on tensors `a` and `b`.
    - Set the name 'out' for the output tensor.
    - Return the output tensor `out`.
- **Output**: Returns a pointer to the output tensor `out`, which is the result of the accumulation operation on tensors `a` and `b`.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_acc`](../ggml/src/ggml.c.driver.md#ggml_acc)
- **See also**: [`test_acc`](#test_acc)  (Data Structure)



---
### test\_pad<!-- {{#data_structure:test_pad}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne_a`: An array representing the dimensions of the tensor.
    - `pad_0`: The amount of padding to apply on the first dimension.
    - `pad_1`: The amount of padding to apply on the second dimension.
- **Description**: The `test_pad` struct is designed to represent a test case for padding operations on tensors, inheriting from `test_case`. It contains fields for the tensor's data type, its dimensions, and the padding amounts for two dimensions, allowing for flexible configuration of padding behavior in tensor operations.
- **Member Functions**:
    - [`test_pad::vars`](#test_padvars)
    - [`test_pad::test_pad`](#test_padtest_pad)
    - [`test_pad::build_graph`](#test_padbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_pad::vars<!-- {{#callable:test_pad::vars}} -->
The `vars` method returns a string representation of the object's member variables.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR4` with the member variables `type`, `ne_a`, `pad_0`, and `pad_1`.
    - The `VARS_TO_STR4` macro formats these variables into a string representation.
- **Output**: The method returns a string that represents the values of the member variables in a formatted manner.
- **See also**: [`test_pad`](#test_pad)  (Data Structure)


---
#### test\_pad::test\_pad<!-- {{#callable:test_pad::test_pad}} -->
The `test_pad` constructor initializes a test case for padding operations on tensors.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne_a`: An array representing the dimensions of the input tensor, defaulting to {512, 512, 1, 1}.
    - `pad_0`: The amount of padding to apply on the first dimension, defaulting to 1.
    - `pad_1`: The amount of padding to apply on the second dimension, defaulting to 1.
- **Control Flow**:
    - The constructor initializes the member variables `type`, `ne_a`, `pad_0`, and `pad_1` with the provided arguments.
    - These member variables are used to define the properties of the padding operation that will be tested.
- **Output**: The constructor does not return a value but sets up the state for the `test_pad` instance.
- **See also**: [`test_pad`](#test_pad)  (Data Structure)


---
#### test\_pad::build\_graph<!-- {{#callable:test_pad::build_graph}} -->
The `build_graph` function constructs a computational graph for a tensor operation, creating and padding a tensor.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions from `ne_a`.
    - Sets the name of tensor `a` to 'a' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Pads tensor `a` using [`ggml_pad`](../ggml/src/ggml.c.driver.md#ggml_pad) with specified padding values `pad_0` and `pad_1` to create tensor `out`.
    - Sets the name of tensor `out` to 'out' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Returns the padded tensor `out`.
- **Output**: Returns a pointer to the padded tensor `out`.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_pad`](../ggml/src/ggml.c.driver.md#ggml_pad)
- **See also**: [`test_pad`](#test_pad)  (Data Structure)



---
### test\_pad\_reflect\_1d<!-- {{#data_structure:test_pad_reflect_1d}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne_a`: An array representing the dimensions of the tensor.
    - `pad_0`: The amount of padding to apply on one side.
    - `pad_1`: The amount of padding to apply on the other side.
- **Description**: The `test_pad_reflect_1d` struct is designed to test the reflection padding operation for 1D tensors, inheriting from `test_case`. It contains fields for the tensor's data type, its dimensions, and the padding amounts to be applied on both sides of the tensor.
- **Member Functions**:
    - [`test_pad_reflect_1d::vars`](#test_pad_reflect_1dvars)
    - [`test_pad_reflect_1d::test_pad_reflect_1d`](#test_pad_reflect_1dtest_pad_reflect_1d)
    - [`test_pad_reflect_1d::build_graph`](#test_pad_reflect_1dbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_pad\_reflect\_1d::vars<!-- {{#callable:test_pad_reflect_1d::vars}} -->
The `vars` method returns a string representation of the object's member variables formatted for debugging.
- **Inputs**:
    - `none`: This method does not take any input parameters.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR4` with the member variables `type`, `ne_a`, `pad_0`, and `pad_1`.
    - The `VARS_TO_STR4` macro formats these variables into a string representation.
- **Output**: The method returns a formatted string that represents the values of the member variables of the class.
- **See also**: [`test_pad_reflect_1d`](#test_pad_reflect_1d)  (Data Structure)


---
#### test\_pad\_reflect\_1d::test\_pad\_reflect\_1d<!-- {{#callable:test_pad_reflect_1d::test_pad_reflect_1d}} -->
The `test_pad_reflect_1d` constructor initializes a test case for a 1D reflect padding operation with specified tensor type, dimensions, and padding values.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne_a`: An array of four integers representing the dimensions of the input tensor, defaulting to {512, 34, 2, 1}.
    - `pad_0`: An integer specifying the amount of padding to add to the start of the tensor, defaulting to 10.
    - `pad_1`: An integer specifying the amount of padding to add to the end of the tensor, defaulting to 9.
- **Control Flow**:
    - The constructor initializes the member variables `type`, `ne_a`, `pad_0`, and `pad_1` with the provided arguments or default values.
    - The constructor is part of the `test_pad_reflect_1d` struct, which inherits from `test_case`.
- **Output**: The constructor does not return a value but initializes an instance of `test_pad_reflect_1d` with the specified parameters.
- **See also**: [`test_pad_reflect_1d`](#test_pad_reflect_1d)  (Data Structure)


---
#### test\_pad\_reflect\_1d::build\_graph<!-- {{#callable:test_pad_reflect_1d::build_graph}} -->
The `build_graph` function constructs a computational graph for a tensor operation involving padding.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions from `ne_a`.
    - Sets the name of tensor `a` to 'a' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Calls [`ggml_pad_reflect_1d`](../ggml/src/ggml.c.driver.md#ggml_pad_reflect_1d) to create a new tensor `out` that reflects the padding of tensor `a` with specified padding values `pad_0` and `pad_1`.
    - Sets the name of tensor `out` to 'out' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Returns the tensor `out`.
- **Output**: Returns a pointer to the output tensor `out`, which is the padded version of tensor `a`.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_pad_reflect_1d`](../ggml/src/ggml.c.driver.md#ggml_pad_reflect_1d)
- **See also**: [`test_pad_reflect_1d`](#test_pad_reflect_1d)  (Data Structure)



---
### test\_arange<!-- {{#data_structure:test_arange}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `start`: The starting value of the range.
    - `stop`: The stopping value of the range.
    - `step`: The increment between each value in the range.
- **Description**: The `test_arange` struct is a derived class from `test_case` that defines a test case for generating a range of values from `start` to `stop` with a specified `step`. It includes members to specify the data type of the tensor and the parameters for the range generation.
- **Member Functions**:
    - [`test_arange::vars`](#test_arangevars)
    - [`test_arange::test_arange`](#test_arangetest_arange)
    - [`test_arange::build_graph`](#test_arangebuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_arange::vars<!-- {{#callable:test_arange::vars}} -->
The `vars` method returns a string representation of the object's parameters.
- **Inputs**:
    - `type`: The data type of the tensor.
    - `start`: The starting value of the range.
    - `stop`: The ending value of the range.
    - `step`: The increment between each value in the range.
- **Control Flow**:
    - The method calls the `VARS_TO_STR4` macro with the parameters `type`, `start`, `stop`, and `step`.
    - The result of the macro is returned as a string.
- **Output**: A string that represents the parameters of the object in a formatted manner.
- **See also**: [`test_arange`](#test_arange)  (Data Structure)


---
#### test\_arange::test\_arange<!-- {{#callable:test_arange::test_arange}} -->
The `test_arange` constructor initializes a test case for generating a tensor with a range of values.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `start`: The starting value of the range, defaulting to 0.f.
    - `stop`: The stopping value of the range, defaulting to 10.f.
    - `step`: The step size between values in the range, defaulting to 1.f.
- **Control Flow**:
    - The constructor initializes the member variables `type`, `start`, `stop`, and `step` with the provided arguments or default values.
    - These member variables are used to define the parameters for generating a tensor with a range of values.
- **Output**: The constructor does not return a value but initializes the `test_arange` object with the specified parameters.
- **See also**: [`test_arange`](#test_arange)  (Data Structure)


---
#### test\_arange::build\_graph<!-- {{#callable:test_arange::build_graph}} -->
The `build_graph` function constructs a computation graph for generating a tensor using the [`ggml_arange`](../ggml/src/ggml.c.driver.md#ggml_arange) function.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations.
- **Control Flow**:
    - Calls [`ggml_arange`](../ggml/src/ggml.c.driver.md#ggml_arange) to create a tensor with values ranging from `start` to `stop` with a specified `step`.
    - Sets the name of the output tensor to 'out' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Returns the created tensor.
- **Output**: Returns a pointer to a `ggml_tensor` that contains a sequence of values generated by the [`ggml_arange`](../ggml/src/ggml.c.driver.md#ggml_arange) function.
- **Functions called**:
    - [`ggml_arange`](../ggml/src/ggml.c.driver.md#ggml_arange)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
- **See also**: [`test_arange`](#test_arange)  (Data Structure)



---
### test\_timestep\_embedding<!-- {{#data_structure:test_timestep_embedding}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne_a`: An array representing the dimensions of the tensor.
    - `dim`: The dimensionality of the embedding.
    - `max_period`: The maximum period for the timestep embedding.
- **Description**: The `test_timestep_embedding` struct is designed to represent a test case for a timestep embedding operation, inheriting from `test_case`. It contains fields for the tensor's data type, its dimensions, the embedding dimensionality, and the maximum period, allowing for the construction of a tensor and the evaluation of its properties in the context of machine learning operations.
- **Member Functions**:
    - [`test_timestep_embedding::vars`](#test_timestep_embeddingvars)
    - [`test_timestep_embedding::test_timestep_embedding`](#test_timestep_embeddingtest_timestep_embedding)
    - [`test_timestep_embedding::build_graph`](#test_timestep_embeddingbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_timestep\_embedding::vars<!-- {{#callable:test_timestep_embedding::vars}} -->
The `vars` method returns a string representation of the member variables of the `test_timestep_embedding` class.
- **Inputs**:
    - `none`: This method does not take any input parameters.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR4` with the member variables `type`, `ne_a`, `dim`, and `max_period`.
    - The `VARS_TO_STR4` macro constructs a formatted string representation of these variables.
- **Output**: The method returns a string that represents the values of the member variables in a formatted manner.
- **See also**: [`test_timestep_embedding`](#test_timestep_embedding)  (Data Structure)


---
#### test\_timestep\_embedding::test\_timestep\_embedding<!-- {{#callable:test_timestep_embedding::test_timestep_embedding}} -->
The `test_timestep_embedding` constructor initializes a test case for timestep embedding with specified parameters.
- **Inputs**:
    - `type`: The data type for the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne_a`: An array representing the dimensions of the tensor, defaulting to {2, 1, 1, 1}.
    - `dim`: The dimensionality of the embedding, defaulting to 320.
    - `max_period`: The maximum period for the embedding, defaulting to 10000.
- **Control Flow**:
    - The constructor initializes member variables `type`, `ne_a`, `dim`, and `max_period` with the provided arguments or default values.
    - These member variables are used to configure the behavior of the timestep embedding test case.
- **Output**: The constructor does not return a value but initializes the state of the `test_timestep_embedding` object.
- **See also**: [`test_timestep_embedding`](#test_timestep_embedding)  (Data Structure)


---
#### test\_timestep\_embedding::build\_graph<!-- {{#callable:test_timestep_embedding::build_graph}} -->
The `build_graph` function constructs a computation graph for a timestep embedding operation.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and state for tensor operations.
- **Control Flow**:
    - Creates a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions from `ne_a`.
    - Sets the name of tensor `a` to 'a' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Calls [`ggml_timestep_embedding`](../ggml/src/ggml.c.driver.md#ggml_timestep_embedding) to compute the timestep embedding using tensor `a`, `dim`, and `max_period`, storing the result in `out`.
    - Sets the name of tensor `out` to 'out' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Returns the output tensor `out`.
- **Output**: Returns a pointer to the output tensor `out`, which contains the result of the timestep embedding operation.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_timestep_embedding`](../ggml/src/ggml.c.driver.md#ggml_timestep_embedding)
- **See also**: [`test_timestep_embedding`](#test_timestep_embedding)  (Data Structure)



---
### test\_leaky\_relu<!-- {{#data_structure:test_leaky_relu}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne_a`: An array representing the dimensions of the tensor.
    - `negative_slope`: The slope for negative input values in the leaky ReLU function.
- **Description**: The `test_leaky_relu` struct is a test case for evaluating the leaky ReLU activation function in a neural network context. It inherits from `test_case` and contains parameters for the tensor type, its dimensions, and the negative slope used in the leaky ReLU function. The struct is designed to facilitate the construction of a computational graph for testing the behavior of the leaky ReLU operation.
- **Member Functions**:
    - [`test_leaky_relu::vars`](#test_leaky_reluvars)
    - [`test_leaky_relu::test_leaky_relu`](#test_leaky_relutest_leaky_relu)
    - [`test_leaky_relu::build_graph`](#test_leaky_relubuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_leaky\_relu::vars<!-- {{#callable:test_leaky_relu::vars}} -->
The `vars` method returns a string representation of the class's member variables.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR3` with the class's member variables: `type`, `ne_a`, and `negative_slope`.
    - The `VARS_TO_STR3` macro formats these variables into a string representation.
- **Output**: The method returns a string that represents the values of the member variables in a formatted manner.
- **See also**: [`test_leaky_relu`](#test_leaky_relu)  (Data Structure)


---
#### test\_leaky\_relu::test\_leaky\_relu<!-- {{#callable:test_leaky_relu::test_leaky_relu}} -->
The `test_leaky_relu` constructor initializes a test case for the Leaky ReLU activation function with specified tensor type, dimensions, and negative slope.
- **Inputs**:
    - `type`: The data type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne_a`: An array representing the dimensions of the input tensor, defaulting to {10, 5, 4, 3}.
    - `negative_slope`: The slope for negative input values, defaulting to 0.1.
- **Control Flow**:
    - The constructor initializes the member variables `type`, `ne_a`, and `negative_slope` with the provided arguments or default values.
    - The `vars` method is overridden to return a string representation of the test case variables for logging purposes.
- **Output**: The constructor does not return a value but initializes an instance of the `test_leaky_relu` class, which can be used to run tests on the Leaky ReLU operation.
- **See also**: [`test_leaky_relu`](#test_leaky_relu)  (Data Structure)


---
#### test\_leaky\_relu::build\_graph<!-- {{#callable:test_leaky_relu::build_graph}} -->
The `build_graph` function constructs a computational graph for a leaky ReLU operation.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages the memory and state for tensor operations.
- **Control Flow**:
    - Creates a new tensor `a` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified type and dimensions based on `ne_a`.
    - Sets the name of tensor `a` to 'a' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Applies the leaky ReLU operation on tensor `a` with the specified `negative_slope` and stores the result in tensor `out`.
    - Sets the name of tensor `out` to 'out' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Returns the output tensor `out`.
- **Output**: Returns a pointer to the output tensor `out`, which contains the result of applying the leaky ReLU activation function to tensor `a`.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_leaky_relu`](../ggml/src/ggml.c.driver.md#ggml_leaky_relu)
- **See also**: [`test_leaky_relu`](#test_leaky_relu)  (Data Structure)



---
### test\_flash\_attn\_ext<!-- {{#data_structure:test_flash_attn_ext}} -->
- **Type**: `struct`
- **Members**:
    - `hsk`: The head size for keys.
    - `hsv`: The head size for values.
    - `nh`: The number of attention heads.
    - `nr`: The repeat count for grouped-query attention.
    - `kv`: The size of key-value pairs.
    - `nb`: The batch size.
    - `mask`: Indicates whether to use a mask.
    - `max_bias`: The maximum bias for ALiBi.
    - `logit_softcap`: The soft cap for logits.
    - `prec`: The precision type for computations.
    - `type_KV`: The type of key-value pairs.
    - `permute`: An array defining the permutation of dimensions.
- **Description**: The `test_flash_attn_ext` struct is designed to facilitate testing of the flash attention mechanism in neural networks, particularly focusing on the attention mechanism's parameters such as head sizes, number of heads, and batch sizes. It includes various attributes to configure the attention mechanism, including the use of masks, maximum bias, and logit soft caps, as well as types for key-value pairs and a permutation array for tensor dimensions.
- **Member Functions**:
    - [`test_flash_attn_ext::vars`](#test_flash_attn_extvars)
    - [`test_flash_attn_ext::max_nmse_err`](#test_flash_attn_extmax_nmse_err)
    - [`test_flash_attn_ext::op_flops`](#test_flash_attn_extop_flops)
    - [`test_flash_attn_ext::test_flash_attn_ext`](#test_flash_attn_exttest_flash_attn_ext)
    - [`test_flash_attn_ext::build_graph`](#test_flash_attn_extbuild_graph)
    - [`test_flash_attn_ext::grad_precise`](#test_flash_attn_extgrad_precise)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_flash\_attn\_ext::vars<!-- {{#callable:test_flash_attn_ext::vars}} -->
The `vars` method returns a string representation of the internal state variables of the `test_flash_attn_ext` class.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR12` with the class's member variables as arguments.
    - The `VARS_TO_STR12` macro formats these member variables into a string representation.
- **Output**: The method outputs a string that contains the formatted values of the member variables of the `test_flash_attn_ext` class.
- **See also**: [`test_flash_attn_ext`](#test_flash_attn_ext)  (Data Structure)


---
#### test\_flash\_attn\_ext::max\_nmse\_err<!-- {{#callable:test_flash_attn_ext::max_nmse_err}} -->
The `max_nmse_err` function returns a constant maximum normalized mean squared error value.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a predefined constant value without any conditional logic or loops.
- **Output**: The output is a double precision floating-point number representing the maximum normalized mean squared error, specifically 5e-4.
- **See also**: [`test_flash_attn_ext`](#test_flash_attn_ext)  (Data Structure)


---
#### test\_flash\_attn\_ext::op\_flops<!-- {{#callable:test_flash_attn_ext::op_flops}} -->
The `op_flops` function calculates the number of floating-point operations (FLOPs) required for matrix multiplication in a specific attention mechanism.
- **Inputs**:
    - `t`: A pointer to a `ggml_tensor` structure, which is unused in this function.
- **Control Flow**:
    - The function begins by marking the input tensor `t` as unused to avoid compiler warnings.
    - It calculates the FLOPs based on the formula: 2 * nh * nr * nb * (hsk + hsv) * kv, where nh is the number of heads, nr is the repeat count, nb is the batch size, hsk is the key head size, hsv is the value head size, and kv is the key-value size.
    - The calculated FLOPs are returned as a 64-bit unsigned integer.
- **Output**: Returns a `uint64_t` value representing the total number of floating-point operations required for the matrix multiplications involved in the attention mechanism.
- **See also**: [`test_flash_attn_ext`](#test_flash_attn_ext)  (Data Structure)


---
#### test\_flash\_attn\_ext::test\_flash\_attn\_ext<!-- {{#callable:test_flash_attn_ext::test_flash_attn_ext}} -->
The `test_flash_attn_ext` function initializes a test case for the flash attention mechanism with various configurable parameters.
- **Inputs**:
    - `hsk`: The size of the key head.
    - `hsv`: The size of the value head.
    - `nh`: The number of attention heads.
    - `nr`: The number of repetitions in the query, used for grouped-query attention.
    - `kv`: The size of the key-value pairs.
    - `nb`: The batch size.
    - `mask`: A boolean indicating whether to use a mask.
    - `max_bias`: The maximum bias value for the attention mechanism.
    - `logit_softcap`: The soft cap for logits.
    - `prec`: The precision type for the computation.
    - `type_KV`: The data type for key-value pairs.
    - `permute`: An array defining the permutation of dimensions.
- **Control Flow**:
    - The function begins by padding the `hsk` and `hsv` values based on the block size of the `type_KV`.
    - It defines a lambda function `create_permuted` to create and permute tensors based on the specified dimensions.
    - It creates the query tensor `q` using the padded `hsk`, batch size `nb`, and number of heads multiplied by repetitions `nh*nr`.
    - It creates the key tensor `k` and value tensor `v` using the respective sizes and types.
    - If masking is enabled, it creates a mask tensor `m`.
    - Finally, it calls the `ggml_flash_attn_ext` function with the created tensors and returns the output tensor.
- **Output**: The function returns a tensor representing the output of the flash attention mechanism, which is computed based on the input query, key, and value tensors.
- **See also**: [`test_flash_attn_ext`](#test_flash_attn_ext)  (Data Structure)


---
#### test\_flash\_attn\_ext::build\_graph<!-- {{#callable:test_flash_attn_ext::build_graph}} -->
The `build_graph` method constructs a computational graph for a flash attention mechanism using padded input tensors.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` which manages memory and tensor operations.
- **Control Flow**:
    - Calculate padded sizes for `hsk` and `hsv` using `GGML_PAD` and [`ggml_blck_size`](../ggml/src/ggml.c.driver.md#ggml_blck_size).
    - Define a lambda function `create_permuted` to create and optionally permute 4D tensors.
    - Create query tensor `q` using `create_permuted` with type `GGML_TYPE_F32`.
    - Create key tensor `k` using `create_permuted` with type `type_KV`.
    - Create value tensor `v` using `create_permuted` with type `type_KV`.
    - If `mask` is true, create a mask tensor `m`.
    - Call [`ggml_flash_attn_ext`](../ggml/src/ggml.c.driver.md#ggml_flash_attn_ext) with the created tensors and additional parameters to compute the output tensor.
    - Set the precision of the output tensor and return it.
- **Output**: Returns a pointer to the output tensor resulting from the flash attention computation.
- **Functions called**:
    - [`ggml_blck_size`](../ggml/src/ggml.c.driver.md#ggml_blck_size)
    - [`test_case::ggml_new_tensor_4d`](#test_caseggml_new_tensor_4d)
    - [`ggml_permute`](../ggml/src/ggml.c.driver.md#ggml_permute)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_flash_attn_ext`](../ggml/src/ggml.c.driver.md#ggml_flash_attn_ext)
    - [`ggml_flash_attn_ext_set_prec`](../ggml/src/ggml.c.driver.md#ggml_flash_attn_ext_set_prec)
- **See also**: [`test_flash_attn_ext`](#test_flash_attn_ext)  (Data Structure)


---
#### test\_flash\_attn\_ext::grad\_precise<!-- {{#callable:test_flash_attn_ext::grad_precise}} -->
The `grad_precise` function always returns true, indicating that gradient estimation should be performed with high precision.
- **Inputs**: None
- **Control Flow**:
    - The function contains no control flow statements; it directly returns a boolean value.
- **Output**: The output is a boolean value, specifically 'true'.
- **See also**: [`test_flash_attn_ext`](#test_flash_attn_ext)  (Data Structure)



---
### test\_cross\_entropy\_loss<!-- {{#data_structure:test_cross_entropy_loss}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
- **Description**: The `test_cross_entropy_loss` struct is a derived class from `test_case` that encapsulates the parameters and methods necessary to evaluate the cross-entropy loss function in a neural network context, specifically designed to handle tensor operations and gradient calculations.
- **Member Functions**:
    - [`test_cross_entropy_loss::vars`](#test_cross_entropy_lossvars)
    - [`test_cross_entropy_loss::test_cross_entropy_loss`](#test_cross_entropy_losstest_cross_entropy_loss)
    - [`test_cross_entropy_loss::build_graph`](#test_cross_entropy_lossbuild_graph)
    - [`test_cross_entropy_loss::initialize_tensors`](#test_cross_entropy_lossinitialize_tensors)
    - [`test_cross_entropy_loss::grad_eps`](#test_cross_entropy_lossgrad_eps)
    - [`test_cross_entropy_loss::grad_precise`](#test_cross_entropy_lossgrad_precise)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_cross\_entropy\_loss::vars<!-- {{#callable:test_cross_entropy_loss::vars}} -->
The `vars` method returns a string representation of the `type` and `ne` attributes of the `test_cross_entropy_loss` class.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR2` with `type` and `ne` as arguments.
    - The `VARS_TO_STR2` macro formats these attributes into a string representation.
- **Output**: The method returns a string that represents the `type` and `ne` attributes of the instance.
- **See also**: [`test_cross_entropy_loss`](#test_cross_entropy_loss)  (Data Structure)


---
#### test\_cross\_entropy\_loss::test\_cross\_entropy\_loss<!-- {{#callable:test_cross_entropy_loss::test_cross_entropy_loss}} -->
The `test_cross_entropy_loss` class defines a test case for evaluating the cross-entropy loss function in a neural network context.
- **Inputs**:
    - `type`: The data type of the tensors used in the test, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array representing the dimensions of the input tensors, defaulting to {10, 5, 4, 3}.
- **Control Flow**:
    - The constructor initializes the `type` and `ne` member variables with the provided arguments.
    - The `build_graph` method creates the computation graph for the cross-entropy loss calculation.
    - It creates tensors for logits and labels, applies softmax to the labels, and computes the cross-entropy loss.
- **Output**: The output is a tensor representing the computed cross-entropy loss.
- **See also**: [`test_cross_entropy_loss`](#test_cross_entropy_loss)  (Data Structure)


---
#### test\_cross\_entropy\_loss::build\_graph<!-- {{#callable:test_cross_entropy_loss::build_graph}} -->
The `build_graph` function constructs a computational graph for calculating the cross-entropy loss between logits and labels.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages the memory and state for the graph.
- **Control Flow**:
    - Creates a new tensor `logits` using [`ggml_new_tensor`](#test_caseggml_new_tensor) with the specified context, type, and dimensions.
    - Sets the `logits` tensor as a parameter using [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param).
    - Names the `logits` tensor as 'logits' using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Creates another tensor `labels` in a similar manner, assuming it to be constant (no gradients).
    - Normalizes the `labels` tensor using [`ggml_soft_max`](../ggml/src/ggml.c.driver.md#ggml_soft_max) to ensure the values sum to 1.
    - Names the normalized `labels` tensor as 'labels_normalized'.
    - Calculates the cross-entropy loss using [`ggml_cross_entropy_loss`](../ggml/src/ggml.c.driver.md#ggml_cross_entropy_loss) with `logits` and `labels` as inputs.
    - Names the output tensor as 'out'.
    - Returns the output tensor containing the cross-entropy loss.
- **Output**: Returns a pointer to the output tensor containing the computed cross-entropy loss.
- **Functions called**:
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_soft_max`](../ggml/src/ggml.c.driver.md#ggml_soft_max)
    - [`ggml_cross_entropy_loss`](../ggml/src/ggml.c.driver.md#ggml_cross_entropy_loss)
- **See also**: [`test_cross_entropy_loss`](#test_cross_entropy_loss)  (Data Structure)


---
#### test\_cross\_entropy\_loss::initialize\_tensors<!-- {{#callable:test_cross_entropy_loss::initialize_tensors}} -->
The `initialize_tensors` function initializes all tensors in the given `ggml_context` with uniform random values between -100.0 and 100.0.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that holds the state of the tensor context.
- **Control Flow**:
    - The function iterates over all tensors in the context using [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor) and [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor).
    - For each tensor, it calls [`init_tensor_uniform`](#init_tensor_uniform) to initialize the tensor with random values in the range of -100.0 to 100.0.
- **Output**: The function does not return a value; it modifies the tensors in place within the provided `ggml_context`.
- **Functions called**:
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`init_tensor_uniform`](#init_tensor_uniform)
- **See also**: [`test_cross_entropy_loss`](#test_cross_entropy_loss)  (Data Structure)


---
#### test\_cross\_entropy\_loss::grad\_eps<!-- {{#callable:test_cross_entropy_loss::grad_eps}} -->
The `grad_eps` function returns a constant float value of 1.0.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a constant value without any conditional statements or loops.
- **Output**: The output is a float value of 1.0.
- **See also**: [`test_cross_entropy_loss`](#test_cross_entropy_loss)  (Data Structure)


---
#### test\_cross\_entropy\_loss::grad\_precise<!-- {{#callable:test_cross_entropy_loss::grad_precise}} -->
The `grad_precise` function returns a boolean value indicating whether to use a precise gradient estimation method.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the boolean value `true` without any conditions or computations.
- **Output**: The output is a boolean value, specifically `true`.
- **See also**: [`test_cross_entropy_loss`](#test_cross_entropy_loss)  (Data Structure)



---
### test\_cross\_entropy\_loss\_back<!-- {{#data_structure:test_cross_entropy_loss_back}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
- **Description**: The `test_cross_entropy_loss_back` struct is a derived class from `test_case` that is designed to test the backward pass of the cross-entropy loss operation in a neural network context. It contains a tensor type and its dimensions, and it overrides the `build_graph` method to create a computational graph for calculating the gradients of the loss with respect to the logits and labels.
- **Member Functions**:
    - [`test_cross_entropy_loss_back::vars`](#test_cross_entropy_loss_backvars)
    - [`test_cross_entropy_loss_back::test_cross_entropy_loss_back`](#test_cross_entropy_loss_backtest_cross_entropy_loss_back)
    - [`test_cross_entropy_loss_back::build_graph`](#test_cross_entropy_loss_backbuild_graph)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_cross\_entropy\_loss\_back::vars<!-- {{#callable:test_cross_entropy_loss_back::vars}} -->
The `vars` method returns a string representation of the class's member variables.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR2` with the class's member variables `type` and `ne`.
    - The result of the macro call is returned as a string.
- **Output**: The output is a string that represents the values of the `type` and `ne` member variables in a formatted manner.
- **See also**: [`test_cross_entropy_loss_back`](#test_cross_entropy_loss_back)  (Data Structure)


---
#### test\_cross\_entropy\_loss\_back::test\_cross\_entropy\_loss\_back<!-- {{#callable:test_cross_entropy_loss_back::test_cross_entropy_loss_back}} -->
The `test_cross_entropy_loss_back` constructor initializes a test case for the backward pass of the cross-entropy loss operation.
- **Inputs**:
    - `type`: The data type for the tensors, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array representing the dimensions of the tensors, defaulting to {10, 5, 4, 3}.
- **Control Flow**:
    - The constructor initializes the member variables `type` and `ne` with the provided arguments.
    - It uses an initializer list to set the values of the member variables.
- **Output**: The constructor does not return a value but initializes an instance of the `test_cross_entropy_loss_back` class.
- **See also**: [`test_cross_entropy_loss_back`](#test_cross_entropy_loss_back)  (Data Structure)


---
#### test\_cross\_entropy\_loss\_back::build\_graph<!-- {{#callable:test_cross_entropy_loss_back::build_graph}} -->
The `build_graph` function constructs a computational graph for a cross-entropy loss operation.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a new 1D tensor `grad` initialized to zero, which will hold the gradients.
    - Creates a new tensor `logits` initialized with the specified type and dimensions, representing the model's output logits.
    - Creates a new tensor `labels` initialized with the specified type and dimensions, representing the true labels for the data.
    - Applies the softmax function to `labels` to ensure they sum to 1, normalizing the labels.
    - Calculates the cross-entropy loss using the [`ggml_cross_entropy_loss_back`](../ggml/src/ggml.c.driver.md#ggml_cross_entropy_loss_back) function, which takes `grad`, `logits`, and the normalized `labels` as inputs.
    - Returns the output tensor containing the computed loss.
- **Output**: Returns a pointer to a `ggml_tensor` that contains the result of the cross-entropy loss calculation.
- **Functions called**:
    - [`test_case::ggml_new_tensor_1d`](#test_caseggml_new_tensor_1d)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`test_case::ggml_new_tensor`](#test_caseggml_new_tensor)
    - [`ggml_soft_max`](../ggml/src/ggml.c.driver.md#ggml_soft_max)
    - [`ggml_cross_entropy_loss_back`](../ggml/src/ggml.c.driver.md#ggml_cross_entropy_loss_back)
- **See also**: [`test_cross_entropy_loss_back`](#test_cross_entropy_loss_back)  (Data Structure)



---
### test\_opt\_step\_adamw<!-- {{#data_structure:test_opt_step_adamw}} -->
- **Type**: `struct`
- **Members**:
    - `type`: The data type of the tensor.
    - `ne`: An array representing the dimensions of the tensor.
- **Description**: The `test_opt_step_adamw` struct is a derived class from `test_case` that encapsulates the parameters and methods necessary to perform an optimization step using the AdamW algorithm, specifically designed for testing tensor operations in the GGML framework.
- **Member Functions**:
    - [`test_opt_step_adamw::vars`](#test_opt_step_adamwvars)
    - [`test_opt_step_adamw::test_opt_step_adamw`](#test_opt_step_adamwtest_opt_step_adamw)
    - [`test_opt_step_adamw::build_graph`](#test_opt_step_adamwbuild_graph)
    - [`test_opt_step_adamw::initialize_tensors`](#test_opt_step_adamwinitialize_tensors)
    - [`test_opt_step_adamw::grad_precise`](#test_opt_step_adamwgrad_precise)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_opt\_step\_adamw::vars<!-- {{#callable:test_opt_step_adamw::vars}} -->
The `vars` method returns a string representation of the `test_opt_step_adamw` class's parameters.
- **Inputs**:
    - `none`: This method does not take any input arguments.
- **Control Flow**:
    - The method calls the macro `VARS_TO_STR2` with `type` and `ne` as arguments.
    - The `VARS_TO_STR2` macro formats these parameters into a string representation.
- **Output**: The method outputs a string that represents the `type` and `ne` attributes of the class.
- **See also**: [`test_opt_step_adamw`](#test_opt_step_adamw)  (Data Structure)


---
#### test\_opt\_step\_adamw::test\_opt\_step\_adamw<!-- {{#callable:test_opt_step_adamw::test_opt_step_adamw}} -->
The `test_opt_step_adamw` constructor initializes a test case for the AdamW optimization step with specified tensor types and dimensions.
- **Inputs**:
    - `type`: The type of the tensor, defaulting to `GGML_TYPE_F32`.
    - `ne`: An array representing the dimensions of the tensor, defaulting to {10, 5, 4, 3}.
- **Control Flow**:
    - The constructor initializes the member variables `type` and `ne` with the provided arguments.
    - If no arguments are provided, it uses the default values for `type` and `ne`.
- **Output**: The constructor does not return a value but initializes an instance of the `test_opt_step_adamw` class with the specified tensor type and dimensions.
- **See also**: [`test_opt_step_adamw`](#test_opt_step_adamw)  (Data Structure)


---
#### test\_opt\_step\_adamw::build\_graph<!-- {{#callable:test_opt_step_adamw::build_graph}} -->
The `build_graph` function constructs a computational graph for a neural network using various tensor operations.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory and resources for tensor operations.
- **Control Flow**:
    - Creates a 4D tensor `a` with specified dimensions and type, and sets it as a parameter tensor.
    - Creates additional 4D tensors for gradients (`grad`, `grad_m`, `grad_v`) and a 1D tensor for AdamW parameters.
    - Calls the [`ggml_opt_step_adamw`](../ggml/src/ggml.c.driver.md#ggml_opt_step_adamw) function to perform an optimization step using the created tensors.
    - Returns the output tensor from the optimization step.
- **Output**: Returns a pointer to the output tensor resulting from the optimization step.
- **Functions called**:
    - [`test_case::ggml_new_tensor_4d`](#test_caseggml_new_tensor_4d)
    - [`ggml_set_param`](../ggml/src/ggml.c.driver.md#ggml_set_param)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`test_case::ggml_new_tensor_1d`](#test_caseggml_new_tensor_1d)
    - [`ggml_opt_step_adamw`](../ggml/src/ggml.c.driver.md#ggml_opt_step_adamw)
- **See also**: [`test_opt_step_adamw`](#test_opt_step_adamw)  (Data Structure)


---
#### test\_opt\_step\_adamw::initialize\_tensors<!-- {{#callable:test_opt_step_adamw::initialize_tensors}} -->
The `initialize_tensors` function initializes all tensors in a given `ggml_context` with uniform random values between 0.0 and 1.0.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that contains the tensors to be initialized.
- **Control Flow**:
    - The function starts a loop that retrieves the first tensor from the context using [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor).
    - Inside the loop, it checks if the current tensor is not NULL.
    - For each tensor, it calls [`init_tensor_uniform`](#init_tensor_uniform) to initialize it with values between 0.0 and 1.0.
    - The loop continues until all tensors in the context have been processed.
- **Output**: The function does not return a value; it modifies the tensors in place by initializing them with random values.
- **Functions called**:
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`init_tensor_uniform`](#init_tensor_uniform)
- **See also**: [`test_opt_step_adamw`](#test_opt_step_adamw)  (Data Structure)


---
#### test\_opt\_step\_adamw::grad\_precise<!-- {{#callable:test_opt_step_adamw::grad_precise}} -->
The `grad_precise` function indicates whether to use a precise gradient estimation method.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a boolean value without any conditional statements or loops.
- **Output**: The function outputs a boolean value, specifically `true`.
- **See also**: [`test_opt_step_adamw`](#test_opt_step_adamw)  (Data Structure)



---
### llm\_norm\_type<!-- {{#data_structure:llm_norm_type}} -->
- **Type**: `enum`
- **Members**:
    - `LLM_NORM`: Represents the LLM normalization type.
    - `LLM_NORM_RMS`: Represents the RMS normalization type for LLM.
- **Description**: The `llm_norm_type` enum defines two types of normalization used in LLM (Large Language Model) operations: `LLM_NORM` for standard normalization and `LLM_NORM_RMS` for root mean square normalization, allowing for flexibility in how model inputs are normalized during processing.


---
### llama\_hparams<!-- {{#data_structure:llama_hparams}} -->
- **Type**: `struct`
- **Members**:
    - `n_vocab`: The size of the vocabulary.
    - `n_embd`: The dimensionality of the embeddings.
    - `n_head`: The number of attention heads.
    - `n_head_kv`: The number of key-value attention heads.
    - `n_rot`: The number of rotational embeddings.
    - `n_embd_head`: The dimension of values (d_v) for each head.
    - `n_ff`: The dimensionality of the feed-forward layer.
    - `f_norm_eps`: Epsilon value for normalization.
    - `f_norm_rms_eps`: Epsilon value for RMS normalization.
    - `n_tokens`: The number of tokens in the input.
- **Description**: The `llama_hparams` struct defines the hyperparameters for a LLaMA model, including dimensions for embeddings, attention heads, and normalization parameters. It contains various attributes that specify the model's architecture, such as the vocabulary size, embedding dimensions, number of heads, and other configuration settings necessary for initializing and running the model.
- **Member Functions**:
    - [`llama_hparams::set_swa_pattern`](../src/llama-hparams.cpp.driver.md#llama_hparamsset_swa_pattern)
    - [`llama_hparams::is_swa_any`](../src/llama-hparams.cpp.driver.md#llama_hparamsis_swa_any)
    - [`llama_hparams::n_head`](../src/llama-hparams.cpp.driver.md#llama_hparamsn_head)
    - [`llama_hparams::n_head_kv`](../src/llama-hparams.cpp.driver.md#llama_hparamsn_head_kv)
    - [`llama_hparams::n_ff`](../src/llama-hparams.cpp.driver.md#llama_hparamsn_ff)
    - [`llama_hparams::n_gqa`](../src/llama-hparams.cpp.driver.md#llama_hparamsn_gqa)
    - [`llama_hparams::n_embd_k_gqa`](../src/llama-hparams.cpp.driver.md#llama_hparamsn_embd_k_gqa)
    - [`llama_hparams::n_embd_v_gqa`](../src/llama-hparams.cpp.driver.md#llama_hparamsn_embd_v_gqa)
    - [`llama_hparams::n_embd_k_s`](../src/llama-hparams.cpp.driver.md#llama_hparamsn_embd_k_s)
    - [`llama_hparams::n_embd_v_s`](../src/llama-hparams.cpp.driver.md#llama_hparamsn_embd_v_s)
    - [`llama_hparams::is_swa`](../src/llama-hparams.cpp.driver.md#llama_hparamsis_swa)
    - [`llama_hparams::n_embd_gqa`](#llama_hparamsn_embd_gqa)

**Methods**

---
#### llama\_hparams::n\_embd\_gqa<!-- {{#callable:llama_hparams::n_embd_gqa}} -->
The `n_embd_gqa` function calculates the dimension of key embeddings across all key-value heads.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the product of `n_embd_head` and `n_head_kv`.
- **Output**: The output is a `uint32_t` representing the total dimension of key embeddings.
- **See also**: [`llama_hparams`](#llama_hparams)  (Data Structure)



---
### test\_llm<!-- {{#data_structure:test_llm}} -->
- **Type**: `struct`
- **Members**:
    - `hp`: Contains hyperparameters for the LLM.
- **Description**: The `test_llm` structure is a derived class from `test_case` that encapsulates the parameters and methods necessary for testing a language model, specifically focusing on the handling of hyperparameters and the construction of various layers within the model.
- **Member Functions**:
    - [`test_llm::test_llm`](#test_llmtest_llm)
    - [`test_llm::llm_build_norm`](#test_llmllm_build_norm)
    - [`test_llm::llm_build_kv_store`](#test_llmllm_build_kv_store)
    - [`test_llm::llm_build_kqv`](#test_llmllm_build_kqv)
    - [`test_llm::initialize_tensors`](#test_llminitialize_tensors)
- **Inherits From**:
    - [`test_case`](#test_case)

**Methods**

---
#### test\_llm::test\_llm<!-- {{#callable:test_llm::test_llm}} -->
The `test_llm` constructor initializes an instance of the `test_llm` class with specified hyperparameters.
- **Inputs**:
    - `hp`: An instance of `llama_hparams` structure containing hyperparameters for the LLM.
- **Control Flow**:
    - The constructor uses an initializer list to move the input hyperparameters into the member variable `hp`.
    - No additional logic or control flow is present in this constructor.
- **Output**: This constructor does not return a value; it initializes the `test_llm` object.
- **See also**: [`test_llm`](#test_llm)  (Data Structure)


---
#### test\_llm::llm\_build\_norm<!-- {{#callable:test_llm::llm_build_norm}} -->
The `llm_build_norm` function normalizes a tensor using specified normalization techniques and applies scaling and bias.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure that manages memory and operations.
    - `cur`: A pointer to the `ggml_tensor` that represents the current tensor to be normalized.
    - `mw`: A pointer to the `ggml_tensor` that represents the scaling weights.
    - `mb`: A pointer to the `ggml_tensor` that represents the bias, which can be null.
    - `type`: An enumeration value of type `llm_norm_type` that specifies the normalization method to use.
- **Control Flow**:
    - The function begins by checking the normalization type specified by the `type` parameter.
    - If the type is `LLM_NORM`, it applies the [`ggml_norm`](../ggml/src/ggml.c.driver.md#ggml_norm) function to the `cur` tensor with a predefined epsilon value.
    - If the type is `LLM_NORM_RMS`, it applies the [`ggml_rms_norm`](../ggml/src/ggml.c.driver.md#ggml_rms_norm) function to the `cur` tensor with a different epsilon value.
    - After normalization, the function multiplies the normalized tensor `cur` by the scaling tensor `mw` using [`ggml_mul`](../ggml/src/ggml.c.driver.md#ggml_mul).
    - If the bias tensor `mb` is not null, it adds the bias to the scaled tensor using [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add).
    - Finally, the function returns the modified tensor `cur`.
- **Output**: The function returns a pointer to the modified `ggml_tensor` after applying normalization, scaling, and optional bias addition.
- **Functions called**:
    - [`ggml_norm`](../ggml/src/ggml.c.driver.md#ggml_norm)
    - [`ggml_rms_norm`](../ggml/src/ggml.c.driver.md#ggml_rms_norm)
    - [`ggml_mul`](../ggml/src/ggml.c.driver.md#ggml_mul)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
- **See also**: [`test_llm`](#test_llm)  (Data Structure)


---
#### test\_llm::llm\_build\_kv\_store<!-- {{#callable:test_llm::llm_build_kv_store}} -->
The `llm_build_kv_store` function updates the key-value store tensors for a language model.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which holds the context for tensor operations.
    - `k_l`: A pointer to the `ggml_tensor` representing the long-term key tensor.
    - `v_l`: A pointer to the `ggml_tensor` representing the long-term value tensor.
    - `k_cur`: A pointer to the `ggml_tensor` representing the current key tensor to be updated.
    - `v_cur`: A pointer to the `ggml_tensor` representing the current value tensor to be updated.
- **Control Flow**:
    - The function begins by reshaping and transposing the current value tensor `v_cur` to match the expected dimensions.
    - It creates a view of the long-term key tensor `k_l` to access its elements in a specific format.
    - It also creates a view of the long-term value tensor `v_l` for similar access.
    - The function then copies the current key tensor `k_cur` into the key cache view created from `k_l`.
    - Finally, it copies the transposed current value tensor `v_cur_t` into the value cache view created from `v_l`.
- **Output**: The function does not return a value; it modifies the key and value tensors in place.
- **Functions called**:
    - [`ggml_transpose`](../ggml/src/ggml.c.driver.md#ggml_transpose)
    - [`ggml_reshape_2d`](../ggml/src/ggml.c.driver.md#ggml_reshape_2d)
    - [`ggml_view_1d`](../ggml/src/ggml.c.driver.md#ggml_view_1d)
    - [`ggml_row_size`](../ggml/src/ggml.c.driver.md#ggml_row_size)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_element_size`](../ggml/src/ggml.c.driver.md#ggml_element_size)
    - [`ggml_cpy`](../ggml/src/ggml.c.driver.md#ggml_cpy)
- **See also**: [`test_llm`](#test_llm)  (Data Structure)


---
#### test\_llm::llm\_build\_kqv<!-- {{#callable:test_llm::llm_build_kqv}} -->
The `llm_build_kqv` function constructs a tensor representing the key-query-value (KQV) attention mechanism for a language model.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages memory and operations for tensor computations.
    - `k_l`: A pointer to the `ggml_tensor` representing the key tensor, which contains the key embeddings.
    - `v_l`: A pointer to the `ggml_tensor` representing the value tensor, which contains the value embeddings.
    - `q_cur`: A pointer to the `ggml_tensor` representing the current query tensor that will be processed.
    - `kq_mask`: A pointer to the `ggml_tensor` used as a mask for the key-query attention, controlling which elements are attended to.
    - `kq_scale`: A float value used to scale the key-query product before applying the softmax operation.
- **Control Flow**:
    - The function begins by permuting the dimensions of the `q_cur` tensor to prepare it for matrix multiplication.
    - It then creates a 3D view of the key tensor `k_l` to facilitate the attention mechanism.
    - The function computes the matrix multiplication of the key tensor and the query tensor to obtain the key-query product.
    - A softmax operation is applied to the key-query product, using the provided mask and scale to normalize the attention scores.
    - Next, a 3D view of the value tensor `v_l` is created to align with the attention mechanism's requirements.
    - The function computes the matrix multiplication of the value tensor and the softmaxed key-query product to obtain the final KQV tensor.
    - The resulting tensor is permuted back to the original dimensions and reshaped into a 2D tensor for further processing.
    - Finally, a new tensor is created for the output, and the function returns the computed tensor.
- **Output**: The function returns a pointer to a `ggml_tensor` that represents the output of the KQV attention mechanism, which is used in the language model's forward pass.
- **Functions called**:
    - [`ggml_permute`](../ggml/src/ggml.c.driver.md#ggml_permute)
    - [`ggml_view_3d`](../ggml/src/ggml.c.driver.md#ggml_view_3d)
    - [`ggml_row_size`](../ggml/src/ggml.c.driver.md#ggml_row_size)
    - [`ggml_mul_mat`](../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_soft_max_ext`](../ggml/src/ggml.c.driver.md#ggml_soft_max_ext)
    - [`ggml_element_size`](../ggml/src/ggml.c.driver.md#ggml_element_size)
    - [`ggml_cont_2d`](../ggml/src/ggml.c.driver.md#ggml_cont_2d)
    - [`test_case::ggml_new_tensor_2d`](#test_caseggml_new_tensor_2d)
- **See also**: [`test_llm`](#test_llm)  (Data Structure)


---
#### test\_llm::initialize\_tensors<!-- {{#callable:test_llm::initialize_tensors}} -->
Initializes all tensors in the given `ggml_context` by populating them with random data or uniform values.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that contains the state and information about the tensors to be initialized.
- **Control Flow**:
    - Iterates through all tensors in the context using [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor) and [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor).
    - Checks the type of each tensor: if it is of type `GGML_TYPE_I32`, it initializes it with random integers; otherwise, it calls [`init_tensor_uniform`](#init_tensor_uniform) to initialize it with uniform values.
- **Output**: This function does not return a value; it modifies the tensors in the provided `ggml_context` directly.
- **Functions called**:
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set)
    - [`init_tensor_uniform`](#init_tensor_uniform)
- **See also**: [`test_llm`](#test_llm)  (Data Structure)



---
### test\_llama<!-- {{#data_structure:test_llama}} -->
- **Type**: `class`
- **Members**:
    - `freq_base`: A static constant representing the base frequency.
    - `freq_scale`: A static constant representing the frequency scale.
    - `ext_factor`: A static constant representing the external factor.
    - `attn_factor`: A static constant representing the attention factor.
    - `beta_fast`: A static constant representing the fast beta value.
    - `beta_slow`: A static constant representing the slow beta value.
- **Description**: The `test_llama` class is a derived class from `test_llm` that implements specific configurations and methods for testing the LLAMA model, including static constants for frequency parameters and methods for building the computation graph for the model's forward pass.
- **Member Functions**:
    - [`test_llama::op_desc`](#test_llamaop_desc)
    - [`test_llama::vars`](#test_llamavars)
    - [`test_llama::max_nmse_err`](#test_llamamax_nmse_err)
    - [`test_llama::test_llama`](#test_llamatest_llama)
    - [`test_llama::build_graph`](#test_llamabuild_graph)
- **Inherits From**:
    - [`test_llm::test_llm`](#test_llmtest_llm)

**Methods**

---
#### test\_llama::op\_desc<!-- {{#callable:test_llama::op_desc}} -->
The `op_desc` method returns a string description of the operation performed by the `test_llama` class.
- **Inputs**:
    - `ggml_tensor * t`: A pointer to a `ggml_tensor` structure, which is unused in this method.
- **Control Flow**:
    - The method starts by marking the input tensor `t` as unused using the `GGML_UNUSED` macro.
    - It then directly returns the string 'LLAMA'.
- **Output**: The output is a string literal 'LLAMA', which serves as a description of the operation.
- **See also**: [`test_llama`](#test_llama)  (Data Structure)


---
#### test\_llama::vars<!-- {{#callable:test_llama::vars}} -->
The `vars` method returns a string representation of the number of tokens in the `test_llama` class.
- **Inputs**: None
- **Control Flow**:
    - The method retrieves the number of tokens from the `hp` member variable.
    - It then calls the macro `VARS_TO_STR1` with the number of tokens as an argument to convert it to a string.
- **Output**: The output is a string representation of the number of tokens.
- **See also**: [`test_llama`](#test_llama)  (Data Structure)


---
#### test\_llama::max\_nmse\_err<!-- {{#callable:test_llama::max_nmse_err}} -->
The `max_nmse_err` function returns a constant maximum normalized mean squared error value.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a constant value without any conditional logic or loops.
- **Output**: The output is a double precision floating-point value of 0.002 (2e-3).
- **See also**: [`test_llama`](#test_llama)  (Data Structure)


---
#### test\_llama::test\_llama<!-- {{#callable:test_llama::test_llama}} -->
The `test_llama` constructor initializes a `test_llm` object with specific hyperparameters for a language model.
- **Inputs**:
    - `n_tokens`: An integer representing the number of tokens to be processed, defaulting to 1.
- **Control Flow**:
    - The constructor initializes the base class `test_llm` with a set of hyperparameters, including vocabulary size, embedding size, number of heads, and others.
    - The hyperparameters are set using a member initializer list, which allows for direct initialization of the base class with the specified values.
- **Output**: The constructor does not return a value but initializes the `test_llama` object with the specified parameters.
- **See also**: [`test_llama`](#test_llama)  (Data Structure)


---
#### test\_llama::build\_graph<!-- {{#callable:test_llama::build_graph}} -->
The `build_graph` function constructs a computational graph for a transformer model using various tensor operations.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure that manages memory and tensor operations.
- **Control Flow**:
    - Creates input tensors for embeddings, positions, and masks.
    - Iterates over the number of layers defined in the model's hyperparameters.
    - For each layer, applies normalization, self-attention, and feed-forward operations.
    - Uses learned weights to compute queries, keys, and values, applying rotary positional encoding.
    - Stores key-value pairs in a cache for efficient attention computation.
    - Applies a feed-forward network to the output of the attention mechanism.
    - Normalizes the final output before producing the final logits.
- **Output**: Returns a pointer to the final output tensor representing the model's predictions.
- **Functions called**:
    - [`test_case::ggml_new_tensor_2d`](#test_caseggml_new_tensor_2d)
    - [`test_case::ggml_new_tensor_1d`](#test_caseggml_new_tensor_1d)
    - [`test_case::ggml_new_tensor_3d`](#test_caseggml_new_tensor_3d)
    - [`test_llm::llm_build_norm`](#test_llmllm_build_norm)
    - [`ggml_mul_mat`](../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`test_llm::llm_build_kv_store`](#test_llmllm_build_kv_store)
    - [`test_llm::llm_build_kqv`](#test_llmllm_build_kqv)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
    - [`ggml_silu`](../ggml/src/ggml.c.driver.md#ggml_silu)
    - [`ggml_mul`](../ggml/src/ggml.c.driver.md#ggml_mul)
- **See also**: [`test_llama`](#test_llama)  (Data Structure)



---
### test\_falcon<!-- {{#data_structure:test_falcon}} -->
- **Type**: `struct`
- **Members**:
    - `freq_base`: Base frequency for the model.
    - `freq_scale`: Scale factor for frequency adjustments.
    - `ext_factor`: External factor for model adjustments.
    - `attn_factor`: Attention factor for model calculations.
    - `beta_fast`: Fast beta parameter for model optimization.
    - `beta_slow`: Slow beta parameter for model optimization.
- **Description**: The `test_falcon` struct is a derived class from `test_llm`, designed to implement a specific language model architecture with parameters for frequency scaling, attention mechanisms, and optimization factors, facilitating the construction and evaluation of the model's computational graph.
- **Member Functions**:
    - [`test_falcon::op_desc`](#test_falconop_desc)
    - [`test_falcon::vars`](#test_falconvars)
    - [`test_falcon::max_nmse_err`](#test_falconmax_nmse_err)
    - [`test_falcon::test_falcon`](#test_falcontest_falcon)
    - [`test_falcon::build_graph`](#test_falconbuild_graph)
- **Inherits From**:
    - [`test_llm::test_llm`](#test_llmtest_llm)

**Methods**

---
#### test\_falcon::op\_desc<!-- {{#callable:test_falcon::op_desc}} -->
The `op_desc` function returns a string description of the operation performed by the `test_falcon` class.
- **Inputs**:
    - `ggml_tensor * t`: A pointer to a `ggml_tensor` structure, which is unused in this function.
- **Control Flow**:
    - The function starts by marking the input tensor `t` as unused using the `GGML_UNUSED` macro.
    - It then directly returns the string 'FALCON'.
- **Output**: The function outputs a string literal 'FALCON', indicating the operation description.
- **See also**: [`test_falcon`](#test_falcon)  (Data Structure)


---
#### test\_falcon::vars<!-- {{#callable:test_falcon::vars}} -->
The `vars` method returns a string representation of the number of tokens in the `test_falcon` class.
- **Inputs**: None
- **Control Flow**:
    - The method retrieves the number of tokens from the `hp` member variable.
    - It then calls the macro `VARS_TO_STR1` with the number of tokens as an argument to generate a string representation.
- **Output**: The output is a string that represents the number of tokens in a specific format defined by the `VARS_TO_STR1` macro.
- **See also**: [`test_falcon`](#test_falcon)  (Data Structure)


---
#### test\_falcon::max\_nmse\_err<!-- {{#callable:test_falcon::max_nmse_err}} -->
The `max_nmse_err` function returns a constant maximum normalized mean squared error value.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a constant value without any conditional statements or loops.
- **Output**: The output is a double value representing the maximum normalized mean squared error, which is set to 2e-3.
- **See also**: [`test_falcon`](#test_falcon)  (Data Structure)


---
#### test\_falcon::test\_falcon<!-- {{#callable:test_falcon::test_falcon}} -->
The `test_falcon` constructor initializes a `test_llm` object with specific hyperparameters for the Falcon model.
- **Inputs**:
    - `n_tokens`: An integer representing the number of tokens to be processed, defaulting to 1.
- **Control Flow**:
    - The constructor initializes the base class `test_llm` with a set of hyperparameters specific to the Falcon model.
    - These hyperparameters include vocabulary size, embedding dimensions, number of heads, and other model-specific configurations.
    - The constructor uses an initializer list to pass these parameters to the `test_llm` constructor.
- **Output**: The constructor does not return a value but initializes the `test_falcon` object with the specified parameters.
- **See also**: [`test_falcon`](#test_falcon)  (Data Structure)


---
#### test\_falcon::build\_graph<!-- {{#callable:test_falcon::build_graph}} -->
The `build_graph` function constructs a computation graph for a transformer model using various tensor operations.
- **Inputs**:
    - `ctx`: A pointer to the `ggml_context` structure, which manages memory and tensor operations.
- **Control Flow**:
    - Creates input tensors for embeddings, positions, and masks.
    - Iterates over the number of layers defined in the model's hyperparameters.
    - For each layer, applies normalization, self-attention, and feed-forward operations.
    - Updates the input tensor for the next layer with the output of the current layer.
    - After processing all layers, applies final normalization and computes the output logits.
- **Output**: Returns a pointer to the final output tensor representing the model's predictions.
- **Functions called**:
    - [`test_case::ggml_new_tensor_2d`](#test_caseggml_new_tensor_2d)
    - [`test_case::ggml_new_tensor_1d`](#test_caseggml_new_tensor_1d)
    - [`test_case::ggml_new_tensor_3d`](#test_caseggml_new_tensor_3d)
    - [`test_llm::llm_build_norm`](#test_llmllm_build_norm)
    - [`ggml_mul_mat`](../ggml/src/ggml.c.driver.md#ggml_mul_mat)
    - [`ggml_cont`](../ggml/src/ggml.c.driver.md#ggml_cont)
    - [`ggml_view_2d`](../ggml/src/ggml.c.driver.md#ggml_view_2d)
    - [`ggml_reshape_3d`](../ggml/src/ggml.c.driver.md#ggml_reshape_3d)
    - [`ggml_rope_ext`](../ggml/src/ggml.c.driver.md#ggml_rope_ext)
    - [`test_llm::llm_build_kv_store`](#test_llmllm_build_kv_store)
    - [`test_llm::llm_build_kqv`](#test_llmllm_build_kqv)
    - [`ggml_gelu`](../ggml/src/ggml.c.driver.md#ggml_gelu)
    - [`ggml_add`](../ggml/src/ggml.c.driver.md#ggml_add)
- **See also**: [`test_falcon`](#test_falcon)  (Data Structure)



# Functions

---
### init\_tensor\_uniform<!-- {{#callable:init_tensor_uniform}} -->
The `init_tensor_uniform` function initializes a given tensor with random values uniformly distributed between specified minimum and maximum values, and handles different tensor types including quantized and integer types.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure that represents the tensor to be initialized.
    - `min`: A float representing the minimum value of the uniform distribution; defaults to -1.0f.
    - `max`: A float representing the maximum value of the uniform distribution; defaults to 1.0f.
- **Control Flow**:
    - Calculate the number of elements in the tensor using [`ggml_nelements`](../ggml/src/ggml.c.driver.md#ggml_nelements) and create a vector `data` to store the random values.
    - Initialize a static vector of random number generators, one for each thread, using `std::random_device` to seed them.
    - Define a lambda function `init_thread` that initializes a segment of the `data` vector with random values from a uniform distribution between `min` and `max`.
    - Launch multiple threads using `std::async` to execute `init_thread` in parallel, dividing the work among available hardware threads.
    - Wait for all threads to complete using `std::future::get`.
    - Depending on the tensor type, handle the data differently:
    - For `GGML_TYPE_F32` and `GGML_TYPE_I32`, directly set the tensor data using [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set).
    - For quantized types and `GGML_TYPE_F16` or `GGML_TYPE_BF16`, perform quantization in parallel by blocks and set the quantized data.
    - For integer types like `GGML_TYPE_I8`, `GGML_TYPE_I16`, and `GGML_TYPE_I32`, set the tensor data directly, noting that the values may not be meaningful integers.
    - For `GGML_TYPE_I64`, set the tensor data by mirroring the float data, splitting the data into two halves.
    - Abort with an error if the tensor type is unsupported.
- **Output**: The function does not return a value; it modifies the tensor in place by setting its data to the initialized values.
- **Functions called**:
    - [`ggml_nelements`](../ggml/src/ggml.c.driver.md#ggml_nelements)
    - [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set)
    - [`ggml_is_quantized`](../ggml/src/ggml.c.driver.md#ggml_is_quantized)
    - [`ggml_blck_size`](../ggml/src/ggml.c.driver.md#ggml_blck_size)
    - [`ggml_quantize_requires_imatrix`](../ggml/src/ggml.c.driver.md#ggml_quantize_requires_imatrix)
    - [`ggml_row_size`](../ggml/src/ggml.c.driver.md#ggml_row_size)
    - [`ggml_quantize_chunk`](../ggml/src/ggml.c.driver.md#ggml_quantize_chunk)
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)


---
### tensor\_to\_float<!-- {{#callable:tensor_to_float}} -->
The function `tensor_to_float` converts a `ggml_tensor` to a vector of floats, handling various data types and quantization.
- **Inputs**:
    - `t`: A pointer to a `ggml_tensor` structure, which contains the tensor data to be converted to a vector of floats.
- **Control Flow**:
    - Initialize an empty vector `tv` to store the resulting floats and reserve space based on the number of elements in the tensor `t`.
    - Create a buffer `buf` to hold the raw data from the tensor, and retrieve the data using [`ggml_backend_tensor_get`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_get).
    - Determine the block size `bs` and check if the tensor type is quantized.
    - Iterate over the tensor dimensions using nested loops to access each element by index, avoiding gaps in views.
    - For each element, check the tensor type and convert the data to a float using the appropriate conversion function or cast.
    - If the tensor type is quantized, use the type traits to convert the data to floats and append them to `tv`.
    - If the tensor type is unsupported, abort the operation with an error.
- **Output**: A `std::vector<float>` containing the converted float values from the input tensor.
- **Functions called**:
    - [`ggml_nelements`](../ggml/src/ggml.c.driver.md#ggml_nelements)
    - [`ggml_backend_tensor_get`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_get)
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)
    - [`ggml_get_type_traits`](../ggml/src/ggml.c.driver.md#ggml_get_type_traits)
    - [`ggml_blck_size`](../ggml/src/ggml.c.driver.md#ggml_blck_size)
    - [`ggml_is_quantized`](../ggml/src/ggml.c.driver.md#ggml_is_quantized)
    - [`ggml_fp16_to_fp32`](../ggml/src/ggml.c.driver.md#ggml_fp16_to_fp32)
    - [`ggml_bf16_to_fp32`](../ggml/src/ggml.c.driver.md#ggml_bf16_to_fp32)


---
### nmse<!-- {{#callable:nmse}} -->
The `nmse` function calculates the normalized mean squared error between two arrays of floats.
- **Inputs**:
    - `a`: A pointer to the first array of floats.
    - `b`: A pointer to the second array of floats.
    - `n`: The number of elements in each array.
- **Control Flow**:
    - Initialize two double variables, `mse_a_b` and `mse_a_0`, to 0.0 to store the mean squared errors.
    - Iterate over each element of the arrays from 0 to n-1.
    - For each element, calculate the squared difference between the corresponding elements of arrays `a` and `b`, and add it to `mse_a_b`.
    - For each element, calculate the square of the element in array `a` and add it to `mse_a_0`.
    - Return the ratio of `mse_a_b` to `mse_a_0` as the normalized mean squared error.
- **Output**: A double representing the normalized mean squared error between the two arrays.


---
### mean\_abs\_asymm<!-- {{#callable:mean_abs_asymm}} -->
The `mean_abs_asymm` function calculates the mean absolute asymmetry between two arrays of floats, optionally filtering based on expected values.
- **Inputs**:
    - `a`: A pointer to the first array of floats.
    - `b`: A pointer to the second array of floats.
    - `n`: The number of elements in the arrays to compare.
    - `expected_vals`: A vector of expected float values for filtering the comparisons; if empty, no filtering is applied.
- **Control Flow**:
    - Initialize a double variable `sum` to 0.0 and a size_t variable `nvalid` to 0.
    - Iterate over each element in the arrays `a` and `b` up to `n`.
    - If `expected_vals` is not empty, check if the current element in `a` matches any value in `expected_vals` within a tolerance of 1e-3; if not, skip to the next iteration.
    - Calculate the asymmetry for the current elements of `a` and `b` using the formula `(a[i] - b[i]) / (a[i] + b[i])`.
    - Add the absolute value of the calculated asymmetry to `sum` and increment `nvalid`.
    - Return the mean absolute asymmetry by dividing `sum` by `nvalid`.
- **Output**: Returns a double representing the mean absolute asymmetry of the valid comparisons.


---
### var\_to\_str<!-- {{#callable:var_to_str}} -->
The `var_to_str` function converts a `ggml_scale_mode` enumeration value to its corresponding string representation.
- **Inputs**:
    - `mode`: A `ggml_scale_mode` enumeration value that specifies the scaling mode to be converted to a string.
- **Control Flow**:
    - The function uses a switch statement to determine the string representation of the `mode` argument.
    - If `mode` is `GGML_SCALE_MODE_NEAREST`, the function returns the string "nearest".
    - If `mode` is `GGML_SCALE_MODE_BILINEAR`, the function returns the string "bilinear".
    - For any other value of `mode`, the function returns the string representation of the integer value of `mode` using `std::to_string`.
- **Output**: A `std::string` representing the name of the scaling mode or its integer value as a string if not recognized.


---
### \_isinf<!-- {{#callable:_isinf}} -->
The function `_isinf` checks if a given float value is infinite.
- **Inputs**:
    - `f`: A float value to be checked for infinity.
- **Control Flow**:
    - The function uses the standard library function `std::isinf` to determine if the input float `f` is infinite.
    - If the macro `GGML_USE_SYCL` is defined, it uses a bitwise operation to check for infinity instead.
- **Output**: A boolean value indicating whether the input float is infinite.


---
### isinf\_or\_max<!-- {{#callable:isinf_or_max}} -->
The function `isinf_or_max` checks if a given float value is either infinite or equal to the maximum or minimum representable float value.
- **Inputs**:
    - `f`: A float value to be checked for being infinite or equal to FLT_MAX or -FLT_MAX.
- **Control Flow**:
    - The function calls `_isinf(f)` to check if the float `f` is infinite.
    - It checks if `f` is equal to `FLT_MAX`, the maximum representable float value.
    - It checks if `f` is equal to `-FLT_MAX`, the minimum representable float value.
    - The function returns true if any of the above conditions are met, otherwise it returns false.
- **Output**: A boolean value indicating whether the input float is infinite or equal to the maximum or minimum float value.
- **Functions called**:
    - [`_isinf`](#_isinf)


---
### ggml\_is\_view\_op<!-- {{#callable:ggml_is_view_op}} -->
The function `ggml_is_view_op` checks if a given operation is one of the view-related operations in the GGML library.
- **Inputs**:
    - `op`: An enumeration value of type `ggml_op` representing the operation to be checked.
- **Control Flow**:
    - The function compares the input operation `op` against four specific operations: `GGML_OP_VIEW`, `GGML_OP_RESHAPE`, `GGML_OP_PERMUTE`, and `GGML_OP_TRANSPOSE`.
    - If `op` matches any of these operations, the function returns `true`.
    - If `op` does not match any of these operations, the function returns `false`.
- **Output**: A boolean value indicating whether the operation is a view-related operation (`true`) or not (`false`).


---
### make\_test\_cases\_eval<!-- {{#callable:make_test_cases_eval}} -->
The function `make_test_cases_eval` generates a collection of test cases for evaluating various GGML operations and configurations.
- **Inputs**: None
- **Control Flow**:
    - Initialize an empty vector `test_cases` to store unique pointers to `test_case` objects.
    - Create a random number generator `rng` with a fixed seed for reproducibility.
    - Iterate over different GGML types and configurations to create test cases for unary operations, adding them to `test_cases`.
    - Add specific test cases for `test_get_rows` and `test_get_rows_back` with various configurations.
    - Generate test cases for 2D pooling operations with different kernel sizes, strides, and paddings.
    - Add test cases for 1D and 2D `im2col` operations with various configurations.
    - Include test cases for depthwise 2D convolution operations with different parameters.
    - Create test cases for 1D transposed convolution operations with various configurations.
    - Add test cases for `test_count_equal` and `test_argmax` operations with different input sizes.
    - Generate test cases for `test_repeat` and `test_repeat_back` operations with different configurations.
    - Add test cases for `test_dup` operations with and without permutations.
    - Create test cases for `test_set` operations with different dimensions.
    - Generate test cases for `test_cpy` operations with various source and destination types and permutations.
    - Add test cases for `test_cont` operations with different input sizes.
    - Define a lambda function `add_test_bin_bcast` to add test cases for binary operations with broadcasting, and use it to add such test cases.
    - Add test cases for `test_add1`, `test_scale`, and `test_silu_back` operations.
    - Generate test cases for normalization operations (`test_norm`, `test_rms_norm`, `test_rms_norm_back`, `test_l2_norm`) with different configurations.
    - Add test cases for `test_ssm_conv` and `test_ssm_scan` operations with specific configurations.
    - Create test cases for `test_rwkv_wkv6`, `test_rwkv_wkv7`, and `test_gla` operations with various parameters.
    - Generate test cases for matrix multiplication operations (`test_mul_mat`, `test_mul_mat_id`) with different types and configurations.
    - Add test cases for `test_out_prod` operations with various configurations.
    - Include test cases for mathematical operations (`test_sqr`, `test_sqrt`, `test_log`, `test_sin`, `test_cos`, `test_clamp`) with different input types.
    - Add test cases for `test_diag_mask_inf` operations with specific configurations.
    - Generate test cases for `test_soft_max` and `test_soft_max_back` operations with different configurations.
    - Add test cases for `test_rope` operations with various configurations, including forward and backward passes.
    - Create test cases for 2D pooling operations (`test_pool2d`) with different configurations.
    - Add test cases for `test_conv_transpose_1d` operations with various configurations.
    - Include test cases for `test_im2col` operations with different configurations.
    - Add test cases for `test_conv_2d_dw` operations with various configurations.
    - Generate test cases for `test_concat` operations with different dimensions and views.
    - Add test cases for `test_argsort` operations with different orders.
    - Create test cases for `test_sum`, `test_sum_rows`, and `test_mean` operations with different input sizes.
    - Add test cases for `test_upscale` and `test_upscale_ext` operations with different configurations.
    - Include test cases for `test_group_norm` operations with specific configurations.
    - Add test cases for `test_acc`, `test_pad`, `test_pad_reflect_1d`, `test_arange`, `test_timestep_embedding`, and `test_leaky_relu` operations.
    - Generate test cases for `test_flash_attn_ext` operations with various configurations.
    - Add test cases for `test_cross_entropy_loss` and `test_cross_entropy_loss_back` operations with different input sizes.
    - Include test cases for `test_opt_step_adamw` operations with specific configurations.
    - Return the populated `test_cases` vector.
- **Output**: A vector of unique pointers to `test_case` objects, each representing a specific test case for evaluating GGML operations.
- **Functions called**:
    - [`ggml_blck_size`](../ggml/src/ggml.c.driver.md#ggml_blck_size)


---
### make\_test\_cases\_perf<!-- {{#callable:make_test_cases_perf}} -->
The `make_test_cases_perf` function creates a vector of unique pointers to `test_case` objects, each representing a performance test case for various GGML operations.
- **Inputs**: None
- **Control Flow**:
    - Initialize an empty vector `test_cases` to store unique pointers to `test_case` objects.
    - Add various `test_case` objects to the `test_cases` vector using `emplace_back`, each representing a different GGML operation with specific parameters.
    - Return the populated `test_cases` vector.
- **Output**: A vector of unique pointers to `test_case` objects, each configured for performance testing of GGML operations.


---
### test\_backend<!-- {{#callable:test_backend}} -->
The `test_backend` function evaluates a given backend's performance and correctness by running a series of tests based on the specified mode and filters.
- **Inputs**:
    - `backend`: A `ggml_backend_t` object representing the backend to be tested.
    - `mode`: A `test_mode` enum value indicating the type of test to run: MODE_TEST, MODE_GRAD, or MODE_PERF.
    - `op_name`: A string representing the name of the operation to filter tests by, or `nullptr` to include all operations.
    - `params_filter`: A string representing a regex pattern to filter test cases by their parameters, or `nullptr` to include all parameters.
- **Control Flow**:
    - Define a lambda function `filter_test_cases` to filter test cases based on `params_filter` using regex.
    - Check if `mode` is MODE_TEST, create test cases using [`make_test_cases_eval`](#make_test_cases_eval), filter them, and initialize a CPU backend for comparison.
    - Iterate over the test cases, evaluate each one using the `eval` method, and count the number of successful tests.
    - Free the CPU backend and return whether all tests passed.
    - Check if `mode` is MODE_GRAD, create and filter test cases, and evaluate gradients using the `eval_grad` method for each test case.
    - Check if `mode` is MODE_PERF, create and filter test cases, and evaluate performance using the `eval_perf` method for each test case.
    - If none of the modes match, call `GGML_ABORT` with a fatal error message.
- **Output**: Returns a boolean indicating whether all tests passed for the given backend and mode.
- **Functions called**:
    - [`make_test_cases_eval`](#make_test_cases_eval)
    - [`ggml_backend_free`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_free)
    - [`make_test_cases_perf`](#make_test_cases_perf)


---
### usage<!-- {{#callable:usage}} -->
The `usage` function prints the usage instructions for the program, detailing the valid modes and options available for execution.
- **Inputs**:
    - `argv`: A pointer to an array of character strings representing the command-line arguments passed to the program.
- **Control Flow**:
    - The function uses `printf` to output the usage instructions to the standard output.
    - It specifies the program name using `argv[0]` and details the valid modes and options available.
- **Output**: The function does not return any value; it simply prints the usage instructions to the standard output.


---
### main<!-- {{#callable:main}} -->
The `main` function processes command-line arguments to set test mode and filters, loads and tests available backends, and reports the results.
- **Inputs**:
    - `argc`: An integer representing the number of command-line arguments.
    - `argv`: An array of C-style strings representing the command-line arguments.
- **Control Flow**:
    - Initialize default test mode to MODE_TEST and filters to nullptr.
    - Iterate over command-line arguments to set mode and filters based on specific flags.
    - Load all available backends using ggml_backend_load_all().
    - Print the number of devices to be tested.
    - Iterate over each backend device, checking if it matches the backend filter.
    - Skip CPU backends if not in MODE_GRAD and no specific backend filter is set.
    - Initialize the backend for each device and set the number of threads if applicable.
    - Print device description and memory information.
    - Call test_backend() for each backend and print the result.
    - Free resources associated with each backend after testing.
    - Free quantization resources using ggml_quantize_free().
    - Print the number of backends that passed the tests and return 0 if all passed, otherwise return 1.
- **Output**: Returns an integer status code: 0 if all backends pass the tests, 1 otherwise.
- **Functions called**:
    - [`usage`](#usage)
    - [`ggml_backend_load_all`](../ggml/src/ggml-backend-reg.cpp.driver.md#ggml_backend_load_all)
    - [`ggml_backend_dev_count`](../ggml/src/ggml-backend-reg.cpp.driver.md#ggml_backend_dev_count)
    - [`ggml_backend_dev_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_name)
    - [`ggml_backend_dev_type`](../ggml/include/ggml-backend.h.driver.md#ggml_backend_dev_type)
    - [`ggml_backend_reg_get_proc_address`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_reg_get_proc_address)
    - [`ggml_backend_dev_description`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_description)
    - [`ggml_backend_dev_memory`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_memory)
    - [`test_backend`](#test_backend)
    - [`ggml_backend_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_name)
    - [`ggml_backend_free`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_free)
    - [`ggml_quantize_free`](../ggml/src/ggml.c.driver.md#ggml_quantize_free)


---
### test\_llm<!-- {{#callable:test_llm::test_llm}} -->
The `test_llm` constructor initializes a `test_llm` object with given `llama_hparams` by moving the parameter into the class member `hp`.
- **Inputs**:
    - `hp`: An instance of `llama_hparams` which contains various hyperparameters for the LLM (Large Language Model) such as vocabulary size, embedding dimensions, number of heads, etc.
- **Control Flow**:
    - The constructor takes a `llama_hparams` object as an argument.
    - It initializes the `hp` member of the `test_llm` class by moving the input `llama_hparams` object into it.
- **Output**: There is no return value as this is a constructor for initializing an object.


