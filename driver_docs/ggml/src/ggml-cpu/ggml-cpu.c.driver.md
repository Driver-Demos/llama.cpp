# Purpose
The provided C source code file is part of a larger software library that appears to be focused on numerical computations, particularly involving tensor operations and multi-threading. The file includes a variety of headers, suggesting it is part of a modular system, likely a machine learning or scientific computing library. The code defines several macros and includes platform-specific headers to handle memory allocation and threading, indicating cross-platform compatibility. It also includes functionality for handling different CPU architectures, such as ARM and x86, and optimizations for specific instruction sets like AVX and NEON.

The file contains implementations for various tensor operations, such as matrix multiplication, element-wise operations, and data type conversions between different floating-point formats (e.g., FP32 to FP16). It also includes threading support, with functions to manage thread pools and distribute computational tasks across multiple threads. The code is structured to handle different CPU capabilities, using conditional compilation to include or exclude features based on the detected architecture. This suggests the library is designed to be efficient and portable, leveraging hardware-specific optimizations where available.
# Imports and Dependencies

---
- `ggml-backend-impl.h`
- `ggml-backend.h`
- `ggml-cpu-traits.h`
- `ggml-cpu-impl.h`
- `ggml-cpu.h`
- `ggml-impl.h`
- `ggml-cpu-quants.h`
- `ggml-threading.h`
- `unary-ops.h`
- `binary-ops.h`
- `vec.h`
- `ops.h`
- `ggml.h`
- `malloc.h`
- `alloca.h`
- `assert.h`
- `errno.h`
- `time.h`
- `math.h`
- `stdlib.h`
- `string.h`
- `stdint.h`
- `inttypes.h`
- `stdio.h`
- `float.h`
- `limits.h`
- `stdarg.h`
- `signal.h`
- `syscall.h`
- `omp.h`
- `llamafile/sgemm.h`
- `windows.h`
- `stdatomic.h`
- `pthread.h`
- `sched.h`
- `pthread_np.h`
- `sys/types.h`
- `sys/stat.h`
- `unistd.h`
- `mach/mach.h`
- `TargetConditionals.h`
- `sys/auxv.h`
- `sys/sysctl.h`
- `sys/resource.h`


# Global Variables

---
### ggml\_arm\_arch\_features
- **Type**: `struct ggml_arm_arch_features_type`
- **Description**: The `ggml_arm_arch_features` is a global variable of type `struct ggml_arm_arch_features_type` that holds information about the ARM architecture features available on the system. It includes flags for various ARM features such as NEON, dot product, integer 8x8 matrix multiplication, SVE, and SME, as well as a count for SVE vector length.
- **Use**: This variable is used to determine the availability of specific ARM architecture features for optimized computation.


---
### type\_traits\_cpu
- **Type**: ``struct ggml_type_traits_cpu``
- **Description**: The `type_traits_cpu` is a static constant array of `struct ggml_type_traits_cpu` that holds type-specific function pointers and properties for different data types used in the GGML library. Each element in the array corresponds to a specific data type, such as `GGML_TYPE_F32`, `GGML_TYPE_F16`, etc., and contains function pointers for operations like vector dot product and conversion from float, as well as metadata like the number of rows processed per operation.
- **Use**: This variable is used to define and access type-specific operations and properties for various data types in the GGML library, facilitating efficient computation on different CPU architectures.


---
### g\_state
- **Type**: `struct ggml_state`
- **Description**: The `g_state` variable is a static instance of the `ggml_state` structure, initialized to zero. This structure contains a `ggml_numa_nodes` member, which is used to manage NUMA (Non-Uniform Memory Access) node information, including CPU and node counts, and the current node on which the main process is executing.
- **Use**: The `g_state` variable is used to store and manage the global state related to NUMA configuration and CPU affinity for the application.


# Data Structures

---
### ggml\_arm\_arch\_features\_type
- **Type**: `struct`
- **Members**:
    - `has_neon`: Indicates if the ARM architecture supports NEON instructions.
    - `has_dotprod`: Indicates if the ARM architecture supports dot product instructions.
    - `has_i8mm`: Indicates if the ARM architecture supports integer 8x8 matrix multiplication instructions.
    - `has_sve`: Indicates if the ARM architecture supports Scalable Vector Extension (SVE) instructions.
    - `sve_cnt`: Stores the count of SVE vector length.
    - `has_sme`: Indicates if the ARM architecture supports Scalable Matrix Extension (SME) instructions.
- **Description**: The `ggml_arm_arch_features_type` structure is used to store information about the capabilities of an ARM architecture, specifically regarding its support for various advanced instruction sets such as NEON, dot product, integer 8x8 matrix multiplication, SVE, and SME. Each member of the structure is an integer that acts as a boolean flag, where a value of -1 indicates that the feature is not supported or not detected, and a positive value indicates support. This structure is initialized with default values indicating no support for these features, and it is likely updated at runtime based on the actual capabilities of the hardware.


---
### memory\_order
- **Type**: `enum`
- **Members**:
    - `memory_order_relaxed`: Specifies relaxed memory ordering, allowing for maximum optimization by the compiler.
    - `memory_order_consume`: Ensures that operations that consume a value are ordered after the operations that produce it.
    - `memory_order_acquire`: Ensures that subsequent operations are not moved before the acquire operation.
    - `memory_order_release`: Ensures that prior operations are not moved after the release operation.
    - `memory_order_acq_rel`: Combines the effects of both acquire and release memory orders.
    - `memory_order_seq_cst`: Provides sequential consistency, ensuring a total order of operations across all threads.
- **Description**: The `memory_order` enum defines various memory ordering constraints for atomic operations in concurrent programming. These constraints dictate how memory operations are ordered with respect to each other, providing different levels of synchronization and visibility guarantees. The enum includes options for relaxed, consume, acquire, release, acquire-release, and sequentially consistent memory orders, each offering a different trade-off between performance and strictness of memory operation ordering.


---
### ggml\_threadpool
- **Type**: `struct`
- **Members**:
    - `mutex`: A mutex used for synchronizing access to the condition variable.
    - `cond`: A condition variable used for waiting for new work.
    - `cgraph`: A pointer to a computation graph structure.
    - `cplan`: A pointer to a computation plan structure.
    - `n_graph`: An atomic integer incremented when there is work to be done.
    - `n_barrier`: An atomic integer used for synchronization barriers.
    - `n_barrier_passed`: An atomic integer tracking the number of barriers passed.
    - `current_chunk`: An atomic integer indicating the current processing chunk during matrix multiplication.
    - `stop`: An atomic boolean used to stop the threadpool altogether.
    - `pause`: An atomic boolean used to pause the threadpool or individual threads.
    - `abort`: An atomic integer used to abort the processing of a graph.
    - `workers`: A pointer to an array of per-thread state structures.
    - `n_threads_max`: The maximum number of threads in the pool.
    - `n_threads_cur`: The number of threads used in the current graph.
    - `prio`: An integer representing the scheduling priority.
    - `poll`: An unsigned integer representing the polling level.
    - `ec`: An enumeration representing the execution status.
- **Description**: The `ggml_threadpool` structure is designed to manage a pool of threads for executing tasks in parallel, particularly for processing computation graphs. It includes synchronization primitives like mutexes and condition variables to coordinate thread activities, atomic variables for managing work distribution and synchronization barriers, and control flags for stopping, pausing, or aborting operations. The structure also maintains information about the number of threads, their scheduling priority, and the current execution status, making it a comprehensive tool for managing multi-threaded computation tasks.


---
### ggml\_compute\_state
- **Type**: `struct`
- **Members**:
    - `thrd`: Represents a thread in the compute state, used when OpenMP is not enabled.
    - `cpumask`: An array indicating the CPU affinity mask for the thread, used when OpenMP is not enabled.
    - `last_graph`: Stores the last graph index processed by the thread, used when OpenMP is not enabled.
    - `pending`: Indicates if there is pending work for the thread, used when OpenMP is not enabled.
    - `threadpool`: Pointer to the threadpool structure managing the compute state.
    - `ith`: Index of the thread within the threadpool.
- **Description**: The `ggml_compute_state` structure is used to manage the state of a computation thread within a threadpool in the GGML library. It contains information about the thread itself, such as its index and CPU affinity, and is used to coordinate the execution of tasks across multiple threads. The structure is conditionally compiled to include additional fields when OpenMP is not used, allowing for manual thread management and CPU affinity settings.


---
### ggml\_numa\_node
- **Type**: `struct`
- **Members**:
    - `cpus`: An array of hardware threads on this NUMA node.
    - `n_cpus`: The number of CPUs (hardware threads) on this node.
- **Description**: The `ggml_numa_node` structure represents a NUMA (Non-Uniform Memory Access) node in a system, which is a grouping of CPUs that share a common memory. This structure contains an array `cpus` that holds the identifiers of the hardware threads (CPUs) associated with this node, and an integer `n_cpus` that indicates the total number of CPUs in this node. This structure is used to manage and optimize memory access patterns in systems with NUMA architecture.


---
### ggml\_numa\_nodes
- **Type**: `struct`
- **Members**:
    - `numa_strategy`: Specifies the NUMA strategy to be used.
    - `nodes`: An array of NUMA nodes, each containing information about CPUs.
    - `n_nodes`: The number of NUMA nodes available.
    - `total_cpus`: The total number of hardware threads on the system.
    - `current_node`: The NUMA node on which the main process is executing.
    - `cpuset`: The CPU set from numactl, used only on Linux systems.
- **Description**: The `ggml_numa_nodes` structure is designed to manage and represent the NUMA (Non-Uniform Memory Access) configuration of a system. It includes a strategy for NUMA allocation, an array of `ggml_numa_node` structures that detail the CPUs available on each NUMA node, and metadata such as the total number of CPUs and the current node where the main process is running. This structure is particularly useful for optimizing memory access patterns in multi-threaded applications on systems with NUMA architecture, ensuring that processes are executed on the most appropriate CPUs to minimize latency and maximize performance. The `cpuset` member is conditionally compiled to support Linux-specific NUMA operations.


---
### ggml\_state
- **Type**: `struct`
- **Members**:
    - `numa`: A structure representing NUMA (Non-Uniform Memory Access) nodes.
- **Description**: The `ggml_state` structure is a simple data structure that encapsulates the state of NUMA nodes within the system. It contains a single member, `numa`, which is a structure of type `ggml_numa_nodes`. This member is responsible for managing and storing information related to the NUMA nodes, such as the strategy for NUMA allocation, the nodes themselves, the total number of CPUs, and the current node on which the main process is executing. This structure is crucial for handling memory access patterns in systems with NUMA architecture, ensuring efficient memory usage and process execution.


---
### mmid\_row\_mapping
- **Type**: `struct`
- **Members**:
    - `i1`: An integer field of type int32_t.
    - `i2`: Another integer field of type int32_t.
- **Description**: The `mmid_row_mapping` structure is a simple data structure that contains two integer fields, `i1` and `i2`, both of type `int32_t`. This structure is likely used to map or associate two integer values, possibly representing indices or identifiers, within a larger context or algorithm.


# Functions

---
### atomic\_store<!-- {{#callable:atomic_store}} -->
Stores a value atomically in an `atomic_int` variable.
- **Inputs**:
    - `ptr`: A pointer to an `atomic_int` variable where the value will be stored.
    - `val`: The value of type `LONG` that will be stored in the atomic variable.
- **Control Flow**:
    - The function calls `InterlockedExchange` with the provided pointer and value.
    - This operation atomically replaces the value at the memory location pointed to by `ptr` with `val`.
- **Output**: The function does not return a value; it performs an atomic store operation.


---
### atomic\_store\_explicit<!-- {{#callable:atomic_store_explicit}} -->
Stores a value in an atomic integer with an explicit memory order.
- **Inputs**:
    - `ptr`: A pointer to an `atomic_int` variable where the value will be stored.
    - `val`: The value of type `LONG` that will be stored in the atomic variable.
    - `mo`: An enumeration of type `memory_order` that specifies the memory ordering constraints for the operation.
- **Control Flow**:
    - The function begins by calling `InterlockedExchange` to store the value `val` into the atomic variable pointed to by `ptr`.
    - The function currently does not implement any logic to handle the `memory_order` parameter, as indicated by the TODO comment.
- **Output**: The function does not return a value; it performs the operation of storing the value atomically.


---
### atomic\_load<!-- {{#callable:atomic_load}} -->
The `atomic_load` function atomically loads the value of an `atomic_int` variable.
- **Inputs**:
    - `ptr`: A pointer to an `atomic_int` variable from which the value is to be loaded.
- **Control Flow**:
    - The function calls `InterlockedCompareExchange` with the pointer to the atomic integer, a value of 0, and a comparand of 0.
    - This operation atomically compares the value at the address pointed to by `ptr` with the comparand (0) and, if they are equal, replaces it with the new value (0) while returning the original value.
- **Output**: The function returns the original value of the atomic integer before the operation.


---
### atomic\_load\_explicit<!-- {{#callable:atomic_load_explicit}} -->
The `atomic_load_explicit` function atomically loads a value from an `atomic_int` pointer with a specified memory order.
- **Inputs**:
    - `ptr`: A pointer to an `atomic_int` variable from which the value will be loaded.
    - `mo`: A `memory_order` enumeration value that specifies the memory ordering constraints for the load operation.
- **Control Flow**:
    - The function calls `InterlockedCompareExchange` with the pointer `ptr`, a value of 0, and a comparand of 0.
    - This effectively reads the value at `ptr` atomically, ensuring that no other thread can modify it during the read operation.
- **Output**: Returns the value that was loaded from the `atomic_int` pointed to by `ptr`.


---
### atomic\_fetch\_add<!-- {{#callable:atomic_fetch_add}} -->
The `atomic_fetch_add` function atomically adds a specified value to an integer and returns the original value.
- **Inputs**:
    - `ptr`: A pointer to an `atomic_int` variable that will be incremented.
    - `inc`: A `LONG` value that specifies the amount to add to the integer pointed to by `ptr`.
- **Control Flow**:
    - The function calls `InterlockedExchangeAdd`, which performs the atomic addition operation.
    - The original value of the integer at `ptr` is returned before the addition is performed.
- **Output**: Returns the original value of the integer before the addition.


---
### atomic\_thread\_fence<!-- {{#callable:atomic_thread_fence}} -->
The `atomic_thread_fence` function provides a memory barrier to enforce ordering constraints on memory operations.
- **Inputs**:
    - `mo`: An enumeration value of type `memory_order` that specifies the memory ordering constraints.
- **Control Flow**:
    - The function calls `MemoryBarrier()` which is a platform-specific function that ensures all memory operations before the barrier are completed before any operations after the barrier.
    - No conditional logic or loops are present in this function, as it directly invokes the memory barrier.
- **Output**: The function does not return a value; it enforces memory ordering but does not produce a direct output.


---
### pthread\_create<!-- {{#callable:pthread_create}} -->
Creates a new thread to execute a specified function with an argument.
- **Inputs**:
    - `out`: A pointer to a `pthread_t` variable where the thread identifier will be stored.
    - `unused`: An unused parameter, typically not utilized in the function.
    - `func`: A pointer to the function that the new thread will execute, which takes a single `void*` argument and returns a `thread_ret_t`.
    - `arg`: A pointer to the argument that will be passed to the function `func`.
- **Control Flow**:
    - The function starts by ignoring the `unused` parameter.
    - It calls the Windows API function `CreateThread` to create a new thread, passing the function pointer `func` and the argument `arg`.
    - If `CreateThread` returns NULL, indicating failure to create the thread, the function returns the error code `EAGAIN`.
    - If the thread is created successfully, the thread handle is stored in the variable pointed to by `out`, and the function returns 0.
- **Output**: Returns 0 on success or `EAGAIN` if the thread could not be created.


---
### pthread\_join<!-- {{#callable:pthread_join}} -->
The `pthread_join` function waits for the specified thread to terminate and then closes its handle.
- **Inputs**:
    - `thread`: A `pthread_t` type representing the thread identifier of the thread to wait for.
    - `unused`: A pointer to unused data, which is ignored in this implementation.
- **Control Flow**:
    - The function starts by casting the `unused` parameter to void to suppress compiler warnings.
    - It then calls `WaitForSingleObject` with the `thread` handle and `INFINITE` to block until the specified thread terminates.
    - After the thread has terminated, it calls `CloseHandle` to release the thread's handle.
    - Finally, it returns the result of the `WaitForSingleObject` call, which indicates the success or failure of the wait operation.
- **Output**: The function returns an integer indicating the result of the wait operation, where a return value of zero indicates success.


---
### sched\_yield<!-- {{#callable:sched_yield}} -->
The `sched_yield` function allows the calling thread to yield the processor, enabling other threads to run.
- **Inputs**: None
- **Control Flow**:
    - The function calls `Sleep(0)`, which causes the calling thread to yield its remaining time slice.
    - This allows other threads that are ready to run to be scheduled.
- **Output**: The function returns 0, indicating successful completion of the yield operation.


---
### ggml\_get\_type\_traits\_cpu<!-- {{#callable:ggml_get_type_traits_cpu}} -->
The `ggml_get_type_traits_cpu` function retrieves the CPU type traits for a specified `ggml_type`.
- **Inputs**:
    - `type`: An enumeration value of type `ggml_type` that specifies the type for which the traits are to be retrieved.
- **Control Flow**:
    - The function accesses a static array `type_traits_cpu` indexed by the input `type`.
    - It returns a pointer to the corresponding `ggml_type_traits_cpu` structure.
- **Output**: Returns a pointer to a `ggml_type_traits_cpu` structure that contains the traits associated with the specified type.


---
### ggml\_thread\_cpu\_relax<!-- {{#callable:ggml_thread_cpu_relax}} -->
The `ggml_thread_cpu_relax` function is a no-operation function used to yield the CPU in a thread-safe manner.
- **Inputs**: None
- **Control Flow**:
    - The function does not contain any control flow statements as it is defined as a no-operation (NOP) function.
    - It is implemented as an inline function that does nothing, effectively serving as a placeholder for CPU relaxation.
- **Output**: The function does not return any value or output, as it is designed solely to yield control without performing any operations.


---
### ggml\_barrier<!-- {{#callable:ggml_barrier}} -->
The `ggml_barrier` function synchronizes threads in a thread pool, ensuring that all threads reach a certain point before any can proceed.
- **Inputs**:
    - `tp`: A pointer to a `ggml_threadpool` structure that contains information about the current thread pool, including the number of threads and synchronization variables.
- **Control Flow**:
    - The function first retrieves the current number of threads in the thread pool using atomic operations.
    - If there is only one thread, the function returns immediately, as no synchronization is needed.
    - If OpenMP is enabled, it uses `#pragma omp barrier` to synchronize threads.
    - If OpenMP is not used, it checks how many threads have passed the barrier and increments the barrier count atomically.
    - If the current thread is the last to reach the barrier, it resets the barrier count and increments the count of passed threads.
    - If not the last thread, it enters a loop, waiting until the count of passed threads increases, indicating that all threads have reached the barrier.
    - Finally, it ensures a full memory fence to maintain proper memory ordering.
- **Output**: The function does not return a value; it ensures that all threads in the pool reach the same point of execution before any can continue.
- **Functions called**:
    - [`atomic_load_explicit`](#atomic_load_explicit)
    - [`atomic_fetch_add_explicit`](#atomic_fetch_add_explicit)
    - [`atomic_store_explicit`](#atomic_store_explicit)
    - [`ggml_thread_cpu_relax`](#ggml_thread_cpu_relax)
    - [`atomic_thread_fence`](#atomic_thread_fence)


---
### ggml\_numa\_init<!-- {{#callable:ggml_numa_init}} -->
Initializes NUMA (Non-Uniform Memory Access) settings for the application.
- **Inputs**:
    - `numa_flag`: An enumeration value that specifies the NUMA strategy to be used.
- **Control Flow**:
    - Checks if NUMA has already been initialized by verifying if `g_state.numa.n_nodes` is greater than 0.
    - If already initialized, it logs a message and exits the function.
    - Sets the NUMA strategy based on the provided `numa_flag`.
    - Retrieves the current NUMA affinity and stores it in `g_state.numa.cpuset`.
    - Enumerates the NUMA nodes by checking the existence of directories in the `/sys/devices/system/node/` path.
    - Counts the total CPUs available by checking the existence of directories in the `/sys/devices/system/cpu/` path.
    - Logs the number of found NUMA nodes and CPUs.
    - Determines the current CPU and its associated NUMA node using the `getcpu` system call.
    - If no nodes or CPUs are found, it resets the node count and exits.
    - Populates the `g_state.numa.nodes` structure with the CPUs associated with each NUMA node.
    - Checks if NUMA balancing is enabled and logs a warning if it is.
- **Output**: The function does not return a value but initializes the global state for NUMA, setting up the number of nodes and CPUs available for the application.
- **Functions called**:
    - [`ggml_get_numa_affinity`](#ggml_get_numa_affinity)
    - [`ggml_is_numa`](#ggml_is_numa)


---
### ggml\_is\_numa<!-- {{#callable:ggml_is_numa}} -->
The `ggml_is_numa` function checks if the system has more than one NUMA node.
- **Inputs**: None
- **Control Flow**:
    - The function accesses the global state variable `g_state` to check the number of NUMA nodes.
    - It evaluates whether the number of nodes is greater than 1.
    - The result of this evaluation is returned as a boolean value.
- **Output**: The function returns a boolean value: true if there are more than one NUMA nodes, otherwise false.


---
### ggml\_init\_arm\_arch\_features<!-- {{#callable:ggml_init_arm_arch_features}} -->
Initializes ARM architecture features for the ggml library.
- **Inputs**: None
- **Control Flow**:
    - Checks if the platform is Linux with ARM architecture or Apple.
    - For Linux, retrieves hardware capabilities using `getauxval` and sets the corresponding feature flags.
    - For Apple, uses `sysctlbyname` to check for specific ARM features and sets the feature flags accordingly.
    - If the platform is neither, it falls back to compile-time checks to set the feature flags.
- **Output**: The function does not return a value but updates the global `ggml_arm_arch_features` structure with the detected capabilities.


---
### ggml\_new\_i32<!-- {{#callable:ggml_new_i32}} -->
Creates a new 1D tensor of type int32 and initializes it with a specified value.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory allocation for the tensor.
    - `value`: An integer value of type int32_t that will be set in the newly created tensor.
- **Control Flow**:
    - The function first asserts that the context does not have allocation disabled using `GGML_ASSERT`.
    - It then calls [`ggml_new_tensor_1d`](../ggml.c.driver.md#ggml_new_tensor_1d) to create a new 1D tensor of type `GGML_TYPE_I32` with a size of 1.
    - The value passed as an argument is set in the newly created tensor using [`ggml_set_i32`](#ggml_set_i32).
    - Finally, the function returns the pointer to the newly created tensor.
- **Output**: Returns a pointer to the newly created `ggml_tensor` initialized with the specified int32 value.
- **Functions called**:
    - [`ggml_get_no_alloc`](../ggml.c.driver.md#ggml_get_no_alloc)
    - [`ggml_new_tensor_1d`](../ggml.c.driver.md#ggml_new_tensor_1d)
    - [`ggml_set_i32`](#ggml_set_i32)


---
### ggml\_new\_f32<!-- {{#callable:ggml_new_f32}} -->
Creates a new 1D tensor of type float (f32) initialized with a specified value.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that manages memory allocation for tensors.
    - `value`: A float value that will be assigned to the newly created tensor.
- **Control Flow**:
    - The function first asserts that the context does not have allocation disabled using `GGML_ASSERT`.
    - It then calls [`ggml_new_tensor_1d`](../ggml.c.driver.md#ggml_new_tensor_1d) to create a new 1D tensor of type `GGML_TYPE_F32` with a size of 1.
    - The value passed as an argument is set to the newly created tensor using [`ggml_set_f32`](#ggml_set_f32).
    - Finally, the function returns the pointer to the newly created tensor.
- **Output**: Returns a pointer to the newly created `ggml_tensor` structure initialized with the specified float value.
- **Functions called**:
    - [`ggml_get_no_alloc`](../ggml.c.driver.md#ggml_get_no_alloc)
    - [`ggml_new_tensor_1d`](../ggml.c.driver.md#ggml_new_tensor_1d)
    - [`ggml_set_f32`](#ggml_set_f32)


---
### ggml\_set\_i32<!-- {{#callable:ggml_set_i32}} -->
Sets the values of a `ggml_tensor` to a specified 32-bit integer based on its data type.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure that holds the data to be modified.
    - `value`: An `int32_t` value that will be set in the tensor.
- **Control Flow**:
    - The function retrieves the number of rows (`n`), number of columns (`nc`), and the size of each row in bytes (`n1`) from the `tensor`.
    - It then accesses the raw data pointer of the tensor.
    - A switch statement is used to determine the tensor's data type and execute the corresponding case.
    - For each case, it asserts that the size of the data type matches the expected size, and then iterates over each row to set the value using the appropriate vector setting function.
    - If the tensor type does not match any of the expected types, it triggers an abort with a fatal error message.
- **Output**: Returns a pointer to the modified `ggml_tensor`.
- **Functions called**:
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_vec_set_i8`](vec.h.driver.md#ggml_vec_set_i8)
    - [`ggml_vec_set_i16`](vec.h.driver.md#ggml_vec_set_i16)
    - [`ggml_vec_set_i32`](vec.h.driver.md#ggml_vec_set_i32)
    - [`ggml_vec_set_f16`](vec.h.driver.md#ggml_vec_set_f16)
    - [`ggml_vec_set_bf16`](vec.h.driver.md#ggml_vec_set_bf16)
    - [`ggml_vec_set_f32`](vec.h.driver.md#ggml_vec_set_f32)


---
### ggml\_set\_f32<!-- {{#callable:ggml_set_f32}} -->
The `ggml_set_f32` function sets all elements of a `ggml_tensor` to a specified float value.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure that represents the tensor to be modified.
    - `value`: A float value that will be assigned to each element of the tensor.
- **Control Flow**:
    - The function retrieves the number of rows (`n`), the number of columns (`nc`), and the size of each row in bytes (`n1`) from the `tensor` structure.
    - It accesses the raw data pointer of the tensor.
    - A switch statement is used to determine the type of the tensor (`tensor->type`).
    - For each case (e.g., `GGML_TYPE_I8`, `GGML_TYPE_I16`, etc.), it asserts the size of the tensor's data type and iterates over each row to set the value using the appropriate helper function (e.g., [`ggml_vec_set_i8`](vec.h.driver.md#ggml_vec_set_i8), [`ggml_vec_set_f32`](vec.h.driver.md#ggml_vec_set_f32), etc.).
    - If the tensor type does not match any case, it calls `GGML_ABORT` to indicate a fatal error.
- **Output**: The function returns a pointer to the modified `ggml_tensor`.
- **Functions called**:
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_vec_set_i8`](vec.h.driver.md#ggml_vec_set_i8)
    - [`ggml_vec_set_i16`](vec.h.driver.md#ggml_vec_set_i16)
    - [`ggml_vec_set_i32`](vec.h.driver.md#ggml_vec_set_i32)
    - [`ggml_vec_set_f16`](vec.h.driver.md#ggml_vec_set_f16)
    - [`ggml_vec_set_bf16`](vec.h.driver.md#ggml_vec_set_bf16)
    - [`ggml_vec_set_f32`](vec.h.driver.md#ggml_vec_set_f32)


---
### ggml\_get\_i32\_1d<!-- {{#callable:ggml_get_i32_1d}} -->
The `ggml_get_i32_1d` function retrieves a 32-bit integer value from a 1-dimensional tensor.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor from which the integer value is to be retrieved.
    - `i`: An integer index specifying the position in the 1-dimensional tensor from which to retrieve the value.
- **Control Flow**:
    - The function first checks if the tensor is contiguous using `ggml_is_contiguous(tensor)`.
    - If the tensor is not contiguous, it unravels the index `i` into a multi-dimensional index using [`ggml_unravel_index`](../ggml.c.driver.md#ggml_unravel_index) and retrieves the value using [`ggml_get_i32_nd`](#ggml_get_i32_nd).
    - If the tensor is contiguous, it checks the type of the tensor and retrieves the value based on the type using a switch statement.
    - For each case in the switch statement, it asserts that the size of the data matches the expected size for that type and retrieves the value accordingly.
    - If the tensor type is not recognized, it calls `GGML_ABORT` to indicate a fatal error.
- **Output**: Returns the 32-bit integer value located at the specified index in the tensor, or triggers an error if the tensor type is unsupported.
- **Functions called**:
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_unravel_index`](../ggml.c.driver.md#ggml_unravel_index)
    - [`ggml_get_i32_nd`](#ggml_get_i32_nd)


---
### ggml\_set\_i32\_1d<!-- {{#callable:ggml_set_i32_1d}} -->
Sets a 32-bit integer value at a specified index in a 1D tensor.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor in which the value will be set.
    - `i`: An integer index specifying the position in the tensor where the value will be set.
    - `value`: The 32-bit integer value to be set at the specified index.
- **Control Flow**:
    - First, the function checks if the tensor is contiguous using `ggml_is_contiguous(tensor)`.
    - If the tensor is not contiguous, it unravels the index `i` into multi-dimensional indices using [`ggml_unravel_index`](../ggml.c.driver.md#ggml_unravel_index) and sets the value using [`ggml_set_i32_nd`](#ggml_set_i32_nd).
    - If the tensor is contiguous, it enters a switch statement based on the tensor's type.
    - For each case in the switch statement, it asserts the size of the data type and sets the value at the specified index in the tensor's data array.
    - If the tensor type does not match any case, it calls `GGML_ABORT` to indicate a fatal error.
- **Output**: The function does not return a value; it modifies the tensor in place.
- **Functions called**:
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_unravel_index`](../ggml.c.driver.md#ggml_unravel_index)
    - [`ggml_set_i32_nd`](#ggml_set_i32_nd)


---
### ggml\_set\_i32\_nd<!-- {{#callable:ggml_set_i32_nd}} -->
Sets a 32-bit integer value at a specified multi-dimensional index in a tensor.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor in which the value will be set.
    - `i0`: The first index for the multi-dimensional access.
    - `i1`: The second index for the multi-dimensional access.
    - `i2`: The third index for the multi-dimensional access.
    - `i3`: The fourth index for the multi-dimensional access.
    - `value`: The 32-bit integer value to be set at the specified index.
- **Control Flow**:
    - Calculates the memory address for the specified index in the tensor's data using the provided indices and the tensor's stride values.
    - Uses a switch statement to determine the tensor's data type and sets the value accordingly.
    - If the tensor type is not recognized, it calls `GGML_ABORT` to indicate a fatal error.
- **Output**: The function does not return a value; it modifies the tensor in place.


---
### ggml\_get\_f32\_1d<!-- {{#callable:ggml_get_f32_1d}} -->
The `ggml_get_f32_1d` function retrieves a 32-bit floating-point value from a specified index of a tensor.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor from which the value is to be retrieved.
    - `i`: An integer index specifying the position in the tensor from which to retrieve the value.
- **Control Flow**:
    - The function first checks if the tensor is contiguous using `ggml_is_contiguous(tensor)`.
    - If the tensor is not contiguous, it unravels the index `i` into a multi-dimensional index using [`ggml_unravel_index`](../ggml.c.driver.md#ggml_unravel_index) and retrieves the value using [`ggml_get_f32_nd`](#ggml_get_f32_nd).
    - If the tensor is contiguous, it uses a switch statement to determine the type of the tensor and retrieves the value directly from the data array based on the tensor's type.
    - For each case in the switch statement, the appropriate data type is cast, and the value at index `i` is returned.
    - If the tensor type is not recognized, the function calls `GGML_ABORT` to indicate a fatal error.
- **Output**: Returns a float value retrieved from the tensor at the specified index, converted appropriately based on the tensor's data type.
- **Functions called**:
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_unravel_index`](../ggml.c.driver.md#ggml_unravel_index)
    - [`ggml_get_f32_nd`](#ggml_get_f32_nd)


---
### ggml\_set\_f32\_1d<!-- {{#callable:ggml_set_f32_1d}} -->
Sets a float value in a 1D tensor at a specified index.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor in which the value will be set.
    - `i`: An integer index specifying the position in the tensor where the value will be set.
    - `value`: A float value that will be assigned to the specified index of the tensor.
- **Control Flow**:
    - The function first checks if the tensor is contiguous using `ggml_is_contiguous(tensor)`.
    - If the tensor is not contiguous, it unravels the index `i` into multi-dimensional indices using [`ggml_unravel_index`](../ggml.c.driver.md#ggml_unravel_index) and sets the value using [`ggml_set_f32_nd`](#ggml_set_f32_nd).
    - If the tensor is contiguous, it uses a switch statement to determine the type of the tensor and sets the value directly in the appropriate data type.
    - If the tensor type is not recognized, it calls `GGML_ABORT` to indicate a fatal error.
- **Output**: The function does not return a value; it modifies the tensor in place.
- **Functions called**:
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_unravel_index`](../ggml.c.driver.md#ggml_unravel_index)
    - [`ggml_set_f32_nd`](#ggml_set_f32_nd)


---
### ggml\_get\_f32\_nd<!-- {{#callable:ggml_get_f32_nd}} -->
The `ggml_get_f32_nd` function retrieves a 32-bit floating-point value from a multi-dimensional tensor at specified indices.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor from which the value is to be retrieved.
    - `i0`: The first index for accessing the tensor's data.
    - `i1`: The second index for accessing the tensor's data.
    - `i2`: The third index for accessing the tensor's data.
    - `i3`: The fourth index for accessing the tensor's data.
- **Control Flow**:
    - The function calculates the data pointer by offsetting the base address of the tensor's data using the provided indices and the tensor's stride values.
    - A switch statement is used to determine the type of the tensor, allowing for different data retrieval methods based on the tensor's type.
    - For each case in the switch statement, the appropriate type cast is performed to retrieve the value from the calculated data pointer.
    - If the tensor type does not match any of the expected types, the function calls `GGML_ABORT` to handle the error.
- **Output**: The function returns a float value retrieved from the tensor at the specified indices, or aborts if the tensor type is unsupported.


---
### ggml\_set\_f32\_nd<!-- {{#callable:ggml_set_f32_nd}} -->
Sets a floating-point value in a multi-dimensional tensor at specified indices.
- **Inputs**:
    - `tensor`: A pointer to a `ggml_tensor` structure representing the tensor in which the value will be set.
    - `i0`: The first index for the multi-dimensional tensor.
    - `i1`: The second index for the multi-dimensional tensor.
    - `i2`: The third index for the multi-dimensional tensor.
    - `i3`: The fourth index for the multi-dimensional tensor.
    - `value`: The floating-point value to be set in the tensor.
- **Control Flow**:
    - Calculates the memory address of the specified index in the tensor's data using the provided indices and the tensor's stride values.
    - Uses a switch statement to determine the tensor's data type and sets the value accordingly.
    - If the tensor type is not recognized, it calls `GGML_ABORT` to indicate a fatal error.
- **Output**: The function does not return a value; it modifies the tensor in place.


---
### ggml\_compute\_forward\_mul\_mat\_one\_chunk<!-- {{#callable:ggml_compute_forward_mul_mat_one_chunk}} -->
The `ggml_compute_forward_mul_mat_one_chunk` function performs matrix multiplication for a specified chunk of data.
- **Inputs**:
    - `params`: Pointer to a structure containing computation parameters.
    - `dst`: Pointer to the destination tensor where the result will be stored.
    - `type`: Enumeration value representing the data type of the tensors.
    - `num_rows_per_vec_dot`: The number of rows to process per vector dot product.
    - `ir0_start`: Starting index for the first input tensor's rows.
    - `ir0_end`: Ending index for the first input tensor's rows.
    - `ir1_start`: Starting index for the second input tensor's rows.
    - `ir1_end`: Ending index for the second input tensor's rows.
- **Control Flow**:
    - Check if the specified row ranges are valid; if not, return immediately.
    - Determine if the second source tensor is contiguous in memory.
    - Calculate the row sizes and strides based on the tensor types.
    - Iterate over the specified ranges of the input tensors in blocks.
    - For each block, compute the dot product of the corresponding rows from the first and second tensors.
    - Store the results in the destination tensor.
- **Output**: The function does not return a value; it directly modifies the destination tensor with the results of the matrix multiplication.
- **Functions called**:
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_row_size`](../ggml.c.driver.md#ggml_row_size)


---
### ggml\_compute\_forward\_mul\_mat<!-- {{#callable:ggml_compute_forward_mul_mat}} -->
Computes the forward multiplication of two matrices and stores the result in a destination tensor.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for the computation, including thread information.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the result of the matrix multiplication.
- **Control Flow**:
    - Extracts the source tensors from the destination tensor.
    - Validates tensor dimensions and types to ensure compatibility for multiplication.
    - Handles special cases for contiguous source tensors and performs matrix multiplication using optimized routines.
    - Distributes the workload across multiple threads based on the number of available threads and the size of the matrices.
    - Processes chunks of the matrices in a loop until all chunks are computed.
- **Output**: The function does not return a value; instead, it modifies the `dst` tensor in place to contain the result of the matrix multiplication.
- **Functions called**:
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`llamafile_sgemm`](llamafile/sgemm.cpp.driver.md#llamafile_sgemm)
    - [`ggml_blck_size`](../ggml.c.driver.md#ggml_blck_size)
    - [`ggml_row_size`](../ggml.c.driver.md#ggml_row_size)
    - [`atomic_store_explicit`](#atomic_store_explicit)
    - [`ggml_barrier`](#ggml_barrier)
    - [`ggml_is_numa`](#ggml_is_numa)
    - [`ggml_compute_forward_mul_mat_one_chunk`](#ggml_compute_forward_mul_mat_one_chunk)
    - [`atomic_fetch_add_explicit`](#atomic_fetch_add_explicit)


---
### ggml\_compute\_forward\_mul\_mat\_id\_one\_chunk<!-- {{#callable:ggml_compute_forward_mul_mat_id_one_chunk}} -->
The `ggml_compute_forward_mul_mat_id_one_chunk` function performs matrix multiplication using a specific mapping of rows based on provided indices.
- **Inputs**:
    - `dst`: A pointer to the destination `ggml_tensor` where the result of the multiplication will be stored.
    - `src0`: A pointer to the first source `ggml_tensor` that will be multiplied.
    - `src1`: A pointer to the second source `ggml_tensor` that will be multiplied.
    - `ids`: A pointer to a `ggml_tensor` containing the indices used to map rows during multiplication.
    - `cur_a`: An integer representing the current index of the expert being processed.
    - `ir0_start`: The starting index for the first dimension of the multiplication.
    - `ir0_end`: The ending index for the first dimension of the multiplication.
    - `ir1_start`: The starting index for the second dimension of the multiplication.
    - `ir1_end`: The ending index for the second dimension of the multiplication.
    - `src0_cur`: A pointer to the current position in the `src0` tensor data.
    - `matrix_rows`: A pointer to a structure that maps the rows of the matrix for the current expert.
    - `row_size`: The size of each row in the matrix.
    - `src1_cont`: A boolean indicating whether `src1` is a contiguous memory block.
    - `wdata`: A pointer to the workspace data used for intermediate calculations.
- **Control Flow**:
    - The function initializes local variables and retrieves the type of the first source tensor.
    - It defines block sizes for processing the multiplication in chunks.
    - It iterates over the specified range of indices for the second dimension, processing in blocks.
    - For each block, it further iterates over the first dimension, applying the matrix multiplication using a vector dot product function.
    - The results are temporarily stored in a buffer and then copied to the destination tensor.
- **Output**: The function does not return a value; instead, it populates the `dst` tensor with the result of the matrix multiplication.


---
### incr\_ptr\_aligned<!-- {{#callable:incr_ptr_aligned}} -->
The `incr_ptr_aligned` function increments a pointer to a memory location by a specified size, ensuring that the new pointer is aligned to a specified boundary.
- **Inputs**:
    - `p`: A pointer to a pointer (`void **`) that points to the current memory location to be incremented.
    - `size`: A `size_t` value representing the number of bytes to increment the pointer by.
    - `align`: A `size_t` value specifying the alignment boundary to which the pointer should be adjusted.
- **Control Flow**:
    - The function retrieves the current pointer value from `*p`.
    - It adjusts the pointer to ensure it is aligned to the specified `align` using the `GGML_PAD` macro.
    - The pointer is then incremented by the specified `size`.
    - The updated pointer is stored back in `*p`.
- **Output**: The function returns the newly aligned pointer after the increment.


---
### ggml\_compute\_forward\_mul\_mat\_id<!-- {{#callable:ggml_compute_forward_mul_mat_id}} -->
The `ggml_compute_forward_mul_mat_id` function performs a matrix multiplication operation with an identity mapping based on specified indices.
- **Inputs**:
    - `params`: A pointer to a `ggml_compute_params` structure containing parameters for the computation, including thread information and workspace.
    - `dst`: A pointer to a `ggml_tensor` structure that will hold the result of the matrix multiplication.
- **Control Flow**:
    - The function retrieves the source tensors from the destination tensor's source array.
    - It checks the types of the source tensors and ensures they are compatible for the operation.
    - It initializes workspace variables and allocates memory for intermediate results.
    - The function groups rows based on the provided indices and counts the number of rows for each group.
    - It uses a barrier to synchronize threads before processing the matrix multiplication in chunks.
    - For each chunk, it computes the matrix multiplication using the specified indices and stores the results in the destination tensor.
- **Output**: The function does not return a value; instead, it populates the `dst` tensor with the results of the matrix multiplication operation.
- **Functions called**:
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`incr_ptr_aligned`](#incr_ptr_aligned)
    - [`ggml_row_size`](../ggml.c.driver.md#ggml_row_size)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_blck_size`](../ggml.c.driver.md#ggml_blck_size)
    - [`ggml_barrier`](#ggml_barrier)
    - [`ggml_is_numa`](#ggml_is_numa)
    - [`ggml_compute_forward_mul_mat_id_one_chunk`](#ggml_compute_forward_mul_mat_id_one_chunk)
    - [`atomic_fetch_add_explicit`](#atomic_fetch_add_explicit)


---
### ggml\_compute\_forward<!-- {{#callable:ggml_compute_forward}} -->
The `ggml_compute_forward` function executes the forward computation for various tensor operations based on the operation type specified in the tensor.
- **Inputs**:
    - `params`: A pointer to a `struct ggml_compute_params` that contains parameters for the computation, including thread information.
    - `tensor`: A pointer to a `struct ggml_tensor` that specifies the operation to be performed and its associated data.
- **Control Flow**:
    - The function begins by asserting that the `params` pointer is not null.
    - It checks if the operation type of the `tensor` is `GGML_OP_NONE` or if the tensor is empty, in which case it returns immediately.
    - If the tensor requires extra computation, it calls [`ggml_cpu_extra_compute_forward`](ggml-cpu-traits.cpp.driver.md#ggml_cpu_extra_compute_forward) and returns if that function indicates to do so.
    - The function then uses a switch statement to determine the operation type of the tensor and calls the corresponding computation function for that operation.
- **Output**: The function does not return a value; instead, it performs computations in place on the `tensor` based on the specified operation.
- **Functions called**:
    - [`ggml_is_empty`](../ggml.c.driver.md#ggml_is_empty)
    - [`ggml_cpu_extra_compute_forward`](ggml-cpu-traits.cpp.driver.md#ggml_cpu_extra_compute_forward)
    - [`ggml_compute_forward_sub`](binary-ops.cpp.driver.md#ggml_compute_forward_sub)
    - [`ggml_compute_forward_mul`](binary-ops.cpp.driver.md#ggml_compute_forward_mul)
    - [`ggml_compute_forward_div`](binary-ops.cpp.driver.md#ggml_compute_forward_div)
    - [`ggml_compute_forward_sqr`](unary-ops.cpp.driver.md#ggml_compute_forward_sqr)
    - [`ggml_compute_forward_sqrt`](unary-ops.cpp.driver.md#ggml_compute_forward_sqrt)
    - [`ggml_compute_forward_log`](unary-ops.cpp.driver.md#ggml_compute_forward_log)
    - [`ggml_compute_forward_sin`](unary-ops.cpp.driver.md#ggml_compute_forward_sin)
    - [`ggml_compute_forward_cos`](unary-ops.cpp.driver.md#ggml_compute_forward_cos)
    - [`ggml_compute_forward_mul_mat`](#ggml_compute_forward_mul_mat)
    - [`ggml_compute_forward_mul_mat_id`](#ggml_compute_forward_mul_mat_id)
    - [`ggml_get_op_params_i32`](../ggml-impl.h.driver.md#ggml_get_op_params_i32)


---
### set\_numa\_thread\_affinity<!-- {{#callable:set_numa_thread_affinity}} -->
Sets the NUMA thread affinity for a specified thread.
- **Inputs**:
    - `thread_n`: An integer representing the thread number for which the NUMA affinity is to be set.
- **Control Flow**:
    - Checks if NUMA is supported on the system.
    - Determines the node number based on the NUMA strategy.
    - Allocates a CPU set for the specified NUMA node.
    - Sets the thread affinity to the CPUs associated with the determined NUMA node.
- **Output**: The function does not return a value but sets the thread's CPU affinity based on the NUMA configuration.


---
### clear\_numa\_thread\_affinity<!-- {{#callable:clear_numa_thread_affinity}} -->
The `clear_numa_thread_affinity` function clears the NUMA thread affinity settings for the current thread.
- **Inputs**: None
- **Control Flow**:
    - Checks if NUMA is supported using `ggml_is_numa()`.
    - If NUMA is supported, it allocates a CPU set and clears it.
    - Sets the affinity of the current thread to all CPUs in the system.
    - Handles any errors that occur during the setting of thread affinity.
- **Output**: The function does not return a value; it modifies the thread's CPU affinity settings.


---
### ggml\_get\_n\_tasks<!-- {{#callable:ggml_get_n_tasks}} -->
The `ggml_get_n_tasks` function determines the number of tasks for a given tensor operation based on the operation type and the number of available threads.
- **Inputs**:
    - `node`: A pointer to a `ggml_tensor` structure representing the operation to be performed.
    - `n_threads`: An integer representing the number of threads available for parallel processing.
- **Control Flow**:
    - Check if the `node` is empty using `ggml_is_empty(node)`, if true, set `n_tasks` to 1 and return.
    - Use a switch statement to determine the operation type (`node->op`) and set `n_tasks` based on the operation type and the number of threads.
    - For operations that can utilize multiple threads (like `GGML_OP_CPY`, `GGML_OP_ADD`, etc.), set `n_tasks` to `n_threads`.
    - For operations that should only use a single task (like `GGML_OP_SUB`, `GGML_OP_SQR`, etc.), set `n_tasks` to 1.
    - Handle special cases for unary operations and custom operations by checking their specific parameters.
    - Ensure that `n_tasks` is greater than 0 before returning.
- **Output**: Returns an integer representing the number of tasks that can be executed for the given tensor operation.
- **Functions called**:
    - [`ggml_is_empty`](../ggml.c.driver.md#ggml_is_empty)
    - [`ggml_get_unary_op`](../ggml.c.driver.md#ggml_get_unary_op)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_op_name`](../ggml.c.driver.md#ggml_op_name)


---
### ggml\_thread\_apply\_affinity<!-- {{#callable:ggml_thread_apply_affinity}} -->
The `ggml_thread_apply_affinity` function sets the CPU affinity for the current thread based on a provided mask.
- **Inputs**:
    - `mask`: A pointer to a boolean array representing the CPU affinity mask, where each element indicates whether a specific CPU core should be used.
- **Control Flow**:
    - The function begins by marking the `mask` parameter as unused to avoid compiler warnings.
    - It then returns true immediately without performing any operations, indicating that the function does not currently apply any affinity settings.
- **Output**: The function returns a boolean value, which is always true, indicating that the operation was successful, even though no actual affinity was applied.


---
### ggml\_thread\_apply\_priority<!-- {{#callable:ggml_thread_apply_priority}} -->
The `ggml_thread_apply_priority` function sets the thread priority for the current thread.
- **Inputs**:
    - `prio`: An integer representing the desired thread priority level.
- **Control Flow**:
    - The function begins by marking the input parameter `prio` as unused, indicating that it is not utilized within the function body.
    - The function then immediately returns `true`, indicating successful execution without any actual priority adjustment.
- **Output**: The function returns a boolean value, which is always `true` in this implementation.


---
### ggml\_thread\_cpumask\_is\_valid<!-- {{#callable:ggml_thread_cpumask_is_valid}} -->
The `ggml_thread_cpumask_is_valid` function checks if any CPU mask in a given boolean array is set to true.
- **Inputs**:
    - `mask`: A pointer to a boolean array representing the CPU mask, where each element indicates whether a corresponding CPU is enabled (true) or disabled (false).
- **Control Flow**:
    - The function iterates over the elements of the `mask` array up to `GGML_MAX_N_THREADS`.
    - If it finds any element in the `mask` array that is true, it immediately returns true.
    - If the loop completes without finding any true values, it returns false.
- **Output**: Returns a boolean value indicating whether at least one CPU in the mask is valid (true) or not (false).


---
### ggml\_thread\_cpumask\_next<!-- {{#callable:ggml_thread_cpumask_next}} -->
The `ggml_thread_cpumask_next` function updates a local CPU mask based on a global mask and an iteration index.
- **Inputs**:
    - `global_mask`: A pointer to a boolean array representing the global CPU mask, indicating which CPUs are available.
    - `local_mask`: A pointer to a boolean array where the local CPU mask will be stored.
    - `strict`: A boolean flag that determines the behavior of the function; if false, the local mask is directly copied from the global mask.
    - `iter`: A pointer to an integer that keeps track of the current iteration index for selecting CPUs.
- **Control Flow**:
    - If the `strict` flag is false, the function copies the `global_mask` to the `local_mask` and returns immediately.
    - If `strict` is true, the function initializes the `local_mask` to zero.
    - The function then iterates over the range of `GGML_MAX_N_THREADS`, calculating the index based on the current `iter` value.
    - If the calculated index exceeds `GGML_MAX_N_THREADS`, it wraps around using modulo operation.
    - If the `global_mask` at the calculated index is true, it sets the corresponding index in `local_mask` to true, updates the `iter` pointer, and returns.
- **Output**: The function does not return a value; instead, it modifies the `local_mask` in place and updates the `iter` pointer.


---
### ggml\_threadpool\_free<!-- {{#callable:ggml_threadpool_free}} -->
Frees the resources allocated for a `ggml_threadpool` structure.
- **Inputs**:
    - `threadpool`: A pointer to a `ggml_threadpool` structure that needs to be freed.
- **Control Flow**:
    - Checks if the `threadpool` pointer is NULL; if it is, the function returns immediately.
    - Locks the mutex associated with the threadpool to ensure thread safety while stopping the threadpool.
    - Sets the `stop` flag to true and the `pause` flag to false, signaling the threads to stop processing.
    - Broadcasts a condition variable to wake up any threads that may be waiting.
    - Unlocks the mutex after signaling the threads.
    - Joins all worker threads, ensuring they have completed execution.
    - Destroys the mutex and condition variable associated with the threadpool.
    - Frees the memory allocated for the worker states and the threadpool itself.
- **Output**: The function does not return a value, but it deallocates the memory and resources associated with the `ggml_threadpool`.
- **Functions called**:
    - [`ggml_aligned_free`](../ggml.c.driver.md#ggml_aligned_free)


---
### ggml\_threadpool\_pause\_locked<!-- {{#callable:ggml_threadpool_pause_locked}} -->
Pauses the execution of a thread pool.
- **Inputs**:
    - `threadpool`: A pointer to a `ggml_threadpool` structure that represents the thread pool to be paused.
- **Control Flow**:
    - Logs a debug message indicating that the thread pool is being paused.
    - Sets the `pause` flag of the `threadpool` to true, indicating that the thread pool should be paused.
    - Broadcasts a condition variable to wake up any threads that may be waiting for work, allowing them to check the new state of the `pause` flag.
- **Output**: This function does not return a value; it modifies the state of the thread pool to indicate that it is paused.


---
### ggml\_threadpool\_resume\_locked<!-- {{#callable:ggml_threadpool_resume_locked}} -->
Resumes the execution of a paused threadpool.
- **Inputs**:
    - `threadpool`: A pointer to a `ggml_threadpool` structure that represents the threadpool to be resumed.
- **Control Flow**:
    - Logs a debug message indicating that the threadpool is being resumed.
    - Sets the `pause` flag of the threadpool to false, indicating that it is no longer paused.
    - Broadcasts a condition variable to wake up any threads that are waiting for the threadpool to resume.
- **Output**: The function does not return a value; it modifies the state of the threadpool to allow it to resume processing tasks.


---
### ggml\_threadpool\_pause<!-- {{#callable:ggml_threadpool_pause}} -->
Pauses the execution of a thread pool.
- **Inputs**:
    - `threadpool`: A pointer to a `ggml_threadpool` structure that represents the thread pool to be paused.
- **Control Flow**:
    - If `GGML_USE_OPENMP` is not defined, the function locks the mutex associated with the thread pool.
    - It checks if the thread pool is already paused; if not, it calls [`ggml_threadpool_pause_locked`](#ggml_threadpool_pause_locked) to pause it.
    - Finally, it unlocks the mutex.
- **Output**: The function does not return a value; it modifies the state of the thread pool to indicate that it is paused.
- **Functions called**:
    - [`ggml_threadpool_pause_locked`](#ggml_threadpool_pause_locked)


---
### ggml\_threadpool\_resume<!-- {{#callable:ggml_threadpool_resume}} -->
Resumes a paused thread pool if it is currently paused.
- **Inputs**:
    - `threadpool`: A pointer to a `ggml_threadpool` structure that represents the thread pool to be resumed.
- **Control Flow**:
    - The function first checks if the `GGML_USE_OPENMP` preprocessor directive is not defined.
    - If `GGML_USE_OPENMP` is not defined, it locks the mutex associated with the thread pool to ensure thread safety.
    - It then checks if the thread pool is currently paused.
    - If the thread pool is paused, it calls the [`ggml_threadpool_resume_locked`](#ggml_threadpool_resume_locked) function to resume it.
    - Finally, it unlocks the mutex.
- **Output**: The function does not return a value; it modifies the state of the thread pool by resuming it if it was paused.
- **Functions called**:
    - [`ggml_threadpool_resume_locked`](#ggml_threadpool_resume_locked)


---
### ggml\_graph\_plan<!-- {{#callable:ggml_graph_plan}} -->
The `ggml_graph_plan` function prepares a computation plan for a given computation graph, estimating the required resources and scheduling tasks for execution.
- **Inputs**:
    - `cgraph`: A pointer to a `ggml_cgraph` structure representing the computation graph to be processed.
    - `n_threads`: An integer specifying the number of threads to be used for computation.
    - `threadpool`: A pointer to a `ggml_threadpool` structure that manages the threads for executing the graph.
- **Control Flow**:
    - If `threadpool` is NULL, a disposable threadpool will be created.
    - If `n_threads` is less than or equal to 0, it is set to the maximum number of threads available in the threadpool or a default value.
    - The function initializes a `ggml_cplan` structure to hold the computation plan.
    - It iterates over each node in the computation graph to determine the number of tasks and the required work size for each operation.
    - For each node, it checks the operation type and calculates the necessary work size based on the operation's characteristics.
    - The maximum number of tasks across all nodes is tracked to optimize thread usage.
    - Finally, the function populates the `ggml_cplan` structure with the determined values and returns it.
- **Output**: Returns a `ggml_cplan` structure containing the planned number of threads, work size, and a reference to the threadpool.
- **Functions called**:
    - [`ggml_get_n_tasks`](#ggml_get_n_tasks)
    - [`ggml_cpu_extra_work_size`](ggml-cpu-traits.cpp.driver.md#ggml_cpu_extra_work_size)
    - [`ggml_is_quantized`](../ggml.c.driver.md#ggml_is_quantized)
    - [`ggml_type_size`](../ggml.c.driver.md#ggml_type_size)
    - [`ggml_row_size`](../ggml.c.driver.md#ggml_row_size)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_up`](../ggml-impl.h.driver.md#ggml_up)


---
### ggml\_graph\_compute\_thread<!-- {{#callable:ggml_graph_compute_thread}} -->
The `ggml_graph_compute_thread` function processes nodes in a computational graph using a thread from a thread pool.
- **Inputs**:
    - `data`: A pointer to a `ggml_compute_state` structure that contains the state of the thread, including its index and associated thread pool.
- **Control Flow**:
    - The function retrieves the computational graph and plan from the thread pool associated with the current thread state.
    - It sets the NUMA thread affinity based on the thread index to optimize memory access.
    - A `ggml_compute_params` structure is initialized with the current thread index, the number of threads, and other relevant data.
    - A loop iterates over the nodes in the computational graph, processing each node using the [`ggml_compute_forward`](#ggml_compute_forward) function.
    - If the first thread detects an abort condition via a callback, it updates the abort status in the thread pool.
    - A barrier is used to synchronize threads after processing each node.
- **Output**: The function returns a thread return type (0) indicating successful completion of the thread's work.
- **Functions called**:
    - [`set_numa_thread_affinity`](#set_numa_thread_affinity)
    - [`atomic_load_explicit`](#atomic_load_explicit)
    - [`ggml_compute_forward`](#ggml_compute_forward)
    - [`atomic_store_explicit`](#atomic_store_explicit)
    - [`ggml_barrier`](#ggml_barrier)


---
### ggml\_graph\_compute\_thread\_ready<!-- {{#callable:ggml_graph_compute_thread_ready}} -->
Checks if a compute thread in a thread pool is ready to process new work.
- **Inputs**:
    - `state`: A pointer to a `struct ggml_compute_state` that holds the state of the compute thread, including its thread pool and last processed graph.
- **Control Flow**:
    - First, it retrieves the thread pool associated with the given `state`.
    - It checks if the thread is pending work, if the thread pool is stopped, or if it is paused; if any of these conditions are true, it returns true.
    - It then checks for a new graph by loading the current graph count atomically.
    - If the current graph differs from the last processed graph, it updates the `pending` state by calling [`ggml_graph_compute_thread_active`](#ggml_graph_compute_thread_active) and updates the `last_graph` to the new graph.
- **Output**: Returns a boolean indicating whether the thread is ready to process new work.
- **Functions called**:
    - [`atomic_load_explicit`](#atomic_load_explicit)
    - [`ggml_graph_compute_thread_active`](#ggml_graph_compute_thread_active)


---
### ggml\_graph\_compute\_thread\_sync<!-- {{#callable:ggml_graph_compute_thread_sync}} -->
The `ggml_graph_compute_thread_sync` function synchronizes threads in a compute state by using atomic operations to ensure memory visibility.
- **Inputs**:
    - `state`: A pointer to a `struct ggml_compute_state` that holds the state of the current compute thread.
- **Control Flow**:
    - Checks if thread sanitizer (TSAN) is enabled.
    - If TSAN is enabled, it performs a dummy read-modify-write operation using [`atomic_fetch_add_explicit`](#atomic_fetch_add_explicit).
    - If TSAN is not enabled, it calls [`atomic_thread_fence`](#atomic_thread_fence) to ensure memory operations are completed in the correct order.
    - The `state` parameter is marked as unused to avoid compiler warnings.
- **Output**: The function does not return a value; it performs synchronization operations to ensure proper memory visibility across threads.
- **Functions called**:
    - [`atomic_fetch_add_explicit`](#atomic_fetch_add_explicit)
    - [`atomic_thread_fence`](#atomic_thread_fence)


---
### ggml\_graph\_compute\_poll\_for\_work<!-- {{#callable:ggml_graph_compute_poll_for_work}} -->
Polls for work in a thread pool for a given compute state.
- **Inputs**:
    - `state`: A pointer to a `struct ggml_compute_state` that contains the state of the compute thread, including its thread pool and work status.
- **Control Flow**:
    - Checks if the thread is active using [`ggml_graph_compute_thread_active`](#ggml_graph_compute_thread_active).
    - If the thread is not active, it returns the current pending status.
    - Calculates the number of polling rounds based on the thread pool's polling level.
    - Enters a loop that continues polling until the thread is ready or the maximum number of rounds is reached.
    - During polling, it calls [`ggml_thread_cpu_relax`](#ggml_thread_cpu_relax) to yield the CPU.
- **Output**: Returns a boolean indicating whether there is pending work in the compute state.
- **Functions called**:
    - [`ggml_graph_compute_thread_active`](#ggml_graph_compute_thread_active)
    - [`ggml_graph_compute_thread_ready`](#ggml_graph_compute_thread_ready)
    - [`ggml_thread_cpu_relax`](#ggml_thread_cpu_relax)


---
### ggml\_graph\_compute\_check\_for\_work<!-- {{#callable:ggml_graph_compute_check_for_work}} -->
Checks if there is work available for a thread in the computation graph and manages thread synchronization.
- **Inputs**:
    - `state`: A pointer to a `struct ggml_compute_state` that holds the state of the current thread, including its thread pool and work status.
- **Control Flow**:
    - The function first checks if there is any pending work by calling [`ggml_graph_compute_poll_for_work`](#ggml_graph_compute_poll_for_work).
    - If there is pending work, it synchronizes the thread state using [`ggml_graph_compute_thread_sync`](#ggml_graph_compute_thread_sync) and returns the pending status.
    - If no work is pending, it locks the thread pool's mutex to check if the thread is ready for new work.
    - If the thread is not ready, it waits for a signal indicating new work is available.
    - Once signaled, it unlocks the mutex and returns the pending status.
- **Output**: Returns a boolean indicating whether there is pending work for the thread.
- **Functions called**:
    - [`ggml_graph_compute_poll_for_work`](#ggml_graph_compute_poll_for_work)
    - [`ggml_graph_compute_thread_sync`](#ggml_graph_compute_thread_sync)
    - [`ggml_graph_compute_thread_ready`](#ggml_graph_compute_thread_ready)


---
### ggml\_graph\_compute\_secondary\_thread<!-- {{#callable:ggml_graph_compute_secondary_thread}} -->
The `ggml_graph_compute_secondary_thread` function manages the execution of tasks in a secondary thread of a thread pool for computing graph operations.
- **Inputs**:
    - `data`: A pointer to a `ggml_compute_state` structure that contains the state information for the thread, including its index and associated thread pool.
- **Control Flow**:
    - The function begins by casting the input `data` to a `ggml_compute_state` pointer and retrieves the associated thread pool.
    - It applies the thread's priority and CPU affinity based on the state information.
    - An infinite loop is initiated to continuously check for work to process.
    - Within the loop, it first checks if the thread pool is paused, and if so, it waits for a signal to resume.
    - If the thread pool is marked to stop, the loop breaks, terminating the thread.
    - The function checks for new work using [`ggml_graph_compute_check_for_work`](#ggml_graph_compute_check_for_work), which determines if there are tasks to execute.
    - If there is pending work, it processes the task by calling [`ggml_graph_compute_thread`](#ggml_graph_compute_thread) with the current state.
- **Output**: The function returns a thread return type (thread_ret_t), which is typically 0, indicating successful completion of the thread's execution.
- **Functions called**:
    - [`ggml_thread_apply_priority`](#ggml_thread_apply_priority)
    - [`ggml_thread_cpumask_is_valid`](#ggml_thread_cpumask_is_valid)
    - [`ggml_thread_apply_affinity`](#ggml_thread_apply_affinity)
    - [`ggml_graph_compute_check_for_work`](#ggml_graph_compute_check_for_work)
    - [`ggml_graph_compute_thread`](#ggml_graph_compute_thread)


---
### ggml\_graph\_compute\_kickoff<!-- {{#callable:ggml_graph_compute_kickoff}} -->
`ggml_graph_compute_kickoff` initializes and starts the processing of a computation graph in a thread pool.
- **Inputs**:
    - `threadpool`: A pointer to a `ggml_threadpool` structure that manages the threads for computation.
    - `n_threads`: An integer representing the number of threads to be used for processing the graph.
- **Control Flow**:
    - Locks the mutex to ensure thread safety while modifying shared resources.
    - Logs the current and requested number of threads for debugging purposes.
    - Updates the current number of threads in the thread pool using atomic operations.
    - Increments the graph count to indicate that a new graph is ready for processing.
    - Checks if the thread pool is paused; if so, it applies the thread priority and CPU affinity settings, then resumes the thread pool.
    - If not paused, it broadcasts a condition to wake up any waiting threads.
    - Unlocks the mutex to allow other threads to access shared resources.
- **Output**: This function does not return a value but modifies the state of the thread pool to prepare it for processing a new computation graph.
- **Functions called**:
    - [`atomic_store_explicit`](#atomic_store_explicit)
    - [`atomic_fetch_add_explicit`](#atomic_fetch_add_explicit)
    - [`ggml_thread_apply_priority`](#ggml_thread_apply_priority)
    - [`ggml_thread_cpumask_is_valid`](#ggml_thread_cpumask_is_valid)
    - [`ggml_thread_apply_affinity`](#ggml_thread_apply_affinity)
    - [`ggml_threadpool_resume_locked`](#ggml_threadpool_resume_locked)


---
### ggml\_threadpool\_new\_impl<!-- {{#callable:ggml_threadpool_new_impl}} -->
Creates and initializes a new thread pool for executing tasks in parallel.
- **Inputs**:
    - `tpp`: A pointer to a `ggml_threadpool_params` structure containing parameters for the thread pool, such as the number of threads and whether it should be paused.
    - `cgraph`: A pointer to a `ggml_cgraph` structure representing the computation graph that the thread pool will work on.
    - `cplan`: A pointer to a `ggml_cplan` structure that contains the plan for executing the computation graph.
- **Control Flow**:
    - Allocates memory for a new `ggml_threadpool` structure using [`ggml_aligned_malloc`](../ggml.c.driver.md#ggml_aligned_malloc).
    - Initializes the thread pool's parameters, including the computation graph, current chunk, and thread states.
    - Allocates memory for the worker states based on the number of threads specified in `tpp`.
    - Initializes each worker's state, linking it to the thread pool and assigning an index.
    - If not using OpenMP, initializes mutex and condition variables for thread synchronization.
    - Creates worker threads and assigns CPU affinity based on the provided CPU mask, if applicable.
    - Sets the priority and affinity for the main thread if the thread pool is not paused.
- **Output**: Returns a pointer to the newly created `ggml_threadpool` structure.
- **Functions called**:
    - [`ggml_aligned_malloc`](../ggml.c.driver.md#ggml_aligned_malloc)
    - [`ggml_thread_cpumask_next`](#ggml_thread_cpumask_next)
    - [`ggml_thread_apply_priority`](#ggml_thread_apply_priority)
    - [`ggml_thread_cpumask_is_valid`](#ggml_thread_cpumask_is_valid)
    - [`ggml_thread_apply_affinity`](#ggml_thread_apply_affinity)


---
### ggml\_threadpool\_new<!-- {{#callable:ggml_threadpool_new}} -->
Creates a new thread pool for managing concurrent tasks.
- **Inputs**:
    - `tpp`: A pointer to a `struct ggml_threadpool_params` that contains parameters for configuring the thread pool, such as the number of threads and whether the pool should be paused.
- **Control Flow**:
    - Calls [`ggml_threadpool_new_impl`](#ggml_threadpool_new_impl) with the provided parameters and NULL for the graph and plan.
    - The [`ggml_threadpool_new_impl`](#ggml_threadpool_new_impl) function allocates memory for the thread pool and initializes its members.
    - It sets up worker threads based on the number of threads specified in the parameters.
- **Output**: Returns a pointer to the newly created `struct ggml_threadpool`.
- **Functions called**:
    - [`ggml_threadpool_new_impl`](#ggml_threadpool_new_impl)


---
### ggml\_graph\_compute<!-- {{#callable:ggml_graph_compute}} -->
The `ggml_graph_compute` function executes a computational graph using a specified thread pool.
- **Inputs**:
    - `cgraph`: A pointer to a `ggml_cgraph` structure representing the computational graph to be executed.
    - `cplan`: A pointer to a `ggml_cplan` structure containing the plan for executing the graph, including thread count and work data.
- **Control Flow**:
    - Initializes the CPU environment by calling `ggml_cpu_init()`.
    - Asserts that the `cplan` is valid and contains a positive number of threads.
    - Checks if a thread pool is provided; if not, creates a disposable thread pool.
    - If using OpenMP and more than one thread is specified, parallelizes the computation using OpenMP directives.
    - If not using OpenMP, checks the number of threads against the maximum available in the thread pool and starts the computation.
    - Calls [`ggml_graph_compute_thread`](#ggml_graph_compute_thread) for each worker thread to process the graph.
    - Clears thread affinity settings after computation.
    - Returns the status of the computation from the thread pool.
- **Output**: Returns an enumeration of type `ggml_status` indicating the success or failure of the graph computation.
- **Functions called**:
    - [`ggml_cpu_init`](#ggml_cpu_init)
    - [`ggml_threadpool_params_default`](../ggml.c.driver.md#ggml_threadpool_params_default)
    - [`ggml_threadpool_new_impl`](#ggml_threadpool_new_impl)
    - [`atomic_store_explicit`](#atomic_store_explicit)
    - [`ggml_graph_compute_thread`](#ggml_graph_compute_thread)
    - [`ggml_graph_compute_kickoff`](#ggml_graph_compute_kickoff)
    - [`clear_numa_thread_affinity`](#clear_numa_thread_affinity)
    - [`ggml_threadpool_free`](#ggml_threadpool_free)


---
### ggml\_graph\_compute\_with\_ctx<!-- {{#callable:ggml_graph_compute_with_ctx}} -->
Computes the graph of operations defined in `ggml_cgraph` using a specified context and number of threads.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure that holds the context for memory allocation and other state.
    - `cgraph`: A pointer to a `ggml_cgraph` structure that defines the computational graph to be executed.
    - `n_threads`: An integer specifying the number of threads to use for computation.
- **Control Flow**:
    - Calls [`ggml_graph_plan`](#ggml_graph_plan) to create a computation plan (`cplan`) based on the provided `cgraph` and `n_threads`.
    - Allocates a buffer for work data in `cplan` using [`ggml_new_buffer`](../ggml.c.driver.md#ggml_new_buffer).
    - Invokes [`ggml_graph_compute`](#ggml_graph_compute) with the `cgraph` and the prepared `cplan` to execute the computation.
- **Output**: Returns the status of the computation as an `enum ggml_status`, indicating success or failure.
- **Functions called**:
    - [`ggml_graph_plan`](#ggml_graph_plan)
    - [`ggml_new_buffer`](../ggml.c.driver.md#ggml_new_buffer)
    - [`ggml_graph_compute`](#ggml_graph_compute)


---
### ggml\_cpu\_fp32\_to\_fp16<!-- {{#callable:ggml_cpu_fp32_to_fp16}} -->
Converts an array of 32-bit floating-point numbers to 16-bit floating-point numbers.
- **Inputs**:
    - `x`: Pointer to an array of `float` values that need to be converted.
    - `y`: Pointer to an array of `ggml_fp16_t` where the converted values will be stored.
    - `n`: The number of elements in the input array to be converted.
- **Control Flow**:
    - Initializes an index variable `i` to 0.
    - Checks if the `__F16C__` macro is defined, indicating support for F16C instructions.
    - If `__AVX512F__` is defined, processes 16 elements at a time using AVX512 instructions.
    - If not, processes 8 elements at a time using AVX2 instructions if available.
    - If neither AVX512 nor AVX2 is available, processes 4 elements at a time using SSE instructions.
    - Finally, processes any remaining elements one at a time using a standard conversion function.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the converted 16-bit floating-point values.


---
### ggml\_cpu\_fp16\_to\_fp32<!-- {{#callable:ggml_cpu_fp16_to_fp32}} -->
Converts an array of half-precision floating-point numbers (FP16) to single-precision floating-point numbers (FP32).
- **Inputs**:
    - `x`: Pointer to an array of half-precision floating-point numbers (ggml_fp16_t) to be converted.
    - `y`: Pointer to an array where the converted single-precision floating-point numbers (float) will be stored.
    - `n`: The number of elements in the input array to convert.
- **Control Flow**:
    - The function initializes an index variable 'i' to zero.
    - If the compiler supports F16C and AVX512F, it processes 16 elements at a time using vectorized instructions.
    - If AVX512F is not available, it processes 8 elements at a time using AVX2 instructions.
    - If neither AVX512F nor AVX2 is available, it processes 4 elements at a time using SSE instructions.
    - For any remaining elements (less than 4), it processes them one at a time using a standard conversion macro.
- **Output**: The function does not return a value; instead, it populates the output array 'y' with the converted FP32 values.


---
### ggml\_cpu\_fp32\_to\_bf16<!-- {{#callable:ggml_cpu_fp32_to_bf16}} -->
Converts an array of 32-bit floating-point numbers to an array of 16-bit brain floating-point numbers.
- **Inputs**:
    - `x`: Pointer to an array of `float` values that need to be converted.
    - `y`: Pointer to an array of `ggml_bf16_t` where the converted values will be stored.
    - `n`: The number of elements in the input array to be converted.
- **Control Flow**:
    - Initializes an index variable `i` to 0.
    - Iterates over the range from 0 to `n` (exclusive).
    - For each index `i`, converts the `float` value at `x[i]` to `ggml_bf16_t` and stores it in `y[i]`.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the converted values.


---
### ggml\_cpu\_bf16\_to\_fp32<!-- {{#callable:ggml_cpu_bf16_to_fp32}} -->
Converts an array of `ggml_bf16_t` values to an array of `float` values.
- **Inputs**:
    - `x`: Pointer to an array of `ggml_bf16_t` values to be converted.
    - `y`: Pointer to an array of `float` where the converted values will be stored.
    - `n`: The number of elements in the input array.
- **Control Flow**:
    - Initializes an index variable `i` to 0.
    - Checks if the AVX2 instruction set is available.
    - If AVX512F is defined, processes 16 elements at a time using vectorized instructions.
    - If AVX2 is defined, processes 8 elements at a time using vectorized instructions.
    - For any remaining elements, processes them one by one using a scalar conversion function.
- **Output**: The function does not return a value; instead, it populates the output array `y` with the converted float values.


---
### ggml\_cpu\_has\_avx<!-- {{#callable:ggml_cpu_has_avx}} -->
The `ggml_cpu_has_avx` function checks if the CPU supports AVX (Advanced Vector Extensions) instructions.
- **Inputs**: None
- **Control Flow**:
    - The function uses preprocessor directives to check if the `__AVX__` macro is defined.
    - If `__AVX__` is defined, it returns 1, indicating that AVX is supported.
    - If `__AVX__` is not defined, it returns 0, indicating that AVX is not supported.
- **Output**: The function returns an integer: 1 if AVX is supported, and 0 if it is not.


---
### ggml\_cpu\_has\_avx\_vnni<!-- {{#callable:ggml_cpu_has_avx_vnni}} -->
The `ggml_cpu_has_avx_vnni` function checks if the AVX VNNI instruction set is available on the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function uses preprocessor directives to check if the `__AVXVNNI__` macro is defined.
    - If `__AVXVNNI__` is defined, the function returns 1, indicating that AVX VNNI is supported.
    - If `__AVXVNNI__` is not defined, the function returns 0, indicating that AVX VNNI is not supported.
- **Output**: The function returns an integer value: 1 if AVX VNNI is supported, and 0 otherwise.


---
### ggml\_cpu\_has\_avx2<!-- {{#callable:ggml_cpu_has_avx2}} -->
The `ggml_cpu_has_avx2` function checks if the CPU supports the AVX2 instruction set.
- **Inputs**: None
- **Control Flow**:
    - The function uses preprocessor directives to check if the `__AVX2__` macro is defined.
    - If `__AVX2__` is defined, the function returns 1, indicating that AVX2 is supported.
    - If `__AVX2__` is not defined, the function returns 0, indicating that AVX2 is not supported.
- **Output**: The function returns an integer: 1 if AVX2 is supported, 0 otherwise.


---
### ggml\_cpu\_has\_avx512<!-- {{#callable:ggml_cpu_has_avx512}} -->
The `ggml_cpu_has_avx512` function checks if the AVX512 instruction set is supported by the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function uses preprocessor directives to check if the `__AVX512F__` macro is defined.
    - If `__AVX512F__` is defined, the function returns 1, indicating that AVX512 is supported.
    - If `__AVX512F__` is not defined, the function returns 0, indicating that AVX512 is not supported.
- **Output**: The function returns an integer value: 1 if AVX512 is supported, and 0 if it is not.


---
### ggml\_cpu\_has\_avx512\_vbmi<!-- {{#callable:ggml_cpu_has_avx512_vbmi}} -->
The `ggml_cpu_has_avx512_vbmi` function checks if the CPU supports the AVX512 VBMI instruction set.
- **Inputs**: None
- **Control Flow**:
    - The function uses preprocessor directives to check if the `__AVX512VBMI__` macro is defined.
    - If the macro is defined, the function returns 1, indicating that AVX512 VBMI is supported.
    - If the macro is not defined, the function returns 0, indicating that AVX512 VBMI is not supported.
- **Output**: The function returns an integer value: 1 if AVX512 VBMI is supported, and 0 otherwise.


---
### ggml\_cpu\_has\_avx512\_vnni<!-- {{#callable:ggml_cpu_has_avx512_vnni}} -->
The `ggml_cpu_has_avx512_vnni` function checks if the AVX512 VNNI instruction set is supported by the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function uses preprocessor directives to check if the `__AVX512VNNI__` macro is defined.
    - If the macro is defined, it returns 1, indicating that AVX512 VNNI is supported.
    - If the macro is not defined, it returns 0, indicating that AVX512 VNNI is not supported.
- **Output**: The function returns an integer value: 1 if AVX512 VNNI is supported, and 0 otherwise.


---
### ggml\_cpu\_has\_avx512\_bf16<!-- {{#callable:ggml_cpu_has_avx512_bf16}} -->
This function checks if the CPU supports AVX512 BF16 instructions.
- **Inputs**: None
- **Control Flow**:
    - The function uses preprocessor directives to check if the `__AVX512BF16__` macro is defined.
    - If the macro is defined, the function returns 1, indicating support for AVX512 BF16.
    - If the macro is not defined, the function returns 0, indicating no support.
- **Output**: The function returns an integer: 1 if AVX512 BF16 is supported, otherwise 0.


---
### ggml\_cpu\_has\_amx\_int8<!-- {{#callable:ggml_cpu_has_amx_int8}} -->
The `ggml_cpu_has_amx_int8` function checks if the AMX (Advanced Matrix Extensions) for INT8 operations is supported by the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function uses preprocessor directives to check if the `__AMX_INT8__` macro is defined.
    - If the macro is defined, it returns 1, indicating support for AMX INT8.
    - If the macro is not defined, it returns 0, indicating no support for AMX INT8.
- **Output**: The function returns an integer: 1 if AMX INT8 is supported, otherwise 0.


---
### ggml\_cpu\_has\_bmi2<!-- {{#callable:ggml_cpu_has_bmi2}} -->
The `ggml_cpu_has_bmi2` function checks if the CPU supports the BMI2 instruction set.
- **Inputs**: None
- **Control Flow**:
    - The function uses preprocessor directives to check if the `__BMI2__` macro is defined.
    - If `__BMI2__` is defined, the function returns 1, indicating support for BMI2.
    - If `__BMI2__` is not defined, the function returns 0, indicating no support for BMI2.
- **Output**: The function returns an integer: 1 if BMI2 is supported, otherwise 0.


---
### ggml\_cpu\_has\_fma<!-- {{#callable:ggml_cpu_has_fma}} -->
The `ggml_cpu_has_fma` function checks if the Fused Multiply-Add (FMA) instruction is supported by the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function uses preprocessor directives to check if the `__FMA__` macro is defined.
    - If `__FMA__` is defined, it returns 1, indicating that FMA is supported.
    - If `__FMA__` is not defined, it returns 0, indicating that FMA is not supported.
- **Output**: The function returns an integer value: 1 if FMA is supported, and 0 if it is not.


---
### ggml\_cpu\_has\_arm\_fma<!-- {{#callable:ggml_cpu_has_arm_fma}} -->
The `ggml_cpu_has_arm_fma` function checks if the ARM architecture supports the Fused Multiply-Add (FMA) feature.
- **Inputs**: None
- **Control Flow**:
    - The function uses preprocessor directives to check if the `__ARM_FEATURE_FMA` macro is defined.
    - If the macro is defined, it returns 1, indicating that the FMA feature is supported.
    - If the macro is not defined, it returns 0, indicating that the FMA feature is not supported.
- **Output**: The function returns an integer value: 1 if FMA is supported, and 0 if it is not.


---
### ggml\_cpu\_has\_riscv\_v<!-- {{#callable:ggml_cpu_has_riscv_v}} -->
The `ggml_cpu_has_riscv_v` function checks if the RISC-V V extension is available.
- **Inputs**: None
- **Control Flow**:
    - The function uses preprocessor directives to check if the `__riscv_v_intrinsic` macro is defined.
    - If the macro is defined, the function returns 1, indicating that the RISC-V V extension is supported.
    - If the macro is not defined, the function returns 0, indicating that the RISC-V V extension is not supported.
- **Output**: The function returns an integer value: 1 if the RISC-V V extension is supported, and 0 otherwise.


---
### ggml\_cpu\_has\_f16c<!-- {{#callable:ggml_cpu_has_f16c}} -->
The `ggml_cpu_has_f16c` function checks if the F16C instruction set is available on the CPU.
- **Inputs**: None
- **Control Flow**:
    - The function uses preprocessor directives to check if the `__F16C__` macro is defined.
    - If `__F16C__` is defined, the function returns 1, indicating that the F16C instruction set is supported.
    - If `__F16C__` is not defined, the function returns 0, indicating that the F16C instruction set is not supported.
- **Output**: The function returns an integer value: 1 if the F16C instruction set is supported, and 0 otherwise.


---
### ggml\_cpu\_has\_fp16\_va<!-- {{#callable:ggml_cpu_has_fp16_va}} -->
The `ggml_cpu_has_fp16_va` function checks if the CPU supports FP16 vector arithmetic.
- **Inputs**: None
- **Control Flow**:
    - The function uses preprocessor directives to check if the macro `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` is defined.
    - If the macro is defined, the function returns 1, indicating support for FP16 vector arithmetic.
    - If the macro is not defined, the function returns 0, indicating no support.
- **Output**: The function returns an integer value: 1 if FP16 vector arithmetic is supported, otherwise 0.


---
### ggml\_cpu\_has\_wasm\_simd<!-- {{#callable:ggml_cpu_has_wasm_simd}} -->
The `ggml_cpu_has_wasm_simd` function checks if the current compilation supports WebAssembly SIMD.
- **Inputs**: None
- **Control Flow**:
    - The function uses preprocessor directives to check if the `__wasm_simd128__` macro is defined.
    - If the macro is defined, it returns 1, indicating that WebAssembly SIMD is supported.
    - If the macro is not defined, it returns 0, indicating that WebAssembly SIMD is not supported.
- **Output**: The function returns an integer: 1 if WebAssembly SIMD is supported, otherwise 0.


---
### ggml\_cpu\_has\_llamafile<!-- {{#callable:ggml_cpu_has_llamafile}} -->
The `ggml_cpu_has_llamafile` function checks if the `GGML_USE_LLAMAFILE` macro is defined and returns a corresponding integer value.
- **Inputs**: None
- **Control Flow**:
    - The function uses preprocessor directives to check if `GGML_USE_LLAMAFILE` is defined.
    - If defined, it returns 1.
    - If not defined, it returns 0.
- **Output**: The function returns an integer: 1 if `GGML_USE_LLAMAFILE` is defined, otherwise 0.


---
### ggml\_cpu\_has\_sse3<!-- {{#callable:ggml_cpu_has_sse3}} -->
The `ggml_cpu_has_sse3` function checks if the CPU supports the SSE3 instruction set.
- **Inputs**: None
- **Control Flow**:
    - The function uses preprocessor directives to check if the `__SSE3__` macro is defined.
    - If `__SSE3__` is defined, the function returns 1, indicating that SSE3 is supported.
    - If `__SSE3__` is not defined, the function returns 0, indicating that SSE3 is not supported.
- **Output**: The function returns an integer value: 1 if SSE3 is supported, and 0 if it is not.


---
### ggml\_cpu\_has\_ssse3<!-- {{#callable:ggml_cpu_has_ssse3}} -->
The `ggml_cpu_has_ssse3` function checks if the SSSE3 CPU feature is available.
- **Inputs**: None
- **Control Flow**:
    - The function uses preprocessor directives to check if the `__SSSE3__` macro is defined.
    - If `__SSSE3__` is defined, the function returns 1, indicating that SSSE3 is supported.
    - If `__SSSE3__` is not defined, the function returns 0, indicating that SSSE3 is not supported.
- **Output**: The function returns an integer: 1 if SSSE3 is supported, 0 otherwise.


---
### ggml\_cpu\_has\_vsx<!-- {{#callable:ggml_cpu_has_vsx}} -->
Determines if the CPU has VSX (Vector-Scalar Extension) support.
- **Inputs**: None
- **Control Flow**:
    - Checks if the `__POWER9_VECTOR__` macro is defined.
    - If defined, returns 1 indicating VSX support.
    - If not defined, returns 0 indicating no VSX support.
- **Output**: Returns an integer: 1 if VSX is supported, 0 otherwise.


---
### ggml\_cpu\_has\_vxe<!-- {{#callable:ggml_cpu_has_vxe}} -->
The `ggml_cpu_has_vxe` function checks if the VXE or VXE2 CPU features are enabled.
- **Inputs**: None
- **Control Flow**:
    - The function uses preprocessor directives to check if either `__VXE__` or `__VXE2__` is defined.
    - If either of these macros is defined, the function returns 1, indicating that the VXE features are available.
    - If neither macro is defined, the function returns 0, indicating that the VXE features are not available.
- **Output**: The function returns an integer value: 1 if VXE or VXE2 features are enabled, and 0 otherwise.


---
### ggml\_cpu\_has\_dotprod<!-- {{#callable:ggml_cpu_has_dotprod}} -->
The `ggml_cpu_has_dotprod` function checks if the CPU architecture supports the dot product feature.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the `__ARM_ARCH` and `__ARM_FEATURE_DOTPROD` macros are defined.
    - If both macros are defined, it returns the value of `ggml_arm_arch_features.has_dotprod`.
    - If either macro is not defined, it returns 0.
- **Output**: The function returns an integer: 1 if the dot product feature is supported, otherwise 0.


---
### ggml\_cpu\_has\_sve<!-- {{#callable:ggml_cpu_has_sve}} -->
The `ggml_cpu_has_sve` function checks if the CPU supports the Scalable Vector Extension (SVE) feature.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the `__ARM_ARCH` and `__ARM_FEATURE_SVE` macros are defined, indicating that the code is being compiled for an ARM architecture that supports SVE.
    - If both macros are defined, it returns the value of `ggml_arm_arch_features.has_sve`, which indicates whether SVE is supported on the current CPU.
    - If either macro is not defined, it returns 0, indicating that SVE is not supported.
- **Output**: The function returns an integer: 1 if SVE is supported, 0 otherwise.


---
### ggml\_cpu\_has\_matmul\_int8<!-- {{#callable:ggml_cpu_has_matmul_int8}} -->
The `ggml_cpu_has_matmul_int8` function checks if the CPU supports the ARM architecture feature for integer matrix multiplication.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the `__ARM_ARCH` and `__ARM_FEATURE_MATMUL_INT8` macros are defined.
    - If both macros are defined, it returns the value of `ggml_arm_arch_features.has_i8mm`.
    - If either macro is not defined, it returns 0.
- **Output**: The function returns an integer: 1 if the feature is supported, otherwise 0.


---
### ggml\_cpu\_get\_sve\_cnt<!-- {{#callable:ggml_cpu_get_sve_cnt}} -->
The `ggml_cpu_get_sve_cnt` function retrieves the count of Scalable Vector Extension (SVE) registers available on ARM architectures.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the code is being compiled for an ARM architecture with SVE support using preprocessor directives.
    - If SVE is supported, it returns the count of SVE registers from the `ggml_arm_arch_features` structure.
    - If SVE is not supported, it returns 0.
- **Output**: The function returns an integer representing the number of SVE registers available, or 0 if SVE is not supported.


---
### ggml\_cpu\_has\_sme<!-- {{#callable:ggml_cpu_has_sme}} -->
The `ggml_cpu_has_sme` function checks if the CPU supports Scalable Matrix Extension (SME) features.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the `__ARM_ARCH` and `__ARM_FEATURE_SME` macros are defined, indicating that the code is being compiled for an ARM architecture that supports SME.
    - If both macros are defined, it returns the value of `ggml_arm_arch_features.has_sme`, which indicates whether SME is supported.
    - If either macro is not defined, it returns 0, indicating that SME is not supported.
- **Output**: The function returns an integer: 1 if SME is supported, 0 otherwise.


---
### ggml\_cpu\_init<!-- {{#callable:ggml_cpu_init}} -->
Initializes CPU-related settings and data structures for the GGML library.
- **Inputs**: None
- **Control Flow**:
    - The function begins by initializing floating-point half (f16) tables using a context created with [`ggml_init`](../ggml.c.driver.md#ggml_init).
    - A critical section is started to ensure thread safety during initialization.
    - A static boolean variable `is_first_call` is used to check if this is the first invocation of the function.
    - If it is the first call, it initializes various mathematical tables (GELU, Quick GELU, SILU, and EXP) by iterating through a range of values.
    - The time taken for the initialization is logged for debugging purposes.
    - If the OpenMP library is used, it sets the KMP block time environment variable if it is not already set.
    - If the ARM architecture is detected, it initializes ARM-specific features.
    - Finally, the `is_first_call` variable is set to false to prevent re-initialization on subsequent calls.
    - The critical section is ended after the initialization process.
- **Output**: The function does not return a value but initializes necessary data structures and settings for subsequent operations in the GGML library.
- **Functions called**:
    - [`ggml_init`](../ggml.c.driver.md#ggml_init)
    - [`ggml_free`](../ggml.c.driver.md#ggml_free)
    - [`ggml_critical_section_start`](../ggml-threading.cpp.driver.md#ggml_critical_section_start)
    - [`ggml_gelu_f32`](vec.h.driver.md#ggml_gelu_f32)
    - [`ggml_gelu_quick_f32`](vec.h.driver.md#ggml_gelu_quick_f32)
    - [`ggml_init_arm_arch_features`](#ggml_init_arm_arch_features)
    - [`ggml_critical_section_end`](../ggml-threading.cpp.driver.md#ggml_critical_section_end)


# Function Declarations (Public API)

---
### ggml\_graph\_compute\_secondary\_thread<!-- {{#callable_declaration:ggml_graph_compute_secondary_thread}} -->
Executes tasks in a secondary thread for a thread pool.
- **Description**: This function is designed to be run in a secondary thread as part of a thread pool. It continuously checks for tasks to execute, applying thread priority and CPU affinity settings as specified. The function will pause execution if the thread pool is paused and will terminate if the thread pool is stopped. It is intended to be used in environments where tasks are dispatched by a main thread, and the secondary thread is responsible for executing these tasks when available.
- **Inputs**:
    - `data`: A pointer to a `ggml_compute_state` structure, which must not be null. This structure contains the state information for the thread, including the thread pool it belongs to. The function assumes the data is valid and correctly initialized.
- **Output**: Returns a `thread_ret_t` value, typically 0, indicating the thread has completed execution. The function does not modify the input data directly.
- **See also**: [`ggml_graph_compute_secondary_thread`](#ggml_graph_compute_secondary_thread)  (Implementation)


