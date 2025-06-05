# Purpose
This C header file is an internal component of the "ggml-backend" system, which appears to be a framework for managing and interfacing with various computational backends. The file defines a series of structures and function prototypes that facilitate the creation, management, and operation of backend buffers, devices, and streams. The primary focus is on abstracting the details of memory management and computational operations across different hardware or software backends, such as CPUs or GPUs. The file provides a comprehensive API for backend developers to implement custom backends that can handle tensor operations, manage memory buffers, and synchronize operations across different devices.

Key components of this file include the definitions of structures like `ggml_backend_buffer`, `ggml_backend`, `ggml_backend_device`, and `ggml_backend_reg`, each of which encapsulates specific functionalities related to buffer management, backend operations, device properties, and backend registration, respectively. The file also includes function pointers for operations such as buffer allocation, tensor initialization, asynchronous data access, and event synchronization, allowing for flexible and extensible backend implementations. Additionally, the file supports dynamic loading of backends, enabling the system to adapt to different hardware configurations and optimize performance by selecting the most suitable backend based on system capabilities. Overall, this header file is crucial for extending the ggml-backend framework to support a wide range of computational environments.
# Imports and Dependencies

---
- `ggml-backend.h`


# Data Structures

---
### ggml\_backend\_buffer\_type\_i
- **Type**: `struct`
- **Members**:
    - `get_name`: A function pointer that returns the name of the buffer type.
    - `alloc_buffer`: A function pointer that allocates a buffer of the specified type and size.
    - `get_alignment`: A function pointer that returns the alignment requirement for the buffer type.
    - `get_max_size`: An optional function pointer that returns the maximum buffer size that can be allocated.
    - `get_alloc_size`: An optional function pointer that returns the data size needed to allocate the tensor, including padding.
    - `is_host`: An optional function pointer that checks if the tensor data is in host memory and uses the standard layout.
- **Description**: The `ggml_backend_buffer_type_i` structure defines an interface for backend buffer types in the GGML library. It includes function pointers for operations such as retrieving the buffer type's name, allocating buffers, determining alignment requirements, and optionally checking maximum buffer sizes, allocation sizes, and whether the data is in host memory. This structure allows for flexible and customizable buffer management in different backend implementations.


---
### ggml\_backend\_buffer\_type
- **Type**: `struct`
- **Members**:
    - `iface`: An instance of `ggml_backend_buffer_type_i` that provides function pointers for buffer operations.
    - `device`: A `ggml_backend_dev_t` type representing the device associated with this buffer type.
    - `context`: A pointer to a context, which can be used to store additional data or state information.
- **Description**: The `ggml_backend_buffer_type` structure is designed to encapsulate the properties and operations associated with a specific type of backend buffer in the GGML framework. It includes an interface (`iface`) that provides function pointers for various buffer operations, such as allocation and alignment, a device identifier (`device`) that specifies the hardware device associated with the buffer, and a generic context pointer (`context`) for storing additional state or configuration data. This structure is essential for managing different types of buffers that may be used in various backend implementations, allowing for flexible and efficient memory management across different hardware platforms.


---
### ggml\_backend\_buffer\_i
- **Type**: `struct`
- **Members**:
    - `free_buffer`: A function pointer to optionally free the buffer.
    - `get_base`: A function pointer to retrieve the base address of the buffer.
    - `init_tensor`: A function pointer to optionally initialize a tensor in the buffer.
    - `memset_tensor`: A function pointer to set a range of tensor data to a specific value.
    - `set_tensor`: A function pointer to set tensor data from a source to a destination within the buffer.
    - `get_tensor`: A function pointer to retrieve tensor data from the buffer.
    - `cpy_tensor`: A function pointer to optionally copy tensor data between buffers.
    - `clear`: A function pointer to clear the entire buffer with a specified value.
    - `reset`: A function pointer to optionally reset any internal state due to tensor initialization.
- **Description**: The `ggml_backend_buffer_i` structure defines an interface for managing backend buffers in a generalized machine learning library. It provides function pointers for various operations on buffers, such as freeing, initializing, setting, getting, and copying tensor data, as well as clearing and resetting the buffer. This structure allows for flexible and customizable buffer management, supporting different backend implementations and optimizations.


---
### ggml\_backend\_buffer
- **Type**: `struct`
- **Members**:
    - `iface`: A structure containing function pointers for buffer operations.
    - `buft`: Specifies the type of the backend buffer.
    - `context`: A pointer to a context-specific data structure.
    - `size`: The size of the buffer in bytes.
    - `usage`: Indicates the usage pattern of the buffer.
- **Description**: The `ggml_backend_buffer` structure is designed to represent a buffer in the GGML backend system, encapsulating both the buffer's type and its operational interface. It includes a function interface for buffer operations, a type identifier for the buffer, a context pointer for additional data, the size of the buffer, and an enumeration to specify how the buffer is used. This structure is integral to managing memory and operations within the GGML backend, allowing for flexible and efficient handling of tensor data.


---
### ggml\_backend\_i
- **Type**: `struct`
- **Members**:
    - `get_name`: A function pointer to retrieve the name of the backend.
    - `free`: A function pointer to free the resources associated with the backend.
    - `set_tensor_async`: A function pointer for asynchronously setting tensor data.
    - `get_tensor_async`: A function pointer for asynchronously getting tensor data.
    - `cpy_tensor_async`: A function pointer for asynchronously copying tensor data between backends.
    - `synchronize`: A function pointer to synchronize and complete all pending operations.
    - `graph_plan_create`: A function pointer to create a computation graph plan.
    - `graph_plan_free`: A function pointer to free a computation graph plan.
    - `graph_plan_update`: A function pointer to update an existing computation graph plan.
    - `graph_plan_compute`: A function pointer to compute a graph using a plan.
    - `graph_compute`: A function pointer to compute a graph, always asynchronously if supported.
    - `event_record`: A function pointer to record an event on the backend stream.
    - `event_wait`: A function pointer to wait for an event on a different stream.
- **Description**: The `ggml_backend_i` structure defines an interface for a backend in the ggml library, providing function pointers for various operations such as tensor data access, graph computation, and event synchronization. It supports both synchronous and asynchronous operations, allowing for flexible backend implementations that can handle complex computation graphs and manage resources efficiently. The structure includes optional functions for creating and managing computation graph plans, as well as event recording and waiting, which are useful for advanced backend implementations that require precise control over execution order and resource management.


---
### ggml\_backend
- **Type**: `struct`
- **Members**:
    - `guid`: A unique identifier for the backend.
    - `iface`: An interface structure containing function pointers for backend operations.
    - `device`: Represents the device associated with the backend.
    - `context`: A pointer to a context-specific data structure for the backend.
- **Description**: The `ggml_backend` structure is a core component of the GGML backend system, encapsulating the necessary information and interfaces for backend operations. It includes a globally unique identifier (`guid`) to distinguish different backends, an interface (`iface`) that provides function pointers for various backend operations, a device (`device`) that specifies the hardware or software device the backend operates on, and a context pointer (`context`) for storing backend-specific data. This structure is essential for managing and interacting with different backend implementations in the GGML framework.


---
### ggml\_backend\_event
- **Type**: `struct`
- **Members**:
    - `device`: A pointer to a ggml_backend_device structure, representing the device associated with the event.
    - `context`: A void pointer to user-defined data or context associated with the event.
- **Description**: The `ggml_backend_event` structure is used to represent an event in the ggml backend system, typically for synchronization purposes. It contains a pointer to a backend device and a context pointer, allowing it to be associated with specific devices and user-defined data. This structure is part of the backend's event synchronization mechanism, enabling operations like recording and waiting for events across different streams or devices.


---
### ggml\_backend\_device\_i
- **Type**: `struct`
- **Members**:
    - `get_name`: Function pointer to retrieve the device's name.
    - `get_description`: Function pointer to retrieve a short description of the device.
    - `get_memory`: Function pointer to retrieve the device's memory information in bytes.
    - `get_type`: Function pointer to retrieve the device's type.
    - `get_props`: Function pointer to retrieve the device's properties.
    - `init_backend`: Function pointer to initialize the backend stream for the device.
    - `get_buffer_type`: Function pointer to retrieve the preferred buffer type for the device.
    - `get_host_buffer_type`: Optional function pointer to retrieve the host buffer type, typically for faster transfers.
    - `buffer_from_host_ptr`: Optional function pointer to create a buffer from a host pointer.
    - `supports_op`: Function pointer to check if the backend can compute a given operation.
    - `supports_buft`: Function pointer to check if the backend can use tensors allocated in a specific buffer type.
    - `offload_op`: Optional function pointer to check if the backend wants to run an operation despite incompatible buffer allocation.
    - `event_new`: Optional function pointer to create a new event for synchronization.
    - `event_free`: Optional function pointer to free an event.
    - `event_synchronize`: Optional function pointer to synchronize an event.
- **Description**: The `ggml_backend_device_i` structure defines an interface for interacting with backend devices in the GGML framework. It provides function pointers for obtaining device-specific information such as name, description, memory, and type, as well as for initializing backend streams and managing buffer types. Additionally, it includes optional functions for handling host buffers, checking operation support, and managing event synchronization, allowing for flexible and efficient device management and operation execution.


---
### ggml\_backend\_device
- **Type**: `struct`
- **Members**:
    - `iface`: A structure containing function pointers for device-specific operations.
    - `reg`: A registration type for the backend device.
    - `context`: A pointer to a context-specific data structure for the device.
- **Description**: The `ggml_backend_device` structure represents a backend device in the GGML framework, encapsulating device-specific operations, registration information, and context data. It is designed to interface with various backend devices, providing a flexible and extensible way to manage device-specific functionalities and properties.


---
### ggml\_backend\_reg\_i
- **Type**: `struct`
- **Members**:
    - `get_name`: A function pointer that returns the name of the backend registry.
    - `get_device_count`: A function pointer that returns the number of available devices in the backend registry.
    - `get_device`: A function pointer that retrieves a specific device from the backend registry by index.
    - `get_proc_address`: An optional function pointer that retrieves a pointer to a custom function in the backend.
- **Description**: The `ggml_backend_reg_i` structure defines an interface for interacting with a backend registry in the ggml-backend system. It provides function pointers for obtaining the name of the registry, enumerating available devices, retrieving specific devices, and optionally accessing custom functions that are not part of the standard interface. This structure is essential for managing and interfacing with different backend devices and their capabilities within the ggml-backend framework.


---
### ggml\_backend\_reg
- **Type**: `struct`
- **Members**:
    - `api_version`: An integer representing the API version, initialized to GGML_BACKEND_API_VERSION.
    - `iface`: A structure of type ggml_backend_reg_i that provides interface functions for the backend registry.
    - `context`: A pointer to a context, which can be used to store additional data or state information related to the backend registry.
- **Description**: The `ggml_backend_reg` structure is a part of the backend registry system in the GGML library, designed to manage and interface with different backend implementations. It contains an API version to ensure compatibility, an interface structure (`ggml_backend_reg_i`) that provides function pointers for backend operations, and a context pointer for storing additional backend-specific data. This structure is essential for registering and managing backends, allowing for dynamic loading and interaction with various backend devices and functionalities.


# Function Declarations (Public API)

---
### ggml\_backend\_buffer\_copy\_tensor<!-- {{#callable_declaration:ggml_backend_buffer_copy_tensor}} -->
Copies data from a source tensor to a destination tensor using the backend buffer.
- **Description**: This function attempts to copy data from a source tensor to a destination tensor using the backend buffer associated with the destination tensor. It should be used when you need to transfer tensor data between different buffers, potentially across different backends. The function requires that the destination tensor's buffer supports the copy operation; otherwise, it will return false. This function is intended for internal use and should not be called directly; instead, use the higher-level `ggml_backend_tensor_copy` function.
- **Inputs**:
    - `src`: A pointer to the source tensor from which data will be copied. Must not be null and should be a valid tensor.
    - `dst`: A pointer to the destination tensor where data will be copied to. Must not be null and should be a valid tensor. The buffer associated with this tensor must support the copy operation.
- **Output**: Returns true if the copy operation is successful, otherwise returns false if the destination buffer does not support the operation.
- **See also**: [`ggml_backend_buffer_copy_tensor`](ggml-backend.cpp.driver.md#ggml_backend_buffer_copy_tensor)  (Implementation)


---
### ggml\_backend\_buffer\_is\_multi\_buffer<!-- {{#callable_declaration:ggml_backend_buffer_is_multi_buffer}} -->
Checks if a buffer is a multi-buffer.
- **Description**: Use this function to determine if a given backend buffer is a multi-buffer, which is a buffer that contains a collection of other buffers. This function is useful when you need to handle buffers differently based on whether they are single or multi-buffer types. Ensure that the buffer passed is valid and properly initialized before calling this function.
- **Inputs**:
    - `buffer`: A `ggml_backend_buffer_t` representing the buffer to be checked. It must be a valid, initialized buffer. Passing an invalid or uninitialized buffer may lead to undefined behavior.
- **Output**: Returns `true` if the buffer is a multi-buffer, otherwise returns `false`.
- **See also**: [`ggml_backend_buffer_is_multi_buffer`](ggml-backend.cpp.driver.md#ggml_backend_buffer_is_multi_buffer)  (Implementation)


---
### ggml\_backend\_multi\_buffer\_set\_usage<!-- {{#callable_declaration:ggml_backend_multi_buffer_set_usage}} -->
Sets the usage type for all buffers within a multi-buffer.
- **Description**: This function is used to set the usage type for each buffer contained within a multi-buffer. It should be called when you need to update the usage type of all buffers in a multi-buffer to a specific usage type. The function assumes that the provided buffer is a multi-buffer, and it will assert if this is not the case. Therefore, it is important to ensure that the buffer is indeed a multi-buffer before calling this function.
- **Inputs**:
    - `buffer`: A multi-buffer whose individual buffers' usage types are to be set. Must be a valid multi-buffer; otherwise, the function will assert.
    - `usage`: The usage type to set for each buffer within the multi-buffer. This should be a valid value of the enum ggml_backend_buffer_usage.
- **Output**: None
- **See also**: [`ggml_backend_multi_buffer_set_usage`](ggml-backend.cpp.driver.md#ggml_backend_multi_buffer_set_usage)  (Implementation)


---
### ggml\_backend\_register<!-- {{#callable_declaration:ggml_backend_register}} -->
Registers a backend with the internal backend registry.
- **Description**: This function is used to register a backend with the internal backend registry, allowing it to be recognized and utilized by the system. It should be called with a valid backend registration structure that conforms to the expected interface. This function is typically used during the initialization phase of a backend to ensure it is properly integrated into the system. The backend registration structure must be correctly initialized before calling this function.
- **Inputs**:
    - `reg`: A backend registration structure of type `ggml_backend_reg_t`. This structure must be properly initialized and conform to the expected interface. The caller retains ownership of this structure, and it must not be null.
- **Output**: None
- **See also**: [`ggml_backend_register`](ggml-backend-reg.cpp.driver.md#ggml_backend_register)  (Implementation)


