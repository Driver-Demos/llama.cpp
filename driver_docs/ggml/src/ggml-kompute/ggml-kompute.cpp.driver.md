# Purpose
This C++ source file is part of a software library that provides a backend for executing machine learning operations using Vulkan, a graphics and compute API. The file integrates with the Kompute library, which is a high-level Vulkan compute framework, to perform tensor operations on GPUs. The code is structured to manage Vulkan resources, such as buffers and memory, and to execute shader operations for various tensor computations.

The file includes a series of Vulkan shader operations, such as addition, multiplication, scaling, and activation functions like ReLU and GELU, which are compiled into SPIR-V format and executed on the GPU. It defines a `kompute_manager` class to manage Vulkan resources and a [`ggml_kompute_context`](#ggml_kompute_contextggml_kompute_context) structure to maintain the state of the Vulkan context. The file also implements functions to allocate and manage Vulkan memory, check device features, and execute tensor operations using Kompute sequences. Additionally, it provides a backend interface for integration with a larger machine learning framework, allowing the framework to offload computations to the GPU using this Vulkan-based backend.
# Imports and Dependencies

---
- `ggml-impl.h`
- `ggml-backend.h`
- `ggml-backend-impl.h`
- `ggml-kompute.h`
- `shaderop_scale.h`
- `shaderop_scale_8.h`
- `shaderop_add.h`
- `shaderop_addrow.h`
- `shaderop_mul.h`
- `shaderop_silu.h`
- `shaderop_relu.h`
- `shaderop_gelu.h`
- `shaderop_softmax.h`
- `shaderop_norm.h`
- `shaderop_rmsnorm.h`
- `shaderop_diagmask.h`
- `shaderop_mul_mat_f16.h`
- `shaderop_mul_mat_q8_0.h`
- `shaderop_mul_mat_q4_0.h`
- `shaderop_mul_mat_q4_1.h`
- `shaderop_mul_mat_q4_k.h`
- `shaderop_mul_mat_q6_k.h`
- `shaderop_mul_mat_mat_f32.h`
- `shaderop_getrows_f32.h`
- `shaderop_getrows_f16.h`
- `shaderop_getrows_q4_0.h`
- `shaderop_getrows_q4_1.h`
- `shaderop_getrows_q6_k.h`
- `shaderop_rope_norm_f16.h`
- `shaderop_rope_norm_f32.h`
- `shaderop_rope_neox_f16.h`
- `shaderop_rope_neox_f32.h`
- `shaderop_cpy_f16_f16.h`
- `shaderop_cpy_f16_f32.h`
- `shaderop_cpy_f32_f16.h`
- `shaderop_cpy_f32_f32.h`
- `algorithm`
- `array`
- `cassert`
- `cstdint`
- `cstdio`
- `cstring`
- `iostream`
- `memory`
- `mutex`
- `stdexcept`
- `string`
- `unordered_map`
- `utility`
- `vector`
- `kompute/Kompute.hpp`
- `vulkan/vulkan.hpp`
- `cstdlib`


# Global Variables

---
### s\_kompute\_context
- **Type**: `ggml_kompute_context*`
- **Description**: The `s_kompute_context` is a static global pointer to a `ggml_kompute_context` structure, which is initialized to `nullptr`. This context structure holds information about the device, its name, and a shared pointer to a Vulkan descriptor pool.
- **Use**: This variable is used to manage and store the state of the Kompute context, which is essential for Vulkan operations and device management in the application.


---
### komputeManager
- **Type**: `kompute_manager`
- **Description**: The `komputeManager` is a static instance of the `kompute_manager` class, which is responsible for managing a `kp::Manager` instance. This manager is used to handle Vulkan-based operations, particularly for device discovery and management of computational tasks using the Kompute library.
- **Use**: This variable is used to manage and provide access to a `kp::Manager` instance, ensuring that Vulkan operations are properly initialized and managed.


---
### ggml\_backend\_kompute\_buffer\_type\_get\_name
- **Type**: `const char *`
- **Description**: The `ggml_backend_kompute_buffer_type_get_name` is a static function that returns the name of a buffer type in the Kompute backend. It takes a `ggml_backend_buffer_type_t` as an argument and retrieves the name from the context associated with the buffer type.
- **Use**: This function is used to obtain the name of a specific buffer type within the Kompute backend, which is useful for identifying and managing different buffer types.


# Data Structures

---
### ggml\_kompute\_context<!-- {{#data_structure:ggml_kompute_context}} -->
- **Type**: `struct`
- **Members**:
    - `device`: An integer representing the device identifier.
    - `name`: A string representing the formatted name of the Kompute context based on the device.
    - `pool`: A shared pointer to a Vulkan DescriptorPool, used for managing descriptor sets.
- **Description**: The `ggml_kompute_context` struct is a data structure used to manage the context for Kompute operations in a Vulkan environment. It holds information about the device being used, a formatted name for the context, and a descriptor pool for managing Vulkan descriptor sets. This context is essential for setting up and managing the resources needed for executing compute operations on a specified device using the Kompute library.
- **Member Functions**:
    - [`ggml_kompute_context::ggml_kompute_context`](#ggml_kompute_contextggml_kompute_context)

**Methods**

---
#### ggml\_kompute\_context::ggml\_kompute\_context<!-- {{#callable:ggml_kompute_context::ggml_kompute_context}} -->
The `ggml_kompute_context` constructor initializes a context for Kompute operations by setting the device and generating a formatted name for the context.
- **Inputs**:
    - `device`: An integer representing the device ID for which the Kompute context is being initialized.
- **Control Flow**:
    - The constructor initializes the `device` member variable with the provided `device` argument.
    - It calls the [`ggml_kompute_format_name`](#ggml_kompute_format_name) function with the `device` argument to generate a formatted name and assigns it to the `name` member variable.
- **Output**: A `ggml_kompute_context` object with initialized `device` and `name` attributes.
- **Functions called**:
    - [`ggml_kompute_format_name`](#ggml_kompute_format_name)
- **See also**: [`ggml_kompute_context`](#ggml_kompute_context)  (Data Structure)



---
### kompute\_manager<!-- {{#data_structure:kompute_manager}} -->
- **Type**: `class`
- **Members**:
    - `s_mgr`: A pointer to a kp::Manager object, initialized to nullptr.
- **Description**: The `kompute_manager` class is a simple manager for handling a `kp::Manager` instance. It provides an overloaded function call operator to ensure that a `kp::Manager` instance is created if it does not already exist, and a `destroy` method to delete the instance and reset the pointer to nullptr. This class is designed to manage the lifecycle of a `kp::Manager` object, ensuring that it is properly initialized and destroyed as needed.
- **Member Functions**:
    - [`kompute_manager::operator()`](#kompute_manageroperator())
    - [`kompute_manager::destroy`](#kompute_managerdestroy)

**Methods**

---
#### kompute\_manager::operator\(\)<!-- {{#callable:kompute_manager::operator()}} -->
The `operator()` function in the `kompute_manager` class ensures a singleton instance of `kp::Manager` is created and returned, destroying any existing instance if it is not valid.
- **Inputs**: None
- **Control Flow**:
    - Check if `s_mgr` is not null and does not have a valid instance using `hasInstance()`, then call `destroy()` to delete it.
    - If `s_mgr` is null, create a new instance of `kp::Manager` and assign it to `s_mgr`.
    - Return the `s_mgr` pointer.
- **Output**: A pointer to a `kp::Manager` instance.
- **Functions called**:
    - [`kompute_manager::destroy`](#kompute_managerdestroy)
- **See also**: [`kompute_manager`](#kompute_manager)  (Data Structure)


---
#### kompute\_manager::destroy<!-- {{#callable:kompute_manager::destroy}} -->
The `destroy` function deallocates the memory used by the `s_mgr` pointer and sets it to `nullptr`.
- **Inputs**: None
- **Control Flow**:
    - The function calls `delete` on the `s_mgr` pointer to deallocate the memory it points to.
    - It then sets `s_mgr` to `nullptr` to indicate that it no longer points to a valid object.
- **Output**: The function does not return any value.
- **See also**: [`kompute_manager`](#kompute_manager)  (Data Structure)



---
### ggml\_vk\_memory<!-- {{#data_structure:ggml_vk_memory}} -->
- **Type**: `struct`
- **Members**:
    - `data`: A pointer to the memory data, initialized to nullptr.
    - `size`: The size of the memory in bytes, initialized to 0.
    - `primaryMemory`: A pointer to the primary Vulkan device memory, initialized to nullptr.
    - `primaryBuffer`: A pointer to the primary Vulkan buffer, initialized to nullptr.
    - `stagingMemory`: A pointer to the staging Vulkan device memory, initialized to nullptr.
    - `stagingBuffer`: A pointer to the staging Vulkan buffer, initialized to nullptr.
- **Description**: The `ggml_vk_memory` struct is designed to manage Vulkan memory resources, including both primary and staging memory and buffers. It provides a structure to hold pointers to Vulkan device memory and buffer objects, as well as a pointer to the actual data and the size of the memory. This struct is essential for handling memory allocation and management in Vulkan-based applications, ensuring that both device-local and host-visible memory can be accessed and manipulated efficiently.


---
### PushConstants<!-- {{#data_structure:ggml_vk_cpy::PushConstants}} -->
- **Type**: `struct`
- **Members**:
    - `inOff`: Represents the input offset as a 32-bit unsigned integer.
    - `outOff`: Represents the output offset as a 32-bit unsigned integer.
    - `ne00`: Represents a 32-bit signed integer, possibly used for dimensions or counts.
    - `ne01`: Represents a 32-bit signed integer, possibly used for dimensions or counts.
    - `ne02`: Represents a 32-bit signed integer, possibly used for dimensions or counts.
    - `nb00`: Represents a 32-bit unsigned integer, possibly used for buffer sizes or counts.
    - `nb01`: Represents a 32-bit unsigned integer, possibly used for buffer sizes or counts.
    - `nb02`: Represents a 32-bit unsigned integer, possibly used for buffer sizes or counts.
    - `nb03`: Represents a 32-bit unsigned integer, possibly used for buffer sizes or counts.
    - `ne0`: Represents a 32-bit signed integer, possibly used for dimensions or counts.
    - `ne1`: Represents a 32-bit signed integer, possibly used for dimensions or counts.
    - `ne2`: Represents a 32-bit signed integer, possibly used for dimensions or counts.
    - `nb0`: Represents a 32-bit unsigned integer, possibly used for buffer sizes or counts.
    - `nb1`: Represents a 32-bit unsigned integer, possibly used for buffer sizes or counts.
    - `nb2`: Represents a 32-bit unsigned integer, possibly used for buffer sizes or counts.
    - `nb3`: Represents a 32-bit unsigned integer, possibly used for buffer sizes or counts.
- **Description**: The `PushConstants` struct is a data structure used to store various offsets and dimensions, both signed and unsigned, which are likely used in shader operations or GPU computations. It contains a mix of 32-bit signed and unsigned integers, which are typically used to define offsets, dimensions, or buffer sizes for operations involving GPU resources. This struct is initialized with specific values using the `safe_divide` function, indicating its role in managing data alignment or partitioning in GPU tasks.


---
### ggml\_backend\_kompute\_buffer\_type\_context<!-- {{#data_structure:ggml_backend_kompute_buffer_type_context}} -->
- **Type**: `struct`
- **Members**:
    - `device`: An integer representing the device identifier.
    - `device_ref`: An integer initialized to 0, used to track device references.
    - `buffer_alignment`: A 64-bit unsigned integer specifying the buffer alignment.
    - `max_alloc`: A 64-bit unsigned integer indicating the maximum allocation size.
    - `name`: A string representing the name of the buffer type context.
- **Description**: The `ggml_backend_kompute_buffer_type_context` struct is designed to manage and store context information for a buffer type in the Kompute backend. It includes details about the device, such as its identifier and reference count, as well as buffer-specific properties like alignment and maximum allocation size. The struct also holds a name for the context, which is generated based on the device identifier. This context is crucial for managing Vulkan resources and ensuring proper device initialization and memory management in the Kompute backend.
- **Member Functions**:
    - [`ggml_backend_kompute_buffer_type_context::ggml_backend_kompute_buffer_type_context`](#ggml_backend_kompute_buffer_type_contextggml_backend_kompute_buffer_type_context)

**Methods**

---
#### ggml\_backend\_kompute\_buffer\_type\_context::ggml\_backend\_kompute\_buffer\_type\_context<!-- {{#callable:ggml_backend_kompute_buffer_type_context::ggml_backend_kompute_buffer_type_context}} -->
The `ggml_backend_kompute_buffer_type_context` constructor initializes a context for a Kompute buffer type with specified device, buffer alignment, and maximum allocation size.
- **Inputs**:
    - `device`: An integer representing the device ID for which the buffer type context is being created.
    - `buffer_alignment`: A 64-bit unsigned integer specifying the alignment requirement for buffers.
    - `max_alloc`: A 64-bit unsigned integer indicating the maximum allocation size for buffers.
- **Control Flow**:
    - The constructor initializes the `device` member with the provided `device` argument.
    - The `buffer_alignment` member is set to the provided `buffer_alignment` argument.
    - The `max_alloc` member is initialized with the `max_alloc` argument.
    - The `name` member is initialized by calling `ggml_kompute_format_name(device)` to generate a formatted name string based on the device ID.
- **Output**: The constructor does not return a value; it initializes the members of the `ggml_backend_kompute_buffer_type_context` structure.
- **Functions called**:
    - [`ggml_kompute_format_name`](#ggml_kompute_format_name)
- **See also**: [`ggml_backend_kompute_buffer_type_context`](#ggml_backend_kompute_buffer_type_context)  (Data Structure)



---
### ggml\_backend\_kompute\_device\_context<!-- {{#data_structure:ggml_backend_kompute_device_context}} -->
- **Type**: `struct`
- **Members**:
    - `device`: An integer representing the device identifier.
    - `name`: A string representing the name of the device context.
    - `description`: A string providing a description of the device context.
- **Description**: The `ggml_backend_kompute_device_context` struct is used to encapsulate information about a specific device in the Kompute backend. It includes an integer `device` to identify the device, a `name` string to label the device context, and a `description` string to provide additional details about the device context. This struct is likely used to manage and reference device-specific resources and configurations within the Kompute backend.


# Functions

---
### ggml\_kompute\_format\_name<!-- {{#callable:ggml_kompute_format_name}} -->
The `ggml_kompute_format_name` function generates a formatted string representing the Kompute device name.
- **Inputs**:
    - `device`: An integer representing the device identifier.
- **Control Flow**:
    - The function concatenates the string 'Kompute' with the string representation of the input integer `device`.
    - It uses `std::to_string` to convert the integer to a string.
- **Output**: Returns a `std::string` that combines 'Kompute' with the device identifier, e.g., 'Kompute0', 'Kompute1', etc.


---
### enable\_sam<!-- {{#callable:enable_sam}} -->
The `enable_sam` function sets an environment variable to enable the SAM performance test for the RADV Vulkan driver.
- **Inputs**:
    - `None`: This function does not take any input parameters.
- **Control Flow**:
    - The function is marked with the `constructor` attribute, which means it will be executed automatically before the main function is called.
    - It calls the `setenv` function to set the environment variable `RADV_PERFTEST` to the value `sam`, with the overwrite flag set to false.
- **Output**: This function does not return any value; it modifies the environment variable for the process.


---
### ggml\_vk\_checkPhysicalDeviceFeatures<!-- {{#callable:ggml_vk_checkPhysicalDeviceFeatures}} -->
Checks if a Vulkan physical device supports specific features required for operations.
- **Inputs**:
    - `physical_device`: A `vk::PhysicalDevice` object representing the Vulkan physical device to check.
- **Control Flow**:
    - Retrieve the available features of the `physical_device` using `getFeatures`.
    - Check if the `shaderInt16` feature is supported; if not, return false.
    - Retrieve Vulkan 1.1 and 1.2 features using `getFeatures2`.
    - Check if the required 16-bit and 8-bit storage buffer access features are supported; if any are missing, return false.
    - If all checks pass, return true.
- **Output**: Returns a boolean indicating whether the physical device supports the required features.


---
### ggml\_vk\_getVendorName<!-- {{#callable:ggml_vk_getVendorName}} -->
The `ggml_vk_getVendorName` function returns the vendor name associated with a given Vulkan vendor ID.
- **Inputs**:
    - `vendorID`: A 32-bit unsigned integer representing the Vulkan vendor ID.
- **Control Flow**:
    - The function uses a `switch` statement to match the `vendorID` against known vendor IDs.
    - If a match is found, the corresponding vendor name is returned as a string.
    - If no match is found, the function returns the string 'unknown'.
- **Output**: The function outputs a pointer to a string containing the vendor name, or 'unknown' if the vendor ID is not recognized.


---
### ggml\_vk\_available\_devices\_internal<!-- {{#callable:ggml_vk_available_devices_internal}} -->
The `ggml_vk_available_devices_internal` function retrieves a list of Vulkan devices that meet specified memory requirements.
- **Inputs**:
    - `memoryRequired`: A size_t value representing the minimum amount of device local memory required for the Vulkan devices.
- **Control Flow**:
    - Check if Vulkan is available and if an instance exists; if not, return an empty results vector.
    - Attempt to list available physical devices; if an exception occurs, log the error and return an empty results vector.
    - Iterate through each physical device and check its properties, including API version, memory properties, and supported features.
    - For each valid device, check if it has sufficient memory and supports necessary extensions.
    - Create a `ggml_vk_device` structure for each valid device, populating its properties, and add it to the results vector.
    - Sort the results vector based on device type and heap size before returning it.
- **Output**: Returns a vector of `ggml_vk_device` structures representing the available Vulkan devices that meet the specified memory requirements.
- **Functions called**:
    - [`ggml_vk_checkPhysicalDeviceFeatures`](#ggml_vk_checkPhysicalDeviceFeatures)
    - [`ggml_vk_getVendorName`](#ggml_vk_getVendorName)


---
### ggml\_vk\_available\_devices<!-- {{#callable:ggml_vk_available_devices}} -->
The `ggml_vk_available_devices` function returns a reference to a static vector containing Vulkan devices available for use.
- **Inputs**: None
- **Control Flow**:
    - The function defines a static vector of `ggml_vk_device` that is initialized by calling [`ggml_vk_available_devices_internal`](#ggml_vk_available_devices_internal) with an argument of 0.
    - The static vector is returned, ensuring that the same vector is used across multiple calls to the function.
- **Output**: The output is a reference to a static vector of `ggml_vk_device` objects, which represent the available Vulkan devices.
- **Functions called**:
    - [`ggml_vk_available_devices_internal`](#ggml_vk_available_devices_internal)


---
### ggml\_vk\_filterByVendor<!-- {{#callable:ggml_vk_filterByVendor}} -->
Filters a vector of `ggml_vk_device` objects to retain only those that match a specified vendor.
- **Inputs**:
    - `devices`: A reference to a vector of `ggml_vk_device` objects that will be filtered in place.
    - `targetVendor`: A string representing the vendor name to filter the devices by.
- **Control Flow**:
    - The function uses `std::remove_if` to identify devices that do not match the `targetVendor`.
    - The lambda function checks each device's `vendor` attribute against `targetVendor`.
    - Devices that do not match are removed from the `devices` vector using `erase`.
- **Output**: The function modifies the input vector `devices` in place, removing all devices that do not match the specified `targetVendor`.


---
### ggml\_vk\_filterByName<!-- {{#callable:ggml_vk_filterByName}} -->
Filters a vector of `ggml_vk_device` objects by removing those that do not match the specified target name.
- **Inputs**:
    - `devices`: A reference to a vector of `ggml_vk_device` objects that will be filtered in place.
    - `targetName`: A string representing the name to filter the devices by; only devices with this name will be retained.
- **Control Flow**:
    - The function uses `std::remove_if` to identify devices that do not match the `targetName`.
    - A lambda function is provided to `remove_if`, which checks if each device's `name` does not equal `targetName`.
    - The `erase` method is then called on the `devices` vector to remove the identified devices.
- **Output**: The function does not return a value; it modifies the input vector `devices` directly by removing non-matching devices.


---
### ggml\_vk\_get\_device<!-- {{#callable:ggml_vk_get_device}} -->
Retrieves a Vulkan device based on specified memory requirements and device name.
- **Inputs**:
    - `device`: A pointer to a `ggml_vk_device` structure where the selected device information will be stored.
    - `memoryRequired`: The amount of memory required for the device.
    - `name`: A C-style string representing the name of the device to be retrieved.
- **Control Flow**:
    - The function calls another overloaded version of [`ggml_vk_get_device`](#ggml_vk_get_device) that accepts a `std::string` instead of a C-style string.
    - It passes the same arguments to the overloaded function, ensuring that the device is retrieved based on the provided parameters.
- **Output**: Returns a boolean indicating whether the device was successfully retrieved.
- **Functions called**:
    - [`ggml_vk_get_device`](#ggml_vk_get_device)


---
### ggml\_vk\_has\_vulkan<!-- {{#callable:ggml_vk_has_vulkan}} -->
Checks if Vulkan is available through the Kompute manager.
- **Inputs**: None
- **Control Flow**:
    - Calls the `komputeManager()` function to get the instance of the Kompute manager.
    - Invokes the `hasVulkan()` method on the Kompute manager to check for Vulkan support.
- **Output**: Returns a boolean value indicating whether Vulkan is available.


---
### ggml\_vk\_has\_device<!-- {{#callable:ggml_vk_has_device}} -->
Checks if a Vulkan device is available.
- **Inputs**: None
- **Control Flow**:
    - Calls the `komputeManager()` function to get the current instance of the kompute manager.
    - Invokes the `hasDevice()` method on the kompute manager to check for device availability.
    - Returns the result of the `hasDevice()` method.
- **Output**: Returns a boolean value indicating whether a Vulkan device is available.


---
### ggml\_vk\_current\_device<!-- {{#callable:ggml_vk_current_device}} -->
The `ggml_vk_current_device` function retrieves the current Vulkan device being used by the Kompute manager.
- **Inputs**: None
- **Control Flow**:
    - Check if the Kompute manager has a device available using `komputeManager()->hasDevice()`.
    - If no device is available, return a default [`ggml_vk_device`](../../include/ggml-kompute.h.driver.md#ggml_vk_device) object.
    - Retrieve the list of available Vulkan devices using `ggml_vk_available_devices()`.
    - Filter the list of devices to only include those that match the name of the physical device currently in use.
    - Assert that the filtered list of devices is not empty.
    - Return the first device from the filtered list.
- **Output**: Returns a [`ggml_vk_device`](../../include/ggml-kompute.h.driver.md#ggml_vk_device) object representing the current Vulkan device.
- **Functions called**:
    - [`ggml_vk_device`](../../include/ggml-kompute.h.driver.md#ggml_vk_device)
    - [`ggml_vk_available_devices`](#ggml_vk_available_devices)
    - [`ggml_vk_filterByName`](#ggml_vk_filterByName)


---
### ggml\_vk\_allocate\_descriptor\_pool<!-- {{#callable:ggml_vk_allocate_descriptor_pool}} -->
Allocates a Vulkan descriptor pool for managing descriptor sets.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_kompute_context` structure that holds the Vulkan context.
    - `size`: A size_t value representing the maximum number of descriptor sets that can be allocated.
- **Control Flow**:
    - Creates a vector of `vk::DescriptorPoolSize` to define the types and counts of descriptors to allocate.
    - Initializes a `vk::DescriptorPoolCreateInfo` structure with the specified size and descriptor pool sizes.
    - Creates a shared pointer for the descriptor pool in the context.
    - Calls `createDescriptorPool` on the Vulkan device to allocate the descriptor pool.
    - Checks the result of the allocation and prints an error message if it fails.
- **Output**: The function does not return a value; it modifies the `ctx` structure to hold the allocated descriptor pool.


---
### ggml\_vk\_free\_descriptor\_pool<!-- {{#callable:ggml_vk_free_descriptor_pool}} -->
Frees the Vulkan descriptor pool associated with the given `ggml_kompute_context`.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_kompute_context` structure that contains the Vulkan descriptor pool to be freed.
- **Control Flow**:
    - Checks if the `pool` member of the `ctx` is not null.
    - If the pool is valid, it calls the `destroy` method of the Vulkan device to free the resources associated with the descriptor pool.
    - Sets the `pool` member of the `ctx` to null after freeing.
- **Output**: This function does not return a value; it performs a cleanup operation on the Vulkan descriptor pool.


---
### ggml\_vk\_allocate\_buffer<!-- {{#callable:ggml_vk_allocate_buffer}} -->
Allocates a Vulkan buffer with specified size and usage flags.
- **Inputs**:
    - `size`: The size in bytes of the buffer to be allocated.
- **Control Flow**:
    - Creates a `vk::BufferCreateInfo` object to specify the buffer's size and usage.
    - Sets the buffer usage to include storage, transfer source, and transfer destination.
    - Allocates a new `vk::Buffer` object.
    - Calls the Vulkan API to create the buffer using the specified create info.
    - Checks the result of the buffer creation; if unsuccessful, logs an error message.
- **Output**: Returns a pointer to the allocated `vk::Buffer` object.


---
### ggml\_vk\_allocate<!-- {{#callable:ggml_vk_allocate}} -->
Allocates Vulkan memory and buffers for a specified size.
- **Inputs**:
    - `size`: The size in bytes of the memory to allocate.
- **Control Flow**:
    - Creates a primary buffer using [`ggml_vk_allocate_buffer`](#ggml_vk_allocate_buffer) with the specified size.
    - Retrieves memory requirements for the primary buffer.
    - Allocates primary memory with device-local properties.
    - Binds the primary memory to the primary buffer.
    - If the primary memory is host-visible, it maps the memory to a data pointer.
    - If the primary memory is not host-visible, it allocates a staging buffer and memory with host-visible properties.
    - Binds the staging memory to the staging buffer and maps it to a data pointer.
    - Sets the size of the allocated memory structure.
- **Output**: Returns a `ggml_vk_memory` structure containing pointers to the allocated buffers and memory, along with the size of the allocated memory.
- **Functions called**:
    - [`ggml_vk_allocate_buffer`](#ggml_vk_allocate_buffer)
    - [`ggml_vk_allocate`](#ggml_vk_allocate)


---
### ggml\_vk\_aligned\_offset<!-- {{#callable:ggml_vk_aligned_offset}} -->
Calculates the aligned offset for a given buffer based on its alignment requirements.
- **Inputs**:
    - `buffer`: A `ggml_backend_buffer_t` structure representing the buffer whose alignment is to be checked.
    - `offset`: A `size_t` value representing the offset to be aligned.
- **Control Flow**:
    - Retrieve the minimum storage buffer offset alignment from the `buffer` using [`ggml_backend_buffer_get_alignment`](../ggml-backend.cpp.driver.md#ggml_backend_buffer_get_alignment).
    - Check if the provided `offset` is already aligned by verifying if it is divisible by the alignment value.
    - If aligned, return the `offset` directly.
    - If not aligned, calculate and return the largest multiple of the alignment that is less than the `offset`.
- **Output**: Returns a `size_t` value representing the aligned offset.
- **Functions called**:
    - [`ggml_backend_buffer_get_alignment`](../ggml-backend.cpp.driver.md#ggml_backend_buffer_get_alignment)


---
### ggml\_vk\_free\_memory<!-- {{#callable:ggml_vk_free_memory}} -->
Frees Vulkan memory and associated buffers.
- **Inputs**:
    - `memory`: A reference to a `ggml_vk_memory` structure containing Vulkan memory and buffer information.
- **Control Flow**:
    - Calls `destroy` on the `primaryBuffer` to release the primary Vulkan buffer.
    - Checks if `stagingBuffer` is not null, and if so, calls `destroy` on it to release the staging Vulkan buffer.
    - Calls `freeMemory` on the `primaryMemory` to free the associated Vulkan memory.
    - Checks if `stagingMemory` is not null, and if so, calls `freeMemory` on it to free the staging Vulkan memory.
- **Output**: The function does not return a value; it performs cleanup operations on Vulkan resources.


---
### ggml\_vk\_find\_tensor<!-- {{#callable:ggml_vk_find_tensor}} -->
Finds the Vulkan memory context for a given tensor and calculates its offset.
- **Inputs**:
    - `t`: A pointer to a `ggml_tensor` structure representing the tensor whose Vulkan memory context is to be found.
    - `offset`: A reference to a `uint64_t` variable that will store the calculated offset of the tensor within the Vulkan memory.
- **Control Flow**:
    - Retrieve the appropriate buffer from the tensor's source or directly from the tensor.
    - Assert that the buffer is valid and compatible with the expected Vulkan backend.
    - Cast the buffer's context to `ggml_vk_memory` to access Vulkan-specific memory details.
    - Calculate the offset of the tensor's data relative to the Vulkan memory context's data.
    - Assert that the calculated offset is valid and does not exceed the buffer's size.
    - Assign the calculated offset to the provided reference and return the Vulkan memory context.
- **Output**: Returns a pointer to the `ggml_vk_memory` structure representing the Vulkan memory context associated with the tensor.
- **Functions called**:
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)


---
### ggml\_vk\_get\_tensor<!-- {{#callable:ggml_vk_get_tensor}} -->
Retrieves a Vulkan tensor from a given `ggml_tensor`, adjusting offsets as necessary.
- **Inputs**:
    - `t`: A pointer to a `ggml_tensor` structure representing the tensor to retrieve.
    - `alignedOffset`: An optional pointer to a `uint32_t` where the aligned offset will be stored.
- **Control Flow**:
    - The function starts by initializing an `originalOffset` variable to zero.
    - It calls [`ggml_vk_find_tensor`](#ggml_vk_find_tensor) to find the Vulkan tensor associated with the input `ggml_tensor` and retrieves the original offset.
    - If the tensor is not found, it returns a static null tensor.
    - It calculates the number of elements and the number of bytes required for the tensor.
    - It computes the Vulkan offset using [`ggml_vk_aligned_offset`](#ggml_vk_aligned_offset).
    - If `alignedOffset` is provided, it calculates the difference between the original offset and the Vulkan offset, adjusting the number of bytes accordingly.
    - Finally, it calls `komputeManager()->tensor` to create and return a new Vulkan tensor using the calculated parameters.
- **Output**: Returns a shared pointer to a `kp::Tensor` representing the Vulkan tensor, or a null tensor if the input tensor was not found.
- **Functions called**:
    - [`ggml_vk_find_tensor`](#ggml_vk_find_tensor)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_nbytes`](../ggml.c.driver.md#ggml_nbytes)
    - [`ggml_vk_aligned_offset`](#ggml_vk_aligned_offset)


---
### getSpirvShader<!-- {{#callable:getSpirvShader}} -->
The `getSpirvShader` function converts raw shader data into a vector of `uint32_t`.
- **Inputs**:
    - `rawData`: A pointer to the raw shader data in the form of an array of unsigned characters.
    - `size`: The size of the raw data in bytes.
- **Control Flow**:
    - The function first checks if the provided size is divisible by the size of `uint32_t`.
    - If the size is not valid, it throws a runtime error.
    - It then casts the `rawData` pointer to a pointer of `uint32_t`.
    - The number of `uint32_t` elements is calculated by dividing the size by the size of `uint32_t`.
    - Finally, it constructs and returns a vector of `uint32_t` initialized with the data from the cast pointer.
- **Output**: Returns a vector of `uint32_t` containing the shader data converted from the raw input.


---
### safe\_divide<!-- {{#callable:safe_divide}} -->
The `safe_divide` function performs a safe division of two unsigned integers, ensuring that the divisor is valid and that the division does not result in a remainder.
- **Inputs**:
    - `a`: The numerator, an unsigned 32-bit integer.
    - `b`: The denominator, an unsigned 32-bit integer.
- **Control Flow**:
    - The function first checks if the denominator `b` is less than or equal to 1; if so, it returns the numerator `a` directly.
    - Next, it checks if the numerator `a` is divisible by `b` by evaluating the expression `(a % b) != 0`. If this condition is true, it logs an error message and aborts the program.
    - If the division is valid, it proceeds to return the result of the division `a / b`.
- **Output**: Returns the result of the division of `a` by `b` as an unsigned 32-bit integer, or the value of `a` if `b` is less than or equal to 1.


---
### ggml\_vk\_add<!-- {{#callable:ggml_vk_add}} -->
The `ggml_vk_add` function performs element-wise addition of two input tensors using Vulkan compute shaders.
- **Inputs**:
    - `seq`: A reference to a `kp::Sequence` object that records the operations to be executed.
    - `inA`: A shared pointer to the first input tensor.
    - `inB`: A shared pointer to the second input tensor.
    - `out`: A shared pointer to the output tensor where the result will be stored.
    - `inAOff`: An offset for the first input tensor.
    - `inBOff`: An offset for the second input tensor.
    - `outOff`: An offset for the output tensor.
    - `ne00`: The first dimension size of the first input tensor.
    - `ne01`: The second dimension size of the first input tensor.
    - `ne02`: The third dimension size of the first input tensor.
    - `ne03`: The fourth dimension size of the first input tensor.
    - `nb00`: The byte size of the first dimension of the first input tensor.
    - `nb01`: The byte size of the second dimension of the first input tensor.
    - `nb02`: The byte size of the third dimension of the first input tensor.
    - `nb03`: The byte size of the fourth dimension of the first input tensor.
    - `ne10`: The first dimension size of the second input tensor.
    - `ne11`: The second dimension size of the second input tensor.
    - `ne12`: The third dimension size of the second input tensor.
    - `ne13`: The fourth dimension size of the second input tensor.
    - `nb10`: The byte size of the first dimension of the second input tensor.
    - `nb11`: The byte size of the second dimension of the second input tensor.
    - `nb12`: The byte size of the third dimension of the second input tensor.
    - `nb13`: The byte size of the fourth dimension of the second input tensor.
    - `ne0`: The size of the output tensor.
    - `nb0`: The byte size of the first dimension of the output tensor.
    - `nb1`: The byte size of the second dimension of the output tensor.
    - `nb2`: The byte size of the third dimension of the output tensor.
    - `nb3`: The byte size of the fourth dimension of the output tensor.
- **Control Flow**:
    - The function begins by retrieving the SPIR-V shader code for the addition operation.
    - A structure `PushConstants` is defined to hold the offsets and sizes of the tensors.
    - The function checks if an algorithm for this operation already exists in the kompute manager.
    - If not, it creates a new algorithm using the provided tensors and the SPIR-V shader.
    - If the algorithm already exists, it updates the existing algorithm with the new tensor data and parameters.
    - Finally, the operation is recorded in the provided sequence for execution.
- **Output**: The function does not return a value; instead, it records the addition operation in the provided sequence for later execution.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`safe_divide`](#safe_divide)


---
### ggml\_vk\_addrow<!-- {{#callable:ggml_vk_addrow}} -->
The `ggml_vk_addrow` function adds a specified row from one tensor to another tensor using Vulkan compute shaders.
- **Inputs**:
    - `seq`: A reference to a `kp::Sequence` object that records the operations to be executed.
    - `inA`: A shared pointer to the input tensor `inA` from which a row will be added.
    - `inB`: A shared pointer to the input tensor `inB` which will receive the added row.
    - `out`: A shared pointer to the output tensor where the result will be stored.
    - `inAOff`: An offset for the input tensor `inA`.
    - `inBOff`: An offset for the input tensor `inB`.
    - `outOff`: An offset for the output tensor.
    - `size`: The size of the operation to be performed.
    - `row`: The specific row index to be added, defaulting to 0.
- **Control Flow**:
    - The function begins by retrieving the SPIR-V shader code for the add row operation.
    - A structure `PushConstants` is defined to hold the offsets and row index, which is then initialized.
    - The function checks if an algorithm for this operation already exists in the kompute manager.
    - If it does not exist, a new algorithm is created with the specified parameters and the SPIR-V shader.
    - If it does exist, the existing algorithm is updated with the new tensor references and parameters.
    - Finally, the operation is recorded in the sequence for execution.
- **Output**: The function does not return a value; it records an operation in the provided sequence for later execution.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`safe_divide`](#safe_divide)


---
### ggml\_vk\_mul<!-- {{#callable:ggml_vk_mul}} -->
The `ggml_vk_mul` function performs element-wise multiplication of two input tensors using Vulkan compute shaders.
- **Inputs**:
    - `seq`: A reference to a `kp::Sequence` object that manages the sequence of operations to be executed.
    - `inA`: A shared pointer to the first input tensor (`kp::Tensor`) for the multiplication operation.
    - `inB`: A shared pointer to the second input tensor (`kp::Tensor`) for the multiplication operation.
    - `out`: A shared pointer to the output tensor (`kp::Tensor`) where the result of the multiplication will be stored.
    - `inAOff`: An offset for the first input tensor, used to determine the starting point for the multiplication.
    - `inBOff`: An offset for the second input tensor, used to determine the starting point for the multiplication.
    - `outOff`: An offset for the output tensor, used to determine where to store the result.
    - `ne00, ne01, ne02, ne03`: Dimensions of the first input tensor, representing the number of elements in each dimension.
    - `nb00, nb01, nb02, nb03`: Strides for the first input tensor, indicating the byte offset between elements in each dimension.
    - `ne10, ne11, ne12, ne13`: Dimensions of the second input tensor, representing the number of elements in each dimension.
    - `nb10, nb11, nb12, nb13`: Strides for the second input tensor, indicating the byte offset between elements in each dimension.
    - `ne0`: The total number of elements in the output tensor.
    - `nb0, nb1, nb2, nb3`: Strides for the output tensor, indicating the byte offset between elements in each dimension.
- **Control Flow**:
    - The function begins by retrieving the SPIR-V shader code for the multiplication operation.
    - A structure `PushConstants` is defined to hold the offsets and dimensions for the input and output tensors.
    - The function checks if an algorithm for this operation already exists in the `komputeManager`.
    - If not, it creates a new algorithm using the provided tensors and the SPIR-V shader.
    - If the algorithm already exists, it updates the tensor references and workgroup sizes.
    - Finally, the function records the algorithm dispatch operation in the provided sequence.
- **Output**: The function does not return a value; instead, it records the multiplication operation in the provided sequence for execution.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`safe_divide`](#safe_divide)


---
### ggml\_vk\_scale<!-- {{#callable:ggml_vk_scale}} -->
The `ggml_vk_scale` function scales the input tensor by a specified factor and records the operation in a Vulkan sequence.
- **Inputs**:
    - `seq`: A reference to a `kp::Sequence` object that manages the sequence of operations to be executed.
    - `in`: A shared pointer to the input tensor (`kp::Tensor`) that will be scaled.
    - `out`: A shared pointer to the output tensor (`kp::Tensor`) where the scaled result will be stored.
    - `inOff`: An offset in the input tensor, used for indexing.
    - `outOff`: An offset in the output tensor, used for indexing.
    - `size`: The number of elements to scale, which may be adjusted based on the scaling operation.
    - `scale`: A float value representing the scaling factor to apply to the input tensor.
- **Control Flow**:
    - The function begins by retrieving two SPIR-V shaders for scaling operations, one for general scaling and another optimized for sizes divisible by 8.
    - It defines a structure for push constants that includes the input and output offsets and the scaling factor.
    - The function checks if the size is divisible by 8; if so, it adjusts the size and selects the appropriate shader.
    - It then checks if an algorithm for the scaling operation already exists in the kompute manager; if not, it creates a new algorithm.
    - If the algorithm exists, it updates the existing algorithm's tensors, workgroup size, and push constants.
    - Finally, the function records the algorithm dispatch operation in the provided sequence.
- **Output**: The function does not return a value; instead, it records the scaling operation in the Vulkan sequence for later execution.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`safe_divide`](#safe_divide)


---
### ggml\_vk\_xxlu<!-- {{#callable:ggml_vk_xxlu}} -->
The `ggml_vk_xxlu` function dispatches a Vulkan compute algorithm for tensor operations based on provided SPIR-V shaders.
- **Inputs**:
    - `spirv`: A vector of SPIR-V shader instructions used for the compute operation.
    - `suffix`: A string suffix to differentiate the algorithm name.
    - `seq`: A reference to a `kp::Sequence` object for recording operations.
    - `in`: A shared pointer to the input tensor.
    - `out`: A shared pointer to the output tensor.
    - `inOff`: An offset for the input tensor.
    - `outOff`: An offset for the output tensor.
    - `size`: The size of the workgroup for the compute operation.
- **Control Flow**:
    - Defines a structure `PushConstants` to hold input and output offsets.
    - Calculates the algorithm name by appending the `suffix` to the function name.
    - Checks if the algorithm with the generated name already exists in the `komputeManager`.
    - If the algorithm does not exist, it creates a new algorithm using the provided SPIR-V and push constants.
    - If the algorithm exists, it updates the existing algorithm's tensors, workgroup size, and push constants.
    - Records the algorithm dispatch operation in the provided sequence.
- **Output**: The function does not return a value; it records a dispatch operation for a Vulkan compute algorithm in the provided sequence.
- **Functions called**:
    - [`safe_divide`](#safe_divide)


---
### ggml\_vk\_silu<!-- {{#callable:ggml_vk_silu}} -->
The `ggml_vk_silu` function applies the sigmoid linear unit (SiLU) activation function using a Vulkan compute shader.
- **Inputs**:
    - `args`: A variable number of arguments that are forwarded to the [`ggml_vk_xxlu`](#ggml_vk_xxlu) function, which includes the Vulkan compute shader parameters.
- **Control Flow**:
    - The function begins by retrieving the SPIR-V shader code for the SiLU operation using the [`getSpirvShader`](#getSpirvShader) function.
    - It then calls the [`ggml_vk_xxlu`](#ggml_vk_xxlu) function, passing the retrieved SPIR-V code, the suffix 'silu', and the forwarded arguments.
- **Output**: The function does not return a value; instead, it executes the SiLU activation operation on the GPU using Vulkan.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`ggml_vk_xxlu`](#ggml_vk_xxlu)


---
### ggml\_vk\_relu<!-- {{#callable:ggml_vk_relu}} -->
The `ggml_vk_relu` function applies the ReLU activation function using a Vulkan compute shader.
- **Inputs**:
    - `args`: Variadic arguments that are forwarded to the [`ggml_vk_xxlu`](#ggml_vk_xxlu) function, which includes the Vulkan sequence and tensor parameters.
- **Control Flow**:
    - The function begins by retrieving the SPIR-V shader code for the ReLU operation using the [`getSpirvShader`](#getSpirvShader) function.
    - It then calls the [`ggml_vk_xxlu`](#ggml_vk_xxlu) function, passing the retrieved shader code, the suffix 'relu', and the forwarded arguments.
- **Output**: The function does not return a value; it executes the ReLU operation on the provided tensors using Vulkan.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`ggml_vk_xxlu`](#ggml_vk_xxlu)


---
### ggml\_vk\_gelu<!-- {{#callable:ggml_vk_gelu}} -->
The `ggml_vk_gelu` function applies the Gaussian Error Linear Unit (GELU) activation function using a Vulkan compute shader.
- **Inputs**:
    - `args`: A variadic template parameter pack that forwards any number of arguments to the Vulkan compute shader.
- **Control Flow**:
    - The function retrieves the SPIR-V shader code for the GELU operation using the [`getSpirvShader`](#getSpirvShader) function.
    - It then calls the [`ggml_vk_xxlu`](#ggml_vk_xxlu) function, passing the retrieved shader code, the operation name 'gelu', and the forwarded arguments.
- **Output**: The function does not return a value; it executes the Vulkan compute shader to perform the GELU operation on the provided input tensors.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`ggml_vk_xxlu`](#ggml_vk_xxlu)


---
### ggml\_vk\_soft\_max<!-- {{#callable:ggml_vk_soft_max}} -->
The `ggml_vk_soft_max` function computes the softmax operation on input tensors using Vulkan.
- **Inputs**:
    - `seq`: A reference to a `kp::Sequence` object that records the operations to be executed.
    - `inA`: A shared pointer to the first input tensor.
    - `inB`: A shared pointer to the second input tensor, which can be null.
    - `out`: A shared pointer to the output tensor where the result will be stored.
    - `inAOff`: An offset for the first input tensor.
    - `inBOff`: An offset for the second input tensor.
    - `outOff`: An offset for the output tensor.
    - `ne00`: The first dimension size of the input tensors.
    - `ne01`: The second dimension size of the input tensors.
    - `ne02`: The third dimension size of the input tensors.
    - `ne03`: The fourth dimension size of the input tensors.
    - `scale`: A scaling factor applied during the softmax computation.
    - `max_bias`: A bias value used to adjust the maximum value in the softmax calculation.
    - `m0`: A computed value related to the maximum bias.
    - `m1`: Another computed value related to the maximum bias.
    - `n_head_log2`: Log base 2 of the number of heads used in the softmax operation.
- **Control Flow**:
    - The function begins by retrieving the SPIR-V shader code for the softmax operation.
    - A structure `PushConstants` is defined to hold various parameters needed for the shader.
    - The function checks if `inB` is null and assigns `inA` to `inB_` if it is.
    - It then checks if an algorithm for this function has already been created; if not, it creates a new algorithm using the Vulkan API.
    - If the algorithm already exists, it updates the existing algorithm with the new tensor and push constant values.
    - Finally, the function records the operation in the provided sequence.
- **Output**: The function does not return a value; instead, it records the softmax operation in the provided sequence for execution.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`safe_divide`](#safe_divide)


---
### ggml\_vk\_norm\_<!-- {{#callable:ggml_vk_norm_}} -->
The `ggml_vk_norm_` function computes the normalization of a tensor using Vulkan and records the operation in a command sequence.
- **Inputs**:
    - `spirv`: A vector of `uint32_t` representing the SPIR-V shader code used for the normalization operation.
    - `suffix`: A string suffix used to differentiate the algorithm name for the normalization operation.
    - `seq`: A reference to a `kp::Sequence` object that records the operations to be executed.
    - `in`: A shared pointer to a `kp::Tensor` representing the input tensor to be normalized.
    - `out`: A shared pointer to a `kp::Tensor` representing the output tensor where the result of the normalization will be stored.
    - `inOff`: An offset in bytes for the input tensor.
    - `outOff`: An offset in bytes for the output tensor.
    - `ne00`: An integer representing the first dimension size of the input tensor.
    - `nb01`: An integer representing the byte size of the second dimension of the input tensor.
    - `nrows`: An integer representing the number of rows in the input tensor.
    - `epsilon`: A float value used to prevent division by zero during normalization.
- **Control Flow**:
    - The function begins by asserting that `nb01` and `ne00` are divisible by the size of a float to ensure proper memory alignment.
    - A structure `PushConstants` is defined to hold the offsets, dimensions, and epsilon value for the normalization operation.
    - The function constructs a unique algorithm name by appending the provided suffix to the function name.
    - It checks if the algorithm with the constructed name already exists in the kompute manager.
    - If the algorithm does not exist, it creates a new algorithm using the provided SPIR-V code and the push constants.
    - If the algorithm exists, it updates the existing algorithm's tensor references, workgroup size, and push constants.
    - Finally, the function records the algorithm dispatch operation in the provided command sequence.
- **Output**: The function does not return a value; instead, it records the normalization operation in the command sequence for later execution.
- **Functions called**:
    - [`safe_divide`](#safe_divide)


---
### ggml\_vk\_norm<!-- {{#callable:ggml_vk_norm}} -->
The `ggml_vk_norm` function invokes a Vulkan shader to perform normalization on input tensors.
- **Inputs**:
    - `args`: A variadic list of arguments that are forwarded to the [`ggml_vk_norm_`](#ggml_vk_norm_) function, which includes the input tensor, output tensor, offsets, dimensions, and other parameters required for normalization.
- **Control Flow**:
    - The function begins by retrieving the SPIR-V shader code for the normalization operation using the [`getSpirvShader`](#getSpirvShader) function.
    - It then calls the [`ggml_vk_norm_`](#ggml_vk_norm_) function, passing the retrieved shader, a string identifier for the operation, and the forwarded arguments.
- **Output**: The function does not return a value; instead, it performs the normalization operation on the provided tensors using Vulkan.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`ggml_vk_norm_`](#ggml_vk_norm_)


---
### ggml\_vk\_rms\_norm<!-- {{#callable:ggml_vk_rms_norm}} -->
The `ggml_vk_rms_norm` function computes the root mean square normalization using a Vulkan compute shader.
- **Inputs**:
    - `args`: A variadic template parameter pack that forwards any number of arguments to the [`ggml_vk_norm_`](#ggml_vk_norm_) function, which includes input tensors and their properties.
- **Control Flow**:
    - The function begins by retrieving the SPIR-V shader code for the RMS normalization operation using the [`getSpirvShader`](#getSpirvShader) function.
    - It then calls the [`ggml_vk_norm_`](#ggml_vk_norm_) function, passing the retrieved shader code along with the forwarded arguments.
- **Output**: The function does not return a value; instead, it performs the normalization operation on the provided tensors using Vulkan.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`ggml_vk_norm_`](#ggml_vk_norm_)


---
### ggml\_vk\_diag\_mask\_inf<!-- {{#callable:ggml_vk_diag_mask_inf}} -->
The `ggml_vk_diag_mask_inf` function applies a diagonal mask to a tensor using Vulkan compute shaders.
- **Inputs**:
    - `seq`: A reference to a `kp::Sequence` object that records the operations to be executed.
    - `in`: A shared pointer to the input tensor that will be processed.
    - `out`: A shared pointer to the output tensor where the result will be stored.
    - `inOff`: An offset for the input tensor, indicating where to start reading data.
    - `outOff`: An offset for the output tensor, indicating where to start writing data.
    - `n_past`: An integer representing the number of past elements to consider for masking.
    - `ne00`: An integer representing the first dimension size of the tensor.
    - `ne01`: An integer representing the second dimension size of the tensor.
    - `ne02`: An integer representing the third dimension size of the tensor.
- **Control Flow**:
    - The function begins by retrieving the SPIR-V shader code for the diagonal mask operation.
    - A structure `PushConstants` is defined to hold the necessary parameters for the shader, including offsets and dimensions.
    - The function checks if an algorithm for this operation already exists in the kompute manager.
    - If the algorithm does not exist, it creates a new algorithm using the shader and the input/output tensors.
    - If the algorithm exists, it updates the existing algorithm with the new tensor data and parameters.
    - Finally, the function records the operation in the provided sequence for execution.
- **Output**: The function does not return a value; instead, it records the operation in the sequence for later execution.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`safe_divide`](#safe_divide)


---
### ggml\_vk\_mul\_mat\_f16<!-- {{#callable:ggml_vk_mul_mat_f16}} -->
Multiplies two matrices represented as tensors in half-precision floating point format using Vulkan compute shaders.
- **Inputs**:
    - `seq`: A reference to a `kp::Sequence` object that manages the sequence of operations to be executed.
    - `inA`: A shared pointer to the first input tensor (`kp::Tensor`) representing the first matrix.
    - `inB`: A shared pointer to the second input tensor (`kp::Tensor`) representing the second matrix.
    - `out`: A shared pointer to the output tensor (`kp::Tensor`) where the result of the matrix multiplication will be stored.
    - `inAOff`: An offset for the first input tensor, used for indexing into the tensor's data.
    - `inBOff`: An offset for the second input tensor, used for indexing into the tensor's data.
    - `outOff`: An offset for the output tensor, used for indexing into the tensor's data.
    - `ne00`: The first dimension size of the first input tensor.
    - `ne01`: The second dimension size of the first input tensor.
    - `ne02`: The third dimension size of the first input tensor.
    - `nb00`: The byte size of the first dimension of the first input tensor.
    - `nb01`: The byte size of the second dimension of the first input tensor.
    - `nb02`: The byte size of the third dimension of the first input tensor.
    - `nb03`: The byte size of the fourth dimension of the first input tensor.
    - `ne10`: The first dimension size of the second input tensor.
    - `ne11`: The second dimension size of the second input tensor.
    - `ne12`: The third dimension size of the second input tensor.
    - `ne13`: The fourth dimension size of the second input tensor.
    - `nb10`: The byte size of the first dimension of the second input tensor.
    - `nb11`: The byte size of the second dimension of the second input tensor.
    - `nb12`: The byte size of the third dimension of the second input tensor.
    - `nb13`: The byte size of the fourth dimension of the second input tensor.
    - `ne0`: The first dimension size of the output tensor.
    - `ne1`: The second dimension size of the output tensor.
    - `r2`: A parameter related to the second dimension of the output tensor.
    - `r3`: A parameter related to the third dimension of the output tensor.
- **Control Flow**:
    - The function begins by retrieving the SPIR-V shader code for the matrix multiplication operation.
    - A structure `PushConstants` is defined to hold various offsets and sizes needed for the shader.
    - The function calculates the number of workgroups required based on the dimensions of the input tensors.
    - It checks if an algorithm for this operation already exists in the `komputeManager`.
    - If not, it creates a new algorithm using the shader and the input/output tensors, setting the appropriate workgroup sizes and push constants.
    - If the algorithm already exists, it updates the tensor references and workgroup sizes.
    - Finally, the operation is recorded in the sequence for execution.
- **Output**: The function does not return a value; instead, it records the matrix multiplication operation in the provided sequence for later execution.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`safe_divide`](#safe_divide)
    - [`ggml_vk_current_device`](#ggml_vk_current_device)


---
### ggml\_vk\_mul\_mat\_impl<!-- {{#callable:ggml_vk_mul_mat_impl}} -->
Implements matrix multiplication using Vulkan with specified input tensors and parameters.
- **Inputs**:
    - `spirv`: A vector of uint32_t representing the SPIR-V shader code for the matrix multiplication operation.
    - `suffix`: A string suffix used to differentiate algorithm names.
    - `block_size`: The size of the blocks used for processing the matrices.
    - `seq`: A reference to a `kp::Sequence` object that records the operations to be executed.
    - `inA`: A shared pointer to the input tensor A.
    - `inB`: A shared pointer to the input tensor B.
    - `out`: A shared pointer to the output tensor where the result will be stored.
    - `inAOff`: Offset for the input tensor A.
    - `inBOff`: Offset for the input tensor B.
    - `outOff`: Offset for the output tensor.
    - `ne00`: Dimension size for the first dimension of input tensor A.
    - `ne01`: Dimension size for the second dimension of input tensor A.
    - `ne02`: Dimension size for the third dimension of input tensor A.
    - `ne10`: Dimension size for the first dimension of input tensor B.
    - `ne11`: Dimension size for the second dimension of input tensor B.
    - `ne12`: Dimension size for the third dimension of input tensor B.
    - `ne13`: Dimension size for the fourth dimension of input tensor B.
    - `ne0`: Dimension size for the first dimension of the output tensor.
    - `ne1`: Dimension size for the second dimension of the output tensor.
    - `nb01`: Byte offset for the second dimension of input tensor A.
    - `nb02`: Byte offset for the third dimension of input tensor A.
    - `nb03`: Byte offset for the fourth dimension of input tensor A.
    - `nb11`: Byte offset for the second dimension of input tensor B.
    - `nb12`: Byte offset for the third dimension of input tensor B.
    - `nb13`: Byte offset for the fourth dimension of input tensor B.
    - `r2`: Dimension size for the second dimension of the output tensor.
    - `r3`: Dimension size for the third dimension of the output tensor.
- **Control Flow**:
    - Defines a structure `PushConstants` to hold various offsets and dimension sizes for the tensors involved in the multiplication.
    - Calculates the name of the algorithm based on the function name and the provided suffix.
    - Checks if the algorithm with the generated name already exists in the `komputeManager`.
    - If the algorithm does not exist, creates a new algorithm using the provided SPIR-V code and the push constants.
    - If the algorithm exists, updates its tensor references and workgroup sizes.
    - Records the operation for execution in the provided sequence.
- **Output**: The function does not return a value; instead, it records the matrix multiplication operation in the provided sequence for later execution.
- **Functions called**:
    - [`safe_divide`](#safe_divide)
    - [`ggml_vk_current_device`](#ggml_vk_current_device)


---
### ggml\_vk\_mul\_mat\_q4\_0<!-- {{#callable:ggml_vk_mul_mat_q4_0}} -->
The `ggml_vk_mul_mat_q4_0` function performs matrix multiplication using a Vulkan shader for quantized 4-bit matrices.
- **Inputs**:
    - `args`: A variadic template parameter pack that forwards multiple arguments to the [`ggml_vk_mul_mat_impl`](#ggml_vk_mul_mat_impl) function, which includes input tensors and their offsets.
- **Control Flow**:
    - The function begins by retrieving the SPIR-V shader code for the matrix multiplication operation specific to the quantized 4-bit format.
    - It then calls the [`ggml_vk_mul_mat_impl`](#ggml_vk_mul_mat_impl) function, passing the shader, a suffix identifier for the operation, a block size of 1 (indicating unaligned access), and the forwarded arguments.
- **Output**: The function does not return a value; instead, it executes the matrix multiplication operation on the GPU using Vulkan, with the results stored in the specified output tensor.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`ggml_vk_mul_mat_impl`](#ggml_vk_mul_mat_impl)


---
### ggml\_vk\_mul\_mat\_q4\_1<!-- {{#callable:ggml_vk_mul_mat_q4_1}} -->
The `ggml_vk_mul_mat_q4_1` function performs matrix multiplication using a Vulkan shader for quantized data.
- **Inputs**:
    - `seq`: A reference to a `kp::Sequence` object that manages the sequence of operations to be executed.
    - `inA`: A shared pointer to the first input tensor for the matrix multiplication.
    - `inB`: A shared pointer to the second input tensor for the matrix multiplication.
    - `out`: A shared pointer to the output tensor where the result of the multiplication will be stored.
    - `inAOff`: An offset for the first input tensor.
    - `inBOff`: An offset for the second input tensor.
    - `outOff`: An offset for the output tensor.
    - `ne00`: The first dimension size of the first input tensor.
    - `ne01`: The second dimension size of the first input tensor.
    - `ne02`: The third dimension size of the first input tensor.
    - `ne10`: The first dimension size of the second input tensor.
    - `ne11`: The second dimension size of the second input tensor.
    - `ne12`: The third dimension size of the second input tensor.
    - `ne13`: The fourth dimension size of the second input tensor.
    - `ne0`: The first dimension size of the output tensor.
    - `ne1`: The second dimension size of the output tensor.
    - `nb01`: The byte size of the first dimension of the first input tensor.
    - `nb02`: The byte size of the second dimension of the first input tensor.
    - `nb03`: The byte size of the third dimension of the first input tensor.
    - `nb11`: The byte size of the first dimension of the second input tensor.
    - `nb12`: The byte size of the second dimension of the second input tensor.
    - `nb13`: The byte size of the third dimension of the second input tensor.
    - `r2`: The second dimension size for the output tensor.
    - `r3`: The third dimension size for the output tensor.
- **Control Flow**:
    - The function retrieves the SPIR-V shader code for the `q4_1` matrix multiplication operation.
    - It then calls the [`ggml_vk_mul_mat_impl`](#ggml_vk_mul_mat_impl) function, passing the shader code and the provided arguments.
    - The [`ggml_vk_mul_mat_impl`](#ggml_vk_mul_mat_impl) function handles the actual execution of the matrix multiplication using Vulkan.
- **Output**: The function does not return a value; it performs the matrix multiplication operation and stores the result in the specified output tensor.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`ggml_vk_mul_mat_impl`](#ggml_vk_mul_mat_impl)


---
### ggml\_vk\_mul\_mat\_q8\_0<!-- {{#callable:ggml_vk_mul_mat_q8_0}} -->
Multiplies two matrices using Vulkan with quantization level q8_0.
- **Inputs**:
    - `seq`: A reference to a `kp::Sequence` object that manages the sequence of operations to be executed.
    - `inA`: A shared pointer to the first input tensor (`kp::Tensor`) for the matrix multiplication.
    - `inB`: A shared pointer to the second input tensor (`kp::Tensor`) for the matrix multiplication.
    - `out`: A shared pointer to the output tensor (`kp::Tensor`) where the result of the multiplication will be stored.
    - `inAOff`: An offset for the first input tensor, indicating where to start reading data.
    - `inBOff`: An offset for the second input tensor, indicating where to start reading data.
    - `outOff`: An offset for the output tensor, indicating where to start writing data.
    - `ne00`: The first dimension size of the first input tensor.
    - `ne01`: The second dimension size of the first input tensor.
    - `ne02`: The third dimension size of the first input tensor.
    - `ne10`: The first dimension size of the second input tensor.
    - `ne11`: The second dimension size of the second input tensor.
    - `ne12`: The third dimension size of the second input tensor.
    - `ne13`: The fourth dimension size of the second input tensor.
    - `ne0`: The first dimension size of the output tensor.
    - `ne1`: The second dimension size of the output tensor.
    - `nb01`: The byte offset for the first dimension of the first input tensor.
    - `nb02`: The byte offset for the second dimension of the first input tensor.
    - `nb03`: The byte offset for the third dimension of the first input tensor.
    - `nb11`: The byte offset for the first dimension of the second input tensor.
    - `nb12`: The byte offset for the second dimension of the second input tensor.
    - `nb13`: The byte offset for the third dimension of the second input tensor.
    - `r2`: The second dimension size of the output tensor.
    - `r3`: The third dimension size of the output tensor.
- **Control Flow**:
    - The function begins by retrieving the SPIR-V shader code for the q8_0 matrix multiplication operation.
    - It then calls the [`ggml_vk_mul_mat_impl`](#ggml_vk_mul_mat_impl) function, passing the retrieved SPIR-V code along with the specified suffix 'q8_0' and the input arguments.
    - The [`ggml_vk_mul_mat_impl`](#ggml_vk_mul_mat_impl) function handles the actual execution of the matrix multiplication using Vulkan.
- **Output**: The function does not return a value; instead, it performs the matrix multiplication operation and stores the result in the specified output tensor.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`ggml_vk_mul_mat_impl`](#ggml_vk_mul_mat_impl)


---
### ggml\_vk\_mul\_mat\_q4\_k<!-- {{#callable:ggml_vk_mul_mat_q4_k}} -->
Performs matrix multiplication of two quantized matrices using Vulkan.
- **Inputs**:
    - `seq`: A reference to a `kp::Sequence` object that manages the sequence of operations to be executed.
    - `inA`: A shared pointer to the first input tensor (`kp::Tensor`) representing the first matrix.
    - `inB`: A shared pointer to the second input tensor (`kp::Tensor`) representing the second matrix.
    - `out`: A shared pointer to the output tensor (`kp::Tensor`) where the result of the multiplication will be stored.
    - `inAOff`: An offset for the first input tensor, indicating where to start reading data.
    - `inBOff`: An offset for the second input tensor, indicating where to start reading data.
    - `outOff`: An offset for the output tensor, indicating where to start writing data.
    - `ne00`: The first dimension size of the first input tensor.
    - `ne01`: The second dimension size of the first input tensor.
    - `ne02`: The third dimension size of the first input tensor.
    - `ne10`: The first dimension size of the second input tensor.
    - `ne11`: The second dimension size of the second input tensor.
    - `ne12`: The third dimension size of the second input tensor.
    - `ne13`: The fourth dimension size of the second input tensor.
    - `ne0`: The first dimension size of the output tensor.
    - `ne1`: The second dimension size of the output tensor.
    - `nb01`: The byte size of the first dimension of the first input tensor.
    - `nb02`: The byte size of the second dimension of the first input tensor.
    - `nb03`: The byte size of the third dimension of the first input tensor.
    - `nb11`: The byte size of the first dimension of the second input tensor.
    - `nb12`: The byte size of the second dimension of the second input tensor.
    - `nb13`: The byte size of the third dimension of the second input tensor.
    - `r2`: A parameter related to the second dimension of the output tensor.
    - `r3`: A parameter related to the third dimension of the output tensor.
- **Control Flow**:
    - The function begins by retrieving the SPIR-V shader code for the matrix multiplication operation.
    - A structure `PushConstants` is defined to hold various parameters needed for the shader execution.
    - The function checks if an algorithm for this operation already exists in the `komputeManager`.
    - If the algorithm does not exist, it creates a new algorithm using the provided tensors and shader.
    - If the algorithm exists, it updates the existing algorithm with the new tensor data and parameters.
    - Finally, the operation is recorded in the sequence for execution.
- **Output**: The function does not return a value; instead, it records the matrix multiplication operation in the provided sequence for later execution.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`safe_divide`](#safe_divide)


---
### ggml\_vk\_mul\_mat\_q6\_k<!-- {{#callable:ggml_vk_mul_mat_q6_k}} -->
Performs matrix multiplication on quantized 6-bit tensors using Vulkan.
- **Inputs**:
    - `seq`: A reference to a `kp::Sequence` object that manages the sequence of operations to be executed.
    - `inA`: A shared pointer to the first input tensor (`kp::Tensor`) for the multiplication.
    - `inB`: A shared pointer to the second input tensor (`kp::Tensor`) for the multiplication.
    - `out`: A shared pointer to the output tensor (`kp::Tensor`) where the result of the multiplication will be stored.
    - `inAOff`: An offset for the first input tensor, indicating where to start reading data.
    - `inBOff`: An offset for the second input tensor, indicating where to start reading data.
    - `outOff`: An offset for the output tensor, indicating where to start writing data.
    - `ne00`: The size of the first dimension of the first input tensor.
    - `ne01`: The size of the second dimension of the first input tensor.
    - `ne02`: The size of the third dimension of the first input tensor.
    - `ne10`: The size of the first dimension of the second input tensor.
    - `ne11`: The size of the second dimension of the second input tensor.
    - `ne12`: The size of the third dimension of the second input tensor.
    - `ne13`: The size of the fourth dimension of the second input tensor.
    - `ne0`: The size of the first dimension of the output tensor.
    - `ne1`: The size of the second dimension of the output tensor.
    - `nb01`: The byte size of the second dimension of the first input tensor.
    - `nb02`: The byte size of the third dimension of the first input tensor.
    - `nb03`: The byte size of the fourth dimension of the first input tensor.
    - `nb11`: The byte size of the second dimension of the second input tensor.
    - `nb12`: The byte size of the third dimension of the second input tensor.
    - `nb13`: The byte size of the fourth dimension of the second input tensor.
    - `r2`: A parameter related to the second dimension of the output tensor.
    - `r3`: A parameter related to the third dimension of the output tensor.
- **Control Flow**:
    - The function begins by retrieving the SPIR-V shader code for the matrix multiplication operation.
    - A structure `PushConstants` is defined to hold various parameters needed for the shader execution.
    - The function checks if an algorithm for this operation already exists in the `komputeManager`.
    - If the algorithm does not exist, it creates a new algorithm using the provided tensors and shader.
    - If the algorithm exists, it updates the existing algorithm with the new tensor data and parameters.
    - Finally, the operation is recorded in the sequence for execution.
- **Output**: The function does not return a value; instead, it records the matrix multiplication operation in the provided sequence for later execution.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`safe_divide`](#safe_divide)
    - [`ggml_vk_current_device`](#ggml_vk_current_device)


---
### ggml\_vk\_get\_rows<!-- {{#callable:ggml_vk_get_rows}} -->
The `ggml_vk_get_rows` function retrieves specific rows from input tensors and stores them in an output tensor using Vulkan compute shaders.
- **Inputs**:
    - `spirv`: A vector of unsigned 32-bit integers representing the SPIR-V shader code to be executed.
    - `suffix`: A string suffix used to create a unique name for the algorithm.
    - `element_size`: An unsigned integer representing the size of each element in bytes.
    - `qk`: An unsigned integer that may be used to validate the divisibility of `ne00`.
    - `seq`: A reference to a `kp::Sequence` object that records the operations to be executed.
    - `inA`: A shared pointer to the input tensor `inA`.
    - `inB`: A shared pointer to the input tensor `inB`.
    - `out`: A shared pointer to the output tensor where the results will be stored.
    - `inAOff`: An unsigned integer offset for the input tensor `inA`.
    - `inBOff`: An unsigned integer offset for the input tensor `inB`.
    - `outOff`: An unsigned integer offset for the output tensor.
    - `ne00`: An integer representing the first dimension size of the input tensor.
    - `nb01`: An integer representing the second dimension size of the input tensor.
    - `nb1`: An integer representing the size of the last dimension of the input tensor.
    - `size`: An unsigned integer representing the number of rows to retrieve.
- **Control Flow**:
    - The function begins by asserting that `nb01` is divisible by `element_size` and `nb1` is divisible by the size of a float.
    - If `qk` is non-zero, it asserts that `ne00` is divisible by `qk`.
    - A structure `PushConstants` is defined to hold offsets and sizes for the input and output tensors.
    - The function constructs a unique algorithm name using the function name and the provided suffix.
    - It checks if the algorithm with the constructed name already exists in the kompute manager.
    - If the algorithm does not exist, it creates a new algorithm using the provided SPIR-V code and push constants.
    - If the algorithm exists, it updates the tensor references, workgroup size, and push constants for the existing algorithm.
    - Finally, it records the algorithm dispatch operation in the provided sequence.
- **Output**: The function does not return a value; instead, it records a Vulkan compute operation that will be executed later, which retrieves specified rows from the input tensors and stores them in the output tensor.
- **Functions called**:
    - [`safe_divide`](#safe_divide)


---
### ggml\_vk\_get\_rows\_f32<!-- {{#callable:ggml_vk_get_rows_f32}} -->
Retrieves rows of floating-point data from a tensor using Vulkan compute shaders.
- **Inputs**:
    - `seq`: A reference to a `kp::Sequence` object that manages the sequence of operations to be executed.
    - `inA`: A shared pointer to the input tensor from which rows are to be retrieved.
    - `inB`: A shared pointer to the tensor that specifies which rows to retrieve.
    - `out`: A shared pointer to the output tensor where the retrieved rows will be stored.
    - `inAOff`: An offset into the input tensor `inA`.
    - `inBOff`: An offset into the input tensor `inB`.
    - `outOff`: An offset into the output tensor `out`.
    - `ne00`: The first dimension size of the input tensor `inA`.
    - `nb01`: The byte size of the second dimension of the input tensor `inB`.
    - `nb1`: The byte size of the output tensor `out`.
    - `size`: The number of rows to retrieve.
- **Control Flow**:
    - The function begins by asserting that the byte sizes of the input and output tensors are valid based on their element sizes.
    - It constructs a `PushConstants` structure to hold offsets and sizes for the Vulkan shader.
    - The function checks if a Vulkan algorithm for this operation already exists; if not, it creates one using the provided SPIR-V shader.
    - The algorithm is then recorded into the sequence for execution.
- **Output**: The function does not return a value; instead, it records the operation in the provided sequence for later execution.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`ggml_vk_get_rows`](#ggml_vk_get_rows)


---
### ggml\_vk\_get\_rows\_f16<!-- {{#callable:ggml_vk_get_rows_f16}} -->
The `ggml_vk_get_rows_f16` function retrieves rows of data in half-precision floating-point format using a Vulkan compute shader.
- **Inputs**:
    - `args`: Variadic arguments that are forwarded to the [`ggml_vk_get_rows`](#ggml_vk_get_rows) function, which may include input and output tensor references and their respective offsets.
- **Control Flow**:
    - The function begins by defining a static variable `spirv` that holds the compiled SPIR-V shader code for the operation, obtained from the [`getSpirvShader`](#getSpirvShader) function.
    - It then calls the [`ggml_vk_get_rows`](#ggml_vk_get_rows) function, passing the `spirv` variable, the string 'f16' to indicate the data type, the size of the `half` type, a zero for the `qk` parameter, and the forwarded arguments.
- **Output**: The function does not return a value; instead, it performs operations that modify the output tensor based on the input tensors and the specified parameters.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`ggml_vk_get_rows`](#ggml_vk_get_rows)


---
### ggml\_vk\_get\_rows\_q4\_0<!-- {{#callable:ggml_vk_get_rows_q4_0}} -->
Retrieves rows from a quantized tensor using Vulkan compute shaders.
- **Inputs**:
    - `args`: Variadic arguments that are forwarded to the [`ggml_vk_get_rows`](#ggml_vk_get_rows) function, which includes input tensors and their offsets.
- **Control Flow**:
    - The function begins by defining a static variable `spirv` that holds the compiled SPIR-V shader code for the operation.
    - It then calls the [`ggml_vk_get_rows`](#ggml_vk_get_rows) function, passing the `spirv`, a suffix string 'q4_0', an element size of 1, and a constant `QK4_0` along with the forwarded arguments.
- **Output**: The function does not return a value; instead, it performs operations that modify the output tensor based on the input tensors and the specified shader.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`ggml_vk_get_rows`](#ggml_vk_get_rows)


---
### ggml\_vk\_get\_rows\_q4\_1<!-- {{#callable:ggml_vk_get_rows_q4_1}} -->
The `ggml_vk_get_rows_q4_1` function retrieves rows from a tensor using a Vulkan shader for quantized data.
- **Inputs**:
    - `args`: Variadic arguments that are forwarded to the [`ggml_vk_get_rows`](#ggml_vk_get_rows) function, which include the input and output tensors and their respective offsets and dimensions.
- **Control Flow**:
    - The function begins by defining a static variable `spirv` that holds the compiled SPIR-V shader code for the operation, obtained from the [`getSpirvShader`](#getSpirvShader) function.
    - It then calls the [`ggml_vk_get_rows`](#ggml_vk_get_rows) function, passing the `spirv` shader, a suffix string 'q4_1', a block size of 1 (indicating unaligned access), a constant `QK4_1`, and the forwarded arguments.
- **Output**: The function does not return a value; instead, it performs operations that modify the output tensor based on the input tensor and the specified shader.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`ggml_vk_get_rows`](#ggml_vk_get_rows)


---
### ggml\_vk\_get\_rows\_q6\_k<!-- {{#callable:ggml_vk_get_rows_q6_k}} -->
Retrieves rows from a quantized tensor using Vulkan compute shaders.
- **Inputs**:
    - `args`: Variadic arguments that are forwarded to the [`ggml_vk_get_rows`](#ggml_vk_get_rows) function, which include the input and output tensors and their respective offsets and dimensions.
- **Control Flow**:
    - The function begins by retrieving the SPIR-V shader code for the `op_getrows_q6_k` operation.
    - It then calls the [`ggml_vk_get_rows`](#ggml_vk_get_rows) function, passing the shader, a suffix identifier, a block size of 1 (indicating unaligned access), a constant `QK_NL`, and the forwarded arguments.
- **Output**: The function does not return a value; instead, it performs operations that modify the output tensor based on the input tensor rows retrieved using the specified shader.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`ggml_vk_get_rows`](#ggml_vk_get_rows)


---
### ggml\_vk\_rope<!-- {{#callable:ggml_vk_rope}} -->
The `ggml_vk_rope` function performs a tensor operation using Vulkan, applying a specific type of rotation encoding based on the provided parameters.
- **Inputs**:
    - `seq`: A reference to a `kp::Sequence` object that manages the sequence of operations to be executed.
    - `inA`: A shared pointer to the first input tensor.
    - `inB`: A shared pointer to the second input tensor.
    - `inC`: A shared pointer to the third input tensor, which can be null.
    - `out`: A shared pointer to the output tensor.
    - `inAOff`: An offset for the first input tensor.
    - `inBOff`: An offset for the second input tensor.
    - `inCOff`: An offset for the third input tensor.
    - `outOff`: An offset for the output tensor.
    - `src0t`: The data type of the source tensor, which can be either `GGML_TYPE_F16` or `GGML_TYPE_F32`.
    - `n_dims`: The number of dimensions for the tensors.
    - `mode`: An integer representing the mode of operation, which determines the type of rotation encoding.
    - `n_ctx_orig`: An integer representing the original context size.
    - `freq_base`: A float representing the base frequency for the encoding.
    - `freq_scale`: A float representing the scale of the frequency.
    - `has_freq_factors`: A boolean indicating if frequency factors are present.
    - `ext_factor`: A float representing an external factor for the operation.
    - `attn_factor`: A float representing the attention factor.
    - `beta_fast`: A float representing the fast beta parameter.
    - `beta_slow`: A float representing the slow beta parameter.
    - `ne01`: An integer representing the first dimension size for the operation.
    - `ne02`: An integer representing the second dimension size for the operation.
    - `ne03`: An integer representing the third dimension size for the operation.
    - `nb00`: An unsigned integer representing the byte size of the first dimension.
    - `nb01`: An unsigned integer representing the byte size of the second dimension.
    - `nb02`: An unsigned integer representing the byte size of the third dimension.
    - `nb03`: An unsigned integer representing the byte size of the fourth dimension.
    - `ne0`: An integer representing the size of the output tensor.
    - `nb0`: An unsigned integer representing the byte size of the output tensor's first dimension.
    - `nb1`: An unsigned integer representing the byte size of the output tensor's second dimension.
    - `nb2`: An unsigned integer representing the byte size of the output tensor's third dimension.
    - `nb3`: An unsigned integer representing the byte size of the output tensor's fourth dimension.
- **Control Flow**:
    - The function begins by asserting that the data type of the source tensor is either `GGML_TYPE_F16` or `GGML_TYPE_F32`.
    - It retrieves the appropriate SPIR-V shader based on the data type and mode of operation.
    - The function calculates the size of the data type and asserts that the byte sizes of the input and output tensors are correctly aligned.
    - A structure for push constants is defined and initialized with the provided parameters.
    - The function checks if the third input tensor is null and assigns it to the first input tensor if it is.
    - It constructs a unique name for the algorithm based on the function name, mode, and data type.
    - If the algorithm does not already exist in the kompute manager, it creates a new algorithm using the appropriate SPIR-V shader and parameters.
    - If the algorithm already exists, it updates the tensor references and push constants.
    - Finally, the function records the algorithm dispatch operation in the sequence.
- **Output**: The function does not return a value; instead, it records an operation in the provided sequence for execution in a Vulkan context.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`safe_divide`](#safe_divide)


---
### ggml\_vk\_cpy<!-- {{#callable:ggml_vk_cpy}} -->
Copies data from one tensor to another using Vulkan compute shaders.
- **Inputs**:
    - `spirv`: A vector of uint32_t representing the SPIR-V shader code used for the copy operation.
    - `in_element_size`: The size in bytes of each element in the input tensor.
    - `out_element_size`: The size in bytes of each element in the output tensor.
    - `seq`: A reference to a `kp::Sequence` object that records the operations to be executed.
    - `in`: A shared pointer to the input tensor (`kp::Tensor`) from which data will be copied.
    - `out`: A shared pointer to the output tensor (`kp::Tensor`) to which data will be copied.
    - `inOff`: The offset in the input tensor from which to start copying data.
    - `outOff`: The offset in the output tensor where the copied data will be written.
    - `ne00, ne01, ne02, ne03`: Dimensions of the input tensor, used to determine the size of the data to copy.
    - `nb00, nb01, nb02, nb03`: Strides for the input tensor, indicating the byte offset between consecutive elements in each dimension.
    - `ne0, ne1, ne2`: Dimensions of the output tensor, used to determine the size of the data to copy.
    - `nb0, nb1, nb2, nb3`: Strides for the output tensor, indicating the byte offset between consecutive elements in each dimension.
- **Control Flow**:
    - A `PushConstants` structure is defined to hold the necessary parameters for the shader, including offsets and dimensions.
    - The function constructs a unique name for the algorithm based on the input and output element sizes.
    - It checks if an algorithm with the generated name already exists in the `komputeManager`.
    - If the algorithm does not exist, it creates a new algorithm using the provided SPIR-V code and parameters.
    - If the algorithm exists, it updates the existing algorithm's tensors and parameters.
    - Finally, the function records the algorithm dispatch operation in the provided sequence.
- **Output**: The function does not return a value; it records the copy operation in the provided sequence for execution.
- **Functions called**:
    - [`safe_divide`](#safe_divide)


---
### ggml\_vk\_cpy\_f32\_f16<!-- {{#callable:ggml_vk_cpy_f32_f16}} -->
Copies data from a tensor of type `float32` to a tensor of type `float16` using a Vulkan compute shader.
- **Inputs**:
    - `args`: Variadic arguments that are forwarded to the [`ggml_vk_cpy`](#ggml_vk_cpy) function, which includes the input and output tensors and their respective offsets and dimensions.
- **Control Flow**:
    - The function retrieves the SPIR-V shader code for the copy operation from the shader data.
    - It then calls the [`ggml_vk_cpy`](#ggml_vk_cpy) function, passing the shader code along with the input and output tensor parameters.
- **Output**: The function does not return a value; it performs the copy operation directly on the GPU.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`ggml_vk_cpy`](#ggml_vk_cpy)


---
### ggml\_vk\_cpy\_f32\_f32<!-- {{#callable:ggml_vk_cpy_f32_f32}} -->
Copies data from one tensor to another using Vulkan compute shaders.
- **Inputs**:
    - `args`: Variadic arguments that are forwarded to the [`ggml_vk_cpy`](#ggml_vk_cpy) function, which includes the input tensor, output tensor, offsets, and dimensions.
- **Control Flow**:
    - The function retrieves the SPIR-V shader code for the copy operation using [`getSpirvShader`](#getSpirvShader).
    - It then calls the [`ggml_vk_cpy`](#ggml_vk_cpy) function with the retrieved shader, input and output element sizes, and the forwarded arguments.
- **Output**: The function does not return a value; it performs the copy operation asynchronously using Vulkan.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`ggml_vk_cpy`](#ggml_vk_cpy)


---
### ggml\_vk\_cpy\_f16\_f16<!-- {{#callable:ggml_vk_cpy_f16_f16}} -->
Copies data from one tensor to another using Vulkan compute shaders.
- **Inputs**:
    - `args`: Variadic arguments that are forwarded to the [`ggml_vk_cpy`](#ggml_vk_cpy) function, which includes input and output tensors and their respective offsets and dimensions.
- **Control Flow**:
    - The function begins by retrieving the SPIR-V shader code for the copy operation using the [`getSpirvShader`](#getSpirvShader) function.
    - It then calls the [`ggml_vk_cpy`](#ggml_vk_cpy) function, passing the retrieved shader, input and output element sizes, and the forwarded arguments.
- **Output**: The function does not return a value; it performs the copy operation asynchronously using Vulkan.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`ggml_vk_cpy`](#ggml_vk_cpy)


---
### ggml\_vk\_cpy\_f16\_f32<!-- {{#callable:ggml_vk_cpy_f16_f32}} -->
Copies data from a tensor of half-precision floating-point values to a tensor of single-precision floating-point values using a Vulkan compute shader.
- **Inputs**:
    - `args`: Variadic arguments that are forwarded to the [`ggml_vk_cpy`](#ggml_vk_cpy) function, which includes the Vulkan shader, input tensor, output tensor, offsets, and dimensions.
- **Control Flow**:
    - The function begins by retrieving the SPIR-V shader code for the half-to-float copy operation using the [`getSpirvShader`](#getSpirvShader) function.
    - It then calls the [`ggml_vk_cpy`](#ggml_vk_cpy) function, passing the retrieved shader, input and output element sizes, and the forwarded arguments.
- **Output**: The function does not return a value; it performs the copy operation asynchronously using Vulkan.
- **Functions called**:
    - [`getSpirvShader`](#getSpirvShader)
    - [`ggml_vk_cpy`](#ggml_vk_cpy)


---
### ggml\_backend\_kompute\_device\_supports\_op<!-- {{#callable:ggml_backend_kompute_device_supports_op}} -->
Determines if a specific `ggml_backend_dev_t` device supports a given operation defined by a `ggml_tensor`.
- **Inputs**:
    - `dev`: A device of type `ggml_backend_dev_t` that represents the backend device to check for operation support.
    - `op`: A pointer to a `ggml_tensor` structure that defines the operation to be checked for support.
- **Control Flow**:
    - The function retrieves the number of elements in the tensor operation using `ggml_nelements(op)`.
    - A switch statement is used to determine the type of operation defined in `op->op`.
    - For unary operations, it checks if the number of elements is divisible by 4 and further checks specific unary operations for additional conditions.
    - For certain operations like `GGML_OP_NONE`, `GGML_OP_RESHAPE`, and others, it returns true, indicating support.
    - For the `GGML_OP_ROPE`, it checks specific modes to determine support.
    - For operations like `GGML_OP_DUP`, `GGML_OP_CPY`, and `GGML_OP_CONT`, it checks the types of the source tensors to ensure they are either `GGML_TYPE_F32` or `GGML_TYPE_F16`.
    - For `GGML_OP_DIAG_MASK_INF`, it checks if the fourth dimension of the tensor is equal to 1.
    - For matrix multiplication operations (`GGML_OP_MUL_MAT`), it checks the type of the second source tensor and whether either source tensor is transposed.
    - If none of the cases match, it returns false.
- **Output**: Returns a boolean value indicating whether the specified device supports the operation defined by the tensor.
- **Functions called**:
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_get_unary_op`](../ggml.c.driver.md#ggml_get_unary_op)
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_is_transposed`](../ggml.c.driver.md#ggml_is_transposed)


---
### ggml\_vk\_graph\_compute<!-- {{#callable:ggml_vk_graph_compute}} -->
The `ggml_vk_graph_compute` function executes a computational graph using Vulkan, distributing the workload across multiple sequences.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_kompute_context` structure that holds the Vulkan context and device information.
    - `gf`: A pointer to a `ggml_cgraph` structure representing the computational graph to be executed.
- **Control Flow**:
    - Allocate a descriptor pool based on the number of nodes in the graph.
    - Create a vector of sequences to manage the execution of graph nodes.
    - Distribute the nodes of the graph across multiple sequences for parallel execution.
    - For each node, retrieve its source tensors and destination tensor, and check if the operation is valid.
    - Record the appropriate Vulkan commands for each operation based on the type of operation (e.g., addition, multiplication, etc.).
    - Evaluate the recorded commands asynchronously for each sequence.
    - Wait for all sequences to complete execution before freeing the descriptor pool.
- **Output**: The function does not return a value but performs the computation defined by the graph and manages the execution flow using Vulkan.
- **Functions called**:
    - [`ggml_vk_allocate_descriptor_pool`](#ggml_vk_allocate_descriptor_pool)
    - [`ggml_is_empty`](../ggml.c.driver.md#ggml_is_empty)
    - [`ggml_vk_get_tensor`](#ggml_vk_get_tensor)
    - [`ggml_nelements`](../ggml.c.driver.md#ggml_nelements)
    - [`ggml_is_contiguous`](../ggml.c.driver.md#ggml_is_contiguous)
    - [`ggml_vk_addrow`](#ggml_vk_addrow)
    - [`ggml_vk_add`](#ggml_vk_add)
    - [`ggml_vk_mul`](#ggml_vk_mul)
    - [`ggml_vk_scale`](#ggml_vk_scale)
    - [`ggml_get_unary_op`](../ggml.c.driver.md#ggml_get_unary_op)
    - [`ggml_vk_silu`](#ggml_vk_silu)
    - [`ggml_vk_relu`](#ggml_vk_relu)
    - [`ggml_vk_gelu`](#ggml_vk_gelu)
    - [`ggml_op_name`](../ggml.c.driver.md#ggml_op_name)
    - [`ggml_nrows`](../ggml.c.driver.md#ggml_nrows)
    - [`ggml_vk_soft_max`](#ggml_vk_soft_max)
    - [`ggml_vk_diag_mask_inf`](#ggml_vk_diag_mask_inf)
    - [`ggml_vk_norm`](#ggml_vk_norm)
    - [`ggml_vk_rms_norm`](#ggml_vk_rms_norm)
    - [`ggml_is_transposed`](../ggml.c.driver.md#ggml_is_transposed)
    - [`ggml_vk_mul_mat_f16`](#ggml_vk_mul_mat_f16)
    - [`ggml_vk_mul_mat_q8_0`](#ggml_vk_mul_mat_q8_0)
    - [`ggml_vk_mul_mat_q4_0`](#ggml_vk_mul_mat_q4_0)
    - [`ggml_vk_mul_mat_q4_1`](#ggml_vk_mul_mat_q4_1)
    - [`ggml_vk_mul_mat_q4_k`](#ggml_vk_mul_mat_q4_k)
    - [`ggml_vk_mul_mat_q6_k`](#ggml_vk_mul_mat_q6_k)
    - [`ggml_vk_get_rows_f32`](#ggml_vk_get_rows_f32)
    - [`ggml_vk_get_rows_f16`](#ggml_vk_get_rows_f16)
    - [`ggml_vk_get_rows_q4_0`](#ggml_vk_get_rows_q4_0)
    - [`ggml_vk_get_rows_q4_1`](#ggml_vk_get_rows_q4_1)
    - [`ggml_vk_get_rows_q6_k`](#ggml_vk_get_rows_q6_k)
    - [`ggml_vk_rope`](#ggml_vk_rope)
    - [`ggml_vk_cpy_f32_f16`](#ggml_vk_cpy_f32_f16)
    - [`ggml_vk_cpy_f32_f32`](#ggml_vk_cpy_f32_f32)
    - [`ggml_vk_cpy_f16_f16`](#ggml_vk_cpy_f16_f16)
    - [`ggml_vk_cpy_f16_f32`](#ggml_vk_cpy_f16_f32)
    - [`ggml_vk_free_descriptor_pool`](#ggml_vk_free_descriptor_pool)


---
### dataType<!-- {{#callable:kp::TensorT<uint8_t>::dataType}} -->
Returns the data type of the `TensorT<uint8_t>` as an unsigned integer.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns a predefined constant value without any conditional logic or loops.
- **Output**: Returns `TensorDataTypes::eUnsignedInt`, indicating that the data type of the tensor is an unsigned integer.


---
### ggml\_backend\_kompute\_device\_ref<!-- {{#callable:ggml_backend_kompute_device_ref}} -->
Increments the reference count for a Vulkan device context and initializes it if it is not already initialized.
- **Inputs**:
    - `buft`: A pointer to a `ggml_backend_buffer_type_t` structure that contains the context of the Vulkan device.
- **Control Flow**:
    - The function retrieves the context from the input buffer type.
    - If the device reference count is zero, it initializes the Vulkan device with specific extensions.
    - It asserts that a Vulkan device is available.
    - Finally, it increments the device reference count.
- **Output**: The function does not return a value; it modifies the state of the device reference count.
- **Functions called**:
    - [`ggml_vk_has_device`](#ggml_vk_has_device)


---
### ggml\_backend\_kompute\_device\_unref<!-- {{#callable:ggml_backend_kompute_device_unref}} -->
The `ggml_backend_kompute_device_unref` function decrements the reference count of a device context and destroys the device if the reference count reaches zero.
- **Inputs**:
    - `buft`: A pointer to a `ggml_backend_buffer_type_t` structure representing the backend buffer type context.
- **Control Flow**:
    - The function retrieves the context associated with the provided `buft` argument.
    - It asserts that the device reference count is greater than zero to ensure that it can be decremented.
    - The reference count is decremented by one.
    - If the reference count reaches zero, the `komputeManager` is destroyed, releasing any resources associated with the device.
- **Output**: The function does not return a value; it modifies the state of the device context and may destroy the device if no references remain.


---
### ggml\_backend\_kompute\_buffer\_free\_buffer<!-- {{#callable:ggml_backend_kompute_buffer_free_buffer}} -->
Frees the Vulkan memory associated with a given backend buffer.
- **Inputs**:
    - `buffer`: A `ggml_backend_buffer_t` structure that contains the Vulkan memory context to be freed.
- **Control Flow**:
    - The function retrieves the Vulkan memory context from the provided `buffer`.
    - It checks if a Vulkan device is available using `ggml_vk_has_device()`.
    - If a device is available, it calls `ggml_vk_free_memory()` to free the Vulkan memory.
    - Finally, it deletes the memory context.
- **Output**: This function does not return a value; it performs memory deallocation.
- **Functions called**:
    - [`ggml_vk_has_device`](#ggml_vk_has_device)
    - [`ggml_vk_free_memory`](#ggml_vk_free_memory)


---
### ggml\_backend\_kompute\_buffer\_get\_base<!-- {{#callable:ggml_backend_kompute_buffer_get_base}} -->
The `ggml_backend_kompute_buffer_get_base` function retrieves the base data pointer from a Vulkan memory buffer.
- **Inputs**:
    - `buffer`: A `ggml_backend_buffer_t` structure that represents the Vulkan buffer from which the base data pointer is to be retrieved.
- **Control Flow**:
    - The function casts the `context` member of the `buffer` to a `ggml_vk_memory` pointer.
    - It accesses the `data` member of the `ggml_vk_memory` structure to retrieve the base pointer.
- **Output**: Returns a pointer to the base data of the Vulkan memory buffer.


---
### ggml\_backend\_kompute\_buffer\_set\_tensor<!-- {{#callable:ggml_backend_kompute_buffer_set_tensor}} -->
Sets the data of a `ggml_tensor` in a Vulkan backend buffer.
- **Inputs**:
    - `buffer`: A handle to the backend buffer where the tensor data is stored.
    - `tensor`: A pointer to the `ggml_tensor` structure that holds the tensor data.
    - `data`: A pointer to the new data that will be copied into the tensor.
    - `offset`: The offset in the tensor's data where the new data will be written.
    - `size`: The size in bytes of the data to be copied into the tensor.
- **Control Flow**:
    - The function begins by marking the `buffer` parameter as unused to avoid compiler warnings.
    - It retrieves the Vulkan tensor associated with the provided `tensor` using [`ggml_vk_get_tensor`](#ggml_vk_get_tensor).
    - An assertion checks that the retrieved tensor is valid.
    - The function then copies the specified `data` into the tensor's data buffer at the specified `offset` using `memcpy`.
    - Finally, it synchronizes the device with the updated tensor data by evaluating a command in the Vulkan sequence.
- **Output**: This function does not return a value; it modifies the tensor's data in place.
- **Functions called**:
    - [`ggml_vk_get_tensor`](#ggml_vk_get_tensor)


---
### ggml\_backend\_kompute\_buffer\_get\_tensor<!-- {{#callable:ggml_backend_kompute_buffer_get_tensor}} -->
Retrieves a tensor from a Vulkan buffer and copies it to a specified memory location.
- **Inputs**:
    - `buffer`: A handle to the Vulkan buffer from which the tensor is being retrieved.
    - `tensor`: A pointer to the `ggml_tensor` structure representing the tensor to be retrieved.
    - `data`: A pointer to the memory location where the tensor data will be copied.
    - `offset`: The offset in the tensor data from which to start copying.
    - `size`: The number of bytes to copy from the tensor data.
- **Control Flow**:
    - The function begins by marking the `buffer` parameter as unused.
    - It retrieves the Vulkan tensor associated with the provided `tensor` using [`ggml_vk_get_tensor`](#ggml_vk_get_tensor).
    - An assertion checks that the retrieved tensor is valid.
    - The function then synchronizes the local tensor data with the device using `komputeManager()->sequence()->eval<kp::OpTensorSyncLocal>`.
    - Finally, it copies the specified range of bytes from the tensor's data to the provided `data` pointer using `memcpy`.
- **Output**: This function does not return a value; it directly modifies the memory pointed to by the `data` parameter.
- **Functions called**:
    - [`ggml_vk_get_tensor`](#ggml_vk_get_tensor)


---
### ggml\_backend\_kompute\_buffer\_clear<!-- {{#callable:ggml_backend_kompute_buffer_clear}} -->
Clears the contents of a Vulkan buffer by setting all bytes to a specified value.
- **Inputs**:
    - `buffer`: A pointer to a `ggml_backend_buffer_t` structure representing the Vulkan buffer to be cleared.
    - `value`: An 8-bit unsigned integer value that will be used to set all bytes in the buffer.
- **Control Flow**:
    - The function retrieves the Vulkan memory context associated with the provided `buffer`.
    - It uses `memset` to fill the memory area pointed to by `memory->data` with the specified `value` for the size of the buffer.
    - If a staging buffer exists, it synchronizes the primary buffer with the staging buffer using the `komputeManager`.
- **Output**: The function does not return a value; it performs an in-place operation to clear the buffer.


---
### ggml\_backend\_kompute\_buffer\_type\_get\_name<!-- {{#callable:ggml_backend_kompute_buffer_type_get_name}} -->
This function retrieves the name of a buffer type associated with a given backend buffer type.
- **Inputs**:
    - `buft`: A pointer to a `ggml_backend_buffer_type_t` structure representing the backend buffer type.
- **Control Flow**:
    - The function casts the `context` member of the `buft` structure to a pointer of type `ggml_backend_kompute_buffer_type_context`.
    - It accesses the `name` member of the context structure.
    - The function returns the C-style string representation of the name using `c_str()`.
- **Output**: Returns a pointer to a constant character string representing the name of the buffer type.


---
### ggml\_backend\_kompute\_buffer\_type\_alloc\_buffer<!-- {{#callable:ggml_backend_kompute_buffer_type_alloc_buffer}} -->
Allocates a buffer of a specified size for a given backend buffer type.
- **Inputs**:
    - `buft`: The type of backend buffer to allocate.
    - `size`: The size of the buffer to allocate in bytes.
- **Control Flow**:
    - Calls [`ggml_backend_kompute_device_ref`](#ggml_backend_kompute_device_ref) to ensure the device is referenced before allocation.
    - Allocates memory using [`ggml_vk_allocate`](#ggml_vk_allocate) for the specified size.
    - Initializes the buffer using `ggml_backend_buffer_init` with the allocated memory context.
- **Output**: Returns a `ggml_backend_buffer_t` that represents the allocated buffer.
- **Functions called**:
    - [`ggml_backend_kompute_device_ref`](#ggml_backend_kompute_device_ref)
    - [`ggml_vk_allocate`](#ggml_vk_allocate)


---
### ggml\_backend\_kompute\_buffer\_type\_get\_alignment<!-- {{#callable:ggml_backend_kompute_buffer_type_get_alignment}} -->
The `ggml_backend_kompute_buffer_type_get_alignment` function retrieves the buffer alignment value for a specified backend buffer type.
- **Inputs**:
    - `buft`: A pointer to a `ggml_backend_buffer_type_t` structure representing the backend buffer type from which the alignment is to be retrieved.
- **Control Flow**:
    - The function casts the `context` member of the `buft` structure to a pointer of type `ggml_backend_kompute_buffer_type_context`.
    - It accesses the `buffer_alignment` member of the context structure and returns its value.
- **Output**: Returns a `size_t` value representing the buffer alignment for the specified backend buffer type.


---
### ggml\_backend\_vk\_buffer\_type\_get\_max\_size<!-- {{#callable:ggml_backend_vk_buffer_type_get_max_size}} -->
The `ggml_backend_vk_buffer_type_get_max_size` function retrieves the maximum allocation size for a Vulkan buffer type.
- **Inputs**:
    - `buft`: A pointer to a `ggml_backend_buffer_type_t` structure representing the buffer type for which the maximum size is being queried.
- **Control Flow**:
    - The function casts the `context` member of the `buft` structure to a pointer of type `ggml_backend_kompute_buffer_type_context`.
    - It then accesses the `max_alloc` member of the context structure and returns its value.
- **Output**: Returns a `size_t` value representing the maximum allocation size for the specified Vulkan buffer type.


---
### ggml\_backend\_kompute\_buffer\_type<!-- {{#callable:ggml_backend_kompute_buffer_type}} -->
This function retrieves the buffer type for a specified device in the Kompute backend.
- **Inputs**:
    - `device`: An integer representing the index of the device for which the buffer type is requested.
- **Control Flow**:
    - A mutex is locked to ensure thread safety during the execution of the function.
    - The function retrieves a list of available Vulkan devices and checks the count.
    - Assertions are made to ensure the requested device index is valid and does not exceed the maximum number of devices.
    - If the buffer types have not been initialized, they are set up for each available device.
    - The function returns a pointer to the buffer type corresponding to the specified device.
- **Output**: Returns a pointer to a `ggml_backend_buffer_type` structure that contains information about the buffer type for the specified device.
- **Functions called**:
    - [`ggml_vk_available_devices`](#ggml_vk_available_devices)
    - [`ggml_backend_kompute_reg`](#ggml_backend_kompute_reg)


---
### ggml\_backend\_kompute\_name<!-- {{#callable:ggml_backend_kompute_name}} -->
Returns the name of the `ggml_backend_kompute` context.
- **Inputs**:
    - `backend`: A pointer to a `ggml_backend_t` structure representing the backend context.
- **Control Flow**:
    - The function casts the `context` member of the `backend` structure to a `ggml_kompute_context` pointer.
    - It accesses the `name` member of the `ggml_kompute_context` structure.
    - The function returns the C-style string representation of the `name`.
- **Output**: Returns a pointer to a constant character string representing the name of the `ggml_kompute_context`.


---
### ggml\_backend\_kompute\_free<!-- {{#callable:ggml_backend_kompute_free}} -->
Frees the resources associated with the `ggml_backend_t` structure and its context.
- **Inputs**:
    - `backend`: A pointer to a `ggml_backend_t` structure representing the backend context to be freed.
- **Control Flow**:
    - The function starts by casting the `backend->context` to a `ggml_kompute_context` pointer.
    - It asserts that the context being freed is the same as the global `s_kompute_context`.
    - The global `s_kompute_context` is set to nullptr to indicate that it is no longer in use.
    - If the context pointer is not null, it deletes the context to free its resources.
    - Finally, it deletes the `backend` itself to free the associated memory.
- **Output**: This function does not return a value; it performs cleanup operations to free memory.


---
### ggml\_backend\_kompute\_graph\_compute<!-- {{#callable:ggml_backend_kompute_graph_compute}} -->
Computes the graph defined in the `ggml_cgraph` using the specified `ggml_backend`.
- **Inputs**:
    - `backend`: A pointer to a `ggml_backend_t` structure that contains the context and device information for the computation.
    - `cgraph`: A pointer to a `ggml_cgraph` structure that represents the computation graph to be executed.
- **Control Flow**:
    - The function retrieves the `ggml_kompute_context` from the `backend` parameter.
    - It then calls the [`ggml_vk_graph_compute`](#ggml_vk_graph_compute) function, passing the context and the computation graph.
    - Finally, it returns a success status indicating the computation was initiated successfully.
- **Output**: Returns a `ggml_status` indicating the success of the computation initiation, specifically `GGML_STATUS_SUCCESS`.
- **Functions called**:
    - [`ggml_vk_graph_compute`](#ggml_vk_graph_compute)


---
### ggml\_backend\_kompute\_guid<!-- {{#callable:ggml_backend_kompute_guid}} -->
Returns a static GUID for the Kompute backend.
- **Inputs**: None
- **Control Flow**:
    - The function defines a static variable `guid` initialized with a specific byte sequence.
    - The function returns the address of the static `guid` variable.
- **Output**: Returns a pointer to a static `ggml_guid_t` structure containing the GUID.


---
### ggml\_backend\_kompute\_init<!-- {{#callable:ggml_backend_kompute_init}} -->
Initializes the Kompute backend for GPU computation.
- **Inputs**:
    - `device`: An integer representing the device index to be used for the Kompute backend.
- **Control Flow**:
    - Asserts that the global `s_kompute_context` is null to ensure that the backend is not already initialized.
    - Creates a new `ggml_kompute_context` instance with the specified device.
    - Allocates a new `ggml_backend` structure, initializing its fields with a unique GUID, the backend interface, the device registration, and the context.
    - Returns the newly created `ggml_backend` instance.
- **Output**: Returns a pointer to a `ggml_backend_t` structure representing the initialized Kompute backend.
- **Functions called**:
    - [`ggml_backend_kompute_guid`](#ggml_backend_kompute_guid)
    - [`ggml_backend_kompute_reg`](#ggml_backend_kompute_reg)


---
### ggml\_backend\_is\_kompute<!-- {{#callable:ggml_backend_is_kompute}} -->
The `ggml_backend_is_kompute` function checks if a given backend is of type Kompute.
- **Inputs**:
    - `backend`: A pointer to a `ggml_backend_t` structure representing the backend to be checked.
- **Control Flow**:
    - The function first checks if the `backend` pointer is not NULL.
    - If the pointer is valid, it then calls [`ggml_guid_matches`](../ggml.c.driver.md#ggml_guid_matches) to compare the GUID of the backend with the GUID of the Kompute backend.
    - The function returns the result of the comparison.
- **Output**: Returns a boolean value indicating whether the provided backend is a Kompute backend (true) or not (false).
- **Functions called**:
    - [`ggml_guid_matches`](../ggml.c.driver.md#ggml_guid_matches)
    - [`ggml_backend_kompute_guid`](#ggml_backend_kompute_guid)


---
### ggml\_backend\_kompute\_get\_device\_count<!-- {{#callable:ggml_backend_kompute_get_device_count}} -->
The `ggml_backend_kompute_get_device_count` function retrieves the number of available Vulkan devices.
- **Inputs**: None
- **Control Flow**:
    - Calls the [`ggml_vk_available_devices`](#ggml_vk_available_devices) function to get a list of available Vulkan devices.
    - Returns the size of the devices vector, which represents the count of available devices.
- **Output**: Returns the number of Vulkan devices available for computation.
- **Functions called**:
    - [`ggml_vk_available_devices`](#ggml_vk_available_devices)


---
### ggml\_backend\_kompute\_get\_device\_description<!-- {{#callable:ggml_backend_kompute_get_device_description}} -->
Retrieves a description of a specified Vulkan device.
- **Inputs**:
    - `device`: An integer representing the index of the device for which the description is requested.
    - `description`: A pointer to a character array where the device description will be stored.
    - `description_size`: The size of the character array to ensure that the description does not exceed this limit.
- **Control Flow**:
    - Calls `ggml_vk_available_devices()` to retrieve a list of available Vulkan devices.
    - Asserts that the provided device index is valid by checking it against the size of the devices vector.
    - Uses `snprintf` to format the device name into the provided description buffer, ensuring it does not exceed the specified size.
- **Output**: The function does not return a value but populates the `description` buffer with the name of the specified device.
- **Functions called**:
    - [`ggml_vk_available_devices`](#ggml_vk_available_devices)


---
### ggml\_backend\_kompute\_get\_device\_memory<!-- {{#callable:ggml_backend_kompute_get_device_memory}} -->
Retrieves the total and free memory available on a specified Vulkan device.
- **Inputs**:
    - `device`: An integer representing the index of the Vulkan device.
    - `free`: A pointer to a size_t variable where the amount of free memory will be stored.
    - `total`: A pointer to a size_t variable where the total memory will be stored.
- **Control Flow**:
    - Calls `ggml_vk_available_devices()` to get a list of available Vulkan devices.
    - Asserts that the provided device index is valid by checking it against the size of the devices vector.
    - Sets the total memory to the heap size of the specified device.
    - Sets the free memory to the same value as total memory, indicating all memory is available.
- **Output**: The function does not return a value; instead, it updates the values pointed to by the `free` and `total` pointers with the respective memory sizes.
- **Functions called**:
    - [`ggml_vk_available_devices`](#ggml_vk_available_devices)


---
### ggml\_backend\_kompute\_device\_get\_name<!-- {{#callable:ggml_backend_kompute_device_get_name}} -->
The `ggml_backend_kompute_device_get_name` function retrieves the name of a specified device in the Kompute backend.
- **Inputs**:
    - `dev`: A pointer to a `ggml_backend_dev_t` structure representing the device whose name is to be retrieved.
- **Control Flow**:
    - The function casts the `context` member of the `dev` structure to a `ggml_backend_kompute_device_context` pointer.
    - It accesses the `name` member of the context structure and returns it as a C-style string using `c_str()`.
- **Output**: Returns a pointer to a constant character string representing the name of the device.


---
### ggml\_backend\_kompute\_device\_get\_description<!-- {{#callable:ggml_backend_kompute_device_get_description}} -->
This function retrieves the description of a specified compute device.
- **Inputs**:
    - `dev`: A pointer to a `ggml_backend_dev_t` structure representing the compute device.
- **Control Flow**:
    - The function casts the `context` member of the `dev` structure to a `ggml_backend_kompute_device_context` pointer.
    - It accesses the `description` member of the context and returns its C-style string representation.
- **Output**: Returns a pointer to a constant character string that describes the compute device.


---
### ggml\_backend\_kompute\_device\_get\_memory<!-- {{#callable:ggml_backend_kompute_device_get_memory}} -->
Retrieves the memory information (free and total) for a specified device in the Kompute backend.
- **Inputs**:
    - `dev`: A pointer to a `ggml_backend_dev_t` structure representing the device for which memory information is being retrieved.
    - `free`: A pointer to a `size_t` variable where the amount of free memory will be stored.
    - `total`: A pointer to a `size_t` variable where the total amount of memory will be stored.
- **Control Flow**:
    - The function casts the `dev` pointer to a `ggml_backend_kompute_device_context` structure to access device-specific context.
    - It then calls the [`ggml_backend_kompute_get_device_memory`](#ggml_backend_kompute_get_device_memory) function, passing the device index and the pointers to `free` and `total` to retrieve the memory information.
- **Output**: This function does not return a value; instead, it populates the `free` and `total` variables with the respective memory information.
- **Functions called**:
    - [`ggml_backend_kompute_get_device_memory`](#ggml_backend_kompute_get_device_memory)


---
### ggml\_backend\_kompute\_device\_get\_buffer\_type<!-- {{#callable:ggml_backend_kompute_device_get_buffer_type}} -->
Retrieves the buffer type associated with a given Kompute device.
- **Inputs**:
    - `dev`: A pointer to a `ggml_backend_dev_t` structure representing the device from which to retrieve the buffer type.
- **Control Flow**:
    - The function casts the `dev->context` to a `ggml_backend_kompute_device_context` structure to access device-specific information.
    - It then calls [`ggml_backend_kompute_buffer_type`](#ggml_backend_kompute_buffer_type) with the device identifier to obtain the corresponding buffer type.
- **Output**: Returns a `ggml_backend_buffer_type_t` representing the buffer type associated with the specified device.
- **Functions called**:
    - [`ggml_backend_kompute_buffer_type`](#ggml_backend_kompute_buffer_type)


---
### ggml\_backend\_kompute\_device\_supports\_buft<!-- {{#callable:ggml_backend_kompute_device_supports_buft}} -->
Determines if a given device supports a specific buffer type.
- **Inputs**:
    - `dev`: A handle to the device being queried for support.
    - `buft`: A handle to the buffer type being checked for support.
- **Control Flow**:
    - Check if the `get_name` function of the buffer type interface matches the `ggml_backend_kompute_buffer_type_get_name` function.
    - If the names do not match, return false indicating the device does not support the buffer type.
    - Cast the device context and buffer type context to their respective structures.
    - Compare the device IDs of the buffer type context and the device context.
    - Return true if they match, indicating support for the buffer type.
- **Output**: Returns a boolean value indicating whether the device supports the specified buffer type.


---
### ggml\_backend\_kompute\_device\_get\_type<!-- {{#callable:ggml_backend_kompute_device_get_type}} -->
The function `ggml_backend_kompute_device_get_type` returns the type of the specified device as a GPU.
- **Inputs**:
    - `dev`: An instance of `ggml_backend_dev_t` representing the device whose type is to be retrieved.
- **Control Flow**:
    - The function begins by marking the input parameter `dev` as unused to avoid compiler warnings.
    - It then directly returns the constant value `GGML_BACKEND_DEVICE_TYPE_GPU`, indicating that the device type is a GPU.
- **Output**: The function outputs an enumeration value of type `ggml_backend_dev_type`, specifically indicating that the device is of type GPU.


---
### ggml\_backend\_kompute\_device\_get\_props<!-- {{#callable:ggml_backend_kompute_device_get_props}} -->
Retrieves properties of a specified compute device.
- **Inputs**:
    - `dev`: A `ggml_backend_dev_t` representing the compute device whose properties are to be retrieved.
    - `props`: A pointer to a `ggml_backend_dev_props` structure where the properties of the device will be stored.
- **Control Flow**:
    - Calls [`ggml_backend_kompute_device_get_name`](#ggml_backend_kompute_device_get_name) to get the device name and assigns it to `props->name`.
    - Calls [`ggml_backend_kompute_device_get_description`](#ggml_backend_kompute_device_get_description) to get the device description and assigns it to `props->description`.
    - Calls [`ggml_backend_kompute_device_get_type`](#ggml_backend_kompute_device_get_type) to determine the type of the device and assigns it to `props->type`.
    - Calls [`ggml_backend_kompute_device_get_memory`](#ggml_backend_kompute_device_get_memory) to retrieve the free and total memory of the device and assigns them to `props->memory_free` and `props->memory_total` respectively.
    - Initializes the `props->caps` structure with specific capabilities set to false.
- **Output**: The function does not return a value; instead, it populates the `props` structure with the device's properties.
- **Functions called**:
    - [`ggml_backend_kompute_device_get_name`](#ggml_backend_kompute_device_get_name)
    - [`ggml_backend_kompute_device_get_description`](#ggml_backend_kompute_device_get_description)
    - [`ggml_backend_kompute_device_get_type`](#ggml_backend_kompute_device_get_type)
    - [`ggml_backend_kompute_device_get_memory`](#ggml_backend_kompute_device_get_memory)


---
### ggml\_backend\_kompute\_device\_init<!-- {{#callable:ggml_backend_kompute_device_init}} -->
Initializes a `ggml_backend` for a specified device using the Kompute framework.
- **Inputs**:
    - `dev`: A pointer to a `ggml_backend_dev_t` structure representing the device to be initialized.
    - `params`: A string containing parameters for device initialization, which is unused in this function.
- **Control Flow**:
    - The function begins by marking the `params` argument as unused to avoid compiler warnings.
    - It retrieves the context associated with the provided device by casting `dev->context` to `ggml_backend_kompute_device_context`.
    - The function then calls [`ggml_backend_kompute_init`](#ggml_backend_kompute_init) with the device ID to initialize the backend.
    - Finally, it returns the initialized backend.
- **Output**: Returns a pointer to a `ggml_backend_t` structure representing the initialized backend.
- **Functions called**:
    - [`ggml_backend_kompute_init`](#ggml_backend_kompute_init)


---
### ggml\_backend\_kompute\_device\_offload\_op<!-- {{#callable:ggml_backend_kompute_device_offload_op}} -->
Determines if a given operation can be offloaded to a compute device based on its batch size and operation type.
- **Inputs**:
    - `dev`: A `ggml_backend_dev_t` representing the compute device.
    - `op`: A pointer to a `ggml_tensor` structure representing the operation to be evaluated.
- **Control Flow**:
    - Defines a constant `min_batch_size` set to 32.
    - Checks if the second dimension of the tensor `op` is greater than or equal to `min_batch_size` and the operation type is not `GGML_OP_GET_ROWS`.
    - Alternatively, checks if the third dimension of the tensor `op` is greater than or equal to `min_batch_size` and the operation type is `GGML_OP_MUL_MAT_ID`.
    - Returns true if either of the above conditions is satisfied, otherwise returns false.
- **Output**: Returns a boolean indicating whether the operation can be offloaded to the compute device.


---
### ggml\_backend\_kompute\_reg\_get\_name<!-- {{#callable:ggml_backend_kompute_reg_get_name}} -->
The `ggml_backend_kompute_reg_get_name` function returns the name of the Kompute backend.
- **Inputs**: None
- **Control Flow**:
    - The function uses the `GGML_UNUSED` macro to suppress warnings about the unused `reg` parameter.
    - It directly returns the string literal 'Kompute'.
- **Output**: The output is a constant string 'Kompute', representing the name of the backend.


---
### ggml\_backend\_kompute\_reg\_get\_device\_count<!-- {{#callable:ggml_backend_kompute_reg_get_device_count}} -->
The `ggml_backend_kompute_reg_get_device_count` function retrieves the number of available devices in the Kompute backend.
- **Inputs**: None
- **Control Flow**:
    - The function does not contain any control flow statements as it directly calls another function.
    - It uses the [`ggml_backend_kompute_get_device_count`](#ggml_backend_kompute_get_device_count) function to obtain the device count.
- **Output**: The function returns a size_t value representing the number of devices available in the Kompute backend.
- **Functions called**:
    - [`ggml_backend_kompute_get_device_count`](#ggml_backend_kompute_get_device_count)


---
### ggml\_backend\_kompute\_reg\_get\_device<!-- {{#callable:ggml_backend_kompute_reg_get_device}} -->
The `ggml_backend_kompute_reg_get_device` function retrieves a device from the Kompute backend registry.
- **Inputs**:
    - `reg`: A reference to the backend registry from which the device is to be retrieved.
    - `device`: An index representing the specific device to retrieve from the registry.
- **Control Flow**:
    - The function uses a static vector to store devices and a static boolean to track initialization.
    - A mutex is used to ensure thread safety during the initialization process.
    - If the devices have not been initialized, it iterates over the available devices, creating a context for each device and storing it in the vector.
    - After initialization, it asserts that the requested device index is valid and returns the corresponding device.
- **Output**: Returns a pointer to the specified device in the Kompute backend registry.
- **Functions called**:
    - [`ggml_backend_kompute_get_device_count`](#ggml_backend_kompute_get_device_count)
    - [`ggml_backend_kompute_get_device_description`](#ggml_backend_kompute_get_device_description)


---
### ggml\_backend\_kompute\_reg<!-- {{#callable:ggml_backend_kompute_reg}} -->
The `ggml_backend_kompute_reg` function returns a pointer to a static `ggml_backend_reg` structure that contains backend registration information.
- **Inputs**: None
- **Control Flow**:
    - The function defines a static variable `reg` of type `ggml_backend_reg`.
    - The `reg` structure is initialized with the API version, interface pointer, and a null context.
    - The function returns the address of the `reg` variable.
- **Output**: The output is a pointer to the static `ggml_backend_reg` structure, which contains information about the backend's API version, interface, and context.


