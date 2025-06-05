# Purpose
This C header file defines an interface for interacting with Vulkan devices, specifically tailored for a backend system named "Kompute." It includes structures and function prototypes to manage and query Vulkan devices, such as `ggml_vk_device`, which holds information about a Vulkan device, and functions like [`ggml_vk_available_devices`](#ggml_vk_available_devices) and [`ggml_vk_get_device`](#ggml_vk_get_device) to retrieve and select devices based on memory requirements. The file also declares a backend API for initializing and interacting with the Kompute backend, providing functions like [`ggml_backend_kompute_init`](#ggml_backend_tggml_backend_kompute_init) and [`ggml_backend_is_kompute`](#ggml_backend_is_kompute) to manage backend operations. The use of `#pragma once` ensures the file is included only once per compilation, and the `extern "C"` block allows the header to be used in C++ projects, maintaining C linkage.
# Imports and Dependencies

---
- `ggml.h`
- `ggml-backend.h`
- `stdbool.h`
- `stddef.h`
- `stdint.h`


# Data Structures

---
### ggml\_vk\_device
- **Type**: `struct`
- **Members**:
    - `index`: An integer representing the index of the Vulkan device.
    - `type`: An integer representing the type of the Vulkan device, corresponding to VkPhysicalDeviceType.
    - `heapSize`: A size_t value indicating the total heap size available on the device.
    - `name`: A pointer to a constant character string representing the name of the device.
    - `vendor`: A pointer to a constant character string representing the vendor of the device.
    - `subgroupSize`: An integer representing the size of the subgroup for the device.
    - `bufferAlignment`: A 64-bit unsigned integer specifying the alignment requirement for buffers on the device.
    - `maxAlloc`: A 64-bit unsigned integer indicating the maximum allocation size supported by the device.
- **Description**: The `ggml_vk_device` structure is used to represent a Vulkan device in the GGML library, encapsulating various properties such as the device index, type, heap size, name, vendor, subgroup size, buffer alignment, and maximum allocation size. This structure is essential for managing and interfacing with Vulkan devices, providing necessary information for device selection and resource management in Vulkan-based computations.


# Function Declarations (Public API)

---
### ggml\_vk\_available\_devices<!-- {{#callable_declaration:ggml_vk_available_devices}} -->
Retrieves a list of available Vulkan devices that meet the specified memory requirement.
- **Description**: Use this function to obtain a list of Vulkan devices that are available on the system and meet a specified memory requirement. This is useful for applications that need to select a suitable Vulkan device for computation based on available resources. The function returns a pointer to an array of `ggml_vk_device` structures, each representing a device, and populates the `count` parameter with the number of devices found. The caller must ensure that the `count` pointer is valid and can be written to. The function will return `NULL` if no devices meet the criteria or if Vulkan is not supported on the system.
- **Inputs**:
    - `memoryRequired`: Specifies the minimum amount of memory (in bytes) that a device must have to be included in the list. Must be a non-negative value.
    - `count`: A pointer to a `size_t` variable where the function will store the number of devices found. Must not be null.
- **Output**: Returns a pointer to an array of `ggml_vk_device` structures representing the available devices, or `NULL` if no suitable devices are found.
- **See also**: [`ggml_vk_available_devices`](../src/ggml-kompute/ggml-kompute.cpp.driver.md#ggml_vk_available_devices)  (Implementation)


---
### ggml\_vk\_has\_vulkan<!-- {{#callable_declaration:ggml_vk_has_vulkan}} -->
Check if Vulkan is available on the system.
- **Description**: Use this function to determine if Vulkan support is present on the system, which is necessary for utilizing Vulkan-based operations. This function should be called before attempting to perform any Vulkan-related tasks to ensure that the required Vulkan capabilities are available. It is a simple check that does not require any parameters and can be used to conditionally enable Vulkan features in your application.
- **Inputs**: None
- **Output**: Returns true if Vulkan is available on the system, otherwise false.
- **See also**: [`ggml_vk_has_vulkan`](../src/ggml-kompute/ggml-kompute.cpp.driver.md#ggml_vk_has_vulkan)  (Implementation)


---
### ggml\_vk\_has\_device<!-- {{#callable_declaration:ggml_vk_has_device}} -->
Check if a Vulkan device is available.
- **Description**: Use this function to determine if there is at least one Vulkan-compatible device available for use. This is typically called before attempting to initialize or use Vulkan resources to ensure that the necessary hardware support is present. It is a simple check that returns a boolean indicating the presence of a Vulkan device, and it does not require any parameters.
- **Inputs**: None
- **Output**: Returns a boolean value: `true` if a Vulkan device is available, `false` otherwise.
- **See also**: [`ggml_vk_has_device`](../src/ggml-kompute/ggml-kompute.cpp.driver.md#ggml_vk_has_device)  (Implementation)


---
### ggml\_vk\_current\_device<!-- {{#callable_declaration:ggml_vk_current_device}} -->
Retrieves the current Vulkan device in use.
- **Description**: This function returns the Vulkan device currently being used by the system. It should be called when you need to obtain information about the active Vulkan device, such as its index, type, and memory properties. The function assumes that a Vulkan device is available and will assert if no device is found. It is important to ensure that Vulkan is supported and a device is available before calling this function to avoid assertion failures.
- **Inputs**: None
- **Output**: Returns a `ggml_vk_device` structure containing details about the current Vulkan device, such as its index, type, heap size, name, vendor, subgroup size, buffer alignment, and maximum allocation size.
- **See also**: [`ggml_vk_current_device`](../src/ggml-kompute/ggml-kompute.cpp.driver.md#ggml_vk_current_device)  (Implementation)


---
### ggml\_backend\_is\_kompute<!-- {{#callable_declaration:ggml_backend_is_kompute}} -->
Check if a backend is a Kompute backend.
- **Description**: Use this function to determine if a given backend is specifically a Kompute backend. This is useful when you need to verify the type of backend you are working with, especially in environments where multiple backend types may be present. The function should be called with a valid backend handle, and it will return a boolean indicating whether the backend is of the Kompute type. Ensure that the backend is not null before calling this function to avoid undefined behavior.
- **Inputs**:
    - `backend`: A handle to a backend object. Must not be null. The function checks if this backend is of the Kompute type.
- **Output**: Returns true if the backend is a Kompute backend, false otherwise.
- **See also**: [`ggml_backend_is_kompute`](../src/ggml-kompute/ggml-kompute.cpp.driver.md#ggml_backend_is_kompute)  (Implementation)


