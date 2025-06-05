# Purpose
This C header file defines an interface for a backend system using SYCL (a C++-based parallel programming model) within the GGML library, which is likely used for machine learning or numerical computations. It includes function declarations for initializing a SYCL backend, checking if a backend is SYCL-based, and managing device buffers, including splitting tensor buffers across multiple devices and handling pinned host buffers for efficient CPU-GPU data transfers. The file also provides functions to print available SYCL devices, retrieve a list of GPU IDs, get device descriptions, and query device memory statistics. The use of `extern "C"` indicates that these functions are intended to be callable from C++ code, ensuring compatibility across C and C++ projects.
# Imports and Dependencies

---
- `ggml.h`
- `ggml-backend.h`


# Function Declarations (Public API)

---
### ggml\_backend\_is\_sycl<!-- {{#callable_declaration:ggml_backend_is_sycl}} -->
Check if a backend is using the SYCL API.
- **Description**: Use this function to determine if a given backend is utilizing the SYCL API. This is useful when you need to verify the type of backend being used, especially in environments where multiple backend types might be present. The function requires a valid backend object and will return false if the backend is null or does not match the SYCL backend identifier.
- **Inputs**:
    - `backend`: A handle to a backend object. It must not be null, as passing a null value will result in the function returning false. The caller retains ownership of the backend object.
- **Output**: Returns true if the backend is a SYCL backend, otherwise returns false.
- **See also**: [`ggml_backend_is_sycl`](../src/ggml-sycl/ggml-sycl.cpp.driver.md#ggml_backend_is_sycl)  (Implementation)


---
### ggml\_backend\_sycl\_print\_sycl\_devices<!-- {{#callable_declaration:ggml_backend_sycl_print_sycl_devices}} -->
Prints information about available SYCL devices.
- **Description**: Use this function to display detailed information about all available SYCL devices on the system. It provides a summary of each device's capabilities, including device type, name, version, compute units, work group size, memory size, and driver version. This function is useful for debugging and understanding the hardware resources available for SYCL-based computations. It does not require any parameters and does not return any values, making it straightforward to call when device information is needed.
- **Inputs**: None
- **Output**: None
- **See also**: [`ggml_backend_sycl_print_sycl_devices`](../src/ggml-sycl/ggml-sycl.cpp.driver.md#ggml_backend_sycl_print_sycl_devices)  (Implementation)


---
### ggml\_backend\_sycl\_get\_gpu\_list<!-- {{#callable_declaration:ggml_backend_sycl_get_gpu_list}} -->
Populates a list with available GPU device IDs.
- **Description**: This function retrieves the list of available GPU device IDs and populates the provided array with these IDs. It is useful for applications that need to enumerate and select from available GPU devices for SYCL-based computations. The function initializes the array with -1 to indicate unused slots, and fills it with device IDs up to the specified maximum length. It must be called with a valid pointer to an integer array and a non-negative maximum length. If the number of available devices exceeds the maximum length, only the first 'max_len' device IDs are stored.
- **Inputs**:
    - `id_list`: A pointer to an integer array where the function will store the GPU device IDs. The array must be pre-allocated by the caller and should have at least 'max_len' elements. The caller retains ownership of the array.
    - `max_len`: The maximum number of device IDs to store in the 'id_list' array. Must be a non-negative integer. If it is zero, the function will not store any device IDs.
- **Output**: None
- **See also**: [`ggml_backend_sycl_get_gpu_list`](../src/ggml-sycl/ggml-sycl.cpp.driver.md#ggml_backend_sycl_get_gpu_list)  (Implementation)


---
### ggml\_backend\_sycl\_get\_device\_description<!-- {{#callable_declaration:ggml_backend_sycl_get_device_description}} -->
Retrieves the description of a specified SYCL device.
- **Description**: Use this function to obtain a human-readable description of a SYCL device identified by its index. This is useful for displaying device information to users or for logging purposes. Ensure that the `description` buffer is large enough to hold the device description, as specified by `description_size`. The function must be called with a valid device index, and the `description` buffer must not be null. If the device index is invalid or an error occurs, the function will terminate the program.
- **Inputs**:
    - `device`: The index of the SYCL device for which the description is requested. Must be a valid device index; otherwise, the function will terminate the program.
    - `description`: A pointer to a character buffer where the device description will be stored. Must not be null, and the buffer should be large enough to hold the description.
    - `description_size`: The size of the `description` buffer. It should be sufficient to store the device description.
- **Output**: None
- **See also**: [`ggml_backend_sycl_get_device_description`](../src/ggml-sycl/ggml-sycl.cpp.driver.md#ggml_backend_sycl_get_device_description)  (Implementation)


---
### ggml\_backend\_sycl\_get\_device\_count<!-- {{#callable_declaration:ggml_backend_sycl_get_device_count}} -->
Returns the number of available SYCL devices.
- **Description**: Use this function to determine how many SYCL-compatible devices are available on the system. This is useful for applications that need to manage or distribute workloads across multiple devices. The function does not take any parameters and simply returns the count of devices, which can be used to iterate over available devices or to make decisions about resource allocation. It is expected to be called when initializing or configuring SYCL-based operations.
- **Inputs**: None
- **Output**: Returns an integer representing the number of SYCL devices available.
- **See also**: [`ggml_backend_sycl_get_device_count`](../src/ggml-sycl/ggml-sycl.cpp.driver.md#ggml_backend_sycl_get_device_count)  (Implementation)


---
### ggml\_backend\_sycl\_get\_device\_memory<!-- {{#callable_declaration:ggml_backend_sycl_get_device_memory}} -->
Retrieve the available and total memory for a specified SYCL device.
- **Description**: Use this function to obtain the amount of free and total memory available on a specific SYCL device. This is useful for managing resources and ensuring that your application does not exceed the memory limits of the device. The function must be called with a valid device identifier, and it will populate the provided pointers with the memory information. It is important to handle any potential exceptions that may arise during the execution of this function, as it may terminate the program if an error occurs.
- **Inputs**:
    - `device`: An integer representing the SYCL device identifier. It must be a valid device index within the range of available devices.
    - `free`: A pointer to a size_t variable where the function will store the amount of free memory on the device. The pointer must not be null.
    - `total`: A pointer to a size_t variable where the function will store the total memory available on the device. The pointer must not be null.
- **Output**: None
- **See also**: [`ggml_backend_sycl_get_device_memory`](../src/ggml-sycl/ggml-sycl.cpp.driver.md#ggml_backend_sycl_get_device_memory)  (Implementation)


