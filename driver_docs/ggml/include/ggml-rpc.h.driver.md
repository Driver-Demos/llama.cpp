# Purpose
This C header file defines an interface for a remote procedure call (RPC) backend within a larger system, likely related to the GGML library, which is suggested by the included headers "ggml.h" and "ggml-backend.h". The file specifies several function prototypes that facilitate the initialization and management of RPC backends, such as initializing an RPC backend, checking if a backend is RPC, determining buffer types, retrieving device memory information, starting an RPC server, registering the backend, and adding devices. It also defines versioning constants for the RPC protocol and a maximum number of servers, indicating a structured approach to managing multiple RPC connections. The use of `extern "C"` ensures compatibility with C++ compilers, allowing the functions to be used in C++ projects.
# Imports and Dependencies

---
- `ggml.h`
- `ggml-backend.h`


# Function Declarations (Public API)

---
### ggml\_backend\_is\_rpc<!-- {{#callable_declaration:ggml_backend_is_rpc}} -->
Checks if a backend is an RPC backend.
- **Description**: Use this function to determine if a given backend is configured as an RPC backend. This is useful when you need to verify the type of backend you are working with, especially in environments where multiple backend types may be present. The function expects a valid backend object and will return false if the backend is null or does not match the RPC backend identifier.
- **Inputs**:
    - `backend`: A handle to a backend object. It must not be null. If the backend is null or does not match the RPC backend identifier, the function returns false.
- **Output**: Returns true if the backend is an RPC backend, otherwise returns false.
- **See also**: [`ggml_backend_is_rpc`](../src/ggml-rpc/ggml-rpc.cpp.driver.md#ggml_backend_is_rpc)  (Implementation)


---
### ggml\_backend\_rpc\_get\_device\_memory<!-- {{#callable_declaration:ggml_backend_rpc_get_device_memory}} -->
Retrieve the free and total device memory from a remote endpoint.
- **Description**: This function is used to query a remote endpoint for the amount of free and total device memory available. It is typically called when there is a need to monitor or manage memory resources on a remote device. The function requires a valid endpoint string to establish a connection. If the connection to the endpoint cannot be established, both memory values are set to zero. This function should be used when the memory status of a remote device needs to be known, and it assumes that the endpoint is correctly configured and accessible.
- **Inputs**:
    - `endpoint`: A string representing the remote endpoint to query. It must be a valid and accessible endpoint. If the endpoint is invalid or unreachable, the function will set both memory values to zero.
    - `free`: A pointer to a size_t variable where the function will store the amount of free device memory. The caller must provide a valid pointer.
    - `total`: A pointer to a size_t variable where the function will store the total device memory. The caller must provide a valid pointer.
- **Output**: None
- **See also**: [`ggml_backend_rpc_get_device_memory`](../src/ggml-rpc/ggml-rpc.cpp.driver.md#ggml_backend_rpc_get_device_memory)  (Implementation)


---
### ggml\_backend\_rpc\_start\_server<!-- {{#callable_declaration:ggml_backend_rpc_start_server}} -->
Starts an RPC server for the specified backend.
- **Description**: This function initializes and starts an RPC server for the given backend, allowing it to handle client connections. It requires a valid endpoint to bind the server, and optionally a cache directory for local storage. The function also takes memory parameters to specify the available and total memory for the backend. It is essential to ensure that the endpoint is correctly formatted and that the backend is properly initialized before calling this function. The server will continuously accept client connections until it is manually stopped or an error occurs.
- **Inputs**:
    - `backend`: The backend instance for which the RPC server is being started. It must be a valid and initialized backend object.
    - `endpoint`: A string representing the network endpoint where the server will listen for connections. It must be a valid endpoint format, such as 'host:port'.
    - `cache_dir`: An optional string specifying the directory for local cache storage. If null, no local caching will be used.
    - `free_mem`: The amount of free memory available for the backend, specified in bytes. It should be a non-negative value.
    - `total_mem`: The total memory available for the backend, specified in bytes. It should be a non-negative value.
- **Output**: None
- **See also**: [`ggml_backend_rpc_start_server`](../src/ggml-rpc/ggml-rpc.cpp.driver.md#ggml_backend_rpc_start_server)  (Implementation)


